// src/kernels/cuda/s2_gemv_fast.cu - two levers on the hottest kernel, BOTH MEASURED NEGATIVE OR NEUTRAL.
//
// *** THIS FILE IS NAMED "fast" AND IS NOT FASTER.  DO NOT ADOPT IT AS THE FAST PATH. ***
// It is kept because the negative result is worth more than the file costs: it closes two plausible
// optimisations and it is the experiment that established the measurement floor.
//
// L5 got the S2 GEMV from 10.5 to 86.2 G weights/s by amortising the loads, and left the instruction-count
// argument confirmed but unfinished.  Two further reductions were obvious, so both were built and measured:
//
// LEVER 1 - STAGE `x` IN SHARED MEMORY.  `x` is n_in halves (5 KB at n_embd 2560) and every output row reads
// all of it, so the generic kernel issues n_out * n_in activation loads per GEMV.  Staging it once per block
// measured WORSE in every single paired comparison (59-76 G w/s staged against 64-87 G w/s unstaged, across
// four sweep passes and three thread counts).  The __syncthreads and the copy cost more than the L1/L2 hits
// they replace - the same 5 KB is already resident and already broadcast.
//
// LEVER 2 - A CONSTANT-MEMORY CODE TABLE.  Turn the per-element shifts, masks and int-to-float convert into
// one broadcast `float4` load per four elements.  Measured NEUTRAL: the two orderings swap between passes
// with no consistent winner.
//
// WHY THE NEGATIVE RESULT IS THE REAL FINDING: run-to-run variance on this machine is now ~20% within a
// single configuration (quads tpr=32 measured 64.3 to 79.5 G w/s across four passes), because `verify` holds
// ~97% of six CPU cores.  **The noise floor now EXCEEDS the effect sizes being chased**, so further
// micro-optimisation of this kernel cannot be evaluated here and should wait for a quiet machine rather than
// produce more numbers that cannot be distinguished from noise.
//
// Both changes alter the summation order or the expression, so this kernel is checked against the naive
// reference like every other one - being faster is never a reason to be trusted, and being neutral is not
// either.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/s_gemv.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK_S2 = 64;
constexpr int MAX_SHARED_HALVES = 4096;      // 8 KB of shared for x; n_embd 2560 fits with room

// byte -> the four code values with the -1 bias already applied, in element order (bits 0,2,4,6).
inline dpct::constant_memory<float, 2>& c_codes = *new dpct::constant_memory<float, 2>(256, 4);   // never freed: exit-order safe

bool g_lut_ready[64] = {};   // per device: __constant__ memory is per device (a layer split runs on two)

void ensure_lut() try {
    int dev = 0;
    dev = dpct::get_current_device_id();
    if (dev < 0 || dev >= 64) dev = 0;
    if (g_lut_ready[dev]) return;
    float host[256][4];
    for (int b = 0; b < 256; ++b) {
        for (int k = 0; k < 4; ++k) {
            host[b][k] = (float) (((b >> (2 * k)) & 3) - 1);
        }
    }
    const dpct::err0 e =
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(c_codes.get_ptr(), host, sizeof(host))
                             .wait());

    g_lut_ready[dev] = true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// One block per output row.  `staged` says whether x was copied to shared, so the SAME kernel covers both
// configurations and the bench can measure them against each other with nothing else changed.
template <bool STAGE_X>
/*
DPCT1110: The total declared local variable size in device function
s2_gemv_fast_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void s2_gemv_fast_kernel(
    const uint16_t *__restrict__ x, const uint8_t *__restrict__ codes,
    const float *__restrict__ scales, float *__restrict__ y, long long n_in,
    long long n_out, int threads_per_row, uint8_t *dpct_local,
    dpct::accessor<float, dpct::constant, 2> c_codes) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem = (float *)dpct_local;
    sycl::half *sx = reinterpret_cast<sycl::half *>(
        smem); // [threads_per_row, ...] partials live after
    float* partial = smem + (STAGE_X ? (MAX_SHARED_HALVES / 2) : 0);

    const long long o = item_ct1.get_group(2);
    if (o >= n_out) return;
    const int tid = item_ct1.get_local_id(2);

    if (STAGE_X) {
        const sycl::half *xh = reinterpret_cast<const sycl::half *>(x);
#pragma unroll
        for (long long i = tid; i < n_in; i += threads_per_row) sx[i] = xh[i];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];                          // ONE load for four codes
        const float d = s[q >> 4];                          // (q*4) >> 6
        const sycl::float4 cv = *reinterpret_cast<const sycl::float4 *>(
            &c_codes[byte][0]); // one broadcast load
        if (STAGE_X) {
            const sycl::half2 h01 =
                *reinterpret_cast<const sycl::half2 *>(&sx[q * 4]);
            const sycl::half2 h23 =
                *reinterpret_cast<const sycl::half2 *>(&sx[q * 4 + 2]);
            a0 += cv.x() * d * h01[0];
            a1 += cv.y() * d * h01[1];
            a2 += cv.z() * d * h23[0];
            a3 += cv.w() * d * h23[1];
        } else {
            const sycl::uint2 xw2 =
                *reinterpret_cast<const sycl::uint2 *>(x + q * 4);
            const sycl::half2 h01 =
                *reinterpret_cast<const sycl::half2 *>(&xw2.x());
            const sycl::half2 h23 =
                *reinterpret_cast<const sycl::half2 *>(&xw2.y());
            a0 += cv.x() * d * h01[0];
            a1 += cv.y() * d * h01[1];
            a2 += cv.z() * d * h23[0];
            a3 += cv.w() * d * h23[1];
        }
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (tid == 0) y[o] = partial[0];
}

}  // namespace

void s2_gemv_fast(const uint16_t *x, const uint8_t *codes, const float *scales,
                  float *y, int64_t n_in, int64_t n_out, int threads_per_row,
                  bool stage_x) try {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    ensure_lut();
    if (stage_x && n_in > MAX_SHARED_HALVES) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld exceeds the %d-half shared staging limit\n",
                     (long long) n_in, MAX_SHARED_HALVES);
        std::exit(1);
    }
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t smem = (stage_x ? MAX_SHARED_HALVES * sizeof(sycl::half) : 0) +
                        (size_t)threads_per_row * sizeof(float);
    if (stage_x) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        c_codes.init();

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp16});

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem), cgh);
            auto c_codes_acc_ct1 = c_codes.get_access(cgh);

            cgh.parallel_for<dpct_kernel_name<class s2_gemv_fast_kernel_3b9618,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *
                                      sycl::range(1, 1, threads_per_row),
                                  sycl::range(1, 1, threads_per_row)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    s2_gemv_fast_kernel<true>(
                        x, codes, scales, y, n_in, n_out, threads_per_row,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        c_codes_acc_ct1);
                });
        });
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        c_codes.init();

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp16});

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem), cgh);
            auto c_codes_acc_ct1 = c_codes.get_access(cgh);

            cgh.parallel_for<dpct_kernel_name<class s2_gemv_fast_kernel_61f77f,
                                              dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *
                                      sycl::range(1, 1, threads_per_row),
                                  sycl::range(1, 1, threads_per_row)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    s2_gemv_fast_kernel<false>(
                        x, codes, scales, y, n_in, n_out, threads_per_row,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        c_codes_acc_ct1);
                });
        });
    }
    const dpct::err0 e =
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
