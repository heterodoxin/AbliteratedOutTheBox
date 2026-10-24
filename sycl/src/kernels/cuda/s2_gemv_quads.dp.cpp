// src/kernels/cuda/s2_gemv_quads.cu - S2 with the code load amortised over four elements.
//
// THE HYPOTHESIS THIS TESTS.  L4 measured 66 G weights/s = 0.55 weights/cycle/SM, and at an estimated ~13
// instructions per weight that is ~7.2 instructions/cycle/SM against an SM's 4/cycle issue ceiling.  Compute
// (0.4% of FP32 peak) and bandwidth (2%) are excluded by measurement and the dependency chain was excluded by
// experiment in L4, so instruction ISSUE is what remains.  If that is right, cutting instructions per weight
// must move the rate; if the rate does not move, the hypothesis is wrong and that is worth knowing too.
//
// WHERE THE INSTRUCTIONS GO.  For S2 four consecutive elements share ONE byte (element i at byte i/4, bits
// (i%4)*2), and four halves of x are eight contiguous bytes.  The generic kernel walks with a thread stride
// and loads one byte and one half PER ELEMENT; a thread that takes a contiguous quad instead loads one byte
// and one 64-bit word for all four.  Memory instructions per element drop from 2 to 0.5, and the address
// arithmetic is computed once rather than four times.
//
// S2 IS SPECIALISED DELIBERATELY: it is 31.64 GiB of the 38 GiB pack, so it is the kernel that matters.  The
// generic `s_gemv_split` still serves S4 and S8.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/s_gemv.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// QK = 64 elements per group; a quad of 4 elements is 1/16 of a group, so
//     group = (quad * 4) >> 6 = quad >> 4
constexpr int QK_S2 = 64;
constexpr int CODES_PER_BYTE_S2 = 4;

__dpct_inline__ void s2_gemv_quads_kernel(const uint16_t *__restrict__ x,
                                          const uint8_t *__restrict__ codes,
                                          const float *__restrict__ scales,
                                          float *__restrict__ y, long long n_in,
                                          long long n_out, int threads_per_row,
                                          uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto partial = (float *)dpct_local;
    const long long o = item_ct1.get_group(2);
    if (o >= n_out) return;
    const int tid = item_ct1.get_local_id(2);

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;             // exactly one code byte per quad
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];                      // ONE load for four codes
        const float d = s[q >> 4];                      // (q*4) >> 6, the group index as a shift
        // ONE 64-bit load for four halves.  `x` is 256-byte aligned and four halves are eight bytes, so
        // this is an aligned uint2 - if it were not, the misaligned access would fault rather than be slow.
        const sycl::uint2 xw =
            *reinterpret_cast<const sycl::uint2 *>(x + q * 4);
        const sycl::half2 h01 = *reinterpret_cast<const sycl::half2 *>(&xw.x());
        const sycl::half2 h23 = *reinterpret_cast<const sycl::half2 *>(&xw.y());
        // the four codes, each carrying the -1 bias in the INTEGER domain; the scale is applied once per
        // element here rather than once per group, which is the same expression the generic kernel uses
        const float w0 = (float) ((int) (byte & 3) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
        a0 += w0 * h01[0];
        a1 += w1 * h01[1];
        a2 += w2 * h23[0];
        a3 += w3 * h23[1];
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

void s2_gemv_quads(const uint16_t *x, const uint8_t *codes, const float *scales,
                   float *y, int64_t n_in, int64_t n_out,
                   int threads_per_row) try {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0) {
        std::fprintf(stderr, "s2_gemv_quads: n_in %lld is not a multiple of 4\n", (long long) n_in);
        std::exit(1);
    }
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t smem = (size_t)threads_per_row * sizeof(float);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp16});

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem), cgh);

            cgh.parallel_for<
                dpct_kernel_name<class s2_gemv_quads_kernel_5797f4>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *
                                      sycl::range(1, 1, threads_per_row),
                                  sycl::range(1, 1, threads_per_row)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    s2_gemv_quads_kernel(
                        x, codes, scales, y, n_in, n_out, threads_per_row,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
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
