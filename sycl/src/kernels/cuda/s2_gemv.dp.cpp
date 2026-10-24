// src/kernels/cuda/s2_gemv.cu - P2.S2: the S2 GEMV, one thread per output row.
//
// Naive per the phase rule: dequantize on the fly, FP32 accumulation inside the row, no shared memory, no
// vector loads, no __ldg hints.  Phase 3 changes this file; the parity test is what says whether a change is
// still right.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/s2_gemv.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK = 64;

__dpct_inline__ void s2_gemv_kernel(const uint16_t *__restrict__ x,
                                    const uint8_t *__restrict__ codes,
                                    const float *__restrict__ scales,
                                    float *__restrict__ y, long long n_in,
                                    long long n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long o =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (o >= n_out) return;

    const long long nb = n_in / QK;
    const uint8_t* c = codes + o * nb * (QK / 4);
    const float* s = scales + o * nb;

    float acc = 0.0f;
    for (long long b = 0; b < nb; ++b) {
        const float d = s[b];
        const uint8_t* cb = c + b * (QK / 4);
        const uint16_t* xb = x + b * QK;
        for (int j = 0; j < QK; ++j) {
            // (code - 1) in the INTEGER domain, then the group scale, then the activation; the CPU reference
            // below is written in the same order so the two differ only by floating-point contraction.
            const int code = (cb[j >> 2] >> ((j & 3) * 2)) & 0x03;
            acc += (float)(code - 1) * d *
                   sycl::vec<sycl::half, 1>(
                       sycl::bit_cast<sycl::half, unsigned short>(xb[j]))
                       .convert<float, sycl::rounding_mode::automatic>()[0];
        }
    }
    y[o] = acc;
}

}  // namespace

void s2_gemv(const uint16_t *x, const uint8_t *codes, const float *scales,
             float *y, int64_t n_in, int64_t n_out) try {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK != 0) {
        std::fprintf(stderr, "s2_gemv: n_in %lld is not a multiple of %d\n", (long long) n_in, QK);
        std::exit(1);
    }
    const int threads = 128;
    const long long blocks = (n_out + threads - 1) / threads;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        dpct::get_in_order_queue()
            .parallel_for<dpct_kernel_name<class s2_gemv_kernel_163eb8>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)blocks) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    s2_gemv_kernel(x, codes, scales, y, n_in, n_out);
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
