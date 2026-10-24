// include/strata/kernels/q8_1_finite.hpp - #606: the q8_1 activation block's scale and sum, kept finite.
//
// A q8_1 block stores its scale d = amax / 127 and the sum of its 32 raw activations as fp16. A sum past 65504 (one
// massive-activation dimension among 32 values is enough) rounds to inf, and d does past amax = 8.3M; the dot
// products then read inf * 0 = NaN, and a NaN in the residual ends as one token repeated (ggml-org/llama.cpp#23606
// is the same defect). Clamped to the largest finite half, such a block gives an approximate product instead of NaN,
// and the values are held to the int8 range they then exceed. Every block that was finite before is stored bit for
// bit as before: values in (65504, 65520) round to 65504 anyway, and a finite block's |x / d| is at most 127. A NaN
// stays NaN (the comparisons are false for it), so STRATA_DBG_NAN still sees one.
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace strata::kernels {

constexpr float kHalfMax = 65504.0f;

__dpct_inline__ float q8_1_finite(const float v) {
    return sycl::fabs(v) > kHalfMax ? sycl::copysign((float)kHalfMax, (float)v)
                                    : v;
}

/// The block's quantized value of `xi` for the (finite) scale `d`; `amax` = 0 is an all-zero block.
__dpct_inline__ int8_t q8_1_quant(const float xi, const float d,
                                  const float amax) {
    if (amax == 0.0f) return 0;
    const float q = sycl::round(xi / d);
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

__dpct_inline__ sycl::half2 q8_1_ds(const float d, const float sum) {
    return sycl::half2(
        sycl::vec<float, 1>(q8_1_finite(d))
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
        sycl::vec<float, 1>(q8_1_finite(sum))
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
}

}  // namespace strata::kernels
