// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{dsv4-hc.cu,scale.cu,unary.cu,unary.cuh}.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <cmath>

namespace strata::kernels {
namespace {
constexpr int THREADS = 256;
__dpct_inline__ float sigmoid(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}
// ggml SCALE uses scale*x+bias, including its +0 bias. Retain this operation
// explicitly so compile-time zero does not change signed-zero behavior.
__dpct_inline__ float scale_zero_bias(float x, float scale) {
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    return sycl::fma(scale, x, 0.0f);
}
__dpct_inline__ void down_silu(float *lo, int count, float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const std::size_t i =
        std::size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= std::size_t(count)) return;
    const float x = scale_zero_bias(lo[i], scale);
    lo[i] = x / (1.0f + sycl::native::exp(-x));
}
template <bool Fused>
__dpct_inline__ void
pre_gated(const float *__restrict__ xn, float *__restrict__ gate,
          float *__restrict__ mixed, int n_embd, int hc, float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const std::size_t d =
        std::size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (d >= std::size_t(n_embd)) return;
    const std::size_t t = item_ct1.get_group(1);
    xn += t * std::size_t(n_embd) * hc;
    gate += t * std::size_t(n_embd) * hc;
    mixed += t * std::size_t(n_embd);
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const std::size_t i = std::size_t(c) * n_embd + d;
        const float x = xn[i], w = sigmoid(gate[i]);
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const float product = x * w;
        gate[i] = product;
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        if constexpr (Fused) sum = sycl::fma(x, w, sum);
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        else sum = c == 0 ? product : sum + product;
    }
    if constexpr (Fused) mixed[d] = scale * sum;
    else mixed[d] = scale_zero_bias(sum, scale);
}
__dpct_inline__ void post(const float *residual,
                          const float *__restrict__ block_out,
                          const float *__restrict__ inject, float *output,
                          int n_embd, int hc, float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const std::size_t i =
        std::size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= std::size_t(n_embd) * hc) return;
    const std::size_t t = item_ct1.get_group(1);
    residual += t * std::size_t(n_embd) * hc;
    block_out += t * std::size_t(n_embd);
    inject += t * std::size_t(hc);
    output += t * std::size_t(n_embd) * hc;
    const int c = int(i / n_embd), d = int(i % n_embd);
    const float weight = scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
    // Exact residual/output alias is supported; no other thread reads residual[i].
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    output[i] = sycl::fma(block_out[d], weight, residual[i]);
}
void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float))
        throw std::invalid_argument("native GR postops require non-null four-byte aligned pointers");
}
void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}
unsigned blocks(std::size_t n) { return unsigned((n + THREADS - 1) / THREADS); }
void check_launch() {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error != 0)
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        throw std::runtime_error(std::string("native GR postops launch: ") +
                                 dpct::get_error_string_dummy(error));
}
} // namespace

void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream) {
    check_shape(hc_lr, hc);
    check_pointer(lo);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->submit([&](sycl::handler &cgh) {
                auto float_hc_ct2 = 1.0f / float(hc);

                cgh.parallel_for<dpct_kernel_name<class down_silu_24f4fb>>(
                    sycl::nd_range<3>(sycl::range(1, 1, blocks(hc_lr)) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        down_silu(lo, hc_lr, float_hc_ct2);
                    });
            });
    }
    check_launch();
}
void native_gr_pre_gated(const float* xn, float* gate, float* mixed,
                         int n_embd, int hc, bool fused_layer, void* stream) {
    native_gr_pre_gated_multi(xn, gate, mixed, n_embd, hc, 1, fused_layer, stream);
}
void native_gr_pre_gated_multi(const float* xn, float* gate, float* mixed,
                               int n_embd, int hc, int n_tok, bool fused_layer, void* stream) {
    if (n_tok <= 0)
        throw std::invalid_argument("native GR postops require positive n_tok");
    check_shape(n_embd, hc);
    check_pointer(xn); check_pointer(gate); check_pointer(mixed);
    const dpct::dim3 grid{blocks(n_embd), unsigned(n_tok), 1u};
    if (fused_layer)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->submit([&](sycl::handler &cgh) {
                auto float_hc_ct5 = 1.0f / float(hc);

                cgh.parallel_for<dpct_kernel_name<class pre_gated_e6e3d1,
                                                  dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        pre_gated<true>(xn, gate, mixed, n_embd, hc,
                                        float_hc_ct5);
                    });
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->submit([&](sycl::handler &cgh) {
                auto float_hc_ct5 = 1.0f / float(hc);

                cgh.parallel_for<dpct_kernel_name<class pre_gated_10bc88,
                                                  dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        pre_gated<false>(xn, gate, mixed, n_embd, hc,
                                         float_hc_ct5);
                    });
            });
    }
    check_launch();
}
void native_gr_post(const float* residual, const float* block_out, const float* inject,
                    float* output, int n_embd, int hc, void* stream) {
    native_gr_post_multi(residual, block_out, inject, output, n_embd, hc, 1, stream);
}
void native_gr_post_multi(const float* residual, const float* block_out, const float* inject,
                          float* output, int n_embd, int hc, int n_tok, void* stream) {
    if (n_tok <= 0)
        throw std::invalid_argument("native GR postops require positive n_tok");
    check_shape(n_embd, hc);
    check_pointer(residual); check_pointer(block_out); check_pointer(inject); check_pointer(output);
    const dpct::dim3 grid{blocks(std::size_t(n_embd) * hc), unsigned(n_tok),
                          1u};
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->submit([&](sycl::handler &cgh) {
                auto float_hc_ct6 = 1.0f / float(hc);

                cgh.parallel_for<dpct_kernel_name<class post_a7f844>>(
                    sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        post(residual, block_out, inject, output, n_embd, hc,
                             float_hc_ct6);
                    });
            });
    }
    check_launch();
}
} // namespace strata::kernels
