// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu,unary.cuh,ssm-conv.cu,scale.cu}.
// Compile with --use_fast_math, as the pinned backend does.
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
#include "strata/kernels/native_gdn_preprocess.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int S = 128;

__dpct_inline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        value += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            value, offset);
    return value;
}
__dpct_inline__ float norm_sum(float value, float *sums) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) % 32;
    value = warp_sum(value);
    if (lane == 0) sums[item_ct1.get_local_id(2) / 32] = value;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    value = lane < 8 ? sums[lane] : 0.0f;
    return warp_sum(value);
}
__dpct_inline__ float sigmoid(float value) {
    return 1.0f / (1.0f + sycl::native::exp(-value));
}

__dpct_inline__ void conv_silu(float *__restrict__ history,
                               const float *__restrict__ input,
                               const float *__restrict__ weights,
                               float *__restrict__ raw_output,
                               float *__restrict__ silu_output, int channels) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    float values[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], input[c]};
    float sum = 0.0f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) sum += values[tap] * weights[c * 4 + tap];
    // The native SSM kernel adds its zero bias even when there is no bias input.
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    sum = sum + 0.0f;
    raw_output[c] = sum;
    silu_output[c] = sum / (1.0f + sycl::native::exp(-sum));
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) history[c * 3 + tap] = values[tap + 1];
}

__dpct_inline__ void l2_norm(float *input, float epsilon, float scale_after) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int col = item_ct1.get_local_id(2);
    input += size_t(item_ct1.get_group(2)) * S;
    const float value = col < S ? input[col] : 0.0f;
    float partial = 0.0f;
    if (col < S) partial += value * value;
    auto &sums =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    partial = norm_sum(partial, sums);
    const float scale = sycl::rsqrt(partial / S + epsilon);
    if (col < S) {
        // Preserve the FP32 store boundary between RMSNorm and ggml_scale.
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const float normalized = scale * value;
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        input[col] = sycl::fma(normalized, scale_after, 0.0f);
    }
}

__dpct_inline__ void beta_sigmoid(float *beta, int count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < count) beta[i] = sigmoid(beta[i]);
}
__dpct_inline__ void gate_softplus(const float *__restrict__ alpha,
                                   const float *__restrict__ dt,
                                   const float *__restrict__ ssm_a,
                                   float *__restrict__ gate, int count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= count) return;
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float value = alpha[i] + dt[i];
    const float softplus = value > 20.0f
                               ? value
                               : sycl::log1p(sycl::native::exp(
                                     value)); // 1 + e^v loses e^v below ~1e-7
    gate[i] = softplus * ssm_a[i];
}

__dpct_inline__ void out_norm(const float *__restrict__ input,
                              const float *__restrict__ z,
                              const float *__restrict__ gamma,
                              float *__restrict__ output, float epsilon) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int col = item_ct1.get_local_id(2);
    const size_t offset = size_t(item_ct1.get_group(2)) * S;
    const float value = col < S ? input[offset + col] : 0.0f;
    float partial = 0.0f;
    if (col < S) partial += value * value;
    auto &sums =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    partial = norm_sum(partial, sums);
    const float scale = sycl::rsqrt(partial / S + epsilon);
    if (col < S) {
        // RMSNorm+gamma is one pinned fused operator, followed by sigmoid*mul.
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const float weighted = scale * value * gamma[col];
        output[offset + col] = weighted * sigmoid(z[offset + col]);
    }
}

struct Span { const void* pointer; size_t bytes; };
void valid(Span span) {
    const auto address = reinterpret_cast<uintptr_t>(span.pointer);
    if (!span.pointer || address % sizeof(float) || span.bytes > UINTPTR_MAX - address)
        throw std::invalid_argument("native GDN preprocessing requires aligned nonnull valid spans");
}
void disjoint(Span a, Span b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.pointer), bp = reinterpret_cast<uintptr_t>(b.pointer);
    if (ap < bp + b.bytes && bp < ap + a.bytes)
        throw std::invalid_argument("native GDN preprocessing requires disjoint writable spans");
}
void count_and_stream(int64_t count, void* stream) {
    if (!stream || count <= 0 || count > 65535)
        throw std::invalid_argument("native GDN preprocessing requires a stream and count in [1,65535]");
}
void norm_geometry(int64_t rows, int64_t cols, float epsilon, void* stream) {
    count_and_stream(rows, stream);
    if (cols != S || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GDN norm requires width 128 and finite nonnegative epsilon");
}
void check_launch() {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}
}

void native_gdn_conv_silu(float* history, const float* input, const float* weights,
                          float* raw_output, float* silu_output, int64_t channels,
                          int64_t d_conv, void* stream) {
    count_and_stream(channels, stream);
    if (d_conv != 4) throw std::invalid_argument("native GDN convolution requires four taps");
    const size_t bytes = size_t(channels) * sizeof(float);
    const Span writable[] = {{history, 3 * bytes}, {raw_output, bytes}, {silu_output, bytes}};
    const Span inputs[] = {{input, bytes}, {weights, 4 * bytes}};
    for (auto span : writable) valid(span);
    for (auto span : inputs) valid(span);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) disjoint(writable[i], writable[j]);
        for (auto span : inputs) disjoint(writable[i], span);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class conv_silu_1ba4e6>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, unsigned((channels + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    conv_silu(history, input, weights, raw_output, silu_output,
                              int(channels));
                });
    }
    check_launch();
}
void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    norm_geometry(rows, cols, epsilon, stream);
    valid({input, size_t(rows) * S * sizeof(float)});
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->submit([&](sycl::handler &cgh) {
                auto epsilon_S_ct1 = epsilon / S;
                auto sqrtf_float_S_ct2 = 1.0f / sqrtf(float(S));

                cgh.parallel_for<dpct_kernel_name<class l2_norm_bcf2af>>(
                    sycl::nd_range<3>(sycl::range(1, 1, unsigned(rows)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            l2_norm(input, epsilon_S_ct1, sqrtf_float_S_ct2);
                        });
            });
    }
    check_launch();
}
void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    valid({beta, size_t(heads) * sizeof(float)});
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class beta_sigmoid_fcf68f>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, unsigned((heads + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    beta_sigmoid(beta, int(heads));
                });
    }
    check_launch();
}
void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
                     float* gate, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    const size_t bytes = size_t(heads) * sizeof(float);
    const Span output{gate, bytes};
    valid(output);
    for (auto input : {Span{alpha, bytes}, Span{dt, bytes}, Span{ssm_a, bytes}}) {
        valid(input);
        disjoint(output, input);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class gate_softplus_a95cee>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, unsigned((heads + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gate_softplus(alpha, dt, ssm_a, gate, int(heads));
                });
    }
    check_launch();
}
void native_gdn_out_norm(const float* output, const float* z, const float* gamma,
                         float* destination, int64_t heads, int64_t cols,
                         float epsilon, void* stream) {
    norm_geometry(heads, cols, epsilon, stream);
    const size_t bytes = size_t(heads) * S * sizeof(float);
    const Span writable{destination, bytes};
    valid(writable);
    for (auto input : {Span{output, bytes}, Span{z, bytes}, Span{gamma, S * sizeof(float)}}) {
        valid(input);
        disjoint(writable, input);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class out_norm_a49e35>>(
                sycl::nd_range<3>(sycl::range(1, 1, unsigned(heads)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        out_norm(output, z, gamma, destination, epsilon);
                    });
    }
    check_launch();
}
} // namespace strata::kernels
