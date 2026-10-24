// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh}. Scope: contiguous weighted F32 RMSNorm.
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
#include "strata/kernels/native_gr_norm.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

__dpct_inline__ float norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        value += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            value, offset);
    }
    return value;
}

template <int BlockSize>
__dpct_inline__ void weighted_rms_norm(const float *__restrict__ input,
                                       const float *__restrict__ gamma,
                                       float *__restrict__ output, int n_cols,
                                       float epsilon) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid = item_ct1.get_local_id(2);
    const std::size_t row =
        std::size_t(item_ct1.get_group(1)) * item_ct1.get_group_range(2) +
        item_ct1.get_group(2);
    const std::size_t row_offset = row * n_cols;
    input += row_offset;
    gamma += std::size_t(item_ct1.get_group(2)) * n_cols;
    output += row_offset;
    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }

    // Pinned block_reduce<SUM,BlockSize>: every warp repeats the final XOR
    // reduction. There is no warp-0-only broadcast or downward-shuffle tree.
    auto &sums =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    partial = 0.0f;
    if (lane < BlockSize / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);

    const float mean = partial / n_cols;
    const float scale = sycl::rsqrt(mean + epsilon);
#pragma unroll
    for (int col = tid; col < n_cols; col += BlockSize) {
        output[col] = scale * input[col] * gamma[col];
    }
}

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float) != 0)
        throw std::invalid_argument("native GR RMSNorm requires non-null four-byte aligned pointers");
}

} // namespace

void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                 int n_cols, int n_rows, float epsilon, void* stream) {
    native_gr_rms_norm_weighted_multi(input, gamma, output, n_cols, n_rows, 1, epsilon, stream);
}

void native_gr_rms_norm_weighted_multi(const float* input, const float* gamma, float* output,
                                       int n_cols, int n_rows, int n_tok, float epsilon, void* stream) {
    if (n_cols <= 0 || n_rows <= 0 || n_tok <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GR RMSNorm requires positive dimensions and finite nonnegative epsilon");
    check_pointer(input);
    check_pointer(gamma);
    check_pointer(output);
    const auto cuda_stream = strata::q_of(stream);
    const dpct::dim3 grid{unsigned(n_rows), unsigned(n_tok), 1u};
    if (n_cols < 1024)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cuda_stream->parallel_for<dpct_kernel_name<
            class weighted_rms_norm_60b509, dpct_kernel_scalar<256>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                weighted_rms_norm<256>(input, gamma, output, n_cols, epsilon);
            });
    } else
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed
    the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cuda_stream->parallel_for<dpct_kernel_name<
            class weighted_rms_norm_9382c8, dpct_kernel_scalar<1024>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, 1024),
                              sycl::range(1, 1, 1024)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                weighted_rms_norm<1024>(input, gamma, output, n_cols, epsilon);
            });
    }
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
        throw std::runtime_error(std::string("native GR RMSNorm launch: ") +
                                 dpct::get_error_string_dummy(error));
}

} // namespace strata::kernels
