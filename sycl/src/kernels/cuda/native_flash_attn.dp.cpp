// Specialized from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// ggml/src/ggml-cuda/{fattn-vec.cuh,fattn-common.cuh,common.cuh}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
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
#include "strata/kernels/native_flash_attn.hpp"
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <cmath>

namespace strata::kernels {
namespace {
template <int Width> __dpct_inline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = Width / 2; offset; offset >>= 1)
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        x += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            offset, Width);
    return x;
}
__dpct_inline__ float warp_max(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        x = sycl::fmax(x,
                       dpct::experimental::permute_sub_group_by_xor(
                           0xffffffffu,
                           sycl::ext::oneapi::this_work_item::get_sub_group(),
                           x, offset));
    return x;
}

// D=256,ncols=1,F16/F16; 128 threads, nthreads_KQ=nthreads_V=8,
// four values (float2) per load, four V columns per iteration. Padded length256
// gives ntiles_KV=ceil(256/D)=1, so the pinned launcher selects grid.y=1.
/*
DPCT1110: The total declared local variable size in device function attend
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attend(const float *__restrict__ q, const sycl::half *__restrict__ k,
       const sycl::half *__restrict__ v, const int32_t *__restrict__ step,
       int max_context, int padded_length, float scale, float *__restrict__ out,
       int32_t *__restrict__ status, const sycl::half *__restrict__ mask) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2), warp = item_ct1.get_local_id(1),
              tid = warp * 32 + lane;
    const int head = item_ct1.get_group(2), kv = head / 12;
    const int width = step[kStepWidth], nkv = step[kStepNKv];
    const bool valid = width >= 1 && width <= max_context && nkv == width &&
                       step[kStepPos] == width - 1 && step[kStepNBid] == width / 4;
    if (head == 0 && tid == 0)
        *status = valid ? kNativeFlashAttnSuccess : kNativeFlashAttnUnsupportedStep;
    if (!valid) {
        out[head * 256 + tid] = sycl::bit_cast<float>(0x7fc00000);
        out[head * 256 + tid + 128] = sycl::bit_cast<float>(0x7fc00000);
        return;
    }
    sycl::float2 qreg[16];
    sycl::float2 vkq[16] = {};
    auto &tile = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[4 * 4 * 256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &max_shared =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sum_shared =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float maximum = -FLT_MAX / 2.0f, sum = 0.0f;
#pragma unroll
    for (int i0 = 0; i0 < 128; i0 += 32) {
        const int i = i0 + (lane % 8) * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = 2 * (i + j);
            qreg[i0 / 8 + j] = sycl::float2(q[head * 256 + d] * scale,
                                            q[head * 256 + d + 1] * scale);
        }
    }
    for (int base = 0; base < padded_length; base += 128) {
        float score = 0.0f, next_max = maximum;
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int cell = base + warp * 32 + (lane & ~7) + row;
            float dot = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < 128; i0 += 32) {
                const int i = i0 + (lane % 8) * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = 2 * (i + j);
                    const float a =
                        cell < width
                            ? sycl::vec<sycl::half, 1>(
                                  k[(cell * 2 + kv) * 256 + d])
                                  .convert<float,
                                           sycl::rounding_mode::automatic>()[0]
                            : 0.0f;
                    const float b =
                        cell < width
                            ? sycl::vec<sycl::half, 1>(
                                  k[(cell * 2 + kv) * 256 + d + 1])
                                  .convert<float,
                                           sycl::rounding_mode::automatic>()[0]
                            : 0.0f;
                    dot += a * qreg[i0 / 8 + j].x();
                    dot += b * qreg[i0 / 8 + j].y();
                }
            }
            dot = warp_sum<8>(dot);
            dot +=
                cell < width
                    ? (mask ? sycl::vec<sycl::half, 1>(mask[cell])
                                  .convert<float,
                                           sycl::rounding_mode::automatic>()[0]
                            : 0.0f)
                    : sycl::bit_cast<float, int>(0xff800000u);
            next_max = sycl::fmax(next_max, dot + (3.0f * 0.6931f));
            if (lane % 8 == row) score = dot;
        }
#pragma unroll
        for (int offset = 8; offset < 32; offset <<= 1)
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            next_max = sycl::fmax(
                next_max,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    next_max, offset));
        const float rescale = sycl::native::exp(maximum - next_max);
        maximum = next_max;
        score = sycl::native::exp(score - maximum);
        sum = sum * rescale + score;
        tile[tid] = score;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
        for (int k0 = 0; k0 < 32; k0 += 4) {
            const int local = warp * 32 + k0 + lane / 8, cell = base + local;
            const float weight = tile[local];
#pragma unroll
            for (int i0 = 0; i0 < 128; i0 += 32) {
                const int i = i0 + (lane % 8) * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = 2 * (i + j);
                    const float a =
                        cell < width
                            ? sycl::vec<sycl::half, 1>(
                                  v[(cell * 2 + kv) * 256 + d])
                                  .convert<float,
                                           sycl::rounding_mode::automatic>()[0]
                            : 0.0f;
                    const float b =
                        cell < width
                            ? sycl::vec<sycl::half, 1>(
                                  v[(cell * 2 + kv) * 256 + d + 1])
                                  .convert<float,
                                           sycl::rounding_mode::automatic>()[0]
                            : 0.0f;
                    // The pinned sm120a binary contracts the old-accumulator
                    // rescale with the FIRST addition: fma(rescale,old,round(V*w)).
                    // Later columns use fma(V,w,acc). Explicit intrinsics retain
                    // that order despite this adapter's masked-load branches.
                    if (k0 == 0) {
                        /*
                        DPCT1013: The rounding mode could not be specified
                        and the generated code may have different accuracy than
                        the original code. Verify the correctness. SYCL math
                        built-in function rounding mode is aligned with OpenCL
                        C 1.2 standard.
                        */
                        vkq[i0 / 8 + j].x() =
                            sycl::fma(rescale, vkq[i0 / 8 + j].x(), a * weight);
                        /*
                        DPCT1013: The rounding mode could not be specified
                        and the generated code may have different accuracy than
                        the original code. Verify the correctness. SYCL math
                        built-in function rounding mode is aligned with OpenCL
                        C 1.2 standard.
                        */
                        vkq[i0 / 8 + j].y() =
                            sycl::fma(rescale, vkq[i0 / 8 + j].y(), b * weight);
                    } else {
                        /*
                        DPCT1013: The rounding mode could not be specified
                        and the generated code may have different accuracy than
                        the original code. Verify the correctness. SYCL math
                        built-in function rounding mode is aligned with OpenCL
                        C 1.2 standard.
                        */
                        vkq[i0 / 8 + j].x() =
                            sycl::fma(a, weight, vkq[i0 / 8 + j].x());
                        /*
                        DPCT1013: The rounding mode could not be specified
                        and the generated code may have different accuracy than
                        the original code. Verify the correctness. SYCL math
                        built-in function rounding mode is aligned with OpenCL
                        C 1.2 standard.
                        */
                        vkq[i0 / 8 + j].y() =
                            sycl::fma(b, weight, vkq[i0 / 8 + j].y());
                    }
                }
            }
        }
    }
    if (warp == 0) { max_shared[lane] = -FLT_MAX / 2.0f; sum_shared[lane] = 0.0f; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (lane == 0) max_shared[warp] = maximum;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float global_max = warp_max(max_shared[lane]);
    const float rescale = sycl::native::exp(maximum - global_max);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        vkq[i].x() = vkq[i].x() * rescale;
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        vkq[i].y() = vkq[i].y() * rescale;
    }
#pragma unroll
    for (int i0 = 0; i0 < 128; i0 += 32) {
        const int start = warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tile[start + 2 * j] = vkq[i0 / 8 + j].x();
            tile[start + 2 * j + 1] = vkq[i0 / 8 + j].y();
        }
    }
    sum *= rescale;
    sum = warp_sum<32>(sum);
    if (lane == 0) sum_shared[warp] = sum;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    sum = warp_sum<32>(sum_shared[lane]);
#pragma unroll
    for (int i0 = 0; i0 < 256; i0 += 128) {
        float result = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int group = 0; group < 4; ++group)
                result += tile[w * 4 * 256 + group * 256 + i0 + tid];
        }
        out[head * 256 + i0 + tid] = result / sum;
    }
}
struct Span { const void* p; std::size_t n, alignment; };
void validate_spans(const Span* spans, int count) {
    for (int i = 0; i < count; ++i) {
        const auto a = reinterpret_cast<std::uintptr_t>(spans[i].p);
        if (!a || a % spans[i].alignment || spans[i].n > UINTPTR_MAX - a)
            throw std::invalid_argument("native FlashAttention requires nonnull aligned bounded spans");
        for (int j = 0; j < i; ++j) {
            const auto b = reinterpret_cast<std::uintptr_t>(spans[j].p);
            if (a < b + spans[j].n && b < a + spans[i].n)
                throw std::invalid_argument("native FlashAttention requires disjoint buffers");
        }
    }
}
} // namespace

void native_flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v,
                                  const int32_t* step, int64_t capacity, int max_context,
                                  const QsaShapes& shapes, float* output, int32_t* status,
                                  const uint16_t* mask, void* stream) {
    if (!stream || shapes.n_head != 24 || shapes.n_head_kv != 2 || shapes.head_dim != 256 ||
        shapes.idx_block != 4 || shapes.idx_top_k < 256 || capacity < 256 ||
        uint64_t(capacity) > std::numeric_limits<std::size_t>::max() / 1024 ||
        max_context < 1 || max_context > 256)
        throw std::invalid_argument("native FlashAttention supports only Q24x256/KV2x256, capacity>=256 and context1..256 on an explicit stream");
    const std::size_t kv_bytes = std::size_t(capacity) * 1024;
    const Span spans[] = {{q, 24 * 256 * 4, 4}, {k, kv_bytes, 2}, {v, kv_bytes, 2},
                          {step, kStepCount * 4, 4}, {output, 24 * 256 * 4, 4},
                          {status, 4, 4}, {mask, 256 * 2, 2}};
    validate_spans(spans, mask ? 7 : 6);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            ((sycl::queue *)(strata::q_of(stream)))
                ->get_device(),
            {sycl::aspect::fp16});

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class attend_9d5505>>(
                sycl::nd_range<3>(sycl::range(1, 1, 24) * sycl::range(1, 4, 32),
                                  sycl::range(1, 4, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attend(q, reinterpret_cast<const sycl::half *>(k),
                               reinterpret_cast<const sycl::half *>(v), step,
                               max_context, 256, 0.0625f, output, status,
                               reinterpret_cast<const sycl::half *>(mask));
                    });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto result = 0;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (result != 0)
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        throw std::runtime_error(std::string("native FlashAttention launch: ") +
                                 dpct::get_error_string_dummy(result));
}
} // namespace strata::kernels
