// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
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
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_math.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "s26_tsum.dp.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <cstdio>
#include <vector>
#include <cmath>

namespace strata::kernels {
namespace {

#ifndef STRATA_MMVQ_UNROLL
#define STRATA_MMVQ_UNROLL 2
#endif
constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    sycl::half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    sycl::half d;
};
struct IQ4XSBlock {
    sycl::half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    sycl::half d;
};
struct Q40Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q50Block {
    sycl::half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    sycl::half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    sycl::half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__dpct_inline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        x += dpct::experimental::permute_sub_group_by_xor(
            0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            offset);
    }
    return x;
}

template <int SG> __dpct_inline__ float sg_sum(float x) {   // SYCL port: warp_sum for a SG-wide sub-group
#pragma unroll
    for (int offset = SG / 2; offset > 0; offset >>= 1)
        x += sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset);
    return x;
}
// SYCL port: the decode mmvq kernels as Xe2-native SIMD16 sub-groups (twice the registers per lane, same 128
// threads per group) or as the CUDA-shaped 32-lane warp (default). STRATA_MMVQ_SG: 16 = all SIMD16, 1 = IQ4_XS
// only, 2 = short outputs (n_out <= 1024) only, 3 = both. mmvq_sg_bench alone: SIMD16 wins on IQ4_XS (up to
// 1.45x) and short outputs (up to 1.3x), loses 10-25% on the large K-quant projections.
inline int mmvq_sg() {
    static const int v = std::getenv("STRATA_MMVQ_SG") ? std::atoi(std::getenv("STRATA_MMVQ_SG")) : 32;
    return v;
}
inline bool mmvq_sg16(bool iq4xs, int n_out) {
    const int v = mmvq_sg();
    return v == 16 || ((v & 1) && v < 4 && iq4xs) || ((v & 2) && v < 4 && n_out <= 1024);
}

__dpct_inline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        x = sycl::fmax(x,
                       dpct::experimental::permute_sub_group_by_xor(
                           0xffffffff,
                           sycl::ext::oneapi::this_work_item::get_sub_group(),
                           x, offset));
    }
    return x;
}

__dpct_inline__ void native_quantize_q8_1_kernel(const float *__restrict__ x,
                                                 Q81Block *__restrict__ y,
                                                 int n_in) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = int(item_ct1.get_group(2)) * QUANT_THREADS +
                  int(item_ct1.get_local_id(2));
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(sycl::fabs(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

__dpct_inline__ void
native_swiglu_quantize_q8_1_kernel(const float *__restrict__ gate,
                                   const float *__restrict__ up,
                                   Q81Block *__restrict__ y, int n_in) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = int(item_ct1.get_group(2)) * QUANT_THREADS +
                  int(item_ct1.get_local_id(2));
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float gi = gate[i];
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float xi = gi / (1.0f + sycl::native::exp(-gi)) * up[i];
    const float amax = warp_max(sycl::fabs(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606, as native_quantize_q8_1_kernel: finite blocks bit for bit
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__dpct_inline__ float
q5_q8_dot_impl(const int *__restrict__ vl, const int *__restrict__ vh,
               const int *__restrict__ u, const uint8_t *__restrict__ sc,
               const uint8_t *__restrict__ m, const sycl::half2 &dm5,
               const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = STRATA_DP4A(v0i, u[2 * i], STRATA_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i], STRATA_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm5f =
        dm5.template convert<float, sycl::rounding_mode::automatic>();
    return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}

/*
DPCT1110: The total declared local variable size in device function
q5_q8_dot exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ float q5_q8_dot(const Q5KBlock *__restrict__ bq5,
                                const Q81Block *__restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template <bool SmallK>
__dpct_inline__ void native_q5_k_mmvq_kernel(const Q5KBlock *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__dpct_inline__ float q2_q8_dot(const Q20Block *__restrict__ w,
                                const Q81Block *__restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 0);
        const int qo = dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 2);
        const int qx = dpct::byte_level_permute(qe, qo, 0x5140);
        const int qy = dpct::byte_level_permute(qe, qo, 0x7362);
        sumi = STRATA_DP4A(u, qx, sumi);
        sumi = STRATA_DP4A(v, qy, sumi);
    }
    const float d8 = chunk->ds[0];
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template <bool SmallK>
__dpct_inline__ void native_q2_0_mmvq_kernel(const Q20Block *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__dpct_inline__ int load_int_b2(const void *ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__dpct_inline__ float q3_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                                     const uint8_t *__restrict__ scales,
                                     int scale_offset, float d3,
                                     const float *__restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi =
            dpct::vectorized_binary<sycl::char4>(vil, vih, dpct::sub_sat());
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__dpct_inline__ float q3_q8_dot(const Q3KBlock *__restrict__ w,
                                const Q81Block *__restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = x[bq8_offset + i].ds[0];
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template <bool SmallK>
__dpct_inline__ void native_q3_k_mmvq_kernel(const Q3KBlock *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
inline dpct::global_memory<int8_t, 1>& iq4nl_values = *new dpct::global_memory<int8_t, 1>(sycl::range(16), {-127, -104, -83, -65, -49, -35, -22, -10, 1,
                                   13, 25, 38, 53, 69, 89, 113});

static constexpr int8_t kIq4nlTable[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__dpct_inline__ sycl::int2 iq4_table_lookup(int q4, int8_t *iq4nl_values) {
#if defined(STRATA_HIP_GFX906)
    // AMD: llama.cpp's HIP lookup (see iq_kernels.cu get_int_from_table_16): 4 v_perm_b32 per 8 values
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low =
            dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
        const uint32_t high =
            dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
        tmp[i] = dpct::byte_level_permute(low, high,
                                          low_high_selection_indices >> shift);
    }
    return sycl::int2(dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
                      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
#endif
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__dpct_inline__ float iq4_xs_q8_dot(const IQ4XSBlock *__restrict__ w,
                                    const Q81Block *__restrict__ x, int iqs,
                                    int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const sycl::int2 v = iq4_table_lookup(aux_q4, iq4nl_values);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = STRATA_DP4A(v.x(), u0, sumi);
        sumi = STRATA_DP4A(v.y(), u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = sycl::vec<sycl::half, 1>(w->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    x[iqs / 4].ds[0];
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template <bool SmallK>
__dpct_inline__ void
native_iq4_xs_mmvq_kernel(const IQ4XSBlock *__restrict__ w,
                          const Q81Block *__restrict__ x, float *__restrict__ y,
                          int n_in, int n_out, int8_t *iq4nl_values) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs, iq4nl_values);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__dpct_inline__ float
q4_q8_dot_impl(const int *__restrict__ v, const int *__restrict__ u,
               const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
               const sycl::half2 &dm4, const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm4f =
        dm4.template convert<float, sycl::rounding_mode::automatic>();
    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

__dpct_inline__ float q4_q8_dot(const Q4KBlock *__restrict__ bq4,
                                const Q81Block *__restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template <bool SmallK>
__dpct_inline__ void native_q4_k_mmvq_kernel(const Q4KBlock *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__dpct_inline__ float q6_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                                     const int8_t *__restrict__ scales, float d,
                                     const float *__restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = dpct::vectorized_binary<sycl::char4>(
            vil | vih, 0x20202020, dpct::sub_sat());
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__dpct_inline__ float q6_q8_dot(const Q6KBlock *__restrict__ w,
                                const Q81Block *__restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = x[bq8_offset + 2 * i].ds[0];
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template <bool SmallK>
__dpct_inline__ void native_q6_k_mmvq_kernel(const Q6KBlock *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__dpct_inline__ float small_q8_dot(const Q40Block *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds =
        (x->ds).template convert<float, sycl::rounding_mode::automatic>();
    const float d = w->d;
    return d * (sumi * ds.x() - 4 * ds.y());
}

__dpct_inline__ float small_q8_dot(const Q50Block *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds =
        (x->ds).template convert<float, sycl::rounding_mode::automatic>();
    const float d = w->d;
    return d * (sumi * ds.x() - 8 * ds.y());
}

__dpct_inline__ float small_q8_dot(const Q80Block *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = STRATA_DP4A(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = x->ds[0];
    return d0 * d1 * float(sumi);
}

__dpct_inline__ float small_q8_dot(const IQ4NLBlock *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const sycl::int2 v =
            iq4_table_lookup(load_int_b2(w->qs, iqs + i), iq4nl_values);
        sumi = STRATA_DP4A(v.x(), q8[i], sumi);
        sumi = STRATA_DP4A(v.y(), q8[i + 4], sumi);
    }
    const float d = sycl::vec<sycl::half, 1>(w->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    x->ds[0];
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template <typename Weight, int Qi, bool SmallK>
__dpct_inline__ void native_small_mmvq_kernel(const Weight *__restrict__ w,
                                              const Q81Block *__restrict__ x,
                                              float *__restrict__ y, int n_in,
                                              int n_out, int8_t *iq4nl_values) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs, iq4nl_values);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W {
        int vl[2], vh[2]; uint16_t aux[2]; sycl::half2 dm; int bq8_offset;
    };
    static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; sycl::half2 dm; int bq8_offset; };
    static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe =
                dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 0);
            const int qo =
                dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = dpct::byte_level_permute(qe, qo, 0x5140);
            r.qy[j] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = STRATA_DP4A(q8[j * 2], r.qx[j], sumi);
            sumi = STRATA_DP4A(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = chunk->ds[0];
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = x[r.bq8_offset + i].ds[0];
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    static int kqs(int tid) { return tid % 32; }
    // SYCL port: the weight unpack (shifts, masks, the saturating byte subtract) is column-independent, so it is
    // done once in load(); apply() is two dp4a and two fmas per column.
    struct W { int vi[2]; float dsc[2]; int bq8_offset; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vl = load_int_b2(w->ql, iqs);
        const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        const float d = w->d;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            r.vi[i] = dpct::vectorized_binary<sycl::char4>(vil | vih, 0x20202020, dpct::sub_sat());
            r.dsc[i] = d * (float) w->scales[scale_offset + 4 * i];
        }
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            sumf += x[r.bq8_offset + 2 * i].ds[0] * r.dsc[i] * (float) strata::dp4a(r.vi[i], u, 0);
        }
        return sumf;
    }
    // SYCL port, row-blocked kernel: the activation's part for this (block, lane), loaded once per column and
    // reused across the warp's rows.
    struct A { int u[2]; float ds[2]; };
    static A acts(const Q81Block* __restrict__ x, int iqs) {
        A a;
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            a.u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
            a.ds[i] = x[bq8_offset + 2 * i].ds[0];
        }
        return a;
    }
    static float dot(const W& r, const A& a) {
        return a.ds[0] * r.dsc[0] * (float) strata::dp4a(r.vi[0], a.u[0], 0) +
               a.ds[1] * r.dsc[1] * (float) strata::dp4a(r.vi[1], a.u[1], 0);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { sycl::int2 v[4]; int ls; float dw; };
    static W load(const Block* __restrict__ w, int iqs) { return load(w, iqs, const_cast<int8_t*>(kIq4nlTable)); }
    static W load(const Block* __restrict__ w, int iqs, int8_t *iq4nl_values) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(
            reinterpret_cast<const int *>(w->qs)[iqs + j], iq4nl_values);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = sycl::vec<sycl::half, 1>(w->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = STRATA_DP4A(r.v[j].x(), u0, sumi);
            sumi = STRATA_DP4A(r.v[j].y(), u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * x[iqs / 4].ds[0];
        return d * sumi;
    }
};
// The four 32-element formats: `load` decodes the two 32-bit weight chunks and block scale once per (row, block),
// and `apply` runs the exact same STRATA_DP4A and scale arithmetic per activation column as `small_q8_dot`.
template<typename Weight, int Qi>
struct SmallTraits;

template<>
struct SmallTraits<Q40Block, 4> {
    using Block = Q40Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = load_int_b2(w->qs, iqs + i);
            r.vi0[i] = (v >> 0) & 0x0f0f0f0f;
            r.vi1[i] = (v >> 4) & 0x0f0f0f0f;
        }
        r.d = w->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const sycl::float2 ds =
            (x->ds).template convert<float, sycl::rounding_mode::automatic>();
        return r.d * (sumi * ds.x() - 4 * ds.y());
    }
};

template<>
struct SmallTraits<Q50Block, 4> {
    using Block = Q50Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        const int qh = load_int_b2(w->qh, 0);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vl = load_int_b2(w->qs, iqs + i);
            const int vh = qh >> (4 * (iqs + i));
            int vi0 = (vl >> 0) & 0x0f0f0f0f;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            r.vi0[i] = vi0;
            int vi1 = (vl >> 4) & 0x0f0f0f0f;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            r.vi1[i] = vi1;
        }
        r.d = w->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const sycl::float2 ds =
            (x->ds).template convert<float, sycl::rounding_mode::automatic>();
        return r.d * (sumi * ds.x() - 8 * ds.y());
    }
};

template<>
struct SmallTraits<Q80Block, 8> {
    using Block = Q80Block;
    static constexpr int DIV = 32, T = 4, KBY = 1, BPI = 2 * WARPS * WARP / 8;
    static int kqs(int tid) { return 2 * (tid % 4); }
    struct W { int v[2]; float d0; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = w->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
            sumi = STRATA_DP4A(r.v[i], u, sumi);
        }
        const float d1 = x->ds[0];
        return r.d0 * d1 * float(sumi);
    }
};

template<>
struct SmallTraits<IQ4NLBlock, 4> {
    using Block = IQ4NLBlock;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { sycl::int2 v[2]; float dw; };
    static W load(const Block* __restrict__ w, int iqs) { return load(w, iqs, const_cast<int8_t*>(kIq4nlTable)); }
    static W load(const Block* __restrict__ w, int iqs, int8_t *iq4nl_values) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] =
            iq4_table_lookup(load_int_b2(w->qs, iqs + i), iq4nl_values);
        r.dw = sycl::vec<sycl::half, 1>(w->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.v[i].x(), q8[i], sumi);
            sumi = STRATA_DP4A(r.v[i].y(), q8[i + 4], sumi);
        }
        const float d = r.dw * x->ds[0];
        return d * sumi;
    }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)

// S26 STRATA_TSUM=1 (TS): warp 0's NCOLS x ROWS sums as one transposed butterfly (s26_tsum.cuh), bitwise the same
static bool s26_tsum_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    return on;
}

// S26 STRATA_LFUSE (PAIR): blocks n_out.. compute the same rows of w2 into y2 (two matrices of one shape on one
// input in one launch; each output's code is the single matrix's)
template <typename F, int NCOLS, int NW, int ROWS, bool TS = false,
          bool PAIR = false, int SG = WARP>
/*
DPCT1110: The total declared local variable size in device function
native_mmvq_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
native_mmvq_multi_kernel(const typename F::Block *__restrict__ w,
                         const Q81Block *__restrict__ x, float *__restrict__ y,
                         int n_in, int n_out,
                         const typename F::Block *__restrict__ w2 = nullptr,
                         float *__restrict__ y2 = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int BPI =
        F::BPI * NW * SG / (WARPS * WARP); // blocks per iteration scale with the thread count
    const int tid =
        SG * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    int bxi = int(item_ct1.get_group(2));
    if constexpr (PAIR) {
        if (bxi >= n_out / ROWS) { bxi -= n_out / ROWS; w = w2; y = y2; }
    }
    const int row0 = ROWS * bxi;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
#pragma unroll STRATA_MMVQ_UNROLL   // SYCL port: several 256-blocks' loads in flight per warp (bytes in flight, not ALU)
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                /*
                DPCT1084: The function call "Q5KTraits::load" has multiple
                migration results in different template instantiations that
                could not be unified. You may need to adjust the code.
                */
                const typename F::W wv =
                    F::load(w + block, kqs); // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][SG]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                partial[item_ct1.get_local_id(1) - 1][j][i]
                       [item_ct1.get_local_id(2)] = tmp[j][i];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
    if constexpr (TS) {
        constexpr int V = NCOLS * ROWS, P = s26ts::pow2_ceil(V);
        float v[P];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
#pragma unroll
                for (int l = 0; l < NW - 1; ++l) tmp[j][i] +=
                    partial[l][j][i][item_ct1.get_local_id(2)];
                v[j * ROWS + i] = tmp[j][i];
            }
#pragma unroll
        for (int k = V; k < P; ++k) v[k] = 0.0f;
        const float sum = s26ts::tsum<P>(v, int(item_ct1.get_local_id(2)));
        const int k = s26ts::tsum_token<P>(int(item_ct1.get_local_id(2)));
        const int j = k / ROWS, i = k % ROWS;
        if (int(item_ct1.get_local_id(2)) == s26ts::tsum_lane<P>(k) && k < V &&
            row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        return;
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] +=
                partial[l][j][i][item_ct1.get_local_id(2)];
            tmp[j][i] = sg_sum<SG>(tmp[j][i]);
            if (item_ct1.get_local_id(2) == i && row0 + i < n_out)
                y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// ---------------------------------------------------------------- SYCL port: row-blocked mmvq
//
// Measured on the B70 (mmvq_bench): the shared kernel above spends its time re-reading the activation - every
// row's work-group loads all NCOLS columns of it again, and a 4-warp work-group per row ends in a barrier and an
// SLM reduction for ten blocks of work. Here each warp owns RPW rows outright: per 256-block it loads the
// activation parts once for all columns and the RPW weight parts once, does RPW x NCOLS dots, and reduces its
// own lanes at the end. No barrier, no SLM. Only for traits with acts()/dot() (Q6_K so far).
template <typename F, int NCOLS, int RPW>
void native_mmvq_rowwarp_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row0 = (int(item.get_group(2)) * WARPS + warp) * RPW;
    const int blocks_per_row = n_in / F::DIV, x_stride = n_in / Q8K;
    const int kqs = F::kqs(lane);
    float tmp[RPW][NCOLS] = {};
    for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += WARP / F::T) {
        const int kby = kbx * F::KBY;
        typename F::A a[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) a[j] = F::acts(x + std::size_t(j) * x_stride + kby, kqs);
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (row0 + r >= n_out) break;
            const typename F::W wv = F::load(w + std::size_t(row0 + r) * blocks_per_row + kbx, kqs);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) tmp[r][j] += F::dot(wv, a[j]);
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = tmp[r][j];
#pragma unroll
            for (int o = WARP / 2; o > 0; o >>= 1) v += dpct::experimental::permute_sub_group_by_xor(0xffffffffu, sg, v, o);
            if (lane == 0 && row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = v;
        }
}

// ---------------------------------------------------------------- SYCL port: Q6_K with 16-byte loads
//
// bw probe on the B70: 4-byte loads per lane stream at 395 GB/s, 16-byte at 596; the shared kernel above does
// 4-byte loads and measures 130-140 GB/s on the 8192-wide Q6_K projections. Here 8 lanes share a 256-block:
// lane g takes positions iqs = 4g..4g+3, so its ql (16 B), qh (16 B) and each column's q8 ints (16 B) are one
// vector load each; a 32-lane sub-group has four blocks in flight. One row per warp, WARPS rows per group.
// SYCL port: a 16-byte load from a 2-byte aligned address as two aligned 16-byte loads and a shift. A Q6_K block
// is 210 bytes, so its ql/qh runs start 2-byte aligned, and the B70 splits a misaligned vector load into pieces:
// the wide kernel streamed 145 GB/s that way against 640 GB/s for the same blocks repacked at a 16-byte stride.
// Both aligned loads stay inside the 210-byte block for the ql and qh runs (last byte read <= block + 207).
// SYCL port: a 4-int vector load from a 4-byte aligned address (a Q8_1 block is 36 bytes, its qs sit at offset 4): one dword-vector
// message instead of four scalar loads - the decode kernels issue 8 of these per column and 4-byte step.
__dpct_inline__ sycl::int4 ld_i4_a4(const int* p) {
    sycl::int4 v;
    v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p));
    return v;
}
// Q6_K: per-byte (v - 32) of four 6-bit values (bytes 0..63): no byte can saturate, so this is the saturating char4 subtract's
// result exactly, in four integer ops (SWAR: no borrow crosses a byte) instead of the per-byte vector builtin.
__dpct_inline__ int q6_sub32(int x) {
    const uint32_t u = (uint32_t) x;
    return (int) (((u | 0x80808080u) - 0x20202020u) ^ (~u & 0x80808080u));
}
__dpct_inline__ sycl::int4 load16_a2(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::int4* q = reinterpret_cast<const sycl::int4*>(a & ~uintptr_t(15));
    const int k = int(a >> 2) & 3;          // dword offset within the first aligned chunk
    const bool half = (a & 2) != 0;          // and a 2-byte shift (the address is always 2-byte aligned)
    // the second chunk only when the address is unaligned: then it holds needed bytes, so (16-byte chunks never
    // straddle a page) it cannot fault even at the very end of an allocation
    const sycl::int4 lo = q[0], hi = (a & 15) ? q[1] : lo;
    const uint32_t d[8] = {(uint32_t) lo.x(), (uint32_t) lo.y(), (uint32_t) lo.z(), (uint32_t) lo.w(),
                           (uint32_t) hi.x(), (uint32_t) hi.y(), (uint32_t) hi.z(), (uint32_t) hi.w()};
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i)              // w = d[k .. k+4] by selects (no indexed register access)
        w[i] = k == 0 ? d[i] : k == 1 ? d[i + 1] : k == 2 ? d[i + 2] : d[(i + 3) & 7];
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (w[i] >> 16) | (w[i + 1] << 16) : w[i];
    return sycl::int4((int) o[0], (int) o[1], (int) o[2], (int) o[3]);
}
inline bool q6k_a2() {   // STRATA_MMVQ_A2=0 turns the aligned-load path off
    static const bool v = std::getenv("STRATA_MMVQ_A2") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_A2")) != 0;
    return v;
}

template <int NCOLS, int RPW, int SG = WARP, int BS = (int) sizeof(Q6KBlock), bool A2 = false>   // BS: block stride
void native_mmvq_q6k_wide_kernel(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x,
                                 float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    constexpr int NWG = WARPS * WARP / SG;              // sub-groups per work-group
    const int row0 = (int(item.get_group(2)) * NWG + warp) * RPW;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;            // position group within the block, block within the iteration
    const int vh_shift = 2 * ((g & 3) >> 1);
    const int qh4_idx = 2 * (g >> 2) + (g & 1);
    const int scale_offset = 8 * (g >> 2) + (g & 3);
    const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
    const int u_int4 = g & 1;                           // ints 4(g%2)..+3 of the q8 block = one int4
    float acc[RPW][NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += SG / 8) {
        const int kby = kbx * (QK / Q8K);
        sycl::int4 ql4[RPW], qh4[RPW]; float dsc0[RPW], dsc1[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const Q6KBlock* b = reinterpret_cast<const Q6KBlock*>(reinterpret_cast<const uint8_t*>(w) +
                (std::size_t(row0 + (row0 + r < n_out ? r : 0)) * blocks_per_row + kbx) * BS);
            if constexpr (A2) {
                ql4[r] = load16_a2(b->ql + 16 * g);
                qh4[r] = load16_a2(b->qh + 16 * qh4_idx);
            } else {
                ql4[r] = reinterpret_cast<const sycl::int4*>(b->ql)[g];
                qh4[r] = reinterpret_cast<const sycl::int4*>(b->qh)[qh4_idx];
            }
            const float d = b->d;
            dsc0[r] = d * (float) b->scales[scale_offset]; dsc1[r] = d * (float) b->scales[scale_offset + 4];
        }
        // the activations once per column, shared by the RPW rows
        sycl::int4 u0[NCOLS], u1[NCOLS];
        float ds0[NCOLS], ds1[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * x_stride + kby + bq8_offset;
            if constexpr (A2) {   // Q8_1 qs sits at offset 4 of a 36-byte block: 4-byte aligned, so four int loads
                u0[j] = ld_i4_a4(reinterpret_cast<const int*>(xb[0].qs) + 4 * u_int4);
                u1[j] = ld_i4_a4(reinterpret_cast<const int*>(xb[2].qs) + 4 * u_int4);
            } else {
                u0[j] = reinterpret_cast<const sycl::int4*>(xb[0].qs)[u_int4];
                u1[j] = reinterpret_cast<const sycl::int4*>(xb[2].qs)[u_int4];
            }
            ds0[j] = xb[0].ds[0]; ds1[j] = xb[2].ds[0];
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int vl[4] = {ql4[r].x(), ql4[r].y(), ql4[r].z(), ql4[r].w()};
            const int vh[4] = {qh4[r].x() >> vh_shift, qh4[r].y() >> vh_shift, qh4[r].z() >> vh_shift, qh4[r].w() >> vh_shift};
            int vi0[4], vi1[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                vi0[p] = q6_sub32((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030));
                vi1[p] = q6_sub32(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030));
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                int s0 = 0, s1 = 0;
                s0 = strata::dp4a(vi0[0], u0[j].x(), s0); s0 = strata::dp4a(vi0[1], u0[j].y(), s0);
                s0 = strata::dp4a(vi0[2], u0[j].z(), s0); s0 = strata::dp4a(vi0[3], u0[j].w(), s0);
                s1 = strata::dp4a(vi1[0], u1[j].x(), s1); s1 = strata::dp4a(vi1[1], u1[j].y(), s1);
                s1 = strata::dp4a(vi1[2], u1[j].z(), s1); s1 = strata::dp4a(vi1[3], u1[j].w(), s1);
                acc[r][j] += ds0[j] * dsc0[r] * (float) s0 + ds1[j] * dsc1[r] * (float) s1;
            }
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[r][j];
#pragma unroll
            for (int o = SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
            if (lane == 0 && row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = v;
        }
}
// SYCL port: the wide Q6_K kernel with its weight loads issued U steps ahead. The wide kernel's 2560-wide rows
// are ~3 steps of load -> wait -> dot per sub-group, so each sub-group has one 16-byte load per lane in flight
// at a time and the kernel is latency-bound (~146 GB/s). Here a sub-group first issues the ql/qh/scale loads
// of U steps (4U blocks), then does their dots; the activation reads stay inline (they are L1 hits, shared by
// every row). One row per sub-group, WARPS rows per group.
template <int NCOLS, int U, bool A2 = false>
void native_mmvq_q6k_pf_kernel(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x,
                               float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const int vh_shift = 2 * ((g & 3) >> 1);
    const int qh4_idx = 2 * (g >> 2) + (g & 1);
    const int scale_offset = 8 * (g >> 2) + (g & 3);
    const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
    const int u_int4 = g & 1;
    const Q6KBlock* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kb0 = 0; kb0 < blocks_per_row; kb0 += 4 * U) {
        sycl::int4 ql4[U], qh4[U]; float dsc0[U], dsc1[U]; bool ok[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {      // all U steps' weight loads first
            const int kbx = kb0 + 4 * u + sub;
            ok[u] = kbx < blocks_per_row;
            const Q6KBlock* b = wr + (ok[u] ? kbx : 0);
            if constexpr (A2) { ql4[u] = load16_a2(b->ql + 16 * g); qh4[u] = load16_a2(b->qh + 16 * qh4_idx); }
            else { ql4[u] = reinterpret_cast<const sycl::int4*>(b->ql)[g];
            qh4[u] = reinterpret_cast<const sycl::int4*>(b->qh)[qh4_idx]; }
            const float d = b->d;
            dsc0[u] = d * (float) b->scales[scale_offset]; dsc1[u] = d * (float) b->scales[scale_offset + 4];
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (!ok[u]) continue;
            const int kby = (kb0 + 4 * u + sub) * (QK / Q8K);
            const int vl[4] = {ql4[u].x(), ql4[u].y(), ql4[u].z(), ql4[u].w()};
            const int vh[4] = {qh4[u].x() >> vh_shift, qh4[u].y() >> vh_shift, qh4[u].z() >> vh_shift, qh4[u].w() >> vh_shift};
            int vi0[4], vi1[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                vi0[p] = q6_sub32((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030));
                vi1[p] = q6_sub32(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030));
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kby + bq8_offset;
                sycl::int4 a0, a1;
                if constexpr (A2) {   // Q8_1 qs at offset 4 of a 36-byte block: four int loads (as the wide kernel)
                    const int* p0 = reinterpret_cast<const int*>(xb[0].qs) + 4 * u_int4;
                    const int* p1 = reinterpret_cast<const int*>(xb[2].qs) + 4 * u_int4;
                    a0 = sycl::int4(p0[0], p0[1], p0[2], p0[3]);
                    a1 = sycl::int4(p1[0], p1[1], p1[2], p1[3]);
                } else {
                    a0 = reinterpret_cast<const sycl::int4*>(xb[0].qs)[u_int4];
                    a1 = reinterpret_cast<const sycl::int4*>(xb[2].qs)[u_int4];
                }
                int s0 = 0, s1 = 0;
                s0 = strata::dp4a(vi0[0], a0.x(), s0); s0 = strata::dp4a(vi0[1], a0.y(), s0);
                s0 = strata::dp4a(vi0[2], a0.z(), s0); s0 = strata::dp4a(vi0[3], a0.w(), s0);
                s1 = strata::dp4a(vi1[0], a1.x(), s1); s1 = strata::dp4a(vi1[1], a1.y(), s1);
                s1 = strata::dp4a(vi1[2], a1.z(), s1); s1 = strata::dp4a(vi1[3], a1.w(), s1);
                acc[j] += (float) xb[0].ds[0] * dsc0[u] * (float) s0 + (float) xb[2].ds[0] * dsc1[u] * (float) s1;
            }
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}
inline int q6k_pf() {   // STRATA_MMVQ_PF: weight-load steps issued ahead in the wide Q6_K kernel (0 = off)
    static const int v = std::getenv("STRATA_MMVQ_PF") ? std::atoi(std::getenv("STRATA_MMVQ_PF")) : 0;
    return v;
}
inline int q6k_wide_rpw() {
    static const int v = std::getenv("STRATA_MMVQ_WIDE_RPW") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_RPW")) : 1;
    return v;
}
// Large-GRF mode (256 registers per thread instead of 128) for the wide kernels (Q6_K a2 and the generic one) at 2-8
// columns: on the A770 they spill at 128, and mmvq_bench Q6_K 4 columns goes 70-80 -> 150-237 GB/s with the same
// checksum. Large GRF halves the threads an Xe core keeps resident, so it is the default on Arc A-series (Alchemist)
// only, where it was measured; Battlemage and anything else, and 1 column everywhere, launch the same kernels as before.
// STRATA_MMVQ_LARGE_GRF=1 forces it on, =0 off.
inline bool mmvq_large_grf(dpct::queue_ptr s) {
    static const bool v = [s] {
        const char* e = std::getenv("STRATA_MMVQ_LARGE_GRF");
        if (e != nullptr) return std::atoi(e) != 0;
        return strata::intel_gpu_gen(s->get_device()) == strata::IntelGpuGen::Alchemist;
    }();
    return v;
}
template <int NCOLS>
void launch_q6k_wide(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const int rpw = q6k_wide_rpw();
    if (const int pf = q6k_pf(); pf > 0) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        const sycl::range<3> t(1, WARPS, WARP);
#define STRATA_Q6PF(UU) if (pf == UU && q6k_a2()) { \
            s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_pf_a2, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<UU>>>( \
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * t, t), \
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_q6k_pf_kernel<NCOLS, UU, true>(w, x, y, n_in, n_out); }); \
            return; } \
        if (pf == UU) { \
            s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_pf, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<UU>>>( \
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * t, t), \
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_q6k_pf_kernel<NCOLS, UU>(w, x, y, n_in, n_out); }); \
            return; }
        STRATA_Q6PF(2) STRATA_Q6PF(3) STRATA_Q6PF(4) STRATA_Q6PF(6)
#undef STRATA_Q6PF
    }
    if (rpw == 1 && q6k_a2()) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        if constexpr (NCOLS > 1) {
            if (mmvq_large_grf(s)) {
                s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_wide_a2_grf256, dpct_kernel_scalar<NCOLS>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                    sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
                    [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                        native_mmvq_q6k_wide_kernel<NCOLS, 1, WARP, (int) sizeof(Q6KBlock), true>(w, x, y, n_in, n_out);
                    });
                return;
            }
        }
        s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_wide_a2, dpct_kernel_scalar<NCOLS>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                native_mmvq_q6k_wide_kernel<NCOLS, 1, WARP, (int) sizeof(Q6KBlock), true>(w, x, y, n_in, n_out);
            });
        return;
    }
    if (rpw == 1 && mmvq_sg16(false, n_out)) {
        constexpr int NWG = WARPS * WARP / 16;
        const unsigned blocks = unsigned((std::size_t(n_out) + NWG - 1) / NWG);
        s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_wide16, dpct_kernel_scalar<NCOLS>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, NWG, 16), sycl::range(1, NWG, 16)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] { native_mmvq_q6k_wide_kernel<NCOLS, 1, 16>(w, x, y, n_in, n_out); });
        return;
    }
    // RP rows per sub-group share each column's activation loads (the rows' own lanes, k order and butterfly: every
    // output the same bits); with the aligned loads (A2) unless STRATA_MMVQ_A2=0
#define STRATA_Q6W(RP) if (rpw == RP) { \
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS * RP - 1) / (WARPS * RP)); \
        if (q6k_a2()) { \
            s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_wide_a2r, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<RP>>>( \
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)), \
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { \
                    native_mmvq_q6k_wide_kernel<NCOLS, RP, WARP, (int) sizeof(Q6KBlock), true>(w, x, y, n_in, n_out); }); \
            return; } \
        s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_wide, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<RP>>>( \
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)), \
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_q6k_wide_kernel<NCOLS, RP>(w, x, y, n_in, n_out); }); \
        return; }
    STRATA_Q6W(1) STRATA_Q6W(2) STRATA_Q6W(4)
#undef STRATA_Q6W
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_wide1, dpct_kernel_scalar<NCOLS>>>(
        sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_q6k_wide_kernel<NCOLS, 1>(w, x, y, n_in, n_out); });
}
// ---------------------------------------------------------------- SYCL port: wide kernels for Q4_K, Q5_K, IQ4_XS
//
// The Q6_K wide kernel's shape for the other 256-element formats: 8 lanes share a block, each lane's weights are one
// or two 16-byte loads, a 32-lane sub-group has four blocks in flight, one row per sub-group. The multi kernel these
// replace does 4-byte loads (Q4_K/Q5_K, 230-280 GB/s) or global-memory codebook lookups (IQ4_XS, ~100 GB/s).
// Lane g of a block: Q4_K/Q5_K take qs bytes [16g, 16g+16) = sub-blocks 2(g/2) (low nibbles) and 2(g/2)+1 (high
// nibbles) at positions 16(g%2)..+15; IQ4_XS takes sub-block g (16 bytes, low nibbles = positions 0-15).
__dpct_inline__ sycl::int4 ld_q8_16(const Q81Block* b, int half) {   // Q8_1 qs is 4-byte aligned: one dword-vector load
    return ld_i4_a4(reinterpret_cast<const int*>(b->qs) + 4 * half);
}
__dpct_inline__ int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = strata::dp4a(a.x(), b.x(), acc); acc = strata::dp4a(a.y(), b.y(), acc);
    acc = strata::dp4a(a.z(), b.z(), acc); return strata::dp4a(a.w(), b.w(), acc);
}
__dpct_inline__ void k4_scale_min(const uint8_t* sc, int j, float& s, float& m) {   // ggml get_scale_min_k4
    if (j < 4) { s = (float) (sc[j] & 63); m = (float) (sc[j + 4] & 63); }
    else { s = (float) ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)); m = (float) ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)); }
}
struct WideQ4K {
    using Block = Q4KBlock;
    struct W { sycl::int4 lo, hi; float dl, dh, ml, mh; int sb, half; };
    static W load(const Block* b, int g) {
        W r; r.sb = 2 * (g >> 1); r.half = g & 1;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];          // 16-byte aligned (144 = 9 x 16)
        r.lo = sycl::int4(q.x() & 0x0f0f0f0f, q.y() & 0x0f0f0f0f, q.z() & 0x0f0f0f0f, q.w() & 0x0f0f0f0f);
        r.hi = sycl::int4((q.x() >> 4) & 0x0f0f0f0f, (q.y() >> 4) & 0x0f0f0f0f, (q.z() >> 4) & 0x0f0f0f0f, (q.w() >> 4) & 0x0f0f0f0f);
        const sycl::float2 dm = b->dm.convert<float, sycl::rounding_mode::automatic>();
        float s0, m0, s1, m1; k4_scale_min(b->scales, r.sb, s0, m0); k4_scale_min(b->scales, r.sb + 1, s1, m1);
        r.dl = dm.x() * s0; r.ml = dm.y() * m0; r.dh = dm.x() * s1; r.mh = dm.y() * m1;
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) {   // xb = this column's Q8_1 blocks of the 256-block
        const Q81Block* a = xb + r.sb; const Q81Block* c = xb + r.sb + 1;
        const sycl::int4 ua = ld_q8_16(a, r.half), uc = ld_q8_16(c, r.half);
        const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
        const float da = (float) a->ds[0], dc = (float) c->ds[0];
        return da * (r.dl * (float) dp4a4(r.lo, ua, 0) - r.ml * (float) dp4a4(ones, ua, 0)) +
               dc * (r.dh * (float) dp4a4(r.hi, uc, 0) - r.mh * (float) dp4a4(ones, uc, 0));
    }
    // apply() in two parts for several rows of one sub-group: the activations depend on the lane alone (sb, half from g)
    struct A { sycl::int4 ua, uc; float da, dc; };
    static A act(const Q81Block* xb, int g) {
        const int sb = 2 * (g >> 1), half = g & 1;
        const Q81Block* a = xb + sb; const Q81Block* c = xb + sb + 1;
        A v; v.ua = ld_q8_16(a, half); v.uc = ld_q8_16(c, half); v.da = (float) a->ds[0]; v.dc = (float) c->ds[0];
        return v;
    }
    static float dot(const W& r, const A& v) {   // apply's expression, the same operations in the same order
        const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
        return v.da * (r.dl * (float) dp4a4(r.lo, v.ua, 0) - r.ml * (float) dp4a4(ones, v.ua, 0)) +
               v.dc * (r.dh * (float) dp4a4(r.hi, v.uc, 0) - r.mh * (float) dp4a4(ones, v.uc, 0));
    }
};
struct WideQ5K {
    using Block = Q5KBlock;
    using W = WideQ4K::W;
    static W load(const Block* b, int g) {
        W r; r.sb = 2 * (g >> 1); r.half = g & 1;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];          // qs at 48, 176 = 11 x 16
        const sycl::int4 h = reinterpret_cast<const sycl::int4*>(b->qh)[r.half];     // qh bytes of positions 16h..+15
        const int sl = r.sb, sh = r.sb + 1;
        auto hb = [](int v, int bit) { return ((v >> bit) & 0x01010101) << 4; };
        r.lo = sycl::int4((q.x() & 0x0f0f0f0f) | hb(h.x(), sl), (q.y() & 0x0f0f0f0f) | hb(h.y(), sl),
                          (q.z() & 0x0f0f0f0f) | hb(h.z(), sl), (q.w() & 0x0f0f0f0f) | hb(h.w(), sl));
        r.hi = sycl::int4(((q.x() >> 4) & 0x0f0f0f0f) | hb(h.x(), sh), ((q.y() >> 4) & 0x0f0f0f0f) | hb(h.y(), sh),
                          ((q.z() >> 4) & 0x0f0f0f0f) | hb(h.z(), sh), ((q.w() >> 4) & 0x0f0f0f0f) | hb(h.w(), sh));
        const sycl::float2 dm = b->dm.convert<float, sycl::rounding_mode::automatic>();
        float s0, m0, s1, m1; k4_scale_min(b->scales, r.sb, s0, m0); k4_scale_min(b->scales, r.sb + 1, s1, m1);
        r.dl = dm.x() * s0; r.ml = dm.y() * m0; r.dh = dm.x() * s1; r.mh = dm.y() * m1;
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) { return WideQ4K::apply(r, xb); }
    using A = WideQ4K::A;
    static A act(const Q81Block* xb, int g) { return WideQ4K::act(xb, g); }
    static float dot(const W& r, const A& v) { return WideQ4K::dot(r, v); }
};
__dpct_inline__ uint32_t iq4_lut4(uint32_t q4) {   // four nibbles -> four codebook bytes, codebook in registers
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t n = (q4 >> (8 * i)) & 0xF;
        const uint32_t t = n < 4 ? t0 : n < 8 ? t1 : n < 12 ? t2 : t3;
        r |= ((t >> (8 * (n & 3))) & 0xFF) << (8 * i);
    }
    return r;
}
struct WideIQ4XS {
    using Block = IQ4XSBlock;
    struct W { sycl::int4 lo, hi; float d; int sb; };
    static W load(const Block* b, int g) {
        W r; r.sb = g;
        const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b->qs + 16 * g);   // 8-byte aligned (136 = 17 x 8)
        const sycl::int2 qa = q2[0], qb = q2[1];
        const uint32_t v[4] = {(uint32_t) qa.x(), (uint32_t) qa.y(), (uint32_t) qb.x(), (uint32_t) qb.y()};
        int lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { lo[i] = (int) iq4_lut4(v[i] & 0x0f0f0f0f); hi[i] = (int) iq4_lut4((v[i] >> 4) & 0x0f0f0f0f); }
        r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
        const int ls = ((b->scales_l[g / 2] >> (4 * (g & 1))) & 0x0f) | (((b->scales_h >> (2 * g)) & 0x03) << 4);
        r.d = (float) b->d * (float) (ls - 32);
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) {
        const Q81Block* a = xb + r.sb;
        return r.d * (float) a->ds[0] * (float) dp4a4(r.hi, ld_q8_16(a, 1), dp4a4(r.lo, ld_q8_16(a, 0), 0));
    }
    struct A { sycl::int4 u0, u1; float ds; };
    static A act(const Q81Block* xb, int g) {
        const Q81Block* a = xb + g;
        A v; v.u0 = ld_q8_16(a, 0); v.u1 = ld_q8_16(a, 1); v.ds = (float) a->ds[0];
        return v;
    }
    static float dot(const W& r, const A& v) { return r.d * v.ds * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};
// SYCL port: native_mmvq_wide_kernel with RPW rows per sub-group (STRATA_MMVQ_WIDE_K_RPW): each column's activations
// are loaded once (act) and used by the RPW rows (dot); a row's lanes, its k order and its butterfly are the
// one-row kernel's, so every output is the same bits. At 2-4 columns the one-row kernel reads ~5x its weight bytes
// in activations (each row re-reads the whole vector per column).
template <typename F, int NCOLS, int RPW>
void native_mmvq_wide_rpw_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                 float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row0 = (int(item.get_group(2)) * WARPS + warp) * RPW;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const typename F::Block* wr[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) wr[r] = w + std::size_t(row0 + (row0 + r < n_out ? r : 0)) * blocks_per_row;
    float acc[RPW][NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += 4) {
        typename F::W wv[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) wv[r] = F::load(wr[r] + kbx, g);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const typename F::A a = F::act(x + std::size_t(j) * x_stride + kbx * (QK / Q8K), g);
#pragma unroll
            for (int r = 0; r < RPW; ++r) acc[r][j] += F::dot(wv[r], a);
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[r][j];
#pragma unroll
            for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
            if (lane == 0 && row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = v;
        }
}
inline int wide_k_rpw() {   // STRATA_MMVQ_WIDE_K_RPW: rows per sub-group of the Q4_K / Q5_K / IQ4_XS wide kernel (1, 2, 4)
    static const int v = std::getenv("STRATA_MMVQ_WIDE_K_RPW") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_K_RPW")) : 1;
    return v;
}
template <typename F, int NCOLS>
void native_mmvq_wide_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                             float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const typename F::Block* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += 4) {
        const typename F::W wv = F::load(wr + kbx, g);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] += F::apply(wv, x + std::size_t(j) * x_stride + kbx * (QK / Q8K));
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}
template <typename F, int NCOLS>
void launch_wide(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
#define STRATA_WKR(RP) if (wide_k_rpw() == RP) { \
        const unsigned b = unsigned((std::size_t(n_out) + WARPS * RP - 1) / (WARPS * RP)); \
        s->parallel_for<dpct_kernel_name<class native_mmvq_wide_rpw, F, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<RP>>>( \
            sycl::nd_range<3>(sycl::range(1, 1, b) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)), \
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_wide_rpw_kernel<F, NCOLS, RP>(w, x, y, n_in, n_out); }); \
        return; }
    STRATA_WKR(2) STRATA_WKR(4)
#undef STRATA_WKR
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    if constexpr (NCOLS > 1) {
        if (mmvq_large_grf(s)) {
            s->parallel_for<dpct_kernel_name<class native_mmvq_wide_grf256, F, dpct_kernel_scalar<NCOLS>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_wide_kernel<F, NCOLS>(w, x, y, n_in, n_out); });
            return;
        }
    }
    s->parallel_for<dpct_kernel_name<class native_mmvq_wide, F, dpct_kernel_scalar<NCOLS>>>(
        sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_wide_kernel<F, NCOLS>(w, x, y, n_in, n_out); });
}
inline bool wide_k() {   // STRATA_MMVQ_WIDE_K=0: Q4_K/Q5_K/IQ4_XS through the multi kernel (the old path)
    static const bool v = std::getenv("STRATA_MMVQ_WIDE_K") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_WIDE_K")) != 0;
    return v;
}
template <typename F>
bool try_wide(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!wide_k() || ncols < 1 || ncols > 8 || n_in % QK != 0) return false;
    const auto s = strata::q_of(stream);
    switch (ncols) {
        case 1: launch_wide<F, 1>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 2: launch_wide<F, 2>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 3: launch_wide<F, 3>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 4: launch_wide<F, 4>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 5: launch_wide<F, 5>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 6: launch_wide<F, 6>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 7: launch_wide<F, 7>(weights, x_q8_1, y, n_in, n_out, s); return true;
        default: launch_wide<F, 8>(weights, x_q8_1, y, n_in, n_out, s); return true;
    }
}

// SYCL port: wide kernels for the 32-element formats Q8_0 (34-byte blocks) and IQ4_NL (18). Both are only 2-byte
// aligned, so the weights come through load16_a2 (page-safe). Q8_0: 2 lanes per block (16 quants each), 16 blocks
// per sub-group step; IQ4_NL: 1 lane per block (its 16 bytes = 32 nibbles), 32 blocks per step. One row per
// sub-group. The multi/small kernels these replace measured 26 GB/s (IQ4_NL 640 -> 2560) to ~300 GB/s.
struct Wide32Q8 {
    using Block = Q80Block;
    static constexpr int LPB = 2;
    struct W { sycl::int4 q; float d; int half; };
    static W load(const Block* b, int l) {
        W r; r.half = l; r.q = load16_a2(b->qs + 16 * l); r.d = (float) b->d; return r;
    }
    static float apply(const W& r, const Q81Block* xb) {
        return r.d * (float) xb->ds[0] * (float) dp4a4(r.q, ld_q8_16(xb, r.half), 0);
    }
};
struct Wide32IQ4NL {
    using Block = IQ4NLBlock;
    static constexpr int LPB = 1;
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int) {
        W r;
        const sycl::int4 q = load16_a2(b->qs);
        const uint32_t v[4] = {(uint32_t) q.x(), (uint32_t) q.y(), (uint32_t) q.z(), (uint32_t) q.w()};
        int lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { lo[i] = (int) iq4_lut4(v[i] & 0x0f0f0f0f); hi[i] = (int) iq4_lut4((v[i] >> 4) & 0x0f0f0f0f); }
        r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
        r.d = (float) b->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) {
        return r.d * (float) xb->ds[0] * (float) dp4a4(r.hi, ld_q8_16(xb, 1), dp4a4(r.lo, ld_q8_16(xb, 0), 0));
    }
};
template <typename F, int NCOLS>
void native_mmvq_wide32_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                               float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    constexpr int BPS = WARP / F::LPB;                  // blocks per sub-group step
    const int blocks_per_row = n_in / 32;
    const int l = lane % F::LPB, sub = lane / F::LPB;
    const typename F::Block* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kb = sub; kb < blocks_per_row; kb += BPS) {
        const typename F::W wv = F::load(wr + kb, l);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] += F::apply(wv, x + std::size_t(j) * blocks_per_row + kb);
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}
template <typename F, int NCOLS>
void launch_wide32(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    s->parallel_for<dpct_kernel_name<class native_mmvq_wide32, F, dpct_kernel_scalar<NCOLS>>>(
        sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_wide32_kernel<F, NCOLS>(w, x, y, n_in, n_out); });
}
inline bool wide_32() {   // STRATA_MMVQ_WIDE_32=0: Q8_0/IQ4_NL through the small/multi kernels (the old path)
    static const bool v = std::getenv("STRATA_MMVQ_WIDE_32") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_WIDE_32")) != 0;
    return v;
}
template <typename F>
bool try_wide32(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!wide_32() || ncols < 1 || ncols > 8 || n_in % 32 != 0) return false;
    const auto s = strata::q_of(stream);
    switch (ncols) {
        case 1: launch_wide32<F, 1>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 2: launch_wide32<F, 2>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 3: launch_wide32<F, 3>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 4: launch_wide32<F, 4>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 5: launch_wide32<F, 5>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 6: launch_wide32<F, 6>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 7: launch_wide32<F, 7>(weights, x_q8_1, y, n_in, n_out, s); return true;
        default: launch_wide32<F, 8>(weights, x_q8_1, y, n_in, n_out, s); return true;
    }
}

// SYCL port: alignment experiment - Q6_K blocks repacked at a 224-byte stride (every block 16-byte aligned).
template <int NCOLS>
void launch_q6k_wide224(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_wide224, dpct_kernel_scalar<NCOLS>>>(
        sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_q6k_wide_kernel<NCOLS, 1, WARP, 224>(w, x, y, n_in, n_out); });
}
bool g_q6k_wide = std::getenv("STRATA_MMVQ_WIDE") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_WIDE")) != 0;

template <typename F, int NCOLS, int RPW>
void launch_rowwarp(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS * RPW - 1) / (WARPS * RPW));
    s->parallel_for<dpct_kernel_name<class native_mmvq_rowwarp, F, dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<RPW>>>(
        sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { native_mmvq_rowwarp_kernel<F, NCOLS, RPW>(w, x, y, n_in, n_out); });
}
inline int rowwarp_rpw() {   // STRATA_MMVQ_RPW: 0 = the shared kernel, 1/2/4 = rows per warp (bench knob)
    static const int v = std::getenv("STRATA_MMVQ_RPW") ? std::atoi(std::getenv("STRATA_MMVQ_RPW")) : 0;   // default: the shared kernel (measured equal at warm clocks)
    return v;
}

template <typename F, int NCOLS>
void launch_multi_n(const void *weights, const void *x_q8_1, float *y, int n_in,
                    int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        constexpr int ROWS = 1;   // SYCL port: one row per work-group (more groups: the kernel is latency-bound here)
        const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
        s->parallel_for<dpct_kernel_name<
            class native_mmvq_multi_kernel_646c3d, F,
            dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<NW>,
            dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, NW, WARP),
                              sycl::range(1, NW, WARP)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_mmvq_multi_kernel<F, NCOLS, NW, ROWS>(w, x, y, n_in, n_out);
            });
        return;
    }
    if (mmvq_sg16(std::is_same_v<F, IQ4XSTraits>, n_out)) {   // same 128 threads as 8 SIMD16 sub-groups
        constexpr int NW16 = WARPS * WARP / 16;
        const sycl::range<3> t16(1, NW16, 16);
        if (n_in / F::DIV < F::BPI) {
            const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
            s->parallel_for<dpct_kernel_name<class native_mmvq_multi16_small, F, dpct_kernel_scalar<NCOLS>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * t16, t16),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                    native_mmvq_multi_kernel<F, NCOLS, NW16, WARPS, false, false, 16>(w, x, y, n_in, n_out);
                });
        } else {
            s->parallel_for<dpct_kernel_name<class native_mmvq_multi16, F, dpct_kernel_scalar<NCOLS>>>(
                sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * t16, t16),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                    native_mmvq_multi_kernel<F, NCOLS, NW16, 1, false, false, 16>(w, x, y, n_in, n_out);
                });
        }
        return;
    }
    const dpct::dim3 threads(WARP, WARPS);
    // #783 PR-h (stuchapin909): two rows per block for every K, not only small K - each row keeps its own partial sums and
    // reduction, so a row's result does not depend on its neighbour; STRATA_NO_MMVQ_ROWS2=1 keeps one row for large K
    static const bool rows1 = [] {
        const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (rows1 && n_in / F::DIV >= F::BPI) {
        if (s26_tsum_on()) {
            s->parallel_for<dpct_kernel_name<
                class native_mmvq_multi_kernel_2a7dda, F,
                dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<WARPS>,
                dpct_kernel_scalar<1>, dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * threads, threads),
                exp_props,
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                    native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, true>(w, x, y, n_in, n_out);
                });
        } else {
            s->parallel_for<dpct_kernel_name<
                class native_mmvq_multi_kernel_23124e, F,
                dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<WARPS>,
                dpct_kernel_scalar<1>, dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * threads, threads),
                exp_props,
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                    native_mmvq_multi_kernel<F, NCOLS, WARPS, 1>(w, x, y, n_in, n_out);
                });
        }
        return;
    }
    constexpr int ROWS = 2;
    const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
    if (s26_tsum_on()) {
        s->parallel_for<dpct_kernel_name<
            class native_mmvq_multi_kernel_279861, F,
            dpct_kernel_scalar<NCOLS>, dpct_kernel_scalar<WARPS>,
            dpct_kernel_scalar<ROWS>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * threads, threads),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS, true>(w, x, y, n_in, n_out);
            });
    } else {
        s->parallel_for<dpct_kernel_name<
            class native_mmvq_multi_kernel_75e8ad, F, dpct_kernel_scalar<NCOLS>,
            dpct_kernel_scalar<WARPS>, dpct_kernel_scalar<ROWS>,
            dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * threads, threads),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS>(w, x, y, n_in, n_out);
            });
    }
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = strata::q_of(stream);
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
void launch_check() {
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
    if (error != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        throw std::runtime_error(std::string("native MMVQ launch: ") +
                                 dpct::get_error_string_dummy(error));
    }
}


#if defined(STRATA_HIP_GFX906)
// ---- AMD (wave64) layout: one wavefront per R rows, four wavefronts per block, the row's blocks strided over the
// 64 lanes exactly as the CUDA kernels stride them over a block (kbx = lane / T, stride 64 / T), and a 64-lane
// butterfly instead of the LDS partials and __syncthreads of the one-block-per-row layout: on gfx906 a block per
// 2 KB row spent most of its time being launched and joined.  The SAME kernel serves every column count 1..8, so a
// column's sums do not depend on how many columns (verify tokens) ride along.  STRATA_MMVQ_WAVE=0: the CUDA layout.
bool g_wave_off = std::getenv("STRATA_MMVQ_WAVE") && std::string(std::getenv("STRATA_MMVQ_WAVE")) == "0";
template<typename F, int NCOLS, int R>
__launch_bounds__(256)
__global__ void native_mmvq_wave_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPIW = 64 / F::T;
    static_assert(64 % F::T == 0, "a block's threads must tile the wavefront");
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(lane);
    float tmp[NCOLS][R] = {};
    for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += BPIW) {
        const int kby = kbx * F::KBY;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = v;
        }
}
template<typename F, int NCOLS>
void wave_launch_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    constexpr int R = 1;
    const unsigned blocks = unsigned((std::size_t(n_out) + 4 * R - 1) / (4 * R));
    native_mmvq_wave_kernel<F, NCOLS, R><<<blocks, 256, 0, s>>>(w, x, y, n_in, n_out);
}
template<typename F>
void wave_launch(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: wave_launch_n<F, 1>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 2: wave_launch_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: wave_launch_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: wave_launch_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: wave_launch_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: wave_launch_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: wave_launch_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: wave_launch_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ (wave) requires 1 <= ncols <= 8");
    }
}
#define STRATA_WAVE_MMVQ(...) \
    if (!g_wave_off) { wave_launch<__VA_ARGS__>(weights, x_q8_1, y, n_in, n_out, ncols, stream); launch_check(); return; }
#else
#define STRATA_WAVE_MMVQ(...)
#endif

template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(SmallTraits<Weight, Qi>)
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            iq4nl_values.init(*s);

            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->submit([&](sycl::handler &cgh) {
                auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();

                cgh.parallel_for<dpct_kernel_name<
                    class native_small_mmvq_kernel_95f41f, Weight,
                    dpct_kernel_scalar<Qi>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, blocks) * threads,
                                      threads),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            native_small_mmvq_kernel<Weight, Qi, true>(
                                w, x, y, n_in, n_out, iq4nl_values_ptr_ct1);
                        });
            });
        }
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        iq4nl_values.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();

            cgh.parallel_for<dpct_kernel_name<
                class native_small_mmvq_kernel_d00fb9, Weight,
                dpct_kernel_scalar<Qi>, dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * threads,
                                  threads),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_small_mmvq_kernel<Weight, Qi, false>(
                            w, x, y, n_in, n_out, iq4nl_values_ptr_ct1);
                    });
        });
    }
    launch_check();
}

template<typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1,
               float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

// ============================ STRATA_Q8_PACKED=1: the packed Q8_0 decode layout (opt-in) ============================
//
// A lossless load-time repack of a Q8_0 matrix: the SAME int8 values in a qs plane (row-major, n_in bytes per row)
// followed by the SAME fp16 scales in a d plane (row-major, n_in / 32 per row) - no requantization, same bytes.
// The GGUF's 34-byte blocks leave every int 2-byte aligned (load_int_b2 = two 16-bit loads); the planes give each
// thread one aligned 8-byte run, read with streaming (non-temporal) loads. Every thread owns exactly the bytes of
// the ncols = 1 / multi-column EXACT layout (block kbx = tid / 4 + 32 i, ints 2 (tid % 4) and +1), accumulates them
// in the same order with the same expression, and the cross-warp sum and XOR tree are the same: every output is
// BITWISE equal to native_small_mmvq_kernel / native_mmvq_multi_kernel<SmallTraits<Q80Block, 8>, NC, 4, 1>.
// Eligible: 1024 <= n_in <= 4096 and n_out >= 2048, run as 640 persistent workgroups (a grid-stride over rows, two
// partial buffers). Measured on gfx1151 (S26, 8 real UD-IQ4_XS layers per shape, paired, 1-5 columns): 2560 ->
// 6144/10240/12288 +10-12% per call (engine trace: -10.7% on those calls). 6144 -> 2560 (one workgroup per row)
// was +2-4% standalone but -3% in the engine, and rows < 2048 lose: neither is packed.
#if defined(__HIPCC__)
#define STRATA_Q8P_LOAD(p) __builtin_nontemporal_load(p)
#else
/*
DPCT1098: The '*' expression is used instead of the __ldcs call. These two
expressions do not provide the exact same functionality. Check the generated
code for potential precision and/or performance issues.
*/
/*
DPCT1064: Migrated __ldcs call is used in a macro/template definition and
may not be valid for all macro/template uses. Adjust the code.
*/
#define STRATA_Q8P_LOAD(p) (*(p))
#endif
constexpr int Q8P_PERSIST_GRID = 640;
constexpr int Q8P_PERSIST_MAX_IN = 4096;

template <int NCOLS, bool TS = false>
/*
DPCT1110: The total declared local variable size in device function
native_q8_0_packed_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void
native_q8_0_packed_kernel(const int8_t *__restrict__ qs,
                          const sycl::half *__restrict__ dpl,
                          const Q81Block *__restrict__ x, float *__restrict__ y,
                          int n_in, int n_out, int row_step) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int nb = n_in / 32;
    const int kqs = 2 * (tid % 4);
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2][WARPS - 1][NCOLS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int buf = 0;
    for (int row = int(item_ct1.get_group(2)); row < n_out;
         row += row_step, buf ^= 1) {
        float tmp[NCOLS] = {};
        for (int kbx = tid / 4; kbx < nb; kbx += 2 * WARPS * WARP / 8) {
            const std::size_t block = std::size_t(row) * nb + kbx;
            const int* p = reinterpret_cast<const int*>(qs + block * 32) + kqs;
            const int v0 = STRATA_Q8P_LOAD(p);
            const int v1 = STRATA_Q8P_LOAD(p + 1);
            const float d0 = dpl[block];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * nb + kbx;
                const int* u = reinterpret_cast<const int*>(xb->qs);
                int sumi = 0;
                sumi = STRATA_DP4A(v0, u[kqs], sumi);
                sumi = STRATA_DP4A(v1, u[kqs + 1], sumi);
                const float d1 = xb->ds[0];
                tmp[j] += d0 * d1 * float(sumi);
            }
        }
        if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j)
                partial[buf][item_ct1.get_local_id(1) - 1][j]
                       [item_ct1.get_local_id(2)] = tmp[j];
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (TS && item_ct1.get_local_id(1) ==
                      0) { // S26 STRATA_TSUM=1: one transposed butterfly,
                           // bitwise the same sums
            constexpr int P = s26ts::pow2_ceil(NCOLS);
            float v[P];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] +=
                    partial[buf][l][j][item_ct1.get_local_id(2)];
                v[j] = tmp[j];
            }
#pragma unroll
            for (int j = NCOLS; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, int(item_ct1.get_local_id(2)));
            const int j = s26ts::tsum_token<P>(int(item_ct1.get_local_id(2)));
            if (int(item_ct1.get_local_id(2)) == s26ts::tsum_lane<P>(j) &&
                j < NCOLS) y[std::size_t(j) * n_out + row] = sum;
        } else if (item_ct1.get_local_id(1) == 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] +=
                    partial[buf][l][j][item_ct1.get_local_id(2)];
                tmp[j] = warp_sum(tmp[j]);
                if (item_ct1.get_local_id(2) == 0)
                    y[std::size_t(j) * n_out + row] = tmp[j];
            }
        }
    }
}

struct Q8Packed { const int8_t *qs; const sycl::half * d; int n_in, n_out; };
std::unordered_map<const void*, Q8Packed>& q8_packed_registry() {
    static std::unordered_map<const void*, Q8Packed> registry;
    return registry;
}

template <int NCOLS>
void q8_packed_launch(const Q8Packed &w, const Q81Block *x, float *y,
                      dpct::queue_ptr s) {
    const bool persist = w.n_in <= Q8P_PERSIST_MAX_IN;
    const int grid = persist ? (std::min)(w.n_out, Q8P_PERSIST_GRID) : w.n_out;
    if (s26_tsum_on())
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class native_q8_0_packed_kernel_851f07,
                                         dpct_kernel_scalar<NCOLS>,
                                         dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, unsigned(grid)) *
                                  sycl::range(1, WARPS, WARP),
                              sycl::range(1, WARPS, WARP)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_q8_0_packed_kernel<NCOLS, true>(
                    w.qs, w.d, x, y, w.n_in, w.n_out, persist ? grid : w.n_out);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->/*
DPCT1050: The template argument of the dpct_kernel_name could not be
deduced. You need to update this code.
*/
            parallel_for<
                dpct_kernel_name<class native_q8_0_packed_kernel_56c826,
                                 dpct_kernel_scalar<NCOLS>,
                                 dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, unsigned(grid)) *
                                      sycl::range(1, WARPS, WARP),
                                  sycl::range(1, WARPS, WARP)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q8_0_packed_kernel<NCOLS>(
                            w.qs, w.d, x, y, w.n_in, w.n_out,
                            persist ? grid : w.n_out);
                    });
    }
}

bool q8_packed_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q8_packed_registry();
    if (registry.empty() || (ncols > 1 && !g_multi_exact)) return false;   // packed = the EXACT layout only
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, ncols, 32);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    switch (ncols) {
        case 1: q8_packed_launch<1>(found->second, x, y, s); break;
        case 2: q8_packed_launch<2>(found->second, x, y, s); break;
        case 3: q8_packed_launch<3>(found->second, x, y, s); break;
        case 4: q8_packed_launch<4>(found->second, x, y, s); break;
        case 5: q8_packed_launch<5>(found->second, x, y, s); break;
        case 6: q8_packed_launch<6>(found->second, x, y, s); break;
        case 7: q8_packed_launch<7>(found->second, x, y, s); break;
        case 8: q8_packed_launch<8>(found->second, x, y, s); break;
        default: throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
    }
    launch_check();
    return true;
}


// ============================ STRATA_Q6_PACKED=1: the packed Q6_K head layout (opt-in) ============================
//
// The output heads (the main head, 248320 rows, and the MTP draft head's row subset) are Q6_K matrices of 2560-wide
// rows: 10 blocks of 210 bytes each, so every int load is 2-byte aligned and a row is 2100 bytes. The packed copy
// holds the SAME bytes as four planes - ql (128 B per block), qh (64 B), scales (16 B), d (2 B) - so a thread's
// two words are aligned dword loads (non-temporal). Thread ownership (block tid / 32 + 4 i, word tid % 32), the
// q6_q8_dot_impl expression, the per-thread order and the reduction are those of native_q6_k_mmvq_kernel<false> /
// native_mmvq_multi_kernel<Q6KTraits, NC, 4, 1>: every output is bitwise equal. Persistent workgroups.
int g_q6p_grid_override = 0;   // the self-test's grid sweep (STRATA_Q6P_SELFTEST=2)
int q6p_grid() {
    if (g_q6p_grid_override > 0) return g_q6p_grid_override;
    static const int grid = [] {
        const char* v = std::getenv("STRATA_Q6P_GRID");
        const int g = v ? std::atoi(v) : 0;
        return g > 0 ? g : 640;
    }();
    return grid;
}

template <int NCOLS, bool TS = false>
/*
DPCT1110: The total declared local variable size in device function
native_q6_k_packed_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void native_q6_k_packed_kernel(
    const int *__restrict__ ql, const int *__restrict__ qh,
    const int8_t *__restrict__ sc, const sycl::half *__restrict__ dpl,
    const Q81Block *__restrict__ x, float *__restrict__ y, int n_in, int n_out,
    int row_step) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int nb = n_in / 256;
    const int x_stride = n_in / Q8K;
    const int iqs = tid % 32;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int qh_word = 8 * (iqs / 16) + iqs % 8;
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2][WARPS - 1][NCOLS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int buf = 0;
    for (int row = int(item_ct1.get_group(2)); row < n_out;
         row += row_step, buf ^= 1) {
        float tmp[NCOLS] = {};
        for (int kbx = tid / 32; kbx < nb; kbx += WARPS * WARP / 32) {
            const std::size_t block = std::size_t(row) * nb + kbx;
            const int vl = STRATA_Q8P_LOAD(ql + block * 32 + iqs);
            const int vh = STRATA_Q8P_LOAD(qh + block * 16 + qh_word) >> vh_shift;
            const int8_t* scales = sc + block * 16 + scale_offset;
            const float d = dpl[block];
            const int kby = kbx * 8;
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kby;
                int u[2];
                float d8[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    u[i] = reinterpret_cast<const int*>(xb[bq8_offset + 2 * i].qs)[iqs % 8];
                    d8[i] = xb[bq8_offset + 2 * i].ds[0];
                }
                tmp[j] += q6_q8_dot_impl(vl, vh, u, scales, d, d8);
            }
        }
        if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j)
                partial[buf][item_ct1.get_local_id(1) - 1][j]
                       [item_ct1.get_local_id(2)] = tmp[j];
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (TS && item_ct1.get_local_id(1) ==
                      0) { // S26 STRATA_TSUM=1: one transposed butterfly,
                           // bitwise the same sums
            constexpr int P = s26ts::pow2_ceil(NCOLS);
            float v[P];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] +=
                    partial[buf][l][j][item_ct1.get_local_id(2)];
                v[j] = tmp[j];
            }
#pragma unroll
            for (int j = NCOLS; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, int(item_ct1.get_local_id(2)));
            const int j = s26ts::tsum_token<P>(int(item_ct1.get_local_id(2)));
            if (int(item_ct1.get_local_id(2)) == s26ts::tsum_lane<P>(j) &&
                j < NCOLS) y[std::size_t(j) * n_out + row] = sum;
        } else if (item_ct1.get_local_id(1) == 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] +=
                    partial[buf][l][j][item_ct1.get_local_id(2)];
                tmp[j] = warp_sum(tmp[j]);
                if (item_ct1.get_local_id(2) == 0)
                    y[std::size_t(j) * n_out + row] = tmp[j];
            }
        }
    }
}

bool g_q6_bypass = false;   // the load-time self-test times the GGUF-layout kernels with the copy registered

// the packed planes of one Q6_K matrix (owned: allocated by native_q6_k_pack, freed by native_q6_k_unpack)
struct Q6Packed {
    void *base; const int * ql; const int * qh; const int8_t * sc;
    const sycl::half * d;
    int n_in, n_out;
};
std::unordered_map<const void*, Q6Packed>& q6_packed_registry() {
    static std::unordered_map<const void*, Q6Packed> registry;
    return registry;
}

template <int NCOLS>
void q6_packed_launch(const Q6Packed &w, const Q81Block *x, float *y,
                      dpct::queue_ptr s) {
    const int grid = (std::min)(w.n_out, q6p_grid());
    if (s26_tsum_on())
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class native_q6_k_packed_kernel_ee650b,
                                         dpct_kernel_scalar<NCOLS>,
                                         dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, unsigned(grid)) *
                                  sycl::range(1, WARPS, WARP),
                              sycl::range(1, WARPS, WARP)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_q6_k_packed_kernel<NCOLS, true>(
                    w.ql, w.qh, w.sc, w.d, x, y, w.n_in, w.n_out, grid);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->/*
DPCT1050: The template argument of the dpct_kernel_name could not be
deduced. You need to update this code.
*/
            parallel_for<
                dpct_kernel_name<class native_q6_k_packed_kernel_ca2554,
                                 dpct_kernel_scalar<NCOLS>,
                                 dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, unsigned(grid)) *
                                      sycl::range(1, WARPS, WARP),
                                  sycl::range(1, WARPS, WARP)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q6_k_packed_kernel<NCOLS>(
                            w.ql, w.qh, w.sc, w.d, x, y, w.n_in, w.n_out, grid);
                    });
    }
}

bool q6_packed_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q6_packed_registry();
    if (registry.empty() || g_q6_bypass || (ncols > 1 && !g_multi_exact)) return false;
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, ncols, 256);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    switch (ncols) {
        case 1: q6_packed_launch<1>(found->second, x, y, s); break;
        case 2: q6_packed_launch<2>(found->second, x, y, s); break;
        case 3: q6_packed_launch<3>(found->second, x, y, s); break;
        case 4: q6_packed_launch<4>(found->second, x, y, s); break;
        case 5: q6_packed_launch<5>(found->second, x, y, s); break;
        case 6: q6_packed_launch<6>(found->second, x, y, s); break;
        case 7: q6_packed_launch<7>(found->second, x, y, s); break;
        case 8: q6_packed_launch<8>(found->second, x, y, s); break;
        default: throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
    }
    launch_check();
    return true;
}

// ============================ STRATA_Q6_PLANAR: the Q6_K heads as 16-byte-aligned planes (SYCL port) ============================
//
// native_mmvq_q6k_wide_kernel<..., A2 = true> reads each lane's ql / qh run as two aligned 16-byte loads and a funnel shift
// because a 210-byte block is only 2-byte aligned. The planar layout holds the same bytes as four planes (ql 128 B, qh 64 B,
// scales 16 B, d 2 B per block): a lane's runs are single aligned int4 loads, its two scales one 16-bit load (the pair
// scales[so], scales[so + 4] stored adjacent, at byte 2g). The kernel is the A2 kernel's, line for line, from the loads on:
// every output is bitwise equal, ~14% faster on the 521 MB LM head and the draft head. Same bytes, so a matrix is CONVERTED
// (planar allocation made, GGUF-layout one freed): no VRAM is spent. Only the output heads are converted - the prompt path
// reads the other projections in the GGUF layout.
struct Q6Planar {
    void* base;
    const uint8_t* ql; const uint8_t* qh; const uint8_t* sc; const sycl::half* d;
    int n_in, n_out;
};
std::unordered_map<const void*, Q6Planar>& q6_planar_registry() {
    static std::unordered_map<const void*, Q6Planar> registry;
    return registry;
}

template <int NCOLS>
void native_mmvq_q6k_planar_kernel(const uint8_t* __restrict__ qlp, const uint8_t* __restrict__ qhp,
                                   const uint8_t* __restrict__ scp, const sycl::half* __restrict__ dp,
                                   const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const int vh_shift = 2 * ((g & 3) >> 1);
    const int qh4_idx = 2 * (g >> 2) + (g & 1);
    const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
    const int u_int4 = g & 1;
    float acc[NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += WARP / 8) {
        const int kby = kbx * (QK / Q8K);
        const std::size_t bi = std::size_t(row) * blocks_per_row + kbx;
        const sycl::int4 ql4 = reinterpret_cast<const sycl::int4*>(qlp + bi * 128)[g];
        const sycl::int4 qh4 = reinterpret_cast<const sycl::int4*>(qhp + bi * 64)[qh4_idx];
        const uint16_t s2 = reinterpret_cast<const uint16_t*>(scp + bi * 16)[g];
        const float d = dp[bi];
        const float dsc0 = d * (float) (int8_t) (s2 & 0xff), dsc1 = d * (float) (int8_t) (s2 >> 8);
        sycl::int4 u0[NCOLS], u1[NCOLS];
        float ds0[NCOLS], ds1[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * x_stride + kby + bq8_offset;
            u0[j] = ld_i4_a4(reinterpret_cast<const int*>(xb[0].qs) + 4 * u_int4);
            u1[j] = ld_i4_a4(reinterpret_cast<const int*>(xb[2].qs) + 4 * u_int4);
            ds0[j] = xb[0].ds[0]; ds1[j] = xb[2].ds[0];
        }
        const int vl[4] = {ql4.x(), ql4.y(), ql4.z(), ql4.w()};
        const int vh[4] = {qh4.x() >> vh_shift, qh4.y() >> vh_shift, qh4.z() >> vh_shift, qh4.w() >> vh_shift};
        int vi0[4], vi1[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            vi0[p] = q6_sub32((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030));
            vi1[p] = q6_sub32(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030));
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            int s0 = 0, s1 = 0;
            s0 = strata::dp4a(vi0[0], u0[j].x(), s0); s0 = strata::dp4a(vi0[1], u0[j].y(), s0);
            s0 = strata::dp4a(vi0[2], u0[j].z(), s0); s0 = strata::dp4a(vi0[3], u0[j].w(), s0);
            s1 = strata::dp4a(vi1[0], u1[j].x(), s1); s1 = strata::dp4a(vi1[1], u1[j].y(), s1);
            s1 = strata::dp4a(vi1[2], u1[j].z(), s1); s1 = strata::dp4a(vi1[3], u1[j].w(), s1);
            acc[j] += ds0[j] * dsc0 * (float) s0 + ds1[j] * dsc1 * (float) s1;
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

template <int NCOLS>
void q6_planar_launch(const Q6Planar& w, const Q81Block* x, float* y, dpct::queue_ptr s) {
    const unsigned blocks = unsigned((std::size_t(w.n_out) + WARPS - 1) / WARPS);
    const uint8_t *ql = w.ql, *qh = w.qh, *sc = w.sc;
    const sycl::half* d = w.d;
    const int n_in = w.n_in, n_out = w.n_out;
    if constexpr (NCOLS > 1) {   // same rule as the GGUF-layout wide kernel (#2207): the multi-column kernel spills at 128 GRF on Alchemist
        if (mmvq_large_grf(s)) {
            s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_planar_grf256, dpct_kernel_scalar<NCOLS>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                    native_mmvq_q6k_planar_kernel<NCOLS>(ql, qh, sc, d, x, y, n_in, n_out);
                });
            return;
        }
    }
    s->parallel_for<dpct_kernel_name<class native_mmvq_q6k_planar, dpct_kernel_scalar<NCOLS>>>(
        sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_mmvq_q6k_planar_kernel<NCOLS>(ql, qh, sc, d, x, y, n_in, n_out);
        });
}

// every registered matrix is served here whatever the column count (more than 8: in chunks of 8), so a converted matrix is
// never read through a GGUF-layout kernel
bool q6_planar_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q6_planar_registry();
    if (registry.empty() || ncols < 1) return false;
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, std::min(ncols, 8), 256);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto s = strata::q_of(stream);
    const std::size_t x_col = std::size_t(n_in / Q8K);   // Q8_1 blocks per column
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = std::min(8, ncols - c0);
        const auto* x = static_cast<const Q81Block*>(x_q8_1) + std::size_t(c0) * x_col;
        float* yy = y + std::size_t(c0) * n_out;
        switch (nc) {
            case 1: q6_planar_launch<1>(found->second, x, yy, s); break;
            case 2: q6_planar_launch<2>(found->second, x, yy, s); break;
            case 3: q6_planar_launch<3>(found->second, x, yy, s); break;
            case 4: q6_planar_launch<4>(found->second, x, yy, s); break;
            case 5: q6_planar_launch<5>(found->second, x, yy, s); break;
            case 6: q6_planar_launch<6>(found->second, x, yy, s); break;
            case 7: q6_planar_launch<7>(found->second, x, yy, s); break;
            default: q6_planar_launch<8>(found->second, x, yy, s); break;
        }
    }
    launch_check();
    return true;
}

// plane offsets of a matrix of nb blocks in one allocation: ql | qh | scales | d, each 256-byte aligned
struct Q6PlanarLayout { std::size_t o_qh, o_sc, o_d, total; };
Q6PlanarLayout q6_planar_layout(std::size_t nb) {
    Q6PlanarLayout l;
    l.o_qh = (nb * 128 + 255) & ~std::size_t(255);
    l.o_sc = (l.o_qh + nb * 64 + 255) & ~std::size_t(255);
    l.o_d = (l.o_sc + nb * 16 + 255) & ~std::size_t(255);
    l.total = l.o_d + nb * 2;
    return l;
}
Q6Planar q6_planar_make(void* dev, std::size_t nb, int n_in, int n_out) {
    const Q6PlanarLayout l = q6_planar_layout(nb);
    auto* base = static_cast<uint8_t*>(dev);
    return Q6Planar{dev, base, base + l.o_qh, base + l.o_sc, reinterpret_cast<const sycl::half*>(base + l.o_d), n_in, n_out};
}
} // namespace

bool native_q8_0_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q8_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

bool native_q8_0_packed_eligible(int n_in, int n_out) {
    return n_in > 0 && n_in % 32 == 0 && n_in / 32 >= 2 * WARPS * WARP / 8 && n_in <= Q8P_PERSIST_MAX_IN &&
           n_out >= 2048;
}

void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out) {
    const auto* blocks = static_cast<const Q80Block*>(gguf_blocks);
    auto* qs = static_cast<int8_t*>(out);
    auto *d = reinterpret_cast<sycl::half *>(qs + std::size_t(n_in) * n_out);
    const std::size_t count = std::size_t(n_in / 32) * n_out;
    for (std::size_t b = 0; b < count; ++b) {
        std::memcpy(qs + b * 32, blocks[b].qs, 32);
        std::memcpy(d + b, &blocks[b].d, sizeof(sycl::half));
    }
}

void native_q8_0_packed_register(const void* gguf_weights, const void* packed, int n_in, int n_out) {
    validate_pointer(gguf_weights);
    validate_pointer(packed);
    if (!native_q8_0_packed_eligible(n_in, n_out)) throw std::invalid_argument("Q8_0 packed: ineligible shape");
    const auto* qs = static_cast<const int8_t*>(packed);
    q8_packed_registry()[gguf_weights] = Q8Packed{
        qs,
        reinterpret_cast<const sycl::half *>(qs + std::size_t(n_in) * n_out),
        n_in, n_out};
}

void native_q8_0_packed_unregister(const void* gguf_weights) { q8_packed_registry().erase(gguf_weights); }

bool native_q6_k_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q6_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

bool native_q6_k_planar_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q6_PLANAR");
        return !(v && v[0] == '0' && v[1] == '\0');
    }();
    return enabled;
}

bool native_q6_k_is_planar(const void* weights) { return q6_planar_registry().count(weights) != 0; }

// Converts a GGUF-layout Q6_K matrix (a device allocation of the caller's) to the planar layout: a new allocation holds the
// planes, the old one is freed, the returned pointer (the new allocation's, to be freed with sycl::free as the old one would
// have been, after native_q6_k_unpack) is the key native_q6_k_mmvq serves it under. Not eligible or no room: `weights` back.
void* native_q6_k_planar_adopt(void* weights, int n_in, int n_out, const char* what) try {
    if (!weights || n_in <= 0 || n_in % 256 || n_out <= 0 || !native_q6_k_planar_enabled()) return weights;
    if (q6_planar_registry().count(weights)) return weights;
    auto& q = dpct::get_in_order_queue();
    const std::size_t nb = std::size_t(n_in / 256) * std::size_t(n_out);
    const Q6PlanarLayout l = q6_planar_layout(nb);
    void* dev = nullptr;
    try { dev = strata::malloc_device_guarded(l.total, q, what); } catch (...) { dev = nullptr; }
    if (dev == nullptr) {
        std::fprintf(stderr, "strata: STRATA_Q6_PLANAR: no VRAM for the %s's planar copy (+%.1f MiB), keeping the GGUF layout\n", what,
                     l.total / 1048576.0);
        return weights;
    }
    const uint8_t* src = static_cast<const uint8_t*>(weights);
    uint8_t* base = static_cast<uint8_t*>(dev);
    const std::size_t o_qh = l.o_qh, o_sc = l.o_sc, o_d = l.o_d;
    q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
        const std::size_t b = id[0];
        const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src + b * 210);
        uint16_t* qlo = reinterpret_cast<uint16_t*>(base + b * 128);
        for (int i = 0; i < 64; ++i) qlo[i] = s16[i];
        uint16_t* qho = reinterpret_cast<uint16_t*>(base + o_qh + b * 64);
        for (int i = 0; i < 32; ++i) qho[i] = s16[64 + i];
        const int8_t* sc = reinterpret_cast<const int8_t*>(src + b * 210 + 192);
        uint8_t* sco = base + o_sc + b * 16;
        for (int g = 0; g < 8; ++g) {
            const int so = 8 * (g >> 2) + (g & 3);
            sco[2 * g] = (uint8_t) sc[so]; sco[2 * g + 1] = (uint8_t) sc[so + 4];
        }
        reinterpret_cast<uint16_t*>(base + o_d)[b] = s16[104];
    });
    q.wait();
    sycl::free(weights, q);
    q6_planar_registry()[dev] = q6_planar_make(dev, nb, n_in, n_out);
    std::fprintf(stderr, "strata: STRATA_Q6_PLANAR: the %s (%d x %d) is stored as 16-byte-aligned planes (same bytes)\n", what, n_out, n_in);
    return dev;
}
catch (sycl::exception const& exc) {
    std::fprintf(stderr, "strata: STRATA_Q6_PLANAR conversion failed: %s\n", exc.what());
    return weights;
}

// A planar matrix's rows `rows_dev[0 .. n_sel)` as a new planar matrix (the draft head's token subset of the LM head).
void* native_q6_k_planar_gather(const void* planar, int n_in, const int32_t* rows_dev, int64_t n_sel, const char* what) try {
    const auto found = q6_planar_registry().find(planar);
    if (found == q6_planar_registry().end() || found->second.n_in != n_in || n_sel <= 0) return nullptr;
    auto& q = dpct::get_in_order_queue();
    const int nbr = n_in / 256;
    const std::size_t nb = std::size_t(nbr) * std::size_t(n_sel);
    const Q6PlanarLayout l = q6_planar_layout(nb);
    void* dev = nullptr;
    try { dev = strata::malloc_device_guarded(l.total, q, what); } catch (...) { dev = nullptr; }
    if (dev == nullptr) return nullptr;
    const Q6Planar src = found->second;
    uint8_t* base = static_cast<uint8_t*>(dev);
    const std::size_t o_qh = l.o_qh, o_sc = l.o_sc, o_d = l.o_d;
    q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
        const std::size_t b = id[0];
        const std::size_t i = b / nbr, j = b - i * nbr;
        const std::size_t sb = std::size_t(rows_dev[i]) * nbr + j;
        const sycl::int4* sq = reinterpret_cast<const sycl::int4*>(src.ql + sb * 128);
        sycl::int4* dq = reinterpret_cast<sycl::int4*>(base + b * 128);
        for (int k = 0; k < 8; ++k) dq[k] = sq[k];
        const sycl::int4* sh = reinterpret_cast<const sycl::int4*>(src.qh + sb * 64);
        sycl::int4* dh = reinterpret_cast<sycl::int4*>(base + o_qh + b * 64);
        for (int k = 0; k < 4; ++k) dh[k] = sh[k];
        *reinterpret_cast<sycl::int4*>(base + o_sc + b * 16) = *reinterpret_cast<const sycl::int4*>(src.sc + sb * 16);
        reinterpret_cast<uint16_t*>(base + o_d)[b] = reinterpret_cast<const uint16_t*>(src.d)[sb];
    });
    q.wait();
    q6_planar_registry()[dev] = q6_planar_make(dev, nb, n_in, (int) n_sel);
    std::fprintf(stderr, "strata: STRATA_Q6_PLANAR: the %s (%lld x %d) is gathered from the planar head\n", what, (long long) n_sel, n_in);
    return dev;
}
catch (sycl::exception const& exc) {
    std::fprintf(stderr, "strata: STRATA_Q6_PLANAR gather failed: %s\n", exc.what());
    return nullptr;
}

bool native_q6_k_pack(const void *weights, int n_in, int n_out,
                      const char *what) try {
    if (!weights || n_in <= 0 || n_in % 256 || n_in / 256 < WARPS * WARP / 32 || n_out < 2048) return false;
    if (q6_packed_registry().count(weights)) return true;
    const std::size_t nb = std::size_t(n_in / 256) * std::size_t(n_out);
    const std::size_t bytes = nb * sizeof(Q6KBlock);
    std::vector<Q6KBlock> host(nb);
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(host.data(), weights, bytes)
                             .wait()) != 0) {
        ; return false;
    }
    // planes: ql | qh | scales | d, each 256-byte aligned
    const std::size_t o_qh = (nb * 128 + 255) & ~std::size_t(255);
    const std::size_t o_sc = (o_qh + nb * 64 + 255) & ~std::size_t(255);
    const std::size_t o_d = (o_sc + nb * 16 + 255) & ~std::size_t(255);
    const std::size_t total = o_d + nb * 2;
    std::vector<uint8_t> packed(total, 0);
    for (std::size_t b = 0; b < nb; ++b) {
        std::memcpy(packed.data() + b * 128, host[b].ql, 128);
        std::memcpy(packed.data() + o_qh + b * 64, host[b].qh, 64);
        std::memcpy(packed.data() + o_sc + b * 16, host[b].scales, 16);
        std::memcpy(packed.data() + o_d + b * 2, &host[b].d, 2);
    }
    void* dev = nullptr;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(dev = (void *)sycl::malloc_device(
                             total, dpct::get_in_order_queue())) != 0) {
        ; return false;
    }
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    if (DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(dev, packed.data(),
                                                           total).wait()) != 0) {
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        sycl::free(dev, dpct::get_in_order_queue());; return false;
    }
    auto* base = static_cast<uint8_t*>(dev);
    q6_packed_registry()[weights] =
        Q6Packed{dev,
                 reinterpret_cast<const int *>(base),
                 reinterpret_cast<const int *>(base + o_qh),
                 reinterpret_cast<const int8_t *>(base + o_sc),
                 reinterpret_cast<const sycl::half *>(base + o_d),
                 n_in,
                 n_out};
    std::fprintf(stderr, "strata: STRATA_Q6_PACKED=1 packed the %s (%d x %d, +%.1f MiB)\n", what, n_out, n_in,
                 total / 1048576.0);
    // STRATA_Q6P_SELFTEST=1: bitwise check against the GGUF-layout kernels (1..4 columns) and paired timings
    if (const char* v = std::getenv("STRATA_Q6P_SELFTEST"); v && (v[0] == '1' || v[0] == '2')) {
        dpct::queue_ptr s = &dpct::get_in_order_queue();
        /*
        DPCT1025: The SYCL queue is created ignoring the flag and priority
        options.
        */
        s = dpct::get_current_device().create_queue(true);
        const int nq = n_in / Q8K;
        std::vector<Q81Block> hx(std::size_t(4) * nq);
        uint32_t r = 12345u;
        for (auto& blk : hx) {
            for (auto& q : blk.qs) { r = r * 1664525u + 1013904223u; q = int8_t(int((r >> 24) & 255u) - 128); }
            r = r * 1664525u + 1013904223u;
            blk.ds = sycl::float2(0.001f + float(r >> 8) * 1e-9f, 0.0f)
                         .convert<sycl::half, sycl::rounding_mode::rte>();
        }
        void *dx = nullptr, *ya = nullptr, *yb = nullptr;
        dx = (void *)sycl::malloc_device(hx.size() * sizeof(Q81Block),
                                         dpct::get_in_order_queue());
        ya = (void *)sycl::malloc_device(std::size_t(4) * n_out * 4,
                                         dpct::get_in_order_queue());
        yb = (void *)sycl::malloc_device(std::size_t(4) * n_out * 4,
                                         dpct::get_in_order_queue());
        dpct::get_in_order_queue()
            .memcpy(dx, hx.data(), hx.size() * sizeof(Q81Block))
            .wait();
        std::vector<float> a(std::size_t(4) * n_out), c(std::size_t(4) * n_out);
        bool same = true;
        for (int nc = 1; nc <= 4; ++nc) {
            g_q6_bypass = true;
            native_q6_k_mmvq(weights, dx, (float*) ya, n_in, n_out, nc, s);
            g_q6_bypass = false;
            native_q6_k_mmvq(weights, dx, (float*) yb, n_in, n_out, nc, s);
            s->wait();
            dpct::get_in_order_queue()
                .memcpy(a.data(), ya, std::size_t(nc) * n_out * 4)
                .wait();
            dpct::get_in_order_queue()
                .memcpy(c.data(), yb, std::size_t(nc) * n_out * 4)
                .wait();
            same = same && std::memcmp(a.data(), c.data(), std::size_t(nc) * n_out * 4) == 0;
        }
        dpct::event_ptr e0, e1;
        e0 = new sycl::event(); e1 = new sycl::event();
        for (int nc : {1, 2, 4}) {
            std::vector<float> tb, tc;
            for (int k = 0; k < 11; ++k) {
                for (int pass = 0; pass < 2; ++pass) {
                    const bool cand = (pass == 0) == (k % 2 == 1);
                    g_q6_bypass = !cand;
                    dpct::sync_barrier(e0, s);
                    for (int it = 0; it < 5; ++it) native_q6_k_mmvq(weights, dx, (float*) (cand ? yb : ya), n_in, n_out, nc, s);
                    dpct::sync_barrier(e1, s);
                    e1->wait_and_throw();
                    float ms = 0;
                        ms = (e1->get_profiling_info<
                                  sycl::info::event_profiling::command_end>() -
                              e0->get_profiling_info<
                                  sycl::info::event_profiling::
                                      command_start>()) /
                             1000000.0f;
                    (cand ? tc : tb).push_back(ms * 1000.0f / 5);
                }
            }
            g_q6_bypass = false;
            std::sort(tb.begin(), tb.end()); std::sort(tc.begin(), tc.end());
            std::fprintf(stderr, "strata: Q6_K packed self-test %s ncols %d: GGUF %.1f us (%.1f-%.1f), packed %.1f us "
                         "(%.1f-%.1f), 11 paired samples\n", what, nc, tb[5], tb[0], tb[10], tc[5], tc[0], tc[10]);
        }
        std::fprintf(stderr, "strata: Q6_K packed self-test %s: outputs %s (1-4 columns)\n", what,
                     same ? "BITWISE EQUAL" : "DIFFER");
        if (v[0] == '2') {   // grid sweep of the packed kernel
            for (int grid : {160, 320, 640, 1280, 2560, 5120, 20480}) {
                g_q6p_grid_override = grid;
                for (int nc : {1, 2, 4}) {
                    std::vector<float> tc;
                    for (int k = 0; k < 7; ++k) {
                        dpct::sync_barrier(e0, s);
                        for (int it = 0; it < 5; ++it) native_q6_k_mmvq(weights, dx, (float*) yb, n_in, n_out, nc, s);
                        dpct::sync_barrier(e1, s);
                        e1->wait_and_throw();
                        float ms = 0; ms = (e1->get_profiling_info<
                                                sycl::info::event_profiling::
                                                    command_end>() -
                                            e0->get_profiling_info<
                                                sycl::info::event_profiling::
                                                    command_start>()) /
                                           1000000.0f;
                        tc.push_back(ms * 1000.0f / 5);
                    }
                    std::sort(tc.begin(), tc.end());
                    std::fprintf(stderr, "strata: Q6_K packed grid sweep %s grid %d ncols %d: %.1f us (%.1f-%.1f)\n",
                                 what, grid, nc, tc[3], tc[0], tc[6]);
                }
            }
            g_q6p_grid_override = 0;
        }
        dpct::destroy_event(e0); dpct::destroy_event(e1);
        sycl::free(dx, dpct::get_in_order_queue());
            sycl::free(ya, dpct::get_in_order_queue());
            sycl::free(yb, dpct::get_in_order_queue());
        dpct::get_current_device().destroy_queue(s);
        if (!same) { native_q6_k_unpack(weights); return false; }
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void native_q6_k_unpack(const void* weights) {
    q6_planar_registry().erase(weights);   // planar matrices are the caller's allocation (native_q6_k_planar_adopt): the caller frees it
    auto& registry = q6_packed_registry();
    const auto found = registry.find(weights);
    if (found == registry.end()) return;
    sycl::free(found->second.base, dpct::get_in_order_queue());
    registry.erase(found);
}

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
void native_mmvq_set_q6k_wide(bool on) { g_q6k_wide = on; }
void native_q6_k_mmvq_stride224(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                                void* stream) {
    const auto s = strata::q_of(stream);
    switch (ncols) {
        case 1: launch_q6k_wide224<1>(weights, x_q8_1, y, n_in, n_out, s); return;
        case 2: launch_q6k_wide224<2>(weights, x_q8_1, y, n_in, n_out, s); return;
        case 3: launch_q6k_wide224<3>(weights, x_q8_1, y, n_in, n_out, s); return;
        case 4: launch_q6k_wide224<4>(weights, x_q8_1, y, n_in, n_out, s); return;
        default: launch_q6k_wide224<6>(weights, x_q8_1, y, n_in, n_out, s); return;
    }
}
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<
                dpct_kernel_name<class native_quantize_q8_1_kernel_b480d2>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, QUANT_THREADS),
                                  sycl::range(1, 1, QUANT_THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_quantize_q8_1_kernel(
                            x, static_cast<Q81Block *>(x_q8_1), n_total);
                    });
    }
    launch_check();
}

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            ((sycl::queue *)(strata::q_of(stream)))
                ->get_device(),
            {sycl::aspect::fp16});

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<
                class native_swiglu_quantize_q8_1_kernel_85e13e>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, QUANT_THREADS),
                                  sycl::range(1, 1, QUANT_THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_swiglu_quantize_q8_1_kernel(
                            gate, up, static_cast<Q81Block *>(x_q8_1), n_total);
                    });
    }
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (try_wide<WideQ5K>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    STRATA_WAVE_MMVQ(Q5KTraits)
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->parallel_for<
                dpct_kernel_name<class native_q5_k_mmvq_kernel_b1d5a2,
                                 dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * threads, threads),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q5_k_mmvq_kernel<true>(w, x, y, n_in, n_out);
                    });
        }
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class native_q5_k_mmvq_kernel_938b09,
                                         dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * threads,
                              threads),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_q5_k_mmvq_kernel<false>(w, x, y, n_in, n_out);
            });
    }
    launch_check();
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q20Traits)
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<
                dpct_kernel_name<class native_q2_0_mmvq_kernel_dec305,
                                 dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * threads, threads),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q2_0_mmvq_kernel<true>(w, x, y, n_in, n_out);
                    });
        }
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class native_q2_0_mmvq_kernel_ef5f5c,
                                         dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * threads,
                              threads),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_q2_0_mmvq_kernel<false>(w, x, y, n_in, n_out);
            });
    }
    launch_check();
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q3KTraits)
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<
                dpct_kernel_name<class native_q3_k_mmvq_kernel_208be9,
                                 dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * threads, threads),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q3_k_mmvq_kernel<true>(w, x, y, n_in, n_out);
                    });
        }
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class native_q3_k_mmvq_kernel_926817,
                                         dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * threads,
                              threads),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_q3_k_mmvq_kernel<false>(w, x, y, n_in, n_out);
            });
    }
    launch_check();
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q3_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (try_wide<WideIQ4XS>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    STRATA_WAVE_MMVQ(IQ4XSTraits)
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            iq4nl_values.init(*s);

            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->submit([&](sycl::handler &cgh) {
                auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();

                cgh.parallel_for<
                    dpct_kernel_name<class native_iq4_xs_mmvq_kernel_22a688,
                                     dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, blocks) * threads,
                                      threads),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            native_iq4_xs_mmvq_kernel<true>(
                                w, x, y, n_in, n_out, iq4nl_values_ptr_ct1);
                        });
            });
        }
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        iq4nl_values.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();

            cgh.parallel_for<
                dpct_kernel_name<class native_iq4_xs_mmvq_kernel_3ba00d,
                                 dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * threads,
                                  threads),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_iq4_xs_mmvq_kernel<false>(w, x, y, n_in, n_out,
                                                         iq4nl_values_ptr_ct1);
                    });
        });
    }
    launch_check();
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_iq4_xs_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (try_wide<WideQ4K>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    STRATA_WAVE_MMVQ(Q4KTraits)
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->parallel_for<
                dpct_kernel_name<class native_q4_k_mmvq_kernel_da128d,
                                 dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * threads, threads),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q4_k_mmvq_kernel<true>(w, x, y, n_in, n_out);
                    });
        }
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class native_q4_k_mmvq_kernel_b29f4f,
                                         dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * threads,
                              threads),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_q4_k_mmvq_kernel<false>(w, x, y, n_in, n_out);
            });
    }
    launch_check();
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q4_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    if (q6_packed_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // STRATA_Q6_PACKED=1 heads
    if (q6_planar_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // STRATA_Q6_PLANAR copies
    // SYCL port: 16-byte-load kernel for the decode shapes (ncols <= 4)
    // With the aligned loads (load16_a2) the wide kernel wins on every Q6_K shape and window width: 2.6-3.4x the
    // misaligned-load kernels in q6k_align_bench, decode 44.9 -> 54 tok/s on the Coder, output tokens identical.
    // Without them (STRATA_MMVQ_A2=0) it won only on n_out >= 4096 at 1-4 columns, the old gate.
    static const int wide_min = std::getenv("STRATA_MMVQ_WIDE_MIN") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_MIN")) : q6k_a2() ? 0 : 4096;
    static const int wide_maxc = std::getenv("STRATA_MMVQ_WIDE_MAXC") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_MAXC")) : q6k_a2() ? 8 : 4;
    if (g_q6k_wide && ncols >= 1 && ncols <= wide_maxc && n_in % 256 == 0 && n_out >= wide_min) {
        const auto s = strata::q_of(stream);
        switch (ncols) {
            case 1: launch_q6k_wide<1>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 2: launch_q6k_wide<2>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 3: launch_q6k_wide<3>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 4: launch_q6k_wide<4>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 5: launch_q6k_wide<5>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 6: launch_q6k_wide<6>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 7: launch_q6k_wide<7>(weights, x_q8_1, y, n_in, n_out, s); return;
            default: launch_q6k_wide<8>(weights, x_q8_1, y, n_in, n_out, s); return;
        }
    }
    // SYCL port: the row-blocked kernel for the decode shapes (ncols <= 4)
    if (ncols >= 1 && ncols <= 4 && rowwarp_rpw() > 0 && n_in % 256 == 0) {
        dpct::queue_ptr s = strata::q_of(stream);
        const int rpw = rowwarp_rpw();
#define STRATA_RW(NC, RP) if (ncols == NC && rpw == RP) { launch_rowwarp<Q6KTraits, NC, RP>(weights, x_q8_1, y, n_in, n_out, s); return; }
        STRATA_RW(1, 1) STRATA_RW(1, 2) STRATA_RW(1, 4) STRATA_RW(2, 1) STRATA_RW(2, 2) STRATA_RW(2, 4)
        STRATA_RW(3, 1) STRATA_RW(3, 2) STRATA_RW(3, 4) STRATA_RW(4, 1) STRATA_RW(4, 2) STRATA_RW(4, 4)
#undef STRATA_RW
    }

    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q6KTraits)
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<
                dpct_kernel_name<class native_q6_k_mmvq_kernel_4d887f,
                                 dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) * threads, threads),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q6_k_mmvq_kernel<true>(w, x, y, n_in, n_out);
                    });
        }
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class native_q6_k_mmvq_kernel_19484e,
                                         dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, unsigned(n_out)) * threads,
                              threads),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_q6_k_mmvq_kernel<false>(w, x, y, n_in, n_out);
            });
    }
    launch_check();
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

template <int NCOLS>
void q8_0_pair_launch(const Q80Block *w1, const Q80Block *w2, const Q81Block *x,
                      float *y1, float *y2, int n_in, int n_out,
                      dpct::queue_ptr s) {
    using F = SmallTraits<Q80Block, 8>;
    const dpct::dim3 threads(WARP, WARPS);
    if (s26_tsum_on())
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<
            class native_mmvq_multi_kernel_ab153b, F, dpct_kernel_scalar<NCOLS>,
            dpct_kernel_scalar<WARPS>, dpct_kernel_scalar<1>,
            dpct_kernel_scalar<true>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, unsigned(2 * n_out)) * threads,
                              threads),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, true, true>(
                    w1, x, y1, n_in, n_out, w2, y2);
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
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<
            class native_mmvq_multi_kernel_50d9c5, F, dpct_kernel_scalar<NCOLS>,
            dpct_kernel_scalar<WARPS>, dpct_kernel_scalar<1>,
            dpct_kernel_scalar<false>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, unsigned(2 * n_out)) * threads,
                              threads),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, false, true>(
                    w1, x, y1, n_in, n_out, w2, y2);
            });
    }
}

bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream) {
    using F = SmallTraits<Q80Block, 8>;
    // only where both calls would run native_mmvq_multi_kernel<F, ncols, WARPS, 1> (exact layout, not packed)
    if (ggml_type != 8 || !g_multi_exact || ncols < 2 || ncols > 8 || n_out <= 0 || n_in <= 0 || n_in % 32 != 0 ||
        n_in / F::DIV < F::BPI || !w1 || !w2 || !x_q8_1 || !y1 || !y2 || !stream)
        return false;
    const auto& reg = q8_packed_registry();
    if (reg.count(w1) || reg.count(w2)) return false;
    const auto* a = static_cast<const Q80Block*>(w1);
    const auto* b = static_cast<const Q80Block*>(w2);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::q_of(stream);
    switch (ncols) {
        case 2: q8_0_pair_launch<2>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 3: q8_0_pair_launch<3>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 4: q8_0_pair_launch<4>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 5: q8_0_pair_launch<5>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 6: q8_0_pair_launch<6>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 7: q8_0_pair_launch<7>(a, b, x, y1, y2, n_in, n_out, s); break;
        default: q8_0_pair_launch<8>(a, b, x, y1, y2, n_in, n_out, s); break;
    }
    launch_check();
    return true;
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    if (q8_packed_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // STRATA_Q8_PACKED=1 tensors
    if (try_wide32<Wide32Q8>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    if (try_wide32<Wide32IQ4NL>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

} // namespace strata::kernels
