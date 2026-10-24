// src/kernels/cuda/iq_kernels.cu - see include/strata/kernels/iq_kernels.hpp.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at the commit in third_party/ggml/VERSION.txt;
// MIT license, third_party/ggml/LICENSE).  The block structs and codebook grids come from its ggml-common.h,
// included unchanged.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_math.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "s26_tsum.dp.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include "ggml-common-tq.h"                   // APR trellis types (agention llama.cpp fork)
#include "strata/kernels/tq_lut_i8_sycl.hpp"   // int8 copy of the trellis codebook (one global scale) for the dp4a dot

#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <algorithm>
#include <type_traits>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <type_traits>
#include <map>
#include <mutex>
#include <tuple>
#include <cmath>

namespace strata::kernels {
namespace {

void check(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__dpct_inline__ int get_int_b2(const void *x, const int &i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__dpct_inline__ int get_int_b4(const void *x, const int &i32) {
    return ((const int *)x)[i32];
}
__dpct_inline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = sycl::popcount(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__dpct_inline__ sycl::int2 get_int_from_table_16(const int &q4,
                                                 const int8_t *table) {
#if defined(STRATA_HIP_GFX906)
    // AMD: llama.cpp's HIP lookup (vecdotq.cuh) - v_perm_b32 takes 3-bit byte indices, so the low and high halves
    // of the table are looked up and the index MSB picks between them: 4 perms per 8 values.
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(table);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
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
#define ggml_cuda_dp4a(a, b, c) strata::dp4a((a), (b), (c))   // SYCL port: the one dp4a (strata/sycl_math.hpp)

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__dpct_inline__ float vec_dot_q2_0_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
        const int qo = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
        const int qx = dpct::byte_level_permute(qe, qo, 0x5140);
        const int qy = dpct::byte_level_permute(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = bq8_1_chunk->ds[0];
    return d2 * d8 * sumi;
}

// SYCL port: ggml's __vcmpne4 / __vsub4 came out of dpct as per-byte loops (extract, compare, insert, x4); on Xe the
// expert dot kernels were ALU-bound with them (77% XVE active, metrics 2026-09-30). SWAR forms of the two uses:
//   swar_ne4(x): 0xFF in every byte of x that is non-zero (the sign masks: at most one bit per byte)
//   swar_sub4(a, s) with s from swar_ne4: per byte a - s, i.e. a + 1 where the sign byte is set (two's complement
//   of the grid value after the xor: the same bits as __vsub4(a, s) for s in {0x00, 0xFF})
#ifndef STRATA_SWAR
#define STRATA_SWAR 1   // 1: the SWAR forms; 0: dpct's per-byte loops (A/B 2026-09-30: both correct)
#endif
#if STRATA_SWAR
__dpct_inline__ int swar_ne4(unsigned x) {
    x = (x | (x >> 4)) & 0x0F0F0F0Fu;
    x = (x | (x >> 2)) & 0x03030303u;
    x = (x | (x >> 1)) & 0x01010101u;
    return (int) (x * 0xFFu);
}
__dpct_inline__ int swar_sub4(unsigned a, unsigned s) {
    return (int) (((a & 0x7F7F7F7Fu) + (s & 0x01010101u)) ^ (a & 0x80808080u));
}
#else
__dpct_inline__ int swar_ne4(unsigned x) { return dpct::vectorized_binary<sycl::uchar4>(x, 0, std::not_equal_to<>()); }
__dpct_inline__ int swar_sub4(unsigned a, unsigned s) { return dpct::vectorized_binary<sycl::uchar4>(a, s, std::minus<>()); }
#endif

__dpct_inline__ float vec_dot_iq2_xxs_q8_1(const void *__restrict__ vbq,
                                           const block_q8_1 *__restrict__ bq8_1,
                                           const int &kbx, const int &iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = swar_ne4(signs & 0x08040201);
        const int grid0 = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = swar_ne4(signs & 0x80402010);
        const int grid1 = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq2_xs_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    // SYCL port: the four 16-bit codes by shifts. ggml reads them through a uint16_t* to an int2, which is type
    // punning the SYCL device compiler does not honour (non-char aliasing): IQ2_XS products came out garbage
    // (iq_multi_parity, Swift 1.5 IQ2_XS) while the dequantizer, reading the codes directly, was exact.
    const uint32_t q2_lo = (uint32_t) get_int_b2(bq2->qs, iqs + 0), q2_hi = (uint32_t) get_int_b2(bq2->qs, iqs + 1);
    const uint16_t q2[4] = {(uint16_t) q2_lo, (uint16_t) (q2_lo >> 16), (uint16_t) q2_hi, (uint16_t) (q2_hi >> 16)};
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = swar_ne4(signs & 0x08040201);
        const int grid_l = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = swar_ne4(signs & 0x80402010);
        const int grid_h = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq2_s_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = swar_ne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                ((signs_packed_8[l0 / 2] & 0x0C) << 21));
        const int signs1 = swar_ne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                ((signs_packed_8[l0 / 2] & 0xC0) << 17));
        const int grid_l = swar_sub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = swar_sub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq3_xxs_q8_1(const void *__restrict__ vbq,
                                           const block_q8_1 *__restrict__ bq8_1,
                                           const int &kbx, const int &iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const sycl::int2 q3_packed =
        sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::int2 grid_pos =
            sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = swar_ne4(signs & 0x08040201);
        const int grid_l = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = swar_ne4(signs & 0x80402010);
        const int grid_h = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = sycl::vec<sycl::half, 1>(bq3->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq3_s_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const sycl::int2 qs_packed =
        sycl::int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::int2 grid_pos =
            sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                       iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = swar_ne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                ((signs_packed_8[l0 / 2] & 0x0C) << 21));
        const int signs1 = swar_ne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                ((signs_packed_8[l0 / 2] & 0xC0) << 17));
        const int grid_l = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int grid_h = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = sycl::vec<sycl::half, 1>(bq3->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq1_m_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = sycl::vec<sycl::half, 1>(scale.f16)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs].ds[0];
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

// SYCL port: the IQ4_NL codebook in registers and its 2-byte-aligned weights as aligned loads. ggml reads 4 uint32 of
// the codebook from global memory per lookup and each weight int as two 16-bit loads (block_iq4_nl is 18 bytes, its
// qs 2-byte aligned); the expert down projection (IQ4_NL in 39 of 48 layers) was the largest expert kernel. Same ints
// out, so the dots are bitwise unchanged.
__dpct_inline__ uint32_t iq4nl_lut4(uint32_t q4) {   // four nibbles -> four kvalues_iq4nl bytes
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
__dpct_inline__ sycl::int2 iq4nl_pair(int aux) {   // = get_int_from_table_16(aux)
    return sycl::int2((int) iq4nl_lut4((uint32_t) aux & 0x0f0f0f0fu), (int) iq4nl_lut4(((uint32_t) aux >> 4) & 0x0f0f0f0fu));
}
__dpct_inline__ sycl::int2 load8_a2(const uint8_t* p) {   // 8 bytes at a 2-byte-aligned address, as two ints
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0], w1 = q[1];
    if (!(a & 2)) return sycl::int2((int) w0, (int) w1);
    const uint32_t w2 = q[2];   // only when unaligned: it then holds needed bytes, so it cannot cross a page
    return sycl::int2((int) ((w0 >> 16) | (w1 << 16)), (int) ((w1 >> 16) | (w2 << 16)));
}
__dpct_inline__ int load4_a2(const uint8_t* p) {   // 4 bytes at a 2-byte-aligned address (= get_int_b2)
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0];
    if (!(a & 2)) return (int) w0;
    return (int) ((w0 >> 16) | (q[1] << 16));   // the second word holds needed bytes: page-safe
}
// SYCL port: the gate/up kernels can stage their format's codebook grid in local memory (STRATA_GRID_SLM=1, compile
// time). Measured slower and off: copying the grid into each of ~1,200 work-groups per call costs more than the
// lookups save (IQ2_S 52.5 -> 62.7 us, IQ3_S 60.8 -> 70.0, IQ3_XXS 60.3 -> 62.1 per call; unitrace 2026-10-01).
#ifndef STRATA_GRID_SLM
#define STRATA_GRID_SLM 0
#endif
#ifndef STRATA_IQ4NL_FAST
#define STRATA_IQ4NL_FAST 1   // compile-time: -DSTRATA_IQ4NL_FAST=0 for ggml's table/16-bit-load path
#endif

__dpct_inline__ float vec_dot_iq4_nl_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#if STRATA_IQ4NL_FAST
    const sycl::int2 qq = load8_a2(bq4->qs + 4 * iqs);
#endif
#pragma unroll
    for (int l = 0; l < 2; ++l) {
#if STRATA_IQ4NL_FAST
        const sycl::int2 v = iq4nl_pair(l ? qq.y() : qq.x());
#else
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
#endif
        sumi = ggml_cuda_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y(), q8[l + 4], sumi);
    }
    const float d = sycl::vec<sycl::half, 1>(bq4->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1->ds[0];
    return d * sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  The GSQ-RCO IQ3_S
// file keeps one layer's routed gate/up experts in this format.
__dpct_inline__ float vec_dot_iq4_xs_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x(), u0, sumi);
        sumi = ggml_cuda_dp4a(v.y(), u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = sycl::vec<sycl::half, 1>(bq4->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 4].ds[0];
    return d * sumi;
}

// ---------------------------------------------------------------- Unsloth's UD-Q4_K_XL experts
// Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down: llama.cpp's vec_dot_*_q8_1 (vecdotq.cuh, VDR 2 each), transcribed; the
// Q5_1 min term is the one departure (below).  Via eddoursul/Strata 8029fa9 (iq_dot.cuh) and #255 (Q8_0,
// gopinath87607), which agree with llama.cpp and with each other.
constexpr int VDR_Q4_K = 2, VDR_Q5_K = 2, VDR_Q5_1 = 2, VDR_Q5_0 = 2, VDR_Q4_1 = 2, VDR_Q4_0 = 2, VDR_Q8_0 = 2, VDR_Q6_K = 1;

__dpct_inline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);   // the min times the sum of the QUANTIZED activations
    }
    const sycl::float2 dm4f =
        dm4.template convert<float, sycl::rounding_mode::automatic>();
    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}
__dpct_inline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const sycl::half2 &dm5,
    const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm5f =
        dm5.template convert<float, sycl::rounding_mode::automatic>();
    return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}
// the 6-bit scales and mins of the 32-value group pair bq8_offset / 2, branchless (llama.cpp; shared by Q4_K, Q5_K)
__dpct_inline__ void k_scale_min(const uint8_t *scales8, int bq8_offset,
                                 uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*) scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}
__dpct_inline__ float vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;
    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    uint16_t aux[2];
    k_scale_min(bq4_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
__dpct_inline__ float vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*) (bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scale_min(bq5_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}
// Q5_1: llama.cpp's integer chain, but the min term multiplies the sum of the QUANTIZED activations (dp4a with
// 0x01010101, times d8) instead of the q8_1 block's `ds.y`, which our quantizer (like llama.cpp's) fills with the sum
// of the ORIGINAL activations.  That is ggml-cpu's convention (its q8_1 `s` is d * sum(q)) and the one the K-quant
// mins above use; the scaled and the min term then see the same activation (eddoursul/Strata measured 1.1-1.2%
// against 1.9% relative error per expert).  Result: sumi * (d5 * d8) + sumu * (m5 * d8).
// Q5_0: llama.cpp's integer chain verbatim (vecdotq.cuh vec_dot_q5_0_q8_1_impl) -- symmetric (one delta, no
// learned min), so unlike Q5_1 there is no min-term rounding choice to make: the constant -16 per-weight offset
// is accounted for via the q8_1 block's `ds.y` (sum of the ORIGINAL activations), exactly as upstream.
__dpct_inline__ float vec_dot_q5_0_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q5_0* bq5_0 = (const block_q5_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_0; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float d5 = sycl::vec<sycl::half, 1>(bq5_0->d)
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    const float d8 = bq8_1->ds[0];
    const float s8 = bq8_1->ds[1];
    return d5 * (sumi * d8 - (16.0f * VDR_Q5_0 / QI5_0) * s8);
}
// Q4_0 (plain llama-quantize Q4_0 files, e.g. bartowski's: gate/up and most down projections): the codes are centred
// (q - 8 as signed bytes) before the dot, the exact integer form ggml-cpu's q4_0 x q8_0 dot uses, rather than
// llama.cpp's CUDA chain (uncentred codes minus 8 x the q8_1 block's float sum `ds.y`).  On a real Q4_0 expert
// (2560 x 640) the CUDA chain measured 1.4e-2 relative error against the float product, the centred form 5.3e-3;
// with the CUDA chain native_expert_parity failed its 3e-2 limit on two layers of a Q4_0 Flash-Next file.  The
// centring is the same idea as the Q5_1 min term above: the offset is applied to the activations' own int8 codes.
__dpct_inline__ int q4_0_centred(const int v) {
    // Unsigned: (b & 0x08080808) * 0x1E reaches 0xF0F0F0F0, which overflows a signed int - undefined behaviour the
    // sm_60 build (no __dp4a, strata_dp4a fallback) optimised into a wrong value (native_expert_parity gpu rel 1.7e4).
    const uint32_t b = (uint32_t) v ^ 0x08080808u;   // per byte: 0..15 -> (q - 8) as a 4-bit two's complement value
    return (int) (b | ((b & 0x08080808u) * 0x1Eu));  // sign-extend each nibble into its byte (0x08 * 0x1E = 0xF0)
}
__dpct_inline__ float vec_dot_q4_0_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q4_0* bq4 = (const block_q4_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_0; ++i) {
        const int v = get_int_b2(bq4->qs, iqs + i);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 0) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i), sumi);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 4) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i + QI4_0), sumi);
    }
    return sycl::vec<sycl::half, 1>(bq4->d)
               .convert<float, sycl::rounding_mode::automatic>()[0] *
           bq8_1->ds[0] * sumi;
}
// Q4_1 (llama-quantize puts it on a few down projections of its Q4_0 files): the Q5_1 chain above without the fifth
// bit, the same min-term choice (the activations' own int8 codes): sumi * (d4 * d8) + sumu * (m4 * d8).
__dpct_inline__ float vec_dot_q4_1_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q4_1* bq4_1 = (const block_q4_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_1; ++i) {
        const int v = get_int_b4(bq4_1->qs, iqs + i);
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
        sumi = ggml_cuda_dp4a((v >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v >> 4) & 0x0F0F0F0F, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const sycl::float2 dm4 =
        (bq4_1->dm).template convert<float, sycl::rounding_mode::automatic>();
    const float d8 = bq8_1->ds[0];
    return sumi * (dm4.x() * d8) + sumu * (dm4.y() * d8);
}
__dpct_inline__ float vec_dot_q5_1_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const sycl::float2 dm5 =
        (bq5_1->dm).template convert<float, sycl::rounding_mode::automatic>();
    const float d8 = bq8_1->ds[0];
    return sumi * (dm5.x() * d8) + sumu * (dm5.y() * d8);
}
// Q6_K: llama.cpp's integer chain verbatim (vecdotq.cuh vec_dot_q6_K_q8_1_impl_mmvq / vec_dot_q6_K_q8_1) -- signed
// per-16-element scales, the -32 offset folded into the per-byte value, the q8_1 block's ds.low for the scales,
// upstream exactly as pulled (no min-term choice exists: Q6_K has no learned min).
__dpct_inline__ float
vec_dot_q6_K_q8_1_impl_mmvq(const int vl, const int vh,
                            const int *__restrict__ u,
                            const int8_t *__restrict__ scales, const float d,
                            const float *__restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = dpct::vectorized_binary<sycl::char4>(
            vil | vih, 0x20202020, dpct::sub_sat()); // vi = (vil | vih) - 32
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}
__dpct_inline__ float vec_dot_q6_K_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q6_K* bq6_K = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
    const int8_t* scales = bq6_K->scales + scale_offset;
    int u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + 2 * i].ds[0];
    }
    return vec_dot_q6_K_q8_1_impl_mmvq(
        vl, vh, u, scales,
        sycl::vec<sycl::half, 1>(bq6_K->d)
            .convert<float, sycl::rounding_mode::automatic>()[0],
        d8);
}
__dpct_inline__ float vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i)
        sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    const float d8_0 = sycl::vec<sycl::half, 1>(bq8_0->d)
                           .convert<float, sycl::rounding_mode::automatic>()[0],
                d8_1 = bq8_1->ds[0];
    return d8_0 * d8_1 * ((float) sumi);
}

// ---------------------------------------------------------------- APR trellis types (ggml 144 TQ2_T, 145 TQK6, 146 TQK7)
// Transcribed from src/kernels/cuda/iq_kernels.cu (which has them from the agention llama.cpp fork's ggml-cuda/tq.cuh,
// 6d3f2b1b8; format in ggml-common-tq.h).  A 128-value block is 8 lanes of 4 trellis steps (16 values); the dot of lane l
// reads q8_1 block l/2, ints 4(l&1)..  Same integer and float operations in the same order as the CUDA kernels: the
// decode reads the int8 codebook tq_lut_i8 (one global scale) against q8_1 activations.
template<int TY> struct TqInfo;
template<> struct TqInfo<144> { using block_t = block_tq2_t; };   // byte-aligned, no K
template<> struct TqInfo<145> { using block_t = block_tqk6; static constexpr int K = 6; };
template<> struct TqInfo<146> { using block_t = block_tqk7; static constexpr int K = 7; };
__dpct_inline__ float tq_h2f(uint32_t bits) { return (float) sycl::bit_cast<sycl::half>((uint16_t) bits); }
// the four unscaled weights of the step with state s: (a.x, a.y, b.x, b.y)
__dpct_inline__ void tq_step(uint32_t s, float (&w)[4]) {
    const uint32_t* lut = (const uint32_t*) tq_lut_f16;
    const uint32_t x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
    const uint32_t pa = lut[x >> 21], pb = lut[(x >> 10) & 2047u];
    w[0] = tq_h2f(pa & 0xFFFFu);
    w[1] = tq_h2f(pa >> 16);
    w[2] = tq_h2f(pb & 0xFFFFu);
    w[3] = tq_h2f(pb >> 16);
}
// the states of steps 4l..4l+3 of one block (qs is 2-byte aligned: every trellis block has an even size)
template<int TY>
__dpct_inline__ void tq_states4(const uint8_t* qs, int l, uint32_t s[4]) {
    const uint16_t* qs16 = (const uint16_t*) qs;
    if constexpr (TY == 144) {   // s_t = qs[(t + 31) % 32] << 8 | qs[t]
        const uint32_t w01 = qs16[2 * l + 0], w23 = qs16[2 * l + 1];
        const uint32_t prev = qs[(4 * l + 31) & 31];
        const uint32_t b0 = w01 & 0xFF, b1 = w01 >> 8, b2 = w23 & 0xFF, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0;
        s[1] = (b0 << 8) | b1;
        s[2] = (b1 << 8) | b2;
        s[3] = (b2 << 8) | b3;
    } else {
        // step t's state: the 16 stream bits from bit (31 - t)*K, circular over 2K u16 words; steps 4l..4l+3 all
        // lie in the four words from (28 - 4l)*K >> 4
        constexpr int K = TqInfo<TY>::K, nwords = 2 * K;
        const int off3 = (28 - 4 * l) * K, wb = off3 >> 4, r = off3 & 15;
        const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
        const uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) | ((uint64_t) qs16[wi2] << 32) |
                           ((uint64_t) qs16[wi3] << 48);
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = (uint32_t) (v >> (r + (3 - i) * K)) & 0xFFFFu;
    }
}
// The lane's 16 codebook values as 4 ints of packed int8 (tq_load: the weight side only), the activation side in
// tq_apply; vec_dot_tq_q8_1 is apply(load()).  Numerics as in the CUDA file: the int8 codebook is within 0.85% RMS of
// the fp16 one; the prompt path decodes through the fp16 codebook (dq_tq below).
struct TqLane { int q[4]; float d; };
template<int TY>
__dpct_inline__ TqLane tq_load(const void* __restrict__ vbq, int kbx, int iqs) {
    const auto* b = (const typename TqInfo<TY>::block_t*) vbq + kbx;
    uint32_t s[4];
    tq_states4<TY>(b->qs, iqs / 4, s);
    TqLane r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t x = (s[i] & TQ_STATE_MASK) * 0x9e3779b1u;
        r.q[i] = (int) ((uint32_t) tq_lut_i8[x >> 21] | ((uint32_t) tq_lut_i8[(x >> 10) & 2047u] << 16));
    }
    r.d = tq_h2f(b->d);
    return r;
}
__dpct_inline__ float tq_apply(const TqLane& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
    const int l = iqs / 4;
    const block_q8_1* b8 = bq8_1 + l / 2;
    const int* q8 = (const int*) b8->qs + 4 * (l & 1);
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) sumi = ggml_cuda_dp4a(r.q[i], q8[i], sumi);
    return r.d * ((float) b8->ds[0] * TQ_LUT_I8_SCALE) * (float) sumi;
}
template<int TY>
__dpct_inline__ float vec_dot_tq_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                      const int& kbx, const int& iqs) {
    return tq_apply(tq_load<TY>(vbq, kbx, iqs), bq8_1, iqs);
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> 
struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = QI4_K / VDR_Q4_K, step = VDR_Q4_K;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = QI5_K / VDR_Q5_K, step = VDR_Q5_K;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<14> { static constexpr int qk = 256, ipb = QI6_K / VDR_Q6_K, step = VDR_Q6_K;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q6_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = QI5_1 / VDR_Q5_1, step = VDR_Q5_1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<2> { static constexpr int qk = 32, ipb = QI4_0 / VDR_Q4_0, step = VDR_Q4_0;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<3> { static constexpr int qk = 32, ipb = QI4_1 / VDR_Q4_1, step = VDR_Q4_1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<6> { static constexpr int qk = 32, ipb = QI5_0 / VDR_Q5_0, step = VDR_Q5_0;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = QI8_0 / VDR_Q8_0, step = VDR_Q8_0;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };
// APR trellis types: 8 lanes of 4 steps per 128-value block, iqs = 4 * lane
template<> struct Fmt<144> { static constexpr int qk = 128, ipb = 8, step = 4;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_tq_q8_1<144>(v, y, kbx, iqs); } };
template<> struct Fmt<145> { static constexpr int qk = 128, ipb = 8, step = 4;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_tq_q8_1<145>(v, y, kbx, iqs); } };
template<> struct Fmt<146> { static constexpr int qk = 128, ipb = 8, step = 4;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_tq_q8_1<146>(v, y, kbx, iqs); } };

// The formats of each role, one list each so a type cannot be in one switch and missing from another.  Every
// entry is a kernel template for each CUDA architecture of the build, hence two lists rather than one.
#ifdef STRATA_Q6K_EXPERTS   // opt-in build (-DSTRATA_Q6K_EXPERTS=ON): one more instance per kernel, loaded at start
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(14) X(6) X(2) X(3) X(8) X(144) X(145)
#else
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(6) X(2) X(3) X(8) X(144) X(145)
#endif
#define STRATA_D_FMTS(X) X(20) X(23) X(42) X(7) X(6) X(2) X(3) X(8) X(144) X(146)
#define STRATA_MMVQ_FMTS(X) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(7) X(6) X(2) X(3) X(8) X(144) X(145) X(146)

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}

// SYCL port: lanes per row for the grouped expert kernels. A 2560-wide row is 80 (block, part) calls; over 32
// lanes that is 2.5 calls each and a 5-step reduction - the work per sub-group is too small for the fixed cost.
// STRATA_EXPERT_LANES (compile time) lanes share a row; 256 / that many rows per work-group.
#ifndef STRATA_EXPERT_LANES
#define STRATA_EXPERT_LANES 8
#endif
constexpr int kExpertLanes = STRATA_EXPERT_LANES;
constexpr int kExpertRows = 256 / kExpertLanes;
template <int LANES>
__dpct_inline__ float lanes_sum(float v) {
#pragma unroll
    for (int o = LANES / 2; o > 0; o >>= 1)
        v += dpct::experimental::permute_sub_group_by_xor(0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}
template <int TY, int LANES>
__dpct_inline__ float row_dot_lanes(const uint8_t *row, const block_q8_1 *x, int nb, int sub) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = sub; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return lanes_sum<LANES>(s);
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template <int TY>
__dpct_inline__ float row_dot(const uint8_t *row, const block_q8_1 *x, int nb,
                              int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        /*
        DPCT1084: The function call "Fmt::dot" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

template <int TY>
__dpct_inline__ void
mmvq_kernel(const uint8_t *__restrict__ w, size_t row_bytes,
            const block_q8_1 *__restrict__ x, float *__restrict__ y, int n_in,
            int n_out, int ncols) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 4 + item_ct1.get_local_id(1);
    if (row >= n_out) return;
    const int lane = item_ct1.get_local_id(2);
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s =
            row_dot<TY>(wr, x + (size_t)c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- decode once, apply to every column
// The kernels above call Fmt<TY>::dot once per (call, column): each column re-reads the weight words and redoes
// the grid lookups and sign unpacking.  Here each dot is split, as native_mmvq.cu's multi-column traits are, into
// `load` (everything that depends only on the weight: the signed grid words, the integer scales, the fp16 block
// scale as a float) and `apply` (the activation loads, the dp4a chain in the same order, the same integer scale
// step and the same float expression).  `apply(load(...))` does the dot's integer and float operations in the same
// order on the same values, so a column of the kernels below is BITWISE equal to the same column of mmvq_kernel /
// native_gu_kernel / native_down_kernel (iq_multi_parity checks it; STRATA_OLD_IQ_MMVQ=1 keeps the old kernels).
template<int TY> struct Split;
// Formats with a Split below take the decode-once kernels; the others (Q4_K, Q5_K, Q5_1, Q8_0: UD-Q4_K_XL) the
// per-entry ones, which call Fmt<TY>::dot per column exactly as before #242 (the launchers test kSplit at compile
// time, so the multi kernels are never instantiated for a type without a Split).
template<int TY> inline constexpr bool kSplit = false;
template<int TY> inline constexpr bool kStageIqGrid = (TY == 16 || TY == 17 || TY == 22 || TY == 29);

template<> inline constexpr bool kSplit<16> = true;
template<> struct Split<16> {   // IQ2_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        const sycl::uint2 *grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const sycl::uint2 *)s_grid;
        else grid_lut = (const sycl::uint2 *)iq2xxs_grid;
        W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const sycl::uint2 grid_pos = grid_lut[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x08040201, 0, std::not_equal_to<>());
            r.g[k0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x80402010, 0, std::not_equal_to<>());
            r.g[k0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.ls = aux32 >> 27 | 1;
        r.dw = sycl::vec<sycl::half, 1>(bq2->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = sumi * r.ls / 8;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
// IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
struct SplitLs2 {
    struct W { int g[8]; int ls0, ls1; float dw; };
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<17> = true;
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        // SYCL port: the codes by shifts, not through a uint16_t* to an int2 (see vec_dot_iq2_xs_q8_1)
        const uint32_t q2_lo = (uint32_t) get_int_b2(bq2->qs, iqs + 0), q2_hi = (uint32_t) get_int_b2(bq2->qs, iqs + 1);
        const uint16_t q2[4] = {(uint16_t) q2_lo, (uint16_t) (q2_lo >> 16), (uint16_t) q2_hi, (uint16_t) (q2_hi >> 16)};
        const sycl::uint2 *grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const sycl::uint2 *)s_grid;
        else grid_lut = (const sycl::uint2 *)iq2xs_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::uint2 grid_pos = grid_lut[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x08040201, 0, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x80402010, 0, std::not_equal_to<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.dw = sycl::vec<sycl::half, 1>(bq2->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
};
template<> inline constexpr bool kSplit<22> = true;
template<> struct Split<22> : SplitLs2 {   // IQ2_S
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        const uint64_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint64_t*) s_grid;
        else grid_lut = iq2s_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (grid_lut + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                0x00000000, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                0x00000000, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos[0] ^ signs0, signs0, std::minus<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos[1] ^ signs1, signs1, std::minus<>());
        }
        r.dw = sycl::vec<sycl::half, 1>(bq2->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
};
template<> inline constexpr bool kSplit<18> = true;
template<> struct Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const sycl::int2 q3_packed =
            sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos =
                sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x08040201, 0, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x80402010, 0, std::not_equal_to<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.ls = aux32 >> 28;
        r.dw = sycl::vec<sycl::half, 1>(bq3->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<21> = true;
template<> struct Split<21> {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const sycl::int2 qs_packed = sycl::int2(get_int_b2(bq3->qs, iqs + 0),
                                                get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos =
                sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                           iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                0x00000000, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                0x00000000, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = sycl::vec<sycl::half, 1>(bq3->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi *= r.ls;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<29> = true;
template<> struct Split<29> {   // IQ1_M
    struct W { int g[8]; float delta[4]; int sc0, sc1; float dw; };
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
        const int qs_packed = get_int_b4(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const uint32_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = s_grid;
        else grid_lut = iq1s_grid_gpu;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = grid_lut[qs[l0 / 2] | ((qhl & 0x07) << 8)];
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
        const uint16_t* sc = (const uint16_t*) bq1->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        r.dw = sycl::vec<sycl::half, 1>(scale.f16)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
            const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
            sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] += r.delta[l0 / 2] * sumy;
        }
        const float d = r.dw * bq8_1[iqs].ds[0];
        return d * ((sumi[0] + sumf[0]) * r.sc0 + (sumi[1] + sumf[1]) * r.sc1);
    }
};
template<> inline constexpr bool kSplit<20> = true;
template<> struct Split<20> {   // IQ4_NL
    struct W { sycl::int2 v[2]; float dw; };
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = iq4nl_pair(get_int_b2(bq4->qs, iqs + l));   // SYCL port: = get_int_from_table_16(q)
        r.dw = sycl::vec<sycl::half, 1>(bq4->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = ggml_cuda_dp4a(r.v[l].x(), q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(r.v[l].y(), q8[l + 4], sumi);
        }
        const float d = r.dw * bq8_1->ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<23> = true;
template<> struct Split<23> {   // IQ4_XS
    struct W { sycl::int2 v[4]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4nl_pair(get_int_b4(bq4->qs, iqs + j));   // SYCL port: = get_int_from_table_16(q)
        r.ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = sycl::vec<sycl::half, 1>(bq4->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
            const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
            sumi = ggml_cuda_dp4a(r.v[j].x(), u0, sumi);
            sumi = ggml_cuda_dp4a(r.v[j].y(), u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * bq8_1[iqs / 4].ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<42> = true;
template<> struct Split<42> {   // Q2_0
    struct W { int qx[4], qy[4]; float d2; };
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        W r;
        r.d2 = bq2_0->d;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe =
                dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
            const int qo =
                dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
            r.qx[j] = dpct::byte_level_permute(qe, qo, 0x5140);
            r.qy[j] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
            const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
            sumi = ggml_cuda_dp4a(u, r.qx[j], sumi);
            sumi = ggml_cuda_dp4a(v, r.qy[j], sumi);
        }
        const float d8 = bq8_1_chunk->ds[0];
        return r.d2 * d8 * sumi;
    }
};

// APR trellis types: the lane's 16 decoded codebook values (tq_load / tq_apply above)
template<> inline constexpr bool kSplit<144> = true;
template<> inline constexpr bool kSplit<145> = true;
template<> inline constexpr bool kSplit<146> = true;
template<int TY> struct SplitTq {
    using W = TqLane;
    template<bool STAGE_GRID = false>   // (no grid table to stage: the codebook is read where it is)
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        return tq_load<TY>(vbq, kbx, iqs);
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) { return tq_apply(r, bq8_1, iqs); }
};
template<> struct Split<144> : SplitTq<144> {};
template<> struct Split<145> : SplitTq<145> {};
template<> struct Split<146> : SplitTq<146> {};

// One row against the n <= NC activations x + off[0..n) (n >= 1, warp-uniform; offsets in q8_1 blocks, 32-bit to
// spare registers), the whole warp.  Per activation this is row_dot: the same calls k, lane-strided the same way,
// summed in the same order, then the same warp_sum.  Only the weight side moves out of the per-activation loop.
template <int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__dpct_inline__ void
row_dot_multi(const uint8_t *row, const block_q8_1 *x, const int (&off)[NC],
              int n, int nb, int lane, float (&s)[NC],
              const uint32_t *__restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        /*
        DPCT1084: The function call "Split::load" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        const typename S::W w =
            S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (EXACT_N || c < n) s[c] = warp_sum(s[c]);
}

// 8-lane sub-warp row dot for K = nb * ipb == 40 (TD == 20, IQ4_NL with n_ff == 640):
// 4 rows per warp (32 rows per 256-thread block), 100% active lanes, bitwise identical addition tree to row_dot_multi.
template <int TY, int NC, bool EXACT_N = false>
/*
DPCT1110: The total declared local variable size in device function
row_dot_40_sub8 exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void row_dot_40_sub8(const uint8_t *row, const block_q8_1 *x,
                                     const int (&off)[NC], int n, int t,
                                     float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = t; k < 40; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (EXACT_N || c < n) {
                float s16 = 0.0f;
                s16 += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
                /*
                DPCT1013: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                s[c] = s[c] + s16;
            }
        }
    }
    float s8[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s8[c] = 0.0f;
    {
        const int k = t + 8, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s8[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    {
        const int k = t + 24, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (EXACT_N || c < n) {
                float s24 = 0.0f;
                s24 += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
                /*
                DPCT1013: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                s[c] = s[c] + s8[c] + s24;
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 4; o > 0; o >>= 1) s[c] +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), s[c],
                    o);
        }
    }
}

// 16-lane sub-warp row dot for K = nb * ipb == 20 (TD == 42, Q2_0 with n_ff == 640):
// 2 rows per warp (16 rows per 256-thread block), bitwise identical addition tree to row_dot_multi.
template <int TY, int NC, bool EXACT_N = false>
/*
DPCT1110: The total declared local variable size in device function
row_dot_20_sub16 exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void row_dot_20_sub16(const uint8_t *row, const block_q8_1 *x,
                                      const int (&off)[NC], int n, int t,
                                      float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    {
        const int k = t, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    if (t < 4) {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            /*
            DPCT1013: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            s[c] = s[c] + s16[c];
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 8; o > 0; o >>= 1) s[c] +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), s[c],
                    o);
        }
    }
}

// 16-lane sub-warp row dot for K = nb * ipb == 80 (all kSplit gate/up formats with n_embd == 2560):
// 2 rows per warp (16 rows per 256-thread block), 100% active lanes, bitwise identical addition tree to row_dot_multi.
template <int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
/*
DPCT1110: The total declared local variable size in device function
row_dot_80_sub16 exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
row_dot_80_sub16(const uint8_t *row, const block_q8_1 *x, const int (&off)[NC],
                 int n, int t, float (&s)[NC],
                 const uint32_t *__restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = t; k < 80; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        /*
        DPCT1084: The function call "Split::load" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        const typename S::W w =
            S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    for (int k = t + 16; k < 64; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        /*
        DPCT1084: The function call "Split::load" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        const typename S::W w =
            S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            /*
            DPCT1013: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            s[c] = s[c] + s16[c];
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 8; o > 0; o >>= 1) s[c] +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), s[c],
                    o);
        }
    }
}

template <int TY, bool STAGE_GRID = kStageIqGrid<TY>>
__dpct_inline__ const uint32_t *
stage_iq_grid(uint32_t *s_buf, int tid, int nthreads) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if constexpr (!STAGE_GRID) {
        return nullptr;
    } else if constexpr (TY == 16) {
        sycl::uint2 *dst = (sycl::uint2 *)s_buf;
        const sycl::uint2 *src = (const sycl::uint2 *)iq2xxs_grid;
#pragma unroll
        for (int i = tid; i < 256; i += nthreads) dst[i] = src[i];
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        return s_buf;
    } else if constexpr (TY == 17) {
        sycl::uint2 *dst = (sycl::uint2 *)s_buf;
        const sycl::uint2 *src = (const sycl::uint2 *)iq2xs_grid;
#pragma unroll
        for (int i = tid; i < 512; i += nthreads) dst[i] = src[i];
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        return s_buf;
    } else if constexpr (TY == 22) {
        sycl::uint2 *dst = (sycl::uint2 *)s_buf;
        const sycl::uint2 *src = (const sycl::uint2 *)iq2s_grid;
#pragma unroll
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        return s_buf;
    } else if constexpr (TY == 29) {
        sycl::uint2 *dst = (sycl::uint2 *)s_buf;
        const sycl::uint2 *src = (const sycl::uint2 *)iq1s_grid_gpu;
#pragma unroll
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        return s_buf;
    } else {
        return nullptr;
    }
}

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
struct IqGridWords {
    static constexpr int value = !STAGE_GRID ? 4 : (TY == 22 || TY == 29) ? 2048 : (TY == 17) ? 1024 : (TY == 16) ? 512 : 4;
};

// mmvq_kernel with the columns taken NC at a time.  After warp_sum every lane holds the same sum, so lane c stores
// column c.
// #778: `__shared__ alignas(16)` does not compile for gfx1030 (HIP); the attribute after the declarator does, but nvcc
// with MSVC as host does not take it.  Same layout either way.
#if defined(__HIPCC__)
#define STRATA_SHARED_ALIGN16_U32(name, ...) __shared__ uint32_t name[__VA_ARGS__] __attribute__((aligned(16)))
#else
#define STRATA_SHARED_ALIGN16_U32(name, ...) alignas(16) uint32_t name[__VA_ARGS__]
#endif
template <int TY, int NC, bool STAGE_GRID = kStageIqGrid<TY>,
          bool EXACT_N = false>
__dpct_inline__ void
mmvq_multi_kernel(const uint8_t *__restrict__ w, size_t row_bytes,
                  const block_q8_1 *__restrict__ x, float *__restrict__ y,
                  int n_in, int n_out, int ncols) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_grid_buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    uint32_t[IqGridWords<TY, STAGE_GRID>::value]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const uint32_t *s_grid = stage_iq_grid<TY, STAGE_GRID>(
        s_grid_buf, item_ct1.get_local_id(1) * 32 + item_ct1.get_local_id(2),
        128);
    const int row = item_ct1.get_group(2) * 4 + item_ct1.get_local_id(1);
    if (row >= n_out) return;
    const int lane = item_ct1.get_local_id(2);
    const int nb = n_in / Fmt<TY>::qk, xb = n_in / 32;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    if constexpr (EXACT_N) {
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = c * xb;
        float s[NC];
        row_dot_multi<TY, NC, true, STAGE_GRID>(wr, x, off, NC, nb, lane, s, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (lane == c) y[(size_t) c * n_out + row] = s[c];
    } else {
        for (int c0 = 0; c0 < ncols; c0 += NC) {
            const int n = sycl::min(NC, ncols - c0);
            int off[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c)
                off[c] = (c0 + sycl::min(c, n - 1)) * xb;
            float s[NC];
            row_dot_multi<TY, NC, false, STAGE_GRID>(wr, x, off, n, nb, lane, s, s_grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n && lane == c) y[(size_t) (c0 + c) * n_out + row] = s[c];
        }
    }
}

// ---------------------------------------------------------------- grouped native experts

// ---------------------------------------------------------------- SYCL port: multi-entry dots
//
// A window routes the same expert for several tokens (up to 6). The single-entry dot above recomputes the
// dequantised weights - grid lookups, sign unpacking, the byte-wise compare/xor/subtract emulations - once per
// token; only the dp4a against the activation and the scale differ. Multi<TY> splits each format into `prep`
// (weights, once) and `acts` + `finish` (per token). NW packed int32s of weights per call.
struct MultiW { int w[8]; int a = 0, b = 0; float d = 0.f; };

template <int TY> struct Multi { static constexpr bool has = false; static constexpr int NW = 0; };

template <> struct Multi<18> {   // iq3_xxs
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* grid = nullptr) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
#if STRATA_IQ4NL_FAST
        const sycl::int2 q3_packed = load8_a2(bq3->qs + 4 * iqs);
        const uint32_t aux32 = (uint32_t) load4_a2(bq3->qs + 4 * (QK_K / 16 + iqs / 2));
#else
        const sycl::int2 q3_packed = sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#endif
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t* G = (const uint32_t*) grid;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos = G ? sycl::int2(G[q3[l0 + 0]], G[q3[l0 + 1]])
                                          : sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = swar_ne4(signs & 0x08040201);
            const int signs1 = swar_ne4(signs & 0x80402010);
            m.w[l0 + 0] = swar_sub4(grid_pos.x() ^ signs0, signs0);
            m.w[l0 + 1] = swar_sub4(grid_pos.y() ^ signs1, signs1);
        }
        m.a = (int) (aux32 >> 28);
        m.d = sycl::vec<sycl::half, 1>(bq3->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        const int sumi = s0 + s1;
        return m.d * ds * (float) ((m.a * sumi + sumi / 2) / 2);
    }
};

template <> struct Multi<22> {   // iq2_s
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* grid = nullptr) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
#if STRATA_IQ4NL_FAST
        const int qs_packed = load4_a2(bq2->qs + 4 * (iqs / 2));
        const int signs_packed_32 = load4_a2(bq2->qs + 4 * (QK_K / 32 + iqs / 2));
#else
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
#endif
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const uint8_t* sp = (const uint8_t*) &signs_packed_32;
        const uint64_t* G = (const uint64_t*) grid;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int gi = qs[l0 / 2] | ((qh << (8 - l0)) & 0x300);
            const int* grid_pos = G ? (const int*) (G + gi) : (const int*) (iq2s_grid + gi);
            const int signs0 = swar_ne4(((sp[l0 / 2] & 0x03) << 7) | ((sp[l0 / 2] & 0x0C) << 21));
            const int signs1 = swar_ne4(((sp[l0 / 2] & 0x30) << 3) | ((sp[l0 / 2] & 0xC0) << 17));
            m.w[l0 + 0] = swar_sub4(grid_pos[0] ^ signs0, signs0);
            m.w[l0 + 1] = swar_sub4(grid_pos[1] ^ signs1, signs1);
        }
        m.a = bq2->scales[iqs / 2] & 0x0F;
        m.b = bq2->scales[iqs / 2] >> 4;
        m.d = sycl::vec<sycl::half, 1>(bq2->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        return m.d * ds * (float) ((s0 * m.a + s1 * m.b + (s0 + s1) / 2) / 4);
    }
};

template <> struct Multi<21> {   // iq3_s
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* grid = nullptr) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
#if STRATA_IQ4NL_FAST
        const sycl::int2 qs_packed = load8_a2(bq3->qs + 4 * iqs);
        const int signs_packed_32 = load4_a2(bq3->signs + 4 * (iqs / 2));
#else
        const sycl::int2 qs_packed = sycl::int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
#endif
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const uint8_t* sp = (const uint8_t*) &signs_packed_32;
        const uint32_t* G = (const uint32_t*) grid;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int i0 = qs[l0 + 0] | ((qh << (8 - l0)) & 0x100), i1 = qs[l0 + 1] | ((qh << (7 - l0)) & 0x100);
            const sycl::int2 grid_pos = G ? sycl::int2(G[i0], G[i1]) : sycl::int2(iq3s_grid[i0], iq3s_grid[i1]);
            const int signs0 = swar_ne4(((sp[l0 / 2] & 0x03) << 7) | ((sp[l0 / 2] & 0x0C) << 21));
            const int signs1 = swar_ne4(((sp[l0 / 2] & 0x30) << 3) | ((sp[l0 / 2] & 0xC0) << 17));
            m.w[l0 + 0] = swar_sub4(grid_pos.x() ^ signs0, signs0);
            m.w[l0 + 1] = swar_sub4(grid_pos.y() ^ signs1, signs1);
        }
        m.a = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        m.d = sycl::vec<sycl::half, 1>(bq3->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        return m.d * ds * (float) ((s0 + s1) * m.a);
    }
};

template <> struct Multi<20> {   // iq4_nl (down)
    static constexpr bool has = true; static constexpr int NW = 4;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* = nullptr) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
#if STRATA_IQ4NL_FAST
        const sycl::int2 qq = load8_a2(bq4->qs + 4 * iqs);
#endif
#pragma unroll
        for (int l = 0; l < 2; ++l) {
#if STRATA_IQ4NL_FAST
            const sycl::int2 v = iq4nl_pair(l ? qq.y() : qq.x());
#else
            const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
            const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
#endif
            m.w[2 * l + 0] = v.x();
            m.w[2 * l + 1] = v.y();
        }
        m.d = sycl::vec<sycl::half, 1>(bq4->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
        const int* q8 = (const int*) y->qs + iqs;
#pragma unroll
        for (int l = 0; l < 2; ++l) { u[2 * l + 0] = q8[l + 0]; u[2 * l + 1] = q8[l + 4]; }
        ds = y->ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) { return m.d * ds * (float) (s0 + s1); }
};

template <> struct Multi<42> {   // q2_0 (down)
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* = nullptr) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
            const int qo = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
            m.w[2 * j + 0] = dpct::byte_level_permute(qe, qo, 0x5140);
            m.w[2 * j + 1] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        m.d = bq2_0->d;
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
        const block_q8_1* chunk = y + iqs;
#pragma unroll
        for (int j = 0; j < 4; ++j) { u[2 * j + 0] = get_int_b4(chunk->qs, j * 2 + 0); u[2 * j + 1] = get_int_b4(chunk->qs, j * 2 + 1); }
        ds = chunk->ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) { return m.d * ds * (float) (s0 + s1); }
};

// MODEL-UPDATE (Intel): Unsloth's formats in the lane kernels' multi-entry path.  Each Fmt<TY>::dot of the K-quants,
// Q5_1 and Q8_0 split into `load` (every read of the weight block and what is computed from it alone) and `apply`
// (the activation reads and the rest of the dot), the same statements in the same order as vec_dot_*_q8_1 above, so
// apply(load(...)) is the dot's value bit for bit; a group's 2-4 entries then read and decode each weight word once
// instead of once per entry.  Every entry keeps its own lane-strided k order and lanes_sum: the outputs are bitwise
// those of row_dot_lanes (native_grouped_parity --kq checks it).  STRATA_KQ_SPLIT=0: the per-entry kernels.
template <int TY> struct KqSplit { static constexpr bool has = false; };
// APR trellis types: tq_load decodes the lane's 16 codebook values once, tq_apply is the activation side per entry (the same
// operations as vec_dot_tq_q8_1, so the outputs are the per-entry kernels' bits)
template <int TY> struct KqSplitTq {
    static constexpr bool has = true;
    using W = TqLane;
    __dpct_inline__ static W load(const void* __restrict__ vbq, int kbx, int iqs) { return tq_load<TY>(vbq, kbx, iqs); }
    __dpct_inline__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) { return tq_apply(r, bq8_1, iqs); }
};
template <> struct KqSplit<144> : KqSplitTq<144> {};
template <> struct KqSplit<145> : KqSplitTq<145> {};
template <> struct KqSplit<146> : KqSplitTq<146> {};
template <> struct KqSplit<12> {   // Q4_K (vec_dot_q4_K_q8_1)
    static constexpr bool has = true;
    struct W { int v[2]; uint16_t aux[2]; sycl::half2 dm; };
    __dpct_inline__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q4_K* b = (const block_q4_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        const int* q4 = (const int*) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = q4[0];
        r.v[1] = q4[4];
        k_scale_min(b->scales, bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    __dpct_inline__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        int u[2 * QR4_K];
        float d8[QR4_K];
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        const uint8_t* m = sc + 2;
        return vec_dot_q4_K_q8_1_impl_vmmq(r.v, u, sc, m, r.dm, d8);
    }
};
template <> struct KqSplit<13> {   // Q5_K (vec_dot_q5_K_q8_1)
    static constexpr bool has = true;
    struct W { int vl[2]; int vh[2]; uint16_t aux[2]; sycl::half2 dm; };
    __dpct_inline__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q5_K* b = (const block_q5_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        const int* ql = (const int*) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = (const int*) (b->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> bq8_offset;
        r.vh[1] = qh[4] >> bq8_offset;
        k_scale_min(b->scales, bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    __dpct_inline__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        int u[2 * QR5_K];
        float d8[QR5_K];
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        const uint8_t* m = sc + 2;
        return vec_dot_q5_K_q8_1_impl_vmmq(r.vl, r.vh, u, sc, m, r.dm, d8);
    }
};
template <> struct KqSplit<14> {   // Q6_K (vec_dot_q6_K_q8_1)
    static constexpr bool has = true;
    struct W { int vl, vh; int8_t sc[5]; float d; };
    __dpct_inline__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q6_K* bq6_K = (const block_q6_K*) vbq + kbx;
        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
        W r;
        r.vl = get_int_b2(bq6_K->ql, iqs);
        r.vh = get_int_b2(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
        const int8_t* scales = bq6_K->scales + scale_offset;
        r.sc[0] = scales[0];
        r.sc[4] = scales[4];   // the impl reads scales[4 * i], i < QR6_K = 2
        r.d = sycl::vec<sycl::half, 1>(bq6_K->d).convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    __dpct_inline__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        int u[QR6_K];
        float d8[QR6_K];
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            u[i] = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
            d8[i] = bq8_1[bq8_offset + 2 * i].ds[0];
        }
        return vec_dot_q6_K_q8_1_impl_mmvq(r.vl, r.vh, u, r.sc, r.d, d8);
    }
};
template <> struct KqSplit<7> {   // Q5_1 (vec_dot_q5_1_q8_1)
    static constexpr bool has = true;
    struct W { int vl[VDR_Q5_1]; int qh; sycl::half2 dm; };
    __dpct_inline__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q5_1* b = (const block_q5_1*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) r.vl[i] = get_int_b4(b->qs, iqs + i);
        r.qh = get_int_b4(b->qh, 0);
        r.dm = b->dm;
        return r;
    }
    __dpct_inline__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0, sumu = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) {
            const int vl = r.vl[i];
            const int vh = r.qh >> (4 * (iqs + i));
            const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
            int vi0 = (vl >> 0) & 0x0F0F0F0F;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            sumi = ggml_cuda_dp4a(vi0, u0, sumi);
            int vi1 = (vl >> 4) & 0x0F0F0F0F;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            sumi = ggml_cuda_dp4a(vi1, u1, sumi);
            sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
        }
        const sycl::float2 dm5 = (r.dm).template convert<float, sycl::rounding_mode::automatic>();
        const float d8 = bq8_1->ds[0];
        return sumi * (dm5.x() * d8) + sumu * (dm5.y() * d8);
    }
};
template <> struct KqSplit<8> {   // Q8_0 (vec_dot_q8_0_q8_1)
    static constexpr bool has = true;
    struct W { int q[VDR_Q8_0]; float d; };
    __dpct_inline__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q8_0* b = (const block_q8_0*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) r.q[i] = get_int_b2(b->qs, iqs + i);
        r.d = sycl::vec<sycl::half, 1>(b->d).convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    __dpct_inline__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) sumi = ggml_cuda_dp4a(r.q[i], get_int_b4(bq8_1->qs, iqs + i), sumi);
        const float d8_1 = bq8_1->ds[0];
        return r.d * d8_1 * ((float) sumi);
    }
};
// the lane kernels' multi-entry path: the IQ formats' Multi always, the K-quant splits unless KM is false
template <int TY, bool KM> inline constexpr bool kEntryMulti = Multi<TY>::has || (KM && KqSplit<TY>::has);

// One row against E activations at once, LANES lanes per row: the K-quant / Q5_1 / Q8_0 split (KqSplit).
template <int TY, int LANES, int E>
__dpct_inline__ void row_dot_ksplit(const uint8_t* row, const block_q8_1* const* xs, int nb, int sub, float* out) {
    using F = Fmt<TY>;
    using K = KqSplit<TY>;
    float s[E];
#pragma unroll
    for (int e = 0; e < E; ++e) s[e] = 0.f;
    for (int k = sub; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename K::W w = K::load(row, kbx, iqs);
#pragma unroll
        for (int e = 0; e < E; ++e) s[e] += K::apply(w, xs[e] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int e = 0; e < E; ++e) out[e] = lanes_sum<LANES>(s[e]);
}

// One row against E activations at once, LANES lanes per row: the IQ formats' Multi.
template <int TY, int LANES, int E>
__dpct_inline__ void row_dot_multi_iq(const uint8_t* row, const block_q8_1* const* xs, int nb, int sub, float* out,
                                      const void* grid = nullptr) {
    using F = Fmt<TY>;
    using M = Multi<TY>;
    float s[E];
#pragma unroll
    for (int e = 0; e < E; ++e) s[e] = 0.f;
    for (int k = sub; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        MultiW m;
        M::prep(row, kbx, iqs, m, grid);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            int u[8];
            float ds;
            M::acts(xs[e] + kbx * (F::qk / 32), iqs, u, ds);
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < M::NW / 2; ++j) s0 = ggml_cuda_dp4a(m.w[j], u[j], s0);
#pragma unroll
            for (int j = M::NW / 2; j < M::NW; ++j) s1 = ggml_cuda_dp4a(m.w[j], u[j], s1);
            s[e] += M::finish(s0, s1, m, ds);
        }
    }
#pragma unroll
    for (int e = 0; e < E; ++e) out[e] = lanes_sum<LANES>(s[e]);
}

template <int TY, int LANES, int E>
__dpct_inline__ void row_dot_multi(const uint8_t* row, const block_q8_1* const* xs, int nb, int sub, float* out,
                                   const void* grid = nullptr) {
    if constexpr (KqSplit<TY>::has) row_dot_ksplit<TY, LANES, E>(row, xs, nb, sub, out);
    else row_dot_multi_iq<TY, LANES, E>(row, xs, nb, sub, out, grid);
}

// The entries e0..e1 of one row: chunks of 4 through the multi-entry dot, the rest one at a time.
template <int TY, int LANES, bool KM = true>
__dpct_inline__ void row_entries(const uint8_t* wr, const block_q8_1* x, int x_stride, const int32_t* ent_idx,
                                 int e0, int e1, int nb, int sub, float* dst, size_t dst_stride,
                                 const void* grid = nullptr) {
    int e = e0;
    if constexpr (kEntryMulti<TY, KM>) {
        for (; e + 4 <= e1; e += 4) {
            const block_q8_1* xs[4] = {x + (size_t) ent_idx[e] * x_stride, x + (size_t) ent_idx[e + 1] * x_stride,
                                       x + (size_t) ent_idx[e + 2] * x_stride, x + (size_t) ent_idx[e + 3] * x_stride};
            float o[4];
            row_dot_multi<TY, LANES, 4>(wr, xs, nb, sub, o, grid);
            if (sub == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) dst[(size_t) (e + i) * dst_stride] = o[i];
        }
        if (e + 2 <= e1) {
            const block_q8_1* xs[2] = {x + (size_t) ent_idx[e] * x_stride, x + (size_t) ent_idx[e + 1] * x_stride};
            float o[2];
            row_dot_multi<TY, LANES, 2>(wr, xs, nb, sub, o, grid);
            if (sub == 0) { dst[(size_t) e * dst_stride] = o[0]; dst[(size_t) (e + 1) * dst_stride] = o[1]; }
            e += 2;
        }
    }
    for (; e < e1; ++e) {
        const float v = row_dot_lanes<TY, LANES>(wr, x + (size_t) ent_idx[e] * x_stride, nb, sub);
        if (sub == 0) dst[(size_t) e * dst_stride] = v;
    }
}

constexpr int GU_ROWS = 8;     // rows per block (one warp each)

template <int TG, int LN = kExpertLanes, bool KM = true>
__dpct_inline__ void native_gu_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_tok, const block_q8_1 *__restrict__ xq,
    NativeExpertLayout L, float *__restrict__ gate, float *__restrict__ up) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    // #363: group g is computed by block row g % gridDim.y (a launch sized for fewer rows than possible groups strides);
    // a block row with no group returns before the barrier below (uniform over the work-group)
    const int ng = *n_groups;
    if ((int) item_ct1.get_group(1) >= ng) return;
    const void* grid = nullptr;
#if STRATA_GRID_SLM
    if constexpr (TG == 18 || TG == 21 || TG == 22) {
        using GT = std::conditional_t<TG == 22, uint64_t, uint32_t>;
        constexpr int GN = TG == 22 ? 1024 : TG == 21 ? 512 : 256;
        auto& sg = *sycl::ext::oneapi::group_local_memory_for_overwrite<GT[GN]>(item_ct1.get_group());
        for (int i = (int) item_ct1.get_local_id(2); i < GN; i += (int) item_ct1.get_local_range(2)) {
            if constexpr (TG == 22) sg[i] = iq2s_grid[i];
            else if constexpr (TG == 21) sg[i] = iq3s_grid[i];
            else sg[i] = iq3xxs_grid[i];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        grid = &sg[0];
    }
#endif
    const int rib = item_ct1.get_local_id(2) / LN,
              sub = item_ct1.get_local_id(2) % LN;
    const int row = item_ct1.get_group(2) * (256 / LN) + rib; // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    for (int g = (int) item_ct1.get_group(1); g < ng; g += (int) item_ct1.get_group_range(1)) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        row_entries<TG, LN, KM>(wr, xq, xb, ent_tok, e0, e1, nb, sub, (is_up ? up : gate) + r, (size_t) L.n_ff, grid);
    }
}

// native_gu_kernel with the group's entries taken GRP_NC at a time, each weight part decoded once per pass.
// A group has at most one entry per token of the window (kVerifyMaxT = 8; setup writes --spec 4, and a split window
// has halves of <= 4), so 4 takes such windows in one pass; 8 would take ~64-80 registers against ~48 (ptxas -v).
constexpr int GRP_NC = 4;

template <int TG, bool STAGE_GRID = kStageIqGrid<TG>, bool SUB16 = true>
/*
DPCT1110: The total declared local variable size in device function
native_gu_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void native_gu_multi_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_tok, const block_q8_1 *__restrict__ xq,
    NativeExpertLayout L, float *__restrict__ gate, float *__restrict__ up) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ng = *n_groups;
    if (item_ct1.get_group(1) >= ng) return;
    auto &s_grid_buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        uint32_t[IqGridWords<TG, STAGE_GRID>::value]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const uint32_t *s_grid = stage_iq_grid<TG, STAGE_GRID>(
        s_grid_buf, item_ct1.get_local_id(2), 256);
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    if constexpr (SUB16) {
        if (xb == 80 && nb * Fmt<TG>::ipb == 80) {   // row_dot_80_sub16 covers exactly 80 dots (not the 128-value TQ blocks)
            const int subwarp = item_ct1.get_local_id(2) >> 4,
                      t = item_ct1.get_local_id(2) & 15;
            const int row = item_ct1.get_group(2) * 16 + subwarp; // 0 .. 2*n_ff
            if (row >= 2 * L.n_ff) return;
            const bool is_up = row >= L.n_ff;
            const int r = is_up ? row - (int) L.n_ff : row;
            const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
            float* dst = is_up ? up : gate;
            for (int g = item_ct1.get_group(1); g < ng;
                 g += item_ct1.get_group_range(1)) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = sycl::min(GRP_NC, e1 - e);
                    if (n == 1) {
                        const int off1[1] = { ent_tok[e] * 80 };
                        float s1[1];
                        row_dot_80_sub16<TG, 1, true, STAGE_GRID>(
                            wr, xq, off1, 1, t, s1, s_grid);
                        if (t == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
                    } else if (n == 2) {
                        const int off2[2] = { ent_tok[e] * 80, ent_tok[e + 1] * 80 };
                        float s2[2];
                        row_dot_80_sub16<TG, 2, true, STAGE_GRID>(
                            wr, xq, off2, 2, t, s2, s_grid);
                        if (t < 2) dst[(size_t) (e + t) * L.n_ff + r] = s2[t];
                    } else if (n == 3) {
                        const int off3[3] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80 };
                        float s3[3];
                        row_dot_80_sub16<TG, 3, true, STAGE_GRID>(
                            wr, xq, off3, 3, t, s3, s_grid);
                        if (t < 3) dst[(size_t) (e + t) * L.n_ff + r] = s3[t];
                    } else {
                        const int off4[4] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80, ent_tok[e + 3] * 80 };
                        float s4[4];
                        row_dot_80_sub16<TG, 4, true, STAGE_GRID>(
                            wr, xq, off4, 4, t, s4, s_grid);
                        if (t < 4) dst[(size_t) (e + t) * L.n_ff + r] = s4[t];
                    }
                }
            }
            return;
        }
    }
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int row = item_ct1.get_group(2) * GU_ROWS + warp; // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    float* dst = is_up ? up : gate;
    for (int g = item_ct1.get_group(1); g < ng;
         g +=
         item_ct1.get_group_range(1)) { // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = sycl::min(GRP_NC, e1 - e);
            if (n == 1) {
                const int off1[1] = { ent_tok[e] * xb };
                float s1[1];
                row_dot_multi<TG, 1, true, STAGE_GRID>(
                    wr, xq, off1, 1, nb, lane, s1, s_grid);
                if (lane == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { ent_tok[e] * xb, ent_tok[e + 1] * xb };
                float s2[2];
                row_dot_multi<TG, 2, true, STAGE_GRID>(
                    wr, xq, off2, 2, nb, lane, s2, s_grid);
                if (lane < 2) dst[(size_t) (e + lane) * L.n_ff + r] = s2[lane];
            } else if (n == 3) {
                const int off3[3] = { ent_tok[e] * xb, ent_tok[e + 1] * xb, ent_tok[e + 2] * xb };
                float s3[3];
                row_dot_multi<TG, 3, true, STAGE_GRID>(
                    wr, xq, off3, 3, nb, lane, s3, s_grid);
                if (lane < 3) dst[(size_t) (e + lane) * L.n_ff + r] = s3[lane];
            } else {
                const int off4[4] = { ent_tok[e] * xb, ent_tok[e + 1] * xb, ent_tok[e + 2] * xb, ent_tok[e + 3] * xb };
                float s4[4];
                row_dot_multi<TG, 4, true, STAGE_GRID>(
                    wr, xq, off4, 4, nb, lane, s4, s_grid);
                if (lane < 4) dst[(size_t) (e + lane) * L.n_ff + r] = s4[lane];
            }
        }
    }
}

// APR (prism.hadamard): y = H_block (signs * x) per block of each row, block = the work-group size (a power of two <= 1024),
// one work-group per (row block, row).  The butterflies are the CPU's (hadamard.cpp) in the same order, so the
// rotated values are the same floats.
template <typename T> __dpct_inline__ float ld_f(const T* p) { return (float) *p; }
template <> __dpct_inline__ float ld_f<sycl::half>(const sycl::half* p) { return (float) *p; }
template <typename T> __dpct_inline__ void st_f(T* p, float v) { *p = v; }
template <> __dpct_inline__ void st_f<sycl::half>(sycl::half* p, float v) {
    *p = sycl::vec<float, 1>(v).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}
template <typename T>
__dpct_inline__ void hadamard_rows_kernel(const T* __restrict__ x, T* __restrict__ y, int64_t width,
                                          const float* __restrict__ signs, float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto& hbuf = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(item_ct1.get_group());
    const int n = (int) item_ct1.get_local_range(2), i = (int) item_ct1.get_local_id(2);
    const int64_t col = (int64_t) item_ct1.get_group(2) * n + i, at = (int64_t) item_ct1.get_group(1) * width + col;
    float v = ld_f<T>(x + at);
    if (signs) v *= signs[col];
    hbuf[i] = v;
    for (int len = 1; len < n; len <<= 1) {
        sycl::group_barrier(item_ct1.get_group());
        if ((i & len) == 0) {
            const float a = hbuf[i], c = hbuf[i + len];
            hbuf[i] = a + c;
            hbuf[i + len] = a - c;
        }
    }
    sycl::group_barrier(item_ct1.get_group());
    st_f<T>(y + at, hbuf[i] * scale);
}
template <typename T> class hadamard_rows_name;
template <typename T>
void launch_hadamard_rows(const T* x, T* y, int64_t rows, int64_t width, int block, const float* signs, dpct::queue_ptr s) {
    const float scale = 1.0f / std::sqrt((float) block);
    s->parallel_for<hadamard_rows_name<T>>(
        sycl::nd_range<3>(sycl::range<3>(1, (size_t) rows, (size_t) (width / block) * (size_t) block),
                          sycl::range<3>(1, 1, (size_t) block)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { hadamard_rows_kernel<T>(x, y, width, signs, scale); });
}

__dpct_inline__ void swiglu_entries_kernel(const float *__restrict__ gate,
                                           const float *__restrict__ up,
                                           float *__restrict__ h, long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + sycl::native::exp(-g))) * up[i];
}

template <int TD, int LN = kExpertLanes, bool KM = true>
__dpct_inline__ void native_down_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_dst, const block_q8_1 *__restrict__ hq,
    NativeExpertLayout L, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ng = *n_groups;   // #363: the group stride, as native_gu_kernel
    const int rib = item_ct1.get_local_id(2) / LN,
              sub = item_ct1.get_local_id(2) % LN;
    const int r = item_ct1.get_group(2) * (256 / LN) + rib;
    if (r >= L.n_embd) return;
    const size_t off = L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    // down: entry e's activation is row e of hq (identity index), its output row is ent_dst[e]
    for (int g = (int) item_ct1.get_group(1); g < ng; g += (int) item_ct1.get_group_range(1)) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        int e = e0;
        if constexpr (kEntryMulti<TD, KM>) {
            for (; e + 4 <= e1; e += 4) {
                const block_q8_1* xs[4] = {hq + (size_t) e * hb, hq + (size_t) (e + 1) * hb, hq + (size_t) (e + 2) * hb, hq + (size_t) (e + 3) * hb};
                float o[4];
                row_dot_multi<TD, LN, 4>(wr, xs, nb, sub, o);
                if (sub == 0)
#pragma unroll
                    for (int i = 0; i < 4; ++i) out[(size_t) ent_dst[e + i] * L.n_embd + r] = o[i];
            }
            if (e + 2 <= e1) {
                const block_q8_1* xs[2] = {hq + (size_t) e * hb, hq + (size_t) (e + 1) * hb};
                float o[2];
                row_dot_multi<TD, LN, 2>(wr, xs, nb, sub, o);
                if (sub == 0) { out[(size_t) ent_dst[e] * L.n_embd + r] = o[0]; out[(size_t) ent_dst[e + 1] * L.n_embd + r] = o[1]; }
                e += 2;
            }
        }
        for (; e < e1; ++e) {
            const float v = row_dot_lanes<TD, LN>(wr, hq + (size_t) e * hb, nb, sub);
            if (sub == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = v;
        }
    }
}

// native_down_kernel with the entries taken GRP_NC at a time
template <int TD, bool SUB_DOWN = true>
/*
DPCT1110: The total declared local variable size in device function
native_down_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void native_down_multi_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_dst, const block_q8_1 *__restrict__ hq,
    NativeExpertLayout L, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ng = *n_groups;
    if (item_ct1.get_group(1) >= ng) return;
    auto &s_hq_buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        uint32_t[GRP_NC * 20 * 9]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    if constexpr (SUB_DOWN && TD == 20) {
        if (hb == 20) {
            const int r =
                item_ct1.get_group(2) * 32 + (item_ct1.get_local_id(2) >> 3);
            const int t = item_ct1.get_local_id(2) & 7;
            const bool valid_r = (r < L.n_embd);
            const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int g = item_ct1.get_group(1); g < ng;
                 g += item_ct1.get_group_range(1)) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = sycl::min(GRP_NC, e1 - e);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                    const int words = n * (20 * 9);
                    /*
                    DPCT1118: SYCL group functions and algorithms must be
                    encountered in converged control flow. You may need to
                    adjust the code.
                    */
                    /*
                    DPCT1065: Consider replacing sycl::nd_item::barrier()
                    with
                    sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                    for better performance if there is no access to global
                    memory.
                    */
                    item_ct1.barrier();
#pragma unroll
                    for (int i = item_ct1.get_local_id(2); i < words; i += 256)
                        s_hq_buf[i] = src[i];
                    /*
                    DPCT1118: SYCL group functions and algorithms must be
                    encountered in converged control flow. You may need to
                    adjust the code.
                    */
                    /*
                    DPCT1065: Consider replacing sycl::nd_item::barrier()
                    with
                    sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                    for better performance if there is no access to global
                    memory.
                    */
                    item_ct1.barrier();
                    if (valid_r) {
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            row_dot_40_sub8<TD, 1, true>(wr, s_hq, off1, 1, t,
                                                         s1);
                            if (t == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0];
                        } else if (n == 2) {
                            const int off2[2] = { 0, 20 };
                            float s2[2];
                            row_dot_40_sub8<TD, 2, true>(wr, s_hq, off2, 2, t,
                                                         s2);
                            if (t < 2) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s2[t];
                        } else if (n == 3) {
                            const int off3[3] = { 0, 20, 40 };
                            float s3[3];
                            row_dot_40_sub8<TD, 3, true>(wr, s_hq, off3, 3, t,
                                                         s3);
                            if (t < 3) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s3[t];
                        } else {
                            const int off4[4] = { 0, 20, 40, 60 };
                            float s4[4];
                            row_dot_40_sub8<TD, 4, true>(wr, s_hq, off4, 4, t,
                                                         s4);
                            if (t < 4) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s4[t];
                        }
                    }
                }
            }
            return;
        }
    } else if constexpr (SUB_DOWN && TD == 42) {
        if (hb == 20) {
            const int r =
                item_ct1.get_group(2) * 16 + (item_ct1.get_local_id(2) >> 4);
            const int t = item_ct1.get_local_id(2) & 15;
            const bool valid_r = (r < L.n_embd);
            const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int g = item_ct1.get_group(1); g < ng;
                 g += item_ct1.get_group_range(1)) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = sycl::min(GRP_NC, e1 - e);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                    const int words = n * (20 * 9);
                    /*
                    DPCT1118: SYCL group functions and algorithms must be
                    encountered in converged control flow. You may need to
                    adjust the code.
                    */
                    /*
                    DPCT1065: Consider replacing sycl::nd_item::barrier()
                    with
                    sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                    for better performance if there is no access to global
                    memory.
                    */
                    item_ct1.barrier();
#pragma unroll
                    for (int i = item_ct1.get_local_id(2); i < words; i += 256)
                        s_hq_buf[i] = src[i];
                    /*
                    DPCT1118: SYCL group functions and algorithms must be
                    encountered in converged control flow. You may need to
                    adjust the code.
                    */
                    /*
                    DPCT1065: Consider replacing sycl::nd_item::barrier()
                    with
                    sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                    for better performance if there is no access to global
                    memory.
                    */
                    item_ct1.barrier();
                    if (valid_r) {
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            row_dot_20_sub16<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                            if (t == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0];
                        } else if (n == 2) {
                            const int off2[2] = { 0, 20 };
                            float s2[2];
                            row_dot_20_sub16<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                            if (t < 2) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s2[t];
                        } else if (n == 3) {
                            const int off3[3] = { 0, 20, 40 };
                            float s3[3];
                            row_dot_20_sub16<TD, 3, true>(wr, s_hq, off3, 3, t, s3);
                            if (t < 3) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s3[t];
                        } else {
                            const int off4[4] = { 0, 20, 40, 60 };
                            float s4[4];
                            row_dot_20_sub16<TD, 4, true>(wr, s_hq, off4, 4, t, s4);
                            if (t < 4) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s4[t];
                        }
                    }
                }
            }
            return;
        }
    }
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int r = item_ct1.get_group(2) * 8 + warp;
    const bool valid_r = (r < L.n_embd);
    const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
    for (int g = item_ct1.get_group(1); g < ng;
         g +=
         item_ct1.get_group_range(1)) { // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if (hb == 20) {
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = sycl::min(GRP_NC, e1 - e);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                const int words = n * (20 * 9);
                /*
                DPCT1118: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
#pragma unroll
                for (int i = item_ct1.get_local_id(2); i < words; i += 256)
                    s_hq_buf[i] = src[i];
                /*
                DPCT1118: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                if (valid_r) {
                    if (n == 1) {
                        const int off1[1] = { 0 };
                        float s1[1];
                        row_dot_multi<TD, 1, true>(wr, s_hq, off1, 1, nb, lane,
                                                   s1);
                        if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0];
                    } else if (n == 2) {
                        const int off2[2] = { 0, 20 };
                        float s2[2];
                        row_dot_multi<TD, 2, true>(wr, s_hq, off2, 2, nb, lane,
                                                   s2);
                        if (lane < 2) out[(size_t) ent_dst[e + lane] * L.n_embd + r] = s2[lane];
                    } else {
                        int off[GRP_NC];
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c)
                            off[c] = sycl::min(c, n - 1) * 20;
                        float s[GRP_NC];
                        row_dot_multi<TD, GRP_NC>(wr, s_hq, off, n, nb, lane, s);
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c)
                            if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
                    }
                }
            }
        } else if (valid_r) {
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = sycl::min(GRP_NC, e1 - e);
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    off[c] = (e + sycl::min(c, n - 1)) * hb;
                float s[GRP_NC];
                row_dot_multi<TD, GRP_NC>(wr, hq, off, n, nb, lane, s);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
            }
        }
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
// value i of a q8_1 row set (the warp holds block i / 32, lane = i % 32)
__dpct_inline__ void q8_1_store(const float xi, block_q8_1 *__restrict__ y,
                                const long long i) {
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        amax = sycl::fmax(
            amax,
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                amax, o));
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        sum += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = q8_1_ds(d, sum);
}

__dpct_inline__ void quantize_q8_1_kernel(const float *__restrict__ x,
                                          block_q8_1 *__restrict__ y,
                                          long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    q8_1_store(x[i], y, i);
}

// swiglu_entries_kernel and quantize_q8_1_kernel in one pass, over the call's own entries only:
// [grp_start[0], grp_start[*n_groups]) (a call's entries are contiguous; the verify window's PCIe call starts after
// its VRAM call) instead of all cap_entries twice - nothing reads the others: the down kernel reads its groups'.
// n_ff is a multiple of 32, so a warp is one q8_1 block and the bounds are warp-uniform.  The values are the two
// kernels': the SwiGLU product is rounded on its own - it must not contract into the first add of q8_1_store's block
// sum, as it could not when it went through memory (CUDA: __fmul_rn; HIP's __fmul_rn is a plain product, so there
// the product is written here under contract(off)) - then q8_1_store unchanged: the same blocks, bit for bit.
__dpct_inline__ void swiglu_q8_1_entries_kernel(
    const float *__restrict__ gate, const float *__restrict__ up,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    int n_ff, block_q8_1 *__restrict__ hq) {
    // the product rounded on its own (no contraction into q8_1_store's block sum), as through memory in the v1 kernels
    // - HIP's form; icpx is clang too (SYCL port)
#pragma clang fp contract(off)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long lo = (long long)grp_start[0] * n_ff,
                    hi = (long long)grp_start[*n_groups] * n_ff;
    for (long long i =
             lo +
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < hi; i += (long long)item_ct1.get_group_range(2) *
                      item_ct1.get_local_range(2)) {
        const float g = gate[i];
#if defined(__HIPCC__)
        const float h = (g / (1.0f + __expf(-g))) * up[i];
#else
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const float h = (g / (1.0f + sycl::native::exp(-g))) * up[i];
#endif
        q8_1_store(h, hq, i);
    }
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template <typename dst_t> __dpct_inline__ dst_t cvt(float v);
template <> __dpct_inline__ float cvt<float>(float v) { return v; }
template <>
__dpct_inline__ sycl::half cvt<sycl::half>(float v) {
    return sycl::vec<float, 1>(v)
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

// SYCL port: the dequant functions below wrote one 2-byte value at a time (dq_* = llama.cpp's per-thread loops); on Xe
// the FP16 expert dequant ran at ~80 GB/s of writes, 40% of the prompt path. Each thread's run of values goes out as one
// vector store instead (the offsets are multiples of the run length, so the stores are aligned).
template <typename dst_t, int N>
__dpct_inline__ void store_run(dst_t* y, const float* v) {
    if constexpr (std::is_same_v<dst_t, sycl::half> && (N == 4 || N == 8)) {
        sycl::vec<sycl::half, N> h;
#pragma unroll
        for (int j = 0; j < N; ++j) h[j] = sycl::half(v[j]);
        *reinterpret_cast<sycl::vec<sycl::half, N>*>(y) = h;
    } else {
#pragma unroll
        for (int j = 0; j < N; ++j) y[j] = cvt<dst_t>(v[j]);
    }
}
template <typename dst_t>
inline void dq_iq2_xxs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq2_xs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq2_s(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq3_xxs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
        v[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq3_s(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
        v[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq1_m(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template <typename dst_t>
inline void dq_iq4_nl(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    // SYCL port: thread (ib, il) writes the block's values [8 il, 8 il + 8) as one run: il < 2 the low nibbles of
    // qs[8 il ..], il >= 2 the high nibbles of qs[8 (il - 2) ..] (the same values as llama.cpp's 4 + 4 split)
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q4 = x[ib].qs + 8 * (il & 1);
    const float d = (float) x[ib].d;
    float v[8];
    if (il < 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = d * kvalues_iq4nl[q4[j] & 0xf];
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = d * kvalues_iq4nl[q4[j] >> 4];
    }
    store_run<dst_t, 8>(y, v);
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template <typename dst_t>
inline void dq_q3_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
#pragma unroll
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(
            dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template <typename dst_t>
inline void dq_iq4_xs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    float lo[4], hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo[j] = d * kvalues_iq4nl[q4[j] & 0xf];
        hi[j] = d * kvalues_iq4nl[q4[j] >> 4];
    }
    store_run<dst_t, 4>(y, lo);
    store_run<dst_t, 4>(y + 16, hi);
}
template <typename dst_t>
inline void dq_q2_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        v[j] = d * (float) (code - 1);
    }
    store_run<dst_t, 8>(yy + b * 64 + part * 8, v);
}

// llama.cpp's dequantize_q4_K / dequantize_q5_K (dequantize.cuh; q5_K's 64 threads folded onto 32) and the 32-value
// blocks of Q5_1 / Q8_0, 8 of them per 256-value "superblock" (thread tid writes 8 values of block tid / 4; it was block
// tid % 8, which scattered a sub-group's 2-byte stores 64 bytes apart: B70 Q5_1 / Q8_0 at 88-93 GB/s of FP16 against 250+).
__dpct_inline__ void get_scale_min_k4(int j, const uint8_t *q, uint8_t &d,
                                      uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template <typename dst_t>
inline void dq_q4_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = x[ibs].dm[0];
    const float dmin = x[ibs].dm[1];
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
#pragma unroll
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}
template <typename dst_t>
inline void dq_q5_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        dst_t* y = yy + 64 * il + 2 * ir;
        const float dall = x[ibs].dm[0];
        const float dmin = x[ibs].dm[1];
        const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * sc, m2 = dmin * m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = cvt<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = cvt<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = cvt<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = cvt<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
template <typename dst_t>
inline void dq_q5_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    // symmetric: one delta (no min), the high bit folded in then centered by the format's fixed -16 offset
    const block_q5_0* x = (const block_q5_0*) vx + ibs * (QK_K / QK5_0);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const float d = sycl::vec<sycl::half, 1>(x[ib].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    float v0[4], v1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        v0[j] = ((float) ((x[ib].qs[iqs] & 0xf) | xh_0) - 16.0f) * d;
        v1[j] = ((float) ((x[ib].qs[iqs] >> 4) | xh_1) - 16.0f) * d;
    }
    store_run<dst_t, 4>(y + 4 * il, v0);
    store_run<dst_t, 4>(y + 16 + 4 * il, v1);
}
template <typename dst_t>
inline void dq_q4_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q4_0* x = (const block_q4_0*) vx + ibs * (QK_K / QK4_0);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const float d = sycl::vec<sycl::half, 1>(x[ib].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    dst_t* y = yy + 32 * ib;
    float v0[4], v1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_0 for value pairs iqs, iqs + 16
        v0[j] = ((float) (x[ib].qs[iqs] & 0xf) - 8.0f) * d;
        v1[j] = ((float) (x[ib].qs[iqs] >> 4) - 8.0f) * d;
    }
    store_run<dst_t, 4>(y + 4 * il, v0);
    store_run<dst_t, 4>(y + 16 + 4 * il, v1);
}
template <typename dst_t>
inline void dq_q4_1(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q4_1* x = (const block_q4_1*) vx + ibs * (QK_K / QK4_1);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const sycl::float2 dm =
        (x[ib].dm).template convert<float, sycl::rounding_mode::automatic>();
    dst_t* y = yy + 32 * ib;
    float v0[4], v1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_1 for value pairs iqs, iqs + 16
        v0[j] = (float)(x[ib].qs[iqs] & 0xf) * dm.x() + dm.y();
        v1[j] = (float)(x[ib].qs[iqs] >> 4) * dm.x() + dm.y();
    }
    store_run<dst_t, 4>(y + 4 * il, v0);
    store_run<dst_t, 4>(y + 16 + 4 * il, v1);
}
template <typename dst_t>
inline void dq_q5_1(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const sycl::float2 dm =
        (x[ib].dm).template convert<float, sycl::rounding_mode::automatic>();
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    float v0[4], v1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_1 for value pairs iqs, iqs + 16
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        v0[j] = (float)((x[ib].qs[iqs] & 0xf) | xh_0) * dm.x() + dm.y();
        v1[j] = (float)((x[ib].qs[iqs] >> 4) | xh_1) * dm.x() + dm.y();
    }
    store_run<dst_t, 4>(y + 4 * il, v0);
    store_run<dst_t, 4>(y + 16 + 4 * il, v1);
}
template<typename dst_t>
void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // llama.cpp's dequantize_q6_K (dequantize.cuh, written for 64 threads) folded onto the kernel's 32: each of the
    // 64 thread roles runs at tid and tid + 32.
    const block_q6_K* x = (const block_q6_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int64_t ip = tt / 32;                 // 0 or 1
        const int64_t il = tt - 32 * ip;            // 0...31
        const int64_t is = 8 * ip + il / 16;
        dst_t* y = yy + 128 * ip + il;
        const float d =
            sycl::vec<sycl::half, 1>(x[ibs].d)
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const uint8_t* ql = x[ibs].ql + 64 * ip + il;
        const uint8_t qh = x[ibs].qh[32 * ip + il];
        const int8_t* sc = x[ibs].scales + is;
        y[0]  = cvt<dst_t>(d * sc[0] * ((int8_t) ((ql[0]  & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32] = cvt<dst_t>(d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64] = cvt<dst_t>(d * sc[4] * ((int8_t) ((ql[0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96] = cvt<dst_t>(d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}
template <typename dst_t>
inline void dq_q8_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const float d = sycl::vec<sycl::half, 1>(x[ib].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    dst_t* y = yy + 32 * ib + 8 * il;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = (float)x[ib].qs[8 * il + j] * d;
    store_run<dst_t, 8>(y, v);
}

// BF16 (the token embedding as the checkpoint ships it, tools/embd_bf16_pack.py): 8 values per thread.
template <typename dst_t>
inline void dq_bf16(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const uint16_t* x = (const uint16_t*) vx + ibs * 256 + tid * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) yy[tid * 8 + j] =
        cvt<dst_t>(sycl::bit_cast<float>((uint32_t)x[j] << 16));
}

// APR trellis types: one 256-value call = two 128-blocks; thread tid decodes block tid/16, lane (tid/2)%8, steps
// 2(tid%2)..2(tid%2)+1 (8 values)
template <int TY, typename dst_t>
inline void dq_tq(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const int blk = tid >> 4, l = (tid >> 1) & 7, half = tid & 1;
    const auto* b = (const typename TqInfo<TY>::block_t*) vx + 2 * ibs + blk;
    const float d = tq_h2f(b->d);
    uint32_t s[4];
    tq_states4<TY>(b->qs, l, s);
    dst_t* y = yy + blk * 128 + 16 * l + 8 * half;
    float v[8];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        float w[4];
        tq_step(s[2 * half + i], w);
        v[4 * i + 0] = d * w[0];
        v[4 * i + 1] = d * w[1];
        v[4 * i + 2] = d * w[2];
        v[4 * i + 3] = d * w[3];
    }
    store_run<dst_t, 8>(y, v);
}

// Every type below must also be in is_iq() (BF16: embed_type_supported): the host entry points refuse the others,
// so the default is unreachable.
template <typename dst_t>
__dpct_inline__ void
dq_dispatch(int ty, const void *vx, int64_t ibs, dst_t *y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid);
            break;
        case 17: dq_iq2_xs(vx, ibs, y, tid);
            break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid);
            break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
#ifdef STRATA_Q6K_EXPERTS
        case 14: dq_q6_k(vx, ibs, y, tid); break;
#endif
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 6: dq_q5_0(vx, ibs, y, tid); break;
        case 2: dq_q4_0(vx, ibs, y, tid); break;
        case 3: dq_q4_1(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 30: dq_bf16(vx, ibs, y, tid); break;
        case 144: dq_tq<144>(vx, ibs, y, tid); break;
        case 145: dq_tq<145>(vx, ibs, y, tid); break;
        case 146: dq_tq<146>(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template <typename dst_t>
__dpct_inline__ void dequant_flat_kernel(
    int ty, const void *__restrict__ vx, dst_t *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = item_ct1.get_group(2);
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, item_ct1.get_local_id(2));
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
__dpct_inline__ void dequant_gu_kernel(
    int ty, const void *__restrict__ gate, const void *__restrict__ up,
    int64_t per_row, sycl::half *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = item_ct1.get_group(2);
    const int parity = item_ct1.get_group(1);
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<sycl::half>(ty, parity ? up : gate, i,
                            y + ((2 * r + parity) * per_row + c) * QK_K,
                            item_ct1.get_local_id(2));
}

// the types dq_dispatch dequantizes
bool is_iq(int t) {
    return t == 144 || t == 145 || t == 146 || t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
#ifdef STRATA_Q6K_EXPERTS
           t == 12 || t == 13 || t == 14 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8;
#else
           t == 12 || t == 13 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8;
#endif
}
// values per block of the types the grouped expert kernels take (0 = none)
int gu_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_GU_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_D_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
bool gu_split(int t) {
    switch (t) {
#define STRATA_SP(T) case T: return kSplit<T>;
        STRATA_GU_FMTS(STRATA_SP)
#undef STRATA_SP
        default: return false;
    }
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}
// STRATA_OLD_IQ_MMVQ=1 keeps the per-column kernels (bitwise equal to the new ones; kept for A/B timing)
bool g_old_kernels = env_on("STRATA_OLD_IQ_MMVQ");
// SYCL port (0.1.31 merge): upstream sends the split formats (the i-quants) to native_gu/down_multi_kernel, one warp
// per row, 8 rows a group. On the B70 the port's grouped kernels (8 lanes a row, 32 rows a group, the multi-entry
// dots with aligned loads and the IQ4 codebook in registers) are the measured ones, so they stay the default for
// every format; STRATA_EXPERT_SPLIT=1 takes upstream's multi kernels, launched with their own grid (GU_ROWS rows).
bool g_split_multi = env_on("STRATA_EXPERT_SPLIT");
bool g_no_sub16_gu = env_on("STRATA_NO_SUB16_GU");
// STRATA_IQ_STAGE_GRID=0 disables staging 64-bit i-quant codebook tables into shared memory
bool g_stage_grid = [] {
    const char* v = std::getenv("STRATA_IQ_STAGE_GRID");
    return v == nullptr || v[0] == '\0' || v[0] != '0';
}();
// The single-matrix mmvq (128-thread blocks, one per 4 rows) pays the 2-8 KB staging per block: on the RTX 5070 its
// IQ2_XS / IQ2_S / IQ1_M calls were 10-20% slower staged at 3-8 columns (iq_multi_parity --bench), while the grouped
// expert kernels gain.  So mmvq stages only with STRATA_IQ_STAGE_GRID_MMVQ=1 (bitwise the same either way).
bool g_stage_grid_mmvq = g_stage_grid && [] {
    const char* v = std::getenv("STRATA_IQ_STAGE_GRID_MMVQ");
    return v != nullptr && v[0] == '1';
}();

template <int TY, bool STAGE_GRID = kStageIqGrid<TY>>
void launch_mmvq_multi_nc(dpct::dim3 grid, dpct::dim3 block, dpct::queue_ptr s,
                          const uint8_t *W, size_t rb, const block_q8_1 *X,
                          float *y, int n_in, int n_out, int ncols) {
    switch (ncols) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 1: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class mmvq_multi_kernel_8df6c9, dpct_kernel_scalar<TY>,
                dpct_kernel_scalar<1>, dpct_kernel_scalar<STAGE_GRID>,
                dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 1, STAGE_GRID, true>(
                            W, rb, X, y, n_in, n_out, 1);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 2: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class mmvq_multi_kernel_9c43f2, dpct_kernel_scalar<TY>,
                dpct_kernel_scalar<2>, dpct_kernel_scalar<STAGE_GRID>,
                dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 2, STAGE_GRID, true>(
                            W, rb, X, y, n_in, n_out, 2);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 3: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class mmvq_multi_kernel_f8d14a, dpct_kernel_scalar<TY>,
                dpct_kernel_scalar<3>, dpct_kernel_scalar<STAGE_GRID>,
                dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 3, STAGE_GRID, true>(
                            W, rb, X, y, n_in, n_out, 3);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 4: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class mmvq_multi_kernel_88cd9a, dpct_kernel_scalar<TY>,
                dpct_kernel_scalar<4>, dpct_kernel_scalar<STAGE_GRID>,
                dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 4, STAGE_GRID, true>(
                            W, rb, X, y, n_in, n_out, 4);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 5: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class mmvq_multi_kernel_d55b02, dpct_kernel_scalar<TY>,
                dpct_kernel_scalar<5>, dpct_kernel_scalar<STAGE_GRID>,
                dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 5, STAGE_GRID, true>(
                            W, rb, X, y, n_in, n_out, 5);
                    });
        });
    } break;
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 6: {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class mmvq_multi_kernel_3404b2, dpct_kernel_scalar<TY>,
                dpct_kernel_scalar<6>, dpct_kernel_scalar<STAGE_GRID>,
                dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 6, STAGE_GRID, true>(
                            W, rb, X, y, n_in, n_out, 6);
                    });
        });
    } break;
        default:
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
        if (ncols <= 1) {

            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<dpct_kernel_name<
                    class mmvq_multi_kernel_1f7e16, dpct_kernel_scalar<TY>,
                    dpct_kernel_scalar<1>, dpct_kernel_scalar<STAGE_GRID>,
                    dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(grid * block, block), exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            mmvq_multi_kernel<TY, 1, STAGE_GRID, false>(
                                W, rb, X, y, n_in, n_out, ncols);
                        });
            });
        }
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        else {

            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<dpct_kernel_name<
                    class mmvq_multi_kernel_95b544, dpct_kernel_scalar<TY>,
                    dpct_kernel_scalar<8>, dpct_kernel_scalar<STAGE_GRID>,
                    dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(grid * block, block), exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            mmvq_multi_kernel<TY, 8, STAGE_GRID, false>(
                                W, rb, X, y, n_in, n_out, ncols);
                        });
            });
        }
            break;
    }
}

template <int TY>
void launch_mmvq(const uint8_t *W, size_t rb, const block_q8_1 *X, float *y,
                 int n_in, int n_out, int ncols, dpct::queue_ptr s) {
    const dpct::dim3 grid((unsigned)((n_out + 3) / 4)), block(32, 4);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    if constexpr (!kSplit<TY>) {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_c04bb8,
                                              dpct_kernel_scalar<TY>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<TY>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    else if (g_old_kernels) {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_18057f,
                                              dpct_kernel_scalar<TY>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<TY>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    } else if constexpr (kStageIqGrid<TY>) {
        if (!g_stage_grid_mmvq) {
            launch_mmvq_multi_nc<TY, false>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
            return;
        }
        launch_mmvq_multi_nc<TY, true>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
    } else {
        launch_mmvq_multi_nc<TY, false>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
    }
}


// SYCL port: upstream's opt-in AMD-tuned S26 (STRATA_EXPERT_V2) and S27 (STRATA_EXPERT_V2K) grouped-expert kernels are
// not migrated: dpct cannot name their instantiations and they are tuned for gfx1151. The port's lane kernels below are
// the measured ones for every format; the two switches only print a note.
#if 0
// ---------------------------------------------------------------- S26 STRATA_EXPERT_V2=1 (opt-in): IQ3_S gate/up + IQ4_NL down
// Bitwise equal to native_gu_multi_kernel / native_down_multi_kernel (S26 harness, real UD-IQ4_XS expert blobs, T 1-3,
// memcmp of every output): each row's lane sums and warp_sum are the same; a warp interleaves RPW rows (more loads in
// flight) and reads the chunk's activation rows from LDS and, for IQ3_S, the grid table from LDS (the same values).
#if defined(__HIPCC__)
#define STRATA_NT_LOAD(p) __builtin_nontemporal_load(p)
#else   // (opt-in S26 kernels, AMD-tuned: a CUDA build compiles them with plain loads)
#define STRATA_NT_LOAD(p) (*(p))
#endif
template <bool NT> __dpct_inline__ int ld16i(const uint16_t *p) {
    if constexpr (NT) return (int) STRATA_NT_LOAD(p); else return (int) *p;
}
template <bool NT> __dpct_inline__ int ldb2(const void *x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = ld16i<NT>(x16 + 2 * i32 + 0) << 0;
    x32 |= ld16i<NT>(x16 + 2 * i32 + 1) << 16;
    return x32;
}
template <bool NT> __dpct_inline__ int ld8i(const uint8_t *p) {
    if constexpr (NT) return (int) STRATA_NT_LOAD(p); else return (int) *p;
}
template <bool NT> __dpct_inline__ float ldhalf(const sycl::half *p) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(
                   (unsigned short)ld16i<NT>(
                       reinterpret_cast<const uint16_t *>(p))))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
template<int TY, bool NT> struct S26Split;
template<bool NT> struct S26Split<21, NT> {   // IQ3_S: Split<21>::load with the loads made explicit
    using W = Split<21>::W;
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const sycl::int2 qs_packed =
            make_int2(ldb2<NT>(bq3->qs, iqs + 0), ldb2<NT>(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = ld8i<NT>(bq3->qh + iqs / 2);
        const int signs_packed_32 = ldb2<NT>(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos =
                sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                           iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                0x00000000, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                0x00000000, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.ls = 1 + 2 * ((ld8i<NT>(bq3->scales + iqs / 4) >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = ldhalf<NT>(&bq3->d);
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ b, int iqs) { return Split<21>::apply(r, b, iqs); }
};
template<bool NT> struct S26Split<20, NT> {   // IQ4_NL
    using W = Split<20>::W;
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(ldb2<NT>(bq4->qs, iqs + l), kvalues_iq4nl);
        r.dw = ldhalf<NT>(&bq4->d);
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ b, int iqs) { return Split<20>::apply(r, b, iqs); }
};

// S26 v13+: v6 (4 interleaved rows per warp) with the IQ3_S grid table (LT) and / or the chunk's activation rows (LX)
// read from LDS: the same table values and the same q8_1 bytes, so the same arithmetic and bits.
struct S26IQ3S {   // Split<21>::load with the grid from `grid` (LT) or iq3s_grid itself
    using W = Split<21>::W;
    template<bool LT = true>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* grid) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const sycl::int2 qs_packed = sycl::int2(get_int_b2(bq3->qs, iqs + 0),
                                                get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int i0 = qs[l0 + 0] | ((qh << (8 - l0)) & 0x100), i1 = qs[l0 + 1] | ((qh << (7 - l0)) & 0x100);
            const sycl::int2 grid_pos =
                LT ? sycl::int2(grid[i0], grid[i1])
                   : sycl::int2(iq3s_grid[i0], iq3s_grid[i1]);
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                0x00000000, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                0x00000000, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = sycl::vec<sycl::half, 1>(bq3->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
};

constexpr int S26_XMAX = 2560 / 32;   // q8_1 blocks of one activation row (n_embd 2560)

// S26 BAL: step I of a warp's RPW x NI items (item p = 32 I + lane): rows QA / QB (lanes >= THR take QB)
template<int I, int NI, int RPW> struct S26Bal {
    static constexpr int P0 = 32 * I, QA = P0 / NI, QB0 = (P0 + 31) / NI, QB = QB0 < RPW ? QB0 : RPW - 1;
    static constexpr int THR = (QA + 1) * NI - P0;
    static_assert(NI >= 32, "a step spans at most two rows");
};

template <bool LT, bool LX, int RPW, bool SL = false, bool TS = false>
/*
DPCT1110: The total declared local variable size in device function
s26_gu_l_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void s26_gu_l_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_tok, const block_q8_1 *__restrict__ xq,
    NativeExpertLayout L, float *__restrict__ gate, float *__restrict__ up) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_grid = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    uint32_t[LT ? 512 : 1]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_x = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        block_q8_1[LX ? GRP_NC * S26_XMAX : 1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    if constexpr (LT)
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < 512; i += 256)
            s_grid[i] = iq3s_grid[i];
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int row0 = item_ct1.get_group(2) * GU_ROWS * RPW + warp;
    const bool live = row0 + GU_ROWS * (RPW - 1) < 2 * L.n_ff;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
    int rr[RPW];
    bool upq[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int row = row0 + q * GU_ROWS;
        upq[q] = row >= L.n_ff;
        rr[q] = upq[q] ? row - (int) L.n_ff : row;
        wr[q] = blob + (upq[q] ? L.up_off : 0) + (size_t) rr[q] * L.gu_row;
    }
    const uint32_t* grid = LT ? s_grid : reinterpret_cast<const uint32_t*>(iq3s_grid);
    const int nb = (int) (L.n_embd / 256), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = sycl::min(GRP_NC, e1 - e);
        const block_q8_1* xbase = xq;
        int off[GRP_NC];
        if constexpr (LX) {
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(); // the previous chunk is done with s_x
            for (int i = item_ct1.get_local_id(2); i < n * xb; i += 256) {
                const int c = i / xb, j = i - c * xb;
                memcpy(&s_x[c * S26_XMAX + j], &xq[(size_t) ent_tok[e + c] * xb + j], sizeof(s_x[0]));
            }
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = sycl::min(c, n - 1) * S26_XMAX;
            xbase = s_x;
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = ent_tok[e + sycl::min(c, n - 1)] * xb;
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
        if constexpr (LT || LX) item_ct1.barrier(); // s_grid / s_x ready
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        for (int k = lane; k < nb * 8; k += 32) {
            const int kbx = k / 8, iqs = 2 * (k % 8);
            Split<21>::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) {
                if constexpr (SL && !LT) w[q] =
                    S26Split<21, false>::load(wr[q], kbx, iqs);
                else w[q] = S26IQ3S::load<LT>(wr[q], kbx, iqs, grid);
            }
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += Split<21>::apply(w[q], xbase + off[c] + kbx * 8, iqs);
        }
        if constexpr (TS) {   // S26: all RPW x GRP_NC sums in one transposed butterfly (bitwise the same sums)
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC) {
                const int c = j % GRP_NC;
#pragma unroll
                for (int q = 0; q < RPW; ++q)
                    if (j / GRP_NC == q && c < n) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = sum;
            }
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = s[q][c];
    }
}

constexpr int S26_HMAX = 640 / 32;    // q8_1 blocks of one SwiGLU row (n_ff 640)

template <bool LX, int RPW, bool SL = false, bool TS = false, int BAL = 0>
/*
DPCT1110: The total declared local variable size in device function
s26_down_l_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
s26_down_l_kernel(const unsigned long long *__restrict__ grp_ptr,
                  const int32_t *__restrict__ grp_start,
                  const int32_t *__restrict__ n_groups,
                  const int32_t *__restrict__ ent_dst,
                  const block_q8_1 *__restrict__ hq, NativeExpertLayout L,
                  float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_h = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    block_q8_1[LX ? GRP_NC * S26_HMAX : 1]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int r0 = item_ct1.get_group(2) * 8 * RPW + warp;
    const bool live = r0 + 8 * (RPW - 1) < L.n_embd;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) wr[q] = blob + L.down_off + (size_t) (r0 + 8 * q) * L.d_row;
    const int nb = (int) (L.n_ff / 32), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = sycl::min(GRP_NC, e1 - e);
        const block_q8_1* hbase = hq;
        int off[GRP_NC];
        if constexpr (LX) {
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            for (int i = item_ct1.get_local_id(2); i < n * hb; i += 256) {
                const int c = i / hb, j = i - c * hb;
                memcpy(&s_h[c * S26_HMAX + j], &hq[(size_t) (e + c) * hb + j], sizeof(s_h[0]));
            }
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = sycl::min(c, n - 1) * S26_HMAX;
            hbase = s_h;
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = (e + sycl::min(c, n - 1)) * hb;
        }
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        if constexpr (BAL > 0) {   // S26: 5 items per lane instead of 8 / 4, in BAL groups (loads first)
            constexpr int NI = 2 * S26_HMAX, IT = RPW * NI / 32;
            static_assert(RPW * NI % 32 == 0, "whole steps");
            /*
            DPCT1110: The total declared local variable size in device
            function operator() exceeds 128 bytes and may cause high register
            pressure. Consult with your hardware vendor to find the total
            register size available and adjust the code, or use smaller
            sub-group size to avoid high register pressure.
            */
            auto group = [&]<int G0, int... J>(std::integer_sequence<int, J...>) {
                Split<20>::W w[sizeof...(J)];
                int kk[sizeof...(J)];
                bool hh[sizeof...(J)];
                (
                    [&] {
                    using B = S26Bal<G0 + J, NI, RPW>;
                    hh[J] = B::QB != B::QA && lane >= B::THR;
                    kk[J] = B::P0 + lane - (hh[J] ? B::QB : B::QA) * NI;
                    const uint8_t* rp = hh[J] ? wr[B::QB] : wr[B::QA];
                    if constexpr (SL) w[J] = S26Split<20, false>::load(
                        rp, kk[J] / 2, 2 * (kk[J] % 2));
                    else w[J] = Split<20>::load(rp, kk[J] / 2, 2 * (kk[J] % 2));
                    }(),
                    ...);
                (
                    [&] {
                    using B = S26Bal<G0 + J, NI, RPW>;
#pragma unroll
                    for (int c = 0; c < GRP_NC; ++c)
                        if (c < n) {
                            const float v = Split<20>::apply(w[J], hbase + off[c] + kk[J] / 2, 2 * (kk[J] % 2));
                            if (hh[J]) s[B::QB][c] += v; else s[B::QA][c] += v;
                        }
                    }(),
                    ...);
            };
            constexpr int G = (IT + BAL - 1) / BAL;
            [&]<int... Q>(std::integer_sequence<int, Q...>) {
                (group.template operator()<Q * G>(std::make_integer_sequence<int, (IT - Q * G < G ? IT - Q * G : G)>{}), ...);
            }(std::make_integer_sequence<int, BAL>{});
        } else
        for (int k = lane; k < nb * 2; k += 32) {
            const int kbx = k / 2, iqs = 2 * (k % 2);
            Split<20>::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) {
                if constexpr (SL) w[q] =
                    S26Split<20, false>::load(wr[q], kbx, iqs);
                else w[q] = Split<20>::load(wr[q], kbx, iqs);
            }
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += Split<20>::apply(w[q], hbase + off[c] + kbx, iqs);
        }
        if constexpr (TS) {   // S26: all RPW x GRP_NC sums in one transposed butterfly (bitwise the same sums)
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            const int q = j / GRP_NC, c = j % GRP_NC;
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC && c < n)
                out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = sum;
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = s[q][c];
    }
}


// S26: swiglu_entries_kernel + quantize_q8_1_kernel in one launch (the same h expression, then the same block quantizer,
// contraction off so the product cannot fuse into the first sum); h is still written (it is the scratch the old pair used)
__dpct_inline__ void s26_swiglu_q8_1_kernel(const float *__restrict__ gate,
                                            const float *__restrict__ up,
                                            float *__restrict__ h,
                                            block_q8_1 *__restrict__ y,
                                            long long n) {
#pragma clang fp contract(off)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;   // n is a multiple of 32: whole warps only
    const float g = gate[i];
    const float xi = (g / (1.0f + sycl::native::exp(-g))) * up[i];
    h[i] = xi;
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        amax = sycl::fmax(
            amax,
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                amax, o));
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        sum += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sum, o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = sycl::half2(d, sum);
}
template <bool LT, bool LX, int RG, int RD, bool SL = false, bool FQ = false,
          bool TS = false, int BD = 0>
void s26_launch_l(const NativeExpertLayout &L, int64_t cap_groups,
                  dpct::queue_ptr s, const unsigned long long *grp_ptr,
                  const int32_t *grp_start, const int32_t *n_groups,
                  const int32_t *ent_dst, const int32_t *ent_tok,
                  const block_q8_1 *X, float *gate, float *up, float *h,
                  block_q8_1 *hq, float *out, long long nh) {
    const dpct::dim3 ggu((unsigned)(2 * L.n_ff / (GU_ROWS * RG)),
                         (unsigned)cap_groups);
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class s26_gu_l_kernel_46f2dc, dpct_kernel_scalar<LT>,
                dpct_kernel_scalar<LX>, dpct_kernel_scalar<RG>,
                dpct_kernel_scalar<SL>, dpct_kernel_scalar<TS>>>(
                sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        s26_gu_l_kernel<LT, LX, RG, SL, TS>(
                            grp_ptr, grp_start, n_groups, ent_tok, X, L, gate,
                            up);
                    });
        });
    }
    if (FQ) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class s26_swiglu_q8_1_kernel_a7b554>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                s26_swiglu_q8_1_kernel(gate, up, h, hq, nh);
            });
    } else {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<
                dpct_kernel_name<class swiglu_entries_kernel_a4959c>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    swiglu_entries_kernel(gate, up, h, nh);
                });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->parallel_for<
                dpct_kernel_name<class quantize_q8_1_kernel_6b0f01>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        quantize_q8_1_kernel(h, hq, nh);
                    });
        }
    }
    const dpct::dim3 gd((unsigned)(L.n_embd / (8 * RD)), (unsigned)cap_groups);
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class s26_down_l_kernel_669590, dpct_kernel_scalar<LX>,
                dpct_kernel_scalar<RD>, dpct_kernel_scalar<SL>,
                dpct_kernel_scalar<TS>, dpct_kernel_scalar<BD>>>(
                sycl::nd_range<3>(gd * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        s26_down_l_kernel<LX, RD, SL, TS, BD>(
                            grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
                    });
        });
    }
}
// ---------------------------------------------------------------- stream B: UD-Q4_K_XL grouped decode experts
// STRATA_EXPERT_V2K=1: s26_gu_l_kernel / s26_down_l_kernel's structure (4 interleaved rows per warp, the chunk's q8_1
// activation rows in LDS, SwiGLU + q8_1 fused, optionally STRATA_TSUM's transposed butterfly) for Q4_K / Q5_K gate/up
// and Q5_1 / Q8_0 down, which otherwise run native_gu_kernel / native_down_kernel (one entry at a time, the weight
// words re-read per column).  Each Fmt<TY>::dot is split into S27<TY>::load (everything that depends on the weight) and
// apply (the activation words and the same integer / float expression), the same impl functions, so a row's lane-strided
// k order, its terms and warp_sum are the old kernels': every output is bitwise the old one's (iq harness, memcmp).
template<int TY> struct S27;
template<> struct S27<12> {   // Q4_K
    struct W { int v[2]; uint16_t aux[2]; sycl::half2 dm; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q4_K* b = (const block_q4_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        const int* q4 = (const int*) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = q4[0];
        r.v[1] = q4[4];
        k_scale_min(b->scales, bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        int u[2 * QR4_K];
        float d8[QR4_K];
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        const uint8_t* m = sc + 2;
        return vec_dot_q4_K_q8_1_impl_vmmq(r.v, u, sc, m, r.dm, d8);
    }
};
template<> struct S27<13> {   // Q5_K
    struct W { int vl[2]; int vh[2]; uint16_t aux[2]; sycl::half2 dm; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q5_K* b = (const block_q5_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        const int* ql = (const int*) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = (const int*) (b->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> bq8_offset;
        r.vh[1] = qh[4] >> bq8_offset;
        k_scale_min(b->scales, bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        int u[2 * QR5_K];
        float d8[QR5_K];
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        const uint8_t* m = sc + 2;
        return vec_dot_q5_K_q8_1_impl_vmmq(r.vl, r.vh, u, sc, m, r.dm, d8);
    }
};
template<> struct S27<7> {   // Q5_1
    struct W { int vl[VDR_Q5_1]; int qh; sycl::half2 dm; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q5_1* b = (const block_q5_1*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) r.vl[i] = get_int_b4(b->qs, iqs + i);
        r.qh = get_int_b4(b->qh, 0);
        r.dm = b->dm;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0, sumu = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) {
            const int vl = r.vl[i];
            const int vh = r.qh >> (4 * (iqs + i));
            const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
            int vi0 = (vl >> 0) & 0x0F0F0F0F;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            sumi = ggml_cuda_dp4a(vi0, u0, sumi);
            int vi1 = (vl >> 4) & 0x0F0F0F0F;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            sumi = ggml_cuda_dp4a(vi1, u1, sumi);
            sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
        }
        const sycl::float2 dm5 =
            (r.dm).template convert<float, sycl::rounding_mode::automatic>();
        const float d8 = bq8_1->ds[0];
        return sumi * (dm5.x() * d8) + sumu * (dm5.y() * d8);
    }
};
template<> struct S27<8> {   // Q8_0
    struct W { int q[VDR_Q8_0]; float d; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q8_0* b = (const block_q8_0*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) r.q[i] = get_int_b2(b->qs, iqs + i);
        r.d = sycl::vec<sycl::half, 1>(b->d)
                  .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) sumi = ggml_cuda_dp4a(r.q[i], get_int_b4(bq8_1->qs, iqs + i), sumi);
        const float d8_1 = bq8_1->ds[0];
        return r.d * d8_1 * ((float) sumi);
    }
};

template <int TG, bool LX, int RPW, bool TS>
/*
DPCT1110: The total declared local variable size in device function
s27_gu_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void s27_gu_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_tok, const block_q8_1 *__restrict__ xq,
    NativeExpertLayout L, float *__restrict__ gate, float *__restrict__ up) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    using F = Fmt<TG>;
    using S = S27<TG>;
    auto &s_x = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        block_q8_1[LX ? GRP_NC * S26_XMAX : 1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int row0 = item_ct1.get_group(2) * GU_ROWS * RPW + warp;
    const bool live = row0 + GU_ROWS * (RPW - 1) < 2 * L.n_ff;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
    int rr[RPW];
    bool upq[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int row = row0 + q * GU_ROWS;
        upq[q] = row >= L.n_ff;
        rr[q] = upq[q] ? row - (int) L.n_ff : row;
        wr[q] = blob + (upq[q] ? L.up_off : 0) + (size_t) rr[q] * L.gu_row;
    }
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = sycl::min(GRP_NC, e1 - e);
        const block_q8_1* xbase = xq;
        int off[GRP_NC];
        if constexpr (LX) {
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(); // the previous chunk is done with s_x
            for (int i = item_ct1.get_local_id(2); i < n * xb; i += 256) {
                const int c = i / xb, j = i - c * xb;
                memcpy(&s_x[c * S26_XMAX + j], &xq[(size_t) ent_tok[e + c] * xb + j], sizeof(s_x[0]));
            }
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = sycl::min(c, n - 1) * S26_XMAX;
            xbase = s_x;
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = ent_tok[e + sycl::min(c, n - 1)] * xb;
        }
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            typename S::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) w[q] = S::load(wr[q], kbx, iqs);
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += S::apply(w[q], xbase + off[c] + kbx * (F::qk / 32), iqs);
        }
        if constexpr (TS) {
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC) {
                const int c = j % GRP_NC;
#pragma unroll
                for (int q = 0; q < RPW; ++q)
                    if (j / GRP_NC == q && c < n) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = sum;
            }
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = s[q][c];
    }
}

template <int TD, bool LX, int RPW, bool TS>
/*
DPCT1110: The total declared local variable size in device function
s27_down_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void s27_down_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_dst, const block_q8_1 *__restrict__ hq,
    NativeExpertLayout L, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    using F = Fmt<TD>;
    using S = S27<TD>;
    auto &s_h = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        block_q8_1[LX ? GRP_NC * S26_HMAX : 1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int r0 = item_ct1.get_group(2) * 8 * RPW + warp;
    const bool live = r0 + 8 * (RPW - 1) < L.n_embd;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) wr[q] = blob + L.down_off + (size_t) (r0 + 8 * q) * L.d_row;
    const int nb = (int) (L.n_ff / F::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = sycl::min(GRP_NC, e1 - e);
        const block_q8_1* hbase = hq;
        int off[GRP_NC];
        if constexpr (LX) {
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            for (int i = item_ct1.get_local_id(2); i < n * hb; i += 256) {
                const int c = i / hb, j = i - c * hb;
                memcpy(&s_h[c * S26_HMAX + j], &hq[(size_t) (e + c) * hb + j], sizeof(s_h[0]));
            }
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = sycl::min(c, n - 1) * S26_HMAX;
            hbase = s_h;
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = (e + sycl::min(c, n - 1)) * hb;
        }
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            typename S::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) w[q] = S::load(wr[q], kbx, iqs);
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += S::apply(w[q], hbase + off[c] + kbx * (F::qk / 32), iqs);
        }
        if constexpr (TS) {
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            const int q = j / GRP_NC, c = j % GRP_NC;
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC && c < n)
                out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = sum;
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = s[q][c];
    }
}

template <int TG, int TD, bool LX, int RG, int RD, bool TS>
void s27_launch(const NativeExpertLayout &L, int64_t cap_groups,
                dpct::queue_ptr s, const unsigned long long *grp_ptr,
                const int32_t *grp_start, const int32_t *n_groups,
                const int32_t *ent_dst, const int32_t *ent_tok,
                const block_q8_1 *X, float *gate, float *up, float *h,
                block_q8_1 *hq, float *out, long long nh) {
    const dpct::dim3 ggu((unsigned)(2 * L.n_ff / (GU_ROWS * RG)),
                         (unsigned)cap_groups);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class s27_gu_kernel_7906c8, dpct_kernel_scalar<TG>,
                             dpct_kernel_scalar<LX>, dpct_kernel_scalar<RG>,
                             dpct_kernel_scalar<TS>>>(
            sycl::nd_range<3>(ggu * sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                s27_gu_kernel<TG, LX, RG, TS>(grp_ptr, grp_start, n_groups,
                                              ent_tok, X, L, gate, up);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class s26_swiglu_q8_1_kernel_a7f255>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                s26_swiglu_q8_1_kernel(gate, up, h, hq, nh);
            });
    }
    const dpct::dim3 gd((unsigned)(L.n_embd / (8 * RD)), (unsigned)cap_groups);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class s27_down_kernel_d38356,
                             dpct_kernel_scalar<TD>, dpct_kernel_scalar<LX>,
                             dpct_kernel_scalar<RD>, dpct_kernel_scalar<TS>>>(
            sycl::nd_range<3>(gd * sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                s27_down_kernel<TD, LX, RD, TS>(grp_ptr, grp_start, n_groups,
                                                ent_dst, hq, L, out);
            });
    }
}
template <int TG, int TD>
void s27_launch_ts(bool ts, const NativeExpertLayout &L, int64_t cap_groups,
                   dpct::queue_ptr s, const unsigned long long *grp_ptr,
                   const int32_t *grp_start, const int32_t *n_groups,
                   const int32_t *ent_dst, const int32_t *ent_tok,
                   const block_q8_1 *X, float *gate, float *up, float *h,
                   block_q8_1 *hq, float *out, long long nh) {
    if (ts) s27_launch<TG, TD, true, 4, 4, true>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh);
    else s27_launch<TG, TD, true, 4, 4, false>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh);
}

#endif

// SYCL port: lanes per row of the port's grouped expert kernels at run time (STRATA_GU_LANES / STRATA_DOWN_LANES:
// 4, 8, 16 or 32; default STRATA_EXPERT_LANES), 256 / lanes rows per work-group. Gate/up rows are 80 dot calls
// (2560 wide), the IQ4_NL down rows 40 (640 wide), so the best split can differ by role.
inline int lanes_env(const char* name) {
    const char* v = std::getenv(name);
    const int n = v ? std::atoi(v) : kExpertLanes;
    return n == 4 || n == 8 || n == 16 || n == 32 ? n : kExpertLanes;
}
inline int gu_lanes() { static const int v = lanes_env("STRATA_GU_LANES"); return v; }
inline int down_lanes() { static const int v = lanes_env("STRATA_DOWN_LANES"); return v; }
// STRATA_KQ_GU_LANES (opt-in): the lane split of the K-quant / Q8_0 gate/up rows alone (Unsloth's files), default STRATA_GU_LANES.
// B70, 4-token windows, split on: 16 lanes reads Q4_K/Q5_1 experts at 439 GB/s against 426 at 8; the reduction order differs
// (other bits), so it stays opt-in.
inline int kq_gu_lanes() { static const int v = std::getenv("STRATA_KQ_GU_LANES") ? lanes_env("STRATA_KQ_GU_LANES") : gu_lanes(); return v; }
template <int TG, int LN, bool KM = true>
void launch_gu_port(unsigned groups, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                    const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X,
                    const NativeExpertLayout &L, float *gate, float *up) {
    constexpr int ROWS = 256 / LN;
    const unsigned gx = (unsigned) ((2 * L.n_ff + ROWS - 1) / ROWS);
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    s->parallel_for<dpct_kernel_name<class native_gu_port, dpct_kernel_scalar<TG>, dpct_kernel_scalar<LN>, dpct_kernel_scalar<KM>>>(
        sycl::nd_range<3>(sycl::range<3>(1, groups, (size_t) gx * 256), sycl::range<3>(1, 1, 256)), exp_props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_gu_kernel<TG, LN, KM>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        });
}
// STRATA_KQ_SPLIT=0 (or native_expert_set_kq_split(false)): the K-quant / Q5_1 / Q8_0 lane kernels without the
// multi-entry split (one weight read per entry, as before); the outputs are the same bits either way
bool g_kq_split = [] { const char* v = std::getenv("STRATA_KQ_SPLIT"); return !(v && v[0] == '0'); }();
template <int TG, int LN>
void gu_port_km(unsigned groups, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X, const NativeExpertLayout &L,
                float *gate, float *up) {
    if constexpr (KqSplit<TG>::has) {
        if (!g_kq_split) { launch_gu_port<TG, LN, false>(groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); return; }
    }
    launch_gu_port<TG, LN, true>(groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
}
template <int TG>
void launch_gu_lanes(dpct::dim3 grid, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                     const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X,
                     const NativeExpertLayout &L, float *gate, float *up) {
    switch (KqSplit<TG>::has ? kq_gu_lanes() : gu_lanes()) {
        case 4: gu_port_km<TG, 4>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 16: gu_port_km<TG, 16>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 32: gu_port_km<TG, 32>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        default: gu_port_km<TG, 8>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
    }
}
template <int TD, int LN, bool KM = true>
void launch_down_port(unsigned groups, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                      const int32_t *n_groups, const int32_t *ent_dst, const block_q8_1 *hq,
                      const NativeExpertLayout &L, float *out) {
    constexpr int ROWS = 256 / LN;
    const unsigned gx = (unsigned) ((L.n_embd + ROWS - 1) / ROWS);
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    s->parallel_for<dpct_kernel_name<class native_down_port, dpct_kernel_scalar<TD>, dpct_kernel_scalar<LN>, dpct_kernel_scalar<KM>>>(
        sycl::nd_range<3>(sycl::range<3>(1, groups, (size_t) gx * 256), sycl::range<3>(1, 1, 256)), exp_props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_down_kernel<TD, LN, KM>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        });
}
template <int TD, int LN>
void down_port_km(unsigned groups, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                  const int32_t *n_groups, const int32_t *ent_dst, const block_q8_1 *hq, const NativeExpertLayout &L,
                  float *out) {
    if constexpr (KqSplit<TD>::has) {
        if (!g_kq_split) { launch_down_port<TD, LN, false>(groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); return; }
    }
    launch_down_port<TD, LN, true>(groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}
template <int TD>
void launch_down_lanes(dpct::dim3 grid, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                       const int32_t *n_groups, const int32_t *ent_dst, const block_q8_1 *hq,
                       const NativeExpertLayout &L, float *out) {
    switch (down_lanes()) {
        case 4: down_port_km<TD, 4>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 16: down_port_km<TD, 16>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 32: down_port_km<TD, 32>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        default: down_port_km<TD, 8>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
    }
}

// STRATA_EXPERT_SPLIT=1 (opt-in): upstream's multi-entry kernels, one warp per row, GU_ROWS rows a group, launched with
// their own grid.  The sub-warp variants (SUB16 / SUB_DOWN) are not launched here: they are bitwise equal to the
// generic path and need their own row counts; the port's lane kernels are the default for every format.
template <int TG, bool STAGE_GRID>
void launch_gu_multi(dpct::dim3 grid, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                     const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X,
                     const NativeExpertLayout &L, float *gate, float *up) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    const sycl::range<3> g(1, grid.y, (size_t) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS));
    s->parallel_for<dpct_kernel_name<class native_gu_multi_split, dpct_kernel_scalar<TG>,
                                     dpct_kernel_scalar<STAGE_GRID>>>(
        sycl::nd_range<3>(sycl::range<3>(1, g[1], g[2] * 256), sycl::range<3>(1, 1, 256)), exp_props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_gu_multi_kernel<TG, STAGE_GRID, false>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        });
}

template <int TG>
void launch_gu(dpct::dim3 grid, dpct::queue_ptr s,
               const unsigned long long *grp_ptr, const int32_t *grp_start,
               const int32_t *n_groups, const int32_t *ent_tok,
               const block_q8_1 *X, const NativeExpertLayout &L, float *gate,
               float *up) {
    if constexpr (!kSplit<TG>) {
        launch_gu_lanes<TG>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else if (g_old_kernels || !g_split_multi) {
        launch_gu_lanes<TG>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else {
        if constexpr (kStageIqGrid<TG>) {
            if (g_stage_grid) {
                launch_gu_multi<TG, true>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
                return;
            }
        }
        launch_gu_multi<TG, false>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    }
}

template <int TD>
void launch_down_multi(dpct::dim3 grid, dpct::queue_ptr s, const unsigned long long *grp_ptr,
                       const int32_t *grp_start, const int32_t *n_groups, const int32_t *ent_dst,
                       const block_q8_1 *hq, const NativeExpertLayout &L, float *out) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    const size_t gx = (size_t) ((L.n_embd + GU_ROWS - 1) / GU_ROWS);
    s->parallel_for<dpct_kernel_name<class native_down_multi_split, dpct_kernel_scalar<TD>>>(
        sycl::nd_range<3>(sycl::range<3>(1, grid.y, gx * 256), sycl::range<3>(1, 1, 256)), exp_props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_down_multi_kernel<TD, false>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        });
}

template <int TD>
void launch_down(dpct::dim3 grid, dpct::queue_ptr s,
                 const unsigned long long *grp_ptr, const int32_t *grp_start,
                 const int32_t *n_groups, const int32_t *ent_dst,
                 const block_q8_1 *hq, const NativeExpertLayout &L,
                 float *out) {
    if constexpr (!kSplit<TD>) {
        launch_down_lanes<TD>(grid, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else if (g_old_kernels || !g_split_multi) {
        launch_down_lanes<TD>(grid, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else {
        launch_down_multi<TD>(grid, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    }
}

}  // namespace

void iq_set_old_kernels(bool old) { g_old_kernels = old; }
void native_expert_set_kq_split(bool on) { g_kq_split = on; }
bool iq_old_kernels() { return g_old_kernels; }

// APR (prism.hadamard): see iq_kernels.hpp.  A block that is not a power of two in [2, 1024] dividing the row width exits with a
// message, like check() does for a failed launch.
namespace {
void hadamard_bad_block(const char* what, int block, int64_t width) {
    std::fprintf(stderr, "%s: Hadamard block %d is not a power of two in [2, 1024] dividing the row width %lld "
                         "(hadamard.txt's block and the expert widths disagree)\n", what, block, (long long) width);
    std::exit(1);
}
constexpr int64_t kHadamardMaxRows = 65535;   // the CUDA grid limit; a prompt chunk's rows go up in the same slices
bool hadamard_block_ok(int block, int64_t width) {
    return block >= 2 && block <= 1024 && (block & (block - 1)) == 0 && width % block == 0;
}
}  // namespace

void hadamard_rows(const float* x, float* y, int64_t rows, int64_t width, int block, const float* signs, void* stream) {
    if (rows <= 0) return;
    if (!hadamard_block_ok(block, width)) hadamard_bad_block("hadamard_rows", block, width);
    dpct::queue_ptr s = strata::q_of(stream);
    for (int64_t r0 = 0; r0 < rows; r0 += kHadamardMaxRows) {
        const int64_t n = std::min<int64_t>(kHadamardMaxRows, rows - r0);
        launch_hadamard_rows<float>(x + r0 * width, y + r0 * width, n, width, block, signs, s);
    }
    check("hadamard_rows");
}

void hadamard_rows_f16(uint16_t* x, int64_t rows, int64_t width, int block, const float* signs, void* stream) {
    if (rows <= 0) return;
    if (!hadamard_block_ok(block, width)) hadamard_bad_block("hadamard_rows_f16", block, width);
    dpct::queue_ptr s = strata::q_of(stream);
    for (int64_t r0 = 0; r0 < rows; r0 += kHadamardMaxRows) {
        const int64_t n = std::min<int64_t>(kHadamardMaxRows, rows - r0);
        launch_hadamard_rows<sycl::half>((const sycl::half*) x + r0 * width, (sycl::half*) x + r0 * width, n, width, block,
                                         signs, s);
    }
    check("hadamard_rows_f16");
}

const float* hadamard_device_signs(const float* host, int width) {
    if (host == nullptr || width <= 0) return nullptr;
    static std::mutex mu;
    static std::map<std::tuple<int, const float*, int>, float*> table;   // (device, host vector, width)
    const int dev = dpct::get_current_device_id();
    std::lock_guard<std::mutex> lock(mu);
    float*& d = table[std::make_tuple(dev, host, width)];
    if (d == nullptr) {
        sycl::queue& q = dpct::get_in_order_queue();
        d = (float*) strata::malloc_device_guarded((size_t) width * sizeof(float), q, "Hadamard signs");
        q.memcpy(d, host, (size_t) width * sizeof(float)).wait();
    }
    return d;
}

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30; }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 3: return (size_t) (n / 32) * sizeof(block_q4_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 144: return (size_t) (n / 128) * sizeof(block_tq2_t);   // APR trellis types
        case 145: return (size_t) (n / 128) * sizeof(block_tqk6);
        case 146: return (size_t) (n / 128) * sizeof(block_tqk7);
        case 30: return (size_t) n * 2;   // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class quantize_q8_1_kernel_422964>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((n + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        quantize_q8_1_kernel(x, (block_q8_1 *)y, n);
                    });
    }
    check("quantize_q8_1_rows");
}

// SYCL port: a wide kernel for the dense IQ3_S matrices (attn_q, attn_gate, ssm_out, attn_k, the shared experts'
// gate/up), shaped like the Q4_K..IQ4_XS wide kernels: 8 lanes share a 256-block, lane g takes sub-block g, so a
// 32-lane sub-group has four blocks in flight. Each sub-block's value is vec_dot_iq3_s_q8_1's; only the order the
// sub-blocks are summed in differs from mmvq_multi_kernel, so the output differs in the last bits.
template <int NCOLS>
__dpct_inline__ void iq3s_wide_kernel(const uint8_t* __restrict__ w, size_t rb, const block_q8_1* __restrict__ x,
                                      float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = (int) item.get_local_id(2), warp = (int) item.get_local_id(1);
    const int row = (int) item.get_group(2) * (int) item.get_local_range(1) + warp;
    if (row >= n_out) return;
    const int nb = n_in / QK_K, xs = n_in / QK8_1;
    const int g = lane & 7, sub = lane >> 3;
    const block_iq3_s* wr = (const block_iq3_s*) (w + (size_t) row * rb);
    float acc[NCOLS];
#pragma unroll
    for (int c = 0; c < NCOLS; ++c) acc[c] = 0.0f;
    for (int kbx = sub; kbx < nb; kbx += 4) {
        const block_iq3_s* b = wr + kbx;
        const sycl::int2 qs_packed = load8_a2(b->qs + 8 * g);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = b->qh[g];
        const int signs_packed_32 = load4_a2(b->signs + 4 * g);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        const int scale = 1 + 2 * ((b->scales[g / 2] >> (4 * (g & 1))) & 0x0F);
        const float d3 = sycl::vec<sycl::half, 1>(b->d).convert<float, sycl::rounding_mode::automatic>()[0];
        int wv[8];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos = sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                                   iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = swar_ne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21));
            const int signs1 = swar_ne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17));
            wv[l0 + 0] = swar_sub4(grid_pos.x() ^ signs0, signs0);
            wv[l0 + 1] = swar_sub4(grid_pos.y() ^ signs1, signs1);
        }
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const block_q8_1* xb = x + (size_t) c * xs + (size_t) kbx * (QK_K / QK8_1) + g;
            const int* u = reinterpret_cast<const int*>(xb->qs);
            int sumi = 0;
#pragma unroll
            for (int l = 0; l < 8; ++l) sumi = ggml_cuda_dp4a(wv[l], u[l], sumi);
            sumi *= scale;
            const float d = d3 * xb->ds[0];
            acc[c] += d * sumi;
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        float v = acc[c];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[(size_t) c * n_out + row] = v;
    }
}
// STRATA_IQ3S_WIDE=1: the wide kernel; unset or 0: the multi kernel (default).
bool iq3s_wide_on() {
    static const bool v = [] { const char* e = std::getenv("STRATA_IQ3S_WIDE"); return e && std::atoi(e) != 0; }();
    return v;
}
template <int NCOLS>
void launch_iq3s_wide_t(const uint8_t* W, size_t rb, const block_q8_1* X, float* y, int n_in, int n_out, dpct::queue_ptr s) {
    constexpr int WPG = 4;   // sub-groups per work-group
    const unsigned groups = unsigned((n_out + WPG - 1) / WPG);
    s->parallel_for<dpct_kernel_name<class iq3s_wide_k, dpct_kernel_scalar<NCOLS>>>(
        sycl::nd_range<3>(sycl::range(1, WPG, groups * 32), sycl::range(1, WPG, 32)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { iq3s_wide_kernel<NCOLS>(W, rb, X, y, n_in, n_out); });
}
bool launch_iq3s_wide(const uint8_t* W, size_t rb, const block_q8_1* X, float* y, int n_in, int n_out, int ncols, dpct::queue_ptr s) {
    if (!iq3s_wide_on() || n_in % QK_K != 0) return false;
    switch (ncols) {
        case 1: launch_iq3s_wide_t<1>(W, rb, X, y, n_in, n_out, s); return true;
        case 2: launch_iq3s_wide_t<2>(W, rb, X, y, n_in, n_out, s); return true;
        case 3: launch_iq3s_wide_t<3>(W, rb, X, y, n_in, n_out, s); return true;
        case 4: launch_iq3s_wide_t<4>(W, rb, X, y, n_in, n_out, s); return true;
        case 5: launch_iq3s_wide_t<5>(W, rb, X, y, n_in, n_out, s); return true;
        case 6: launch_iq3s_wide_t<6>(W, rb, X, y, n_in, n_out, s); return true;
        case 7: launch_iq3s_wide_t<7>(W, rb, X, y, n_in, n_out, s); return true;
        case 8: launch_iq3s_wide_t<8>(W, rb, X, y, n_in, n_out, s); return true;
        default: return false;
    }
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    dpct::queue_ptr s = strata::q_of(stream);
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    if (t == 21 && launch_iq3s_wide(W, rb, X, y, n_in, n_out, ncols, s)) { check("iq_mmvq"); return; }
    switch (t) {
#define STRATA_MMVQ(T) case T: launch_mmvq<T>(W, rb, X, y, n_in, n_out, ncols, s); break;
        STRATA_MMVQ_FMTS(STRATA_MMVQ)
#undef STRATA_MMVQ
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream, const float* scale) {
    // scale: NVFP4's s_down (CUDA only; the SYCL port has no NVFP4 experts, so it is never given here)
    if (n % 256 != 0 || !is_iq(t) || scale != nullptr) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<dpct_kernel_name<
                    class dequant_flat_kernel_c7cec2, sycl::half>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)(n / 256)) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dequant_flat_kernel<sycl::half>(
                            t, src, (sycl::half *)dst);
                    });
            });
    }
    check("iq_dequant_f16");
}

namespace {
__dpct_inline__ void embed_rows_kernel(
    int ty, const uint8_t *__restrict__ table, size_t row_bytes,
    const int32_t *__restrict__ tokens, int64_t n_embd, float *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t b = item_ct1.get_group(2);
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t)t * n_embd + b * QK_K,
                       item_ct1.get_local_id(2));
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<
                    dpct_kernel_name<class embed_rows_kernel_284a43>>(
                    sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                                  (unsigned)(n_embd / 256)) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        embed_rows_kernel(
                            t, (const uint8_t *)table, row_bytes, tokens,
                            n_embd, out);
                    });
            });
    }
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<
                    dpct_kernel_name<class dequant_flat_kernel_9052bc, float>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)(n / 256)) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dequant_flat_kernel<float>(
                            t, src, dst);
                    });
            });
    }
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream,
                       const float* tail) {
    // checked like the other entry points: an unknown type used to leave `dst` unwritten, a wrong prompt and no error
    // (tail: NVFP4's per-expert scales, CUDA only - never given on SYCL)
    if (n_embd % 256 != 0 || !is_iq(t) || tail != nullptr) { std::fprintf(stderr, "iq_dequant_gu_f16: type %d / %lld\n", t, (long long) n_embd); std::exit(1); }
    const int64_t per_row = n_embd / 256;
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<
                    dpct_kernel_name<class dequant_gu_kernel_2397ca>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 2, (unsigned)(n_ff * per_row)) *
                            sycl::range(1, 1, 32),
                        sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dequant_gu_kernel(
                            t, gate, up, per_row, (sycl::half *)dst);
                    });
            });
    }
    check("iq_dequant_gu_f16");
}

// 0.1.43: gate/up and down of one expert in one launch.  Work-group g of the first 2 * n_ff * per_row is the gu kernel's
// (row-superblock g / 2, parity g % 2); the rest are the flat kernel's superblocks of the down matrix.
namespace {
__dpct_inline__ void dequant_expert_kernel(int gt, int dt, const void *__restrict__ gate, const void *__restrict__ up,
                                           const void *__restrict__ down, int64_t per_row, int64_t gu_wgs,
                                           sycl::half *__restrict__ ygu, sycl::half *__restrict__ yd) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t g = item_ct1.get_group(2);
    if (g < gu_wgs) {
        const int64_t i = g >> 1;
        const int parity = (int) (g & 1);
        const int64_t r = i / per_row, c = i % per_row;
        dq_dispatch<sycl::half>(gt, parity ? up : gate, i, ygu + ((2 * r + parity) * per_row + c) * QK_K,
                                item_ct1.get_local_id(2));
    } else {
        const int64_t j = g - gu_wgs;
        dq_dispatch<sycl::half>(dt, down, j, yd + j * QK_K, item_ct1.get_local_id(2));
    }
}
}  // namespace

void iq_dequant_expert_f16(int gt, int dt, const void* gate, const void* up, const void* down, int64_t n_ff,
                           int64_t n_embd, uint16_t* dst_gu, uint16_t* dst_d, void* stream) {
    if (n_embd % 256 != 0 || (n_ff * n_embd) % 256 != 0 || !is_iq(gt) || !is_iq(dt)) {
        std::fprintf(stderr, "iq_dequant_expert_f16: types %d/%d, %lld x %lld\n", gt, dt, (long long) n_ff, (long long) n_embd);
        std::exit(1);
    }
    const int64_t per_row = n_embd / 256, gu_wgs = 2 * n_ff * per_row, d_wgs = n_ff * n_embd / 256;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(strata::q_of(stream)->get_device(), {sycl::aspect::fp16});
        strata::q_of(stream)->submit([&](sycl::handler &cgh) {
            cgh.parallel_for<dpct_kernel_name<class dequant_expert_kernel_7d2c41>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned) (gu_wgs + d_wgs)) * sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    dequant_expert_kernel(gt, dt, gate, up, down, per_row, gu_wgs, (sycl::half *) dst_gu,
                                          (sycl::half *) dst_d);
                });
        });
    }
    check("iq_dequant_expert_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}


#if defined(STRATA_HIP_GFX906)
// ---- AMD layouts for the grouped native experts (STRATA_EXP_MODE; 0 = the CUDA one above).
// 1 (W64): a row per 64-lane wavefront - 4x the wavefronts, ~1-2 calls per lane, a 64-lane butterfly.
// 2 (R2):  a 32-lane logical warp computes TWO rows in one loop - two independent load chains in flight.
// Either way a (row, entry) sum does not depend on the window size.
template<int TY>
__device__ __forceinline__ float row_dot64(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 64) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) s += __shfl_xor(s, o, 64);
    return s;
}
template<int TY>
__device__ __forceinline__ void row_dot2(const uint8_t* r0, const uint8_t* r1, const block_q8_1* x, int nb, int lane,
                                         float& o0, float& o1) {
    using F = Fmt<TY>;
    float s0 = 0.0f, s1 = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const block_q8_1* xk = x + kbx * (F::qk / 32);
        const float a = F::dot(r0, xk, kbx, iqs);
        const float b = F::dot(r1, xk, kbx, iqs);
        s0 += a;
        s1 += b;
    }
    o0 = warp_sum(s0);
    o1 = warp_sum(s1);
}
template<int TY, int NR>
__device__ __forceinline__ void row_dotn(const uint8_t* const* r, const block_q8_1* x, int nb, int lane, float* o) {
    using F = Fmt<TY>;
    float acc[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) acc[i] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const block_q8_1* xk = x + kbx * (F::qk / 32);
        float d[NR];
#pragma unroll
        for (int i = 0; i < NR; ++i) d[i] = F::dot(r[i], xk, kbx, iqs);
#pragma unroll
        for (int i = 0; i < NR; ++i) acc[i] += d[i];
    }
#pragma unroll
    for (int i = 0; i < NR; ++i) o[i] = warp_sum(acc[i]);
}
template<int TG, int MODE>
__global__ void __launch_bounds__(256) native_gu_amd_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_tok,
                                                            const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                            float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = 2 * (int) L.n_ff;
    auto wrow = [&](int row) -> const uint8_t* {
        const bool is_up = row >= L.n_ff;
        return blob + (is_up ? L.up_off : 0) + (size_t) (is_up ? row - (int) L.n_ff : row) * L.gu_row;
    };
    auto put = [&](int row, int e, float v) {
        const bool is_up = row >= L.n_ff;
        (is_up ? up : gate)[(size_t) e * L.n_ff + (is_up ? row - (int) L.n_ff : row)] = v;
    };
    if constexpr (MODE == 1) {
        const int lane = threadIdx.x & 63, row = blockIdx.x * 4 + (threadIdx.x >> 6);
        if (row >= nrow) return;
        const uint8_t* wr = wrow(row);
        for (int e = e0; e < e1; ++e) {
            const float v = row_dot64<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
            if (lane == 0) put(row, e, v);
        }
    } else if constexpr (MODE == 4) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int row0 = blockIdx.x * 32 + warp;
        if (row0 >= nrow) return;
        const uint8_t* w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = wrow(row0 + 8 * i < nrow ? row0 + 8 * i : row0);
        for (int e = e0; e < e1; ++e) {
            float o[4];
            row_dotn<TG, 4>(w, xq + (size_t) ent_tok[e] * xb, nb, lane, o);
            if (lane == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) if (row0 + 8 * i < nrow) put(row0 + 8 * i, e, o[i]);
        }
    } else {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int row0 = blockIdx.x * 16 + warp, row1 = row0 + 8;
        if (row0 >= nrow) return;
        const bool two = row1 < nrow;
        const uint8_t* w0 = wrow(row0);
        const uint8_t* w1 = wrow(two ? row1 : row0);
        for (int e = e0; e < e1; ++e) {
            float a, b;
            row_dot2<TG>(w0, w1, xq + (size_t) ent_tok[e] * xb, nb, lane, a, b);
            if (lane == 0) { put(row0, e, a); if (two) put(row1, e, b); }
        }
    }
}
template<int TD, int MODE>
__global__ void __launch_bounds__(256) native_down_amd_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_dst,
                                                              const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                              float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = (int) L.n_embd;
    if constexpr (MODE == 1) {
        const int lane = threadIdx.x & 63, r = blockIdx.x * 4 + (threadIdx.x >> 6);
        if (r >= nrow) return;
        const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
        for (int e = e0; e < e1; ++e) {
            const float v = row_dot64<TD>(wr, hq + (size_t) e * hb, nb, lane);
            if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = v;
        }
    } else if constexpr (MODE == 4) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int r0 = blockIdx.x * 32 + warp;
        if (r0 >= nrow) return;
        const uint8_t* w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = blob + L.down_off + (size_t) (r0 + 8 * i < nrow ? r0 + 8 * i : r0) * L.d_row;
        for (int e = e0; e < e1; ++e) {
            float o[4];
            row_dotn<TD, 4>(w, hq + (size_t) e * hb, nb, lane, o);
            if (lane == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) if (r0 + 8 * i < nrow) out[(size_t) ent_dst[e] * L.n_embd + r0 + 8 * i] = o[i];
        }
    } else {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int r0 = blockIdx.x * 16 + warp, r1 = r0 + 8;
        if (r0 >= nrow) return;
        const bool two = r1 < nrow;
        const uint8_t* w0 = blob + L.down_off + (size_t) r0 * L.d_row;
        const uint8_t* w1 = blob + L.down_off + (size_t) (two ? r1 : r0) * L.d_row;
        for (int e = e0; e < e1; ++e) {
            float a, b;
            row_dot2<TD>(w0, w1, hq + (size_t) e * hb, nb, lane, a, b);
            if (lane == 0) {
                out[(size_t) ent_dst[e] * L.n_embd + r0] = a;
                if (two) out[(size_t) ent_dst[e] * L.n_embd + r1] = b;
            }
        }
    }
}
// ---- 5 (LDS): gate/up of the grid formats (IQ2_S, IQ3_XXS, IQ3_S) with the codebook grid and the group's q8_1
// activations in LDS.  The bench showed gate/up compute-bound, not bandwidth-bound (T = 2 costs ~1.8x T = 1 on the
// same weights, 110-130 GB/s): each call issues its grid lookups and eight activation loads through the vector
// memory path after the weight load.  Same calls, same lane-strided k, same R2 row pairs, same warp_sum: the
// output is bitwise the mode-2 one.
template<int TY> struct GridOf;
template<> struct GridOf<22> { using T = uint64_t; static constexpr int N = 1024; __device__ static const T* src() { return iq2s_grid; } };
template<> struct GridOf<18> { using T = uint32_t; static constexpr int N = 256; __device__ static const T* src() { return iq3xxs_grid; } };
template<> struct GridOf<21> { using T = uint32_t; static constexpr int N = 512; __device__ static const T* src() { return iq3s_grid; } };
template<> struct GridOf<23> { using T = uint32_t; static constexpr int N = 1; __device__ static const T* src() { return iq3s_grid; } };   // IQ4_XS: no grid

// SG = 1 (mode 6): the signs without byte-SIMD ops, which gfx906 lacks (strata_vcmpne4 / strata_vsub4 unpack to
// ~12 word ops each).  m = the sign nibble as 0x00/0xff bytes; (g ^ m) is g or -g-1 per byte, so
// dp4a(g ^ m, u) - dp4a(m, u) = sum(s g u) exactly - the same integers, the same floats.
__device__ __forceinline__ int nib_mask(uint32_t n) {
    const uint32_t b = __umul24(n & 0xFu, 0x204081u) & 0x01010101u;
    return (int) ((b << 8) - b);
}
template<int TY, int SG> struct DotG;
template<int SG> struct DotG<22, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint64_t* grid) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        int& acc = l0 < 4 ? sumi0 : sumi1;
        if constexpr (SG == 1) {
            const int m0 = nib_mask(signs_packed_8[l0 / 2]), m1 = nib_mask(signs_packed_8[l0 / 2] >> 4);
            acc = ggml_cuda_dp4a(grid_pos[0] ^ m0, u0, acc);
            acc = ggml_cuda_dp4a(grid_pos[1] ^ m1, u1, acc);
            acc -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
            acc = ggml_cuda_dp4a(grid_l, u0, acc);
            acc = ggml_cuda_dp4a(grid_h, u1, acc);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };
template<int SG> struct DotG<18, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint32_t* grid) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if constexpr (SG == 1) {
            const int m0 = nib_mask(signs), m1 = nib_mask(signs >> 4);
            sumi = ggml_cuda_dp4a(grid_pos.x ^ m0, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_pos.y ^ m1, u1, sumi);
            sumi -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
        }
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };
template<int SG> struct DotG<21, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint32_t* grid) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if constexpr (SG == 1) {
            const int m0 = nib_mask(signs_packed_8[l0 / 2]), m1 = nib_mask(signs_packed_8[l0 / 2] >> 4);
            sumi = ggml_cuda_dp4a(grid_pos.x ^ m0, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_pos.y ^ m1, u1, sumi);
            sumi -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
        }
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };

template<int SG> struct DotG<23, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint32_t*) { return vec_dot_iq4_xs_q8_1(vbq, bq8_1, kbx, iqs); } };

constexpr int LDS_RB = 64;    // gate/up rows per block (4 R2 passes of 16)
constexpr int LDS_NT = 4;     // entries whose activations sit in LDS at once

template<int TG, int SG, bool TI = false>
__global__ void __launch_bounds__(256) native_gu_lds_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_tok,
                                                            const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                            float* __restrict__ gate, float* __restrict__ up) {
    using GT = typename GridOf<TG>::T;
    using F = Fmt<TG>;
    __shared__ GT sgrid[GridOf<TG>::N];
    extern __shared__ int sx_raw[];   // LDS_NT tokens x xb blocks of q8_1 (36 bytes = 9 ints each)
    const int g = blockIdx.y;
    if (g >= *n_groups) return;       // uniform over the block
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const GT* gsrc = GridOf<TG>::src();
    for (int i = tid; i < GridOf<TG>::N; i += 256) sgrid[i] = gsrc[i];
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = 2 * (int) L.n_ff;
    const block_q8_1* sx = (const block_q8_1*) sx_raw;
    auto wrow = [&](int row) -> const uint8_t* {
        const bool is_up = row >= L.n_ff;
        return blob + (is_up ? L.up_off : 0) + (size_t) (is_up ? row - (int) L.n_ff : row) * L.gu_row;
    };
    auto put = [&](int row, int e, float v) {
        const bool is_up = row >= L.n_ff;
        (is_up ? up : gate)[(size_t) e * L.n_ff + (is_up ? row - (int) L.n_ff : row)] = v;
    };
    for (int c0 = e0; c0 < e1; c0 += LDS_NT) {
        const int cn = min(LDS_NT, e1 - c0);
        __syncthreads();
        for (int j = 0; j < cn; ++j) {
            const int* src = (const int*) (xq + (size_t) ent_tok[c0 + j] * xb);
            for (int i = tid; i < xb * 9; i += 256) sx_raw[j * xb * 9 + i] = src[i];
        }
        __syncthreads();
        for (int p = 0; p < LDS_RB / 16; ++p) {
            const int row0 = blockIdx.x * LDS_RB + p * 16 + warp, row1 = row0 + 8;
            if (row0 >= nrow) break;
            const bool two = row1 < nrow;
            const uint8_t* w0 = wrow(row0);
            const uint8_t* w1 = wrow(two ? row1 : row0);
            if constexpr (TI) {   // mode 7: the tokens inside the k loop - one weight load per k for all of them
                auto pass = [&](auto ntc) {
                    constexpr int NTC = decltype(ntc)::value;
                    float s0[NTC], s1[NTC];
#pragma unroll
                    for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                    for (int k = lane; k < nb * F::ipb; k += 32) {
                        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                        for (int j = 0; j < NTC; ++j) {
                            const block_q8_1* xk = sx + (size_t) j * xb + kbx * (F::qk / 32);
                            const float a = DotG<TG, SG>::f(w0, xk, kbx, iqs, sgrid);
                            const float b = DotG<TG, SG>::f(w1, xk, kbx, iqs, sgrid);
                            s0[j] += a;
                            s1[j] += b;
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
                        if (lane == 0) { put(row0, c0 + j, a); if (two) put(row1, c0 + j, b); }
                    }
                };
                switch (cn) {
                    case 1: pass(std::integral_constant<int, 1>{}); break;
                    case 2: pass(std::integral_constant<int, 2>{}); break;
                    case 3: pass(std::integral_constant<int, 3>{}); break;
                    default: pass(std::integral_constant<int, 4>{}); break;
                }
                continue;
            }
            for (int j = 0; j < cn; ++j) {
                const block_q8_1* x = sx + (size_t) j * xb;
                float s0 = 0.0f, s1 = 0.0f;
                for (int k = lane; k < nb * F::ipb; k += 32) {
                    const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
                    const block_q8_1* xk = x + kbx * (F::qk / 32);
                    const float a = DotG<TG, SG>::f(w0, xk, kbx, iqs, sgrid);
                    const float b = DotG<TG, SG>::f(w1, xk, kbx, iqs, sgrid);
                    s0 += a;
                    s1 += b;
                }
                s0 = warp_sum(s0);
                s1 = warp_sum(s1);
                if (lane == 0) { put(row0, c0 + j, s0); if (two) put(row1, c0 + j, s1); }
            }
        }
    }
}

// ---- 7 (down): the group's q8_1 h rows in LDS, the tokens inside the k loop, the R2 row pairs of mode 2 -
// the same per-(row, entry) sums in the same order: bitwise the mode-2 output.
template<int TD>
__global__ void __launch_bounds__(256) native_down_lds_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_dst,
                                                              const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                              float* __restrict__ out) {
    using F = Fmt<TD>;
    extern __shared__ int sh_raw[];   // LDS_NT entries x hb blocks of q8_1
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_ff / F::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = (int) L.n_embd;
    const block_q8_1* sh = (const block_q8_1*) sh_raw;
    for (int c0 = e0; c0 < e1; c0 += LDS_NT) {
        const int cn = min(LDS_NT, e1 - c0);
        __syncthreads();
        {
            const int* src = (const int*) (hq + (size_t) c0 * hb);   // the chunk's entries are contiguous
            for (int i = tid; i < cn * hb * 9; i += 256) sh_raw[i] = src[i];
        }
        __syncthreads();
        for (int p = 0; p < LDS_RB / 16; ++p) {
            const int r0 = blockIdx.x * LDS_RB + p * 16 + warp, r1 = r0 + 8;
            if (r0 >= nrow) break;
            const bool two = r1 < nrow;
            const uint8_t* w0 = blob + L.down_off + (size_t) r0 * L.d_row;
            const uint8_t* w1 = blob + L.down_off + (size_t) (two ? r1 : r0) * L.d_row;
            auto pass = [&](auto ntc) {
                constexpr int NTC = decltype(ntc)::value;
                float s0[NTC], s1[NTC];
#pragma unroll
                for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                for (int k = lane; k < nb * F::ipb; k += 32) {
                    const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const block_q8_1* xk = sh + (size_t) j * hb + kbx * (F::qk / 32);
                        const float a = F::dot(w0, xk, kbx, iqs);
                        const float b = F::dot(w1, xk, kbx, iqs);
                        s0[j] += a;
                        s1[j] += b;
                    }
                }
#pragma unroll
                for (int j = 0; j < NTC; ++j) {
                    const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
                    if (lane == 0) {
                        out[(size_t) ent_dst[c0 + j] * L.n_embd + r0] = a;
                        if (two) out[(size_t) ent_dst[c0 + j] * L.n_embd + r1] = b;
                    }
                }
            };
            switch (cn) {
                case 1: pass(std::integral_constant<int, 1>{}); break;
                case 2: pass(std::integral_constant<int, 2>{}); break;
                case 3: pass(std::integral_constant<int, 3>{}); break;
                default: pass(std::integral_constant<int, 4>{}); break;
            }
        }
    }
}

// ---- 8: mode 7's gate/up with SwiGLU and the q8_1 quantization in the epilogue.  A block takes h rows
// [r0, r0 + 32): gate rows r0.. and up rows r0.. (64 weight rows, the same per-(row, entry) sums as mode 7), so
// one 32-lane warp per entry holds exactly one q8_1 block of h and runs quantize_q8_1_kernel's own code on it -
// bitwise the separate kernels' hq, two launches fewer per call.
template<int TG>
__global__ void __launch_bounds__(256) native_gu_fused_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_tok,
                                                              const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                              block_q8_1* __restrict__ hq) {
    using GT = typename GridOf<TG>::T;
    using F = Fmt<TG>;
    __shared__ GT sgrid[GridOf<TG>::N];
    __shared__ float res[LDS_NT][64];
    extern __shared__ int sx_raw[];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const GT* gsrc = GridOf<TG>::src();
    for (int i = tid; i < GridOf<TG>::N; i += 256) sgrid[i] = gsrc[i];
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int r0 = blockIdx.x * 32;
    const block_q8_1* sx = (const block_q8_1*) sx_raw;
    auto wrow = [&](int i) -> const uint8_t* {   // i < 32: gate row r0 + i, else up row r0 + i - 32
        return blob + (i < 32 ? (size_t) 0 : L.up_off) + (size_t) (r0 + (i & 31)) * L.gu_row;
    };
    for (int c0 = e0; c0 < e1; c0 += LDS_NT) {
        const int cn = min(LDS_NT, e1 - c0);
        __syncthreads();
        for (int j = 0; j < cn; ++j) {
            const int* src = (const int*) (xq + (size_t) ent_tok[c0 + j] * xb);
            for (int i = tid; i < xb * 9; i += 256) sx_raw[j * xb * 9 + i] = src[i];
        }
        __syncthreads();
        for (int p = 0; p < 4; ++p) {
            const int i0 = p * 16 + warp, i1 = i0 + 8;
            const uint8_t* w0 = wrow(i0);
            const uint8_t* w1 = wrow(i1);
            auto pass = [&](auto ntc) {
                constexpr int NTC = decltype(ntc)::value;
                float s0[NTC], s1[NTC];
#pragma unroll
                for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                for (int k = lane; k < nb * F::ipb; k += 32) {
                    const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const block_q8_1* xk = sx + (size_t) j * xb + kbx * (F::qk / 32);
                        const float a = DotG<TG, 1>::f(w0, xk, kbx, iqs, sgrid);
                        const float b = DotG<TG, 1>::f(w1, xk, kbx, iqs, sgrid);
                        s0[j] += a;
                        s1[j] += b;
                    }
                }
#pragma unroll
                for (int j = 0; j < NTC; ++j) {
                    const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
                    if (lane == 0) { res[j][i0] = a; res[j][i1] = b; }
                }
            };
            switch (cn) {
                case 1: pass(std::integral_constant<int, 1>{}); break;
                case 2: pass(std::integral_constant<int, 2>{}); break;
                case 3: pass(std::integral_constant<int, 3>{}); break;
                default: pass(std::integral_constant<int, 4>{}); break;
            }
        }
        __syncthreads();
        if (warp < cn) {   // swiglu_entries_kernel + quantize_q8_1_kernel, one block of 32 h values per entry
            const float gg = res[warp][lane], uu = res[warp][32 + lane];
            const float xi = (gg / (1.0f + __expf(-gg))) * uu;
            float amax = fabsf(xi), sum = xi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            const float d = q8_1_finite(amax / 127.0f);   // #606, as q8_1_store: finite blocks bit for bit
            const int8_t q = q8_1_quant(xi, d, amax);
            block_q8_1* y = hq + (size_t) (c0 + warp) * hb + blockIdx.x;
            y->qs[lane] = q;
            if (lane == 0) y->ds = q8_1_ds(d, sum);
        }
    }
}

int g_exp_mode = -1;   // native_expert_set_mode (the bench); -1 = STRATA_EXP_MODE, default 7
int exp_mode() {
    static const int m = [] { const char* v = std::getenv("STRATA_EXP_MODE"); return v ? std::atoi(v) : 7; }();
    return g_exp_mode >= 0 ? g_exp_mode : m;
}
#endif
int g_exp_phase = 0;   // the bench: 0 all, 1 gate/up + swiglu + quantize only, 2 down only

void native_expert_set_mode(int mode, int phase) {
#if defined(STRATA_HIP_GFX906)
    g_exp_mode = mode;
#else
    (void) mode;
#endif
    g_exp_phase = phase;
}

namespace {
bool g_grouped_v1 = env_on("STRATA_GROUPED_V1");
}  // namespace

void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0) { std::fprintf(stderr, "native_expert_grouped: n_ff %lld\n", (long long) L.n_ff); std::exit(1); }
    dpct::queue_ptr s = strata::q_of(stream);
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    // SYCL port: STRATA_EXPERT_V2 / STRATA_EXPERT_V2K (upstream's AMD-tuned S26 / S27 kernels) are not built here
    static const bool v2_note = [] {
        const char* a = std::getenv("STRATA_EXPERT_V2");
        const char* b = std::getenv("STRATA_EXPERT_V2K");
        if ((a && a[0] == '1') || (b && b[0] == '1'))
            std::fprintf(stderr, "note: STRATA_EXPERT_V2 / STRATA_EXPERT_V2K are AMD-only experiments, ignored on SYCL\n");
        return true;
    }();
    (void) v2_note;
    const bool v1 = g_grouped_v1;
    // #363: a verify window's PCIe call usually has no group: `grid_groups` block rows stride over the groups instead
    // of one row per possible group (the results do not depend on it)
    const int64_t gy = (v1 || grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;
    const dpct::dim3 ggu((unsigned)((2 * L.n_ff + kExpertRows - 1) / kExpertRows),
                         (unsigned)gy);
#if defined(STRATA_HIP_GFX906)
    const int em0 = exp_mode();
#endif
#if defined(STRATA_HIP_GFX906)
    const bool fused_gu = em0 == 8 && (L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 || L.gu_type == 23) &&
                          L.n_ff % 32 == 0;
    if (fused_gu && g_exp_phase != 2) {
        const dim3 gl((unsigned) (L.n_ff / 32), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
        switch (L.gu_type) {
            case 18: native_gu_fused_kernel<18><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 21: native_gu_fused_kernel<21><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 23: native_gu_fused_kernel<23><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            default: native_gu_fused_kernel<22><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        }
        check("native_expert_grouped/gu fused");
    }
    if (g_exp_phase != 2 && !fused_gu) {
#else
    if (g_exp_phase != 2) {
#endif
#if defined(STRATA_HIP_GFX906)
    const bool lds_gu = (em0 == 5 || em0 == 6 || em0 == 7 || em0 == 8) && (L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 || L.gu_type == 23);
    if (lds_gu) {
        const dim3 gl((unsigned) ((2 * L.n_ff + LDS_RB - 1) / LDS_RB), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
        switch (L.gu_type) {
            case 18: if (em0 >= 7) native_gu_lds_kernel<18, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<18, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<18, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            case 21: if (em0 >= 7) native_gu_lds_kernel<21, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<21, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<21, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            case 23: if (em0 >= 7) native_gu_lds_kernel<23, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<23, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            default: if (em0 >= 7) native_gu_lds_kernel<22, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<22, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<22, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        }
    } else {
    const int em = exp_mode() >= 5 ? 2 : exp_mode();
    const dim3 ggu_amd((unsigned) ((2 * L.n_ff + (em == 1 ? 3 : em == 4 ? 31 : 15)) / (em == 1 ? 4 : em == 4 ? 32 : 16)), (unsigned) cap_groups);
#define STRATA_GU_AMD(T) \
    case T: if (em == 1) native_gu_amd_kernel<T, 1><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); \
            else if (em == 4) native_gu_amd_kernel<T, 4><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); \
            else native_gu_amd_kernel<T, 2><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
    // the AMD layouts cover the IQ packs' gate/up types; Unsloth's Q4_K / Q5_K take the CUDA layout below
    const bool amd_gu = L.gu_type == 16 || L.gu_type == 17 || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 ||
                        L.gu_type == 23 || L.gu_type == 29 || L.gu_type == 42;
    if ((em == 1 || em == 2 || em == 4) && amd_gu) {
        switch (L.gu_type) {
            STRATA_GU_AMD(16) STRATA_GU_AMD(17) STRATA_GU_AMD(18) STRATA_GU_AMD(21) STRATA_GU_AMD(22) STRATA_GU_AMD(23)
            STRATA_GU_AMD(29) STRATA_GU_AMD(42)
        }
    } else
#undef STRATA_GU_AMD
#endif
    switch (L.gu_type) {
#define STRATA_GU(T) case T: launch_gu<T>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        STRATA_GU_FMTS(STRATA_GU)
#undef STRATA_GU
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
#if defined(STRATA_HIP_GFX906)
    }
#endif
    check("native_expert_grouped/gu");
    const long long nh = (long long) cap_entries * L.n_ff;
#if defined(STRATA_HIP_GFX906)
    // gfx906: the separate SwiGLU and q8_1 passes stay the default (the A/B baseline); the RDNA one-pass
    // kernel (bitwise the same) with STRATA_HIP_SWIGLU_FUSED=1
    static const bool sw_fused = env_on("STRATA_HIP_SWIGLU_FUSED");
    const bool sw_v1 = v1 || !sw_fused;
#else
    const bool sw_v1 = v1;
#endif
    // APR: Hadamard-folded down weights take the rotated h, so h exists as floats first (the separate passes)
    if (sw_v1 || L.had_block > 0) {
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                s->parallel_for<
                    dpct_kernel_name<class swiglu_entries_kernel_a4959c>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        swiglu_entries_kernel(gate, up, h, nh);
                    });
            }
            if (L.had_block > 0)   // APR: the down weights are Hadamard-folded, so is their input (in place, every entry row)
                hadamard_rows(h, h, cap_entries, L.n_ff, L.had_block, L.d_signs, stream);
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(s->get_device(),
                                             {sycl::aspect::fp16});

                s->parallel_for<
                    dpct_kernel_name<class quantize_q8_1_kernel_f68c59>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            quantize_q8_1_kernel(h, hq, nh);
                        });
            }
    } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<
                dpct_kernel_name<class swiglu_q8_1_entries_kernel_115291>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        swiglu_q8_1_entries_kernel(gate, up, grp_start,
                                                   n_groups, (int)L.n_ff, hq);
                    });
    }
    check("native_expert_grouped/swiglu");
    }
    if (g_exp_phase == 1) return;
    const int d_rows = (!g_old_kernels && !g_no_sub16_gu && L.n_ff == 640) ? (L.d_type == 20 ? 32 : (L.d_type == 42 ? 16 : 8)) : 8;
    const dpct::dim3 gd((unsigned)((L.n_embd + d_rows - 1) / d_rows),
                        (unsigned)gy);
#if defined(STRATA_HIP_GFX906)
    if ((em0 == 7 || em0 == 8) && (L.d_type == 20 || L.d_type == 42)) {
        const dim3 gl((unsigned) ((L.n_embd + LDS_RB - 1) / LDS_RB), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_ff / 32) * sizeof(block_q8_1);
        if (L.d_type == 20) native_down_lds_kernel<20><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        else native_down_lds_kernel<42><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        check("native_expert_grouped/down");
        return;
    }
    const int em = exp_mode() >= 5 ? 2 : exp_mode();
    const dim3 gd_amd((unsigned) ((L.n_embd + (em == 1 ? 3 : em == 4 ? 31 : 15)) / (em == 1 ? 4 : em == 4 ? 32 : 16)), (unsigned) cap_groups);
#define STRATA_D_AMD(T) \
    case T: if (em == 1) native_down_amd_kernel<T, 1><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); \
            else if (em == 4) native_down_amd_kernel<T, 4><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); \
            else native_down_amd_kernel<T, 2><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
    // the AMD layouts cover the IQ packs' down types; the others (Unsloth's Q4_K / Q5_K / Q5_1 / Q8_0) take the
    // CUDA layout below
    if ((em == 1 || em == 2 || em == 4) && (L.d_type == 20 || L.d_type == 23 || L.d_type == 42)) {
        switch (L.d_type) {
            STRATA_D_AMD(20) STRATA_D_AMD(23) STRATA_D_AMD(42)
        }
    } else
#undef STRATA_D_AMD
#endif
    switch (L.d_type) {
#define STRATA_DOWN(T) case T: launch_down<T>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        STRATA_D_FMTS(STRATA_DOWN)
#undef STRATA_DOWN
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}


// A770 port (STRATA_PF_XMX=2): a group of experts' products in ONE launch, the weight tiles decoded straight from
// the quantized blobs into local memory (no FP16 copy of the expert in global memory).  Sub-group 8, joint_matrix
// 8x8x16 (DG2), 256 GRF.  Per expert i of the group: Y_i = X_i . W_i^T, rows back to back (cnt[i] each).
// gate/up: up_i != nullptr, virtual row 2r = gate row r, 2r+1 = up row r (iq_dequant_gu_f16's interleave).
namespace xmxg {
// Sub-group width = the XMX tile N: 8 (DG2, 8x8x16) or 16 (Xe2, 8x16x16; CMake STRATA_XMX_SG16 for an AOT bmg / lnl build)
#ifdef STRATA_XMX_SG16
constexpr int kSgW = 16;
#else
constexpr int kSgW = 8;
#endif
namespace jm = sycl::ext::oneapi::experimental::matrix;
struct Grp {
    const uint8_t* a[kXmxGroupMax];
    const uint8_t* b[kXmxGroupMax];
    int off[kXmxGroupMax + 1];
    int tstart[kXmxGroupMax + 1];
    int n;
};
template <int SGM_, int SGN_, int MT_, int NT_>
struct Cfg {
    static constexpr int SGM = SGM_, SGN = SGN_, MT = MT_, NT = NT_, KS = 32, SG = kSgW;
    static constexpr int BM = SGM * MT * 8, BN = SGN * NT * kSgW, WG = SGM * SGN * kSgW;
};
template <class C, class Name>
void launch(sycl::queue& q, int ty, bool gu, const sycl::half* X, float* Y, const Grp& gp, int ntiles, int N, int K) {
    constexpr int BM = C::BM, BN = C::BN, KS = C::KS, WG = C::WG, SG = C::SG;
    constexpr int LDA = KS + 8, LDB = KS + 8;
    const int nb = N / BN;
#ifndef STRATA_NO_DG2_XMX
    constexpr int TN = kSgW;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> sa(BM * LDA, h), sb(BN * LDB, h);
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        h.parallel_for<Name>(sycl::nd_range<2>({(size_t) ntiles, (size_t) nb * WG}, {1, (size_t) WG}), props,
                             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSgW)]] {
            auto sg = it.get_sub_group();
            const int t = (int) it.get_group(0);
            int e = 0;
            while (e + 1 < gp.n && t >= gp.tstart[e + 1]) ++e;
            const int m0 = (t - gp.tstart[e]) * BM;
            const int rows = sycl::min(BM, gp.off[e + 1] - gp.off[e] - m0);
            const sycl::half* Xe = X + ((size_t) gp.off[e] + m0) * K;
            float* Ye = Y + ((size_t) gp.off[e] + m0) * N;
            const uint8_t* wa = gp.a[e];
            const uint8_t* wb = gp.b[e];
            const int n0 = (int) it.get_group(1) * BN;
            const int lid = (int) it.get_local_id(1), sgid = lid / SG;
            const int sgm = sgid / C::SGN, sgn = sgid % C::SGN;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, TN> acc[C::MT][C::NT];
            for (int i = 0; i < C::MT; ++i)
                for (int j = 0; j < C::NT; ++j) jm::joint_matrix_fill(sg, acc[i][j], 0.0f);
            sycl::half* pa = sa.template get_multi_ptr<sycl::access::decorated::no>().get();
            sycl::half* pb = sb.template get_multi_ptr<sycl::access::decorated::no>().get();
            for (int k0 = 0; k0 < K; k0 += KS) {
                constexpr int VA = BM * KS / 8;
                for (int v = lid; v < VA; v += WG) {
                    const int r = v / (KS / 8), c = (v % (KS / 8)) * 8;
                    sycl::vec<sycl::half, 8> x(0);
                    if (r < rows) x = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(Xe + (size_t) r * K + k0 + c);
                    *reinterpret_cast<sycl::vec<sycl::half, 8>*>(pa + r * LDA + c) = x;
                }
                // B: row r of the tile, 32 values from k0 = 4 runs of 8, each one thread of the format's decoder
                for (int v = lid; v < BN * 4; v += WG) {
                    const int r = v >> 2, sub = v & 3, nr = n0 + r;
                    const uint8_t* base = gu ? ((nr & 1) ? wb : wa) : wa;
                    const int64_t row = gu ? (nr >> 1) : nr;
                    const int64_t flat = row * K + k0;            // a multiple of 32
                    const int64_t ibs = flat >> 8;
                    const int o32 = (int) (flat & 255), o = o32 + 8 * sub;
                    // the decoder thread that writes values [o, o + 8) of the superblock: the IQ2/IQ3/IQ4_NL/IQ1_M and
                    // Q2_0 decoders write [8 tid, 8 tid + 8) (coalesced runs); IQ4_XS and the rest keep the
                    // (sub-run, 32-block) split
                    // (the trellis decoder thread tid writes values [8 tid, 8 tid + 8) of the 256-value call as well)
                    const bool run8 = ty == 16 || ty == 17 || ty == 18 || ty == 20 || ty == 21 || ty == 22 ||
                                      ty == 29 || ty == 42 || ty == 144 || ty == 145 || ty == 146;
                    const int tid = run8 ? (o >> 3) : (sub * 8 + (o32 >> 5));
                    dq_dispatch<sycl::half>(ty, base, ibs, pb + r * LDB - o32, tid);
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int kk = 0; kk < KS; kk += 16) {
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> am[C::MT];
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, TN, jm::layout::col_major> bm[C::NT];
                    for (int i = 0; i < C::MT; ++i)
                        jm::joint_matrix_load(sg, am[i], sa.template get_multi_ptr<sycl::access::decorated::no>() +
                                                             (sgm * C::MT * 8 + i * 8) * LDA + kk, LDA);
                    for (int j = 0; j < C::NT; ++j)
                        jm::joint_matrix_load(sg, bm[j], sb.template get_multi_ptr<sycl::access::decorated::no>() +
                                                             (sgn * C::NT * TN + j * TN) * LDB + kk, LDB);
                    for (int i = 0; i < C::MT; ++i)
                        for (int j = 0; j < C::NT; ++j) jm::joint_matrix_mad(sg, acc[i][j], am[i], bm[j], acc[i][j]);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            for (int i = 0; i < C::MT; ++i) {
                const int r = sgm * C::MT * 8 + i * 8;
                if (r >= rows) continue;
                for (int j = 0; j < C::NT; ++j) {
                    const int c = n0 + sgn * C::NT * TN + j * TN;
                    if (r + 8 <= rows) {
                        jm::joint_matrix_store(sg, acc[i][j],
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                Ye + (size_t) r * N + c), N, jm::layout::row_major);
                    } else {
                        float* st = reinterpret_cast<float*>(pa) + sgid * 8 * TN;
                        jm::joint_matrix_store(sg, acc[i][j],
                            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(st),
                            TN, jm::layout::row_major);
                        sycl::group_barrier(sg);
                        const int lim = rows - r, ln = (int) sg.get_local_linear_id();
                        for (int rr = 0; rr < lim; ++rr) Ye[(size_t) (r + rr) * N + c + ln] = st[rr * TN + ln];
                        sycl::group_barrier(sg);
                    }
                }
            }
        });
    });
#else
    (void) q; (void) X; (void) Y; (void) gp; (void) ntiles; (void) N; (void) K;
    std::fprintf(stderr, "grouped XMX: the 8x8x16 / sub-group 8 kernel is not built for this device (STRATA_NO_DG2_XMX)\n");
    std::exit(1);
#endif
}
class k_xg_small; class k_xg_big;
#ifdef STRATA_XMX_SG16
using Small = Cfg<2, 4, 4, 2>;   // 64 x 128 (8 sub-groups of 16)
using Big = Cfg<4, 2, 4, 4>;     // 128 x 128
#else
using Small = Cfg<2, 8, 4, 2>;   // 64 x 128
using Big = Cfg<4, 4, 4, 4>;     // 128 x 128
#endif
template <class C>
int fill(Grp& gp, const int32_t* cnt, int n) {
    int t = 0, o = 0;
    for (int i = 0; i < n; ++i) { gp.off[i] = o; gp.tstart[i] = t; o += cnt[i]; t += (cnt[i] + C::BM - 1) / C::BM; }
    gp.off[n] = o; gp.tstart[n] = t; gp.n = n;
    return t;
}
}  // namespace xmxg

void iq_xmx_grouped(int ty, const uint8_t* const* a, const uint8_t* const* b, const int32_t* cnt, int n,
                    const uint16_t* X, float* Y, int N, int K, void* stream) {
    if (n <= 0) return;
    if (n > kXmxGroupMax || N % 128 != 0 || K % 32 != 0 || !is_iq(ty)) {
        std::fprintf(stderr, "iq_xmx_grouped: bad arguments (type %d, n %d, N %d, K %d)\n", ty, n, N, K);
        std::exit(1);
    }
    xmxg::Grp gp{};
    int64_t tot = 0;
    for (int i = 0; i < n; ++i) { gp.a[i] = a[i]; gp.b[i] = b ? b[i] : nullptr; tot += cnt[i]; }
    if (tot <= 0) return;
    sycl::queue& q = *strata::q_of(stream);
    const auto* Xh = reinterpret_cast<const sycl::half*>(X);
    const bool gu = b != nullptr;
    if (tot / n < 96) {
        const int nt = xmxg::fill<xmxg::Small>(gp, cnt, n);
        xmxg::launch<xmxg::Small, xmxg::k_xg_small>(q, ty, gu, Xh, Y, gp, nt, N, K);
    } else {
        const int nt = xmxg::fill<xmxg::Big>(gp, cnt, n);
        xmxg::launch<xmxg::Big, xmxg::k_xg_big>(q, ty, gu, Xh, Y, gp, nt, N, K);
    }
    check("iq_xmx_grouped");
}


// ================================================================ SYCL port: XMX GEMM straight from the quantized rows
//
// The prompt path dequantizes every routed expert to FP16 (8 MB written, then read back by oneMKL) before its GEMMs:
// 39-41% of the prompt time on the B70 (2026-09-30), the GEMMs themselves another 16%. This kernel dequantizes a
// 16-feature strip of the weight matrix into local memory, packs it for the XMX B operand and multiplies it against
// every activation row with joint_matrix (16x16x16, FP16 in, FP32 out; sub-group 16 as XMX wants). The FP16 weights
// never touch global memory.
//
//   Y[M][n_out] = X[M][K] . W[n_out][K]^T      X FP16 row-major (lda K), Y FP32 (ldy), W quantized rows (a ggml IQ type)
//
// Gate/up: `up != nullptr`, virtual row 2r = gate row r, 2r+1 = up row r (the order iq_dequant_gu_f16 produces, which
// swiglu_interleaved expects). K must be a multiple of 256 (a QK_K block per slab) or the whole row must fit a slab
// (K = 640: 16 rows are 40 blocks, block-aligned since 16 * 640 / 256 = 40).
namespace xmx {
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int SG = 16;                 // XMX sub-group size
constexpr int NSG = 8;                 // sub-groups per work-group
constexpr int WG = SG * NSG;
constexpr int NT = 16;                 // features per work-group: one B tile
constexpr int MAXT = 4;                // C tiles per sub-group -> 512 rows per chunk
constexpr int APAD = 8;                // halves of padding on the staged A tile rows
struct Src { const uint8_t* gate; const uint8_t* up; int ty; int64_t per_row; };
struct Smem { sycl::half* stage; sycl::half* packed; sycl::half* atile; float* ctile; };

// the block index (x[ibs] in dq_dispatch) of virtual row v's k-range [k0, k0 + 256) - rows are per_row blocks long
__dpct_inline__ const void* block_of(const Src& s, int v, int cblk, int64_t& ibs) {
    if (s.up != nullptr) { ibs = (int64_t) (v >> 1) * s.per_row + cblk; return (v & 1) ? s.up : s.gate; }
    ibs = (int64_t) v * s.per_row + cblk;
    return s.gate;
}

// one work-group: features [f0, f0 + 16) of the weight matrix against all M rows; slab_k halves of K per pass
void gemm_kernel(Src src, const sycl::half* X, int K, int slab_k, float* Y, int64_t ldy, int M, int n_out, Smem sm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = it.get_sub_group();
    const int sgid = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0], t = (int) it.get_local_id(2);
    const int f0 = (int) it.get_group(2) * NT;
    const int per_slab = slab_k / 256;                    // blocks of one row per slab (K % 256 == 0), or
    const int whole = (slab_k == K) ? 1 : 0;              // the whole 16-row strip as NT * K / 256 flat blocks
    const int nblk = whole ? NT * K / 256 : NT * per_slab;
    auto gptr = [](auto* p) { return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p); };
    auto lptr = [](auto* p) { return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p); };
    sycl::half* atile = sm.atile + (size_t) sgid * 16 * (16 + APAD);
    float* ctile = sm.ctile + (size_t) sgid * 256;
    for (int m_base = 0; m_base < M; m_base += MAXT * NSG * 16) {
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> C[MAXT];
#pragma unroll
        for (int k = 0; k < MAXT; ++k) jm::joint_matrix_fill(sg, C[k], 0.0f);
        const int tiles = std::min((M - m_base + 15) / 16, MAXT * NSG);
        for (int k0 = 0; k0 < K; k0 += slab_k) {
            it.barrier(sycl::access::fence_space::local_space);
            // dequantize the strip's blocks of this slab into stage[NT][slab_k] (flat, row-major): 32 "threads" per
            // block in dq_dispatch's terms, two calls per lane
            for (int b = sgid; b < nblk; b += NSG) {
                int v, kk;
                if (whole) { v = (b * 256) / K; kk = (b * 256) % K; }
                else { v = b / per_slab; kk = (b % per_slab) * 256; }
                sycl::half* dst = sm.stage + (size_t) v * slab_k + kk;
                const int row = f0 + v;
                if (row < n_out) {
                    int64_t ibs;
                    const void* blk = src.gate;
                    if (whole) ibs = ((int64_t) f0 * K) / 256 + b;   // flat blocks of the [n_out][K] matrix
                    else blk = block_of(src, row, k0 / 256 + b % per_slab, ibs);
                    dq_dispatch<sycl::half>(src.ty, blk, ibs, dst, lane);
                    dq_dispatch<sycl::half>(src.ty, blk, ibs, dst, lane + 16);
                } else {
                    for (int i = lane; i < 256; i += SG) dst[i] = sycl::half(0.0f);
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            // pack for the XMX B operand: element (k, n) at packed[(k / 2) * (NT * 2) + n * 2 + (k & 1)]
            for (int i = t; i < NT * slab_k; i += WG) {
                const int n = i / slab_k, k = i - n * slab_k;
                sm.packed[(k >> 1) * (NT * 2) + n * 2 + (k & 1)] = sm.stage[i];
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int ks = 0; ks < slab_k; ks += 16) {
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> B;
                jm::joint_matrix_load(sg, B, lptr(sm.packed + (ks >> 1) * (NT * 2)), NT * 2);
#pragma unroll
                for (int tt = 0; tt < MAXT; ++tt) {
                    const int tile = sgid + tt * NSG;
                    if (tile >= tiles) break;
                    const int m0 = m_base + tile * 16;
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> A;
                    if (m0 + 16 <= M) {
                        jm::joint_matrix_load(sg, A, gptr(X + (size_t) m0 * K + k0 + ks), (size_t) K);
                    } else {   // the tail tile: valid rows staged, the rest zero
                        for (int i = lane; i < 16 * 16; i += SG) {
                            const int r = i >> 4, c = i & 15;
                            atile[r * (16 + APAD) + c] = (m0 + r < M) ? X[(size_t) (m0 + r) * K + k0 + ks + c] : sycl::half(0.0f);
                        }
                        sycl::group_barrier(sg);
                        jm::joint_matrix_load(sg, A, lptr(atile), (size_t) (16 + APAD));
                        sycl::group_barrier(sg);
                    }
                    jm::joint_matrix_mad(sg, C[tt], A, B, C[tt]);
                }
            }
        }
        // the chunk's results
#pragma unroll
        for (int tt = 0; tt < MAXT; ++tt) {
            const int tile = sgid + tt * NSG;
            if (tile >= tiles) break;
            const int m0 = m_base + tile * 16;
            if (m0 + 16 <= M && f0 + NT <= n_out) {
                jm::joint_matrix_store(sg, C[tt], gptr(Y + (size_t) m0 * ldy + f0), (size_t) ldy, jm::layout::row_major);
            } else {
                jm::joint_matrix_store(sg, C[tt], lptr(ctile), (size_t) 16, jm::layout::row_major);
                sycl::group_barrier(sg);
                for (int i = lane; i < 256; i += SG) {
                    const int r = i >> 4, c = i & 15;
                    if (m0 + r < M && f0 + c < n_out) Y[(size_t) (m0 + r) * ldy + f0 + c] = ctile[i];
                }
                sycl::group_barrier(sg);
            }
        }
    }
}
}  // namespace xmx

bool xmx_gemm_iq(int ty, const void* gate, const void* up, int64_t K, int n_out, const uint16_t* X, int M, float* Y,
                 int64_t ldy, void* stream) {
    if (!is_iq(ty) || ty == 16 || ty == 17 || ty == 29 || ty == 23 || ty == 11) return false;   // dq_dispatch covers these; the rest untested here
    if (M <= 0 || n_out <= 0 || n_out % xmx::NT != 0 || K <= 0) return false;
    const bool by_block = (K % 256) == 0;
    const bool whole = !by_block && ((int64_t) xmx::NT * K) % 256 == 0 && K % 16 == 0;
    if (!by_block && !whole) return false;
    const int slab_k = by_block ? 256 : (int) K;
    if (up != nullptr && !by_block) return false;   // gate/up rows are per_row blocks each
    const size_t stage_halves = (size_t) xmx::NT * slab_k;
    const size_t bytes = stage_halves * 2 /*stage*/ + stage_halves * 2 /*packed*/ + (size_t) xmx::NSG * 16 * (16 + xmx::APAD) * 2 +
                         (size_t) xmx::NSG * 256 * 4;
    if (bytes > 96 * 1024) return false;
    dpct::queue_ptr q = strata::q_of(stream);
    xmx::Src src{(const uint8_t*) gate, (const uint8_t*) up, ty, by_block ? K / 256 : 0};
    const int groups = n_out / xmx::NT;
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(bytes), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) groups * xmx::WG), sycl::range<3>(1, 1, xmx::WG)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                uint8_t* base = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                xmx::Smem sm;
                sm.stage = (sycl::half*) base;
                sm.packed = sm.stage + stage_halves;
                sm.atile = sm.packed + stage_halves;
                sm.ctile = (float*) (sm.atile + (size_t) xmx::NSG * 16 * (16 + xmx::APAD));
                xmx::gemm_kernel(src, (const sycl::half*) X, (int) K, slab_k, Y, ldy, M, n_out, sm);
            });
    });
    return true;
}
}  // namespace strata::kernels
