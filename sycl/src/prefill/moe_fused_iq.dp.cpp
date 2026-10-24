// src/prefill/moe_fused_iq.cu - see include/strata/prefill/moe_fused_iq.hpp (#136: the native packs' prompt experts on
// the fused int8 kernels of moe_fused.cu).
//
// The arithmetic.  An activation block of 32 values is x = d_x * a, a = round(x / d_x) in -127..127, stored in natural
// order (per 64 values: 64 codes, then {d_0, d_1} and 8 unused bytes - 80 bytes, the Q2_0 path's size).  A weight
// block of 32 (or 16) values is w = d_w * q with q an int8: the i-quants' codebook entries with their signs applied
// (IQ2_XXS / IQ2_XS / IQ2_S: 0..43, IQ3_XXS: 0..62, IQ3_S: 0..15, IQ4_XS / IQ4_NL: -127..113), Q2_0's code - 1.  So
//     sum w x = d_w * d_x * sum q a,     |sum q a| <= 32 * 127 * 127 < 2^22,
// and the int32 dot becomes a float exactly with one float add: the mma starts from C = 0x4B400000 (the bits of
// 1.5 * 2^23), so as_float(D) - 1.5 * 2^23 is the dot.  Per output and 32 values: that add, d_w * d_x, one fma (the
// formats with a scale per 16 values: two dots, each times its d_w, then d_x).  d_w is llama.cpp's own: IQ2_XXS
// d (2s + 1) / 8, IQ2_XS / IQ2_S d (2s + 1) / 8 per 16, IQ3_XXS d (2s + 1) / 4, IQ3_S d (2s + 1), IQ4_XS d (s - 32),
// IQ4_NL and Q2_0 d (dequantize_row_* in ggml-quants.c, mmq-load-tiles.cuh).
//
// The load stage.  Per 64 values of K (a stage), a thread decodes one 32-value sub-block of one of the work item's
// weight rows: its block bytes are read from the blob with 16-bit loads (the GGUF blocks are 2-byte aligned: 66 / 74 /
// 82 / 98 / 110 / 136 / 18 bytes) one stage ahead into registers (and two 256-value blocks ahead into L2), then turned
// into 32 int8 and the sub-block's scales in shared memory, double-buffered.  The codebooks live in shared memory (up
// to 8 KB, IQ2_S).  The activations arrive by cp.async, four stages deep, as in moe_fused.cu; the fragments come by
// ldmatrix.  The products are mma.sync m16n8k32 (m16n8k16 for the formats with a scale per 16 values); the epilogues
// (SwiGLU, H to int8 per 32 features; down into the per-slot rows) are moe_fused.cu's.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::prefill::fused {
namespace {

void ck(dpct::err0 e, const char *what) {
}

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1}, 8 unused
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// 16 warps, each 64 weight rows x 16 routed rows; WW of them along the weight rows and 16 / WW along the routed rows: a
// work item is (WR = 64 WW weight rows) x (TR = 256 / WW routed rows of one expert) - 256 x 64 or 128 x 128, chosen
// per launch by the rows per expert (pick_ww)
constexpr int THREADS = 512;
constexpr int WLD = 80;                 // bytes per decoded weight row of a stage: 64 int8 + 16 (no bank conflicts)
constexpr int tile_rows(int ww) { return 256 / ww; }
constexpr int weight_rows(int ww) { return 64 * ww; }
constexpr int ASTAGES = 4;              // cp.async depth of the activations (64 values of K a stage)
constexpr int GU_ROWS_K = 2560, D_ROWS_K = 640;   // K of gate/up (n_embd) and of down (n_ff)

// the formats (ggml type ids)
constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_XS = GGML_TYPE_IQ4_XS,
              T_IQ4_NL = GGML_TYPE_IQ4_NL, T_Q2_0 = GGML_TYPE_Q2_0,
              // Aurora (opt-in, HIP gfx11 only): Unsloth UD-Q4_K_XL's experts - gate/up Q4_K (one layer Q5_K), down Q5_1
              // (five layers Q8_0).  Their weights are w = d_w q + m_w (a minimum), so a sub-block's dot is
              //   sum w x = d_x (d_w sum q a + m_w sum a):
              // the activation blocks carry sum a per 32 values in their eight spare bytes (quant_act_nat_kernel and
              // the H epilogue write it for every path; no other format reads it)
              T_Q4_K = GGML_TYPE_Q4_K, T_Q5_K = GGML_TYPE_Q5_K, T_Q5_1 = GGML_TYPE_Q5_1, T_Q8_0 = GGML_TYPE_Q8_0;
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 &&
              sizeof(block_iq4_nl) == 18 && sizeof(block_q2_0) == 18 && sizeof(block_q4_K) == 144 &&
              sizeof(block_q5_K) == 176 && sizeof(block_q5_1) == 24 && sizeof(block_q8_0) == 34,
              "the block layouts this file decodes");

// block bytes, scale per 16 values, codebook bytes in shared memory
constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110
         : t == T_IQ4_XS ? 136 : t == T_Q4_K ? 144 : t == T_Q5_K ? 176 : t == T_Q5_1 ? 24 : t == T_Q8_0 ? 34 : 18;
}
// the formats with a minimum (w = d q + m), and the raw words a sub-block's load stage holds
constexpr bool has_min(int t) { return t == T_Q4_K || t == T_Q5_K || t == T_Q5_1; }
constexpr int raw_words(int t) {
    return t == T_Q4_K ? 10 : t == T_Q5_K ? 18 : t == T_Q5_1 ? 6 : t == T_Q8_0 ? 9 : 5;
}
constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4
         : t == T_IQ3_S ? 512 * 4 : 0;
}
constexpr size_t smem_bytes(int t, int ww) {
    return (size_t) 2 * weight_rows(ww) * (WLD + 16) + (size_t) ASTAGES * tile_rows(ww) * AB +
           (size_t) tile_rows(ww) * 4 + grid_bytes(t);
}

// moe_fused.cu's grouping tables (the same layout: fused::group writes them)
struct Tables {
    int32_t *cnt, *off, *fill, *ts;
    sycl::int2 *tiles;
};
size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (sycl::int2 *)((uint8_t *)scratch + tiles_at(n_expert))};
}

// ---- activations in natural order: one warp per 64 values
__dpct_inline__ void quant_act_nat_kernel(const float *__restrict__ x,
                                          int64_t nblk,
                                          uint8_t *__restrict__ xa) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t w =
        (int64_t)item_ct1.get_group(2) * (item_ct1.get_local_range(2) / 32) +
        item_ct1.get_local_id(2) / 32;
    const int lane = item_ct1.get_local_id(2) & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = sycl::fabs(v0), a1 = sycl::fabs(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        a0 = sycl::fmax(
            a0, dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), a0, o));
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        a1 = sycl::fmax(
            a1, dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), a1, o));
    }
    uint8_t* out = xa + w * AB;
    const int c0 = a0 > 0.0f ? sycl::vec<float, 1>{(v0 * (127.0f / a0))}
                                   .convert<int, sycl::rounding_mode::rte>()[0]
                             : 0,
              c1 = a1 > 0.0f ? sycl::vec<float, 1>{(v1 * (127.0f / a1))}
                                   .convert<int, sycl::rounding_mode::rte>()[0]
                             : 0;
    out[lane] = (uint8_t) (int8_t) c0;
    out[32 + lane] = (uint8_t) (int8_t) c1;
    int s0 = c0, s1 = c1;   // the codes' sums per 32 values (the formats with a minimum read them)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        s0 += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), s0,
            o);
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        s1 += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), s1,
            o);
    }
    if (lane == 0) * (sycl::float4 *)(out + 64) =
        sycl::float4(a0 / 127.0f, a1 / 127.0f, (float)s0, (float)s1);
}

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800
__dpct_inline__ void cp16(void *dst, const void *src) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     (unsigned)__cvta_generic_to_shared(dst)),
                 "l"(src));
#else
    *(((uint32_t *)(uintptr_t)(unsigned)__cvta_generic_to_shared(dst))) =
        *(((uint32_t *)(uintptr_t)src));
    if (16 > 4)
        *(((uint32_t *)(uintptr_t)(unsigned)__cvta_generic_to_shared(dst)) +
          1) = *(((uint32_t *)(uintptr_t)src) + 1);
    if (16 > 8)
        *(((uint32_t *)(uintptr_t)(unsigned)__cvta_generic_to_shared(dst)) +
          2) = *(((uint32_t *)(uintptr_t)src) + 2);
    if (16 > 12)
        *(((uint32_t *)(uintptr_t)(unsigned)__cvta_generic_to_shared(dst)) +
          3) = *(((uint32_t *)(uintptr_t)src) + 3);
#endif
}
/*
DPCT1026: The call to "cp.async.commit_group;
" was removed because current "cp.async" is migrated to synchronous copy
operation. You may need to adjust the code to tune the performance.
*/
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
#else
__device__ __forceinline__ void cp_commit() {
#endif }
/*
DPCT1053: Migration of device assembly code is not supported.
*/
__dpct_inline__ void pf_l2(const void *p) {
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}
/*
DPCT1026: The call to "cp.async.wait_group %0;
" was removed because current "cp.async" is migrated to synchronous copy
operation. You may need to adjust the code to tune the performance.
*/
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    template <int N> __device__ __forceinline__ void cp_wait() {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#else
    template <int N> __device__ __forceinline__ void cp_wait() {
#endif }
// d = MAGIC + A (16 x 32 s8, row) * B (32 x 8 s8, col): the int32 dot with the magic bias already added (the C
// operand), so as_float(d) - 1.5 * 2^23 is the dot as a float
__dpct_inline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                           uint32_t b1) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, "
                "{%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                  "r"(MAGIC));
#else
            {
                volatile void *d_mat_frag_ct1[4] = {&d[0], &d[1], &d[2], &d[3]};
                sycl::vec<uint32_t, 4> a_mat_frag_ct1(a[0], a[1], a[2], a[3]);
                sycl::vec<uint32_t, 2> b_mat_frag_ct1(b0, b1);
                sycl::vec<int32_t, 4> c_mat_frag_ct1(MAGIC, MAGIC, MAGIC,
                                                     MAGIC);
                dpct::experimental::matrix::mma<16, 8, 32, int8_t, int32_t>(
                    reinterpret_cast<volatile void **>(d_mat_frag_ct1),
                    &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
            }
#endif
}
// the same at m16n8k16 (A 16 x 16, B 16 x 8)
__dpct_inline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, "
                "{%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
#else
            {
                volatile void *d_mat_frag_ct1[4] = {&d[0], &d[1], &d[2], &d[3]};
                sycl::vec<uint32_t, 2> a_mat_frag_ct1(a0, a1);
                sycl::vec<uint32_t, 1> b_mat_frag_ct1(b0);
                sycl::vec<int32_t, 4> c_mat_frag_ct1(MAGIC, MAGIC, MAGIC,
                                                     MAGIC);
                dpct::experimental::matrix::mma<16, 8, 16, int8_t, int32_t>(
                    reinterpret_cast<volatile void **>(d_mat_frag_ct1),
                    &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
            }
#endif
}
__dpct_inline__ float dotf(int d) {
    return sycl::bit_cast<float>(d) - MAGICF;
}
// four 8 x 16-byte matrices, row addresses from lanes 0-7, 8-15, 16-23, 24-31: lane (g, t) gets bytes 4t..4t+3 of row g
// of each - an mma fragment
__dpct_inline__ void ldsm4(uint32_t (&r)[4], const void *p) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0,%1,%2,%3}, [%4];\n"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                         : "r"((unsigned)__cvta_generic_to_shared(p))
                         : "memory");
#else
            dpct::experimental::matrix::ldmatrix(
                (uintptr_t)(unsigned)__cvta_generic_to_shared(p), &r[0], &r[1],
                &r[2], &r[3]);
#endif
}

#endif
#if (defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800) ||    \
    defined(__HIPCC__)
#if defined(__HIPCC__)
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
#endif
// ---- the load stage: a 32-value sub-block's bytes into registers, then int8 and scales
__dpct_inline__ uint32_t ld16(const uint8_t *p) {
    return *(const uint16_t *)p;
}
__dpct_inline__ uint32_t ld32(const uint8_t *p) {
    return ld16(p) | (ld16(p + 2) << 16);
}
__dpct_inline__ float half_at(uint32_t w) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>((unsigned short)w))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity (bit 7 of v may be anything)
__dpct_inline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = sycl::popcount(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte `s` (broadcast) applied: bits 0-3 to .x, 4-7 to .y
__dpct_inline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t &qx,
                             uint32_t &qy) {
    const uint32_t m0 = dpct::vectorized_binary<sycl::uchar4>(
                       s & 0x08040201u, 0, std::not_equal_to<>()),
                   m1 = dpct::vectorized_binary<sycl::uchar4>(
                       s & 0x80402010u, 0, std::not_equal_to<>());
    qx = dpct::vectorized_binary<sycl::uchar4>(gx ^ m0, m0, std::minus<>());
    qy = dpct::vectorized_binary<sycl::uchar4>(gy ^ m1, m1, std::minus<>());
}
#if defined(__HIPCC__)
// gfx11 (Aurora): the signs from a shared-memory table instead of the compare / subtract emulation.  An entry (one per
// sign byte b; for the 7-bit sign formats per 7 bits, the parity bit folded in) is {m0, c0, m1, c1}: m = 0xFF in the
// bytes whose sign bit is set (bits 0-3 for .x, 4-7 for .y), c = m & 0x01010101.  q = (g ^ m) + c is -g where the sign
// is set - the two's complement, byte-wise - and g elsewhere; no byte carries into the next, because no codebook entry
// has a zero byte (256 - g with g in 1..127; the grids' bytes are 1..62), so it is the bytes signed8 gives.
__device__ __forceinline__ void signed8t(uint32_t gx, uint32_t gy, const uint4 sg, uint32_t& qx, uint32_t& qy) {
    qx = (gx ^ sg.x) + sg.y;
    qy = (gy ^ sg.z) + sg.w;
}
__host__ __device__ constexpr int sign_entries(int t) { return t == T_IQ2_S || t == T_IQ3_S ? 256 : 128; }
__device__ __forceinline__ uint4 sign_entry(int i, bool parity) {
    uint32_t b = (uint32_t) i;
    if (parity) b ^= (__popc(b) & 1) << 7;
    const uint32_t m0 = __vcmpne4(((b & 15) * 0x01010101u) & 0x08040201u, 0),
                   m1 = __vcmpne4(((b >> 4) * 0x01010101u) & 0x08040201u, 0);
    return make_uint4(m0, m0 & 0x01010101u, m1, m1 & 0x01010101u);
}
#endif
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in .x, the high ones' in .y
// W (X5): direct-selector form on gfx: p = low 3 bits of each nibble picks within the table half, bit 3 picks the half;
// exhaustively bit-exact over all 2^32 inputs (X5). Disable with -DSTRATA_W_NO_T16.
#if defined(__HIPCC__) && !defined(STRATA_W_NO_T16)
__device__ __forceinline__ uint32_t table16_one(uint32_t x, const uint32_t (&t)[4]) {
    const uint32_t p = x & 0x07070707u;
    const uint32_t a = __builtin_amdgcn_perm(t[1], t[0], p), b = __builtin_amdgcn_perm(t[3], t[2], p);
    return __builtin_amdgcn_perm(b, a, ((x >> 1) & 0x04040404u) | 0x03020100u);
}
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    lo = table16_one(q4, t);
    hi = table16_one(q4 >> 4, t);
}
#else
__dpct_inline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t &lo,
                             uint32_t &hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = dpct::byte_level_permute(t[0], t[1], q4 >> sh),
                       h = dpct::byte_level_permute(t[2], t[3], q4 >> sh);
        tmp[i] = dpct::byte_level_permute(l, h, sel >> sh);
    }
    lo = dpct::byte_level_permute(tmp[0], tmp[1], 0x6420);
    hi = dpct::byte_level_permute(tmp[0], tmp[1], 0x7531);
}
#endif

// Raw bytes of sub-block `ib` of the block at `bp` (IQ: the 256-value super-block, ib 0..7; Q2_0: the 64-value block,
// ib 0..1; IQ4_NL: the 32-value block).
// 32 bits at a 2-byte aligned address (one load when it is 4-byte aligned)
__dpct_inline__ uint32_t ld32a(const uint8_t *p) {
    return ((uintptr_t) p & 3) == 0 ? *(const uint32_t*) p : ld32(p);
}
template <int T, int NW>
__dpct_inline__ void load_unit(const uint8_t *bp, int ib, uint32_t (&w)[NW]) {
    static_assert(NW >= raw_words(T), "raw words");
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 8 + 16 * ib + 4 * k);
        const uint32_t ls = ((bp[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((ld16(bp + 2) >> (2 * ib)) & 3) << 4);
        w[4] = ld16(bp) | (ls << 16);
    } else if constexpr (T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    } else if constexpr (T == T_Q4_K || T == T_Q5_K) {
        // qs: 32 bytes of the sub-block pair (ib / 2), the nibble chosen in convert; w[8] = d | dmin << 16; w[9]: the
        // scale/min bytes (get_scale_min_k4's: ib < 4: sc[ib], sc[ib + 4]; else sc[ib + 4], sc[ib - 4], sc[ib]), ib << 24
        constexpr int QS = T == T_Q4_K ? 16 : 48;
        const uint8_t* q = bp + QS + 32 * (ib >> 1);
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = ld32a(q + 4 * k);
        w[8] = ld32a(bp);
        const uint8_t* sc = bp + 4;
        w[9] = ib < 4 ? (uint32_t) sc[ib] | ((uint32_t) sc[ib + 4] << 8)
                      : (uint32_t) sc[ib + 4] | ((uint32_t) sc[ib - 4] << 8) | ((uint32_t) sc[ib] << 16);
        w[9] |= (uint32_t) ib << 24;
        if constexpr (T == T_Q5_K) {
#pragma unroll
            for (int k = 0; k < 8; ++k) w[10 + k] = ld32a(bp + 16 + 4 * k);   // qh
        }
    } else if constexpr (T == T_Q5_1) {   // d, m, qh, qs[16]
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32a(bp + 8 + 4 * k);
        w[4] = ld32a(bp + 4);
        w[5] = ld32a(bp);
    } else if constexpr (T == T_Q8_0) {   // d, qs[32]
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[8] = ld16(bp);
    } else {   // Q2_0
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
}

// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T, int NW>
/*
DPCT1110: The total declared local variable size in device function convert
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void convert(
    const uint32_t (&w)[NW], const uint8_t *grid, const uint32_t (&kv)[4],
    uint32_t (&q)[8], float &s0, float &s1, const sycl::uint4 *sgn = nullptr) {
    if constexpr (T == T_Q4_K || T == T_Q5_K) {
        const int ib = (int) (w[9] >> 24), sh = 4 * (ib & 1);
        uint32_t sc, mn;
        if (ib < 4) { sc = w[9] & 63; mn = (w[9] >> 8) & 63; }
        else {
            sc = ((w[9] & 0xF) | (((w[9] >> 8) & 0xC0) >> 2));   // sc[ib + 4] & 15 | (sc[ib - 4] >> 6) << 4
            mn = (((w[9]) >> 4) & 0xF) | (((w[9] >> 16) & 0xC0) >> 2);   // sc[ib + 4] >> 4 | (sc[ib] >> 6) << 4
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q[k] = (w[k] >> sh) & 0x0F0F0F0Fu;
            if constexpr (T == T_Q5_K) q[k] |= ((w[10 + k] >> ib) & 0x01010101u) << 4;
        }
        s0 = half_at(w[8]) * (float) sc;
        s1 = -(half_at(w[8] >> 16) * (float) mn);
    } else if constexpr (T == T_Q5_1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t lo = ((w[4] >> (4 * k)) & 0xF) * 0x00204081u, hi = ((w[4] >> (16 + 4 * k)) & 0xF) * 0x00204081u;
            q[k] = (w[k] & 0x0F0F0F0Fu) | ((lo & 0x01010101u) << 4);
            q[4 + k] = ((w[k] >> 4) & 0x0F0F0F0Fu) | ((hi & 0x01010101u) << 4);
        }
        s0 = half_at(w[5]);
        s1 = half_at(w[5] >> 16);
    } else if constexpr (T == T_Q8_0) {
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = w[k];
        s0 = s1 = half_at(w[8]);
    } else if constexpr (T == T_IQ2_XXS) {
        const sycl::uint2 *g = (const sycl::uint2 *)grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const sycl::uint2 e = g[(w[0] >> (8 * l)) & 255];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x(), e.y(), unpack_ksigns(w[1] >> (7 * l)), q[2 * l],
                    q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const sycl::uint2 *g = (const sycl::uint2 *)grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const sycl::uint2 e = g[c & 511];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(c >> 9) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x(), e.y(), unpack_ksigns(c >> 9), q[2 * l],
                    q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const sycl::uint2 *g = (const sycl::uint2 *)grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const sycl::uint2 e =
                g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x(), e.y(), ((w[1] >> (8 * l)) & 255) * 0x01010101u,
                    q[2 * l], q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0], g[i1], sgn[(w[2] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                     sgn[(w[2] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                    ((w[2] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else if constexpr (T == T_IQ4_XS || T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = T == T_IQ4_XS ? half_at(w[4]) * (float) ((int) (w[4] >> 16) - 32) : half_at(w[4]);
    } else {   // Q2_0: code - 1 via a byte table {-1, 0, 1, 2}
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe =
                dpct::byte_level_permute(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = dpct::byte_level_permute(
                0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = dpct::byte_level_permute(qe, qo, 0x5140);
            q[2 * h + 1] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        s0 = s1 = half_at(w[2]);
    }
}

template <int T>
__dpct_inline__ const void *grid_src(
    const uint64_t *iq2xxs_grid, const uint64_t *iq2xs_grid,
    const uint64_t *iq2s_grid, const uint32_t *iq3xxs_grid,
    const uint32_t *iq3s_grid) {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else return nullptr;
}
#endif

// One work item = (TR routed rows of expert e - one or two of group()'s 64-row tiles -, a block of WR weight rows);
// a persistent grid walks the batch's items.  GU: gate/up rows of WR / 2 features (format WT) against the rows'
// tokens' activations, SwiGLU, H to int8 per 32 features into `out`.  !GU: down rows (format WT) against H, FP32 into
// `dm`.  Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, gate rows as mma rows 0-7 and
// up rows as 8-15, so a lane holds gate and up of one feature), routed rows 16 wt.. as two n8.
template <int WT, bool GU, int WW>
/*
DPCT1110: The total declared local variable size in device function
native_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void native_kernel(
    const Batch b, const NativeGeom geo, const Tables tb,
    const uint8_t *__restrict__ act, const int32_t *__restrict__ src,
    uint8_t *__restrict__ out, float *__restrict__ dm, uint8_t *dpct_local,
    uint64_t *iq2xxs_grid, uint64_t *iq2xs_grid, uint64_t *iq2s_grid,
    uint32_t *iq3xxs_grid, uint32_t *iq3s_grid, const int8_t *kvalues_iq4nl) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int TR = tile_rows(WW), WR = weight_rows(WW);
    constexpr int WT_BYTES = WR * WLD, WS_FLOATS = WR * 4, ACT_STAGE = TR * AB;
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;   // 64-value stages along K
    constexpr int NFB = (GU ? 1280 : 2560) / WR;          // weight-row blocks per tile
    constexpr int ACT_LD = NS * AB;                       // bytes per activation row: 3200 (a token), 800 (a row's H)
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    auto smem = (uint8_t *)dpct_local;
    uint8_t* wt = smem;                                               // [2][WR][WLD] decoded int8 weights
    float* ws = (float*) (smem + 2 * WT_BYTES);                       // [2][WR][4] their scales per 16 values
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);                // [ASTAGES][TR][AB] activations
    int* srow = (int*) (stages + ASTAGES * ACT_STAGE);                // [TR] the activation row of each tile row
    uint8_t* sgrid = (uint8_t*) (srow + TR);                          // the codebook

    const int tid = item_ct1.get_local_id(2), lane = tid & 31, warp = tid >> 5,
              g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t *gs = (const uint32_t *)grid_src<WT>(
            iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid, iq3s_grid);
#pragma unroll
        for (int i = tid; i < grid_bytes(WT) / 4;
             i += THREADS)((uint32_t *)sgrid)[i] = gs[i];
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    // Local weight rows: m tile i of warp wf is rows 64 wf + 16 i.. (mma rows g and g + 8: rows +g and +8+g).  For
    // gate/up a tile is 8 features, its gate rows then its up rows, so a lane holds gate and up of one feature.  The
    // fragments come by ldmatrix: lane l gives the address of row (l & 7) of matrix l >> 3 - A: {rows 0-7, rows 8-15}
    // x {bytes 0-15, 16-31} of a 32-value chunk; B: {tokens nb0 + 0-7, + 8-15} x {bytes 0-15, 16-31}.
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLD + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    // this thread's decode unit (the first 2 WR threads): local weight row ur, sub-block uj of each stage
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = item_ct1.get_group(2); w < nwork;
         w += item_ct1.get_group_range(2)) {
        const sycl::int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x(), row0 = tl.y();
        if ((row0 - tb.off[e]) % TR != 0) continue;              // a 64-row tile inside an earlier item's TR rows
        const int nrows = sycl::min(TR, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WR;
        // local row ur: gate/up - feature (WR / 2) fb + 8 (ur >> 4) + (ur & 7), its up row when bit 3 is set
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* {                  // the block of stage s's sub-block
            if (GU) return wrow + (s >> 2) * BS;
            return WT == T_IQ4_NL ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return GU ? 2 * (s & 3) + uj : uj; };
        // the weight bytes into L2 ahead of the register loads (one stage ahead only - less than a DRAM round trip):
        // gate/up PF super-blocks ahead (the row's two threads take a block's first and last line), down the whole row
        // slice at once (180 or 360 bytes a row)
        constexpr int PF = 2;
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < GU_ROWS_K / 256) pf_l2(wrow + sb * BS + (uj ? BS - 1 : 0));
        };
        if (GU) {
#pragma unroll
            for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        } else if (dec) {
#pragma unroll
            for (size_t o = 128 * (size_t)uj; o < geo.d_row + 127; o += 256)
                pf_l2(wrow + dpct::min(o, geo.d_row - 1));
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
        item_ct1.barrier(); // the previous item is done with the buffers
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // srow
        // the rows past the item's end are not loaded: a warp entirely past it skips the products, and the columns
        // of a partial one are never written (no reduction mixes columns)
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % ASTAGES) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += THREADS) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[5], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1);
            sycl::uint4 *d =
                (sycl::uint4 *)(wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = sycl::uint4(q[0], q[1], q[2], q[3]);
            d[1] = sycl::uint4(q[4], q[5], q[6], q[7]);
            *(sycl::float2 *)(ws + buf * WS_FLOATS + ur * 4 + 2 * uj) =
                sycl::float2(s0, s1);
        };
#pragma unroll
        for (int s = 0; s < ASTAGES - 1; ++s) {
            if (s < NS) load_act(s);
            cp_commit();
        }
        uint32_t raw[5] = {0, 0, 0, 0, 0};
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<ASTAGES - 2>();
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
            if (s + ASTAGES - 1 < NS) load_act(s + ASTAGES - 1);
            cp_commit();
            if (GU && (s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % ASTAGES) * ACT_STAGE;
                // the stage's scales: per weight row 4 (per 16 values), per routed row 2 (per 32)
                sycl::float2 dx[2][2]; // [n8 tile][column 2 tig + cc]
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] =
                        *(const sycl::float2 *)(sa +
                                                (nb0 + 8 * n + 2 * tig + cc) *
                                                    AB +
                                                64);
#pragma unroll
                for (int h = 0; h < 2; ++h) {                     // the stage's two 32-value halves
                    uint32_t bq[4];                               // {n0 bytes 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];                            // {rows g, g + 8} x {bytes 0-15, 16-31}
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const sycl::float2 swa =
                            *(const sycl::float2 *)(S +
                                                    (64 * wf + 16 * i + g) * 4 +
                                                    2 * h);
                        const sycl::float2 swb =
                            *(const sycl::float2 *)(S +
                                                    (64 * wf + 16 * i + 8 + g) *
                                                        4 +
                                                    2 * h);
                        const float wa0 = swa.x(), wa1 = swa.y(), wb0 = swb.x(),
                                    wb1 = swb.y();
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {
                                // a scale per 16 weights: the two halves' dots, each times its scale, then d_x
                                int d0[4], d1[4];
                                mma16(d0, a[0], a[1], bq[2 * n]);
                                mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float v = sycl::fma(
                                        (float)(q < 2 ? wa1 : wb1), dotf(d1[q]),
                                        (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                    acc[i][n][q] = sycl::fma(
                                        h ? dx[n][q & 1].y() : dx[n][q & 1].x(),
                                        (float)v, acc[i][n][q]);
                                }
                            } else {
                                int d[4];
                                mma32(d, a, bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float p = (q < 2 ? wa0 : wb0) *
                                                    (h ? dx[n][q & 1].y()
                                                       : dx[n][q & 1].x());
                                    acc[i][n][q] = sycl::fma(
                                        (float)p, dotf(d[q]), acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
            }
            // the next stage's weights (into the other buffer, read after the next barrier).  (Decoding them before
            // this stage's products instead was slower on the RTX 5070: a layer of 2048 / 3584 / 8192 tokens, IQ2_S
            // 5.2 / 9.5 / 17.5 ms -> 6.0 / 10.2 / 18.3.)
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature (WR / 2) fb + 32 wf + 8 i + g (lane g of m tile i: rows 64 wf + 16 i + g and + 8), tile
            // rows nb0 + 8 n + 2 tig + c; H to int8 per row over the warp's 32 features (one 32-value half-block of the
            // down product's K)
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + sycl::native::exp(-gt)) * up;
                        am = sycl::fmax(am, sycl::fabs(h[i]));
                    }
#pragma unroll
                    /*
                    DPCT1108: '__shfl_xor_sync' was migrated with the
                    experimental feature masked sub_group function which may not
                    be supported by all compilers or runtimes. You may need to
                    adjust the code.
                    */
                    for (int o = 4; o < 32; o <<= 1) am = sycl::fmax(
                        am,
                        dpct::experimental::permute_sub_group_by_xor(
                            0xffffffffu,
                            sycl::ext::oneapi::this_work_item::get_sub_group(),
                            am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] =
                            (uint8_t)(int8_t)sycl::vec<float, 1>{(h[i] * inv)}
                                .convert<int, sycl::rounding_mode::rte>()[0];
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * 2560 + rbase + 64 * wf + 16 * i + 8 * (q >> 1) + g] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}

#if defined(__HIPCC__)
// ---- Aurora (S23): the native packs' fused experts on gfx11 (RDNA3 / RDNA3.5) matrix cores, v_wmma_i32_16x16x16_iu8
// (wave32).  The CUDA kernel's arithmetic: a 32-value sub-block decoded to int8 (load_unit / convert above) and its
// scales; per 32 values the int32 dot starts from MAGIC (the C operand), so as_float(d) - 1.5 * 2^23 is the dot; the
// formats with a scale per 16 values take each 16-value k-step's dot alone.  Fragments (gfx11): A lane l holds the 16
// k of row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l the 16 k of column l % 16, C/D lane l holds
// D[2i + l / 16][l % 16].  Both operands are in natural order: the decoded weights (LDS, double-buffered) and the
// activations (quant_act_nat_kernel; read by each lane from global/L2, a stage ahead).
// A work item = NW_ROWS weight rows x a 64-row tile; 8 waves: 4 along the weight rows (32 each) x 2 along the tile
// (32 each), each 2 x 2 WMMA tiles.  Gate/up: local row r is feature r / 2, its gate (r even) or up (r odd) row, so a
// lane pair l, l + 16 holds gate and up of one feature.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_NAT_W11 1
#else
#define STRATA_NAT_W11 0
#endif
typedef int nw_i4 __attribute__((ext_vector_type(4)));
typedef int nw_i8 __attribute__((ext_vector_type(8)));
constexpr int NW_ROWS = 128;
constexpr int NW_THREADS = 256;

__device__ __forceinline__ nw_i8 nw_wmma(nw_i4 a, nw_i4 b, nw_i8 c) {
#if STRATA_NAT_W11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}
__device__ __forceinline__ nw_i4 nw_u4(uint4 v) { return nw_i4{(int) v.x, (int) v.y, (int) v.z, (int) v.w}; }

// W (X1): ALIAS = the H tile lives on the weight buffers (one extra barrier); LB > 0 = amdgpu_waves_per_eu(LB) (VGPR cap).
// Disable both with -DSTRATA_W_NO_OCC (the launch sites then use <WT, GU, false, 0> and grid factor wgp_blocks).
template <int WT, bool GU, bool ALIAS = false, int LB = 0>
__global__ void __launch_bounds__(NW_THREADS) __attribute__((amdgpu_waves_per_eu(LB > 0 ? LB : 1)))
native_w11_kernel(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
                  const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if STRATA_NAT_W11
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;  // 64-value stages along K
    constexpr int NFB = (GU ? 1280 : 2560) / NW_ROWS;     // work items per tile
    constexpr int ACT_LD = NS * AB;
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    constexpr int GB = grid_bytes(WT) > 0 ? grid_bytes(WT) : 16;
    __shared__ __align__(16) uint8_t wtbuf[2 * NW_ROWS * WLD];
    uint8_t(*wt)[NW_ROWS][WLD] = reinterpret_cast<uint8_t(*)[NW_ROWS][WLD]>(wtbuf);
    __shared__ __align__(16) float ws[2][NW_ROWS][4];           // their scales per 16 values
    __shared__ __align__(16) uint8_t sgrid[GB];
    constexpr int SGN = (WT == T_IQ2_XXS || WT == T_IQ2_XS || WT == T_IQ2_S || WT == T_IQ3_XXS || WT == T_IQ3_S)
                            ? sign_entries(WT) : 1;
    __shared__ __align__(16) uint4 ssign[SGN];                   // the signs of a sign byte, as byte masks
    __shared__ int srow[kTileRows];
    __shared__ float hs_sep[(GU && !ALIAS) ? kTileRows : 1][65];
    static_assert(!ALIAS || (size_t) kTileRows * 65 * 4 <= (size_t) 2 * NW_ROWS * WLD, "H tile must fit the weight buffers");
    float(*hs)[65] = ALIAS ? reinterpret_cast<float(*)[65]>(wtbuf) : hs_sep;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += NW_THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    if constexpr (SGN > 1) {
        for (int i = tid; i < SGN; i += NW_THREADS) ssign[i] = sign_entry(i, SGN == 128);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    const int ur = tid >> 1, uj = tid & 1;                       // this thread's decode unit: row ur, sub-block uj
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y, nrows = min(kTileRows, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * NW_ROWS;
        const uint8_t* wrow = GU ? blob + ((ur & 1) ? geo.up_off : 0) + (size_t) (fb * (NW_ROWS / 2) + (ur >> 1)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* {
            if (GU) return wrow + (s >> 2) * BS;
            return (WT == T_IQ4_NL || WT == T_Q5_1 || WT == T_Q8_0) ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return GU ? 2 * (s & 3) + uj : uj; };
        auto put = [&](const uint32_t (&raw)[raw_words(WT)], int buf) {
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1, ssign);
            uint4* d = (uint4*) &wt[buf][ur][32 * uj];
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) &ws[buf][ur][2 * uj] = make_float2(s0, s1);
        };
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);             // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        uint32_t raw[raw_words(WT)] = {};
        load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (NS > 1) load_unit<WT>(unit(1), sub(1), raw);
        __syncthreads();                                          // srow
        const bool on = 32 * wn < nrows;
        const uint8_t* brow[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) brow[nt] = act + (size_t) srow[32 * wn + 16 * nt + l16] * ACT_LD;
        float acc[2][2][8];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][nt][i] = 0.0f;
        uint4 bq[2][4];
        float2 bx[2], bsum[2];   // the activation scales, and (the formats with a minimum) the codes' sums per 32 values
        auto fetch_b = [&](int s, uint4 (&q)[2][4], float2 (&x)[2], float2 (&sm)[2]) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint4* p = reinterpret_cast<const uint4*>(brow[nt] + s * AB);
                q[nt][0] = p[0]; q[nt][1] = p[1]; q[nt][2] = p[2]; q[nt][3] = p[3];
                x[nt] = *reinterpret_cast<const float2*>(brow[nt] + s * AB + 64);
                if constexpr (has_min(WT)) sm[nt] = *reinterpret_cast<const float2*>(brow[nt] + s * AB + 72);
                else sm[nt] = make_float2(0.0f, 0.0f);
            }
        };
        if (on) fetch_b(0, bq, bx, bsum);
        for (int s = 0; s < NS; ++s) {
            __syncthreads();                                      // stage s's weights are in buffer s & 1
            uint4 nbq[2][4];
            float2 nbx[2], nbsum[2];
            if (on && s + 1 < NS) fetch_b(s + 1, nbq, nbx, nbsum);
            if (on) {
                const int bf = s & 1;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int rb = 32 * wm + 16 * mt;
                        const uint4* ap = reinterpret_cast<const uint4*>(&wt[bf][rb + l16][32 * h]);
                        const nw_i4 A0 = nw_u4(ap[0]), A1 = nw_u4(ap[1]);
                        float w0[8], w1[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float2 sw = *reinterpret_cast<const float2*>(&ws[bf][rb + 2 * i + hi][2 * h]);
                            w0[i] = sw.x; w1[i] = sw.y;
                        }
#pragma unroll
                        for (int nt = 0; nt < 2; ++nt) {
                            const float dx = h ? bx[nt].y : bx[nt].x;
                            const nw_i4 B0 = nw_u4(bq[nt][2 * h]), B1 = nw_u4(bq[nt][2 * h + 1]);
                            const nw_i8 m = nw_i8{MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC, MAGIC};
                            if constexpr (K16) {
                                const nw_i8 d0 = nw_wmma(A0, B0, m), d1 = nw_wmma(A1, B1, m);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    const float v = fmaf(w1[i], dotf(d1[i]), w0[i] * dotf(d0[i]));
                                    acc[mt][nt][i] = fmaf(dx, v, acc[mt][nt][i]);
                                }
                            } else {
                                const nw_i8 d = nw_wmma(A1, B1, nw_wmma(A0, B0, m));
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    acc[mt][nt][i] = fmaf(w0[i] * dx, dotf(d[i]), acc[mt][nt][i]);
                                    if constexpr (has_min(WT)) acc[mt][nt][i] = fmaf(w1[i] * dx, h ? bsum[nt].y : bsum[nt].x, acc[mt][nt][i]);
                                }
                            }
                        }
                    }
                }
            }
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);                            // the other buffer: its readers passed this barrier
                if (s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
                if (on) {
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        bx[nt] = nbx[nt];
                        bsum[nt] = nbsum[nt];
#pragma unroll
                        for (int j = 0; j < 4; ++j) bq[nt][j] = nbq[nt][j];
                    }
                }
            }
        }
        if constexpr (GU) {
            if constexpr (ALIAS) __syncthreads();                 // every wave is done reading the weight buffers
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float up = __shfl_xor_sync(0xffffffffu, acc[mt][nt][i], 16);
                            const float gt = acc[mt][nt][i];
                            if (hi == 0) hs[32 * wn + 16 * nt + l16][16 * wm + 8 * mt + i] = gt / (1.0f + __expf(-gt)) * up;
                        }
            }
            __syncthreads();
            // H block fb (features 64 fb ..) to int8 per 32, natural order: a thread per (tile row, half)
            if (tid < 2 * kTileRows) {
                const int r = tid >> 1, hh = tid & 1;
                if (r < nrows) {
                    float am = 0.0f;
#pragma unroll 8
                    for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(hs[r][32 * hh + j]));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (size_t) fb * AB;
                    uint32_t wd[8] = {};
                    int csum = 0;
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const int c = __float2int_rn(hs[r][32 * hh + j] * inv);
                        csum += c;
                        wd[j >> 2] |= (uint32_t) (uint8_t) (int8_t) c << (8 * (j & 3));
                    }
                    uint4* o4 = reinterpret_cast<uint4*>(o + 32 * hh);
                    o4[0] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                    o4[1] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
                    *reinterpret_cast<float*>(o + 64 + 4 * hh) = am / 127.0f;
                    *reinterpret_cast<float*>(o + 72 + 4 * hh) = (float) csum;   // (the formats with a minimum)
                }
            }
        } else {
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const int r = 32 * wn + 16 * nt + l16;
                        if (r < nrows) {
                            float* d = dm + (size_t) (row0 + r) * 2560 + rbase + 32 * wm + 16 * mt + hi;
#pragma unroll
                            for (int i = 0; i < 8; ++i) d[2 * i] = acc[mt][nt][i];
                        }
                    }
            }
        }
    }
#else
    __builtin_trap();
#endif
}
#endif  // __HIPCC__

unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

// per device: whether every kernel here runs (sm_80+, device code in this build, fits), and their occupancy
struct DevInfo {
    bool done = false, ok = false;
    int sms = 0, occ = 1;
};
std::mutex g_mu;
DevInfo g_dev[32];

#if !defined(__HIPCC__)
template <int T, bool GU, int WW> bool setup_ww(int& occ) {
    dpct::kernel_function_info fa{};
    if (cudaFuncGetAttributes(&fa, native_kernel<T, GU, WW>) != 0 ||
        fa.ptxVersion < 80) return false;
    if (cudaFuncSetAttribute(native_kernel<T, GU, WW>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes(T, WW)) != 0)
        return false;
    int o = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &o, native_kernel<T, GU, WW>, THREADS, smem_bytes(T, WW)) != 0 ||
        o < 1)
        return false;
    occ = std::min(occ, o);
    return true;
}
template <int T, bool GU> bool setup_one(int& occ) { return setup_ww<T, GU, 4>(occ) && setup_ww<T, GU, 2>(occ); }
#endif

#if defined(__HIPCC__)
// Resident blocks per multiprocessor for the gfx11 WMMA prompt-expert kernels' persistent grids.  HIP counts a gfx11
// WGP (two CUs, which these kernels use as one in the default WGP mode) as one multiprocessor, and the occupancy query
// answers 1 block for these kernels where 2 run side by side - so the grid was half the GPU's room.  gfx1151 (20
// WGPs), 8192 tokens x top 10, IQ3_S / IQ4_NL, one layer: 1 block per WGP 60.7 ms, 2 44.7, 3 49.9, 4 45.2 (the same
// work items and arithmetic: the results do not change).  STRATA_PF_OCC=N sets the blocks per WGP (an experiment knob).
static int wgp_blocks(int occ) {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_OCC");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    return env > 0 ? env : 2 * std::max(occ, 1);
}
#endif
const DevInfo& dev_info() {
    int dev = 0;
    dev = dpct::get_current_device_id();
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
#if defined(STRATA_HIP_GFX906)
    // gfx906 reports compute capability 9.0 through HIP, but the mma.sync bodies above are CUDA sm_80+ only and
    // empty in a hipcc build: never available here (the prompt path keeps its own expert GEMMs)
    return d;
#elif defined(__HIPCC__)
    {   // gfx11 with this build's code: the WMMA kernels (one occupancy for all: the same shape and LDS budget)
        cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceProp prop;
        hipFuncAttributes fa{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || std::strncmp(prop.gcnArchName, "gfx11", 5) != 0 ||
            hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(native_w11_kernel<T_IQ3_XXS, true>)) != hipSuccess) {
            cudaGetLastError();
            return d;
        }
        int o = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&o, native_w11_kernel<T_IQ2_S, true>, NW_THREADS, 0) != hipSuccess)
            o = 0;
        d.ok = o >= 1;
        d.occ = d.ok ? o : 1;
        cudaGetLastError();
        return d;
    }
#else
    int major = 0;
    major = dpct::get_device(dev).get_major_version();
    d.sms = dpct::get_device(dev).get_max_compute_units();
    if (major < 8) return d;
    int occ = 1 << 20;
    d.ok = setup_one<T_IQ2_XXS, true>(occ) && setup_one<T_IQ2_XS, true>(occ) && setup_one<T_IQ2_S, true>(occ) &&
           setup_one<T_IQ3_XXS, true>(occ) && setup_one<T_IQ3_S, true>(occ) && setup_one<T_IQ4_XS, true>(occ) &&
           setup_one<T_Q2_0, false>(occ) && setup_one<T_IQ4_NL, false>(occ);
    d.occ = d.ok ? occ : 1;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
#endif
    return d;
}

// opt-in on top of STRATA_PF_FUSED=1: STRATA_PF_FUSED_KQ=1 takes UD-Q4_K_XL's formats (the output differs from the FP16
// expert path's at rounding level; KL-gated, see aurora_s23.md)
bool kq_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_FUSED_KQ");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}
bool gu_covered(int t) {
    return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS || t == T_IQ3_S || t == T_IQ4_XS
#if defined(__HIPCC__)
           || ((t == T_Q4_K || t == T_Q5_K) && kq_on())   // the gfx11 kernels only
#endif
        ;
}
bool d_covered(int t) {
    return t == T_Q2_0 || t == T_IQ4_NL
#if defined(__HIPCC__)
           || ((t == T_Q5_1 || t == T_Q8_0) && kq_on())
#endif
        ;
}

template <int T, bool GU>
void launch(int ww, unsigned grid, const Batch &b, const NativeGeom &g,
            const Tables &tb, const void *act, const int32_t *src, void *out,
            float *dm, dpct::queue_ptr s) {
    const uint8_t* a = (const uint8_t*) act;
    uint8_t* o = (uint8_t*) out;
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
            if (ww == 4) {
                iq2xxs_grid.init(*s);
                iq2xs_grid.init(*s);
                iq2s_grid.init(*s);
                iq3xxs_grid.init(*s);
                iq3s_grid.init(*s);
                kvalues_iq4nl.init(*s);

                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                s->submit([&](sycl::handler &cgh) {
                    auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();
                    auto iq2xs_grid_ptr_ct1 = iq2xs_grid.get_ptr();
                    auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
                    auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
                    auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
                    auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem_bytes(T, 4)), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class native_kernel_6643fb, dpct_kernel_scalar<T>,
                        dpct_kernel_scalar<GU>, dpct_kernel_scalar<4>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, grid) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                native_kernel<T, GU, 4>(
                                    b, g, tb, a, src, o, dm,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    iq2xxs_grid_ptr_ct1, iq2xs_grid_ptr_ct1,
                                    iq2s_grid_ptr_ct1, iq3xxs_grid_ptr_ct1,
                                    iq3s_grid_ptr_ct1, kvalues_iq4nl_ptr_ct1);
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
                iq2xxs_grid.init(*s);
                iq2xs_grid.init(*s);
                iq2s_grid.init(*s);
                iq3xxs_grid.init(*s);
                iq3s_grid.init(*s);
                kvalues_iq4nl.init(*s);

                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                s->submit([&](sycl::handler &cgh) {
                    auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();
                    auto iq2xs_grid_ptr_ct1 = iq2xs_grid.get_ptr();
                    auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
                    auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
                    auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
                    auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem_bytes(T, 2)), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class native_kernel_e6c346, dpct_kernel_scalar<T>,
                        dpct_kernel_scalar<GU>, dpct_kernel_scalar<2>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, grid) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                native_kernel<T, GU, 2>(
                                    b, g, tb, a, src, o, dm,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    iq2xxs_grid_ptr_ct1, iq2xs_grid_ptr_ct1,
                                    iq2s_grid_ptr_ct1, iq3xxs_grid_ptr_ct1,
                                    iq3s_grid_ptr_ct1, kvalues_iq4nl_ptr_ct1);
                            });
                });
            }
}

// The work item's shape for a layer of `n` routed rows over `n_expert` experts: 128 routed rows when an expert has
// about one such tile (56 to 112 rows on average), else 64.  Measured on the RTX 5070 (tests/cuda/prefill_fused_iq_test,
// a layer of 2048 / 3584 / 8192 tokens, ms, 64 -> 128): IQ2_S 5.2 -> 6.7, 9.5 -> 9.3, 17.5 -> 19.5; IQ2_XXS 4.2 ->
// 5.8, 7.4 -> 7.6, 13.6 -> 16.5; IQ3_S 5.6 -> 6.6, 10.2 -> 8.8, 18.4 -> 19.0; IQ3_XXS 5.1 -> 6.3, 9.1 -> 8.3, 16.5 ->
// 17.5 (256 rows was slower still).  STRATA_PF_FUSED_TILE=64|128 forces one.
int pick_ww(int64_t n, int n_expert) {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_PF_FUSED_TILE");
        const int t = v ? std::atoi(v) : 0;
        return t == 64 ? 4 : t == 128 ? 2 : 0;
    }();
    if (forced) return forced;
    const double avg = (double) n / std::max(n_expert, 1);
    return avg > 56.0 && avg <= 112.0 ? 2 : 4;
}

}  // namespace

bool native_supported(int gu_type, int d_type) {
    static const bool off = [] {   // STRATA_PF_FUSED_NATIVE=0: the native packs keep MMQ under STRATA_PF_FUSED=1 (A/B)
        const char* v = std::getenv("STRATA_PF_FUSED_NATIVE");
        return v != nullptr && v[0] == '0';
    }();
    return !off && requested() && gu_covered(gu_type) && d_covered(d_type) && dev_info().ok;   // opt-in (=1)
}

void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class quant_act_nat_kernel_7d58c5>>(
                    sycl::nd_range<3>(sycl::range(1, 1, blocks(nblk, 8)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            quant_act_nat_kernel(x, nblk, (uint8_t *)xa);
                        });
        }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "quantize_act_native");
}

void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    if (!d.ok || !gu_covered(g.gu_type) || !d_covered(g.d_type)) {
        std::fprintf(stderr, "prefill fused experts (native): types %d / %d are not covered here\n", g.gu_type, g.d_type);
        std::exit(1);
    }
    const dpct::queue_ptr s = strata::q_of(stream);
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
#if defined(__HIPCC__)
    {
        const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
        // W (X1): LDS alias of the H tile + waves_per_eu(8) (192 VGPRs) + 4 blocks per WGP, only for the types whose
        // kernels do not spill under the cap (measured: gate/up IQ2_*, IQ3_*, IQ4_XS 0-5 regs; down IQ4_NL 11, Q8_0 / Q2_0 0;
        // Q4_K / Q5_K gate/up and Q5_1 down spill 100+ and keep the old launch).
#ifdef STRATA_W_NO_OCC
        const bool occ_gu = false, occ_d = false;
#else
        const bool occ_gu = g.gu_type == T_IQ2_XXS || g.gu_type == T_IQ2_XS || g.gu_type == T_IQ2_S ||
                            g.gu_type == T_IQ3_XXS || g.gu_type == T_IQ3_S || g.gu_type == T_IQ4_XS;
        const bool occ_d = g.d_type == T_Q2_0 || g.d_type == T_Q8_0 || g.d_type == T_IQ4_NL;
#endif
        const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / NW_ROWS), (int64_t) d.sms * (occ_gu ? 4 : wgp_blocks(d.occ)));
        const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / NW_ROWS), (int64_t) d.sms * (occ_d ? 4 : wgp_blocks(d.occ)));
        const uint8_t* xa8 = (const uint8_t*) xa;
        uint8_t* ha8 = (uint8_t*) ha;
#define STRATA_NW_GU(T) do { if (occ_gu) native_w11_kernel<T, true, true, 8><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr); else native_w11_kernel<T, true><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr); } while (0)
        switch (g.gu_type) {
            case T_IQ2_XXS: STRATA_NW_GU(T_IQ2_XXS); break;
            case T_IQ2_XS: STRATA_NW_GU(T_IQ2_XS); break;
            case T_IQ2_S: STRATA_NW_GU(T_IQ2_S); break;
            case T_IQ3_XXS: STRATA_NW_GU(T_IQ3_XXS); break;
            case T_IQ3_S: STRATA_NW_GU(T_IQ3_S); break;
            case T_Q4_K: STRATA_NW_GU(T_Q4_K); break;
            case T_Q5_K: STRATA_NW_GU(T_Q5_K); break;
            default: STRATA_NW_GU(T_IQ4_XS); break;
        }
#undef STRATA_NW_GU
#define STRATA_NW_D(T) do { if (occ_d) native_w11_kernel<T, false, true, 8><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm); else native_w11_kernel<T, false><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm); } while (0)
        if (g.d_type == T_Q2_0) STRATA_NW_D(T_Q2_0);
        else if (g.d_type == T_Q5_1) STRATA_NW_D(T_Q5_1);
        else if (g.d_type == T_Q8_0) STRATA_NW_D(T_Q8_0);
        else STRATA_NW_D(T_IQ4_NL);
#undef STRATA_NW_D
        ck(cudaGetLastError(), "experts_native");
        return;
    }
#endif
    const int ww = pick_ww(n, n_expert);
    // the most 64-row tiles the batch can have (every row in it, plus a partial tile per expert) - the items of the
    // ones inside a larger item end at once
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    switch (g.gu_type) {
        case T_IQ2_XXS: launch<T_IQ2_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_S: launch<T_IQ2_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_XXS: launch<T_IQ3_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_S: launch<T_IQ3_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        default: launch<T_IQ4_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
    }
    if (g.d_type == T_Q2_0) launch<T_Q2_0, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    else launch<T_IQ4_NL, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "experts_native");
}

}  // namespace strata::prefill::fused
