// src/prefill/moe_fused.cu - see include/strata/prefill/moe_fused.hpp (#136: the Q2_0 pack's prompt experts on
// Strata's own int8 tensor-core kernels, with the grouping on the GPU).
//
// The arithmetic.  A Q2_0 block is 64 weights w = d_w * (q - 1), q in 0..3, four codes per byte, weight j of a block
// in byte j / 4 at bits 2 * (j % 4).  An activation block of 32 values is x = d_x * a with a = round(x / d_x) in
// -127..127.  Over one 32-value half-block
//     sum w x = d_w * d_x * (sum q a - sum a),
// so the int8 product takes the codes as they are (0..3 is a valid s8) and the offset is the activations' own sum,
// stored with the block.  The dot product (|sum q a| <= 32 * 3 * 127 < 2^22) becomes a float exactly with one
// integer add: as_float(0x4B400000 + dot) = 1.5 * 2^23 + dot, and the block stores c = -(1.5 * 2^23 + sum a), so
// as_float(0x4B400000 + dot) + c = dot - sum a with no rounding.  Per output and 64-block that is
//     acc += d_w * (d_x0 * t0 + d_x1 * t1)        (t = dot - sum a of each half)
// - two integer adds, two float adds and three multiply-adds.
//
// The K order.  mma.sync m16n8k32 takes four consecutive k of a row (A) or column (B) per register.  Four consecutive
// weights are one code byte; four codes of one register instead come from the same 2-bit field of four consecutive
// bytes - (word >> 2t) & 0x03030303 holds weights t, t+4, t+8, t+12 of a 16-weight word.  So the kernels use that order
// for the k of a half-block: mma k index 16h + 4t + j stands for weight (and activation) 16h + t + 4j.  The
// activations are stored in it (perm32), and a lane's two B registers (h = 0 and 1) are adjacent: one 8-byte load.
#include "strata/prefill/moe_fused.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/core/emulate.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::prefill::fused {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fused experts: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, then {d0, c0, d1, c1}
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// The Strata Q2_0 blob (moe_mmq.cu strata_q2_kernel, kernels.cu blob_dequant_kernel): gate/up codes [1280][640 B]
// (rows interleaved: gate of feature f at row 2f, its up at 2f + 1), down codes [2560][160 B], gate/up scales
// [1280][40] fp16, down scales [2560][10] fp16.
constexpr size_t O_GU_CODES = 0, O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                 O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
constexpr int THREADS = 512;            // 16 warps: 4 along the weight rows (64 each) x 4 along the tile rows (16 each)
constexpr int WROWS = 256;              // weight rows per work item: 128 features of gate/up, or 256 outputs of down
constexpr int STAGES = 4;               // cp.async pipeline depth, one 64-weight block of K per stage
constexpr int STAGE_BYTES = WROWS * 16 + kTileRows * AB;
constexpr size_t smem_bytes(bool gu) {
    return (size_t) WROWS * (gu ? 40 : 10) * 2 + STAGES * STAGE_BYTES + kTileRows * 4;
}

// the stored position of value k of a 32-value half-block (see "The K order")
__host__ __device__ __forceinline__ int perm32(int k) { return 8 * (k & 3) + 4 * (k >> 4) + ((k & 15) >> 2); }
// where value k of a 32-value half-block is stored: perm32 for the CUDA kernel; on HIP the gfx11 WMMA kernel's order
// (value 4b + t of a 16-value word at 4t + b: see expert_w11_kernel)
__host__ __device__ __forceinline__ int act_pos(int k) {
#if defined(__HIPCC__)
    return (k & 16) | ((k & 3) << 2) | ((k >> 2) & 3);
#else
    return perm32(k);
#endif
}

struct Tables {
    int32_t *cnt, *off, *fill, *ts;      // per expert: rows, first row (E + 1), placement cursor, first tile (E + 1)
    int2* tiles;                         // {expert, first row}, in expert order
};
size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

// ---- activations: one warp per 64 values
__global__ void quant_act_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    const int q0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0;
    const int q1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    int s0 = q0, s1 = q1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    uint8_t* out = xa + w * AB;
    out[act_pos(lane)] = (uint8_t) (int8_t) q0;
    out[32 + act_pos(lane)] = (uint8_t) (int8_t) q1;
    if (lane == 0)
        *(float4*) (out + 64) = make_float4(a0 / 127.0f, -(MAGICF + (float) s0), a1 / 127.0f, -(MAGICF + (float) s1));
}

// ---- grouping: counts, offsets and tiles, placement
__global__ void count_kernel(const int32_t* __restrict__ ids, int64_t n, int E, int32_t* __restrict__ cnt) {
    extern __shared__ int32_t hist[];
    for (int e = threadIdx.x; e < E; e += blockDim.x) hist[e] = 0;
    __syncthreads();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int e = ids[i];
        if ((unsigned) e < (unsigned) E) atomicAdd(&hist[e], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        if (hist[e]) atomicAdd(&cnt[e], hist[e]);
}

// exclusive prefix sum over the block (blockDim a multiple of 32); *total = the block's sum
__device__ int block_excl_sum(int v, int* total, int* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[w] = x;
    __syncthreads();
    if (w == 0) {
        int s = lane < nw ? sh[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        sh[lane] = s;
    }
    __syncthreads();
    const int r = x - v + (w > 0 ? sh[w - 1] : 0);
    *total = sh[nw - 1];
    __syncthreads();   // sh is reused by the next call
    return r;
}

// one block, a thread per expert (E <= 1024)
__global__ void scan_kernel(Tables tb, int E) {
    __shared__ int sh[32];
    const int e = threadIdx.x;
    const int c = e < E ? tb.cnt[e] : 0, nt = (c + kTileRows - 1) / kTileRows;
    int rows = 0, tiles = 0;
    const int o = block_excl_sum(c, &rows, sh), to = block_excl_sum(nt, &tiles, sh);
    if (e < E) {
        tb.off[e] = o;
        tb.fill[e] = o;
        tb.ts[e] = to;
        for (int i = 0; i < nt; ++i) tb.tiles[to + i] = make_int2(e, o + i * kTileRows);
    }
    if (e == 0) {
        tb.off[E] = rows;
        tb.ts[E] = tiles;
    }
}

__global__ void place_kernel(const int32_t* __restrict__ ids, int64_t n, int k, int E, Tables tb,
                             int32_t* __restrict__ slot, int32_t* __restrict__ src) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = ids[i];
    if ((unsigned) e >= (unsigned) E) { slot[i] = 0; return; }   // (the router never writes one)
    const int p = atomicAdd(&tb.fill[e], 1);
    slot[i] = p;
    src[p] = (int32_t) (i / k);
}

// ---- the expert products
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = A (16 x 32 s8, row) * B (32 x 8 s8, col), int32, from zero
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
// Turing (STRATA_PF_FUSED_SM75): no cp.async - the stages are filled by plain copies (the same bytes; the barriers
// around them stay) - and no m16n8k32 with s8 operands: it is four m8n8k16, rows g / g + 8 by k 0-15 / 16-31 (the
// m16n8k32 fragments: a[0] / a[1] rows g / g + 8 at k 4 tig.., a[2] / a[3] the same at k 16 + 4 tig.., b0 / b1 the
// two k halves).  The int32 sums are exact in any order: the same d as the sm_80 instruction.
__device__ __forceinline__ void cp16(void* dst, const void* src) { *(uint4*) dst = __ldg((const uint4*) src); }
__device__ __forceinline__ void cp4(void* dst, const void* src) { *(uint32_t*) dst = __ldg((const uint32_t*) src); }
__device__ __forceinline__ void cp_commit() {}
template <int N> __device__ __forceinline__ void cp_wait() {}
__device__ __forceinline__ void mma8_s8(int& c0, int& c1, uint32_t a, uint32_t b) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    d[0] = d[1] = d[2] = d[3] = 0;
    mma8_s8(d[0], d[1], a[0], b0);
    mma8_s8(d[0], d[1], a[2], b1);
    mma8_s8(d[2], d[3], a[1], b0);
    mma8_s8(d[2], d[3], a[3], b1);
}
#endif

// One work item = (tile of up to 64 routed rows of expert e, a block of 256 weight rows); a persistent grid walks
// the batch's items (their count is on the device).  GU: gate/up rows of 128 features against the rows' tokens'
// activations (`act` per token, 2560 values), SwiGLU, H to int8 per 32 features into `out` (per row, 640 values).
// !GU: down rows (256 outputs) against H (`act` per row), FP32 into `dm` per row.
// Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, its gate rows as mma rows 0-7 and
// its up rows as 8-15, so a lane holds gate and up of one feature: SwiGLU in registers), tile rows 16 wt.. as two n8.
template <bool GU>
__global__ void __launch_bounds__(THREADS, 1)
expert_kernel(const Batch b, const Tables tb, const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
              uint8_t* __restrict__ out, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    constexpr int NKB = GU ? 40 : 10;                 // 64-weight blocks along K
    constexpr int NFB = GU ? 5 : 10;                  // weight-row blocks per tile
    constexpr int CODE_LD = GU ? 640 : 160;           // code bytes per weight row
    constexpr size_t O_CODES = GU ? O_GU_CODES : O_D_CODES, O_SC = GU ? O_GU_SC : O_D_SC;
    constexpr int ACT_LD = NKB * AB;                  // bytes per activation row: 3200 (a token), 800 (a row's H)
    constexpr unsigned M2 = 0x03030303u;
    extern __shared__ __align__(16) uint8_t smem[];
    const __half* wsc = (const __half*) smem;                     // [256][NKB] the work item's weight scales
    uint8_t* stages = smem + WROWS * NKB * 2;                     // [STAGES] x {codes [256][16], act [64][AB]}
    int* srow = (int*) (stages + STAGES * STAGE_BYTES);           // [64] the activation row of each tile row

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp & 3, nb0 = 16 * (warp >> 2);
    int ra[4], rb[4];                                             // the local weight rows of mma rows g and g + 8
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (GU) { ra[i] = 2 * (32 * wf + 8 * i + g); rb[i] = ra[i] + 1; }
        else { ra[i] = 64 * wf + 16 * i + g; rb[i] = ra[i] + 8; }
    }
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y, nrows = min(kTileRows, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WROWS;
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);             // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        for (int c = tid; c < WROWS * 5; c += THREADS) {          // the scales: 5 x 16 B (GU) or 5 x 4 B a row
            const int r = c / 5, q = c % 5;
            if (GU) cp16(smem + r * 80 + q * 16, blob + O_SC + (size_t) (rbase + r) * 80 + q * 16);
            else cp4(smem + r * 20 + q * 4, blob + O_SC + (size_t) (rbase + r) * 20 + q * 4);
        }
        __syncthreads();                                          // srow
        auto load = [&](int kb) {
            uint8_t* st = stages + (kb % STAGES) * STAGE_BYTES;
            if (tid < WROWS) cp16(st + tid * 16, blob + O_CODES + (size_t) (rbase + tid) * CODE_LD + kb * 16);
            const int a = tid - (THREADS - kTileRows * 5);        // the last 320 threads: 64 rows x 5 x 16 B
            if (a >= 0) {
                const int r = a / 5, q = a % 5;
                cp16(st + WROWS * 16 + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + kb * AB + q * 16);
            }
        };
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < NKB) load(s);
            cp_commit();
        }
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int kb = 0; kb < NKB; ++kb) {
            cp_wait<STAGES - 2>();
            __syncthreads();
            if (kb + STAGES - 1 < NKB) load(kb + STAGES - 1);
            cp_commit();
            if (!on0) continue;
            const uint8_t* st = stages + (kb % STAGES) * STAGE_BYTES;
            const uint8_t* sa = st + WROWS * 16;
            uint2 bq[2][2];                                       // [n8 tile][half]: the B registers
            float4 sc[2][2];                                      // [n8 tile][column 2 tig + c]: {d0, c0, d1, c1}
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
                const uint8_t* r = sa + (nb0 + 8 * n + g) * AB + 8 * tig;
                bq[n][0] = *(const uint2*) r;
                bq[n][1] = *(const uint2*) (r + 32);
                sc[n][0] = *(const float4*) (sa + (nb0 + 8 * n + 2 * tig) * AB + 64);
                sc[n][1] = *(const float4*) (sa + (nb0 + 8 * n + 2 * tig + 1) * AB + 64);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4 ca = *(const uint4*) (st + ra[i] * 16), cb = *(const uint4*) (st + rb[i] * 16);
                const float da = __half2float(wsc[ra[i] * NKB + kb]), db = __half2float(wsc[rb[i] * NKB + kb]);
                const int s = 2 * tig;
                const uint32_t a0[4] = {(ca.x >> s) & M2, (cb.x >> s) & M2, (ca.y >> s) & M2, (cb.y >> s) & M2};
                const uint32_t a1[4] = {(ca.z >> s) & M2, (cb.z >> s) & M2, (ca.w >> s) & M2, (cb.w >> s) & M2};
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
                    int d0[4], d1[4];
                    mma_s8(d0, a0, bq[n][0].x, bq[n][0].y);
                    mma_s8(d1, a1, bq[n][1].x, bq[n][1].y);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const float4 z = sc[n][q & 1];
                        const float u0 = __int_as_float(MAGIC + d0[q]) + z.y, u1 = __int_as_float(MAGIC + d1[q]) + z.w;
                        acc[i][n][q] = fmaf(q < 2 ? da : db, fmaf(z.z, u1, z.x * u0), acc[i][n][q]);
                    }
                }
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature 32 wf + 8 i + g (lane g of m tile i), tile rows nb0 + 8 n + 2 tig + c; H to int8 per
            // row over the warp's 32 features (= one 32-value half-block of the down product's K)
            const int blk = 4 * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    int qv[4], sum = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) { qv[i] = __float2int_rn(h[i] * inv); sum += qv[i]; }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + perm32(8 * i + g)] = (uint8_t) (int8_t) qv[i];
                        if (g == 0) *(float2*) (o + 64 + 8 * hh) = make_float2(am / 127.0f, -(MAGICF + (float) sum));
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
                        if (r < nrows) dm[(size_t) (row0 + r) * 2560 + rbase + (q < 2 ? ra[i] : rb[i])] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}

#if defined(__HIPCC__)
// ---- Aurora (S23): the same products on gfx11 (RDNA3 / RDNA3.5) matrix cores, v_wmma_i32_16x16x16_iu8 (wave32).
// Fragments (gfx11): A lane l holds the 16 k of row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l the 16 k of
// column l % 16, C/D lane l holds D[2i + l / 16][l % 16], i = 0..7.  A = 16 weight rows, B = 16 tile rows (tokens):
// a lane's eight results are eight weight rows of ONE tile row, whose activation scales it already holds.
// The K order of a 16-weight word (one k-step): A register t = (word >> 2t) & 0x03030303 holds weights t, t+4, t+8,
// t+12, so k index p = 4t + b stands for weight 4b + t; the activations are stored in that order (act_pos).  The
// integer dot of each 32-value half (two k-steps) is exact and the epilogue is the CUDA kernel's, so the numbers are
// those of expert_kernel (another K order inside exact integer sums).
// No LDS staging: a wave reads its weight rows (16 B per row and 64-weight block) and its tile rows' activation
// blocks (80 B) straight from global memory (L2: an expert's rows are read by every tile, a token's by every weight
// block), one block ahead in registers; the work item's weight scales go to LDS once.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_FUSED_W11 1
#else
#define STRATA_FUSED_W11 0
#endif
typedef int w11_i4 __attribute__((ext_vector_type(4)));
typedef int w11_i8 __attribute__((ext_vector_type(8)));
constexpr int W_ROWS = 128;                       // weight rows per work item (GU: 64 features = one H block)
constexpr int W_THREADS = 256;                    // 8 waves: 4 along the weight rows (32 each) x 2 along the tile (32)

__device__ __forceinline__ w11_i8 wmma_iu8(w11_i4 a, w11_i4 b, w11_i8 c) {
#if STRATA_FUSED_W11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, true, b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}

template <bool GU>
__global__ void __launch_bounds__(W_THREADS)
expert_w11_kernel(const Batch b, const Tables tb, const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
                  uint8_t* __restrict__ out, float* __restrict__ dm) {
#if STRATA_FUSED_W11
    constexpr int NKB = GU ? 40 : 10;                 // 64-weight blocks along K
    constexpr int NFB = (GU ? 1280 : 2560) / W_ROWS;  // work items per tile
    constexpr int CODE_LD = GU ? 640 : 160;           // code bytes per weight row
    constexpr size_t O_CODES = GU ? O_GU_CODES : O_D_CODES, O_SC = GU ? O_GU_SC : O_D_SC;
    constexpr int ACT_LD = NKB * AB;                  // bytes per activation row
    constexpr unsigned M2 = 0x03030303u;
    __shared__ __align__(16) uint16_t wsc[W_ROWS * NKB];        // the item's weight scales (fp16), [row][kb]
    __shared__ int srow[kTileRows];
    __shared__ float hs[GU ? kTileRows : 1][GU ? 65 : 1];      // GU: H of the item's 64 features, [tile row][feature]
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y, nrows = min(kTileRows, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * W_ROWS;
        __syncthreads();                                          // the previous item is done with srow / wsc / hs
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);             // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        {
            const uint4* s4 = reinterpret_cast<const uint4*>(blob + O_SC + (size_t) rbase * NKB * 2);
            for (int i = tid; i < W_ROWS * NKB / 8; i += W_THREADS) reinterpret_cast<uint4*>(wsc)[i] = s4[i];
        }
        __syncthreads();
        const bool on = 32 * wn < nrows;                          // a wave past the tile's rows only stages
        float acc[2][2][8];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][nt][i] = 0.0f;
        if (on) {
            const uint8_t* arow[2];
            const uint8_t* brow[2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) arow[mt] = blob + O_CODES + (size_t) (rbase + 32 * wm + 16 * mt + l16) * CODE_LD;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) brow[nt] = act + (size_t) srow[32 * wn + 16 * nt + l16] * ACT_LD;
            uint4 ac[2], bq[2][4];
            float4 bs[2];
            auto fetch = [&](int kb, uint4 (&a)[2], uint4 (&q)[2][4], float4 (&s)[2]) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) a[mt] = *reinterpret_cast<const uint4*>(arow[mt] + kb * 16);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint4* p = reinterpret_cast<const uint4*>(brow[nt] + kb * AB);
                    q[nt][0] = p[0]; q[nt][1] = p[1]; q[nt][2] = p[2]; q[nt][3] = p[3];
                    s[nt] = *reinterpret_cast<const float4*>(brow[nt] + kb * AB + 64);
                }
            };
            fetch(0, ac, bq, bs);
            for (int kb = 0; kb < NKB; ++kb) {
                uint4 nac[2], nbq[2][4];
                float4 nbs[2];
                if (kb + 1 < NKB) fetch(kb + 1, nac, nbq, nbs);
                w11_i8 c[2][2][2];                                // [half][mt][nt]
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int nt = 0; nt < 2; ++nt) c[h][mt][nt] = w11_i8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 4; ++ks) {                  // k-step = 16-weight word ks of the block
                    w11_i4 A[2];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const unsigned wd = ks == 0 ? ac[mt].x : ks == 1 ? ac[mt].y : ks == 2 ? ac[mt].z : ac[mt].w;
                        A[mt] = w11_i4{(int) (wd & M2), (int) ((wd >> 2) & M2), (int) ((wd >> 4) & M2),
                                       (int) ((wd >> 6) & M2)};
                    }
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const uint4 q = bq[nt][ks];
                        const w11_i4 B = w11_i4{(int) q.x, (int) q.y, (int) q.z, (int) q.w};
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt) c[ks >> 1][mt][nt] = wmma_iu8(A[mt], B, c[ks >> 1][mt][nt]);
                    }
                }
                // the epilogue: per 64-weight block acc += d_w * (d_x0 * (dot0 - sum0) + d_x1 * (dot1 - sum1))
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    float dw[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int m = 32 * wm + 16 * mt + 2 * i + hi;
                        dw[i] = __half2float(__ushort_as_half(wsc[m * NKB + kb]));
                    }
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const float4 z = bs[nt];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float u0 = __int_as_float(MAGIC + c[0][mt][nt][i]) + z.y;
                            const float u1 = __int_as_float(MAGIC + c[1][mt][nt][i]) + z.w;
                            acc[mt][nt][i] = fmaf(dw[i], fmaf(z.z, u1, z.x * u0), acc[mt][nt][i]);
                        }
                    }
                }
                if (kb + 1 < NKB) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) ac[mt] = nac[mt];
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        bs[nt] = nbs[nt];
#pragma unroll
                        for (int j = 0; j < 4; ++j) bq[nt][j] = nbq[nt][j];
                    }
                }
            }
        }
        if constexpr (GU) {
            // SwiGLU: row 2f is feature f's gate (lanes 0..15), 2f + 1 its up (lanes 16..31) - the same (mt, i)
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
            // H block fb (features 64 fb .. 64 fb + 63) to int8: a thread per (tile row, half)
            if (tid < 2 * kTileRows) {
                const int r = tid >> 1, hh = tid & 1;
                if (r < nrows) {
                    float am = 0.0f;
#pragma unroll 8
                    for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(hs[r][32 * hh + j]));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (size_t) fb * AB;
                    int sum = 0;
                    uint32_t wd[8] = {};
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const int qv = __float2int_rn(hs[r][32 * hh + j] * inv);
                        sum += qv;
                        const int p = act_pos(j);
                        wd[p >> 2] |= (uint32_t) (uint8_t) (int8_t) qv << (8 * (p & 3));
                    }
                    uint4* o4 = reinterpret_cast<uint4*>(o + 32 * hh);
                    o4[0] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                    o4[1] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
                    *reinterpret_cast<float2*>(o + 64 + 8 * hh) = make_float2(am / 127.0f, -(MAGICF + (float) sum));
                }
            }
        } else if (on) {
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
#else
    __builtin_trap();
#endif
}
#endif  // __HIPCC__

unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

struct DevInfo {
    bool done = false;
    int cc = 0, sms = 0, occ_gu = 0, occ_d = 0;
};
std::mutex g_mu;
DevInfo g_dev[32];

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
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
#if defined(STRATA_HIP_GFX906)
    // gfx906 reports compute capability 9.0 through HIP, but the mma.sync bodies above are CUDA sm_80+ only and
    // empty in a hipcc build: never available here (the prompt path keeps its own expert GEMMs)
    return d;
#elif defined(__HIPCC__)
    {   // gfx11 (RDNA3 / RDNA3.5) with this build's code for it: the WMMA kernels
        cudaDeviceProp prop;
        cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
        hipFuncAttributes fa{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || !strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName) ||
            hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(expert_w11_kernel<true>)) != hipSuccess) {
            cudaGetLastError();
            return d;
        }
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_gu, expert_w11_kernel<true>, W_THREADS, 0) != hipSuccess ||
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_d, expert_w11_kernel<false>, W_THREADS, 0) != hipSuccess)
            d.occ_gu = d.occ_d = 0;
        d.cc = d.occ_gu >= 1 && d.occ_d >= 1 ? 80 : 0;
        cudaGetLastError();
        return d;
    }
#else
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    d.cc = 10 * strata::cc_major_of(major) + strata::cc_minor_of(minor);   // STRATA_EMULATE_CC counts (tests)
    // Turing (sm_75): the same kernels with plain copies and m8n8k16 MMAs, opt-in (STRATA_PF_FUSED_SM75=1)
    static const bool sm75 = [] { const char* v = std::getenv("STRATA_PF_FUSED_SM75"); return v != nullptr && v[0] == '1'; }();
    if (d.cc < 75 || (d.cc < 80 && !sm75)) { d.cc = d.cc < 80 ? 0 : d.cc; return d; }
    // the device code of this card's arch must have the kernels (a build without an sm_75+ target has an empty body)
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, expert_kernel<true>) != cudaSuccess || fa.ptxVersion < (d.cc >= 80 ? 80 : 75)) {
        cudaGetLastError();
        d.cc = 0;
        return d;
    }
    ck(cudaFuncSetAttribute(expert_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes(true)),
       "smem gu");
    ck(cudaFuncSetAttribute(expert_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes(false)),
       "smem down");
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_gu, expert_kernel<true>, THREADS, smem_bytes(true));
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_d, expert_kernel<false>, THREADS, smem_bytes(false));
    if (d.occ_gu < 1 || d.occ_d < 1) d.cc = 0;                   // does not fit this card: the MMQ path
    cudaGetLastError();
#endif
    return d;
}

}  // namespace

bool built() { return true; }
bool available() { return dev_info().cc >= 75; }
// 0.1.36: on by default for the Q2_0 pack (+13-23% prompts on an RTX 5070; teacher-forced against the FP16 path as
// close as MMQ at 8K and closer at 32K: phaseA-tests s18-tf); STRATA_PF_FUSED=0 keeps MMQ.  The native packs' kernels
// (moe_fused_iq) stay opt-in: STRATA_PF_FUSED=1 (requested()).
bool enabled() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_PF_FUSED");
#if defined(__HIPCC__)
        return v != nullptr && v[0] == '1';   // Aurora: the gfx11 WMMA kernels are opt-in (STRATA_PF_FUSED=1)
#else
        return v == nullptr || v[0] != '0';
#endif
    }();
    return env && available();
}
bool requested() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_PF_FUSED");
        return v != nullptr && v[0] == '1';
    }();
    return env && available();
}

size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) (cols / 64) * AB; }
size_t group_bytes(int64_t n, int n_expert) {
    return tiles_at(n_expert) + ((size_t) (n + kTileRows - 1) / kTileRows + (size_t) n_expert) * sizeof(int2);
}

void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
    quant_act_kernel<<<blocks(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_act");
}

void group(const int32_t* ids, int64_t n, int k, int n_expert, void* scratch, int32_t* slot, int32_t* src,
           void* stream) {
    if (n_expert < 1 || n_expert > 1024) {
        std::fprintf(stderr, "prefill fused experts: %d experts (the grouping takes 1 to 1024)\n", n_expert);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(scratch, n_expert);
    ck(cudaMemsetAsync(tb.cnt, 0, (size_t) n_expert * 4, s), "group memset");
    if (n > 0) {
        const unsigned nb = std::min<unsigned>(blocks(n, 1024), 4u * (unsigned) std::max(dev_info().sms, 1));
        count_kernel<<<nb, 1024, (size_t) n_expert * 4, s>>>(ids, n, n_expert, tb.cnt);
    }
    scan_kernel<<<1, (unsigned) ((n_expert + 31) / 32 * 32), 0, s>>>(tb, n_expert);
    if (n > 0) place_kernel<<<blocks(n, 256), 256, 0, s>>>(ids, n, k, n_expert, tb, slot, src);
    ck(cudaGetLastError(), "group");
}

void experts(const Batch& b, int n_expert, int64_t n, const void* scratch, const void* xa, const int32_t* src,
             void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    // the most tiles the batch can have: every row in it, plus a partial tile per expert
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
#if defined(__HIPCC__)
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / W_ROWS), (int64_t) d.sms * wgp_blocks(d.occ_gu));
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / W_ROWS), (int64_t) d.sms * wgp_blocks(d.occ_d));
    expert_w11_kernel<true><<<g_gu, W_THREADS, 0, s>>>(b, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr);
    expert_w11_kernel<false><<<g_d, W_THREADS, 0, s>>>(b, tb, (const uint8_t*) ha, src, nullptr, dm);
#else
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * 5, (int64_t) d.sms * d.occ_gu);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * 10, (int64_t) d.sms * d.occ_d);
    expert_kernel<true><<<g_gu, THREADS, smem_bytes(true), s>>>(b, tb, (const uint8_t*) xa, src, (uint8_t*) ha,
                                                                nullptr);
    expert_kernel<false><<<g_d, THREADS, smem_bytes(false), s>>>(b, tb, (const uint8_t*) ha, src, nullptr, dm);
#endif
    ck(cudaGetLastError(), "experts");
}

}  // namespace strata::prefill::fused
