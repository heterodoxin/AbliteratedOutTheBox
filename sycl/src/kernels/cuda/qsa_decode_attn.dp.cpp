// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include "strata/core/geometry.hpp"
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv, "qsa_decode_attn is compiled for core/geometry.hpp");
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

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
__dpct_inline__ float warp_max(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(
        v, dpct::experimental::permute_sub_group_by_xor(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
// Per-format bodies; `load8` below is the KV_MODE dispatcher. value=false is the K side, true the V side.
__dpct_inline__ void load8_f16(const QsaAttnPools &p, bool value, long long row,
                               int d0, float *out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const sycl::uint4 raw = *reinterpret_cast<const sycl::uint4 *>(base);
    const sycl::half2 *h2 = reinterpret_cast<const sycl::half2 *>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const sycl::float2 f =
            h2[j].template convert<float, sycl::rounding_mode::automatic>();
        out[2 * j] = f.x();
        out[2 * j + 1] = f.y();
    }
}
__dpct_inline__ void load8_q8(const QsaAttnPools &p, bool value, long long row,
                              int d0, float *out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = sycl::vec<sycl::half, 1>(
                         sycl::bit_cast<sycl::half, unsigned short>(sbits))
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    const sycl::uint2 raw = *reinterpret_cast<const sycl::uint2 *>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__dpct_inline__ void load8_q4(const QsaAttnPools &p, bool value, long long row,
                              int d0, float *out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = sycl::vec<sycl::half, 1>(
                        sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
    }
}
template <int KV_MODE>
__dpct_inline__ void load8(const QsaAttnPools &p, bool value, long long row,
                           int d0, float *out) {
    if constexpr (KV_MODE == 0) load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1) load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        // hybrid K8V4: both sides are defined - K unrotated INT8, V rotated Q4_0 - so a value=true call
        // reads the Q4_0 pool instead of dereferencing the null v_q (no call site does today; PR review)
        if (value) load8_q4(p, value, row, d0, out);
        else load8_q8(p, value, row, d0, out);
    } else load8_q4(p, value, row, d0, out);
}

template <int KV_MODE, bool LANE_CELL = false, int RS = 0>
/*
DPCT1110: The total declared local variable size in device function
attn_chunk_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attn_chunk_kernel(const float *__restrict__ q, QsaAttnPools p,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ step, int n_kv_heads,
                  int page_size, float scale, float *__restrict__ part_acc,
                  float *__restrict__ part_m, float *__restrict__ part_l,
                  int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    q += (size_t)item_ct1.get_group(0) * (size_t)(n_kv_heads * G) * HD;
    ids += (size_t)item_ct1.get_group(0) * (size_t)cap;
    step += (size_t)item_ct1.get_group(0) * kStepCount;
    part_acc += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][HD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // 12 KB: this KV head's query heads
    auto &sp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G * CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // scores, then probabilities
    auto &srow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // pool row of each cell (page, kv head, slot)
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    // the scores and probabilities: [head][cell], or - RS - [cell][head] so the value phase reads a cell's 12 probabilities as 3 vectors
    const auto spi = [](int h, int c) { return RS != 0 ? c * G + h : h * CHUNK + c; };
    const int n_ids = *(step + kStepWidth);
    const int chunk = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = sycl::min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        sq[i / HD][i % HD] = q[(size_t)(kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if constexpr (LANE_CELL) {
        // S25 (STRATA_ATTN_LANECELL=1): thread t scores cell t % 64 for heads 3 (t / 64) .. +2 by itself - the same
        // 8-dimension partial per former lane (the identical expression), then the former warp_sum's butterfly as
        // lane 0 ran it (a[l] += a[l + o] for o = 16 .. 1), so every score is bit-identical without 12 shuffle
        // reductions per cell
        constexpr int HPT = G * CHUNK / THREADS;   // heads per thread (3)
        const int c = t % CHUNK, h0 = (t / CHUNK) * HPT;
        if (c >= n_here || srow[c] < 0) {
#pragma unroll
            for (int j = 0; j < HPT; ++j) sp[spi(h0 + j, c)] = -FLT_MAX;
        } else {
            float a[HPT][32];
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], l * 8, k8);
#pragma unroll
                for (int j = 0; j < HPT; ++j) {
                    const sycl::float4 qa =
                        *reinterpret_cast<const sycl::float4 *>(
                            &sq[h0 + j][l * 8]);
                    const sycl::float4 qb =
                        *reinterpret_cast<const sycl::float4 *>(
                            &sq[h0 + j][l * 8 + 4]);
                    float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                              k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                              k8[6] * qb.z() + k8[7] * qb.w();
                    a[j][l] = s;
                }
            }
#pragma unroll
            for (int j = 0; j < HPT; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                    for (int l = 0; l < o; ++l) a[j][l] = a[j][l] + a[j][l + o];
                sp[spi(h0 + j, c)] = a[j][0] * scale;
            }
        }
    } else
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[spi(lane, c)] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
        if constexpr (RS == 1) {
            // 151-b70: the 12 heads' 32-lane butterfly sums as ONE reduce-scatter - 16 shuffles instead of 60.  Each stage pairs the
            // same lanes in the same order as warp_sum (xor 16, 8, 4, 2, 1) and adds own + partner, so every head's sum is the
            // same tree and the same bits; the heads are padded to 16, halved at each stage (the lane keeps the half its bit
            // selects, sends the other), and after the xor-2 stage lane 2h + {0, 1} holds head h's partial pair; xor 1 finishes it.
            float part[16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const sycl::float4 qa =
                    *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8]);
                const sycl::float4 qb =
                    *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8 + 4]);
                part[h] = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                          k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                          k8[6] * qb.z() + k8[7] * qb.w();
            }
#pragma unroll
            for (int h = G; h < 16; ++h) part[h] = 0.0f;
            const auto xr = [](float v, int o) {
                return dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
            };
            float c8[8];
            const bool b16 = (lane & 16) != 0, b8 = (lane & 8) != 0, b4 = (lane & 4) != 0, b2 = (lane & 2) != 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float send = b16 ? part[i] : part[8 + i];
                const float keep = b16 ? part[8 + i] : part[i];
                c8[i] = keep + xr(send, 16);
            }
            float c4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float send = b8 ? c8[i] : c8[4 + i];
                const float keep = b8 ? c8[4 + i] : c8[i];
                c4[i] = keep + xr(send, 8);
            }
            float c2[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const float send = b4 ? c4[i] : c4[2 + i];
                const float keep = b4 ? c4[2 + i] : c4[i];
                c2[i] = keep + xr(send, 4);
            }
            const float send1 = b2 ? c2[0] : c2[1];
            const float keep1 = b2 ? c2[1] : c2[0];
            float s = keep1 + xr(send1, 2);
            s = s + xr(s, 1);
            const int head = (lane >> 1) & 15;
            if ((lane & 1) == 0 && head < G) sp[spi(head, c)] = s * scale;
        } else {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 qa =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8]);
            const sycl::float4 qb =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                      k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                      k8[6] * qb.z() + k8[7] * qb.w();
            s = warp_sum(s);
            if (lane == 0) sp[spi(h, c)] = s * scale;
        }
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[spi(h, lane)], b = sp[spi(h, lane + 32)];
        const float m = warp_max(sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0)
                             ? sycl::native::exp(a - m)
                             : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0)
                             ? sycl::native::exp(b - m)
                             : 0.0f;
        sp[spi(h, lane)] = ea;
        sp[spi(h, lane + 32)] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[srow[c] * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[srow[c] * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
        if constexpr (RS != 0) {
            // the cell's 12 probabilities as three vector loads (the layout is [cell][head]); the same fma per head in the same order
#pragma unroll
            for (int q4 = 0; q4 < G / 4; ++q4) {
                const sycl::float4 pv = *reinterpret_cast<const sycl::float4 *>(&sp[c * G + 4 * q4]);
                acc[4 * q4 + 0] = sycl::fma(pv.x(), v, acc[4 * q4 + 0]);
                acc[4 * q4 + 1] = sycl::fma(pv.y(), v, acc[4 * q4 + 1]);
                acc[4 * q4 + 2] = sycl::fma(pv.z(), v, acc[4 * q4 + 2]);
                acc[4 * q4 + 3] = sycl::fma(pv.w(), v, acc[4 * q4 + 3]);
            }
        } else {
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[spi(h, c)], v, acc[h]);
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// PR #540 (sskver): the same attention with fewer shuffles and less shared-memory traffic, bit for bit the kernel
// above (same operands, same order); +7% prompt speed on a V100, where this kernel reads every prompt chunk.  It
// uses more registers (80 vs 38 on sm_70), so it runs on cards below sm_75 only, and exists only in the experimental
// build (-DSTRATA_EXPERIMENTAL_SM60=ON): the ready-made engine keeps exactly the kernel above.
#if defined(STRATA_EXPERIMENTAL_SM60)
// The 12 heads' lane sums (`part[12..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
// every output adds the same two operands at every level, so each sum is bit for bit warp_sum's (float add commutes),
// for 16 shuffles instead of 12 x 5.  Lane l ends holding head head_of_lane(l); lanes l and l^1 agree.
__device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own thread index), per KV format.
template <int KV_MODE>
__device__ __forceinline__ float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return __half2float(__ushort_as_half(p.v_pool[row * HD + d]));
    } else if constexpr (KV_MODE == 1) {
        const float sc = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
        return (float) p.v_q[row * HD + d] * sc;
    } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = __half2float(__ushort_as_half(blk->d));
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nibble * dd;
    }
}

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel_pre75(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads (dead once the scores are done)
    __shared__ float sp[G][CHUNK];                // scores
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    // probabilities, cell-major: the value loop reads one cell's 12 as three float4.  They are first written after the
    // __syncthreads that ends the score phase, the last reader of sq, so they share its storage (64 x 12 of its 256 x 12
    // floats) and the block's shared memory stays what it was before the cell-major layout.
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sq[0][0]);
    static_assert(CHUNK * G <= G * HD, "spt must fit in sq");
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.  NC cells per step
    // (warp + 8 * (i + j)): a lane's q slice comes out of shared memory once for all of them, which cuts the traffic
    // that bounds this phase (24 x 16 B per lane per cell, against the FMAs) by NC; every dot product is the same
    // expression as for a single cell, so the scores are unchanged.
    constexpr int NC = 2;   // 4 measured slower on a V100 (the registers cost occupancy)
    const int hd = head_of_lane(lane);
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) load8<KV_MODE>(p, false, srow[cc[j]], lane * 8, kk[j]);
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < NC; ++k) {   // a masked or out-of-range cell in the group: one cell at a time
                const int c = cc[k];
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    // One cell's 12 weights (three broadcast float4 loads) into the 12 accumulators; the cells are folded in
    // ascending order, so the sums are the plain loop's, bit for bit.
    const auto fold = [&](int c, float v) {
        const float4* w4 = reinterpret_cast<const float4*>(spt[c]);
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        acc[0] = fmaf(w0.x, v, acc[0]);  acc[1] = fmaf(w0.y, v, acc[1]);
        acc[2] = fmaf(w0.z, v, acc[2]);  acc[3] = fmaf(w0.w, v, acc[3]);
        acc[4] = fmaf(w1.x, v, acc[4]);  acc[5] = fmaf(w1.y, v, acc[5]);
        acc[6] = fmaf(w1.z, v, acc[6]);  acc[7] = fmaf(w1.w, v, acc[7]);
        acc[8] = fmaf(w2.x, v, acc[8]);  acc[9] = fmaf(w2.y, v, acc[9]);
        acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
    };
    int c = 0;
    // four cells at a time with their V loads issued together (the loop was one dependent load chain per cell);
    // a group holding a masked cell (page -1, rare) leaves this loop and the scalar one below finishes the chunk.
    for (; c + 4 <= n_here; c += 4) {
        const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) break;
        const float v0 = load_v1<KV_MODE>(p, r0, t), v1 = load_v1<KV_MODE>(p, r1, t);
        const float v2 = load_v1<KV_MODE>(p, r2, t), v3 = load_v1<KV_MODE>(p, r3, t);
        fold(c, v0);
        fold(c + 1, v1);
        fold(c + 2, v2);
        fold(c + 3, v3);
    }
    for (; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        fold(c, load_v1<KV_MODE>(p, srow[c], t));
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}
#endif  // STRATA_EXPERIMENTAL_SM60

__dpct_inline__ void attn_merge_kernel(const float *__restrict__ part_acc,
                                       const float *__restrict__ part_m,
                                       const float *__restrict__ part_l,
                                       int n_chunks, float *__restrict__ attn,
                                       long long scratch_stride = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    part_acc += (size_t)item_ct1.get_group(1) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(1) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(1) * (size_t)scratch_stride;
    attn += (size_t)item_ct1.get_group(1) *
            (size_t)item_ct1.get_group_range(2) * HD;
    const int h = item_ct1.get_group(2); // global query head
    const int kvh = h / G, hl = h % G;
    const int d = item_ct1.get_local_id(2);
    float M = -FLT_MAX;
#pragma unroll
    for (int c = 0; c < n_chunks; ++c)
        M = sycl::fmax(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = sycl::native::exp(m - M);
        L = sycl::fma((float)(part_l[slot * G + hl]), (float)w, L);
        acc = sycl::fma((float)(part_acc[((size_t)slot * G + hl) * HD + d]),
                        (float)w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

#if defined(STRATA_EXPERIMENTAL_SM60)
// the current device is below sm_75 (per device: a layer split can mix cards); STRATA_ATTN_PRE75=0 turns PR #540's
// kernel off (A/B)
bool pre75_attn() {
    static const bool off = [] {
        const char* e = std::getenv("STRATA_ATTN_PRE75");
        return e != nullptr && e[0] == '0';
    }();
    static int cc[64] = {};
    int dev = 0;
    if (off) return false;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = 10 * major + minor;
    }
    return cc[dev] < 75;
}
#define STRATA_ATTN_CHUNK(M) (pre75_attn() ? attn_chunk_kernel_pre75<M> : attn_chunk_kernel<M>)
#else
#define STRATA_ATTN_CHUNK(M) attn_chunk_kernel<M>
#endif

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dpct::dim3 grid((unsigned)n_chunks, (unsigned)s.n_head_kv,
                          (unsigned)n_q);
    dpct::queue_ptr st = strata::q_of(stream);
    // S25: STRATA_ATTN_LANECELL=1 - the score phase one cell per thread (bit-identical scores, no shuffle reductions)
    static const bool lane_cell = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
#define STRATA_ATTN_LC(M)                                                      \
    {                                                                          \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});  \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto q_ct0 = q;                                                    \
            QsaAttnPools pools_ct1 = pools;                                    \
            auto ids_ct2 = ids;                                                \
            auto steps_ct3 = steps;                                            \
            auto s_n_head_kv_ct4 = (int)s.n_head_kv;                           \
            auto s_page_size_ct5 = (int)s.page_size;                           \
            auto scale_ct6 = scale;                                            \
            auto part_acc_ct7 = part_acc;                                      \
            auto part_m_ct8 = part_m;                                          \
            auto part_l_ct9 = part_l;                                          \
            auto n_chunks_ct10 = n_chunks;                                     \
            auto cap_ct11 = (int)cap;                                          \
            auto stride_ct12 = stride;                                         \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<class attn_chunk_kernel_6e87d5,  \
                                              dpct_kernel_scalar<M>,           \
                                              dpct_kernel_scalar<true>>>(      \
                sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),           \
                                  sycl::range(1, 1, THREADS)),                 \
                exp_props,                                                     \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        attn_chunk_kernel<M, true>(                            \
                            q_ct0, pools_ct1, ids_ct2, steps_ct3,              \
                            s_n_head_kv_ct4, s_page_size_ct5, scale_ct6,       \
                            part_acc_ct7, part_m_ct8, part_l_ct9,              \
                            n_chunks_ct10, cap_ct11, stride_ct12);             \
                    });                                                        \
        });                                                                    \
    }
#define STRATA_ATTN_RSK(M, V)                                                      \
    {                                                                          \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});  \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto q_ct0 = q;                                                    \
            QsaAttnPools pools_ct1 = pools;                                    \
            auto ids_ct2 = ids;                                                \
            auto steps_ct3 = steps;                                            \
            auto s_n_head_kv_ct4 = (int)s.n_head_kv;                           \
            auto s_page_size_ct5 = (int)s.page_size;                           \
            auto scale_ct6 = scale;                                            \
            auto part_acc_ct7 = part_acc;                                      \
            auto part_m_ct8 = part_m;                                          \
            auto part_l_ct9 = part_l;                                          \
            auto n_chunks_ct10 = n_chunks;                                     \
            auto cap_ct11 = (int)cap;                                          \
            auto stride_ct12 = stride;                                         \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<class attn_chunk_kernel_rs_7c2a91,  \
                                              dpct_kernel_scalar<M>,           \
                                              dpct_kernel_scalar<V>>>(      \
                sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),           \
                                  sycl::range(1, 1, THREADS)),                 \
                exp_props,                                                     \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        attn_chunk_kernel<M, false, V>(                            \
                            q_ct0, pools_ct1, ids_ct2, steps_ct3,              \
                            s_n_head_kv_ct4, s_page_size_ct5, scale_ct6,       \
                            part_acc_ct7, part_m_ct8, part_l_ct9,              \
                            n_chunks_ct10, cap_ct11, stride_ct12);             \
                    });                                                        \
        });                                                                    \
    }
#undef STRATA_ATTN_RSK_DEFINED
    // 151-b70: STRATA_ATTN_RS=2 (default): the probabilities in [cell][head] order and the value phase reading a cell's 12 as three vectors - the same
    // bits (first-token logits md5 equal at 4K and 16K), +0.8..0.9% prompt tok/s on the B70; =1 adds a 16-shuffle reduce-scatter for the scores
    // (bit-exact too, but -1.5% here: the score phase is not shuffle bound); =0 the layout before
    static const int rs_on = [] { const char* v = std::getenv("STRATA_ATTN_RS"); return v ? std::atoi(v) : 2; }();
    if (lane_cell) {
        if (kv_mode == 3) { STRATA_ATTN_LC(3); }
        else if (kv_mode == 2) { STRATA_ATTN_LC(2); }
        else if (kv_mode == 1) { STRATA_ATTN_LC(1); }
        else { STRATA_ATTN_LC(0); }
    } else if (rs_on == 1) {
        if (kv_mode == 3) { STRATA_ATTN_RSK(3, 1); }
        else if (kv_mode == 2) { STRATA_ATTN_RSK(2, 1); }
        else if (kv_mode == 1) { STRATA_ATTN_RSK(1, 1); }
        else { STRATA_ATTN_RSK(0, 1); }
    } else if (rs_on == 2) {
        if (kv_mode == 3) { STRATA_ATTN_RSK(3, 2); }
        else if (kv_mode == 2) { STRATA_ATTN_RSK(2, 2); }
        else if (kv_mode == 1) { STRATA_ATTN_RSK(1, 2); }
        else { STRATA_ATTN_RSK(0, 2); }
    } else
#undef STRATA_ATTN_LC
#undef STRATA_ATTN_RSK
    if (kv_mode == 3)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_dfacfc,
                                          dpct_kernel_scalar<3>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<3>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    } else if (kv_mode == 2)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_b0d448,
                                          dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<2>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    } else if (kv_mode == 1)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_2ad0a4,
                                          dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<1>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_5c04f7,
                                          dpct_kernel_scalar<0>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<0>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class attn_merge_kernel_338175>>(
            sycl::nd_range<3>(
                sycl::range(1, (unsigned)n_q, (unsigned)s.n_head) *
                    sycl::range(1, 1, HD),
                sycl::range(1, 1, HD)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_merge_kernel(part_acc, part_m, part_l, n_chunks, attn,
                                  stride);
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dpct::dim3 grid((unsigned)n_chunks, (unsigned)s.n_head_kv);
    dpct::queue_ptr st = strata::q_of(stream);
    if (kv_mode == 3)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_f262e6,
                                          dpct_kernel_scalar<3>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<3>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    } else if (kv_mode == 2)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_4fd923,
                                          dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<2>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    } else if (kv_mode == 1)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_f6855e,
                                          dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<1>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_7c8e09,
                                          dpct_kernel_scalar<0>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<0>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class attn_merge_kernel_c055e7>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)s.n_head) *
                                  sycl::range(1, 1, HD),
                              sycl::range(1, 1, HD)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_merge_kernel(part_acc, part_m, part_l, n_chunks, attn, 0);
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

}  // namespace strata::kernels
