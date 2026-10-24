// src/kernels/cuda/qsa_prompt_attn.cu - see include/strata/kernels/qsa_prompt_attn.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include "strata/core/geometry.hpp"
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/emulate.hpp"
#include <cstdlib>
#include <cstring>
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_prompt_attn_xmx.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace strata::kernels {
namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv, "qsa_prompt_attn is compiled for core/geometry.hpp");
#ifndef D1_CH
#define D1_CH 32
#endif
constexpr int CH = D1_CH;         // cells per chunk
constexpr int THREADS = 128;      // 4 warps: scores by cell (8 each), p.v by dimension (64 each = one int8 scale group)
constexpr int QS = HD + 8;        // q row stride in halves (bank-conflict-free fragment loads)

// The MMA below needs sm_75 or newer (Turing runs it as two k=8 steps); cp.async needs sm_80. Builds for pre-sm_75
// cards compile the MMA to a trap; qsa_prompt_attn_batch refuses such a device at run time, so the old kernel runs
// there.  Turing compiles cp_async16 to a trap as well and takes the v1 kernel instead of launch_i8.
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the old kernel (below)
#define STRATA_PA_SM80 0
#elif 0   // SYCL: inline PTX (mma/ldmatrix/cp.async) - the XMX port is pending; see tools/fixups.py
#define STRATA_PA_SM80 1
#else
#define STRATA_PA_SM80 0
#endif

// m16n8k16 with f16 inputs needs sm_80.  Turing (sm_75) has m16n8k8 with the SAME A/B/C register mapping, so the
// k=16 step is two k=8 steps on the fragments as they are already laid out: a[0]/a[1] are rows gid/gid+8 at k columns
// 2*tig..2*tig+1 (b[0]'s k rows), a[2]/a[3] the same rows at k columns 2*tig+8..2*tig+9 (b[1]'s k rows).  The
// products then add into the same FP32 C registers in the order hi-part-0, hi-part-1, which is the order the k16
// instruction accumulates in as well - but the sum now rounds twice, so the two paths do not agree bit for bit.
__dpct_inline__ void mma16816(float *c, const uint32_t *a, const uint32_t *b) {
#if !STRATA_PA_SM80 &&                                                         \
    (defined(__HIPCC__) || !defined(DPCT_COMPATIBILITY_TEMP) ||                \
     DPCT_COMPATIBILITY_TEMP < 750)
    /* unreachable on SYCL: the launcher refuses this device */   // AMD and pre-Turing builds: no mma.sync (the host keeps the old kernel there)
#elif 0   // SYCL: Turing's m16n8k8 PTX (unreachable: the launcher refuses this device)
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]));
#else
    {
        volatile void *d_mat_frag_ct1[4] = {&c[0], &c[1], &c[2], &c[3]};
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(a[0], a[1], a[2], a[3]);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(b[0], b[1]);
        sycl::vec<float, 4> c_mat_frag_ct1(c[0], c[1], c[2], c[3]);
        dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1,
            &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#endif
#endif
}

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152.
__dpct_inline__ uint32_t i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    sycl::half2 h = *reinterpret_cast<sycl::half2 *>(&y);
    h = h - sycl::half2(
                sycl::vec<float, 1>(1152.f)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
                sycl::vec<float, 1>(1152.f)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
    return *reinterpret_cast<uint32_t*>(&h);
}

__dpct_inline__ uint32_t pack_h2(float lo_k,
                                 float hi_k) { // element k in the low half
    sycl::half2 h = sycl::float2(lo_k, hi_k)
                        .convert<sycl::half, sycl::rounding_mode::rte>();
    return *reinterpret_cast<uint32_t*>(&h);
}

// KV_MODE 1: int8 codes + fp16 scale per 64 values. KV_MODE 0: fp16 values (scales 1).
// KV_MODE 3 (hybrid K8V4): K as mode 1, V as mode 0 - the row's q4_0 blocks are dequantized to fp16 at
// gather, so everything downstream of the load is the mode-0 V path; the caller un-rotates the output.
// KV_MODE 4 (Q4_0 K and V, `--kv q4_0`): mode 1 with a scale per 32 values - each q4_0 block's codes enter as exact
// int8 (code - 8) and its fp16 scale is applied in FP32, as mode 1's are; the caller rotates q and un-rotates the
// output (kv_q4.hpp).
template <int KV_MODE>
struct Smem {
    using KElem =
        typename std::conditional<KV_MODE == 0, sycl::half, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1 || KV_MODE == 4,
                                            int8_t, sycl::half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;   // elements; 16-byte aligned rows, banks spread
    static constexpr int VROW = (KV_MODE == 1 || KV_MODE == 4) ? HD + 16 : HD + 8;
    static constexpr int NG = KV_MODE == 4 ? HD / QK4_0 : 4;      // scale groups per row (q4_0: 8 of 32, else 4 of 64)
    sycl::half qh[16][QS];
    sycl::half ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][NG];
    float vs[CH][NG];
    float s[16][CH + 1];
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

template <int KV_MODE>
/*
DPCT1110: The total declared local variable size in device function
prompt_attn_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
prompt_attn_kernel(const float *__restrict__ q, QsaAttnPools p,
                   const int32_t *__restrict__ ids,
                   const int32_t *__restrict__ steps, int n_kv_heads,
                   int page_size, float scale_log2, float *__restrict__ attn,
                   int cap, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem_raw = (unsigned char *)dpct_local;
    Smem<KV_MODE>& S = *reinterpret_cast<Smem<KV_MODE>*>(smem_raw);
    const int qi = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n = *(steps + (size_t)qi * kStepCount + kStepWidth);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        qm = sycl::fmax(qm, sycl::fabs(q[i]));
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) qm = sycl::fmax(
        qm, dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    qm = sycl::fmax(sycl::fmax(S.qmax[0], S.qmax[1]),
                    sycl::fmax(S.qmax[2], S.qmax[3]));
    int qe = 0;
    /*
    DPCT1017: The sycl::frexp call is used instead of the frexpf call.
    These two calls do not provide exactly the same functionality. Check the
    potential precision and/or performance issues for the generated code.
    */
    if (qm > 0.0f) sycl::frexp(
        qm, sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::yes>(
                &qe)); // qm < 2^qe
    const float qup = sycl::ldexp(1.0f, 14 - qe),
                qdown = sycl::ldexp(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const sycl::half hi =
            sycl::vec<float, 1>(x)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
        S.qh[h][d] = hi;
        S.ql[h][d] =
            sycl::vec<float, 1>(
                x - sycl::vec<sycl::half, 1>(hi)
                        .convert<float, sycl::rounding_mode::automatic>()[0])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = sycl::min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
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
        item_ct1.barrier(); // rows ready; the previous chunk's p.v is done with
                            // k, v, s
        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        if constexpr (KV_MODE == 4) {
            // q4_0 K and V: one (K or V, cell, block) per item - the block's 16 code bytes (2-byte aligned: eight
            // 16-bit loads) become 32 exact int8 codes (element j in the low nibble of byte j, j + 16 in the high
            // one), its fp16 scale goes to ks/vs
            constexpr int BLKS = HD / QK4_0;
            constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
            for (int i = t; i < 2 * CH * BLKS; i += THREADS) {
                const int kv = i / (CH * BLKS), rem = i % (CH * BLKS), c = rem / BLKS, b = rem % BLKS;
                const long long r = S.row[c];
                sycl::uint4 lo = sycl::uint4(0, 0, 0, 0),
                            hi = sycl::uint4(0, 0, 0, 0);
                float d = 0.0f;
                if (r >= 0) {
                    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((kv == 0 ? p.k_q4 : p.v_q4) + r * BYTES) + b;
                    /*
                    DPCT1098: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    d = sycl::vec<sycl::half, 1>(
                            sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                            .convert<float,
                                     sycl::rounding_mode::automatic>()[0];
                    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(blk->qs);
                    uint32_t w[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        /*
                        DPCT1098: The '*' expression is used instead of the
                        __ldg call. These two expressions do not provide the
                        exact same functionality. Check the generated code for
                        potential precision and/or performance issues.
                        */
                        w[j] = (uint32_t)*(q16 + 2 * j) |
                               ((uint32_t)*(q16 + 2 * j + 1) << 16);
                    // per byte: the low nibble minus 8 is element j, the high one minus 8 element j + 16
                    lo = sycl::uint4(
                        dpct::vectorized_binary<sycl::uchar4>(
                            w[0] & 0x0F0F0F0Fu, 0x08080808u, std::minus<>()),
                        dpct::vectorized_binary<sycl::uchar4>(
                            w[1] & 0x0F0F0F0Fu, 0x08080808u, std::minus<>()),
                        dpct::vectorized_binary<sycl::uchar4>(
                            w[2] & 0x0F0F0F0Fu, 0x08080808u, std::minus<>()),
                        dpct::vectorized_binary<sycl::uchar4>(
                            w[3] & 0x0F0F0F0Fu, 0x08080808u, std::minus<>()));
                    hi = sycl::uint4(dpct::vectorized_binary<sycl::uchar4>(
                                         (w[0] >> 4) & 0x0F0F0F0Fu, 0x08080808u,
                                         std::minus<>()),
                                     dpct::vectorized_binary<sycl::uchar4>(
                                         (w[1] >> 4) & 0x0F0F0F0Fu, 0x08080808u,
                                         std::minus<>()),
                                     dpct::vectorized_binary<sycl::uchar4>(
                                         (w[2] >> 4) & 0x0F0F0F0Fu, 0x08080808u,
                                         std::minus<>()),
                                     dpct::vectorized_binary<sycl::uchar4>(
                                         (w[3] >> 4) & 0x0F0F0F0Fu, 0x08080808u,
                                         std::minus<>()));
                }
                int8_t* dst = kv == 0 ? &S.k[c][b * QK4_0] : &S.v[c][b * QK4_0];
                reinterpret_cast<sycl::uint4 *>(dst)[0] = lo;
                reinterpret_cast<sycl::uint4 *>(dst)[1] = hi;
                (kv == 0 ? S.ks : S.vs)[c][b] = d;
            }
        } else {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            for (int i = t; i < CH * KPIECES; i += THREADS) {
                const int c = i / KPIECES, pc = i % KPIECES;
                const long long r = S.row[c];
                sycl::uint4 kx = sycl::uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        /*
                        DPCT1098: The '*' expression is used instead of the
                        __ldg call. These two expressions do not provide the
                        exact same functionality. Check the generated code for
                        potential precision and/or performance issues.
                        */
                        kx = *(reinterpret_cast<const sycl::uint4 *>(p.k_pool +
                                                                     r * HD) +
                               pc);
                    else   // modes 1 and 3: the K side is INT8
                        /*
                        DPCT1098: The '*' expression is used instead of the
                        __ldg call. These two expressions do not provide the
                        exact same functionality. Check the generated code for
                        potential precision and/or performance issues.
                        */
                        kx = *(reinterpret_cast<const sycl::uint4 *>(p.k_q +
                                                                     r * HD) +
                               pc);
                }
                *reinterpret_cast<sycl::uint4 *>(
                    reinterpret_cast<unsigned char *>(&S.k[c][0]) + pc * 16) =
                    kx;
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j)
                        S.v[c][b * QK4_0 + j] = sycl::half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d =
                            sycl::vec<sycl::half, 1>(
                                sycl::bit_cast<sycl::half, unsigned short>(
                                    blk->d))
                                .convert<float,
                                         sycl::rounding_mode::automatic>()[0];
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.v[c][b * QK4_0 + j] =
                                sycl::vec<float, 1>(
                                    (float)((int)(blk->qs[j] & 0x0F) - 8) * d)
                                    .convert<sycl::half,
                                             sycl::rounding_mode::rte>()[0];
                            S.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                sycl::vec<float, 1>(
                                    (float)((int)(blk->qs[j] >> 4) - 8) * d)
                                    .convert<sycl::half,
                                             sycl::rounding_mode::rte>()[0];
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                for (int i = t; i < CH * VPIECES; i += THREADS) {
                    const int c = i / VPIECES, pc = i % VPIECES;
                    const long long r = S.row[c];
                    sycl::uint4 vx = sycl::uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            /*
                            DPCT1098: The '*' expression is used instead of
                            the __ldg call. These two expressions do not provide
                            the exact same functionality. Check the generated
                            code for potential precision and/or performance
                            issues.
                            */
                            vx = *(reinterpret_cast<const sycl::uint4 *>(
                                       p.v_q + r * HD) +
                                   pc);
                        else
                            /*
                            DPCT1098: The '*' expression is used instead of
                            the __ldg call. These two expressions do not provide
                            the exact same functionality. Check the generated
                            code for potential precision and/or performance
                            issues.
                            */
                            vx = *(reinterpret_cast<const sycl::uint4 *>(
                                       p.v_pool + r * HD) +
                                   pc);
                    }
                    *reinterpret_cast<sycl::uint4 *>(
                        reinterpret_cast<unsigned char *>(&S.v[c][0]) +
                        pc * 16) = vx;
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = sycl::vec<sycl::half, 1>(
                                sycl::bit_cast<sycl::half, unsigned short>(
                                    p.k_scale[r * (HD / KV_Q8_GROUP) + g]))
                                .convert<float,
                                         sycl::rounding_mode::automatic>()[0];
                        b = sycl::vec<sycl::half, 1>(
                                sycl::bit_cast<sycl::half, unsigned short>(
                                    p.v_scale[r * (HD / KV_Q8_GROUP) + g]))
                                .convert<float,
                                         sycl::rounding_mode::automatic>()[0];
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = sycl::vec<sycl::half, 1>(
                                sycl::bit_cast<sycl::half, unsigned short>(
                                    p.k_scale[r * (HD / KV_Q8_GROUP) + g]))
                                .convert<float,
                                         sycl::rounding_mode::automatic>()[0];
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
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
        // scores: warp w takes cells 8w..8w+7 (one n-tile) over all 256 dims, per scale group (64 dims; q4_0's 32)
        constexpr int NG = Smem<KV_MODE>::NG, KPG = HD / 16 / NG;   // groups per row, 16-dim MMA steps per group
#pragma unroll
        for (int nt = 0; nt < CH / 32; ++nt) {
            const int cb = (warp + 4 * nt) * 8;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk = 0; kk < KPG; ++kk) {
                    const int k0 = (g * KPG + kk) * 16;
                    uint32_t ah[4], al[4], b[2];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig + 8]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig + 8]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig + 8]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig + 8]);
                    if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes
                        b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig]));
                        b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]));
                    } else {
                        b[0] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig]);
                        b[1] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]);
                    }
                    mma16816(tg, ah, b);
#ifndef D1_NO_QLO
                    mma16816(tg, al, b);
#endif
                }
                const float s0 = S.ks[cb + 2 * tig][g], s1 = S.ks[cb + 2 * tig + 1][g];
                sc[0] = sycl::fma(tg[0], (float)s0, sc[0]);
                sc[1] = sycl::fma(tg[1], (float)s1, sc[1]);
                sc[2] = sycl::fma(tg[2], (float)s0, sc[2]);
                sc[3] = sycl::fma(tg[3], (float)s1, sc[3]);
            }
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
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
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = S.s[r][sub * PER + j]; mx = sycl::fmax(mx, x[j]);
            }
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 1; o < 8; o <<= 1) mx = sycl::fmax(
                mx,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), mx, o));
            const float m_old = S.mrow[r];
            const float m_new = sycl::fmax(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e =
                    x[j] == -INFINITY ? 0.0f : sycl::exp2(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 1; o < 8; o <<= 1) sum +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), sum, o);
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            if (sub == 0) {
                const float a =
                    m_old == -INFINITY ? 0.0f : sycl::exp2(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = sycl::fma(S.lsum[r], (float)a, sum);
                S.mrow[r] = m_new;
            }
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
        // p.v: warp w owns dims [64w, 64w+64), which is int8 scale group w (q4_0: groups 2w and 2w+1, four n-tiles
        // each). A group's scale is folded into p relative to the chunk's largest magnitude (q4_0's scales are signed:
        // ggml's d = max / -8), times 2^14 (|p'| <= 2^14: inside FP16's range, its lo half out of the subnormals); the
        // chunk's sum is then added to the running one in FP32 with the factor taken back out
        {
            constexpr int GPW = NG / 4, JPG = 8 / GPW;   // scale groups per warp, n-tiles per group
            float vdown_g[GPW];
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int gi = 0; gi < GPW; ++gi) {
            const int vg = warp * GPW + gi;
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32)
                vmax = sycl::fmax(vmax, sycl::fabs(S.vs[c][vg]));
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 16; o > 0; o >>= 1) vmax = sycl::fmax(
                vmax, dpct::experimental::permute_sub_group_by_xor(
                          0xffffffffu,
                          sycl::ext::oneapi::this_work_item::get_sub_group(),
                          vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f;
            vdown_g[gi] = vmax * (1.0f / 16384.0f);
#pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.vs[cA][vg] * vup, w1 = S.vs[cA + 1][vg] * vup, w2 = S.vs[cB][vg] * vup,
                            w3 = S.vs[cB + 1][vg] * vup;
                const float p00 = S.s[gid][cA] * w0, p01 = S.s[gid][cA + 1] * w1;
                const float p10 = S.s[gid + 8][cA] * w0, p11 = S.s[gid + 8][cA + 1] * w1;
                const float p02 = S.s[gid][cB] * w2, p03 = S.s[gid][cB + 1] * w3;
                const float p12 = S.s[gid + 8][cB] * w2, p13 = S.s[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const sycl::half2 *h =
                        reinterpret_cast<const sycl::half2 *>(ah);
                    sycl::float2 f;
                    f = h[0].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[0] = pack_h2(p00 - f.x(), p01 - f.y());
                    f = h[1].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[1] = pack_h2(p10 - f.x(), p11 - f.y());
                    f = h[2].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[2] = pack_h2(p02 - f.x(), p03 - f.y());
                    f = h[3].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[3] = pack_h2(p12 - f.x(), p13 - f.y());
                }
#pragma unroll
                for (int jj = 0; jj < JPG; ++jj) {
                    const int j = gi * JPG + jj;
                    const int d = warp * 64 + j * 8 + gid;
                    uint32_t b[2];
                    if constexpr (KV_MODE == 1 || KV_MODE == 4) {
                        const uint32_t x0 = (uint8_t) S.v[cA][d] | ((uint32_t) (uint8_t) S.v[cA + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[cB][d] | ((uint32_t) (uint8_t) S.v[cB + 1][d] << 8);
                        b[0] = i8x2_to_h2(x0);
                        b[1] = i8x2_to_h2(x1);
                    } else {
                        const sycl::half2 h0 =
                            sycl::half2(S.v[cA][d], S.v[cA + 1][d]);
                        const sycl::half2 h1 =
                            sycl::half2(S.v[cB][d], S.v[cB + 1][d]);
                        b[0] = *reinterpret_cast<const uint32_t*>(&h0);
                        b[1] = *reinterpret_cast<const uint32_t*>(&h1);
                    }
                    mma16816(tmp[j], ah, b);
#ifndef D1_NO_PLO
                    mma16816(tmp[j], al, b);
#endif
                }
            }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float vdown = vdown_g[j / JPG];
                acc[j][0] = sycl::fma(acc[j][0], (float)a0, tmp[j][0] * vdown);
                acc[j][1] = sycl::fma(acc[j][1], (float)a0, tmp[j][1] * vdown);
                acc[j][2] = sycl::fma(acc[j][2], (float)a1, tmp[j][2] * vdown);
                acc[j][3] = sycl::fma(acc[j][3], (float)a1, tmp[j][3] * vdown);
            }
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + 2 * tig;
        *reinterpret_cast<sycl::float2 *>(attn + (size_t)gid * HD + d) =
            sycl::float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<sycl::float2 *>(attn + (size_t)(gid + 8) * HD +
                                              d) =
                sycl::float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

// ---- v2 (int8 KV): warp w owns dims [64w, 64w+64) for both q.k and p.v, which is also int8 scale group w. So a
// warp needs only its own 64-byte slice of each K and V row: it gathers it itself with cp.async into its own
// double-buffered stage while it computes the previous chunk, and q stays in registers. Only the q.k partial sums
// (one per dim group) cross warps, and they are added in a fixed order: the result is deterministic.
constexpr int CH2 = 32;

struct Smem2 {
    int8_t kv[2][4][2][CH2][64];   // stage, warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[2][4][2][CH2];        // stage, warp, K/V scale of the cell for the warp's group
    float part[4][16][CH2 + 1];    // q.k per dim group
    float p[16][CH2 + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__dpct_inline__ int
swz(int cell, int byte) { // byte offset of (cell, byte) in a stage slice
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
__dpct_inline__ void cp_async16(void *smem, const void *gmem, bool valid) {
#if !STRATA_PA_SM80
    /* unreachable on SYCL: the launcher refuses this device */
#else
    auto sa = smem;
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa),
                 "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__dpct_inline__ void cp_async_commit() {
#if STRATA_PA_SM80
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.commit_group;\n" ::);
#else

#endif
#endif
}
__dpct_inline__ void cp_async_wait1() {
#if STRATA_PA_SM80
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.wait_group 1;\n" ::);
#else

#endif
#endif
}

/*
DPCT1110: The total declared local variable size in device function
prompt_attn_i8_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
prompt_attn_i8_kernel(const float *__restrict__ q, QsaAttnPools p,
                      const int32_t *__restrict__ ids,
                      const int32_t *__restrict__ steps, int n_kv_heads,
                      int page_size, float scale_log2, float *__restrict__ attn,
                      int cap, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem_raw = (unsigned char *)dpct_local;
    Smem2& S = *reinterpret_cast<Smem2*>(smem_raw);
    const int qi = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n = *(steps + (size_t)qi * kStepCount + kStepWidth);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;

    // q: the power-of-two prescale over all 12 heads (as v1), then this warp's 64 dims as hi/lo A fragments
    float qm = 0.0f;
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        qm = sycl::fmax(qm, sycl::fabs(q[i]));
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) qm = sycl::fmax(
        qm, dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    qm = sycl::fmax(sycl::fmax(S.qmax[0], S.qmax[1]),
                    sycl::fmax(S.qmax[2], S.qmax[3]));
    int qe = 0;
    /*
    DPCT1017: The sycl::frexp call is used instead of the frexpf call.
    These two calls do not provide exactly the same functionality. Check the
    potential precision and/or performance issues for the generated code.
    */
    if (qm > 0.0f) sycl::frexp(
        qm, sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::yes>(&qe));
    const float qup = sycl::ldexp(1.0f, 14 - qe),
                qdown = sycl::ldexp(scale_log2, qe - 14);
    uint32_t qh[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            sycl::float2 x = sycl::float2(0.f, 0.f);
            if (row < G) x = *reinterpret_cast<const sycl::float2 *>(
                q + (size_t)row * HD + col);
            x.x() *= qup;
            x.y() *= qup;
            const sycl::half2 hi =
                sycl::float2(x.x(), x.y())
                    .convert<sycl::half, sycl::rounding_mode::rte>();
            const sycl::float2 hf =
                hi.template convert<float, sycl::rounding_mode::automatic>();
            const sycl::half2 lo =
                sycl::float2(x.x() - hf.x(), x.y() - hf.y())
                    .convert<sycl::half, sycl::rounding_mode::rte>();
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    }

    // the chunk pipeline: cells two chunks ahead, their pool rows one chunk ahead, the data (cp.async) one ahead
    const int n_chunks = (n + CH2 - 1) / CH2;
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    auto cell_of = [&](int c) -> int {
                                       return c < n ? *(ids + c) : -1;
    };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const long long page = (long long)*(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    auto issue = [&](long long r, int st, float& ksr, float& vsr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            /*
            DPCT1108: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            /*
            DPCT1121: Make sure that the "r" which is used in the SYCL group
            function/algorithm is initialized.
            */
            const long long rr = dpct::experimental::select_from_sub_group(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
            cp_async16(&S.kv[st][warp][0][0][0] + swz(cell, pc * 16), p.k_q + off, ok);
            cp_async16(&S.kv[st][warp][1][0][0] + swz(cell, pc * 16), p.v_q + off, ok);
        }
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        ksr = r >= 0 ? sycl::vec<sycl::half, 1>(
                           sycl::bit_cast<sycl::half, unsigned short>(
                               *(p.k_scale + r * (HD / KV_Q8_GROUP) + warp)))
                           .convert<float, sycl::rounding_mode::automatic>()[0]
                     : 0.0f;
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        vsr = r >= 0 ? sycl::vec<sycl::half, 1>(
                           sycl::bit_cast<sycl::half, unsigned short>(
                               *(p.v_scale + r * (HD / KV_Q8_GROUP) + warp)))
                           .convert<float, sycl::rounding_mode::automatic>()[0]
                     : 0.0f;
    };
    float ksn, vsn;
    issue(row_of(cell_of(lane)), 0, ksn, vsn);
    cp_async_commit();
    S.sc[0][warp][0][lane] = ksn;
    S.sc[0][warp][1][lane] = vsn;
    long long r_next = row_of(cell_of(CH2 + lane));
    int cell_next2 = cell_of(2 * CH2 + lane);

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int st = ci & 1, c0 = ci * CH2;
        const bool more = ci + 1 < n_chunks;
        if (more) issue(r_next, st ^ 1, ksn, vsn);
        cp_async_commit();
        r_next = row_of(cell_next2);
        cell_next2 = cell_of((ci + 3) * CH2 + lane);
        cp_async_wait1();
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const int8_t* K = &S.kv[st][warp][0][0][0];
        const int8_t* V = &S.kv[st][warp][1][0][0];
        // q.k over this warp's 64 dims, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b[2];
                b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig)));
                b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig + 8)));
                mma16816(tg, qh[kk], b);
                mma16816(tg, ql[kk], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[st][warp][0][c], s1 = S.sc[st][warp][0][c + 1];
            S.part[warp][gid][c] = tg[0] * s0;
            S.part[warp][gid][c + 1] = tg[1] * s1;
            S.part[warp][gid + 8][c] = tg[2] * s0;
            S.part[warp][gid + 8][c + 1] = tg[3] * s1;
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
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = sycl::fmax(mx, x[j]);
            }
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 1; o < 8; o <<= 1) mx = sycl::fmax(
                mx,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), mx, o));
            const float m_old = S.mrow[r];
            const float m_new = sycl::fmax(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e =
                    x[j] == -INFINITY ? 0.0f : sycl::exp2(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 1; o < 8; o <<= 1) sum +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), sum, o);
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            if (sub == 0) {
                const float a =
                    m_old == -INFINITY ? 0.0f : sycl::exp2(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = sycl::fma(S.lsum[r], (float)a, sum);
                S.mrow[r] = m_new;
            }
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
        // p.v over this warp's 64 dims (as v1)
        {
            float vmax = S.sc[st][warp][1][lane];
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 16; o > 0; o >>= 1) vmax = sycl::fmax(
                vmax, dpct::experimental::permute_sub_group_by_xor(
                          0xffffffffu,
                          sycl::ext::oneapi::this_work_item::get_sub_group(),
                          vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH2 / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                            w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const sycl::half2 *h =
                        reinterpret_cast<const sycl::half2 *>(ah);
                    sycl::float2 f;
                    f = h[0].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[0] = pack_h2(p00 - f.x(), p01 - f.y());
                    f = h[1].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[1] = pack_h2(p10 - f.x(), p11 - f.y());
                    f = h[2].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[2] = pack_h2(p02 - f.x(), p03 - f.y());
                    f = h[3].template convert<float,
                                              sycl::rounding_mode::automatic>();
                        al[3] = pack_h2(p12 - f.x(), p13 - f.y());
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = j * 8 + gid;
                    const uint32_t x0 = (uint8_t) V[swz(cA, d)] | ((uint32_t) (uint8_t) V[swz(cA + 1, d)] << 8);
                    const uint32_t x1 = (uint8_t) V[swz(cB, d)] | ((uint32_t) (uint8_t) V[swz(cB + 1, d)] << 8);
                    uint32_t b[2];
                    b[0] = i8x2_to_h2(x0);
                    b[1] = i8x2_to_h2(x1);
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = sycl::fma(acc[j][0], (float)a0, tmp[j][0] * vdown);
                acc[j][1] = sycl::fma(acc[j][1], (float)a0, tmp[j][1] * vdown);
                acc[j][2] = sycl::fma(acc[j][2], (float)a1, tmp[j][2] * vdown);
                acc[j][3] = sycl::fma(acc[j][3], (float)a1, tmp[j][3] * vdown);
            }
        }
        if (more) {
            S.sc[st ^ 1][warp][0][lane] = ksn;
            S.sc[st ^ 1][warp][1][lane] = vsn;
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<sycl::float2 *>(attn + (size_t)gid * HD + d) =
            sycl::float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<sycl::float2 *>(attn + (size_t)(gid + 8) * HD +
                                              d) =
                sycl::float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

bool launch_i8(const float *q, const QsaAttnPools &pools, const int32_t *ids,
               const int32_t *steps, int64_t cap, const QsaShapes &s,
               float *attn, int64_t n_q, dpct::queue_ptr st) try {
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs this on several)
    int dev = 0;
    dev = dpct::get_current_device_id();
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const int bytes = (int)sizeof(Smem2);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        /*
        DPCT1027: The call to cudaFuncSetAttribute was replaced with 0
        because SYCL currently does not support corresponding setting.
        */
        if (0 != 0) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(st->get_device(),
                                         {sycl::aspect::fp16});

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(bytes), cgh);

                auto q_q0_s_n_head_HD_ct0 = q + q0 * s.n_head * HD;
                auto ids_q0_cap_ct2 = ids + q0 * cap;
                auto steps_q0_kStepCount_ct3 = steps + q0 * kStepCount;
                auto attn_q0_s_n_head_HD_ct7 = attn + q0 * s.n_head * HD;

                cgh.parallel_for<
                    dpct_kernel_name<class prompt_attn_i8_kernel_c3d463>>(
                    sycl::nd_range<3>(
                        sycl::range(1, (unsigned)s.n_head_kv, (unsigned)nb) *
                            sycl::range(1, 1, THREADS),
                        sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        prompt_attn_i8_kernel(
                            q_q0_s_n_head_HD_ct0, pools, ids_q0_cap_ct2,
                            steps_q0_kStepCount_ct3, (int)s.n_head_kv,
                            (int)s.page_size, scale_log2,
                            attn_q0_s_n_head_HD_ct7, (int)cap,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <int KV_MODE>
bool launch(const float *q, const QsaAttnPools &pools, const int32_t *ids,
            const int32_t *steps, int64_t cap, const QsaShapes &s, float *attn,
            int64_t n_q, dpct::queue_ptr st) try {
    static bool attr[64] = {};   // per device, as above
    int dev = 0;
    dev = dpct::get_current_device_id();
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const int bytes = (int)sizeof(Smem<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        /*
        DPCT1027: The call to cudaFuncSetAttribute was replaced with 0
        because SYCL currently does not support corresponding setting.
        */
        if (0 != 0) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(st->get_device(),
                                         {sycl::aspect::fp16});

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(bytes), cgh);

                auto q_q0_s_n_head_HD_ct0 = q + q0 * s.n_head * HD;
                auto ids_q0_cap_ct2 = ids + q0 * cap;
                auto steps_q0_kStepCount_ct3 = steps + q0 * kStepCount;
                auto attn_q0_s_n_head_HD_ct7 = attn + q0 * s.n_head * HD;

                cgh.parallel_for<
                    dpct_kernel_name<class prompt_attn_kernel_e09b15,
                                     dpct_kernel_scalar<KV_MODE>>>(
                    sycl::nd_range<3>(
                        sycl::range(1, (unsigned)s.n_head_kv, (unsigned)nb) *
                            sycl::range(1, 1, THREADS),
                        sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        prompt_attn_kernel<KV_MODE>(
                            q_q0_s_n_head_HD_ct0, pools, ids_q0_cap_ct2,
                            steps_q0_kStepCount_ct3, (int)s.n_head_kv,
                            (int)s.page_size, scale_log2,
                            attn_q0_s_n_head_HD_ct7, (int)cap,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// PR #600: compiled only into the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON), the one build that runs on a
// Volta card; the ready-made engine has none of it (no extra kernels to load, the same code as before).
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
// ---- sm_70 (Volta): v1's structure on mma.m8n8k4 ------------------------------------------------------------------
// Volta's tensor cores only have m8n8k4 (FP16 in, FP32 accumulate): per warp FOUR independent 8x8x4 products, one per
// quad-pair (QP q = lanes 4q..4q+3 and 4q+16..4q+19; thread j = (lane & 3) + 4 * (lane >> 4) within it). Layout, measured on
// a V100 (tools-free microtest, and equal to turbomind's SM70_MMA_884): the thread holds row j of A (k 0..3 as two
// registers) and column j of B (k 0..3); C register i is row (lane & 1) + 4 * (lane >> 4) + 2 * ((i >> 1) & 1),
// column (lane & 2) + 4 * (i >> 2) + (i & 1) of its QP's 8x8 tile.
//   q.k: warp w owns dims [64w, 64w+64) = int8 scale group w; QP q takes cells 8q..8q+7 of the 32-cell chunk; 16 k-steps
//        of 4 dims, two 8-row blocks (12 heads + 4 zero rows), q as hi + lo halves. The warp's partial (times the
//        cells' K scale) goes to S.part[w]; the softmax adds the four groups in a fixed order (deterministic).
//   p.v: warp w owns dims [64w, 64w+64) again; QP q takes the 16 dims 64w+16q.. (two n-tiles of 8); k = the 32 cells in
//        steps of 4; p carries the V scale (as v1) and is split into hi + lo halves.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
#define STRATA_PA_VOLTA 1
#else
#define STRATA_PA_VOLTA 0
#endif

__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if STRATA_PA_VOLTA
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    /* unreachable on SYCL: the launcher refuses this device */   // m8n8k4 is a Volta instruction; the host only launches this kernel on sm_70
#endif
}

template <int KV_MODE>
struct Smem70 {
    using KElem = typename std::conditional<KV_MODE == 0, __half, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1, int8_t, __half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;
    static constexpr int VROW = KV_MODE == 1 ? HD + 16 : HD + 8;
    __half qh[16][QS];
    __half ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][4];
    float vs[CH][4];
    float part[4][16][CH + 4];   // q.k per dim group (scaled by the cell's K scale)
    float s[16][CH + 4];         // the chunk's probabilities
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) prompt_attn_v70_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                  const int32_t* __restrict__ ids,
                                                                  const int32_t* __restrict__ steps, int n_kv_heads,
                                                                  int page_size, float scale_log2,
                                                                  float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem70<KV_MODE>& S = *reinterpret_cast<Smem70<KV_MODE>*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int qp = (lane >> 2) & 3, jj = (lane & 3) + 4 * (lane >> 4);   // quad-pair, thread within it
    const int crow = (lane & 1) + 4 * (lane >> 4);                        // C row of register 0 (+2 per register pair)
    const int ccol = lane & 2;                                           // C column of register 0 (+4 per half, +1 per reg)

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[2][2][8];
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[rb][nt][i] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
        }
        __syncthreads();   // rows ready; the previous chunk's p.v is done with k, v, s
        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            for (int i = t; i < CH * KPIECES; i += THREADS) {
                const int c = i / KPIECES, pc = i % KPIECES;
                const long long r = S.row[c];
                uint4 kx = make_uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_pool + r * HD) + pc);
                    else   // modes 1 and 3: the K side is INT8
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD) + pc);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.k[c][0]) + pc * 16) = kx;
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j) S.v[c][b * QK4_0 + j] = __half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d = __half2float(__ushort_as_half(blk->d));
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.v[c][b * QK4_0 + j] = __float2half_rn((float) ((int)(blk->qs[j] & 0x0F) - 8) * d);
                            S.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                __float2half_rn((float) ((int)(blk->qs[j] >> 4) - 8) * d);
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                for (int i = t; i < CH * VPIECES; i += THREADS) {
                    const int c = i / VPIECES, pc = i % VPIECES;
                    const long long r = S.row[c];
                    uint4 vx = make_uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_q + r * HD) + pc);
                        else
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_pool + r * HD) + pc);
                    }
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.v[c][0]) + pc * 16) = vx;
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        __syncthreads();
        // scores: warp w = dim group w (64 dims), QP q = cells 8q..8q+7
        {
            float tg[2][8];
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int i = 0; i < 8; ++i) tg[rb][i] = 0.0f;
            const int cell_b = qp * 8 + jj;
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                const int d0 = warp * 64 + ks * 4;
                uint32_t b0, b1;
                if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes, exact in FP16
                    const uint32_t raw = *reinterpret_cast<const uint32_t*>(&S.k[cell_b][d0]);
                    b0 = i8x2_to_h2(raw & 0xffffu);
                    b1 = i8x2_to_h2(raw >> 16);
                } else {
                    const uint2 raw = *reinterpret_cast<const uint2*>(&S.k[cell_b][d0]);
                    b0 = raw.x;
                    b1 = raw.y;
                }
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const uint2 ah = *reinterpret_cast<const uint2*>(&S.qh[rb * 8 + jj][d0]);
                    const uint2 al = *reinterpret_cast<const uint2*>(&S.ql[rb * 8 + jj][d0]);
                    mma884(tg[rb], ah.x, ah.y, b0, b1);
                    mma884(tg[rb], al.x, al.y, b0, b1);
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = rb * 8 + crow + 2 * ((i >> 1) & 1);
                    const int cell = qp * 8 + ccol + 4 * (i >> 2) + (i & 1);
                    S.part[warp][row][cell] = tg[rb][i] * S.ks[cell][warp];
                }
        }
        __syncthreads();
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int c = sub * PER + j;
                const float sum4 = ((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c];
                x[j] = c < nh ? sum4 * qdown : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        __syncthreads();
        // p.v: warp w owns dims [64w, 64w+64) (int8 scale group w), QP q its 16 dims 64w+16q..; the V scale is folded
        // into p relative to the chunk's largest, times 2^14 (as v1)
        {
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32) vmax = fmaxf(vmax, S.vs[c][warp]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[2][2][8];
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) tmp[rb][nt][i] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH / 4; ++ks) {
                const int c0 = ks * 4;
                float w[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) w[e] = S.vs[c0 + e][warp] * vup;
                uint2 ah[2], al[2];
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const float4 pv = *reinterpret_cast<const float4*>(&S.s[rb * 8 + jj][c0]);
                    const float p0 = pv.x * w[0], p1 = pv.y * w[1], p2 = pv.z * w[2], p3 = pv.w * w[3];
                    ah[rb].x = pack_h2(p0, p1);
                    ah[rb].y = pack_h2(p2, p3);
                    const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&ah[rb].x));
                    const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&ah[rb].y));
                    al[rb].x = pack_h2(p0 - f01.x, p1 - f01.y);
                    al[rb].y = pack_h2(p2 - f23.x, p3 - f23.y);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int d = warp * 64 + qp * 16 + nt * 8 + jj;
                    uint32_t b0, b1;
                    if constexpr (KV_MODE == 1) {
                        const uint32_t x0 = (uint8_t) S.v[c0][d] | ((uint32_t) (uint8_t) S.v[c0 + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[c0 + 2][d] | ((uint32_t) (uint8_t) S.v[c0 + 3][d] << 8);
                        b0 = i8x2_to_h2(x0);
                        b1 = i8x2_to_h2(x1);
                    } else {
                        const __half2 h0 = __halves2half2(S.v[c0][d], S.v[c0 + 1][d]);
                        const __half2 h1 = __halves2half2(S.v[c0 + 2][d], S.v[c0 + 3][d]);
                        b0 = *reinterpret_cast<const uint32_t*>(&h0);
                        b1 = *reinterpret_cast<const uint32_t*>(&h1);
                    }
#pragma unroll
                    for (int rb = 0; rb < 2; ++rb) {
                        mma884(tmp[rb][nt], ah[rb].x, ah[rb].y, b0, b1);
                        mma884(tmp[rb][nt], al[rb].x, al[rb].y, b0, b1);
                    }
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                float a_[2];
                a_[0] = S.alpha[rb * 8 + crow];
                a_[1] = S.alpha[rb * 8 + crow + 2];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        acc[rb][nt][i] = fmaf(acc[rb][nt][i], a_[(i >> 1) & 1], tmp[rb][nt][i] * vdown);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = rb * 8 + crow + 2 * h;
            if (row >= G) continue;
            const float l = S.lsum[row];
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int i = hh * 4 + h * 2;               // registers i, i+1: row h, columns ccol + 4 * hh + {0, 1}
                    const int d = warp * 64 + qp * 16 + nt * 8 + ccol + 4 * hh;
                    *reinterpret_cast<float2*>(attn + (size_t) row * HD + d) =
                        make_float2(acc[rb][nt][i] * inv, acc[rb][nt][i + 1] * inv);
                }
        }
}

template <int KV_MODE>
bool launch70(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
              const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem70<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_v70_kernel<KV_MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_v70_kernel<KV_MODE><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (sm_70): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#else   // AMD, or a build without STRATA_EXPERIMENTAL_SM60: no m8n8k4 kernel (the caller keeps the old one)
template <int KV_MODE>
bool launch70(const float *, const QsaAttnPools &, const int32_t *,
              const int32_t *, int64_t, const QsaShapes &, float *, int64_t,
              dpct::queue_ptr) {
    return false;
}
#endif  // !__HIPCC__ && STRATA_EXPERIMENTAL_SM60

#if defined(__HIPCC__)
// ---- S6: the int8-KV prompt attention on RDNA4 matrix cores (opt-in: STRATA_HIP_WMMA=1, gfx12 only). The design of
// the v2 kernel above with gfx12's v_wmma_f32_16x16x16_f16 (wave32) in place of m16n8k16: wave w owns dims
// [64w, 64w+64) (int8 scale group w) for q.k and p.v; q and p are split into FP16 hi + lo parts, the int8 codes enter
// exactly as FP16, the scales are applied in FP32 and the four groups' q.k partials are added in a fixed order.
// FP32-level accuracy, deterministic, but not bitwise equal to qsa_decode_attn_batch (another summation order).
// Fragment layout (16x16x16, wave32, checked on gfx1201): A lane l holds A[l % 16][(l / 16) * 8 + i], B lane l holds
// B[(l / 16) * 8 + i][l % 16], C/D lane l holds D[(l / 16) * 8 + i][l % 16], i = 0..7.
// gfx11 (RDNA3 / RDNA3.5: gfx110x, gfx115x) has the same instruction with another layout: A lane l holds all 16 k of
// row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l all 16 k of column l % 16, and C/D lane l holds
// D[2 * i + l / 16][l % 16]. PA_KL is the k values per lane (8 or 16), PA_KOFF(half) the first one, PA_ROW the C row.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_PA_WMMA 1
#define STRATA_PA_GFX11 0
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || \
    defined(__gfx1151__)
#define STRATA_PA_WMMA 1
#define STRATA_PA_GFX11 1
#else
#define STRATA_PA_WMMA 0
#define STRATA_PA_GFX11 0
#endif
#if STRATA_PA_GFX11
constexpr int PA_KL = 16;
#define PA_KOFF(half) 0
#define PA_ROW(i, half) (2 * (i) + (half))
#else
constexpr int PA_KL = 8;
#define PA_KOFF(half) ((half) * 8)
#define PA_ROW(i, half) ((half) * 8 + (i))
#endif
typedef _Float16 wh8 __attribute__((ext_vector_type(PA_KL)));
typedef float wf8 __attribute__((ext_vector_type(8)));
constexpr int WCH = 32;          // cells per chunk (two 16-cell tiles)
constexpr int WVS = 80;          // staged V row stride in bytes (64 codes, 16-byte aligned, banks spread)
struct alignas(16) SmemW {
    float part[4][16][WCH + 1];  // q.k per dim group
    float p[16][WCH + 1];
    float vs[4][WCH];            // V scale per (group, cell)
    int valid[WCH];              // the cell has a resident pool row
    uint8_t v[4][WCH][WVS];      // per wave: the chunk's V codes of its group, one row per cell
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ wf8 wmma_f16(wh8 a, wh8 b, wf8 c) {
#if STRATA_PA_WMMA && STRATA_PA_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif STRATA_PA_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

__device__ __forceinline__ wh8 i8_to_h(const uint32_t* x) {   // PA_KL codes, exact: |code| <= 128
    wh8 h;
#pragma unroll
    for (int i = 0; i < PA_KL; ++i) h[i] = (_Float16) (int) (int8_t) (x[i / 4] >> (8 * (i % 4)));
    return h;
}

// SPLIT = true (default): q and p carried as FP16 hi + lo pairs (two WMMAs each: FP32-level). S23 (opt-in
// STRATA_PA_FAST=1): SPLIT = false, one FP16 image each - half the matrix work, rounding-level (FP16 attention as the
// usual flash-attention kernels), quality-gated
template <bool SPLIT>
__global__ void __launch_bounds__(THREADS) prompt_attn_wmma_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ steps, int n_kv_heads,
                                                                   int page_size, float scale_log2,
                                                                   float* __restrict__ attn, int cap) {
#if STRATA_PA_WMMA
    __shared__ SmemW S;
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int col = lane & 15, half = lane >> 4;   // fragment column / k half
    const int dim0 = warp * 64;

    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    // q A fragments of this wave's 64 dims: row = col (heads 12..15 zero), k = dim0 + kk*16 + PA_KOFF(half) + i
    wh8 qh[4], ql[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        float x[PA_KL];
        if (col < G) {
            const float4* src = reinterpret_cast<const float4*>(q + (size_t) col * HD + dim0 + kk * 16 + PA_KOFF(half));
#pragma unroll
            for (int j = 0; j < PA_KL / 4; ++j) {
                const float4 a = src[j];
                x[4 * j] = a.x; x[4 * j + 1] = a.y; x[4 * j + 2] = a.z; x[4 * j + 3] = a.w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < PA_KL; ++i) x[i] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < PA_KL; ++i) {
            const float v = x[i] * qup;
            const _Float16 hi = (_Float16) v;
            qh[kk][i] = hi;
            ql[kk][i] = (_Float16) (v - (float) hi);
        }
    }

    wf8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int n_chunks = (n + WCH - 1) / WCH;
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int c0 = ci * WCH;
        // this lane's cell of the chunk (lane = cell index): its pool row, or -1 (past the selection, or a block the
        // KV streaming left non-resident: masked as in the decode kernel)
        long long myrow = -1;
        if (c0 + lane < n) {
            const int cell = __ldg(ids + c0 + lane);
            const long long page = (long long) __ldg(p.page_table + cell / page_size);
            if (page >= 0) myrow = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        // stage this wave's V slice of the chunk (64 bytes of each row) and the V scale; the K fragments come straight
        // from global memory (8 consecutive codes per lane)
        {
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0, v2 = v0, v3 = v0;
            float vsc = 0.0f;
            if (myrow >= 0) {
                const uint4* src = reinterpret_cast<const uint4*>(p.v_q + myrow * HD + dim0);
                v0 = __ldg(src); v1 = __ldg(src + 1); v2 = __ldg(src + 2); v3 = __ldg(src + 3);
                vsc = __half2float(__ushort_as_half(__ldg(p.v_scale + myrow * (HD / KV_Q8_GROUP) + warp)));
            }
            uint4* dst = reinterpret_cast<uint4*>(&S.v[warp][lane][0]);
            dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
            S.vs[warp][lane] = vsc;
            if (warp == 0) S.valid[lane] = myrow >= 0;
        }
        // q.k over this wave's 64 dims for two 16-cell tiles, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const long long rr = __shfl(myrow, nt * 16 + col);
            wf8 s = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float ksc = 0.0f;
            if (rr >= 0) ksc = __half2float(__ushort_as_half(__ldg(p.k_scale + rr * (HD / KV_Q8_GROUP) + warp)));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t kx[PA_KL / 4] = {};
                if (rr >= 0) {
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(p.k_q + rr * HD + dim0 + kk * 16 + PA_KOFF(half));
#if STRATA_PA_GFX11
                    const uint4 v = __ldg(reinterpret_cast<const uint4*>(src));
                    kx[0] = v.x; kx[1] = v.y; kx[2] = v.z; kx[3] = v.w;
#else
                    const uint2 v = __ldg(reinterpret_cast<const uint2*>(src));
                    kx[0] = v.x; kx[1] = v.y;
#endif
                }
                const wh8 b = i8_to_h(kx);
                s = wmma_f16(qh[kk], b, s);
                if constexpr (SPLIT) s = wmma_f16(ql[kk], b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[warp][PA_ROW(i, half)][nt * 16 + col] = s[i] * ksc;
        }
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                // past the selection, or a masked cell (non-resident page, as in the decode kernel): no weight
                x[j] = S.valid[c] ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this wave's 64 dims: 4 tiles of 16 dims, 2 k-steps of 16 cells
        {
            float vmax = S.vs[warp][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            wf8 tmp[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int cb = ks * 16 + PA_KOFF(half);   // this lane's PA_KL cells
                wh8 ah, al;
#pragma unroll
                for (int i = 0; i < PA_KL; ++i) {
                    const float pv = S.p[col][cb + i] * (S.vs[warp][cb + i] * vup);
                    const _Float16 hi = (_Float16) pv;
                    ah[i] = hi;
                    al[i] = (_Float16) (pv - (float) hi);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = j * 16 + col;
                    wh8 b;
#pragma unroll
                    for (int i = 0; i < PA_KL; ++i) b[i] = (_Float16) (int) (int8_t) S.v[warp][cb + i][d];
                    tmp[j] = wmma_f16(ah, b, tmp[j]);
                    if constexpr (SPLIT) tmp[j] = wmma_f16(al, b, tmp[j]);
                }
            }
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = S.alpha[PA_ROW(i, half)];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[j][i] = fmaf(acc[j][i], a[i], tmp[j][i] * vdown);
        }
        // no barrier here: `v`/`vs` are this wave's own, and `p`/`alpha`/`valid` are rewritten only after the next
        // chunk's first barrier (which every wave reaches after its p.v)
    }
    float inv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float l = S.lsum[PA_ROW(i, half)];
        inv[i] = l > 0.0f ? 1.0f / l : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = PA_ROW(i, half);
            if (row < G) attn[(size_t) row * HD + dim0 + j * 16 + col] = acc[j][i] * inv[i];
        }
#else
    __builtin_trap();
#endif
}

// gfx12 (RDNA4) and gfx11 (RDNA3 / RDNA3.5), only on request: the output differs from the default kernel's in its
// last bits
bool hip_wmma_usable() {
    static const bool want = [] {
        const char* e = std::getenv("STRATA_HIP_WMMA");
        return e != nullptr && e[0] == '1';
    }();
    if (!want) return false;
    static int arch[64] = {};   // 0 unknown, 1 gfx11 / gfx12, 2 other
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { (void) hipGetLastError(); return false; }
    if (arch[dev] == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        arch[dev] = std::strncmp(prop.gcnArchName, "gfx12", 5) == 0 || strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName)
                        ? 1 : 2;
        static bool told = false;
        if (!told) {
            told = true;
            std::fprintf(stderr, "strata: STRATA_HIP_WMMA: the prompt attention on matrix cores %s (%s)\n",
                         arch[dev] == 1 ? "on" : "unavailable", prop.gcnArchName);
        }
    }
    return arch[dev] == 1;
}

bool launch_wmma(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                 const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        static const bool fast = [] { const char* e = std::getenv("STRATA_PA_FAST"); return e != nullptr && e[0] == '1'; }();
        static const int64_t min_t = [] { const char* e = std::getenv("STRATA_PF_SWITCH_MIN_T"); return e ? (int64_t) std::atoll(e) : (int64_t) 0; }();
        if (fast && n_q >= min_t)
            prompt_attn_wmma_kernel<false><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
                q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
                (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
        else
            prompt_attn_wmma_kernel<true><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
                q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
                (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (wmma): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#endif

}  // namespace

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    bool volta = false, turing = false;   // per call, from the CURRENT device (a layer split can mix Turing with newer cards)
    {   // sm_75 or newer: the MMA above compiles for both.  sm_80+ runs the cp.async kernel (launch_i8); Turing has
        // no cp.async, so it runs the v1 kernel (launch<1>, same accuracy, another summation order).  An older card
        // keeps the old kernel.
        // #371: the compute capability with its minor - sm_70 (V100) has no m16n8k8 (the kernels trap below sm_75)
#if defined(STRATA_HIP_GFX906)
        return false;   // gfx906: the tensor-core and WMMA kernels are not for it; the FP32 kernel runs
#endif
        static int cc[64] = {};
        int dev = 0;
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 ||
            dev < 0 || dev >= 64) {
            ; return false;
        }
        if (cc[dev] == 0) {
            int major = 0, minor = 0;
            if (DPCT_CHECK_ERROR(
                    major = dpct::get_device(dev).get_major_version()) != 0 ||
                DPCT_CHECK_ERROR(
                    minor = dpct::get_device(dev).get_minor_version()) != 0) {
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                return false;
            }
            // STRATA_QSA_WARP=1|attn (an A/B arm): the pre-sm_80 kernels on any card, as RTX 20 runs them
            const char* w = std::getenv("STRATA_QSA_WARP");
            cc[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "attn")) ? 75
                      : 10 * strata::cc_major_of(major) + strata::cc_minor_of(minor);
        }
        (void) cc[dev];
        // SYCL: the mma.sync kernel below is not ported (upstream's Turing/sm_80 split - `turing` - does not
        // apply); its XMX port takes the same arguments
        if (qsa_prompt_attn_xmx(q, pools, ids, steps, cap, s, attn, n_q, stream)) return true;
        return false;   // the caller takes the older kernel
    }
#if defined(__HIPCC__)
    // the tensor-core kernels are compiled out on AMD (its major version is not a CUDA sm); RDNA4 has its own int8-KV
    // matrix-core kernel, opt-in (STRATA_HIP_WMMA=1); everything else keeps the old kernel
    (void) turing; (void) volta;
    if (pools.k_q != nullptr && pools.v_q != nullptr && pools.k_scale != nullptr && pools.v_scale != nullptr &&
        pools.k_q4 == nullptr && pools.v_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv &&
        cap > 0 && ids && steps && pools.page_table && hip_wmma_usable())
        return launch_wmma(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);
    return false;
#endif
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table)
        return false;
    dpct::queue_ptr st = strata::q_of(stream);
    if (pools.k_q4 != nullptr) {   // Q4_0 K and V (--kv q4_0): mode 4.  STRATA_PROMPT_ATTN_Q4=0: the old kernel (A/B)
        static const bool q4_off = [] {
            const char* v = std::getenv("STRATA_PROMPT_ATTN_Q4");
            return v != nullptr && v[0] == '0';
        }();
        // sm_80+ only: on Turing mode 4 would run as pairs of m16n8k8 MMAs, which no parity run has checked yet
        if (q4_off || turing || pools.v_q4 == nullptr) return false;
        return launch<4>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return false;
        if (volta) return launch70<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
        return launch<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        // STRATA_PROMPT_ATTN_V1=1 (debug): the first version, same accuracy, another summation order - the control
        // for how far the model amplifies an FP32-level change.  Turing always takes it: v2's cp.async does not
        // exist before sm_80.
        static const bool v1 = std::getenv("STRATA_PROMPT_ATTN_V1") != nullptr;
        if (volta) return launch70<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
        if (v1 || turing) return launch<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
        return launch_i8(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    if (volta) return launch70<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
    return launch<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
}

}  // namespace strata::kernels
