// SYCL port: the prompt attention on XMX (joint_matrix) - the port of qsa_prompt_attn.cu's tensor-core kernel, v2.
//
// One work-group per (query position, KV head): 16 query rows (the group's 12 heads + 4 zero rows) against the
// position's selected cells, CH cells per chunk. Scores S = Q K^T and the update O += P V run on joint_matrix (FP16 x
// FP16 -> FP32, 16x16x16, sub-group 16). Q and P are split into hi + lo FP16 halves exactly as the CUDA kernel does, and
// the per-64-dim INT8 scales of K and V are applied outside the products the same way (K's after each scale group's
// product, V's folded into P relative to the chunk's largest). KV modes: 0 FP16 K/V, 1 INT8 K/V with scales, 3 hybrid
// K8V4 (INT8 K + Q4_0 V dequantized to FP16).
//
// v1 (16 cells per chunk) was correct but 3x slower than the FP32 fallback: its time went to staging (2-byte scattered
// stores into the packed B layouts) and to 5 barriers + a softmax pass + an accumulator read-modify-write per 16 cells.
// v2: 64-cell chunks (a quarter of the per-cell overhead), V packed two cells at a time as 32-byte contiguous stores,
// K^T packed as 4-byte pairs with INT8 converted on the load, hi and lo halves of Q accumulated into one accumulator per
// scale group, and one accumulator update per chunk after the whole P.V product has run in registers.
#include "strata/kernels/qsa_prompt_attn_xmx.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int HD = 256, G = 12, SG = 16, NSG = 8, THREADS = SG * NSG, QS = HD + 8;
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_prompt_attn_xmx is compiled for core/geometry.hpp");
constexpr int QK4_0 = 32;
struct q4_0_block { uint16_t d; uint8_t qs[QK4_0 / 2]; };

// local memory layout for CH cells per chunk; the P tiles reuse the K^T region (K is done once the scores are)
template <int CH>
struct Layout {
    static constexpr int PS = CH + 8;                        // P / score row stride
    static constexpr int NT = CH / 16;                       // cell tiles
    static constexpr int UNITS = NT * 4;                     // (cell tile, scale group) score units
    static constexpr size_t q = 2 * 16 * QS * 2;             // Q hi, Q lo (halves)
    static constexpr size_t kp = (size_t) CH * HD * 2;       // packed K^T, then the P tiles
    static constexpr size_t pv = (size_t) 4 * 2 * 16 * PS * 2;
    static constexpr size_t kp_or_pv = kp > pv ? kp : pv;
    static constexpr size_t vp = (size_t) CH * HD * 2;       // packed V
    static constexpr size_t parts = (size_t) (UNITS > 16 ? UNITS : 16) * 256 * 4;   // score units, then P.V tiles
    static constexpr size_t s = (size_t) 16 * PS * 4;        // probabilities
    static constexpr size_t acc = (size_t) 16 * HD * 4;
    static constexpr size_t scal = (size_t) CH * 4 * 4 * 2;  // ks, vs
    static constexpr size_t misc = 80 * 4;
    static constexpr size_t row = (size_t) CH * 8;
    static constexpr size_t bytes = q + kp_or_pv + vp + parts + s + acc + scal + misc + row + 16 * 12;
};

template <typename T>
__dpct_inline__ auto lptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

template <int KV_MODE, int CH>
void prompt_attn_xmx2(const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ids,
                      const int32_t* __restrict__ steps, int n_kv_heads, int page_size, float scale_log2,
                      float* __restrict__ attn, int cap, uint8_t* smem) {
    using L = Layout<CH>;
    constexpr int PS = L::PS, NT = L::NT, UNITS = L::UNITS;
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = it.get_sub_group();
    uint8_t* at = smem;
    auto take = [&](size_t b) { uint8_t* r = at; at += (b + 15) & ~(size_t) 15; return r; };
    sycl::half* qh = (sycl::half*) take(L::q / 2);
    sycl::half* ql = (sycl::half*) take(L::q / 2);
    sycl::half* kp = (sycl::half*) take(L::kp_or_pv);
    sycl::half* ptile = kp;                                   // [g][hi/lo][16][PS], after the scores
    sycl::half* vp = (sycl::half*) take(L::vp);
    float* parts = (float*) take(L::parts);
    float* sp = (float*) take(L::s);
    float* acc = (float*) take(L::acc);
    float* ks = (float*) take(L::scal / 2);
    float* vs = (float*) take(L::scal / 2);
    float* misc = (float*) take(L::misc);
    long long* rows = (long long*) take(L::row);
    float* qmax = misc;            // [8]
    float* mrow = misc + 8;        // [16]
    float* lsum = misc + 24;       // [16]
    float* alpha = misc + 40;      // [16]
    float* vup = misc + 56;        // [4]
    float* vdown = misc + 60;      // [4]

    const int qi = (int) it.get_group(2), kvh = (int) it.get_group(1);
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = steps[(size_t) qi * kStepCount + kStepWidth];
    const int t = (int) it.get_local_id(2), lane = (int) sg.get_local_id()[0], sgid = (int) sg.get_group_id()[0];

    // q: scaled so its largest value sits near 2^14 (exact), split into hi + lo halves (as the CUDA kernel)
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = sycl::fmax(qm, sycl::fabs(q[i]));
    qm = sycl::reduce_over_group(sg, qm, sycl::maximum<float>());
    if (lane == 0) qmax[sgid] = qm;
    it.barrier(sycl::access::fence_space::local_space);
    qm = 0.0f;
    for (int i = 0; i < NSG; ++i) qm = sycl::fmax(qm, qmax[i]);
    int qe = 0;   // frexp's exponent from the bits (qm < 2^qe); qm > 0 is a normal number here
    if (qm > 0.0f) qe = (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFFu) - 126;
    const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const sycl::half hi = sycl::half(x);
        qh[h * QS + d] = hi;
        ql[h * QS + d] = sycl::half(x - (float) hi);
    }
    for (int i = t; i < 16 * HD; i += THREADS) acc[i] = 0.0f;
    if (t < 16) { mrow[t] = -INFINITY; lsum[t] = 0.0f; }

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = sycl::min(CH, n - c0);
        // the chunk's rows (the previous chunk's P.V only reads vp/ptile, not rows)
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            rows[t] = r;
        }
        it.barrier(sycl::access::fence_space::local_space);   // rows ready; the previous chunk is done with kp/vp/parts
        // K rows -> packed K^T: element (dim k, cell c) at kp[(k/2)*(CH*2) + c*2 + (k&1)]; one item = 8 dims of a cell
        for (int i = t; i < CH * (HD / 8); i += THREADS) {
            const int c = i / (HD / 8), piece = i % (HD / 8);
            const long long r = rows[c];
            uint32_t w[4] = {0u, 0u, 0u, 0u};
            if (r >= 0) {
                if constexpr (KV_MODE == 0) {
                    const sycl::vec<uint32_t, 4> v = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(p.k_pool + r * HD + piece * 8);
                    for (int j = 0; j < 4; ++j) w[j] = v[j];
                } else {
                    const sycl::vec<int8_t, 8> v = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.k_q + r * HD + piece * 8);
                    for (int j = 0; j < 4; ++j) {
                        const sycl::half2 h2((float) v[2 * j], (float) v[2 * j + 1]);
                        w[j] = sycl::bit_cast<uint32_t>(h2);
                    }
                }
            }
            uint32_t* dst = reinterpret_cast<uint32_t*>(kp);
            for (int j = 0; j < 4; ++j) dst[(size_t) (piece * 4 + j) * CH + c] = w[j];
        }
        // V rows -> packed V: element (cell c, dim d) at vp[(c/2)*(HD*2) + d*2 + (c&1)]; one item = a cell pair x 8 dims,
        // written as 8 contiguous uint32 (32 bytes)
        if constexpr (KV_MODE == 3) {
            constexpr int BLKS = HD / QK4_0;
            constexpr int BYTES = BLKS * (int) sizeof(q4_0_block);
            for (int i = t; i < (CH / 2) * BLKS; i += THREADS) {
                const int cp = i / BLKS, b = i % BLKS;
                float v0[QK4_0], v1[QK4_0];
                for (int e = 0; e < 2; ++e) {
                    const long long r = rows[2 * cp + e];
                    float* v = e ? v1 : v0;
                    if (r < 0) { for (int j = 0; j < QK4_0; ++j) v[j] = 0.0f; continue; }
                    const q4_0_block* blk = reinterpret_cast<const q4_0_block*>(p.v_q4 + r * BYTES) + b;
                    const float d = (float) sycl::bit_cast<sycl::half>(blk->d);
                    for (int j = 0; j < QK4_0 / 2; ++j) {
                        v[j] = (float) ((int) (blk->qs[j] & 0x0F) - 8) * d;
                        v[j + QK4_0 / 2] = (float) ((int) (blk->qs[j] >> 4) - 8) * d;
                    }
                }
                uint32_t* dst = reinterpret_cast<uint32_t*>(vp) + (size_t) cp * HD + b * QK4_0;
                for (int j = 0; j < QK4_0; j += 8) {
                    sycl::vec<uint32_t, 8> o;
                    for (int u = 0; u < 8; ++u) o[u] = sycl::bit_cast<uint32_t>(sycl::half2(v0[j + u], v1[j + u]));
                    *reinterpret_cast<sycl::vec<uint32_t, 8>*>(dst + j) = o;
                }
            }
        } else {
            for (int i = t; i < (CH / 2) * (HD / 8); i += THREADS) {
                const int cp = i / (HD / 8), piece = i % (HD / 8);
                sycl::half h0[8], h1[8];
                for (int e = 0; e < 2; ++e) {
                    const long long r = rows[2 * cp + e];
                    sycl::half* h = e ? h1 : h0;
                    if (r < 0) { for (int j = 0; j < 8; ++j) h[j] = sycl::half(0.0f); continue; }
                    if constexpr (KV_MODE == 1) {
                        const sycl::vec<int8_t, 8> v = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.v_q + r * HD + piece * 8);
                        for (int j = 0; j < 8; ++j) h[j] = sycl::half((float) v[j]);
                    } else {
                        const sycl::vec<uint16_t, 8> v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.v_pool + r * HD + piece * 8);
                        for (int j = 0; j < 8; ++j) h[j] = sycl::bit_cast<sycl::half>(v[j]);
                    }
                }
                sycl::vec<uint32_t, 8> o;
                for (int j = 0; j < 8; ++j) o[j] = sycl::bit_cast<uint32_t>(sycl::half2(h0[j], h1[j]));
                *reinterpret_cast<sycl::vec<uint32_t, 8>*>(reinterpret_cast<uint32_t*>(vp) + (size_t) cp * HD + piece * 8) = o;
            }
        }
        for (int i = t; i < CH * 4; i += THREADS) {
            const int c = i / 4, g = i % 4;
            const long long r = rows[c];
            float a = 0.0f, b = 0.0f;
            if (r >= 0) {
                if constexpr (KV_MODE == 1) {
                    a = (float) sycl::bit_cast<sycl::half>(p.k_scale[r * (HD / KV_Q8_GROUP) + g]);
                    b = (float) sycl::bit_cast<sycl::half>(p.v_scale[r * (HD / KV_Q8_GROUP) + g]);
                } else if constexpr (KV_MODE == 3) {
                    a = (float) sycl::bit_cast<sycl::half>(p.k_scale[r * (HD / KV_Q8_GROUP) + g]);
                    b = 1.0f;
                } else {
                    a = b = 1.0f;
                }
            }
            ks[c * 4 + g] = a;
            vs[c * 4 + g] = b;
        }
        it.barrier(sycl::access::fence_space::local_space);
        // scores: unit u = (cell tile u / 4, scale group u % 4); hi and lo of Q into one accumulator (same K scale)
        for (int u = sgid; u < UNITS; u += NSG) {
            const int nt = u / 4, g = u % 4;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> C;
            jm::joint_matrix_fill(sg, C, 0.0f);
            for (int kk = 0; kk < 4; ++kk) {
                const int k0 = (g * 4 + kk) * 16;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> B;
                jm::joint_matrix_load(sg, B, lptr(kp + (k0 >> 1) * (CH * 2) + nt * 16 * 2), (size_t) (CH * 2));
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> A;
                jm::joint_matrix_load(sg, A, lptr(qh + k0), (size_t) QS);
                jm::joint_matrix_mad(sg, C, A, B, C);
                jm::joint_matrix_load(sg, A, lptr(ql + k0), (size_t) QS);
                jm::joint_matrix_mad(sg, C, A, B, C);
            }
            jm::joint_matrix_store(sg, C, lptr(parts + (size_t) u * 256), (size_t) 16, jm::layout::row_major);
        }
        // the V scale per group, folded into P relative to the chunk's largest (times 2^14, as in CUDA)
        if (t < 4) {
            float vmax = 0.0f;
            for (int c = 0; c < CH; ++c) vmax = sycl::fmax(vmax, vs[c * 4 + t]);
            vup[t] = vmax > 0.0f ? 16384.0f / vmax : 0.0f;
            vdown[t] = vmax * (1.0f / 16384.0f);
        }
        it.barrier(sycl::access::fence_space::local_space);
        // online softmax: row r = t / 8, CH / 8 cells per thread, the 8 lanes of a row in one sub-group
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
            for (int j = 0; j < PER; ++j) {
                const int c = sub * PER + j, nt = c >> 4, cc = c & 15;
                float v = 0.0f;
                for (int g = 0; g < 4; ++g) v += parts[(size_t) (nt * 4 + g) * 256 + r * 16 + cc] * ks[c * 4 + g];
                x[j] = c < nh ? v * qdown : -INFINITY;
                mx = sycl::fmax(mx, x[j]);
            }
            for (int o = 1; o < 8; o <<= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, o));
            const float m_old = mrow[r];
            const float m_new = sycl::fmax(m_old, mx);
            float sum = 0.0f;
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : sycl::exp2(x[j] - m_new);
                sp[r * PS + sub * PER + j] = e;
                sum += e;
            }
            for (int o = 1; o < 8; o <<= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
            sycl::group_barrier(sg);   // every lane of the row read m_old before lane 0 moves it
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : sycl::exp2(m_old - m_new);
                alpha[r] = a;
                lsum[r] = sycl::fma(lsum[r], a, sum);
                mrow[r] = m_new;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
        // P tiles into the K^T region: groups share one pair when V carries no per-group scale (modes 0 and 3)
        constexpr int NGP = KV_MODE == 1 ? 4 : 1;
        for (int i = t; i < NGP * 16 * (CH / 2); i += THREADS) {
            const int g = i / (16 * (CH / 2)), rc = i % (16 * (CH / 2)), r = rc / (CH / 2), c = (rc % (CH / 2)) * 2;
            const float w0 = vs[c * 4 + g] * vup[g], w1 = vs[(c + 1) * 4 + g] * vup[g];
            const float p0 = sp[r * PS + c] * w0, p1 = sp[r * PS + c + 1] * w1;
            const sycl::half h0 = sycl::half(p0), h1 = sycl::half(p1);
            *reinterpret_cast<sycl::half2*>(ptile + ((size_t) (g * 2 + 0) * 16 + r) * PS + c) = sycl::half2(h0, h1);
            *reinterpret_cast<sycl::half2*>(ptile + ((size_t) (g * 2 + 1) * 16 + r) * PS + c) =
                sycl::half2(sycl::half(p0 - (float) h0), sycl::half(p1 - (float) h1));
        }
        it.barrier(sycl::access::fence_space::local_space);
        // P.V: sub-group sg owns dims [32 sg, 32 sg + 32) (two B tiles) = V scale group sg / 2; the whole chunk's product
        // in registers, then one update of its own accumulator columns
        {
            const int g = sgid >> 1, gp = KV_MODE == 1 ? g : 0;
            const sycl::half* ah = ptile + (size_t) (gp * 2 + 0) * 16 * PS;
            const sycl::half* al = ptile + (size_t) (gp * 2 + 1) * 16 * PS;
            float* tmp = parts + (size_t) sgid * 512;
            for (int dt = 0; dt < 2; ++dt) {
                const int d0 = sgid * 32 + dt * 16;
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> C;
                jm::joint_matrix_fill(sg, C, 0.0f);
                for (int ks16 = 0; ks16 < CH; ks16 += 16) {
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> B;
                    jm::joint_matrix_load(sg, B, lptr(vp + (size_t) (ks16 >> 1) * (HD * 2) + d0 * 2), (size_t) (HD * 2));
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> A;
                    jm::joint_matrix_load(sg, A, lptr(ah + ks16), (size_t) PS);
                    jm::joint_matrix_mad(sg, C, A, B, C);
                    jm::joint_matrix_load(sg, A, lptr(al + ks16), (size_t) PS);
                    jm::joint_matrix_mad(sg, C, A, B, C);
                }
                jm::joint_matrix_store(sg, C, lptr(tmp + dt * 256), (size_t) 16, jm::layout::row_major);
            }
            sycl::group_barrier(sg);
            const float vd = vdown[g];
            for (int i = lane; i < 512; i += SG) {
                const int dt = i >> 8, rc = i & 255, r = rc >> 4, cc = rc & 15;
                float* a = acc + r * HD + sgid * 32 + dt * 16 + cc;
                *a = sycl::fma(*a, alpha[r], tmp[i] * vd);
            }
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    for (int i = t; i < G * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float l = lsum[h];
        attn[(size_t) h * HD + d] = l > 0.0f ? acc[h * HD + d] / l : 0.0f;
    }
}

template <int KV_MODE, int CH>
bool launch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
            const QsaShapes& s, float* attn, int64_t n_q, dpct::queue_ptr st) {
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    constexpr size_t bytes = Layout<CH>::bytes;
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        const float* qq = q + q0 * s.n_head * HD;
        const int32_t* idq = ids + q0 * cap;
        const int32_t* stq = steps + q0 * kStepCount;
        float* aq = attn + q0 * s.n_head * HD;
        const int nkv = (int) s.n_head_kv, ps = (int) s.page_size, capi = (int) cap;
        st->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(bytes), cgh);
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, (size_t) nkv, (size_t) nb * THREADS), sycl::range<3>(1, 1, THREADS)),
                             [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                                 prompt_attn_xmx2<KV_MODE, CH>(qq, pools, idq, stq, nkv, ps, scale_log2, aq, capi,
                                                               slm.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    return true;
}

template <int CH>
bool dispatch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
              const QsaShapes& s, float* attn, int64_t n_q, dpct::queue_ptr st) {
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {
        if (!pools.k_scale) return false;
        return launch<3, CH>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        return launch<1, CH>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    return launch<0, CH>(q, pools, ids, steps, cap, s, attn, n_q, st);
}
}  // namespace

bool qsa_prompt_attn_xmx(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                         const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    // STRATA_PROMPT_ATTN_XMX=1 (64-cell chunks) or =32 (32-cell chunks); unset or 0: the caller's fallback
    static const int mode = [] { const char* v = std::getenv("STRATA_PROMPT_ATTN_XMX"); return v ? std::atoi(v) : 0; }();
    if (mode == 0 || n_q <= 0) return false;
    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps ||
        !pools.page_table)
        return false;
    dpct::queue_ptr st = strata::q_of(stream);
    const size_t have = st->get_device().get_info<sycl::info::device::local_mem_size>();
    const bool ch64 = mode != 32 && have >= Layout<64>::bytes;
    static bool told = false;
    if (!told) {
        told = true;
        std::fprintf(stderr, "strata: prompt attention on XMX, %d-cell chunks (%zu of %zu bytes of local memory)\n",
                     ch64 ? 64 : 32, ch64 ? Layout<64>::bytes : Layout<32>::bytes, have);
    }
    if (!ch64 && have < Layout<32>::bytes) return false;
    return ch64 ? dispatch<64>(q, pools, ids, steps, cap, s, attn, n_q, st)
                : dispatch<32>(q, pools, ids, steps, cap, s, attn, n_q, st);
}
}  // namespace strata::kernels
