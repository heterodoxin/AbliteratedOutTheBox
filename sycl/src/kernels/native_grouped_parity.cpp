// src/kernels/native_grouped_parity.cpp - native_expert_grouped's group stride (grid_groups) and its fused SwiGLU +
// q8_1 pass, against the launches before them (native_grouped_set_v1: a block row per possible group, SwiGLU and the
// q8_1 quantization as two kernels over all cap_entries).  Synthetic blobs, no model.
//
//     build/native_grouped_parity           the checks (a ctest; needs the GPU)
//     build/native_grouped_parity --bench   the checks, then v1/new timings of 48 calls in a graph (a window's layers)
//
// Every gate/up format with every down format; calls of 0..cap groups of 1..T entries each (an empty call is what the
// verify window's PCIe call usually is), entries from 0 and from past 0 (the PCIe call's follow the VRAM call's),
// scattered destinations, stale scratch, grid_groups 0 (= cap), 1, 2, 3, 4 and above cap: the output must be BITWISE
// equal to v1's, the rows the call must not write included (both start from the same NaN pattern).
//
// Random bytes are valid codes for every format here (every grid index is in range); only the fp16 block scales are
// set, small enough that the SwiGLU outputs keep a finite fp16 q8_1 scale.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/iq_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {
// SYCL port only (iq_kernels.dp.cpp): the K-quant / Q5_1 / Q8_0 lane kernels with (true, the default) or without the
// multi-entry split; STRATA_KQ_SPLIT=0 at startup is false
void native_expert_set_kq_split(bool on);
}
namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(dpct::err0 e, const char *what) {
}

template<typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(DPCT_CHECK_ERROR(p = (T *)sycl::malloc_device(
                            n * sizeof(T) + 256, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(p, 0, n * sizeof(T) + 256).wait()),
       "memset");
    return p;
}

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        case 42: return "Q2_0";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 || t == 7 || t == 8 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of its four scale words (bytes 48..55), the sign and the
            // exponent's top bits in the last: 0x1 / 0x2 there (0x9 / 0xA negative) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t) ((sgn(rng) ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else {   // the other formats start their block with the fp16 scale: 2^-13 .. 2^-6
            // (Q4_K / Q5_K: 2^-14 .. 2^-11, their 6-bit sub-scales multiply it by up to 63)
            const int e = (t == 12 || t == 13) ? 1 + ex(rng) % 4 : ex(rng);
            const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (e << 10) | man(rng));
            std::memcpy(&w[o + (t == 14 ? 208 : 0)], &h, 2);   // Q6_K keeps its fp16 scale last (ql, qh, scales, d)
            if (t == 12 || t == 13 || t == 7) {   // Q4_K / Q5_K: dmin, Q5_1: m - the second fp16, as small
                const int em = (t == 7) ? ex(rng) : 1 + ex(rng) % 4;
                const uint16_t m = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (em << 10) | man(rng));
                std::memcpy(&w[o + 2], &m, 2);
            }
        }
    }
    return w;
}

// one window's worth of expert calls on one format pair: blobs, T tokens' q8_1 activations, cap = T * K entries
struct Setup {
    k::NativeExpertLayout L;
    int T = 0, K = 0, cap = 0, n_blobs = 0;
    size_t slot = 0;
    uint8_t* blobs = nullptr;
    uint8_t* xq = nullptr;
    uint8_t* scr = nullptr;
    size_t scr_bytes = 0;
    float* out = nullptr;
    size_t out_floats = 0;
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *n = nullptr, *dst = nullptr, *tok = nullptr;
    std::vector<std::vector<uint8_t>> hb;          // host copies: the blobs, the q8_1 activations, the last plan
    std::vector<uint8_t> h_xq;
    std::vector<int> h_blob;
    std::vector<int32_t> h_start, h_tok, h_dst;

    Setup(int gu, int dt, int64_t H, int64_t FF, int tokens, int k_per_token,
          int blobs_n, std::mt19937 &rng, dpct::queue_ptr s) {
        L = k::native_expert_layout(gu, dt, H, FF);
        T = tokens; K = k_per_token; cap = T * K; n_blobs = blobs_n;
        slot = (L.bytes + 255) & ~(size_t) 255;
        blobs = dalloc<uint8_t>(slot * (size_t) n_blobs);
        for (int b = 0; b < n_blobs; ++b) {
            // gate rows | up rows (format gu, H values each) | down rows (format dt, FF values each)
            auto blob = random_rows(gu, 2 * FF, H, rng);
            const auto down = random_rows(dt, H, FF, rng);
            blob.insert(blob.end(), down.begin(), down.end());
            hb.push_back(blob);
            if (blob.size() != L.bytes) { std::fprintf(stderr, "blob %zu != %zu bytes\n", blob.size(), L.bytes); std::exit(2); }
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                   blobs + slot * (size_t)b, blob.data(), blob.size()).wait()),
               "blob");
        }
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) T * H);
        for (auto& v : x) v = nd(rng);
        float* dx = dalloc<float>(x.size());
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(dx, x.data(), x.size() * 4).wait()),
           "x");
        xq = dalloc<uint8_t>((size_t) T * (H / 32) * 36);
        k::quantize_q8_1_rows(dx, T, H, xq, s);
        h_xq.resize((size_t) T * (H / 32) * 36);
        s->memcpy(h_xq.data(), xq, h_xq.size()).wait();
        ck(DPCT_CHECK_ERROR(s->wait()), "quantize");
        sycl::free(dx, dpct::get_in_order_queue());
        scr_bytes = k::native_expert_scratch_bytes(cap, FF);
        scr = dalloc<uint8_t>(scr_bytes);
        out_floats = (size_t) cap * H;
        out = dalloc<float>(out_floats);
        ptr = dalloc<unsigned long long>(cap);
        start = dalloc<int32_t>(cap + 1);
        n = dalloc<int32_t>(1);
        dst = dalloc<int32_t>(cap);
        tok = dalloc<int32_t>(cap);
    }
    ~Setup() {
        sycl::free(blobs, dpct::get_in_order_queue());
            sycl::free(xq, dpct::get_in_order_queue());
            sycl::free(scr, dpct::get_in_order_queue());
            sycl::free(out, dpct::get_in_order_queue());
        sycl::free(ptr, dpct::get_in_order_queue());
            sycl::free(start, dpct::get_in_order_queue());
            sycl::free(n, dpct::get_in_order_queue());
            sycl::free(dst, dpct::get_in_order_queue());
            sycl::free(tok, dpct::get_in_order_queue());
    }
    // a plan: `groups` groups of 1..T entries from entry `base` on (as many as fit in cap)
    int plan(int groups, int base, std::mt19937& rng) {
        std::uniform_int_distribution<int> size(1, T);
        std::vector<int> sizes;
        int used = base;
        for (int g = 0; g < groups; ++g) {
            const int m = std::min(size(rng), cap - used);
            if (m <= 0) break;
            sizes.push_back(m);
            used += m;
        }
        return plan_sizes(sizes, base, rng);
    }
    // groups of these sizes from entry `base` on, each its own blob (as long as there are enough: a window's groups
    // are different experts); scattered destinations, random tokens
    int plan_sizes(const std::vector<int>& sizes, int base, std::mt19937& rng) {
        std::uniform_int_distribution<int> tk(0, T - 1);
        std::vector<int> order(n_blobs);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<unsigned long long> p;
        std::vector<int32_t> st(1, base), tk_v, ds(cap);
        h_blob.clear();
        for (const int m : sizes) {
            h_blob.push_back(order[p.size() % order.size()]);
            p.push_back((unsigned long long) (blobs + slot * (size_t) order[p.size() % order.size()]));
            st.push_back(st.back() + m);
        }
        if (st.back() > cap) { std::fprintf(stderr, "plan: %d entries > cap %d\n", st.back(), cap); std::exit(2); }
        const int ng = (int) p.size();
        std::iota(ds.begin(), ds.end(), 0);
        std::shuffle(ds.begin(), ds.end(), rng);                  // scattered rows of `out`
        for (int e = 0; e < cap; ++e) tk_v.push_back(tk(rng));
        h_start = st; h_tok = tk_v; h_dst = ds;   // host copies for the CPU reference
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        if (ng) ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                       ptr, p.data(), p.size() * 8).wait()),
                   "ptr");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(start, st.data(),
                                                              st.size() * 4).wait()),
           "start");
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(n, &ng, 4).wait()),
           "n");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(dst, ds.data(),
                                                              ds.size() * 4).wait()),
           "dst");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(tok, tk_v.data(),
                                                              tk_v.size() * 4).wait()),
           "tok");
        return ng;
    }
    void call(int64_t grid_groups, dpct::queue_ptr s) {
        k::native_expert_grouped(L, ptr, start, n, dst, tok, cap, cap, xq, scr, out, s, grid_groups);
    }
    // `out` after one call from the NaN pattern, the scratch holding `junk` (what earlier calls left there)
    std::vector<uint32_t> result(bool v1, int64_t grid_groups, int junk,
                                 dpct::queue_ptr s) {
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memset(out, 0xFF, out_floats * 4)
                                .wait()),
           "memset");
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(scr, junk, scr_bytes).wait()),
           "memset");
        k::native_grouped_set_v1(v1);
        call(grid_groups, s);
        k::native_grouped_set_v1(false);
        ck(DPCT_CHECK_ERROR(s->wait()), "sync");
        std::vector<uint32_t> o(out_floats);
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(o.data(), out, o.size() * 4)
                                .wait()),
           "out");
        return o;
    }
};

void check(int gu, int dt, int64_t H, int64_t FF, dpct::queue_ptr s,
           std::mt19937 &rng) {
    Setup S(gu, dt, H, FF, 6, 4, 9, rng, s);   // 6 tokens x 4 experts: cap 24 entries
    int calls = 0, bad = 0;
    size_t written = 0;
    for (const int base : {0, 7}) {
        for (const int groups : {0, 1, 2, 3, 5, 8, 24}) {
            const int ng = S.plan(groups, base, rng);
            const auto ref = S.result(true, 0, 0x00, s);
            size_t w = 0;
            for (const uint32_t v : ref) w += v != 0xFFFFFFFFu;
            written += w;
            for (const int64_t gy : {(int64_t) 0, (int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 4, (int64_t) S.cap + 3}) {
                const auto got = S.result(false, gy, (calls & 1) ? 0x7F : 0x00, s);
                ++calls;
                size_t diff = 0;
                for (size_t i = 0; i < ref.size(); ++i) diff += ref[i] != got[i];
                if (diff) {
                    std::printf("  %-8s/%-7s base %d, %d groups, grid_groups %lld: %zu of %zu floats differ  FAIL\n",
                                name_of(gu), name_of(dt), base, ng, (long long) gy, diff, ref.size());
                    ++bad;
                }
            }
        }
    }
    std::printf("%-8s/%-7s %5lld x %4lld  %d calls: %s (%zu output floats written in all)\n", name_of(gu), name_of(dt),
                (long long) H, (long long) FF, calls, bad ? "FAIL" : "bitwise equal to v1", written);
    if (written == 0) { std::printf("  nothing was written: the check checked nothing  FAIL\n"); ++bad; }
    g_fail += bad;
}

// ------------------------------------------------------------------------------------------------ --kq
// MODEL-UPDATE (Intel): the K-quant / Q5_1 / Q8_0 multi-entry split (native_expert_set_kq_split) against the per-entry
// kernels, bitwise; and both against a CPU reference in float: ggml's dequantize_row_* of the weights, the q8_1
// activations' own values (d x q), SwiGLU in float (the GPU re-quantizes h to q8_1 for the down product, the reference
// does not: that rounding is the tolerance, 3e-2 of the output's largest magnitude).
float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return (float) sycl::bit_cast<sycl::half>(h);
}
// ggml-quants.c dequantize_row_* (one block)
void dq_block(int t, const uint8_t* b, float* y) {
    switch (t) {
        case 8: {   // Q8_0: d, qs[32]
            const float d = h2f(b);
            for (int i = 0; i < 32; ++i) y[i] = d * (float) (int8_t) b[2 + i];
            break;
        }
        case 7: {   // Q5_1: d, m, qh[4], qs[16]
            const float d = h2f(b), m = h2f(b + 2);
            uint32_t qh;
            std::memcpy(&qh, b + 4, 4);
            const uint8_t* qs = b + 8;
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh0 = ((qh >> j) << 4) & 0x10, xh1 = ((qh >> (j + 12))) & 0x10;
                y[j] = (float) ((qs[j] & 0x0F) | xh0) * d + m;
                y[j + 16] = (float) ((qs[j] >> 4) | xh1) * d + m;
            }
            break;
        }
        case 12: case 13: {   // Q4_K: d, dmin, scales[12], qs[128]; Q5_K: d, dmin, scales[12], qh[32], qs[128]
            const float d = h2f(b), dmin = h2f(b + 2);
            const uint8_t* sc = b + 4;
            const uint8_t* qh = b + 16;
            const uint8_t* ql = t == 12 ? b + 16 : b + 48;
            auto smk = [&](int j, uint8_t& s, uint8_t& m) {
                if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
                else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
            };
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s, m;
                smk(is, s, m);
                const float d1 = d * s, m1 = dmin * m;
                smk(is + 1, s, m);
                const float d2 = d * s, m2 = dmin * m;
                for (int l = 0; l < 32; ++l)
                    *y++ = d1 * (float) ((ql[l] & 0xF) + (t == 13 && (qh[l] & u1) ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l)
                    *y++ = d2 * (float) ((ql[l] >> 4) + (t == 13 && (qh[l] & u2) ? 16 : 0)) - m2;
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
            break;
        }
        case 14: {   // Q6_K: ql[128], qh[64], scales[16], d
            const uint8_t* ql = b;
            const uint8_t* qh = b + 128;
            const int8_t* sc = (const int8_t*) (b + 192);
            const float d = h2f(b + 208);
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = (int8_t) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int8_t) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d * sc[is] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
            break;
        }
        default: std::fprintf(stderr, "dq_block: type %d\n", t); std::exit(2);
    }
}
std::vector<float> dq_rows(int t, const uint8_t* w, int64_t rows, int64_t n) {
    const int bv = block_values(t);
    const size_t bs = k::iq_row_bytes(t, bv);
    std::vector<float> y((size_t) rows * n);
    for (size_t b = 0; b < y.size() / bv; ++b) dq_block(t, w + b * bs, &y[b * bv]);
    return y;
}
// the CPU reference of the last plan's call: out rows ent_dst[e] (n_embd floats each)
std::vector<float> reference(const Setup& S) {
    const int64_t H = S.L.n_embd, FF = S.L.n_ff;
    std::vector<float> ref(S.out_floats, std::nanf(""));
    std::vector<float> x(H);
    for (size_t g = 0; g + 1 < S.h_start.size(); ++g) {
        const auto& blob = S.hb[S.h_blob[g]];
        const auto G = dq_rows(S.L.gu_type, blob.data(), FF, H);
        const auto U = dq_rows(S.L.gu_type, blob.data() + S.L.up_off, FF, H);
        const auto D = dq_rows(S.L.d_type, blob.data() + S.L.down_off, H, FF);
        for (int e = S.h_start[g]; e < S.h_start[g + 1]; ++e) {
            const uint8_t* xb = &S.h_xq[(size_t) S.h_tok[e] * (H / 32) * 36];
            for (int64_t i = 0; i < H; ++i) x[i] = h2f(xb + (i / 32) * 36) * (float) (int8_t) xb[(i / 32) * 36 + 4 + i % 32];
            std::vector<float> h(FF);
            for (int64_t r = 0; r < FF; ++r) {
                double gs = 0, us = 0;
                for (int64_t i = 0; i < H; ++i) { gs += (double) G[r * H + i] * x[i]; us += (double) U[r * H + i] * x[i]; }
                h[r] = (float) (gs / (1.0 + std::exp(-gs)) * us);
            }
            for (int64_t r = 0; r < H; ++r) {
                double s = 0;
                for (int64_t i = 0; i < FF; ++i) s += (double) D[r * FF + i] * h[i];
                ref[(size_t) S.h_dst[e] * H + r] = (float) s;
            }
        }
    }
    return ref;
}
void check_kq(int gu, int dt, int64_t H, int64_t FF, dpct::queue_ptr s, std::mt19937& rng) {
    Setup S(gu, dt, H, FF, 6, 4, 9, rng, s);   // groups of 1..6 entries: the split's chunks of 4 and 2 and the rest
    int calls = 0, bad = 0;
    double worst = 0;
    for (const int groups : {1, 2, 3, 5, 8}) {
        const int ng = S.plan(groups, 0, rng);
        k::native_expert_set_kq_split(false);
        const auto a = S.result(false, 0, 0x00, s);
        k::native_expert_set_kq_split(true);
        const auto b = S.result(false, 0, 0x7F, s);
        ++calls;
        size_t diff = 0;
        for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
        if (diff) {
            std::printf("  %-6s/%-6s %d groups: split vs per-entry: %zu of %zu floats differ  FAIL\n", name_of(gu), name_of(dt),
                        ng, diff, a.size());
            ++bad;
        }
        const auto ref = reference(S);
        double amax = 0, emax = 0;
        size_t n = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (std::isnan(ref[i])) continue;
            float g;
            std::memcpy(&g, &b[i], 4);
            amax = std::max(amax, (double) std::fabs(ref[i]));
            emax = std::max(emax, (double) std::fabs(g - ref[i]));
            ++n;
        }
        const double rel = amax > 0 ? emax / amax : 1.0;
        worst = std::max(worst, rel);
        if (n == 0 || !(rel < 3e-2)) {
            std::printf("  %-6s/%-6s %d groups: GPU vs CPU reference: max err %.3g of %.3g (%zu values)  FAIL\n", name_of(gu),
                        name_of(dt), ng, emax, amax, n);
            ++bad;
        }
    }
    std::printf("%-6s/%-6s %5lld x %4lld  %d calls: split %s, CPU reference worst %.2e of the largest output\n", name_of(gu),
                name_of(dt), (long long) H, (long long) FF, calls, bad ? "FAIL" : "bitwise equal to per-entry", worst);
    g_fail += bad;
}

// ------------------------------------------------------------------------------------------------ --bench
// A window's 48 calls, one per layer, each with ITS OWN experts as the layers have: copies of the Setup's blobs in one
// arena, so the weights stream from DRAM as in the engine.  (48 calls re-reading the same 28 experts - 73 MB, about an
// Ada L2 - would read part of them from L2; the engine's layers share none.)
struct Layers {
    uint8_t* arena = nullptr;
    std::vector<unsigned long long*> ptr;           // per layer: its groups' blobs (device)
    Layers(Setup& S, int ng) {
        const int per = std::max(ng, 1);
        arena = dalloc<uint8_t>(S.slot * (size_t) (48 * per));
        for (int l = 0; l < 48; ++l) {
            std::vector<unsigned long long> p;
            for (int j = 0; j < per; ++j) {
                uint8_t* d = arena + S.slot * (size_t) (l * per + j);
                ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                       d,
                       S.blobs + S.slot * (size_t)((l * per + j) % S.n_blobs),
                       S.L.bytes).wait()),
                   "blob copy");
                p.push_back((unsigned long long) d);
            }
            unsigned long long* dp = dalloc<unsigned long long>(p.size());
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                   dp, p.data(), p.size() * 8).wait()),
               "ptr");
            ptr.push_back(dp);
        }
    }
    Layers(const Layers&) = delete;
    Layers& operator=(const Layers&) = delete;
    ~Layers() {
        for (auto *p : ptr)
            DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
        sycl::free(arena, dpct::get_in_order_queue());
    }
};

// the 48 calls of one variant captured in one graph
dpct::experimental::command_graph_exec_ptr
make_graph(dpct::queue_ptr s, Setup &S, const Layers &Ls, bool v1, int64_t gy) {
    k::native_grouped_set_v1(v1);
    dpct::experimental::command_graph_ptr g;
    dpct::experimental::command_graph_exec_ptr ge;
    ck(DPCT_CHECK_ERROR(dpct::experimental::begin_recording(s)), "capture");
    for (int l = 0; l < 48; ++l)
        k::native_expert_grouped(S.L, Ls.ptr[l], S.start, S.n, S.dst, S.tok, S.cap, S.cap, S.xq, S.scr, S.out, s, gy);
    ck(DPCT_CHECK_ERROR(dpct::experimental::end_recording(s, &g)), "capture");
    ck(DPCT_CHECK_ERROR(
           ge = new sycl::ext::oneapi::experimental::command_graph<
               sycl::ext::oneapi::experimental::graph_state::executable>(
               g->finalize())),
       "instantiate");
    ck(DPCT_CHECK_ERROR(delete (g)), "graph");
    k::native_grouped_set_v1(false);
    return ge;
}

// one launch of the graph: microseconds per call
float launch_us(dpct::queue_ptr s,
                dpct::experimental::command_graph_exec_ptr ge,
                dpct::event_ptr e0, dpct::event_ptr e1) {
    /*
    DPCT1024: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    ck(DPCT_CHECK_ERROR(dpct::sync_barrier(e0, s)), "record");
    ck(DPCT_CHECK_ERROR(s->ext_oneapi_graph(*ge)), "launch");
    /*
    DPCT1024: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    ck(DPCT_CHECK_ERROR(dpct::sync_barrier(e1, s)), "record");
    ck(DPCT_CHECK_ERROR(e1->wait_and_throw()), "bench");
    float ms = 0;
    ck(DPCT_CHECK_ERROR(
           ms = (e1->get_profiling_info<
                     sycl::info::event_profiling::command_end>() -
                 e0->get_profiling_info<
                     sycl::info::event_profiling::command_start>()) /
                1000000.0f),
       "elapsed");
    return 1e3f * ms / 48.0f;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// The two variants' launches ALTERNATE (v1-new, new-v1, ...) after a warm-up, and each reports its median: a card
// under sustained load steps its clock down after a fraction of a second (an RTX 4080 SUPER went from 80 to 89 us
// per gate/up call within one run), and whichever variant ran before the step would look faster.
constexpr int kWarmPairs = 20, kPairs = 40;

void bench(dpct::queue_ptr s, std::mt19937 &rng) {
    std::printf("\n--bench: microseconds per call, a window's 48 calls in a graph, each layer with its own experts (the\n"
                "weights stream from DRAM as in the engine); v1 / new (grid_groups): medians of %d launches each,\n"
                "alternating, after %d warm-up pairs - idle GPU assumed\n", kPairs, kWarmPairs);
    // a verify window's shape: 4 tokens x 10 experts, 2560 x 768 IQ3_S / IQ4_XS experts
    Setup S(21, 23, 2560, 768, 4, 10, 32, rng, s);
    std::vector<int> vram(16, 1);
    vram.insert(vram.end(), 12, 2);                                   // 28 groups, 40 entries
    const struct { const char* what; std::vector<int> sizes; int64_t gy; } rows[] = {
        {"no group (the PCIe call at pcie_frac 0)", {}, 4},
        {"1 group of 2 entries (a PCIe call)", {2}, 4},
        {"28 groups, 40 entries (a VRAM call)", vram, 0},
    };
    dpct::event_ptr e0, e1;
    ck(DPCT_CHECK_ERROR(e0 = new sycl::event()), "event");
    ck(DPCT_CHECK_ERROR(e1 = new sycl::event()), "event");
    for (const auto& r : rows) {
        const int ng = S.plan_sizes(r.sizes, 0, rng);
        const Layers Ls(S, ng);                                       // 48 x 28 experts: 3.7 GB for the VRAM call
        dpct::experimental::command_graph_exec_ptr ga = make_graph(s, S, Ls,
                                                                   true, 0),
                                                   gb = make_graph(s, S, Ls,
                                                                   false, r.gy);
        std::vector<float> ta, tb;
        for (int i = 0; i < kWarmPairs + kPairs; ++i) {
            const bool a_first = i % 2 == 0;
            const float x = launch_us(s, a_first ? ga : gb, e0, e1), y = launch_us(s, a_first ? gb : ga, e0, e1);
            if (i < kWarmPairs) continue;
            ta.push_back(a_first ? x : y);
            tb.push_back(a_first ? y : x);
        }
        ck(DPCT_CHECK_ERROR(delete (ga)), "graph");
        ck(DPCT_CHECK_ERROR(delete (gb)), "graph");
        std::printf("  %-42s %8.2f / %8.2f us (grid_groups %lld)\n", r.what, median(ta), median(tb), (long long) r.gy);
    }
    dpct::destroy_event(e0);
    dpct::destroy_event(e1);
}

// ------------------------------------------------------------------------------------------------ --kqbench [NL [host]]
// Unsloth's formats against the IQ packs' at the model's shapes (2560 x 640): microseconds per grouped call and the
// expert bytes it reads per microsecond.  NL layers (default 16), each with its own experts, so the weights stream from
// DRAM; a call reads each of its groups' whole blobs (gate + up + down) once.  Shapes: one token (10 groups of 1), a
// window of 2 tokens (16 groups, 20 entries) and of 4 (28 groups, 40 entries; native_grouped_parity's --bench).
// STRATA_GU_LANES / STRATA_DOWN_LANES pick the lane split as in the engine.
void kq_bench(dpct::queue_ptr s, std::mt19937& rng, int nl, bool host) {
    const int pairs[][2] = {{21, 20}, {23, 20}, {18, 42}, {12, 7}, {12, 8}, {13, 8},
#ifdef STRATA_Q6K_EXPERTS
                            {14, 8},
#endif
                            {8, 8}};
    std::vector<int> w1(10, 1), w2(12, 1), w4(16, 1);
    w2.insert(w2.end(), 4, 2);                                  // 16 groups, 20 entries
    w4.insert(w4.end(), 12, 2);                                 // 28 groups, 40 entries
    const struct { const char* what; int T; const std::vector<int>* sizes; } shapes[] = {
        {"1 token, 10x1", 1, &w1}, {"2 tokens, 16 groups / 20", 2, &w2}, {"4 tokens, 28 groups / 40", 4, &w4}};
    dpct::event_ptr e0 = new sycl::event(), e1 = new sycl::event();
    std::printf("\n--kqbench: %d layers per graph, each its own experts (%s); us per call (median of 40), expert MB per call, GB/s\n",
                nl, host ? "pinned host memory" : "device memory");
    for (const auto& pr : pairs) {
        for (const auto& sh : shapes) {
            Setup S(pr[0], pr[1], 2560, 640, sh.T, 10, 28, rng, s);
            const int ng = S.plan_sizes(*sh.sizes, 0, rng);
            // host: the arena in pinned host memory, read by the kernels over PCIe (the B70's mirrored experts)
            uint8_t* arena = host ? (uint8_t*) sycl::malloc_host(S.slot * (size_t) (nl * ng) + 256, dpct::get_in_order_queue())
                                  : dalloc<uint8_t>(S.slot * (size_t) (nl * ng));
            std::vector<unsigned long long*> lp;
            for (int l = 0; l < nl; ++l) {
                std::vector<unsigned long long> p;
                for (int j = 0; j < ng; ++j) {
                    uint8_t* d = arena + S.slot * (size_t) (l * ng + j);
                    dpct::get_in_order_queue().memcpy(d, S.blobs + S.slot * (size_t) (j % S.n_blobs), S.L.bytes).wait();
                    p.push_back((unsigned long long) d);
                }
                unsigned long long* dp = dalloc<unsigned long long>(p.size());
                dpct::get_in_order_queue().memcpy(dp, p.data(), p.size() * 8).wait();
                lp.push_back(dp);
            }
            dpct::experimental::command_graph_ptr g;
            dpct::experimental::begin_recording(s);
            for (int l = 0; l < nl; ++l)
                k::native_expert_grouped(S.L, lp[l], S.start, S.n, S.dst, S.tok, S.cap, S.cap, S.xq, S.scr, S.out, s, 0);
            dpct::experimental::end_recording(s, &g);
            auto* ge = new sycl::ext::oneapi::experimental::command_graph<
                sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());
            delete g;
            std::vector<float> t;
            for (int i = 0; i < 60; ++i) {
                const float us = launch_us(s, ge, e0, e1) * 48.0f / (float) nl;
                if (i >= 20) t.push_back(us);
            }
            delete ge;
            const double mb = (double) ng * (double) S.L.bytes / 1e6, us = median(t);
            std::printf("  %-7s/%-6s %-26s %8.2f us  %6.2f MB  %6.1f GB/s\n", name_of(pr[0]), name_of(pr[1]), sh.what, us,
                        mb, mb * 1e3 / us);
            for (auto* p : lp) sycl::free(p, dpct::get_in_order_queue());
            sycl::free(arena, dpct::get_in_order_queue());
        }
    }
    dpct::destroy_event(e0);
    dpct::destroy_event(e1);
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    dpct::queue_ptr s;
    ck(DPCT_CHECK_ERROR(s = dpct::get_current_device().create_queue(true)),
       "stream");
    std::mt19937 rng(316);
    if (argc > 1 && std::string(argv[1]) == "--kqbench") {   // timings only (the checks: run without arguments)
        kq_bench(s, rng, argc > 2 ? std::atoi(argv[2]) : 16, argc > 3 && std::string(argv[3]) == "host");
        dpct::get_current_device().destroy_queue(s);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--kq") {   // the K-quant split and the CPU reference only
        const int pairs[][2] = {{12, 7}, {12, 8}, {13, 8}, {14, 8}, {8, 8}, {13, 7}};
        for (const auto& pr : pairs) check_kq(pr[0], pr[1], 2560, 640, s, rng);   // the model's shapes
        for (const auto& pr : pairs) check_kq(pr[0], pr[1], 512, 256, s, rng);
        std::printf("native_grouped_parity --kq: %d failures\n", g_fail);
        dpct::get_current_device().destroy_queue(s);
        return g_fail ? 1 : 0;
    }
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 42, 12, 13,
#ifdef STRATA_Q6K_EXPERTS
                   14,
#endif
                   8})        // STRATA_GU_FMTS
        for (int dt : {20, 23, 42, 7, 8}) check(gu, dt, 512, 256, s, rng);   // STRATA_D_FMTS; IQ4_XS: n_ff % 256
    check(21, 20, 2560, 640, s, rng);                                   // a model's shapes
    check(21, 23, 2560, 768, s, rng);
    if (do_bench) bench(s, rng);
    std::printf("native_grouped_parity: %d failures\n", g_fail);
    dpct::get_current_device().destroy_queue(s);
    return g_fail ? 1 : 0;
}
