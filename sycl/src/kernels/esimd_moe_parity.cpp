// esimd_moe_parity (151-b70): the fused int8 DPAS expert products (prefill/esimd_moe.dp.cpp) against
//   (a) the engine's dequantize + oneMKL path (FP16 weights, FP16 activations), the path they replace,
//   (b) an FP64 host reference from the blobs with the unquantized activations,
//   (c) an exact host reference on the SAME int8 activations the device produced (isolates the kernel arithmetic from the
//       activation quantization: only the float accumulation order differs),
// for every gate/up x down type pair the kernels handle, and times both paths per expert.
//   esimd_moe_parity [mean_rows=80] [experts=16] [reps=10] [seed=1] [gate_up_type down_type]
//   types: gate/up 21 IQ3_S, 18 IQ3_XXS, 22 IQ2_S; down 20 IQ4_NL, 42 Q2_0.
// Exit status 1 when (c) is off by more than 2e-5 of the largest value or (b) exceeds the int8 activation error budget.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/prefill/esimd_moe.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "../prefill/iq_tables.inc"

namespace i8 = strata::prefill::i8;
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

static int block_bytes(int t) { return t == 21 ? 110 : t == 18 ? 98 : t == 22 ? 82 : 18; }
static int block_vals(int t) { return (t == 21 || t == 18 || t == 22) ? 256 : t == 42 ? 64 : 32; }

// host decode of one row (K values) -> double
static void dec_row(int t, const uint8_t* row, int K, double* out) {
    const int bb = block_bytes(t), bv = block_vals(t);
    for (int b = 0; b < K / bv; ++b) {
        const uint8_t* blk = row + (size_t) b * bb;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        const double d = h2f(dh);
        double* o = out + (size_t) b * bv;
        if (t == 21) {
            const uint8_t *qs = blk + 2, *qh = blk + 66, *sgn = blk + 74, *scl = blk + 106;
            for (int ib = 0; ib < 8; ++ib) {
                const double dd = d * (1 + 2 * ((scl[ib / 2] >> (4 * (ib % 2))) & 0xf));
                for (int r = 0; r < 8; ++r) {
                    const int idx = qs[8 * ib + r] | (((qh[ib] >> r) & 1) << 8);
                    const uint8_t* g = (const uint8_t*) &iq3s_grid[idx];
                    const uint8_t sb = sgn[4 * ib + r / 2];
                    for (int j = 0; j < 4; ++j) o[ib * 32 + 4 * r + j] = dd * ((sb >> (4 * (r & 1) + j) & 1) ? -(int) g[j] : (int) g[j]);
                }
            }
        } else if (t == 18) {
            const uint8_t* qs = blk + 2;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux; std::memcpy(&aux, blk + 66 + 4 * ib, 4);
                const double dd = d * (0.5 + (aux >> 28)) * 0.5;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sb = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                    const uint8_t *g1 = (const uint8_t*) &iq3xxs_grid[qs[8 * ib + 2 * l]], *g2 = (const uint8_t*) &iq3xxs_grid[qs[8 * ib + 2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + 8 * l + j] = dd * ((sb >> j & 1) ? -(int) g1[j] : (int) g1[j]);
                        o[ib * 32 + 8 * l + 4 + j] = dd * ((sb >> (4 + j) & 1) ? -(int) g2[j] : (int) g2[j]);
                    }
                }
            }
        } else if (t == 22) {
            const uint8_t *qs = blk + 2, *qh = blk + 66, *scl = blk + 74, *sgn = blk + 34;
            for (int ib = 0; ib < 8; ++ib)
                for (int l = 0; l < 4; ++l) {
                    const int idx = qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
                    const uint8_t* g = (const uint8_t*) &iq2s_grid[idx];
                    const double dd = d * (0.5 + ((scl[ib] >> (4 * (l / 2))) & 0xf)) * 0.25;
                    const uint8_t sb = sgn[4 * ib + l];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + 8 * l + j] = dd * ((sb >> j & 1) ? -(int) g[j] : (int) g[j]);
                }
        } else if (t == 20) {
            for (int j = 0; j < 16; ++j) {
                o[j] = d * kvalues_iq4nl[blk[2 + j] & 15];
                o[16 + j] = d * kvalues_iq4nl[blk[2 + j] >> 4];
            }
        } else {   // 42 Q2_0
            for (int i = 0; i < 64; ++i) o[i] = d * (((blk[2 + i / 4] >> ((i % 4) * 2)) & 3) - 1);
        }
    }
}

struct Result { bool ok; double t_ref, t_fused; };

static Result run_case(sycl::queue* q, dpct::blas::descriptor_ptr hd, int gt, int dt, int mean, int n, int reps, int seed) {
    const int K = 2560, NFF = 640, NE = 2560;
    std::mt19937 rng(seed);
    std::vector<int32_t> cnt(n);
    std::gamma_distribution<double> gd(1.5, mean / 1.5);
    int tot = 0;
    for (auto& c : cnt) { c = std::max(1, (int) gd(rng)); tot += c; }
    if (n >= 3) { cnt[0] = 1; cnt[1] = 7; cnt[2] = 9; tot = 0; for (int c : cnt) tot += c; }   // pad edge cases
    i8::Layout lo;
    i8::make_layout(lo, cnt.data(), n);

    const size_t rowb_gu = (size_t) (K / block_vals(gt)) * block_bytes(gt), rowb_dn = (size_t) (NFF / block_vals(dt)) * block_bytes(dt);
    const size_t gate_b = NFF * rowb_gu, down_b = NE * rowb_dn, blob_b = 2 * gate_b + down_b;
    std::vector<std::vector<uint8_t>> hb(n);
    std::vector<uint8_t*> db(n);
    for (int i = 0; i < n; ++i) {
        hb[i].resize(blob_b);
        for (auto& x : hb[i]) x = (uint8_t) rng();
        for (size_t o = 0; o < 2 * gate_b; o += block_bytes(gt)) { const uint16_t d = f2h(0.0004f + 0.00002f * (rng() % 50)); std::memcpy(&hb[i][o], &d, 2); }
        for (size_t o = 2 * gate_b; o < blob_b; o += block_bytes(dt)) { const uint16_t d = f2h(0.0004f + 0.00002f * (rng() % 50)); std::memcpy(&hb[i][o], &d, 2); }
        db[i] = sycl::malloc_device<uint8_t>(blob_b + 256, *q);
        q->memcpy(db[i], hb[i].data(), blob_b);
    }
    std::vector<uint16_t> hx((size_t) tot * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hx) { float v = nd(rng) * 0.5f; if (rng() % 97 == 0) v *= 12.f; x = f2h(v); }   // a few outliers
    uint16_t* dxh = sycl::malloc_device<uint16_t>(hx.size(), *q);
    q->memcpy(dxh, hx.data(), hx.size() * 2);
    int8_t* Xq = sycl::malloc_device<int8_t>(i8::xq_bytes(lo, K), *q);
    float* dX = sycl::malloc_device<float>(i8::dx_bytes(lo, K) / 4, *q);
    int8_t* Hq = sycl::malloc_device<int8_t>(i8::xq_bytes(lo, NFF), *q);
    float* dH = sycl::malloc_device<float>(i8::dx_bytes(lo, NFF) / 4, *q);
    float *GU = sycl::malloc_device<float>((size_t) tot * 2 * NFF, *q), *Dm = sycl::malloc_device<float>((size_t) tot * NE, *q);
    float *GUr = sycl::malloc_device<float>((size_t) tot * 2 * NFF, *q), *Dmr = sycl::malloc_device<float>((size_t) tot * NE, *q);
    uint16_t* Hh = sycl::malloc_device<uint16_t>((size_t) tot * NFF, *q);
    uint16_t* dwgu = sycl::malloc_device<uint16_t>((size_t) 2 * NFF * K, *q);
    uint16_t* dwd = sycl::malloc_device<uint16_t>((size_t) NE * NFF, *q);
    q->wait();
    std::vector<const uint8_t*> pb(n);
    for (int i = 0; i < n; ++i) pb[i] = db[i];

    auto fused = [&]() {
        i8::quantize_rows(dxh, lo, K, Xq, dX, q);
        if (!i8::gate_up(gt, pb.data(), gate_b, NFF, lo, K, Xq, dX, GU, q)) { std::printf("gate_up refused (type %d)\n", gt); std::exit(2); }
        i8::swiglu_quantize(GU, lo, NFF, Hq, dH, q);
        if (!i8::down(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hq, dH, Dm, q)) { std::printf("down refused (type %d)\n", dt); std::exit(2); }
    };
    auto ref = [&]() {
        int64_t o = 0;
        for (int i = 0; i < n; ++i) {
            strata::kernels::iq_dequant_expert_f16(gt, dt, db[i], db[i] + gate_b, db[i] + 2 * gate_b, NFF, K, dwgu, dwd, q);
            const float a = 1.f, b = 0.f;
            dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, 2 * NFF, cnt[i], K, &a, dwgu,
                             dpct::library_data_t::real_half, K, dxh + o * K, dpct::library_data_t::real_half, K, &b,
                             GUr + o * 2 * NFF, dpct::library_data_t::real_float, 2 * NFF, dpct::compute_type::f32);
            strata::prefill::swiglu_interleaved(GUr + o * 2 * NFF, Hh + o * NFF, cnt[i], q);
            dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, NE, cnt[i], NFF, &a, dwd,
                             dpct::library_data_t::real_half, NFF, Hh + o * NFF, dpct::library_data_t::real_half, NFF, &b,
                             Dmr + o * NE, dpct::library_data_t::real_float, NE, dpct::compute_type::f32);
            o += cnt[i];
        }
    };
    auto tm = [&](auto f) {
        f(); q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5) { f(); q->wait(); }   // GPU clocks up
        const auto a = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) f();
        q->wait();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / reps;
    };

    fused(); q->wait();
    std::vector<int8_t> hXq(i8::xq_bytes(lo, K));
    std::vector<float> hdX(i8::dx_bytes(lo, K) / 4), hGU((size_t) tot * 2 * NFF);
    q->memcpy(hXq.data(), Xq, hXq.size());
    q->memcpy(hdX.data(), dX, hdX.size() * 4);
    q->memcpy(hGU.data(), GU, hGU.size() * 4).wait();
    const int nkb = K / 32;
    auto xq_at = [&](int tile, int r, int k) { return (int) hXq[(((size_t) tile * nkb + k / 32) * 8 + r) * 32 + (k % 32)]; };
    auto dx_at = [&](int tile, int r, int kb) { return (double) hdX[((size_t) tile * nkb + kb) * 8 + r]; };
    bool ok = true;
    std::vector<double> wrow(K);
    double err_c = 0, mx_c = 0, err_b = 0, rms_bref = 0;
    int sampled = 0;
    for (int e = 0; e < n; ++e)
        for (int rl = 0; rl < cnt[e]; rl += (cnt[e] > 12 ? 5 : 1)) {
            const int tile = lo.tile0[e] + rl / 8, r = rl % 8, row = lo.row0[e] + rl;
            for (int v = 0; v < 2 * NFF; v += 37) {
                dec_row(gt, &hb[e][(size_t) (v & 1 ? gate_b : 0) + (size_t) (v >> 1) * rowb_gu], K, wrow.data());
                double refc = 0, refb = 0;
                for (int kb = 0; kb < nkb; ++kb) {
                    double s = 0;
                    for (int k = 0; k < 32; ++k) s += (double) xq_at(tile, r, kb * 32 + k) * wrow[kb * 32 + k];
                    refc += s * dx_at(tile, r, kb);
                }
                for (int k = 0; k < K; ++k) refb += (double) h2f(hx[(size_t) row * K + k]) * wrow[k];
                const double got = hGU[(size_t) row * 2 * NFF + v];
                err_c = std::max(err_c, std::fabs(got - refc)); mx_c = std::max(mx_c, std::fabs(refc));
                err_b += (got - refb) * (got - refb); rms_bref += refb * refb;
                ++sampled;
            }
        }
    const bool ok1 = err_c <= 2e-5 * mx_c && std::sqrt(err_b / rms_bref) < 0.02;
    ok = ok && ok1;
    std::printf("  gate/up type %2d: exact-int8 ref max err %.2e of %.2e (%.1e)  vs FP64 true rms rel err %.2e (%d samples)  %s\n", gt,
                err_c, mx_c, err_c / mx_c, std::sqrt(err_b / rms_bref), sampled, ok1 ? "OK" : "FAIL");
    {
        std::vector<int8_t> hHq(i8::xq_bytes(lo, NFF));
        std::vector<float> hdH(i8::dx_bytes(lo, NFF) / 4), hDm((size_t) tot * NE);
        q->memcpy(hHq.data(), Hq, hHq.size());
        q->memcpy(hdH.data(), dH, hdH.size() * 4);
        q->memcpy(hDm.data(), Dm, hDm.size() * 4).wait();
        const int nkh = NFF / 32;
        double ec = 0, mc = 0, eb = 0, rb = 0, hq_bad = 0;
        std::vector<double> drow(NFF);
        int smp = 0;
        for (int e = 0; e < n; ++e)
            for (int rl = 0; rl < cnt[e]; rl += (cnt[e] > 12 ? 5 : 1)) {
                const int tile = lo.tile0[e] + rl / 8, r = rl % 8, row = lo.row0[e] + rl;
                std::vector<double> H(NFF);
                for (int j = 0; j < NFF; ++j) {
                    const double g = hGU[(size_t) row * 2 * NFF + 2 * j], u = hGU[(size_t) row * 2 * NFF + 2 * j + 1];
                    H[j] = g / (1.0 + std::exp(-g)) * u;
                }
                for (int kb = 0; kb < nkh; ++kb) {
                    double amax = 0; for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(H[kb * 32 + k]));
                    const double dd = hdH[((size_t) tile * nkh + kb) * 8 + r];
                    if (std::fabs(dd - amax / 127.0) > 1e-5 * amax + 1e-30) hq_bad += 1;
                }
                for (int v = 0; v < NE; v += 41) {
                    dec_row(dt, &hb[e][2 * gate_b + (size_t) v * rowb_dn], NFF, drow.data());
                    double refc = 0, refb = 0;
                    for (int kb = 0; kb < nkh; ++kb) {
                        double s = 0;
                        for (int k = 0; k < 32; ++k) s += (double) hHq[(((size_t) tile * nkh + kb) * 8 + r) * 32 + k] * drow[kb * 32 + k];
                        refc += s * hdH[((size_t) tile * nkh + kb) * 8 + r];
                    }
                    for (int k = 0; k < NFF; ++k) refb += H[k] * drow[k];
                    const double got = hDm[(size_t) row * NE + v];
                    ec = std::max(ec, std::fabs(got - refc)); mc = std::max(mc, std::fabs(refc));
                    eb += (got - refb) * (got - refb); rb += refb * refb;
                    ++smp;
                }
            }
        const bool ok2 = ec <= 2e-5 * mc && std::sqrt(eb / rb) < 0.03 && hq_bad == 0;
        ok = ok && ok2;
        std::printf("  down type %2d   : exact-int8 ref max err %.2e of %.2e (%.1e)  vs FP64 true rms rel err %.2e, swiglu scale mismatches %.0f (%d samples)  %s\n",
                    dt, ec, mc, ec / mc, std::sqrt(eb / rb), hq_bad, smp, ok2 ? "OK" : "FAIL");
    }
    ref(); q->wait();
    {
        std::vector<float> a((size_t) tot * NE), b((size_t) tot * NE);
        q->memcpy(a.data(), Dmr, a.size() * 4);
        q->memcpy(b.data(), Dm, b.size() * 4).wait();
        double e2 = 0, r2 = 0;
        for (size_t i = 0; i < a.size(); ++i) { e2 += (double) (a[i] - b[i]) * (a[i] - b[i]); r2 += (double) a[i] * a[i]; }
        std::printf("  chain vs dequantize + oneMKL (FP16 path): rms rel diff %.2e (rows %d)\n", std::sqrt(e2 / r2), tot);
    }
    Result res{ok, tm(ref) / n, tm(fused) / n};
    const double t_q = tm([&] { i8::quantize_rows(dxh, lo, K, Xq, dX, q); }) / n;
    const double t_gu = tm([&] { i8::gate_up(gt, pb.data(), gate_b, NFF, lo, K, Xq, dX, GU, q); }) / n;
    const double t_sw = tm([&] { i8::swiglu_quantize(GU, lo, NFF, Hq, dH, q); }) / n;
    const double t_dn = tm([&] { i8::down(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hq, dH, Dm, q); }) / n;
    std::printf("  rows %d (%d experts, mean %.0f): dequant+oneMKL %.1f us/expert | fused int8 %.1f us/expert (%.2fx)  [quantize %.1f, gate/up %.1f, swiglu+q %.1f, down %.1f]\n",
                tot, n, (double) tot / n, res.t_ref, res.t_fused, res.t_ref / res.t_fused, t_q, t_gu, t_sw, t_dn);
    for (int i = 0; i < n; ++i) sycl::free(db[i], *q);
    sycl::free(dxh, *q); sycl::free(Xq, *q); sycl::free(dX, *q); sycl::free(Hq, *q); sycl::free(dH, *q); sycl::free(GU, *q);
    sycl::free(Dm, *q); sycl::free(GUr, *q); sycl::free(Dmr, *q); sycl::free(Hh, *q); sycl::free(dwgu, *q); sycl::free(dwd, *q);
    return res;
}

int main(int argc, char** argv) {
    const int mean = argc > 1 ? std::atoi(argv[1]) : 80;
    const int n = argc > 2 ? std::atoi(argv[2]) : 16;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int seed = argc > 4 ? std::atoi(argv[4]) : 1;
    sycl::queue* q = &dpct::get_in_order_queue();
    if (!i8::available(q)) { std::printf("esimd_moe_parity: SKIP (no ESIMD device with the s8 8x16x32 DPAS)\n"); return 0; }
    dpct::blas::descriptor_ptr hd = new dpct::blas::descriptor();
    hd->set_queue(q);
    std::vector<std::pair<int, int>> cases = {{21, 20}, {18, 20}, {22, 20}, {21, 42}, {18, 42}, {22, 42}};
    if (argc > 6) cases = {{std::atoi(argv[5]), std::atoi(argv[6])}};
    int fails = 0;
    for (auto [gt, dt] : cases) {
        std::printf("gate/up %d, down %d, mean %d rows, %d experts\n", gt, dt, mean, n);
        fails += !run_case(q, hd, gt, dt, mean, n, reps, seed).ok;
    }
    std::printf("%s\n", fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
