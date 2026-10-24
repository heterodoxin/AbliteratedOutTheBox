// xmx_gemm_bench: the XMX quantized GEMM against dequant + oneMKL on random experts.
//   xmx_gemm_bench [ty=18] [M=96] [mode=gu|down] [reps=50]
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
int main(int argc, char** argv) {
    const int ty = argc > 1 ? std::atoi(argv[1]) : 18;
    const int M = argc > 2 ? std::atoi(argv[2]) : 96;
    const std::string mode = argc > 3 ? argv[3] : "gu";
    const int reps = argc > 4 ? std::atoi(argv[4]) : 50;
    const bool gu = mode == "gu";
    const int64_t K = gu ? 2560 : 640, n_out = gu ? 1280 : 2560, n_ff = 640;
    const int blk_elems = (ty == 20) ? 32 : (ty == 42) ? 64 : 256;
    const size_t blk_bytes = strata::kernels::iq_row_bytes(ty, blk_elems);
    const size_t row_bytes = strata::kernels::iq_row_bytes(ty, K);
    if (row_bytes == 0) { std::printf("type %d not supported\n", ty); return 2; }
    const int64_t rows_per_blob = gu ? n_ff : n_out;
    std::mt19937 rng(7);
    auto blob = [&]() {
        std::vector<uint8_t> b(rows_per_blob * row_bytes);
        for (auto& x : b) x = (uint8_t) rng();
        for (size_t o = 0; o + 1 < b.size(); o += blk_bytes) { b[o] = 0x00; b[o + 1] = 0x3C; }   // d = 1.0 in every block
        return b;
    };
    std::vector<uint8_t> hg = blob(), hu = gu ? blob() : std::vector<uint8_t>();
    std::vector<uint16_t> hx((size_t) M * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hx) x = f2h(nd(rng));
    sycl::queue* q = &dpct::get_in_order_queue();
    uint8_t* dg = sycl::malloc_device<uint8_t>(hg.size(), *q);
    uint8_t* du = gu ? sycl::malloc_device<uint8_t>(hu.size(), *q) : nullptr;
    uint16_t* dx = sycl::malloc_device<uint16_t>(hx.size(), *q);
    uint16_t* dw = sycl::malloc_device<uint16_t>((size_t) n_out * K, *q);
    float* y_ref = sycl::malloc_device<float>((size_t) M * n_out, *q);
    float* y_xmx = sycl::malloc_device<float>((size_t) M * n_out, *q);
    q->memcpy(dg, hg.data(), hg.size()).wait();
    if (gu) q->memcpy(du, hu.data(), hu.size()).wait();
    q->memcpy(dx, hx.data(), hx.size() * 2).wait();
    // the reference path: the engine's dequant kernels + oneMKL (what prefill.cpp runs today)
    dpct::blas::descriptor_ptr h = new dpct::blas::descriptor();
    h->set_queue(q);
    auto dequant = [&]() {
        if (gu) strata::kernels::iq_dequant_gu_f16(ty, dg, du, n_ff, K, dw, q);
        else strata::kernels::iq_dequant_f16(ty, dg, n_out * K, dw, q);
    };
    auto mkl = [&]() {
        const float alpha = 1.0f, beta = 0.0f;
        dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) n_out, M, (int) K, &alpha, dw,
                         dpct::library_data_t::real_half, (int) K, dx, dpct::library_data_t::real_half, (int) K, &beta, y_ref,
                         dpct::library_data_t::real_float, (int) n_out, dpct::compute_type::f32);
    };
    dequant(); mkl(); q->wait();
    if (!strata::kernels::xmx_gemm_iq(ty, dg, du, K, (int) n_out, dx, M, y_xmx, n_out, q)) { std::printf("xmx_gemm_iq refused\n"); return 3; }
    q->wait();
    std::vector<float> r((size_t) M * n_out), x((size_t) M * n_out);
    q->memcpy(r.data(), y_ref, r.size() * 4).wait();
    q->memcpy(x.data(), y_xmx, x.size() * 4).wait();
    double maxref = 0, maxerr = 0, sumerr = 0; size_t bad = 0, nonfinite = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        if (!std::isfinite(r[i]) || !std::isfinite(x[i])) { ++nonfinite; continue; }
        maxref = std::max(maxref, (double) std::fabs(r[i]));
        const double e = std::fabs((double) r[i] - x[i]);
        maxerr = std::max(maxerr, e); sumerr += e;
        if (e > 1e-2 * (1.0 + std::fabs(r[i]))) ++bad;
    }
    std::printf("ty %d %s M %d: max |ref| %.3g, max err %.3g, mean err %.3g, %zu of %zu outside 1%% (nonfinite %zu)\n", ty,
                mode.c_str(), M, maxref, maxerr, sumerr / r.size(), bad, r.size(), nonfinite);
    auto time = [&](auto&& fn) {
        for (int i = 0; i < 3; ++i) fn(); q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        q->wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    const double t_dq = time([&] { dequant(); }), t_mkl = time([&] { mkl(); }), t_both = time([&] { dequant(); mkl(); });
    const double t_xmx = time([&] { strata::kernels::xmx_gemm_iq(ty, dg, du, K, (int) n_out, dx, M, y_xmx, n_out, q); });
    const double flop = 2.0 * M * n_out * K;
    std::printf("dequant %.3f ms + mkl %.3f ms = %.3f ms (%.0f GFLOP/s on the GEMM)  |  xmx %.3f ms (%.0f GFLOP/s, %.2fx)\n", t_dq,
                t_mkl, t_both, flop / t_mkl / 1e6, t_xmx, flop / t_xmx / 1e6, t_both / t_xmx);
    return bad > r.size() / 1000 ? 1 : 0;
}
