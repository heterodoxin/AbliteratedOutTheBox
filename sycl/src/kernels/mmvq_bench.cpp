// sycl/src/kernels/mmvq_bench.cpp - SYCL port: time the dense native_mmvq kernels on random weights.
//   mmvq_bench [n_in] [n_out] [ncols]      (weights are random bytes: dequant cost is data-independent)
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
int main(int argc, char** argv) {
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2560, n_out = argc > 2 ? std::atoi(argv[2]) : 2560, ncols = argc > 3 ? std::atoi(argv[3]) : 4;
    sycl::queue* s = &dpct::get_in_order_queue();
    // Q6_K: 210 bytes per 256 weights
    const size_t wbytes = (size_t) n_out * (n_in / 256) * 210;
    std::vector<uint8_t> hw(wbytes);
    for (size_t i = 0; i < wbytes; ++i) hw[i] = (uint8_t) (i * 2654435761u >> 13);
    for (size_t b = 0; b < wbytes / 210; ++b) { hw[b * 210 + 208] = 0x00; hw[b * 210 + 209] = 0x3c; }   // d = 1.0 (finite)
    std::vector<float> hx((size_t) ncols * n_in);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((i * 7919) % 1000) / 500.f - 1.f;
    void* w = sycl::malloc_device(wbytes, *s);
    float* x = sycl::malloc_device<float>(hx.size(), *s);
    void* xq = sycl::malloc_device((size_t) ncols * (n_in / 32) * 36, *s);
    float* y = sycl::malloc_device<float>((size_t) ncols * n_out, *s);
    s->memcpy(w, hw.data(), wbytes).wait(); s->memcpy(x, hx.data(), hx.size() * 4).wait();
    strata::kernels::native_quantize_q8_1(x, xq, n_in, ncols, s);
    {   // correctness: the 16-byte kernel against the shared one
        std::vector<float> y0((size_t) ncols * n_out), y1(y0.size());
        strata::kernels::native_mmvq_set_q6k_wide(false);
        strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, ncols, s); s->wait(); s->memcpy(y0.data(), y, y0.size() * 4).wait();
        strata::kernels::native_mmvq_set_q6k_wide(true);
        strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, ncols, s); s->wait(); s->memcpy(y1.data(), y, y1.size() * 4).wait();
        double num = 0, den = 0; for (size_t i = 0; i < y0.size(); ++i) { num += std::fabs((double) y0[i] - y1[i]); den += std::fabs((double) y0[i]); }
        std::printf("wide vs shared: rel %.3e over %zu outputs (first %g vs %g)\n", num / (den > 0 ? den : 1), y0.size(), y0[0], y1[0]);
        strata::kernels::native_mmvq_set_q6k_wide(std::getenv("STRATA_MMVQ_WIDE") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_WIDE")) != 0);
    }
    {   // warm up until the clocks are up (~300 ms of work), then time
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 300) {
            for (int i = 0; i < 20; ++i) strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, ncols, s);
            s->wait();
        }
    }
    const int it = 400;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; ++i) strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, ncols, s);
    s->wait();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    std::vector<float> hy((size_t) ncols * n_out);
    s->memcpy(hy.data(), y, hy.size() * 4).wait();
    unsigned long long cs = 1469598103934665603ull;
    for (float v : hy) { unsigned u; std::memcpy(&u, &v, 4); cs = (cs ^ u) * 1099511628211ull; }
    std::printf("q6_k mmvq %d x %d, %d cols: %.1f us  (%.1f GB/s of weights)  checksum %016llx\n", n_in, n_out, ncols, us, wbytes / us / 1e3, cs);
    // STRATA_BENCH_PLANAR=1 (#2340): the same matrix as STRATA_Q6_PLANAR planes, bitwise against the GGUF-layout result above; then a
    // gathered row subset (the MTP draft head's case) against the same rows of the full result. Exit 1 on any difference.
    const char* pe = std::getenv("STRATA_BENCH_PLANAR");
    if (pe && std::atoi(pe) != 0) {
        void* w2 = sycl::malloc_device(wbytes, *s);
        s->memcpy(w2, hw.data(), wbytes).wait();
        void* wp = strata::kernels::native_q6_k_planar_adopt(w2, n_in, n_out, "bench matrix");
        if (!strata::kernels::native_q6_k_is_planar(wp)) { std::printf("planar: adopt declined\n"); return 1; }
        float* y2 = sycl::malloc_device<float>((size_t) ncols * n_out, *s);
        strata::kernels::native_q6_k_mmvq(wp, xq, y2, n_in, n_out, ncols, s); s->wait();
        std::vector<float> hp((size_t) ncols * n_out);
        s->memcpy(hp.data(), y2, hp.size() * 4).wait();
        const bool same = std::memcmp(hp.data(), hy.data(), hp.size() * 4) == 0;
        for (int i = 0; i < 20; ++i) strata::kernels::native_q6_k_mmvq(wp, xq, y2, n_in, n_out, ncols, s);
        s->wait();
        const auto p0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) strata::kernels::native_q6_k_mmvq(wp, xq, y2, n_in, n_out, ncols, s);
        s->wait();
        const double pus = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - p0).count() / it;
        std::printf("q6_k planar %d x %d, %d cols: %.1f us  (%.1f GB/s of weights)  bitwise %s\n", n_in, n_out, ncols, pus, wbytes / pus / 1e3,
                    same ? "EQUAL" : "DIFFERENT");
        int rc = same ? 0 : 1;
        const int nsel = std::max(1, n_out / 3);
        std::vector<int32_t> rows(nsel);
        for (int i = 0; i < nsel; ++i) rows[i] = (int32_t) (((uint64_t) i * 2654435761ull + 12345) % (uint64_t) n_out);
        int32_t* rd = sycl::malloc_device<int32_t>(nsel, *s);
        s->memcpy(rd, rows.data(), nsel * 4).wait();
        void* wg = strata::kernels::native_q6_k_planar_gather(wp, n_in, rd, nsel, "bench subset");
        if (!wg) { std::printf("planar gather: declined\n"); return 1; }
        float* y3 = sycl::malloc_device<float>((size_t) ncols * nsel, *s);
        strata::kernels::native_q6_k_mmvq(wg, xq, y3, n_in, nsel, ncols, s); s->wait();
        std::vector<float> hg((size_t) ncols * nsel);
        s->memcpy(hg.data(), y3, hg.size() * 4).wait();
        size_t bad = 0;
        for (int c = 0; c < ncols; ++c)
            for (int i = 0; i < nsel; ++i)
                if (std::memcmp(&hg[(size_t) c * nsel + i], &hy[(size_t) c * n_out + rows[i]], 4) != 0) ++bad;
        std::printf("q6_k planar gather %d of %d rows, %d cols: %zu of %zu outputs differ\n", nsel, n_out, ncols, bad, hg.size());
        if (bad) rc = 1;
        return rc;
    }
    return 0;
}
