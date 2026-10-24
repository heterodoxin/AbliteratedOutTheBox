// sycl/src/kernels/gr_bench.cpp - SYCL port: time fused_gr_read_multi (norm + down + up) for 1-6 tokens on random
// weights at the Coder's shapes; prints a checksum of every output so two builds/switches can be compared bitwise.
// Both GR decompositions: STRATA_GR_MORE_GROUPS=0 / =1 (default: on for Arc A-series only); STRATA_GR_LARGE_GRF the same.
#include "strata/kernels/fused_gr.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    using namespace strata::kernels;
    sycl::queue* s = &dpct::get_in_order_queue();
    constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, TM = kFusedGrMaxT;
    auto bf16 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); };
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    uint32_t r = 12345;
    auto rnd = [&] { r = r * 1664525u + 1013904223u; return ((r >> 8) & 0xffff) / 65536.0f - 0.5f; };
    for (auto& v : wd) v = bf16(rnd() * 0.05f);
    for (auto& v : wu) v = bf16(rnd() * 0.05f);
    for (auto& v : wi) v = bf16(rnd() * 0.05f);
    std::vector<float> hR((size_t) TM * D), hn(D);
    for (auto& v : hR) v = rnd();
    for (auto& v : hn) v = 1.0f + 0.1f * rnd();
    auto dev = [&](const void* h, size_t b) { void* p = sycl::malloc_device(b, *s); s->memcpy(p, h, b).wait(); return p; };
    auto* d_wd = (uint16_t*) dev(wd.data(), wd.size() * 2);
    auto* d_wu = (uint16_t*) dev(wu.data(), wu.size() * 2);
    auto* d_wi = (uint16_t*) dev(wi.data(), wi.size() * 2);
    auto* d_R = (float*) dev(hR.data(), hR.size() * 4);
    auto* d_n = (float*) dev(hn.data(), hn.size() * 4);
    float* xn = sycl::malloc_device<float>((size_t) TM * D, *s);
    float* out = sycl::malloc_device<float>((size_t) TM * (LR + HC + HC + N), *s);
    FusedGrArgs a[TM];
    for (int t = 0; t < TM; ++t) {
        float* o = out + (size_t) t * (LR + HC + HC + N);
        a[t].R = d_R + (size_t) t * D; a[t].w_norm = d_n; a[t].w_down = d_wd; a[t].w_up = d_wu; a[t].w_inject = d_wi;
        a[t].lo = o; a[t].rs = o + LR; a[t].inject_out = o + LR + HC; a[t].mixed = o + LR + 2 * HC;
    }
    for (int T = 1; T <= 6; ++T) {
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 150) {
            for (int i = 0; i < 20; ++i) fused_gr_read_multi(a, T, xn, s);
            s->wait();
        }
        const int it = 500;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) fused_gr_read_multi(a, T, xn, s);
        s->wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        std::vector<float> h((size_t) T * (LR + HC + HC + N));
        s->memcpy(h.data(), out, h.size() * 4).wait();
        uint64_t cs = 1469598103934665603ull;
        double sum = 0;
        for (float v : h) { uint32_t u; std::memcpy(&u, &v, 4); cs = (cs ^ u) * 1099511628211ull; sum += v; }
        std::printf("fused GR read, %d tokens: %7.1f us (%.0f GB/s of weights)  checksum %016llx  sum %.9e\n", T, us,
                    (wd.size() + wu.size() + wi.size()) * 2.0 / us / 1e3, (unsigned long long) cs, sum);
    }
    return 0;
}
