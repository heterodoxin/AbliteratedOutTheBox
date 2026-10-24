// sycl/src/kernels/q6k_align_bench.cpp - SYCL port: does Q6_K's 210-byte (2-byte aligned) block stride cost the wide
// mmvq kernel its bandwidth? Same weights at stride 210 and repacked at 224 (16-byte aligned); outputs must match.
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>
int main() {
    sycl::queue* s = &dpct::get_in_order_queue();
    const int shapes[][2] = {{2560, 10240}, {2560, 12288}, {6144, 2560}, {2560, 248320}};
    for (auto& sh : shapes) {
        const int n_in = sh[0], n_out = sh[1];
        const size_t nb = (size_t) n_out * (n_in / 256);
        std::vector<uint8_t> hw(nb * 210), hp(nb * 224, 0);
        for (size_t i = 0; i < hw.size(); ++i) hw[i] = (uint8_t) (i * 2654435761u >> 13) & 0x3b;
        for (size_t b = 0; b < nb; ++b) std::memcpy(&hp[b * 224], &hw[b * 210], 210);
        void* w = sycl::malloc_device(hw.size(), *s);
        void* wp = sycl::malloc_device(hp.size(), *s);
        s->memcpy(w, hw.data(), hw.size()).wait(); s->memcpy(wp, hp.data(), hp.size()).wait();
        for (int nc : {1, 2, 4, 6}) {
            std::vector<float> hx((size_t) nc * n_in);
            for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((i * 7919) % 1000) / 500.f - 1.f;
            float* x = sycl::malloc_device<float>(hx.size(), *s);
            void* xq = sycl::malloc_device((size_t) nc * (n_in / 32) * 36, *s);
            float* y = sycl::malloc_device<float>((size_t) nc * n_out, *s);
            s->memcpy(x, hx.data(), hx.size() * 4).wait();
            strata::kernels::native_quantize_q8_1(x, xq, n_in, nc, s);
            auto time = [&](const std::function<void()>& run, std::vector<float>& out) {
                const auto w0 = std::chrono::steady_clock::now();
                while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 150) {
                    for (int i = 0; i < 10; ++i) run();
                    s->wait();
                }
                const int it = 200;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) run();
                s->wait();
                out.resize((size_t) nc * n_out);
                s->memcpy(out.data(), y, out.size() * 4).wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
            };
            std::vector<float> y0, y1;
            const double a = time([&] { strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, nc, s); }, y0);
            const double b = time([&] { strata::kernels::native_q6_k_mmvq_stride224(wp, xq, y, n_in, n_out, nc, s); }, y1);
            double num = 0, den = 0;
            for (size_t i = 0; i < y0.size(); ++i) { num += std::fabs((double) y0[i] - y1[i]); den += std::fabs((double) y0[i]); }
            const double gb = nb * 210.0 / 1e3;
            std::printf("q6_k %5d x %6d cols %d: stride 210 %7.1f us %6.1f GB/s | stride 224 %7.1f us %6.1f GB/s (%.2fx) | rel diff %.1e\n",
                        n_in, n_out, nc, a, gb / a, b, gb / b, a / b, num / (den > 0 ? den : 1));
            sycl::free(x, *s); sycl::free(xq, *s); sycl::free(y, *s);
        }
        sycl::free(w, *s); sycl::free(wp, *s);
    }
    return 0;
}
