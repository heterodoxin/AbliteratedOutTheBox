// sycl/src/kernels/mmvq_sg_bench.cpp - SYCL port: the decode mmvq kernels on the Coder's dense shapes and types.
//   mmvq_sg_bench   (run with STRATA_MMVQ_SG=16 and =32; prints us per call and a checksum of the outputs)
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    sycl::queue* s = &dpct::get_in_order_queue();
    struct T { const char* name; int type, bytes, elems; };   // ggml type, bytes per block of `elems` weights
    const T types[] = {{"q6_k", 14, 210, 256}, {"q5_k", 13, 176, 256}, {"q4_k", 12, 144, 256}, {"iq4_xs", 23, 136, 256},
                        {"q8_0", 8, 34, 32}, {"iq4_nl", 20, 18, 32}};
    const int shapes[][2] = {{2560, 10240}, {2560, 12288}, {6144, 2560}, {2560, 640}, {640, 2560}, {2560, 2560}};
    const int cols[] = {2, 4, 6};
    for (const T& t : types)
        for (auto& sh : shapes) {
            const int n_in = sh[0], n_out = sh[1];
            if (n_in % t.elems != 0) continue;   // 640 wide: the 32-element formats only
            const size_t wbytes = (size_t) n_out * (n_in / t.elems) * t.bytes;
            std::vector<uint8_t> hw(wbytes);
            for (size_t i = 0; i < wbytes; ++i) hw[i] = (uint8_t) (i * 2654435761u >> 13) & 0x3b;   // keeps fp16 fields finite
            void* w = sycl::malloc_device(wbytes, *s);
            s->memcpy(w, hw.data(), wbytes).wait();
            for (int nc : cols) {
                std::vector<float> hx((size_t) nc * n_in);
                for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((i * 7919) % 1000) / 500.f - 1.f;
                float* x = sycl::malloc_device<float>(hx.size(), *s);
                void* xq = sycl::malloc_device((size_t) nc * (n_in / 32) * 36, *s);
                float* y = sycl::malloc_device<float>((size_t) nc * n_out, *s);
                s->memcpy(x, hx.data(), hx.size() * 4).wait();
                strata::kernels::native_quantize_q8_1(x, xq, n_in, nc, s);
                auto run = [&] { strata::kernels::native_mmvq(t.type, w, xq, y, n_in, n_out, nc, s); };
                const auto w0 = std::chrono::steady_clock::now();
                while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 150) {
                    for (int i = 0; i < 20; ++i) run();
                    s->wait();
                }
                const int it = 300;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) run();
                s->wait();
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
                std::vector<float> hy((size_t) nc * n_out);
                s->memcpy(hy.data(), y, hy.size() * 4).wait();
                double cs = 0; for (float v : hy) cs += std::isfinite(v) ? std::fabs((double) v) : 1e30;
                std::printf("%-7s %5d x %5d  cols %d  %7.1f us  %6.1f GB/s  sum %.6e\n", t.name, n_in, n_out, nc, us,
                            wbytes / us / 1e3, cs);
                sycl::free(x, *s); sycl::free(xq, *s); sycl::free(y, *s);
            }
            sycl::free(w, *s);
        }
    return 0;
}
