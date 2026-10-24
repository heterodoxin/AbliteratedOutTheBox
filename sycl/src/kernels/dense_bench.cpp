// sycl/src/kernels/dense_bench.cpp - the decode window's dense GEMVs (native_mmvq) at the model's shapes and formats:
// NMAT different matrices per shape captured in one graph, so the weights stream from DRAM as in the engine (a window
// reads every layer's matrix once); us per call, GB/s of weights, and an FNV hash of the outputs so two builds or two
// STRATA_MMVQ_* settings can be compared bit for bit.
//
//     dense_bench [type ...]      (default: 14 13 12 23 = Q6_K Q5_K Q4_K IQ4_XS)
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
const char* name_of(int t) {
    switch (t) { case 14: return "Q6_K"; case 13: return "Q5_K"; case 12: return "Q4_K"; case 23: return "IQ4_XS";
                 case 8: return "Q8_0"; default: return "?"; }
}
// random blocks with a finite, small fp16 scale (at the format's offset) so the outputs are finite
void fill(int t, std::vector<uint8_t>& w, std::mt19937& rng) {
    std::uniform_int_distribution<int> b(0, 255);
    for (auto& v : w) v = (uint8_t) b(rng);
    const size_t bs = t == 14 ? 210 : t == 13 ? 176 : t == 12 ? 144 : t == 23 ? 136 : 34;
    const size_t off = t == 14 ? 208 : 0;
    for (size_t o = 0; o + bs <= w.size(); o += bs) {
        const uint16_t h = (uint16_t) (0x1400 | (b(rng) & 0x3ff));          // ~2^-10
        std::memcpy(&w[o + off], &h, 2);
        if (t == 12 || t == 13) std::memcpy(&w[o + 2], &h, 2);             // dmin
    }
}
uint64_t fnv(const float* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint32_t* u = (const uint32_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= u[i]; h *= 1099511628211ull; }
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> types;
    for (int i = 1; i < argc; ++i) types.push_back(std::atoi(argv[i]));
    if (types.empty()) types = {14, 13, 12, 23};
    dpct::queue_ptr s = dpct::get_current_device().create_queue(true);
    std::mt19937 rng(7);
    const struct { int n_in, n_out; const char* what; } shapes[] = {
        {2560, 10240, "GDN qkv"}, {2560, 6144, "GDN z"}, {6144, 2560, "out-proj"}, {2560, 12288, "QSA q"}, {2560, 512, "k/v"}};
    constexpr int NMAT = 24;
    std::printf("dense_bench: %d matrices per shape in one graph; us per call (median of 30), GB/s of weights, hash\n", NMAT);
    for (const int t : types) {
        for (const auto& sh : shapes) {
            const size_t wb = k::native_mmvq_weight_bytes(t, sh.n_in, sh.n_out), slot = (wb + 255) & ~(size_t) 255;
            uint8_t* w = (uint8_t*) sycl::malloc_device(slot * NMAT, *s);
            std::vector<uint8_t> hw(wb);
            for (int m = 0; m < NMAT; ++m) { fill(t, hw, rng); s->memcpy(w + slot * m, hw.data(), wb).wait(); }
            std::vector<float> hx((size_t) 4 * sh.n_in);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : hx) v = nd(rng);
            float* x = sycl::malloc_device<float>(hx.size(), *s);
            s->memcpy(x, hx.data(), hx.size() * 4).wait();
            void* xq = sycl::malloc_device(k::native_q8_1_bytes(sh.n_in, 4) + 256, *s);
            float* y = sycl::malloc_device<float>((size_t) 4 * sh.n_out * NMAT, *s);
            for (int nc = 1; nc <= 4; ++nc) {
                k::native_quantize_q8_1(x, xq, sh.n_in, nc, s);
                s->wait();
                dpct::experimental::command_graph_ptr g;
                dpct::experimental::begin_recording(s);
                for (int m = 0; m < NMAT; ++m)
                    k::native_mmvq(t, w + slot * m, xq, y + (size_t) 4 * sh.n_out * m, sh.n_in, sh.n_out, nc, s);
                dpct::experimental::end_recording(s, &g);
                auto* ge = new sycl::ext::oneapi::experimental::command_graph<
                    sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());
                delete g;
                std::vector<double> us;
                for (int i = 0; i < 40; ++i) {
                    sycl::event e0 = s->ext_oneapi_submit_barrier();
                    s->ext_oneapi_graph(*ge);
                    sycl::event e1 = s->ext_oneapi_submit_barrier();
                    e1.wait();
                    const double ns = (double) (e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                                e0.get_profiling_info<sycl::info::event_profiling::command_start>());
                    if (i >= 10) us.push_back(ns / 1e3 / NMAT);
                }
                delete ge;
                std::sort(us.begin(), us.end());
                std::vector<float> hy((size_t) 4 * sh.n_out * NMAT);
                s->memcpy(hy.data(), y, hy.size() * 4).wait();
                uint64_t h = 1469598103934665603ull;
                for (int m = 0; m < NMAT; ++m) h ^= fnv(&hy[(size_t) 4 * sh.n_out * m], (size_t) nc * sh.n_out) + m;
                std::printf("%-6s %-8s %5d x %5d  %d cols: %8.2f us  %6.1f GB/s  hash %016llx\n", name_of(t), sh.what,
                            sh.n_in, sh.n_out, nc, us[us.size() / 2], wb / us[us.size() / 2] / 1e3, (unsigned long long) h);
            }
            sycl::free(w, *s); sycl::free(x, *s); sycl::free(xq, *s); sycl::free(y, *s);
        }
    }
    dpct::get_current_device().destroy_queue(s);
    return 0;
}
