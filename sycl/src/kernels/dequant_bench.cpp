// sycl/src/kernels/dequant_bench.cpp - time the prompt path's expert dequantizers (iq_dequant_gu_f16 / iq_dequant_f16)
// on random blocks and print a hash of the FP16 output, so two builds can be compared bit for bit.
//   dequant_bench [type|all] [experts]       (gate/up 1280 x 2560 and down 2560 x 1280 per expert, the model's shape)
#include "strata/kernels/iq_kernels.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

namespace {
struct Ty { int id; const char* name; int bytes_per_256; };
const Ty kTypes[] = {{16, "IQ2_XXS", 66}, {17, "IQ2_XS", 74}, {22, "IQ2_S", 82}, {18, "IQ3_XXS", 98},
                     {21, "IQ3_S", 110}, {20, "IQ4_NL", 144}, {42, "Q2_0", 72},
                     // Unsloth's files (MODEL-UPDATE): gate/up Q4_K / Q5_K / Q6_K / Q8_0, down Q5_1 / Q8_0
                     {12, "Q4_K", 144}, {13, "Q5_K", 176}, {14, "Q6_K", 210}, {7, "Q5_1", 192}, {8, "Q8_0", 272},
                     {2, "Q4_0", 144}, {3, "Q4_1", 160}, {6, "Q5_0", 176}};
uint64_t fnv(const uint16_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    const int want = argc > 1 && std::strcmp(argv[1], "all") != 0 ? std::atoi(argv[1]) : -1;
    const int experts = argc > 2 ? std::atoi(argv[2]) : 32;
    const int64_t n_ff = 1280, n_embd = 2560;
    sycl::queue* s = &dpct::get_in_order_queue();
    for (const Ty& t : kTypes) {
        if (want >= 0 && t.id != want) continue;
        const size_t per_mat = (size_t) n_ff * n_embd / 256 * t.bytes_per_256;   // one of gate / up / down
        std::vector<uint8_t> hb(per_mat);
        uint32_t x = 12345u + (uint32_t) t.id;
        for (auto& b : hb) { x = x * 1664525u + 1013904223u; b = (uint8_t) (x >> 24); }
        uint8_t* src = (uint8_t*) sycl::malloc_device(per_mat * 3 * experts, *s);   // gate, up, down per expert
        for (int e = 0; e < experts * 3; ++e) s->memcpy(src + (size_t) e * per_mat, hb.data(), per_mat);
        const size_t gu_out = (size_t) 2 * n_ff * n_embd, d_out = (size_t) n_embd * n_ff;
        uint16_t* dst = (uint16_t*) sycl::malloc_device((gu_out + d_out) * 2 * experts, *s);
        s->wait();
        auto run = [&]() {
            for (int e = 0; e < experts; ++e) {
                const uint8_t* b = src + (size_t) e * 3 * per_mat;
                uint16_t* o = dst + (size_t) e * (gu_out + d_out);
                strata::kernels::iq_dequant_gu_f16(t.id, b, b + per_mat, n_ff, n_embd, o, s);
                strata::kernels::iq_dequant_f16(t.id, b + 2 * per_mat, n_embd * n_ff, o + gu_out, s);
            }
            s->wait();
        };
        for (int i = 0; i < 3; ++i) run();
        const int reps = 10;
        std::vector<double> ms;
        for (int i = 0; i < reps; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            run();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[reps / 2];
        const double wr = (double) (gu_out + d_out) * 2 * experts, rd = (double) per_mat * 3 * experts;
        std::vector<uint16_t> ho((gu_out + d_out) * 2);   // hash expert 0 and the last expert
        s->memcpy(ho.data(), dst, (gu_out + d_out) * 2).wait();
        const uint64_t h0 = fnv(ho.data(), gu_out + d_out);
        s->memcpy(ho.data(), dst + (size_t) (experts - 1) * (gu_out + d_out), (gu_out + d_out) * 2).wait();
        const uint64_t h1 = fnv(ho.data(), gu_out + d_out);
        std::printf("%-8s %2d experts: median %7.2f ms  write %6.1f GB/s  (read %5.1f GB/s)  hash %016llx %016llx\n", t.name,
                    experts, med, wr / med / 1e6, rd / med / 1e6, (unsigned long long) h0, (unsigned long long) h1);
        sycl::free(src, *s);
        sycl::free(dst, *s);
    }
    return 0;
}
