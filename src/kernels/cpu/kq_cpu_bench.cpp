// src/kernels/cpu/kq_cpu_bench.cpp - MODEL-UPDATE: how close the CPU pool's expert rows come to the RAM bandwidth, per
// expert format.  No GPU, no model: --mb MB of expert blobs (ggml-quantized random weights, far more than any L3) are
// computed by T threads at once through the engine's own entry points (native_gu_rows + native_down_rows, the pool's
// calls), each thread walking its own experts, nt tokens per expert - the decode case, where every CPU expert is a
// fresh read from DRAM.  Prints the weight bytes per second against the --ram-gbs given (a separate read test).
//
//     kq_cpu_bench [--pairs q4_K/q5_1,q5_K/q8_0,q6_K/q8_0,q8_0/q8_0,iq3_s/iq4_nl] [--threads 1,8,16,32] [--nt 1,2,4]
//                  [--mb 4096] [--reps 3] [--ram-gbs N]
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t kH = 2560, kFF = 640;
constexpr int kMaxT = 8;

int type_of(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n != nullptr && s == n) return t;
    }
    return -1;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(s.substr(from, c == std::string::npos ? std::string::npos : c - from));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}

// one expert blob in the GGUF form: gate rows, up rows (n_ff x n_embd each), down rows (n_embd x n_ff), from random
// weights quantized by ggml-cpu (valid scales, as a real file has)
void make_blob(const cpu::NativeFmt& f, uint8_t* blob, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> row((size_t) std::max(kH, kFF));
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) f.d_type);
    for (int64_t r = 0; r < 2 * kFF; ++r) {
        for (int64_t i = 0; i < kH; ++i) row[(size_t) i] = nd(rng);
        tg->from_float(row.data(), blob + (size_t) r * f.gu_row, kH);
    }
    for (int64_t r = 0; r < kH; ++r) {
        for (int64_t i = 0; i < kFF; ++i) row[(size_t) i] = nd(rng);
        td->from_float(row.data(), blob + f.down_off + (size_t) r * f.d_row, kFF);
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> pairs = split("q4_K/q5_1,q5_K/q8_0,q6_K/q8_0,q8_0/q8_0,iq3_s/iq4_nl");
    std::vector<int> threads = {1, 8, 16, 32}, nts = {1, 2, 4};
    int mb = 4096, reps = 3;
    double ram = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        auto ints = [](const std::string& s) { std::vector<int> v; for (auto& x : split(s)) v.push_back(std::atoi(x.c_str())); return v; };
        if (a == "--pairs" && more) pairs = split(argv[++i]);
        else if (a == "--threads" && more) threads = ints(argv[++i]);
        else if (a == "--nt" && more) nts = ints(argv[++i]);
        else if (a == "--mb" && more) mb = std::atoi(argv[++i]);
        else if (a == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--ram-gbs" && more) ram = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: kq_cpu_bench [--pairs a/b,...] [--threads 1,8] [--nt 1,2,4] [--mb 4096] [--reps 3] [--ram-gbs N]\n");
            return 2;
        }
    }
    ggml_cpu_init();
    std::printf("kq_cpu_bench: %s (avx2 %d, avx512 %d), %d MB of blobs per format\n", cpu::cpu_name().c_str(),
                (int) cpu::cpu_avx2_ok(), (int) cpu::cpu_avx512_ok(), mb);
    for (const std::string& pair : pairs) {
        const size_t slash = pair.find('/');
        const int gt = slash == std::string::npos ? -1 : type_of(pair.substr(0, slash));
        const int dt = slash == std::string::npos ? -1 : type_of(pair.substr(slash + 1));
        cpu::NativeFmt f;
        std::string err;
        if (gt < 0 || dt < 0 || !cpu::native_fmt(gt, dt, kH, kFF, f, err)) {
            std::printf("%s: %s\n", pair.c_str(), err.empty() ? "unknown types" : err.c_str());
            continue;
        }
        ggml_quantize_init((ggml_type) gt);   // the i-quants' quantizers need their grids first
        ggml_quantize_init((ggml_type) dt);
        const size_t E = std::max<size_t>(64, ((size_t) mb << 20) / f.bytes);
        std::unique_ptr<uint8_t[]> mem(new uint8_t[E * f.bytes + 4096]);
        uint8_t* blobs = (uint8_t*) (((uintptr_t) mem.get() + 4095) & ~(uintptr_t) 4095);
        {   // 16 distinct experts, copied over the whole buffer (the bytes in DRAM are what is measured)
            std::mt19937 rng(1234 + gt * 64 + dt);
            const size_t U = std::min<size_t>(16, E);
            for (size_t e = 0; e < U; ++e) make_blob(f, blobs + e * f.bytes, rng);
            for (size_t e = U; e < E; ++e) std::memcpy(blobs + e * f.bytes, blobs + (e % U) * f.bytes, f.bytes);
        }
        // activations: kMaxT tokens of x (n_embd) and h (n_ff), quantized the way the pool quantizes them
        std::mt19937 rng(77);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<std::vector<uint8_t>> xa(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes)),
            ha(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
        const void* act[kMaxT];
        const void* hq[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            std::vector<float> x(kH), h(kFF);
            for (auto& v : x) v = nd(rng);
            for (auto& v : h) v = 0.1f * nd(rng);
            cpu::native_quant_act(f, x.data(), xa[t].data());
            cpu::native_quant_h(f, h.data(), ha[t].data());
            act[t] = xa[t].data();
            hq[t] = ha[t].data();
        }
        std::printf("\n%s: %zu experts of %.3f MB (%.0f MB)\n", pair.c_str(), E, f.bytes / 1e6, E * (double) f.bytes / 1e6);
        for (int nt : nts) {
            for (int T : threads) {
                if (T < 1) continue;
                std::vector<double> gbs;
                for (int rep = 0; rep < reps; ++rep) {
                    std::atomic<size_t> next{0};
                    auto work = [&] {
                        std::vector<float> ffb((size_t) kMaxT * kFF), outb((size_t) kMaxT * kH);
                        float* ff[kMaxT];
                        float* out[kMaxT];
                        for (int t = 0; t < kMaxT; ++t) { ff[t] = &ffb[(size_t) t * kFF]; out[t] = &outb[(size_t) t * kH]; }
                        std::vector<std::vector<uint8_t>> hb(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
                        for (;;) {
                            const size_t e = next.fetch_add(1);
                            if (e >= E) break;
                            const uint8_t* blob = blobs + e * f.bytes;
                            cpu::native_gu_rows(f, blob, act, nt, ff, 0, (int) kFF);
                            cpu::native_down_rows(f, blob, hq, nt, out, 0, (int) kH);
                        }
                        volatile float sink = outb[0] + ffb[0];
                        (void) sink;
                    };
                    const auto t0 = std::chrono::steady_clock::now();
                    std::vector<std::thread> th;
                    for (int k = 0; k < T; ++k) th.emplace_back(work);
                    for (auto& x : th) x.join();
                    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    gbs.push_back(E * (double) f.bytes / s / 1e9);
                }
                std::sort(gbs.begin(), gbs.end());
                const double med = gbs[gbs.size() / 2];
                std::printf("  nt %d threads %2d: %6.1f GB/s of weights (%.3f ms per expert per thread)%s\n", nt, T, med,
                            f.bytes / (med * 1e9 / T) * 1e3,
                            ram > 0 ? (" = " + std::to_string((int) std::lround(100 * med / ram)) + "% of RAM").c_str() : "");
            }
        }
    }
    return 0;
}
