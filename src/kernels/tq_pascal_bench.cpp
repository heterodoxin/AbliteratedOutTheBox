// src/kernels/tq_pascal_bench.cpp - the Gyro trellis decode kernels (TQ2_T 144, TQK6 145, TQK7 146) on synthetic blocks.
//
//     build/tq_pascal_bench            checks: the decode-once kernels against the per-column ones (bitwise), and a
//                                      checksum of every output, so two builds can be compared byte for byte
//     build/tq_pascal_bench --bench    the checks, then microseconds per call
//
// Random bytes are valid trellis codes (any 15-bit state indexes the codebook); only the fp16 block scales are set,
// small enough that the grouped path's SwiGLU output keeps a finite fp16 q8_1 scale.  No model, needs the GPU.
// The checksum lines ("SUM ...") are what two builds are compared by: same lines, same bits.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template<typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}

const char* name_of(int t) { return t == 144 ? "TQ2_T" : t == 145 ? "TQK6" : t == 146 ? "TQK7" : t == 42 ? "Q2_0" : "?"; }

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const int bv = t == 42 ? 64 : 128;
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, bv);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));   // 2^-13 .. 2^-6
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

std::vector<float> random_x(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    return x;
}

void check_mmvq(int t, int n_in, int n_out, cudaStream_t s, std::mt19937& rng) {
    const auto w = random_rows(t, n_out, n_in, rng);
    uint8_t* dw = dalloc<uint8_t>(w.size());
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    const int max_cols = 8;
    const auto x = random_x((size_t) max_cols * n_in, rng);
    float* dx = dalloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    uint8_t* dxq = dalloc<uint8_t>((size_t) max_cols * (n_in / 32) * 36);
    k::quantize_q8_1_rows(dx, max_cols, n_in, dxq, s);
    float* dy_old = dalloc<float>((size_t) max_cols * n_out);
    float* dy_new = dalloc<float>((size_t) max_cols * n_out);
    std::vector<float> y_old((size_t) max_cols * n_out), y_new(y_old.size());
    uint64_t sum = 1469598103934665603ull;
    int bad = 0;
    for (int nc = 1; nc <= max_cols; ++nc) {
        ck(cudaMemset(dy_old, 0xFF, y_old.size() * 4), "memset");
        ck(cudaMemset(dy_new, 0xFF, y_new.size() * 4), "memset");
        k::iq_set_old_kernels(true);
        k::iq_mmvq(t, dw, dxq, dy_old, n_in, n_out, nc, s);
        k::iq_set_old_kernels(false);
        k::iq_mmvq(t, dw, dxq, dy_new, n_in, n_out, nc, s);
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(y_old.data(), dy_old, y_old.size() * 4, cudaMemcpyDeviceToHost), "y_old");
        ck(cudaMemcpy(y_new.data(), dy_new, y_new.size() * 4, cudaMemcpyDeviceToHost), "y_new");
        size_t diff = 0;
        bool finite = true;
        for (size_t i = 0; i < (size_t) nc * n_out; ++i) {
            diff += std::memcmp(&y_old[i], &y_new[i], 4) != 0;
            finite = finite && std::isfinite(y_new[i]);
        }
        if (diff || !finite) { std::printf("  %s mmvq nc %d: %zu differ%s FAIL\n", name_of(t), nc, diff, finite ? "" : ", non-finite"); ++bad; }
        sum = fnv(y_new.data(), (size_t) nc * n_out * 4, sum);
    }
    std::printf("SUM mmvq %-6s %5d x %5d  %016llx  %s\n", name_of(t), n_out, n_in, (unsigned long long) sum,
                bad ? "FAIL" : "decode-once == per-column");
    g_fail += bad;
    cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy_old); cudaFree(dy_new);
}

struct Grouped {
    k::NativeExpertLayout L;
    int n_groups = 0, n_ent = 0, cap_groups = 0, cap_ent = 0, n_tok = 0;
    std::vector<uint8_t*> blobs;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    float* dx = nullptr;
    uint8_t *dxq = nullptr, *dscr = nullptr;
    float* dout = nullptr;
    size_t out_floats = 0;

    Grouped(int gu, int dt, int64_t H, int64_t FF, const std::vector<int>& counts, int tokens, std::mt19937& rng) {
        L = k::native_expert_layout(gu, dt, H, FF);
        n_groups = (int) counts.size();
        cap_groups = n_groups + 2;
        n_tok = tokens;
        std::vector<int32_t> start(1, 0), tok, dst;
        for (int c : counts) start.push_back(start.back() + c);
        n_ent = start.back();
        cap_ent = n_ent + 3;
        std::uniform_int_distribution<int> tk(0, tokens - 1);
        for (int e = 0; e < n_ent; ++e) tok.push_back(tk(rng));
        dst.resize(n_ent);
        std::iota(dst.begin(), dst.end(), 0);
        std::shuffle(dst.begin(), dst.end(), rng);
        std::vector<unsigned long long> ptr;
        for (int g = 0; g < n_groups; ++g) {
            auto b = random_rows(gu, 2 * FF, H, rng);
            const auto d = random_rows(dt, H, FF, rng);
            b.insert(b.end(), d.begin(), d.end());
            if (b.size() != L.bytes) { std::fprintf(stderr, "blob size %zu != %zu\n", b.size(), L.bytes); std::exit(2); }
            uint8_t* db = dalloc<uint8_t>(b.size());
            ck(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice), "blob");
            blobs.push_back(db);
            ptr.push_back((unsigned long long) db);
        }
        dptr = dalloc<unsigned long long>(cap_groups);
        dstart = dalloc<int32_t>(cap_groups + 1);
        dn = dalloc<int32_t>(1);
        ddst = dalloc<int32_t>(cap_ent);
        dtok = dalloc<int32_t>(cap_ent);
        ck(cudaMemcpy(dptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "ptr");
        ck(cudaMemcpy(dstart, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start");
        ck(cudaMemcpy(dn, &n_groups, 4, cudaMemcpyHostToDevice), "n");
        if (n_ent) {
            ck(cudaMemcpy(ddst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
            ck(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "tok");
        }
        const auto x = random_x((size_t) tokens * H, rng);
        dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        dxq = dalloc<uint8_t>((size_t) tokens * (H / 32) * 36);
        dscr = dalloc<uint8_t>(k::native_expert_scratch_bytes(cap_ent, FF));
        out_floats = (size_t) cap_ent * H;
        dout = dalloc<float>(out_floats);
    }
    ~Grouped() {
        for (auto* b : blobs) cudaFree(b);
        cudaFree(dptr); cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
        cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout);
    }
    void run(bool old, cudaStream_t s) {
        k::iq_set_old_kernels(old);
        k::quantize_q8_1_rows(dx, n_tok, L.n_embd, dxq, s);
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, cap_groups, cap_ent, dxq, dscr, dout, s);
        k::iq_set_old_kernels(false);
    }
    std::vector<float> result(bool old, cudaStream_t s) {
        ck(cudaMemset(dout, 0xFF, out_floats * 4), "memset");
        ck(cudaMemset(dscr, 0, k::native_expert_scratch_bytes(cap_ent, L.n_ff)), "memset");
        run(old, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> o(out_floats);
        ck(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost), "out");
        return o;
    }
};

void check_grouped(int gu, int dt, int64_t H, int64_t FF, cudaStream_t s, std::mt19937& rng) {
    Grouped G(gu, dt, H, FF, {1, 3, 4, 5, 0, 8, 2, 11, 1}, 8, rng);
    const auto a = G.result(true, s), b = G.result(false, s);
    size_t diff = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += std::memcmp(&a[i], &b[i], 4) != 0;
        if (i < (size_t) G.n_ent * H) finite = finite && std::isfinite(b[i]);
    }
    const bool ok = diff == 0 && finite;
    std::printf("SUM grouped %-6s/%-6s %5lld x %4lld  %016llx  %s\n", name_of(gu), name_of(dt), (long long) H, (long long) FF,
                (unsigned long long) fnv(b.data(), b.size() * 4), ok ? "decode-once == per-column" : "FAIL");
    if (!ok) { std::printf("  %zu of %zu floats differ%s\n", diff, a.size(), finite ? "" : ", non-finite outputs"); ++g_fail; }
}

float time_ms(cudaStream_t s, int it, const auto& fn) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    fn();
    ck(cudaStreamSynchronize(s), "warm");
    cudaEventRecord(e0, s);
    for (int i = 0; i < it; ++i) fn();
    cudaEventRecord(e1, s);
    ck(cudaEventSynchronize(e1), "bench");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / it;
}

void bench(cudaStream_t s, std::mt19937& rng) {
    std::printf("\n--bench: microseconds per call (idle GPU assumed)\n");
    const int n_in = 2560, n_out = 8192, it = 100;
    for (int t : {144, 145, 146, 42}) {
        const auto w = random_rows(t, n_out, n_in, rng);
        uint8_t* dw = dalloc<uint8_t>(w.size());
        ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
        const auto x = random_x((size_t) 8 * n_in, rng);
        float* dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        uint8_t* dxq = dalloc<uint8_t>((size_t) 8 * (n_in / 32) * 36);
        k::quantize_q8_1_rows(dx, 8, n_in, dxq, s);
        float* dy = dalloc<float>((size_t) 8 * n_out);
        const double gb = (double) w.size() / 1e9;
        std::printf("BENCH mmvq %-6s %d x %d (%.3f GB):", name_of(t), n_out, n_in, gb);
        for (int nc : {1, 2, 3, 4, 8}) {
            const float us = 1e3f * time_ms(s, it, [&] { k::iq_mmvq(t, dw, dxq, dy, n_in, n_out, nc, s); });
            std::printf("  nc%d %.1f us (%.0f GB/s)", nc, us, gb / (us * 1e-6));
        }
        std::printf("\n");
        cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
    }
    // the same 21M weights as rows of different lengths: a per-row cost shows as a falling time with longer rows
    for (int t : {145}) {
        for (auto rc : {std::pair<int, int>{8192, 2560}, {4096, 5120}, {2048, 10240}, {1024, 20480}}) {
            const int ni = rc.second, no = rc.first;
            const auto w = random_rows(t, no, ni, rng);
            uint8_t* dw = dalloc<uint8_t>(w.size());
            ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
            const auto x = random_x((size_t) 8 * ni, rng);
            float* dx = dalloc<float>(x.size());
            ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
            uint8_t* dxq = dalloc<uint8_t>((size_t) 8 * (ni / 32) * 36);
            k::quantize_q8_1_rows(dx, 8, ni, dxq, s);
            float* dy = dalloc<float>((size_t) 8 * no);
            const float us = 1e3f * time_ms(s, it, [&] { k::iq_mmvq(t, dw, dxq, dy, ni, no, 1, s); });
            std::printf("BENCH rows %-6s %5d x %5d nc1 %.1f us\n", name_of(t), no, ni, us);
            cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
        }
    }
    // the verify window's shape: 2560 x 640 experts, 16 groups of m entries each
    for (auto gd : {std::pair<int, int>{145, 146}, {145, 144}, {144, 144}, {144, 146}}) {
        std::printf("BENCH grouped %-6s/%-6s 16 groups of m entries:", name_of(gd.first), name_of(gd.second));
        for (int m : {1, 2, 3, 4, 8}) {
            Grouped G(gd.first, gd.second, 2560, 640, std::vector<int>(16, m), 8, rng);
            const float us = 1e3f * time_ms(s, 30, [&] { G.run(false, s); });
            const double gb = 16.0 * (double) G.L.bytes / 1e9;
            std::printf("  m=%d %.1f us (%.0f GB/s)", m, us, gb / (us * 1e-6));
        }
        std::printf("\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device %s cc %d.%d\n", prop.name, prop.major, prop.minor);
    std::mt19937 rng(18);
    for (int t : {144, 145, 146}) {
        check_mmvq(t, 2560, 67, s, rng);
        check_mmvq(t, 1024, 5, s, rng);
    }
    for (int gu : {144, 145})
        for (int dt : {144, 146}) check_grouped(gu, dt, 2560, 640, s, rng);
    // other geometries: a longer hidden row (the staged gate/up falls back from its 80-block activation buffer), a shorter
    // expert row (the 8-lane down rows need n_ff 640)
    for (int gu : {144, 145}) {
        check_grouped(gu, 146, 5120, 640, s, rng);
        check_grouped(gu, 144, 2560, 512, s, rng);
        check_grouped(gu, 146, 2048, 256, s, rng);
    }
    if (do_bench) bench(s, rng);
    std::printf("tq_pascal_bench: %d failures\n", g_fail);
    cudaStreamDestroy(s);
    return g_fail ? 1 : 0;
}
