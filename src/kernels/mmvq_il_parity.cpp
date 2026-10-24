// src/kernels/mmvq_il_parity.cpp - native_mmvq_il (2-4 columns from the interleaved q8_1 copy, fork F4) against
// native_mmvq's multi-column kernels, bitwise, for every rows-a-warp choice; --bench times both per call.
//
//     build/mmvq_il_parity [--bench [--emit-table]]
//
// --emit-table (with --bench) prints the rows-a-warp table this card measured, in the source form of kIlRows in
// native_mmvq.cu: per cell the rows count (1/2/4) that takes at least 3% off native_mmvq's time, else 0.
//
// The same Q8_1 bytes feed both paths (quantize_q8_1_rows, then native_q8_1_interleave). Weights are random bytes with
// the block scales rewritten as normal fp16 values, so every output is finite (a NaN would compare equal whatever
// produced it). The bench cycles through enough copies of the weights to miss the L2 cache, as a layer's call does.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace strata::kernels;

struct Cell { int type, T, n_out; double ref; double il[5]; };
std::vector<Cell> g_cells;
struct Case {
    const char* name;
    int type, n_in, n_out, block_elems, block_bytes, scale_at[2];
};
// the engine's dense shapes: qkv 10240 / gate 6144 / out 2560 (GDN), q 12288 / k,v 512 / o 2560 (QSA), the head
const Case CASES[] = {
    {"IQ4_XS 2560x10240", 23, 2560, 10240, 256, 136, {0, -1}},
    {"IQ4_XS 2560x6144", 23, 2560, 6144, 256, 136, {0, -1}},
    {"IQ4_XS 6144x2560", 23, 6144, 2560, 256, 136, {0, -1}},
    {"IQ4_XS 2560x512", 23, 2560, 512, 256, 136, {0, -1}},
    {"Q4_K 2560x10240", 12, 2560, 10240, 256, 144, {0, 2}},
    {"Q4_K 6144x2560", 12, 6144, 2560, 256, 144, {0, 2}},
    {"Q4_K odd rows 2560x2051", 12, 2560, 2051, 256, 144, {0, 2}},
    {"Q5_K 2560x12288", 13, 2560, 12288, 256, 176, {0, 2}},
    {"Q5_K 2560x6144", 13, 2560, 6144, 256, 176, {0, 2}},
    {"Q5_K 6144x2560", 13, 6144, 2560, 256, 176, {0, 2}},
    {"Q5_K head 2560x248320", 13, 2560, 248320, 256, 176, {0, 2}},
    {"Q6_K 2560x12288", 14, 2560, 12288, 256, 210, {208, -1}},
    {"Q6_K 2560x10240", 14, 2560, 10240, 256, 210, {208, -1}},
    {"Q6_K 6144x2560", 14, 6144, 2560, 256, 210, {208, -1}},
    {"Q6_K odd rows 2560x2051", 14, 2560, 2051, 256, 210, {208, -1}},
    {"Q6_K 2560x512", 14, 2560, 512, 256, 210, {208, -1}},
    {"IQ4_XS 2560x12288", 23, 2560, 12288, 256, 136, {0, -1}},
    {"IQ4_XS 2560x640", 23, 2560, 640, 256, 136, {0, -1}},
    {"Q4_K 2560x6144", 12, 2560, 6144, 256, 144, {0, 2}},
    {"Q4_K 2560x640", 12, 2560, 640, 256, 144, {0, 2}},
    {"Q4_K 2560x512", 12, 2560, 512, 256, 144, {0, 2}},
    {"Q5_K 2560x10240", 13, 2560, 10240, 256, 176, {0, 2}},
    {"Q5_K 2560x640", 13, 2560, 640, 256, 176, {0, 2}},
    {"Q5_K 2560x512", 13, 2560, 512, 256, 176, {0, 2}},
    {"Q6_K 2560x6144", 14, 2560, 6144, 256, 210, {208, -1}},
    {"Q6_K 2560x640", 14, 2560, 640, 256, 210, {208, -1}},
    {"Q3_K 2560x10240", 11, 2560, 10240, 256, 110, {108, -1}},
    {"Q3_K 2560x12288", 11, 2560, 12288, 256, 110, {108, -1}},
    {"Q3_K 2560x6144", 11, 2560, 6144, 256, 110, {108, -1}},
    {"Q3_K 6144x2560", 11, 6144, 2560, 256, 110, {108, -1}},
    {"Q3_K 2560x640", 11, 2560, 640, 256, 110, {108, -1}},
    {"Q3_K 2560x512", 11, 2560, 512, 256, 110, {108, -1}},
    {"Q3_K 2560x2051 odd", 11, 2560, 2051, 256, 110, {108, -1}},
};

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

int run_case(const Case& c, bool bench, cudaStream_t s) {
    if (std::strstr(c.name, "odd") != nullptr) bench = false;   // parity-only shape: not a timing cell of the table
    if (c.type == 13 && c.n_out > 100000 && !bench) { /* the head: parity too, just big */ }
    const std::size_t wbytes = native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    std::mt19937 rng(77u + (unsigned) c.type * 31u + (unsigned) c.n_out);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%s: weight bytes mismatch\n", c.name);
        return 1;
    }
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    const int copies = bench ? (int) std::max<std::size_t>(1, std::min<std::size_t>(24, (192u << 20) / wbytes)) : 1;
    std::vector<void*> dw(copies, nullptr);
    for (int i = 0; i < copies; ++i) {
        if (!ck(cudaMalloc(&dw[i], wbytes), "malloc w") || !ck(cudaMemcpy(dw[i], w.data(), wbytes, cudaMemcpyHostToDevice), "copy w"))
            return 1;
    }
    int bad = 0;
    const int t_max = std::getenv("MMVQ_IL_TMAX") ? std::atoi(std::getenv("MMVQ_IL_TMAX")) : 8;   // 147-verify: 5-8 columns
    for (int T = 1; T <= t_max; ++T) {
        std::vector<float> x((std::size_t) T * c.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        float* dx = nullptr;
        void *xq = nullptr, *xil = nullptr;
        float *yref = nullptr, *yil = nullptr;
        const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
        if (!ck(cudaMalloc(&dx, x.size() * 4), "malloc x") ||
            !ck(cudaMalloc(&xq, native_q8_1_bytes(c.n_in, T)), "malloc xq") ||
            !ck(cudaMalloc(&xil, native_q8_1_il_bytes(c.n_in, T)), "malloc xil") ||
            !ck(cudaMalloc((void**) &yref, ybytes), "malloc yref") || !ck(cudaMalloc((void**) &yil, ybytes), "malloc yil"))
            return 1;
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x");
        quantize_q8_1_rows(dx, T, c.n_in, xq, s);
        if (T >= 2) native_q8_1_interleave(xq, xil, c.n_in, T, s);
        native_mmvq(c.type, dw[0], xq, yref, c.n_in, c.n_out, T, s);
        ck(cudaStreamSynchronize(s), "ref");
        std::vector<uint32_t> a((std::size_t) T * c.n_out), b(a.size());
        ck(cudaMemcpy(a.data(), yref, ybytes, cudaMemcpyDeviceToHost), "read ref");
        double best_ref = 0, best_il[5] = {0, 0, 0, 0, 0};
        int table_r = 0;
        for (int r : {0, 1, 2, 4}) {
            native_mmvq_il_tune(r);
            cudaMemset(yil, 0xff, ybytes);
            native_mmvq_il(c.type, dw[0], xq, xil, yil, c.n_in, c.n_out, T, s);
            ck(cudaStreamSynchronize(s), "il");
            ck(cudaMemcpy(b.data(), yil, ybytes, cudaMemcpyDeviceToHost), "read il");
            long long diff = 0, nonfinite = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                float fa;
                std::memcpy(&fa, &a[i], 4);
                if (!std::isfinite(fa)) ++nonfinite;
                if (a[i] != b[i]) ++diff;
            }
            const bool used = r != 0 || native_mmvq_il_supported(c.type, T, c.n_out);
            if (diff != 0 || nonfinite != 0) {
                ++bad;
                std::printf("FAIL %-26s T=%d rows=%d: %lld bits differ, %lld non-finite of %zu\n", c.name, T, r, diff, nonfinite, a.size());
            }
            (void) used;
            if (bench && r != 0) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                const int reps = 200;
                for (int rep = 0; rep < 2; ++rep) {   // rep 0 warms, rep 1 times
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < reps; ++i) native_mmvq_il(c.type, dw[i % copies], xq, xil, yil, c.n_in, c.n_out, T, s);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                }
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                best_il[r] = 1000.0 * ms / reps;
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
            }
        }
        native_mmvq_il_tune(0);
        if (bench) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            const int reps = 200;
            for (int rep = 0; rep < 2; ++rep) {
                cudaEventRecord(e0, s);
                for (int i = 0; i < reps; ++i) native_mmvq(c.type, dw[i % copies], xq, yref, c.n_in, c.n_out, T, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
            }
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            best_ref = 1000.0 * ms / reps;
            (void) table_r;
            g_cells.push_back({c.type, T, c.n_out, best_ref, {0, best_il[1], best_il[2], 0, best_il[4]}});
            std::printf("BENCH %-26s T=%d  multi %.1f us | il rows1 %.1f rows2 %.1f rows4 %.1f us\n", c.name, T, best_ref,
                        best_il[1], best_il[2], best_il[4]);
        }
        cudaFree(dx); cudaFree(xq); cudaFree(xil); cudaFree(yref); cudaFree(yil);
    }
    for (void* p : dw) cudaFree(p);
    return bad;
}

// the table as kIlRows prints it: per (type, ncols, n_out class) the fastest rows count if >= 3% under native_mmvq, else 0.
// A cell with several measured shapes takes the worst (smallest) gain of the best choice, as the shipped table does for
// two cards; a class with no measured shape prints the value of its neighbour below it (or 0).
void emit_table() {
    const int types[] = {23, 12, 13, 14, 11};
    const char* names[] = {"IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q3_K"};
    std::printf("// measured on this card by mmvq_il_parity --bench --emit-table\n");
    // per cell: the rows count whose WORST ratio to native_mmvq over the cell's measured shapes is the best, taken only
    // when that worst ratio is under 0.97 (else 0); a class with no measured shape takes its lower neighbour's value
    auto pick_cell = [&](int type, int nc, int cls, int prev) {
        double best_worst = 1e9;
        int best = -1;
        bool any = false;
        for (int r : {1, 2, 4}) {
            double worst = 0;
            bool have = false;
            for (const Cell& c : g_cells) {
                if (c.type != type || c.T != nc) continue;
                const int cc = c.n_out < 2048 ? 0 : c.n_out < 4096 ? 1 : c.n_out < 8192 ? 2 : c.n_out < 12288 ? 3 : 4;
                if (cc != cls || c.il[r] <= 0) continue;
                have = true;
                worst = std::max(worst, c.il[r] / c.ref);
            }
            any = any || have;
            if (have && worst < best_worst) { best_worst = worst; best = r; }
        }
        if (!any) return prev;
        return best_worst < 0.97 ? best : 0;
    };
    for (int ti = 0; ti < 5; ++ti) {
        std::printf("    {%d, {", types[ti]);
        for (int nc = 2; nc <= 4; ++nc) {
            int prev = 0;
            std::printf("{");
            for (int cls = 0; cls < 5; ++cls) {
                prev = pick_cell(types[ti], nc, cls, prev);
                std::printf("%d%s", prev, cls < 4 ? ", " : "");
            }
            std::printf("}%s", nc < 4 ? ", " : "");
        }
        std::printf("}},   // %s\n", names[ti]);
    }
    std::printf("// 5-8 columns (kIlRows58_<cc>; rows 4 runs as 2 there):\n");
    for (int ti = 0; ti < 5; ++ti) {
        std::printf("    {%d, {", types[ti]);
        for (int nc = 5; nc <= 8; ++nc) {
            int prev = 0;
            std::printf("{");
            for (int cls = 0; cls < 5; ++cls) {
                prev = pick_cell(types[ti], nc, cls, prev);
                std::printf("%d%s", prev, cls < 4 ? ", " : "");
            }
            std::printf("}%s", nc < 8 ? ", " : "");
        }
        std::printf("}},   // %s\n", names[ti]);
    }
    std::printf("// one column (kIl1RowsShared):\n");
    for (int ti = 0; ti < 5; ++ti) {
        std::printf("    {%d, {", types[ti]);
        int prev = 0;
        for (int cls = 0; cls < 5; ++cls) {
            prev = pick_cell(types[ti], 1, cls, prev);
            std::printf("%d%s", prev, cls < 4 ? ", " : "");
        }
        std::printf("}},   // %s\n", names[ti]);
    }
}

// the per-architecture lookup, no GPU needed for its answers
int check_tables() {
    int bad = 0;
    for (int type : {23, 12, 13, 14, 11})
        for (int nc = 2; nc <= 4; ++nc)
            for (int n_out : {512, 2560, 6144, 10240, 12288, 248320}) {
                const int a = native_mmvq_il_rows_for(86, type, nc, n_out);
                // sm_89 has no table of its own: it answers the shared table (what sm_86 answered before its own
                // 143-dense table, which differs only in Q3_K, a type the shared table does not use); sm_120 has its own
                // since 147-verify
                const int b = native_mmvq_il_rows_for(89, type, nc, n_out);
                if ((type != 11 && b != a) || (type == 11 && b != 0)) {
                    ++bad;
                    std::printf("FAIL table: sm_89 differs from sm_86 at type %d ncols %d n_out %d\n", type, nc, n_out);
                }
                if (a != 0 && a != 1 && a != 2 && a != 4) { ++bad; std::printf("FAIL table: rows %d\n", a); }
            }
    // Volta has its own table (PR 1401): Q6_K at 2 columns, 2560 rows -> 1 row a warp; the Ampere table says 0 there
    if (native_mmvq_il_rows_for(70, 14, 2, 2560) != 1 || native_mmvq_il_rows_for(86, 14, 2, 2560) != 0) {
        ++bad;
        std::printf("FAIL table: the sm_70 table is not the V100's\n");
    }
    // Pascal has its own table too (the P100's): Q6_K, 3 columns, 12288 rows -> 4 rows a warp; sm_75 still has none (the shared one)
    if (native_mmvq_il_rows_for(60, 14, 3, 12288) != 4 || native_mmvq_il_rows_for(60, 23, 2, 512) != 0) {
        ++bad;
        std::printf("FAIL table: the sm_60 table is not the P100's\n");
    }
    if (native_mmvq_il_rows_for(89, 99, 2, 2560) != 0 || native_mmvq_il_rows_for(89, 12, 5, 2560) != 0 ||
        native_mmvq_il_rows_for(86, 12, 5, 2560) != 0 || native_mmvq_il_rows_for(120, 12, 9, 2560) != 0) {
        ++bad;
        std::printf("FAIL table: an unknown type or ncols must be 0\n");
    }
    return bad;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool emit = bench && argc > 2 && std::string(argv[2]) == "--emit-table";
    cudaStream_t s;
    cudaStreamCreate(&s);
    int bad = 0;
    bad += check_tables();
    const char* only = std::getenv("MMVQ_IL_ONLY");   // 143-dense: only the cases whose name contains this
    for (const Case& c : CASES)
        if (only == nullptr || std::strstr(c.name, only) != nullptr) bad += run_case(c, bench, s);
    if (emit) emit_table();
    std::printf("%s\n", bad ? "mmvq_il_parity: FAIL" : "mmvq_il_parity: OK (every case, T 1-8, rows 1/2/4 and the table, bitwise)");
    return bad ? 1 : 0;
}
