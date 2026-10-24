// esimd_iq3s_decode_bench (151-b70): IQ3_S rows decoded to DPAS B fragments (int8) by an ESIMD kernel, lanes = 16 rows.
//   esimd_iq3s_decode_bench [N=1280] [K=2560] [experts=16] [write=1]
// Output per expert: Wq [N/16][K/32][8][16] dwords (the B tile of a DPAS s8 8x16x32: dword r of row n at r*16+n), sw [N/16][K/32][16] floats
// (d * (1 + 2 s) per row and 32-block).  Checked against a host decode with ggml's own grid.  Reports us per expert and GB/s of
// compressed weights consumed.  write=0 keeps the fragments in registers (an xor checksum) to separate decode from the plane write.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <utility>
#include <type_traits>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

using namespace sycl::ext::intel::esimd;

constexpr int BLK = 110;   // sizeof(block_iq3_s)

template <int N, class F>
SYCL_ESIMD_FUNCTION __attribute__((always_inline)) inline void sfor(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) SYCL_ESIMD_FUNCTION {
        (f(std::integral_constant<int, I>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

// unaligned 32-bit read at a 2-byte aligned byte offset (per lane) through two dword gathers
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld32(const uint8_t* base, simd<uint32_t, 16> off) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 16> lo = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a);
    simd<uint32_t, 16> hi = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a + 4u);
    simd<uint32_t, 16> sh = (off & 2u) << 3;   // 0 or 16
    return (lo >> sh) | ((hi << 16) << (16u - sh));
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld16(const uint8_t* base, simd<uint32_t, 16> off) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 16> lo = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a);
    return (lo >> ((off & 2u) << 3)) & 0xffffu;
}

template <int WRITE>
void run(sycl::queue& q, const uint8_t* W, const uint32_t* grid, uint32_t* Wq, float* sw, uint32_t* sink, int N, int K, int E) {
    const int nsb = K / 256, ntiles = N / 16, rowb = nsb * BLK;
    const size_t nthr = (size_t) E * ntiles;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<128>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nthr), props, [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
            const int t = (int) id[0], nt = t % ntiles, e = t / ntiles;
            const uint8_t* We = W + (size_t) e * N * rowb;
            simd<uint32_t, 16> rowoff = (simd<uint32_t, 16>(0, 1) + (uint32_t) (nt * 16)) * (uint32_t) rowb;
            simd<uint32_t, 16> chk = 0;
            for (int sb = 0; sb < nsb; ++sb) {
                const uint32_t P = sb * BLK;
                simd<uint32_t, 16> d16 = ld16(We, rowoff + P);
                simd<uint16_t, 16> dlo = d16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
                simd<sycl::half, 16> dh16 = dlo.template bit_cast_view<sycl::half>();
                simd<float, 16> d = dh16;
                simd<uint32_t, 16> sc4 = ld32(We, rowoff + (P + 106));
                simd<uint32_t, 16> qh4 = ld32(We, rowoff + (P + 66));
                simd<uint32_t, 16> qh8 = ld32(We, rowoff + (P + 70));
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    simd<uint32_t, 16> q0 = ld32(We, rowoff + (P + 2 + 8 * ib));
                    simd<uint32_t, 16> q1 = ld32(We, rowoff + (P + 6 + 8 * ib));
                    simd<uint32_t, 16> sg = ld32(We, rowoff + (P + 74 + 4 * ib));
                    simd<uint32_t, 16> qhv = ((ib < 4 ? qh4 : qh8) >> (8 * (ib & 3))) & 0xffu;
                    simd<uint32_t, 16> scn = ((sc4 >> (8 * (ib >> 1))) >> (4 * (ib & 1))) & 0xfu;
                    simd<float, 16> swv = d * simd<float, 16>(simd<int, 16>(scn * 2u + 1u));
                    simd<uint32_t, 128> frag;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                        simd<uint32_t, 16> idx = ((qsw >> (8 * (r & 3))) & 0xffu) | (((qhv >> r) & 1u) << 8);
                        simd<uint32_t, 16> g = gather<uint32_t, 16>(grid, idx << 2);
                        simd<uint32_t, 16> nib = ((sg >> (8 * (r >> 1))) >> (4 * (r & 1))) & 0xfu;
                        simd<uint32_t, 16> m = ((nib * 0x00204081u) & 0x01010101u);
                        simd<uint32_t, 16> mask = m * 255u;
                        frag.template select<16, 1>(16 * r) = (g ^ mask) + m;
                    }
                    if constexpr (WRITE) {
                        const size_t fi = ((size_t) e * ntiles + nt) * (K / 32) + sb * 8 + ib;
                        block_store<uint32_t, 128>(Wq + fi * 128, frag);
                        block_store<float, 16>(sw + fi * 16, swv);
                    } else {
                        simd<uint32_t, 16> x = swv.template bit_cast_view<uint32_t>();
#pragma unroll
                        for (int r = 0; r < 8; ++r) x ^= frag.template select<16, 1>(16 * r);
                        chk ^= x;
                    }
                }
            }
            if constexpr (!WRITE) block_store<uint32_t, 16>(sink + (size_t) t * 16, chk);
        });
    });
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 1280, K = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int E = argc > 3 ? std::atoi(argv[3]) : 16, WR = argc > 4 ? std::atoi(argv[4]) : 1;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device %s, N %d K %d, %d experts, write %d\n", q.get_device().get_info<sycl::info::device::name>().c_str(), N, K, E, WR);
    const int nsb = K / 256, ntiles = N / 16, rowb = nsb * BLK;
    std::mt19937 rng(11);
    std::vector<uint8_t> hW((size_t) E * N * rowb);
    for (size_t i = 0; i < hW.size(); ++i) hW[i] = (uint8_t) rng();
    for (size_t b = 0; b < hW.size() / BLK; ++b) {   // sane fp16 d
        const sycl::half dh = sycl::half(0.002f + 0.0001f * (rng() % 50));
        std::memcpy(&hW[b * BLK], &dh, 2);
    }
    uint8_t* dW = sycl::malloc_device<uint8_t>(hW.size() + 4096, q);
    uint32_t* dgrid = sycl::malloc_device<uint32_t>(512, q);
    uint32_t* dWq = sycl::malloc_device<uint32_t>((size_t) E * ntiles * (K / 32) * 128, q);
    float* dsw = sycl::malloc_device<float>((size_t) E * ntiles * (K / 32) * 16, q);
    uint32_t* dsink = sycl::malloc_device<uint32_t>((size_t) E * ntiles * 16, q);
    q.memcpy(dW, hW.data(), hW.size()); q.memcpy(dgrid, iq3s_grid, 512 * 4); q.wait();
    auto go = [&] { if (WR) run<1>(q, dW, dgrid, dWq, dsw, dsink, N, K, E); else run<0>(q, dW, dgrid, dWq, dsw, dsink, N, K, E); q.wait(); };
    go();
    double best = 1e30;
    for (int r = 0; r < 10; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        go();
        best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("decode: %.1f us per expert (%.0f GB/s of compressed weights)\n", best / E, (double) N * rowb / (best / E) / 1e3);
    if (WR) {   // host check of expert 1
        const int e = E > 1 ? 1 : 0;
        std::vector<uint32_t> hQ((size_t) ntiles * (K / 32) * 128);
        std::vector<float> hS((size_t) ntiles * (K / 32) * 16);
        q.memcpy(hQ.data(), dWq + (size_t) e * hQ.size(), hQ.size() * 4).wait();
        q.memcpy(hS.data(), dsw + (size_t) e * hS.size(), hS.size() * 4).wait();
        long bad = 0, badS = 0;
        for (int n = 0; n < N; n += 3) {
            for (int sb = 0; sb < nsb; ++sb) {
                const uint8_t* blk = &hW[((size_t) e * N + n) * rowb + (size_t) sb * BLK];
                sycl::half dh; std::memcpy(&dh, blk, 2);
                const uint8_t *qs = blk + 2, *qh = blk + 66, *sgn = blk + 74, *scl = blk + 106;
                for (int ib = 0; ib < 8; ++ib) {
                    const int sc = (scl[ib / 2] >> (4 * (ib % 2))) & 0xf;
                    const float swr = (float) dh * (1 + 2 * sc);
                    const size_t fi = ((size_t) (n / 16) * (K / 32) + sb * 8 + ib);
                    if (std::fabs(hS[fi * 16 + n % 16] - swr) > 1e-6f * std::fabs(swr)) ++badS;
                    for (int r = 0; r < 8; ++r) {
                        const int idx = qs[8 * ib + r] | (((qh[ib] >> r) & 1) << 8);
                        const uint8_t* g = (const uint8_t*) &iq3s_grid[idx];
                        const uint8_t sbyte = sgn[4 * ib + r / 2];
                        uint32_t want = 0;
                        for (int j = 0; j < 4; ++j) {
                            const int kk = 4 * (r & 1) + j;
                            const int v = (sbyte >> kk & 1) ? -(int) g[j] : (int) g[j];
                            want |= (uint32_t) (uint8_t) (int8_t) v << (8 * j);
                        }
                        if (hQ[fi * 128 + r * 16 + n % 16] != want) ++bad;
                    }
                }
            }
        }
        std::printf("check expert %d: %ld bad dwords, %ld bad scales\n", e, bad, badS);
    }
    return 0;
}
