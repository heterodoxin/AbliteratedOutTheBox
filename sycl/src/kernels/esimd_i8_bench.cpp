// esimd_i8_bench (151-b70): an ESIMD int8 DPAS expert GEMM on pre-decoded weights, us per expert.
//   esimd_i8_bench [M=80] [experts=64] [reps=10]
// Y[M][N] = sum_kb dx[m][kb] * sw[n][kb] * (Xq[m][kb*32..] . Wq[n][kb*32..])  (int8 x int8 -> int32 per 32 of K, then a float
// scale: the shape of a quantized-weight x q8_1 product).  Operands are pre-tiled in DPAS register order:
//   Xq   [E][M/8][K/32][64 dwords]   (8 rows x 32 bytes: an 8x32 A tile, row-major)
//   dx   [E][M/8][K/32][8 floats]
//   Wq   [E][N/16][K/32][8][16 dwords]  (the B tile: dword r of column n at r*16+n)
//   sw   [E][N/16][K/32][16 floats]
// Build: icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 esimd_i8_bench.cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <utility>
#include <type_traits>

#ifndef EPIV
#define EPIV 1
#endif
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

struct Args {
    const uint32_t* Xq; const float* dx; const uint32_t* Wq; const float* sw; float* Y;
    int M, N, K, MTILES, NTILES, alias;
};

template <int N, class F>
SYCL_ESIMD_FUNCTION inline void sfor(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) SYCL_ESIMD_FUNCTION {
        (f(std::integral_constant<int, I>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

template <int MT, int NT, int GRF, int EPI>
void run(sycl::queue& q, const Args a, int E) {
    const int nkb = a.K / 32;
    const int mtg = (a.MTILES + MT - 1) / MT, ntg = a.NTILES / NT;
    const size_t nthr = (size_t) E * mtg * ntg;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<GRF>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nthr), props, [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
            struct St {
                simd<uint32_t, 128 * NT> b; simd<float, 16 * NT> sw; simd<uint32_t, 64 * MT> a; simd<float, 8 * MT> dx;
            };
            const int t = (int) id[0];
            const int nt0 = (t % ntg) * NT, mt0 = ((t / ntg) % mtg) * MT, e = t / (ntg * mtg);
            simd<float, 128 * MT * NT> acc = 0.f;
            simd<int, 128 * MT * NT> acci = 0;
            const uint32_t* Xe = a.Xq + (size_t) e * a.MTILES * nkb * 64;
            const float* dxe = a.dx + (size_t) e * a.MTILES * nkb * 8;
            const uint32_t* We = a.Wq + (size_t) (a.alias ? 0 : e) * a.NTILES * nkb * 128;
            const float* swe = a.sw + (size_t) (a.alias ? 0 : e) * a.NTILES * nkb * 16;
            St sA, sB;
            auto ld = [&](int kb, St& s) SYCL_ESIMD_FUNCTION {
                sfor<NT>([&](auto jj) SYCL_ESIMD_FUNCTION {
                    constexpr int j = decltype(jj)::value;
                    s.b.template select<128, 1>(128 * j) = block_load<uint32_t, 128>(We + ((size_t) (nt0 + j) * nkb + kb) * 128);
                    s.sw.template select<16, 1>(16 * j) = block_load<float, 16>(swe + ((size_t) (nt0 + j) * nkb + kb) * 16);
                });
                sfor<MT>([&](auto ii) SYCL_ESIMD_FUNCTION {
                    constexpr int i = decltype(ii)::value;
                    const int mi = (mt0 + i < a.MTILES) ? mt0 + i : a.MTILES - 1;
                    s.a.template select<64, 1>(64 * i) = block_load<uint32_t, 64>(Xe + ((size_t) mi * nkb + kb) * 64);
                    s.dx.template select<8, 1>(8 * i) = block_load<float, 8>(dxe + ((size_t) mi * nkb + kb) * 8);
                });
            };
            auto cmp = [&](int kb, St& s) SYCL_ESIMD_FUNCTION {
                sfor<MT>([&](auto ii) SYCL_ESIMD_FUNCTION {
                    constexpr int i = decltype(ii)::value;
                    sfor<NT>([&](auto jj) SYCL_ESIMD_FUNCTION {
                        constexpr int j = decltype(jj)::value;
                        constexpr int o = 128 * (i * NT + j);
                        simd<int, 128> bv = s.b.template select<128, 1>(128 * j).template bit_cast_view<int>();
                        simd<int, 64> av = s.a.template select<64, 1>(64 * i).template bit_cast_view<int>();
                        if constexpr (EPI == 0) {
                            simd<int, 128> c0 = acci.template select<128, 1>(o);
                            acci.template select<128, 1>(o) = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(c0, bv, av);
                        } else if constexpr (EPI == 1) {
                            simd<int, 128> tmp = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(
                                simd<int, 128>(0), bv, av);
                            simd<float, 128> tf = tmp;
                            simd<float, 16> sw = s.sw.template select<16, 1>(16 * j);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float dxr = s.dx[8 * i + r];
                                acc.template select<16, 1>(o + 16 * r) += tf.template select<16, 1>(16 * r) * (sw * dxr);
                            }
                        } else {
                            simd<int, 128> tmp = xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(
                                simd<int, 128>(0), bv, av);
                            simd<float, 16> sw = s.sw.template select<16, 1>(16 * j);
                            simd<int, 16> isw = convert<int>(sw * 50.f);
#pragma unroll
                            for (int r = 0; r < 8; ++r)
                                acci.template select<16, 1>(o + 16 * r) += tmp.template select<16, 1>(16 * r) * isw;
                            if ((kb & 3) == 3) {
                                simd<float, 128> tf = acci.template select<128, 1>(o);
                                acci.template select<128, 1>(o) = 0;
#pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    const float dxr = s.dx[8 * i + r];
                                    acc.template select<16, 1>(o + 16 * r) += tf.template select<16, 1>(16 * r) * (sw * dxr);
                                }
                            }
                        }
                    });
                });
            };
            ld(0, sA);
            for (int kb = 0; kb < nkb; kb += 2) {
                ld(kb + 1, sB);
                cmp(kb, sA);
                if (kb + 2 < nkb) ld(kb + 2, sA);
                cmp(kb + 1, sB);
            }
            if constexpr (EPI == 0) acc = acci;
            sfor<MT>([&](auto ii) SYCL_ESIMD_FUNCTION {
                constexpr int i = decltype(ii)::value;
                sfor<NT>([&](auto jj) SYCL_ESIMD_FUNCTION {
                    constexpr int j = decltype(jj)::value;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int m = (mt0 + i) * 8 + r;
                        if (m < a.M) block_store<float, 16>(a.Y + ((size_t) e * a.M + m) * a.N + (nt0 + j) * 16,
                                                            acc.template select<16, 1>(128 * (i * NT + j) + 16 * r));
                    }
                });
            });
        });
    });
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 80;
    const int E = argc > 2 ? std::atoi(argv[2]) : 64;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device %s, M %d, %d experts\n", q.get_device().get_info<sycl::info::device::name>().c_str(), M, E);
    struct Shape { const char* nm; int N, K; } shapes[2] = {{"gate+up", 1280, 2560}, {"down", 2560, 640}};
    std::mt19937 rng(7);
    for (const Shape& sh : shapes) {
        const int N = sh.N, K = sh.K, nkb = K / 32, MTILES = (M + 7) / 8, NTILES = N / 16;
        const size_t nX = (size_t) E * MTILES * nkb * 64, nW = (size_t) E * NTILES * nkb * 128;
        std::vector<uint32_t> hX(nX), hW(nW);
        std::vector<float> hdx((size_t) E * MTILES * nkb * 8), hsw((size_t) E * NTILES * nkb * 16);
        for (auto& x : hX) x = rng(); for (auto& x : hW) x = rng() & 0x0f0f0f0fu;   // small weights, |w| <= 15 like a grid
        for (auto& x : hdx) x = 0.01f * (1 + rng() % 8);
        for (auto& x : hsw) x = 0.02f * (1 + rng() % 8);
        uint32_t *dX = sycl::malloc_device<uint32_t>(nX, q), *dW = sycl::malloc_device<uint32_t>(nW, q);
        float *ddx = sycl::malloc_device<float>(hdx.size(), q), *dsw = sycl::malloc_device<float>(hsw.size(), q);
        float* dY = sycl::malloc_device<float>((size_t) E * M * N, q);
        q.memcpy(dX, hX.data(), nX * 4); q.memcpy(dW, hW.data(), nW * 4);
        q.memcpy(ddx, hdx.data(), hdx.size() * 4); q.memcpy(dsw, hsw.data(), hsw.size() * 4); q.wait();
        Args a{dX, ddx, dW, dsw, dY, M, N, K, MTILES, NTILES, argc > 4 ? std::atoi(argv[4]) : 0};
        auto time = [&](auto&& fn) {
            fn(); q.wait();
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                fn(); q.wait();
                best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            return best / E;
        };
        const double ops = 2.0 * M * N * K;
        struct V { const char* nm; double t; } vs[6];
        vs[0] = {"2x2g256", time([&] { run<2, 2, 256, EPIV>(q, a, E); })};
        vs[1] = {"2x2g128", time([&] { run<2, 2, 128, EPIV>(q, a, E); })};
        vs[2] = {"1x2g128", time([&] { run<1, 2, 128, EPIV>(q, a, E); })};
        vs[3] = {"2x1g128", time([&] { run<2, 1, 128, EPIV>(q, a, E); })};
        vs[4] = {"1x1g128", time([&] { run<1, 1, 128, EPIV>(q, a, E); })};
        vs[5] = {"1x4g256", time([&] { run<1, 4, 256, EPIV>(q, a, E); })};
        // check expert 3 (last run's output: MT8 NT1) against a host reference
        std::vector<float> hY((size_t) M * N);
        q.memcpy(hY.data(), dY + (size_t) 3 * M * N, hY.size() * 4).wait();
        double maxerr = 0, maxref = 0;
        for (int m = 0; m < M; m += 5) for (int n = 0; n < N; n += 7) {
            double ref = 0;
            const int mt = m / 8, r = m % 8, nt = n / 16, c = n % 16;
            for (int kb = 0; kb < nkb; ++kb) {
                long long s = 0;
                for (int k = 0; k < 32; ++k) {
                    // A tile: row r at dwords r*8 .. ; B tile: dword (k/4) of column c at (k/4)*16 + c
                    const uint32_t aw = hX[(((size_t) 3 * MTILES + mt) * nkb + kb) * 64 + r * 8 + k / 4];
                    const uint32_t bw = hW[(((size_t) 3 * NTILES + nt) * nkb + kb) * 128 + (k / 4) * 16 + c];
                    s += (long long) (int8_t) (aw >> (8 * (k % 4))) * (int8_t) (bw >> (8 * (k % 4)));
                }
                ref += (double) s * hdx[(((size_t) 3 * MTILES + mt) * nkb + kb) * 8 + r] * hsw[(((size_t) 3 * NTILES + nt) * nkb + kb) * 16 + c];
            }
            maxref = std::max(maxref, std::fabs(ref)); maxerr = std::max(maxerr, std::fabs(ref - hY[(size_t) m * N + n]));
        }
        std::printf("%-8s N %d K %d:", sh.nm, N, K);
        for (auto& v : vs) std::printf("  %s %.1f us (%.0f TOPS)", v.nm, v.t, ops / v.t / 1e6);
        std::printf("\n          max err %.3g of %.3g\n", maxerr, maxref);
        sycl::free(dX, q); sycl::free(dW, q); sycl::free(ddx, q); sycl::free(dsw, q); sycl::free(dY, q);
    }
    return 0;
}
