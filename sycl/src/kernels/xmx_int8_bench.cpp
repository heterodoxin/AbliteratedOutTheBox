// xmx_int8_bench (0.1.44): what a DPAS int8 expert GEMM costs next to the FP16 joint_matrix one, on one device.
//   xmx_int8_bench [M=160] [experts=64] [reps=20]      build: -DSTRATA_XMX_SG16 for Xe2 (8x16x32), none for DG2 (8x8x32)
// Shape of a Strata expert: gate+up N=1280, K=2560 (down: N=2560, K=640).  Modes:
//   f16    : half A (M x K) and half W (N x K), float accumulate - the GEMM the current dequant + oneMKL path runs
//   i8     : int8 A and W, int32 accumulate over the whole K (no per-block scale): the DPAS floor
//   i8blk  : int8 A and W, one DPAS per 32 of K into a zeroed int32 tile, then acc += float(tile) * dw[n] * dx[m][kb]
//            (the per-32 scales of the activations and of the IQ weights): the real fused-kernel epilogue
// Weights are random int8 (decode cost excluded; a real kernel adds the IQ grid decode).  E different experts, so the
// weights come from DRAM, not L2.  Prints us per expert and the effective TOPS; checks i8blk against a host reference.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace jm = sycl::ext::oneapi::experimental::matrix;
using half = sycl::half;
#ifdef STRATA_XMX_SG16
constexpr int SGW = 16;
#else
constexpr int SGW = 8;
#endif
constexpr int SGM = 2, SGN = 4, MT = 4, NT = 2;          // 8 sub-groups: tile 64 x (4*2*SGW)
constexpr int BM = SGM * MT * 8, BN = SGN * NT * SGW, NSG = SGM * SGN, WG = NSG * SGW;

// mode 0: f16
template <class T>
struct Slm;
void run_f16(sycl::queue& q, const half* A, const half* W, float* C, int M, int N, int K, int E) {
    constexpr int KS = 32, LD = KS + 8;
    const int nmt = (M + BM - 1) / BM, nnt = N / BN;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sa(BM * LD, h), sb(BN * LD, h);
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        h.parallel_for<class k_f16>(sycl::nd_range<3>({(size_t) E, (size_t) nmt, (size_t) nnt * WG}, {1, 1, (size_t) WG}), props,
                                    [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SGW)]] {
            auto sg = it.get_sub_group();
            const int e = (int) it.get_group(0), m0 = (int) it.get_group(1) * BM, n0 = (int) it.get_group(2) * BN;
            const int lid = (int) it.get_local_id(2), sgid = lid / SGW, sgm = sgid / SGN, sgn = sgid % SGN;
            const half* We = W + (size_t) e * N * K;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, SGW> acc[MT][NT];
            for (int i = 0; i < MT; ++i) for (int j = 0; j < NT; ++j) jm::joint_matrix_fill(sg, acc[i][j], 0.0f);
            half* pa = sa.template get_multi_ptr<sycl::access::decorated::no>().get();
            half* pb = sb.template get_multi_ptr<sycl::access::decorated::no>().get();
            for (int k0 = 0; k0 < K; k0 += KS) {
                for (int v = lid; v < BM * KS / 8; v += WG) {
                    const int r = v / (KS / 8), c = (v % (KS / 8)) * 8;
                    sycl::vec<half, 8> x(0);
                    if (m0 + r < M) x = *reinterpret_cast<const sycl::vec<half, 8>*>(A + (size_t) (m0 + r) * K + k0 + c);
                    *reinterpret_cast<sycl::vec<half, 8>*>(pa + r * LD + c) = x;
                }
                for (int v = lid; v < BN * KS / 8; v += WG) {
                    const int r = v / (KS / 8), c = (v % (KS / 8)) * 8;
                    *reinterpret_cast<sycl::vec<half, 8>*>(pb + r * LD + c) =
                        *reinterpret_cast<const sycl::vec<half, 8>*>(We + (size_t) (n0 + r) * K + k0 + c);
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int kk = 0; kk < KS; kk += 16) {
                    jm::joint_matrix<sycl::sub_group, half, jm::use::a, 8, 16, jm::layout::row_major> a[MT];
                    jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, SGW, jm::layout::col_major> b[NT];
                    for (int i = 0; i < MT; ++i)
                        jm::joint_matrix_load(sg, a[i], sa.template get_multi_ptr<sycl::access::decorated::no>() + (sgm * MT * 8 + i * 8) * LD + kk, LD);
                    for (int j = 0; j < NT; ++j)
                        jm::joint_matrix_load(sg, b[j], sb.template get_multi_ptr<sycl::access::decorated::no>() + (sgn * NT * SGW + j * SGW) * LD + kk, LD);
                    for (int i = 0; i < MT; ++i) for (int j = 0; j < NT; ++j) jm::joint_matrix_mad(sg, acc[i][j], a[i], b[j], acc[i][j]);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            for (int i = 0; i < MT; ++i) {
                const int r = m0 + sgm * MT * 8 + i * 8;
                if (r + 8 > M) continue;   // the bench uses M a multiple of 8; rows beyond are zero-padded work only
                for (int j = 0; j < NT; ++j)
                    jm::joint_matrix_store(sg, acc[i][j], sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        C + ((size_t) e * M + r) * N + n0 + sgn * NT * SGW + j * SGW), N, jm::layout::row_major);
            }
        });
    });
}

// mode 1/2: int8.  BLK: per-32 float epilogue (dw: [E][N][K/32] float, dx: [M][K/32] float)
constexpr int KI = 32;   // joint_matrix int8 K
template <bool BLK, class Name>
void run_i8(sycl::queue& q, const int8_t* A, const int8_t* W, const float* dw, const float* dx, float* C, int M, int N, int K, int E) {
    constexpr int KS = 64, LD = KS + 16;   // bytes
    const int nmt = (M + BM - 1) / BM, nnt = N / BN, nkb = K / 32;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int8_t, 1> sa(BM * LD, h), sb(BN * LD, h);
        sycl::local_accessor<float, 1> sdw(BN * 2, h), sdx(BM * 2, h);
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        h.parallel_for<Name>(sycl::nd_range<3>({(size_t) E, (size_t) nmt, (size_t) nnt * WG}, {1, 1, (size_t) WG}), props,
                             [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SGW)]] {
            auto sg = it.get_sub_group();
            const int e = (int) it.get_group(0), m0 = (int) it.get_group(1) * BM, n0 = (int) it.get_group(2) * BN;
            const int lid = (int) it.get_local_id(2), sgid = lid / SGW, sgm = sgid / SGN, sgn = sgid % SGN;
            const int8_t* We = W + (size_t) e * N * K;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, SGW> acc[MT][NT];
            jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, SGW> tmp[MT][NT];
            for (int i = 0; i < MT; ++i) for (int j = 0; j < NT; ++j) {
                jm::joint_matrix_fill(sg, acc[i][j], 0.0f);
                jm::joint_matrix_fill(sg, tmp[i][j], 0);
            }
            int8_t* pa = sa.template get_multi_ptr<sycl::access::decorated::no>().get();
            int8_t* pb = sb.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* pdw = sdw.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* pdx = sdx.template get_multi_ptr<sycl::access::decorated::no>().get();
            for (int k0 = 0; k0 < K; k0 += KS) {
                for (int v = lid; v < BM * KS / 16; v += WG) {
                    const int r = v / (KS / 16), c = (v % (KS / 16)) * 16;
                    sycl::vec<int8_t, 16> x(0);
                    if (m0 + r < M) x = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(A + (size_t) (m0 + r) * K + k0 + c);
                    *reinterpret_cast<sycl::vec<int8_t, 16>*>(pa + r * LD + c) = x;
                }
                for (int v = lid; v < BN * KS / 16; v += WG) {
                    const int r = v / (KS / 16), c = (v % (KS / 16)) * 16;
                    *reinterpret_cast<sycl::vec<int8_t, 16>*>(pb + r * LD + c) =
                        *reinterpret_cast<const sycl::vec<int8_t, 16>*>(We + (size_t) (n0 + r) * K + k0 + c);
                }
                if (BLK) {
                    for (int v = lid; v < BN * 2; v += WG) pdw[v] = dw[((size_t) e * N + n0 + v / 2) * nkb + k0 / 32 + v % 2];
                    for (int v = lid; v < BM * 2; v += WG) pdx[v] = (m0 + v / 2 < M) ? dx[(size_t) (m0 + v / 2) * nkb + k0 / 32 + v % 2] : 0.f;
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int kk = 0; kk < KS; kk += KI) {
                    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, KI, jm::layout::row_major> a[MT];
                    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, KI, SGW, jm::layout::col_major> b[NT];
                    for (int i = 0; i < MT; ++i)
                        jm::joint_matrix_load(sg, a[i], sa.template get_multi_ptr<sycl::access::decorated::no>() + (sgm * MT * 8 + i * 8) * LD + kk, LD);
                    for (int j = 0; j < NT; ++j)
                        jm::joint_matrix_load(sg, b[j], sb.template get_multi_ptr<sycl::access::decorated::no>() + (sgn * NT * SGW + j * SGW) * LD + kk, LD);
                    if (BLK) {
                        for (int i = 0; i < MT; ++i) for (int j = 0; j < NT; ++j) {
                            jm::joint_matrix_fill(sg, tmp[i][j], 0);
                            jm::joint_matrix_mad(sg, tmp[i][j], a[i], b[j], tmp[i][j]);
                            const float sw = pdw[(sgn * NT * SGW + j * SGW + (int) sg.get_local_linear_id()) * 2 + kk / 32];
                            const int rbase = sgm * MT * 8 + i * 8;
                            int ei = 0;
                            jm::joint_matrix_apply(sg, tmp[i][j], acc[i][j], [&](int32_t& t, float& ac) {
                                ac = sycl::fma((float) t * sw, pdx[(rbase + ei) * 2 + kk / 32], ac);
                                ++ei;
                            });
                        }
                    } else {
                        for (int i = 0; i < MT; ++i) for (int j = 0; j < NT; ++j) jm::joint_matrix_mad(sg, tmp[i][j], a[i], b[j], tmp[i][j]);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            for (int i = 0; i < MT; ++i) {
                const int r = m0 + sgm * MT * 8 + i * 8;
                if (r + 8 > M) continue;
                for (int j = 0; j < NT; ++j) {
                    if (!BLK) {   // int32 -> float through a per-element copy
                        jm::joint_matrix_apply(sg, tmp[i][j], acc[i][j], [&](int32_t& t, float& ac) { ac = (float) t; });
                    }
                    jm::joint_matrix_store(sg, acc[i][j], sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        C + ((size_t) e * M + r) * N + n0 + sgn * NT * SGW + j * SGW), N, jm::layout::row_major);
                }
            }
        });
    });
}
class k_i8; class k_i8blk;

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 160;
    const int E = argc > 2 ? std::atoi(argv[2]) : 64;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device %s, sub-group %d, M %d, %d experts\n", q.get_device().get_info<sycl::info::device::name>().c_str(), SGW, M, E);
    struct Shape { const char* nm; int N, K; } shapes[2] = {{"gate+up", 1280, 2560}, {"down", 2560, 640}};
    std::mt19937 rng(5);
    for (const Shape& sh : shapes) {
        const int N = sh.N, K = sh.K, nkb = K / 32;
        const size_t nw = (size_t) E * N * K, na = (size_t) M * K;
        std::vector<int8_t> hA(na), hW(nw);
        std::vector<half> hAh(na);
        for (size_t i = 0; i < na; ++i) { hA[i] = (int8_t) ((int) (rng() % 255) - 127); hAh[i] = half((float) hA[i] / 64.f); }
        for (auto& x : hW) x = (int8_t) ((int) (rng() % 31) - 15);
        std::vector<float> hdw((size_t) E * N * nkb), hdx((size_t) M * nkb);
        for (auto& x : hdw) x = 0.01f * (1 + rng() % 8);
        for (auto& x : hdx) x = 0.02f * (1 + rng() % 8);
        int8_t *dA = sycl::malloc_device<int8_t>(na, q), *dW = sycl::malloc_device<int8_t>(nw, q);
        half *dAh = sycl::malloc_device<half>(na, q), *dWh = sycl::malloc_device<half>(nw, q);
        float *ddw = sycl::malloc_device<float>(hdw.size(), q), *ddx = sycl::malloc_device<float>(hdx.size(), q);
        float* dC = sycl::malloc_device<float>((size_t) E * M * N, q);
        q.memcpy(dA, hA.data(), na).wait(); q.memcpy(dW, hW.data(), nw).wait();
        q.memcpy(dAh, hAh.data(), na * 2).wait();
        { std::vector<half> t(nw); for (size_t i = 0; i < nw; ++i) t[i] = half((float) hW[i]); q.memcpy(dWh, t.data(), nw * 2).wait(); }
        q.memcpy(ddw, hdw.data(), hdw.size() * 4).wait(); q.memcpy(ddx, hdx.data(), hdx.size() * 4).wait();
        auto time = [&](auto&& fn) {
            fn(); q.wait();
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                fn(); q.wait();
                best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            return best / E;   // us per expert (launch overhead amortised over E experts)
        };
        const double ops = 2.0 * M * N * K;
        const double t0 = time([&] { run_f16(q, dAh, dWh, dC, M, N, K, E); });
        const double t1 = time([&] { run_i8<false, k_i8>(q, dA, dW, ddw, ddx, dC, M, N, K, E); });
        const double t2 = time([&] { run_i8<true, k_i8blk>(q, dA, dW, ddw, ddx, dC, M, N, K, E); });
        // check i8blk expert 3 against a host reference
        std::vector<float> hC((size_t) M * N);
        q.memcpy(hC.data(), dC + (size_t) 3 * M * N, hC.size() * 4).wait();
        double maxerr = 0, maxref = 0;
        for (int m = 0; m < M; m += 7) for (int n = 0; n < N; n += 11) {
            double ref = 0;
            for (int kb = 0; kb < nkb; ++kb) {
                long long s = 0;
                for (int k = 0; k < 32; ++k) s += (long long) hA[(size_t) m * K + kb * 32 + k] * hW[((size_t) 3 * N + n) * K + kb * 32 + k];
                ref += (double) s * hdw[((size_t) 3 * N + n) * nkb + kb] * hdx[(size_t) m * nkb + kb];
            }
            maxref = std::max(maxref, std::fabs(ref)); maxerr = std::max(maxerr, std::fabs(ref - hC[(size_t) m * N + n]));
        }
        std::printf("%-8s N %d K %d: f16 %.1f us/expert (%.1f TFLOPs), i8 %.1f us (%.1f TOPS), i8blk %.1f us (%.1f TOPS); i8blk max err %.3g of %.3g\n",
                    sh.nm, N, K, t0, ops / t0 / 1e6, t1, ops / t1 / 1e6, t2, ops / t2 / 1e6, maxerr, maxref);
        sycl::free(dA, q); sycl::free(dW, q); sycl::free(dAh, q); sycl::free(dWh, q); sycl::free(ddw, q); sycl::free(ddx, q); sycl::free(dC, q);
    }
    return 0;
}
