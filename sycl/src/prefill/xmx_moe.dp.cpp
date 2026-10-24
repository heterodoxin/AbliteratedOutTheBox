// A770 port: grouped FP16 XMX GEMM for the prompt path's experts (STRATA_PF_XMX=1).
// Y[r][n] = sum_k X[r][k] * W_e[n][k] for each expert e of a group, its rows r in [row0 + off[e], row0 + off[e+1]).
// One launch per group instead of one oneMKL call per expert.  joint_matrix 8x8x16 (DG2), SLM-tiled, 256 GRF.
// Measured standalone on the A770 (64 experts, gamma-distributed rows, vs per-expert oneMKL): down 6.1x at mean 40
// rows, 2.8x at 160, 1.75x at 320; gate/up 3.0x / 1.46x / 0.79x (oneMKL wins above ~250 rows: the caller picks).
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdlib>
#include <cstdio>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/xmx_moe.hpp"

namespace strata::prefill::xmx {
namespace {
using half = sycl::half;
namespace jm = sycl::ext::oneapi::experimental::matrix;
// Sub-group width = the XMX tile's N: 8 on Alchemist (DG2, joint_matrix 8x8x16), 16 on Xe2 (Battlemage, 8x16x16; the
// CMake switch STRATA_XMX_SG16 for an AOT bmg / lnl build).  A build has one shape: the other is not valid there.
#ifdef STRATA_XMX_SG16
constexpr int kSgW = 16;
#else
constexpr int kSgW = 8;
#endif

struct Grp {
    const half* w[kMaxGroup];
    int off[kMaxGroup + 1];     // rows, relative to row0
    int tstart[kMaxGroup + 1];  // first M-tile of each expert
    int n;
};

template <int SGM_, int SGN_, int MT_, int NT_, int KS_>
struct Cfg {
    static constexpr int SGM = SGM_, SGN = SGN_, MT = MT_, NT = NT_, KS = KS_;
    static constexpr int SG = kSgW;
    static constexpr int BM = SGM * MT * 8, BN = SGN * NT * kSgW, NSG = SGM * SGN, WG = NSG * SG;
};

template <class C, class Name>
void launch(sycl::queue& q, const half* X, float* Y, const Grp& gp, int ntiles, int N, int K) {
    constexpr int BM = C::BM, BN = C::BN, KS = C::KS, WG = C::WG, SG = C::SG;
    constexpr int LDA = KS + 8, LDB = KS + 8;
    const int nb = N / BN;
#ifndef STRATA_NO_DG2_XMX
    constexpr int TN = kSgW;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sa(BM * LDA, h), sb(BN * LDB, h);
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        h.parallel_for<Name>(sycl::nd_range<2>({(size_t) ntiles, (size_t) nb * WG}, {1, (size_t) WG}), props,
                             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSgW)]] {
            auto sg = it.get_sub_group();
            const int t = (int) it.get_group(0);
            int e = 0;
            while (e + 1 < gp.n && t >= gp.tstart[e + 1]) ++e;
            const int m0 = (t - gp.tstart[e]) * BM;
            const int rows = sycl::min(BM, gp.off[e + 1] - gp.off[e] - m0);
            const half* Xe = X + ((size_t) gp.off[e] + m0) * K;
            float* Ye = Y + ((size_t) gp.off[e] + m0) * N;
            const half* We = gp.w[e];
            const int n0 = (int) it.get_group(1) * BN;
            const int lid = (int) it.get_local_id(1), sgid = lid / SG;
            const int sgm = sgid / C::SGN, sgn = sgid % C::SGN;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, TN> acc[C::MT][C::NT];
            for (int i = 0; i < C::MT; ++i)
                for (int j = 0; j < C::NT; ++j) jm::joint_matrix_fill(sg, acc[i][j], 0.0f);
            half* pa = sa.template get_multi_ptr<sycl::access::decorated::no>().get();
            half* pb = sb.template get_multi_ptr<sycl::access::decorated::no>().get();
            for (int k0 = 0; k0 < K; k0 += KS) {
                constexpr int VA = BM * KS / 8, VB = BN * KS / 8;
                for (int v = lid; v < VA; v += WG) {
                    const int r = v / (KS / 8), c = (v % (KS / 8)) * 8;
                    sycl::vec<half, 8> x(0);
                    if (r < rows) x = *reinterpret_cast<const sycl::vec<half, 8>*>(Xe + (size_t) r * K + k0 + c);
                    *reinterpret_cast<sycl::vec<half, 8>*>(pa + r * LDA + c) = x;
                }
                for (int v = lid; v < VB; v += WG) {
                    const int r = v / (KS / 8), c = (v % (KS / 8)) * 8;
                    *reinterpret_cast<sycl::vec<half, 8>*>(pb + r * LDB + c) =
                        *reinterpret_cast<const sycl::vec<half, 8>*>(We + (size_t) (n0 + r) * K + k0 + c);
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int kk = 0; kk < KS; kk += 16) {
                    jm::joint_matrix<sycl::sub_group, half, jm::use::a, 8, 16, jm::layout::row_major> a[C::MT];
                    jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, TN, jm::layout::col_major> b[C::NT];
                    for (int i = 0; i < C::MT; ++i)
                        jm::joint_matrix_load(sg, a[i], sa.template get_multi_ptr<sycl::access::decorated::no>() +
                                                            (sgm * C::MT * 8 + i * 8) * LDA + kk, LDA);
                    for (int j = 0; j < C::NT; ++j)
                        jm::joint_matrix_load(sg, b[j], sb.template get_multi_ptr<sycl::access::decorated::no>() +
                                                            (sgn * C::NT * TN + j * TN) * LDB + kk, LDB);
                    for (int i = 0; i < C::MT; ++i)
                        for (int j = 0; j < C::NT; ++j) jm::joint_matrix_mad(sg, acc[i][j], a[i], b[j], acc[i][j]);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            for (int i = 0; i < C::MT; ++i) {
                const int r = sgm * C::MT * 8 + i * 8;
                if (r >= rows) continue;
                for (int j = 0; j < C::NT; ++j) {
                    const int c = n0 + sgn * C::NT * TN + j * TN;
                    if (r + 8 <= rows) {
                        jm::joint_matrix_store(sg, acc[i][j],
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                Ye + (size_t) r * N + c), N, jm::layout::row_major);
                    } else {   // partial rows: through this sub-group's own SLM slice
                        float* st = reinterpret_cast<float*>(pa) + sgid * 8 * TN;
                        jm::joint_matrix_store(sg, acc[i][j],
                            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(st),
                            TN, jm::layout::row_major);
                        sycl::group_barrier(sg);
                        const int lim = rows - r, ln = (int) sg.get_local_linear_id();
                        for (int rr = 0; rr < lim; ++rr) Ye[(size_t) (r + rr) * N + c + ln] = st[rr * TN + ln];
                        sycl::group_barrier(sg);
                    }
                }
            }
        });
    });
#else
    (void) q; (void) X; (void) Y; (void) gp; (void) ntiles; (void) N; (void) K;
    std::fprintf(stderr, "grouped XMX: the 8x8x16 / sub-group 8 kernel is not built for this device (STRATA_NO_DG2_XMX)\n");
    std::exit(1);
#endif
}

class k_small; class k_big;
#ifdef STRATA_XMX_SG16
using Small = Cfg<2, 4, 4, 2, 32>;   // 64 x 128 tile: few rows per expert (8 sub-groups of 16)
using Big = Cfg<4, 2, 4, 4, 32>;     // 128 x 128 tile
#else
using Small = Cfg<2, 8, 4, 2, 32>;   // 64 x 128 tile: few rows per expert
using Big = Cfg<4, 4, 4, 4, 32>;     // 128 x 128 tile
#endif

template <class C>
int fill(Grp& gp, const int32_t* cnt, int n) {
    int t = 0, o = 0;
    for (int i = 0; i < n; ++i) {
        gp.off[i] = o; gp.tstart[i] = t;
        o += cnt[i]; t += (cnt[i] + C::BM - 1) / C::BM;
    }
    gp.off[n] = o; gp.tstart[n] = t; gp.n = n;
    return t;
}
}  // namespace

int mode() {
    // The kernels are joint_matrix 8x8x16 on sub-group 8 (DG2, Arc A-series) or, in a STRATA_XMX_SG16 build, 8x16x16 on
    // sub-group 16 (Xe2, Battlemage).
    static const int v = [] {
        const char* e = std::getenv("STRATA_PF_XMX");
        // 0.1.48 (owner approved 2026-10-18): mode 2 (the fused grouped XMX launch) is the default on Arc Alchemist (A-series): A750 prompts +30..39% (4K), +32..35% (1-2K tokens),
        // first-token KL 0.002 against a 0.0005 floor, HumanEval / reasoning / agent / CJK / needles inside the run-to-run spread; Battlemage is
        // 7-11% slower with it, so it stays off there.  STRATA_PF_XMX=0 turns it off, 1 is the dequantize + grouped XMX variant.
        const bool alchemist = strata::intel_gpu_gen(strata::q_of(nullptr)->get_device()) == strata::IntelGpuGen::Alchemist;
        const int want = e ? std::atoi(e) : alchemist ? 2 : 0;
        if (want == 0) return 0;
#ifdef STRATA_NO_DG2_XMX
        std::fprintf(stderr, "STRATA_PF_XMX ignored: this build is AOT for a device without the sub-group 8 XMX kernels\n");
        return 0;
#else
#ifndef STRATA_XMX_SG16
        if (strata::intel_gpu_gen(strata::q_of(nullptr)->get_device()) == strata::IntelGpuGen::Battlemage) {
            std::fprintf(stderr, "STRATA_PF_XMX ignored: this build has the sub-group 8 / 8x8x16 kernels (Arc A-series); Battlemage needs the AOT bmg build\n");
            return 0;
        }
#endif
        return want;
#endif
    }();
    return v;
}
bool enabled() { return mode() != 0; }

int gu_max_mean() {
    static const int v = [] { const char* e = std::getenv("STRATA_PF_XMX_GU_MAX"); return e ? std::atoi(e) : 250; }();
    return v;
}

void grouped_f16(const uint16_t* X, const uint16_t* const* W, const int32_t* cnt, int n, float* Y, int N, int K,
                 void* stream) {
    if (n <= 0) return;
    sycl::queue& q = *strata::q_of(stream);
    Grp gp{};
    int64_t tot = 0;
    for (int i = 0; i < n; ++i) { gp.w[i] = reinterpret_cast<const half*>(W[i]); tot += cnt[i]; }
    if (tot <= 0) return;
    const half* Xh = reinterpret_cast<const half*>(X);
    if (tot / n < 96) {
        const int nt = fill<Small>(gp, cnt, n);
        launch<Small, k_small>(q, Xh, Y, gp, nt, N, K);
    } else {
        const int nt = fill<Big>(gp, cnt, n);
        launch<Big, k_big>(q, Xh, Y, gp, nt, N, K);
    }
}
}  // namespace strata::prefill::xmx
