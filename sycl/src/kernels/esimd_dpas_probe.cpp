// esimd_dpas_probe (151-b70): the DPAS peak an ESIMD kernel reaches on one device, registers only.
//   esimd_dpas_probe [reps=4096] [threads=8192]
// Modes: fp16 (8x16x16, fp32 acc), s8 (8x16x32, int32 acc).  RC=8 rows per DPAS, MT row tiles x NT column tiles of
// accumulators per hardware thread; the A/B operands are loaded once (random bits), so the loop is pure DPAS.
// Build: icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 esimd_dpas_probe.cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

template <bool S8, int MT, int NT, int GRF>
double run(sycl::queue& q, int reps, int threads, uint32_t* src, float* out) {
    constexpr int KW = S8 ? 8 : 8;               // dwords per row of A (K=32 s8 / K=16 fp16: 8 dwords)
    constexpr auto PA = S8 ? xmx::dpas_argument_type::s8 : xmx::dpas_argument_type::fp16;
    using CT = std::conditional_t<S8, int, float>;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<GRF>};
    auto t0 = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(threads, 8), props, [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int tid = it.get_global_id(0);
            simd<int, 8 * KW> a[MT];
            simd<int, 16 * KW> b[NT];
            simd<CT, 128> acc[MT][NT];
            for (int i = 0; i < MT; ++i) a[i] = simd<int, 8 * KW>(0x3c003c00u + i * 0x01010101 + tid, 1);
            for (int j = 0; j < NT; ++j) b[j] = simd<int, 16 * KW>(0x3c003c00u + j * 0x00010001, 1);
            for (int i = 0; i < MT; ++i) for (int j = 0; j < NT; ++j) acc[i][j] = 0;
            for (int r = 0; r < reps; ++r) {
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int j = 0; j < NT; ++j)
                        acc[i][j] = xmx::dpas<8, 8, CT, CT, int, int, PA, PA>(acc[i][j], b[j], a[i]);
            }
            float s = 0;
            for (int i = 0; i < MT; ++i) for (int j = 0; j < NT; ++j) { simd<float, 128> f = acc[i][j]; s += reduce<float>(f, std::plus<>()); }
            out[tid] = s;
        });
    }).wait();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 4096, threads = argc > 2 ? atoi(argv[2]) : 8192;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    float* out = sycl::malloc_device<float>(threads, q);
    uint32_t* src = nullptr;
    auto rep = [&](const char* nm, double (*fn)(sycl::queue&, int, int, uint32_t*, float*), int mt, int nt, bool s8) {
        fn(q, 64, threads, src, out);
        double best = 1e30;
        for (int k = 0; k < 3; ++k) best = std::min(best, fn(q, reps, threads, src, out));
        const double ops = 2.0 * 8 * 16 * (s8 ? 32 : 16) * (double) mt * nt * reps * threads;
        std::printf("%-22s %.1f T%s/s  (%.3f s)\n", nm, ops / best / 1e12, s8 ? "OPS" : "FLOPS", best);
    };
    rep("fp16 4x2 grf256", run<false, 4, 2, 256>, 4, 2, false);
    rep("fp16 4x2 grf128", run<false, 4, 2, 128>, 4, 2, false);
    rep("fp16 2x2 grf256", run<false, 2, 2, 256>, 2, 2, false);
    rep("s8   4x2 grf256", run<true, 4, 2, 256>, 4, 2, true);
    rep("s8   4x2 grf128", run<true, 4, 2, 128>, 4, 2, true);
    rep("s8   2x2 grf256", run<true, 2, 2, 256>, 2, 2, true);
    return 0;
}
