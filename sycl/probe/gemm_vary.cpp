// gemm_vary: the expert GEMMs of a prompt see a different row count every call.  Does oneMKL pay per distinct M (kernel
// generation / lookup), and how much do rows that are not a multiple of 8, 16 or 32 cost?
//   gemm_vary [calls=3000]
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int calls = argc > 1 ? std::atoi(argv[1]) : 3000;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    using half = sycl::half;
    const int64_t N = 2560, maxT = 1536;
    half* X = sycl::malloc_device<half>(maxT * N, q);
    half* W = sycl::malloc_device<half>(N * 1280, q);
    float* Y = sycl::malloc_device<float>(maxT * N, q);
    q.memset(X, 0, maxT * N * 2); q.memset(W, 0, N * 1280 * 2); q.wait();
    struct Shape { const char* name; int64_t n, k; };
    const Shape shapes[] = {{"gate/up", 1280, N}, {"down", N, 640}};
    std::mt19937 rng(7);
    std::vector<int> ts(calls);
    // skewed like a prompt's per-expert row counts: many small, a few large
    for (int& t : ts) { const double u = std::generate_canonical<double, 20>(rng); t = 1 + (int) (400.0 * u * u * u * u * 3.8); if (t > 1500) t = 1500; }
    double mean_t = 0; for (int t : ts) mean_t += t; mean_t /= calls;
    std::printf("rows per call: mean %.0f\n", mean_t);
    for (const Shape& s : shapes) {
        auto pass = [&](bool round8) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < calls; ++i) {
                int64_t T = ts[i];
                if (round8) T = (T + 15) / 16 * 16;
                oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, s.n, T, s.k, 1.0f, W, s.k, X, s.k,
                                                      0.0f, Y, s.n);
            }
            q.wait();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / calls;
        };
        const double first = pass(false);
        const double second = pass(false);
        const double rounded = pass(true);
        std::printf("%-8s first pass %7.1f us/call, second %7.1f, rows rounded up to 16: %7.1f\n", s.name, first, second, rounded);
    }
    return 0;
}
