// gemm_launch: what does one expert-sized FP16 GEMM (oneMKL, half in, float out) cost back to back, by row count?
// The prompt path issues one gate/up and one down GEMM per routed expert (~25k experts per 4K prompt on the 48 layers); if the
// per-call time barely moves with the rows, the path is bound by launches, not by the XMX units.
//   gemm_launch [calls=2000]
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const int calls = argc > 1 ? std::atoi(argv[1]) : 2000;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const int64_t N = 2560;
    struct Shape { const char* name; int64_t n, k; };
    const Shape shapes[] = {{"gate/up 1280x2560", 1280, N}, {"down 2560x640", N, 640}};
    const int64_t Ts[] = {8, 32, 80, 160, 320, 640, 1280};
    const int64_t maxT = 1280;
    using half = sycl::half;
    half* X = sycl::malloc_device<half>(maxT * N, q);
    half* W = sycl::malloc_device<half>(N * 1280 * 4, q);
    float* Y = sycl::malloc_device<float>(maxT * N, q);
    q.memset(X, 0, maxT * N * sizeof(half));
    q.memset(W, 0, N * 1280 * 4 * sizeof(half));
    q.wait();
    for (const Shape& s : shapes) {
        for (int64_t T : Ts) {
            auto run = [&](int n) {
                for (int i = 0; i < n; ++i)
                    oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, s.n, T, s.k, 1.0f,
                                                          W, s.k, X, s.k, 0.0f, Y, s.n);
                q.wait();
            };
            run(50);
            const auto t0 = std::chrono::steady_clock::now();
            run(calls);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / calls;
            auto* h = new dpct::blas::descriptor();
            h->set_queue(&q);
            const float alpha = 1.0f, beta = 0.0f;
            auto run2 = [&](int cnt) {
                for (int i = 0; i < cnt; ++i)
                    dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) s.n, (int) T, (int) s.k, &alpha, W,
                                     dpct::library_data_t::real_half, (int) s.k, X, dpct::library_data_t::real_half, (int) s.k, &beta, Y,
                                     dpct::library_data_t::real_float, (int) s.n, dpct::compute_type::f32);
                q.wait();
            };
            run2(50);
            const auto t1 = std::chrono::steady_clock::now();
            run2(calls);
            const double us2 = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / calls;
            delete h;
            std::printf("%-20s T=%5lld: mkl %7.1f us/call   dpct::blas::gemm %7.1f us/call\n", s.name, (long long) T, us, us2);
        }
    }
    return 0;
}
