// src/prefill/gemm_bf16_parity.cu - Gemm::bf16 against cuBLAS's own BF16 product (GPU, synthetic, no model).
//
// Below sm_80 (no BF16 tensor cores) Gemm::bf16 converts the weight and the activations to FP16 and runs the FP16
// tensor-core GEMM (Volta by default, Turing with STRATA_BF16_TC=1) or widens them to fp32 (Pascal); everywhere else
// it is the cuBLAS BF16 call itself.  (On Pascal the cuBLAS BF16 reference itself may be refused: the test then fails
// at the reference, which says so.)  This
// checks the result against cublasGemmEx on the same BF16 inputs, within fp32-accumulation rounding, over the prompt
// path's shapes: the hyper-connection down / up projections, the router and indexer rows, the PLE value matrix, a T
// large enough to slice the activations, beta = 1 accumulation (the bf16x2 low parts) and an output row stride wider
// than N.  --bench adds the time of each against the cuBLAS BF16 product.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/prefill/gemm.hpp"
#include <dpct/blas_utils.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <dpct/lib_common_utils.hpp>

namespace {

uint16_t to_bf16(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

bool ok_or(dpct::err0 e, const char *what) {
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */

    return e == 0;
}

struct Shape {
    int64_t T, N, K, ldy;
    const char* what;
};

}  // namespace

int main(int argc, char **argv) try {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int dev = 0, maj = 0, min = 0;
    dev = dpct::get_current_device_id();
    maj = dpct::get_device(dev).get_major_version();
    min = dpct::get_device(dev).get_minor_version();
    std::printf("gemm_bf16_parity: compute capability %d.%d\n", maj, min);

    dpct::queue_ptr st = &dpct::get_in_order_queue();
    if (!ok_or(DPCT_CHECK_ERROR(
                   st = dpct::get_current_device().create_queue(true)),
               "stream")) return 1;
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(st, 0, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    dpct::blas::descriptor_ptr h = nullptr;
    if (DPCT_CHECK_ERROR(h = new dpct::blas::descriptor()) != 0) {
        std::printf("FAIL: cublasCreate\n"); return 1;
    }
    h->set_queue(st);

    const Shape shapes[] = {
        {4096, 320, 10240, 0, "hc down (activations sliced)"},
        {4096, 10240, 320, 0, "hc up"},
        {777, 2560, 2560, 0, "PLE value"},
        {512, 512, 2560, 0, "router / indexer q"},
        {300, 4, 10240, 0, "hc inject"},
        {2048, 1, 2560, 0, "single row"},
        {333, 128, 2560, 136, "row stride wider than N"},
    };
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int failures = 0;
    for (const Shape& s : shapes) {
        const int64_t ldy = s.ldy > 0 ? s.ldy : s.N;
        std::vector<uint16_t> x((size_t) (s.T * s.K)), xlo(x.size()), w((size_t) (s.N * s.K));
        for (size_t i = 0; i < x.size(); ++i) {
            const float v = 3.0f * nd(rng);          // activations: normalized, a few units
            x[i] = to_bf16(v);
            xlo[i] = to_bf16(1e-3f * nd(rng));       // a bf16x2 low part
        }
        for (auto& v : w) v = to_bf16(0.02f * nd(rng));
        uint16_t *dx = nullptr, *dxlo = nullptr, *dw = nullptr;
        float *dy = nullptr, *dr = nullptr;
        const size_t ybytes = (size_t) (s.T * ldy) * 4;
        bool ok =
            ok_or(
                DPCT_CHECK_ERROR(dx = (uint16_t *)sycl::malloc_device(
                                     x.size() * 2, dpct::get_in_order_queue())),
                "x") &&
            ok_or(
                DPCT_CHECK_ERROR(dxlo = (uint16_t *)sycl::malloc_device(
                                     x.size() * 2, dpct::get_in_order_queue())),
                "xlo") &&
            ok_or(
                DPCT_CHECK_ERROR(dw = (uint16_t *)sycl::malloc_device(
                                     w.size() * 2, dpct::get_in_order_queue())),
                "w") &&
            ok_or(DPCT_CHECK_ERROR(dy = (float *)sycl::malloc_device(
                                       ybytes, dpct::get_in_order_queue())),
                  "y") &&
            ok_or(DPCT_CHECK_ERROR(dr = (float *)sycl::malloc_device(
                                       ybytes, dpct::get_in_order_queue())),
                  "ref");
        if (!ok) return 1;
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        dpct::get_in_order_queue().memcpy(dx, x.data(), x.size() * 2).wait();
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        dpct::get_in_order_queue().memcpy(dxlo, xlo.data(), xlo.size() * 2).wait();
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        dpct::get_in_order_queue().memcpy(dw, w.data(), w.size() * 2).wait();
        dpct::get_in_order_queue().memset(dy, 0, ybytes).wait();
        dpct::get_in_order_queue().memset(dr, 0, ybytes).wait();

        // the product under test: X . W^T, then the low part added with beta = 1
        gemm.bf16(dx, dw, dy, s.T, s.N, s.K, ldy);
        gemm.bf16(dxlo, dw, dy, s.T, s.N, s.K, ldy, 1.0f);
        // the reference: cuBLAS on the BF16 inputs, the same two calls
        const float one = 1.0f, zero = 0.0f;
        auto ref = [&](const uint16_t* X, float beta) {
            /*
            DPCT1020: Migration of cublasGemmEx, if it is called from
            __global__ or __device__ function, is not supported. You may need to
            redesign the code to use the host-side dpct::blas::gemm instead,
            which submits this call to the SYCL queue automatically.
            */
            return cublasGemmEx(
                h, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, (int)s.N, (int)s.T, (int)s.K,
                &one, dw, dpct::library_data_t::real_bfloat16, (int)s.K, X,
                dpct::library_data_t::real_bfloat16, (int)s.K, &beta, dr,
                dpct::library_data_t::real_float, (int)ldy,
                dpct::compute_type::f32, -1);
        };
        if (ref(dx, zero) != 0 || ref(dxlo, one) != 0) {
            std::printf("FAIL: reference cublasGemmEx\n");
            return 1;
        }
        if (!ok_or(DPCT_CHECK_ERROR(st->wait()), "sync")) return 1;
        std::vector<float> y((size_t) (s.T * ldy)), r(y.size());
        dpct::get_in_order_queue().memcpy(y.data(), dy, ybytes).wait();
        dpct::get_in_order_queue().memcpy(r.data(), dr, ybytes).wait();
        double worst = 0.0, mag = 1e-30;
        int bad = 0;
        for (int64_t t = 0; t < s.T; ++t)
            for (int64_t c = 0; c < s.N; ++c) {
                const float a = y[(size_t) (t * ldy + c)], b = r[(size_t) (t * ldy + c)];
                if (!std::isfinite(a)) ++bad;
                worst = std::max(worst, (double) std::fabs(a - b));
                mag = std::max(mag, (double) std::fabs(b));
            }
        // the untouched columns of a wider row stride stay as they were
        for (int64_t t = 0; t < s.T && ldy > s.N; ++t)
            for (int64_t c = s.N; c < ldy; ++c)
                if (y[(size_t) (t * ldy + c)] != 0.0f) ++bad;
        const bool pass = bad == 0 && worst <= 1e-4 * mag;
        if (!pass) ++failures;
        std::printf("  %-32s T %5lld N %5lld K %5lld ldy %5lld: worst |diff| %.3e of max |ref| %.3e (rel %.2e) %s\n",
                    s.what, (long long) s.T, (long long) s.N, (long long) s.K, (long long) ldy, worst, mag, worst / mag,
                    pass ? "pass" : "FAIL");

        if (bench) {
            auto time = [&](auto&& f) {
#pragma unroll
                for (int i = 0; i < 3; ++i) f();
                st->wait();
                const auto t0 = std::chrono::steady_clock::now();
#pragma unroll
                for (int i = 0; i < 20; ++i) f();
                st->wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
            };
            const double us_gemm = time([&] { gemm.bf16(dx, dw, dy, s.T, s.N, s.K, ldy); });
            const double us_ref = time([&] { ref(dx, zero); });
            std::printf("      bench: Gemm::bf16 %9.1f us   cuBLAS BF16 %9.1f us   (%.2fx)\n", us_gemm, us_ref,
                        us_ref / us_gemm);
        }
        sycl::free(dx, dpct::get_in_order_queue());
            sycl::free(dxlo, dpct::get_in_order_queue());
            sycl::free(dw, dpct::get_in_order_queue());
            sycl::free(dy, dpct::get_in_order_queue());
            sycl::free(dr, dpct::get_in_order_queue());
    }
    delete (h);
    dpct::get_current_device().destroy_queue(st);
    std::printf("gemm_bf16_parity: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
