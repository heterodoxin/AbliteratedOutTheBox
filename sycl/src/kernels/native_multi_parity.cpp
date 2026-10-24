// src/kernels/native_multi_parity.cpp - the multi-token router and the k = 10 combine, bitwise against their
// single-token contracts (#783 PR-c: route_multi, combine_k10_vec4).
//
//   * native_router_top10_multi (one warp per token, 8 tokens per CTA) against native_router_top10 token by token,
//     for n = 1..19 (a partial last CTA, and more than one CTA), on plain logits, on logits with exact ties and on
//     logits with NaN (the router maps NaN to -FLT_MAX): ids and weights are compared with memcmp;
//   * native_moe_combine / native_moe_combine_multi (k = 10 with a shared row takes the float4 kernel; a misaligned
//     output, k = 9 and a null shared row take the scalar one) against a host replay of the documented contract:
//     the first product rounds to F32, the next ones accumulate with FMA in expert order, the shared row is added
//     once: memcmp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(dpct::err0 e, const char *what) {
}

template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    check(DPCT_CHECK_ERROR(
              p = (void *)sycl::malloc_device(h.size() * sizeof(T) + 64,
                                              dpct::get_in_order_queue())),
          "malloc");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              p, h.data(), h.size() * sizeof(T)).wait()),
          "h2d");
    return (T*) p;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memcpy(h.data(), d, n * sizeof(T))
                               .wait()),
          "d2h");
    return h;
}

}  // namespace

int main(int argc, char**) {
    (void) argc;
    using namespace strata::kernels;
    std::mt19937 rng(783);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;
    dpct::queue_ptr st = &dpct::get_in_order_queue();
    check(DPCT_CHECK_ERROR(st = dpct::get_current_device().create_queue(true)),
          "stream");

    // ---- the multi-token router
    for (int mode = 0; mode < 3; ++mode) {
        const char* names[] = {"random logits", "exact ties", "NaN logits"};
        int mode_bad = 0, cases = 0;
        for (int n = 1; n <= 19; ++n) {
            std::vector<float> lg((size_t) n * 512);
            for (auto& v : lg) v = gauss(rng) * 2.0f;
            if (mode == 1)
                for (size_t i = 0; i < lg.size(); ++i) lg[i] = (float) (int) (lg[i] * 2.0f) * 0.5f;   // coarse: many exact ties
            if (mode == 2)
                for (int t = 0; t < n; ++t) lg[(size_t) t * 512 + (size_t) (rng() % 512)] = std::nanf("");
            float* d_l = up(lg);
            int32_t *d_i, *d_ir;
            float *d_w, *d_wr;
            check(DPCT_CHECK_ERROR(
                      d_i = (int32_t *)sycl::malloc_device(
                          (size_t)n * 10 * 4, dpct::get_in_order_queue())),
                  "i");
            check(DPCT_CHECK_ERROR(
                      d_ir = (int32_t *)sycl::malloc_device(
                          (size_t)n * 10 * 4, dpct::get_in_order_queue())),
                  "ir");
            check(DPCT_CHECK_ERROR(
                      d_w = (float *)sycl::malloc_device(
                          (size_t)n * 10 * 4, dpct::get_in_order_queue())),
                  "w");
            check(DPCT_CHECK_ERROR(
                      d_wr = (float *)sycl::malloc_device(
                          (size_t)n * 10 * 4, dpct::get_in_order_queue())),
                  "wr");
            native_router_top10_multi(d_l, d_i, d_w, n, st);
            for (int t = 0; t < n; ++t) native_router_top10(d_l + (size_t) t * 512, d_ir + t * 10, d_wr + t * 10, st);
            check(DPCT_CHECK_ERROR(
                      dpct::get_current_device().queues_wait_and_throw()),
                  "sync");
            const auto i1 = down(d_i, (size_t) n * 10), i2 = down(d_ir, (size_t) n * 10);
            const auto w1 = down(d_w, (size_t) n * 10), w2 = down(d_wr, (size_t) n * 10);
            ++cases;
            if (std::memcmp(i1.data(), i2.data(), i1.size() * 4) != 0 || std::memcmp(w1.data(), w2.data(), w1.size() * 4) != 0) {
                std::printf("    *** native_router_top10_multi n=%d (%s) differs from native_router_top10 ***\n", n, names[mode]);
                ++mode_bad;
            }
            sycl::free(d_l, dpct::get_in_order_queue());
                sycl::free(d_i, dpct::get_in_order_queue());
                sycl::free(d_ir, dpct::get_in_order_queue());
                sycl::free(d_w, dpct::get_in_order_queue());
                sycl::free(d_wr, dpct::get_in_order_queue());
        }
        std::printf("  %-52s %s (%d cases)\n", (std::string("router multi == single, ") + names[mode]).c_str(),
                    mode_bad ? "*** NO ***" : "bitwise", cases);
        bad += mode_bad;
    }

    // ---- the combine
    struct Cfg { const char* name; int k; bool shared; int out_off; };
    const Cfg cfgs[] = {{"k=10 + shared (float4 kernel)", 10, true, 0}, {"k=10 + shared, output misaligned", 10, true, 1},
                        {"k=10 without shared", 10, false, 0}, {"k=9 + shared", 9, true, 0}};
    const int N = 2048;
    for (const Cfg& c : cfgs) {
        int cfg_bad = 0, cases = 0;
        for (int n = 1; n <= 8; ++n) {
            std::vector<float> parts((size_t) n * c.k * N), w((size_t) n * c.k), sh((size_t) n * N);
            for (auto& v : parts) v = gauss(rng);
            for (auto& v : w) v = (float) (rng() % 1000) / 1000.0f;
            for (auto& v : sh) v = gauss(rng);
            float* d_p = up(parts);
            float* d_w = up(w);
            float* d_s = up(sh);
            float* d_o = nullptr;
            check(DPCT_CHECK_ERROR(
                      d_o = (float *)sycl::malloc_device(
                          (size_t)n * N * 4 + 64, dpct::get_in_order_queue())),
                  "o");
            float* out = d_o + c.out_off;
            if (c.out_off) {
                // a misaligned output cannot take a multi launch of rows with the 4-float stride contract either way:
                // the rows are still n*N floats from `out`
            }
            native_moe_combine_multi(d_p, d_w, c.shared ? d_s : nullptr, out, N, c.k, n, st);
            check(DPCT_CHECK_ERROR(
                      dpct::get_current_device().queues_wait_and_throw()),
                  "sync");
            const auto got = down(out, (size_t) n * N);
            std::vector<float> want((size_t) n * N);
            for (int t = 0; t < n; ++t)
                for (int col = 0; col < N; ++col) {
                    float s = parts[((size_t) t * c.k) * N + col] * w[(size_t) t * c.k];
                    for (int e = 1; e < c.k; ++e)
                        s = std::fmaf(parts[((size_t) t * c.k + e) * N + col], w[(size_t) t * c.k + e], s);
                    if (c.shared) s += sh[(size_t) t * N + col];
                    want[(size_t) t * N + col] = s;
                }
            ++cases;
            if (std::memcmp(got.data(), want.data(), got.size() * 4) != 0) {
                std::printf("    *** native_moe_combine_multi n=%d (%s) differs from the contract's host replay ***\n", n, c.name);
                ++cfg_bad;
            }
            // the single-token call is the same arithmetic
            if (n == 1) {
                check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                           .memset(d_o, 0, (size_t)N * 4 + 64)
                                           .wait()),
                      "z");
                native_moe_combine(d_p, d_w, c.shared ? d_s : nullptr, out, N, c.k, st);
                check(DPCT_CHECK_ERROR(
                          dpct::get_current_device().queues_wait_and_throw()),
                      "sync");
                const auto g1 = down(out, (size_t) N);
                ++cases;
                if (std::memcmp(g1.data(), want.data(), g1.size() * 4) != 0) {
                    std::printf("    *** native_moe_combine (%s) differs from the contract's host replay ***\n", c.name);
                    ++cfg_bad;
                }
            }
            sycl::free(d_p, dpct::get_in_order_queue());
                sycl::free(d_w, dpct::get_in_order_queue());
                sycl::free(d_s, dpct::get_in_order_queue());
                sycl::free(d_o, dpct::get_in_order_queue());
        }
        std::printf("  %-52s %s (%d cases)\n", (std::string("combine ") + c.name).c_str(), cfg_bad ? "*** NO ***" : "bitwise", cases);
        bad += cfg_bad;
    }

    std::printf("\nnative_multi: %d failures\n", bad);
    if (bad) return 1;
    std::printf("native_multi_parity OK\n");
    return 0;
}
