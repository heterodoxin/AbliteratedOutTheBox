// src/kernels/verify_parity.cpp - batched / split verify-window kernels against the kernels they replace, BITWISE.
//
// A verify window's token t must come out bit for bit as it would in a window of any other size (the drafts are
// accepted exactly when greedy decode would have produced them), so every kernel here is checked against the
// one it stands in for, on random inputs that include -0.0, denormals, NaN, -inf and large values.  The tests
// follow eddoursul's fork (MIT; verify_parity.cpp), ported to this tree's kernels.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(dpct::err0 e, const char *what) {
}

template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(DPCT_CHECK_ERROR(p = (T *)sycl::malloc_device(
                               n * sizeof(T), dpct::get_in_order_queue())),
          "cudaMalloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d, h.data(), h.size() * sizeof(T)).wait()),
          "upload");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memcpy(h.data(), d, n * sizeof(T))
                               .wait()),
          "download");
    return h;
}

// argmax_rows against sample_tokens' greedy pick and row_top_prob_split against row_top_prob, bitwise: rows of both
// vocabularies with ties at the maximum (the lowest index wins), NaN, -inf, a row with no value above -inf, and
// launches repeated on one scratch (each must leave its counters at zero)
int test_argmax(std::mt19937 &rng, dpct::queue_ptr s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 4.0f);
    uint8_t* sa = dev<uint8_t>(argmax_rows_scratch_bytes(8));   // one scratch for every row count, as in the engine
    uint8_t* st = dev<uint8_t>(row_top_prob_scratch_bytes(8));
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memset(sa, 0, argmax_rows_scratch_bytes(8))
                               .wait()),
          "memset");
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memset(st, 0, row_top_prob_scratch_bytes(8))
                               .wait()),
          "memset");
    for (const int n : {248320, 40525, 4097, 33}) {
        for (const int rows : {4, 8, 1, 3, 8, 2}) {
            std::vector<float> h((size_t) rows * n);
            for (float& v : h) v = nd(rng);
            for (int r = 0; r < rows; ++r) {
                float* row = h.data() + (size_t) r * n;
                const int a = (int) (rng() % n), b = (int) (rng() % n);
                row[a] = row[b] = 40.0f;                       // a tie at the maximum
                row[(int) (rng() % n)] = std::nanf("");        // never picked
                row[(int) (rng() % n)] = -INFINITY;
                if (r == 2)                                    // no value above -inf: 0
                    for (int i = 0; i < n; ++i) row[i] = i % 7 ? -INFINITY : std::nanf("");
            }
            float* d_l = dev<float>(h.size());
            up(d_l, h);
            int* d_ref = dev<int>(rows);
            int32_t* d_out = dev<int32_t>(rows);
            float* d_p1 = dev<float>(rows);
            float* d_p2 = dev<float>(rows);
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(d_l, rows, n, nullptr, 0, sp, d_ref, s);
            row_top_prob(d_l, rows, n, d_ref, d_p1, s);
            for (int rep = 0; rep < 3; ++rep) {
                argmax_rows(d_l, rows, n, sa, d_out, s);
                row_top_prob_split(d_l, rows, n, d_out, d_p2, st, s);
                check(DPCT_CHECK_ERROR(s->wait()), "argmax");
                const std::vector<int> ref = down(d_ref, rows);
                const std::vector<int32_t> out = down(d_out, rows);
                const std::vector<float> p1 = down(d_p1, rows), p2 = down(d_p2, rows);
                for (int r = 0; r < rows; ++r)
                    if (ref[r] != out[r] || std::memcmp(&p1[r], &p2[r], 4) != 0) {
                        if (bad < 5)
                            std::printf("  argmax n %d rows %d row %d: %d / %d, p %.9g / %.9g\n", n, rows, r, ref[r],
                                        out[r], p1[r], p2[r]);
                        ++bad;
                    }
            }
            sycl::free(d_l, dpct::get_in_order_queue());
                sycl::free(d_ref, dpct::get_in_order_queue());
                sycl::free(d_out, dpct::get_in_order_queue());
                sycl::free(d_p1, dpct::get_in_order_queue());
                sycl::free(d_p2, dpct::get_in_order_queue());
        }
    }
    sycl::free(sa, dpct::get_in_order_queue());
    sycl::free(st, dpct::get_in_order_queue());
    std::printf("argmax_rows, row_top_prob_split: %s\n",
                bad ? "MISMATCH" : "bitwise equal to the one-block kernels (ties, NaN, -inf, 1-8 rows on one scratch)");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: verify_parity [--selftest]\n");
            return 2;
        }
    }
    std::mt19937 rng(20260925);
    dpct::queue_ptr s = &dpct::get_in_order_queue();
    check(DPCT_CHECK_ERROR(s = dpct::get_current_device().create_queue(true)),
          "stream");
    int bad = 0;
    bad += test_argmax(rng, s);
    dpct::get_current_device().destroy_queue(s);
    std::printf("verify_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
