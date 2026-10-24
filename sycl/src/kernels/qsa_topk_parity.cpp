// src/kernels/qsa_topk_parity.cpp - the QSA top-k the engine dispatches (qsa_block_topk) against the original kernel
// (qsa_block_topk_ref): the selected cells must be identical (GPU, synthetic, no model).
//
// A case is a context under a capacity (the engine's --max-context; max_blocks = capacity / 4 + 2, and without an
// active-block count the dispatch follows it), with consecutive queries at the context's end so that the tail block
// holds 0 to 3 cells.  Three score sets per case: continuous, 64 levels (many ties at the threshold) and all equal
// (the rule's ties go to the lowest index).  The dispatch is called as a decode window calls it (no block count) and as
// the prompt path does (with the call's largest block count).  The reference and the dispatched kernel are also timed.
//
// Usage: qsa_topk_parity --selftest
//        qsa_topk_parity CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] [COUNT=1]   (COUNT=0: no block count)
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(dpct::err0 e, const char *w) {
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

bool run_case(int64_t ctx, int64_t nq, int reps, int64_t capacity, bool counted) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    std::mt19937 rng((uint32_t) (ctx * 31 + capacity));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> scores((size_t) (nq * max_blocks)), tied(scores.size()), equal(scores.size(), 0.25f);
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* st = steps.data() + i * k::kStepCount;
        float* row = scores.data() + i * max_blocks;
        for (int64_t b = 0; b < max_blocks; ++b) row[b] = nd(rng);
        // the tail block as the scorers leave it: far above the others when it holds cells
        if (st[k::kStepNKv] % s.idx_block != 0) row[st[k::kStepNBid]] += 1e9f;
        for (int64_t b = 0; b < max_blocks; ++b)
            tied[(size_t) (i * max_blocks + b)] = row[b] > 1e8f ? row[b] : std::floor(row[b] * 8.0f) / 8.0f;   // ~64 levels
    }
    int32_t *d_steps = nullptr, *ids_ref = nullptr, *ids_new = nullptr;
    float* d_scores = nullptr;
    ck(DPCT_CHECK_ERROR(d_steps = (int32_t *)sycl::malloc_device(
                            steps.size() * 4, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(d_scores = (float *)sycl::malloc_device(
                            scores.size() * 4, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           ids_ref = (int32_t *)sycl::malloc_device(
               (size_t)(nq * cap) * 4, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           ids_new = (int32_t *)sycl::malloc_device(
               (size_t)(nq * cap) * 4, dpct::get_in_order_queue())),
       "malloc");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_steps, steps.data(),
                                                          steps.size() * 4).wait()),
       "upload");
    const int64_t active = counted ? steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1 : -1;
    std::vector<int32_t> a((size_t) (nq * cap)), b(a.size());
    auto same = [&](const std::vector<float> &sc) -> int64_t {
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        try {
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_scores, sc.data(),
                                                          sc.size() * 4).wait()),
       "upload");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memset(ids_ref, 0xff, a.size() * 4)
                                .wait()),
           "memset");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memset(ids_new, 0xff, a.size() * 4)
                                .wait()),
           "memset");
        k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr);
        k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active);
        ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
           "sync");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(a.data(), ids_ref, a.size() * 4)
                                .wait()),
           "down");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(b.data(), ids_new, b.size() * 4)
                                .wait()),
           "down");
        int64_t n = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
            n += std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap);
        }
        return n;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    const int64_t same_tied = same(tied), same_equal = same(equal), same_cont = same(scores);   // d_scores: continuous
    dpct::event_ptr e0, e1;
    ck(DPCT_CHECK_ERROR(e0 = new sycl::event()), "event");
    ck(DPCT_CHECK_ERROR(e1 = new sycl::event()), "event");
    auto timed = [&](auto f) {
        /*
        DPCT1024: The original code returned the error code that was
        further consumed by the program logic. This original code was replaced
        with 0. You may need to rewrite the program logic consuming the error
        code.
        */
        try {
    ck(DPCT_CHECK_ERROR(dpct::sync_barrier(e0)), "record");
        for (int r = 0; r < reps; ++r) f();
        /*
        DPCT1024: The original code returned the error code that was
        further consumed by the program logic. This original code was replaced
        with 0. You may need to rewrite the program logic consuming the error
        code.
        */
        ck(DPCT_CHECK_ERROR(dpct::sync_barrier(e1)), "record");
        ck(DPCT_CHECK_ERROR(e1->wait_and_throw()), "time");
        float ms = 0;
        ck(DPCT_CHECK_ERROR(
               ms = (e1->get_profiling_info<
                         sycl::info::event_profiling::command_end>() -
                     e0->get_profiling_info<
                         sycl::info::event_profiling::command_start>()) /
                    1000000.0f),
           "elapsed");
        return ms / (float) reps;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    const float t_ref = timed([&] { k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr); });
    const float t_new = timed([&] { k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active); });
    dpct::destroy_event(e0); dpct::destroy_event(e1);
    sycl::free(d_steps, dpct::get_in_order_queue());
        sycl::free(d_scores, dpct::get_in_order_queue());
        sycl::free(ids_ref, dpct::get_in_order_queue());
        sycl::free(ids_new, dpct::get_in_order_queue());
    const bool ok = same_cont == nq && same_tied == nq && same_equal == nq;
    std::printf("%s context %lld under a capacity of %lld (%lld blocks), %lld queries %s: reference %.3f ms, "
                "dispatched %.3f ms (%.1fx); identical ids: continuous %lld/%lld, 64 levels %lld/%lld, equal scores "
                "%lld/%lld\n", ok ? "PASS" : "FAIL", (long long) ctx, (long long) capacity, (long long) max_blocks,
                (long long) nq, counted ? "with their block count" : "without a block count", t_ref, t_new, t_ref / t_new,
                (long long) same_cont, (long long) nq, (long long) same_tied,
                (long long) nq, (long long) same_equal, (long long) nq);
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0) {
        // {context, capacity}: both sides of the register kernel's reach (33,792 blocks = 135,168 cells), short
        // contexts under a long capacity (fewer blocks than the kernel's threads; a prompt batch below the dispatch's
        // 4,608 blocks), and the long contexts themselves
        const int64_t cases[][2] = {{9000, 9000},    {135160, 135160}, {135176, 135176}, {2100, 262144},
                                    {9000, 262144},  {32768, 262144},  {262144, 262144}, {300001, 524288},
                                    {524288, 524288}};
        bool ok = true;
        for (const auto& c : cases) {
            ok = run_case(c[0], 8, 3, c[1], false) && ok;     // a decode window's call
            ok = run_case(c[0], 64, 3, c[1], true) && ok;     // the prompt path's
        }
        std::printf("%s\n", ok ? "qsa_topk_parity: PASS" : "qsa_topk_parity: FAIL");
        return ok ? 0 : 1;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: qsa_topk_parity --selftest | CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] "
                             "[COUNT=1]\n");
        return 2;
    }
    const int64_t ctx = std::atoll(argv[1]);
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : ctx;
    if (ctx < nq || nq <= 0 || reps <= 0 || capacity < ctx) {
        std::fprintf(stderr, "qsa_topk_parity: CONTEXT >= QUERIES > 0, REPS > 0, CAPACITY >= CONTEXT\n");
        return 2;
    }
    return run_case(ctx, nq, reps, capacity, argc <= 5 || std::atoi(argv[5]) != 0) ? 0 : 1;
}
