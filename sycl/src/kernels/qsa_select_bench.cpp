// src/kernels/qsa_select_bench.cpp - the prompt path's QSA selection (qsa_select.hpp) timed per stage, block scores
// (the warp kernel and the tensor-core one) and top-k, for a batch of consecutive queries at a given context, and
// the two scorers compared: score difference and how many selections differ (GPU, synthetic, no model).
// Usage: qsa_select_bench [context=131072] [queries=256] [reps=10] [capacity_cells]
// capacity_cells (the engine's --max-context): the score buffers and the top-k dispatch follow the CAPACITY
// (max_blocks = capacity / 4 + 2), the work follows the context. Default: capacity = context (max_blocks = ctx / 4 + 1).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(DPCT_CHECK_ERROR(
           d = (T *)sycl::malloc_device(h.size() * sizeof(T) + 64,
                                        dpct::get_in_order_queue())),
       "malloc");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
           d, h.data(), h.size() * sizeof(T)).wait()),
       "upload");
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 131072;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : 0;
    const int64_t max_blocks = capacity > 0 ? capacity / 4 + 2 : ctx / 4 + 1, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // keys with a shared direction plus noise, so the scores have a spread like a real indexer's
    std::vector<float> dir(128), pooled((size_t) (max_blocks * 128)), dead(128), q((size_t) (nq * 512));
    for (auto& x : dir) x = nd(rng);
    for (int64_t b = 0; b < max_blocks; ++b) {
        const float a = nd(rng);
        for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * a * dir[d] + nd(rng);
    }
    for (auto& x : dead) x = nd(rng);
    for (int64_t i = 0; i < nq; ++i)
        for (int d = 0; d < 512; ++d) q[(size_t) (i * 512 + d)] = 0.2f * dir[d % 128] + 0.1f * nd(rng);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {   // qsa_step_fill's arithmetic (kept here so the bench links only qsa_select)
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_pooled = up(pooled);
    const float* d_dead = up(dead);
    const float* d_q = up(q);
    const int32_t* d_steps = up(steps);
    float *sc_old = nullptr, *sc_new = nullptr;
    int32_t *ids_old = nullptr, *ids_new = nullptr;
    ck(DPCT_CHECK_ERROR(
           sc_old = (float *)sycl::malloc_device((size_t)(nq * max_blocks) * 4,
                                                 dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           sc_new = (float *)sycl::malloc_device((size_t)(nq * max_blocks) * 4,
                                                 dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           ids_old = (int32_t *)sycl::malloc_device(
               (size_t)(nq * cap) * 4, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           ids_new = (int32_t *)sycl::malloc_device(
               (size_t)(nq * cap) * 4, dpct::get_in_order_queue())),
       "malloc");
    const int64_t active = steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1;
    auto run_old = [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_old, nullptr, active); };
    // a device without the tensor-core scorer (HIP other than gfx12): time the warp scorer and the top-k only
    const bool have_tc = k::qsa_block_scores_tc(d_pooled, d_dead, d_q, d_steps, 1, max_blocks, s, sc_new, nullptr, active);
    auto run_new = [&] {
        if (!k::qsa_block_scores_tc(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_new, nullptr, active)) {
            std::fprintf(stderr, "tc scorer refused\n");
            std::exit(2);
        }
    };
    run_old();
    if (have_tc) run_new();
    else ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                sc_new, sc_old, (size_t)(nq * max_blocks) * 4).wait()),
            "copy");
    k::qsa_block_topk_ref(sc_old, d_steps, nq, max_blocks, cap, s, ids_old, nullptr);
    k::qsa_block_topk(sc_new, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active);
    int32_t* ids_reg = nullptr;   // the register top-k on the OLD scores: must equal the reference exactly
    ck(DPCT_CHECK_ERROR(
           ids_reg = (int32_t *)sycl::malloc_device(
               (size_t)(nq * cap) * 4, dpct::get_in_order_queue())),
       "malloc");
    k::qsa_block_topk(sc_old, d_steps, nq, max_blocks, cap, s, ids_reg, nullptr, active);
    ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
       "warm");
    // compare
    std::vector<float> a((size_t) (nq * max_blocks)), b(a.size());
    std::vector<int32_t> ia((size_t) (nq * cap)), ib(ia.size());
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(a.data(), sc_old, a.size() * 4)
                            .wait()),
       "down");
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(b.data(), sc_new, b.size() * 4)
                            .wait()),
       "down");
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(ia.data(), ids_old, ia.size() * 4)
                            .wait()),
       "down");
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(ib.data(), ids_new, ib.size() * 4)
                            .wait()),
       "down");
    std::vector<int32_t> ir(ia.size());
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(ir.data(), ids_reg, ir.size() * 4)
                            .wait()),
       "down");
    int64_t reg_same = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
        reg_same += std::equal(ia.begin() + i * cap, ia.begin() + i * cap + w, ir.begin() + i * cap);
    }
    double max_rel = 0, sum_rel = 0, n_rel = 0;
    int64_t same_sel = 0, cells_diff = 0, cells_all = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* st = steps.data() + i * k::kStepCount;
        for (int64_t j = 0; j <= st[k::kStepNBid]; ++j) {
            const double x = a[(size_t) (i * max_blocks + j)], y = b[(size_t) (i * max_blocks + j)];
            const double r = std::fabs(x - y) / std::max(1e-6, std::fabs(x));
            max_rel = std::max(max_rel, r);
            sum_rel += r;
            n_rel += 1;
        }
        const int64_t w = st[k::kStepWidth];
        std::vector<int32_t> x(ia.begin() + i * cap, ia.begin() + i * cap + w), y(ib.begin() + i * cap, ib.begin() + i * cap + w);
        same_sel += x == y;
        std::vector<int32_t> d;
        std::set_symmetric_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(d));
        cells_diff += (int64_t) d.size() / 2;
        cells_all += w;
    }
    // accuracy against an FP64 host reference on a sample (blocks below n_bid; the tail block is the warp kernel's own
    // arithmetic in both scorers). Gate, as the prompt-attention harness's: the scorer under test is no worse than 4x
    // the warp kernel's error, floored at 1e-5 of the score scale.
    double err_old = 0, err_new = 0, scale = 0;
    {
        std::mt19937 srng(11);
        const int64_t nqs = std::min<int64_t>(nq, 32);
        for (int64_t qs = 0; qs < nqs; ++qs) {
            const int64_t i = qs * nq / nqs;
            const int64_t nbid = steps[(size_t) (i * k::kStepCount + k::kStepNBid)];
            if (nbid <= 0) continue;
            for (int sidx = 0; sidx < 1024; ++sidx) {
                const int64_t j = sidx < 64 ? std::min<int64_t>(nbid - 1, sidx) : (int64_t) (srng() % (uint64_t) nbid);
                double ref = 0;
                for (int h = 0; h < 4; ++h) {
                    double d = 0;
                    for (int c = 0; c < 128; ++c) d += (double) q[(size_t) (i * 512 + h * 128 + c)] * (double) pooled[(size_t) (j * 128 + c)];
                    ref += d > 0 ? d : 0;
                }
                scale = std::max(scale, std::fabs(ref));
                err_old = std::max(err_old, std::fabs(ref - (double) a[(size_t) (i * max_blocks + j)]));
                err_new = std::max(err_new, std::fabs(ref - (double) b[(size_t) (i * max_blocks + j)]));
            }
        }
    }
    const bool acc_ok = !have_tc || err_new <= std::max(4.0 * err_old, 1e-5 * scale);
    // time
    dpct::event_ptr e0, e1;
    e0 = new sycl::event(); e1 = new sycl::event();
    auto timed = [&](auto f) {
        try {
    dpct::sync_barrier(e0);
        for (int r = 0; r < reps; ++r) f();
        dpct::sync_barrier(e1);
        ck(DPCT_CHECK_ERROR(e1->wait_and_throw()), "time");
        float ms = 0;
        ms = (e1->get_profiling_info<
                  sycl::info::event_profiling::command_end>() -
              e0->get_profiling_info<
                  sycl::info::event_profiling::command_start>()) /
             1000000.0f;
        return ms / reps;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    const float t_old = timed(run_old), t_new = have_tc ? timed(run_new) : t_old;
    const float t_tk = timed([&] { k::qsa_block_topk_ref(sc_old, d_steps, nq, max_blocks, cap, s, ids_old, nullptr); });
    const float t_tk2 = timed([&] { k::qsa_block_topk(sc_old, d_steps, nq, max_blocks, cap, s, ids_reg, nullptr, active); });
    std::printf("top-k %.3f -> %.3f ms (%.1fx), register top-k identical to the reference %lld/%lld\n", t_tk, t_tk2,
                t_tk / t_tk2, (long long) reg_same, (long long) nq);
    std::printf("%s accuracy vs FP64 (score scale %.3g): warp kernel max err %.3g, tensor-core max err %.3g (%.2g of scale)\n",
                !have_tc ? "SKIP" : acc_ok ? "PASS" : "FAIL", scale, err_old, err_new, scale > 0 ? err_new / scale : 0.0);
    if (!have_tc) std::printf("tensor-core scorer not available on this device: warp scorer %.3f ms, top-k %.3f ms (%.0f%% of the two)\n", t_old, t_tk2, 100.0 * t_tk2 / (t_old + t_tk2));
    std::printf("ctx %lld, %lld queries x %lld blocks: scores %.3f -> %.3f ms (%.1fx), top-k %.3f ms; score rel diff "
                "mean %.2g max %.2g; selections identical %lld/%lld, cells differing %.4f%%\n", (long long) ctx,
                (long long) nq, (long long) active, t_old, t_new, t_old / t_new, t_tk, sum_rel / std::max(1.0, n_rel),
                max_rel, (long long) same_sel, (long long) nq, cells_all ? 100.0 * (double) cells_diff / (double) cells_all : 0.0);
    return acc_ok && reg_same == nq ? 0 : 1;
}
