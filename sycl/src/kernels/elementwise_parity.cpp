// src/kernels/elementwise_parity.cpp - P2.S5's test for the layer glue.
//
// The ops are small; the CONVENTIONS in them are not, and each one is a place where a plausible reading
// gives a plausible number:
//
//   1. `softplus` HAS A LARGE-x BRANCH.  `log1p(exp(x))` overflows f32 at x > 88 and loses relative precision
//      well before that; `ggml_compute_softplus_f32` returns `x` above 20.  A rival without the branch is
//      asserted observable, because a gate that saturates to +inf makes `exp(gate)` inf and the whole
//      recurrence NaN - which reads as a state bug and not as a missing branch.
//   2. `ssm_a` IS NEGATIVE.  `gate = softplus(...) * ssm_a`, so `exp(gate) < 1` and the state DECAYS.  A
//      fixture with a positive `ssm_a` would still produce finite output and the recurrence would blow up
//      instead of decaying, so the sign is checked as a property and not assumed.
//   3. `silu` IS COMPUTED IN DOUBLE then cast, because `ref/gdn.py`'s numpy does.  f32 `expf` differs in the
//      last bits, and the test measures that rather than asserting it away.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(dpct::err0 e, const char *what) {
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: elementwise_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    const int64_t H_V = 48;

    // ---- 1. gdn_gate, with the reference's own softplus
    {
        const int n = (int) H_V;
        std::mt19937 rng(11);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> alpha(n), dt(n), a(n), want(n);
        for (int i = 0; i < n; ++i) {
            // A MIXTURE THAT ACTUALLY CROSSES THE BRANCH.  The first version drew alpha from a normal with
            // sigma 3, so the largest value was about 9 and `softplus` never took its `x > 20` path - the
            // "branch is observable" check then reported **0.00% apart, 0 non-finite**, which is the fixture
            // saying it cannot see the thing it was written to see.  Every third head is now large.
            const bool large = (i % 3) == 0;
            // Up to ~120, so the fixture spans WELL PAST f32's `exp` overflow at 88.  A fixture that stopped
            // at 88 would show the branch as unobservable and would be right: `log1pf(expf(x))` equals `x` to
            // f32 precision for the whole range 20..88, so ggml's threshold of 20 is CONSERVATIVE and the
            // behaviour only actually changes where `expf` overflows.
            alpha[i] = large ? (22.0f + 6.0f * (float) (i % 17)) : g(rng) * 3.0f;
            dt[i] = g(rng) * 0.5f;
            // NEGATIVE, as the artifact's `ssm_a = -exp(A_log)` is.  Checked below as a property.
            a[i] = -(std::fabs(g(rng)) + 0.1f);
            const double x = (double) alpha[i] + (double) dt[i];
            const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
            want[(size_t) i] = (float) (sp * (double) a[i]);
        }
        // the fixture must EXERCISE the branch, or check 1 below measures nothing
        {
            int over = 0, past_overflow = 0;
            for (int i = 0; i < n; ++i) {
                const float x = alpha[(size_t) i] + dt[(size_t) i];
                if (x > 20.0f) ++over;
                if (x > 88.0f) ++past_overflow;
            }
            std::printf("  %-44s %s (%d above 20, %d above 88)\n", "the fixture crosses the branch",
                        over ? "yes" : "*** NO ***", over, past_overflow);
            if (!over) ++bad;
        }
        // the sign property, asserted rather than assumed
        int positive = 0;
        for (int i = 0; i < n; ++i) if (a[(size_t) i] >= 0.0f) ++positive;
        std::printf("  %-44s %s (%d of %d non-negative)\n", "the fixture's ssm_a is negative",
                    positive ? "*** NO ***" : "yes", positive, n);
        if (positive) ++bad;

        float *d_a = nullptr, *d_dt = nullptr, *d_sa = nullptr, *d_g = nullptr;
        check(DPCT_CHECK_ERROR(d_a = (float *)sycl::malloc_device(
                                   n * 4, dpct::get_in_order_queue())),
              "a");
        check(DPCT_CHECK_ERROR(d_dt = (float *)sycl::malloc_device(
                                   n * 4, dpct::get_in_order_queue())),
              "dt");
        check(DPCT_CHECK_ERROR(d_sa = (float *)sycl::malloc_device(
                                   n * 4, dpct::get_in_order_queue())),
              "sa");
        check(DPCT_CHECK_ERROR(d_g = (float *)sycl::malloc_device(
                                   n * 4, dpct::get_in_order_queue())),
              "g");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_a, alpha.data(), n * 4).wait()),
              "ca");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_dt, dt.data(), n * 4).wait()),
              "cd");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_sa, a.data(), n * 4).wait()),
              "cs");
        strata::kernels::gdn_gate(d_a, d_dt, d_sa, d_g, 1, H_V, nullptr);
        std::vector<float> got((size_t) n);
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got.data(), d_g, n * 4)
                                   .wait()),
              "cg");

        const double rel = rel_l1(want, got);
        std::printf("  %-44s rel %.3e\n", "gdn_gate vs the reference", rel);
        // double softplus and double multiply on both sides, then one cast
        if (!(rel <= 1e-6)) { std::printf("    *** over 1e-6 ***\n"); ++bad; }

        // TRAP: no large-x branch.  `log1p(exp(x))` in f32, with no `x > 20` path.
        //
        // AND THE MEASUREMENT SAYS WHERE THE BRANCH ACTUALLY MATTERS.  The first version of this fixture
        // spanned 20..88 and reported **0 heads differ**: `log1pf(expf(x))` agrees with `x` to f32 precision
        // over that whole range, so ggml's threshold of 20 is CONSERVATIVE - the branch changes the answer only
        // where `expf` OVERFLOWS, at about 88.  The check reports the smallest x at which the two readings
        // part company, so the number is in the output rather than in this comment.
        {
            int wrong = 0, differing = 0;
            float first_differ = -1.0f;
            std::vector<float> rival((size_t) n);
            for (int i = 0; i < n; ++i) {
                const float x = alpha[(size_t) i] + dt[(size_t) i];
                const float sp = std::log1p(std::exp(x));       // no branch, f32
                rival[(size_t) i] = sp * a[(size_t) i];
                if (!std::isfinite(sp)) ++wrong;
                const float branched = x > 20.0f ? x : std::log1pf(std::exp(x));
                if (sp != branched) {
                    ++differing;
                    if (first_differ < 0.0f || x < first_differ) first_differ = x;
                }
            }
            const double r = rel_l1(want, rival);
            const bool visible = r > 0.05 || wrong > 0;
            std::printf("  %-44s %s (%.2f%% apart, %d non-finite, %d heads differ, first at x = %.1f)\n",
                        "the softplus large-x branch is observable", visible ? "yes" : "*** NO ***",
                        r * 100, wrong, differing, (double) first_differ);
            if (!visible) ++bad;
        }
        // PROPERTY: exp(gate) < 1 for every element, which is what makes the state decay
        {
            int over = 0;
            for (float v : got) if (!(std::exp(v) < 1.0f)) ++over;
            std::printf("  %-44s %s (%d of %d not < 1)\n", "exp(gate) < 1 for every head",
                        over ? "*** NO ***" : "yes", over, n);
            if (over) ++bad;
        }
        sycl::free(d_a, dpct::get_in_order_queue());
            sycl::free(d_dt, dpct::get_in_order_queue());
            sycl::free(d_sa, dpct::get_in_order_queue());
            sycl::free(d_g, dpct::get_in_order_queue());
    }

    // ---- 2. silu, in double then cast
    {
        const int n = 4096;
        std::mt19937 rng(22);
        std::normal_distribution<float> g(0.0f, 3.0f);
        std::vector<float> x((size_t) n), want((size_t) n);
        for (int i = 0; i < n; ++i) {
            x[(size_t) i] = g(rng);
            const double v = (double) x[(size_t) i];
            want[(size_t) i] = (float) (v / (1.0 + std::exp(-v)));
        }
        float* d_x = nullptr;
        check(DPCT_CHECK_ERROR(d_x = (float *)sycl::malloc_device(
                                   n * 4, dpct::get_in_order_queue())),
              "sx");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_x, x.data(), n * 4).wait()),
              "csx");
        strata::kernels::silu_inplace(d_x, n, nullptr);
        std::vector<float> got((size_t) n);
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got.data(), d_x, n * 4)
                                   .wait()),
              "csg");
        const double rel = rel_l1(want, got);
        std::printf("\n  %-44s rel %.3e\n", "silu (double) vs the reference", rel);
        if (!(rel <= 1e-7)) { std::printf("    *** over 1e-7 ***\n"); ++bad; }
        sycl::free(d_x, dpct::get_in_order_queue());
    }

    // ---- 3. scale and the f32->f16 bridge
    {
        const int n = 1024;
        std::vector<float> x((size_t) n), want((size_t) n);
        for (int i = 0; i < n; ++i) x[(size_t) i] = (float) (i - n / 2) * 0.013f;
        const float s = 1.0f / std::sqrt(128.0f);
        for (int i = 0; i < n; ++i) want[(size_t) i] = x[(size_t) i] * s;

        float* d_x = nullptr;
        uint16_t* d_h = nullptr;
        check(DPCT_CHECK_ERROR(d_x = (float *)sycl::malloc_device(
                                   n * 4, dpct::get_in_order_queue())),
              "ex");
        check(DPCT_CHECK_ERROR(d_h = (uint16_t *)sycl::malloc_device(
                                   n * 2, dpct::get_in_order_queue())),
              "eh");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_x, x.data(), n * 4).wait()),
              "cex");
        strata::kernels::scale_inplace(d_x, n, s, nullptr);
        std::vector<float> got((size_t) n);
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got.data(), d_x, n * 4)
                                   .wait()),
              "ceg");
        int diff = 0;
        for (int i = 0; i < n; ++i) if (got[(size_t) i] != want[(size_t) i]) ++diff;
        std::printf("  %-44s %d of %d differ\n", "scale_inplace is exact", diff, n);
        if (diff) ++bad;

        strata::kernels::f32_to_f16_bulk(d_x, d_h, n, nullptr);
        std::vector<uint16_t> h((size_t) n);
        check(
            DPCT_CHECK_ERROR(
                (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(h.data(), d_h, n * 2).wait()),
            "ceh");
        int hbad = 0;
        for (int i = 0; i < n; ++i)
            if (h[(size_t) i] != strata::kernels::f16_from_f32(got[(size_t) i])) ++hbad;
        std::printf("  %-44s %d of %d differ\n", "f32_to_f16_bulk uses the shared conversion", hbad, n);
        if (hbad) ++bad;
        sycl::free(d_x, dpct::get_in_order_queue());
            sycl::free(d_h, dpct::get_in_order_queue());
    }

    // ---- 4. rms_norm_weighted, and the TWO RIVAL READINGS it exists to be told apart from.
    //
    // This kernel's whole reason for being a separate entry point from `gdn_l2_norm` is that the two differ by
    // one `/ cols`, and that a swap produces a well-scaled, plausible tensor.  So the test computes BOTH wrong
    // readings and requires each to differ from the right one by more than the tolerance before it judges the
    // kernel - the same discipline the shared-expert test uses for silu-on-gate.
    {
        const int64_t rows = 24, cols = 256;          // the real `attn_q`-norm shape
        const float eps = 1e-6f;
        std::vector<float> x((size_t) (rows * cols));
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float) std::sin((double) i * 0.017) * 1.7f + 0.4f;
        std::vector<float> w((size_t) cols);
        for (int64_t c = 0; c < cols; ++c) w[(size_t) c] = 0.5f + 0.002f * (float) (c % 97);

        // the oracle: f64, mean, times w
        auto ref = [&](bool use_sum, bool use_w) {
            std::vector<double> y((size_t) (rows * cols));
            for (int64_t r = 0; r < rows; ++r) {
                double ss = 0;
                for (int64_t c = 0; c < cols; ++c) {
                    const double v = x[(size_t) (r * cols + c)];
                    ss += v * v;
                }
                const double den = std::sqrt((use_sum ? ss : ss / (double) cols) + (double) eps);
                for (int64_t c = 0; c < cols; ++c) {
                    const double v = x[(size_t) (r * cols + c)];
                    y[(size_t) (r * cols + c)] = (v / den) * (use_w ? (double) w[(size_t) c] : 1.0);
                }
            }
            std::vector<float> out((size_t) (rows * cols));
            for (size_t i = 0; i < out.size(); ++i) out[i] = (float) y[i];
            return out;
        };
        const std::vector<float> want = ref(false, true);
        const std::vector<float> rival_sum = ref(true, false);    // what gdn_l2_norm computes
        const std::vector<float> rival_now = ref(false, false);   // the (1+w)-vs-w / no-weight confusion

        auto rel = [&](const std::vector<float>& a, const std::vector<float>& b) {
            double d = 0, m = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                d += std::fabs((double) a[i] - (double) b[i]);
                m += std::fabs((double) a[i]);
            }
            return d / (m > 1e-30 ? m : 1e-30);
        };
        const double r_sum = rel(want, rival_sum), r_now = rel(want, rival_now);
        std::printf("  %-44s %.2f%% apart\n", "sum-vs-mean is observable", r_sum * 100);
        std::printf("  %-44s %.2f%% apart\n", "with-w-vs-without-w is observable", r_now * 100);
        // `sqrt(cols)` = 16 for the first and the weight's own spread for the second; a fixture that cannot
        // separate them is not testing the kernel that was written.
        if (!(r_sum > 0.5)) { std::printf("  *** the sum reading is NOT observable ***\n"); ++bad; }
        if (!(r_now > 0.5)) { std::printf("  *** the no-weight reading is NOT observable ***\n"); ++bad; }

        float* d_x = nullptr;
        float* d_w = nullptr;
        check(DPCT_CHECK_ERROR(d_x = (float *)sycl::malloc_device(
                                   x.size() * 4, dpct::get_in_order_queue())),
              "m rmsx");
        check(DPCT_CHECK_ERROR(d_w = (float *)sycl::malloc_device(
                                   w.size() * 4, dpct::get_in_order_queue())),
              "m rmsw");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_x, x.data(),
                                                                 x.size() * 4).wait()),
              "c rmsx");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_w, w.data(),
                                                                 w.size() * 4).wait()),
              "c rmsw");
        strata::kernels::rms_norm_weighted(d_x, d_w, rows, cols, eps, nullptr);
        std::vector<float> got((size_t) (rows * cols));
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got.data(), d_x, got.size() * 4)
                                   .wait()),
              "c rmsg");

        const double r_got = rel(want, got);
        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double m = std::fabs((double) want[i]);
            const double e = std::fabs((double) got[i] - (double) want[i]) / (m > 1e-30 ? m : 1e-30);
            if (e > worst) worst = e;
        }
        int nonfinite = 0;
        for (float v : got) if (!std::isfinite(v)) ++nonfinite;
        std::printf("  %-44s rel %.3e  worst %.3e  nonfinite %d\n", "rms_norm_weighted vs f64 oracle", r_got,
                    worst, nonfinite);
        if (nonfinite) ++bad;
        // The kernel accumulates in f32 over 256 terms; 1e-6 is loose enough for that and far tighter than the
        // 16x the rival readings are off by.
        if (!(r_got < 1e-6)) { std::printf("  *** rms_norm_weighted is WRONG ***\n"); ++bad; }
        // and a null `w` must be legal, because `ref/qsa.py` allows it.
        //
        // THE INPUT HAS TO BE RE-UPLOADED FIRST.  The kernel is IN PLACE, so `d_x` currently holds the
        // NORMALISED tensor from the call above; running it again normalises a second time and every element
        // differs.  That is exactly what the first version of this check reported - "6144 of 6144 differ" -
        // and it is a fixture bug rather than a kernel bug: the oracle reads the ORIGINAL `x`, so the two
        // sides were never looking at the same input.
        const std::vector<float> want_null = ref(false, false);
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_x, x.data(),
                                                                 x.size() * 4).wait()),
              "c rmsx2");
        strata::kernels::rms_norm_weighted(d_x, nullptr, rows, cols, eps, nullptr);
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got.data(), d_x, got.size() * 4)
                                   .wait()),
              "c rmsn");
        int null_bad = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (std::fabs((double) got[i] - (double) want_null[i]) > 1e-6) ++null_bad;
        std::printf("  %-44s %d of %zu differ\n", "a null weight is legal", null_bad, got.size());
        if (null_bad) ++bad;
        sycl::free(d_x, dpct::get_in_order_queue());
        sycl::free(d_w, dpct::get_in_order_queue());
    }

    // Packed embedding rows: every code width, scale-group size, optional offset,
    // row boundary and partial CUDA block. Compare bits with a scalar CPU decoder.
    // Volatile materializes the multiply so this oracle cannot silently use FMA.
    {
        dpct::queue_ptr stream;
        /*
        DPCT1025: The SYCL queue is created ignoring the flag and priority
        options.
        */
        check(DPCT_CHECK_ERROR(
                  stream = dpct::get_current_device().create_queue(true)),
              "embedding stream");
        std::mt19937 rng(77);
        std::uniform_real_distribution<float> values(-2.0f, 2.0f);
        int mismatches = 0, guards = 0, fma_diff = 0, cases = 0;
        for (const int bits : {2, 4, 8}) {
            const int bias = bits == 2 ? -1 : bits == 4 ? -7 : -16;
            const int per_byte = 8 / bits;
            const unsigned mask = (1u << bits) - 1u;
            for (const int group : {16, 32, 64}) {
                for (const int n : {320, 2560}) {
                    const int row_bytes = n / per_byte, row_groups = n / group;
                    std::vector<uint8_t> codes((size_t) 3 * row_bytes, 0);
                    std::vector<float> scales((size_t) 3 * row_groups), offsets(scales.size());
                    std::vector<int> unpacked((size_t) 3 * n);
                    for (size_t i = 0; i < unpacked.size(); ++i) {
                        const unsigned code = (unsigned) (i * 13 + i / n * 7) & mask;
                        unpacked[i] = (int) code;
                        codes[i / per_byte] |= (uint8_t) (code << ((i % per_byte) * bits));
                    }
                    for (size_t i = 0; i < scales.size(); ++i) {
                        scales[i] = values(rng);
                        offsets[i] = values(rng);
                    }
                    scales[0] = -0.0f;
                    offsets[0] = 0.0f;
                    uint8_t* dc = nullptr;
                    float *ds = nullptr, *dof = nullptr, *out = nullptr;
                    check(DPCT_CHECK_ERROR(
                              dc = (uint8_t *)sycl::malloc_device(
                                  codes.size(), dpct::get_in_order_queue())),
                          "embedding codes");
                    check(DPCT_CHECK_ERROR(
                              ds = sycl::malloc_device<float>(
                                  scales.size(), dpct::get_in_order_queue())),
                          "embedding scales");
                    check(DPCT_CHECK_ERROR(
                              dof = sycl::malloc_device<float>(
                                  offsets.size(), dpct::get_in_order_queue())),
                          "embedding offsets");
                    check(DPCT_CHECK_ERROR(
                              out = sycl::malloc_device<float>(
                                  ((size_t)n + 2), dpct::get_in_order_queue())),
                          "embedding output");
                    /*
                    DPCT1124: cudaMemcpyAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    check(DPCT_CHECK_ERROR(
                              stream->memcpy(dc, codes.data(), codes.size())),
                          "embedding codes upload");
                    /*
                    DPCT1124: cudaMemcpyAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    check(
                        DPCT_CHECK_ERROR(stream->memcpy(
                            ds, scales.data(), scales.size() * sizeof(float))),
                        "embedding scales upload");
                    /*
                    DPCT1124: cudaMemcpyAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    check(DPCT_CHECK_ERROR(
                              stream->memcpy(dof, offsets.data(),
                                             offsets.size() * sizeof(float))),
                          "embedding offsets upload");
                    check(DPCT_CHECK_ERROR(stream->wait()),
                          "embedding upload sync");

                    for (const bool with_offset : {false, true}) {
                        for (int row = 0; row < 3; ++row) {
                            std::vector<float> want((size_t) n), got((size_t) n + 2, -12345.0f);
                            for (int i = 0; i < n; ++i) {
                                const size_t gi = (size_t) row * row_groups + i / group;
                                const float code = (float) (unpacked[(size_t) row * n + i] + bias);
                                volatile float product = code * scales[gi];
                                const float offset = with_offset ? offsets[gi] : 0.0f;
                                want[(size_t) i] = product + offset;
                                const float fused = std::fma(code, scales[gi], offset);
                                if (std::memcmp(&fused, &want[(size_t) i], sizeof(float)) != 0) ++fma_diff;
                            }
                            /*
                            DPCT1124: cudaMemcpyAsync is migrated to
                            asynchronous memcpy API. While the origin API might
                            be synchronous, it depends on the type of operand
                            memory, so you may need to call wait() on event
                            return by memcpy API to ensure synchronization
                            behavior.
                            */
                            check(DPCT_CHECK_ERROR(stream->memcpy(
                                      out, got.data(),
                                      got.size() * sizeof(float))),
                                  "embedding output guards");
                            strata::kernels::embedding_gather(dc + (size_t) row * row_bytes,
                                ds + (size_t) row * row_groups,
                                with_offset ? dof + (size_t) row * row_groups : nullptr,
                                n, bits, bias, group, out + 1, stream);
                            /*
                            DPCT1124: cudaMemcpyAsync is migrated to
                            asynchronous memcpy API. While the origin API might
                            be synchronous, it depends on the type of operand
                            memory, so you may need to call wait() on event
                            return by memcpy API to ensure synchronization
                            behavior.
                            */
                            check(DPCT_CHECK_ERROR(stream->memcpy(
                                      got.data(), out,
                                      got.size() * sizeof(float))),
                                  "embedding result");
                            check(DPCT_CHECK_ERROR(stream->wait()),
                                  "embedding sync");
                            for (int i = 0; i < n; ++i) {
                                if (std::memcmp(&want[(size_t) i], &got[(size_t) i + 1], sizeof(float)) != 0) ++mismatches;
                            }
                            if (got.front() != -12345.0f || got.back() != -12345.0f) ++guards;
                            ++cases;

                            // Capturing the gather proves that it has no hidden synchronization.
                            // A following scale also checks that it uses the caller's stream.
                            if (row == 1) {
                                dpct::experimental::command_graph_ptr graph;
                                dpct::experimental::command_graph_exec_ptr exec;
                                check(DPCT_CHECK_ERROR(
                                          dpct::experimental::begin_recording(
                                              stream)),
                                      "embedding capture");
                                strata::kernels::embedding_gather(dc + (size_t) row * row_bytes,
                                    ds + (size_t) row * row_groups,
                                    with_offset ? dof + (size_t) row * row_groups : nullptr,
                                    n, bits, bias, group, out + 1, stream);
                                strata::kernels::scale_inplace(out + 1, n, 2.0f, stream);
                                check(DPCT_CHECK_ERROR(
                                          dpct::experimental::end_recording(
                                              stream, &graph)),
                                      "embedding capture end");
                                check(
                                    DPCT_CHECK_ERROR(
                                        exec = new sycl::ext::oneapi::
                                            experimental::command_graph<
                                                sycl::ext::oneapi::
                                                    experimental::graph_state::
                                                        executable>(
                                                graph->finalize())),
                                    "embedding instantiate");
                                check(DPCT_CHECK_ERROR(
                                          stream->ext_oneapi_graph(*exec)),
                                      "embedding replay");
                                /*
                                DPCT1124: cudaMemcpyAsync is migrated to
                                asynchronous memcpy API. While the origin API
                                might be synchronous, it depends on the type of
                                operand memory, so you may need to call wait()
                                on event return by memcpy API to ensure
                                synchronization behavior.
                                */
                                check(DPCT_CHECK_ERROR(stream->memcpy(
                                          got.data(), out,
                                          got.size() * sizeof(float))),
                                      "embedding graph result");
                                check(DPCT_CHECK_ERROR(stream->wait()),
                                      "embedding graph sync");
                                for (int i = 0; i < n; ++i) {
                                    const float expected = want[(size_t) i] * 2.0f;
                                    if (std::memcmp(&expected, &got[(size_t) i + 1], sizeof(float)) != 0) ++mismatches;
                                }
                                if (got.front() != -12345.0f || got.back() != -12345.0f) ++guards;
                                check(DPCT_CHECK_ERROR(delete (exec)),
                                      "embedding exec destroy");
                                check(DPCT_CHECK_ERROR(delete (graph)),
                                      "embedding graph destroy");
                            }
                        }
                    }
                    sycl::free(dc, dpct::get_in_order_queue());
                        sycl::free(ds, dpct::get_in_order_queue());
                        sycl::free(dof, dpct::get_in_order_queue());
                        sycl::free(out, dpct::get_in_order_queue());
                }
            }
        }
        check(
            DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream)),
            "embedding stream destroy");
        std::printf("  embedding gather: %d row cases, %d bit mismatches, %d guard failures, %d FMA differences\n",
                    cases, mismatches, guards, fma_diff);
        if (mismatches || guards || fma_diff == 0) ++bad;
    }

    // ---- THE f32 -> bf16 CONVERSIONS KEEP A NaN A NaN.  `f32_to_bf16_bulk` (the header's `bf16_from_f32`
    // on the device) and the prompt path's dequantizer (its own `f2bf`) against ggml_compute_fp32_to_bf16's rule;
    // before the fix 0x7FFFFFFF came back as -0 and 0x7F800001 as +inf.  `bf16_bits_test` covers every f32 on
    // the host; this is the same header compiled by nvcc, plus the second copy in dequant_bf16.cu.
    {
        auto ggml_bf16 = [](uint32_t u) -> uint16_t {
            if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64);
            return (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
        };
        std::vector<uint32_t> bits = {0x7FFFFFFFu, 0xFFFFFFFFu, 0x7F800001u, 0xFF800001u, 0x7FC00000u, 0x7FBFFFFFu,
                                      0x7F800000u, 0xFF800000u, 0x7F7FFFFFu, 0x00000000u, 0x80000000u, 0x3F808000u,
                                      0x3F818000u, 0x00008000u, 0x7F7F8000u};
        std::mt19937 rng(13);
        while (bits.size() < 4096) bits.push_back((uint32_t) rng());
        const size_t n = bits.size();
        float* dx = nullptr;
        uint16_t* dy = nullptr;
        check(DPCT_CHECK_ERROR(dx = (float *)sycl::malloc_device(
                                   n * 4, dpct::get_in_order_queue())),
              "bf16 x");
        check(DPCT_CHECK_ERROR(dy = (uint16_t *)sycl::malloc_device(
                                   n * 2, dpct::get_in_order_queue())),
              "bf16 y");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(dx, bits.data(), n * 4).wait()),
              "bf16 cx");
        strata::kernels::f32_to_bf16_bulk(dx, dy, (int64_t) n, nullptr);
        std::vector<uint16_t> got(n);
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(got.data(), dy, n * 2)
                                   .wait()),
              "bf16 cy");
        int wrong = 0;
        for (size_t i = 0; i < n; ++i) wrong += got[i] != ggml_bf16(bits[i]);

        // A Q8_0 block (type 8) whose fp16 scale is a NaN: every dequantized value is NaN * q, a NaN, and it must
        // still be one in BF16.  A second block with an ordinary scale checks the finite path did not move.
        std::vector<uint8_t> blk(2 * 34, 0);
        blk[0] = 0x00; blk[1] = 0x7E;                        // fp16 quiet NaN
        blk[34] = 0x00; blk[35] = 0x3C;                      // fp16 1.0
        for (int j = 0; j < 32; ++j) {
            blk[(size_t) (2 + j)] = (uint8_t) (int8_t) (j - 16);
            blk[(size_t) (36 + j)] = (uint8_t) (int8_t) (3 * j - 50);
        }
        uint8_t* db = nullptr;
        uint16_t* dq = nullptr;
        check(DPCT_CHECK_ERROR(db = (uint8_t *)sycl::malloc_device(
                                   blk.size(), dpct::get_in_order_queue())),
              "q8 blk");
        check(DPCT_CHECK_ERROR(dq = (uint16_t *)sycl::malloc_device(
                                   64 * 2, dpct::get_in_order_queue())),
              "q8 out");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(db, blk.data(),
                                                                 blk.size()).wait()),
              "q8 cblk");
        strata::kernels::dequant_bf16(8, db, 0, 2, 32, dq, nullptr);
        check(DPCT_CHECK_ERROR(
                  dpct::get_current_device().queues_wait_and_throw()),
              "dequant_bf16");
        std::vector<uint16_t> q(64);
        check(
            DPCT_CHECK_ERROR(
                (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(q.data(), dq, 64 * 2).wait()),
            "q8 cout");
        int dq_wrong = 0;
        for (int j = 0; j < 32; ++j) {
            if (!((q[(size_t) j] & 0x7FFFu) > 0x7F80u)) ++dq_wrong;           // row 0: NaN scale -> NaN
            const float v = (float) (3 * j - 50);                                // row 1: d = 1.0 -> the integer
            uint32_t vb;
            std::memcpy(&vb, &v, 4);
            if (q[(size_t) (32 + j)] != ggml_bf16(vb)) ++dq_wrong;
        }
        std::printf("  f32 -> bf16 (NaN kept): bulk %d of %zu differ from ggml, dequant_bf16 %d of 64 wrong\n",
                    wrong, n, dq_wrong);
        if (wrong || dq_wrong) ++bad;
        sycl::free(dx, dpct::get_in_order_queue());
            sycl::free(dy, dpct::get_in_order_queue());
            sycl::free(db, dpct::get_in_order_queue());
            sycl::free(dq, dpct::get_in_order_queue());
    }

    std::printf("\nelementwise: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("elementwise_parity OK\n");
    return 0;
}
