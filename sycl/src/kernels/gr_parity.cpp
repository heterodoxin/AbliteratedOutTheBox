// src/kernels/gr_parity.cpp - P2.S2's test for the gated residual / hyper-connection.
//
// `ref/gr.py` opens by listing the details "a prose reading gets wrong", and every one of them is a reading
// that has the RIGHT SHAPES and produces a PLAUSIBLE number.  So each is computed here the wrong way round and
// required to differ materially BEFORE the kernel is judged against the right one.  A test that only does the
// second half passes against either reading:
//
//   1. PER-STREAM RMSNorm.  The source comment reads ambiguously; the code reduces over ne[0] = n_embd, i.e.
//      one RMS per stream.  The rival reading is one RMS over the whole hc*n_embd stack.
//   2. `/ hc` INSIDE the silu: `lo = silu(proj / hc)`.  Moving it outside is a one-line change that keeps
//      every shape.
//   3. SiLU on `lo` and SIGMOID on the gate, a few lines apart in the same function.  Swapping them is the
//      obvious slip, and both are monotone saturating functions so the magnitudes stay comparable.
//   4. `mean` over the stream axis, not `sum`.  A factor of hc = 4, which is exactly the kind of error that
//      looks like a scale problem rather than a structural one.
//   5. ACTIVATION PRECISION. The historical BF16-rounded CPU contract and the pinned CUDA single-token
//      FP32-activation contract must be distinguishable. Both are tested against explicit references;
//      neither is selected merely because the weight is BF16. The scalar fixture isolates xn and lo
//      rounding from reduction order, and graph replay must preserve the precision chosen at capture.
//   6. `gr_write`'s `2*sigmoid`, which centres the gate on 1 so a ZERO injection is a plain residual add.
//      Asserted as a property, not as a value, because that is what the source comment claims.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/gr.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(dpct::err0 e, const char *what) {
}

/// The same rule the kernel uses, on the host - `ref/quant.py::bf16`.
float to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    float o;
    std::memcpy(&o, &i, 4);
    return o;
}

uint16_t bf16_bits(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    return (uint16_t) (i >> 16);
}

struct Opts {
    bool per_stream_norm = true;   ///< false: one RMS over the whole stack (WRONG)
    bool scale_inside_silu = true; ///< false: silu(proj)/hc instead of silu(proj/hc) (WRONG)
    bool sigmoid_on_gate = true;   ///< false: silu on the gate and sigmoid on lo (WRONG)
    bool mean_over_streams = true; ///< false: sum over streams instead of mean (WRONG)
    bool round_activation = true;  ///< false: pinned CUDA single-token BF16 MMVF contract
};

/// `ref/gr.py::gr_read`, transcribed with switches for the rival readings.
void reference(const std::vector<float>& R, const std::vector<float>& w_norm,
               const std::vector<float>& w_down, const std::vector<float>& w_up,
               const std::vector<float>& w_inject, float eps, long long n_embd, long long hc, long long hc_lr,
               const Opts& o, std::vector<float>& mixed, std::vector<float>& inject) {
    const long long hc_dim = hc * n_embd;
    std::vector<float> xn((size_t) hc_dim);
    if (o.per_stream_norm) {
        for (long long c = 0; c < hc; ++c) {
            double ms = 0;
            for (long long d = 0; d < n_embd; ++d) ms += (double) R[(size_t) (c * n_embd + d)] * R[(size_t) (c * n_embd + d)];
            ms /= (double) n_embd;
            const float rs = (float) (1.0 / std::sqrt(ms + (double) eps));
            for (long long d = 0; d < n_embd; ++d) {
                const size_t i = (size_t) (c * n_embd + d);
                xn[i] = R[i] * rs * w_norm[i];
            }
        }
    } else {
        double ms = 0;
        for (long long i = 0; i < hc_dim; ++i) ms += (double) R[(size_t) i] * R[(size_t) i];
        ms /= (double) hc_dim;
        const float rs = (float) (1.0 / std::sqrt(ms + (double) eps));
        for (long long i = 0; i < hc_dim; ++i) xn[(size_t) i] = R[(size_t) i] * rs * w_norm[(size_t) i];
    }

    std::vector<float> act((size_t) hc_dim);
    for (long long i = 0; i < hc_dim; ++i) act[(size_t) i] = o.round_activation ? to_bf16(xn[(size_t) i]) : xn[(size_t) i];

    // lo
    std::vector<float> lo((size_t) hc_lr);
    for (long long k = 0; k < hc_lr; ++k) {
        double a = 0;
        for (long long i = 0; i < hc_dim; ++i) a += (double) act[(size_t) i] * (double) w_down[(size_t) (k * hc_dim + i)];
        const float p = (float) a;
        if (o.sigmoid_on_gate) lo[(size_t) k] = o.scale_inside_silu ? (p / (float) hc) / (1.0f + std::exp(-(p / (float) hc)))
                                                                  : (p / (1.0f + std::exp(-p))) / (float) hc;
        else                   lo[(size_t) k] = o.scale_inside_silu ? 1.0f / (1.0f + std::exp(-(p / (float) hc)))
                                                                    : (1.0f / (1.0f + std::exp(-p))) / (float) hc;
    }
    std::vector<float> lq((size_t) hc_lr);
    for (long long k = 0; k < hc_lr; ++k)
        lq[(size_t) k] = o.round_activation ? to_bf16(lo[(size_t) k]) : lo[(size_t) k];

    // gate and the gated mean
    mixed.assign((size_t) n_embd, 0.0f);
    for (long long d = 0; d < n_embd; ++d) {
        float m = 0.0f;
        for (long long c = 0; c < hc; ++c) {
            const long long i = c * n_embd + d;
            double a = 0;
            for (long long k = 0; k < hc_lr; ++k) a += (double) lq[(size_t) k] * (double) w_up[(size_t) (i * hc_lr + k)];
            const float g = (float) a;
            const float s = o.sigmoid_on_gate ? 1.0f / (1.0f + std::exp(-g)) : g / (1.0f + std::exp(-g));
            m += xn[(size_t) i] * s;
        }
        mixed[(size_t) d] = o.mean_over_streams ? m / (float) hc : m;
    }

    inject.assign((size_t) hc, 0.0f);
    for (long long c = 0; c < hc; ++c) {
        double a = 0;
        for (long long i = 0; i < hc_dim; ++i) a += (double) act[(size_t) i] * (double) w_inject[(size_t) (c * hc_dim + i)];
        inject[(size_t) c] = (float) a;
    }
}

/// Normalised L1 difference.  `mag_out` returns the mean |a|, for reporting.
double rel_diff(const std::vector<float>& a, const std::vector<float>& b, double* mag_out = nullptr) {
    double d = 0, mag = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        mag += std::fabs((double) a[i]);
    }
    if (mag_out) *mag_out = mag / (double) (a.empty() ? 1 : a.size());
    return d / (mag > 1e-30 ? mag : 1e-30);
}

/// Relative error against the magnitude of the TERMS, not of the result.
///
/// `R + block_out * w` cancels wherever block_out*w is close to -R, and a plain |want-got|/|want| then
/// reports the CONDITION NUMBER instead of the arithmetic.  This is the same metric mistake the project has
/// now made four times: rounds 169 (Q4_K), 189 (RoPE), 194 (shared_expert) and this one.  For a sum, the
/// denominator is the sum of the term magnitudes.
double rel_terms(double want, double got, double term_a, double term_b) {
    const double den = std::fabs(term_a) + std::fabs(term_b);
    return std::fabs(want - got) / (den > 1e-30 ? den : 1e-30);
}

const char* activation_mode_name(int mode) {
    return mode == 2 ? "native pinned GR" : (mode == 1 ? "FP32" : "BF16");
}
void select_activation_mode(int mode) {
    strata::kernels::gr_set_native_mmvf(mode == 2);
    // In mode 2 the precision-only switch is deliberately false: native MMVF must imply FP32 by itself.
    strata::kernels::gr_set_fp32_activations(mode == 1);
}

// A diagonal GR fixture makes activation precision observable without summation-order ambiguity.
// Two-wide matrices satisfy native MMVF's pair ABI, while each nonzero dot has only one nonzero product.
// At R=1 and eps=0, xn=gamma exactly; any rounding in xn or lo comes from the selected activation contract.
int scalar_activation_contract() {
    using namespace strata::kernels;
    const GrShapes sh{2, 1, 2};
    float *d_R = nullptr, *d_norm = nullptr, *d_mixed = nullptr, *d_inject = nullptr;
    uint16_t* d_weights = nullptr;
    void* d_scratch = nullptr;
    check(DPCT_CHECK_ERROR(
              d_R = sycl::malloc_device<float>(2, dpct::get_in_order_queue())),
          "scalar R");
    check(DPCT_CHECK_ERROR(d_norm = sycl::malloc_device<float>(
                               2, dpct::get_in_order_queue())),
          "scalar norm");
    check(DPCT_CHECK_ERROR(d_mixed = sycl::malloc_device<float>(
                               2, dpct::get_in_order_queue())),
          "scalar mixed");
    check(DPCT_CHECK_ERROR(d_inject = sycl::malloc_device<float>(
                               1, dpct::get_in_order_queue())),
          "scalar inject");
    check(DPCT_CHECK_ERROR(d_weights = sycl::malloc_device<uint16_t>(
                               10, dpct::get_in_order_queue())),
          "scalar weights");
    check(DPCT_CHECK_ERROR(
              d_scratch = (void *)sycl::malloc_device(
                  gr_workspace_bytes(sh), dpct::get_in_order_queue())),
          "scalar scratch");
    GrWorkspace ws;
    gr_workspace_init(sh, d_scratch, ws);
    const float ones[] = {1.0f, 1.0f};
    const uint16_t weights[] = {
        bf16_bits(0.5f), 0, 0, 0,             // down: lo[0]=silu(xn[0]/2), lo[1]=0
        bf16_bits(1.5f), 0, bf16_bits(1.5f), 0, // both gate rows consume lo[0]
        bf16_bits(2.0f), 0                     // injection: 2*xn[0]
    };
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR(
              (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_R, ones, sizeof(ones)).wait()),
          "scalar upload R");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_weights, weights,
                                                             sizeof(weights)).wait()),
          "scalar upload weights");
    dpct::queue_ptr stream;
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    check(DPCT_CHECK_ERROR(stream =
                               dpct::get_current_device().create_queue(true)),
          "scalar stream");
    int bad = 0;
    for (float gamma : {1.0f, 1.00390625f}) {
        const float gammas[] = {gamma, gamma};
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_norm, gammas, sizeof(gammas)).wait()),
              "scalar upload norm");
        float mixed_by_mode[3][2] = {}, inject_by_mode[3] = {};
        for (int mode = 0; mode < 3; ++mode) {
            select_activation_mode(mode);
            gr_read(d_R, d_norm, d_weights, d_weights + 4, d_weights + 8, 0.0f,
                    sh, ws, d_mixed, d_inject, stream);
            check(DPCT_CHECK_ERROR(stream->wait()), "scalar warmup");
            dpct::experimental::command_graph_ptr graph;
            dpct::experimental::command_graph_exec_ptr executable;
            check(DPCT_CHECK_ERROR(dpct::experimental::begin_recording(stream)),
                  "scalar capture begin");
            gr_read(d_R, d_norm, d_weights, d_weights + 4, d_weights + 8, 0.0f,
                    sh, ws, d_mixed, d_inject, stream);
            check(DPCT_CHECK_ERROR(
                      dpct::experimental::end_recording(stream, &graph)),
                  "scalar capture end");
            check(DPCT_CHECK_ERROR(
                      executable =
                          new sycl::ext::oneapi::experimental::command_graph<
                              sycl::ext::oneapi::experimental::graph_state::
                                  executable>(graph->finalize())),
                  "scalar instantiate");
            // Changing both host options after capture cannot change the graph's selected kernels.
            select_activation_mode((mode + 1) % 3);
            // Warmup results must not make a missing/no-op captured launch pass.
            // Poison the outputs and intermediate workspace before replay.
            check(DPCT_CHECK_ERROR(
                      stream->memset(d_mixed, 0xa5, 2 * sizeof(float))),
                  "scalar poison mixed");
            check(
                DPCT_CHECK_ERROR(stream->memset(d_inject, 0xa5, sizeof(float))),
                "scalar poison inject");
            check(DPCT_CHECK_ERROR(stream->memset(d_scratch, 0xa5, ws.bytes)),
                  "scalar poison workspace");
            check(DPCT_CHECK_ERROR(stream->ext_oneapi_graph(*executable)),
                  "scalar replay");
            check(DPCT_CHECK_ERROR(stream->wait()), "scalar replay sync");
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                       .memcpy(mixed_by_mode[mode], d_mixed,
                                               2 * sizeof(float))
                                       .wait()),
                  "scalar mixed copy");
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                       .memcpy(&inject_by_mode[mode], d_inject,
                                               sizeof(float))
                                       .wait()),
                  "scalar inject copy");
            const float activation = mode ? gamma : to_bf16(gamma);
            const float projection = activation * 0.5f;
            const float lo = projection / (1.0f + std::exp(-projection));
            const float gate = (mode ? lo : to_bf16(lo)) * 1.5f;
            const float expected_mixed = gamma / (1.0f + std::exp(-gate));
            const float expected_inject = activation * 2.0f;
            // Each mode has its own activation contract and bounded scalar-formula
            // check. Native fast exp/div need not match ordinary FP32 postops bit
            // for bit: native_gr_postops_parity establishes that arithmetic with
            // actual pinned CUDA graphs. The nonzero projection/injection here
            // has only one exactly representable product and remains exact.
            const bool ok = std::fabs(mixed_by_mode[mode][0] - expected_mixed) <= 2e-7f &&
                            mixed_by_mode[mode][0] == mixed_by_mode[mode][1] &&
                            inject_by_mode[mode] == expected_inject;
            std::printf("  scalar gamma %.8f %s activation/capture %s (mixed %.9f, ref %.9f)\n",
                        gamma, activation_mode_name(mode), ok ? "pass" : "FAIL",
                        mixed_by_mode[mode][0], expected_mixed);
            if (!ok) ++bad;
            check(DPCT_CHECK_ERROR(delete (executable)),
                  "scalar graph exec destroy");
            check(DPCT_CHECK_ERROR(delete (graph)), "scalar graph destroy");

            const float sentinel = -73.25f;
            check(
                DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                     .memcpy(d_inject, &sentinel, sizeof(float))
                                     .wait()),
                "scalar sentinel");
            select_activation_mode(mode);
            gr_read(d_R, d_norm, d_weights, d_weights + 4, nullptr, 0.0f,
                    sh, ws, d_mixed, d_inject, stream);
            check(DPCT_CHECK_ERROR(stream->wait()), "scalar final mixer sync");
            float final_mixed[2] = {}, untouched = 0.0f;
            check(DPCT_CHECK_ERROR(
                      (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                          .memcpy(final_mixed, d_mixed, sizeof(final_mixed))
                          .wait()),
                  "scalar final mixed");
            check(DPCT_CHECK_ERROR(
                      (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                          .memcpy(&untouched, d_inject, sizeof(float))
                          .wait()),
                  "scalar sentinel copy");
            const bool final_ok = std::memcmp(final_mixed, mixed_by_mode[mode], sizeof(final_mixed)) == 0 && untouched == sentinel;
            std::printf("  scalar final mixer %s %s\n", activation_mode_name(mode), final_ok ? "pass" : "FAIL");
            if (!final_ok) ++bad;
        }
        if (std::fabs(mixed_by_mode[1][0] - mixed_by_mode[0][0]) <= 1e-5f ||
            (gamma != 1.0f && inject_by_mode[0] == inject_by_mode[1])) {
            std::printf("  scalar precision fixture is not observable\n");
            ++bad;
        }
        if (inject_by_mode[1] != inject_by_mode[2]) {
            std::printf("  native MMVF changed the exact scalar FP32 injection projection\n");
            ++bad;
        }
    }
    select_activation_mode(0);
    check(DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream)),
          "scalar stream destroy");
    sycl::free(d_R, dpct::get_in_order_queue());
        sycl::free(d_norm, dpct::get_in_order_queue());
        sycl::free(d_mixed, dpct::get_in_order_queue());
        sycl::free(d_inject, dpct::get_in_order_queue());
    sycl::free(d_weights, dpct::get_in_order_queue());
        sycl::free(d_scratch, dpct::get_in_order_queue());
    return bad;
}

int fused_multi_lds_parity(const float* d_norm, const uint16_t* d_down, const uint16_t* d_up,
                           const uint16_t* d_inject, float eps) {
    using namespace strata::kernels;
    constexpr int N = 2560, HC = 4, LR = 320, D = N * HC, T = kFusedGrMaxT;
    std::mt19937 rng(0x6f8a);
    std::normal_distribution<float> normal(0.0f, 0.3f);
    std::vector<float> r((size_t) T * D), bo((size_t) T * N), inj((size_t) T * HC);
    for (auto& x : r) x = normal(rng);
    for (auto& x : bo) x = normal(rng);
    for (auto& x : inj) x = normal(rng);

    float *d_r = nullptr, *d_r_out = nullptr, *d_bo = nullptr, *d_inj = nullptr;
    float *d_lo = nullptr, *d_rs = nullptr, *d_inj_out = nullptr, *d_mixed = nullptr, *d_xn = nullptr;
    check(DPCT_CHECK_ERROR(d_r = sycl::malloc_device<float>(
                               r.size(), dpct::get_in_order_queue())),
          "multi R");
    check(DPCT_CHECK_ERROR(d_r_out = sycl::malloc_device<float>(
                               r.size(), dpct::get_in_order_queue())),
          "multi R_out");
    check(DPCT_CHECK_ERROR(d_bo = sycl::malloc_device<float>(
                               bo.size(), dpct::get_in_order_queue())),
          "multi bo");
    check(DPCT_CHECK_ERROR(d_inj = sycl::malloc_device<float>(
                               inj.size(), dpct::get_in_order_queue())),
          "multi inj");
    check(DPCT_CHECK_ERROR(d_lo = sycl::malloc_device<float>(
                               (size_t)T * LR, dpct::get_in_order_queue())),
          "multi lo");
    check(DPCT_CHECK_ERROR(d_rs = sycl::malloc_device<float>(
                               (size_t)T * HC, dpct::get_in_order_queue())),
          "multi rs");
    check(DPCT_CHECK_ERROR(d_inj_out = sycl::malloc_device<float>(
                               (size_t)T * HC, dpct::get_in_order_queue())),
          "multi injection");
    check(DPCT_CHECK_ERROR(d_mixed = sycl::malloc_device<float>(
                               (size_t)T * N, dpct::get_in_order_queue())),
          "multi mixed");
    check(DPCT_CHECK_ERROR(d_xn = sycl::malloc_device<float>(
                               (size_t)T * D, dpct::get_in_order_queue())),
          "multi xn");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_r, r.data(), r.size() * sizeof(float)).wait()),
          "multi copy R");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_bo, bo.data(), bo.size() * sizeof(float)).wait()),
          "multi copy bo");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_inj, inj.data(), inj.size() * sizeof(float)).wait()),
          "multi copy inj");

    std::vector<FusedGrArgs> args(T);
    for (int t = 0; t < T; ++t) {
        auto& a = args[t];
        a.R = d_r + (size_t) t * D;
        a.R_out = d_r_out + (size_t) t * D;
        a.apply = true;
        a.bo_prev = d_bo + (size_t) t * N;
        a.inj_prev = d_inj + (size_t) t * HC;
        a.w_norm = d_norm;
        a.w_down = d_down;
        a.w_up = d_up;
        a.w_inject = d_inject;
        a.eps = eps;
        a.lo = d_lo + (size_t) t * LR;
        a.rs = d_rs + (size_t) t * HC;
        a.inject_out = d_inj_out + (size_t) t * HC;
        a.mixed = d_mixed + (size_t) t * N;
    }

    struct Snapshot {
        std::vector<float> r_out, lo, rs, inject, mixed;
    };
    auto snapshot = [&]() {
        try {
    Snapshot s;
        s.r_out.resize((size_t) T * D);
        s.lo.resize((size_t) T * LR);
        s.rs.resize((size_t) T * HC);
        s.inject.resize((size_t) T * HC);
        s.mixed.resize((size_t) T * N);
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(s.r_out.data(), d_r_out,
                                           s.r_out.size() * sizeof(float))
                                   .wait()),
              "multi read R_out");
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                      .memcpy(s.lo.data(), d_lo, s.lo.size() * sizeof(float))
                      .wait()),
              "multi read lo");
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                      .memcpy(s.rs.data(), d_rs, s.rs.size() * sizeof(float))
                      .wait()),
              "multi read rs");
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(s.inject.data(), d_inj_out,
                                           s.inject.size() * sizeof(float))
                                   .wait()),
              "multi read inject");
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(s.mixed.data(), d_mixed,
                                           s.mixed.size() * sizeof(float))
                                   .wait()),
              "multi read mixed");
        return s;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    // the split read (STRATA_GR_V3=1) sums in another order than the single-token kernel: equal within float
    // rounding, not to the bit, so it is compared with a relative tolerance; the default kernels bit for bit
    static const bool v3 = [] { const char* v = std::getenv("STRATA_GR_V3"); return v != nullptr && std::atoi(v) != 0; }();
    auto close = [](const std::vector<float>& x, const std::vector<float>& y) {
        double worst = 0.0, mag = 1e-30;
        for (size_t i = 0; i < x.size(); ++i) {
            worst = std::max(worst, (double) std::fabs(x[i] - y[i]));
            mag = std::max(mag, (double) std::fabs(y[i]));
        }
        if (worst > 2e-6 * mag)
            std::printf("    tolerance: worst %.3e of max |ref| %.3e (rel %.3e), n %zu\n", worst, mag, worst / mag, x.size());
        return worst <= 2e-6 * mag;
    };
    auto same = [&](const Snapshot& a, const Snapshot& b) {
        if (v3)   // `lo` is the default kernels' workspace between down and up; the split read keeps it in shared memory
            return close(a.r_out, b.r_out) && close(a.rs, b.rs) && close(a.inject, b.inject) && close(a.mixed, b.mixed);
        return std::memcmp(a.r_out.data(), b.r_out.data(), a.r_out.size() * sizeof(float)) == 0 &&
               std::memcmp(a.lo.data(), b.lo.data(), a.lo.size() * sizeof(float)) == 0 &&
               std::memcmp(a.rs.data(), b.rs.data(), a.rs.size() * sizeof(float)) == 0 &&
               std::memcmp(a.inject.data(), b.inject.data(), a.inject.size() * sizeof(float)) == 0 &&
               std::memcmp(a.mixed.data(), b.mixed.data(), a.mixed.size() * sizeof(float)) == 0;
    };

    dpct::queue_ptr stream = &dpct::get_in_order_queue();
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    check(DPCT_CHECK_ERROR(stream =
                               dpct::get_current_device().create_queue(true)),
          "multi stream");
    // Max T forces the HIP kernel's full dynamic-LDS request: 8 * 1280 * sizeof(float) = 40 KiB.
    fused_gr_read_multi(args.data(), T, d_xn, stream);
    check(DPCT_CHECK_ERROR(stream->wait()), "multi max-T sync");
    const Snapshot multi = snapshot();
    for (int t = 0; t < T; ++t) fused_gr_read(args[t], stream);
    check(DPCT_CHECK_ERROR(stream->wait()), "single reference sync");
    const Snapshot single = snapshot();
    int bad = 0;
    if (!same(multi, single)) {
        std::printf("  fused GR multi max-T differs from single-token calls\n");
        ++bad;
    }

    dpct::experimental::command_graph_ptr graph = nullptr;
    dpct::experimental::command_graph_exec_ptr graph_exec = nullptr;
    check(DPCT_CHECK_ERROR(dpct::experimental::begin_recording(stream)),
          "multi graph begin");
    fused_gr_read_multi(args.data(), T, d_xn, stream);
    check(DPCT_CHECK_ERROR(dpct::experimental::end_recording(stream, &graph)),
          "multi graph end");
    check(DPCT_CHECK_ERROR(
              graph_exec = new sycl::ext::oneapi::experimental::command_graph<
                  sycl::ext::oneapi::experimental::graph_state::executable>(
                  graph->finalize())),
          "multi graph instantiate");
    check(DPCT_CHECK_ERROR(stream->ext_oneapi_graph(*graph_exec)),
          "multi graph initial replay");
    check(DPCT_CHECK_ERROR(stream->wait()), "multi graph initial sync");
    const Snapshot captured = snapshot();
    if (!same(multi, captured)) {
        std::printf("  fused GR multi captured graph differs from direct max-T call\n");
        ++bad;
    }

    // Reuse the same captured pointers with new payloads; then independently run the single-token path again.
    for (size_t i = 0; i < r.size(); ++i) r[i] = -0.7f * r[i] + 0.001f * (float) (i % 17);
    for (size_t i = 0; i < bo.size(); ++i) bo[i] = -0.4f * bo[i] + 0.02f;
    for (size_t i = 0; i < inj.size(); ++i) inj[i] += 0.3f;
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    check(DPCT_CHECK_ERROR(
              stream->memcpy(d_r, r.data(), r.size() * sizeof(float))),
          "multi replay R");
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    check(DPCT_CHECK_ERROR(
              stream->memcpy(d_bo, bo.data(), bo.size() * sizeof(float))),
          "multi replay bo");
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    check(DPCT_CHECK_ERROR(
              stream->memcpy(d_inj, inj.data(), inj.size() * sizeof(float))),
          "multi replay inj");
    check(DPCT_CHECK_ERROR(stream->ext_oneapi_graph(*graph_exec)),
          "multi graph changed replay");
    check(DPCT_CHECK_ERROR(stream->wait()), "multi graph changed sync");
    const Snapshot replay = snapshot();
    if (std::memcmp(multi.mixed.data(), replay.mixed.data(), multi.mixed.size() * sizeof(float)) == 0) {
        std::printf("  fused GR graph replay ignored changed inputs\n");
        ++bad;
    }
    for (int t = 0; t < T; ++t) fused_gr_read(args[t], stream);
    check(DPCT_CHECK_ERROR(stream->wait()), "changed single reference sync");
    if (!same(replay, snapshot())) {
        std::printf("  fused GR changed graph replay differs from single-token calls\n");
        ++bad;
    }


    // S26 STRATA_QFUSE: the read's own q8_1 image of `mixed` must be the bytes native_quantize_q8_1 writes from it -
    // for every T (1..8), directly and through a captured graph replayed twice (the group counters must reset)
    {
        uint8_t *d_q = nullptr, *d_ref = nullptr;
        unsigned* d_cnt = nullptr;
        const size_t qbytes = (size_t) T * (N / 32) * 36;
        check(DPCT_CHECK_ERROR(d_q = (uint8_t *)sycl::malloc_device(
                                   qbytes, dpct::get_in_order_queue())),
              "qfuse q8");
        check(DPCT_CHECK_ERROR(d_ref = (uint8_t *)sycl::malloc_device(
                                   qbytes, dpct::get_in_order_queue())),
              "qfuse ref");
        check(DPCT_CHECK_ERROR(d_cnt = sycl::malloc_device<unsigned int>(
                                   (N / 32), dpct::get_in_order_queue())),
              "qfuse counters");
        check(
            DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                 .memset(d_cnt, 0, (N / 32) * sizeof(unsigned))
                                 .wait()),
            "qfuse counters zero");
        int qbad = 0;
        std::vector<uint8_t> hq(qbytes), hr(qbytes);
        for (int tt = 1; tt <= T && !qbad; ++tt) {
            std::vector<FusedGrArgs> qa(args.begin(), args.begin() + tt);
            for (int t = 0; t < tt; ++t) { qa[t].q8_mixed = d_q + (size_t) t * (N / 32) * 36; qa[t].q8_cnt = d_cnt; }
            for (int rep = 0; rep < 3 && !qbad; ++rep) {
                check(DPCT_CHECK_ERROR(stream->memset(d_q, 0x5a, qbytes)),
                      "qfuse poison");
                bool wrote = false;
                dpct::experimental::command_graph_ptr qg = nullptr;
                dpct::experimental::command_graph_exec_ptr qx = nullptr;
                if (rep == 0) {
                    wrote = fused_gr_read_multi(qa.data(), tt, d_xn, stream);
                } else {   // a captured read, replayed (twice: rep 1 and 2 use fresh captures, each replayed twice)
                    check(DPCT_CHECK_ERROR(
                              dpct::experimental::begin_recording(stream)),
                          "qfuse begin");
                    wrote = fused_gr_read_multi(qa.data(), tt, d_xn, stream);
                    check(DPCT_CHECK_ERROR(
                              dpct::experimental::end_recording(stream, &qg)),
                          "qfuse end");
                    check(
                        DPCT_CHECK_ERROR(
                            qx = new sycl::ext::oneapi::experimental::
                                command_graph<sycl::ext::oneapi::experimental::
                                                  graph_state::executable>(
                                    qg->finalize())),
                        "qfuse instantiate");
                    check(DPCT_CHECK_ERROR(stream->ext_oneapi_graph(*qx)),
                          "qfuse replay 1");
                    check(DPCT_CHECK_ERROR(stream->memset(d_q, 0x5a, qbytes)),
                          "qfuse poison 2");
                    check(DPCT_CHECK_ERROR(stream->ext_oneapi_graph(*qx)),
                          "qfuse replay 2");
                }
                strata::kernels::native_quantize_q8_1(d_mixed, d_ref, N, tt, stream);
                check(DPCT_CHECK_ERROR(stream->wait()), "qfuse sync");
                if (qx) { delete (qx); delete (qg); }
                if (v3) { std::printf("  QFUSE: the v3 read writes no q8_1 (%s)\n", wrote ? "WRONG: it says it did" : "ok"); qbad += wrote; break; }
                check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                           .memcpy(hq.data(), d_q,
                                                   (size_t)tt * (N / 32) * 36)
                                           .wait()),
                      "qfuse read");
                check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                           .memcpy(hr.data(), d_ref,
                                                   (size_t)tt * (N / 32) * 36)
                                           .wait()),
                      "qfuse ref read");
                if (!wrote || std::memcmp(hq.data(), hr.data(), (size_t) tt * (N / 32) * 36) != 0) {
                    std::printf("  QFUSE: T=%d rep %d: %s\n", tt, rep, wrote ? "q8_1 bytes differ" : "not written");
                    ++qbad;
                }
            }
        }
        std::printf("  fused GR read + q8_1 (STRATA_QFUSE), T 1..8, direct and graph replays: %s\n", qbad ? "FAIL" : "pass");
        bad += qbad;
        sycl::free(d_q, dpct::get_in_order_queue());
            sycl::free(d_ref, dpct::get_in_order_queue());
            sycl::free(d_cnt, dpct::get_in_order_queue());
    }
    std::printf("  fused GR multi max-T=8 LDS launch and changing graph replay %s\n",
                bad == 0 ? "pass" : "FAIL");
    check(DPCT_CHECK_ERROR(delete (graph_exec)), "multi graph exec destroy");
    check(DPCT_CHECK_ERROR(delete (graph)), "multi graph destroy");
    check(DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream)),
          "multi stream destroy");
    check(DPCT_CHECK_ERROR(sycl::free(d_xn, dpct::get_in_order_queue())),
          "free multi xn");
    check(DPCT_CHECK_ERROR(sycl::free(d_mixed, dpct::get_in_order_queue())),
          "free multi mixed");
    check(DPCT_CHECK_ERROR(sycl::free(d_inj_out, dpct::get_in_order_queue())),
          "free multi injection");
    check(DPCT_CHECK_ERROR(sycl::free(d_rs, dpct::get_in_order_queue())),
          "free multi rs");
    check(DPCT_CHECK_ERROR(sycl::free(d_lo, dpct::get_in_order_queue())),
          "free multi lo");
    check(DPCT_CHECK_ERROR(sycl::free(d_inj, dpct::get_in_order_queue())),
          "free multi inj");
    check(DPCT_CHECK_ERROR(sycl::free(d_bo, dpct::get_in_order_queue())),
          "free multi bo");
    check(DPCT_CHECK_ERROR(sycl::free(d_r_out, dpct::get_in_order_queue())),
          "free multi R_out");
    check(DPCT_CHECK_ERROR(sycl::free(d_r, dpct::get_in_order_queue())),
          "free multi R");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: gr_parity [--selftest]\n"); return 2; }
    }

    const long long n_embd = 256, hc = 4, hc_lr = 32;
    const long long hc_dim = hc * n_embd;
    const float eps = 1e-6f;
    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // Weights are BF16-VALUED f32, which is what the pack holds for a BF16 source type.  Generating plain f32
    // weights would make the kernel (which stores 16 bits) differ from the reference by the rounding itself,
    // and the test would then be measuring the fixture.
    auto bf16_weight = [&](size_t n, float sigma) {
        std::vector<float> v(n);
        for (auto& x : v) x = to_bf16(gauss(rng) * sigma);
        return v;
    };
    std::vector<float> R((size_t) hc_dim);
    for (auto& x : R) x = gauss(rng);
    // STREAM SCALES 1, 4, 16, 64.  Without them trap 1 is only ~5% observable, because four streams of i.i.d.
    // normal noise have nearly the same RMS by accident - so a whole-stack RMS and a per-stream RMS agree to
    // within the noise, and the fixture would be asserting a distinction it cannot see.  Giving the streams
    // different magnitudes is what makes "per-stream" a claim with content.
    for (long long c = 0; c < hc; ++c)
        for (long long d = 0; d < n_embd; ++d) R[(size_t) (c * n_embd + d)] *= std::pow(4.0f, (float) c);
    // gamma stored as (1 + w), so values near 1 - a reader expecting a plain scale is the one this catches
    std::vector<float> w_norm((size_t) hc_dim);
    for (auto& x : w_norm) x = 1.0f + 0.1f * gauss(rng);
    std::vector<float> w_down = bf16_weight((size_t) (hc_lr * hc_dim), 0.2f);
    std::vector<float> w_up = bf16_weight((size_t) (hc_dim * hc_lr), 0.2f);
    std::vector<float> w_inject = bf16_weight((size_t) (hc * hc_dim), 0.05f);

    // NO PERMUTATION: the kernel takes both matrices exactly as the manifest (and `ref/gr.py`) store them.
    // `w_down` is (hc_lr, hc_dim) row-major and `w_up` is (hc_dim, hc_lr) row-major, so the bit-copy below is
    // the whole conversion.  An earlier version of this kernel wanted both TRANSPOSED - the test built them
    // that way and asserted the orientation observable - and that is gone because the warp-per-row mapping
    // makes it unnecessary.  What replaces it is the general orientation trap further down: a weight array
    // read with the wrong index expression must still produce a visibly different answer.
    std::vector<uint16_t> q_down(w_down.size()), q_up(w_up.size()), q_inject(w_inject.size());
    for (size_t i = 0; i < w_down.size(); ++i) q_down[i] = bf16_bits(w_down[i]);
    for (size_t i = 0; i < w_up.size(); ++i) q_up[i] = bf16_bits(w_up[i]);
    for (size_t i = 0; i < w_inject.size(); ++i) q_inject[i] = bf16_bits(w_inject[i]);

    // ---- device side
    float *d_R = nullptr, *d_norm = nullptr, *d_mixed = nullptr, *d_inject = nullptr;
    uint16_t *d_down = nullptr, *d_up = nullptr, *d_inj = nullptr;
    check(DPCT_CHECK_ERROR(d_R = sycl::malloc_device<float>(
                               R.size(), dpct::get_in_order_queue())),
          "m R");
    check(DPCT_CHECK_ERROR(d_norm = sycl::malloc_device<float>(
                               w_norm.size(), dpct::get_in_order_queue())),
          "m norm");
    check(DPCT_CHECK_ERROR(d_down = sycl::malloc_device<uint16_t>(
                               q_down.size(), dpct::get_in_order_queue())),
          "m down");
    check(DPCT_CHECK_ERROR(d_up = sycl::malloc_device<uint16_t>(
                               q_up.size(), dpct::get_in_order_queue())),
          "m up");
    check(DPCT_CHECK_ERROR(d_inj = sycl::malloc_device<uint16_t>(
                               q_inject.size(), dpct::get_in_order_queue())),
          "m inj");
    check(DPCT_CHECK_ERROR(d_mixed = sycl::malloc_device<float>(
                               (size_t)n_embd, dpct::get_in_order_queue())),
          "m mixed");
    check(DPCT_CHECK_ERROR(d_inject = sycl::malloc_device<float>(
                               (size_t)hc, dpct::get_in_order_queue())),
          "m inject");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_R, R.data(), R.size() * sizeof(float)).wait()),
          "c R");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_norm, w_norm.data(), w_norm.size() * sizeof(float)).wait()),
          "c norm");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_down, q_down.data(), q_down.size() * sizeof(uint16_t)).wait()),
          "c down");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_up, q_up.data(), q_up.size() * sizeof(uint16_t)).wait()),
          "c up");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_inj, q_inject.data(), q_inject.size() * sizeof(uint16_t)).wait()),
          "c inj");

    const strata::kernels::GrShapes sh{n_embd, hc, hc_lr};
    // DEVICE memory: the workspace is written by the kernel.
    void* d_ws_raw = nullptr;
    check(DPCT_CHECK_ERROR(d_ws_raw = (void *)sycl::malloc_device(
                               strata::kernels::gr_workspace_bytes(sh),
                               dpct::get_in_order_queue())),
          "m ws");
    strata::kernels::GrWorkspace ws;
    strata::kernels::gr_workspace_init(sh, d_ws_raw, ws);
    strata::kernels::gr_read(d_R, d_norm, d_down, d_up, d_inj, eps, sh, ws, d_mixed, d_inject, nullptr);
    std::vector<float> got_mixed((size_t) n_embd), got_inject((size_t) hc);
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memcpy(got_mixed.data(), d_mixed,
                                       got_mixed.size() * sizeof(float))
                               .wait()),
          "c mixed");
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memcpy(got_inject.data(), d_inject,
                                       got_inject.size() * sizeof(float))
                               .wait()),
          "c inject");

    int bad = 0;

    // ---- the traps, asserted OBSERVABLE before the kernel is judged against the right reading
    Opts right;
    std::vector<float> want_mixed, want_inject;
    reference(R, w_norm, w_down, w_up, w_inject, eps, n_embd, hc, hc_lr, right, want_mixed, want_inject);

    struct Trap { const char* name; Opts o; double floor; };
    const Trap traps[] = {
        {"per-stream RMSNorm vs whole-stack", {false, true, true, true, true}, 0.05},
        {"/hc inside the silu vs outside", {true, false, true, true, true}, 0.05},
        {"SiLU on lo / sigmoid on gate, swapped", {true, true, false, true, true}, 0.05},
        {"mean over streams vs sum", {true, true, true, false, true}, 0.50},
        {"BF16 vs FP32 activation contracts", {true, true, true, true, false}, 1e-4},
    };
    for (const Trap& t : traps) {
        std::vector<float> m, i;
        reference(R, w_norm, w_down, w_up, w_inject, eps, n_embd, hc, hc_lr, t.o, m, i);
        const double rel = rel_diff(want_mixed, m);
        const bool observable = rel > t.floor;
        std::printf("  %-40s %-4s (%.4f%% apart, floor %.3f%%)\n", t.name, observable ? "yes" : "*** NO ***",
                    rel * 100, t.floor * 100);
        if (!observable) {
            std::printf("      *** the fixture cannot see this trap, so passing proves nothing about it ***\n");
            ++bad;
        }
    }

    // ---- the kernel against the correct reference
    double mag = 0;
    const double rel_mixed = rel_diff(want_mixed, got_mixed, &mag);
    std::printf("\n  %-40s worst %.3e (mean |ref| %.4f)\n", "mixed vs reference", rel_mixed, mag);

    // The tolerance is set by SUMMATION ORDER, not by the structure: both sides round the activation to bf16
    // and both weights are bf16-valued, so every product is exact in f32 and only the order differs.  A
    // structural error is several orders of magnitude larger - the trap floors above say so.
    if (!(rel_mixed <= 1e-4)) {
        std::printf("    *** over 1e-4, which summation order does not explain - look for a structural bug ***\n");
        ++bad;
    }

    const double rel_inject = rel_diff(want_inject, got_inject);
    std::printf("  %-40s rel %.3e\n", "inject vs reference", rel_inject);
    if (!(rel_inject <= 1e-4)) {
        // Say WHICH elements and by how much: a whole-array L1 hides whether this is one stream wrong (an
        // indexing bug) or all of them slightly wrong (a precision bug), and those need different responses.
        std::printf("    want:");
        for (long long c = 0; c < hc; ++c) std::printf(" %12.6f", (double) want_inject[(size_t) c]);
        std::printf("\n    got :");
        for (long long c = 0; c < hc; ++c) std::printf(" %12.6f", (double) got_inject[(size_t) c]);
        std::printf("\n");
        ++bad;
    }

    // Same geometry and weights with the CUDA single-token activation contract.  The scalar fixture
    // below is the independent precision check; this case exercises multi-stream reductions and indexing.
    {
        Opts fp32;
        fp32.round_activation = false;
        std::vector<float> wm, wi, gm((size_t) n_embd), gi((size_t) hc);
        reference(R, w_norm, w_down, w_up, w_inject, eps, n_embd, hc, hc_lr, fp32, wm, wi);
        strata::kernels::gr_set_fp32_activations(true);
        strata::kernels::gr_read(d_R, d_norm, d_down, d_up, d_inj, eps, sh, ws, d_mixed, d_inject, nullptr);
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                      .memcpy(gm.data(), d_mixed, gm.size() * sizeof(float))
                      .wait()),
              "FP32 mixed");
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                      .memcpy(gi.data(), d_inject, gi.size() * sizeof(float))
                      .wait()),
              "FP32 inject");
        strata::kernels::gr_set_fp32_activations(false);
        const double rm = rel_diff(wm, gm), ri = rel_diff(wi, gi);
        const bool ok = rm <= 1e-4 && ri <= 1e-4;
        std::printf("  FP32 activations vs reference: %s (mixed %.3e, inject %.3e)\n",
                    ok ? "pass" : "FAIL", rm, ri);
        if (!ok) ++bad;
        strata::kernels::gr_read(d_R, d_norm, d_down, d_up, d_inj, eps, sh, ws, d_mixed, d_inject, nullptr);
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                      .memcpy(gm.data(), d_mixed, gm.size() * sizeof(float))
                      .wait()),
              "restored mixed");
        check(DPCT_CHECK_ERROR(
                  (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                      .memcpy(gi.data(), d_inject, gi.size() * sizeof(float))
                      .wait()),
              "restored inject");
        const bool restored = std::memcmp(gm.data(), got_mixed.data(), gm.size() * sizeof(float)) == 0 &&
                              std::memcmp(gi.data(), got_inject.data(), gi.size() * sizeof(float)) == 0;
        std::printf("  restoring default activation contract: %s\n", restored ? "byte-identical" : "FAIL");
        if (!restored) ++bad;
    }

    // ---- THE WEIGHT ORIENTATION MUST BE OBSERVABLE.  The kernel now takes both matrices exactly as the
    // manifest stores them, so there is no permutation for a loader to forget - but the INDEX EXPRESSIONS
    // inside the kernel are still a place to be wrong, and a wrong one gives right-shaped, wrong-valued
    // output.  Feed transposed copies through the same code path and require the answer to differ.
    {
        uint16_t *d_down_bad = nullptr, *d_up_bad = nullptr;
        check(DPCT_CHECK_ERROR(d_down_bad = sycl::malloc_device<uint16_t>(
                                   q_down.size(), dpct::get_in_order_queue())),
              "m down_bad");
        check(DPCT_CHECK_ERROR(d_up_bad = sycl::malloc_device<uint16_t>(
                                   q_up.size(), dpct::get_in_order_queue())),
              "m up_bad");
        std::vector<uint16_t> tr_down(q_down.size()), tr_up(q_up.size());
        for (long long k = 0; k < hc_lr; ++k)
            for (long long i = 0; i < hc_dim; ++i)
                tr_down[(size_t) (i * hc_lr + k)] = q_down[(size_t) (k * hc_dim + i)];
        for (long long i = 0; i < hc_dim; ++i)
            for (long long k = 0; k < hc_lr; ++k)
                tr_up[(size_t) (k * hc_dim + i)] = q_up[(size_t) (i * hc_lr + k)];
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_down_bad, tr_down.data(), tr_down.size() * 2).wait()),
              "cb d");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  d_up_bad, tr_up.data(), tr_up.size() * 2).wait()),
              "cb u");
        std::vector<float> bad_mixed((size_t) n_embd);
        float* d_bad = nullptr;
        check(
            DPCT_CHECK_ERROR(d_bad = sycl::malloc_device<float>(
                                 bad_mixed.size(), dpct::get_in_order_queue())),
            "m bad");
        strata::kernels::gr_read(d_R, d_norm, d_down_bad, d_up_bad, d_inj, eps, sh, ws, d_bad, d_inject,
                                 nullptr);
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                   .memcpy(bad_mixed.data(), d_bad,
                                           bad_mixed.size() * sizeof(float))
                                   .wait()),
              "cb b");
        const double rel = rel_diff(want_mixed, bad_mixed);
        const bool visible = rel > 0.05;
        std::printf("  %-40s %s (%.1f%% apart)\n", "weight orientation is observable",
                    visible ? "yes" : "*** NO ***", rel * 100);
        if (!visible) ++bad;
        sycl::free(d_down_bad, dpct::get_in_order_queue());
            sycl::free(d_up_bad, dpct::get_in_order_queue());
            sycl::free(d_bad, dpct::get_in_order_queue());
    }

    // ---- the FINAL mixer passes w_inject = nullptr, and then nothing may be written
    float sentinel = -12345.0f;
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memcpy(d_inject, &sentinel, sizeof(float))
                               .wait()),
          "c sentinel");
    strata::kernels::gr_read(d_R, d_norm, d_down, d_up, nullptr, eps, sh, ws, d_mixed, d_inject, nullptr);
    float after = 0.0f;
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memcpy(&after, d_inject, sizeof(float))
                               .wait()),
          "c after");
    const bool untouched = (after == sentinel);
    std::printf("  %-40s %s\n", "null w_inject writes nothing", untouched ? "yes" : "*** NO ***");
    if (!untouched) ++bad;

    // ---- gr_write: the `2*sigmoid` centring, asserted as the PROPERTY the source comment claims
    std::vector<float> block_out((size_t) n_embd);
    for (auto& x : block_out) x = gauss(rng);
    std::vector<float> zero_inj((size_t) hc, 0.0f);
    float *d_Rw = nullptr, *d_bo = nullptr, *d_zi = nullptr, *d_outw = nullptr;
    check(DPCT_CHECK_ERROR(d_Rw = sycl::malloc_device<float>(
                               R.size(), dpct::get_in_order_queue())),
          "m Rw");
    check(DPCT_CHECK_ERROR(d_bo = sycl::malloc_device<float>(
                               block_out.size(), dpct::get_in_order_queue())),
          "m bo");
    check(DPCT_CHECK_ERROR(d_zi = sycl::malloc_device<float>(
                               zero_inj.size(), dpct::get_in_order_queue())),
          "m zi");
    check(DPCT_CHECK_ERROR(d_outw = sycl::malloc_device<float>(
                               R.size(), dpct::get_in_order_queue())),
          "m outw");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_Rw, R.data(), R.size() * sizeof(float)).wait()),
          "c Rw");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_bo, block_out.data(), block_out.size() * sizeof(float)).wait()),
          "c bo");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_zi, zero_inj.data(), zero_inj.size() * sizeof(float)).wait()),
          "c zi");

    strata::kernels::gr_write(d_Rw, d_bo, d_zi, sh, d_outw, nullptr);
    std::vector<float> got_write((size_t) hc_dim);
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memcpy(got_write.data(), d_outw,
                                       got_write.size() * sizeof(float))
                               .wait()),
          "c outw");

    double worst_plain = 0;
    for (long long c = 0; c < hc; ++c)
        for (long long d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) (c * n_embd + d);
            const double want = (double) R[i] + (double) block_out[(size_t) d];
            worst_plain = std::fmax(worst_plain, rel_terms(want, (double) got_write[i], (double) R[i],
                                                           (double) block_out[(size_t) d]));
        }
    const bool plain = worst_plain < 1e-6;
    std::printf("  %-40s %s (worst rel %.3e)\n", "zero injection -> plain residual add",
                plain ? "yes" : "*** NO ***", worst_plain);
    if (!plain) ++bad;

    // the rival reading: sigmoid without the 2 halves the update, and must be visible.
    //
    // The comparison is on the DELTA `out - R`, not on `out`.  The claim is about the gate WEIGHT, and `R`
    // now reaches 64 in the last stream, so measuring `R + block_out*w` against the wrong reading dilutes a
    // 100% error in `w` down to 2.27% - which is how this check first passed as "not observable" while being
    // perfectly observable.  Measure the quantity the claim is about.
    {
        std::vector<float> got_delta((size_t) hc_dim), want_delta((size_t) hc_dim), wrong_delta((size_t) hc_dim);
        for (long long c = 0; c < hc; ++c)
            for (long long d = 0; d < n_embd; ++d) {
                const size_t i = (size_t) (c * n_embd + d);
                got_delta[i] = (float) ((double) got_write[i] - (double) R[i]);
                want_delta[i] = block_out[(size_t) d];        // 2*sigmoid(0) = 1.0
                wrong_delta[i] = block_out[(size_t) d] * 0.5f; // 1*sigmoid(0) = 0.5
            }
        const double rel = rel_diff(want_delta, wrong_delta);
        const double rel_got = rel_diff(want_delta, got_delta);
        const bool visible = rel > 0.05 && rel_got < 0.05;
        std::printf("  %-40s %s (readings %.0f%% apart, kernel within %.2e)\n",
                    "1*sigmoid vs 2*sigmoid is observable", visible ? "yes" : "*** NO ***", rel * 100, rel_got);
        if (!visible) ++bad;
    }

    // a non-zero injection: every stream receives the SAME block output, differing only by weight
    std::vector<float> inj((size_t) hc);
    for (auto& x : inj) x = gauss(rng) * 3.0f;
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_zi, inj.data(), inj.size() * sizeof(float)).wait()),
          "c inj2");
    strata::kernels::gr_write(d_Rw, d_bo, d_zi, sh, d_outw, nullptr);
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                               .memcpy(got_write.data(), d_outw,
                                       got_write.size() * sizeof(float))
                               .wait()),
          "c outw2");
    double worst_stream = 0;
    for (long long c = 0; c < hc; ++c) {
        const double w = 2.0 / (1.0 + std::exp(-(double) inj[(size_t) c] / (double) hc));
        for (long long d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) (c * n_embd + d);
            const double want = (double) R[i] + (double) block_out[(size_t) d] * w;
            worst_stream = std::fmax(worst_stream, rel_terms(want, (double) got_write[i], (double) R[i],
                                                             (double) block_out[(size_t) d] * w));
        }
    }
    const bool streams_ok = worst_stream < 1e-6;
    std::printf("  %-40s %s (worst rel %.3e)\n", "every stream adds the same block output",
                streams_ok ? "yes" : "*** NO ***", worst_stream);
    if (!streams_ok) ++bad;

    // ---- THE REAL DIMENSIONS. Exercise all activation/projection variants at n_embd=2560, hc=4, hc_lr=320.
    // Workspace sizing and every full weight row must agree with the model geometry.
    {
        const long long rn = 2560, rhc = 4, rlr = 320, rdim = rhc * rn;
        std::mt19937 rrng(99);
        std::normal_distribution<float> rg(0.0f, 1.0f);
        auto rbf16 = [&](size_t n, float sigma) {
            std::vector<float> v(n);
            for (auto& x : v) x = to_bf16(rg(rrng) * sigma);
            return v;
        };
        std::vector<float> rR((size_t) rdim);
        for (auto& x : rR) x = rg(rrng);
        for (long long c = 0; c < rhc; ++c)
            for (long long d = 0; d < rn; ++d) rR[(size_t) (c * rn + d)] *= std::pow(4.0f, (float) c);
        std::vector<float> rnorm((size_t) rdim);
        for (auto& x : rnorm) x = 1.0f + 0.1f * rg(rrng);
        std::vector<float> rdown = rbf16((size_t) (rlr * rdim), 0.2f);
        std::vector<float> rup = rbf16((size_t) (rdim * rlr), 0.2f);
        std::vector<float> rinj = rbf16((size_t) (rhc * rdim), 0.05f);

        std::vector<uint16_t> qd(rdown.size()), qu(rup.size()), qi(rinj.size());
        for (size_t i = 0; i < rdown.size(); ++i) qd[i] = bf16_bits(rdown[i]);
        for (size_t i = 0; i < rup.size(); ++i) qu[i] = bf16_bits(rup[i]);
        for (size_t i = 0; i < rinj.size(); ++i) qi[i] = bf16_bits(rinj[i]);

        float *dR = nullptr, *dN = nullptr, *dM = nullptr, *dI = nullptr;
        uint16_t *dD = nullptr, *dU = nullptr, *dJ = nullptr;
        check(DPCT_CHECK_ERROR(dR = (float *)sycl::malloc_device(
                                   rR.size() * 4, dpct::get_in_order_queue())),
              "rR");
        check(
            DPCT_CHECK_ERROR(dN = (float *)sycl::malloc_device(
                                 rnorm.size() * 4, dpct::get_in_order_queue())),
            "rN");
        check(DPCT_CHECK_ERROR(dD = (uint16_t *)sycl::malloc_device(
                                   qd.size() * 2, dpct::get_in_order_queue())),
              "rD");
        check(DPCT_CHECK_ERROR(dU = (uint16_t *)sycl::malloc_device(
                                   qu.size() * 2, dpct::get_in_order_queue())),
              "rU");
        check(DPCT_CHECK_ERROR(dJ = (uint16_t *)sycl::malloc_device(
                                   qi.size() * 2, dpct::get_in_order_queue())),
              "rJ");
        check(DPCT_CHECK_ERROR(dM = (float *)sycl::malloc_device(
                                   (size_t)rn * 4, dpct::get_in_order_queue())),
              "rM");
        check(
            DPCT_CHECK_ERROR(dI = (float *)sycl::malloc_device(
                                 (size_t)rhc * 4, dpct::get_in_order_queue())),
            "rI");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  dR, rR.data(), rR.size() * 4).wait()),
              "crR");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  dN, rnorm.data(), rnorm.size() * 4).wait()),
              "crN");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  dD, qd.data(), qd.size() * 2).wait()),
              "crD");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  dU, qu.data(), qu.size() * 2).wait()),
              "crU");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
                  dJ, qi.data(), qi.size() * 2).wait()),
              "crJ");

        const strata::kernels::GrShapes rsh{rn, rhc, rlr};
        void* rws_raw = nullptr;
        check(DPCT_CHECK_ERROR(rws_raw = (void *)sycl::malloc_device(
                                   strata::kernels::gr_workspace_bytes(rsh),
                                   dpct::get_in_order_queue())),
              "m rws");
        strata::kernels::GrWorkspace rws;
        strata::kernels::gr_workspace_init(rsh, rws_raw, rws);
        for (int mode = 0; mode < 3; ++mode) {
            select_activation_mode(mode);
            strata::kernels::gr_read(dR, dN, dD, dU, dJ, eps, rsh, rws, dM, dI, nullptr);
            std::vector<float> gm((size_t) rn), gi((size_t) rhc);
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                       .memcpy(gm.data(), dM, gm.size() * 4)
                                       .wait()),
                  "cgm");
            check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                       .memcpy(gi.data(), dI, gi.size() * 4)
                                       .wait()),
                  "cgi");

            Opts precision;
            precision.round_activation = mode == 0;
            std::vector<float> wm, wi;
            reference(rR, rnorm, rdown, rup, rinj, eps, rn, rhc, rlr, precision, wm, wi);
            const double rm = rel_diff(wm, gm), ri = rel_diff(wi, gi);
            const bool ok = rm <= 1e-4 && ri <= 1e-4;
            std::printf("\n  real dims 2560/4/320 %s %s (mixed %.3e, inject %.3e)\n",
                        activation_mode_name(mode), ok ? "pass" : "*** FAIL ***", rm, ri);
            if (!ok) ++bad;
        }
        bad += fused_multi_lds_parity(dN, dD, dU, dJ, eps);
        select_activation_mode(0);
        sycl::free(rws_raw, dpct::get_in_order_queue());
        sycl::free(dR, dpct::get_in_order_queue());
            sycl::free(dN, dpct::get_in_order_queue());
            sycl::free(dD, dpct::get_in_order_queue());
            sycl::free(dU, dpct::get_in_order_queue());
            sycl::free(dJ, dpct::get_in_order_queue());
            sycl::free(dM, dpct::get_in_order_queue());
            sycl::free(dI, dpct::get_in_order_queue());
    }

    bad += scalar_activation_contract();
    std::printf("\ngr_read/gr_write: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("gr_parity OK\n");
    return 0;
}
