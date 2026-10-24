// src/kernels/kv_q8_parity.cpp - plan v0.3 P7: INT8 KV append/gather against a host reference (GPU, no model).
//
// Appends random K/V cells at scattered positions through a non-identity page table into both the INT8 pools
// (kv_append_q8_step) and the FP16 pools (kv_append_step), gathers random cell sets from both, and checks:
//   1. INT8 codes and FP16 scales are BITWISE equal to the host reference (same f16_from_f32 rounding);
//   2. the INT8 gather is BITWISE equal to the host dequantization of those codes;
//   3. against the FP16 path, the INT8 values differ by at most half a quantization step of their group plus the
//      fp16 rounding of both sides (0.624 steps).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(dpct::err0 e, const char *w) {
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}
template <typename T>
T *dalloc(size_t n) {
    T *p = nullptr;
    ck(DPCT_CHECK_ERROR(p = (T *)sycl::malloc_device(
                            n * sizeof(T) + 64, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(p, 0, n * sizeof(T) + 64).wait()),
       "memset");
    return p;
}
}  // namespace

int main() {
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;                                                              // several pages in a small pool
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int pages = 8, cells = pages * P;
    std::mt19937 rng(17);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 5 + 3) % pages;            // a non-identity permutation
    int32_t* d_table = dalloc<int32_t>(pages);
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_table, table.data(), pages * 4).wait()),
       "table");
    int8_t *kq = dalloc<int8_t>((size_t) cells * H * D), *vq = dalloc<int8_t>((size_t) cells * H * D);
    uint16_t *ks = dalloc<uint16_t>((size_t) cells * H * G), *vs = dalloc<uint16_t>((size_t) cells * H * G);
    uint16_t *kp = dalloc<uint16_t>((size_t) cells * H * D), *vp = dalloc<uint16_t>((size_t) cells * H * D);
    float *kcur = dalloc<float>(H * D), *vcur = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    std::vector<std::vector<float>> hk(cells), hv(cells);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    const int n_fill = cells - 37;                                                  // leave some cells empty
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        const float scale = (n % 7 == 0) ? 1e-3f : (n % 11 == 0 ? 40.f : 1.f);    // small and large groups
        for (auto& x : kv) x = nd(rng) * scale;
        for (auto& x : vv) x = nd(rng) * scale;
        if (n % 13 == 0) std::fill(kv.begin(), kv.begin() + 64, 0.f);              // an all-zero group
        hk[pos] = kv; hv[pos] = vv;
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(step, hstep, sizeof hstep).wait()),
           "step");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(kcur, kv.data(),
                                                              kv.size() * 4).wait()),
           "k");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(vcur, vv.data(),
                                                              vv.size() * 4).wait()),
           "v");
        k::kv_append_q8_step(kq, vq, ks, vs, d_table, step, kcur, vcur, s, nullptr);
        k::kv_append_step(kp, vp, d_table, step, kcur, vcur, s, nullptr);
        ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
           "append");
    }
    // 1. codes and scales bitwise vs the host reference
    std::vector<int8_t> hkq((size_t) cells * H * D), hvq(hkq.size());
    std::vector<uint16_t> hks((size_t) cells * H * G), hvs(hks.size());
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(hkq.data(), kq, hkq.size())
                            .wait()),
       "d2h");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(hvq.data(), vq, hvq.size())
                            .wait()),
       "d2h");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(hks.data(), ks, hks.size() * 2)
                            .wait()),
       "d2h");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(hvs.data(), vs, hvs.size() * 2)
                            .wait()),
       "d2h");
    long bad_codes = 0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int kvsel = 0; kvsel < 2; ++kvsel)
            for (int h = 0; h < H; ++h)
                for (int g = 0; g < G; ++g) {
                    const float* x = (kvsel ? hv[pos] : hk[pos]).data() + h * D + g * 64;
                    float amax = 0.f;
                    for (int t = 0; t < 64; ++t) amax = std::fmax(amax, std::fabs(x[t]));
                    const uint16_t sb = k::f16_from_f32(amax / 127.0f);
                    const float sf = k::f32_from_f16(sb);
                    const long long row = ((long long) table[pos / P] * H + h) * P + pos % P;
                    if ((kvsel ? hvs : hks)[row * G + g] != sb) ++bad_codes;
                    for (int t = 0; t < 64; ++t) {
                        int q = sf > 0.f ? (int) std::nearbyint(x[t] / sf) : 0;
                        q = std::clamp(q, -127, 127);
                        if ((kvsel ? hvq : hkq)[row * D + g * 64 + t] != (int8_t) q) ++bad_codes;
                    }
                }
    }
    if (bad_codes) { std::fprintf(stderr, "FAIL: %ld codes/scales differ from the host reference\n", bad_codes); ++g_fail; }
    // 2 + 3. gather random selections from both paths
    const int max_ids = 200;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t *k8 = dalloc<uint16_t>((size_t) max_ids * H * D), *v8 = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t *k16 = dalloc<uint16_t>((size_t) max_ids * H * D), *v16 = dalloc<uint16_t>((size_t) max_ids * H * D);
    double worst = 0.0;
    for (int trial = 0; trial < 20; ++trial) {
        const int n_ids = 1 + (int) (rng() % max_ids);
        std::vector<int32_t> ids(n_ids);
        for (auto& id : ids) id = positions[rng() % n_fill];
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_ids, ids.data(), n_ids * 4).wait()),
           "ids");
        int32_t hstep[k::kStepCount] = {0, 0, 0, n_ids};
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(step, hstep, sizeof hstep).wait()),
           "step");
        k::kv_gather_q8_step(kq, vq, ks, vs, d_table, d_ids, step, max_ids, s, k8, v8, nullptr);
        k::kv_gather_step(kp, vp, d_table, d_ids, step, max_ids, s, k16, v16, nullptr);
        ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
           "gather");
        std::vector<uint16_t> a8((size_t) n_ids * H * D), b8(a8.size()), a16(a8.size()), b16(a8.size());
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(a8.data(), k8, a8.size() * 2)
                                .wait()),
           "d2h");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(b8.data(), v8, b8.size() * 2)
                                .wait()),
           "d2h");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(a16.data(), k16, a16.size() * 2)
                                .wait()),
           "d2h");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(b16.data(), v16, b16.size() * 2)
                                .wait()),
           "d2h");
        for (int j = 0; j < n_ids; ++j)
            for (int h = 0; h < H; ++h) {
                const long long row = ((long long) table[ids[j] / P] * H + h) * P + ids[j] % P;
                for (int d = 0; d < D; ++d) {
                    const size_t o = ((size_t) j * H + h) * D + d;
                    for (int kvsel = 0; kvsel < 2; ++kvsel) {
                        const float sf = k::f32_from_f16((kvsel ? hvs : hks)[row * G + d / 64]);
                        const int8_t q = (kvsel ? hvq : hkq)[row * D + d];
                        const uint16_t want = k::f16_from_f32((float) q * sf);
                        const uint16_t got = (kvsel ? b8 : a8)[o];
                        if (got != want) { if (g_fail < 5) std::fprintf(stderr, "FAIL: gather id %d h %d d %d\n", j, h, d); ++g_fail; }
                        const float ref = k::f32_from_f16((kvsel ? b16 : a16)[o]);
                        const double err = std::fabs(k::f32_from_f16(got) - ref) / (sf > 0 ? sf : 1.0);
                        worst = std::max(worst, err);
                    }
                }
            }
    }
    // |x - q*sf| <= sf/2 (the stored scale keeps |x|/sf <= 127.06, so no clamping), plus the fp16 rounding of each
    // side: at most 2^-11 relative of a value up to 127*sf, i.e. 127/2048 = 0.062 steps each -> 0.5 + 0.124
    if (!(worst <= 0.5 + 2.0 * 127.0 / 2048.0 + 1e-3)) { std::fprintf(stderr, "FAIL: INT8 vs FP16 error %.3f steps\n", worst); ++g_fail; }
    std::printf("kv_q8_parity: %s (worst INT8-vs-FP16 error %.3f quantization steps)\n", g_fail ? "FAILED" : "OK", worst);
    return g_fail ? 1 : 0;
}
