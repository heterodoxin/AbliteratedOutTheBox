// src/kernels/fused_gr_bench.cpp - fused_gr_read_multi: the AMD fast kernels (STRATA_GR_FAST) against the old
// ones, bit for bit on every output (lo, rs, inject, mixed, the in-place R), and each one's time.
//
//     build/fused_gr_bench [iters] [T min] [T max]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/fused_gr.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 6;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix;
    uint16_t *dwd, *dwu, *dwi;
    dR = (float *)sycl::malloc_device(R.size() * 4, dpct::get_in_order_queue());
    dbo =
        (float *)sycl::malloc_device(bo.size() * 4, dpct::get_in_order_queue());
    dinj = (float *)sycl::malloc_device(inj.size() * 4,
                                        dpct::get_in_order_queue());
    dwn =
        (float *)sycl::malloc_device(wn.size() * 4, dpct::get_in_order_queue());
    dxn = (float *)sycl::malloc_device((size_t)TM * D * 4,
                                       dpct::get_in_order_queue());
    dlo = (float *)sycl::malloc_device((size_t)TM * LR * 4,
                                       dpct::get_in_order_queue());
    drs = (float *)sycl::malloc_device((size_t)TM * HC * 4,
                                       dpct::get_in_order_queue());
    dio = (float *)sycl::malloc_device((size_t)TM * HC * 4,
                                       dpct::get_in_order_queue());
    dmix = (float *)sycl::malloc_device((size_t)TM * N * 4,
                                        dpct::get_in_order_queue());
    dwd = (uint16_t *)sycl::malloc_device(wd.size() * 2,
                                          dpct::get_in_order_queue());
    dwu = (uint16_t *)sycl::malloc_device(wu.size() * 2,
                                          dpct::get_in_order_queue());
    dwi = (uint16_t *)sycl::malloc_device(wi.size() * 2,
                                          dpct::get_in_order_queue());
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dbo, bo.data(), bo.size() * 4).wait();
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dinj, inj.data(), inj.size() * 4).wait();
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dwn, wn.data(), wn.size() * 4).wait();
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dwd, wd.data(), wd.size() * 2).wait();
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dwu, wu.data(), wu.size() * 2).wait();
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dwi, wi.data(), wi.size() * 2).wait();
    dpct::queue_ptr s;
    s = dpct::get_current_device().create_queue(true);
    dpct::event_ptr e0, e1;
    e0 = new sycl::event();
    e1 = new sycl::event();
    int failures = 0;
    for (int T = t_lo; T <= t_hi; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                K::FusedGrArgs a[TM];
                for (int t = 0; t < T; ++t) {
                    a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
                    a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                    a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                    a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                    a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
                }
                std::vector<float> out[2];
                for (int f = 0; f < 2; ++f) {
                    dpct::get_in_order_queue()
                        .memcpy(dR, R.data(), R.size() * 4)
                        .wait();
                    dpct::get_in_order_queue()
                        .memset(dlo, 0xff, (size_t)TM * LR * 4)
                        .wait();
                    dpct::get_in_order_queue()
                        .memset(drs, 0xff, (size_t)TM * HC * 4)
                        .wait();
                    dpct::get_in_order_queue()
                        .memset(dio, 0xff, (size_t)TM * HC * 4)
                        .wait();
                    dpct::get_in_order_queue()
                        .memset(dmix, 0xff, (size_t)TM * N * 4)
                        .wait();
                    K::fused_gr_set_fast(f);
                    K::fused_gr_read_multi(a, T, dxn, s);
                    s->wait();
                    auto& o = out[f];
                    o.resize((size_t) TM * (D + LR + HC + HC + N));
                    float* p = o.data();
                    dpct::get_in_order_queue()
                        .memcpy(p, dR, (size_t)TM * D * 4)
                        .wait();
                        p += (size_t)TM * D;
                    dpct::get_in_order_queue()
                        .memcpy(p, dlo, (size_t)TM * LR * 4)
                        .wait();
                        p += (size_t)TM * LR;
                    dpct::get_in_order_queue()
                        .memcpy(p, drs, (size_t)TM * HC * 4)
                        .wait();
                        p += (size_t)TM * HC;
                    dpct::get_in_order_queue()
                        .memcpy(p, dio, (size_t)TM * HC * 4)
                        .wait();
                        p += (size_t)TM * HC;
                    dpct::get_in_order_queue()
                        .memcpy(p, dmix, (size_t)TM * N * 4)
                        .wait();
                }
                const bool same = std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
                double us[2] = {1e30, 1e30};
                for (int round = 0; round < 3; ++round)
                    for (int f = 0; f < 2; ++f) {
                        K::fused_gr_set_fast(f);
                        for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        dpct::sync_barrier(e0, s);
                        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        dpct::sync_barrier(e1, s);
                        e1->wait_and_throw();
                        float ms = 0;
                        ms = (e1->get_profiling_info<
                                  sycl::info::event_profiling::command_end>() -
                              e0->get_profiling_info<
                                  sycl::info::event_profiling::
                                      command_start>()) /
                             1000000.0f;
                        us[f] = std::fmin(us[f], 1e3 * ms / iters);
                    }
                std::printf("T %d apply %d inject %d | old %6.1f us  fast %6.1f us | %s\n", T, apply, inject, us[0], us[1],
                            same ? "bitwise equal" : "DIFFERS");
                if (!same) ++failures;
            }
    std::printf("fused_gr_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
