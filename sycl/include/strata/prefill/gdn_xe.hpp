// sycl/include/strata/prefill/gdn_xe.hpp - the prompt GDN recurrence for Intel GPUs (variant 6 of
// gdn_recurrence_variant, src/prefill/kernels.dp.cpp).  The column kernel's four row groups and their 32-long FMA
// chains, in the same order, but a 16-lane sub-group holds four columns x four row groups (lane % 4 = column,
// lane / 4 = row group), so the two cross-row-group sums are sub-group shuffles instead of SLM round trips, and a
// token needs one work-group barrier instead of five.  Bit-identical to variant 0 on an Arc A770.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>

namespace strata::prefill::detail {
constexpr int GDN_XE_S = 128, GDN_XE_HK = 16, GDN_XE_HV = 48, GDN_XE_C = 10240;
constexpr int GDN_XE_CB = 16, GDN_XE_NCB = GDN_XE_S / GDN_XE_CB;
constexpr int GDN_XE_NT = GDN_XE_CB * 4;

// Do not replace with reduce_over_group: keep ((p0+p1)+p2)+p3,
// exactly the old red[0][c]+red[1][c]+red[2][c]+red[3][c] expression.
inline float gdn_xe_sum4(sycl::sub_group sg, float x, unsigned c) {
    const float p0 = sycl::select_from_group(sg, x, c);
    const float p1 = sycl::select_from_group(sg, x, c + 4);
    const float p2 = sycl::select_from_group(sg, x, c + 8);
    const float p3 = sycl::select_from_group(sg, x, c + 12);
    return p0 + p1 + p2 + p3;
}

inline void gdn_xe_body(sycl::nd_item<1> it, float* __restrict__ state,
                        const float* __restrict__ h, const float* __restrict__ gate,
                        const float* __restrict__ beta, float* __restrict__ out,
                        int64_t T, sycl::local_accessor<float, 1> inputs) {
    const auto sg = it.get_sub_group();
    const unsigned lane = sg.get_local_linear_id();
    const unsigned rg = lane / 4, c = lane % 4;
    const unsigned tid = it.get_local_linear_id();
    const unsigned head = it.get_group_linear_id() / GDN_XE_NCB;
    const unsigned cb = it.get_group_linear_id() % GDN_XE_NCB;
    const unsigned col = cb * GDN_XE_CB + sg.get_group_linear_id() * 4 + c;
    const unsigned qh = head % GDN_XE_HK;
    constexpr unsigned RPG = GDN_XE_S / 4;
    constexpr size_t rs = GDN_XE_HV * GDN_XE_S;
    float* base = state + (rg * RPG * GDN_XE_HV + head) * GDN_XE_S + col;
    float s[RPG];
#pragma unroll
    for (unsigned r = 0; r < RPG; ++r) s[r] = base[r * rs];
    float nq[2], nk[2], nv = 0, ng = 0, nb = 0;
    auto fetch = [&](int64_t t) {
        const float* ht = h + t * GDN_XE_C;
#pragma unroll
        for (unsigned u = 0; u < 2; ++u) {
            nq[u] = ht[qh * GDN_XE_S + tid + u * GDN_XE_NT];
            nk[u] = ht[GDN_XE_HK * GDN_XE_S + qh * GDN_XE_S + tid + u * GDN_XE_NT];
        }
        nv = ht[2 * GDN_XE_HK * GDN_XE_S + head * GDN_XE_S + col];
        ng = gate[t * GDN_XE_HV + head];
        nb = beta[t * GDN_XE_HV + head];
    };
    if (T > 0) fetch(0);
    for (int64_t t = 0; t < T; ++t) {
        // Alternate q/k banks. The barrier at t-1 guarantees every read of
        // bank t-2 has finished before we overwrite it at t. No end barrier.
        const unsigned bank = (t & 1) * 2 * GDN_XE_S;
#pragma unroll
        for (unsigned u = 0; u < 2; ++u) {
            inputs[bank + tid + u * GDN_XE_NT] = nq[u];
            inputs[bank + GDN_XE_S + tid + u * GDN_XE_NT] = nk[u];
        }
        const float cv = nv, cg = ng, cbt = nb;
        it.barrier(sycl::access::fence_space::local_space);
        if (t + 1 < T) fetch(t + 1);
        const float g = sycl::native::exp(cg);
        float kv = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], inputs[bank + GDN_XE_S + rg * RPG + r], kv);
        const float kv_col = gdn_xe_sum4(sg, kv, c);
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < RPG; ++r) {
            s[r] = sycl::fma(g, s[r], inputs[bank + GDN_XE_S + rg * RPG + r] * delta);
            o = sycl::fma(s[r], inputs[bank + rg * RPG + r], o);
        }
        const float oc = gdn_xe_sum4(sg, o, c) * sycl::rsqrt(float(GDN_XE_S));
        if (rg == 0) out[t * GDN_XE_HV * GDN_XE_S + head * GDN_XE_S + col] = oc;
    }
#pragma unroll
    for (unsigned r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// Intel GPUs only (the caller checks gdn_xe_device).  256 GRFs a thread: an offline DG2 (acm-g10) compile of this kernel
// spilled 4000 bytes a thread at the default 128.  Keeping the state in SLM instead of registers removed the spill
// but ran ~4x slower on an A770 (13.7 vs 3.1-3.3 ms at T=1000, same bits), so the state stays in registers.
inline sycl::event gdn_xe_launch(sycl::queue& q, float* state, const float* h,
                                const float* gate, const float* beta, float* out, int64_t T) {
    const auto range = sycl::nd_range<1>(GDN_XE_HV * GDN_XE_NCB * GDN_XE_NT, GDN_XE_NT);
    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> inputs(4 * GDN_XE_S, cgh);   // two q/k banks, 2048 bytes a work-group
        cgh.parallel_for<class GdnXeLargeGrf>(range,
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                gdn_xe_body(it, state, h, gate, beta, out, T, inputs);
            });
    });
}
} // namespace strata::prefill::detail
