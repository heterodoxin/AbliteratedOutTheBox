// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include "strata/core/geometry.hpp"
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/resident_plan_mirror.hpp"
#include "strata/kernels/dp4a.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef STRATA_PLAN_LOCAL
#define STRATA_PLAN_LOCAL 1   // 0: the original one-thread plan kernel (A/B)
#endif
namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
static_assert(S == strata::core::kCompiledGeometry.ssm_state_size, "verify_kernels is compiled for core/geometry.hpp");
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

// `commit` (one token): the history then keeps it - [hist1, hist2, x_0], each thread its channel's, after reading them
__dpct_inline__ void gdn_conv_l2_multi_kernel(float *hist,
                                              const float *__restrict__ qkv,
                                              const float *__restrict__ w,
                                              float *__restrict__ h, int C,
                                              int qk_heads, float eps,
                                              int t_begin, bool commit) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = t_begin + item_ct1.get_group(1);
    const int c = item_ct1.get_group(2) * S + item_ct1.get_local_id(2);
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    if (commit) {
        hist[c * 3] = v1;
        hist[c * 3 + 1] = v2;
        hist[c * 3 + 2] = x;
    }
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + sycl::native::exp(-sum));
    if ((int)item_ct1.get_group(2) < qk_heads) {
        float sq = y * y;
        /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                sq, o);
        if ((item_ct1.get_local_id(2) & 31) == 0)
            part[item_ct1.get_local_id(2) >> 5] = sq;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= sycl::rsqrt(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__dpct_inline__ void
gdn_conv_commit_kernel(float *__restrict__ hist, const float *__restrict__ qkv,
                       int C, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

template <int MAX_T = kVerifyMaxT, bool EXACT_T = false>
/*
DPCT1110: The total declared local variable size in device function
gdn_ab_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_ab_multi_kernel(
    const float *__restrict__ x, const uint16_t *__restrict__ wa,
    const uint16_t *__restrict__ wb, const float *__restrict__ dt,
    const float *__restrict__ ssm_a, float *__restrict__ gate,
    float *__restrict__ beta, int n, int h_v, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 2 + (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
        (is_beta ? wb : wa) + (size_t)r * n);
    float acc[MAX_T];
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wv = *(w4 + j);
        const float w0 = sycl::bit_cast<float>(wv.x() << 16),
                    w1 = sycl::bit_cast<float>(wv.x() & 0xffff0000u);
        const float w2 = sycl::bit_cast<float>(wv.y() << 16),
                    w3 = sycl::bit_cast<float>(wv.y() & 0xffff0000u);
        const float w4f = sycl::bit_cast<float>(wv.z() << 16),
                    w5 = sycl::bit_cast<float>(wv.z() & 0xffff0000u);
        const float w6 = sycl::bit_cast<float>(wv.w() << 16),
                    w7 = sycl::bit_cast<float>(wv.w() & 0xffff0000u);
#pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (!EXACT_T && t >= T) break;
            const float* xt = x + (size_t) t * n;
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const sycl::float4 xa = *reinterpret_cast<const sycl::float4 *>(xt + j * 8);
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const sycl::float4 xb = *reinterpret_cast<const sycl::float4 *>(xt + j * 8 + 4);
            float a = acc[t];
            a = sycl::fma((float)w0, (float)(xa.x()), a);
                a = sycl::fma((float)w1, (float)(xa.y()), a);
            a = sycl::fma((float)w2, (float)(xa.z()), a);
                a = sycl::fma((float)w3, (float)(xa.w()), a);
            a = sycl::fma((float)w4f, (float)(xb.x()), a);
                a = sycl::fma((float)w5, (float)(xb.y()), a);
            a = sycl::fma((float)w6, (float)(xb.z()), a);
                a = sycl::fma((float)w7, (float)(xb.w()), a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        if (!EXACT_T && t >= T) break;
        float a = acc[t];
#pragma unroll
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = 16; o > 0; o >>= 1) a +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t)t * h_v + r] = 1.0f / (1.0f + sycl::native::exp(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : sycl::log1p(sycl::native::exp(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

// State-only commit kernel: each head's 128 independent state columns are split across 4 blocks of 32 columns
// (48 * 4 = 192 blocks of 128 threads across all SMs, vs 48 blocks of 512 threads), with no sq/o/norm/z/y work.
/*
DPCT1110: The total declared local variable size in device function
gdn_step_commit_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_step_commit_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta, int h_k,
    int h_v, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int n = *n_keep;
    if (n <= 0) return;
    auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col_local = item_ct1.get_local_id(2);
    const int col = item_ct1.get_group(1) * 32 + col_local;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        sk[tid] = ht[qk + qh * S + tid];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

// S26 (STRATA_QFUSE=1): the q8_1 image of a 32-column block, written by the warp that holds those columns with
// native_quantize_q8_1_kernel's quantizer (the same XOR-tree max and sum, d = amax / 127, roundf(x / d), ds = (d, sum)).
struct GdnQ81 { sycl::half2 ds; int8_t qs[32]; };
__dpct_inline__ void gdn_q8_1_store(GdnQ81 *__restrict__ xq, size_t idx,
                                    float xi) {
    // no contraction: xi is a product here, and fma(a, b, shfl) would round the first sum unlike the separate
    // quantizer, which reads xi from memory (S26 harness: ds.sum differed by 1 ulp without this)
#pragma clang fp contract(off)
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(
        amax, dpct::experimental::permute_sub_group_by_xor(
                  0xffffffffu,
                  sycl::ext::oneapi::this_work_item::get_sub_group(), amax, o));
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) sum +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sum, o);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
    xq[idx / 32].qs[idx % 32] = q;
    if (idx % 32 == 0) xq[idx / 32].ds = sycl::half2(d, sum);
}

template <bool ALL_OUT, bool Q>
/*
DPCT1110: The total declared local variable size in device function
gdn_step_norm_multi_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void gdn_step_norm_multi_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta,
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_k, int h_v, int T,
    const int32_t *__restrict__ n_keep, int t_out_begin,
    GdnQ81 *__restrict__ xq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_kv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_o =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wsum =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col = item_ct1.get_local_id(2);
    const int rg = item_ct1.get_local_id(1);
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = ALL_OUT ? T : (n_keep ? *n_keep : T);
    if (n > 0) {
        if (rg == 0) sk[0][col] = hbuf[qk + qh * S + col];
        else if (rg == 1 && (ALL_OUT || 0 >= t_out_begin)) sq[0][col] = hbuf[qh * S + col];
    }
    const float gam = (rg == 0) ? gamma[col] : 0.0f;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    for (int r = 0; r < RPG; ++r) s[r] = ALL_OUT ? base[r * row_stride]
                                                 : base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        const float* ht = hbuf + (size_t) t * C;
        const bool need_out = ALL_OUT || (t >= t_out_begin);
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (t + 1 < n) {
            const float* ht_next = ht + C;
            const bool need_next = ALL_OUT || (t + 1 >= t_out_begin);
            if (rg == 0) sk[nxt][col] = ht_next[qk + qh * S + col];
            else if (rg == 1 && need_next) sq[nxt][col] = ht_next[qh * S + col];
        }
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[cur][rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r)
                s[r] = sycl::fma((float)g, s[r], sk[cur][rg * RPG + r] * delta);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[cur][rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[cur][rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] +
                  red_o[3][col]) *
                 sycl::rsqrt((float)S);
            sq_part = oc * oc;
        }
        // every thread takes the shuffles, as the single-token kernel does: inside the `rg == 0` branch hipcc on
        // gfx1151 gave another rounding for 3% of the outputs (gdn_parity section 5 on Aurora)
        /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                sq_part, o2);
        if (rg == 0 && (col & 31) == 0) wsum[col >> 5] = sq_part;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = sycl::rsqrt(ss / (float)S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            const float yv =
                oc * scale * gam * (1.0f / (1.0f + sycl::native::exp(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            if constexpr (Q) gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
        }
    }
    if (!ALL_OUT && n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

// S25 (STRATA_GDN_SPLIT=1): the same recurrence with each head's 128 state columns over 4 blocks of 32 columns
// (128 threads: the same (column, row group) threads, each with the same 32 state rows), so 4x the blocks of
// gdn_step_norm_multi_kernel - which ran one 512-thread block per head (32 blocks) and read the 2 MB state of a layer
// at ~60 GB/s. Every column's arithmetic is the original's; the output norm, which needs all 128 columns of a head,
// moves to gdn_out_norm_kernel: this kernel leaves the unnormalized output `oc` in y and the second kernel sums the
// squares by the same warps (columns 32w..32w+31, the same butterfly) in the same order.
// S26: OUT = false (the commit, whose outputs nobody reads) drops the output reduction and its two barriers; the
// norm kernel below has the old kernel's code shape (S x RG threads, rg 0 holding oc), which makes it bitwise equal -
// the 128-thread version differed from the old kernel by 1 ulp in ~7% of the outputs.
constexpr int GS_COLS = 32;
template <bool OUT>
/*
DPCT1110: The total declared local variable size in device function
gdn_step_split_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
gdn_step_split_kernel(float *__restrict__ state, const float *__restrict__ hbuf,
                      int C, const float *__restrict__ gate,
                      const float *__restrict__ beta, float *__restrict__ y,
                      int h_k, int h_v, int T,
                      const int32_t *__restrict__ n_keep, int t_out_begin) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RG][GS_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int c0 = item_ct1.get_group(1) * GS_COLS;
    const int lc = item_ct1.get_local_id(2); // local column
    const int col = c0 + lc;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        if (!OUT) continue;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        red[rg][lc] = o;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (rg == 0 && t >= t_out_begin)
            y[(size_t)t * value_dim + head * S + col] =
                (red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc]) *
                sycl::rsqrt((float)S);
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}


// S26: the state-only replay (the commit) with its per-token inputs prefetched - gate, beta and the v column of every
// row up front, the next row's k / q while the current one computes. The arithmetic and its order are
// gdn_step_split_kernel<false>'s (gfx1151 harness: state bitwise equal; 44.5 -> 39.1 us at 1 row, 47.4 -> 40.5 at 2,
// 53.8 -> 42.9 at 4, h_v 48). With outputs the prefetch lost (register pressure), so the window keeps the plain split.
template <int MAXT>
/*
DPCT1110: The total declared local variable size in device function
gdn_state_replay_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_state_replay_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta, int h_k,
    int h_v, int T, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RG][GS_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int c0 = item_ct1.get_group(1) * GS_COLS;
    const int lc = item_ct1.get_local_id(2);
    const int col = c0 + lc;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int n = n_keep ? *n_keep : T;
    float gv[MAXT], bv[MAXT], vv[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < n) {
            gv[t] = gate[(size_t) t * h_v + head];
            bv[t] = beta[(size_t) t * h_v + head];
            vv[t] = hbuf[(size_t) t * C + 2 * qk + head * S + col];
        }
    }
    float kn = 0.0f, qn = 0.0f;
    if (n > 0) { kn = hbuf[qk + qh * S + tid]; qn = hbuf[qh * S + tid]; }
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t >= n) break;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        sk[tid] = kn; sq[tid] = qn;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (t + 1 < n) {
            const float* hn = hbuf + (size_t) (t + 1) * C;
            kn = hn[qk + qh * S + tid]; qn = hn[qh * S + tid];
        }
        const float g = sycl::native::exp(gv[t]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (vv[t] - g * kv_col) * bv[t];
        float o = 0.0f;   // unused: kept because the measured code had it (without it: 51 us instead of 39 at 1 row)
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        (void) o;
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}
// y = oc * rsqrt(mean(oc^2) + eps) * gamma * sigmoid(z) per head and token: gdn_step_norm_multi_kernel's tail with its
// code shape (S x RG threads; rg 0 holds oc, the other row groups contribute 0 to wsum), reading oc from y
template <bool Q>
__dpct_inline__ void gdn_out_norm_kernel(
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_v, int T, const int32_t *__restrict__ n_keep,
    int t_out_begin, GdnQ81 *__restrict__ xq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &wsum =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S * RG / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2),
              t = item_ct1.get_group(1) + t_out_begin,
              col = item_ct1.get_local_id(2), rg = item_ct1.get_local_id(1);
    const int tid = rg * S + col;
    const int n = n_keep ? *n_keep : T;
    if (t >= n) return;
    const int value_dim = S * h_v;
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = y[(size_t) t * value_dim + head * S + col];
        sq_part = oc * oc;
    }
    /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sq_part, o2);
    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = sycl::rsqrt(ss / (float)S + eps);
        const float zz = z[(size_t) t * value_dim + head * S + col];
        if constexpr (Q) {
            const float yv = oc * scale * gamma[col] *
                             (1.0f / (1.0f + sycl::native::exp(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
        } else {
            y[(size_t)t * value_dim + head * S + col] =
                oc * scale * gamma[col] *
                (1.0f / (1.0f + sycl::native::exp(-zz)));
        }
    }
}

__dpct_inline__ void embedding_gather_dev_kernel(
    const uint8_t *__restrict__ codes, const float *__restrict__ scales,
    const float *__restrict__ offsets, const int32_t *__restrict__ tokens,
    int64_t n, int code_bits, int code_bias, int group_elems,
    unsigned long long row_codes, unsigned long long row_groups,
    float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float product = (float)(code + code_bias) * sc[group];
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    out[(size_t)t * n + i] = product + (of ? of[group] : 0.0f);
}

__dpct_inline__ void broadcast_streams_kernel(const float *__restrict__ x,
                                              float *__restrict__ R, int64_t n,
                                              int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__dpct_inline__ void copy_indexed_kernel(float *__restrict__ dst,
                                         const float *__restrict__ src,
                                         int64_t stride,
                                         const int32_t *__restrict__ index,
                                         int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx = *index;
    if (idx < 0) return;
#pragma unroll
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n; i += (int64_t)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2))
        dst[i] = src[(size_t) idx * stride + i];
}

__dpct_inline__ void
fetch_blobs_kernel(const unsigned long long *__restrict__ src,
                   const int32_t *__restrict__ n, sycl::uint4 *__restrict__ dst,
                   long long per) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = (long long)*n * per;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const sycl::uint4 *)src[k])[off];
    }
}

__dpct_inline__ void rebase_ptrs_kernel(unsigned long long *ptr,
                                        const int32_t *n,
                                        unsigned long long base,
                                        long long bytes) {
    const int k =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__dpct_inline__ void add_streams_broadcast_kernel(const float *__restrict__ h,
                                                  const float *__restrict__ e,
                                                  float *__restrict__ R,
                                                  int64_t n, int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__dpct_inline__ void ident_hits_kernel(const int32_t *__restrict__ ids, int n,
                                       int32_t *__restrict__ slot,
                                       int32_t *__restrict__ dst,
                                       int32_t *__restrict__ count) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template <typename E>
__dpct_inline__ void gather_rows_kernel(const E *__restrict__ src,
                                        long long row_e,
                                        const int32_t *__restrict__ ids,
                                        long long n, E *__restrict__ dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = n * row_e;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__dpct_inline__ void map_ids_kernel(int32_t *ids,
                                    const int32_t *__restrict__ table, int n) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) ids[i] = table[ids[i]];
}

__dpct_inline__ void row_top_prob_kernel(const float *__restrict__ logits,
                                         int n_vocab,
                                         const int32_t *__restrict__ ids,
                                         float *__restrict__ probs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2);
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n_vocab;
         i += item_ct1.get_local_range(2)) s += sycl::native::exp(l[i] - m);
    /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), s,
            o);
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = s;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < (int)(item_ct1.get_local_range(2) >> 5); ++w) tot +=
            part[w];
        probs[t] = 1.0f / tot;
    }
}

__dpct_inline__ void
mtp_select_kernel(const float *__restrict__ R_src, int64_t stride,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ row_dev,
                  float *__restrict__ R_dst, int32_t *__restrict__ tok_dst,
                  int32_t *out, int j, const float *probs, float *out_p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = *row_dev;
#pragma unroll
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < stride; i += (int64_t)item_ct1.get_group_range(2) *
                          item_ct1.get_local_range(2))
        R_dst[i] = R_src[(size_t) row * stride + i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        /*
        DPCT1078: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::system);
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
    }
}

__dpct_inline__ void dense_steps_kernel(const int32_t *__restrict__ cells,
                                        int n, int32_t *__restrict__ steps) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

// --pipeline-windows: the drafter's chain, teacher forced (see force_token)
__dpct_inline__ void force_token_kernel(int32_t *tok, const int32_t *force,
                                        int j) {
    const int32_t f = ((const volatile int32_t*) force)[j];
    if (f >= 0) *tok = f;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto blob_bytes_ct3 = (long long)(blob_bytes / 16);

                cgh.parallel_for<
                    dpct_kernel_name<class fetch_blobs_kernel_14b6f5>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        fetch_blobs_kernel(src, n, (sycl::uint4 *)dst,
                                           blob_bytes_ct3);
                    });
            });
    }
    check("fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class rebase_ptrs_kernel_436f27>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    rebase_ptrs_kernel(ptr, n, (unsigned long long)base,
                                       (long long)blob_bytes);
                });
    }
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class add_streams_broadcast_kernel_7de8f5>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok,
                                (unsigned)((n_embd * hc + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    add_streams_broadcast_kernel(h, e, R, n_embd, hc);
                });
    }
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class ident_hits_kernel_287489>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1024),
                                  sycl::range(1, 1, 1024)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    ident_hits_kernel(ids, n, slot, dst, count);
                });
    }
    check("ident_hits");
}

namespace {
__dpct_inline__ void
copy_row_to_first_kernel(const int32_t *__restrict__ row_dev, float *a,
                         int64_t a_n, float *b, int64_t b_n, float *c,
                         int64_t c_n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t row = *row_dev;
    if (row == 0) return;
    const int64_t i0 = (int64_t)item_ct1.get_group(2) *
                           item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2),
                  st = (int64_t)item_ct1.get_group_range(2) *
                       item_ct1.get_local_range(2);
#pragma unroll
    for (int64_t i = i0; i < a_n; i += st) a[i] = a[row * a_n + i];
#pragma unroll
    for (int64_t i = i0; i < b_n; i += st) b[i] = b[row * b_n + i];
#pragma unroll
    for (int64_t i = i0; i < c_n; i += st) c[i] = c[row * c_n + i];
}
}  // namespace

void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n,
                       void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class copy_row_to_first_kernel_993785>>(
                sycl::nd_range<3>(sycl::range(1, 1, 16) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_row_to_first_kernel(row_dev, a, a_n, b, b_n, c, c_n);
                });
    }
    check("copy_row_to_first");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class mtp_select_kernel_4611fe>>(
                sycl::nd_range<3>(sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    mtp_select_kernel(R_src, R_stride, ids, row_dev, R_dst,
                                      tok_dst, out, j, probs, out_p);
                });
    }
    check("mtp_select");
}

void force_token(int32_t* tok, const int32_t* force, int j, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class force_token_kernel_a9fa72>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    force_token_kernel(tok, force, j);
                });
    }
    check("force_token");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    dpct::queue_ptr s = strata::q_of(stream);
    if (row_bytes % 16 == 0)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            long long row_bytes_ct1 = row_bytes / 16;

            cgh.parallel_for<
                dpct_kernel_name<class gather_rows_kernel_b3ae5a, sycl::uint4>>(
                sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    gather_rows_kernel((const sycl::uint4 *)src, row_bytes_ct1,
                                       ids, n, (sycl::uint4 *)dst);
                });
        });
    } else if (row_bytes % 4 == 0)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            long long row_bytes_ct1 = row_bytes / 4;

            cgh.parallel_for<
                dpct_kernel_name<class gather_rows_kernel_f256ac, uint32_t>>(
                sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    gather_rows_kernel((const uint32_t *)src, row_bytes_ct1,
                                       ids, n, (uint32_t *)dst);
                });
        });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class gather_rows_kernel_c5b171, uint8_t>>(
            sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                gather_rows_kernel(src, row_bytes, ids, n, dst);
            });
    }
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class map_ids_kernel_43a113>>(
                sycl::nd_range<3>(sycl::range(1, 1, 64), sycl::range(1, 1, 64)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    map_ids_kernel(ids, table, n);
                });
    }
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class row_top_prob_kernel_6adbe2>>(
                sycl::nd_range<3>(sycl::range(1, 1, n_rows) *
                                      sycl::range(1, 1, 1024),
                                  sycl::range(1, 1, 1024)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        row_top_prob_kernel(logits, n_vocab, ids, probs);
                    });
    }
    check("row_top_prob");
}

#if !defined(__HIPCC__)   // warp-32 kernels: an AMD card keeps the one-block row_top_prob and sampler_greedy_kernel
namespace {

constexpr int kArgMaxBlocks = 128, kArgThreads = 256, kTopBlocks = 8;

// the larger value, on equality the lower index: sampler_greedy_kernel's order
__dpct_inline__ void arg_take(float ov, int oi, float &bv, int &bi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// a block's (value, index) pick, in thread 0; `none` is the index of no candidate
__dpct_inline__ void arg_block(float &bv, int &bi, int none) {
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &si = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int off = 16; off > 0; off >>= 1) {
        /*
        DPCT1108: '__shfl_down_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        /*
        DPCT1121: Make sure that the "bv" which is used in the SYCL group
        function/algorithm is initialized.
        */
        const float ov = dpct::experimental::shift_sub_group_left(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bv,
            off);
        /*
        DPCT1108: '__shfl_down_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        /*
        DPCT1121: Make sure that the "bi" which is used in the SYCL group
        function/algorithm is initialized.
        */
        const int oi = dpct::experimental::shift_sub_group_left(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bi,
            off);
        arg_take(ov, oi, bv, bi);
    }
    const int warp = (int)(item_ct1.get_local_id(2) >> 5),
              lane = (int)(item_ct1.get_local_id(2) & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) {
        const int nw = (int)(item_ct1.get_local_range(2) >> 5);
        bv = lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            /*
            DPCT1121: Make sure that the "bv" which is used in the SYCL
            group function/algorithm is initialized.
            */
            const float ov = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off);
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            /*
            DPCT1121: Make sure that the "bi" which is used in the SYCL
            group function/algorithm is initialized.
            */
            const int oi = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bi, off);
            arg_take(ov, oi, bv, bi);
        }
    }
}

// A row's scratch, whatever the launch's row count (one scratch serves windows of every size): its counter, then
// the blocks' values and indices.
struct ArgRow {
    unsigned counter;
    unsigned pad[3];
    float v[kArgMaxBlocks];
    int i[kArgMaxBlocks];
};

// grid (blocks, rows): block b scans [b * per_block, +per_block) of its row
__dpct_inline__ void argmax_rows_kernel(const float *__restrict__ logits, int n,
                                        int per_block,
                                        ArgRow *__restrict__ rows,
                                        int32_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int)item_ct1.get_group(1), b = (int)item_ct1.get_group(2),
              nb = (int)item_ct1.get_group_range(2);
    unsigned* counter = &rows[row].counter;
    float* pv = rows[row].v;
    int* pi = rows[row].i;
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = sycl::min(n, lo + per_block);
    float bv = sycl::bit_cast<float, int>(0xff800000); // -inf
    int bi = n;
    for (int v = lo + (int)item_ct1.get_local_id(2); v < hi;
         v += (int)item_ct1.get_local_range(2)) {
        const float s = l[v];
        if (s > bv) { bv = s; bi = v; }
    }
    arg_block(bv, bi, n);
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<bool>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) {
        pv[b] = bv;
        pi[b] = bi;
        /*
        DPCT1078: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        last =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                counter, 1u) == (unsigned)(nb - 1);
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    bv = sycl::bit_cast<float, int>(0xff800000);
    bi = n;
    if ((int)item_ct1.get_local_id(2) < nb) {
        bv = ((volatile const float *)pv)[item_ct1.get_local_id(2)];
        bi = ((volatile const int *)pi)[item_ct1.get_local_id(2)];
    }
    arg_block(bv, bi, n);
    if (item_ct1.get_local_id(2) == 0) {
        out[row] = bi < n ? bi : 0;   // no value above -inf: 0, as the one-block kernel
        *counter = 0;
    }
}

// a row's top-probability scratch: its counter, then its 32 warps' sums
struct TopRow {
    unsigned counter;
    unsigned pad[3];
    float parts[32];
};

// grid (kTopBlocks, rows), 1024 / kTopBlocks threads: thread j of block b is row_top_prob's thread b * blockDim + j
__dpct_inline__ void row_top_prob_split_kernel(const float *__restrict__ logits,
                                               int n_vocab,
                                               const int32_t *__restrict__ ids,
                                               float *__restrict__ probs,
                                               TopRow *__restrict__ rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int)item_ct1.get_group(1);
    float* parts = rows[row].parts;
    const float* l = logits + (size_t) row * n_vocab;
    const float m = l[ids[row]];
    const int vt = (int)(item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2));
    float s = 0.0f;
#pragma unroll
    for (int i = vt; i < n_vocab; i += 1024) s += sycl::native::exp(l[i] - m);
    /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), s,
            o);
    if ((item_ct1.get_local_id(2) & 31) == 0) parts[vt >> 5] = s;
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<bool>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &rows[row].counter, 1u) ==
        (unsigned)(item_ct1.get_group_range(2) - 1);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last || item_ct1.get_local_id(2) != 0) return;
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 32; ++w) tot += ((volatile const float *)parts)[w];
    probs[row] = 1.0f / tot;
    rows[row].counter = 0;
}

}  // namespace

uint64_t argmax_rows_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(ArgRow); }

void argmax_rows(const float* logits, int n_rows, int n, void* scratch, int32_t* out, void* stream) {
    if (n_rows <= 0) return;
    const int nb = std::min(kArgMaxBlocks, std::max(1, (n + 4095) / 4096));
    const int per_block = (n + nb - 1) / nb;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class argmax_rows_kernel_d5bcf6>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_rows, (unsigned)nb) *
                        sycl::range(1, 1, kArgThreads),
                    sycl::range(1, 1, kArgThreads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        argmax_rows_kernel(logits, n, per_block,
                                           (ArgRow *)scratch, out);
                    });
    }
    check("argmax_rows");
}

uint64_t row_top_prob_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(TopRow); }

void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* scratch,
                        void* stream) {
    if (n_rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class row_top_prob_split_kernel_9495b0>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_rows, kTopBlocks) *
                                      sycl::range(1, 1, 1024 / kTopBlocks),
                                  sycl::range(1, 1, 1024 / kTopBlocks)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        row_top_prob_split_kernel(logits, n_vocab, ids, probs,
                                                  (TopRow *)scratch);
                    });
    }
    check("row_top_prob_split");
}
#else
uint64_t argmax_rows_scratch_bytes(int) { return 0; }
void argmax_rows(const float*, int, int, void*, int32_t*, void*) {
    std::fprintf(stderr, "argmax_rows: not built for this backend\n");
    std::exit(1);
}
uint64_t row_top_prob_scratch_bytes(int) { return 0; }
void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void*,
                        void* stream) {
    row_top_prob(logits, n_rows, n_vocab, ids, probs, stream);
}
#endif

bool multi_block_head_ops() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MULTI_BLOCK_ARGMAX");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

bool argmax_rows_wanted() try {
#if defined(__HIPCC__)
    return false;
#else
    // sm_80 to sm_89: sample_tokens' greedy pick there is the one-block kernel (sm_90+ has the cluster kernel)
    static const bool want = [] {
        int dev = 0, major = 0;
        if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 ||
            DPCT_CHECK_ERROR(
                major = dpct::get_device(dev).get_major_version()) != 0)
            return false;
        return major == 8;
    }();
    return want && multi_block_head_ops();
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

namespace {
__dpct_inline__ void window_ids_kernel(int32_t *steps, int window, int32_t *ids,
                                       long long stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int q = item_ct1.get_group(1);
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
#pragma unroll
    for (int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         j < width;
         j += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        ids[q * stride + j] = start + j;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) st[3] =
        width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class window_ids_kernel_581536>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n, 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    window_ids_kernel(steps, window, ids,
                                      (long long)ids_stride);
                });
    }
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class dense_steps_kernel_fd7245>>(
                sycl::nd_range<3>(sycl::range(1, 1, 64), sycl::range(1, 1, 64)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    dense_steps_kernel(cells, n, steps);
                });
    }
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin, bool commit) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT ||
        (commit && (n_tok != 1 || t_begin != 0))) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class gdn_conv_l2_multi_kernel_b3f737>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok, (unsigned)(channels / S)) *
                        sycl::range(1, 1, S),
                    sycl::range(1, 1, S)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_conv_l2_multi_kernel(
                            const_cast<float *>(history), qkv, conv_w, h,
                            channels, qk_heads, eps, t_begin, commit);
                    });
    }
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class gdn_conv_commit_kernel_dc1d7f>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((channels + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gdn_conv_commit_kernel(history, qkv, channels, n_keep);
                });
    }
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const unsigned blocks = (unsigned) ((2 * h_v + 1) / 2);
    dpct::queue_ptr st = strata::q_of(stream);
    switch (n_tok) {
    case 1: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<1>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<1, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 2: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<2>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<2, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 3: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<3>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<3, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 4: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<4>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<4, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 5: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<5>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<5, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    case 6: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                             dpct_kernel_scalar<6>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<6, true>(x, w_alpha, w_beta, dt, ssm_a,
                                             gate, beta, n_embd, h_v, n_tok);
            });
    } break;
    default: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gdn_ab_multi_kernel_34dcce,
                                          dpct_kernel_scalar<kVerifyMaxT>,
                                          dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel<kVerifyMaxT, false>(x, w_alpha, w_beta, dt,
                                                        ssm_a, gate, beta,
                                                        n_embd, h_v, n_tok);
            });
    } break;
    }
    check("gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin, void* xq_out) {
    GdnQ81* const xq = static_cast<GdnQ81*>(xq_out);   // S26: non-null = also the outputs' q8_1 (STRATA_QFUSE)
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    static const bool split = [] { const char* v = std::getenv("STRATA_GDN_SPLIT"); return v && v[0] == '1'; }();
    // S26: the split runs where it is faster (gfx1151 harness, h_v 48: the commit 8-19% per call; a window of 2+ rows
    // 3-13%; one row: no gain, the old kernel stays) - bitwise equal in every case
    if (split && (t_out_begin >= n_tok || n_tok - t_out_begin >= 2)) {
        const dpct::dim3 grid((unsigned)h_v, S / GS_COLS), block(GS_COLS, RG);
        if (t_out_begin >= n_tok) {   // no outputs (the commit): the state only
                                      /*
                                      DPCT1049: The work-group size passed to the SYCL kernel may
                                      exceed the limit. To get the device limit, query
                                      info::device::max_work_group_size. Adjust the work-group size if
                                      needed.
                                      */
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class gdn_state_replay_kernel_e4d4c4,
                                     dpct_kernel_scalar<kVerifyMaxT>>>(
                    sycl::nd_range<3>(grid * block, block), exp_props,
                    [=](sycl::nd_item<3> item_ct1) {
                        gdn_state_replay_kernel<kVerifyMaxT>(
                            state, h, conv_channels, gate, beta, h_k, h_v,
                            n_tok, n_keep);
                    });
        } else {
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                strata::q_of(stream)
                    ->parallel_for<
                        dpct_kernel_name<class gdn_step_split_kernel_83c676,
                                         dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(grid * block, block), exp_props,
                        [=](sycl::nd_item<3> item_ct1) {
                            gdn_step_split_kernel<true>(
                                state, h, conv_channels, gate, beta, y, h_k,
                                h_v, n_tok, n_keep, t_out_begin);
                        });
            }
            const dpct::dim3 ng((unsigned)h_v, (unsigned)(n_tok - t_out_begin));
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            if (xq) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                strata::q_of(stream)
                    ->parallel_for<
                        dpct_kernel_name<class gdn_out_norm_kernel_a650eb,
                                         dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(ng * sycl::range(1, RG, S),
                                          sycl::range(1, RG, S)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gdn_out_norm_kernel<true>(z, gamma, eps, y, h_v,
                                                          n_tok, n_keep,
                                                          t_out_begin, xq);
                            });
            }
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            else {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                strata::q_of(stream)
                    ->parallel_for<
                        dpct_kernel_name<class gdn_out_norm_kernel_f6dfab,
                                         dpct_kernel_scalar<false>>>(
                        sycl::nd_range<3>(ng * sycl::range(1, RG, S),
                                          sycl::range(1, RG, S)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gdn_out_norm_kernel<false>(
                                    z, gamma, eps, y, h_v, n_tok, n_keep,
                                    t_out_begin, nullptr);
                            });
            }
        }
        check("gdn_step_norm_multi (split)");
        return;
    }
    static const bool commit_split = [] {
        const char* e = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        return !e || e[0] != '0';
    }();
    if (commit_split && n_keep != nullptr && t_out_begin >= n_tok) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class gdn_step_commit_kernel_5d9da7>>(
                    sycl::nd_range<3>(sycl::range(1, 4u, (unsigned)h_v) *
                                          sycl::range(1, RG, 32),
                                      sycl::range(1, RG, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        gdn_step_commit_kernel(state, h, conv_channels, gate,
                                               beta, h_k, h_v, n_keep);
                    });
        }
        check("gdn_step_commit");
        return;
    }
    const dpct::dim3 g1((unsigned)h_v), b1(S, RG);
    const bool all_out = n_keep == nullptr && t_out_begin <= 0;
    if (xq && t_out_begin < n_tok) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        if (all_out) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<dpct_kernel_name<
                    class gdn_step_norm_multi_kernel_42ae26,
                    dpct_kernel_scalar<true>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(g1 * b1, b1), exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gdn_step_norm_multi_kernel<true, true>(
                                state, h, conv_channels, gate, beta, z, gamma,
                                eps, y, h_k, h_v, n_tok, nullptr, 0, xq);
                        });
        }
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<dpct_kernel_name<
                    class gdn_step_norm_multi_kernel_42ae26,
                    dpct_kernel_scalar<false>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(g1 * b1, b1), exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gdn_step_norm_multi_kernel<false, true>(
                                state, h, conv_channels, gate, beta, z, gamma,
                                eps, y, h_k, h_v, n_tok, n_keep, t_out_begin,
                                xq);
                        });
        }
    } else if (all_out) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<
                class gdn_step_norm_multi_kernel_42ae26,
                dpct_kernel_scalar<true>, dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(g1 * b1, b1), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_multi_kernel<true, false>(
                            state, h, conv_channels, gate, beta, z, gamma, eps,
                            y, h_k, h_v, n_tok, nullptr, 0, nullptr);
                    });
    } else {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<
                class gdn_step_norm_multi_kernel_42ae26,
                dpct_kernel_scalar<false>, dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(g1 * b1, b1), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_multi_kernel<false, false>(
                            state, h, conv_channels, gate, beta, z, gamma, eps,
                            y, h_k, h_v, n_tok, n_keep, t_out_begin, nullptr);
                    });
    }
    check("gdn_step_norm_multi");
}

namespace {
__dpct_inline__ void wait_flag_ge_kernel(const volatile uint32_t *flag,
                                         uint32_t value, uint32_t spin_max) {
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
}  // namespace

namespace {
// one thread per window entry: up to kVerifyMaxT tokens x 10 routed experts (80), so 128 (#646 had 64: a window of
// 7+ tokens lost its last entries)
constexpr int kResidentPlanMax = 128;
static_assert(kVerifyMaxT * 10 <= kResidentPlanMax, "resident_plan: one thread per entry");
// the parallel scan keeps one partial sum per warp in s_wsum[4] and packs (entries << 16 | groups) in an int
static_assert(kResidentPlanMax <= 128 && kResidentPlanMax % 32 == 0, "resident_plan: at most 4 warps");
static_assert(kResidentPlanMax < 32768, "resident_plan: the packed entry count must stay below 2^15");
/*
DPCT1110: The total declared local variable size in device function
resident_plan_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
resident_plan_kernel(const int32_t *__restrict__ ids, int n, int k,
                     const int32_t *__restrict__ res, int n_expert,
                     const uint8_t *cache_base,
                     const unsigned long long *slot_off, long long blob,
                     int32_t *__restrict__ pl, long long capx, uint32_t *skip,
                     uint32_t ring, const unsigned long long *__restrict__ mir,
                     volatile uint32_t *plan_err, int32_t *usage) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_ids = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    int32_t[kResidentPlanMax]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_excl = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int32_t[kResidentPlanMax]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_wsum =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[4]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_bad = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int tid = item_ct1.get_local_id(2);
    if (tid == 0) s_bad = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    int32_t eid = -1;
    int32_t slot = -1;
    unsigned long long maddr = 0;   // SYCL port: a host-mirrored expert (not in VRAM): its pinned mirror, read over PCIe
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0 && eid >= 0 && eid < n_expert && mir != nullptr) maddr = mir[eid];
        if (usage != nullptr && eid >= 0 && eid < n_expert)   // the adaptive tier's routing counts (device side)
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(usage[eid]).fetch_add(1);
        if (slot < 0 && maddr == 0)
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                &s_bad, 1);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (s_bad) {
        if (tid == 0 && skip != nullptr) *skip = 0;
        if (tid == 0 && skip == nullptr) {   // #871: the all-resident graph has no host plan to fall back on
            pl[0] = 0; pl[1] = 0; pl[2] = 0;   // an empty plan: no expert runs on a stale pointer
            /*
            DPCT1078: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            if (plan_err != nullptr) {
                    *plan_err = 1;
                    sycl::atomic_fence(sycl::memory_order::acq_rel,
                                       sycl::memory_scope::system);
            }
        }
        return;
    }

    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
#pragma unroll
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
    }
    const bool is_first = (tid < n && first_j == tid && (slot >= 0 || maddr != 0));
    const int my_cnt = is_first ? count_same : 0;
    const int my_pack = (my_cnt << 16) | (is_first ? 1 : 0);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int pref = my_pack;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        /*
        DPCT1108: '__shfl_up_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        const int up = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            pref, d);
        if (lane >= d) pref += up;
    }
    s_excl[tid] = pref - my_pack;
    if (lane == 31) {
        s_wsum[warp] = pref;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (is_first) {
        int tot = s_excl[tid];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < warp) tot += s_wsum[w];
        }
        const int grp_idx = tot & 0xffff;
        const int ent_start = tot >> 16;
        ptr[grp_idx] = slot >= 0 ? (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob))
                                 : maddr;
        start[grp_idx] = ent_start;
    }
    if (tid < n) {
        const int fj_warp = first_j >> 5;
        int fj_tot = s_excl[first_j];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < fj_warp) fj_tot += s_wsum[w];
        }
        const int out_idx = (fj_tot >> 16) + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        const int groups = (s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3]) & 0xffff;
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
        if (skip != nullptr) {
            /*
            DPCT1078: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
            strata::sys_store(skip, ring);
        }
    }
}
// The same plan in one block of 128 threads (n <= 128): thread i owns entry i. Groups are the distinct experts in
// order of first occurrence; group g's entries are its occurrences in increasing i - exactly the loop above
// (S26: the one-thread loop took ~44 us per call on gfx1151, 48 per window).
void resident_plan_par_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                         int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                         long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                         uint32_t ring) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_id = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[128]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_first =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_size =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_bad = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int i = item_ct1.get_local_id(2);
    if (i == 0) s_bad = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int32_t e = -1;
    if (i < n) {
        e = ids[i];
        s_id[i] = e;
        if (e < 0 || e >= n_expert || res[e] < 0) s_bad = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (s_bad) { if (i == 0) *skip = 0; return; }
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < n; ++j) {
            const bool same = s_id[j] == e;
            if (same && j < first) first = j;
            if (same && j < i) ++rank;
            size += same;
        }
    }
    if (i < n) { s_first[i] = first; s_size[i] = size; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    if (i < n) {
        // group of entry i = the number of first occurrences before `first`; its start = their sizes' sum
        int group = 0, gstart = 0;
#pragma unroll
        for (int j = 0; j < first; ++j)
            if (s_first[j] == j) { ++group; gstart += s_size[j]; }
        if (i == first) {
            const int32_t slot = res[e];
            ptr[group] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
            start[group] = gstart;
        }
        dst[gstart + rank] = i;
        tok[gstart + rank] = i / k;
        if (i == 0) {
            int groups = 0;
#pragma unroll
            for (int j = 0; j < n; ++j) groups += s_first[j] == j;
            start[groups] = n;
            start2[0] = n;
            counts[0] = groups;
            counts[1] = n;
            counts[2] = 0;
        }
    }
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (i == 0) *skip = ring;
}
__dpct_inline__ void wait_flag_ge_or_kernel(const volatile uint32_t *flag,
                                            uint32_t value,
                                            const volatile uint32_t *skip, uint32_t spin_max) {
    if (strata::sys_load(skip) == value) return;
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
__dpct_inline__ void copy_i32_unless_kernel(int32_t *__restrict__ dst,
                                            const volatile int32_t *src, int n,
                                            const uint32_t *skip,
                                            uint32_t value) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (strata::sys_load(skip) == value) return;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) dst[i] = src[i];
}
/*
DPCT1052: SYCL does not support the member access for a volatile qualified
vector type. The volatile qualifier was removed. You may need to rewrite the
code.
*/
__dpct_inline__ void copy_or_zero_kernel(sycl::float4 *__restrict__ dst,
                                         const sycl::float4 *src, long long n4,
                                         const uint32_t *skip, uint32_t value, bool pair) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const bool zero = *skip == value;
#pragma unroll
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n4; i += (long long)item_ct1.get_group_range(2) *
                      item_ct1.get_local_range(2))
        dst[i] = zero ? sycl::float4(0.f, 0.f, 0.f, 0.f)
                      : (pair ? strata::load_mapped_float4(src + i) : const_cast<const sycl::float4 *>(src)[i]);
}
}  // namespace

namespace {
const int32_t* g_mirror_res = nullptr;
const unsigned long long* g_mirror_table = nullptr;
int32_t* g_usage = nullptr;
long long g_mirror_entries = 0;   // 0: unchecked
// #1980: the layer slice `res_layer` belongs to the table the mirror was set for (a later stage passes its own)
bool in_mirror_table(const int32_t* res_layer) {
    if (g_mirror_res == nullptr || res_layer < g_mirror_res) return false;
    return g_mirror_entries <= 0 || res_layer - g_mirror_res < g_mirror_entries;
}
}
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table, int32_t* usage, long long entries) {
    g_mirror_entries = entries;
    g_usage = usage;
    g_mirror_res = d_res;
    g_mirror_table = mirror_table;
}
void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err) {
    const unsigned long long* mir = nullptr;   // SYCL port: the layer's slice of the host-mirror table, if any
    int32_t* use = nullptr;                    // the layer's slice of the device routing counts, if on
    if (g_mirror_table != nullptr && in_mirror_table(res_layer)) mir = g_mirror_table + (res_layer - g_mirror_res);
    if (g_usage != nullptr && in_mirror_table(res_layer)) use = g_usage + (res_layer - g_mirror_res);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class resident_plan_kernel_db6b7a>>(
                sycl::nd_range<3>(sycl::range(1, 1, kResidentPlanMax),
                                  sycl::range(1, 1, kResidentPlanMax)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        resident_plan_kernel(
                            ids, n_entries, k, res_layer, n_expert, cache_base,
                            slot_off, blob, plan, capx, skip, ring, mir, plan_err, use);
                    });
    }
    check("resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    const uint32_t spin_max = strata::spin_max(*strata::q_of(stream));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class wait_flag_ge_or_kernel_2b2de3>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    wait_flag_ge_or_kernel(flag, value, skip, spin_max);
                });
    }
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class copy_i32_unless_kernel_d9d761>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_i32_unless_kernel(dst, (const volatile int32_t *)src,
                                           (int)n, skip, value);
                });
    }
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const bool pair = strata::a770_fast();
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class copy_or_zero_kernel_5ef081>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_or_zero_kernel((sycl::float4 *)dst,
                                        (const sycl::float4 *)src, n4,
                                        skip, value, pair);
                });
    }
    check("copy_or_zero_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    const uint32_t spin_max = strata::spin_max(*strata::q_of(stream));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class wait_flag_ge_kernel_d7debf>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    wait_flag_ge_kernel(flag, value, spin_max);
                });
    }
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class embedding_gather_dev_kernel_600016>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                              (unsigned)((n + 255) / 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    embedding_gather_dev_kernel(
                        codes, scales, offsets, tokens, n, code_bits, code_bias,
                        group_elems, row_codes, row_groups, out);
                });
    }
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class broadcast_streams_kernel_aa4f6f>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok,
                                (unsigned)((n_embd * hc + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    broadcast_streams_kernel(x, R, n_embd, hc);
                });
    }
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class copy_indexed_kernel_b12e23>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_indexed_kernel(dst, src, stride, index, n);
                });
    }
    check("copy_indexed");
}

// Opt-in GPU stage timestamps. DG2 has no comparable device-wide clock, so
// read a live host USM clock uncached (the fork's gpu_stamp implementation).
// The writer is joined before runtime teardown and exists only in profile mode.
namespace {
uint64_t profile_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct ProfileHostClock {
    sycl::context context;
    uint64_t* word;
    std::atomic<bool> stop{false};
    std::thread writer;
    explicit ProfileHostClock(const sycl::queue& q) : context(q.get_context()),
        word(sycl::malloc_host<uint64_t>(8, context)) {
        if (!word) throw std::runtime_error("gpu_stamp: host clock allocation failed");
        __atomic_store_n(word, profile_now_ns(), __ATOMIC_RELEASE);
        writer = std::thread([this] {
            while (!stop.load(std::memory_order_relaxed)) {
                __atomic_store_n(word, profile_now_ns(), __ATOMIC_RELEASE);
                for (int i = 0; i < 8; ++i) __builtin_ia32_pause();
            }
        });
    }
    ~ProfileHostClock() {
        stop.store(true, std::memory_order_relaxed);
        if (writer.joinable()) writer.join();
        // Keep the word alive until context teardown, as in the fork: a graph
        // may still retain its pointer when process-level statics are destroyed.
    }
};
uint64_t* profile_clock_word(const sycl::queue& q) {
    static std::mutex mu;
    static std::vector<std::unique_ptr<ProfileHostClock>> clocks;
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& clock : clocks) if (clock->context == q.get_context()) return clock->word;
    clocks.push_back(std::make_unique<ProfileHostClock>(q));
    return clocks.back()->word;
}
}

void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    auto& q = *strata::q_of(stream);
    uint64_t* word = profile_clock_word(q);
    q.single_task([=] {
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        buf[i] = strata::sys_load_mapped(word);
    });
}

}  // namespace strata::kernels
