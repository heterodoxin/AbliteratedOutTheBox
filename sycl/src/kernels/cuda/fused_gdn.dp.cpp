// src/kernels/cuda/fused_gdn.cu - see include/strata/kernels/fused_gdn.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include "strata/core/geometry.hpp"
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/fused_gdn.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
static_assert(S == strata::core::kCompiledGeometry.ssm_state_size, "fused_gdn is compiled for core/geometry.hpp");
constexpr int RG = 4;           // row groups
constexpr int RPG = S / RG;     // 32 rows per thread

/*
DPCT1110: The total declared local variable size in device function
gdn_step_norm_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_step_norm_kernel(
    float *__restrict__ state, const float *__restrict__ q,
    const float *__restrict__ k, const float *__restrict__ v,
    const float *__restrict__ gate, const float *__restrict__ beta,
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_k, int h_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_kv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_o =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wsum = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[S * RG / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col = item_ct1.get_local_id(2); // 0..127
    const int rg = item_ct1.get_local_id(1);  // 0..3
    const int tid = rg * S + col;
    const int qh = head % h_k;
    if (tid < S) { sk[tid] = k[qh * S + tid]; sq[tid] = q[qh * S + tid]; }
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float g = sycl::native::exp(gate[head]);
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
    red_kv[rg][col] = kv;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
    const float delta = (v[head * S + col] - g * kv_col) * beta[head];
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
        o = sycl::fma(s[r], sq[rg * RPG + r], o);
        base[r * row_stride] = s[r];
    }
    red_o[rg][col] = o;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) *
             sycl::rsqrt((float)S);
        sq_part = oc * oc;
    }
    // RMS over the head's 128 outputs: warps of row group 0 are threads 0..127.
    /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature masked
sub_group function which may not be supported by all compilers or runtimes. You
may need to adjust the code.
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
        const float zz = z[head * S + col];
        y[head * S + col] =
            oc * scale * gamma[col] * (1.0f / (1.0f + sycl::native::exp(-zz)));
    }
}

__dpct_inline__ void gdn_conv_l2_kernel(float *__restrict__ hist,
                                        const float *__restrict__ qkv,
                                        const float *__restrict__ w,
                                        float *__restrict__ h, int qk_heads,
                                        float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int c = item_ct1.get_group(2) * S + item_ct1.get_local_id(2);
    const float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2], x = qkv[c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    hist[c * 3] = v1;
    hist[c * 3 + 1] = v2;
    hist[c * 3 + 2] = x;
    float y = sum / (1.0f + sycl::native::exp(-sum));
    if ((int)item_ct1.get_group(2) < qk_heads) {
        float sq = y * y;
        /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature masked
sub_group function which may not be supported by all compilers or runtimes. You
may need to adjust the code.
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
    h[c] = y;
}

__dpct_inline__ void
gdn_ab_kernel(const float *__restrict__ x, const uint16_t *__restrict__ wa,
              const uint16_t *__restrict__ wb, const float *__restrict__ dt,
              const float *__restrict__ ssm_a, float *__restrict__ gate,
              float *__restrict__ beta, int n, int h_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 8 + (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
        (is_beta ? wb : wa) + (size_t)r * n);
    float acc = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wv = *(w4 + j);
        const sycl::float4 xa =
            *reinterpret_cast<const sycl::float4 *>(x + j * 8);
        const sycl::float4 xb =
            *reinterpret_cast<const sycl::float4 *>(x + j * 8 + 4);
        acc = sycl::fma(sycl::bit_cast<float>(wv.x() << 16), (float)(xa.x()),
                        acc);
            acc = sycl::fma(sycl::bit_cast<float>(wv.x() & 0xffff0000u),
                            (float)(xa.y()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.y() << 16), (float)(xa.z()),
                        acc);
            acc = sycl::fma(sycl::bit_cast<float>(wv.y() & 0xffff0000u),
                            (float)(xa.w()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.z() << 16), (float)(xb.x()),
                        acc);
            acc = sycl::fma(sycl::bit_cast<float>(wv.z() & 0xffff0000u),
                            (float)(xb.y()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.w() << 16), (float)(xb.z()),
                        acc);
            acc = sycl::fma(sycl::bit_cast<float>(wv.w() & 0xffff0000u),
                            (float)(xb.w()), acc);
    }
    /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature masked
sub_group function which may not be supported by all compilers or runtimes. You
may need to adjust the code.
*/
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            acc, o);
    if (lane != 0) return;
    if (is_beta) {
        beta[r] = 1.0f / (1.0f + sycl::native::exp(-acc));
    } else {
        const float v = acc + dt[r];
        const float sp = v > 20.0f ? v : sycl::log1p(sycl::native::exp(v));
        gate[r] = sp * ssm_a[r];
    }
}

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class gdn_conv_l2_kernel_44cd0c>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)(channels / S)) *
                                      sycl::range(1, 1, S),
                                  sycl::range(1, 1, S)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_conv_l2_kernel(history, qkv, conv_w, h, qk_heads,
                                           eps);
                    });
    }
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

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class gdn_ab_kernel_c211b9>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((2 * h_v + 7) / 8)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_ab_kernel(x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                                      n_embd, h_v);
                    });
    }
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

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        std::exit(1);
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class gdn_step_norm_kernel_51bed3>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)h_v) *
                                      sycl::range(1, RG, S),
                                  sycl::range(1, RG, S)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_kernel(state, q, k, v, gate, beta, z,
                                             gamma, eps, y, h_k, h_v);
                    });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

}  // namespace strata::kernels
