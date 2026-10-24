#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "s26_tsum.dp.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <cmath>

namespace strata::kernels {
namespace {

// The FP32-activation MMVF implementation below is adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/{mmvf.cu,common.cuh}.
// Scope: ordinary contiguous BF16 matrix, one FP32 activation vector, no fusion/ids/channels.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
__dpct_inline__ float mmvf_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        value += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            value, offset);
    return value;
}

template <int BLOCK_SIZE>
__dpct_inline__ void bf16_f32_mmvf_kernel(const float *__restrict__ x,
                                          const uint16_t *__restrict__ w,
                                          float *__restrict__ y, int n_in) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2);
    const uint16_t *row = w + (size_t)item_ct1.get_group(2) * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const sycl::float2 *inputs2 = reinterpret_cast<const sycl::float2 *>(x);
    auto &partials =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (t < 32) partials[t] = 0.0f;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const sycl::float2 input = inputs2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        acc = sycl::fma(f32_from_bf16((uint16_t)weight), input.x(), acc);
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        acc =
            sycl::fma(f32_from_bf16((uint16_t)(weight >> 16)), input.y(), acc);
    }
    acc = mmvf_warp_sum(acc);
    if constexpr (BLOCK_SIZE > 32) {
        // All lanes have the same reduced value; one store avoids a same-value shared-memory race.
        if ((t & 31) == 0) partials[t / 32] = acc;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (t < 32) acc = mmvf_warp_sum(partials[t]);
    }
    if (t == 0) y[item_ct1.get_group(2)] = acc;
}

// the same kernel for up to 8 activation rows - the weight row is read ONCE and every
// token keeps its own accumulator with exactly the single-row kernel's order (pairs, two ordered FMAs, the same warp
// and block reductions), so each output is bit-identical to a bf16_f32_mmvf_kernel launch of its own.
template <int BLOCK_SIZE, int NT, bool EXACT_T = true>
__dpct_inline__ void bf16_f32_mmvf_multi_kernel(
    const float *__restrict__ x, int64_t ldx, const uint16_t *__restrict__ w,
    float *__restrict__ y, int64_t ldy, int n_in, int n_tok) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2);
    const uint16_t *row = w + (size_t)item_ct1.get_group(2) * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    auto &partials =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[NT][32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (t < 32) {
#pragma unroll
        for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const uint32_t weight = *(weights2 + pair);
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (EXACT_T || k < n_tok) {
                /*
                DPCT1098: The '*' expression is used instead of the __ldg
                call. These two expressions do not provide the exact same
                functionality. Check the generated code for potential precision
                and/or performance issues.
                */
                const sycl::float2 input =
                    *(reinterpret_cast<const sycl::float2 *>(x +
                                                             (size_t)k * ldx) +
                      pair);
                /*
                DPCT1013: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                acc[k] = sycl::fma(w0, input.x(), acc[k]);
                /*
                DPCT1013: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                acc[k] = sycl::fma(w1, input.y(), acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k)
        if (EXACT_T || k < n_tok) acc[k] = mmvf_warp_sum(acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (EXACT_T || k < n_tok) partials[k][t / 32] = acc[k];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (EXACT_T || k < n_tok) acc[k] = mmvf_warp_sum(partials[k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (EXACT_T || k < n_tok)
                y[(size_t)k * ldy + item_ct1.get_group(2)] = acc[k];
}

// S25 (STRATA_MMVF_ROWS=1): RPB output rows per block. Each block read its row's weights once but all T activation
// rows again (n_out blocks x T x n_in floats through L2 - the part that grew with T); here a block reads the
// activations once for RPB rows. Per output, thread t still walks pairs t, t + BLOCK_SIZE, ... with the same two
// ordered FMAs and the same warp and block reductions: bit-identical to bf16_f32_mmvf_multi_kernel.
// S26 STRATA_LFUSE=1 (AUX): one more block computes row 0 of w_aux into y_aux[k * ldy_aux] - the same per-output
// code, so each aux output is bitwise what its own bf16_f32_mmvf_multi_kernel launch gave (the shared expert's gate)
template <int BLOCK_SIZE, int NT, int RPB, bool TS = false, bool AUX = false>
/*
DPCT1110: The total declared local variable size in device function
bf16_f32_mmvf_rows_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void bf16_f32_mmvf_rows_kernel(
    const float *__restrict__ x, int64_t ldx, const uint16_t *__restrict__ w,
    float *__restrict__ y, int64_t ldy, int n_in, int n_out, int n_tok,
    const uint16_t *__restrict__ w_aux = nullptr,
    float *__restrict__ y_aux = nullptr, int64_t ldy_aux = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2);
    int o0 = item_ct1.get_group(2) * RPB;
    if constexpr (AUX) {
        if (item_ct1.get_group(2) == item_ct1.get_group_range(2) - 1) {
            w = w_aux; y = y_aux; ldy = ldy_aux; n_out = 1; o0 = 0;
        }
    }
    auto &partials = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RPB][NT][32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) partials[r][k][t] = 0.0f;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    float acc[RPB][NT];
#pragma unroll
    for (int r = 0; r < RPB; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        sycl::float2 in[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) in[k] = reinterpret_cast<const sycl::float2 *>(
                x + (size_t)k * ldx)[pair];
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            if (o0 + r >= n_out) break;
            const uint32_t weight = reinterpret_cast<const uint32_t*>(w + (size_t) (o0 + r) * n_in)[pair];
            const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k < n_tok) {
                    /*
                    DPCT1013: The rounding mode could not be specified and
                    the generated code may have different accuracy than the
                    original code. Verify the correctness. SYCL math built-in
                    function rounding mode is aligned with OpenCL C 1.2
                    standard.
                    */
                    acc[r][k] = sycl::fma(w0, in[k].x(), acc[r][k]);
                    /*
                    DPCT1013: The rounding mode could not be specified and
                    the generated code may have different accuracy than the
                    original code. Verify the correctness. SYCL math built-in
                    function rounding mode is aligned with OpenCL C 1.2
                    standard.
                    */
                    acc[r][k] = sycl::fma(w1, in[k].y(), acc[r][k]);
                }
            }
        }
    }
    if constexpr (TS) {   // S26 STRATA_TSUM=1: the RPB x NT warp sums (both stages) as transposed butterflies, bitwise the same
        constexpr int V = RPB * NT, P = s26ts::pow2_ceil(V);
        const int lane = t & 31, j = s26ts::tsum_token<P>(lane);
        const bool own = lane == s26ts::tsum_lane<P>(j) && j < V;
        float v[P];
#pragma unroll
        for (int q = 0; q < P; ++q) v[q] = q < V ? acc[q / NT][q % NT] : 0.0f;
        float sum = s26ts::tsum<P>(v, lane);
        if constexpr (BLOCK_SIZE > 32) {
            if (own) partials[j / NT][j % NT][t / 32] = sum;
            item_ct1.barrier(sycl::access::fence_space::local_space);
            if (t >= 32) return;
#pragma unroll
            for (int q = 0; q < P; ++q) v[q] = q < V ? partials[q / NT][q % NT][t] : 0.0f;
            sum = s26ts::tsum<P>(v, lane);
        }
        const int r = j / NT, k = j % NT;
        if (own && o0 + r < n_out && k < n_tok) y[(size_t) k * ldy + o0 + r] = sum;
        return;
    }
#pragma unroll
    for (int r = 0; r < RPB; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = mmvf_warp_sum(acc[r][k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) partials[r][k][t / 32] = acc[r][k];
        /*
        DPCT1064: Migrated __syncthreads call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (t < 32)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) acc[r][k] = mmvf_warp_sum(partials[r][k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int r = 0; r < RPB; ++r)
            if (o0 + r < n_out)
#pragma unroll
                for (int k = 0; k < NT; ++k)
                    if (k < n_tok) y[(size_t) k * ldy + o0 + r] = acc[r][k];
}

template <int B>
void launch_rows(const float *x, int64_t ldx, const uint16_t *w, float *y,
                 int64_t ldy, int n_in, int n_out, int n_tok,
                 dpct::queue_ptr st, const uint16_t *w_aux = nullptr,
                 float *y_aux = nullptr, int64_t ldy_aux = 0) {
    constexpr int RPB = 4;
    const unsigned nb = (unsigned) ((n_out + RPB - 1) / RPB);
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (w_aux != nullptr) {   // S26 STRATA_LFUSE: + one aux row block
/*
DPCT1049: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size. Adjust
the work-group size if needed.
*/
/*
DPCT1049: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size. Adjust
the work-group size if needed.
*/
#define S26_AUX(TSV)                                                           \
    if (n_tok <= 4) {                                                          \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto x_ct0 = x;                                                    \
            auto ldx_ct1 = ldx;                                                \
            auto w_ct2 = w;                                                    \
            auto y_ct3 = y;                                                    \
            auto ldy_ct4 = ldy;                                                \
            auto n_in_ct5 = n_in;                                              \
            auto n_out_ct6 = n_out;                                            \
            auto n_tok_ct7 = n_tok;                                            \
            auto w_aux_ct8 = w_aux;                                            \
            auto y_aux_ct9 = y_aux;                                            \
            auto ldy_aux_ct10 = ldy_aux;                                       \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<                                 \
                class bf16_f32_mmvf_rows_kernel_7c05e5, dpct_kernel_scalar<B>, \
                dpct_kernel_scalar<4>, dpct_kernel_scalar<RPB>,                \
                dpct_kernel_scalar<TSV>, dpct_kernel_scalar<true>>>(           \
                sycl::nd_range<3>(sycl::range(1, 1, nb + 1) *                  \
                                      sycl::range(1, 1, B),                    \
                                  sycl::range(1, 1, B)),                       \
                exp_props,                                                     \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        bf16_f32_mmvf_rows_kernel<B, 4, RPB, TSV, true>(       \
                            x_ct0, ldx_ct1, w_ct2, y_ct3, ldy_ct4, n_in_ct5,   \
                            n_out_ct6, n_tok_ct7, w_aux_ct8, y_aux_ct9,        \
                            ldy_aux_ct10);                                     \
                    });                                                        \
        });                                                                    \
    } else {                                                                   \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto x_ct0 = x;                                                    \
            auto ldx_ct1 = ldx;                                                \
            auto w_ct2 = w;                                                    \
            auto y_ct3 = y;                                                    \
            auto ldy_ct4 = ldy;                                                \
            auto n_in_ct5 = n_in;                                              \
            auto n_out_ct6 = n_out;                                            \
            auto n_tok_ct7 = n_tok;                                            \
            auto w_aux_ct8 = w_aux;                                            \
            auto y_aux_ct9 = y_aux;                                            \
            auto ldy_aux_ct10 = ldy_aux;                                       \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<                                 \
                class bf16_f32_mmvf_rows_kernel_681929, dpct_kernel_scalar<B>, \
                dpct_kernel_scalar<8>, dpct_kernel_scalar<RPB>,                \
                dpct_kernel_scalar<TSV>, dpct_kernel_scalar<true>>>(           \
                sycl::nd_range<3>(sycl::range(1, 1, nb + 1) *                  \
                                      sycl::range(1, 1, B),                    \
                                  sycl::range(1, 1, B)),                       \
                exp_props,                                                     \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        bf16_f32_mmvf_rows_kernel<B, 8, RPB, TSV, true>(       \
                            x_ct0, ldx_ct1, w_ct2, y_ct3, ldy_ct4, n_in_ct5,   \
                            n_out_ct6, n_tok_ct7, w_aux_ct8, y_aux_ct9,        \
                            ldy_aux_ct10);                                     \
                    });                                                        \
        });                                                                    \
    }
        if (ts) { S26_AUX(true) } else { S26_AUX(false) }
#undef S26_AUX
        return;
    }
    if (ts) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        if (n_tok <= 4) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->/*
  DPCT1050: The template argument of the dpct_kernel_name could not be
  deduced. You need to update this code.
  */
                parallel_for<dpct_kernel_name<
                    class bf16_f32_mmvf_rows_kernel_ff3610,
                    dpct_kernel_scalar<B>, dpct_kernel_scalar<4>,
                    dpct_kernel_scalar<RPB>, dpct_kernel_scalar<true>,
                    dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, nb) *
                                          sycl::range(1, 1, B),
                                      sycl::range(1, 1, B)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            bf16_f32_mmvf_rows_kernel<B, 4, RPB, true>(
                                x, ldx, w, y, ldy, n_in, n_out, n_tok);
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

            st->/*
  DPCT1050: The template argument of the dpct_kernel_name could not be
  deduced. You need to update this code.
  */
                parallel_for<dpct_kernel_name<
                    class bf16_f32_mmvf_rows_kernel_1b1bb5,
                    dpct_kernel_scalar<B>, dpct_kernel_scalar<8>,
                    dpct_kernel_scalar<RPB>, dpct_kernel_scalar<true>,
                    dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, nb) *
                                          sycl::range(1, 1, B),
                                      sycl::range(1, 1, B)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            bf16_f32_mmvf_rows_kernel<B, 8, RPB, true>(
                                x, ldx, w, y, ldy, n_in, n_out, n_tok);
                        });
        }
        return;
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    if (n_tok <= 4) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->/*
  DPCT1050: The template argument of the dpct_kernel_name could not be
  deduced. You need to update this code.
  */
            parallel_for<dpct_kernel_name<
                class bf16_f32_mmvf_rows_kernel_7ff80d, dpct_kernel_scalar<B>,
                dpct_kernel_scalar<4>, dpct_kernel_scalar<RPB>,
                dpct_kernel_scalar<false>,
                dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, nb) * sycl::range(1, 1, B),
                                  sycl::range(1, 1, B)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        bf16_f32_mmvf_rows_kernel<B, 4, RPB>(
                            x, ldx, w, y, ldy, n_in, n_out, n_tok);
                    });
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->/*
  DPCT1050: The template argument of the dpct_kernel_name could not be
  deduced. You need to update this code.
  */
            parallel_for<dpct_kernel_name<
                class bf16_f32_mmvf_rows_kernel_13db68, dpct_kernel_scalar<B>,
                dpct_kernel_scalar<8>, dpct_kernel_scalar<RPB>,
                dpct_kernel_scalar<false>,
                dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, nb) * sycl::range(1, 1, B),
                                  sycl::range(1, 1, B)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        bf16_f32_mmvf_rows_kernel<B, 8, RPB>(
                            x, ldx, w, y, ldy, n_in, n_out, n_tok);
                    });
    }
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

}  // namespace

bool bf16_gemv_fp32_mmvf_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                   int64_t n_in, int64_t n_out, int n_tok, const uint16_t* w_aux, float* y_aux,
                                   int64_t ldy_aux, void* stream) {
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    // exactly the conditions under which both calls would take the rows / multi kernels at the same block size
    if (!rows || n_out < 64 || n_tok < 2 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || w_aux == nullptr || y_aux == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        return false;
    const dpct::queue_ptr st = strata::q_of(stream);
    const int ni = (int) n_in, no = (int) n_out;
    switch (mmvf_block_size(n_in)) {
        case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 result = 0;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (result != 0)
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        throw std::runtime_error(
            std::string("bf16_gemv_fp32_mmvf_multi_aux launch: ") +
            dpct::get_error_string_dummy(result));
    return true;
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    const dpct::queue_ptr st = strata::q_of(stream);
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    if (rows && n_out >= 64) {
        const int ni = (int) n_in, no = (int) n_out;
        switch (mmvf_block_size(n_in)) {
            case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
        }
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        const dpct::err0 result = 0;
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (result != 0)
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1001: The statement could not be removed.
            */
            throw std::runtime_error(
                std::string("bf16_gemv_fp32_mmvf_multi launch: ") +
                dpct::get_error_string_dummy(result));
        return;
    }
#define STRATA_MMVF_M(N)                                                       \
    case N:                                                                    \
        switch (n_tok) {                                                       \
        case 1: {                                                              \
            auto exp_props = sycl::ext::oneapi::experimental::properties{      \
                sycl::ext::oneapi::experimental::use_root_sync};               \
                                                                               \
            st->submit([&](sycl::handler &cgh) {                               \
                auto x_ct0 = x;                                                \
                auto ldx_ct1 = ldx;                                            \
                auto w_ct2 = w;                                                \
                auto y_ct3 = y;                                                \
                auto ldy_ct4 = ldy;                                            \
                auto n_in_ct5 = (int)n_in;                                     \
                auto n_tok_ct6 = n_tok;                                        \
                                                                               \
                cgh.parallel_for<dpct_kernel_name<                             \
                    class bf16_f32_mmvf_multi_kernel_27194f,                   \
                    dpct_kernel_scalar<N>, dpct_kernel_scalar<1>,              \
                    dpct_kernel_scalar<true>>>(                                \
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *     \
                                          sycl::range(1, 1, N),                \
                                      sycl::range(1, 1, N)),                   \
                    exp_props,                                                 \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            bf16_f32_mmvf_multi_kernel<N, 1, true>(            \
                                x_ct0, ldx_ct1, w_ct2, y_ct3, ldy_ct4,         \
                                n_in_ct5, n_tok_ct6);                          \
                        });                                                    \
            });                                                                \
        } break;                                                               \
        case 2: {                                                              \
            auto exp_props = sycl::ext::oneapi::experimental::properties{      \
                sycl::ext::oneapi::experimental::use_root_sync};               \
                                                                               \
            st->submit([&](sycl::handler &cgh) {                               \
                auto x_ct0 = x;                                                \
                auto ldx_ct1 = ldx;                                            \
                auto w_ct2 = w;                                                \
                auto y_ct3 = y;                                                \
                auto ldy_ct4 = ldy;                                            \
                auto n_in_ct5 = (int)n_in;                                     \
                auto n_tok_ct6 = n_tok;                                        \
                                                                               \
                cgh.parallel_for<dpct_kernel_name<                             \
                    class bf16_f32_mmvf_multi_kernel_ded48c,                   \
                    dpct_kernel_scalar<N>, dpct_kernel_scalar<2>,              \
                    dpct_kernel_scalar<true>>>(                                \
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *     \
                                          sycl::range(1, 1, N),                \
                                      sycl::range(1, 1, N)),                   \
                    exp_props,                                                 \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            bf16_f32_mmvf_multi_kernel<N, 2, true>(            \
                                x_ct0, ldx_ct1, w_ct2, y_ct3, ldy_ct4,         \
                                n_in_ct5, n_tok_ct6);                          \
                        });                                                    \
            });                                                                \
        } break;                                                               \
        case 3: {                                                              \
            auto exp_props = sycl::ext::oneapi::experimental::properties{      \
                sycl::ext::oneapi::experimental::use_root_sync};               \
                                                                               \
            st->submit([&](sycl::handler &cgh) {                               \
                auto x_ct0 = x;                                                \
                auto ldx_ct1 = ldx;                                            \
                auto w_ct2 = w;                                                \
                auto y_ct3 = y;                                                \
                auto ldy_ct4 = ldy;                                            \
                auto n_in_ct5 = (int)n_in;                                     \
                auto n_tok_ct6 = n_tok;                                        \
                                                                               \
                cgh.parallel_for<dpct_kernel_name<                             \
                    class bf16_f32_mmvf_multi_kernel_a6c94f,                   \
                    dpct_kernel_scalar<N>, dpct_kernel_scalar<3>,              \
                    dpct_kernel_scalar<true>>>(                                \
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *     \
                                          sycl::range(1, 1, N),                \
                                      sycl::range(1, 1, N)),                   \
                    exp_props,                                                 \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            bf16_f32_mmvf_multi_kernel<N, 3, true>(            \
                                x_ct0, ldx_ct1, w_ct2, y_ct3, ldy_ct4,         \
                                n_in_ct5, n_tok_ct6);                          \
                        });                                                    \
            });                                                                \
        } break;                                                               \
        case 4: {                                                              \
            auto exp_props = sycl::ext::oneapi::experimental::properties{      \
                sycl::ext::oneapi::experimental::use_root_sync};               \
                                                                               \
            st->submit([&](sycl::handler &cgh) {                               \
                auto x_ct0 = x;                                                \
                auto ldx_ct1 = ldx;                                            \
                auto w_ct2 = w;                                                \
                auto y_ct3 = y;                                                \
                auto ldy_ct4 = ldy;                                            \
                auto n_in_ct5 = (int)n_in;                                     \
                auto n_tok_ct6 = n_tok;                                        \
                                                                               \
                cgh.parallel_for<dpct_kernel_name<                             \
                    class bf16_f32_mmvf_multi_kernel_eba2ed,                   \
                    dpct_kernel_scalar<N>, dpct_kernel_scalar<4>,              \
                    dpct_kernel_scalar<true>>>(                                \
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *     \
                                          sycl::range(1, 1, N),                \
                                      sycl::range(1, 1, N)),                   \
                    exp_props,                                                 \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            bf16_f32_mmvf_multi_kernel<N, 4, true>(            \
                                x_ct0, ldx_ct1, w_ct2, y_ct3, ldy_ct4,         \
                                n_in_ct5, n_tok_ct6);                          \
                        });                                                    \
            });                                                                \
        } break;                                                               \
        case 5: {                                                              \
            auto exp_props = sycl::ext::oneapi::experimental::properties{      \
                sycl::ext::oneapi::experimental::use_root_sync};               \
                                                                               \
            st->submit([&](sycl::handler &cgh) {                               \
                auto x_ct0 = x;                                                \
                auto ldx_ct1 = ldx;                                            \
                auto w_ct2 = w;                                                \
                auto y_ct3 = y;                                                \
                auto ldy_ct4 = ldy;                                            \
                auto n_in_ct5 = (int)n_in;                                     \
                auto n_tok_ct6 = n_tok;                                        \
                                                                               \
                cgh.parallel_for<dpct_kernel_name<                             \
                    class bf16_f32_mmvf_multi_kernel_3318a1,                   \
                    dpct_kernel_scalar<N>, dpct_kernel_scalar<5>,              \
                    dpct_kernel_scalar<true>>>(                                \
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *     \
                                          sycl::range(1, 1, N),                \
                                      sycl::range(1, 1, N)),                   \
                    exp_props,                                                 \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            bf16_f32_mmvf_multi_kernel<N, 5, true>(            \
                                x_ct0, ldx_ct1, w_ct2, y_ct3, ldy_ct4,         \
                                n_in_ct5, n_tok_ct6);                          \
                        });                                                    \
            });                                                                \
        } break;                                                               \
        case 6: {                                                              \
            auto exp_props = sycl::ext::oneapi::experimental::properties{      \
                sycl::ext::oneapi::experimental::use_root_sync};               \
                                                                               \
            st->submit([&](sycl::handler &cgh) {                               \
                auto x_ct0 = x;                                                \
                auto ldx_ct1 = ldx;                                            \
                auto w_ct2 = w;                                                \
                auto y_ct3 = y;                                                \
                auto ldy_ct4 = ldy;                                            \
                auto n_in_ct5 = (int)n_in;                                     \
                auto n_tok_ct6 = n_tok;                                        \
                                                                               \
                cgh.parallel_for<dpct_kernel_name<                             \
                    class bf16_f32_mmvf_multi_kernel_eddebc,                   \
                    dpct_kernel_scalar<N>, dpct_kernel_scalar<6>,              \
                    dpct_kernel_scalar<true>>>(                                \
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *     \
                                          sycl::range(1, 1, N),                \
                                      sycl::range(1, 1, N)),                   \
                    exp_props,                                                 \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            bf16_f32_mmvf_multi_kernel<N, 6, true>(            \
                                x_ct0, ldx_ct1, w_ct2, y_ct3, ldy_ct4,         \
                                n_in_ct5, n_tok_ct6);                          \
                        });                                                    \
            });                                                                \
        } break;                                                               \
        default: {                                                             \
            auto exp_props = sycl::ext::oneapi::experimental::properties{      \
                sycl::ext::oneapi::experimental::use_root_sync};               \
                                                                               \
            st->submit([&](sycl::handler &cgh) {                               \
                auto x_ct0 = x;                                                \
                auto ldx_ct1 = ldx;                                            \
                auto w_ct2 = w;                                                \
                auto y_ct3 = y;                                                \
                auto ldy_ct4 = ldy;                                            \
                auto n_in_ct5 = (int)n_in;                                     \
                auto n_tok_ct6 = n_tok;                                        \
                                                                               \
                cgh.parallel_for<dpct_kernel_name<                             \
                    class bf16_f32_mmvf_multi_kernel_b73f0b,                   \
                    dpct_kernel_scalar<N>, dpct_kernel_scalar<8>,              \
                    dpct_kernel_scalar<false>>>(                               \
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *     \
                                          sycl::range(1, 1, N),                \
                                      sycl::range(1, 1, N)),                   \
                    exp_props,                                                 \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            bf16_f32_mmvf_multi_kernel<N, 8, false>(           \
                                x_ct0, ldx_ct1, w_ct2, y_ct3, ldy_ct4,         \
                                n_in_ct5, n_tok_ct6);                          \
                        });                                                    \
            });                                                                \
        } break;                                                               \
        } break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_M(32); STRATA_MMVF_M(64); STRATA_MMVF_M(96); STRATA_MMVF_M(128);
        STRATA_MMVF_M(160); STRATA_MMVF_M(192); STRATA_MMVF_M(224); STRATA_MMVF_M(256);
    }
#undef STRATA_MMVF_M
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 result = 0;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (result != 0)
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        throw std::runtime_error(
            std::string("bf16_gemv_fp32_mmvf_multi launch: ") +
            dpct::get_error_string_dummy(result));
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    const dpct::queue_ptr st = strata::q_of(stream);
#define STRATA_MMVF_CASE(N)                                                    \
    case N: {                                                                  \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto x_ct0 = x;                                                    \
            auto w_ct1 = w;                                                    \
            auto y_ct2 = y;                                                    \
            auto n_in_ct3 = (int)n_in;                                         \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<                                 \
                class bf16_f32_mmvf_kernel_2903a5, dpct_kernel_scalar<N>>>(    \
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_out) *         \
                                      sycl::range(1, 1, N),                    \
                                  sycl::range(1, 1, N)),                       \
                exp_props,                                                     \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        bf16_f32_mmvf_kernel<N>(x_ct0, w_ct1, y_ct2,           \
                                                n_in_ct3);                     \
                    });                                                        \
        });                                                                    \
    } break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_CASE(32);
        STRATA_MMVF_CASE(64);
        STRATA_MMVF_CASE(96);
        STRATA_MMVF_CASE(128);
        STRATA_MMVF_CASE(160);
        STRATA_MMVF_CASE(192);
        STRATA_MMVF_CASE(224);
        STRATA_MMVF_CASE(256);
    }
#undef STRATA_MMVF_CASE
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 result = 0;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (result != 0)
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") +
                                 dpct::get_error_string_dummy(result));
}



void bf16_gemv_fp32_mmvf_cols(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int ncols,
                              void* stream) {
    // columns contiguous: the existing multi-row kernel (weight read once, each output bitwise its one-row call),
    // up to 8 rows per launch
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = ncols - c0 < 8 ? ncols - c0 : 8;
        if (nc == 1) bf16_gemv_fp32_mmvf(x + (size_t) c0 * n_in, w, y + (size_t) c0 * n_out, n_in, n_out, stream);
        else bf16_gemv_fp32_mmvf_multi(x + (size_t) c0 * n_in, n_in, w, y + (size_t) c0 * n_out, n_out, n_in, n_out, nc,
                                       stream);
    }
}

}  // namespace strata::kernels
