// src/kernels/cuda/shared_expert.cu - P2.S2: the shared expert.
//
// `ref/moe.py::shared_expert`, transcribed:
//
//     h = silu(x @ gate_shexp.T) * (x @ up_shexp.T)      <- SILU GOES ON GATE, not on up
//     h = h @ down_shexp.T                                (nt, n_embd)
//     g = sigmoid(x @ gate_inp_shexp)                     (nt,) ONE SCALAR PER TOKEN
//     return h * g[:, None]
//
// TWO THINGS HERE WERE BELIEVED WRONG FOR MANY ROUNDS and `docs/semantics.md` records both.  SILU GOES ON THE
// GATE TENSOR: the opposite reading is plausible, produces the right shapes, and is wrong.  And
// `ffn_gate_inp_shexp` is `(n_embd,)` whose output is "one value per token" - a SCALAR gate obtained by dotting
// that vector with the hidden state, NOT a per-expert gate and NOT a per-dimension elementwise one.  Both were
// live readings of the same shapes until the source comment settled them.
//
// The result is ADDED to the routed output - not router-weighted, not renormalised against it.
//
// The legacy canonical projections use their explicit Q8_0/Q8_K activation images. The optional native
// BF16 path affects only the scalar gate: it reads the original F32 input and uses pinned CUDA MMVF plus
// an FP32 sigmoid. Native projection overrides independently select CUDA Q8_1 MMVQ and FP32 SwiGLU.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cmath>
#include <climits>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 128;
bool native_bf16 = false;

// `f32_to_f16` used to live here as a private copy.  Round 198 found it - and the identical one in
// `quantize_act.cu` - returning a NaN for every FINITE value that overflows fp16 (1e30, 65536, 1e45) instead
// of saturating to inf, because `exp >= 31` conflates an f32 inf/NaN with an out-of-range finite exponent.
// This file's fixture is all O(1), so it could not see it.  The conversion now lives in one place, with the
// regimes spelled out.

// silu(x) = x / (1 + exp(-x)), written as `ref/moe.py` writes it, in DOUBLE then cast - the reference works in
// float64 and a float32 exp differs in the last bits.  The multiplication by `up` is the reference's order.
__dpct_inline__ void swiglu_kernel(const float *__restrict__ gate,
                                   const float *__restrict__ up,
                                   float *__restrict__ out, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const double x = (double) gate[i];
    out[i] = (float)(x / (1.0 + sycl::exp(-x))) * up[i];
}

__dpct_inline__ void native_swiglu_kernel(const float *gate, const float *up,
                                          float *out, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    // Pinned CUDA unary.cuh op_silu, then unary_gated_op_kernel's multiply.
    // Explicit intrinsics reproduce its --use_fast_math operations without
    // changing compilation of the default legacy arithmetic in this file.
    out[i] = gate[i] / (1.0f + sycl::native::exp(-gate[i])) * up[i];
}


void to_f16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) out[i] = f16_from_f32(in[i]);
}

// The per-token scalar gate: sigmoid(dot(x, w)) with w = `ffn_gate_inp_shexp` (n_embd,).
//
// **ONE THREAD WAS THE LARGEST SINGLE KERNEL IN THE MODEL, AND THE COMMENT DEFENDING IT SAID WHY IT SHOULD
// NOT BE.**  It read: "n_embd is 2560, so this is a short serial loop and the parallelism is elsewhere."
// 2560 iterations is not short when each one is a DOUBLE multiply-add.  This card's double rate is **1/64** of
// its float rate, so the loop is a 2560-long dependency chain through a 1/64-rate unit - tens of microseconds
// from the chain alone, before counting 2560 sequential global loads issued by one thread with no coalescing.
//
// MEASURED, and this is what the number looks like from outside: `shared_expert` is **0.2374 ms of a
// 0.8883 ms block - 26.7% of the whole block, 11.4 ms per token** - for three projections that read 5.5 MB and
// whose memory floor is 0.009 ms.  It is seven kernels, and six of them are GEMVs or elementwise passes over
// 640-2560 elements.  This is the one that cannot be explained by bandwidth.
//
// THE DOUBLE ACCUMULATOR IS KEPT.  `ref/gr.py` reduces in float64 and `block_sum` in `gr.cu` exists for the
// same reason - the summation order of a tree is not the reference's, so the defence is enough precision that
// the order stops mattering.  What changes is the CHAIN: ten terms per thread instead of 2560, then a
// tree-reduce.  Through a sigmoid, a 1e-16 relative difference in the dot is not a difference.
//
// **BOTH OPERANDS ARE BF16, AND THAT IS THE REFERENCE'S CONTRACT, NOT A STORAGE DETAIL.**  `gate_inp_shexp` is a
// BF16 weight, so ggml converts the activation to its `vec_dot_type` - BF16 - exactly as it does for the router
// (`docs/activation-contract.md`).  This kernel used to take `const float* w` while the LOADER had already
// re-rounded the tensor to 2 B/elem, so it read 2560 floats out of a 5120-byte buffer: 5120 B past the end of
// the tensor, inside the 4.5 GiB arena, where nothing faults.  The dot came out astronomically large, sigmoid
// saturated to 1.0, and `scale_kernel` multiplied the shared expert's output by it - which is why the FIRST
// symptom was 2560 non-finite outputs rather than a wrong scalar.
//
// A wrong scalar here is the quiet failure mode: sigmoid bounds the damage to [0,1], so a gate that should be
// 0.5 and reads 1.0 scales the shared expert by 2x and produces perfectly finite, perfectly plausible logits.
__dpct_inline__ double warp_sum_d(double v) {
    /*
DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
    /*
DPCT1121: Make sure that the "v" which is used in the SYCL group
function/algorithm is initialized.
*/
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v +=
        dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            off);
    /*
    DPCT1108: '__shfl_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    /*
    DPCT1121: Make sure that the "v" which is used in the SYCL group
    function/algorithm is initialized.
    */
    return dpct::experimental::select_from_sub_group(
        0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, 0);
}

__dpct_inline__ void scalar_gate_kernel(const uint16_t *__restrict__ x_bf16,
                                        const uint16_t *__restrict__ w_bf16,
                                        float *__restrict__ out, int n_embd) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &scratch = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<
        3>()); // 8 warps: the launch is <<<1, 256>>>
    double acc = 0.0;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n_embd;
         i += item_ct1.get_local_range(2))
        acc += (double) f32_from_bf16(x_bf16[i]) * (double) f32_from_bf16(w_bf16[i]);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    acc = warp_sum_d(acc);
    if (lane == 0) scratch[warp] = acc;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int nw = ((int)item_ct1.get_local_range(2) + 31) >> 5;
    if (warp == 0) {
        acc = (item_ct1.get_local_id(2) < nw)
                  ? scratch[item_ct1.get_local_id(2)]
                  : 0.0;
        acc = warp_sum_d(acc);
        if (item_ct1.get_local_id(2) == 0) out[0] =
            (float)(1.0 / (1.0 + sycl::exp(-acc)));
    }
}

__dpct_inline__ void scale_kernel(float *__restrict__ out,
                                  const float *__restrict__ g, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) out[i] *= g[0];
}

__dpct_inline__ void native_scalar_sigmoid_kernel(float *gate) {
    // Match the single-token CUDA sigmoid's FP32 fast-math operations without changing legacy kernels'
    // compilation flags. The dot product was already reduced by the pinned native MMVF implementation.
    gate[0] = 1.0f / (1.0f + sycl::native::exp(-gate[0]));
}
void native_scalar_sigmoid_multi_kernel(float* gate) {   // thread t = token t, same expression
    /*
    DPCT1064: Migrated __expf call is used in a macro/template definition
    and may not be valid for all macro/template uses. Adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    gate[item_ct1.get_local_id(2)] =
        1.0f / (1.0f + sycl::native::exp(-gate[item_ct1.get_local_id(2)]));
}

/// The MoE block's final combination.  See the header for the two readings it exists to pin.
__dpct_inline__ void moe_combine_kernel(const float *__restrict__ parts,
                                        const float *__restrict__ weights,
                                        const float *__restrict__ shared,
                                        float *__restrict__ y, int n_embd,
                                        int k, int has_shared) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (j >= n_embd) return;
    // Accumulate in DOUBLE, in the reference's own order (`for i in range(k): out[t] += w[t,i] * g[0]`).
    // f32 would be defensible for ten terms, but the reference is float64 and this is one line.
    double acc = 0.0;
#pragma unroll
    for (int e = 0; e < k; ++e) acc +=
        (double)weights[e] * (double)parts[(size_t)e * n_embd + j];
    // The SHARED output is added PLAIN - not router-weighted, not renormalised against the routed sum.
    if (has_shared) acc += (double) shared[j];
    y[j] = (float) acc;
}

}  // namespace

void shared_expert_set_native_bf16(bool enabled) { native_bf16 = enabled; }
bool shared_expert_native_bf16_enabled() { return native_bf16; }

namespace {
__dpct_inline__ void scale_rows_kernel(float *__restrict__ out,
                                       const float *__restrict__ g, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) out[(size_t) t * n + i] *= g[t];
}

__dpct_inline__ void sigmoid_scale_rows_kernel(float *__restrict__ out,
                                               const float *__restrict__ g,
                                               int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) {
        const float gt = 1.0f / (1.0f + sycl::native::exp(-g[t]));
        out[(size_t) t * n + i] *= gt;
    }
}

__dpct_inline__ void
sigmoid_scale_rows_vec4_kernel(sycl::float4 *__restrict__ out4,
                               const float *__restrict__ g, int n4) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n4) {
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const float gt = 1.0f / (1.0f + sycl::native::exp(-*(g + t)));
        sycl::float4 v = out4[(size_t)t * n4 + i];
        v.x() *= gt;
        v.y() *= gt;
        v.z() *= gt;
        v.w() *= gt;
        out4[(size_t) t * n4 + i] = v;
    }
}

void launch_sigmoid_scale_rows(float *out, const float *g, int n_embd,
                               int n_tok, dpct::queue_ptr cs) {
    if ((n_embd & 3) == 0 && ((uintptr_t) out & 15u) == 0) {
        const int n4 = n_embd >> 2;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<
                dpct_kernel_name<class sigmoid_scale_rows_vec4_kernel_723a94>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok,
                                (unsigned)((n4 + THREADS - 1) / THREADS)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    sigmoid_scale_rows_vec4_kernel(
                        reinterpret_cast<sycl::float4 *>(out), g, n4);
                });
        }
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs->parallel_for<
            dpct_kernel_name<class sigmoid_scale_rows_kernel_26f4a0>>(
            sycl::nd_range<3>(
                sycl::range(1, (unsigned)n_tok,
                            (unsigned)((n_embd + THREADS - 1) / THREADS)) *
                    sycl::range(1, 1, THREADS),
                sycl::range(1, 1, THREADS)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                sigmoid_scale_rows_kernel(out, g, n_embd);
            });
    }
}

bool fused_swiglu_q81_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_FUSED_SWIGLU_Q81");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}
}  // namespace

void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream, const void* x_q8_1_ready, int lfuse) {
    const bool gate_deferred = (lfuse & 1) != 0, pair = (lfuse & 2) != 0;
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    dpct::queue_ptr cs = strata::q_of(stream);
    // the activation's q8_1: the caller's (STRATA_VERIFY_QDEDUP, x_q8_1_ready) or the descriptor's, else quantized here
    const void* x_q8_1 = x_q8_1_ready ? x_q8_1_ready : nw.x_q8_1;
    if (!x_q8_1) {
        native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
        x_q8_1 = nw.q8_1;
    }
    // S26 STRATA_LFUSE (gate_deferred): gate and up in one launch where the pair applies (bitwise the two calls)
    if (!(pair && nw.gate_type == nw.up_type &&
          native_mmvq_pair(nw.gate_type, nw.gate_data, nw.up_data, x_q8_1, gate, up, (int) n_embd, (int) n_ff, n_tok, stream))) {
        native_mmvq(nw.gate_type, nw.gate_data, x_q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
        native_mmvq(nw.up_type, nw.up_data, x_q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    }
    const int n = (int) (n_ff * n_tok);
    if (fused_swiglu_q81_enabled()) {
        native_swiglu_quantize_q8_1(gate, up, nw.q8_1, (int) n_ff, n_tok, stream);
    } else {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<
                dpct_kernel_name<class native_swiglu_kernel_c4e755>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((n + THREADS - 1) / THREADS)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    native_swiglu_kernel(gate, up, gate, n);
                });
        }
        native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    }
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    if (gate_deferred) {   // the caller computed g (raw) with the router and scales `out` in the combine
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (e !=
            0) throw std::runtime_error(std::string("shared_expert_multi: ") +
                                        dpct::get_error_string_dummy(e));
        return;
    }
    static const bool batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    if (native_bf16 && batch && n_tok > 1) {   // one gemv for all rows (outputs identical), fused sigmoid+scale
        bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
        launch_sigmoid_scale_rows(out, g, (int) n_embd, n_tok, cs);
    } else if (native_bf16 && n_tok == 1) {
        bf16_gemv_fp32_mmvf(x, gate_inp_bf16, g, n_embd, 1, stream);
        launch_sigmoid_scale_rows(out, g, (int) n_embd, 1, cs);
    } else {
        for (int t = 0; t < n_tok; ++t) {
            if (native_bf16) {
                bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
                {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};

                    cs->submit([&](sycl::handler &cgh) {
                        auto g_t_ct0 = g + t;

                        cgh.parallel_for<dpct_kernel_name<
                            class native_scalar_sigmoid_kernel_9552ea>>(
                            sycl::nd_range<3>(sycl::range(1, 1, 1),
                                              sycl::range(1, 1, 1)),
                            exp_props, [=](sycl::nd_item<3> item_ct1) {
                                native_scalar_sigmoid_kernel(g_t_ct0);
                            });
                    });
                }
            } else {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs->get_device(),
                                             {sycl::aspect::fp64});

                cs->submit([&](sycl::handler &cgh) {
                    auto x_bf16_size_t_t_n_embd_ct0 =
                        x_bf16 + (size_t)t * n_embd;
                    auto g_t_ct2 = g + t;

                    cgh.parallel_for<
                        dpct_kernel_name<class scalar_gate_kernel_76bc5d>>(
                        sycl::nd_range<3>(sycl::range(1, 1, 256),
                                          sycl::range(1, 1, 256)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                scalar_gate_kernel(x_bf16_size_t_t_n_embd_ct0,
                                                   gate_inp_bf16, g_t_ct2,
                                                   (int)n_embd);
                            });
                });
            }
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs->parallel_for<dpct_kernel_name<class scale_rows_kernel_4f17ec>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok,
                                (unsigned)((n_embd + THREADS - 1) / THREADS)) *
                        sycl::range(1, 1, THREADS),
                    sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    scale_rows_kernel(out, g, (int)n_embd);
                });
        }
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
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) throw std::runtime_error(std::string("shared_expert_multi: ") +
                                         dpct::get_error_string_dummy(e));
}

uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}

void shared_expert(const uint8_t *x_q8_0, const uint8_t *x_q8k,
                   const uint16_t *x_bf16, const SForm &gate_form,
                   const uint8_t *gate_codes, const float *gate_scales,
                   const float *gate_off, const SForm &up_form,
                   const uint8_t *up_codes, const float *up_scales,
                   const float *up_off, const SForm &down_form,
                   const uint8_t *down_codes, const float *down_scales,
                   const float *down_off, const uint16_t *gate_inp_bf16,
                   float *scratch, float *out, int64_t n_embd, int64_t n_ff,
                   int tpr, void *stream, const float *x_f32,
                   const NativeSharedWeights *native) try {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool use_native = native_bf16;
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if ((use_native || native_gate || native_up) && !x_f32)
        throw std::invalid_argument("shared_expert native input projection requires unrounded x_f32");
    if (native_projection) {
        if (!native->q8_1 || !stream || n_embd > INT_MAX || n_ff > INT_MAX)
            throw std::invalid_argument("shared_expert native projections require scratch, stream and int32 dimensions");
        // Validate every active shape before any kernel is enqueued.
        if (native_gate) native_mmvq_weight_bytes(native->gate_type, (int) n_embd, (int) n_ff);
        if (native_up) native_mmvq_weight_bytes(native->up_type, (int) n_embd, (int) n_ff);
        if (native_down) native_mmvq_weight_bytes(native->down_type, (int) n_ff, (int) n_embd);
    }
    if (scratch == nullptr) {
        std::fprintf(stderr, "shared_expert: scratch is null; the caller owns it "
                             "(see shared_expert_scratch_bytes)\n");
        std::exit(1);
    }
    // CARVED FROM THE CALLER'S SCRATCH.  This used to be four `cudaMalloc`s and four `cudaFree`s per call, which
    // is illegal during stream capture AND a token-path allocation - P2.T10 forbids both.
    uint8_t* p = (uint8_t*) scratch;
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    float* gate = (float*) p;
    float* up = (float*) (p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = (float*) (p + a * 2 + q0 + qk);

    // WHICH ACTIVATION THIS PROJECTION WANTS, READ FROM ITS OWN FORM.  See `SForm::act_kind`: the three
    // families cannot be told apart by the other fields, so the kind is carried rather than derived.
    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) {
            s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, stream);
        } else if (f.act_kind == 1) {
            s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, stream);
        } else {
            s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, stream);
        }
    };

    const unsigned g_ff = (unsigned) ((n_ff + THREADS - 1) / THREADS);
    const unsigned g_embd = (unsigned) ((n_embd + THREADS - 1) / THREADS);

    // gate and up projections, then silu(gate) * up in place in `gate`
    if (native_gate || native_up)
        native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, stream);
    if (native_gate)
        native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
    if (native_up)
        native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
    if (native_projection && native_down && fused_swiglu_q81_enabled()) {
        native_swiglu_quantize_q8_1(gate, up, native->q8_1, (int) n_ff, 1, stream);
        native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
    } else {
        if (native_projection)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class native_swiglu_kernel_75dcdc>>(
                    sycl::nd_range<3>(sycl::range(1, 1, g_ff) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        native_swiglu_kernel(gate, up, gate, (int)n_ff);
                    });
        } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(
                strata::q_of(stream)->get_device(),
                {sycl::aspect::fp64});

            strata::q_of(stream)
                ->parallel_for<dpct_kernel_name<class swiglu_kernel_fbfcfe>>(
                    sycl::nd_range<3>(sycl::range(1, 1, g_ff) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        swiglu_kernel(gate, up, gate, (int)n_ff);
                    });
        }

        // down: (n_ff) -> (n_embd), and THE INTERMEDIATE IS QUANTIZED TO THE DOWN WEIGHT'S OWN CONTRACT - which is
        // what `ggml_mul_mat` does for every matmul in the model.  It used to be rounded to fp16 with no
        // justification beyond "the kernel takes fp16".
        if (native_down) {
            native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, stream);
            native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
        } else if (down_form.act_kind == 1) {
            if (n_ff % 256 != 0) {
                std::fprintf(stderr, "shared_expert: the down weight wants Q8_K but n_ff %lld is not a multiple "
                                     "of 256; Q8_K is structurally impossible here\n", (long long) n_ff);
                std::exit(1);
            }
            quantize_q8_K(gate, h_q8k, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        } else if (down_form.code_bits == 2) {
            quantize_q8_0(gate, h_q8_0, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        } else {
            quantize_q8_0(gate, h_q8_0, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        }
    }

    // the per-token scalar gate, then the multiply.  Note the gate is computed from `x`, the ORIGINAL hidden
    // state, not from anything the expert produced. The historical branch uses BF16-rounded inputs; the
    // native branch uses the pinned CUDA FP32 activation contract.
    // `<<<1, 256>>>`: one block, because the output is ONE scalar and a second block would only add a global
    // round trip.  256 threads is the reduction's width, not the problem's size.
    if (use_native) {
        bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, stream);
        launch_sigmoid_scale_rows(out, g, (int)n_embd, 1,
                                  strata::q_of(stream));
    } else {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(
                strata::q_of(stream)->get_device(),
                {sycl::aspect::fp64});

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class scalar_gate_kernel_82de77>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            scalar_gate_kernel(x_bf16, gate_inp_bf16, g,
                                               (int)n_embd);
                        });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<dpct_kernel_name<class scale_kernel_a41a71>>(
                    sycl::nd_range<3>(sycl::range(1, 1, g_embd) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        scale_kernel(out, g, (int)n_embd);
                    });
        }
    }

    if (stream == nullptr) {
        const dpct::err0 e = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void moe_combine(const float *parts, const float *weights, const float *shared,
                 float *y, int64_t n_embd, int64_t k, void *stream) try {
    if (n_embd <= 0 || k <= 0) return;
    // k > 64 is refused rather than truncated: silently summing the first 64 of a longer list would be a
    // wrong answer that looks like a right one, and no geometry in this artifact comes close to it.
    if (k > 64) {
        std::fprintf(stderr, "moe_combine: k = %lld exceeds 64\n", (long long) k);
        std::exit(1);
    }
    const unsigned grid = (unsigned) ((n_embd + THREADS - 1) / THREADS);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp64});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                int shared_nullptr_ct6 = shared != nullptr;

                cgh.parallel_for<
                    dpct_kernel_name<class moe_combine_kernel_c03916>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        moe_combine_kernel(parts, weights, shared, y,
                                           (int)n_embd, (int)k,
                                           shared_nullptr_ct6);
                    });
            });
    }
    if (stream == nullptr) {
        const dpct::err0 e = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
