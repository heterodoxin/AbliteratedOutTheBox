// include/strata/prefill/moe_fused.hpp - #136 (prompt speed): the routed experts of a Q2_0 pack in the prompt path on
// Strata's own int8 tensor-core kernels, opt-in with STRATA_PF_FUSED=1 (NVIDIA sm_80 and newer; sm_75 and HIP: MMQ).
//
// The MMQ path (moe_mmq.hpp) per MoE layer: a host sync to group the (token, k) pairs by expert, a gather of each
// expert into a 16-expert group buffer, then per group the q8_1 activations of every routed slot, gate/up, SwiGLU,
// the q8_1 of H and down - five passes over per-slot buffers.  Here:
//   - the activations are rounded to int8 once per TOKEN (not per slot), with a scale per 32 values;
//   - the grouping (counts, offsets, the rows of each expert, a list of 64-row tiles) runs on the GPU from the
//     router's ids, so the host never waits for the routing;
//   - one launch per batch of up to kMaxBatch experts reads each expert where it already is (a resident cache slot
//     or a slot of the streamed ring: a pointer per expert, no gather) - gate/up with SwiGLU and the int8 rounding
//     of H in its epilogue, then down into the per-slot rows the existing combine reads.
// The 2-bit codes are unpacked in registers and multiplied with mma.sync m16n8k32 (s8 x s8 -> s32); Q2_0's offset
// (w = d * (q - 1)) is folded into the activation blocks' sums.  The numbers differ from MMQ's (another int8
// rounding of the activations and of H) by about as much as MMQ differs from FP32 (tests/cuda/prefill_fused_moe_test).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fused {

/// Routed rows (token, k pairs of one expert) per tile.
constexpr int kTileRows = 64;
/// Experts per launch: their blob pointers travel in the launch's parameters (1 KB).
constexpr int kMaxBatch = 128;

/// This build has the kernels (a CUDA build; HIP and builds without MMQ do not).
bool built();
/// built(), and the current device is sm_80 or newer (mma.sync with s8 operands at m16n8k32).
bool available();
/// available(), unless STRATA_PF_FUSED=0 (read once): the Q2_0 pack's fused experts, on by default since 0.1.36.
bool enabled();
/// STRATA_PF_FUSED=1 given explicitly, and available(): the native IQ packs' fused kernels (opt-in).  On gfx12 (RDNA4)
/// only the native kernels exist: available() / enabled() / requested() stay false there and
/// native_supported() (moe_fused_iq.hpp) reads the same opt-in itself.
bool requested();

/// Bytes of `rows` activation rows of `cols` values (a multiple of 64) in the kernels' int8 form: per 64 values 64
/// codes and {d, c} of each half (c = -(1.5 * 2^23 + the half's code sum), what the epilogue adds back).
size_t act_bytes(int64_t rows, int64_t cols);
/// Bytes of the grouping tables for `n` routed rows over `n_expert` experts.
size_t group_bytes(int64_t n, int n_expert);

/// x [rows][cols] FP32 -> `xa` (act_bytes(rows, cols)).
void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream);

/// The routing `ids` [n = tokens * k] -> per expert its rows (counts, offsets, the 64-row tile list, in `scratch`,
/// group_bytes(n, n_expert)), `slot[i]` = the row of pair i (the row of the per-slot outputs the combine reads) and
/// `src[row]` = its token (i / k).  The order of the rows within an expert is not fixed; no output depends on it.
void group(const int32_t* ids, int64_t n, int k, int n_expert, void* scratch, int32_t* slot, int32_t* src,
           void* stream);

/// Experts [e0, e1) of one layer (e1 - e0 <= kMaxBatch); blob[e - e0]: the Strata Q2_0 blob of expert e on the
/// device (16-byte aligned), read only for an expert with rows.
struct Batch {
    int e0 = 0, e1 = 0;
    const uint8_t* blob[kMaxBatch] = {};
};
/// The batch's products: gate/up from `xa` (quantize_act of the layer's input, [tokens][2560]) through the rows
/// `src` of `group`, SwiGLU, H as int8 into `ha` (act_bytes(n, 640)), down into `dm` [n][2560] FP32 at the rows of
/// `group`.  `n`: the layer's routed rows (tokens * k), which bounds the launch grid.
void experts(const Batch& b, int n_expert, int64_t n, const void* scratch, const void* xa, const int32_t* src, void* ha,
             float* dm, void* stream);

}  // namespace strata::prefill::fused
