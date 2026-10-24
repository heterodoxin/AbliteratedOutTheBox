// include/strata/kernels/iq_kernels.hpp - the i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
// IQ4_NL) and Q2_0 on the GPU for the IQ2_XS / IQ3_XXS model files, and Q4_K / Q5_K / Q5_1 / Q8_0 for Unsloth's
// UD-Q4_K_XL (gate/up Q4_K or Q5_K, down Q5_1 or Q8_0, a Q8_0 embedding).
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, see third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// The token-embedding types iq_embed_rows and iq_dequant_f32 read: the i-quants above and BF16 (30).
bool embed_type_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32; fp16 times `*scale` when given (device:
/// an NVFP4 expert's s_down, read by the kernel itself so the source may be released right after it).
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream, const float* scale = nullptr);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
/// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.  `tail` (device, NVFP4 only): the blob's {s_gate, s_up,
/// s_down, 0}; gate rows are scaled by s_gate and up rows by s_up (s_down goes on the down matrix: folded into up,
/// the up weights would sit near 1e-8, deep in FP16's subnormals).
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream, const float* tail = nullptr);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
    size_t tail_off = 0;                // NVFP4: {s_gate, s_up, s_down, 0} at the blob's end; 0 = none
    // APR (prism.hadamard): the down weights are Hadamard-folded, so native_expert_grouped rotates h before
    // quantizing it (hadamard_rows).  had_block 0 = none; d_signs is a DEVICE vector of n_ff signs or nullptr.
    int had_block = 0;
    const float* d_signs = nullptr;
};
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
/// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);

/// A770 port (STRATA_PF_XMX=2): up to kXmxGroupMax experts' products in one XMX launch, the weights decoded from the
/// quantized rows into local memory.  Expert i: Y_i[cnt[i]][N] = X_i . W_i^T (X FP16, Y FP32, rows back to back).
/// gate/up: b != nullptr (a = gate, b = up rows of n_embd = K; the output's virtual rows interleave gate/up);
/// down: b == nullptr, a = the [N][K] rows.  N a multiple of 128, K of 32 (and a whole number of rows per 256 block).
constexpr int kXmxGroupMax = 16;
void iq_xmx_grouped(int ty, const uint8_t* const* a, const uint8_t* const* b, const int32_t* cnt, int n,
                    const uint16_t* X, float* Y, int N, int K, void* stream);

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
/// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.
/// `grid_groups` (1 .. cap_groups; 0 = cap_groups) groups run side by side, a block row each striding over the rest:
/// a call that usually has few groups or none (the verify window's PCIe share) launches less for the ones it does
/// not have.  The results do not depend on it.
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups = 0);
/// true: `native_expert_grouped`'s launches before the group stride (STRATA_GROUPED_V1=1 at startup) - a block row
/// per possible group, SwiGLU and the q8_1 quantization as two kernels over all cap_entries.  Bitwise the same results
/// (native_grouped_parity checks it); kept for A/B timing.  Set before graph capture; captured graphs keep theirs.
void native_grouped_set_v1(bool v1);

/// The bench only: the AMD kernel layout (STRATA_EXP_MODE values; -1 = the environment's) and the phase
/// (0 all, 1 gate/up + SwiGLU + quantize, 2 down).
void native_expert_set_mode(int mode, int phase);

/// APR (prism.hadamard): y = H_block (signs * x) per block of `block` values of each of `rows` rows of `width`
/// (x == y allowed).  `signs`: a device vector of `width` signs, or nullptr (identity).  The same floats as the CPU's
/// hadamard_rotate.  The f16 form rotates in place, computing in float.
void hadamard_rows(const float* x, float* y, int64_t rows, int64_t width, int block, const float* signs, void* stream);
void hadamard_rows_f16(uint16_t* x, int64_t rows, int64_t width, int block, const float* signs, void* stream);
/// A device copy (on the current device) of the host sign vector `host` of `width` signs, made on first use and
/// kept for the process.  Call it at setup, not inside a CUDA graph capture.  nullptr for a nullptr `host`.
const float* hadamard_device_signs(const float* host, int width);

/// EXPDEC (STRATA_EXPDEC=1, docs/EXPERTDEC.md): `native_expert_grouped` stages the group's q8_1 activations in shared
/// memory and computes several rows per lane, the same sums per (row, entry) as the default kernels (bitwise), for
/// gate/up IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S at n_embd 2560 and down IQ4_NL / Q2_0 at n_ff 640; other layouts
/// keep the default kernels.  Set before graph capture; captured graphs keep the kernels they captured.
void native_expert_set_expdec(int on);
int native_expert_expdec();
/// Its knobs, read from the environment at startup (STRATA_EXPDEC_R 1 | 2, _DR 1 | 2 | 4, _MINB 2 | 4, _D20 0 | 1,
/// _RB, _DRB, _MINBLK): gate/up and down rows per lane, the gate/up kernel's __launch_bounds__ minimum blocks per SM
/// (2 = up to 128 registers, 4 = up to 64), EXPDEC for the IQ4_NL down too, gate/up and down rows per block (0 = auto
/// from the launch's groups), the auto rule's minimum blocks.  In the setter a field <= 0 (< 0 for the two rows per
/// block and down_iq4_nl) keeps the current value.
/// `down` (STRATA_EXPDEC_DOWN, default 0): EXPDEC for the down too; by default only the gate/up (in the engine the
/// Q2_0 down was ~10% slower with EXPDEC, docs/EXPERTDEC.md par. 7).
struct NativeExpertExpdecParams {
    int gu_rows_per_lane = 0, down_rows_per_lane = 0, gu_min_blocks_per_sm = 0, down_iq4_nl = -1, down = -1;
    int gu_rows_per_block = -1, down_rows_per_block = -1, min_blocks = 0;
};
void native_expert_set_expdec_params(const NativeExpertExpdecParams& p);
NativeExpertExpdecParams native_expert_expdec_params();
/// Whether EXPDEC has kernels for this layout's gate/up and down (the bench's check).
bool native_expert_expdec_takes(const NativeExpertLayout& L, bool& gu, bool& down);
/// `iq_mmvq` and `native_expert_grouped` decode each weight part once and apply it to every column / entry;
/// true selects the older kernels that decode it again per column (STRATA_OLD_IQ_MMVQ=1 at startup).  Both give
/// bitwise the same results.  Set before graph capture; captured graphs keep the kernels they captured.
/// Volta prompt experts (STRATA_PF_WMMA): Y[rows of expert e] = X[rows] . W_e^T for a group of experts, W_e at
/// W + e * expert_bytes (n_out rows of K values of type ty, GGUF blocks), X FP16 (ldx), rows bounds[e]..bounds[e+1],
/// Y FP32 (ldy).  FP16 tensor cores, FP32 sums.  false: not launched (shape or type not covered).  CUDA only.
bool gemm_iq_f16_grouped(int ty, const void* W, size_t expert_bytes, int n_out, int K, const void* X, int ldx,
                         const int32_t* bounds, int n_experts, int max_rows, float* Y, int ldy, void* stream);
void iq_set_old_kernels(bool old);
bool iq_old_kernels();

}  // namespace strata::kernels
