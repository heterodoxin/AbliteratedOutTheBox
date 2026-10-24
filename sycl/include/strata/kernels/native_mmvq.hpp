#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

// Native GGUF Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL
// and IQ4_XS / CUDA Q8_1 adapters, pinned to llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, sm_120 generic MMVQ.
// All pointers are device pointers, at least 4-byte aligned, with no overlap.
// All calls enqueue on the explicit non-null CUDA stream; no allocation or wait.
// The translation unit must use --use_fast_math, as the pinned CUDA oracle does.
//
// Shapes use GGUF order: n_in is the contiguous reduction dimension, n_out is
// the weight row count, and ncols is the activation column/token count, 1..8
// (plan v0.3 P3). Columns are contiguous: activation column j is x + j * n_in
// (its Q8_1 blocks at j * n_in / 32) and output column j is y + j * n_out. See
// native_mmvq_set_multi_exact for how ncols > 1 relates to ncols == 1.
// n_in must be a positive multiple of 32 for quantization/Q4_0/Q5_0/Q8_0/IQ4_NL,
// 64 for Q2_0, or 256 for the other formats. n_out must be positive. Weights remain
// unmodified row-major GGUF blocks: Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210 and
// IQ4_XS=136 bytes per 256 elements; Q2_0 is 18 bytes per 64 elements.
// Q4_0/IQ4_NL=18, Q5_0=22, Q8_0=34 bytes per 32 elements.
// Q8_1 scratch has 36 bytes per 32 elements, with no extra row padding here.
// Input floats must be finite, and their block scales/sums representable in FP16.
std::size_t native_q8_1_bytes(int n_in, int ncols = 1);

// Layout for ncols > 1. false: llama.cpp's generic multi-column table (upstream), equal to ncols == 1 to
// float rounding, speed not yet measured. true (default): the ncols == 1 layout, every column bitwise equal to a
// single-column call. Set before
// graph capture; captured graphs keep the kernels they captured.
void native_mmvq_set_multi_exact(bool exact);
void native_mmvq_set_q6k_wide(bool on);   // SYCL port: the 16-byte-load Q6_K kernel (default on)
void native_q6_k_mmvq_stride224(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                                void* stream);   // SYCL port: experiment, Q6_K blocks at a 224-byte stride
bool native_mmvq_multi_exact();

// One quantization may serve multiple weight matrices sharing the same input.
// Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it does
// not reconstruct that sum from the quantized integers.
void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols,
                          void* stream);

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream);

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

// Convenience composition: caller owns scratch sized by native_q8_1_bytes.
void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

/// STRATA_Q8_PACKED=1 (opt-in): a lossless load-time repack of a Q8_0 matrix into a qs plane (n_out x n_in int8)
/// followed by a d plane (n_out x n_in / 32 fp16), the same bytes (34 per 32 values). A registered matrix's
/// native_q8_0_mmvq calls (keyed by its GGUF-layout device pointer, which stays valid for the prompt path) read the
/// packed copy instead; every output is bitwise equal to the GGUF-layout kernels.
bool native_q8_0_packed_enabled();
bool native_q8_0_packed_eligible(int n_in, int n_out);
void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out);
void native_q8_0_packed_register(const void* gguf_weights, const void* packed, int n_in, int n_out);
void native_q8_0_packed_unregister(const void* gguf_weights);

/// STRATA_Q6_PACKED=1 (opt-in): a packed copy of a Q6_K matrix (the output heads) - the same bytes as ql / qh /
/// scales / d planes - that native_q6_k_mmvq calls on `weights` read instead (bitwise equal outputs). The copy is
/// owned here: native_q6_k_pack builds it from the device matrix (false: not eligible or failed; nothing changes),
/// native_q6_k_unpack frees it. STRATA_Q6P_SELFTEST=1 checks it bitwise and times it at load.
bool native_q6_k_packed_enabled();
bool native_q6_k_pack(const void* weights, int n_in, int n_out, const char* what);
void native_q6_k_unpack(const void* weights);

/// STRATA_Q6_PLANAR (SYCL port; =0 turns it off): the Q6_K output heads stored as planes - ql (128 B), qh (64 B), the scales
/// (16 B, each lane's pair adjacent), d (2 B) of every 210-byte block - so the decode kernel's weight loads are single
/// aligned 16-byte loads (the GGUF layout is 2-byte aligned: two loads and a shift per run). Same bytes, ~14% faster, outputs
/// bitwise equal. native_q6_k_planar_adopt converts a GGUF-layout matrix (the old allocation is freed, the returned pointer
/// replaces it and is what native_q6_k_mmvq is called with; free it like the old one, after native_q6_k_unpack);
/// native_q6_k_planar_gather takes rows of a planar matrix as a new planar one (nullptr: `planar` is not one).
bool native_q6_k_planar_enabled();
bool native_q6_k_is_planar(const void* weights);
void* native_q6_k_planar_adopt(void* weights, int n_in, int n_out, const char* what);
void* native_q6_k_planar_gather(const void* planar, int n_in, const int32_t* rows_dev, int64_t n_sel, const char* what);

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

// Storage/dispatch helpers take stable GGML type IDs, avoiding a ggml runtime
// dependency in the engine: Q4_0=2, Q5_0=6, Q8_0=8, Q3_K=11, Q4_K=12, Q5_K=13,
// Q6_K=14, IQ4_NL=20, IQ4_XS=23, Q2_0=42. Unsupported IDs throw in the byte-count
// and launch helpers; only the
// capability query returns false.
bool native_mmvq_supported(int ggml_type) noexcept;
std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out);
void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream);
/// S26 STRATA_LFUSE: native_mmvq(w1 -> y1) and native_mmvq(w2 -> y2), same type / shape / input, in ONE launch,
/// bitwise the two calls. Returns false (nothing launched) where that is not the case (Q8_0, 2-8 columns only).
bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream);

} // namespace strata::kernels
