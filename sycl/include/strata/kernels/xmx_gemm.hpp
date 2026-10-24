// SYCL port: XMX (joint_matrix) GEMM straight from a ggml-quantized weight matrix - see iq_kernels.dp.cpp.
#pragma once
#include <cstdint>
namespace strata::kernels {
/// Y[M][n_out] (FP32, row stride ldy) = X[M][K] (FP16, row stride K) . W[n_out][K]^T with W's rows quantized as ggml
/// type `ty` (18 IQ3_XXS, 20 IQ4_NL, 21 IQ3_S, 22 IQ2_S, 42 Q2_0). `up` non-null: the gate/up pair, virtual rows
/// interleaved gate/up as iq_dequant_gu_f16 lays them out (n_out = 2 * n_ff). Returns false when the shape or type is
/// not handled (the caller keeps its dequant + oneMKL path).
bool xmx_gemm_iq(int ty, const void* gate, const void* up, int64_t K, int n_out, const uint16_t* X, int M, float* Y,
                 int64_t ldy, void* stream);
}
