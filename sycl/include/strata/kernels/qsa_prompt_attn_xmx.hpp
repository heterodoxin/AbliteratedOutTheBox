// SYCL port: the prompt attention (QSA, selected cells) on XMX joint_matrix - see qsa_prompt_attn_xmx.dp.cpp.
#pragma once
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include <cstdint>
namespace strata::kernels {
/// The same contract as qsa_prompt_attn_batch; returns false (nothing launched) when the shape, pools or device are not
/// handled, so the caller keeps its fallback. Off by default (slower than the fallback): STRATA_PROMPT_ATTN_XMX=1 enables it.
bool qsa_prompt_attn_xmx(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                         const QsaShapes& s, float* attn, int64_t n_q, void* stream);
}
