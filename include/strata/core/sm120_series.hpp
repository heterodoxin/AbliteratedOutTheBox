// strata/core/sm120_series.hpp - bitwise kernel switches that are ON by default on a compute capability 12.0 card
// (RTX 50 series / RTX PRO Blackwell workstation cards).
//
// 147-verify measured these exact switches on an RTX PRO 5000 Blackwell 72 GB (IQ3_S, every expert in VRAM): together
// a verify round of 4 tokens 13.70 -> 12.23 ms with the same text (notes-147-l620.md).  Each one is bitwise equal to
// the kernels it replaces (their own parity tests and the identity gate), so only the speed changes:
//   STRATA_SIDE_MIXER=2  the mixer's input-only work on branches of the verify graph
//   STRATA_ATTN_DAILY=1  the Daily's decode attention (attn_chunk_kernel_pre75 + merge v2)
//   STRATA_GR_UP75=1     the hyper-connection up projection with its rows' loads hoisted
//   STRATA_GR_DOWN_TG=1  the hyper-connection down projection with one token a block row (T times the blocks); =2 (token
//                        pairs) on a 12.0 card with fewer than 80 SMs
//   STRATA_GU_SWIGLU=1   gate/up, SwiGLU and the q8_1 blocks of the routed experts in one launch
// and one that rounds differently (KL-checked, the AMD gfx12 rule): STRATA_PF_FUSED (the native packs' prompt experts on the
// fused int8 kernels, IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS gate/up only, 12.0 cards with 80+ SMs only; =0 keeps MMQ, =1 takes every
// covered format on any card).
//
//   STRATA_SM120_PERF=0  turns the whole set off on a 12.0 card (the defaults every other card has).
// Each switch's own variable still wins in both directions (=0 off, =1 / =2 on), on any card, as before.
// Nothing here changes a card that is not a 12.0 (an sm_121 / sm_100 card keeps its defaults).
#pragma once

#include "strata/core/emulate.hpp"

#include <cstdlib>

namespace strata {

/// STRATA_SM120_PERF (read once): unset or anything but "0" = on
inline bool sm120_series() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SM120_PERF");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
}

/// the device (raw cudaDevAttrComputeCapabilityMajor / Minor) is a 12.0 card, or emulates one (STRATA_EMULATE_CC=120)
inline bool is_sm120(int major, int minor) { return cc_major_of(major) == 12 && cc_minor_of(minor) == 0; }

/// the default of a bitwise switch of the set: on for a 12.0 card unless STRATA_SM120_PERF=0
inline bool sm120_default(int major, int minor) { return is_sm120(major, minor) && sm120_series(); }

}  // namespace strata
