// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    std::vector<std::pair<std::string, std::string>> t;
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch)) {
        // gfx1200 / gfx1201 (RDNA4): only the PROMPT half of the gfx1151 table.  Reported exact (identical answers) on an R9700
        // in #1478 and re-checked bitwise here (notes-142-amd.md); the decode half measured slower on that card, so it stays out.
        // STRATA_HCD_EXACT is left out: it needs the gfx1151 hipBLASLt tuning table, finds none on gfx12, says so on every start
        // and does nothing (measured: with and without it identical, 0.0 to +0.1%).
        // STRATA_GDN_CONVL2 is AMD-only on purpose: it is not bitwise on CUDA 13.3 sm_120 (#1522), and this table is never
        // consulted by a CUDA build.
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (off12 != nullptr && off12[0] == '0') return t;
        t = {
            {"STRATA_GDN_HEAD", "1"},
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_CONVL2", "1"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_CVEC_FUSE", "1"},
        };
        return t;
    }
    if (!strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    // Exact (bitwise) on Aurora, each with its measured gain in aurora_s23.md:
    t = {
        // prompt
        {"STRATA_GDN_HEAD", "1"},        // the GDN recurrence, four lanes a column + the grid-stride norm (S23)
        {"STRATA_GDN_PP", "2"},          //   two columns per lane: 64K prompt +2.9% (stream A, round 2)
        {"STRATA_GDN_CONVL2", "1"},      // conv + q/k L2 norm in one kernel: 64K -0.3 s (stream A)
        {"STRATA_GDN_NOY", "1"},         // no dead FP32 store of the norm (stream A)
        {"STRATA_CVEC_FUSE", "1"},       // a steered layer's write, control vector and next norm in one pass (S23)
        {"STRATA_HCD_EXACT", "1"},       // the HC down GEMM in hipBLASLt solution 1176's k order: 64K +1.8% (stream S)
        {"STRATA_PF_PAD", "1"},          // padded GEMM row strides (acts with STRATA_PF_GEMM; inert without it) (S23)
        {"STRATA_TOPK_REG_MIN_BLOCKS", "4096"},   // the register top-k from 16K cells, not 28K: same ids; per 2,048 queries the ref takes 1.0 / 2.4 / 3.9 ms at 16K / 20K / 24K cells, the scan 0.87 / 0.93 / 1.00
        // decode
        {"STRATA_Q8_PACKED", "1"},       // the packed Q8_0 decode layout: +2.3% (S26)
        {"STRATA_Q6_PACKED", "1"},       // the packed Q6_K heads (S26)
        {"STRATA_MMVF_ROWS", "1"},       // bf16 multi-row GEMV, 4 rows per block (S25)
        {"STRATA_ATTN_LANECELL", "1"},   // decode attention scores, one cell per thread (S25)
        {"STRATA_EXPERT_V2", "1"},       // the grouped decode experts, IQ3_S gate/up + IQ4_NL down (S26)
        {"STRATA_TSUM", "1"},            // several warp sums as one transposed butterfly (S26)
        {"STRATA_LFUSE", "1"},           // fewer launches around the shared expert and the KV append (S26)
        {"STRATA_GDN_SPLIT", "1"},       // the GDN step over 4 blocks per head (S25/S26)
        {"STRATA_QFUSE", "1"},           // activation q8_1 images written by their producers (S26)
        {"STRATA_PLE_BATCH", "1"},       // the verify window's PLE key / value projections at once (S25)
        {"STRATA_SH_STREAM", "1"},       // the shared expert on its own stream: decode +1.8% / +6.7% (UD-Q4_K_XL) (140-m)
    };
    return t;
}

bool igpu_prompt_defaults_arch(const char* gcn_arch) {
    return strata::kernels::gfx_arch_is(gcn_arch, "gfx1103") || strata::kernels::gfx_arch_is(gcn_arch, "gfx1150") ||
           strata::kernels::gfx_arch_is(gcn_arch, "gfx1151");
}

bool igpu_prompt_set_enabled(const char* gcn_arch, bool windows) {
    const char* off = std::getenv("STRATA_IGPU_DEFAULTS");
    if (off != nullptr && off[0] == '0') return false;
    const char* e = std::getenv("STRATA_IGPU_PROMPT_SET");
    if (e != nullptr && e[0] == '1') return true;
    if (e != nullptr && e[0] == '0') return false;
    return kIgpuPromptSetDefaultOn && !windows && strata::kernels::gfx_arch_is(gcn_arch, "gfx1103");
}

bool igpu_uma_arena_default(const char* gcn_arch, bool integrated, bool windows, bool on) {
    if (!integrated) return false;
    const char* reg = std::getenv("STRATA_UMA_REGISTER");
    if (reg != nullptr && reg[0] == '1') return false;
    // 0.2.1.2: a Strix Halo (gfx1151) on Windows.  Registering the 47 GiB arena with the driver took ALL the free VRAM (91 GiB before,
    // 0 after; BIOS carve-out 64 GB + 32 GB shared): the expert cache got 0 slots and the engine exited ("the embedding buffer
    // failed"), 3 of 3 starts (notes-022-strix.md).  Unregistered, the arena is held by the working-set lock and the cache gets
    // 24,576 slots.  Independent of `on` (that is the 780M's opt-in).  STRATA_ARENA_PIN_GIB=N / 0 still registers it.
    if (windows) return strata::kernels::gfx_arch_is(gcn_arch, "gfx1151");
    return on && strata::kernels::gfx_arch_is(gcn_arch, "gfx1103");
}

std::vector<std::pair<std::string, std::string>> igpu_default_env(const char* gcn_arch, bool integrated) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!integrated || !igpu_prompt_defaults_arch(gcn_arch)) return t;
#if defined(_WIN32)
    const bool win = true;
#else
    const bool win = false;
#endif
    if (!igpu_prompt_set_enabled(gcn_arch, win)) return t;
    // Measured on a Radeon 780M (gfx1103, P2006, Q2_0, notes-153-igpu.md / notes-148-amd.md): prompts 109.5 -> 214.7 tok/s
    // with the first four, 233 with HC_UPMIX, decode unchanged.  Dense projections are 63% of a prompt's GPU time there.
    t = {
        {"STRATA_HIP_WMMA", "1"},      // the prompt attention on matrix cores
        {"STRATA_PF_GEMM", "1"},       // the prompt's FP16 projections on the 128 x 256 WMMA GEMM
        {"STRATA_PA_FAST", "1"},       // the prompt attention with single FP16 q and p
        {"STRATA_SELECT_WMMA", "1"},   // the block scorer on matrix cores
        {"STRATA_HC_UPMIX", "1"},      // the hyper-connection up projection with its epilogue
    };
    return t;
}

static void set_default(const std::pair<std::string, std::string>& kv) {
#if defined(_WIN32)
    _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
    setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
}

std::vector<std::string> apply_arch_defaults(const char* gcn_arch, bool integrated) {
    std::vector<std::string> set;
    for (const auto& kv : arch_default_env(gcn_arch)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
        set_default(kv);
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        const bool g12 = strata::kernels::gfx_arch_is_gfx12(gcn_arch);
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns "
                             "them off; a switch you set is kept): ",
                     g12 ? "gfx12 (RDNA4)" : "gfx1151 (Strix Halo)", set.size(),
                     g12 ? "STRATA_GFX12_DEFAULTS" : "STRATA_GFX1151_DEFAULTS");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    std::vector<std::string> igpu;
    for (const auto& kv : igpu_default_env(gcn_arch, integrated)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;
        set_default(kv);
        igpu.push_back(kv.first);
    }
    if (!igpu.empty()) {
        std::fprintf(stderr, "strata: integrated GPU (%s): the matrix-core prompt set is on (1.6x to 2x faster prompts on a "
                             "Radeon 780M, the answers differ in the last bits): ", gcn_arch);
        for (size_t i = 0; i < igpu.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", igpu[i].c_str() + 7);
        std::fprintf(stderr, ".  Turn one off with NAME=0 (for example STRATA_HC_UPMIX=0), all of them with STRATA_IGPU_PROMPT_SET=0");
#if defined(_WIN32)
        std::fprintf(stderr, ".  Not yet tested on Windows");
#endif
        std::fprintf(stderr, "\n");
        set.insert(set.end(), igpu.begin(), igpu.end());
    }
    return set;
}

}  // namespace strata::core
