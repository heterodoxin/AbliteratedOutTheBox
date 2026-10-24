// src/core/arch_defaults_test.cpp - the per-architecture default table (no GPU needed).
#include "strata/core/arch_defaults.hpp"
#include "strata/core/gfx12_platform_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main() {
    using namespace strata::core;
    unset("STRATA_GFX1151_DEFAULTS");
    // other architectures (and CUDA, which passes an empty name): nothing
    CHECK(arch_default_env("gfx1100").empty());
    CHECK(arch_default_env("gfx1150").empty());
    CHECK(arch_default_env("gfx1152").empty());
    CHECK(arch_default_env("gfx11510").empty());   // not a prefix match
    CHECK(arch_default_env("").empty());
    CHECK(arch_default_env(nullptr).empty());
    // gfx1151, plain and with its feature suffix: the table, and no bit-changing switch in it
    for (const char* a : {"gfx1151", "gfx1151:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() >= 10);
        for (const auto& kv : t) {
            CHECK(kv.first.rfind("STRATA_", 0) == 0);
            for (const char* bits : {"STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_HC_UPMIX", "STRATA_PA_FAST", "STRATA_HIP_WMMA",
                                     "STRATA_SELECT_WMMA", "STRATA_HC_Q8", "STRATA_PF_HCDOWN", "STRATA_WMMA_GEMM"})
                CHECK(kv.first != bits);
        }
    }
    // gfx12: the five prompt switches only, no decode switch, no bit-changing one
    unset("STRATA_GFX12_DEFAULTS");
    for (const char* a : {"gfx1200", "gfx1201", "gfx1201:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() == 5);
        for (const auto& kv : t) CHECK(kv.first != "STRATA_HCD_EXACT");
        for (const auto& kv : t) {
            CHECK(kv.first != "STRATA_Q8_PACKED" && kv.first != "STRATA_SH_STREAM" && kv.first != "STRATA_PF_GEMM" &&
                  kv.first != "STRATA_PF_FUSED");
        }
    }
    CHECK(arch_default_env("gfx12010").empty());
    put("STRATA_GFX12_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1201").empty());
    unset("STRATA_GFX12_DEFAULTS");
    // a user's setting wins, whatever it is; the rest are set
    put("STRATA_TSUM", "0");
    unset("STRATA_QFUSE");
    const auto set = apply_arch_defaults("gfx1151");
    CHECK(std::getenv("STRATA_TSUM") != nullptr && std::string(std::getenv("STRATA_TSUM")) == "0");
    CHECK(std::getenv("STRATA_QFUSE") != nullptr && std::string(std::getenv("STRATA_QFUSE")) == "1");
    for (const auto& n : set) CHECK(n != "STRATA_TSUM");
    // the opt-out
    put("STRATA_GFX1151_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1151").empty());
    CHECK(apply_arch_defaults("gfx1151").empty());
    // the two approved gfx12 prompt defaults (WMMA attention, Q2_0 fused experts): on for Linux, off for Windows
    CHECK(gfx12_prompt_defaults_on(false));
    CHECK(!gfx12_prompt_defaults_on(true));
#if defined(_WIN32)
    CHECK(!kGfx12PromptDefaultsOn);
#else
    CHECK(kGfx12PromptDefaultsOn);
#endif
    // 153-igpu: the matrix-core prompt set of the integrated Radeons, chosen by the UMA flag AND the architecture.
    // Mock device properties: {gcnArchName, integrated}.
    struct Mock { const char* name; const char* gcn; bool integrated; bool expect_on; };
    const Mock mocks[] = {
        {"Radeon 780M (Phoenix, UMA)",          "gfx1103",                 true,  true},
        {"Radeon 780M with feature suffix",     "gfx1103:sramecc-:xnack-", true,  true},
        {"Radeon 890M (Strix Point, UMA)",      "gfx1150",                 true,  true},
        {"Radeon 8060S (Strix Halo, UMA)",      "gfx1151",                 true,  true},
        {"RX 7900 XTX (dGPU)",                  "gfx1100",                 false, false},
        {"RX 7800 XT (dGPU)",                   "gfx1101",                 false, false},
        {"R9700 (dGPU)",                        "gfx1201",                 false, false},
        {"RX 9060 (dGPU)",                      "gfx1200",                 false, false},
        {"RX 6800 (dGPU)",                      "gfx1030",                 false, false},
        {"a gfx1103 reported as a discrete card", "gfx1103",               false, false},
        {"a gfx1151 reported as a discrete card", "gfx1151",               false, false},
        {"Radeon 860M (Krackan, no kernels)",   "gfx1152",                 true,  false},
        {"older iGPU (gfx1036, no kernels)",    "gfx1036",                 true,  false},
        {"an integrated gfx1201 would not get it", "gfx1201",              true,  false},
        {"CUDA (no gcn name)",                  "",                        true,  false},
    };
    unset("STRATA_IGPU_DEFAULTS");
    unset("STRATA_IGPU_PROMPT_SET");
    // 021-minipc: with nothing set the prompt set is the default of the Radeon 780M only (gfx1103, integrated, not Windows)
    CHECK(kIgpuPromptSetDefaultOn);
    CHECK(igpu_prompt_set_enabled("gfx1103", false));
    CHECK(igpu_prompt_set_enabled("gfx1103:sramecc-:xnack-", false));
    CHECK(!igpu_prompt_set_enabled("gfx1103", true));        // Windows: not tested
    for (const char* g : {"gfx1150", "gfx1151", "gfx1100", "gfx1201", "gfx1152", "", "gfx11030"}) CHECK(!igpu_prompt_set_enabled(g, false));
    CHECK(!igpu_prompt_set_enabled(nullptr, false));
    for (const Mock& m : mocks) {
        const bool want = m.expect_on && std::string(m.gcn).rfind("gfx1103", 0) == 0;
#if defined(_WIN32)
        CHECK(igpu_default_env(m.gcn, m.integrated).empty());
#else
        CHECK(igpu_default_env(m.gcn, m.integrated).size() == (want ? 5u : 0u));
#endif
    }
#if !defined(_WIN32)
    for (const char* k : {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA", "STRATA_HC_UPMIX"}) unset(k);
    CHECK(apply_arch_defaults("gfx1103", true).size() == 5);       // a 780M gets the set with nothing set
    for (const char* k : {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA", "STRATA_HC_UPMIX"}) unset(k);
    CHECK(apply_arch_defaults("gfx1150", true).empty());           // 880M / 890M: no hardware, stays as it was
    for (const char* k : {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA", "STRATA_HC_UPMIX"}) unset(k);
#endif
    // the off switches
    put("STRATA_IGPU_PROMPT_SET", "0");
    CHECK(!igpu_prompt_set_enabled("gfx1103", false));
    CHECK(igpu_default_env("gfx1103", true).empty());
    unset("STRATA_IGPU_PROMPT_SET");
    put("STRATA_IGPU_DEFAULTS", "0");
    CHECK(!igpu_prompt_set_enabled("gfx1103", false));
    unset("STRATA_IGPU_DEFAULTS");
    // opt-in per run for the other arches: STRATA_IGPU_PROMPT_SET=1; the checks below run with it on (every mock arch of the list)
    put("STRATA_IGPU_PROMPT_SET", "1");
    CHECK(igpu_prompt_set_enabled("gfx1150", false));
    put("STRATA_IGPU_DEFAULTS", "0");
    CHECK(!igpu_prompt_set_enabled("gfx1150", false));   // the whole-set off switch wins
    unset("STRATA_IGPU_DEFAULTS");
    for (const Mock& m : mocks) {
        const auto t = igpu_default_env(m.gcn, m.integrated);
        CHECK(t.empty() == !m.expect_on);
        if (m.expect_on) {
            CHECK(t.size() == 5);
            const char* want[] = {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA", "STRATA_HC_UPMIX"};
            for (const char* w : want) {
                bool found = false;
                for (const auto& kv : t) found = found || (kv.first == w && kv.second == "1");
                CHECK(found);
            }
            for (const auto& kv : t) CHECK(kv.first != "STRATA_PF_FUSED" && kv.first != "STRATA_HC_Q8" && kv.first != "STRATA_PF_HCDOWN");
        }
        if (!m.expect_on) std::printf("  (not on) %s\n", m.name);
    }
    CHECK(igpu_default_env(nullptr, true).empty());
    CHECK(igpu_prompt_defaults_arch("gfx1103") && igpu_prompt_defaults_arch("gfx1150") && igpu_prompt_defaults_arch("gfx1151"));
    CHECK(!igpu_prompt_defaults_arch("gfx11030") && !igpu_prompt_defaults_arch("gfx1100") && !igpu_prompt_defaults_arch("gfx1201"));
    // the exact tables are untouched: gfx1103 / gfx1150 stay out of arch_default_env, gfx1151 keeps no bit-changing switch
    CHECK(arch_default_env("gfx1103").empty());
    // the opt-out of the whole set
    put("STRATA_IGPU_DEFAULTS", "0");
    CHECK(igpu_default_env("gfx1103", true).empty());
    unset("STRATA_IGPU_DEFAULTS");
    put("STRATA_IGPU_PROMPT_SET", "0");
    CHECK(igpu_default_env("gfx1103", true).empty());
    put("STRATA_IGPU_PROMPT_SET", "1");
    // apply: a discrete card sets nothing, an iGPU sets the five, a switch the user set (even to 0) is kept
    const char* five[] = {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA", "STRATA_HC_UPMIX"};
    for (const char* k : five) unset(k);
    CHECK(apply_arch_defaults("gfx1100", false).empty());
    CHECK(apply_arch_defaults("gfx1201", false).empty());
    CHECK(apply_arch_defaults("gfx1103", false).empty());   // integrated == false: a mixed box (iGPU + dGPU) keeps the old defaults
    for (const char* k : five) CHECK(std::getenv(k) == nullptr);
    put("STRATA_HC_UPMIX", "0");
    const auto on = apply_arch_defaults("gfx1103", true);
    CHECK(on.size() == 4);
    for (const char* k : {"STRATA_HIP_WMMA", "STRATA_PF_GEMM", "STRATA_PA_FAST", "STRATA_SELECT_WMMA"})
        CHECK(std::getenv(k) != nullptr && std::string(std::getenv(k)) == "1");
    CHECK(std::getenv("STRATA_HC_UPMIX") != nullptr && std::string(std::getenv("STRATA_HC_UPMIX")) == "0");
    for (const char* k : five) unset(k);
    unset("STRATA_IGPU_PROMPT_SET");
    // 021-minipc: UMA mode is NOT a default (slower, resets not fixed): off with nothing asked; the logic below runs with on=true
    CHECK(!kIgpuUmaArenaDefaultOn);
    CHECK(!igpu_uma_arena_default("gfx1103", true, false));
    // 021-minipc: UMA mode (the expert arena stays unregistered) is chosen for the Radeon 780M only: gfx1103 + integrated + not Windows
    unset("STRATA_UMA_REGISTER");
    CHECK(igpu_uma_arena_default("gfx1103", true, false, true));
    CHECK(igpu_uma_arena_default("gfx1103:sramecc-:xnack-", true, false, true));
    CHECK(!igpu_uma_arena_default("gfx1103", true, true, true));      // Windows: not measured
    CHECK(!igpu_uma_arena_default("gfx1103", false, false, true));    // a gfx1103 that is not integrated
    for (const char* g : {"gfx1150", "gfx1151", "gfx1152", "gfx1100", "gfx1101", "gfx1200", "gfx1201", "gfx1030", "gfx11030", "", "gfx90a"})
        CHECK(!igpu_uma_arena_default(g, true, false, true));
    // 0.2.1.2: Strix Halo on Windows is unregistered by default (on or off), nothing else on Windows; Linux gfx1151 unchanged
    CHECK(igpu_uma_arena_default("gfx1151", true, true));
    CHECK(igpu_uma_arena_default("gfx1151:sramecc+:xnack-", true, true));
    CHECK(!igpu_uma_arena_default("gfx1151", false, true));           // not integrated
    CHECK(!igpu_uma_arena_default("gfx1151", true, false));           // Linux
    for (const char* g : {"gfx1150", "gfx1152", "gfx1103", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx1030", "", "gfx90a"})
        CHECK(!igpu_uma_arena_default(g, true, true));
    CHECK(!igpu_uma_arena_default(nullptr, true, true));
    put("STRATA_UMA_REGISTER", "1");
    CHECK(!igpu_uma_arena_default("gfx1151", true, true));
    unset("STRATA_UMA_REGISTER");
    CHECK(!igpu_uma_arena_default(nullptr, true, false, true));
    put("STRATA_UMA_REGISTER", "1");
    CHECK(!igpu_uma_arena_default("gfx1103", true, false, true));     // the user registers the arena as before
    unset("STRATA_UMA_REGISTER");
    // both platforms: no Windows guard on the iGPU set (the owner wants it there too; untested on Windows is only said in the log)
    std::printf(fails ? "arch_defaults_test: %d FAILED\n" : "arch_defaults_test: OK\n", fails);
    return fails ? 1 : 0;
}
