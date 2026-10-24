// 151-b70: the default choice of the prompt-expert chain (strata::prefill::i8::choose_mode), no GPU needed.
//   unset            : auto on a Battlemage device whose kernels are available, else the FP16 chain
//   STRATA_PREFILL_MMQ=0 / 1 / auto : off / forced / auto, whatever the device
#include <cstdio>
#include "strata/prefill/esimd_moe.hpp"

using strata::prefill::i8::choose_mode;
static int bad = 0;
#define CHECK(cond, what) do { if (!(cond)) { std::printf("FAIL: %s\n", what); ++bad; } } while (0)

int main() {
    // unset or empty
    CHECK(choose_mode(nullptr, true, true) == 2, "unset on Battlemage -> auto");
    CHECK(choose_mode("", true, true) == 2, "empty on Battlemage -> auto");
    CHECK(choose_mode(nullptr, true, false) == 0, "unset on Battlemage without kernels (AOT stub, failed probe) -> FP16");
    CHECK(choose_mode(nullptr, false, true) == 0, "unset on Alchemist / any non-Battlemage device that can run the kernels -> FP16");
    CHECK(choose_mode(nullptr, false, false) == 0, "unset elsewhere -> FP16");
    // explicit
    CHECK(choose_mode("0", true, true) == 0, "=0 on Battlemage turns it off");
    CHECK(choose_mode("1", true, true) == 1, "=1 forces int8");
    CHECK(choose_mode("1", false, true) == 1, "=1 forces it on a non-Battlemage device too (the caller still needs available())");
    CHECK(choose_mode("auto", false, true) == 2, "=auto");
    CHECK(choose_mode("Auto", true, true) == 2, "=Auto");
    CHECK(choose_mode("0", false, false) == 0, "=0 elsewhere");
    CHECK(choose_mode("00", true, true) == 0, "=00 is 0");
    CHECK(choose_mode("2", true, true) == 1, "any other number forces int8");
    CHECK(choose_mode("off", true, true) == 0, "an unparsable word is 0, as atoi had it");
    if (bad == 0) std::printf("esimd_moe_mode_test: all OK\n");
    return bad == 0 ? 0 : 1;
}
