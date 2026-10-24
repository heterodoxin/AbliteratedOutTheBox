// src/program/batch_groups_test.cpp - how many pipelined slot groups --batch runs with (header-only, CPU).
//
//   1. `auto` picks the most groups, at most one per stage, that divide the slots;
//   2. a layer split with --batch >= 2 and no --batch-groups means `auto` (0.1.41);
//   3. --batch-mtp (or --adapt-async 1) turns that default into one group, and says which feature did it;
//   4. a number or `auto` given explicitly is honoured, with or without those features;
//   5. no layer split: the number given (1) stays.
#include "strata/program/batch_groups.hpp"

#include <cstdio>

namespace bg = strata::program::batch_groups;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bg::Input in(int batch, int stages) {
    bg::Input i;
    i.batch = batch;
    i.later_stages = stages;
    return i;
}
}  // namespace

int main() {
    std::printf("batch_groups_test\n");
    check(bg::auto_groups(0, 4) == 1, "auto: no layer split, one group");
    check(bg::auto_groups(1, 2) == 2 && bg::auto_groups(1, 4) == 2 && bg::auto_groups(1, 3) == 1,
          "auto: two GPUs, 2 groups when the slots divide by 2");
    check(bg::auto_groups(3, 8) == 4 && bg::auto_groups(3, 6) == 3 && bg::auto_groups(3, 2) == 2,
          "auto: at most one group per GPU, and a divisor of the slots");

    {   // the 0.1.41 default
        const auto r = bg::resolve(in(2, 1));
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None, "layer split, --batch 2, nothing given: 2 groups");
    }
    {
        const auto r = bg::resolve(in(1, 1));
        check(r.groups == 1 && !r.from_auto, "--batch 1: no default");
    }
    {
        const auto r = bg::resolve(in(2, 0));
        check(r.groups == 1 && !r.from_auto, "no layer split: one group");
    }

    {   // the production shape: parallel 2 on a 2-GPU split with --batch-mtp
        bg::Input i = in(2, 1);
        i.batch_mtp = true;
        auto r = bg::resolve(i);
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None && r.mtp_dropped,
              "--batch-mtp, nothing given: the pipeline wins, --batch-mtp is dropped");
        i.mtp_serial = true;
        r = bg::resolve(i);
        check(r.groups == 1 && r.from_auto && r.serial == bg::Serial::BatchMtp && r.would_be == 2 && !r.mtp_dropped,
              "STRATA_BATCH_MTP_SERIAL: one group, because of --batch-mtp");
        check(bg::serial_what(r.serial)[0] == '-', "the reason names the feature");
    }
    {
        bg::Input i = in(2, 1);
        i.adapt_async = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.serial == bg::Serial::AdaptAsync, "--adapt-async 1, nothing given: one group");
    }
    {
        bg::Input i = in(2, 1);
        i.batch_mtp = true;
        i.adapt_async = true;
        check(bg::resolve(i).serial == bg::Serial::AdaptAsync, "both: --adapt-async 1 still gives one group");
        i.mtp_serial = true;
        check(bg::resolve(i).serial == bg::Serial::BatchMtp, "both with STRATA_BATCH_MTP_SERIAL: --batch-mtp is named");
    }
    {   // nothing to give way: the slots do not divide, so auto is one group anyway and there is nothing to say
        bg::Input i = in(3, 1);
        i.batch_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.serial == bg::Serial::None && r.would_be == 1, "auto would be 1 group anyway: no reason");
    }
    {   // PROPOSED NVIDIA default: two cards, nothing given: serial windows (measured +12..16% on 2x A4000)
        bg::Input i = in(4, 1);
        i.nv_serial_faster = true;
        auto r = bg::resolve(i);
        check(r.groups == 1 && r.from_auto && r.serial == bg::Serial::TwoCardsSerialFaster && r.would_be == 2,
              "nv_serial_faster, 2 cards, 4 slots, nothing given: one group, named");
        check(bg::serial_what(r.serial)[0] == 's', "the reason names the measurement");
        check(bg::resolve(in(4, 1)).groups == 2, "flag off: the pipeline default stays");
        i = in(8, 1); i.nv_serial_faster = true;   // measured too: +15%
        check(bg::resolve(i).groups == 1, "8 slots (measured +15%): serial too");
        i = in(4, 3); i.nv_serial_faster = true;
        check(bg::resolve(i).groups == 4, "4 cards (not measured): the pipeline stays");
        i = in(4, 1); i.nv_serial_faster = true; i.batch_mtp = true;
        check(bg::resolve(i).groups == 2 && bg::resolve(i).mtp_dropped, "with --batch-mtp the 0.1.43.1 rule is unchanged");
        i = in(4, 1); i.nv_serial_faster = true; i.set = true; i.auto_given = true;
        check(bg::resolve(i).groups == 2, "--batch-groups auto given: the pipeline");
        i.auto_given = false; i.given = 2;
        check(bg::resolve(i).groups == 2, "--batch-groups 2 given: 2");
    }
    {   // explicit values are honoured
        bg::Input i = in(4, 1);
        i.set = true;
        i.given = 2;
        i.batch_mtp = true;
        i.adapt_async = true;
        const auto r = bg::resolve(i);
        check(r.groups == 2 && !r.from_auto && r.serial == bg::Serial::None, "--batch-groups 2 with --batch-mtp: 2");
        i.given = 1;
        check(bg::resolve(i).groups == 1, "--batch-groups 1: 1 (the opt-out)");
        i.given = 4;   // not checked here: generate.cpp refuses what does not divide
        check(bg::resolve(i).groups == 4, "--batch-groups 4 is returned as given");
    }
    {
        bg::Input i = in(2, 1);
        i.set = true;
        i.auto_given = true;
        i.batch_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None,
              "--batch-groups auto with --batch-mtp: auto asked for the pipeline, 2 groups");
    }
    {
        bg::Input i = in(2, 0);
        i.set = true;
        i.given = 2;
        check(bg::resolve(i).groups == 2, "an explicit number without a split is returned (the engine warns)");
    }

    std::printf("%s\n", g_fail == 0 ? "batch_groups_test: all passed" : "batch_groups_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
