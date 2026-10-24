#!/bin/bash
# Regression test for the Arc A750 prompt deadlock (0.1.48): a 2,744-token prompt (a docs excerpt in chat framing, token ids in
# data/prompt_deadlock_2744.tok) made the prompt path hang in the Level Zero driver (the thread spun in sched_yield inside
# libze_intel_gpu while the GPU was idle) when the expert arena was DMA'd straight from pageable memory.  The engine now copies
# every expert that is not USM host memory through the stager's pinned buffers.
#
#   STRATA_BIN=build-jit/strata MODEL_ARGS="--pack ... --native ... --ple-gguf ... --mtp ... --kv int8 --max-context 8192 ..." \
#       sycl/tools/test_prompt_deadlock.sh
#
# MODEL_ARGS are the engine arguments of the box (everything but --tokens-file/--max-new); the A750 needs its usual environment
# (render group, ONEAPI_DEVICE_SELECTOR, IGC_EnableDPEmulation=1, ...).  Passes when the prefill line is printed within
# DEADLOCK_TIMEOUT seconds (default 900; the prompt takes about 60 s on an A750 after the start-up).  Exit 0 pass, 1 fail, 2 usage.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${STRATA_BIN:-}
[ -n "$BIN" ] && [ -x "$BIN" ] && [ -n "${MODEL_ARGS:-}" ] || { echo "usage: STRATA_BIN=<strata> MODEL_ARGS=\"...\" $0"; exit 2; }
OUT=$(mktemp)
timeout "${DEADLOCK_TIMEOUT:-900}" "$BIN" $MODEL_ARGS --tokens-file "$HERE/data/prompt_deadlock_2744.tok" --max-new 4 --greedy > "$OUT" 2>&1
rc=$?
if grep -q "^prefill  *2743 tokens" "$OUT"; then
    echo "PASS: the 2,743-token prompt was read ($(grep -E '^prefill' "$OUT" | head -1 | tr -s ' '))"
    rm -f "$OUT"; exit 0
fi
echo "FAIL: no prefill line (engine exit code $rc); output kept in $OUT"
tail -5 "$OUT"
exit 1
