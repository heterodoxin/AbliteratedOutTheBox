# 04 · Monitor

**Route** `#/monitor` · **File** `src/screens/Monitor.tsx` (284 lines)

The screen that answers "is it working, and is my computer coping?" — live. It
carries the most `Metric`s (10), the most `Bar`s (5), two `Gauge`s and two
`Spark`s, which makes it the densest screen and the one most at risk of reading
as a wall of numbers.

## What it is today

- **A waiting/working hero** — the current phase in plain words, with prompt
  progress (`0 / 8 tokens`) while reading.
- **Output speed and prompt reading** — two large numeric metrics with sparklines.
- **Hardware card** — GPU load bar, memory, GPU temperature with a limit.
- **Context used** — percentage and tokens, with a "room to keep going" note.
- **Model resources** — expert cache, layers on GPU.
- **Totals** — a four-metric summary of the session (tokens, requests, avg
  throughput, time).

## Where it falls short

1. ~~Nothing is scoped in time.~~ **Checked and wrong.** Both speed cards already
   read `Current response · last 60 seconds` and `Current prompt · last 60
   seconds`, and the totals card carries a `This session` pill in place of a start
   date. What genuinely lacks a window is the *hardware* block — VRAM, RAM and
   temperature describe the instant with no stated sampling interval.
2. **The hero is idle most of the time.** The screen's largest element is the
   phase label, which reads "Waiting for a request" whenever nothing is running —
   the common case for anyone who opens this screen to look at history.
3. **Live numbers have no steadying.** Values update every second and the digits
   shift width. Tabular figures are used in places but not everywhere, so rows
   twitch.
4. ~~The hardware card has no thresholds made explicit.~~ **Partly wrong.** The
   temperature metric already prints a verdict word (`comfortable` /
   `throttling risk`) beside the limit, so it is not colour-alone. The gap is that
   VRAM and RAM have no equivalent verdict, and the temperature threshold is a
   bare `limit - 8` with no stated basis.
5. **Context usage is duplicated** with the chat telemetry popover, at different
   precisions and with different labels (`Context used` vs `of memory`).
6. **No time axis anywhere.** History exists in `MonitorState.history` (a rolling
   sample buffer) but only ever renders as unlabelled sparklines.
7. **The totals card has no start date shown**, though `totals.since` exists.

## The plan

### Substance — make every number mean something
1. ~~Extend the window labelling to the hardware block.~~ **Fixed.** The block
   header now reads *Sampled every second*, and VRAM and RAM carry the same
   plain-word verdict the temperature metric already had (*plenty free*,
   *comfortable*), so no reading depends on colour alone.
2. **Give the hero a resting state worth looking at.** When idle, it becomes a
   compact session summary (last request, throughput trend, context) instead of a
   large empty "Waiting for a request".
3. ~~Give VRAM and RAM a verdict.~~ **Done**, and the temperature's threshold is
   now stated rather than implied: `Limit 105°C · flags above 97°C ·
   comfortable`. The margin is a named constant with its reasoning attached, so
   the reader can disagree with it.
4. **Reconcile context vocabulary across the app.** One phrase, one precision,
   shared with the chat popover. Pick `of memory` and use it in both places, or
   `of context used` in both.

### Hierarchy
5. **Three tiers, explicitly.** *Now* (phase + throughput), *This machine*
   (hardware), *Since launch* (totals). Today all three sit at the same weight.
6. **One hero.** The phase/throughput block is the hero; the totals card drops a
   type step and becomes a footer strip.

### Fluidity
7. **Sparklines gain a shared baseline and a hover readout**, so the curve can be
   interrogated rather than only glanced at.
8. **Numbers stop twitching** — `font-variant-numeric: tabular-nums` on every live
   figure, verified rather than assumed.
9. **A visible time axis** on at least one chart, using `history`, so "is this
   normal?" is answerable.

## Motion

| Interaction | Spec |
|---|---|
| Value changing | No transition on the digit itself; the bar/gauge it drives animates `--st-t-local` |
| Phase changing | Label cross-fade, `--st-t-control` — already correct |
| Hero idle → working | `st-enter` on the working block, `--st-t-local` |
| Sparkline appending | No animation; the line simply extends |

Under reduced motion every one of these becomes a value swap with no interpolation.

## Verification

- Assert **no live metric renders without a window label**.
- Assert every threshold-bearing metric renders a word alongside its number —
  greyscale screenshot must still be readable.
- Assert `tabular-nums` computes on every `.num` inside the screen.
- Extend `flow monitor` to capture idle and generating states in both themes.
