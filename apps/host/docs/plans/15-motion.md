# 15 · Motion and fluidity

**Across the app** · **Rules** [interaction-guidelines.md](../interaction-guidelines.md) · **Harness** `scripts/motion.mjs`

Motion is the layer that makes the app feel considered. It is also the layer most
able to make a machine running a language model feel slow, so it is rationed
deliberately.

## What is today

- **One scale**: 100 / 160 / 240 / 320 / 460 ms, one standard curve, springs only
  for direct manipulation.
- **Twenty keyframes**, down from twenty-three, most entrances collapsed into two
  shared recipes.
- **Route changes morph** through `startViewTransition`, including a shared
  element where one exists.
- **Three grid-track animations** — sidebar rail, context rail, workspace tree.
- **Interruptible overlays** throughout, proven by frame-sampled contact sheets.
- **Reduced motion substitutes opacity for movement**, verified: `transform: none`
  and `opacity: 1` under the OS setting.
- **Nothing animates during generation** except the caret and the status dot.

## Where it falls short

1. **Progress and gauge fills are not on the scale everywhere.** Some jump to
   their value; some ease. The inconsistency is visible when two update together.
2. **The `st-enter` recipe carries nearly every entrance**, which makes unrelated
   surfaces move identically. A palette and a toast should not arrive the same way.
3. **No motion exists for the most common action** — sending a message. The user
   message simply appears.
4. ~~Nothing acknowledges a completed generation when scrolled away.~~ **Fixed.**
   A reply that finishes off screen had no way to say so — the text simply
   stopped arriving. The jump-to-latest pill now dips once, **on opacity alone**,
   so the cue is noticed without anything moving and without pulling the eye off
   what the reader is actually reading. Verified: the cue fires with `st-cue` on
   the `--st-t-choreo` token. Under reduced motion it becomes a colour change
   rather than a pulse, so the acknowledgement is not lost.
5. **Two loops can run simultaneously** — the busy dot and a progress bar — where
   the rule allows one moving thing per region.
6. ~~No motion budget is measured.~~ **Fixed for the loop budget.**
   `scripts/motion.mjs` now checks **one loop per region** across nine routes in
   both themes, grouping by the nearest card, panel or table so a spinner in a
   list row is judged against that row rather than the whole screen.

   Proven non-vacuous: a second loop inside a settings card reports
   `light · settings: 20 loops at once in "st-card.st-card--flush"` and exits 1;
   removing it returns to 0.

   **Two bugs in the check itself, both found by trying to prove it.** It only
   covered seven routes, so the probe I first used to test it — on Settings —
   could not have failed. And it toggled the theme by clicking a control that
   the setup modal covers, which timed out on that route. Both fixed; the check
   now sets the theme the way the app does and covers every reachable route. Frame cost is asserted only for lensing,
   which was declined. The rest is assumed cheap.
7. **The reading-progress hairline animates on scroll but nothing else does**,
   though the header hairline and title collapse share the same trigger — three
   scroll-driven effects on one element tree, never measured together.

## The plan

### Substance
1. **Finish the scale's application**: every value-driven fill eases on
   `--st-t-local`; every state change is a token on `--st-t-control`; nothing
   jumps except a value the user typed.
2. **Differentiate entrances by kind**, not by taste: *transient* surfaces lift
   (6 px), *modal* surfaces scale (0.98), *inline* content fades (0 px). Three
   recipes, each with a reason.
3. ~~A send acknowledgement.~~ **Wrong — it already had one.** The claim was that
   "the most frequent action in the app currently has no response at all", but a
   sent message animates in on `st-enter`: **6 px lift over 240 ms**, measured.
   The plan's proposed values (4 px on `--st-t-control`, 160 ms) differ by degree,
   not by existence, and the existing values sit on the scale. No change made.
4. ~~A completion cue.~~ **Done**, on opacity only.

### Discipline
5. **One loop per region, enforced**: the audit gains a rule that two elements
   with running infinite animations may not share a stacking context.
6. **A motion budget check** covering the three scroll-driven effects together:
   p95 frame time while scrolling must stay within 1.2× of a build with them
   disabled.

## Motion

The reference table the audit enforces:

| Band | Token | Travel | Use |
|---|---|---|---|
| micro | 100 ms | ≤ 4 px | hover, press, colour |
| control | 160 ms | ≤ 16 px | menus, rows, small transforms |
| local | 240 ms | ≤ 24 px | drawers, disclosure, value fills |
| surface | 320 ms | — | rails, routes, whole panels |
| choreographed | 460 ms | — | first run only |

## Verification

- `npm run motion` stays green: every surface reverses mid-flight.
- New: **no two infinite animations share a stacking context**.
- New: **p95 scroll frame time within 1.2×** of the scroll-effects-disabled build.
- Reduced-motion sweep asserts `transform: none` on every moving element and
  `opacity: 1` on every fading one.
