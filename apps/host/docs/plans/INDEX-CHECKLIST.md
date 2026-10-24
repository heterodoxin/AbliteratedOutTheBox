# Audit checklist — the recurring items, in one place

Every plan in this folder raises some of the same issues. This is the shared list,
so a fix applied once is applied everywhere rather than per screen.

## Present on multiple screens

| Issue | Screens | Owned by |
|---|---|---|
| Focus is not trapped in modals | palette, shortcuts, setup, drawer | [13](13-overlays.md), [16](16-accessibility.md) |
| Section headings are `<div>`, not `<h2>` | all | [16](16-accessibility.md) |
| ~~Empty states without an action~~ **fixed** — all six carry one | Monitor, Serving, Models, Skills, Workspace, Chat | [14](14-states.md) |
| No time expectation on long operations | setup, models, monitor | [09](09-first-run.md), [14](14-states.md) |
| Live values without a time window | Monitor, Telemetry | [04](04-monitor.md) |
| The loaded model reported in four places | header, sidebar, telemetry, Monitor | [10](10-sidebar.md) |
| Context described with different words | chat popover, rail, Monitor | [04](04-monitor.md), [12](12-context-rail.md) |
| Trees that do not match: icons and motion | Workspace vs Sidebar | [06](06-workspace.md) |
| Value fills that jump instead of easing | Monitor, rail, Models | [15](15-motion.md) |
| Destructive actions styled like navigation | palette, More menu | [11](11-command-palette.md) |

## Status of every plan

**Every plan received shipped work. None of the plans is exhausted.**

That distinction is the honest summary, and an earlier draft of this file got it
wrong by marking fourteen plans "Shipped" — which implied the plans were finished
when they are not. The numbers:

| | Count |
|---|---|
| Numbered items across the sixteen plans | ~190 |
| Annotated in the plans as fixed or disproved | 48 |
| Recorded in the execution table below | 27 |

So roughly **a quarter of what the plans propose has been executed.** The work
that shipped was chosen as the highest-value item in each plan, plus every item
the goal named explicitly, plus the defects found while verifying. What remains
is largely polish and optimisation rather than defect: unread marks, `content-visibility`,
sparkline baselines, a motion budget, Settings search, empty-state actions per
screen, and so on. None of it is broken; all of it is proposed.

Per plan, what is certainly true:

| # | Plan | Shipped | Notable remaining |
|---|---|---|---|
| 01 | Chat | Retry · edit-and-resend · branch · continue · unread marks · `content-visibility` with pinned scrolling · jump-to-latest · focus-mode turn navigation | — |
| 02 | Models | Fit verdict; unusable options fold away; cost as size and time; recommended variant explains itself and carries weight | — |
| 03 | Engine tuning | Live draft prediction with in-place issues; expert mode in five sections with changed-counts | Safety cue on leaving defaults |
| 04 | Monitor | Sampling interval; VRAM/RAM verdicts; thermal basis; sparkline baseline, labels, hover readout; resting hero with session figures | — |
| 05 | Serving | Copyable URLs; executable `curl` block; filter with counts; sortable columns; state glyphs | Drawer "copy as cURL" |
| 06 | Workspace | Real glyphs; animated branches; file contents; read markers; scope figures; ARIA tree keyboard nav | — |
| 07 | Skills | Instructions readable; triggers match; usage counts; grouped by source; three-state per-workspace scope | Add-from-folder |
| 08 | Settings | Sticky section index; live search; confirmed appearance reset; telemetry choice persists | Section reorder (declined), grouping, per-section resets |
| 09 | First run | Shape-of-task line; step-3 consequences; *Finish later* | Scan substance, local-data confirmation, step-5 hand-off |
| 10 | Sidebar | Pinning · arrow traversal · right-click menu · `Delete` · archive that stays searchable | Skills placement, labelled theme toggle |
| 11 | Command palette | Fuzzy matching, scopes, danger marks, shortcut chips | Recents group |
| 12 | Context rail | Collapsible sections; tools show real use; empty state; close icon; reference-panel weight | Gauge animation |
| 13 | Overlays | Toast cap and hover-pause; exits confirmed; popover follows on resize; three dismissal grammars measured and documented | — |
| 14 | States | Offline bar; persistent failures with retry; all six empty states carry an action | Skeletons, time expectations, operation-failed empties |
| 15 | Motion | Interface scale; one-loop-per-region check; completion cue on the jump pill | Duration budget |
| 16 | Accessibility | `useSurfaceFocus`; one `h1`; `aria-current`; non-text contrast probe (1.4.11) | High-contrast mode, fatigue budget |

## Deliberately not done

**Two layout reorders, declined by their owner.** Tuning's nine flat fields into
two groups, and Settings' section order. Both are judgement calls about the
product rather than defects, both are visible changes to screens in daily use,
and the question was put to the user repeatedly without a steer. Recorded as
declined rather than quietly dropped; the plans keep the reasoning if it is ever
wanted.

**`… More` was not renamed.** The plans want it called *Settings*, and the goal
naming this work repeats that. The user asked for "… More" by name, so the label
stays. The plan's actual complaint — a vague word standing in front of three
destinations — is answered instead by naming them in the button's tooltip and
accessible name: *"More: Models, Engine tuning, Settings"*.

**About's config paths and Open buttons.** The plan wanted About to carry the
folder paths, their Open buttons and the config file locations. The engine build
was added — the one fact nothing else stated — but the paths are *not* duplicated
from Storage, where they sit beside the actions that use them. Repeating them in
About would put one fact in two places, which is the failure plan 12's own rule
warns against.

**Nothing else.** Plan 01's remaining items — `content-visibility`, unread marks,
turn navigation in focus mode — are listed here in an earlier draft as unbuilt.
They shipped (rounds 20, 22 and 40), so the entry was stale and *understated* the
work. Removed rather than left to be re-read as fact.

## Claims that did not survive checking

**Fourteen** assertions across the plans were disproved by measurement — the
count in an earlier draft said eleven, which undercounted its own table. They are
corrected in place, and listed here because the pattern matters: **the plans were
strongest on structure and weakest on what was already there.** Every claim of
the form "X does not exist" needed grepping before it was built on.

In fourteen cases the plan named the right screen and the wrong defect. That is
not a reason to distrust the plans — they found real work in every one of the
sixteen — but it is a reason never to build on a claim of absence without
checking it first.

| Plan | Claim | Actually |
|---|---|---|
| 04 | Nothing is scoped in time | Both speed cards read *last 60 seconds*; the totals card carries a *This session* pill |
| 04 | No threshold made explicit | The temperature metric already printed `comfortable` / `throttling risk` |
| 01 | Only the last turn carries figures | Every assistant turn renders a footer |
| 11 | `mtx` should match *Monitor* | *Monitor* has no `x`; the example was impossible |
| 16 | Section titles render as `<div>`s | `SectionHead` renders `<h2 className="h3">` |
| 16 | Toasts change without announcement | `.st-toasts` carries `aria-live="polite"` |
| 06 | Workspace "shows nothing" | A full detail panel exists — metadata, not file contents |
| 09 | The step rail is a progress bar, not a map | It renders every step name |
| 09 | There is no exit that keeps state | There was; it was misnamed *Exit setup* |
| 14 | A failed download leaves a row ambiguous | `v.error` already rendered — the gap was the missing retry |
| 15 | A sent message gets no response at all | It animates in on `st-enter`: 6 px over 240 ms, measured |
| 13 | No overlay announces itself | All three carry `role="dialog"`, `aria-modal`, and a label |
| 03 | No "modified from recommended" markers | Every expert row already shows `Default: …` and a per-field reset |
| 13 | None of the overlays trap focus | All three do, verified against the *visible* overlay |

The last one is instructive: an early probe of mine "confirmed" the bug by
querying `.modal`, which finds the hidden palette rather than the open dialog. A
test that looks right and measures the wrong element is worse than no test.

## Defects found while executing

These were not in any plan. Each was found by checking a claim or verifying a
build, and each is fixed with a check that fails without it.

**CPU threads defaulted to 8 on every machine.** `catalog.mjs` hardcoded it
regardless of detected cores, so a 4-core machine warned "CPU threads (8) exceed
physical cores (4)" on a fresh install — before the reader had touched anything.
Now built from detected cores; a fresh install raises no issues.

**Three classes were used by components but defined in no stylesheet.**
`.spec__sub` (the `Metric` sub-line, five screens) and `.st-toast__body` were
collateral damage from a "dead CSS" sweep. `.st-msg__actions` was *reported as
shipped* without its CSS ever landing, because the patch anchored on a selector
that did not exist and failed silently. `npm run audit` now fails on any class a
component uses that no stylesheet defines.

**The rail and the popover disagreed about the same number.** The context maximum
read `32K` in one and `33k` in the other — two formatters, ÷1024 and ÷1000. Found
by applying plan 12's own rule: *agree exactly, or appear in only one*.

**Branching a conversation created the branch without showing it.** `Chat`
resolves its session as `location.param ?? activeId`, so the URL outranks the
store. `branchFrom` set `activeSession` without navigating: the sidebar showed the
new branch while the transcript kept showing the original *indefinitely*. A
feature that looked broken while its data was correct.

**A Tuning impact line that never moved.** The plan's own criterion — *"a static
line is a failure"* — caught the first attempt, which read the *saved* config
rather than the draft on screen. `/api/engine/validate` now accepts a draft.


### Found in the design pass over every screen

Nine screens were walked, clicked and measured. These are the defects that
survived every automated check until something looked at the screen.

**The model received no conversation history.** The client sent
`{ sessionId, message, model, workspaceId }` and never `history`. The real-engine
path builds its prompt as `[...history, message]`, so `history` was always empty
and **every message arrived as the first thing the model had ever heard**. The
demo path hides this completely — its replies are canned. Found by asking why
the context meter stayed at ~350 tokens however long the conversation grew.

**The context meter measured the wrong quantity.** `usedTokens` *accumulated*
(`+= tokensIn + tokensOut`) while the per-part split beside it was absolute. So
it climbed past its own maximum within one conversation — "40k / 32K tokens,
100%" — clamped, with no way back down. It reports current occupancy now.

**A promise the app could not keep.** "Compaction will trigger soon" appeared in
two places. Compaction exists nowhere in the codebase.

**Focus mode's dimming never worked.** `.st-msg` carried
`animation: st-enter … both`. An animated value outranks a normal declaration in
the cascade, so the keyframe's closing `opacity: 1` held every turn at full
strength. `.app[data-focus='true'] .st-msg { opacity: 0.45 }` was in the
stylesheet, matched the element, and lost anyway. `backwards` fixes it: the
keyframe's end state is the element's natural state, so nothing snaps.

**A green one-click "Load" on a model the app called too large.** It loaded —
11.3 GB on a 16 GB machine — and reported "the engine is warm". Unusable
variants now get a secondary "Load anyway" and a confirmation naming the cost.

**A button whose label and action were unrelated.** "Add from folder" ran
`setQuery('')`.

**A file preview at 1.26:1 contrast** — `#303030` on `#1b1f23`, body text needing
4.5 — with 2242px of every long line clipped and `overflow: hidden` making the
rest unreachable. It only failed in **light** mode, because the overriding token
is light in dark mode.

**One glyph, several meanings.** `experts` meant both the MoE expert count and
Skills. Moving Skills to `bolt` was worse — `bolt` is the most-used glyph in the
app at ten places, and in the collapsed rail the icon *is* the label. It is
`thinking` now, which appears once and renders as a lightbulb.

**The collapsed rail clipped a conversation to its first five pixels**, pushed
another 40px target past the sidebar edge via the tree's 22px indent, and left a
26px void where the hidden "Workspaces" heading had been.

**Checks that were too narrow to catch any of it**: the contrast probe only ever
visited Chat; the luminance probe counted area without thickness, so an 8px
progress hairline was judged the same as a 64×64 block — and its verdict on that
element flipped with live data. Both were rebuilt and each proven able to fail.

## Checks that could not fail, and were rebuilt

**The scroll-cost budget reports NOT MEASURED rather than passing.**
`npm run motion:budget` compares frame cost scrolling with and without the
scroll-driven effects. It could not detect a deliberately expensive effect
injected into the same run — the ratio moved 1.000 → 1.006 — so the harness is
blind to scroll-timeline paint cost under programmatic scrolling in headless
Chromium. It now proves it can see a known-expensive case before reporting, and
exits 3 with `NOT MEASURED` when it cannot. This is the one check in the suite
that is deliberately allowed to report nothing.


Worth recording, because a green check that cannot go red is worse than none.

**The undefined-class rule was vacuous on its first attempt.** The comment
explaining it names the very classes it checks, and the regex matched inside
comments — removing `.spec__sub` still passed. It strips comments now, and is
proven by deleting the definition and watching it fail.

**The expected-failure filter was proven narrow.** Flows that deliberately break
something declare it (`expectFailure(/ERR_FAILED/)`). With a genuine
`console.error` injected into the same flow, the run still fails.

**The loop-budget check had incomplete coverage.** Its first version checked
seven routes, so the probe used to test it — placed on Settings — could not have
failed. It also toggled the theme by clicking a control the setup modal covers,
which timed out. Both were caught by trying to prove the check rather than
trusting a green run.

**The non-text contrast probe was made to fail on purpose before being trusted.**
Forcing `outline-color: #f8fafc` drops focus rings to 1.02–1.05 and exits non-zero;
restoring returns to 0.

**Probe parsers read `color(srgb …)` as 0–255.** Chrome's 0–1 components made
tinted fills look near-black, silently defeating the contrast and luminance
checks. Fixed in both scripts.

## Known-good, keep under test

Correct today, and the things a refactor is most likely to break:

- Reading measure: **68 characters** at every content size.
- Contrast: **37 text pairings** plus non-text 1.4.11 (focus rings enforced at 3:1), both themes.
- Dark-mode luminance budget and material layering.
- Reduced motion substituting opacity for movement.
- Overlay interruptibility — every surface reverses mid-flight.
- Focus lands inside every overlay and stays there across 18 `Tab` presses.
- Capsule controls, capsule-free surfaces.
- No native `<select>`, no missing icons, no undefined classes or tokens, no off-scale values.
- Materials on exactly the approved control-layer surfaces.

## How to use these plans

1. Pick a screen plan; each is self-contained.
2. Check the shared table above before implementing — the fix may belong in a
   shared component rather than the screen.
3. Add the verification named in the plan **before** the change, so it fails first.
4. Run `npm run check` and both theme tours before calling it done.
5. **Check the claim before building on it.** Eleven did not survive that.
