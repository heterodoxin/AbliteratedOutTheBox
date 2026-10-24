# 03 · Engine tuning

**Route** `#/tuning` · **File** `src/screens/Tuning.tsx` (413 lines)

Where a beginner mode and an expert mode have to coexist without either feeling
like a compromise. It is the most control-dense screen in the app.

## What it is today

- **Beginner / Expert tabs** — a segmented control.
- **Nine `Field` rows** in beginner mode: context length, memory mode, image
  support, free-CPU policy, and others, each with a hint line and a `Select`.
- **An "Expected effect" side card** — the consequences of the current settings,
  in plain language.
- **A pinned action** — "Save & restart engine".
- **Expert mode** — a settings table plus raw parameter rows.

## Where it falls short

1. ~~No sense of scale.~~ **Fixed.** The context field now carries the live
   figure, and the host predicts the *draft* rather than the saved config — which
   is what makes the line a preview of the choice instead of a restatement of the
   status quo. Verified: setting 128K changes the line from `about 1.9 GB` to
   `about 2.4 GB` and raises the trained-context warning in place.
2. **Every field has equal weight.** Context length and image support are not
   equally consequential, and the layout says they are.
3. ~~The consequence is not shown at the moment of change.~~ **Fixed** for the
   three beginner fields. The plan's own criterion — "a static line is a
   failure" — is what caught the first attempt: reading `engine.predicted` showed
   the saved config, so the line never moved. `/api/engine/validate` now accepts
   a draft and returns its prediction.
4. **Expert mode is a wall.** A table plus raw rows, with no grouping and no
   explanation of which values are safe.
5. **"Save & restart engine" is the only exit** — restarting is heavy, and there
   is no "these settings apply to the next load" alternative.
6. **No diff against defaults.** After changing three things it is impossible to
   see what is no longer recommended.
7. ~~No validation feedback in place.~~ **Fixed** for beginner mode: issues for
   the draft render above the fields they concern, toned by severity.

## The plan

### Substance

> **Two defects found in a design pass, both fixed.** The model breakdown note
> was printed **twice** in the Expected-effect panel — once as a standalone line
> added in an earlier round, once in the notes list beneath it. And a long memory
> value wrapped after "of", leaving an orphaned "12 GB" on its own line; the
> measurement is now non-breaking, so the break falls before the figure.
 — consequence at the moment of choice
1. **Inline impact per field.** Each row gains a muted right-hand consequence
   line that updates live: context length shows `≈1.9 GB of 24 GB`, memory mode
   shows what it trades. The Expected-effect card stays as the summary, not the
   only place the answer lives.
2. **A live "before → after" on the summary card** whenever a value differs from
   the saved config, so a change is legible before committing.
3. **"Modified from recommended" markers per field**, with a one-click revert on
   the row. The drawer already has this pattern for sampling; reuse it.

### Hierarchy
4. **Split beginner into two groups**: *What the model can do* (context, images)
   and *How it uses your computer* (memory, CPU policy). Nine flat rows is a
   list; two groups of four is a decision.
5. **Expert mode gets sections and a safety cue** — each group labelled, and any
   value outside the recommended band marked, so a dangerous number is visible
   rather than merely permitted.

### Fluidity
6. **Apply-on-next-load becomes the default action**, with a separate, clearly
   marked "Apply now (restarts the engine)" for people who want it immediately.
   This removes the heaviest action from the critical path.
7. **Invalid values are marked in place** with the reason under the field, and
   the save action disables with a stated reason rather than failing on submit.

## Motion

| Interaction | Spec |
|---|---|
| Beginner ↔ Expert | Segmented thumb + panel cross-fade, `--st-t-control` |
| Impact line changing | Number-only cross-fade, `--st-t-control`; no layout shift — the row reserves the line |
| Revert to recommended | `st-enter` on the value, `--st-t-control` |
| Save enabled | Colour and elevation only, `--st-t-control`. No movement |

## Verification

- Assert every field renders a consequence line and that it **changes when the
  value changes** — a static line is a failure.
- Assert "Save" is disabled with a visible reason when a value is invalid.
- `flow tuning` extended to switch modes, modify a value, revert, and save.
