# 14 · Empty, loading and error states

**Across every screen** — no single file

The states nobody designs and everybody sees. For an audience that finds
computers intimidating, an empty screen is the most likely place to give up, and
a silent failure is the most likely place to lose trust.

## What is today

- **`Empty`** is a real component (icon, title, body, optional action) used on
  Chat, Models, Serving, Skills, Monitor, Workspace.
- **`BootSkeleton`** covers the first paint.
- **Errors** surface as toasts and as an inline error on the failed message.
- **Empty states distinguish "nothing yet" from "nothing matches"** on Skills and
  Workspace — the right instinct, applied inconsistently.
- **Loading** is a spinner or an indeterminate `Bar` depending on the surface.

## Where it falls short

1. **Empty states do not say what to do next, only what is absent.** Several have
   an action, several do not.
2. **No empty state exists on Monitor or Serving for a first run** beyond a
   generic message; neither points at the step that would fill them.
3. ~~A failed model download leaves a row in an ambiguous state.~~ **Wrong, and
   the real gap was different.** `v.error` already renders on the row. What was
   missing is what to *do* about it: there was no Retry anywhere in the app
   except Chat's *Ask again*, so a failed download was a dead end and the reader
   had to work out that the Download button was also the retry. Failed rows now
   carry the reason **and** a *Try again* action together.
4. **No retry affordance on failure.** The error is reported, never actionable.
5. **Loading has no time expectation.** Downloads show a percentage but not how
   long; scans show a line with no sense of duration.
6. **The skeleton does not match the final layout**, so the first paint jumps
   when real content arrives.
7. ~~Offline / engine-unreachable has no dedicated state.~~ **Fixed.**
   `connection` had been tracked in the store since the beginning and was
   rendered nowhere, so a dead host looked exactly like an idle one — every panel
   simply empty, with no statement that anything was wrong. A bar now names the
   problem in plain language and offers a reconnect. Verified both ways: with the
   host killed it reports `offline`, and with it running the bar is absent.
8. **Skill and Workspace empties have an action; Monitor and Serving do not.**
   The pattern exists and is not applied.

## The plan

### Substance — every state says what to do
1. **A state checklist per screen**: *first run (nothing exists yet)* ·
   *filtered to nothing* · *engine unreachable* · *operation failed*. Six screens
   × four states is twenty-four states, and today about eight are handled.
2. **Every empty state carries exactly one action** that leads to the thing that
   would fill it. Monitor's empty should offer "Send a prompt", not describe
   itself.
3. ~~Errors carry a way out.~~ **Fixed** for the actions that can fail today.
   Engine failures reported only through a toast, which is gone in 5.2 s; the
   failure is now *also* recorded and rendered in place, with the retry on the
   surface that failed. Verified against a deliberately aborted restart: the
   toast had expired by 8.2 s and the inline error remained, retry re-reported
   the failure while it persisted, and **cleared once the request succeeded** —
   so it is a real retry, not a re-render. Model load failures still report by
   toast alone.
4. ~~An explicit engine-unreachable state.~~ **Done** — driven by the existing
   event-stream status rather than a separate health poll, so it reacts the
   moment the stream drops.
5. **Time expectations on every operation longer than two seconds**, computed
   from observed rate where possible.

### Fluidity
6. **Skeletons that match the final layout**, so nothing jumps. Build the
   skeleton from the real component's box model.
7. **Loading resolves by cross-fade, never by a jump** — the content fades in over
   the skeleton on `--st-t-local`.

## Motion

| State | Spec |
|---|---|
| Skeleton → content | Cross-fade, `--st-t-local`. No layout movement |
| Empty state appearing | `st-enter`, `--st-t-local`, 8 px — once, not per element |
| Error appearing | `st-enter`, `--st-t-control`, 6 px; stays until dismissed |
| Retry in progress | The button becomes busy in place; no spinner elsewhere on screen |
| Indeterminate bar | Loops only while the operation runs, and stops the instant it ends |

## Verification

- New `flow states`: force each of the four states per screen and assert an action
  is present in the empty states and a retry in the failures.
- Assert no operation over two seconds lacks a time expectation.
- Assert the skeleton and the loaded layout have **identical box geometry** — a
  jump of more than 2 px is a failure.
