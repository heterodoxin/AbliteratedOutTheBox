# T07 · Hide what is not built

**Phase 1** · 2 findings (0 high, 1 medium, 1 low) · Depends on: nothing, can start now

## Goal
"Isn't connected in this preview" appears several times, a Session log dialog has only a Done button, and "Demo" labels sit on single values. Hide unbuilt controls and show one "Preview build" badge instead.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/components/ComposerControls.tsx`
- `src/screens/Chat.tsx`
- `src/screens/Monitor.tsx`
- `src/screens/Tasks.tsx`
- `src/screens/Models.tsx`
- `src/components/layout.tsx` (one header badge)
- `src/lib/lifecycle.ts or src/lib/config.ts` (one feature-flag map)

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `GLOBAL` | "isn't connected in this preview/build" footnotes in 5+ places (thinking menu, tool access, compaction, session log, add menu) | hide unbuilt controls, or one "Preview build" badge |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `chat-session-log` | Title Case "Download Session Log"; rest of app is sentence case | "Download session log" |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] `grep -rn "isn.t connected in this preview" src` returns nothing rendered by default.
- [ ] There are no per-value "Demo" labels. One "Preview build" badge appears in the app header when the demo engine is active.
- [ ] Hidden controls are behind a feature flag, not deleted, so they can return when they are built.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `chat-session-log-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t07-hide-unbuilt` with message `T07: Hide what is not built`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
