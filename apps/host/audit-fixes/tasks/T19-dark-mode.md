# T19 · Dark mode: remaining findings

**Phase 3** · 4 findings (0 high, 1 medium, 3 low) · Depends on: T02, T03

## Goal
Fix the dark-mode issues that remain after the shared control and overlay fixes.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/styles/app.css`
- `src/styles/kit/`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `dark-chat-open-website-invalid` | Same plain-text error (no red, no field state) as light | shared field error style |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `dark-models-overview` | Blue "Reading" label (#3b82f6-ish) on near-black is low contrast | ink label in both themes |
| 2 | Low | `dark-monitor-generating` | Sidebar model dot turns blue while generating (same as light) — blue isn't used for state anywhere else in dark | green pulse for "working" |
| 3 | Low | `dark-setup-model` | Heading gets a visible focus outline on step change (same as light setup-prepare-error) | move focus without a visible ring on non-interactive headings (tabindex=-1 + :focus-visible none) |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] `npm run contrast` and `npm run luminance` pass in dark mode.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `models-overview-dark.png`
- `chat-open-website-invalid-dark.png`
- `monitor-generating-dark.png`
- `setup-model-dark.png`

## Done when
- All checks pass.
- Commit on branch `fix/t19-dark-mode` with message `T19: Dark mode: remaining findings`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
