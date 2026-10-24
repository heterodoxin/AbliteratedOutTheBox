# T14 · Monitor: remaining findings

**Phase 3** · 2 findings (0 high, 1 medium, 1 low) · Depends on: T05

## Goal
Fix the remaining Monitor findings once T05 has made the numbers consistent.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/screens/Monitor.tsx`
- `src/components/charts.tsx`
- `src/lib/hardware-summary.ts`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `monitor-*` | Caption "Demo · do not add these pools" is cryptic | "Shared with the model — don't add to Model memory" or a single combined memory bar |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `monitor-computer` | "Shared memory 16 GB · 3.8 GB available" disagrees with "System memory 7.0 / 16 GB" above | one source, same timestamp |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] Each finding above is resolved in the running app, in light and dark mode.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `monitor-computer-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t14-monitor` with message `T14: Monitor: remaining findings`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
