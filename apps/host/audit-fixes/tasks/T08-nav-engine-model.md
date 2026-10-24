# T08 · Engine and Model in the main nav

**Phase 1** · 1 findings (1 high, 0 medium, 0 low) · Depends on: T01

## Goal
Engine tuning is what makes Strata different, but it sits under More. Put Engine and Model in the primary nav and make the sidebar model card open Model.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/app/Sidebar.tsx`
- `src/lib/router.ts`
- `src/lib/commands.ts` (command palette entries)
- `src/app/App.tsx`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | High | `navigation-more` | Engine and Models live in the "More" popover | promote Engine (and Model) to primary nav |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

_None._

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] Sidebar order: Chat, Tasks, Skills, Monitor, Engine, Model, then More (Serving, Settings…). Keep the collapsed rail in the same order.
- [ ] The sidebar model card opens Model.
- [ ] The command palette lists Engine and Model.
- [ ] Existing routes still resolve (keep redirects from old paths).
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `navigation-more-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t08-nav-engine-model` with message `T08: Engine and Model in the main nav`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
