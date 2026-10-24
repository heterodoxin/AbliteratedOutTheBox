# T16 · Settings and Setup: remaining findings

**Phase 3** · 7 findings (0 high, 2 medium, 5 low) · Depends on: T02, T09

## Goal
Make the Settings index follow the visible section, and finish the Setup flow after T09.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/screens/Settings.tsx`
- `src/screens/Setup.tsx`
- `server/setup.mjs`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `settings-search` | While searching "motion", the index highlights "About" but the only result is under Reading & focus (same lag in Privacy state) | highlight follows visible section; during search, highlight the result's section or none |
| 2 | Medium | `setup-advanced` | "Advanced" opens a second modal on top of the setup modal, containing only the same context dial already on the page behind it | inline disclosure, or drop — nothing new inside |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `settings-general` | Reading size value wraps to two lines ("16" / "px") | nowrap, fixed width value |
| 2 | Low | `settings-general` | General contains a single row (Keyboard shortcuts) | fold into Reading & focus, or add Language/Startup here |
| 3 | Low | `settings-reading-focus` | Option label "Standard / system" uses a slash | "Follow system" (matches Motion) |
| 4 | Low | `settings-storage` | "Manage models" (plural) | "Manage model versions" → Model page |
| 5 | Low | `setup-prepare` | Footer line "Global · all chats · A model operation is in progress. View requests" | drop; progress already shown |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] The Settings index highlights the section under the sticky header while scrolling (IntersectionObserver with a top offset).
- [ ] No scrollIntoView calls remain in Settings; use container.scrollTo with an offset.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `settings-search-light.png`
- `settings-general-light.png`
- `settings-reading-focus-light.png`
- `settings-storage-light.png`
- `setup-advanced-light.png`
- `setup-prepare-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t16-settings-setup` with message `T16: Settings and Setup: remaining findings`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
