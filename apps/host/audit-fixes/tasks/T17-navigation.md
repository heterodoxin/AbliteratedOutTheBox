# T17 · Navigation and overlays: remaining findings

**Phase 3** · 4 findings (0 high, 2 medium, 2 low) · Depends on: T02, T08

## Goal
Finish the command palette, shortcuts, sampling drawer and context panel.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/app/CommandPalette.tsx`
- `src/app/Shortcuts.tsx`
- `src/app/SamplingDrawer.tsx`
- `src/app/ContextPanel.tsx`
- `src/app/Sidebar.tsx`
- `src/lib/shortcut-bindings.ts`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `navigation-rename-chat` | "Chat name" label sits inline left of a ~170px input that truncates the current name | stacked label, full-width input, name pre-selected |
| 2 | Medium | `navigation-delete-chat-confirmation` | Copy says "Archive it if you want to keep it" but the dialog has no Archive action | add "Archive instead" secondary button |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `navigation-command-search` | "Restart engine" uses a red icon for a non-destructive action | neutral icon; red only for destructive |
| 2 | Low | `navigation-sidebar-collapsed` | Collapsed rail shows a lone green dot for the model with no label/tooltip | tooltip "Qwen3.8-Flash-Next · running"; same for other rail icons |

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
- `navigation-rename-chat-light.png`
- `navigation-delete-chat-confirmation-light.png`
- `navigation-command-search-light.png`
- `navigation-sidebar-collapsed-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t17-navigation` with message `T17: Navigation and overlays: remaining findings`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
