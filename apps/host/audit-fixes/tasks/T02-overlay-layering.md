# T02 · Overlay layering and dialogs

**Phase 1** · 5 findings (2 high, 3 medium, 0 low) · Depends on: nothing, can start now

## Goal
Menus open behind dialogs, some overlays show the page through them, opening a dialog shifts and clips the sidebar, and the dark-mode scrim lightens the page. Fix this once with a z-index scale, portalled popovers and a stable scrollbar gutter.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/styles/app.css` (z-index tokens: base, sticky, sidebar, popover, dialog, dialog-popover, toast)
- `src/components/Popover.tsx and src/components/FloatingMenu.tsx` (portal to body, layer above the open dialog)
- `src/components/ui.tsx` (Dialog: opaque surface, scrim colour per theme, scroll lock without layout shift)
- `src/app/CommandPalette.tsx`
- `src/screens/Setup.tsx` (dialog max-height + internal scroll)
- `src/components/TaskEditor.tsx` (Repeat dropdown)

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | High | `tasks-repeat-options` | Repeat dropdown opens but its options render behind the dialog (confirmed in capture QA) — can't pick a frequency by mouse | portal the listbox above the dialog layer (z-index / top-layer) |
| 2 | High | `setup-computer` | Setup dialog is see-through: the chat page ("What would you like to do?", suggestion cards, composer) shows through and overlaps the setup content. Either the entrance fade was captured mid-way or the "Translucent" surface setting applies to dialogs | verify; dialogs always opaque (Transparency setting should not reach modal surfaces) |
| 3 | Medium | `navigation-rename-chat` | Opening a dialog shifts the sidebar ~26px left, clipping its contents ("orkspaces", New chat button cut) — also in delete confirmation | fix scrollbar-gutter compensation (scrollbar-gutter: stable on the scroll container, not padding on body) |
| 4 | Medium | `navigation-command-palette` | Open state is see-through — chat content shows through the palette and is unreadable (search/empty states are opaque). Same pattern as Setup | verify entrance animation vs. translucent surface; overlays must be opaque at first paint |
| 5 | Medium | `dark-*` | Dark-mode modal scrim is a light grey wash (#4a4a4a-ish) rather than a darkening overlay, so the page behind looks washed out and the dialog has little contrast against it | scrim rgba(0,0,0,.55) in dark |

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
- [ ] In the task editor, the Repeat dropdown opens above the dialog and an option can be picked with mouse and keyboard.
- [ ] Opening any dialog does not move the sidebar (compare its x position before and after; use scrollbar-gutter: stable or compensate padding).
- [ ] Setup and the command palette have opaque surfaces in both themes. First confirm the see-through look is not just a mid-animation capture; if it is, note that and skip.
- [ ] Dark mode scrim darkens the page (e.g. rgba(0,0,0,.55)).
- [ ] Setup fits a 1280×720 window with its own scroll area; the primary action stays visible.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `tasks-repeat-options-light.png`
- `setup-computer-light.png`
- `navigation-rename-chat-light.png`
- `navigation-command-palette-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t02-overlay-layering` with message `T02: Overlay layering and dialogs`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
