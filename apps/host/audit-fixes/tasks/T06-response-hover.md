# T06 · Fix the response hover layout

**Phase 1** · 1 findings (1 high, 0 medium, 0 low) · Depends on: nothing, can start now

## Goal
Hovering a reply opens a side column about 20px wide, its text wraps one character per line, and the code block is pushed out of view. Message actions must never change layout.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/screens/Chat.tsx` (message actions)
- `src/styles/app.css`
- `src/components/Markdown.tsx` (only if the code block width is involved)

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | High | `chat-response-actions` | LAYOUT BUG: hovering a response opens a ~20px side column; its meta text ("…tok/s · 4.2s") wraps one character per line and the code block is pushed left/clipped ("rt function", "turn notes") | actions row horizontal under the message, absolutely positioned; never reflow content |

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
- [ ] Hovering or focusing a reply shows the actions in a row under the message, without moving any content (measure the message bounding box before and after hover; it must be identical).
- [ ] Meta text ("tok/s · 4.2s") sits on one line.
- [ ] Works with the keyboard (focus-within) and on touch (always visible on narrow screens).
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `chat-response-actions-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t06-response-hover` with message `T06: Fix the response hover layout`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
