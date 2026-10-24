# T15 · Skills and Tasks: remaining findings

**Phase 3** · 9 findings (0 high, 2 medium, 7 low) · Depends on: T02, T03, T04

## Goal
Label row actions, give the task detail view its own controls, and finish the Skills and Tasks states.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/screens/Skills.tsx`
- `src/screens/Tasks.tsx`
- `src/components/TaskEditor.tsx`
- `src/lib/tasks.ts`
- `src/app/TaskTree.tsx`
- `server/tasks.mjs, server/skill-edits.mjs`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `tasks-create-*` | Instructions textareas (and the AI prompt box) render in monospace while every other field is sans | inherit brand font in textareas |
| 2 | Medium | `tasks-created` | A task created with "Run on schedule" checked appears as "Paused" | verify state mapping; new scheduled tasks show "Next: …" |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `skills-added` | In this state the page subtitle, "N enabled here" and the search field disappear | keep header layout stable across states |
| 2 | Low | `skills-edit` | Description is a single-line input that truncates; Instructions field is cut off at the dialog bottom | textarea for both; dialog scroll with sticky footer |
| 3 | Low | `skills-overview` | "3 enabled here" — "here" is ambiguous | "3 enabled in this workspace" |
| 4 | Low | `tasks-populated` | Every row has the same clock icon tile, so it carries no information | drop the tile or vary by state (scheduled / running / paused / failed) |
| 5 | Low | `tasks-empty` | Empty-state icon is a large heavy outline clock, unlike the small boxed icons used in other empty states | shared EmptyState component |
| 6 | Low | `tasks-ai-blank` | Dialog body is clipped at the bottom (Instructions field cut) with no scroll cue | scroll fade + sticky footer |
| 7 | Low | `tasks-result` | Date formats differ: "06/10/2026, 12:24:23" here vs "7 Oct 2026, 13:24" on the row | one date formatter |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] `node --test tests/tasks-and-sharing.test.mjs` passes.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `skills-added-light.png`
- `skills-edit-light.png`
- `skills-overview-light.png`
- `tasks-populated-light.png`
- `tasks-empty-light.png`
- `tasks-ai-blank-light.png`
- `tasks-created-light.png`
- `tasks-result-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t15-skills-tasks` with message `T15: Skills and Tasks: remaining findings`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
