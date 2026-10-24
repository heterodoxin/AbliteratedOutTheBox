# Strata UI fixes — handoff for Claude Code

This package comes from the Strata UI audit of 7 October 2026: 177 findings across 210 screen states. Here they are split into 19 tasks. Each task file lists the findings it covers, the files to change, and how to check the result.

## What's inside
- `tasks/` has one file per task, ready to give to a subagent.
- `findings.json` has all 177 findings with severity, state, issue, suggested fix and assigned task.
- `designs/` holds the reference designs for T09 (Model) and T10 (Engine v3). They are design-component source files: read them for structure, copy, colours and behaviour. They won't run on their own.
- `.claude/agents/strata-fixer.md` is a subagent definition (Sonnet, focused on one task).
- `PROMPTS.md` has the prompts to paste into Claude Code.
- `REVIEW.md` is the checklist for the review after each phase.
- `PROGRESS.md` is the log that each task appends to.

## Setup
1. Copy this folder into the repo root as `audit-fixes/` (so paths read `audit-fixes/tasks/T01-…`).
2. Copy `audit-fixes/.claude/agents/strata-fixer.md` to `.claude/agents/strata-fixer.md` in the repo.
3. Make sure `docs/evidence/audit-2026-10-07/` (the screenshots) is in the repo, because the task files point to it.
4. Run `npm install`, then `npm run check` and `node --test tests/` once. Note anything that already fails, so it isn't blamed on a task.

## Order
Phase 1 fixes shared components, so later tasks get smaller. Tasks in the same phase with no dependency between them can run in parallel on separate branches.

### Phase 1 — root causes (shared components)
| Task | Title | Findings | High / Med / Low | Depends on |
|---|---|---|---|---|
| T01 | One model, one name | 11 | 6 / 2 / 3 | — |
| T02 | Overlay layering and dialogs | 5 | 2 / 3 / 0 | — |
| T03 | Primary, disabled and on states | 11 | 1 / 7 / 3 | — |
| T04 | One error component, no fixture text | 10 | 0 / 9 / 1 | — |
| T05 | Show numbers when they matter | 6 | 2 / 2 / 2 | T01 |
| T06 | Fix the response hover layout | 1 | 1 / 0 / 0 | — |
| T07 | Hide what is not built | 2 | 0 / 1 / 1 | — |
| T08 | Engine and Model in the main nav | 1 | 1 / 0 / 0 | T01 |

### Phase 2 — rebuild Model and Engine
| Task | Title | Findings | High / Med / Low | Depends on |
|---|---|---|---|---|
| T09 | Build the single-model Model page | 17 | 0 / 9 / 8 | T01, T03, T04 |
| T10 | Build Engine v3 | 17 | 2 / 9 / 6 | T01, T02, T03 |

### Phase 3 — remaining findings per screen
| Task | Title | Findings | High / Med / Low | Depends on |
|---|---|---|---|---|
| T11 | Chat: remaining findings | 38 | 0 / 20 / 18 | T01, T05, T06, T07 |
| T12 | Time-skip: remaining findings | 6 | 0 / 3 / 3 | — |
| T13 | Serving: remaining findings | 19 | 0 / 6 / 13 | T04 |
| T14 | Monitor: remaining findings | 2 | 0 / 1 / 1 | T05 |
| T15 | Skills and Tasks: remaining findings | 9 | 0 / 2 / 7 | T02, T03, T04 |
| T16 | Settings and Setup: remaining findings | 7 | 0 / 2 / 5 | T02, T09 |
| T17 | Navigation and overlays: remaining findings | 4 | 0 / 2 / 2 | T02, T08 |
| T18 | Preview panel: remaining findings | 7 | 0 / 3 / 4 | — |
| T19 | Dark mode: remaining findings | 4 | 0 / 1 / 3 | T02, T03 |

## Running it cheaply
- Do high and medium first. Every task file puts low findings under "Optional polish". Skip them in the first pass. That cuts about a third of the work.
- Use a fresh session per task. Each subagent starts clean with only its task file, so context stays small.
- Check with tests first, and use screenshots only for the states listed in the task.
- After each phase, run one review (see REVIEW.md) instead of reviewing after every task.

Rough cost at API list prices (Sonnet 5.5, medium effort, for the subagents; one review per phase):
- Phase 1 (8 tasks): $10–20
- Phase 2 (2 large tasks): $8–16
- Phase 3 (9 tasks): $10–20, or about half that if you skip the low findings
- Reviews: $8–15

## Ground rules for every task
- Stay within the task's scope. If a fix needs a file that another task owns, make the smallest change and note it in PROGRESS.md.
- Don't change the design kit tokens in `src/styles/kit/` unless the task says so. Run `npm run sync:kit` if you do.
- Keep both themes working. Check light and dark.
- Existing tests must stay green. Add tests for new logic (formatters, clamping, fit rules).
- A finding can be closed as "already fixed by Txx" or "not reproducible", with a one-line reason.
