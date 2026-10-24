# Prompts for Claude Code

## 1. Run one task (simplest)
Start a new session in the repo for each task and paste:

```
Use the strata-fixer subagent to complete audit-fixes/tasks/T01-one-model-name.md.
High and medium findings only. Follow the task's "How to check" and "Done when" sections.
```

Change the file name for each task. Set the effort to medium for the subagent (for example with /model, or in your settings) to keep costs down.

## 2. Run a whole phase (orchestrated)
Paste into one session:

```
Read audit-fixes/README.md. Run Phase 1: for each task T01–T08, launch the strata-fixer
subagent with its task file, high and medium findings only. Run tasks whose
dependencies are met in parallel; wait for T01 before T05 and T08.
Each subagent works on its own branch. When all are done, merge the branches into
fix/phase-1 in task order, resolve conflicts, and run npm run check, node --test tests/
and npm run build. Then stop and summarise PROGRESS.md for me.
```

Use the same prompt for Phase 2 (T09–T10) and Phase 3 (T11–T19).

## 3. Review after each phase
Run this in a fresh session, ideally on a stronger model:

```
Review the fix/phase-1 branch against audit-fixes/REVIEW.md and the task files for T01–T08.
For each finding, check the change in the code and, for the states listed, re-capture the
screen and compare it with docs/evidence/audit-2026-10-07/. Report, per task: fixed, not
fixed, regressions. Don't fix anything yourself, just list what to send back.
```
