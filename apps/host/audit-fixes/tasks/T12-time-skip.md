# T12 · Time-skip: remaining findings

**Phase 3** · 6 findings (0 high, 3 medium, 3 low) · Depends on: nothing, can start now

## Goal
Give the two versions stable names across every state, use plain words, and let people use the version that was kept.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/components/TimeSkipPanel.tsx`
- `src/lib/time-skip.ts`
- `src/components/ArtifactPanel.tsx`
- `src/screens/Chat.tsx` (inline card)

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `timeskip-building` | Small amber status text ("Reset needs work", "Content overflows") at ~11px is low contrast on white | darker amber (#8a5a00) or icon + ink text |
| 2 | Medium | `timeskip-*` | Stream A is called "Original", then "Revision", then "Revision A"; B is "Alternative", "Candidate B", "Selected version" across states | fixed names: "Version A" / "Version B", with stage as a sub-label |
| 3 | Medium | `timeskip-complete` | Finished panel offers only "Replay demo" — no way to open/apply the kept version | primary "Open version B" + secondary "Compare" |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `timeskip-building` | Header chips: A shows a ✓, B shows a ▾ — two different affordances for the same kind of item | same control for both; ✓ only on the winner |
| 2 | Low | `timeskip-*` | "First draft saved ▾" chevron: unclear what it opens | make it a plain label or a history menu with a visible title |
| 3 | Low | `timeskip-inspect-original` | Scrolled list slides under the panel header with no divider/shadow; first row half-hidden | sticky header with bottom border when scrolled |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] `node --test tests/time-skip.test.mjs` passes.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `timeskip-building-light.png`
- `timeskip-complete-light.png`
- `timeskip-inspect-original-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t12-time-skip` with message `T12: Time-skip: remaining findings`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
