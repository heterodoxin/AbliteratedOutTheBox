# T05 · Show numbers when they matter

**Phase 1** · 6 findings (2 high, 2 medium, 2 low) · Depends on: T01

## Goal
The composer stats show "—" while generating, Monitor reads 0.0 tok/s at rest, percentages have no labels, and memory can read 20.0 / 16 GB. Show live values, label every unit, and catch impossible ones.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/components/ChatToolbar.tsx`
- `src/lib/chat-metrics.ts`
- `src/components/ContextRing.tsx`
- `src/components/TelemetryPopover.tsx`
- `src/screens/Monitor.tsx`
- `src/lib/hardware-summary.ts`
- `server/monitor.mjs, server/hardware.mjs`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `chat-empty` | Stats row idle: "— tokens/s · 0 total tokens · ◌ —" | hide until first reply |
| 2 | High | `chat-generating` | Stats row shows "— tokens/s · — total tokens" WHILE generating — the one moment speed matters | show live tok/s during stream |
| 3 | Medium | `chat-showcase-code` | "21%" in stats row unlabeled | "21% of context" |
| 4 | High | `monitor-generating` | "System memory 20.0 / 16 GB" — used exceeds total, shown without warning | cap/flag; if real, show "Over by 4 GB · swapping" in red |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `chat-generating` | Sidebar model dot turns blue during generation, no label/tooltip; "Serving 1" badge unexplained | tooltip "Generating"; badge → "1 request" |
| 2 | Low | `monitor-*` | Header right icon (image glyph) has no label | tooltip "Save snapshot" or text button |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] While generating, the stats row shows live tok/s. When idle it shows the last reply (e.g. "43.8 tok/s · last reply") or is hidden.
- [ ] Every percentage has a label ("21% of context").
- [ ] Used memory can never exceed total. Clamp it and log a warning, and make both free-memory figures come from one source.
- [ ] Add unit tests for the clamping and idle formatting.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `chat-empty-light.png`
- `chat-generating-light.png`
- `chat-showcase-code-light.png`
- `monitor-generating-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t05-live-numbers` with message `T05: Show numbers when they matter`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
