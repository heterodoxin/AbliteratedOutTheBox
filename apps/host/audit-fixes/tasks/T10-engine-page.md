# T10 · Build Engine v3

**Phase 2** · 17 findings (2 high, 9 medium, 6 low) · Depends on: T01, T02, T03

## Goal
Replace Tuning with designs/Engine v3.dc.html: auto-tune with the live animation, workloads with the two modes (leave room for other apps / full speed), and an Expert mode with every parameter, presets and the raw config editor.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/screens/Tuning.tsx` (rewrite as Engine)
- `src/components/EngineExperience.tsx`
- `src/lib/tuning-controls.ts`
- `src/lib/config.ts`
- `src/lib/experience.ts`
- `server/engine.mjs`
- `designs/Engine v3.dc.html` (reference: structure, copy, motion timing, colours)

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `engine-beginner` | Dial tick labels (4K 8K 16K 32K) sit in a straight row under a 270° dial, not at their ticks | labels around the arc, or use a slider |
| 2 | Medium | `engine-beginner` | Context range tops out at 32K; model supports 1M | range from engine capabilities |
| 3 | Medium | `engine-beginner` | "Reset" link is very pale grey (~2:1) | ink-muted #6f6f6f minimum |
| 4 | Medium | `engine-*` | Sticky action bar is ~140px tall with "More actions", disabled "Apply now…", disabled "Save for next load", plus two lines of notes; covers content and both actions look disabled at rest | one-line bar that appears only when there are changes ("2 changes · Discard · Save") |
| 5 | Medium | `engine-resource-controls` | Beginner mode exposes GPU/layer split, KV mode (INT8), CPU threads | move to Expert; Beginner gets workload + headroom only |
| 6 | Medium | `engine-expert` | Two stacked segmented controls (Beginner/Expert, Parameters/Raw config) look identical | mode as a toggle in the header; sub-tabs as underline tabs |
| 7 | High | `engine-config-invalid` | Invalid JSON shows no visible error; the error line sits under the sticky action bar; Export silently disables | inline error banner above the editor with line/column; keep editor bottom clear of the bar |
| 8 | Medium | `engine-expert-advanced` | Several Expert parameters (RoPE scaling, Expert cache / RAM mode) are "Preserved only · adapter does not apply this value" — shown as live inputs in small grey caption | disable the control + "Not used by this engine" tag, or hide behind "Show unsupported" |
| 9 | Medium | `engine-expert-changed-only-empty` | Status says "Draft" and "Save for next load" is enabled while "Changed only (0)" | Draft/Save only when changes > 0 |
| 10 | High | `engine-expert-*` | At 1280×720 Expert mode shows ONE parameter per screen: mode tabs, sub-tabs, two disclosures, filter row and the sticky bar take ~85% of the height | compact table rows (name · key · value · revert), collapse header chrome on scroll |
| 11 | Medium | `engine-expert-server` | "Server access (server.host)" duplicates the Serving page setting, and is "Preserved only" here | one home: Serving; link from Expert |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `engine-prompt-batch` | Second dial in blue (context dial is green); "Automatic · recommended 2,048" checkbox unchecked while value is 2,048 | one accent; Automatic on by default, dial disabled until unchecked |
| 2 | Low | `engine-raw-config` | Plain textarea: no line numbers, highlighting or inline validation; bottom clipped by sticky bar | code editor with gutter errors |
| 3 | Low | `engine-*` | Two-column layout leaves right column empty below "Expected experience" | sticky summary card on the right |
| 4 | Low | `engine-discard-confirmation` | "Return all three editors…" — user doesn't know there are three | "Discard all unsaved engine changes" |
| 5 | Low | `engine-actions` | Recommended defaults / Validate / Speed test hidden in "More actions" | surface "Reset to recommended" and "Run speed test" as visible buttons |
| 6 | Low | `engine-adapter-details` | "llama.cpp adapter… native Strata configuration requires a different adapter" | move to Expert › About engine |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] Beginner mode shows no expert controls.
- [ ] Expert mode shows all parameters on one scrollable page grouped in sections; at 1280×720 more than one section is visible.
- [ ] Invalid raw config shows an inline error next to the editor and disables Apply; the sticky action bar never covers content.
- [ ] Auto-tune motion respects prefers-reduced-motion. Run `npm run motion` and `npm run motion:budget`.
- [ ] The header shows the model via formatModelName() and has no "Change model".
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `engine-beginner-light.png`
- `engine-resource-controls-light.png`
- `engine-prompt-batch-light.png`
- `engine-expert-light.png`
- `engine-raw-config-light.png`
- `engine-config-invalid-light.png`
- `engine-discard-confirmation-light.png`
- `engine-actions-light.png`
- `engine-adapter-details-light.png`
- `engine-expert-advanced-light.png`
- `engine-expert-changed-only-empty-light.png`
- `engine-expert-server-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t10-engine-page` with message `T10: Build Engine v3`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
