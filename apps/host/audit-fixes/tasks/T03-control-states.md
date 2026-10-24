# T03 · Primary, disabled and on states

**Phase 1** · 11 findings (1 high, 7 medium, 3 low) · Depends on: nothing, can start now

## Goal
Enabled and disabled primary buttons look the same, dark-mode primary buttons are green on green, toggle "on" is barely visible, and checkboxes use browser blue. Define the states once in the shared components.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/components/ui.tsx` (Button, Toggle, Checkbox)
- `src/styles/app.css`
- `src/styles/kit/` (only if the token lives there; keep sync:kit working)

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | High | `GLOBAL` | Primary buttons (New chat, Send, Approve, Open preview) use the same pale-green tint enabled or disabled; Approve looks disabled, Send-enabled ≈ Send-disabled | solid #10b981 + white for enabled; neutral grey for disabled |
| 2 | Medium | `GLOBAL-chat` | At 1280×720 the composer (3-line empty textarea + toolbar + stats) takes ~250px; reading area is ~420px | 1-line composer that grows; stats inline in toolbar |
| 3 | Medium | `GLOBAL` | Native checkboxes render browser blue (#2563eb) — off-palette | accent-color: #10b981 (or ink) app-wide |
| 4 | Medium | `skills-*` | Toggle "on" state is pale mint with a white knob — barely distinguishable from "off" | solid #10b981 track when on |
| 5 | Medium | `tasks-populated` | Row actions are unlabeled icons (pencil, paper-plane for "run now", toggle); paper-plane reads as "send" | tooltips + text "Run now"; label the toggle "On schedule" |
| 6 | Medium | `tasks-detail-*` | Detail view keeps list chrome (Demo note, "Deleted tasks" checkbox, search, sort) and the generic subtitle; "All tasks" is a centred underlined link above content | breadcrumb "Tasks › Daily briefing" top-left; hide list controls |
| 7 | Medium | `dark-*` | Primary buttons in dark (Approve, Open preview, Save setup and load, New chat) are dark green fill with green text (~3:1) | solid #10b981 with #0b1f17 or white text; same rule as light |
| 8 | Medium | `dark-skills-overview / dark-skills-edit` | "On" toggles in dark are a dark green track with a dark knob — on vs off is hard to tell | bright #10b981 track + white knob when on |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `GLOBAL-sidebar` | Task rows show pencil + trash icons permanently on every row | reveal on hover/focus |
| 2 | Low | `dark-*` | Disabled "Save for next load" in dark is near-invisible green-on-green | neutral disabled style shared with light |
| 3 | Low | `dark-preview-pdf / dark-preview-website` | PDF page and website preview stay bright white/cream inside the dark panel — expected for document content, but there's no frame or dimming option | thin border + optional "Dim document" toggle |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] Primary enabled: solid #10b981 with white text, both themes. Disabled: neutral fill (#eef1f4 light / a neutral grey in dark) with muted text, not a green tint.
- [ ] Toggle on: bright green track and white knob, both themes. Off: neutral track.
- [ ] Checkboxes use the brand green via accent-color or a custom control.
- [ ] `npm run contrast` passes; add the new states to it if they are not covered.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `tasks-populated-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t03-control-states` with message `T03: Primary, disabled and on states`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
