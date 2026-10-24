# T18 · Preview panel: remaining findings

**Phase 3** · 7 findings (0 high, 3 medium, 4 low) · Depends on: nothing, can start now

## Goal
Stop the tab strip overflowing and give every file type the same zoom and paging controls.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/components/ArtifactPanel.tsx`
- `src/components/PdfPreview.tsx`
- `src/components/TablePreview.tsx`
- `src/components/ChatMedia.tsx`
- `src/lib/preview-tabs.ts`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `preview-*` | Tab strip truncates ("fieldnotes-cover.svg" clipped by the full-screen button) and doesn't scroll to the active tab — in preview-code the first tab is pushed off | overflow scroll with fade + scroll active tab into view (scrollLeft, not scrollIntoView) |
| 2 | Medium | `preview-*` | Opening the panel squeezes the chat to ~470px; composer wraps to two rows | resizable split with min chat width ~560px; or panel as overlay below 1280px |
| 3 | Medium | `preview-pdf-*` | Mixed zoom controls: segmented Fit/100% for images, a dropdown for PDF; Previous/Next as large pill buttons | one zoom control pattern; icon buttons for paging |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `preview-pdf-actual` | At 100% the PDF page overflows horizontally with text cut off and no horizontal scrollbar | allow horizontal scroll / pan |
| 2 | Low | `preview-full-screen` | "Page text" disclosure dumps the PDF's text as one run-on paragraph | preserve line breaks or hide |
| 3 | Low | `preview-*` | Header label "Example · Website" etc. | file name + type; drop "Example" |
| 4 | Low | `preview-table` | Filter field and table card are misaligned (filter overhangs right edge) | align to table width |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] One shared toolbar component for zoom and paging is used by every preview type.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `preview-pdf-actual-light.png`
- `preview-full-screen-light.png`
- `preview-table-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t18-preview-panel` with message `T18: Preview panel: remaining findings`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
