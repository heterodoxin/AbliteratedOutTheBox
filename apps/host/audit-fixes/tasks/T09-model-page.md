# T09 · Build the single-model Model page

**Phase 2** · 17 findings (0 high, 9 medium, 8 low) · Depends on: T01, T03, T04

## Goal
Replace the multi-model library (Models) and Setup step 2 with the single-model page in designs/Model.dc.html: a hero for the running version, a ladder of versions marked by how well each fits this PC, a detail panel with one action, and the list of installed versions.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/screens/Models.tsx` (rewrite as Model)
- `src/components/ModelCatalog.tsx` (remove or reduce to version list)
- `src/screens/Setup.tsx` (step 2 becomes "pick a version", reusing the fit logic)
- `src/lib/types.ts` (Version, Fit)
- `server/catalog.mjs, server/models.mjs` (versions + fit for detected hardware)
- `designs/Model.dc.html` (reference: structure, copy, colours, fit rules)

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `models-overview` | Profile legend is a plain-text line with slashes ("Agentic · Tools… / Balanced · … / Performance · …") | drop with profiles; if kept, chips with tooltips |
| 2 | Medium | `models-recommendation-details` | Balanced and Performance explanations are the same sentence twice | unique copy or one line |
| 3 | Medium | `models-context-64k` | At 64K every profile says "No fitting model found", each with its own "Review context" + "Why?" — dead end ×3 | one message: "64K needs 2.1 GB more · Use 32K" with a one-click fix |
| 4 | Medium | `models-none-installed` | Floating "2 of 3 selected · Compare variants" tray overlaps the empty state's help text, selecting items that aren't visible | clear selection when the list filters to empty; never overlay content |
| 5 | Medium | `models-hardware-unavailable` | "Check computer" + "Why?" repeated on three rows plus a fourth line saying the same | one banner with one action |
| 6 | Medium | `models-hardware-unavailable-dialog` | Dialog says "Use guided setup to check this computer" but has no button | add "Run guided setup" primary |
| 7 | Medium | `models-load-confirmation` | Body is a config string ("32K context · batch 2,048 · INT8 cache · Automatic · 2 CPU threads") and "your separate draft" is unexplained | plain sentence: "Switch all chats to Balanced (4.9 GB). Takes ~5 s."; details behind disclosure |
| 8 | Medium | `models-load-error` | Error copy exposes internals ("Fixture: model load failed; installed files retained."); dialog then offers two retries ("Try again" and "Save setup and load") | plain cause + one retry; keep fixture text out of UI |
| 9 | Medium | `setup-model` | Setup dialog is taller than a 720px window; title and footer scroll out of view | max-height: calc(100vh - 48px) with scrolling body and sticky footer |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `models-overview` | "Global · all chats" floats above the buttons as an orphan label | put under the model name |
| 2 | Low | `models-overview` | Reading speed label blue, Answer speed label green — colour code used only on this page | ink labels; one accent |
| 3 | Low | `models-recommendation-details` | "Start answering · Not estimated" shown as a value | hide unknown rows |
| 4 | Low | `models-filters` | Pluralisation: "1 variants", "1 quantizations", "1 installed · 1 variants" | pluralise |
| 5 | Low | `models-filters` | "Include incompatible variants" checkbox label is 16px next to 14px filter pills | match control text size |
| 6 | Low | `models-storage` | Storage shows a raw temp path with no actions | "Open folder" / "Change location" |
| 7 | Low | `models-large-catalog` | Demo button label "Simulate download" | "Download"; mark demo once in header |
| 8 | Low | `models-context-options` | Context menu stops at 64K | list matches engine range (up to 1M) |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] All seven versions show size, speed on this PC and a fit zone (fast / slower / too big). "Recommended" is the best-quality version that is still fast.
- [ ] The detail panel action matches the state: Switch + Delete, Download (progress + cancel), Unload, or "Too big for this PC".
- [ ] Fit is computed from the real hardware the host reports; the reference design's numbers are placeholders.
- [ ] Setup step 2 uses the same component and the same recommendation.
- [ ] Light and dark both pass `npm run contrast`.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `models-overview-light.png`
- `models-recommendation-details-light.png`
- `models-filters-light.png`
- `models-storage-light.png`
- `models-context-64k-light.png`
- `models-none-installed-light.png`
- `models-hardware-unavailable-light.png`
- `models-hardware-unavailable-dialog-light.png`
- `models-load-confirmation-light.png`
- `models-large-catalog-light.png`
- `models-load-error-light.png`
- `models-context-options-light.png`
- `setup-model-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t09-model-page` with message `T09: Build the single-model Model page`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
