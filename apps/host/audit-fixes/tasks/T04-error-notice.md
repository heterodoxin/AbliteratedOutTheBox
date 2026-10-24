# T04 · One error component, no fixture text

**Phase 1** · 10 findings (0 high, 9 medium, 1 low) · Depends on: nothing, can start now

## Goal
Errors render as plain body text in seven places and most start with "Fixture:". Add one ErrorNotice component (icon, what happened, what to do, optional action) and use it everywhere.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/components/ui.tsx` (new ErrorNotice)
- `src/screens/Serving.tsx`
- `src/components/CloudflareDialog.tsx`
- `src/screens/Models.tsx`
- `src/screens/Skills.tsx`
- `src/screens/Tasks.tsx / src/components/TaskEditor.tsx`
- `src/screens/Setup.tsx`
- `src/screens/Chat.tsx` (open-website error)
- `tests/ui-fixture-host.mjs` (check where the "Fixture:" prefix comes from)

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `chat-open-website-invalid` | Error "Enter a full http:// or https:// address." is plain black body text below the helper; field has no error state | red text directly under field + red border + aria-invalid; helper below |
| 2 | Medium | `chat-composer-error` | Composer error squeezed into the toolbar between "Auto" and Send in 11px red | error line above the composer with icon |
| 3 | Medium | `serving-requests-all` | Status column is lowercase plain text (queued / generating / done / error); errors aren't red | status chips: grey queued, blue generating, green done, red error |
| 4 | Medium | `serving-request-error` | Error reason shows fixture text: "Fixture: upstream response timed out after 4 seconds." | strip fixture prefix; plain cause + next step |
| 5 | Medium | `serving-cloudflare-error` | Error is plain 16px black body text ("Fixture: Cloudflare could not establish…") with no error styling; buttons unchanged | red error callout with icon, cause, and "Try again" as the primary |
| 6 | Medium | `skills-refresh-error` | Error is plain text at the bottom: "Fixture: skill service unavailable." — no styling, no retry | error banner under the header with Retry |
| 7 | Medium | `GLOBAL` | Fixture text leaks into UI copy ("Fixture: …") in Serving, Cloudflare, Models, Skills errors | strip in fixture layer; errors use one component (icon, cause, action) |
| 8 | Medium | `tasks-populated` | Failed run shows plain black text "Fixture: the demo model returned an invalid response." under the row | red inline status "Last run failed · View" (no fixture text) |
| 9 | Medium | `setup-prepare-error` | Error state keeps the heading "Preparing your model" (with a visible focus box around it), shows "Preparation stopped" in plain bold, and ends with fixture text | heading "Couldn't prepare the model", red icon, cause, Retry; focus without visible ring on headings |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `chat-markdown-table` | Status column "Ready" as plain text far right | status chip; tighter column widths |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] Every error state listed below uses ErrorNotice.
- [ ] Find the source of "Fixture:". If it is only in the fixture host, leave the host alone but make sure production code never prints it; if production code adds it, remove it.
- [ ] Errors are announced (role="alert" or aria-live="polite").
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `chat-open-website-invalid-light.png`
- `chat-markdown-table-light.png`
- `chat-composer-error-light.png`
- `serving-requests-all-light.png`
- `serving-request-error-light.png`
- `serving-cloudflare-error-light.png`
- `skills-refresh-error-light.png`
- `tasks-populated-light.png`
- `setup-prepare-error-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t04-error-notice` with message `T04: One error component, no fixture text`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
