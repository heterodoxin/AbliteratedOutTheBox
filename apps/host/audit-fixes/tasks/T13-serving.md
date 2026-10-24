# T13 · Serving: remaining findings

**Phase 3** · 19 findings (0 high, 6 medium, 13 low) · Depends on: T04

## Goal
Polish Serving: status chips, unlabelled icon actions, the hidden Connect guide, the table and the API key flow.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/screens/Serving.tsx`
- `src/lib/serving-state.ts`
- `src/components/CloudflareDialog.tsx`
- `server/serving.mjs, server/local-share.mjs`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `serving-request-queued` | Queued request can't be cancelled (only generating shows "Cancel request") | Cancel for queued too |
| 2 | Medium | `serving-key-empty` | Copy icon shown next to an empty key field; "Generate" and "Enter your own" have equal weight | generate on open; "Use my own key" link; copy only when a value exists |
| 3 | Medium | `serving-local-sharing-*` | Local sharing opens from an unlabeled link icon at the end of the Status row | labelled row in the summary card: "Local network · Off · Share" |
| 4 | Medium | `serving-local-sharing-manage` | The shared network URL has no copy button (every other URL does) | add copy |
| 5 | Medium | `serving-local-sharing-active` | While sharing is on, the only sign is the base URL changing to 192.0.2.42 and a pressed icon; Status still says "API listening" | Status: "Shared on local network" with amber dot + Stop |
| 6 | Medium | `serving-empty` | Empty state says "Requests appear here when a client… uses the API" but doesn't offer the Connect-an-app guide, which hides behind the header (i) icon | "Connect an app" button in the empty state and as a labelled header button |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `serving-requests-all` | Summary "1 active · 1 queued · 1 completed" leaves out the error row | add "· 1 failed" |
| 2 | Low | `serving-requests-all` | Status row ends in an unlabeled link icon; header (i) icon unlabeled | tooltip/aria-label, or text button |
| 3 | Low | `serving-request-done` | Footnote "This view updates while the request runs" on a finished request; State values lowercase ("done") | show footnote only while running; sentence case + chip |
| 4 | Low | `serving-guide-*` | Base URL and model ID inside the numbered steps have no copy buttons (only the curl block does); dialog body scrolls with no visible cue, code block runs into the bottom edge | inline copy chips; footer with Done; fade at scroll edge |
| 5 | Low | `serving-guide-*` | Step 4 "This local API has no key configured" reads as a warning but is styled as a normal step | info note with "Set key" link |
| 6 | Low | `serving-key-remove-confirmation` | Key shown as "Required · Ends in DEMO"; body "Existing clients can keep their current key field" is unclear | "Ends in •••3f9a"; "Apps that send the old key will still connect." |
| 7 | Low | `serving-cloudflare-address` | "STEP 1 OF 4" repeats the stepper directly above | drop the eyebrow |
| 8 | Low | `serving-cloudflare-review-quick` | Key limitation ("Turn off streaming in your client") buried in small grey text | amber note callout |
| 9 | Low | `serving-cloudflare-connector-missing` | Only a Homebrew command ("brew install cloudflared") — macOS only | OS-aware command / tabs |
| 10 | Low | `serving-cloudflare-connected` | "Done" and "Stop tunnel" same weight; Stop is destructive | Done primary; Stop as red text button |
| 11 | Low | `serving-tunnel-active` | Header button still reads "Cloudflare tunnel" while a tunnel is live | "Tunnel on" with green dot |
| 12 | Low | `serving-cloudflare-domain-empty` | Stepper scrolls with content and is clipped under the dialog header | pin stepper below header |
| 13 | Low | `serving-key-change` | "Your old key will stop working" is buried in grey helper text | amber note above the buttons |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] `node --test tests/request-flow.test.mjs tests/tasks-and-sharing.test.mjs` passes.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `serving-requests-all-light.png`
- `serving-request-queued-light.png`
- `serving-request-done-light.png`
- `serving-key-empty-light.png`
- `serving-key-remove-confirmation-light.png`
- `serving-cloudflare-address-light.png`
- `serving-cloudflare-review-quick-light.png`
- `serving-cloudflare-connector-missing-light.png`
- `serving-cloudflare-connected-light.png`
- `serving-tunnel-active-light.png`
- `serving-cloudflare-domain-empty-light.png`
- `serving-local-sharing-manage-light.png`
- `serving-local-sharing-active-light.png`
- `serving-key-change-light.png`
- `serving-empty-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t13-serving` with message `T13: Serving: remaining findings`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
