# Browser feedback implementation — 5 October 2026

Implemented the 14 annotated corrections supplied by the user. The DeepSeek screenshots informed hierarchy and density; Strata’s existing fonts, icons, colours and surfaces remain in use.

## Disposition of each comment

| Comment | Change | Verification |
| --- | --- | --- |
| 1 — Setup header spacing | Added 24 px below the setup header before progress steps. | Desktop and narrow setup inspected. |
| 2 — Visible computer specifications | Removed the Computer details disclosure. CPU, every GPU, memory and storage appear in an icon-led grid that wraps when necessary. | Mac hardware and a synthetic two-GPU PC rendered. |
| 3 — RAM + VRAM total | Capacity illustration totals system RAM and dedicated GPU memory. Unified memory is counted once. Individual GPU totals/free readings remain visible when reported. | Unit tests: 16 GB unified stays 16 GB; 32 GB RAM + 24 GB + 12 GB GPUs shows 68 GB. Browser screenshot confirms both GPUs appear. |
| 4 — Simpler model review | Default setup shows readable model name, download state/size, fit and memory preview. One Advanced button opens exact identity, tuning and performance details. | Opened and closed Advanced; focus returns to its trigger. |
| 5 — Context is the highlight | Context dial is always visible beside the recommendation, stacked above it on narrow screens. Circular pointer dragging updates the selected supported stop and memory preview. Approximate numeric entry rounds to the closest supported stop. | Real pointer drag: 32K → 16K, estimated memory 1.9 → 1.8 GB. Arrow key: 16K → 8K. Advanced input 5000 → 4096. Unit coverage for angular wrap and rounding. |
| 6 — Repeated setup text | Shortened preparation and completion copy. Exact variant/profile/context details moved to Advanced or Setup details. Demo remains explicitly labelled. | Resumed choices and completed simulated loading through Start chatting. |
| 7 — Collapse beside logo | Expanded header places an icon-only collapse control beside the logo. Collapsed header contains the expand icon in place of the logo. | Desktop collapse/restore and compact overlay checked. |
| 8 — Serving table spacing | Added 20 px between the filter/search toolbar and request table container. | Measured 21 px to the table itself including its border; rendered a completed demo request. |
| 9 — Skills placement | Skills now follows New chat directly. | Expanded, compact and phone navigation checked. |
| 10 — Workspace search | Search icon sits beside Workspaces and add-folder. Activating it replaces that header with a search input and close action. Escape clears/closes and restores focus. | Opened search, entered a nonmatching query and closed with Escape. |
| 11 — API-key field spacing | Label, masked input, visibility control and help use consistent form spacing. | Blank key dialog opened; label/input gap measured at 8 px. Cancelled without changing credentials. |
| 12 — Simple Skills rows and editor | Icon, title, short description, edit icon and toggle. Scope, trigger phrases and instructions move into one editor. Name, description, instructions and triggers now genuinely save as host-local overrides. | Edited a disposable skill, saved, refreshed and reopened. Host tests cover persistence, invalid content and missing IDs. |
| 13 — Quiet chat header | Removed visible title/model/message/token summary from the left header. Retained an accessible heading and the context control on the right. | Created a chat and inspected its desktop and phone layouts. |
| 14 — Sidebar hierarchy | Shared logo/control row, New chat, Skills, workspace header and indented conversation rows. Removed repeated Recent/Pinned subheads inside each workspace; pins and relative times remain compact. | Sidebar screenshot and navigation interactions checked. |

## Additional defect found during verification

At phone widths the setup surface had a maximum height with `overflow: hidden`, clipping the lower controls and preventing wheel scrolling. It now scrolls vertically and returns to the top when moving to another step. Verified by scrolling to Load model, completing preparation, and entering Chat at 390 px.

## Memory interpretation

The combined number is **installed capacity**, not a guarantee that a model fits. RAM and dedicated GPU memory are separate pools; placement and backend support still determine usable budgets. Existing fit calculations continue to use those budgets. Shared Mac memory is never added twice. Free VRAM is shown only when the host reports it; otherwise the row says it is not reported. Storage is not labelled SSD because the current host does not detect drive type.

The 68 GB PC screenshot uses a synthetic fixture (32 GB system RAM; GPUs with 24 GB and 12 GB). It verifies presentation, not actual detection or multi-GPU execution.

## Skill editing behavior

Edits persist in Strata’s host configuration and overlay the discovered skill by ID. Original bundled/workspace `SKILL.md` files remain intact. Validation rejects missing/oversized required content, malformed trigger phrases and unknown skill IDs. Toggles and workspace scope keep their existing behavior. The editor explains where changes are saved.

## Verification completed

- `npm run typecheck`: passed, including the final scrolling correction.
- `npm run audit`: passed; 28 icon references resolved.
- `node --test tests/*.test.mjs`: 20 passed, 0 failed.
- `npm run build`: passed after the final correction.
- CUA browser interactions: dial drag, keyboard adjustment, approximate entry, reset, Advanced dialog, interrupted setup/resume, simulated model loading, Start chatting, sidebar collapse/restore, workspace search/Escape, skill save/refresh/reopen, blank API-key dialog and request table.
- Visual checks: 1333 px desktop, 1024 px compact and 390 px phone; Light and Dark. No console errors captured in the final review tab.
- Verification used isolated demo hosts and test data. The user’s preview data was not used for test edits. The existing `http://localhost:5199/` preview remains available.

Production engine integration, native model performance and actual PC hardware detection were not exercised by these UI checks.

## Screenshot evidence

All images are in `docs/evidence/user-feedback-2026-10-05/`:

1. [Computer summary](evidence/user-feedback-2026-10-05/01-computer.png)
2. [Model choice and visible context dial](evidence/user-feedback-2026-10-05/02-model-and-dial.png)
3. [Simplified Skills](evidence/user-feedback-2026-10-05/03-skills.png)
4. [Skill editor](evidence/user-feedback-2026-10-05/04-skill-editor.png)
5. [Quiet chat and sidebar](evidence/user-feedback-2026-10-05/05-chat-and-sidebar.png)
6. [API-key dialog spacing](evidence/user-feedback-2026-10-05/06-key-dialog-spacing.png)
7. [Serving filters and table](evidence/user-feedback-2026-10-05/07-serving.png)
8. [Synthetic PC with two GPUs](evidence/user-feedback-2026-10-05/08-pc-multiple-gpus.png)
9. [Narrow setup with reachable actions](evidence/user-feedback-2026-10-05/09-narrow-setup.png)
10. [Narrow Skills](evidence/user-feedback-2026-10-05/10-narrow-skills.png)
11. [Dark Skills and compact navigation](evidence/user-feedback-2026-10-05/11-dark-skills.png)
