# Chat shell and rich-content showcase — 6 October 2026

## Delivered

1. Integrated the chat and preview headers into one aligned top bar. Browser and document tabs share a single horizontally scrolling row; the selected tab scrolls into view. Removed the duplicate document heading.
2. Added the visible chat title, subagent details (description, status, elapsed time, output tokens), Chat template and Download Session Log options, and preview fullscreen/collapse controls. Templates populate an editable composer draft. Session-log download/save is deliberately not connected yet.
3. Split attachment cards into a main sidebar-preview action and a menu with sidebar, file explorer and default-app actions. Local examples reveal/open their actual files. Text/blob previews open an exported local copy. Remote websites open in the browser and explain why no local file is available.
4. Sidebar Tasks + opens the shared task modal without changing the route.
5. Removed Manage tasks. Task leaves have visible edit/delete icons; editing prefills the same modal, and deletion requires confirmation. Archived tasks remain recoverable.
6. Added three independent anchored popovers for tokens/second, total chat tokens and context utilization. Statistics use session data, with active monitor context accepted only for the matching session. Missing usage is shown as unavailable. Compaction is explicitly not connected in this build.
7. More sits at the left of the navigation footer, with the theme toggle at the far right.

## Showcase

Open: http://localhost:5199/#/chat/s_tvgo6un2muia

Title: **Chat showcase · code, tools & media**. Includes a TypeScript block, completed and pending example tool calls, an SVG image, an original four-second MP4, an animated GIF with pause/resume, a Markdown table/quote/list, and six existing website/document/file previews. Agent and metric fixtures are labeled as examples; approving the example does not execute a tool. Recreate via composer **+ → Try chat showcase**.

Media assets are local and authored by `scripts/assets/make-chat-media.py` (Pillow and imageio-ffmpeg; generation-only dependencies). Video responses support byte ranges and media MIME types.

## Verification

- `npm run typecheck`, `npm run audit`, `npm run build`: pass.
- `node --test tests/*.test.mjs`: 30 pass.
- Added tests for session-stat isolation and native file-action path/symlink validation, using a mock launcher.
- Browser: add/edit/delete with unchanged chat URL, prefilled schedule, confirmation, persistent task icons, subagent details, templates, session-log placeholder, unified tabs, expand/collapse, all three independent popovers, and 390 px responsive layout.
- Native: PDF revealed and selected in Finder, then opened in Preview by the default-app action. Test windows closed afterward.
- Media: four-second video loaded and played to its end; GIF pause swaps to the still poster and resumes to the GIF. HTTP range request returned 206 with `video/mp4` and the expected byte count.
- Task mutations were performed only on an isolated host. The main app retains its previous chats and adds the requested showcase.

![Chat, code, approval and integrated image preview](evidence/chat-shell-2026-10-06/chat-showcase.jpg)
