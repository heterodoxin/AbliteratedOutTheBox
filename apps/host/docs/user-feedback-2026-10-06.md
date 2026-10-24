# UI annotations — 6 October 2026

Implemented in the Strata design iteration. Preview: http://localhost:5199/.

## Annotation checklist

| Comments | Change |
| --- | --- |
| 1 | Chat and sidebar dividers align at the same height and use the same neutral border token. |
| 2, 6 | Removed composer keyboard hints and workspace-file icon. |
| 3 | Removed the Workspace page and navigation entries; folders group conversations. Old Workspace URLs fall back to Chat. |
| 4, 5 | Theme-aware scrollbars and sliders; vertical spacing around Seed. |
| 7 | Response settings moved beside Conversation context in the top toolbar. |
| 8 | Composer has +, access menu, model picker and thinking menu. Small screens use two orderly rows. |
| 9 | Tasks below Skills. Empty state, templates, search, sorting, manual editor and model-assisted draft editor. Host persists schedules and results; run, stop, pause, edit, archive and restore actions. |
| 10, 11 | Removed Monitor idle card and recent requests. Requests remain in Serving. |
| 12 | Hardware displayed as persistent icon cards for processor, graphics, memory, storage and OS. |
| 13 | Tool approval cards use the same neutral surfaces, borders and spacing as the app. |
| 14, 15 | Removed pause-new-requests and empty-state Connect an app buttons. |
| 16, 19 | Serving summary has URL, model ID and status only; status includes a local-sharing icon action. |
| 17, 18 | Redesigned API-key dialog with generation, masked entry, copy and explicit apply action. |
| 20, 21, 22, 24 | Removed rerun setup, appearance preview, appearance reset and Diagnostics from Settings. |
| 23 | About logo is inside the Version card. |
| 25 | Models and Model files consolidated into Models & files under Storage. |
| 26, 27 | Removed the Engine section from Settings. |

## Website and document previews

Use **+ → Try preview examples** in Chat. The demo chat also responds to “Show preview examples.” with the three example cards. Click a card to open its preview beside the conversation; switch outputs using tabs. Narrow screens use a document sheet.

- Website: sandboxed HTML or an embedded URL, with an Open outside preview action for sites that block embedding.
- Markdown: readable formatting and a Source toggle.
- PDF: lazily loaded PDF.js viewer, page navigation, zoom and extracted page text.
- Open a file reads a local file for preview only; it does not upload it or add it to the model prompt.

## Runtime boundaries

The current development server uses the **Strata demo engine**, with a llama.cpp/OpenAI-compatible adapter available in the code. There is **no DeepSeek harness integration**. Composer access and thinking menus save per-chat preferences; the adapter does not yet enforce those access modes or adjustable reasoning budgets. Their menus identify this limitation.

Tasks are real host-owned schedules for text prompts using the loaded model. They run while the host is open, skip missed runs, prevent overlapping runs of the same task and preserve results. The demo model returns labelled sample answers. File, shell, web, email and calendar tools are not connected to task execution.

Local sharing starts a separate inference-only listener after an API key exists. The UI can start/stop it; it is not automatically enabled. Host settings and files are excluded. Automated sharing checks bound the listener to loopback; LAN sharing was not enabled on the user's machine.

## Verification

- `npm run typecheck` — passed.
- `npm run audit` — clean.
- `npm run build` — passed.
- `node --test tests/*.test.mjs` — 23 passed, 0 failed.
- Browser review at 1654 × 977 and 390 × 844, including light and dark appearance.
- Verified model-assisted draft → create → run → result → pause in an isolated host. Verified manual editor and retired Workspace route.
- Verified website, Markdown Read/Source and both PDF pages, plus streamed artifact cards.
- Inspected revised Monitor, Serving, blank key dialog, Settings, tool approval cards and advanced response settings.
- No browser console errors in the final inspection.
- Original development server remains available at port 5199; isolated review data was separate from user data.

## Screenshots

Screenshots show the working application with built-in sample outputs.

- [Website beside chat](evidence/user-feedback-2026-10-06/chat-website.jpg)
- [Markdown beside chat](evidence/user-feedback-2026-10-06/chat-markdown.jpg)
- [PDF beside chat](evidence/user-feedback-2026-10-06/chat-pdf.jpg)
- [Composer access menu](evidence/user-feedback-2026-10-06/composer-access-menu.jpg)
- [Streamed Markdown preview in dark appearance](evidence/user-feedback-2026-10-06/chat-markdown-dark.jpg)
- [Mobile chat](evidence/user-feedback-2026-10-06/chat-mobile.jpg)
- [Mobile Markdown preview](evidence/user-feedback-2026-10-06/markdown-mobile.jpg)
- [Task with completed result in dark appearance](evidence/user-feedback-2026-10-06/tasks-dark.jpg)
- [Monitor](evidence/user-feedback-2026-10-06/monitor.jpg)
- [Serving](evidence/user-feedback-2026-10-06/serving.jpg)
- [API key dialog](evidence/user-feedback-2026-10-06/api-key-dialog.jpg)
- [Settings About](evidence/user-feedback-2026-10-06/settings-about.jpg)
- [Tool approval card](evidence/user-feedback-2026-10-06/tool-approval.jpg)
- [Light response settings](evidence/user-feedback-2026-10-06/response-settings-light.jpg)
