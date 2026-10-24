# UI follow-up · 6 October 2026

## Delivered

| Annotation | Change |
| --- | --- |
| 1 | Expandable Tasks tree, right-side add icon, task detail links and recoverable deletion. Deleted tasks can be restored on the Tasks screen. |
| 2 | Model chip shows model name on the first line and quantization on the second. The design demo uses the requested Qwen3.8-Flash-Next / UD-Q4_K_M label; the underlying demo model and API ID remain unchanged. Real models show their own family and quantization. |
| 3 | Neutral Send control in dark mode. |
| 4 | Conversation context control and panel removed from chat. |
| 5 | Tool approval remains inline; the redundant approval toast was removed. |
| 6 | Skills moved into More. |
| 7 | Cloudflare setup dialog offers a temporary URL or an existing Cloudflare domain, installation help, API-key prerequisite, connection state and Stop. |
| 8 | A connected tunnel's API URL appears in Serving, with copy and manage controls. |
| 9 | Monitor header uses the sidebar surface without a backdrop or fade. |
| 10 | Icon-only Monitor export captures a complete PNG, including sources, time, charts, hardware and icons, and opens a native Save dialog. |
| 11 | Engine header uses the same plain surface. |
| 12 | Conversations row removed from Settings Storage. |
| 13 | Keyboard shortcuts can be reassigned, cleared and restored. Conflicts are rejected; bindings persist in browser storage and drive the actual dispatcher and composer. |
| 14 | Shortcut search removed. |
| 15 | Animated sun/moon theme toggle to the left of More; reduced-motion preference respected. |

Created the user-facing chat **Design review · six previews** at `http://localhost:5199/#/chat/s_qti4osdfvux8`. It contains a website, two-page PDF, Markdown brief, SVG cover, TypeScript example and filterable/sortable CSV table. The right panel opens automatically when the chat is reopened. It stays beside chat when the available content width is at least 900 px; smaller layouts use a dialog.

## Verification

- TypeScript check, UI audit and production build passed.
- All 28 automated tests passed, including tunnel lifecycle, key requirement, token privacy, native save contract/cancellation and shortcut persistence/dispatch.
- Browser-verified task creation, tree navigation and deletion in the isolated test workspace.
- Browser-verified shortcut conflict handling and a reassigned shortcut actually opening a new chat.
- Browser-verified all six preview types, PDF page navigation, table filtering/sorting, dark mode, More, Settings and both Cloudflare setup choices.
- Monitor PNG generated through the app and visually inspected. The export fixture saved the actual browser-rendered PNG to `evidence/ui-followup-2026-10-06/monitor-export.png`. A first pass revealed missing external sprite icons; these are now inlined into the export.
- Native macOS Save picker launch reached the host's `osascript` chooser. Computer-use tooling could not access that standalone chooser, so the save contract/cancellation was unit-tested and the complete PNG was saved through the isolated fixture. Windows/Linux native pickers were not exercised here.
- Cloudflared is not installed on this computer. Tunnel process/URL/stop/error behavior was tested with a controlled connector fixture; no live public tunnel was opened. The modal presents the required installation and API-key steps.

## Cloudflare behavior

A dedicated loopback listener exposes only inference endpoints. Host management routes and workspace files are not served through it. An API key is required before connecting and cannot be removed while a tunnel is active. Tunnel tokens are passed in the child process environment and are not persisted, logged, returned to the client or included in command arguments. Tunnels stop with the host and require an explicit new connection after restart.

Temporary tunnels require non-streaming requests. Domain-based tunnels report connector registration; the user must configure the published application route to the displayed local service URL. Cloudflare account/domain changes are performed in Cloudflare's dashboard.

Sources checked on 6 October 2026:

- [Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
- [Tunnel setup](https://developers.cloudflare.com/tunnel/get-started/)
- [Run parameters and TUNNEL_TOKEN](https://developers.cloudflare.com/tunnel/reference/run-parameters/)
