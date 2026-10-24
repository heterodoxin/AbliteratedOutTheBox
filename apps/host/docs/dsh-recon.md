# DSH packaged-app reconnaissance

Source: `/Applications/DeepSeek Harness.app/Contents/Resources/app.asar` (read-only; version `0.2.0-rc.2`, `dsh/desktop-runtime.json` → `hostProtocolVersion: 4`, node `24.18.1`, platform `darwin/arm64`).
Method: asar header parsed (`dataOffset = align4(16 + uint32@12)`), 288 `@deepseek-ai/*` packages extracted to `/tmp/dsh-recon/`, READMEs + `lib/*.js` + `lib/*.css` read and grepped. Every claim below cites a package/file. "not discoverable" = not present in the packaged code.

---

## 1. Runtime topology

**One CLI, layered "profiles", one web surface.** `dsh` is "the sole supported Node application launcher: profiles are ordered stacks of plugin-bundle patch layers" (`dsh/README.md`). Profiles live under `$DSH_HOME/profiles/<name>` (`$DSH_HOME` else `~/.dsh` — `lib/main.js:25` `DSH_HOME_DIR_NAME = ".dsh"`, `dsh-home-paths`). Layer order: bundle patches in `dsh.profile.bundles` order → profile `cordis.patch.yml` → `$DSH_HOME/cordis.patch.yml` → `--patch` overlays. Shipped bundles: `dsh-base`, `dsh-web-app`, `dsh-headless`, `dsh-sdk-app`, `dsh-sdk-minimal`, `dsh-acp-app`. The `desktop` profile name is reserved for Electron and rejected by the npm CLI.

| Surface | Entry | Notes |
|---|---|---|
| Web | `dsh web` / `dsh --profile web` | `dsh-web-app/lib/startup.js`: flags `--host <host>`, `--port <port>` (0 = OS-assigned), `--trusted-host <authority...>`, `--no-open`. `--host 0.0.0.0` is a **usage error**: "it would expose remote code execution to the network; use 127.0.0.1 instead". |
| Desktop | Electron `lib/main.js` | Spawns a child Node process, does not embed the host. |
| Headless / SDK / ACP | `dsh --profile headless\|sdk\|sdk-minimal\|acp` | Same plugin kernel, no browser. |

### Host: HTTP server
`dsh-host-webserver` is a plain `node:http` server that "knows no harness concepts and serves no files". Config `{host: 127.0.0.1 | 0.0.0.0, port: 3000, compression: 'gzip'}` (README example uses port **3000**; `port: 0` → OS-assigned, read back via `ctx.webServer.port`). Matching: **exact table → longest prefix → single fallback handler**; `register(route)` / `registerUpgrade(route)`; duplicate path throws. Index rendering = two layers: `collectIndexInjections()` gathers structured `IndexInjection` rows (one `webserver/index-inject` emit per call), then `applyIndexTaps(html)` raw transforms. No TLS/auth/origin policy of its own — route owners enforce theirs. Shipped web bundle uses gzip level 1 with a 1024-byte threshold.

### Frontend: the built web shell
`@deepseek-ai/dsh-web-frontend` — package description: *"Web application entry: vite build over the `@deepseek-ai/dsh-client-web` shell library; `dist/` served by apps/cli's `dsh web`"* (repo `apps/web`). Contents: `dist/index.html` (825 B, `<div id="root">`, KaTeX fonts under `dist/assets/fonts/`), `assets/index-5SrrfWpU.js` (633 KB), `assets/vendor-CCJJTK99.js` (740 KB), `assets/index-BPHePDI_.css` (76 KB), `assets/vendor-BNsW4eBh.css` (29 KB).

Served by `dsh-host-frontend-static`, which claims the fallback seat: root/configured index render `index.html` through `ctx.webServer.renderIndex`; other existing files served with MIME; unknown extension → `application/octet-stream`; traversal → **403**; missing → empty **404** (no silent SPA fallback); non-GET/HEAD → **405**. It injects `<base href="./">` ahead of every injected row. Index access needs `ctx.connection.authorizeIndex`: a valid process token gets **303 + persistent browser cookie**, a valid cookie serves the index, everything else gets the Connection-owned **401**. Static assets stay public.

### `window.__DSH_BOOT__`
Composed by `dsh-client-modules` (host half): it "scans the host Loader's entries for packages declaring `dsh.client`, composes the `window.__DSH_BOOT__` entry graph … in module-graph order, serves one-or-more-plugin combo scripts …, contributes the registration facade, application preloads, bootstrap scripts, and graph to the webserver's index injection table". The graph is emitted as one injection row:

```
{ kind: "global", name: "__DSH_BOOT__", value: graph }
```

which `dsh-host-webserver/lib/index.js` renders verbatim as
`<script>globalThis["__DSH_BOOT__"] = {…}</script>` (placed in `<head>`, `<` escaped to `\u003c`). Companion rows: `script-preload` for `application`-phase batches, `script-src` for `bootstrap`-phase batches, and a raw `script` row carrying the module-loader queue. The browser bundle then reads it: `await globalThis.__DSH_BOOT_READY__?.promise; const o = n.__ModuleLoader__; const s = globalThis.__DSH_TRANSPORT__; this.modules = o.create({ boot: n.__DSH_BOOT__, staticModules: rM(), ...s?.loadBundle && { loadBundle: s.loadBundle } })` (`dsh-web-frontend/dist/assets/index-5SrrfWpU.js`).

### Client plugin model (slots + `/plugins`)
- A browser plugin declares `dsh.client` in its `package.json` (`platform: 'web'`, exports `./client`, optional `dsh.client.external`). Host bundles are served under **`/plugins`**; `<id>/client` and bare id resolve to the same exports. Boot scripts register factories; module bodies run lazily at first materialization. Failed combo → one retry, then per-row one-resource combo (≤3 requests per missing row).
- Shared baseline module table `PLATFORM_MODULES` (React, Cordis, static UI libraries) is frozen by the shell; only `dsh.client.external` entries may resolve elsewhere. Rejects malformed requests, missing suppliers, self-requests, sync cycles.
- UI composition is **typed slots**: `dsh-client-ui-slots` provides `single` / `list` / `keyed` / `chain` slots plus Component Factories. "Declaring a slot is claiming it" — the only entry allowed to render a key is its declarer; registering into an undeclared slot throws at load. `ChainSelect` elects by ascending `priority`, ties by registration order. The React implementation and install contract live in `dsh-client-ui-renderer` ("after the complete client plugin roster settles, the boot kernel calls `ctx…`").
- Live updates: `dsh-client-hmr` "keeps open Web pages in sync with the Host plugin graph and reloads rebuilt browser bundles"; ordinary enable/disable needs no page reload. The `dsh-client-connection` package carries browser→Host Remote calls, exact Fetch responses and "connection generations" over the `/api` bridge (`dsh-api-gateway` owns the two-sided RPC endpoint and `REMOTE_STREAM_MUX_PATH` websocket mux).

### Desktop boot chain
`app.asar/lib/main.js` → `spawn(node, ["--expose-internals", …, "<runtimeDir>/node_modules/@deepseek-ai/dsh-desktop-host/lib/index.js", runtimeDir, projectDir, primaryRuntime, pnpm, nodeBin], {stdio: [ignore, pipe, pipe, ipc]})`. The child (`@deepseek-ai/dsh-desktop-host`, *"Private Node-mode host process for the Electron desktop application"*, mounts `dsh-host-webserver`, `dsh-agent`, `dsh-client-connection`, `dsh-jobs`, `dsh-schedule`, `dsh-workspace`, `dsh-skill-office`, `dsh-deepseek-account`) replies over IPC `{type:'ready', url, injections}`; main then `window.loadURL(<that url>)`. Other IPC events: `platform-session`, `shutdown-complete`, `fatal`, plus a request/response control channel. Preloads shipped: `preload-app.cjs`, `preload-mandatory.cjs`, `preload-platform-account.cjs`, `preload-update-dialog.cjs`, `preload-welcome.cjs`; separate renderer documents for welcome / mandatory-update / update-dialog. The preload sets `html[data-platform='darwin']` (macOS) and `data-windows-titlebar` (Windows); these switch frame geometry.

URL announcement: `dsh web: <authenticatedUrl> (LAN: …)`, `DSH_WEB_URL` is exported to the model's shell env, and `webRuntime` service + `web-app` plugin add the "You are interacting with the user through the DeepSeek Harness Web GUI at <url>" system-prompt section.

**Ports:** config example 3000; `--port 0` for ephemeral; Electron takes whatever the desktop-host reports; loopback host is hardcoded `127.0.0.1` for the canonical URL (`LOOPBACK_HOST`).

---

## 2. Client package map

### Shell / sidebar / layout
| Package | Responsibility | UI region |
|---|---|---|
| `dsh-client-ui-layout` | Three-column `AppFrame`, edge-column widths, `ctx.layout.selectPanel/toggleSidebar/openRightbar/closeRightbar`, theme presenter | whole frame |
| `dsh-client-ui-renderer` | Mounts the assembled client GUI, React slot renderer + install contract | app root |
| `dsh-client-ui-slots` | Slot/factory registry (React-free) | infra |
| `dsh-client-ui-sidebar` | Brand mark+name, New Session, `sidebar.panellist` global entries, collapse/rail, bottom-pinned Settings | left column |
| `dsh-client-ui-workspace` | Workspace + Session catalog: grouped/flat lists, add/rename/reorder/search/fork/archive/delete, row hover menus | left column body |
| `dsh-client-ui-sidebar-right` | Right dock: one surface per session, two presentations (docked/fullscreen), expand button, `rightbar` controller | right column |
| `dsh-client-ui-sidebar-files` | Workspace file tree with watched auto-refresh | right tab |
| `dsh-client-ui-sidebar-terminal` | Shell picker + persistent terminals, rename, recovery | right tab |
| `dsh-client-ui-sidebar-browser` | HTTP(S) browsing: iframe+managed history (web) / `<webview>` (desktop) | right tab |
| `dsh-client-ui-sidebar-documentpreview` | File preview + renderer choice (Markdown/code/PDF/HTML/images/sheets) | right tab |
| `dsh-client-ui-theme` | light/dark/system, content font size 10–22 px, token stylesheets | global presentation |
| `dsh-client-ui-brand-official` | Swaps DSH mark+name into the sidebar for an `official` build | sidebar brand |
| `dsh-client-ui-open-in-app` | Session-header split button: open `cwd` in remembered app | header |

### Conversation / chat
| Package | Responsibility | UI region |
|---|---|---|
| `dsh-client-ui-conversation` | Target-neutral assembly: event/view registries, per-session bindings, input state, image URL cache, composer shell | center |
| `dsh-client-ui-chat` | The Chat target: recorded-conversation rendering, historical images, scroll restore, "work details" modes | center |
| `dsh-client-ui-trajectory` | Turn-aware ledger + timing overview + record inspector | center (tab) |
| `dsh-client-ui-tool` | Renders every tool call, dispatched by `conversation.tool-call` key | inside chat |
| `dsh-client-ui-primitives` | Controls, icons, overlays, Markdown/TeX, terminal, read, diff, search, web, JSON renderers | everywhere |
| `dsh-client-ui-attachment` | Draft rail under composer, full-viewport drop invitation, durable images, lightbox | composer + messages |
| `dsh-client-ui-commands` | `/` command popups, client actions, host command input; `/model`, `/permission` popupSelect specs | composer |
| `dsh-client-ui-input-trigger` | `/` and `@` grouped menu at caret (commands, files, sessions) with drill-down | composer |
| `dsh-client-ui-reference` | `@` completion ordering: files before sessions | composer |
| `dsh-client-ui-message-feedback` | Like/Dislike in the assistant action strip + feedback dialog + `/feedback` | message actions |
| `dsh-client-ui-approval` | Approval presentation; takes over the composer; Enter=allow, Escape=reject | composer takeover |
| `dsh-client-ui-user-questions` | Agent question panel beside the input: options, text, skip, batch submit | beside composer |
| `dsh-client-ui-plan` | Plan mode chip, plan review card, plan preview resource | composer + card |
| `dsh-client-ui-goal` | Durable goal + activation: edit/pause/resume/clear, `/goal` bubbles | chat node |
| `dsh-client-ui-jobs` | Session background jobs: one header control, lifecycle/elapsed/progress, expandable output | header |
| `dsh-client-ui-deliverables` | Turn-end changed-files card, delivered-file cards, inline-code links into the sidebar | turn tail |
| `dsh-client-ui-skill` | `/skill` and `/name` skill invocation surface | composer |
| `dsh-client-ui-model-selection` | `/model` popup + composer model control (provider-grouped, reasoning effort) | composer |
| `dsh-client-ui-agent-preset` | Preset chooser + new-task default + Creator-mode authoring | hero / settings |
| `dsh-client-ui-permission-presets` | Preset picker for session (composer, `/permission`) vs future default (settings) | composer + settings |
| `dsh-client-ui-subagent` | Browse descendant conversations, running state, tokens, active-turn duration | overlay/panel |
| `dsh-client-ui-workflow-run` | One durable workflow run per Chat node; phases → members | chat node |
| `dsh-client-ui-schedule` | Automation tasks page + right-sidebar task tab | overlay/tab |
| `dsh-client-ui-cordis` | Historical generated-plugin cards + process-local control panel | chat node |
| `dsh-client-ui-directory-picker-browse` / `-native` | In-app directory browser / OS chooser, filling `ui-workspace` directory-flow slots | modal |

### Settings
| Package | Responsibility |
|---|---|
| `dsh-client-ui-settings` | Transport+schema layer: `ctx.configForms` namespaces, atomic multi-field updates, concurrent-write protection, `whileServed` |
| `dsh-client-ui-settings-general` | The Settings shell: `sidebar.settings` occupant, navigation, modal panel, onboarding steps, connection pill, config-file action |
| `dsh-client-ui-settings-models` | Models page: API keys (write-only), per-provider model lists, hand-declared custom routes |
| `dsh-client-ui-settings-plugins` | "Built-in plugins" section shell owning the tab row (`settings.plugins.tab`) |
| `dsh-client-ui-settings-plugin-inventory` | Read-only "Plugin list" tab: agent presets (open) + collapsed global inventory + search |
| `dsh-client-ui-settings-account` | Account/login section (STRIP for Strata) |
| `dsh-client-ui-settings-shell` / `-agent-loop` / `-subagent` / `-web-search` / `-session-log` | Official plugin pages: Shell, Agent loop, Subagent, Web search, session-log upload |
| `dsh-client-ui-shortcuts` | "Keyboard shortcuts" dialog (Mod+/): search, recorder, restore defaults |
| `dsh-client-ui-plugin-manager` | Plugins sidebar entry: install/enable bundles, pnpm output, install-script approval |

### Infra (non-visual)
`dsh-client-store` (React-free observables/snapshot stores, Immer updates, optional persistence), `dsh-client-modules` (boot graph + `/plugins` bundles), `dsh-client-shortcuts` (per-device keybinding registry/persistence), `dsh-client-locale` (en/zh + plugin locales), `dsh-client-connection` (Remote calls + connection generations), `dsh-client-resources` (`dsh-resource://` addressing), `dsh-client-file-upload` (Blob/stream intake per Session), `dsh-client-hmr`.

---

## 3. Screen & navigation inventory

### Left sidebar (`dsh-client-ui-sidebar`, `-workspace`)
Top→bottom: **brand row** (`sidebar.brand.mark` + `sidebar.brand.name`; fallback is the fish mark + localized `common.brand.localBuild` "local build"; a complete build stacks a `version[-commit][-dirty]` badge from `DSH_CLIENT_VERSION` / `DSH_CLIENT_COMMIT_HASH` / `DSH_CLIENT_GIT_DIRTY`) → **New Session** button (label `session.new.label` = "New session"; effective binding shown as trailing grey text on hover/focus) → **global panel entries** (`sidebar.panellist`; *"The shipped composition registers no example panel"*, labels still present: "Global panels") → **workspace/session list** → bottom-pinned **Settings**.
Sidebar labels: "New Session", "Open sidebar", "Collapse sidebar", "Settings", "Global panels".
Session list (`ui-workspace` copy): "Search session names", "Filter sessions", "Group by", "Order by", "Last updated", "Show archived" / "Archived only" / "All conversations (show archived)", "Hide archived", "No sessions yet", "No matching sessions", "No archived sessions yet", "Archived sessions cannot be opened. Unarchive it to view.", "Selecting or adding a workspace", "Loading workspaces…". Row hover buttons + row menu: **Rename session**, **Fork session** ("Could not fork the session. Try again."), **Archive session**, **Pin session** ("Pin failed. Try again later."), "Plan awaiting review" badge. Subagent-origin sessions are hidden; pending interactions show warning dots.
Collapse: 264–420 px (default 280), **56 px rail**, auto-collapse below **1024 px**, macOS desktop hides the column entirely and mounts reopen/New-Session controls into the frame's `shell.leading` seat beside the traffic lights. Scrollbars are pointer affordances: thumb transparent when the pointer is outside the column, drawn for 2 s after it leaves.

### Conversation header (top bar)
`conversation.session.header` with `leading` / `actions` / `corner` / `lineage` / `utilities` slots. Known occupants: open-in-app split button (`workspace.openLocal`, "Open requested", "Opening containing folder…"), **jobs control** ("Background jobs"), **context-occupancy ring + percentage** ("the button stays hidden until context usage and capacity are available"), right-sidebar expand button, more-actions menu (contains **Download session log**, `dsh-session-log-export`), subagent/workflow lineage chips, the View tab bar (**Chat** / **Trajectory**, hidden when fewer than two views are available).

### Settings dialog
Opened from the sidebar's bottom control or the Settings command (desktop `Mod+,`, web `Mod+Alt+,`); a modal panel using the frame's overlay inset with ≥24 px viewport margin; navigation scrolls independently of content; exactly one onboarding step mounts at a time.
Sections = `settings.section` contributions. Observed: **General**, **Models**, **Account**, **Plugins** ("Built-in plugins"), **Keyboard shortcuts** (entry labeled "Edit shortcuts", "follows Coding Tools"). The Plugins section owns a tab row: **Plugin list** (read-only; agent presets open, global inventory collapsed, "Search plugins"), plus the **Official** group pages **Shell**, **Agent loop**, **Subagent**, **Web search**, **Session log**.
General contains: **Coding Tools** switch (`ui-settings.enabled`, default `true`), **Appearance** row (from ui-theme), **Current version** at the bottom, **Open configuration file** (loopback browser only, host-confirmed), **Upload Session Log when using the official model API**, and rows registered into `settings.general.item`.
Shell page copy: "Command timeout (ms)", "Output cap per stream (bytes)", "Overridden" badge, "Reset to default", "Nothing is written until **Save**".
Models page copy: "Add model provider", "Custom model API", "API protocol", "Base URL", "API key", "Display name", "Context window", "Add model", "Delete model", "A capacity must be a number, optionally suffixed K or M.", "Context window must be a positive count, like 131072, 256K, or 1M.", "Anthropic Messages", "Configured — enter a new value to replace", "Customized model catalog".

### Overlays / modals
| Overlay | Evidence |
|---|---|
| Generic modal (mask + card) | `Modal.module.css`: fixed inset 0, `z-index: 1000`, `padding: max(24px, var(--dsh-frame-overlay-top, 24px)) 24px`, dialog `width: min(380px, 100%)`, mask uses `--dsw-mask-blur` + `--dsw-alias-bg-mask-1` |
| Menus | `Menu` / `MenuSurface` / `MenuGroup` modules; menu surface fill `--dsw-menu-surface-fill: #f8f9fa94` |
| Keyboard shortcuts dialog | 480×600 CSS px card, header + search fixed, internally scrolling list, inline recorder, "Restore All Defaults" footer with override count |
| Approval | composer takeover; "Waiting for approval", "Approval details", "Allow once"; Enter allows, Escape rejects |
| Ask-user question panel | "Ask question", "Waiting until you answer", "Type your answer", "Take time", "Skip", "Dismiss all questions" |
| Directory picker | "Select Workspace Directory" browse dialog (list/navigate/create) or the native OS chooser |
| Plugin manager | install guide, pnpm output panel, "Install scripts need permission" dialog, "Allow these scripts and retry" |
| Risk confirmation | `RiskConfirmation.module.css`; "I understand the risks and want to continue" (Full access / Auto review) |
| Image lightbox | `ImageLightbox.module.css`, "Close original image preview" |
| Toasts / hover cards / tooltips | `Toast.module.css`, `HoverCard.module.css` (`--dsw-hovercard-bg: #2C2C2E`), `Tooltip.module.css` |
| Plan review card | "Plan review", "View full plan", "Open plan in sidebar", "Request changes" |
| Update dialogs (desktop only) | renderer/update-dialog.html, renderer/mandatory-update.html |

### Keyboard shortcuts (defaults read from registration code)
| Keys (desktop / web) | Action | Source |
|---|---|---|
| `Mod+N` / `Mod+Alt+N` | New session (`session.new`, aliases "new session", "new chat") | `ui-workspace` |
| `Mod+K` / `Mod+Alt+K` | Search sessions (`session.search`) | `ui-workspace` |
| `Mod+O` / `Mod+Alt+O` | Add workspace / open folder (`workspace.add`) | `ui-workspace` |
| `Mod+Alt+G` | Rename session (`session.rename`) | `ui-workspace` |
| `Mod+Alt+F` / `Mod+Shift+F` | Fork session (`session.fork`) | `ui-workspace` |
| `Mod+Shift+A` / `Mod+Alt+A` | Archive session (`session.archive`) | `ui-workspace` |
| `Mod+B` / `Mod+Alt+B` | Toggle left sidebar (`shell.overlay`, label "Toggle left sidebar") | `ui-layout` |
| `Mod+Alt+B` / `Mod+Shift+B` | Toggle right sidebar float/fullscreen | `ui-sidebar-right` |
| `Mod+,` / `Mod+Alt+,` | Open/close Settings (`settings.open`) | `ui-settings-general` |
| `Mod+/` | Open/close Keyboard shortcuts (`settings.general.item`) | `ui-shortcuts` |
| `Mod+P` / `Mod+Alt+P` | Files sidebar | `ui-sidebar-files` |
| `Ctrl+\`` | Terminal | `ui-sidebar-terminal` |
| `Mod+T` / `Mod+Alt+T` | Browser tab in right sidebar | `ui-sidebar-browser` |
| `Mod+Alt+O` / `Mod+Shift+O` | Open workspace in app (`workspace.openLocal`) | `ui-open-in-app` |
| `Enter` | Send/approve; `Shift+Enter` newline; `Mod+Enter` steers all queued messages | `ui-conversation` |
| `Escape` | Stop response (`response.stop`); also closes top dialog / stops menu | `ui-conversation` |
| double `Escape` | Host `stopSequenceMs` setting, default **500 ms**, range 1…2,147,483,646 | `dsh-client-shortcuts` |
| `Mod+2` / arrows / Enter | segmented-tab & menu navigation | `ui-conversation`, `ui-shortcuts` |

Binding rules: each command declares defaults per profile `desktop:{macos,windows,linux}` and `web:{macos,windows,linux}`; an omitted profile is unbound. Web persists overrides in origin-local `localStorage` key **`dsh.keybindings.v1`**; Desktop in **`userData/keybindings.json`** (v1 → v2 adds optional `secondCode`). Only overrides are stored; **`null` clears a binding**; Restore All affects the current profile only. Effective bindings take priority over native actions, editors, terminals and embedded pages.

### Command palette
**There is no command palette.** The nearest equivalents are (a) the `/` slash popup — `dsh-client-ui-commands` + `ui-input-trigger`, grouped commands / files / sessions with drill-down and a "Loading options…" state, and "a command line is never silently downgraded to a plain prompt"; (b) the searchable **Keyboard shortcuts** dialog (`Mod+/`) which lists every registered command, its English alias and its key, matching "an ordered subsequence within one label, English alias, or key combination, ignoring case"; (c) the `/model` popup for model+effort selection.

---

## 4. Conversation / chat UX mechanics

**Assembly.** `ui-conversation` consumes `SessionEventLikeEntry` feeds (outer `type` = durable vs client-only transient; inner `event` = `{type, seq, time, data}`), with match/start/update Definitions, Turn/Step location, and Group Definitions that own segmentation. Views are pluggable targets; **Chat** and **Trajectory** are the shipped ones (`conversation.view`). "With Coding Tools off, the shell hides only Trajectory; other plugin-contributed Views remain available."

| Mechanic | Behaviour | Evidence |
|---|---|---|
| Message list | Turn/step-grouped timeline with incremental append/prepend, replacement-window rebuilds, "Click to load earlier history" / "Load earlier history…", "Back to bottom", restored scroll position | `ui-conversation`, `ui-chat`, `ui-trajectory` |
| Streaming | Assistant deltas arrive as transient `assistant/live-chunk` entries; settlement removes transient matches and applies one durable entry; text renders token-by-token; stop is `response.stop` (Escape) | `ui-conversation` |
| Reasoning / thinking | "Deep diving" + shimmer tokens (`--dsw-alias-label-deep-diving`, `-shimmer`), "Analyzing the request", disclosure rows; work-details modes: `settings.transcript.{compact,standard,detailed,verbose}` and `settings.performance.{compact,detailed}` | `ui-chat`, `ui-theme`, slot keys |
| Tool calls | One renderer per `conversation.tool-call` key via `ui-tool`; strings: "Called tools" / "Calling tools", "Execution requested", "Result cap reached.", "Content truncated", "Collapse output", "Command failed", "Collapse diff", "Inspect call", "Edited files" / "Editing files" | `ui-tool`, `ui-chat` |
| Approvals | The approval plugin takes over the composer and publishes the pending request through `ctx.uiSession`; `approval.allow` = Enter, `approval.reject` = Escape; copy: "Waiting for approval", "Approval details", "Allow once"; one pending-request lock shared by keyboard and pointer | `ui-approval` |
| Permissions | Presets **Read Only**, **Workspace Write**, **Full access**, **Auto review**; session-scoped pickers in the composer and `/permission`, future-session default in General settings; "Enable Full access?", "I understand the risks and want to continue" | `ui-permission-presets` |
| Edit / retry / branch | Assistant action strip (`conversation.chat.assistant-actions`), "Branch into a new conversation", "Available only on the last message of a completed turn", copy actions, "Cancel editing" | `ui-chat`, `ui-conversation` |
| Feedback | Like/Dislike pair in the finalized assistant strip; both open a dialog collecting a category + optional description; `/feedback` decoration; toasts on ack/failure | `ui-message-feedback` |
| Composer | One Lexical editor per Session (`DraftEditorRuntime` / `DraftEditor`), attachments as an ordered draft rail + full-viewport drop invitation, `@` references (files before sessions, `@"path with spaces"`), `/` commands claimed/highlighted while only arguments change, IME-safe hints, `Cmd/Ctrl+Enter steers all queued messages` | `ui-conversation`, `ui-attachment`, `ui-reference`, `ui-input-trigger` |
| Attachments | Browser stores a Blob / exact bytes / ReadableStream per Session and gets an opaque receipt for a later prompt; durable images render in Chat, Trajectory and Tool results with a lightbox | `dsh-client-file-upload`, `ui-attachment` |
| Context / token metering | Context-occupancy **ring + percentage** below the input card, opening a token breakdown panel; session stats add "Avg time to first token (TTFT)", "Cache hit", "Cache write", "Cached input", decode wall time; compaction: "Compacting context…", "Context compacted", "Compaction summary unavailable", `/compact` | `ui-conversation`, `dsh-session-stats`, `dsh-compaction` |
| Session titles | First eligible human message → title, with optional async LLM generator (`dsh-session-title-llm`, `-first-prompt-llm`), explicit user rename, `session.untitled` fallback | `dsh-session-title*` |
| Session list / persistence | Append-only JSONL per session with checksummed Zstandard frames (raw lines when compression off), versioned format generations v0→v4 with migrations, projection cache for history lists/stats/goals, FTS5 search | `dsh-session-persistence-jsonl`, `dsh-session-format*`, `dsh-session-projection*`, `dsh-session-query-sqlite` |
| Plan mode | Enter with `/plan`, leave with the composer chip ("Plan mode on — click to turn off (/plan off)"), submitted plan becomes a review card ("Plan review", "View full plan", "Request changes"), plan preview is a sidebar resource ("Plan · Markdown") | `ui-plan`, `ui-sidebar-right` |
| Goals | Durable objective across resume/fork/restart plus process-local activation; states "Ongoing Goal", "Paused Goal", "Blocked Goal", "Inactive Goal"; actions Edit / Pause / Resume / Clear / Save goal; `/goal` runs render as "Command input" bubbles; `dsh-goal-round-driver` auto-continues while idle and rounds remain | `ui-goal`, `dsh-goal`, `dsh-goal-round-driver` |
| Jobs | One header control for the session's background jobs with lifecycle, elapsed time and progress; live and settled-with-output jobs expand; **collapsing stops the stream**; "Confirm stop", "Click again to confirm", "no exit code" | `ui-jobs`, `dsh-jobs-local` |
| Deliverables | Turn-end changed-files card with host line counts, each opening the turn's review tab in that file; explicit delivered-file cards; inline-code references in closing prose become links; split/unified diff, "Enable line wrap", "Binary file; changes cannot be shown", "File too large; changes cannot be shown" | `ui-deliverables` |
| User questions | Agent question card beside the chat input: choose options, enter text, skip, submit one answer batch; timeout/abort copy ("ask_user_question timed out before the user answered") | `ui-user-questions` |
| Trajectory | Turn-aware ledger (User/Assistant/Tool/Subtool/compaction), turn+step boundaries, record inspector for tokens, duration, input, output, timing, images, attachment summaries, "Drag to resize. Double-click to reset." | `ui-trajectory` |
| Subagents / workflows | Descendant browsing with running state, token usage, active-turn duration; completed one-shot conversations open read-only; workflow runs render as Chat nodes with phases→members | `ui-subagent`, `ui-workflow-run` (both STRIP for Strata) |

---

## 5. Settings model

**Host side (`dsh-settings`).** "Edit fields that plugins declare with `.settings`…" Forms expose **only volatile fields from active, uniquely addressed profile entries**; ordinary configuration stays editable through Cordis configuration files. Composition:

```yaml
- id: settings
  name: '@deepseek-ai/dsh-settings'
```

On first start after the Loader settles, a legacy harness-home `settings.yaml` is imported once — each section is written into the entry of the same id (`ui-developer-tools` → `ui-settings`, `ui-onboarding` → `ui-settings-general`, `shell` → the platform shell executor entry) — then renamed to `settings.yaml.imported`. **Reset restores the value beneath the profile override (including schema defaults); home patches and command-line overlays take precedence and a form write they would override is refused.** Each form reports `autoGenerate` (default on) for schema-driven pages — "no shipped client does so yet". A plugin with its own page registers `configure({ auto: false }, ctx.fiber)` inside an optional `ctx.inject(['settings'], …)` child from `apply`.

**Schema & persistence.** Declared by plugins with `schemastery` (`z.object({...})`) — e.g. `dsh-web-app` config `{openBrowser, printUrl, surfaceContext, trustedHosts}`, `dsh-host-webserver` `{host, port, compression}`, `dsh-llm-pi-ai` `{providers: {...}}`. Values live in the composed Cordis tree: bundle patch → profile `cordis.patch.yml` → **`$DSH_HOME/cordis.patch.yml`** → `--patch` overlays. Credentials never enter the settings document — they go through `dsh-credentials` and appear to the UI only as write-only descriptions ("Stored outside the settings file. Leave blank to keep the current key.").

**Client side (`dsh-client-ui-settings`).** One `settings.describe` reader per browser; feature adapters get a per-entry handle:

- `ctx.configForms.get(entryId)` → snapshot `{ value, base, user, revision, writable, persistence }` and one write queue shared by every editor of that entry.
- `set` / `unset` submit one operation; `mutate` submits **one atomic operation list**; staged editors pass the revision they read, and conflicts preserve drafts.
- `ctx.configForms.whileServed(namespaces, register)` registers a page only while the host actually serves one of its namespaces ("A deployment that never composed the owner therefore shows no trace of the page").
- Validation is two-stage: browser validates against the serialized Config schema, the host validates the complete configuration including non-serializable checks.

**Pane contributions (slots).** `settings.section` (a page), `settings.plugins.tab` (a tab inside Built-in plugins), `settings.onboarding` (one mounted step at a time), `settings.general.item` (a row inside General), `settings.links.*`, `settings.header`, `settings.launcher`, plus named surfaces `settings.models`, `settings.account`, `settings.plugins`, `settings.pluginInventory`, `settings.shell`, `settings.agentLoop`, `settings.subagent`, `settings.webSearch`, `settings.sessionLog`, `settings.theme`, `settings.permission`, `settings.transcript.*`, `settings.performance.*`.

**Namespaces seen in the wild:** `ui-theme` (theme + content font size, persisted by the local provider into `$DSH_HOME/cordis.patch.yml`; remote Web keeps it browser-local), `ui-settings` (`ui-settings.enabled` Coding Tools switch), `subagent`, `subagent-model-selection`, `shell`, plus keybindings which are **not** settings at all (they live in localStorage / `userData/keybindings.json`). There is **no versioned settings-migration system** for the settings document itself (only session-format migrations exist); "not discoverable" for a settings schema version field.

---

## 6. Design language

All values quoted from `dsh-client-ui-theme/lib/client.js` (embedded stylesheet bundle: `base.css`, `design-platform.css`, `corner-shape.css`, `focus.css`, `onboarding.css`, `scrollbar.css`, `gradient-shadow-text.css`, `shiki.css`) and the CSS-module files under `dsh-client-ui-primitives/lib/`.

**Token layering.** Three tiers: `--dsw-static-*` (raw palette) → `--dsw-alias-*` (semantic) → `--dsw-specific-*`, plus `--dsh-*` (app/layout), `--dsl-*` (markdown/terminal blocks), `--ds-*` (motion/fonts). 483 distinct `--dsw/--dsh/--ds/--dsl/--shiki` tokens are defined in the bundle.

**Palette (light / dark are two blocks: `body{…}` and `body[data-ds-dark-theme]{…}`).** Neutrals are "neutral-bluish": `--dsw-static-neutral-bluish-00:#fff`, `-50:#f9fafb`, `-75:#f1f3f5`, `-100:#ebeef2`, `-150:#e9ecf2`, `-200:#e1e5ee`, `-300:#cfd3d6`, `-400:#adb2b8`, `-500:#979da6`, `-600:#81858c`, `-700:#61666b`, `-750:#43454a`, `-800:#353638`, `-850:#2c2c2e`, `-875:#232324`, `-900:#1b1b1c`, `-950:#151517`, `-1000:#0f1115`. Brand/accent: `--dsw-static-deepseek-500:#4176e6` (light) / `#4176e6` also `--dsw-alias-brand-primary-new-colorprimary-new-color:#4176e6`; deepseek ramp `-50:#edf3fe`, `-100:#e4edfd`, `-200:#d3e2ff`, `-300:#b7c8fe`, `-400:#7aaaff`, `-450:#5686fe`, `-500:#4176e6`, `-600:#4868b2`, `-800:#34415b`, `-900:#283142`. State: green `-400:#4ed17e` / `-500:#22c55e` / `-900:#233c2c`; amber `-400:#f7ad31` / `-500:#f59e0b`; red `-400:#f25a5a` / `-500:#ef4444` / `-600:#ec1313`; blue `-400:#60a5fa` / `-450:#4d93f8` / `-500:#3b82f6` / `-600:#2563eb` / `-900:#0e3074` / `-950:#172554`.
Surfaces: `--dsw-alias-bg-base` = `neutral-bluish-00` (light) / `neutral-bluish-950` (dark); `bg-layer-1/2/3` = `neutral-bluish-00` (light); masks `--dsw-alias-bg-mask-1/2/3` = `#0000003d` / `#0000001f` / `#0000007a` (drop mask `#ffffffb3`); skeleton `#0000000a`. Text: `--dsw-alias-label-primary` = `neutral-bluish-1000` / (dark) inverted; `-secondary` = `-700`; `-tertiary` = `-600`; `-caption` = `-400`; `-dimmed` = `-200`; `-primary-foreground` = `-00`. Borders: `--dsw-alias-border-l1:#0000000a`, `-l2:#0000001a`, `-l2-darkmode-thin:#0000001a`, `-l3:#0000001f`, `-l4:#00000029`. Sidebar: `--dsw-specific-sidebar-fill: var(--dsw-static-neutral-bluish-50)`, nav hover `-75`, active `-100`, active accent `deepseek-100`. Tooltip `--dsw-alias-tooltip-bg: neutral-bluish-850`; toast `-800` with `-00` label; menu surface `#f8f9fa94`; hover card `#2C2C2E`. Diff: added `#e6f4e7` / gutter `#edf7ed` / marker `#01a241`; deleted `#fce6e2` / `#fdece9` / `#ba2723`; `--dsw-alias-code-diff-added: green-500-a08 (#22c55e14)`, deleted `red-600-a08 (#ec131314)`. Shiki token colors are inlined (e.g. `--shiki-token-string:#8ce99a`, `--shiki-token-punctuation:#ced4da`, `--shiki-token-link:#74c0fc`).

**Theme mechanism.** `html { color-scheme }` + `body[data-ds-dark-theme]` attribute; theme ids `light` / `dark` / `system` (`THEME_PREFERENCES = ["light","dark","system"]`); third-party themes register alias-layer overrides. Content font size is a variable: `--dsh-content-font-size` (setting range **10–22 px**, `Only affects conversation content`), with `--dsh-content-font-delta: calc(var(--dsh-content-font-size,14px) - 14px)` driving markdown line-height and secondary size.

**Typography.**
- UI: `--dsw-font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Helvetica, Arial, sans-serif`
- Brand: `--dsw-font-family-brand: "Montserrat", var(--dsw-font-family)` (with `brand-font.css` in `ui-theme/lib/styles/`)
- Code: `--ds-font-family-code: "SF Mono", "JetBrains Mono", "Fira Code", Consolas, "Liberation Mono", Menlo, Courier, "PingFang SC", "Microsoft YaHei"`
- Composite steps: `--dsw-font-base-16: 16px/24px var(--dsw-font-family)` (weight 400), `--dsw-font-base-strong-16: 500 16px/24px`, `--dsw-font-l-20: 500 20px/28px`, `--dsw-font-m-18: 500 16px/28px`, `--dsw-font-markdown-base: 400 var(--dsh-content-font-size,14px)/calc(24px + var(--dsh-content-font-delta))`, markdown strong 600, italic variant, `--dsw-font-markdown-code-block: 11px/19px var(--ds-font-family-code)`; block banner `11px/18px`.
- Component sizes: buttons `font-size: 14px; line-height: 22px; padding: 0 14px; height: 36px` (`.md`) and `12px/18px; padding: 0 10px; height: 28px; radius sm` (`.sm`); sidebar `--dsh-sidebar-inline-padding: 12px`.

**Radii / shape.** `--dsw-radius-xs:4px`, `-sm:8px`, `-md:12px`, `-lg:16px`, `-xl:20px`, `-panel:28px`. Windows content corner `--dsh-windows-content-radius:16px` ("only the content area's top-left corner has a 16px radius; the other corners and the internal divider remain square"). Progressive enhancement: `@supports (corner-shape:superellipse(1.5)){ :root{--dsw-corner-shape:superellipse(1.5)} *,:before,:after{corner-shape:var(--dsw-corner-shape)} }`.

**Elevation / shadows.** `--dsw-shadow-lv1: 0 2px 4px 0 #0000000d`, `-lv1-blur: 0 4px 12px 0 #00000005`, `-lv2: 0 4px 12px 0 #00000005, 0 2px 8px 0 #0000000a`, `-lv3: 0 0 1px 0 #0003, 0 0 4px 0 #00000005, 0 12px 32px 0 #00000014`; composed elevations `--dsw-elevation-stroke: 0 0 0 .5px var(--dsw-elevation-stroke-color)`, `-soft: stroke, 0 4px 16px 0 #00000008, 0 0 24px 0 #00000008`, `-panel: stroke, 0 3px 8px 0 #00000008, 0 0 16px 0 #00000005`, `-prominent: stroke, 0 3px 8px 0 #0000000a, 0 0 20px 0 #0000000d`. Borders are hairlines: `0.5px solid var(--dsw-alias-border-l3)` on outlined buttons; `--dsw-alias-border-l2-darkmode-thin` exists precisely so dark mode can thin a stroke.

**Motion.** `--ds-ease-in-out: cubic-bezier(.4, 0, .2, 1)`; durations `--ds-transition-duration: .2s`, `-fast: .1s`, `-slow: .3s`. Reconnection loader dots advance every **500 ms** and an attempt stays visible ≥ **800 ms**; the Connected pill fades out over **150 ms**; "Dragging has no transition delay"; discrete open/close sets `data-animating` so the track and the occupant's slide share one duration/easing; fullscreen presentation suppresses transitions; reduced-motion disables sidebar transitions.

**Layout metrics.**
| Metric | Value | Source |
|---|---|---|
| Sidebar width | clamp **264–420 px**, default **280 px**, rail **56 px**, auto-collapse below **1024 px** | `ui-layout/lib/client.js` (`clampWidth(sidebar, 264, 420)`, `sidebar: 280`, `SIDEBAR_AUTO_COLLAPSE = 1024`) |
| Right panel | first open **45 %** of frame (`RIGHTBAR_DEFAULT_RATIO = .45`), capped **70 %** (`RIGHTBAR_MAX_RATIO = .7`), floor 300 px, centre protected at 400 px | `ui-layout` |
| Frame clearances | `--dsh-frame-top-clearance: 48px`, `--dsh-frame-overlay-top: calc(var(--dsh-frame-top-clearance) + 20px)`, `--dsh-frame-leading-clearance: 160px`, `--dsh-frame-chrome-top: var(--dsh-windows-titlebar-height)` | `ui-layout` |
| Chat content | `--dsh-chat-content-width: min(calc(100% - 32px), 920px)`, user column `clamp(680px, …)`; other literal caps found in the client bundles: `max-width: 960px`, `843px`, `760px`, `720px` | `ui-conversation`, `ui-chat` |
| Composer | `--dsh-composer-card-max-width: calc(var(--dsh-chat-content-width) + 32px)` (≈952 px), `--dsh-composer-text-max-height: 336px`, `--dsh-composer-dock-inset: 8px`, `--dsh-composer-side-clearance: 16px` (8 px narrow), `--dsh-composer-stack-gap: 6px` | `ui-conversation` |
| Scrollbars | `--dsh-scrollbar-width: 5px \| 6px \| 9px`, session list `--dsh-session-list-scrollbar-width: 5px` | primitives / `ui-workspace` |
| Modal | full-viewport layer `z-index: 1000`, padding `max(24px, var(--dsh-frame-overlay-top,24px)) 24px`, dialog `min(380px,100%)`, gap 20px | `Modal.module.css` |
| Shortcut dialog | 480 × 600 CSS px; rows 42 px high; settings-control grey key blocks | `ui-shortcuts` README |
| z-index ladder (observed values) | 0,1,2,5,10,11,15,100,1000,1004,1100 | grep over all CSS/JS |
| macOS chrome | 52 px top strip in the expanded sidebar; caption icon buttons = 16 px glyphs in 28 px circular controls; `--dsh-windows-menu-start: 84px` collapsed / 48 px expanded | `ui-sidebar` README |
| Focus | `focus.css` (298 B) defines the ring; automatic entry/return focus deliberately omits outlines while Tab/directional navigation keeps visible indicators | `ui-theme`, `ui-shortcuts` |

**Icons / assets.** No icon-font or third-party icon package: hand-written inline SVGs, named exports like `IconPanelLeftOutlineRegular`, `IconNewChatOutlineRegular` (`ui-sidebar/lib/client.js`). Product mark assets: a fish mark slot + wordmark; `ui-brand-official` swaps in the DeepSeek Harness mark for an `official` build; the conversation hero always uses the animated fish. Frontend ships `favicon.svg` / `favicon-dark.svg`, `manifest.webmanifest`, and KaTeX woff/woff2/ttf subsets.

---

## 7. Strip list vs keep list

| Capability | Verdict | Why / replacement |
|---|---|---|
| Profile/bundle boot model (`dsh.profile.bundles` + patch layers, `$DSH_HOME/cordis.patch.yml`) | **KEEP** | It is the extension spine; Strata needs one `strata` profile, not a cloud account. Simplify to 2–3 bundles. |
| Slot/factory client composition (`ui-slots` + `ui-renderer`) | **KEEP** | Lets Models / Engine / Serving / Monitor be separate panes without touching the shell. This is the single most reusable idea in the codebase. |
| `dsh.client` + `/plugins` + `__DSH_BOOT__` boot graph | **KEEP** | Free plugin discovery and lazy bundles; Strata can serve one prebuilt graph. |
| Web-first host (`host-webserver` + `host-frontend-static` + `web-app` flags) | **KEEP** | Exactly Strata's deployment shape; keep `--host/--port/--no-open` and the loopback default, add an explicit LAN opt-in later. |
| Electron desktop shell | **DEFER** | Large surface (updater, welcome, mandatory-update, platform login, tray). Strata is web-first; ship `strata serve` and let the browser be the app. |
| Sign-in / accounts / DeepSeek cloud API | **STRIP** | Remove `dsh-llm-deepseek*`, `dsh-api-account-controller`, `ui-settings-account`, `dsh-deepseek-account`. Replace with the local engine as the only provider. |
| Session-log upload + OTel telemetry + product analytics | **STRIP** | Remove `dsh-session-log-deepseek`, `dsh-session-telemetry*`, `dsh-host-product-telemetry-otel`, `dsh-client-product-analytics`, `ui-settings-session-log`. Strata's Monitor dashboard replaces the *visibility* need locally. |
| Subagents / workflows / agent teams / schedules | **STRIP** | Remove `dsh-subagent*`, `ui-subagent`, `ui-workflow-run`, `dsh-experimental-agent-team*`, `dsh-schedule`, `ui-schedule`. A single local model cannot serve parallel agents usefully. |
| Provider abstraction `ctx.llm` + chunk protocol (`finish`, `BlockAssembler`) | **KEEP** | This is the seam Strata plugs the local engine into; keeping it means the loop, titles and compaction all work unchanged. |
| `dsh-llm-pi-ai` "custom route" config (`api: openai-completions`, `baseURL`, `models[]`) | **KEEP (repurpose)** | It already reaches llama.cpp/MLX-style servers. Strata's Engine/Models tabs should write this config shape. |
| Settings model (`ctx.settings` forms + `configForms` handles + revision-checked writes) | **KEEP** | Schema-driven panes, atomic updates, conflict protection — exactly what Engine tuning needs (beginner/expert). |
| Credentials seam (`dsh-credentials`) | **DEFER** | A local engine needs no API key; keep the seam for remote fallback but ship without a secrets UI. |
| Permission presets (Read Only / Workspace Write / Full access / Auto review) + approvals | **KEEP** | Local models still write files; the Enter/Escape approval takeover is cheap and high-trust. Drop "Auto review" (it is an LLM call). |
| Capability gating (`ui-settings.enabled` Coding Tools switch) | **KEEP** | Free simple/expert mode switch for Strata; extend it to gate Engine-expert controls. |
| Context metering + session stats + Trajectory | **KEEP (adapt)** | Essential for small local context windows; Trajectory's token/timing inspector becomes the Monitor foundation. |
| Compaction (`/compact`, `dsh-compaction-basic`) | **KEEP** | Non-negotiable with 8–32 k local contexts; keep `/compact` and the "Compacting context…" affordances. |
| Artifacts/office skills, `libreoffice-kit`, PTC runtime, MCP, web search/fetch | **DEFER** | Large optional trees; keep the skill *loader* (`dsh-skill-filesystem`, `/skill`) and drop the bundled office/search plugins initially. |
| HMR of client plugins | **DEFER** | Nice for plugin authors; adds a websocket path. Keep the boot graph, enable HMR behind a dev flag. |
| Session persistence (JSONL + zstd frames + format migrations + FTS5 search) | **KEEP (simplify)** | Durable sessions are the product; keep append-only JSONL, one format version, and FTS-less listing at first. |
| Command palette | **ADD (gap)** | DSH has none; Strata should add one — the registry (`ctx.shortcuts` catalog with labels, aliases, keys) already exists to populate it. |
| OpenAI-compatible **serving** (Serving tab + request log) | **ADD (net-new)** | Nothing in the packaged code serves `/v1/*` outbound; only pi-ai *consumes* it. Build the server + log. |
| First-run hardware onboarding wizard | **ADD (net-new)** | The onboarding ledger (`settings.onboarding`, sequential, one step mounted) is the ready-made container. |
| Engine tuning beginner/expert | **ADD** | Render as a `settings.section` page plus a Start-page card; reuse the staged-draft + Save + "Overridden" pattern from `ui-settings-shell`. |
| Monitor dashboard | **ADD** | Reuse `dsh-session-stats` (turns/steps/TTFT/decode), context-occupancy ring, and the Trajectory timing overview as the data layer. |

---

## 8. Local-engine integration points

### Where the provider abstraction lives
`dsh-llm` is the only supported path into a model: *"Any composition that calls a model provider — an agent loop, a session-title generator, a compaction summarizer — streams its requests through this service. Mount it together with at least one provider adapter; the service itself has no configuration and no provider wire code."* Adapters: `dsh-llm-deepseek` (own Messages transport + capabilities), `dsh-llm-deepseek-api-key` / `-account` (auth + model discovery), `dsh-llm-pi-ai` (routes through `@earendil-works/pi-ai` — *"design-verification twin of dsh-llm-deepseek"*), `dsh-llm-retry` (retries at the agent's failed-step extension point), `dsh-deepseek-llm-api-extensions` (provider-specific top-level request fields).

**Interface (from `dsh-llm/README.md`):**
```
for await (const chunk of ctx.llm.stream({
  provider: 'deepseek-official',
  model: 'deepseek-v4-flash',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
})) { /* block-start, text-delta, …, usage, finish */ }
```
- `ctx.llm.listProviders()` → registered routes in registration order; duplicate route registration fails `DUPLICATE_ADAPTER`.
- `ctx.llm.listModels()` is required for GUI selection ("The GUI requires catalog membership for selection and submission"; base returns an empty list).
- Resolution: per-model **context window, output default, reasoning efforts, input modalities, system-prompt update mode** (`'in-history'` when any-position `system` message is effective); `normalizeModelInfo` rejects anything else with `INVALID_MODEL_INFO`.
- Reasoning effort is validated against the exact model **before any provider I/O**; an adapter output cap is materialized when the request omits one.
- **Every stream ends in exactly one terminal `finish` chunk:** `{kind:'error', failure}` or `{kind:'aborted', failure}`. Stable failure codes: `NO_ADAPTER`, `MISSING_CREDENTIAL`, `INVALID_CREDENTIAL`, `AUTH`, `RATE_LIMIT`, `CONTEXT_WINDOW_EXCEEDED`, `QUOTA`, `ACCOUNT_QUOTA`, `DISCOVERY_FAILED`, `IMAGE_OFFLOAD_REQUIRED`. "Consumers route on the code, never on message text."
- Compaction of the stream: `BlockAssembler` / `AssistantStreamAccumulator`, timestamps + token boundaries preserved in one durable attempt settlement.

### Wire protocol actually spoken
`dsh-llm-pi-ai` → `@earendil-works/pi-ai` → **OpenAI Chat Completions via the `openai` SDK client**:
`dsh/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js:557`
```js
return new OpenAI({ apiKey, baseURL: model.baseUrl, dangerouslyAllowBrowser: true, defaultHeaders: headers … })
…
const { data: openaiStream, response } = await retryProviderRequest(
  () => client.chat.completions.create(params, requestOptions).withResponse(), …)
```
Request shaping observed in the same file: `params.stream_options = { include_usage: true }` (line 582), `params.max_completion_tokens = options.maxTokens` (592), reasoning/thinking compatibility switches `thinkingFormat` ∈ `zai | qwen | qwen-chat-template | chat-template | baseten | deepseek | openrouter | ant-ling | together | string-thinking` (619–712), instruction role `model.reasoning && compat.supportsDeveloperRole ? "developer" : "system"` (896).

Stream parsing (same file): `choice.delta.tool_calls[]` → `{ id, function.name, function.arguments }` accumulated with `partialArgs` and re-parsed per delta via `parseStreamingJson` (425–455); reasoning read from `["reasoning_content","reasoning","reasoning_text"]`, first non-empty wins — **the comment names llama.cpp explicitly**: *"Some endpoints return reasoning in reasoning_content (llama.cpp), or reasoning (other openai compatible endpoints)"* (395–399). Usage mapping (1167–1183): reads `usage.cached_tokens` (Kimi-style top-level) as well as the OpenAI nested form, and `completion_tokens_details.reasoning_tokens`; "OpenAI completion_tokens already includes reasoning_tokens" — i.e. it does not double-count.

Model discovery (`dsh-llm-pi-ai/lib/index.js:2130–2180`): OpenAI protocols list at **`GET {baseURL}/models`** with bearer auth; Anthropic Messages at `{root}/v1/models` with `x-api-key` + `anthropic-version`. Response capped at **4 MiB**; reads `data[]` and "the enriched `models` map some compatible gateways expose"; any other protocol reports it cannot be interrogated so the UI falls back to hand-entry.

### Tool-call format
Tools are declared with JSON-schema-style `{ name, description, parameters }` (pi-ai passes an OpenAI `tools` array; a comment notes `tools` must be present "when messages include tool_calls or tool role messages" — line 22, needed by LiteLLM/Anthropic proxies). Calls come back as streaming `tool_calls` deltas (id → name → partial arguments). Results are returned as `role: "tool"` messages correlated by `tool_call_id`, with assistant replay metadata kept on the durable messages. DSH-side, `dsh-system-prompt` supplies "one ordered system prompt and the available tool schemas for each model step"; workspace guidance comes from `dsh-agent-instructions` (user-global + project `AGENTS.md`); history pressure is handled by `dsh-compaction` (`/compact`) plus `dsh-compaction-image-offload`.

### Model-selection flow
1. Adapters register routes and advertise models; configured-provider adapters declare a settings namespace so "configuration surfaces can activate dormant providers and edit connection facts without a restart".
2. `dsh-agent-default-model` answers exactly one question for fresh agents — `currentSelection()` → `{provider, model, reasoningEffort?}`; `saveSelection()` persists it via the config editor (a no-op without one). Config shape: `{ provider: required, model: required, reasoningEffort?: optional }`.
3. Selection UI: `/model` popup and the composer's model control both show "the same provider-grouped choices, and the selected model determines the available effort names and default" (`dsh-client-ui-model-selection`); page copy "Select the model for this conversation", "Model and reasoning effort", "This model provides no reasoning effort levels."
4. The chosen selection enters the session and each request names `provider` + `model` + optional `reasoningEffort`; capability resolution (context window, output cap, modalities, reasoning efforts) happens in `ctx.llm` before I/O.
5. `dsh-client-ui-settings-models` is where a human declares a route by hand: "Add model provider", "Custom model API", "API protocol", "Base URL", "API key", "Display name", "Context window", "Add model".

### Checklist: what an OpenAI-compatible local server must satisfy for Strata
| Requirement | Detail | Why (from DSH code) |
|---|---|---|
| `POST {baseURL}/chat/completions` | OpenAI Chat Completions schema; `model`, `messages[]`, `tools[]`, `max_tokens`/`max_completion_tokens`, `stream: true` | `openai-completions.js:557` uses the `openai` SDK with `baseURL` |
| `GET {baseURL}/models` | `{"data":[{"id":…,"name"?,"context_window"?,"max_tokens"?}]}`; extra `models` map tolerated; ≤4 MiB | pi-ai discovery `LISTABLE_PROTOCOLS` |
| SSE streaming | `text/event-stream`, one JSON object per `data:` line, terminating `data: [DONE]` | pi-ai reads `openaiStream` async-iterable |
| `stream_options.include_usage` | Must honor it and emit a final chunk carrying `usage` | `params.stream_options = { include_usage: true }` |
| Usage fields | `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`, optional `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens`; `completion_tokens` **must** already include reasoning tokens | usage mapping 1167–1183 |
| Tool-call deltas | `choices[0].delta.tool_calls[]` with `index`, `id`, `type:"function"`, `function.name`, `function.arguments` fragments (concatenated, JSON parsed incrementally) | `ensureToolCallBlock` + `partialArgs` |
| Reasoning channel | Emit thinking text in `reasoning_content` (llama.cpp convention) — `reasoning` and `reasoning_text` also accepted, first non-empty wins | 395–399 |
| Tool-call correlation | Accept `role:"tool"` messages with `tool_call_id`, and tolerate `tools` being present even when only tool history exists | line 22 comment |
| Error shape | HTTP error + JSON body the adapter can classify into `AUTH` / `RATE_LIMIT` / `CONTEXT_WINDOW_EXCEEDED` / `QUOTA`; never rely on message text | failure-code contract |
| Capability metadata | Local model must know its own context window; otherwise `defaultContextWindow: 262,144` / `defaultMaxTokens: 32,768` fallbacks apply, which are far too large for local models — Strata must set these per model | `dsh-llm-pi-ai` config table |
| Shims likely needed | (a) `reasoning` vs `reasoning_content` choice, (b) effort mapping into chat-template thinking flags, (c) prompt-cache fields (`openai-prompt-cache.js`) are OpenAI-specific, (d) `image` content parts must be plain base64 data URLs, (e) `systemPromptUpdate: 'in-history'` only if the engine honors mid-history system messages | adapter compat switches |
| Config entry Strata should write | `providers.local: { displayName: "Local", api: "openai-completions", baseURL: "http://127.0.0.1:8080/v1", models: [{ id, name, contextWindow, maxTokens, reasoningEfforts }] }` | `dsh-llm-pi-ai` provider dictionary |

**Not discoverable in the packaged code:** an outbound OpenAI-compatible *serving* endpoint (nothing registers `/v1/chat/completions` as a route; `dsh-host-webserver` consumers are file/UI routes only), any GPU/hardware detection, any engine-process supervision (llama.cpp/MLX lifecycle), and any request-log store. All four are net-new Strata surfaces.

---

### Appendix — evidence shortcuts
- Host/boot: `dsh/README.md`, `dsh-base/README.md`, `dsh-web-app/{README.md,lib/startup.js,lib/index.js}`, `dsh-host-webserver/README.md` + `lib/index.js`, `dsh-host-frontend-static/README.md`, `dsh-client-modules/README.md` + `lib/index.js`, `dsh-desktop-host/{package.json,lib/cli.js}`, `lib/main.js:3656-3730`, `dsh-web-frontend/{package.json,dist/index.html,dist/assets/index-5SrrfWpU.js}`.
- IA/strings: per-package `lib/client.js` literals (e.g. `ui-workspace`, `ui-settings-*`, `ui-shortcuts`, `ui-chat`, `ui-conversation`, `ui-trajectory`, `ui-deliverables`), plus the slot-key census (231 keys) gathered across `dsh-client-*/lib/client.js`.
- Design tokens: `dsh-client-ui-theme/lib/client.js` (embedded `design-platform.css` + friends), `dsh-client-ui-primitives/lib/*.module.css`, `dsh-web-frontend/dist/assets/*.css`.
- Engine: `dsh-llm/README.md`, `dsh-llm-pi-ai/{README.md,lib/index.js}`, `dsh-agent-default-model/README.md`, `dsh-settings/README.md`, `@earendil-works/pi-ai/dist/api/openai-completions.js`.
