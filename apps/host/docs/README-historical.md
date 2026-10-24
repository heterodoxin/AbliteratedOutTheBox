# Strata

Local AI, with one model by default and multiple native instances in Expert mode.
The React UI connects to the official Strata installer and engine, checks the
hardware, prepares supported models, serves compatible APIs and shows measured
engine activity. See [native integration handover](docs/native-engine-handover.md).

This is the web-first iteration of the app. It is built to become a standalone
desktop app later — the browser shell and the engine host are already separate
processes talking over HTTP, which is the same seam Electron would need.

```
┌────────────────────────────┐        ┌──────────────────────────────────────┐
│  web shell  (Vite/React)   │  /api  │  engine host  (node, zero deps)      │
│  :5199 in dev              │ ─────▶ │  :8791                               │
│                            │  SSE   │  hardware · models · engine ·        │
│  Chat Monitor Serving      │ ◀───── │  serving · metrics · inference       │
│  Models Tuning Workspace   │        │                                      │
└────────────────────────────┘        └──────────────────────────────────────┘
```

## Run it

```bash
npm install
npm run dev          # host on :8791 + web on :5199, one command
```

Open <http://localhost:5199>. On a clean machine Strata opens the first-run
Welcome flow: hardware check → choose version → optionally read pictures → download/load → chat. Advanced settings remain collapsed.

Single-port mode (what the desktop build will use):

```bash
npm run build
npm run serve        # everything on http://127.0.0.1:8791
```

Other scripts: `npm run typecheck`, `npm run sync:kit` (re-import the design
kit), `npm run shot -- out.png <url>` and `npm run flow -- chat|expert|setup|light`
(scripted screenshots used to verify the UI).

## What is real, and what is not

### Try the Time-skip sidebar

Send `/time-skip Make a small website for my plant shop` in any chat. Any prompt
text works; no model needs to be loaded. This is a **28-second UI demo**, with
simulated checkpoints and checks, rather than a connection to the OMP harness.
It saves A, builds and checks B, chooses B, then rejects a weaker revision.
Click either stream or the saved first draft to inspect its example evidence.
The run continues when hidden, survives reloads, and can be replayed when done.
The sidebar shares the preview tab bar and respects reduced-motion preferences.

### Implementation status

| Area | Status |
| --- | --- |
| Hardware detection | **Real.** `system_profiler` / `sysctl` / `vm_stat` on macOS, `nvidia-smi` / `/proc` on Linux, `os` module fallbacks everywhere. |
| Native model catalog | **Read from the installed Strata source.** Families, real download sizes and hardware compatibility come from its official MCP catalog. |
| Model files | **Native installer or real GGUF adapter.** Native preparation reuses/resumes shards and companion files. The older design adapter remains simulated unless `STRATA_REAL_ENGINE=1`. |
| Engine tuning | **Native config and measured operations.** Validated parameters/JSON, config backups, per-model GPU placement, measured benchmark and supported NVIDIA calibration. |
| Serving API | **Native gateway.** Chat Completions, Anthropic and supported Responses protocols route by exact model ID. Existing key/network controls and request cancellation remain. |
| Inference | **Native managed processes or a selected API/local adapter.** Native vision sends actual image content, and explicitly enabled MCP calls appear in chat. Design demonstrations remain labelled. |
| Monitor metrics | **Reported native readings.** Per-model TPS/prompt TPS, context and cache; shared host RAM and GPU measurements. Missing readings remain unavailable. Simulated mode is labelled. |

Conversations stay in browser storage. Selected API providers receive their
conversation and skill instructions; enabled MCP servers access their configured
resources. Downloads and update checks contact the official upstream services.
No usage telemetry is sent. The UI request log stores timing/token metadata, and
native API monitoring can retain request content in memory when explicitly enabled.

## Layout

```
strata/
├── index.html               single entry, dark by default
├── src/
│   ├── app/                 shell: App, Sidebar, CommandPalette (⌘K)
│   ├── components/          primitives (ui.tsx), charts, screen chrome
│   ├── screens/             Launch · Setup · Chat · Models · Tuning ·
│   │                        Serving · Monitor · Workspace · Skills · Settings
│   ├── lib/                 types, api client, event stream, store, router, format
│   └── styles/
│       ├── kit/             tokens.css + components.css, synced from ../ui
│       └── app.css          app shell and screen styles
├── server/
│   ├── index.mjs            HTTP surface, SSE hub, static hosting
│   ├── hardware.mjs         real probes and the compatibility verdict
│   ├── catalog.mjs          the model ladder + every expert parameter
│   ├── models.mjs           install/load state machine, disk accounting
│   ├── engine.mjs           beginner goals → expert params, predictions, validation
│   ├── monitor.mjs          live phase + throughput + thermals
│   ├── serving.mjs          API keys, access policy, request log
│   ├── inference.mjs        backend detection (llama.cpp/Ollama/LM Studio) + proxy
│   └── store.mjs            JSON state under $STRATA_HOME (~/.strata)
└── scripts/                 dev runner, design-kit sync, screenshot harness
```

State lives in `~/.strata` (`models/`, `data/state.json`). If the home directory
is not writable — sandboxes, locked-down accounts — the host falls back to
`./.strata`. Override with `STRATA_HOME`, `STRATA_WORKSPACE`, `STRATA_PORT`.

## Design

The interface **is** the Strata UI kit in `../ui`, not a lookalike.

- **Light is the default.** `:root` holds the light palette; `data-theme="dark"`
  swaps the same token names. Both themes are laid out from one stylesheet.
- **Kit classes are the components.** Every component the kit defines is in use:
  `.st-header` (screen headers), `.st-card` (55 call sites), `.st-btn`,
  `.st-pill` and `.st-badge` (status lines vs. table cells), `.st-progress`,
  `.st-tabs`, `.st-table`, `.st-metric` (every number tile in the app),
  `.st-input` / `.st-toggle` / `.st-slider` / `.st-field` / `.st-label`,
  `.st-collapse` (model thinking), `.st-code` (code blocks),
  `.st-gauge` (GPU load, context fill), `.st-drawer`, `.st-composer`,
  `.st-msg` / `.st-bubble`, `.st-toast`.
  `src/components/ui.tsx` is a thin binding layer over them; `src/styles/app.css`
  only adds layout plus a short list of documented extensions at the top.
- **No gradients anywhere.** Surfaces are flat; depth comes from the kit's three
  shadows (`--st-sh-card`, `--st-sh-bar`, `--st-sh-pop`).
- **Kit geometry.** Radius 8 / 14 / 20 / 24, 4pt spacing, 14px controls, 20px
  cards, 24px panels, 64px header, 1160px content column.
- **Icons** are the Anron duotone set, rebuilt into one sprite at
  `public/icons/sprite.svg` by `npm run sync:kit`. The script strips the kit's
  C2PA metadata and carries the source `fill="none"` onto each `<symbol>` — without
  it every stroked path inherits the default black fill and the icons render as
  blobs.

What comes from DSH rather than the kit is the *shape of the product*: sidebar
sessions, a status line that reports what the engine is doing, ⌘K palette,
slash commands, per-tool approvals, and a sampling drawer one click from the
composer.

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | host status, detected backend |
| GET | `/api/setup` · POST `/api/setup/step` · POST `/api/setup/complete` | first-run state |
| GET | `/api/hardware?reprobe=1` | probes, checks, verdict, recommendation |
| GET | `/api/models` | library with per-variant state and progress |
| POST | `/api/models/:id/install` · `/cancel` · `/load` | library actions |
| DELETE | `/api/models/:id` · POST `/api/models/unload` | reclaim disk / VRAM |
| GET/PATCH | `/api/engine` | beginner + expert configuration |
| POST | `/api/engine/restart` · `/reset` · `/validate` · GET `/export` | engine lifecycle |
| GET/PATCH | `/api/serving` | access policy, API key, request log |
| GET | `/api/monitor` | live snapshot (also pushed over SSE) |
| GET | `/api/events` | SSE: `monitor`, `models`, `engine`, `serving`, `setup`, `log` |
| POST | `/api/chat/stream` | NDJSON token stream for the chat UI |
| GET | `/v1/models`, POST `/v1/chat/completions`, POST `/v1/messages` | public, OpenAI/Anthropic compatible |

## Next

1. Engine supervisor: detect a `llama-server` binary, spawn it with the tuned
   flags, parse its logs into the Monitor instead of synthesising them.
2. Real downloads: resolve GGUF files from Hugging Face, verify hashes, resume
   interrupted installs.
3. Desktop shell: wrap the built web app and the host in Electron/Tauri using
   the same `{ url }` handshake the host already exposes.
4. Tool use and approvals in chat (read/write inside the workspace, shell with
   an explicit grant), following the permission-preset pattern.
