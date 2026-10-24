# Native Strata UI integration handover

Implemented 14 October 2026 in the React/Node app under `backups/strata`.

## Consumer flow

The default is one model at a time. First launch uses the visual `GuidedSetup` funnel: **Welcome → Your model → Prepare → Make it yours → Chat**. Welcome has animated vector artwork and one expert skip at the bottom. Model selection offers friendly version cards, context, RAM/VRAM inventory with an aggregate total, and an existing-folder browser. Unified CPU/GPU memory counts once. Memory verdicts are guidance: **May run slower** does not block selection or Continue. There is no compatibility alternatives panel or picture-reading question at this stage. The preparation screen confirms the download size before **Download model**, then shows engine-reported progress alongside a clickable Chat / Files / Your setup tour. Vision-capable families always prepare their mmproj/encoder files. Preparation does not load the model automatically.

After preparation, users choose **Leave 20% for my work** or **Full performance**, and choose whether to **Read pictures when this model loads** (off initially). The optional longer native calibration measures engine settings for approximately 5–10 minutes. **Tune & start chatting** applies the resource and vision preferences, loads one model, performs a short benchmark, and completes setup only when that model is healthy. Chat opens automatically with starter prompts for a new user; selecting a prompt fills the composer without sending it. Progress, choices and operation IDs survive reloads. Finish later preserves the current stage and keeps completed files available.

Normal Models shows Version, optional picture reading, context, Computer use and Load model. The resource preference can be changed here after onboarding. Image generation is a separate collapsed option. The selected model and context remain visible in the toolbar; saved setups and Manage models are available there. Local GGUF import is a secondary entry in Version. Imported models retain the existing metadata/shard validation and reference their files without copying them.

### Guided setup integration

- `choices.onboarding` stores the step, family/version, context, vision, folder, resource preference, calibration choice and operation IDs through the existing setup API. It uses four persisted setup steps; chat is the exit.
- `POST /api/native/directory` accepts `{directory}`. The controller validates an existing readable/writable folder and persists it for native downloads. Existing installed configurations retain their file locations. The official installer checks and reuses matching native files/shards; custom GGUF import remains the separate local-model path.
- `POST /api/native/operations` with `kind: "install"` and `loadAfter: false` prepares files. The UI sends `vision: "yes"` for vision-capable families, regardless of the future loading choice. `kind: "tune"` accepts `{model, profile: "work" | "performance", calibrate: boolean, vision: "yes" | "no", guided: true}` and orchestrates native calibration, resource settings, loading and a measured benchmark. This operation is cancellable.
- `kind: "load"` accepts `vision: "yes" | "no" | "cpu" | "gpu"`. Source configurations retain the prepared vision section and files; only the instance configuration omits vision when disabled. `ui_vision_enabled` and `ui_vision_device` persist the preference after a healthy load. Installed model metadata exposes `visionEnabled`/`visionDevice`, while `images` continues to describe prepared capabilities. An install with `loadAfter: true` passes the separate `loadVision` preference to loading. Changing vision activation reuses the prepared files.
- Apple and SSD/NVMe spilling are part of the requested handover target. Wire this UI to the newest engine's actual backend and out-of-core preparation/load contract. The development checkout at v0.1.38 still rejects Apple preparation in its official installer; removing UI memory gates does not change that older engine or expand supported model architectures. Physical RAM + dedicated VRAM is an inventory total, not a promise of one interchangeable allocation pool. Actual preparation errors and disk-space checks remain visible.
- Headroom caps CPU expert workers to at most 80% of detected physical cores and reserves at least 20% of the selected GPU's memory. It preserves calibrated lower worker counts and the model/vision safety reserve. This is a worker/memory budget, not a hard utilisation or RAM cap. Performance restores the baseline budget while retaining system safety reserves. Values and a baseline are saved in the prepared configuration with an atomic write and `.bak` backup.
- Native `load` also accepts `resourceProfile` for the later Models control. Resource changes are saved only after the engine becomes ready. Cancelling during a health check unloads the cancelled instance.
- Artwork and the tour use SVG/CSS animation with reduced-motion support; no video renderer or animation dependency is required. Radio choices and tour tabs support arrow-key navigation, and the dialog focuses each stage's title.
- Verification: 157 Node tests pass, TypeScript/build pass, and browser checks cover expert skip, folder selection, keyboard choices, reload, stop/resume, tuning and automatic chat handoff. Screenshots in `docs/evidence/guided-onboarding-2026-10-14` use a disposable supported-hardware fixture. No actual weights, calibration or native update ran on the Mac host.

## Expert flow

1. Open Models → Expert mode → Model execution.
2. Turn on **Run multiple models**.
3. Choose a native family/version, select **Run on → GPU 1**, and Load model.
4. Choose another version/family, select **GPU 2**, and Load model. The first instance stays loaded.
5. Use **Split across GPUs…** and layer split points when one model needs several cards.
6. Use in chat changes the default chat instance. An identical load is idempotent; changed settings reload that placement while preserving its public ID.

The native catalog determines supported architectures. Arbitrary GGUF imports use the separate local engine adapter; importing a compatible-looking file does not grant Strata-native optimization. CPU-only native text inference is not offered by this adapter. CPU vision encoding is available when the installed native version supports it. Qwen-Image generation remains its own optional model flow; the public Strata text engine does not provide an image-generation backend.

Preparing new native files unloads managed native instances because the installer can replace shared engine files. The confirmation explains this. Download/prepare all needed versions before loading them concurrently.

## Connecting the native installation

Set `STRATA_NATIVE_REPO` to the absolute path to the official Strata checkout. It must contain `setup.py`, `tools/strata_mcp.py` and `serve/server.py`. Settings → Engine can connect a different existing installation. The source checkout is separate from this UI app.

The catalog adapter requires Python 3.10+. `STRATA_NATIVE_PYTHON` selects it explicitly; otherwise the installed virtual environment or an available system Python is used. The native installer creates the model's `.venv`. Loading uses that environment, rather than installing packages into a random system interpreter.

For desktop distribution, bundle the tested native source and a supported Python runtime, set these paths from the launcher, and retain the installer’s platform checks. Consumers should receive this installation as part of the app; manual source-folder connection is the fallback. The current relative development fallback locates the sibling `Strata` checkout in this workspace.

The native adapter works independently of `STRATA_REAL_ENGINE`. That flag continues to select real versus simulated behavior for the older GGUF/llama adapter. Set it to `1` for actual imported GGUF inference. The existing preview conversations and design demonstrations are retained. Switching into the native execution group stops an owned real GGUF engine; concurrent placement applies to native instances.

## Backend boundaries

| Module | Responsibility |
| --- | --- |
| `server/native-control.mjs` | Official MCP catalog/status/install bridge, job persistence, native configuration copies, process lifecycle, measured benchmark, calibration and updates. |
| `server/native-settings.mjs` | Allowed settings, bounds, MCP configuration validation and credential masks. |
| `server/native-gateway.mjs` | Transparent Chat Completions, Anthropic and Responses forwarding to the selected native instance. |
| `server/chat-images.mjs` | Attached image validation: PNG/JPEG/WebP, maximum 3 × 2 MiB. |
| `src/lib/native.ts` | Shared native state and polling. |
| `src/components/NativeModels.tsx` | Consumer setup, Expert device assignment, version management and load flow. |
| `src/components/NativeSettings.tsx`, `NativeMcpEditor.tsx` | Per-model parameters, JSON and tool server configuration. |
| `src/components/NativeMonitor.tsx` | All-instance rates, per-instance details and shared host measurements. |

Each instance runs `serve/server.py` on a distinct private loopback port with a random private API key and a configuration copy in the UI data directory. The public API URL remains the single URL shown in Serving. Its existing API key and local-network/tunnel access controls still apply. Private host controls are not exposed through the sharing proxy.

The public ID is added to the private native config as an alias, including suffixes for repeated versions on different GPUs. Clients put that exact ID in `model`. Unknown and ambiguous IDs fail before streaming. Responses input, client tool schemas, tool results, reasoning fields and protocol events are preserved. Public clients cannot opt into the host’s configured MCP servers; only local chat's explicit Tools opt-in sends `strata_mcp: true`.

## Settings and tools

Expert parameters include thinking budget, output fitting, Anthropic thinking, GPU/layer order, VRAM reserve, elastic cache, concurrent requests, idle unload, lazy loading, stall timeout, API monitoring, image tokens and MCP limits. Sampling includes temperature, top-p/k, min-p, penalties and seed. JSON also exposes engine arguments and `STRATA_` environment flags. Fields absent from the installed server source are disabled with an update hint; the server rejects newly configured unsupported fields.

Saving validates first, preserves unrelated config keys and non-Strata environment values, writes a `.bak`, and atomically replaces the config. Existing env/header credentials are masked and preserved when the mask is submitted. Clearing a normal field restores its native default. New settings apply on the next load. Context/KV/vision/low-RAM preparation changes go through the official installer so dependent native flags stay consistent.

MCP servers can be local programs or HTTP connections. Arguments are separate argv entries. Add/Edit/Remove and enabled state are available without using JSON; advanced credentials, working directory and raw configuration remain accessible. Saving a server does not run its tools. In chat, **Enable connected tools** explicitly opts into automatic calls and Bypass permissions; calls/results become visible tool traces. Tools stay off by default.

Native images are stored as browser IndexedDB blobs with message references, then sent as image content to an enabled native vision model. They are separate from artifact previews. Photos are not silently sent to text-only/API models: queued photos require a local vision target; earlier photos become filename notes when switching to a model without that target. At most three recent pictures, including the current attachments, are sent per request so repeated photo messages fit the request limit. Older pictures stay in IndexedDB and can be reattached.

## Monitor and Serving

With multiple instances, Monitor opens All models and shows each model's generating TPS, prompt reading TPS, state and queue. Current rates are labelled Now; idle rates are labelled Last request. Missing measurements show —. Select an instance for its speed history, context and expert cache. GPU readings are device measurements, and host RAM is shown once. Speeds and context are not added across independent models.

Serving shows one base URL and all available model IDs/devices. Requests can be filtered by model. The connection guide has Chat Completions, Responses and Anthropic templates and a model selector. Responses is capability-gated to native versions that implement it. Cancelling a request aborts only its forwarded call.

## Files, failures and recovery

Storage retains the existing configurable model directory and rescan flow. New installations default to the user’s Documents/Strata/Models folder. Windows resolves the known Documents folder (including redirected/OneDrive locations); Linux honors XDG_DOCUMENTS_DIR; macOS uses Documents. Saved custom libraries and legacy libraries with real weights remain selected. STRATA_MODELS_DIR is an explicit override for tests and custom deployments. For the default folder, native preparation uses Documents/Strata as data_dir so its models/<variant> layout lands in Models. On case-sensitive volumes, the host creates a relative models → Models alias; existing conflicting folders are never replaced. The native checkout remains unchanged. Existing native configs retain their original paths; retuning and repair target that model’s original data folder rather than the default folder for new downloads. Completed native downloads and partial SDK downloads are retained for resume. SDK preparation running across a host restart is rediscovered and its progress reattached; interrupted UI operations are recorded with a retry message.

Settings → Engine provides the original engine/preparation logs and export. Managed native process groups are stopped on unload and host shutdown; temporary instance configs are removed. Servers started outside this UI are detected and preserved; loading/updating here asks the user to stop that server or connect its API first.

**Tune model** runs a fixed, greedy, thinking-off benchmark and saves reported engine timings. Stop benchmark aborts its request. NVIDIA calibration invokes the installed native `calibrate_config` for the selected existing model; failures are surfaced rather than accepted as a successful measurement.

About checks release availability inline. Explicit update actions run the official native update path; Strata source updates require a clean checkout and the official origin, and use `git pull --ff-only`. Updates require unloading native instances. Engine rollback is enabled only when the installer supports it and a saved `.previous` engine exists. UI-source rollback is not fabricated. These buttons manage the configured native installation; updating the React app itself belongs to the desktop distribution updater.

## Verification and deployment limits

TypeScript and production build pass. The complete Node suite passed **149 tests**. Native fixtures cover independent devices/ports/keys, single-mode switching, idempotence/reload identity, invalid placement preservation, busy-instance guards, settings/backups/masks, benchmarks and aborts, calibration target, dirty-checkout refusal, three-protocol routing, native errors, disconnect cancellation, vision payloads and MCP traces. Browser evidence is in `docs/evidence/native-feature-parity-2026-10-14`.

The real read-only adapter was verified against the local official checkout at **v0.1.38**. This Apple M4/16 GB machine has no native weights/environment installed; that checkout does not offer a supported GPU here. Multi-GPU screenshots and performance values use disposable fixtures, not measurements from this Mac. Before release, run the actual supported GPU test: install/resume, prepare vision, load two placements, send simultaneous requests through all supported protocols, benchmark/calibrate, update/rollback and test process cleanup on Windows as well as Linux. No native weights, calibration, source update or rollback were run during this UI implementation.


## Guided onboarding annotation pass

The guided flow removes the duplicate Step N header, groups the model and context selectors above a RAM/VRAM inventory, and uses independent interactive version cards. Primary buttons inherit the same visual treatment as New chat. The modal scrolls vertically with a sticky action footer, including while browsing folders.

Preparation is headed “Downloading the model...” and has one model/version/quant identity row. Its gently floating counter reads native download.done_gb / total_gb, showing GB left in the current file, rather than multiplying shard progress by the whole-model size. Unknown progress stays indeterminate. Counter interpolation and CSS motion respect reduced motion. Storage shortages show required and free GB with a folder-change action; raw paths and flags are removed from this notice. A resumed setup refreshes the plan before enabling download.

The four interactive tour topics are Time-skip (labelled Preview), Goal (Coming soon), Loop and Subagents. /loop is now a slash alias for opening Tasks; schedules are chosen there. The Goal tour describes future outcome-driven work and does not implement /goal. Subagents explains the natural-language request to spawn agents and bring parallel results together.

This pass passed 39 focused directory, progress, memory, native control, vision and chat tests, plus TypeScript and production build. Browser checks verified shared memory counted once, two GPUs aggregated, exact New chat/Continue computed styling, shard countdown 13.7 → 6.2 GB, all tour tabs and keyboard navigation, /loop → Tasks, readable real disk-space feedback, and modal scrolling. Download progress in the evidence is simulated; no model weights or real calibration were run.


The final resource-choice screen now uses an animated computer and resource-budget bar rather than the priority ring. Both budget modes update the illustration immediately. Native semantics are unchanged: headroom remains a CPU worker / GPU memory preference, not a live utilisation throttle. Onboarding checkbox marks retain native input semantics and keyboard operation while rendering a consistent black tick in light mode.


The first-chat screen now places a concise welcome above the real message box. Starter task cards are removed. When setup closes and a model is ready, the first-chat input receives focus without changing a saved draft. This layout also applies when no conversation exists yet. It returns to the standard transcript layout once a message is present. Dark-mode evidence for all onboarding stages, tuning and first chat is in `docs/evidence/onboarding-dark-2026-10-14`.
