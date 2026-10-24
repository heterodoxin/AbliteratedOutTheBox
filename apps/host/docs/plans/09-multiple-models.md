# Multiple loaded models — UI and integration handover

Implemented 14 October 2026 in the design app at `backups/strata`.

## User flow

Updated 14 October: **one model at a time is the default**.

1. Normal mode shows the model/version, optional image model (Off by default), conversation context, computer use, and Tune model / Load model. Device assignment, instance lists and combined reservations are hidden.
2. **Tune model** applies the recommended configuration to the draft without changing loaded models. It remains an optional heuristic setup action, not a speed benchmark.
3. **Load model** switches to the chosen model automatically, using automatic GPU placement. It becomes the chat default and the only loaded text-model instance. Users do not have to unload the previous model manually.
4. **Expert mode > Model execution** exposes GPU assignment and **Run multiple models**, off by default. Enabling this switch explicitly keeps siblings loaded when loading another version or device assignment. Without the switch, Expert loads still keep one model at a time.
5. A same-version/device load applies changed settings to that instance while preserving its API identity; an identical configuration is idempotent. An ordinary single-mode load removes other instances even when the selected instance is already loaded.
6. Leaving Expert mode restores single-model loading and automatic placement. Opening/closing settings or toggling the switch does not itself unload running work; the next successful single-mode Load performs the switch. Request guards block switching while affected work is active/queued. Validation, availability and memory checks precede all simulation lifecycle mutations, so failed switches preserve the current model.
7. Monitor and Serving show their simple single-model presentations with one model loaded. Instance selection, model lists and routing filters appear when multiple instances have actually been loaded through the advanced flow.

The native host explicitly reports its current single-engine and automatic-placement limitation. Multiple simulated instances are usable in the design host; real multi-process GPU inference is not implemented by this UI change. The “Design preview” label was removed from the Loaded models heading; simulation disclosures in generated responses and telemetry sources remain.

## The five annotations

| Page | Change |
| --- | --- |
| Monitor | All-model overview and individual model selector; loaded-instance rows identify devices; requests show the actual API model ID. Per-model context, throughput and request history are scoped separately. GPU utilization, memory, temperature, CPU and disk readings belong to devices/the host. Selecting a model selects its assigned GPUs by default. |
| Serving | One gateway URL; all loaded API model IDs listed with copy buttons and GPU assignments. Clients specify the `model` ID. Request table has model identity, a model filter, and a model-specific connection guide. |
| Settings > About | Removed Design examples, Chat showcase and Preview examples. Update and Privacy remain. |
| Model setup | Detected-device selector, multiple selected GPUs, instance list, scoped unload and chat default, and combined RAM/VRAM planning. |
| Model actions | Separate Tune model and Load model actions. Tuning does not load or restart an engine. |

## API contract

- `GET /api/runtime`: mode, capabilities, revision, default instance ID, instance list and estimated memory reservations.
- `POST /api/runtime/plan`: `{ variantId, gpuIds, config, executionMode? }`; returns candidate memory, combined budget, `allowed` and `alreadyLoaded`. No lifecycle mutation.
- `POST /api/runtime/load`: same body; omitted `executionMode` means `single`, and only explicit `multiple` preserves sibling instances; validates the engine configuration, requires an installed version and returns `{ state, instance }`. Native multi-instance calls are rejected.
- `POST /api/runtime/:instanceId/unload`: refuses active/queued requests for that instance, removes its queue and releases its reservation; other instances remain.
- `POST /api/runtime/:instanceId/default`: chooses the default for Strata chat. Explicit API requests retain their selected model.
- SSE `runtime`: immutable runtime state after model/queue/phase/default changes.
- `GET /v1/models`: all loaded instance API IDs. Instance ID, GPU IDs, context and simulation mode appear in metadata.
- `POST /v1/chat/completions` and `/v1/messages`: route to the exact requested API model ID before entering the instance queue. Unknown IDs and omitted IDs when multiple instances exist return an error before streaming starts. With one loaded instance, an omitted ID remains compatible.

The first instance of a version uses its version ID. Additional instances of that version use suffixes such as `--2`. These runtime identities last for the host session. In single mode the loaded target becomes the chat default. The legacy `/api/models/:id/load` path also uses single-mode switching, so normal chat/model-picker loads cannot accidentally add instances. The legacy `library.loaded` field is the chat default only; it does not enumerate every loaded instance. Library variant states are projected from the instance registry. Delete requires unloading every instance of the selected version.

The simulated host owns a separate request queue and prompt/context state for each instance. A request on one instance can run while a different instance is active. Requests on the same instance remain serialized and cancellable. Existing request metadata carries `instanceId`, `model` and `gpuIds`.

## Memory and telemetry

System RAM is a single pool shared by all instances. Unified GPU memory uses that same pool and is not added to physical RAM. Dedicated VRAM is budgeted separately for each GPU; spare capacity on an unselected GPU cannot satisfy another GPU's reservation.

Planning includes weights, context/KV cache, parallel request slots, runtime overhead and one host headroom reserve. The largest requested headroom is used once for the host. Pageable weights can share remaining RAM and spill to SSD in the design planner. Fixed context/runtime allocations cannot spill. Non-pageable/locked weights cannot use that escape path. An impossible plan disables Load and preserves existing instances.

Reservations are estimates, not observed allocations. Equal splitting across selected dedicated GPUs is a planning assumption. The current metadata, especially hybrid/MoE context storage, is approximate. The design planner supports automatic/all-layer or CPU placement; explicit partial-layer placement needs a verified engine allocation model and must not be treated as a guaranteed fit. Monitor host readings retain their measured/illustrative/unavailable sources. All-model context fill does not add independent context windows; users select an instance for context detail. The speed card refers to the last completed request when idle; live simulated throughput is left unavailable until a result is reported. Per-instance speed histories and caches are left unavailable instead of copying another model's telemetry.

## Native engine integration still required

1. Refactor `supervisor.mjs` into a factory keyed by instance ID. Each process owns a distinct private port, status, cancellation, logs, metrics and failure cleanup. Validate adapter-specific device visibility and tensor/layer split flags using stable hardware IDs. CPU-only placement must explicitly disable GPU offload.
2. Refactor the inference bridge to dispatch to the selected instance's supervisor URL and immutable configuration. Eliminate use of a single global backend/configuration for routing. Keep the public gateway port, API key, local sharing and tunnel settings global.
3. Allocate/reserve memory atomically during load, verify free dedicated VRAM and shared system memory, and reconcile reservations with observed allocation. Model metadata is a preview until the adapter verifies weights, KV/image workspaces, scratch buffers and placement. Account for other applications; do not present capacity arithmetic as measured free memory.
4. Separate per-instance monitor state, prompt reuse, KV/conversation cache and device telemetry. Do not attribute a whole GPU's utilization or shared RAM reading exclusively to one model using it. Retain device IDs and request-to-instance identity in snapshots and logs.
5. Isolate settings changes and lifecycle operations. Loading/unloading/restarting one engine must preserve healthy siblings. Keep targeted request guards and queued-dispatch validation. Clean up ports, processes and reservations on startup failure/exit.
6. Decide explicit persistence/autostart policy. Current extra design instances are session-only. Migrate legacy singleton state to one default instance without promising other instances survived a host restart.
7. Enable `capabilities.multipleModels` and `deviceAssignment` only after the native adapter supports these paths. Preserve the honest fallback for single-engine installations.

The optional image-model asset library remains separate. Real Qwen-Image generation, companion encoder/VAE/adapters, role-aware routing and combined image working-memory reservations are still required. This change does not make downloaded image GGUF assets a real inference engine or list them as text API models.

## Files and checks

Main implementation: `server/runtime-models.mjs`, runtime routes in `server/index.mjs`, request metadata in `server/serving.mjs`, `src/components/RunningModels.tsx`, `src/lib/runtime-models.ts`, ModelRun/Monitor/Serving/Settings, and shared model-run CSS.

Checks: TypeScript and production build; runtime service tests for two GPUs, idempotence, routing, per-instance queues, targeted unload, invalid devices, shared RAM and native capability gates; Monitor scope isolation; disposable-host HTTP tests for both protocols, simultaneous independent requests, model listing, invalid/ambiguous routing and targeted lifecycle.
