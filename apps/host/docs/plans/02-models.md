# 02 · Models

**Route** `#/models` · **File** `src/screens/Models.tsx` (297 lines)

The screen that answers "what can this computer actually run?". For the audience
this app targets — people who find hardware intimidating — it carries more weight
than any other configuration screen.

## What it is today

- **Current model card** — name, blurb, a spec row (params · layers · experts ·
  RAM class · disk), a Loaded/Needs-files pill, and Load / Unload / Download.
- **Variants list** — `listrow` per variant: label, Recommended pill, blurb, RAM
  class, action button, remove button. Install progress uses a `Bar` with a stage
  caption.
- **Downloads and disk card** — free space, used-by-models bar, "Tune current model".
- **Icons** — `layers` throughout.

## Where it falls short

1. **"8 GB RAM class" is jargon.** The entire value proposition is *will this run
   on my machine*, and the answer is expressed as a memory class the reader has
   to map onto their own computer themselves.
2. **No fit indicator.** There is a Recommended pill, but no explicit "fits /
   tight / will not fit" per row, which is the only question being asked.
3. ~~Download size is the only cost shown.~~ **Fixed.** Rows now read
   `Download · 10 GB · about 13 min`. The formula is shared with the setup
   wizard rather than duplicated — two screens describing the same download must
   not disagree, and a shared helper is the only way to guarantee that. It is
   deliberately coarse: rounded to whole minutes and floored at two, because a
   precise-looking number from an unknown connection is a false promise. Time-to-download and the disk
   consequence are absent, and "Choose · 10 GB" reads as a price.
4. ~~The recommended variant is marked but not explained.~~ **Fixed.** A badge
   told you *which* one and never *why*. It now says so: *"The largest that fits
   comfortably in this computer's 16 GB with room left over — smaller ones would
   trade quality for nothing."* That is literally the host's rule (pick the
   largest variant whose footprint fits), stated in the reader's terms rather
   than left as something to infer from a RAM class.

   Worth recording: my first check of this claimed the host returned no
   recommendation at all, because I queried the models payload instead of the
   hardware one. The recommendation worked; the missing thing was the reason. Five variants, five equal-weight
   rows. The recommended one is distinguished only by a small pill.
5. **The disk card is a dead end.** It reports usage and links to tuning; it does
   not let you reclaim space, which is the actual motivation for visiting.
6. **No download progress detail** — a percentage and a stage string, no rate, no
   remaining time.
7. **`layers` is used for the model glyph** where the kit now ships a real
   `folder`/`file` vocabulary. The model glyph itself is fine, but the vocabulary
   is inconsistent with the Workspace tree.

## The plan

### Substance — answer the real question
1. **A fit verdict per variant**, computed from detected hardware: **Runs well ·
   Runs, tight · Too large**. Three states, plain words, one glance. This replaces
   the reader having to interpret a RAM class.
2. **Move the RAM class into the detail**, behind the verdict, for people who
   want it.
3. **Show download cost as time and space**: `≈10 GB · about 4 min on this
   connection`, with the disk-after figure (`23 GB → 13 GB free`).
4. **A "why this one" line on the recommended variant**, in one sentence, drawn
   from the same hardware detection that produced the verdict.

### Hierarchy
5. ~~Give the recommended variant visual weight.~~ **Done, restrained.** The
   recommended row takes a 2px accent hairline down its leading edge and a title
   one step larger (18px against 16px). Nothing more — it already explains itself
   in words, so this only has to make it findable, not to shout.

   **The edge had to move to `::after`.** `.listrow::before` is the separator
   hairline, so the first version silently replaced it. Invisible here, because
   the recommended row sits at the top where the separator is hidden anyway —
   and broken the moment a recommendation appeared anywhere else.
6. **Collapse the not-installed variants** behind "Show 3 more options" when the
   machine cannot run them. Presenting unusable options as equals is the main
   source of confusion on this screen.

### Fluidity
7. **Switch the model-change animation from a keyed entrance to a shared-element
   morph**, now that the headline and the row are both named. The earlier attempt
   failed because the swap is server-driven; the fix is to capture the old state
   before the request and hold it through the busy phase.
8. **Load progress reads as a phase sequence**, not a bare bar: `Reading → into
   memory → ready`, each step completing rather than one bar filling.

## Motion

| Interaction | Spec |
|---|---|
| Model change | Shared-element morph, `--st-t-surface`; falls back to the current keyed `st-enter` |
| Load phases | Step marks animate `st-mark`, `--st-t-local`, staggered 40 ms |
| Fit verdict appearing | `st-enter`, `--st-t-control`, 4 px — one motion, not three |
| Row expand (show more) | grid track `0fr → 1fr`, `--st-t-local` |

## Verification

- Assert every variant renders a **fit verdict string from the fixed set**; a
  variant without one is a failure.
- Assert the disk-after figure equals `free − downloadSize` for each row.
- Screenshot both themes at 1512 and 900 px: the variant list must not clip.

## Chat and image models, and custom imports — 12 October 2026

**Status: proposal for integration.** The current Models screen still runs one
catalog model. This section describes the next implementation; the example
names supplied by the user are design labels, not verified model releases.

### Simple page flow

1. Keep the current chat model card, context controls and Computer use choice.
2. Add a small **Add image model** action below the chat model card. It expands
   a second card with an image icon, model/version picker and its own load state.
   The empty card stays out of the way for people who only use chat.
3. With both selected, **Tune & run** prepares the pair. The right column shows
   their combined RAM and each GPU's budget, with separate contributions for
   chat, context, images and free space. Count unified memory once.
4. Offer **Keep both ready** when the pair's resident and peak working memory
   fits. If it does not, explain the capacity limit and offer **Load images when
   needed**. Require the user to choose that fallback; do not silently unload
   chat while claiming both models are running.
5. Image controls use size and quality presets. Context remains a chat setting.
   Show image latency in seconds per image only after an adapter supplies a
   defensible estimate or a measured benchmark for that resolution and quality.
6. Keep one Saved settings dropdown. A named setting can eventually remember
   both exact model versions, the loading policy and each model's tuning.
   Migrate existing single-model profiles as chat-only settings.

Example arrangement:

| Main column | Preview column |
| --- | --- |
| Chat · Qwen3.8-Flash-Next | Expected chat speed |
| Images · QwenImage2.1-Turbo, optional | Expected image time, or Not measured |
| Conversation context | Combined memory by RAM and GPU |
| Computer use | This computer: every CPU and GPU |
| Tune & run | |

### Runtime needed before this can be offered as working

- Replace the single `library.loaded` identity and singleton supervisor with a
  model registry and per-role engine instances. Keep independent load/error/
  cancel states and ports. Route chat and image requests by capability.
- Add an image generation adapter with a declared architecture/format/device
  support matrix. Reading images in a chat model is a different capability from
  generating images. The current `beginner.readImages` toggle cannot provide
  this image generation slot by itself.
- Reserve memory for both weights, chat KV cache, image runtime workspaces,
  auxiliaries and system headroom before launching. An image adapter must report
  its peak requirement and supported offload behavior. Do not assume that its
  weights or active tensors can spill to SSD like a chat model.
- Keeping two models loaded and generating from them concurrently are separate
  decisions. Default to scheduling heavy work on a shared GPU. In Expert mode,
  offer device assignment for each model and concurrent execution where the
  backend supports it and the peak memory fits.
- Preserve a healthy running chat model if adding an image model fails. Monitor
  and Serving need model identities per engine; progress and memory must be
  attributed to the right instance.

### Bring your own model

Put **Import model…** in Manage models, with **Choose file/folder** and
**Download from a model URL/repository**. The host inspects metadata and required
files before presenting a review with name, role, format, size and compatibility.

- A model outside Strata's curated catalog can run when an installed adapter
  supports its architecture and format. Start with compatible chat GGUF imports;
  add image bundles after the image adapter's supported formats are defined.
- Imported models receive a neutral **Custom** label. Curated versions can show
  **Strata optimized** only when that exact model/engine/device combination has
  a validated tune. Custom models retain context, resource choices and Expert
  mode, with conservative defaults and an optional benchmark.
- If an adapter does not support the architecture, show the precise missing
  capability before Load is enabled. An import cannot make arbitrary weights
  compatible. Offer the existing custom model API connection when the user
  already has another compatible model server running.
- Do not execute repository code during metadata inspection. Register exact
  source files, hashes, tokenizer/template and dependencies with a stable model
  identity; copy or reference files deliberately and show ownership before
  offering deletion. Partial downloads are resumable; cancellation retains a
  recoverable installation state.
- Unknown performance stays **Not measured**. Unknown memory needs a compatibility
  probe or user-provided limit before promising that a model will fit.

### Delivery order and acceptance

1. Model registry and custom chat import; preserve current catalog behavior.
2. Image adapter and image result routing; validate one supported image family.
3. Shared planner, paired load lifecycle, resource scheduling and device choices.
4. Paired saved settings, Monitor/Serving integration and measured forecasts.

Accept only after chat+image stays loaded on a capable machine, the smaller-PC
fallback is explicit, every GPU budget is respected, import failures preserve
the running model, paired settings survive restart and cancel/unload target the
correct engine. Exercise one GPU, multiple GPUs, unified memory and CPU-only
hardware only where each adapter actually supports those configurations.


### Implemented UI and downloads — 12 October 2026

The model summary is now visible in both normal and Expert modes. The Version
card includes an optional Qwen-Image-2.1-Turbo selector. Off is the initial
choice. Selecting a quant does not start a download; Download is a separate
action with inline progress, cancellation, retry and downloaded states.

| UI choice | GGUF file | Download size |
| --- | --- | --- |
| Q4 | qwen-image-2.1-turbo-Q4_K_M.gguf | 4.20 GB |
| Q5 | qwen-image-2.1-turbo-Q5_K_M.gguf | 5.39 GB |
| Q6 | qwen-image-2.1-turbo-Q6_K.gguf | 6.27 GB |
| Q8 | qwen-image-2.1-turbo-Q8_0.gguf | 7.64 GB |

Source: [Unsloth repository](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF),
revision `5d1f9f759ea877ae556cda88dbec958b62ad3809`. The reviewed byte counts
and SHA-256 values are pinned in `server/image-catalog.json`. Sizes above use
GB in decimal units, matching the source file sizes.

`server/image-models.mjs` owns a separate image library. It persists choice and
installation metadata in `data/image-models.json` under STRATA_HOME and writes
real files beneath `models/image/<quant>/`. It leaves the chat library and
chat engine process independent. Switching Off retains downloaded weights.
The host checks free storage, exact byte count, GGUF magic and SHA-256 before
promoting a partial file to an installed model. Cancel or verification failure
removes the partial file. Retry starts a new download.

Host contract:

- `GET /api/image-models`: source metadata, choices, selected/applied values,
  per-version installation state and generation capability.
- `PATCH /api/image-models` with `{selected: "off" | "q4" | "q5" | "q6" | "q8"}`:
  store a choice without fetching or loading anything.
- `POST /api/image-models/<quant>/install` and `/cancel`: explicit download actions.
- `POST /api/image-models/apply`: validate that the selected choice is installed
  and that this host has image generation capability before applying it.
- SSE event `image-models`: state and progress updates over the existing stream.

Named settings retain `imageModel`. Earlier settings omit this field and migrate
to Off. Restoring settings restores both the engine configuration and image
choice. A pending image selection is retained across refresh.

**Runtime integration still required:** STRATA_REAL_ENGINE=0 uses the existing
prototype convention: installation and application are simulated and no weights
are fetched. STRATA_REAL_ENGINE=1 downloads and verifies the requested diffusion
GGUF, but reports image generation unavailable. Production generation must add a
compatible image adapter, its text encoder and VAE, separate load/process health,
image result routing and shared memory planning. Do not treat a downloaded GGUF
as a runnable image pipeline. The current forecast explicitly labels itself as
chat memory when an image choice is selected; it does not claim combined fit or
image throughput. Real mode refuses to apply an image choice until that adapter
exists; selecting Off retains normal chat operation.


### Family switcher and simplified management — 13 October 2026

- The header replaces the descriptive sentence with a Model family combobox.
  Its options come from `modelFamilies(library)`, so registering future catalog
  families also registers them in this switcher.
- Normal and Expert modes share the model summary in the toolbar, before Saved
  settings and Manage models and above the separator. In Expert mode it retains
  its disclosure behavior for the basic settings.
- Family changes prefer a remembered exact version, then downloaded versions,
  then a matching precision or Balanced. Unapplied configuration, image choice
  and active saved settings are retained per family during the current visit.
  Selecting the same version leaves the draft unchanged. Switching families is
  a setup choice; Tune & run still controls the engine lifecycle.
- The Manage models dialog uses `ManagedModelLibrary`: one compact version list,
  one occurrence of each quantization, one selected-version memory summary and
  one primary action. Running and downloaded states are visible in the list.
  Unload and Delete files live in Version actions; deletion keeps its confirmation
  and running versions must be unloaded first. The existing standalone library
  remains available to its older route.
- Opening Manage models uses the family and exact version currently selected in
  setup, including a choice that has not been downloaded yet.

The prototype catalog now includes Qwen3.6-35B-A3B alongside Qwen3.8-Flash-Next.
Both are design fixtures with illustrative sizes and performance. Qwen3.6's
architecture/context metadata is based on the [official config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json).
Its hybrid attention memory estimate remains approximate. This change does not
add a real GGUF download manifest or engine compatibility validation for that
family. The real catalog remains separate in `server/real-model-catalog.mjs`;
production integration must validate exact files, hashes, architecture support,
context limits and measured memory before exposing native downloads.


## 14 October: multiple model instances

See [Multiple loaded models](09-multiple-models.md) for the implemented Models/Monitor/Serving flows, API contracts, shared memory planning and remaining native adapter work. Tune model and Load model are separate actions.
