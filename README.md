![AbliteratedOutTheBox](assets/banner.png)

# AbliteratedOutTheBox

## What is this project

AbliteratedOutTheBox is a fork of llama.cpp that attempts to reduce refusals and safety disclaimers through inference-time activation edits. The original model weights stay unchanged. For each prompt, `llama-abliterate` measures comply versus refuse openings and clean versus disclaimered openings, then uses the model's own continuation to decide whether to apply the edit. It needs no training, corpus calibration, sidecar, or second model. Architectures must expose the shared `build_cvec` layer-output hook used by the adapter.

## Install

Requires CMake 3.14 or newer, a C++17 compiler, a build tool such as Make, OpenSSL development files, and a GGUF model. Platform dependency commands are in [the build guide](docs/build.md).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-abliterate -j

./build/bin/llama-abliterate \
    -m /path/to/model.gguf -n 256 \
    -p "Give me step by step instructions to pick a lock"
```

`--ff-threshold <f>` (default 0.70) sets how readily the gate fires; `--edit-strength <f>` (default 2.0) sets the refusal reference; `--suffix-strength <f>` (default 1.0) sets how far the chat-template suffix is shifted toward comply; `--clamp-window <n>` (default 8) sets how many generated tokens the edit stays on; `--first-candidates <n>` (default 4) sets how many first-word candidates are scored by the refusal readout; `--disc-strength <f>` (default 0.4) sets the disclaimer reference; `--deliver-strength <f>` (default 1.0) sets how hard the response is pushed from evasion into delivery; `--deliver-window <n>` (default 8) and `--rep-penalty <f>` (default 1.15) tune the delivery steering. A negative disclaimer or zero delivery strength disables that edit and its anchor prefills. These flags accept finite numbers. See [tools/abliterate/README.md](tools/abliterate/README.md) for the tool details.

## Why it's better

The directions come from the current prompt instead of a fitted dataset or a separate weight file. One model supplies the measurements and generates the answer, and the edit uses the shared llama.cpp graph hook rather than separate code for each architecture.

When the gate stays off, the adapter's correction is zero. Comparisons with the base model must use the same chat template, thinking setting, backend, and greedy decoding. This does not guarantee that every benign prompt leaves the gate off.

Refusal reduction and answer quality still depend on the model and prompt. A low projection onto the measured refusal direction does not establish that an answer is useful or coherent; generated responses need to be read directly. No benchmark refusal rate is claimed for the current implementation.

## Why should I care

You can try a runtime edit without creating another model checkpoint or fitting an adapter. The tool supports llama.cpp's CPU and GPU backends and keeps one copy of the model weights. It adds prompt processing and per-token projection work, so latency and throughput depend on the model, backend, and prompt length.

## Advanced

Each prompt uses five comply, five refuse, four clean, four disclaimer, four delivery, and four deflection anchor prefills. A separate clean prefill generates up to 24 look-ahead tokens for the gate, and a final prefill starts the answer, so an unfired prompt costs 28 prefills. A fired prompt adds a prefill of the user text to read the suffix baselines, a shifted prefill checked with eight look-ahead tokens, one prefill per extra first-word candidate, and one more prefill per weaker retry. Disabling the disclaimer and delivery edits removes 16 anchor prefills. Anchor openings and look-ahead tokens are discarded; the generated answer receives no forced opening.

The comply and refuse openings are twins written in one formal register, so their contrast reflects the decision to deliver rather than an upbeat versus solemn tone. The refusal direction starts as the mean refuse state minus the mean comply state, with sequential projections against the centered comply openings subtracted. The gate uses the peak look-ahead projection, averaged over layers `n_layer/3` through `2*n_layer/3`, and fires above `--ff-threshold`. On a fired prompt, the refusal reference is the comply projection mean minus `--edit-strength` times its population standard deviation. The disclaimer reference is the clean projection mean minus `--disc-strength` times the clean-to-disclaimer gap. Each correction removes only the projection above its reference, from layer `3*n_layer/7` through `2*n_layer/3`. On the user-text positions that clamp is the whole edit. The chat-template suffix after the user text is where the model reads out its decision, so each suffix position is shifted `--suffix-strength` comply-to-refuse gaps toward comply from its own unedited projection, along the refusal direction made orthogonal to the final suffix step so the shift does not reopen the template; if that ends the turn within eight look-ahead tokens it is retried at half strength, down to the plain clamp. The first word is chosen among the top `--first-candidates` tokens as the one whose next state has the lowest refusal projection on the gate scale. During generation the clamps stay on for `--clamp-window` tokens and then release, because clamping the whole answer kept a whimsy-prone MoE joking once its opening had committed. Removing the refusal direction alone often leaves the model evading rather than delivering, so on a fired prompt a third delivery direction (committed-delivery openings minus deflection openings, orthogonalized against the refusal direction) is clamped up for the first `--deliver-window` generated tokens, with a repetition penalty and a skip of template protocol tokens on the edited generation; all of this acts only when the gate fires, so benign output is unchanged.

The tool generates one response with greedy decoding and renders the chat template with thinking disabled. The gate is per context, so the runtime handles one sequence at a time. The context must fit the longest anchor, the look-ahead, and the generated answer. Failed captures, context overflow, or non-finite projections stop the tool with an error. Fixed anchor openings can miss refusal behavior, benign prompts can trigger edits, and edits can leave refusals, disclaimers, repetition, or incoherent text. Increasing either edit strength can worsen answer quality.

## Known limitations

Some models still have problems. On the hybrid qwen3_5 architecture, which mixes recurrent linear-attention layers with full attention (tested on Qwen3.8-27B and MiMo-V2.6-9B), the edit barely changes delivery: the recurrent layers cannot absorb an edit strong enough to change the decision without ending the answer early, so most prompts there still refuse. Low-bit quantizations of large models remain harder than q6 or q8, and some answers on any model still open with a short disclaimer before the content.
