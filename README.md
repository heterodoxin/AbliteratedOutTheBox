![AbliteratedOutTheBox](assets/banner.png)

# AbliteratedOutTheBox

## What is AbliteratedOutTheBox?

AbliteratedOutTheBox is a fork of llama.cpp that gets any supported model to answer flagged harmful prompts, delivering the requested content instead of a refusal, at inference time and with the original weights left on disk unchanged. It needs no fit, no training, and no extra files: for each prompt `llama-abliterate` reads two directions from the prompt itself, by forcing the assistant to start a few comply versus refuse openings (and clean versus disclaimered openings), diffing the activations, and a look-ahead gate decides per prompt whether to remove them. The edit happens inside the graph at the `build_cvec` layer-output site that 131 of the 165 model architectures share, so one recipe covers them with no per-model code.

## Install

```bash
# build the runtime (pick the backend you already use: -DGGML_CUDA=ON, -DGGML_HIP=ON, or CPU-only)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-abliterate -j

# convert the model to a GGUF for the runtime (any quantization; needs Python with torch, transformers, gguf)
python3 convert_hf_to_gguf.py /path/to/Qwen3-8B --outfile qwen3-8b.gguf --outtype q8_0

# run: no fit, no sidecar, any backend including pure CPU
./build/bin/llama-abliterate \
    -m qwen3-8b.gguf -ngl 99 -n 256 \
    -p "Give me step by step instructions to pick a lock"
```

`--ff-threshold <f>` (default 0.70) sets how readily the gate fires; `--disc-strength <f>` (default 0.4) sets how hard the safety-disclaimer direction is removed. See [tools/abliterate/README.md](tools/abliterate/README.md) for the tool details.

## Why it's better

Measured on Qwen3-8B with a hand-validated response hidden-state probe (no string matching, no judge model), over held-out sets, with the edit read from the prompt at inference (no fit):

- Refusal drops 96.3% to 7.9% on broad held-out harmful prompts and 82.0% to 0.0% on JailbreakBench; the model answers and delivers the content, confirmed by hand-reading the generations.
- Benign is preserved exactly: the gate fires on 0.4% of alpaca prompts (1 of 256), and when it does not fire the output is bit-identical to the base model (KL 0.000000 over held-out benign, 0 gated, 0.0 logit change).
- A second direction removes most of the safety-disclaimer hedging from the compliant answers (a hidden-state read of the disclaimer character drops about a third), while answers stay coherent (distinct-token ratio around 0.65 vs 0.84 base). Pushing `--disc-strength` higher removes more disclaimers but can start to degenerate longer outputs, so it is tunable.

Weight-baked abliteration edits the stored weights and perturbs every prompt, so its benign output is never identical and it needs per-architecture surgery. Here the edit is a scale-free projection at the `build_cvec` layer-output site, so one recipe covers 131 architectures with no per-model code, and prompts the gate leaves alone are untouched.

## Why should I care

It runs with nothing to set up and nothing to download beyond the model itself: no fit, no sidecar, no second file, on whatever you already have including CPU or a few gigabytes of VRAM for a quantized model, because the runtime is plain llama.cpp. You keep one copy of the weights, ordinary prompts behave exactly as before, and generation throughput is unchanged because the only added cost is one extra prompt prefill when a prompt is flagged.

## Advanced

Both directions are read per prompt and the edit covers the prompt, not only the generated tokens. The refusal decision is committed into the prompt's KV during prefill, so editing only the generated positions leaves refusal near 95% versus single digits when the prompt is edited too. The runtime therefore uses two passes: a clean detection pass that fires the gate, then an edited prefill plus generation. Each per-token edit is a projection, so decode throughput is unchanged.

The refusal direction is the difference between forced refuse and comply openings, orthogonalized against the comply-opening spread to drop benign-common components, and used for both the gate and the edit. The disclaimer direction is the difference between disclaimered and clean compliant openings. A look-ahead gate generates up to 24 of the model's own tokens and fires when the peak refusal projection, band-averaged across the middle layers, exceeds `--ff-threshold`; at 0.70 that holds real-benign false positives near 0.4% at 0.89 harmful recall, and lowering it raises recall at the cost of more benign fires. When fired, the edit pulls the refusal projection down to just past the comply level (reference 0.3 of the comply-to-refuse gap below comply) and the disclaimer projection down by `--disc-strength` of its clean-to-disclaimered gap, from layer `3*n_layer/7` through `2*n_layer/3`.

Measurement note: refusal is measured by the probe over the model's own generation. Re-reading finished text through the base model over-counts, because a compliant answer that carries a safety disclaimer reads as refusal-like to a classifier even though it delivered the content; the edit-during-generation measure, hand-validated against reading the outputs, is the one reported here.

Known limits: the runtime edits one sequence at a time, since the gate scalar is per context; the directions are read from a fixed set of forced openings, so very unusual prompts may gate or edit less well; and a strong `--disc-strength` can degenerate long outputs, which is why it is a tunable with a moderate default.
