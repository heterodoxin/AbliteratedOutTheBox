# AbliteratedOutTheBox for Strata

Inference-time abliteration for the [Strata inference engine](https://github.com/Niko1221/Strata). This branch ports AbliteratedOutTheBox's activation hook into Strata's native residual-stream execution path. Model weights stay unchanged on disk.

## Validation status

**The code builds and passes the checks run so far, but full-model Strata inference has not been tested.** The HIP engine compiled on a Radeon AI PRO R9700. Isolated GPU tests passed for the sequential projections, activation capture, history wrapping, and switching the hook inside a captured graph. Python tests passed for calibration, gate scoring, and request cleanup.

These checks do not establish refusal-removal rates, response quality, throughput, or exact base-model output preservation. Those require an end-to-end run with a compatible Strata model. The results from other backends do not establish results for Strata.

## Build

The build helper downloads a pinned copy of Strata, applies the integration patch, copies in the native hook, and builds Strata. It does not download model weights.

```bash
python3.11 -m pip install -r strata/requirements.txt

# AMD / HIP: use your card's architecture.
python3.11 strata/build.py --backend hip --arch gfx1201 --test -- -DSTRATA_MMQ_KQUANTS=OFF

# NVIDIA / CUDA: use your card's compute capability.
python3.11 strata/build.py --backend cuda --arch 89 --test
```

A C++ toolchain, CMake 3.24 or newer, and the selected GPU toolkit are required. The patched engine is written to `strata/.strata-src/build-abliterate/strata` (`strata.exe` on Windows).

## Run

Use a Strata-supported **Qwen3.8 / qwen4exp GGUF** with its tokenizer and chat template in the file:

```bash
python3.11 strata/abliterate.py \
  --model /path/to/model.gguf \
  --prompt "Explain how a transformer works" \
  --max-new-tokens 256
```

The driver loads one resident Strata process, reads per-prompt directions from forced openings, measures a clean look-ahead, and generates the answer with the edit enabled only when the gate fires. Calibration text is discarded; it is not prepended to the answer. The final prompt is processed again before answering.

- `--ff-threshold 0.70`: the gate threshold; lower values allow more edits.
- `--edit-strength 2.0`: refusal reference, in comply-opening standard deviations below the comply mean.
- `--disc-strength 0.4`: disclaimer-removal strength. A negative value skips that direction and its eight anchor prefills.
- `--context 4096`: context capacity. Prompts and anchor suffixes are checked before model loading.
- `--strata-source PATH` and `--engine PATH`: use an existing patched Strata checkout or executable.
- Put extra Strata engine arguments after `--`, for example `-- --gpu 0`.

## Port details and limits

The native hook applies two sequential clipped projections at Strata's existing layer-output control-vector site, after the FFN residual write. It covers prompt processing and decoding, including captured graphs. Hyper-connection streams share each layer's direction; calibration captures their mean. The gate uses the middle third of the layers, and editing starts at `3 * layers / 7`.

This first port supports one text sequence on one GPU through Strata's Qwen3.8 engine path. Batching, speculative decoding, vision, layer splitting, and simultaneous control vectors are rejected. Prompt and conversation caches are disabled because each request has its own directions and edited state must not leak into another request. CUDA sources are included; CUDA execution has not been tested. Metal, SYCL, and the separate Qwen3.6 engine path are not ported.

The default performs 21 anchor prefills, a clean look-ahead, and a final answer prefill. An early end-of-generation token requires one additional gate prefill. This adds substantial work before the answer. No speed or model-quality improvement is claimed.

See [the integration notes](strata/README.md) for source layout, protocol, and checks.
