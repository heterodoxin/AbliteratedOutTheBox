# Strata integration

This directory ports the inference-time projection from `src/llama-adapter.cpp` and the fit-free recipe from `tools/abliterate/abliterate.cpp` into [Niko1221/Strata](https://github.com/Niko1221/Strata), pinned at `8183c6b67162340c09b67360860177c577a65f2e`.

## Files

- `native/abliterate.cu`: CUDA/HIP projection, activation capture, and protocol implementation.
- `native/abliterate.hpp`: the native hook API.
- `native/abliterate-parity.cpp`: GPU checks using synthetic residuals; no model download.
- `patches/strata-hook.patch`: connects the hook to Strata's existing control-vector call sites and resident engine protocol.
- `abliterate.py`: calibration, gate, and generation driver. It uses Strata's tokenizer and the GGUF's own chat template, with thinking disabled and greedy generation.
- `build.py`: downloads the pinned engine, checks/applies the patch, installs the native sources, and builds it. Re-running it against the same prepared source is supported. It refuses other source revisions and conflicting hook files.

## Native integration

`--abliterate` allocates stable direction, mode, and capture buffers before graph capture. `Cvec::covers` includes the hook's middle-layer band, so the existing prompt, per-layer, and verify-window paths materialize the residual write and invoke the hook. This also prevents a fused read from skipping or applying that write twice. No hook allocation or projection is added to the default path without the flag.

For each residual stream, the edit is:

```text
h = h - max(dot(h, refusal) - refusal_reference, 0) * refusal
h = h - max(dot(h, disclaimer) - disclaimer_reference, 0) * disclaimer
```

The second projection reads the result of the first. When the edit is off, the hook leaves a materialized residual unchanged. If a GR write was pending, it performs that write before capture/editing. Exact end-to-end parity with the unmodified engine remains unverified.

The capture ring stores the last 25 token states at each gate layer, averaged across hyper-connection streams. Each request gets fresh recurrent/KV state through Strata's normal uncached prompt path. Calibration, look-ahead, and answer generation use the same resident model. No response string matching, sidecar, training, or weight rewriting is used.

## Resident protocol

These commands are available between `GEN` requests when the engine starts with `--abliterate`:

| Command | Result |
| --- | --- |
| `ABL INFO` | Layer count, width, residual stream count, history capacity. |
| `ABL CAPTURE` | Disable edits, clear capture history, invalidate previous directions. |
| `ABL READ N` | Read the last N captured states, oldest first; 1 through 25. Missing layers and non-finite states fail. |
| `ABL SET L REF REF2 R... R2...` | Upload two directions and references for edit layer L. Directions must be finite, unit length or zero. |
| `ABL ON` | Enable editing after every edit layer has a direction. |
| `ABL OFF` | Disable editing and capture. |

Mode and directions are device buffers read at graph replay time. Invalid commands disable the edit. The Python driver owns the process exclusively and resets the mode after each answer.

## Checks

```bash
python3.11 -m unittest discover -s strata -p 'test_*.py' -v
python3.11 strata/build.py --backend hip --arch gfx1201 --test -- -DSTRATA_MMQ_KQUANTS=OFF
```

The HIP build and isolated GPU checks were run on a Radeon AI PRO R9700. The CPU tests compare the calibration against a scalar reference to the original recipe and exercise request cleanup. Full-model Strata behavior, performance, and CUDA execution have not been tested.

To prepare an existing checkout without compiling:

```bash
python3.11 strata/build.py --source /path/to/Strata --backend hip --prepare-only
```

The original engine's license remains in its downloaded checkout. The retained upstream source tree in the root of this branch supplies the original hook and the compatible tensor library used by the Strata build; the commands above build and run Strata.
