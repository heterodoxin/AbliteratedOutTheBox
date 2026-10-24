# Strata on an Intel Arc

Strata's engine is CUDA. On an Intel Arc it runs as **Strata's own engine, ported to SYCL** (`sycl/`, the
section "The engine itself on Intel" below), behind the same Strata server: OpenAI and Anthropic APIs, streaming,
tool calls, MCP and the web app all unchanged. llama.cpp's SYCL backend is the comparison point in the tables (it
runs the same GGUF at about a third of the speed).

Everything Intel-specific lives in `sycl/`; no shared file of upstream's is changed, so upstream merges stay clean.
`sycl/setup_intel.py` and `sycl/serve/server_intel.py` wrap upstream's `setup.py` and `serve/server.py` from
outside.

Written for and measured on an **Arc Pro B70 (32 GB)** running the Coder (IQ1_M) on Ubuntu 24.04, in a
PCIe 3.0 x8 slot (the card trains at Gen3 x8 there; it can do Gen5 x16).

## What you get (2026-10-01)

| | NVIDIA (Strata engine) | Intel Arc, Strata SYCL port | Intel Arc, llama.cpp |
|---|---|---|---|
| models | all four | the Coder IQ1_M; the original IQ2_XS too (experts beyond VRAM in the host mirror: 58-64 tok/s) | the Coder (fits VRAM) |
| model files | the same GGUFs | the same GGUFs + a native pack | the same GGUFs |
| where the model lives | experts in RAM, hot ones on the card | every expert in VRAM (`--stream-experts`: no host copy), shard 2's lookup table read from the SSD by row | all of shard 1 on the card, shard 2 paged from disk |
| RAM needed | 32-64 GB | little (23 GB is fine) | little (23 GB is fine) |
| decode, Coder IQ1_M | 44-51 tok/s on an RTX 5070 | **78.2 tok/s** (19-token prompt, 256 greedy tokens), **75.7** after a 2,184-token prompt, 69 after 40K, 66.6 after 128K, 55.9 after 256K; 48-72 through the API | 23-25 tok/s |
| prompt reading | ~1,870 tok/s | **790 tok/s** at 2,184 tokens, 1,117 at 40K, 888 at 128K, 757 at 256K | ~150 tok/s (424 at 105K) |
| speculative decoding (MTP) | yes | yes (the base checkpoint's draft layer; 70-85% of drafts accepted on code) | no |
| images | yes | not yet | not yet |
| context | up to 262K | 256K measured, 56 tok/s decode there (`--kv-resident`: the KV in pinned host memory, the attended window in VRAM) | 131K measured ceiling |

Decode speed with speculative decoding depends on the text: code drafts well, prose less so (see "Speed depends
on the text"). The SYCL numbers are greedy runs of the engine (as in "How to run it by hand" below, 256 new
tokens) on engine 0.1.31-sycl (2026-10-01; 0.1.32-0.1.38-sycl reproduce them exactly);
"Decode round 2" below lists what each change bought.

## Setup

On Ubuntu 24.04 or newer, from nothing. The Intel GPU driver (the `xe` or `i915` kernel driver and the compute runtime;
[INTEL_ARC.md](INTEL_ARC.md), "What you need") is assumed: `ls /dev/dri` shows a `renderD128`.

1. **Docker, once.** The engine runs in a container, so the PC needs Docker and no oneAPI:

        sudo apt install docker.io git
        sudo usermod -aG docker,render $USER

   Log out and in again (or reboot) so the groups apply. `docker ps` must work without `sudo`; the `render` group lets
   the container open the GPU.
2. **Get Strata and build the engine** ("How to build it" below, about 10 minutes the first time).
3. **Run setup:**

        ./setup.sh --backend sycl [setup.py's options, e.g. --model IQ3_S --context 32768 --port 8085]

   `setup.sh` makes its own Python environment in `.venv`, so Ubuntu's "externally managed" system Python is not
   touched. **Do not run `python3 sycl/setup_intel.py` yourself:** it installs setup's packages into whichever Python runs
   it, and Ubuntu's refuses (`externally-managed-environment`). `./setup.sh --backend sycl` is the one command, for the
   first install and every later start.

Setup is upstream's `setup.py` with the Intel steps swapped in (`sycl/setup_intel.py` imports it and replaces those steps;
setup.py itself is unchanged). The model choice, download, pack, tokenizer, MTP draft layer and the context and KV
questions are setup's own. What changes:

- **GPU check:** the Arc is found in sysfs (vendor 8086 under `xe` or `i915`) and offered through setup's AMD
  path. That is the path that builds locally and has no images.
- **Engine step:** it uses the SYCL build (`build-sycl-aot/strata` or `build-sycl/strata`, run in the `strata-sycl-dev`
  image by `sycl/serve/strata-sycl.sh`) instead of compiling CUDA or HIP.
- **RAM rule:** this does not apply on an `xe` card. The CUDA engine keeps every expert in RAM; the port streams them from
  the GGUF into VRAM (`--stream-experts`), so RAM only decides the KV streaming. `--check` lists what fits by VRAM.
- **The config:** it uses the container's paths and `"backend": "sycl"`. The VRAM reserve is 1,024 MiB up to
  32K and 2,048 MiB with 4,096-token prompt chunks above that, 300 MiB on a card under 12 GB; a
  `--vram-reserve-mib N` you give is kept as given. KV streaming (`--kv-resident 32768`) is on from
  64K up when the RAM holds the KV. A `model_switcher` or `sampling` block from an earlier config is kept.
  `run-<model>.sh` starts `sycl/serve/server_intel.py`.
- **Arc A-series (`i915`, e.g. the A750 with 8 GB):** a different config, see "Arc A750 and the other Alchemist cards" below.
- **Engine settings:** the config's `"env"` block reaches the engine. `strata-sycl.sh` forwards every variable starting with
  `STRATA_`, `ONEAPI_`, `UR_`, `IGC_`, `SYCL_` or `ZES_` into the container and sets `STRATA_VERIFY_DEVICE_PLAN=1`,
  `STRATA_VERIFY_NO_HOST=1` and `STRATA_STAGER_THREADS=12` unless the environment says otherwise (for example
  `"env": {"STRATA_VERIFY_NO_HOST": "0"}`).

If a future `setup.py` drops a step this relies on, it stops with a message instead of writing a wrong config.
The models, packs and checkout must sit under the folder `strata-sycl.sh` mounts at `/work` (the one above the
checkout, or `STRATA_SYCL_ROOT`). An earlier Strata install on the same PC keeps its data folder in
`~/.config/strata/settings.json`; pass `--data-dir <folder above the checkout>/Strata-data` to put this one under the
mount. A symlink to a folder outside the mount does not work in the container; a hard link does.

Then `run-<model>.sh` (or `./setup.sh --backend sycl` again) starts the model. The first start of a 30 GB model
takes about two minutes. `--port N` and `--host 0.0.0.0` work as in upstream's setup.

## How to build it

The engine is built in Docker, in the `strata-sycl-dev` image (oneAPI 2026.1 compiler, oneMKL and Level Zero, from a
community llama.cpp SYCL image), so the PC needs no oneAPI. From the checkout (its folder is called `Strata` here; the
folder above it is what the container mounts at `/work`):

    cd Strata
    docker build --pull -t strata-sycl-dev sycl/tools   # once: about 6 minutes, a 13 GB image
    H=$(basename "$PWD")

**Arc B-series** (Pro B70: `bmg-g31`; B580, B570 and Pro B60: `bmg-g21`; `ocloc ids bmg-g21` in the image lists the
device ids). Ahead-of-time code, no compile at the first start:

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

**Arc A-series** (Alchemist: A750, A770, A580, A380): no `AOT=`. The code is compiled from SPIR-V at the first start
(JIT, about a minute). Nobody has run an AOT build for these cards.

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

`--pull` matters: the image starts from a community image whose tag moves, and `docker build` reuses a copy of it from an earlier
build unless it is told to fetch the current one. An old copy can hold a device compiler (ocloc from Compute Runtime
26.22.38646, IGC 2.36.3) that crashes on the Battlemage kernels ("IGC: Internal Compiler Error: Floating point exception", 77 device
images, then `gen compiler command failed with exit code 245` at the last link, #3219). `sycl/tools/build.sh` checks the
version before the compile and stops with the fix. If you cannot update the image, `-e I8_ESIMD=OFF` builds without the fused int8
kernels (stubs: the prompt's experts take the slower XMX path, the answers do not change); `-e ALLOW_BAD_OCLOC=1` builds anyway.

The build takes about 3 minutes and ends with `BUILD EXIT 0` and `errors: 0`. The binary is `build-sycl-aot/strata`
(B-series) or `build-sycl/strata` (A-series); setup finds either one. Run the `docker build` again after pulling a
Strata update that changes `sycl/tools/Dockerfile`, and the `docker run` after every update of the engine (setup
says when the engine is older than it needs).

## Things that matter on this GPU

- **`SYCL_CACHE_PERSISTENT` must be 0.** The persistent JIT cache segfaults on Xe2 during the first
  compile. The start script sets it; if you run llama-server by hand, do too.
- **The whole model goes on the card** (`--n-gpu-layers 999`), except `per_layer_token_embd.weight`,
  the single 28.8 GB tensor of shard 2. `--override-tensor per_layer_token_embd=CPU` keeps it in host
  memory, mmapped and paged from the SSD by row, which is how the model's authors serve it. Host RSS
  stays around 2 GB.
- **Thinking.** The model reasons before it answers. Strata's web app has the setting; for an API
  client, put `{"reasoning_effort": "none"}` in `strata-<model>.shared-settings.json` next to the
  config, or send `chat_template_kwargs: {"enable_thinking": false}` per request. Without one of
  those a short `max_tokens` is spent entirely inside the think block and the answer looks empty.
- **`/health` says 503 while loading**; the server polls `/props` instead.

## How much context fits

The architecture keeps this small: only every fourth layer is full attention (12 of 48); the other
36 are gated-delta-net layers with a fixed-size recurrent state that does not grow with context.
Per token, the 12 attention layers keep 2 KV heads x 256 x (K + V) at q8_0, plus the
sparse-attention indexer's keys. **Measured: about 20 KiB per token** (VRAM grows 0.6 GB per 32k
of context with everything else unchanged).

| context | VRAM in use, model loaded | headroom on 32 GB | status |
|---|---|---|---|
| 32,768 | 28.4 GB | 3.5 GB | measured, the default |
| 65,536 | 29.0 GB | 2.9 GB | measured, loads |
| 98,304 | 29.6 GB | 2.3 GB | measured, loads |
| 131,072 | 30.3 GB | 1.5 GB | **measured, the practical ceiling**: a 104,798-token prompt (79% of the window) read in 247 s at 424 tok/s, then answered |
| 163,840 | ~31.0 GB | ~0.9 GB | not attempted: under the 1.2 GB safety margin |
| 262,144 | ~32.6 GB | none | **does not fit - asking for it took the host down** |

**Do not ask for more than fits.** On this driver a GPU allocation past VRAM does not fail: the xe
driver evicts buffers into host RAM, the kernel runs out of memory, and the machine livelocks until
its hardware watchdog resets it. A 262K request did exactly that on 2026-09-29, and llama.cpp's own
"failed to fit params" check fired too late to prevent it. Compute the KV size first and leave
1.5 GB free.

Prompt reading gets faster with size, because the work batches better: ~150 tok/s on a 2.7k-token
prompt, 424 tok/s on a 105k-token one. So a full 128k window costs about five minutes to read,
not fifteen.

## The engine itself on Intel: the SYCL port (`sycl/`)

Everything above runs the model through llama.cpp. This is the other half: Strata's own engine - the 50 CUDA
kernels and the host code that drives them (streams, events, graph capture, pinned memory) - built for the
Arc with oneAPI. It is a migration of the tree, not a new backend: the engine has no backend seam to slot
into.

**Layout.** `sycl/src` and `sycl/include` mirror the tree and hold only the files the port changes; the
SYCL build (`sycl/CMakeLists.txt`) takes every other source from the original location. No upstream file
is edited. `sycl/include/dpct/` is the vendored SYCLomatic helper library, so the port builds without the
migration tool.

**How it was made, so it can be redone.**

1. `sycl/tools/Dockerfile` - the dev image: the llama.cpp SYCL image plus SYCLomatic (`dpct` 2025.3),
   ninja, and the CUDA 12.8 headers that `sycl/tools/get-cuda-headers.sh` pulls out of NVIDIA's pip wheels
   (dpct parses CUDA; it needs the headers, not the toolkit).
2. `sycl/tools/migrate.sh` - writes a compilation database for the 86 CUDA-touching translation units and
   runs dpct over them. 85 migrate; dpct reports no line it could not migrate, and about 1,400 advisory
   notes.
3. `sycl/tools/fixups.py` - what dpct got wrong or could not do, as an idempotent script with a reason per
   item. The ones that mattered:
   - CUDA's null stream means the default stream; dpct turned it into a null `sycl::queue*`. Every
     stream cast now goes through `strata::q_of()` (`sycl/include/strata/sycl_queue.hpp`).
   - `__ldg((const float*) p)` came out as `*p`, reading one byte of a float scale (two sites, s_gemv).
   - `__fadd_rn(a, b ? c : d)` lost its parentheses (two sites).
   - the ggml lookup tables were threaded through kernel parameters with the wrong table per template;
     they are plain `static const` arrays read from device code, as ggml-sycl does.
   - helper headers renamed since dpct 2025.3 (`entangle`, `chunked_partition`), graph introspection and
     `cudaGraphUpload` (no SYCL equivalents), `%globaltimer` (the stage profiler reads zeros).
4. `sycl/tools/build.sh` - configure + build with icpx inside the image. Two compiler flags are load-bearing:
   `-fp-model=precise` (icpx defaults to a fast FP model) and `-cl-fp32-correctly-rounded-divide-sqrt`
   for the device compiler - the Arc's fp32 divide is not correctly rounded by default (OpenCL allows
   2.5 ulp), and Strata's quantizers are byte-exact against ggml through `amax / 127`. Measured: without
   it `quantize_act_parity` has 303k mismatches, with it none.

**Where it stands (2026-09-29).** The whole tree builds and links: the `strata` binary and 22 kernel
parity tests. On the B70, 19 of the 22 pass byte-for-byte or within their tolerances; the other three
need model fixtures (`iq_parity`, `ple_parity`) or the tensor-core kernel below (`qsa_prompt_attn`).

| parity test | result |
|---|---|
| bf16_gemv, cvec, dequant_s2, elementwise, gdn, gr, kv_q4, kv_q8, kv_stream, qsa, quantize_act, rope, router_top10, s2_gemv, s2_gemv_q8, s_gemv, s_gemv_q8k, sampler, shared_expert | pass |
| iq_parity, ple_parity | `ctest` generates the `iq_parity` fixtures into the build tree (`iq_parity_fixtures`, needs python3 with numpy; skipped without). `ple_parity` needs the Q2_0 shard 2 GGUF and the dense pack: configure with `-DSTRATA_PLE_GGUF=<shard 2 .gguf>` (and `-DSTRATA_PLE_FIXTURE_DIR=<dir with ple_in.bin and ple_out.bin>`); a missing input is a hard error, as in the CUDA build (#1795) |
| qsa_prompt_attn_parity | needs the tensor-core kernel (below) |
| kv_hybrid_parity (0.1.27, hybrid K8V4 KV) | appends, gathers and attention pass; its last step is that same tensor-core kernel |

**The engine end to end (2026-09-29, later the same day).** It generates. Prompt "Write a Python function
that returns the n-th Fibonacci number", greedy, 64 tokens:

> `<think>` The user wants a Python function that returns the n-th Fibonacci number. Let me consider the
> options: 1. **Recursive approach** - Simple but O(2^n) time complexity, very slow for large n. 2.
> **Iterative approach** - O(n) time, O(1) ...

Measured on the same card, same 2,185-token prompt, 64 greedy tokens (AOT build, `--spec 2`,
`--no-prefill-borrow`), against llama.cpp's SYCL build serving the same GGUF:

| | llama.cpp SYCL | Strata SYCL port |
|---|---|---|
| prompt reading | 138-319 tok/s | **560 tok/s** (693 at 2,000 tokens, 180 at 300) |
| decode, suffix drafter only (`--spec 2`) | 24.2-26.0 tok/s | 20.2-20.8 tok/s |
| decode with the MTP draft layer (`--spec 4 --mtp`) | - | **42.7 tok/s** at the 2,185-token context (81% accepted, 2.9 tokens/round); 45.3 on a short prompt |

Prompt reading is where Strata's design pays (oneMKL GEMM over dequantised experts, the whole model on
the card). Decode is at ~80%: a speculative round costs ~53 ms whatever its size and the suffix drafter
is accepted 9% of the time, so nearly every round yields one token; `--spec 4` (16.9 tok/s) was worse
than `--spec 2` (20.8) for that reason. The kernels themselves run at 130-160 GB/s of weights on a card
that streams 600 GB/s (`sycl/probe/bw.cpp`, incompressible data) - the same class llama.cpp reaches -
and five variants of the hot Q6_K matvec (lanes per row, rows per warp, unroll, 16-byte loads, software
pipelining; `mmvq_bench`) all landed in that band. The draft source was the lever: the base
Qwen3.8-Flash-Next checkpoint's MTP layer (`tools/mtp_fetch.py fetch`, `mtp_pack.py --experts q2_0`,
`mtp_rt.py`; 4.9 GB downloaded, 809 MiB of VRAM) drafts for the Coder fine-tune at 87% acceptance:
3.4 tokens per 53 ms round, 45 tok/s, the same greedy tokens. The `mtp.cpp` path ran on the first try
under the same flags.

Without AOT the runtime JIT-compiles every kernel on first use, ~47 s the first time a process runs;
`SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<dir>` keeps that across runs (15 MB).

How to run it by hand (a greedy test run, the way every number in this section was measured; inside the
`strata-sycl-dev` image, AOT build in `build-sycl-aot/`):

```
STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 \
build-sycl/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> \
    --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
    --prefill auto --spec 4 --spec-min-p 0.5 --max-context 8192 --tokens <ids> --max-new 64 --greedy
```

- `--stream-experts` (this port): no resident host copy of the experts; every one of the 12,288 goes from
  the GGUF into the VRAM cache through a small staging ring (`GgufExpertSource`). Upstream needs 32 GB of
  RAM for this model; with the flag the engine runs on 23 GiB.
- `STRATA_VERIFY_DEVICE_PLAN=1`: the GPU plans each layer itself (upstream's E-6; off by default there).
- `STRATA_VERIFY_NO_HOST=1` (this port): the host waits for the whole window graph instead of per-layer
  rings. Only valid with every expert resident, which is the case on a 32 GB card.
- `--no-prefill-borrow`: the prompt path must not lend expert slots (a lent expert is served from the host
  behind a flag the GPU does not see reliably here; long prompts hung without it).

What the port had to get right beyond compiling (each is an entry in `sycl/tools/fixups.py` or a flag):

- **Host-mapped flags.** Strata's decode is a GPU/CPU handshake through host-mapped memory. `volatile`
  device loads do not bypass the caches on Intel; system-scope atomics do, and they are the only thing
  measured to work (`sycl/probe/doorbell.cpp`, six variants). Host-to-device visibility *during* a kernel
  stays unreliable on this platform, which is why the all-resident path waits for the window instead.
- **Bounded spins.** A device spin that never sees its flag is not a hang of one process: the xe driver
  times the queue out and resets the GT node by node - a window graph has 2,400 - and the card stays
  wedged until a reboot (twice). Every spin is capped (`kSpinMax`).
- **32-lane sub-groups.** The kernels are written for warps; dpct pinned 134 of 289 launches, the rest
  would run at Xe2's default 16 (`-fsycl-default-sub-group-size=32`).
- **Synchronous copies.** `cudaMemcpy` blocks; dpct's default-queue `memcpy` did not wait. With the
  streaming ring that filled the expert cache from overwritten buffers: non-deterministic residuals, NaN by
  layer 5. Found with the per-layer residual ladder (`STRATA_VERIFY_DEBUG=1` prints R after every layer).
- `native_expert_parity` hand-ported: the GPU native expert kernel matches ggml's float reference on real
  IQ1_M rows (rel 1.1e-2, the same class as the CPU path).

Profiling: `sycl/tools/Dockerfile.unitrace` builds the dev image with Intel's unitrace; `unitrace -d` around
the engine plus `sycl/rank_kernels.py` gives device time per kernel. `mmvq_bench` and `native_expert_parity
NATIVE_BENCH=1` time the two hot kernel families in isolation (warm the clocks first: a 5 ms run measures
the ramp, not the kernel).

**After the 0.1.25-0.1.27 merge (2026-09-30, same card, same flags, AOT).** Fibonacci prompt (19 tokens),
64 greedy tokens: decode 46.5 tok/s (85% of drafts accepted). The 2,184-token prompt: 566 tok/s prompt,
38.3 tok/s decode (77% accepted on that text); the draft layer's prompt pass is now upstream's batched one
(38.6 ms for 2,184 tokens, from 105 ms for 19 before it). Parity: 19 of 22, byte-exact `quantize_act` back.

**Speed work, 2026-09-30 (after the 0.1.27 merge).** What moved the numbers, measured on the B70 with the
same prompts as above (outputs identical before and after each change):

| change | effect |
|---|---|
| every window graph and the drafter's graphs captured at load (`STRATA_WARM_GRAPHS=0`: as before) | the first request no longer pays 250-290 ms of captures |
| the commit graph left running while the drafter's round runs on its own queue (`Verifier::commit(wait=false)` + `commit_finish`) | decode 38.3 -> 43.0 tok/s at the 2,184-token context, 46.5 -> 49.8 short |
| upstream 0.1.29 merged (GDN recurrence pipelining, the block-scores read, sampler) | 2,184-token prompt 765 -> 792 tok/s, decode at that context 43.1 -> 45.2 |
| the draft layer's batched prompt pass takes prompts under 64 rows (it fell back to per-6-token graphs) | its prompt cost on 19 tokens 106 -> 16 ms |
| one device module per kernel (`-fsycl-device-code-split=per_kernel`) | the first launch in the prompt path (the embedding gather) 245 -> 1 ms |
| the expert dequant writes each thread's run of FP16 values as one vector store (it wrote them one 2-byte store at a time) | dequant per expert 0.085 -> 0.030 ms; prompt 496 -> 575 tok/s at 2,184 tokens, 720 -> 841 at 8,000 |
| SWAR sign compare/subtract in the expert dots, a local-memory resident-plan kernel, a split-K fused down kernel (bit-identical to the single-token one) | parity-clean, no measurable decode change; kept |
| the PLE row reader's default queue depth 16 -> 64 blocking O_DIRECT threads (`STRATA_IO_THREADS`; 128 and 256 change nothing: the drive tops out near 85k IOPS on the 27k random 4 KB reads a 2,184-token prompt needs) | the rows of that prompt 466 -> 330 ms |
| a short first prompt chunk (`STRATA_PREFILL_FIRST`, default 256 tokens) so the GPU starts while the remaining rows are still being read | 2,184 tokens: 711 -> 765 tok/s, 8,000: 857 -> 986; time to first token about 150 ms less. The chunk boundary moves rounding in the expert GEMMs, so a long greedy continuation can diverge late (token 45 of 64 on the test prompt) |

Draft policy sweep (2,184-token prompt, 128 tokens): windows of 6 (`--spec 4`) at 41.6-43.5 tok/s; `--spec 2`
33.6, `--spec 6` 35-37; `--spec-min-p` 0.3-0.7 within noise. Profiles (unitrace, `strata-sycl-dev:metrics` with
Intel's metrics libraries and `dev.xe.observation_paranoid=0`): the expert dot kernels are ALU-bound (77% XVE
active), the dense projections memory-latency bound (92-128 GB/s at 84-93% occupancy), a decode round is 80-85%
kernel time and ~5 us of launch gap per node over 2,400-2,600 nodes.

**XMX.** oneMKL's FP16 GEMMs already run on the XMX units (30-60 TFLOP/s in `xmx_gemm_bench`); the prompt path is
bound by the dequant that feeds them, not by the products. Two joint_matrix kernels were written and are correct
but lose to the existing paths on this card, so both stay opt-in: `xmx_gemm_iq` (a fused dequant + GEMM straight
from the quantized rows, 4-5x slower than dequant + oneMKL) and `qsa_prompt_attn_xmx` (the port of the mma.sync
prompt attention; `qsa_prompt_attn_parity` passes at 1e-6 of scale, 3x slower than the FP32 fallback per chunk;
`STRATA_PROMPT_ATTN_XMX=1`). Where the prompt time goes now (2,184 / 8,000 tokens): dequant 28% / 27%, GEMMs
22% / 20%, the per-layer host grouping 9.5% / 9%, attention 7% / 14%, embeddings + PLE rows 13% / 5%.

A trap worth knowing: with a native pack `--prefill-until N` does not feed the rest of the prompt through the
token loop (it is skipped for native packs), so the tokens after N are dropped and the model free-runs. Compare
output tokens between paths, never only timings.

**An 80,000-token prompt (2026-09-30).** Where the time goes changes completely at this scale:

- **The VRAM plan.** `--expert-cache auto` fills the card down to `--vram-reserve-mib` *before* the KV state
  (about 1 GB in INT8 at 81,920 cells) and the prompt chunk buffers (about 2 GB at `--prefill 8192`) exist. With
  the 1,536 MiB default that put 30.95 of the B70's 32.6 GB in use, the driver started migrating buffers and the
  run never finished. `--max-context 81920 --kv int8 --vram-reserve-mib 3072 --prefill 4096` peaks at 28.8 GB.
  (80k ids also exceed Linux's 128 KB single-argument limit: `--tokens-file`.)
- **Streamed experts.** The reserve costs cache slots (10,348 of 12,288 at 3 GB), and the prompt path streams the
  missing experts from the GGUF for every chunk. Two port bugs sat on this path and are fixed: the stream plan
  held `GgufExpertSource::blob()` pointers across ~1,900 reads of a 512-slot ring (so blobs were overwritten
  before they were copied: the 75 s / 1,065 tok/s measured first was computed partly with the wrong experts), and
  the stream-all walk hangs on this card in its first large chunk (the copy engine stops on a barrier). The stager
  threads now read the blobs themselves, and the port uses the per-layer routed-only walk (`STRATA_PREFILL_RING=8`
  upstream, the port's default; `STRATA_PREFILL_STREAM_ALL=1` restores the other for debugging).
- **Correct result: 80,000 tokens in 101 s = 790 tok/s.** GPU time: expert down GEMM 26.0 s, attention 15.4 s,
  dequant 12.7 s, QSA block selection 9.3 s (0.2 s at 8k: it scans every block of the context per query), gate/up
  GEMM 8.9 s, gather 6.1 s, the per-layer grouping sync 5.2 s, GDN recurrence 4.8 s. The PLE rows are free at this
  scale (97.6% row-cache hits).
- **Decode after such a prompt, and the fix.** Without lending cache slots to the prompt path (`--no-prefill-borrow`,
  which the port used from the start) the reserve evicts ~1,900 experts for good and decode after the prompt runs at
  2 tok/s. Borrowing lends ~940 slots to the prompt path and refills them in about a second afterwards: 80,000
  tokens in 75 s = **1,062 tok/s, decode 35-40 tok/s** after it, at `--vram-reserve-mib 2048 --prefill 4096`.
  It costs ~1 s on short prompts (2,184 tokens: 610 vs 792 tok/s), so the port borrows by default only above a
  32K context (`--prefill-borrow` / `--no-prefill-borrow` decide explicitly).

**Long contexts (2026-10-01, engine 0.1.32-sycl, setup's flags: `--kv-resident 32768 --vram-reserve-mib 2048
--prefill 4096`, the prompt path borrowing cache slots - the default above 32K).** Same text repeated to length,
then 256 greedy tokens:

| context | KV | prompt | decode after | peak VRAM | experts in the host mirror |
|---|---|---|---|---|---|
| 128K | int8 | 888 tok/s (144 s) | **66.6 tok/s** | 29.9 GB | none |
| 128K | q4_0 | 848 tok/s (151 s) | 62.8 tok/s | 29.7 GB | none |
| 128K | k8v4, KV in VRAM | **1,006 tok/s** (127 s) | **66.8 tok/s** | 30.7 GB | 103 (0.2 GB) |
| 256K | int8 | 757 tok/s (338 s) | **55.9 tok/s** | 30.2 GB | none |
| 256K | q4_0 | 737 tok/s (347 s) | 51.7 tok/s | 30.0 GB | none |
| 256K | k8v4, KV in VRAM | 823 tok/s (311 s) | 52.2 tok/s | 30.9 GB | 865 (1.7 GB) |

Borrowing keeps every expert in VRAM (the prompt buffers live in lent cache slots, refilled after the prompt).
Without it (`--no-prefill-borrow`) the buffers keep their own VRAM and ~1,000-2,300 experts move to the pinned host
mirror: the prompt is 2-5% faster, decode after it 10-23% slower (0.1.31 without borrowing: 128K int8 934 / 59.5,
q4_0 887 / 53.4, k8v4 1,026 / 54.4; 256K int8 784 / 48.7, q4_0 763 / 43.2, k8v4 838 / 42.3 tok/s).

KV streaming (`--kv-resident`) keeps the whole KV in pinned host memory and only the attended window in VRAM, so
the KV pushes no experts out; setup.py turns it on from 64K up and keeps INT8 (the faster of the
two at every size here). Without it the KV pushes experts out and decode after the prompt fell to 4-9 tok/s at
128K-256K (2026-09-30). `--kv k8v4` does not support streaming yet: setup keeps its KV in VRAM, which pushes up to
more experts out; those come from the pinned host mirror over PCIe (planned item 2), so decode holds at 42 tok/s at
256K (it was 3.8 before the mirror). k8v4 reads prompts fastest (no streaming copies) and works through the FP32
attention fallback (its dedicated prompt kernel is the XMX one below).

**XMX prompt attention v2.** 64-cell chunks, vector-packed K^T and V, hi+lo Q in one accumulator per scale group,
one accumulator update per chunk: 1.4-1.5x faster than v1 and correct in all modes (`qsa_prompt_attn_parity`,
and `kv_hybrid_parity` passes completely with it), but still ~2x slower than the FP32 fallback (13.4 vs 5.5 ms per
chunk, INT8, 32K context). This attention is gather-bound: each query position selects its own ~2,000 cells, so
the K/V fetch dominates and only 12 of the 16 matrix rows are real heads. Opt-in: `STRATA_PROMPT_ATTN_XMX=1`
(64-cell chunks, 120 KB of local memory) or `=32`.

**XMX v2 in the full matrix, and why a grouped kernel is not next (2026-09-30).** With `STRATA_PROMPT_ATTN_XMX=1`
(stage timing on, ~8-10% overhead) the long prompts run 23-33% slower than with the FP32 attention: 128K int8 666 vs
960 tok/s, k8v4 596 vs 983; 256K k8v4 506 vs 752, int8 536 vs 718 (q4_0 does not use the XMX kernel). The obvious
fix, gathering the union of neighbouring positions' cells once, depends on how much their selections overlap.
Measured on the last chunk of an 80K prompt (first QSA layer, `STRATA_DUMP_SEL=<file>`): the union of 8 consecutive
positions is 3.3x one position's 2,051 cells (16: 5.1x), and only 12% of a selection is shared by all 8. Grouping
would cut the K/V gather to ~40% but multiply the arithmetic by 3-5x: at best 5-8 s of an 80K prompt. Not built.

**Serving the port (2026-09-30).** `serve/server.py --engine strata` runs the SYCL engine unchanged through
`sycl/serve/strata-sycl.sh`, an `exe` that starts the binary inside the oneAPI runtime image with the serve pipes
attached (paths in the config's `args` are the container's, the data root mounted at `/work`). A config:

```json
{"engine": "strata", "exe": "<repo>/sycl/serve/strata-sycl.sh",
 "args": ["--pack", "/work/pack", "--native", "<shard 1>", "--ple-gguf", "<shard 2>",
          "--expert-profile", "data/expert-profile-coder.bin", "--expert-cache", "auto", "--stream-experts",
          "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/work/mtp/rt",
          "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024"],
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}, ...}
```

Measured through the OpenAI API with those sampling defaults: 36 tok/s decode on a first turn, 33 on a follow-up
(which reuses the conversation's cached prompt), against 24-26 for llama.cpp on the same card. After the decode work
(2026-10-01 build): 48-72 tok/s on a ~190-token answer, prompt included (the first request after a start is the slower
one); 55-59 tok/s over 5,000-18,000-token answers on the 2026-09-30 build. The reserve matters:
with `--stream-experts` there is no host copy of the experts, so any expert left out of VRAM is read from the SSD
and computed on the CPU whenever it is routed. At 1,536 MiB the cache came up 128 experts short and decode fell to
5-10 tok/s; 1,024 MiB fits all 12,288 with 2 GB of VRAM still free.

**Logprobs (2026-10-02).** `/v1/chat/completions` takes OpenAI's `"logprobs": true` and `"top_logprobs": 0..20`,
streamed or not, through `sycl/serve/server_intel.py`; the reply carries `choices[0].logprobs.content[]` (`token`,
`logprob`, `bytes`, `top_logprobs`), and with thinking on the thinking tokens under `reasoning_content`. The engine
(the port only) gets `logprobs=K` on its GEN line and writes an `LP logprob id:logprob ...` line after each `T`
line, from the verify window's head logits - the distribution the token was taken from, before temperature,
penalties and sampling. Upstream's server ignores the lines, and an engine never asked writes none. Each token
costs one 1 MB logits row copy when asked. Measured (IQ2_XS, "capital of France"): "Paris" at -0.0019 (p 0.998), the
same answer and no measurable time without it. Not on `/v1/completions` or the Anthropic endpoint yet.

**The Monitor tab on Intel.** `sycl/serve/server_intel.py` is `serve/server.py` with two additions made at run
time:

- **GPU readings.** When NVML has no card, it plugs `sycl/serve/xe_telemetry.py` into `serve/telemetry.py`'s
  `gpu_reader`. That reads the Arc through sysfs (temperature, power and its cap, the PCIe link the card
  trained at). Load and VRAM come from `/run/gpustat.json`: xe reports them per client in `/proc/*/fdinfo`,
  which only root can read, so a small root sampler writes them (`sycl/tools/gpustat.py`; install it with its
  `gpustat.service` as its header says).
- **A Model menu.** A run config may name a `model_switcher` (an RPC taking `{"mode": m}`) for a host that swaps
  models on one card. The web app's header then gets a Model menu (`sycl/serve/web/switcher.js`, injected into
  the page).

**Speed depends on the text.** Decode with speculative decoding tracks how often the draft layer guesses right.
The served engine at 32K context: 45-51 tok/s on the test prompt (continuing a Fibonacci function, 77-85%
of drafts accepted), 30-39 tok/s on a chat answer with prose (55-72%); sampling and the repetition penalty cost
nothing measurable. The engine used to abort at exit in serve mode (a queue wait in a destructor after the
runtime's teardown began); it now exits directly once its requests are done.

**Planned (2026-09-30), in order.** Ranked by payoff on this card; the first three were taken first.

1. Expert dot products on XMX in integer mode: a decode window (up to 6 tokens) fits one INT8 DPAS (1-8 rows), the
   i-quant grids decode to small integers, the activations are already INT8. Today: scalar dp4a, ALU-bound (77%).
   **Parked**: three `joint_matrix` versions (opt-in `STRATA_EXPERT_XMX=1`, do not enable) were 1.4x and 2-3x slower
   than dp4a, and the third (B filled in place with `joint_matrix_apply`) hung the GPU. At 1-6 rows the grid decode
   and the packed-B layout cost more than the DPAS saves.
2. Experts missing from VRAM read from pinned host memory over PCIe instead of the SSD (the 256K decode collapse).
   **Done**: at start-up every expert without a VRAM slot is read into a pinned host mirror (`STRATA_MIRROR_MIB`, by
   default free RAM less 4 GiB), and the device-built verify plan points the expert kernels straight at it. Before,
   that plan dropped non-resident experts, so output with misses was wrong. Forced test (`--expert-cache 8000`,
   1,879 experts / 3.6 GiB out of VRAM): decode 2.6 -> 40.9 tok/s (full cache 43.3), prompt 451 -> 640 tok/s
   (776), output tokens identical to the full-cache run. About 3.6 GiB of VRAM can go to context for ~6% of decode.
3. KV streaming (`--kv-resident`) from 64K up: only the attended window of the KV in VRAM. **Done**: 256K decodes at
   31 tok/s (from 4-5).
4. QSA block selection on XMX: every query against every pooled block, a dense product that grows with the context.
5. The hot decode kernels re-tuned for Xe2's native 16-wide sub-groups (twice the registers per thread).
   **Tried, no gain**: SIMD16 builds of the multi-column mmvq and the wide Q6_K kernel (`STRATA_MMVQ_SG`: 16 all,
   1 IQ4_XS only, 2 short outputs only; default 32). Alone (`mmvq_sg_bench`) SIMD16 is up to 1.45x faster on IQ4_XS
   and 1.3x on the 640-row shared-expert projections, 10-25% slower on the large K-quant ones; in the engine no
   setting beats SIMD32 beyond run-to-run noise (SIMD32 itself lands at 39 or 45 tok/s). Output tokens identical.
   The same bench shows the real headroom: the dense decode kernels stream 100-280 GB/s of a 608 GB/s card
   (Q6_K, the most common type, ~146 GB/s at 2-4 columns).
   **Found and fixed: misaligned loads.** A Q6_K block is 210 bytes, so every block's `ql`/`qh` runs start only
   2-byte aligned, and the B70 splits a misaligned 16-byte load into pieces. The same weights repacked at a
   224-byte stride ran 2.3-4.7x faster (`q6k_align_bench`). Without changing the layout, the wide Q6_K kernel now
   does two aligned 16-byte loads and a shift (`load16_a2`; both stay inside the block) and takes every Q6_K
   shape and window width (it used to be gated to n_out >= 4096, 1-4 columns): 2560->10240 at 1/2/4/6 columns
   160/145/141/101 -> 493/432/331/273 GB/s, the 248K-row head 150 -> 407 GB/s at 1 column. Output tokens
   identical. **Coder decode 44.9 -> 54 tok/s** (2k prompt, greedy). `STRATA_MMVQ_A2=0` restores the old path.
   Still misaligned: IQ4_XS (136 B, 8-aligned, ~100 GB/s), IQ4_NL (18 B), Q8_0 (34 B); Q4_K/Q5_K are aligned
   but use 4-byte loads in the multi kernel (230-280 GB/s).

**Decode round 2 (2026-09-30), from a unitrace of 256 decoded tokens.** Each step measured on its own, 19-token prompt,
256 tokens, two runs each, output tokens identical throughout:

| | decode |
|---|---|
| after the Q6_K alignment fix | 53.9 tok/s |
| + `resident_plan` grouping in parallel (thread 0 alone took 73 us per layer, ~3.5 ms a round) | 57.1 |
| + the GR down kernel reads the activations directly (the 60 KB SLM tile per group capped occupancy) | 58.7 |
| + wide 16-byte-load kernels for Q4_K, Q5_K, IQ4_XS (codebook in registers) | 62.6 |
| + wide kernels for Q8_0 (MTP dense, shexp down of layer 0) and IQ4_NL (shexp down); Q6_K's activation loads aligned | 65.6 |
| + the GR down projection sliced by columns (each group stages one 128-column slice of xn, 4 lanes per row) + a fixed-order reduce | 70.6 |
| + IQ4_NL expert dots (the down projection in 39 of 48 layers): codebook in registers, weights as aligned 4-byte loads, bitwise the same ints (2026-10-01) | 76.1 |
| + GR norm per (token, stream), bitwise the same; aligned weight loads in the IQ3_XXS/IQ3_S/IQ2_S gate/up dots | **77.9** |

The last step changes the output: identical for 146 tokens, then a near-tie after a comma goes the other way (the new
kernels sum in a different order). Q8_0 kernels 4-6x (189 -> 31 us at 2560 x 10240, 2 columns), IQ4_NL 640 -> 2560
28 -> 6.6 us. The aligned-load helper only loads its second chunk when the address is unaligned, so it never reads a
16-byte chunk without a needed byte and cannot cross a page at the end of an allocation. `STRATA_MMVQ_WIDE_32=0`
restores the old Q8_0/IQ4_NL kernels. The sliced GR down kernel (GR read 108.6 -> 76.5 us at 6 tokens, sums equal to
~1e-7) flips that near-tie back: its output is the original one. `STRATA_GR_DOWN_SLICED=0` restores the direct kernel.

Kernel level: IQ4_XS 107 -> 323 GB/s at 2 columns (61 -> 212 at 6), Q4_K/Q5_K 1.3-1.6x; the GR read 135 -> 110 us at
6 tokens (bitwise equal). Switches to the old paths: `STRATA_PLAN_PARALLEL=0`, `STRATA_GR_DOWN_DIRECT=0`,
`STRATA_MMVQ_WIDE_K=0`. (`iq_parity` reports 10 "missing fixture" failures with and without the change: the oracle
fixtures are not in the port's tree.)

`STRATA_IQ3S_WIDE=1` (opt-in, 2026-10-10) reads the dense IQ3_S matrices (attn_q, attn_gate, ssm_out, attn_k, the
shared experts' gate/up) with a wide kernel shaped like the Q4_K..IQ4_XS ones. unitrace, B70, IQ3_XXS, one card, T=1:
worst attn_k 77 -> 123 GB/s, best attn_q 114 -> 307 GB/s; these matrices take 0.60 ms of kernel time per token
instead of 1.36 ms. The greedy output parts from the default at token 198 of 256 (the sub-blocks are summed in
another order). Measured again for 0.1.45 (B70, IQ3_S, stock-like config, 5 interleaved pairs): decode 33.39 -> 33.29 tok/s
(-0.3%), 4K prompt 40.71 -> 40.79, so it stays opt-in; the gain the author reports is on two cards.

**Q6_K decode kernels (0.1.45, #2340 leadgtr7, #2207 Andy Slater).** The Q6_K wide kernel took its six-bit offset (v - 32)
through a per-byte saturating subtract; it is now four integer ops (bitwise equal), and the Q4_K / Q5_K / IQ4_XS / Q6_K
kernels load the Q8_1 activations as one dword-vector. No switch. mmvq_bench Q6_K, B70: 2560x10240 1 column 44.7 -> 26.5 us,
4 columns 64.5 -> 46.2; A750: 2560x10240 4 columns 251 -> 102 us. End to end on the B70 (IQ3_S, 5 interleaved pairs, answers
identical): decode 32.24 -> 33.37 tok/s on a story prompt (+3.5%), 39.11 -> 40.77 after a 4K prompt (+4.2%), prompt rate unchanged.
`STRATA_MMVQ_LARGE_GRF` (default on for Arc A-series only, `=1` forces it on, `=0` off) launches the 2-8 column wide kernels with
256 registers per thread, where they spilled at 128 (the same bytes are produced either way). `STRATA_Q6_PLANAR` (default on,
`=0` off) converts the Q6_K output head and the MTP draft head to ql / qh / scales / d planes at load (same bytes, no VRAM change,
bitwise equal): 2560x248320 on a B70 800 -> 662 us (1 column), 1230 -> 1020 us (4 columns); end to end it adds +0.3% on the B70
(33.16 -> 33.25 tok/s, within noise) on top of the kernel changes above.

**Two-speed runs, explained (2026-09-30).** Identical greedy runs decode at either ~45 or ~39 tok/s. A per-gather
trace of the PLE reader (`STRATA_PLE_TRACE=1`) shows the slow runs pay one 226 ms PLE read stall in the first
decode round, after the window graphs are captured; every other read and round matches the fast runs. Prompt time
plus decode time is the same in both modes (4.95-5.17 s): the stall lands either in the prompt's PLE wait or in the
first decode round, so it is a once-per-process cost, not lost throughput. Ruled out: NVMe APST, the I/O scheduler
(`none`), CPU starvation (93% idle during decode), I/O thread count. Compare runs on time to first token + decode.
6. INT8 prompt GEMMs: experts dequantized to INT8, oneMKL/oneDNN INT8 on XMX (half the dequant bytes, 2x rate).
7. Fewer graph nodes per decode round (~2,500 at ~5 us): norm+rope, scores+top-k, gate+quantize fused.
8. Wider speculation (two draft branches per verify window): the kernels are latency-bound, so it is nearly free.
9. A model bigger than VRAM: the original Qwen3.8-Flash-Next IQ2_XS (35.5 GB of experts; the same path suits Q2_0
   and Swift 1.5) on the B70 with 23 GB of RAM. ~24 GB of experts in VRAM, the rest (~10-12 GB) in the pinned host
   mirror the device plan reads over PCIe (item 2). The port has the IQ2_XS/IQ3_XXS/Q2_0 expert kernels. Open: whether
   a 10+ GB pinned mirror fits beside everything else in 23 GB, and decode with that share of experts on a Gen3 x8
   link (3.6 GB mirrored measured 40.9 tok/s; expect less). Needs the 68 GB download.
   **Done (2026-10-01)**: shard 1 downloaded (39.2 GB; shard 2 is byte-identical to the Coder's, so a hard link),
   a native pack (`tools/iq_pack.py`, 6 s), the original model's expert profile, the same MTP draft layer. 18,329 of
   24,576 experts in VRAM (24.6 GiB), 6,247 in the pinned host mirror (8.4 GiB, 9 s to fill), host RAM never below
   12 GB free. **Decode 50.8 tok/s** on the 19-token prompt and **60.6** after 2,184 tokens, prompt 549 tok/s;
   coherent, correct answers on both. Q2_0 and Swift 1.5 (similar size) should behave the same; IQ3_XXS/IQ3_S
   (43-50 GB of experts) would need 19-26 GB mirrored, more than 23 GB of RAM allows.

**Keeping up with upstream.** A merge of upstream `main` into `b70` leaves the copies in `sycl/` behind
wherever upstream touched a file they mirror. They are refreshed by re-migration, not by hand (done for
0.1.25-0.1.27, 2026-09-30):

1. Merge upstream into `b70`. No shared file should conflict: the Intel code is all in `sycl/`. Then check that
   `sycl/setup_intel.py --check` and `sycl/serve/server_intel.py --engine mock` still run against the new
   setup.py and server.py.
2. Migrate the *old* upstream tree (a `git archive` of the pre-merge commit) with `migrate.sh` into
   `sycl-base`, and the merged tree into `sycl-new`; about 3 minutes each in the dev image.
3. `sycl/tools/normalize.sh <dir> <commit>` on both: dpct's file names, the unchanged files, the
   message serials, then `fixups.py`. Two runs of dpct on the same source now differ only in the kernel
   name hashes it generates.
4. For every file present in both: `git merge-file sycl/F sycl-base/F sycl-new/F`. Files upstream did not
   change are left alone. Copies dpct never produced (`verify.cpp`, `mtp.cpp`, `remote_experts.cpp`
   include no CUDA header directly) take upstream's diff by hand; new parity tests are copied in and
   listed in `sycl/CMakeLists.txt`.
5. A fixup whose pattern upstream changed shows up as a compile error (the PTX gate became an `#elif`
   under upstream's `__HIPCC__` guard); extend the fixup, re-run it, rebuild.

**The 0.1.31 merge (2026-10-01).** Upstream 0.1.29 -> 0.1.31 (132 commits) by the steps above: 45 migrated files
changed for real (27 more only in dpct's kernel-name hashes), 38 conflicts resolved, verify.cpp/mtp.cpp/
conversation_state.cpp ported by hand. A symbol audit (the port's feature identifiers counted before and after) is
part of the procedure now: it caught a dropped mirror hook and four doorbell waits that had lost their spin bound
(upstream renamed the loops; `fixups.py` covers both spellings). What the merge needed beyond conflicts:

- **The expert kernels.** Upstream sends the i-quant experts to new multi kernels (one warp per row, 8 rows a
  group); the port's grid is sized for its own kernels (8 lanes a row, 32 rows a group), so only a quarter of each
  expert's rows were written and the output was end-of-text tokens. The port's kernels stay the default (they also
  serve upstream's new Q4_K/Q5_K/Q5_1/Q8_0 experts); `STRATA_EXPERT_SPLIT=1` runs upstream's, with their grid.
- **Prompt-slot borrowing hung** in the first chunk of any prompt with it on (40K, 128K; GPU busy) - fixed after
  the 0.1.32 merge, see "Prompt-slot borrowing: the hang" below.
- **Upstream's new tests found three dpct mistranslations from the first migration.** dpct writes `__fadd_rn(a, b)`
  as `a + b` without parentheses, so `__fadd_rn(sum, c ? x : y)` became `sum + c ? x : y` (replaced, not added): in
  the native QSA indexer's per-token pooled key (the decode path: blocks completed while generating got wrong
  pooled keys; prompt blocks come from the batched kernel and were right), in the native QSA scores (an opt-in
  path) and in the s2 activation rounding (every value 0; the Q2_0 s2 pack, unused here). Fixed, and `fixups.py`
  now parenthesises. `qsa_parity`'s batch-vs-sequential test passes; decode output after long prompts changed.
- **Upstream's new kernels against the port's on the B70** (decode tok/s, Coder 19 / 2,184-token prompts, IQ2_XS
  19): port defaults 78.1 / 75.7 / 58.6; `STRATA_EXPERT_SPLIT=1` 65.6-69.5 / 67.2 / 49.1 (-11 to -16%);
  `STRATA_GR_V3=1` 73.2 / 72.4 (-4 to -6%). Both stay opt-in.
- **Lanes per row of the port's expert kernels** (`STRATA_GU_LANES` / `STRATA_DOWN_LANES`, 4/8/16/32 at run time):
  gate/up 8 or 16 x down 4 or 8 all decode in 37.7-38.6 ms per verify round (2 runs each, Coder, both prompts); tok/s
  differences between them are draft acceptance (each split rounds differently). The default (8 / 8) stays.
- **Speed after the merge:** Coder 78.2 / 75.7 tok/s (as before), IQ2_XS 58.6 / 64.2 (from 50.8 / 60.6).
- **Still failing:** (`iq_multi_parity` on IQ2_XS: fixed later, see "Two model-specific bugs"), `s2_expert_grouped_parity` (the s2 path), `kv_hybrid_parity`'s last step
  (needs the unported tensor-core prompt kernel). `ple_parity` and `native_expert_parity` need model files.

**The 0.1.32 merge (2026-10-01).** Upstream 0.1.31 -> 0.1.32 (89 commits) by the same steps; the hash-only
differences are now canonicalized before `git merge-file` (the port's kernel-name hashes are kept), which left 10
files with real conflicts. What it needed:

- **Upstream's async commit** (`set_commit_async` / `wait_commit`) maps onto the port's own deferred commit
  (`commit(n, err, false)` + `commit_finish`); `wait_commit` is `commit_finish`.
- **Upstream's new hyper-connection read variants** (split / staged, chosen per card by a bit-for-bit self-test at
  start) are not used by the port, which keeps its sliced down / split norm read; the self-test segfaulted on the
  B70, so on SYCL it runs only with `STRATA_HC_CHECK=1` (open). The port's split-norm kernel was renamed
  (`gr_norm_split_port_kernel`): upstream now has one of the same name.
- Two kernel names collided after hash canonicalization (renamed), and fused_gr's per-block shared-memory query is
  a fixup now.
- **Result:** every output identical to 0.1.31 (Coder 19 / 2,184-token prompts and IQ2_XS, 256 greedy tokens),
  same speeds (Coder 77.7-78.1 / 75.7 tok/s, IQ2_XS 58.5), 40K prompt 1,201 tok/s then 65.1 tok/s decode.
  `kv_hybrid_parity` and `qsa_prompt_attn_parity` pass (the tests now turn on the XMX prompt attention they
  check); `iq_multi_parity` (IQ2_XS) and `s2_expert_grouped_parity` as before.

**The 0.1.33 merge (2026-10-01).** Upstream 0.1.32 -> 0.1.33 (17 commits): no shared file conflicted (the Intel
code is all in `sycl/`); six engine files changed upstream and were merged by the same re-migration (only those six:
the other files dpct produced differently were left alone). Two conflicts: `--resident-cpu-experts` (upstream's new
`resident_cpu_explicit` beside the port's `--stream-experts`) and the prompt attention's compute-capability check
(the port keeps its XMX dispatch). `sycl/setup_intel.py` and `sycl/serve/server_intel.py` ran unchanged against the
new setup.py and server.py. Every output identical (Coder, IQ2_XS, the 40K prompt with borrowing); `gr_parity` and
`qsa_prompt_attn_parity` pass.

**Prompt-slot borrowing: the hang (fixed 2026-10-01).** From 0.1.31 on, a prompt that borrowed cache slots stopped
in its first full chunk with the GPU at 100% and the host waiting on the compute queue. It was never a borrowing
bug: borrowing is just the only way the Coder streams experts during a prompt (every expert is resident otherwise),
and the streamed path's stager deadlocked. Its threads read experts into a ring of 16 page-locked buffers and, to
reuse one, waited on the copy queue's event for the DMA that last read it (the CUDA form, `cudaEventSynchronize`).
Under the Level Zero v2 adapter that event, once a host thread had waited on it, no longer released the other
queue's barrier that also listed it, and the GPU waited forever - only once a layer streamed more than 16 experts
(the ring wrapped). The v1 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) and a 256-buffer ring both ran; so did a sync
after every phase (`STRATA_PREFILL_SYNC=1`, kept as a debug switch). The fix: the copy queue writes a sequence
number into page-locked host memory after each DMA and the stager polls it - no host thread waits on a queue's
event. Output is bit-identical to the run without borrowing; borrowing is the default above 32K again.

**Two model-specific bugs found with Swift 1.5 (fixed 2026-10-01).** UkisAI's Swift 1.5 IQ2_XS (setup's `swift`
family, the same GSQ-RCO layout) decoded token 0 forever: NaN logits from layer 13 on.

- **`--stream-experts` read the wrong shard.** Since upstream 0.1.31, `ExpertLayout::gguf_file` is per layer AND role
  (`3 * layer + role`; a shard boundary can fall inside a layer). The port's `GgufExpertSource` still indexed it by
  layer, so a layer in shard 2 was read from shard 1 at shard 2's offsets: garbage IQ1_M scales, infinities after the
  fp16 dequantization. The original model keeps every expert in shard 1 and never noticed; Swift's layers 13-47 are
  in shard 2. `STRATA_DBG_NAN=1` now also reports the experts' fp16 GEMM inputs (activations, dequantized gate/up).
- **IQ2_XS products were garbage** (the long-failing `iq_multi_parity` IQ2_XS case): ggml's `vec_dot_iq2_xs` reads
  its four 16-bit codes through a `uint16_t*` to an `int2`, type punning the SYCL device compiler does not honour;
  the dequantizer, which reads the codes directly, was exact. The codes are extracted by shifts now (the dot and the
  expert kernels' `Split<17>`); the test checks against a CPU decode from ggml's tables too and passes. No model here
  uses IQ2_XS tensors, Swift included - this was not Swift's bug, only found on the way.

Swift 1.5 IQ2_XS after both: correct answers, 51-118 words of thinking on short questions, 48 tok/s raw decode.

**Load time (2026-10-02).** Starting a model is mostly the expert cache's fill from the GGUF (`--stream-experts`):
the Coder's 23.4 GiB of experts took 64-76 s, the IQ2_XS's 24.9 GiB 91 s, because the fill read each expert's
three slices, copied the blob and waited for the copy before the next read. It is a pipeline now: the slots are
admitted in profile order first (the same placement), the reads go in file order by up to 8 threads into
page-locked batches of 64, and a batch's copies run while the next one is read. Cold page cache: 76.2 s -> 18.8 s
(0.33 -> 1.34 GB/s); a whole Coder start from the engine's launch to its first token 82 s -> 26 s, the IQ2_XS
120 s -> 41 s (its 8.2 GB host mirror is ~9 s of the rest). Output identical. `STRATA_FILL_SERIAL=1` is the old fill.

**The stream-all prompt walk is back (2026-10-03).** It copies every expert the VRAM cache does not hold, layer by
layer ahead of the compute, instead of only the routed ones. It hung on the B70 until the stager and PLE fixes of
2026-10-02 and runs since. It now runs by default when the VRAM holds more than 90% of the (layer, expert) pairs;
past that it would copy several times the routed experts (the IQ2_XS keeps a quarter of them in the RAM mirror: a
2,184-token prompt fell from ~560 to 254 tok/s with it). `STRATA_PREFILL_STREAM_ALL=1` / `=0` force it on or off.
Coder, outputs identical either way:

| prompt | routed-only | stream-all |
|---|---|---|
| 40K int8 | 1,106 tok/s | 1,160 |
| 128K int8, KV streaming | 897 | 920 |
| 128K k8v4 | 1,006 | 1,041 |
| 256K int8, KV streaming | 760 | 780 |

Measured alongside: the 2,184-token prompt reads 784-799 tok/s on both the 0.1.35+PR and the 0.1.38 build (three
runs each), so the 825 seen once on 0.1.35 was session variance, not a regression. The prompt path at 40K (Coder,
GPU timeline): expert dequant 15.7%, expert GEMMs 23.4%, host grouping 6.5%, gather 5.7%, combine 2.4%, prompt
attention 19.2%, QSA select 6.8%, DeltaNet recurrence 5.8%, hyper-connection reads 4.6%. Upstream's sm_90 cluster
kernels (greedy sampler, QSA top-k) would save next to nothing here: the sampler reads ~6 MB a round against a 13 ms
token. Q4_0 tensor-core prompt attention (#452) is not worth porting: the port's XMX prompt attention is 2-3x slower
than the plain kernel it uses for every KV format.

**The 0.1.38 merge (2026-10-03).** Upstream 0.1.35 -> 0.1.38 (83 commits) merged the seven PRs the port had taken
early, so their header forks in `sycl/include` are gone; the merge base for the 3-way merge was the port's own "main +
PRs" migration. New in the port with it: the DeltaNet output norm without its dead FP32 store, two prompt-path
fixes (the fused layout's buffers, the streamed walk's resident lookup), `STRATA_GR_DOWN_MAX4=1` (opt-in; neutral on
the B70: 2,184-token prompt 793 vs 788 tok/s, decode 76.2 vs 76.3). Stubbed in the port: `--peer-device` (a second
GPU as an expert-cache tier: upstream's code is CUDA calls; `open` refuses with a message) and the fused int8 prompt
kernels (#136, part of the MMQ library the port does not build). Off on SYCL: the sm_90 thread-block-cluster greedy
sampler and QSA top-k (no SYCL counterpart; the callers take the plain kernels). `STRATA_GDN_KEYHEAD=1` stays opt-in
(still 8% slower here: 727 vs 788 tok/s). `qsa_prompt_attn_parity` now also tests upstream's Q4_0 tensor-core mode
(#452), which the port does not have: those cases report "refused" (the int8 and fp16 XMX cases pass); a `--kv
q4_0` prompt takes the older kernel, as before. Outputs identical (Coder, IQ2_XS, 40K); 128K int8 KV streaming
892 / 66.8 tok/s.

**0.1.35 and seven open upstream PRs (2026-10-02).** Upstream main 0.1.33 -> 0.1.35 merged as before, then seven open
PRs ported into `sycl/` ahead of upstream (one dpct run of main + all of them, 3-way merged into only the files they
touch; `sycl/tools/merge_upstream.py`). Their header changes live in `sycl/include` until upstream merges them.

| PR | what | on the B70 |
|---|---|---|
| #385 (sergqwer) | stager: a buffer's first job of a generation waits for the previous DMA from it | race fix; same output |
| #463 (constantindjonkam) | decode waits for an adaptive swap before reading the residency table | determinism fix |
| #453 (architectds) | the drafter's batched K/V on the KV-streaming ring | 128K int8: 888 -> 895 tok/s prompt, 66.6 -> 67.0 decode |
| #374 (sergqwer) | the first chunk's PLE rows read beside layer 0 | part of the 2,184-token prompt's 784 -> 825 tok/s |
| #363 (BlueKingMuch) | the PCIe expert call: group stride, fused SwiGLU + q8_1 | IQ2_XS decode +1.7%; re-done on the port's lane kernels |
| #407 (sergqwer) | `--adapt-tuned` (opt-in) | neutral here; stays opt-in |
| #413 (BlueKingMuch) | DeltaNet recurrence per key head | bit-identical but 8% slower prompt here: off (`STRATA_GDN_KEYHEAD=1`) |

#374 needed two port fixes: its next-chunk read assumed every chunk is the full chunk length (the port's first chunk
is 256 tokens: the second chunk's rows were read from the wrong place), and its host wait on the PLE upload's event,
now inside the layer loop, deadlocked the prompt past ~4K tokens under the Level Zero v2 adapter (the stager's bug
again): the upload is marked by a polled sequence number now. #413's gate is an NVIDIA SM-count rule and its parity
test an SM-holding NVIDIA bench (not built). Outputs identical to 0.1.33 (Coder 19 / 2,184 tokens, IQ2_XS, 40K).

**Not ported yet (2026-10-01).**

- Three kernels carry inline PTX (`mma.sync` tensor-core matrix ops, `ldmatrix`, `cp.async`):
  `qsa_prompt_attn`, `qsa_select`'s block scores, `native_qsa_score`. The SYCL build takes the "older card"
  fallback the CUDA build uses below sm_80. `qsa_prompt_attn` also has an XMX version (`joint_matrix`, opt-in
  `STRATA_PROMPT_ATTN_XMX=1`): correct, but slower than the fallback (see "XMX prompt attention v2").
- The ggml MMQ prefill path (`moe_mmq.cu`) needs llama.cpp's ggml-cuda sources; not built.
- AOT device code is what runs: `AOT=bmg-g31 BUILD_DIR=.../build-sycl-aot` (the JIT build costs ~47 s of
  compiling on the first window).

## Arc A750 and the other Alchemist cards (`i915`, 2026-10-07)

Measured on an Arc A750 (8 GB, `i915`, PCIe 4.0) with the Flash-Next IQ3_XXS in a PC with 64 GB of RAM, a Ryzen 5 5600X
(AVX2, no AVX-512). `./setup.sh --backend sycl` writes a different config for an `i915` card (and a 300 MiB VRAM reserve and
`--draft-vocab en` for any card under 12 GB):

| | Arc Pro B70 (`xe`, 32 GB) | Arc A750 (`i915`, 8 GB) |
|---|---|---|
| experts | `--stream-experts`: from the GGUF into VRAM, the rest in a pinned RAM mirror | no `--stream-experts`: all of them in a RAM arena (39.97 GiB for IQ3_XXS), 406 of 24,576 in VRAM, the CPU computes the others |
| `STRATA_VERIFY_NO_HOST` | 1 (the wrapper's default) | 0, written to the config's `"env"` |
| `--vram-reserve-mib` | 1024 / 2048 | 300 |
| `--ple-io` | `ram` (rotational-disk rule of setup) | `ram` |
| FP64 | native | emulated: `IGC_EnableDPEmulation=1`, `OverrideDefaultFP64Settings=1`, `NEOReadDebugKeys=1` in `"env"` |
| build | AOT `bmg-g31` | JIT (no `AOT=`) |

- **Why no `--stream-experts`.** The mirror is one pinned host allocation for everything the card does not hold (40 GB here). A
  single `sycl::malloc_host` above a few GB fails on the A750 (`STRATA_MIRROR_MIB` 3,000 worked, 8,192 and 24,000 did not, also
  with `memlock` unlimited and the relaxed-allocation-limit variables), and the engine then refuses to start with
  `24170 experts are neither in VRAM nor mirrored`. Without the flag the engine loads the GGUF's experts into an ordinary RAM
  arena and the CPU computes the experts that are not in the 0.67 GiB cache. That needs the RAM: about 43 GB for IQ3_XXS plus
  the 29 GB n-gram table with `--ple-io ram` (the table is not locked in the container, so it is reclaimable; do not give the
  container `--ulimit memlock=-1`, the locked table and the arena then do not fit in 64 GB and the PC swaps). Setup warns when the RAM is less than the model's.
- **Why `STRATA_VERIFY_NO_HOST=0`.** With it set the GPU plans each layer itself and the CPU's experts are never asked for:
  the window hangs (the i915 log says `Fence expiration time out`) or, if the pool is refused, the engine says
  `REFUSED: ... neither in VRAM nor mirrored`. It is the switch of the B70's case, where every expert the card lacks is in the mirror.
  The engine takes a value of `0` (or empty) as "off", and `strata-sycl.sh` does not pass such a value on.
- **FP64.** An Alchemist has no FP64 hardware. One kernel on the sampled path declares `double`, and the SYCL runtime refuses it at
  its first launch (`'double' is not supported in 'Intel(R) Arc(TM) A750 Graphics' device`, the engine dies on the first request
  that has a temperature). With the three variables above the driver emulates it; the cost is the sampler's tail
  (see "FP64." in the B70 section).
- **The first request after a start used to return `!!!!!` (token 0) or crash the engine.** The GPU waits for the CPU's experts at
  every layer in a spin that is bounded (`strata::kSpinMax`, `sycl/include/strata/sycl_doorbell.hpp`): a spin that never ends
  hangs the driver. The bound was 20,000 reads, a few tens of milliseconds, which is enough when nothing is waited for (the B70)
  and is not for a CPU layer in the first request (cold pages, 13 s for a 26-token prompt). The GPU gave up, went on with the
  experts' outputs missing, the next layers' routing was garbage (one expert chosen ten times for a token: `nt=10`) and the
  CPU pool wrote past its token arrays and died with a segmentation fault (exit code 139); when it did not die the logits were NaN
  and the sampler's answer was token 0. Later requests ran with warm pages and were right. The bound is now a build option,
  `STRATA_SYCL_SPIN_MAX` (CMake; `SPIN_MAX=` for `sycl/tools/build.sh`): 2,000,000 reads (a few seconds) unless the build is an
  AOT build for a `bmg` card, which keeps 20,000. Before the change 11 of 16 runs of the engine (three requests each, a 26-token prompt) died or gave
  token 0 in the first request; with 2,000,000 reads 4 of 4 were right in all three requests.
- **Kernels misread memory more than 4 GiB into an allocation (fixed in the build).** On an Arc A770 (16 GB, `xe`, Coder
  IQ1_M, `--stream-experts`, a 4.46 GiB expert cache) every prompt answered `!!!!` (token 0): the experts in the cache's
  last 0.46 GiB dequantized to NaN in the prompt path. A copy of those slots read back equal to the GGUF
  (`STRATA_VERIFY_ALL_SLOTS` passed) and a fresh device copy of the same bytes dequantized clean; only a kernel reading
  the slot in place got wrong bytes. In a JIT build IGC addresses a kernel's pointer arguments as a 32-bit offset from
  the allocation's base. The JIT build now passes `-ze-opt-greater-than-4GB-buffer-required` (`sycl/CMakeLists.txt`),
  which keeps the kernels on 64-bit addresses. After the change, prompts of 73 to 542 tokens had 0 non-finite values
  (`STRATA_DBG_NAN=1`) and the 542-token prompt answered right. The speed cost is not measured. The AOT (`bmg-*`) build is
  unchanged. oneMKL's own precompiled kernels do not get the option; the GEMM stages an operand past 4 GiB for that.
- **Speed.** About 10 to 15 tok/s decode (76 to 92% of the drafts accepted), a 26-token first prompt in 13 s and later short prompts in 0.2 to
  1.3 s; the A750's PCIe link probes at 10.6 GB/s.
- **A750 and the xe error counters.** The engine segfaults in a worker thread of the CPU pool when it exits (dmesg only, the server
  has already printed "stopped"); it is not a GPU event.
- **The prompt's GDN recurrence runs a sub-group kernel on Alchemist** (`sycl/include/strata/prefill/gdn_xe.hpp`,
  variant 6 of `gdn_recurrence_variant`). It keeps the column kernel's four row groups and their FMA chains in the same
  order, but a 16-lane sub-group holds four columns x four row groups, so the two cross-row-group sums are sub-group
  shuffles instead of SLM round trips and a token needs one work-group barrier instead of five (256 GRFs a thread).
  Measured on an Arc A770 (16 GB, `xe`, oneAPI 2026.1 JIT), 2026-10-13: `gdn_parity --prefill-tokens 8192` 51.0 ->
  24.5 ms, 32,768 tokens 204-236 -> 110-113 ms (48 heads), 0 mismatches; end to end, Coder IQ1_M, `--max-context
  131072 --kv int8 --kv-resident 20480 --vram-reserve-mib 1200 --resident-experts`, prompt 386 -> 396-407 tok/s at
  8K and 478-480 -> 511-518 tok/s at 32K, the same output tokens. On by default on Alchemist only, the column kernels everywhere else (not measured there). `STRATA_GDN_XE=0`
  turns it off, `STRATA_GDN_XE=1` turns it on for any Intel GPU with 16-lane sub-groups. To check another Intel card:
  `STRATA_GDN_XE=1 ./gdn_parity --selftest --prefill-only` (or `ctest -R prefill_gdn_parity`), and `--prefill-tokens
  8192 --prefill-repeat 5` for the time of both kernels. Not taken: the same kernel with its state in SLM (no register
  spills, the same bits, ~4x slower: 13.7 vs 3.1-3.3 ms at T=1000).

### Prompt speed on Alchemist: the grouped XMX expert launch is the default (engine 0.1.45.3, `STRATA_PF_XMX`)

The A750's prompt path decodes and multiplies a group of experts in one fused launch on the XMX units (`STRATA_PF_XMX=2`, `sycl/src/prefill/xmx_moe.dp.cpp`)
instead of dequantizing each expert to FP16 and calling oneMKL. It is the default on Alchemist cards; `STRATA_PF_XMX=0` turns it off and `=1` is the
variant that dequantizes first and runs one grouped XMX GEMM per product. Battlemage keeps the old path (the same kernels are 7-11% slower there).
Measured on the A750 (IQ3_XXS, default flags, interleaved rounds against the run before): a 1K-token prompt 35.1 -> 46.4 tok/s, 2K 61.7 -> 83.1, 4K 79.3 -> 110.4
(+39%, 5 of 5 rounds), 6K 66.3 -> 114.8; decode unchanged (18.7 tok/s), the same number of cache slots. It rounds differently from the FP16 path:
teacher-forced KL against the default over 6,006 forced tokens 0.0002 (the default against itself: 0.0001; with the default PCIe share the A750's own run-to-run noise
is 0.012), first-token KL 0.0017 (floor 0.0005), top-1 agreement 99.7%. HumanEval (every 2nd, 82) 73, 69 and 74 for the mode and two default runs, reasoning 21 / 23 / 22 of
30, a 35-question Chinese/Japanese/Korean set 35/35 in all, needles at 2K-6K 9/9, an 8K-context tool-call agent 6/6.

### A750: a prompt that never finished is fixed (engine 0.1.45.3, commit `fca3c5958`)

On an Arc A750 some prompts never finished: with default flags a 2,744-token documentation excerpt hung in 6 of 6 runs (the
thread spun in `sched_yield` inside `libze_intel_gpu` with every engine idle), and the same hang was seen on a 5.3K-token needle prompt. The
cause: SYCL has no `cudaHostRegister`, so the expert arena is ordinary pageable memory even where the engine calls it
pinned. Copying a 2.3 MB expert straight out of it went through the Level Zero driver's staging path on the calling thread,
and on some prompts that call never returned. Now only experts in USM host memory (the `--stream-experts` pinned mirror)
are copied directly (the answer is remembered per 2 MiB); every other expert goes through the stager's pinned buffers, like
an unpinned one. **`STRATA_PF_DMA_PAGEABLE=1` restores the old direct copy.**

Checked: the 2,744-token repro finished 3 of 3 times after the fix (0 of 6 before); the first-token logits and the tokens are
identical old against new on the A750 (1.4K, 2.5K and 4K prompts) and on the Arc Pro B70 (stock config, and without
`--stream-experts`); speed is unchanged where it did not hang (A750 4K prompt 79.4 against 79.3 tok/s, B70 stock 1,246
against 1,240), and on the B70 without `--stream-experts` the prompt is faster (680-740 against 77-456 tok/s). The
regression test is `sycl/tools/test_prompt_deadlock.sh` (the 2,744-token prompt; it times out on the old build and passes on
the fix).

## Not done

- Images on the llama.cpp config: not wired. On the SYCL port they are wired, see "Pictures on an Arc" below. Open there: a
  nobody has run the SYCL picture reader on a card of 24 GB or more. Pictures are on for a Battlemage card too since 2026-10-22 (see "Default" below),
  with the CPU reader.
- Speculative decoding on the llama.cpp path (the GGUF carries no draft layer llama.cpp can use). The SYCL
  port has it (MTP draft layer, `--mtp`).

## Pictures on an Arc (2026-10-19, branch f-021-arcvis)

Pictures follow the rule of every other card ([MODELS.md](MODELS.md#the-versions)): a picture reader on the CPU under 24 GB of VRAM, on
the card from 24 GB. `strata_control/vision_reader.py` is the one place for it. On an Arc:

- **The CPU reader** is `strata-vision` (llama.cpp's mtmd). Setup compiles it into `engine-sycl/`, the server runs it on the host, and the
  engine in the container reads the picture's embeddings from the host's `/dev/shm`, which `strata-sycl.sh` shares
  (`STRATA_SYCL_SHM=0` keeps a private one). On the A750 a 448x448 picture takes 1.73 s.
- **The SYCL reader** is `strata-vision-sycl`, the same program on llama.cpp's SYCL backend (`tools/vision`, `STRATA_VISION_SYCL=ON`).
  `setup --backend sycl --vision gpu` compiles it on the host when the oneAPI compiler (`icx`, `icpx`, `intel-oneapi-mkl-devel`) is
  installed; without it setup says so and keeps the CPU reader. It needs the PC's oneAPI runtime, not the container. If it does not
  start, the server falls back to the CPU reader with one log line. On the A750: 0.16 s per picture instead of 1.73 s (after a 30 s
  first start that compiles the kernels), and the embeddings match the CPU reader's (cosine 0.9959 to 0.9999 over 8 pictures). It takes
  VRAM the experts would use, which is why the rule gives it to 24 GB cards only.
- **Default.** On for every Arc card with the CPU reader, a B-series (Battlemage) card included. An Arc of 24 GB or more also keeps the CPU reader by
  default: the SYCL reader has not run on such a card, so `--vision gpu` asks for it and `vision_reader.INTEL_GPU_READER_BY_DEFAULT` is the switch to
  flip once one has. **B-series cards are supported and untested by us** (the owner decision of 2026-10-22: users will report failures). One known
  risk: the first picture on the Arc Pro B70 (engine 0.1.45, IQ3_S, 2026-10-19, CPU reader) logged an xe engine reset (ccs + bcs) and
  `Fault response: Unsuccessful -EINVAL`, and the request ended 503. The CPU reader was in use, so the fault is in the engine on xe, not in a GPU
  reader; it has not been looked at since. It is not a refusal: setup and `strata launch` print one plain line that the card is new to Strata and
  how to report a problem, and `--vision no` (setup) or `--vision off` (`strata launch`) turns pictures off.

Tested on the A750 (IQ3_XXS, 8K context, engine from f-021-int, JIT build, 2026-10-19): a picture with a red circle, a blue square and the
words "STRATA 42", 8 variants, answered correctly by the CPU reader 10 of 10 (a first set with smaller text read "42" as "412"), and
by the SYCL reader 10 of 10; two pictures in one request, a picture kept in the history of a second turn, and a long text prompt after the
pictures all answered; a text request after every picture answered; 47 picture requests, outside and inside the container (inside, one
answer was cut off by the 200-token limit of the test), no GPU fault in the kernel log, no 503.

## Gyro on an Arc (2026-10-20, branch f-022-arcgyro)

Gyro-S and Gyro-M (trellis-coded experts, [GYRO.md](GYRO.md#intel-arc-sycl)) run on the SYCL engine. Tested on the Arc A750 only (i915, 8 GB,
Ryzen 5 5600X, 62 GB of RAM). Setup writes the usual A750 config for them (32K context, 300 MiB VRAM reserve), without `--ple-io ram`: the
28 GB n-gram table stays on the SSD, because the expert arena already takes 24 GB (Gyro-S) of the RAM.

| A750, Gyro-S | |
| --- | ---: |
| prompt, 1,406 tokens | 82 tok/s (40 with dequantize + oneMKL; the grouped XMX launch now takes the trellis types, 5/5 pairs) |
| prompt, 2,377 / 20,000 tokens | 102 / 195 tok/s |
| decode, MTP draft of 4 | 15-16 tok/s (13.7 with the old link share, 5/5 pairs) |
| experts in the card's cache | 410 of 24,576 |

Decode is bound by the CPU pool (5 workers, 8 GB/s) and the PCIe link (10.4 GB/s measured), not by the card's kernels (20 microseconds an
expert). For trellis packs the engine takes the link share against a 12 GB/s reference instead of 20, which gives 0.48 on this card instead
of 0.29. A smaller prompt chunk makes room for more cached experts and slows the prompt: `--prefill 1024` +13% decode, -24% prompt speed;
the default stays `--prefill auto`. Gyro-M needs `--max-context 8192 --vram-reserve-mib 100 --prefill 1024` on 8 GB (14 tok/s decode,
62 tok/s prompt). A second Arc card does not take Gyro's experts (the engine refuses and says so). The Battlemage AOT build (`bmg-g31`)
compiles with the trellis kernels; it was not run, there is no B-series card here.

## Arc Pro B70 with a model that does not fit: Flash-Next IQ3_S (2026-10-07, branch f-141-intel)

The earlier B70 rows are for models whose experts all fit in VRAM. IQ3_S has 24,576 experts of 1.97 MB (46 GB): the 32 GB
card holds 11.5k of them (22 GiB of cache) and the rest sit in a pinned host mirror that the GPU reads over PCIe. Tested on
the B70 VM (xe driver, kernel 7.0, oneAPI 2026.1, PCIe measured at 13.3 GB/s host to device), engine 0.1.40-sycl.

**Two things made it produce garbage, both fixed or documented here.**

- **Aliased pages in big allocations (fixed in the engine).** The xe driver returned the 22 GiB expert-cache arena with two
  of its 2 MiB pages mapped onto the same memory: a write at +2 MiB showed at +1022 MiB. Cache slots there held other
  experts' bytes, and the prompt came out NaN at the first layer that routed to one (layer 4, expert 249); the same run
  with another `--max-context`, or with an MTP window of 4 instead of 6, was clean. `STRATA_VERIFY_ALL_SLOTS=1`
  (`STRATA_VERIFY_FIND=1`, `STRATA_VERIFY_DUMP=dir`) reads every filled slot back and compares it with the GGUF; that is how
  the aliasing was found (the wrong bytes were slot 0's blob). Every allocation of 32 MiB or more now goes through
  `strata::malloc_device_guarded`: it tags every 64 KiB, reads the tags back in a second kernel, and allocates again behind a
  growing spacer until the range is clean (a warning names each retry; the check costs under 0.5 s at startup;
  `STRATA_ARENA_ALIAS_CHECK=0` skips it). `sycl/probe/arena_rw.cpp` and `alias_scan.cpp` do not reproduce it on their own: it
  needs the engine's allocation history.
- **`STRATA_VERIFY_NO_HOST=1` is required whenever part of the experts is not in VRAM.** `sycl/serve/strata-sycl.sh` sets
  it; a by-hand run must too. Without it the per-layer host/GPU handshake is not visible across the bus on xe and the logits
  turn NaN a few tokens into the answer, differently on every run. With it (and the mirror covering every miss) the run is
  deterministic: two runs gave identical tokens.

**Numbers** (greedy, `--spec 4 --spec-min-p 0.5 --mtp`, INT8 KV, 32K context, `--stream-experts --vram-reserve-mib 1024
--ple-io ram`, 11.5k experts resident; the 10-prompt `mg_norepeat` gate passes 10 of 10, the xe error counters did not move):

| | before | after |
|---|---|---|
| 4,095-token prompt, `--prefill auto` (2,048-token chunks) | 618 tok/s | 730 |
| 4,095-token prompt, `--prefill 4096` | 784 | **980-1,002** |
| 8,169-token prompt: auto / 4096 / 8192 lending cache slots / 8192 own buffers | 676 | 953 / 1,037 / 1,117 (9.4k slots) |
| decode, 200-token story (66% of drafts accepted, 1.98 tokens per round) | 31.4 tok/s | 30.5 with 4096-token chunks (700 fewer cache slots) |
| decode, 200-token code answer (90% accepted, 3.65 tokens per round) | not measured | 40.8 with 4096-token chunks |

(Interleaved A/B pairs, medians of 3-5. "Before" is the same build with the old short first chunk; the decode figures are
the old default and `--prefill 4096`.)

**Why the prompt is slow here, and the two changes.** Every prompt chunk streams the experts the cache does not hold over
PCIe, and a 4K prompt touches nearly all 512 experts of every layer: 13.5k missing experts x 1.97 MB = 26 GB, 2 s of the
bus per chunk. So the prompt reads faster in fewer, bigger chunks, and the short first chunk (`STRATA_PREFILL_FIRST`, 256
tokens, there to start the GPU while the PLE rows are read) cost a whole extra pass of that stream. It is now 256 only when
the cache holds every expert, else 0. `setup_intel.py` writes `--prefill 4096` for a card with 24 GB or more (the chunk's
buffers take ~700 cache slots; `--prefill 8192 --prefill-borrow` lends cache slots instead and keeps all of them for decode,
at ~1 s of refill per prompt). Where the 4K prompt's GPU time goes now (GPU timeline, timing on): expert dequant 20%, the two
expert GEMMs 34%, QSA attention 14%, host grouping 6.5%, GDN 9%, hyper-connection reads 4.6%.

**Draft length** (200 tokens, medians of 5 interleaved pairs, `--spec-min-p 0.5`): `--spec 6` against `--spec 4` is +6.5% on
code (43.4 against 40.8 tok/s, 84% accepted against 90%) and -1.5% on prose (29.9 against 30.4). A 2-run sweep of the rest
(`--spec 2/3/5`, min-p 0.3/0.7) was within the run-to-run noise of ~1 tok/s; the setup default (4, 0.5) stays.

**FP64.** Arc Alchemist emulates it (`IGC_EnableDPEmulation=1`); on the A750 one row of the sampler's top_p/temperature tail
costs 451 us in double and 20 us in float (`sycl/probe/fp64_cost.cpp`), and a sampled decode ran at 7.6 tok/s against 12 for
a greedy one. On the B70 the same kernels cost 3.4 against 2.1 us, so FP64 is not a B70 problem. Of the kernels that use
double, the native-pack path (the packs setup builds) runs only the sampler (a 16-token greedy run created 173 kernels, the
double ones among them: `sample_tokens` only; the native router, combine, gate and norm kernels are float). The sampler's tail
now accumulates in `strata::samp_acc_t`, float by default (`-DSTRATA_SYCL_SAMPLER_FP64=1` keeps double): A750 sampled decode
7.62 -> 8.10 tok/s (medians of 5 interleaved pairs, runs 7.1-9.2), `sampler_parity` 0 failures on both cards, and the B70 gate's
sampled cases picked the same tokens.

The sampler's launchers also kept dpct's `aspect::fp64` check, which threw "'double' is not supported" at the first sampled token on an A750 without the emulation keys even though the kernels had been float since the accumulator change; 0.1.41 drops the check (it stays in a `-DSTRATA_SYCL_SAMPLER_FP64=1` build). Measured on the A750 (IQ3_XXS, `--spec 4 --mtp`, sampled, 100 tokens, 4 interleaved pairs, one binary): without `IGC_EnableDPEmulation` / `OverrideDefaultFP64Settings` 11.0 / 11.6 / 12.4 / 12.4 tok/s, with them 9.3 / 9.9 / 9.8 / 9.8 (median 12.0 against 9.8, faster in 4 of 4 pairs). Setup still writes the keys for an A-series card: the kernels other than the sampler's that use double (the oracle router, norms, PLE, QSA indexer) were not exercised without them on long contexts or with vision, so the keys stay the safe default.

**XMX.** oneMKL's FP16 GEMMs already run on the matrix engines; per expert the dequant (42 us for a gate/up matrix) costs
twice the GEMM (20 us), and the fused dequant+XMX kernel (`xmx_gemm_bench`) is 0.22x of the pair on this card. The speed left
on this card for this model is the dequant kernels (at ~200 GB/s against a 600 GB/s card), the host grouping, and the QSA
prompt attention (14%); the multi-column int8 DPAS decode kernels of llama.cpp PR 29864 target K-quant and Q8_0 weights, not
the IQ-quant experts, and were not ported.

## Arc Pro B70, Flash-Next IQ3_S: prompt and decode speed in 0.1.43 (branch f-143-intel)

Measured on the B70 (xe, 32 GB, 60 GB RAM, IQ3_S, 11.5k of 24.6k experts in VRAM, the rest in the pinned mirror; unitrace from pti-gpu
for the kernel timelines). All the changes below leave the tokens as they were (first-token logits compared byte for byte, the 10-prompt
gate identical in the 32K and the 128K configuration); every one has an off switch.

- **Prompt: the expert copies run on the card's copy engine** (`sycl/include/strata/copy_queue.hpp`). The Level Zero v2 adapter runs every
  SYCL queue on the compute engine, a copy-only queue included: the ~10,000 expert DMAs of a 4K prompt (2 MB, ~80 us each) sat between the
  expert kernels and never overlapped one (unitrace: 0 of 2,000 copies overlapped a kernel). An immediate command list on the copy-only queue
  group (ordinal 1) wrapped with `sycl::make_queue` puts them on `Copy Engine<1,0>`; events work across the two queues as before. 4K prompt
  3.82 -> 3.25 s, 16K 15.5 -> 13.5 s. On by default for a Battlemage card; `STRATA_COPY_ENGINE=1` forces it elsewhere, `=0` turns it off (an A750
  gained 1.4% and the prompt hung there once the stager's completion flag was a copy-engine memcpy, so Alchemist stays off).
- **The 128K configuration setup writes** (`--kv-resident 32768`, the prompt borrows cache slots) paid three more costs: `kv_append` wrote the
  host copy of the KV with 1-byte stores (2 s of a 4K prompt, now 4-byte lane stores), every expert GEMM staged two 6.5/3.3 MB operands because they
  sit past 4 GiB in the borrowed region (an A770 oneMKL bug that did not show on this card: off on Battlemage, `STRATA_SYCL_GEMM_STAGE=1`
  brings it back), and the lent slots were refilled by a single-threaded GGUF read (0.9 s for 930 slots, now a DMA from the pinned mirror:
  the experts in the last 10% of the cache slots are mirrored too, `STRATA_LEND_MIRROR=0` off). 4K prompt incl. the refill 7.2 -> 3.3 s,
  1K 3.1 -> 1.6 s, 16K 25.5 -> 13.5 s, 49K 78 -> 44 s; a 684-token request through the server 2.64 -> 1.59 s, a 3,083-token one 6.1 -> 2.95 s (the same answer).
- **Decode: `data/expert-profile-sycl.bin`.** The shipped profile ranks the experts for a document summary; a chat, code or non-English prompt finds
  only 60% of its routed experts in the cache (85% for a summary). `tools/make_profile_mix.py` builds a ranking from several routing traces with
  every trace counting the same, `tools/collect_profile_traces.py` records them (`bench/profile_corpus.json`: 45 prompts) through the engine's
  device-side routing counts (`STRATA_USAGE_DUMP=PATH`). The profile built from the 45 prompts and eight summaries holds 71% / 84%
  (leave-one-out) and gave +5% to +64% decode (mean +21%) on 12 prompts that were in neither set, identical tokens; setup_intel writes it for a
  card on xe (the i915 A750 keeps the shipped one: its CPU computes the misses, so the residency moves roundings).
- Decode graph without the nodes nobody answers under `STRATA_VERIFY_NO_HOST` (`STRATA_VERIFY_LEAN=0` off): -2%.
- Tried and not worth it on this card: the grouped XMX expert GEMM for Battlemage (built, `STRATA_PF_XMX=1/2`, correct to 6e-7, 11-15% slower), a
  miss-first or hit-first group order in the device plan (-3% / -7%), spreading the expert chains over several queues (one compute engine),
  `STRATA_LFUSE`, `STRATA_VERIFY_EAGER`.

## Measured, 2026-10-01, Arc Pro B70, Coder IQ1_M, 32K context, INT8 KV: the SYCL port (engine 0.1.31-sycl)

| | |
|---|---|
| decode | 78.2 tok/s on a 19-token prompt, 75.7 after a 2,184-token prompt (256 greedy tokens, MTP + suffix drafts) |
| prompt | 780 tok/s on 2,184 tokens; 934 tok/s at 128K, 784 at 256K |
| through the API | 48-72 tok/s on a ~190-token chat answer, prompt included (the first request after a start is the slower one) |
| VRAM | all 12,288 experts resident, ~1.9 GB free with everything loaded (`--vram-reserve-mib 1024`) |
| RAM | no host copy of the experts (`--stream-experts`) |

## Measured, 2026-09-29, Arc Pro B70, Coder IQ1_M, 32K context, q8_0 KV: llama.cpp

| | |
|---|---|
| load | ~100 s |
| VRAM | 28.4 of 32 GB |
| decode | 23-25 tok/s; GPU 92% busy at 165 W, CPU idle |
| prefill | 149 tok/s on a 2,701-token prompt; 424 tok/s on a 104,798-token prompt at 131k context |
| quality | correct code on every test; matches the NVIDIA path token for token in spirit, not measured |

## Arc Pro B70: `--prefill 8192` in the long-context config (2026-10-18, branch f-151-b70)

`setup_intel.py` now recommends 8,192-token prompt chunks for a Battlemage card with 24 GB or more when the context is over 32K (the
prompt borrows cache slots; `recommended_prefill()`), 4,096 everywhere else, and it keeps a chunk size you put in the config.
Arc Pro B70, IQ3_S, setup's 128K flags (`--kv-resident 32768 --vram-reserve-mib 2048`), `--pcie-frac 0 --adapt-every 0`, 2 rounds,
32 greedy tokens after the prompt (identical in every run):

| prompt | `--prefill 4096` | `--prefill 8192` | cache slots | decode after |
|---|---|---|---|---|
| 16K | 1,222 tok/s | 1,347-1,362 (+10-11%) | 11,795 both | 33-38 / 38 |
| 32K | 1,187 | 1,361-1,383 (+15-16%) | 11,794-11,795 both | 35 / 38.6 |

The lent slots are refilled after the prompt, so the bigger chunk costs the decode nothing. First-token logits against 4096
(`STRATA_DUMP_FIRST_LOGITS`, whole vocabulary): KL 1.1e-2 nats at 16K and 3.0e-2 at 32K, top-1 the same, largest logit
difference 1.0 (the chunk boundary moves where the QSA block selection is made, so a different chunk size is not bit-identical;
two runs of the same size are, KL 0). With the 32K context (own prompt buffers, no borrowing) a bigger chunk is +20% on a 16K
prompt (1,236 -> 1,478-1,482) but takes 1,397 cache slots (10,926 -> 9,529) and decode after it falls 9-16%, so 4,096 stays there.

## Arc Pro B70: the fused int8 expert path for the prompt (2026-10-18, branch f-151-b70, default on for Battlemage)

On a Battlemage GPU (B70, B580, B570: detected by the device architecture id, `bmg-g21` and `bmg-g31`, not by name) the engine computes the routed experts of the prompt with ESIMD kernels that decode the quantized weights in
registers straight into the s8 DPAS B operand (no FP16 copy of the expert, 2.1 MB of weights read instead of 9.8 MB written and read
back) and multiply them by q8_1-quantized activations; `sycl/src/prefill/esimd_moe.dp.cpp`, behaviour note and switches in
[INTEL_INT8_EXPERTS.md](INTEL_INT8_EXPERTS.md). Gate/up IQ3_S, IQ3_XXS, IQ2_S and down IQ4_NL, Q2_0 are covered (47 of the 48 layers of
the mixed IQ3_S pack; the other layer, IQ4_XS gate/up, and any other format keep the FP16 path). Default (STRATA_PREFILL_MMQ unset) is `auto` on Battlemage and the FP16 chain on every other GPU (Alchemist A750 / A770, Lunar Lake, other vendors); `=0` turns it off, `=1` forces it, `=auto` is the default. The activations are
quantized, so the bits differ from the FP16 path. `auto` times both chains on this GPU at the first layer of each format and takes int8
below the measured crossover (130-250 rows per expert here), per group of experts.

Arc Pro B70, IQ3_S pack, 32K config, `--prefill 4096`, fixed flags (`STRATA_PREFILL_CPU_SHARE=0 STRATA_ROUTE_TAIL_SKIP=0 --pcie-frac 0
--adapt-every 0`), 5 interleaved rounds per size, prefill tok/s (medians; the noise arm is the default run again):

| prompt | default | noise arm | `=1` | `=auto` |
|---|---|---|---|---|
| 1,024 tokens | 645.8 | 648.3 | 775.4 (+20%) | 783.5 (+21%) |
| 4,095 tokens | 1,257.0 | 1,256.8 | 1,413.8 (+12.5%) | 1,407.7 (+12.0%) |

The output tokens are identical in all runs. Quality (teacher-forced KL against the FP16 path over 6,002 positions of 28 prompts): mean
2e-4 nats, p99 6e-3, top-1 99.7%, perplexity ratio 1.0004; a `--prefill 8192` chunk moves the first-token distribution by 1e-2 to 3e-2
for comparison. At more than ~130 rows per expert (a chunk of 8,192 tokens or more) the int8 chain stops winning because every row block
decodes the weights again (1.03x at 194 rows against 1.4-1.9x at 60): use `auto`.

### Arc Pro B70, what else was measured on the prompt path (2026-10-18, branch f-151-b70)

All on the 4,095-token IQ3_S prompt, 32K config, `--prefill 4096`, fixed flags, 3-4 interleaved rounds with a noise arm (prefill tok/s,
medians, output tokens identical in every arm):

| change | result |
|---|---|
| `STRATA_GDN_XE=1` (variant 6, the sub-group recurrence that is the default on Alchemist; bit-identical) | 1,251 -> 1,207 (-3.5%): stays off on Battlemage |
| `STRATA_PROMPT_ATTN_XMX=1` / `=32` (the joint_matrix prompt attention, 64- / 32-cell chunks) | 1,258 -> 952 / 913 (-24% / -27%): stays opt-in |
| QSA prompt attention, probabilities stored `[cell][head]` and read as three vectors per cell in the value phase (`STRATA_ATTN_RS=2`, now the default; first-token logits md5 equal at 4K and 16K) | 1,254 -> 1,265 (+0.9%, 3 of 3 rounds) |
| the same plus a 16-shuffle reduce-scatter of the 12 heads' score sums (`STRATA_ATTN_RS=1`; also bit-exact) | 1,254 -> 1,236 (-1.5%): the score phase is not shuffle bound |
| int8 expert groups of 24 or 32 experts instead of 16 | 1,416 -> 1,420 / 1,416: no effect |
| K-quant / Q5_1 / Q8_0 multi-entry decode kernels (`STRATA_KQ_SPLIT`, on by default; bit-identical to `=0`), `native_grouped_parity --kqbench 16`, synthetic UD-Q4_K_XL-shaped experts | 1 token: 76.4 vs 75.0 us (-1.8%); 2 tokens: 122.5 vs 135.9 us (+10%); 4 tokens (an MTP window): 202.6 vs 233 us (+13%); Q6_K 4 tokens 387 vs 485 (+20%), Q8_0 290 vs 341 (+15%) |

The B70's int8 expert kernels also build and pass the parity test as a SPIR-V JIT build and as an AOT build for `bmg-g21` (compile only: no
G21 card here); an AOT build for another device (Alchemist, Xe-HPC) gets stubs (`STRATA_I8_NO_ESIMD`).
