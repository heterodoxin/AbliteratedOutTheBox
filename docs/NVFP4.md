# NVFP4 routed experts

The engine can serve a Qwen3.8-Flash-Next whose routed experts are NVFP4, as NVIDIA's ModelOpt stores them: 64-value
blocks of E2M1 codes with four UE4M3 sub-block scales, plus one FP32 `weight_scale_2` per expert and projection.
Everything else (attention, GDN, shared experts, routers, the head) stays in the formats the native path already
reads. Measured on Windows 11, RTX 5090 (32 GB), Ryzen 9 9950X3D, 128 GB DDR5-5600.

**The three NVFP4 versions in setup and the app.** `NVFP4` is "NVFP4 (Strata's version)", `NVFP4-NVIDIA` is "NVFP4
(NVIDIA's version)", and `NVFP4-QAD` is "NVFP4 (NVIDIA, trained for FP4)" ([the short guide](MODELS.md#nvfp4)). All three
read pictures, through the original model's picture reader file that setup downloads with them (see "Images" below and
the FP4-Local section). The other sections of this page apply to all three unless they say otherwise.

## Getting a model

```
python tools/nvfp4_convert.py --model <ModelOpt NVFP4 checkpoint dir> --outfile <model>.gguf
python tools/iq_pack.py --gguf <model>.gguf --out <pack dir>
```

- `nvfp4_convert.py` runs llama.cpp's own converter (`third_party/llama.cpp`, the commit setup pins) with the type
  policy the engine needs. The experts are repacked without loss: `tools/nvfp4_verify.py` compares them with the
  checkpoint bit for bit, and `tools/nvfp4_verify_gguf.py` checks the GGUF.
  - The small projections the BF16 kernels read (routers, SSM gates, indexer, PLE and hyper-connection projections)
    are written as BF16.
  - Everything served natively goes to Q8_0: attention, `ssm_out`, the shared experts, the output head, and the
    token embedding.
  - The 51.2e9-value PLE table is left out.
- `iq_pack.py` writes `experts.bin` for an NVFP4 GGUF by itself. Each blob is `[gate | up | down]` plus a 16-byte
  tail `{s_gate, s_up, s_down, 0}`. The GGUF has no room for these scales, so the engine reads NVFP4 experts from
  `experts.bin` only.

Run it like an IQ pack, with the GGUF as its own dense-weight source and the PLE table from an IQ GGUF's PLE shard:

```
strata --pack <pack dir> --native <model>.gguf --native-dense-gguf <model>.gguf \
       --ple-gguf <an IQ model's shard with per_layer_token_embd> --mtp <mtp/rt> \
       --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 ...
```

The NVFP4 checkpoints change only the routed experts. The PLE table, the MTP layer and the expert profile of the
original model therefore work as they are.

Tested end to end with `jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4`. By its
`hf_quant_config.json`, NVIDIA's `nvidia/Qwen3.8-Flash-Next-NVFP4` quantizes the routed experts the same way
(NVFP4, group 16; the other layers BF16, the PLE table FP8, the MTP experts FP8), so it takes the same path. It has
not been run here as downloaded. Setup offers it as a ready-made pack, the size `NVFP4-NVIDIA`.

## Measured

RTX 5090, 64K context, int8 K/V, `--vram-reserve-mib 1500`:

| | NVFP4 pack | IQ2_XS, same engine |
| --- | ---: | ---: |
| experts in RAM | 63.3 GiB | 33.0 GiB |
| expert cache slots | ~8,350 | ~17,300 |
| decode, 400-token answer | 114-116 tokens/s | 132-139 tokens/s |
| a 32K prompt | 3,900 tokens/s | 5,600 tokens/s |

An NVFP4 expert blob is 2.76 MB, against 1.51 MB for IQ2_XS: the cache holds half as many, and a miss costs twice
the bytes over PCIe. The PCIe share of the missed experts is therefore 0.25 for an NVFP4 pack, not the usual 0.55.
At 0.55 the copy kernel was 40% of all GPU kernel time (nsys, 300-token decode). A sweep at 262K context gave the
following decode rates:

| PCIe share | short prompt | 32K prompt |
| --- | ---: | ---: |
| 0.55 | 103 tokens/s | 105 |
| 0.40 | 110 | 117 |
| 0.25 | 117 | 124 |
| 0.10 | 112 | 118 |

## Measured in Strata (2026-10-13)

The numbers above are the fork author's, on his RTX 5090 and 128 GB. These were measured here, on the ready-made GPTQ
pack (`--family nvfp4 --model NVFP4`).

Quality, the quant yardstick (`tools/quant_quality.py`, 4,995 teacher-forced positions of code, docs, English, other
languages and an agent transcript, against Unsloth's Q8_0; RTX 3060 box):

| | same top token as Q8_0 | mean KL | KL p99 |
| --- | ---: | ---: | ---: |
| NVFP4 GPTQ, experts on the CPU | 92.09% | 0.070 | 1.12 |
| NVFP4 GPTQ, 1,500 experts on the GPU | 91.79% | 0.069 | 1.07 |
| the same with `STRATA_ROUTE_TAIL_SKIP=7` | 88.59% | 0.118 | 1.76 |
| NVFP4 GPTQ with the GSQ-RCO IQ4_NL n-gram table instead of the FP8 one | 92.15% | 0.069 | 1.07 |
| for comparison, the same tool: Unsloth UD-Q5_K_XL | 94.55% | 0.038 | 0.64 |
| Unsloth UD-Q4_K_XL | 93.09% | 0.056 | - |
| Unsloth UD-IQ4_XS | 88.91% | 0.101 | 1.60 |
| GSQ-RCO IQ3_S | 87.69% | 0.144 | 1.98 |

The route tail skip moves the answers by more than the quantization itself, so it is off by default for NVFP4
experts. `STRATA_ROUTE_TAIL_SKIP=7` turns it on: +7-25% decode on the RTX 3060, +47-55% on the RTX 5070 (64 GB).

Decode experts (`nvfp4_decode_bench`, 64 distinct experts per call, the share of the card's DRAM bandwidth):

| tokens per expert | RTX 5070 | RTX 3060 | Tesla P100 |
| --- | ---: | ---: | ---: |
| 1 | 0.294 ms (89%) | 0.554 ms (89%) | 0.910 ms |
| 2 | 0.298 ms (88%) | 0.596 ms | 1.247 ms |
| 4 | 0.335 ms (79%) | 0.766 ms | 1.954 ms |

Each weight is decoded once for all the tokens of a verify window (before the port's kernel work: 2.26 ms on the 3060
at 4 tokens), bit for bit the same results.

Prompt experts on the RTX 5070 (`mmq_expert_bench`, one launch of 16 experts with 625 rows each, gate/up + down): w4a8
1.52 ms, w4a4x2 (`STRATA_PREFILL_NVFP4=w4a4x2`) 1.13 ms, w4a4x2 with down too (`STRATA_PREFILL_X2_MIN_K=640`)
0.995 ms, UD-IQ4_XS's experts 1.20 ms. First-token KL against the exact FP16 path (24 prompts, 1.5-8K tokens): w4a8
0.0068, w4a4x2 0.0056, w4a4x2 with down 0.0044. On a 64 GB PC the prompt waits for the experts read from the SSD,
so the modes give the same prompt speed there; w4a8 stays the default until a 96 GB+ RTX 50 PC measures the others.
On RTX 20-40 the NVFP4 prompt products are 1.35-1.6x slower than IQ3_S / IQ4_NL ones at 32K: NVFP4's 16-value scales
are applied in FP32 after every int8 step, which only Blackwell's FP4 tensor cores avoid.

Below its RAM need (68 GB of experts plus room) setup runs the pack in the RAM-budget mode (`--resident-budget-gib`,
the RAM less 24 GB), as for Unsloth's files. Mapped experts (`--mmap-experts`) on a 64 GB Windows PC filled the RAM.

Images: the packs keep Qwen's vision tower unquantized, so the original's image encoder reads pictures for them
(`--vision`; strata-vision opens the dense GGUF for the vocabulary). Checked on an RTX 5070 and an RTX 3060 against
IQ3_XXS's answers on the same picture (a number in the picture read, two shapes with their colours and sides). The
encoder on the GPU takes ~1.1 GiB of VRAM: on the 3060 at 8K context the cache held 1,293 experts with images against
1,734 without. Pictures are on by default; setup and the app pick the CPU reader under 24 GB of graphics memory and the
GPU reader from 24 GB (`--vision cpu` / `--vision gpu` change it).

## Where the global scales go

Each scale multiplies its own projection's FP32 output, as llama.cpp applies `.scale`:
`h = silu(s_gate * (G x)) * (s_up * (U x))`, `y = s_down * (D h)`.

The scales are ~1e-4 each. Folding `s_down` into up is algebraically the same, but it leaves the hidden near 1e-5.
The hidden is then rounded to Q8_0 (CPU) or q8_1 (GPU) blocks with an FP16 scale, and `amax / 127` lands deep in
FP16's subnormals: one expert's output was off by 2-12% instead of 1.1% (`nvfp4_avx512_parity --expert`). The FP16
prompt path had the same problem in its dequantized up weights (~1e-8). Any future fold must be checked against
FP16's range.

## Prompt path precision

`STRATA_PREFILL_NVFP4` picks how the batched prompt path multiplies the experts:

| mode | kernel | activations | one product vs FP64 (`mmq_nvfp4_parity`) |
|---|---|---|---|
| `w4a8` (default) | llama.cpp MMQ, int8 tensor cores | q8_1, float block scales | 0.53% |
| `w4a4` | llama.cpp MMQ, Blackwell FP4 MMA (sm_120a only) | NVFP4 | 8.6% |
| `fp16` | dequantize + cuBLAS FP16 GEMM | FP16 | reference |

`src/prefill/mmq_nvfp4_w4a8.cu` compiles mmq.cuh's int8 NVFP4 path with Blackwell's FP4 MMA hidden from both the
device code and the host's tile config, so it is the same on every architecture; decode already multiplies at this
precision. The FP4 MMA exists only in the arch-specific target, so `src/prefill/mmq_nvfp4_w4a4.cu` (the FP4 x FP4
kernel and its FP4 activation quantizer) is the one unit CMake builds with `12x` -> `12xa` (`strata_mmq_w4a4`);
the rest of the engine keeps the architectures as given and runs on sm_121 too. A card without that unit's image
(another architecture, or sm_121 against a 120a build) falls back to `w4a8` with a note.

First-token KL divergence against `fp16`, 8 prompts (code and prose, 1K-8K tokens). The noise floor is the same
`fp16` path cut into 4096-token chunks instead of 8192: another summation order, same arithmetic.

| mode | KL mean | KL median | KL max | top-1 agreement | prefill, 4-8K prompts |
|---|---|---|---|---|---|
| noise floor | 0.00023 | 0.00003 | 0.0016 | 8/8 | - |
| `w4a8` | 0.0018 | 0.0010 | 0.0096 | 8/8 | 2,503 tok/s |
| `w4a4` | 0.0080 | 0.0027 | 0.040 | 7/8 | 2,904 tok/s |
| `fp16` | reference | | | | 1,905 tok/s |

`w4a8` is the default. It is 4.5x closer to the reference than `w4a4`, at 86% of its speed, and it is the precision
every generated token already runs at. `fp16` is the choice when the prompt must be read exactly.

`STRATA_DUMP_MOE_INPUT` + `mmq_nvfp4_parity --real` test one product on a real prompt's rows: layer 20, its 12
busiest experts, max|x|/rms 4 median and 9 at most. The errors:

| mode | gate/up | down (heavier SwiGLU tails) |
| --- | ---: | ---: |
| `w4a8` | 0.46% | 0.90% |
| `w4a4` | 7.2% | 8.4% |
| `fp16` | 0.017% | 0.021% |

## CPU experts

`src/kernels/cpu/nvfp4_avx512.cpp` computes the CPU pool's NVFP4 rows in 512-bit lanes. Each 64-value block is
decoded once for every token of a verify window: `vpdpbusd` on |E2M1| codes, with the sign moved onto the
activation. The arithmetic is ggml-cpu's `ggml_vec_dot_nvfp4_q8_0`, with the float additions ordered differently
(worst 4.5e-7 relative, `nvfp4_avx512_parity`). `STRATA_NO_NVFP4_512=1` falls back to ggml-cpu.

A CPU without AVX-512 (Zen 1-3, Intel Core 4th-10th and 12th-14th gen, Core Ultra, Xeon E5 v3/v4) runs the same rows in 256-bit lanes
(`src/kernels/cpu/nvfp4_avx2.cpp`): ggml's AVX2 arithmetic step for step, so every row is bit-equal to ggml-cpu's
(`nvfp4_cpu_parity`, 0 of 1280x9 / 2560x9 rows differ on a Ryzen 5 5600X, a Ryzen 7 255 and a Core Ultra 7 265).
`STRATA_NO_NVFP4_256=1` falls back to ggml-cpu. An NVFP4 pack is a native pack (its folder holds `native_experts.txt`),
so the engine starts on such a CPU; the AVX-512 requirement belongs to the canonical Q2_0 pack only (#2347: a `--pack`
folder without `native_experts.txt` is read as that pack, and the engine now says so before it refuses).

| tokens | ggml-cpu | AVX-512 | speedup |
|---|---|---|---|
| 1 | 0.102 ms | 0.058 ms | 1.76x |
| 4 | 0.410 ms | 0.125 ms | 3.29x |
| 7 | 0.719 ms | 0.196 ms | 3.66x |

These are one expert's gate+up (1280 rows of 2560, cache-resident). The pool itself is bound by DRAM.

The pool in DRAM terms (`nvfp4_avx512_parity --bw THREADS TOKENS`: 1536 experts, 4.2 GB, GB/s of weights read; tokens = tokens an
expert serves in one verify window, 1-2 in decode):

| CPU (threads) | kernel | 1 thread, 1 tok | all cores, 1 tok | 2 tok | 4 tok |
|---|---|---|---|---|---|
| Ryzen 7 255, Zen 4 (8) | AVX-512 | 15.4 | 58.6 | 57.6 | 47.7 |
| | AVX2 rows | 13.9 | 58.9 | 58.1 | 42.1 |
| | ggml-cpu | 12.6 | 58.7 | 44.8 | 22.4 |
| Ryzen 5 5600X, Zen 3, DDR4-3200 (6) | AVX2 rows | 12.1 | 42.2 | 41.5 | 30.1 |
| | ggml-cpu | 10.4 | 42.9 | 30.9 | 15.5 |
| Core Ultra 7 265, DDR5-5600 (20) | AVX2 rows | 16.0 | 74.2 | 73.0 | 60.6 |
| | ggml-cpu | 10.3 | 74.1 | 62.7 | 37.1 |

Without AVX-512 a core reads ~10% slower (13.9 vs 15.4 GB/s), but a pool that reads DRAM runs at the same rate up to two tokens a
window and ~12% below at four. The CPU's memory decides, not its vector width.

## Other GPUs

Only the optional `w4a4` needs Blackwell. The default `w4a8` is ggml's int8 MMQ (sm_75 and up), and decode uses the
engine's own q8_1 kernels, which decode UE4M3/E2M1 in software. The engine was built for sm_75/86/89 as PTX and run
on the RTX 5090 with the host answering as that generation. On those builds `mmq_nvfp4_parity` and
`nvfp4_expert_gpu_parity` gave results identical to sm_120.

HIP: the decode kernels use no CUDA-only intrinsics. The NVFP4 MMQ instance and the W4A8 unit are CUDA-only and
guarded (`GGML_USE_HIP`). Neither was built for HIP here.

## Tests

- `nvfp4_expert_gpu_parity`: the decode kernels against ggml's reference.
- `nvfp4_avx512_parity`: the CPU rows against ggml-cpu (with `--expert`: one whole expert).
- `mmq_nvfp4_parity`: the prompt path's products against FP64 (with `--real`: rows dumped by `STRATA_DUMP_MOE_INPUT`).
- `tools/nvfp4_verify.py` and `tools/nvfp4_verify_gguf.py`: the conversion, bit for bit.

## NVIDIA's FP4-Local checkpoint

`nvidia/Qwen3.8-Flash-Next-FP4-Local` (setup size **NVFP4-QAD**) is NVIDIA's own retrained 4-bit model: ModelOpt 0.47
safetensors, experts retrained by quantization-aware distillation. Strata reads it by converting it on the user's PC;
nothing of NVIDIA's weights is redistributed (licenses: NVIDIA Open Model Agreement and Qwen Community License 1.0).

What the checkpoint holds, and what `tools/nvfp4_local_pack.py` does with it:

| In the checkpoint | Becomes |
| --- | --- |
| routed experts: NVFP4 (E2M1, E4M3 scale per 16, F32 scale per tensor); the scales stored in NVIDIA's "CSF" form (`.nvfp4_csf_fixed` U8 + `.nvfp4_csf_exceptions` U32) | the scales are decoded to the plain E4M3 matrix (byte-exact; the format is the one vLLM's loader documents), then repacked exactly like every other NVFP4 pack: `experts.bin` with the same 36-byte super-blocks and the 16-byte scale tail. Checked: 304 complete experts and the whole `experts.bin` (SHA-256) equal the route through a full GGUF |
| attention, GDN projections, shared experts, indexer: MXFP8 (E4M3 values, E8M0 scale per 32) | dequantized exactly, then Q8_0 (same 32-value blocks; +0.6% RMS on top of MXFP8's own ~3%), or BF16 for the small projections the BF16 kernels read |
| n-gram (PLE) table, 26.8 GiB: NVFP4 rows of 160 values | a one-tensor GGUF of 90-byte rows (80 B codes + 10 E4M3 scales, one F32 for the table), read by the new PLE format `NVFP4` (`strata.ple.format = nvfp4x16`) in `ngram.cpp`; same row size as IQ4_NL |
| MTP head: NVFP4 experts, BF16 rest | a Q2_0 draft layer built with `mtp_pack.py` / `mtp_rt.py` (88.7% of drafts accepted over 225 test requests) |
| vision tower | not converted. Pictures are read by the original model's picture reader (the shared `mmproj` file that setup downloads, as for the other NVFP4 versions); the dense GGUF supplies the vocabulary |

`python tools/nvfp4_local_pack.py prepare --model <checkpoint> --out <dir>` runs every step (setup does this): the n-gram
table, the dense GGUF (no experts), the pack (experts written straight from the checkpoint) and the draft layer. It needs
about 110 GB of free disk beside the 102 GB download, which can be deleted afterwards. On a Threadripper PRO with the files
on NVMe it took about 12 minutes.

Measured on an RTX PRO 5000 72 GB (Blackwell), 128K context, int8 KV, every expert in VRAM (`tools/strata_bench.py`):
decode 159 tokens/s (median of chat answers, 137-216), 4K prompt 3,981 tokens/s, 16K 4,813, 32K 4,916 (first token 6.6 s),
agent session 151 tokens/s; the engine holds 69 GB of RAM beside 72 GB of VRAM. Quality is in the table below
(`tools/quant_quality.py`, set v1, against Unsloth's Q8_0): top-1 90.5%, mean KL 0.085, perplexity 3.95 (Q8_0: 4.05).
This is a retrained model, so its distance to the original's Q8_0 includes real weight changes, not only rounding.
