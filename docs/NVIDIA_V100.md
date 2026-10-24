# NVIDIA Tesla V100 / Titan V (Volta, sm_70): the community build

Strata's ready-made engine and the support matrix start at compute capability 7.5 (RTX 20). A V100 (7.0) runs the same
engine compiled for it, through the experimental build that issue #236 added. It is **community-tested, not part of the
supported cards**: no ready-made engine is published for it and the numbers below come from one machine.

## Build

CUDA 13 dropped Volta, so the build needs a **CUDA 12.x** toolkit (12.8 was used) and a host GCC that toolkit accepts.

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON
cmake --build build --target strata -j
```

Setup does the same by itself for a model on a V100 (the experimental CUDA 12 engine, in `engine-cuda12/`: ready-made
on Windows, compiled with a CUDA 12.x toolkit on Linux; [OLDER_GPUS.md](OLDER_GPUS.md)). For a PC with a V100 and a newer card, a build for both architectures (`-DCMAKE_CUDA_ARCHITECTURES="70;86"`) is the
natural choice (a community report in issue #509 ran an RTX 3080 beside a V100 that way; it was not repeated here). On the
measuring PC a Quadro RTX 4000 (sm_75) sits beside the V100s and was hidden with `CUDA_VISIBLE_DEVICES` because the sm_70-only
build has no code for it.

## What runs on Volta

The flag only lowers the floor (CMake and the runtime device check) and replaces two intrinsics Volta lacks. It does not
select "old" kernels everywhere: which kernel runs is decided per architecture when the engine is built.

| Part of the prompt path | On sm_70 |
| --- | --- |
| MoE experts | opt-in (`STRATA_PF_WMMA=1`, only when every visible card is compute capability 7+; MMQ otherwise): `gemm_iq_f16_grouped`: a group of experts per launch on the FP16 tensor cores, the weights dequantized in shared memory (+10% prompt speed against MMQ's `dp4a` kernels, [bench](../bench/results/2026-10-07-v100-prompt-experts/README.md); FP16 activations instead of MMQ's q8_1, so the answers differ) |
| Dense projections (dequantized weights) | FP16 tensor-core GEMMs (cuBLAS / CUTLASS `s884`) |
| BF16 projections (hyper-connection, router, indexer, ...) | converted to FP16 and run on the FP16 tensor cores (#655, #540; `STRATA_BF16_TC=0`: cuBLAS BF16, an FP32 SIMT kernel on Volta) |
| QSA attention for decode and verify windows (and prompts with `STRATA_PROMPT_ATTN_OLD=1`) | #540's kernel (fewer shuffles, bit-exact; `STRATA_ATTN_PRE75=0`: the one other cards run) |
| QSA prompt attention, int8 / FP16 / K8V4 KV | `prompt_attn_v70_kernel` on `mma.m8n8k4` (`STRATA_PROMPT_ATTN_OLD=1`: the decode kernel, one query at a time) |
| QSA prompt attention, Q4_0 KV | the decode kernel |
| QSA block scores | the warp kernel (the tensor-core scorer needs sm_80) |

| Part of a decode / verify window | On sm_70 |
| --- | --- |
| Routed experts in VRAM | gfx906's expert mode 8: the codebook grid and the group's activations in shared memory, SwiGLU fused (`STRATA_EXP_MODE=0`: the CUDA layout other cards run) |
| Dense 2-4 column GEMVs and the head | the interleaved `native_mmvq_il` with a Volta rows table (`STRATA_MMVQ_IL=0`: `native_mmvq`) |
| Hyper-connection read, up projection | gfx906's latency-hidden up kernel (`STRATA_GR_FAST=0`: the plain one) |

All three are bitwise the kernels they replace; on a V100-SXM2 they take a verify window's GPU work from 22.8 to 21.2 ms
([bench/results/2026-10-07-v100-decode-kernels](../bench/results/2026-10-07-v100-decode-kernels/README.md)).

## Measured

One V100-PCIE-32GB, i9-7960X, Unsloth `UD-IQ4_XS` (not a setup model), int8 KV, MTP `--spec 2`; prompt read speed of random-word prompts, tokens/s,
with and without the Volta attention kernel (`STRATA_PROMPT_ATTN_OLD=1`):

| Prompt tokens | old attention path | `prompt_attn_v70_kernel` |
| --- | ---: | ---: |
| 7,194 | 1,010 | 1,164 |
| 28,650 | 1,063 | 1,251 |
| 114,338 | 973 | 1,123 |

Decode was 38-51 tok/s in every run and is not affected. The profile, the parity numbers, the needle test and the limits are in
[bench/results/2026-10-03-v100-prompt-attn](../bench/results/2026-10-03-v100-prompt-attn/README.md).

What the kernel does and does not guarantee: against an FP64 reference its error is about 2-3e-6 (output scale 3.6; the FP32 kernel it replaces: 2e-6; 5e-6 before the hi and lo halves got separate accumulator chains); 15 of 15
needles were found; greedy output was identical on the 28,650- and 114,338-token prompts and differed late in the answer on the 7,194-token one (another
summation order).

## Checking and benchmarking the Volta kernels

```sh
cmake -S . -B build -DSTRATA_PARITY_PROMPT_ATTN=ON ... && cmake --build build --target qsa_prompt_attn_parity
CUDA_VISIBLE_DEVICES=0 ./build/qsa_prompt_attn_parity 32768 2048 5     # synthetic, no model; exit 0 = PASS
```

It compares the tensor-core attention with the FP32 kernel and an FP64 host reference and times both. The Q4_0 cases are skipped below
sm_80.

## Limits

- A layer split over two V100 was not measured with this kernel. The cards work in turn on a single request, so a split adds expert-cache room, not prompt speed.
- sm_60 (Pascal) is covered by the same flag, and community runs exist: two Tesla P40 with a layer split (#1028, bench
  [2026-10-05-community-2x-p40](../bench/results/2026-10-05-community-2x-p40/README.md)), a Tesla P100 alone and with an RTX 2070 SUPER as
  the expert-helper card (#1069: 30.5 tok/s alone, 35.8-38.4 with the helper), and a V100 + P100 pair (#1079: +6-11% decode from 0.1.39 to
  0.1.40 with the P100 as helper cache). One reporter's numbers each; this page's own kernel measurements are V100 only.
- Volta's tensor cores take FP16 only: a model's BF16 weights are converted, and a weight outside FP16's range would saturate (none did in
  the check above).

## The decode kernels measured on a V100 (opt-in: `STRATA_SM70_TABLE=1`)

The interleaved 2-4 column matvec with a Volta rows table, expert mode 8 and the latency-hidden norm/up (bitwise the default
kernels' output, GPU work per verify window 22.8 -> 21.2 ms on a V100-SXM2 in the contributor's measurement) are used on
sm_70 only when `STRATA_SM70_TABLE=1` is set. They become the default once the contributor has confirmed them on a V100 with
the 0.1.41 code. Without the variable sm_70 runs what 0.1.40.3 ran. The single switches (`STRATA_EXP_MODE`, `STRATA_MMVQ_IL`,
`STRATA_GR_FAST`) still apply on top.

## More opt-in switches (CUDA 12 experimental build)

None of these is on by default. Each was measured by its contributor on a V100, not by the maintainers.

| Switch | What it does | Contributor's number |
| --- | --- | --- |
| `STRATA_FUSE_DECODE=1` | the decode GEMV / MoE operator fusions of #1097 (gate/up + SwiGLU in one kernel, SwiGLU + quantizer in one kernel, all activation images of a vector in one pass, two BF16 projections in one kernel); bitwise the unfused kernels | 4-16 us per token saved, not visible end to end |
| `STRATA_PF_WMMA=1` | the FP16 tensor-core prompt experts (above); also takes IQ4_NL down projections (K=640, #1771) | +10% prompt speed (#1401) |
| `STRATA_PF_WMMA2=1` | a second prompt-expert kernel for IQ2_S gate/up and IQ4_NL down (#2356) | +2.5-4.3% prompt speed |
| `STRATA_ARGMAX_ROWS_SM70=1` | the multi-block greedy pick on sm_7x (#2357), bitwise the one-block pick | +0.7-0.8% decode |
| `STRATA_GR_NORM_FAST=1` | the register-split hyper-connection norm on CUDA (#2358) | +0.5-0.6% decode |
| `STRATA_Q8_GATHER_WIDE=0` | turns off the 8-wide int8 KV gather (#1095) that the experimental build uses below compute capability 7.5 | +11% on the gather kernel |
| `STRATA_GDN_REC_WARP=2` | the chain-split DeltaNet recurrence of the prompt on a V100 (#1098); changes the summation order (FP32 level), so the default stays the pipelined kernel; `1` is the bit-exact sync-free variant, `STRATA_GDN_CHUNK=1` the chunked path | +6-9% on the recurrence |
