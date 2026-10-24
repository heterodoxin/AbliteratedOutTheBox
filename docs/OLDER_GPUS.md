# Older GPUs (experimental)

Strata's supported cards are NVIDIA RTX 20 / 30 / 40 / 50 and the AMD cards in [AMD_HIP.md](AMD_HIP.md). The cards
below run through **opt-in, experimental** paths that community members wrote and measured on their own machines. The
maintainers have one of these cards, a Tesla P100, and test on it; they have none of the others: each path is
compile-checked and unit-tested here, and the ready-made engines and their output stay exactly as they were. Numbers are
the reporters' own, on one machine each, unless the text says the maintainers measured them.

## Support matrix

### NVIDIA

| Cards | Compute capability | How it runs | What is different on it | Reported |
| --- | --- | --- | --- | --- |
| Tesla P100 | 6.0 | the CUDA 12 engine | `__dp4a` emulated (bit-exact); BF16 projections through fp32; opt-in: `STRATA_Q8_SM60=1` runs the Q8_0 dense projections and head on a GP100 kernel ([below](#pascal-gp100-a-q8_0-decode-kernel-opt-in)) | 2x P100, IQ3_S, engine 0.1.39: prompt 526-532 tok/s, decode 28-32 tok/s at 128K prompt tokens, 3 of 3 needle checks at 122K ([report](../bench/results/2026-10-06-community-2x-p100/README.md), #1157); PH402 (4 GP100 dies), IQ3_XXS: decode 28.3 -> 38.9 tok/s with `STRATA_Q8_SM60=1` and a Q8_0 dense shard |
| Tesla P40 / P4, GTX 10 series | 6.1 | the CUDA 12 engine | BF16 projections through fp32 (cuBLAS has no BF16 GEMM there, #395); the dense weights dequantized by a wide kernel, expert mode 8 and its own interleaved-matvec table; opt-in `STRATA_DENSE_F32=1`: the large dense GEMMs as `cublasSgemm` ([below](#tesla-p40-sm_61-what-was-measured-and-changed)) | P40, IQ3_S, engine 0.1.30: prompt 217-374 tok/s, decode 30-33 tok/s (#395); 0.1.43.1 with the sm_61 changes: prompt 255 tok/s at 565 tokens (was 205), decode 37-38 tok/s (was 34-35); with `STRATA_DENSE_F32=1` 464-474 tok/s at 7K tokens (was 360-432) |
| Tesla V100, Titan V | 7.0 | the CUDA 12 engine | BF16 projections on the FP16 tensor cores (#655, #540); the prompt attention on `mma.m8n8k4` (#600); a leaner attention kernel (#540) | V100-PCIE-32GB, UD-IQ4_XS: prompt 1,123-1,251 tok/s (#600); V100 32GB, IQ2_XS: prompt +22% from #540 |
| RTX 20 (Turing) | 7.5 | **supported**, the ready-made engine | opt-in: `STRATA_BF16_TC=1` runs the BF16 projections on the FP16 tensor cores | RTX 2080 Ti, Q2_0: prompt +15-18% (#655) |

A card needs enough VRAM to be useful: 12 GB or more is recommended, as for every card (a 2 GB GT 1030 is compute
capability 6.1 too, and cannot hold any of the model).

### AMD

| Cards | Architecture | How it runs | Reported |
| --- | --- | --- | --- |
| RX 6800 / 6900 series | gfx1030 | setup (`--backend hip`), unvalidated; #540's attention kernel is the default there, with 8 cells per step and DPP lane exchanges (bit-exact; `STRATA_ATTN_PRE75=0` runs the standard kernel) | [AMD_HIP.md](AMD_HIP.md#rdna2-gfx1030); RX 6900 XT, IQ3_S: prompts +4-6% alone, +7-12% with #835 ([bench/results/2026-10-04-rdna2-pre75-attention](../bench/results/2026-10-04-rdna2-pre75-attention/README.md)) |
| RX 6700 XT | gfx1031 | setup (`--backend hip`), unvalidated (#524) | used daily by its reporter, one card |
| RX 5500 XT (RDNA1) | gfx1012 | built by hand: `-DCMAKE_HIP_ARCHITECTURES=gfx1012` (HIP 5.7 or 7) | 8 GB card, IQ3_S, 8K prompt: 15.3 tok/s decode (#442) |
| RX 5700 XT (RDNA1) | gfx1010 | built by hand: `-DCMAKE_HIP_ARCHITECTURES=gfx1010`, on ROCm 7.14's `gfx101X-dgpu` wheels; needs `ROCR_VISIBLE_DEVICES=0` in a PC that also has an AMD iGPU | 8 GB card, Coder IQ1_M, 32K context: prompt 115 / 144 / 149 tok/s and decode 18.0 / 20.3 / 23.3 tok/s at 4K / 16K / 30K prompt tokens, 6 of 6 needle checks at 8K and 30K, 61 of 61 ctest on the card, 607 of 12,288 experts in VRAM |
| RX 5700 and the 6 GB RX 5600 (RDNA1) | gfx1010 | as above | the same Navi 10 silicon as the RX 5700 XT, so the same build; a 6 GB card leaves ~50 expert slots, expect the RX 5500 XT's range, not this one's |
| Radeon PRO V520 / Pro 5600M (RDNA1) | gfx1011 | built by hand: `-DCMAKE_HIP_ARCHITECTURES=gfx1011` | untested on hardware: it is the same RDNA1 ISA as gfx1010 (Navi 12), and the engine builds for it with 0 errors on the same wheels |
| Instinct MI50 / MI60, Radeon VII | gfx906 (wave64) | built by hand: `-DSTRATA_HIP_GFX906=ON` | 2x MI50, Coder IQ1_M, 128K context: decode 50.1 / 47.8 / 45.7 tok/s at 4K / 32K / 128K prompt tokens, prompt ~520 tok/s (#677) |

## NVIDIA: the CUDA 12 engine

CUDA 13 dropped Pascal and Volta: it cannot compile for them. Setup therefore keeps a **second engine**, built with
CUDA 12.9 and `-DSTRATA_EXPERIMENTAL_SM60=ON`, in its own folder (`engine-cuda12\`, beside `engine\`). One engine runs
per model, so the choice is made per model, by the oldest card that model runs on:

- **Every card the model uses is RTX 20 or newer:** the ready-made CUDA 13 engine, as always.
- **A card is Pascal or Volta:** the CUDA 12 engine. Setup says so (`CUDA 12: sm_70 is older than CUDA 13 supports
  ...`). On Windows it downloads `strata-windows-x64-cuda12.zip` with NVIDIA's CUDA 12 libraries (from pip, like the
  CUDA 13 ones); on Linux, or with `--build`, it compiles the engine with a CUDA 12.x toolkit.

### Opting in

An older card is used only when you choose it; a PC with a newer card keeps recommending the newer one.

| You | Setup |
| --- | --- |
| have only Pascal / Volta NVIDIA cards (and no AMD card it can use) | uses them, with the CUDA 12 engine |
| name the card: `START-HERE.bat --setup --gpu 1`, or `--gpus 0,1` with a newer card | uses it; the model gets the CUDA 12 engine |
| `--cuda 12` (or `STRATA_CUDA=12`) | the CUDA 12 engine for this model, on any card |
| `--cuda 13` | the CUDA 13 engine even with an older card (a warning: it has no code for that card) |
| `STRATA_EXPERIMENTAL_SM60=1` | the older cards are listed as usable (the setting from #295 still works) |

The choice is kept in the model's config (`"cuda": 12`): its starts and `UPDATE.bat` keep it, and other models keep
their own engine. Setting the model up again chooses again (by its cards; add `--cuda 12` to keep a forced choice). A Pascal / Volta card added to a model at a start (`--gpus`) moves that model to the
CUDA 12 engine.

`--cuda 12` is also the way to run Strata with an NVIDIA driver older than 580: CUDA 12 needs 528 or newer on Windows
(527.41, NVIDIA's minor-version compatibility) and 525 on Linux; setup's "driver too old" stop says so. Such old
drivers were not tested here.

### Mixed cards

A model that shares a V100 with an RTX 30 / 40 card runs both on the CUDA 12 engine (it has code for sm_60 to sm_89,
plus PTX). An RTX 50 card (sm_120) in a CUDA 12 engine is a warning, not a stop: CUDA 12.8 and newer compile for it,
but engines built with 12.8 crashed on long prompts there (#220, #224). Keep the RTX 50 card on its own model (`--gpu
N`), where it runs the CUDA 13 engine.

### Building it yourself

```sh
# Linux (Windows: the same with the toolkit's nvcc.exe)
cmake -S . -B build-cuda12 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON \
      -DCMAKE_CUDA_ARCHITECTURES="61;70" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build-cuda12 --target strata -j
```

Setup does the same when it compiles: it looks for the newest CUDA 12.x toolkit (`STRATA_NVCC=<path to nvcc>` picks
one, #601; on glibc 2.43 use 12.8, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

On GCC 12.3 with nvcc (openEuler 24.03, CUDA 12.8, 2x V100) the build needed `-D_BITS_OPT_RANDOM_H` added to the host flags (#1074; one
report, not reproduced here). The `size_t` error in `vmm.hpp` that the same report hit is fixed in 0.1.40.2.

### What the flag changes, and what it does not

`-DSTRATA_EXPERIMENTAL_SM60=ON` lowers the runtime floor to compute capability 6.0 and compiles the older cards' code
paths into that build only: the Volta prompt attention (#600), #540's attention kernel (used below sm_75), and the
Pascal BF16 path. The ready-made CUDA 13 engine has none of them, so its kernels and its output are unchanged. The
FP16 path for BF16 projections is in every build but runs by default only below sm_75; RTX 20 owners can try it with
`STRATA_BF16_TC=1` (`=0` turns it off on a V100). It is not bitwise the same as cuBLAS's BF16 kernel (FP16 tensor-core
sums round differently): #540 measured a mean KL of 8.4e-3 on the next-token distribution of 24 code prompts on a V100
(the same top-1 in 23), the size of other summation-order changes; #655 a worst relative difference of 3.5e-5 per
product on an RTX 2080 Ti.

### Tesla P100 (sm_60): seven decode switches on by default (engine 0.1.45.3)

On a Tesla P100 the engine now turns on a set of seven switches that change only how the work is scheduled, so the answers
stay bit for bit the same: `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_SIDE_MIXER=2`, `STRATA_FUSE_WAIT=2`,
`STRATA_B6_MMVQ_ROWS=1`, `STRATA_FUSE_DECODE=1` and `STRATA_GR_UP75=1`. `STRATA_SM60_PERF=0` turns the whole set off, and a
switch you set yourself, to any value, is kept. The set is used when every visible card has compute capability 6.0.
Measured by the maintainers on a Tesla P100 16 GB, IQ3_XXS, default flags, 8 interleaved pairs of fresh engines: decode 36.56
-> 41.76 tokens/s (+14.2%, faster in 8 of 8 pairs; a run against itself gave 0.975), with the same number of expert-cache
slots; the identity gate (10 prompts and 4K/20K prompt dumps) is identical with the set on. The full list, the RTX 30 set
and the per-switch results: [DETAILS.md](DETAILS.md#rtx-30-series-sm_86-and-tesla-p100-sm_60-bitwise-decode-switches-on-by-default).

### Pascal GP100: a Q8_0 decode kernel (opt-in)

GP100 (P100, PH402) is the one Pascal chip without `__dp4a`, and the decode kernels are written for the cards that have
it: on a GP100 die the dense projections of a verify window read their weights at 40-80 GB/s of its ~730, decoding K-
and i-quant blocks (Q6_K, IQ4_XS, Q4_K, IQ3_S, ...) one int at a time. Two changes, both measured on a PH402 only:

- **`STRATA_Q8_SM60=1`** packs every Q8_0 dense matrix and the head into the planes of `STRATA_Q8_PACKED` (any shape)
  and runs them on `src/kernels/cuda/q8_sm60.cuh`: a warp (half or quarter warp for narrow rows) per row, 16-byte
  weight loads, the window's q8_1 activations copied once per workgroup into shared memory as an int8 plane, the dot
  on `STRATA_DP4A` (vmad), a shuffle sum - no barrier per row. Shapes it declines (a 6144-wide row with 7-8 columns)
  keep the usual kernel. Not bitwise the exact kernels (the per-32 products are added in another order). The packed
  copies cost their size in VRAM beside the GGUF layout the prompt path reads (Flash-Next: 2.9 GiB of dense
  projections over the stages and 0.6 GiB for the head).
- The GSQ-RCO packs have no Q8_0 dense matrices, so the kernel needs a copy of the model whose dense projections and head are
  Q8_0: `python tools/q8_dense_gguf.py <model>-00001-of-00002.gguf <new folder>` writes one: every shard that holds
  such matrices is converted (Swift 1.5 keeps them in both of its shards) and the rest are hard-linked beside them
  (+3.5 GiB; the experts keep their offsets, so the native pack is reused). Point `--native` and
  `--ple-gguf` at the new folder. Requantizing costs at most 0.4% of a tensor's |w|max (Q8_0 is finer than the
  sources).
- **Opt-in `STRATA_IQ_STAGE_TABLES=1`**: IQ3_XXS's and IQ3_S's codebooks and sign-mask tables are staged in shared memory by
  the grouped expert kernels (IQ3_XXS's ported from shinbunbun/llama-cpp-p100-patches 29 and 30, IQ3_S's the same for its
  sign bytes; bitwise the same output). IQ3_S was the one gate/up format whose codebook those kernels still read from
  global memory: the contributor measured a verify window's layer of IQ3_S experts (20 experts, 24 entries) 24% faster on a
  GP100 die (517 -> 393 us with a Q2_0 down projection, 492 -> 369 us with IQ4_NL), and Swift 1.5 IQ3_XXS (13 of its 48
  layers are IQ3_S) 3.2% faster at 1M context (39.3 -> 40.5 tok/s); IQ3_XXS's tables alone were within noise (+0-4%). The
  maintainers' interleaved A/Bs on a Tesla P100 (PCIe, IQ3_XXS pack and IQ3_S pack, 5-6 pairs each) showed no end-to-end
  change (decode 0.97-1.03x, inside the noise of that box), so it is off by default (the contributor's version had it on
  for compute capability 6.0). On an RTX 3090 Ti, forced on, the same IQ3_S layers took 3-13% longer (those kernels already
  read at 490-665 GB/s there).
- Neither change is measured on compute capability 6.1 (P40, P4, GTX 10 series), which has `__dp4a`. Both build for
  sm_61, and on a card with `__dp4a` the Q8_0 kernel's dot is the hardware instruction (`tools/q8_sm60_check.cu`
  passes on an RTX 3090 Ti); numbers from a 6.1 card with `STRATA_Q8_SM60=1` or `STRATA_IQ_STAGE_TABLES=1` would
  show whether either helps there.

Measured on a PH402 SKU 200 (two boards, four GP100 dies of 48 SMs and 32 GB HBM2), application clocks locked at
1050 MHz, Windows 11, driver 581.80 (TCC), CUDA 12.9, engine 0.1.40.2; Flash-Next IQ3_XXS, 32K, `--layer-split 12,24,36
--prefill 2048`; greedy, 256 tokens, medians of 2-3 interleaved starts, decode tok/s:

| | ~10.7K-token document | the same, cached | short reasoning | code | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSQ-RCO shard, exact kernels | 28.8 | 29.4 | 28.4 | 26.8 | 28.3 |
| Q8_0 dense shard, `STRATA_Q8_PACKED=1` | 27.5 | 31.0 | 34.4 | 31.8 | 31.1 |
| Q8_0 dense shard, `STRATA_Q8_SM60=1` | 37.5 | 38.4 | 41.2 | 38.5 | 38.9 |

GPU time per verify window (`STRATA_VERIFY_PROFILE`): dense projections 19.2 -> 11.2 ms, head 4.25 -> 2.1 ms; the
draft policy then picks longer windows (2.10 -> 2.31 tokens per window). Prompt reads are unchanged (437-438 tok/s at
10.7K). `tools/q8_sm60_check.cu` checks the kernel against a CPU reference on these shapes (1-8 columns) and times it.

Two things that mattered as much on this card: its default application clock is 759 MHz, and a layer split leaves
each die idle most of a window, so the dies decode below 1050 MHz unless the clocks are locked
(`nvidia-smi -i <dies> -ac 715,1050` as administrator; 25.0 -> 29.1 tok/s at 10.7K context). And a second engine on
the same PC that pins its threads to the same logical CPUs halves this one's decode while it runs (the host thread
is time-sliced); keeping the two on different logical CPUs (SMT siblings) avoided it.

A/B switches: `STRATA_DEQUANT_WIDE=0`, `STRATA_SM61_TABLE=0` (sm_61: the kernels 0.1.43.1 ran), `STRATA_DENSE_F32=1` (Pascal, opt-in),
`STRATA_Q8_SM60=0|1` (Pascal GP100, opt-in), `STRATA_BF16_TC=0|1`, `STRATA_PROMPT_ATTN_OLD=1` (the decode kernel for prompts), `STRATA_ATTN_PRE75=0`
(#540's kernel off; on gfx103x with HIP that kernel is the default, see the AMD table above, and `=1` turns it on for
another wave32 AMD card). [NVIDIA_V100.md](NVIDIA_V100.md) has the V100 build, its measurements and the parity test.

### Tesla P40 (sm_61): what was measured and changed

One Tesla P40 24 GB in a 20-vCPU VM (Xeon E5-2695 v4, 98 GB RAM), CUDA 12.4 (cuBLAS 12.4.5.8), IQ3_S with MTP
(`--spec 4`), int8 KV, `--max-context 262144`, engine 0.1.43.1; the numbers and every run are in
[bench/results/2026-10-12-p40-sm61-kernels](../bench/results/2026-10-12-p40-sm61-kernels/README.md).
`STRATA_PREFILL_TIMING=1`, `STRATA_DECODE_TIMING=1` and `STRATA_VERIFY_PROFILE=1` show a card that is the bottleneck
everywhere: the prompt's GPU timeline equals its wall time (waiting for expert copies 0.2%, host grouping 0.2% of a
7K-token prompt) and a decode window is 49 ms of GPU stages with 1-3 ms of CPU experts. Of a 7K-token prompt, 36% is
the dense projections' GEMMs, 28% the experts (ggml's MMQ, which has its own Pascal dp4a configuration), 13% the
hyper-connection read and 12% the attention.

Three things run differently on compute capability 6.1 by default and a fourth is opt-in. Each gives the same bits as
what it replaces:

| Change | What it replaces | Measured on the P40 | Switch |
| --- | --- | --- | --- |
| Wide dequantization (a thread per 8 values, 16-bit loads, 16-byte stores) for Q4_K / Q5_K / Q6_K / IQ4_XS | a thread per 32 values with byte loads: 0.5 s of every prompt chunk, whatever its length | Q6_K 10240 x 2560: 7.2 -> 0.33 ms (10 -> 227 GB/s); Q4_K 5.2x, Q5_K 3.8x, IQ4_XS 3.6x; `dequant_wide_parity` 360 cases bitwise | on; `STRATA_DEQUANT_WIDE=0` |
| Expert mode 8 for the experts in VRAM (decode and verify windows) | the CUDA layout | an 18-expert call at 2 tokens: IQ3_S 390 -> 255 us, IQ3_XXS 345 -> 252, IQ2_S 290 -> 238, IQ4_XS 328 -> 261; `native_expert_bench` bitwise | on; `STRATA_SM61_TABLE=0` (or `STRATA_EXP_MODE=0`) |
| The interleaved small-batch matvec with a table measured on the P40 (`kIlRows_61`) | `native_mmvq` | IQ4_XS 2560 x 10240 at 2 columns 112 -> 65 us, Q6_K 12288 x 2560 at 4 columns 189 -> 127 us, the Q5_K head at 3 columns 2,545 -> 1,766 us; `mmvq_il_parity` bitwise | on; `STRATA_SM61_TABLE=0` (or `STRATA_MMVQ_IL=0`) |
| The large dense GEMMs (qkv, query, gate, output projections) as `cublasSgemm` on FP32 copies of the FP16 operands | cuBLAS's FP16-operand kernels, which run at 4.3 TFLOPS instead of 7.6 for most chunk sizes (a 7K-token prompt reads in 16.3 or 19.5 s by its exact token count) | 8.4-9.8 TFLOPS at every size from 512 tokens; qkv at 7,041 tokens 85 -> 45 ms; `gemm_f32_parity` 5,201 chunk sizes bitwise | **opt-in**, `STRATA_DENSE_F32=1` |

The GEMM route is opt-in because its equality is a property of the cuBLAS build, not of the code: it was measured with
cuBLAS 12.4.5.8 only. Run `gemm_f32_parity` (a build with `-DSTRATA_BUILD_TESTS=ON`; exit 0 = every size bitwise) with
your toolkit's library before you set it.

Through the server, medians of 3 requests per run, two runs per build alternated, each run started with the card at
65 C (see below). By default: a 565-token prompt 203 / 205 -> 256 / 255 tok/s, decode after it 35.0 / 34.2 -> 38.2 /
37.2 tok/s, a 7,047-token prompt 1-3% faster at the same token count. With `STRATA_DENSE_F32=1` as well: the
7,047-token prompt 430 / 364 -> 474 / 473 tok/s (the old build reads it at ~361 or ~431 tok/s depending on whether the
chunk has more than 7,040 tokens; with the route every request read at 464-474), decode after it 32.1 / 33.4 -> 35.9 /
34.0. With `STRATA_DEQUANT_WIDE=0 STRATA_SM61_TABLE=0` and the route off the new build measures as the old one (210 /
365 tok/s, 33.3 tok/s).

Measured and **not** changed: the latency-hidden hyper-connection read (`STRATA_GR_FAST=1`) is slower on the P40 below
5 tokens (82 -> 88 us at one); `STRATA_PREFILL_CPU_SHARE=auto` and `STRATA_ADAPT_OVERLAP=1` gave nothing.

Two things to know when measuring on such a card:

- **Temperature.** A P40 has no fan of its own. This one reaches 87-89 C in a few minutes of work; at 89 C the driver
  reports `SW Thermal Slowdown: Active` and the clock falls from 1,531 to ~1,300 MHz (`nvidia-smi -q -d PERFORMANCE`,
  `nvidia-smi --query-gpu=temperature.gpu,clocks.gr --format=csv -l 2`). Start every arm of an A/B at the same
  temperature, or the later arm loses.
- **Greedy answers are not identical run to run on a CPU without AVX-512**, even with `--prompt-cache 0
  --adapt-swaps 0 --pcie-frac 0`: an expert computed on the CPU rounds differently for a lone token than in a window
  of several (`native_gu_mt_min`), and how many drafts a window holds depends on timing. Add `STRATA_IQ_MT_MIN=1`
  (and `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_ROUTE_TAIL_SKIP=0`) to compare two builds' answers byte for byte.

### Quadro RTX 8000: measurements of existing switches

The Quadro RTX 8000 is Turing (sm_75): it has FP16 tensor cores, but no native BF16 or TF32 tensor cores. Two
**existing, opt-in** CUDA switches are worth comparing on your own workload:

- `STRATA_BF16_TC=1` converts BF16 projection operands to FP16 and uses FP16 tensor cores with FP32 accumulation.
  The default on sm_75 is off; `=0` restores the cuBLAS BF16 path. The switch applies to 7.x cards, not the native
  BF16 path on sm_80 and newer. Conversion clamps finite values outside FP16's range to ±65504; small values and
  differently ordered sums can round differently.
- `STRATA_SELECT_SIMT=1` uses the tiled FP32 QSA block scorer on the pre-sm_80 CUDA path instead of the default
  warp scorer. `=0` keeps the warp scorer. It does not enable TF32 on Turing, and its summation order differs.

The existing paths are in [gemm.cu](../src/prefill/gemm.cu) and [qsa_select.cu](../src/kernels/cuda/qsa_select.cu).

Set the environment before starting the engine, or set the switches in the model config's `env` object, then
restart. For example, this config fragment opts into both (keep the config's other settings):

```json
"env": {
  "STRATA_BF16_TC": "1",
  "STRATA_SELECT_SIMT": "1"
}
```

This section records an investigation, **not a new kernel or a change to defaults**. Measurements on 2026-10-07
used one 48 GiB Quadro RTX 8000 (physical GPU 3), a 260 W power limit, a Xeon W-2295, CUDA 12.4 and GCC 13.
The source-built engine was based on v0.1.40.3 with other local changes integrated; unrelated opt-in changes
were disabled in these arms. These are not measurements of an otherwise clean documentation-only checkout,
not results at the launcher's 200 W setting, and not a speed promise for every sm_75 card.

The model was IQ3_S with MTP (`--spec 4`), `--max-context 262144`, INT8 KV with `--kv-resident 32768`,
`--pcie-frac 0.20`, a fixed `--expert-cache 19000` request and `--prefill auto` (8,192-token ceiling).
The engine reported 21,189 cache slots including the prompt loan. Prompt caching and adaptive swaps were off
(`--prompt-cache 0 --adapt-swaps 0`). Each size used three fresh, seeded synthetic prompts and 128 output tokens
per request; the table gives median engine-reported rates, not end-to-end server latency.

| Existing switch settings | 537 tokens: prompt / decode tok/s | 4,057 tokens: prompt / decode tok/s | 32,057 tokens: prompt / decode tok/s |
| --- | --- | --- | --- |
| BF16_TC=0, SELECT_SIMT=0 | 493 / 86.8 | 1,055 / 79.9 | 1,135 / 70.6 |
| BF16_TC=1, SELECT_SIMT=0 | 661 / 83.7 | 1,262 / 75.4 | 1,416 / 68.4 |
| BF16_TC=1, SELECT_SIMT=1 | 679 / 87.1 | 1,290 / 78.2 | 1,501 / 71.6 |

BF16_TC improved prompt throughput by about 20–34%, but did not improve decode in these comparisons. Adding
SELECT_SIMT improved the 32K prompt rate by about 6% over BF16_TC alone; this is not a universal decode win.
BF16_TC changed replies in 3 of the 9 paired requests, so use it only if different rounding is acceptable.
All five basic arithmetic/JSON checks passed in each arm, which is a smoke check, **not a model-quality study**.
A separate synthetic QSA scorer check at 131,072 context tokens matched selected IDs for all 256 queries and
passed its FP64 accuracy gate. That check does not prove identical selections on arbitrary prompts or model quality.

Two other existing tuning options did not justify a recommendation on this machine:

- `--prefill auto:16384`, with BF16_TC on and SELECT_SIMT off, increased the prompt's cache loan from 4.38 to
  7.38 GiB for only about 0.6% more 32K prompt throughput (1,425 versus 1,416 tok/s). Just 24 MiB of VRAM remained
  at the last graph capture. Keep the 8,192 ceiling rather than spend that memory margin for this small gain.
- `STRATA_PREFILL_CPU_SHARE=auto` gave no gain at the two tested sizes (537 and 4,057 tokens): prompt rates were
  647 and 1,253 tok/s versus 661 and 1,262 with BF16_TC alone; decode was also lower. Larger prompts were not
  tested for this setting.

## AMD: building gfx906 and gfx1012

Setup does not build these; build by hand and run `serve/server.py` with a config, as on any other card.

- **gfx906** (MI50 / MI60 / Radeon VII, wave64): a separate opt-in build, `-DSTRATA_HIP_GFX906=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx906` (not `STRATA_ENABLE_HIP`). Current ROCm no longer ships gfx906 libraries; the
  reporter used a community ROCm 7.14 image. Recipe, kernels and measurements: [AMD_HIP.md](AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source).
- **gfx1012** (RX 5500 XT): the wave32 backend, `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1012`. HIP 5.7
  (Ubuntu's packages) works: older hipBLAS (0.x) is used through rocBLAS, the legacy HIP names and the missing
  `__syncwarp` are version-gated, and RDNA1's missing signed dot4 uses llama.cpp's SDWA sequence
  (`-DSTRATA_GFX1012_PORTABLE_DOT=ON`: the portable one).

## Reports welcome

These paths stay experimental until more people run them. A report with the card, the driver / ROCm version, the
model and the engine log (`strata-<model>.log`) in an issue helps; [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)
has the format for measurements.
