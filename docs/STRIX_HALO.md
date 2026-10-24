# Strix Halo (Ryzen AI Max, gfx1151)

Strata runs on the Ryzen AI Max "Strix Halo" APUs (Radeon 8060S / 8050S, RDNA3.5, `gfx1151`, one pool of LPDDR5X shared by the
CPU and the GPU) through the same HIP backend as the Radeon cards in [AMD_HIP.md](AMD_HIP.md). **Status: experimental.**
Everything here was built and measured on one machine, the maintainers' Strix Halo box (Ryzen AI Max+ 395, 128 GB,
Ubuntu 26.04, kernel 7.0, ROCm 7.14.1). Setup (`setup.sh`, `START-HERE.bat`) recognizes it from 0.1.40, by its PCI id (`1002:1586`, one id for the 8060S, 8050S and 8040S) and by
`gfx1151` exactly: a Strix Point (890M / 880M, gfx1150), Krackan (860M / 840M, gfx1152) or Phoenix / Hawk Point (780M / 760M, gfx1103) Radeon is
named as what it is and not supported. On Linux setup compiles the engine for gfx1151 itself (the steps below are the same, by hand). The 0.2.0-beta release also has ready-made packages: `Strata-0.2.0-beta-linux-amd-strixhalo.AppImage` (engine `halo`, gfx1151, ROCm 7.14.1) and, on Windows, `Strata-0.2.0-beta-windows-hip-rdna3.exe` (engine `rdna3`: gfx1100, gfx1101, gfx1102 and gfx1151, ROCm 10.2.0a20260930); any app installer plus **Update engine** (or `strata update`) fetches the same package. What was run: both packages were built, linked and scanned, and the Linux package refuses to start on a card of another family (tried on an RX 9000 card); the maintainers have no Strix Halo and ran neither on one. The measurements on this page come from source builds on Linux (the maintainers' earlier box and community machines). Nobody has run a Strix Halo on Windows. The packaged engines do not carry the hipBLASLt table of section 3 yet (planned for 0.2.1), so their prompt reading is slower than the numbers below until then. A Windows zip of the 0.1.x START-HERE route without gfx1151 code is refused with a message, never used. Setup counts the GPU's memory as the BIOS carve-out plus the shared memory
(the GTT pool; on Windows half the RAM) and recommends UD-IQ4_XS from about 80 GB of memory; it changes no BIOS or kernel setting.

**Windows start-up (0.2.1.2).** On a Strix Halo under Windows the engine no longer registers the expert arena with the GPU driver: registering it took all free VRAM (91 GiB before, 0 after, on a 64 GB carve-out + 32 GB shared PC), the expert cache got no slots and the engine stopped with "the embedding buffer failed" (0.2.1 and earlier). The arena stays in RAM, held by the working-set lock, and the cache gets its normal slots. `STRATA_ARENA_PIN_GIB=N` (or `0`) registers it as before, `STRATA_UMA_REGISTER=1` too. Linux is unchanged.

What the engine does differently on this chip:

- **Unified memory.** The GPU's "VRAM" is the GTT pool of the same RAM. `device_free_bytes()` (the figure `--expert-cache auto` and
  the memory plan use) counts what Linux can give back (`MemAvailable` less 6 GiB for the system, `STRATA_UMA_HEADROOM_GIB`
  changes the 6) instead of only `MemFree`, per device, for any integrated GPU on Linux (this applies to a DGX Spark too).
- **gfx11 matrix cores.** The prompt attention, the block scorer, the prompt GEMMs and the prompt experts have WMMA kernels for
  gfx1100 / 1101 / 1102 / 1103 / 1150 / 1151. A gfx11 part outside that list (gfx1152) is not given them: it takes the portable path.
- **The hipBLASLt table** `tools/hip/gfx1151-hipblaslt-100401.txt` (ROCm 7.14.1's hipBLASLt 1.x `100401`), including the prompt shapes
  at `--prefill 16384` (T 16383 / 16384). Setup's own ROCm for gfx1151 (the 7.14.0a20260608 wheels, hipBLASLt 1.4.0) takes
  `tools/hip/gfx1151-hipblaslt-100400.txt`: the same shapes, calibrated against that library (docs/AMD_HIP.md, Tuning table).
- **The expert cache and the RAM budget share the RAM.** For a model with a RAM budget (UD-IQ4_XS, UD-Q4_K_XL) setup writes a number for `--expert-cache` (the RAM less 24 GB less the budget), not `auto`: `auto` counts all the memory the OS can give back at start, and with a 55 GiB budget and a desktop open it took the room the OS needed (#1715, a global OOM on a 128 GB box). The number is in the config's `args`; `--expert-cache N` there sets it. An engine run by hand should pass one too.

## 1. The toolchain (no root needed)

ROCm 7.14.1 for gfx1151 from AMD's TheRock builds, extracted into a folder of your own:

```sh
mkdir -p ~/rocm && cd ~/rocm
curl -LO https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.1.tar.gz   # 1.7 GB
sha256sum therock-dist-linux-gfx1151-7.14.1.tar.gz   # c40e8f2bd6630a7d11557c762b99c6fa8afb04c9bd0e51ed1675ee1ca24afb00
mkdir install && tar -xzf therock-dist-linux-gfx1151-7.14.1.tar.gz -C install
export SDK=$HOME/rocm/install
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}
```

You also need CMake 3.24+, Ninja, git and a C++ host toolchain with libstdc++ headers (`sudo apt install build-essential cmake ninja-build git`).
The maintainers' box has no system compiler: they point ROCm's clang at a user-space copy of gcc 15 with
`--gcc-toolchain=<dir>/usr` (added to `CMAKE_HIP_FLAGS` below; leave it out when the system has one).

The kernel's amdgpu driver and `/dev/kfd` access (your user in the `render` and `video` groups) are all the GPU side needs.
To let the GPU use most of a 128 GB machine the maintainers' box boots with `ttm.pages_limit=29360128 ttm.page_pool_size=29360128`
(a 112 GiB GTT; the default is about half of the RAM) and `amd_iommu=off`; both are host settings the engine does not need to run,
only to have the room.

## 2. The engine

llama.cpp's `ggml` sources are pinned by this repository: CMake fetches the revision it needs. For an offline build, point
`STRATA_GGML_DIR` at a checkout of exactly that revision.

```sh
git clone https://github.com/Niko1221/Strata.git && cd Strata
cmake -S . -B build-halo -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode"
cmake --build build-halo --target strata -j 8        # about 25 minutes the first time
```

`gfx1151` is "unvalidated" in the build's own wording (it warns). `STRATA_PREFILL_MMQ=ON` builds ggml's MMQ kernels the prompt
experts need. The tests: `cmake --build build-halo -j 8` builds them all, `ctest --test-dir build-halo` runs them;
`hip_prefill_hcd_exact_parity` and `hip_prefill_hipblaslt_gemm` need the table (next section) and skip without it.

## 3. The prompt table, and running

Hand the engine the table (`setup` does it for the supported cards; for gfx1151 you do it): without it the prompt's dense matrix
products use plain hipBLAS and are slower.

```sh
export STRATA_HIPBLASLT_TUNING=$PWD/tools/hip/gfx1151-hipblaslt-100401.txt
```

The engine accepts it only for the architecture and the hipBLASLt version named on the table's second line
(`STRATA_HIPBLASLT_TUNING_V1 gfx1151 100401`: ROCm 7.14.1) and says so at start-up (`prefill gemm: hipBLASLt tuning enabled (90 rows,
gfx1151, version 100401)`). Then run `strata` with the same arguments as on any other card (the model packs, `--serve`, the
settings of [DETAILS.md](DETAILS.md)); a pack of the Unsloth UD-IQ4_XS or UD-Q4_K_XL GGUFs is what the numbers below used.

## 4. What is on by default on gfx1151

At start the engine looks at GPU 0; on a gfx1151 it sets the switches below unless you set them yourself
(`STRATA_GFX1151_DEFAULTS=0` turns the whole table off, and the start-up line names what was set). **Every one of them gave
byte-identical output on the maintainers' box** (the same token ids, 4K / 32K / 64K and four models, and a bitwise harness per kernel):
they change speed, not answers. Nothing here applies to CUDA or to another AMD architecture.

| Switch | What it does |
|---|---|
| `STRATA_GDN_HEAD`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2`, `STRATA_GDN_NOY` | the prompt's gated-delta-net recurrence: four lanes per column, two columns per lane, the convolution and the norms fused |
| `STRATA_HCD_EXACT` | the hyper-connection down projection (N 320, K 10240) in a WMMA kernel that reproduces hipBLASLt solution 1176 / 1177's summation order (`hip_prefill_hcd_exact_parity` checks it bit for bit); it runs only where the table makes hipBLASLt pick one of those two, and says once if it cannot |
| `STRATA_CVEC_FUSE`, `STRATA_PF_PAD` | a steered layer's write, control vector and next norm in one pass; padded GEMM row strides |
| `STRATA_TOPK_REG_MIN_BLOCKS=4096` | the prompt's QSA top-k uses the register kernel from 16K cells of context instead of 28K (the original kernel is slower there; the selected cells are identical, `qsa_topk_parity` checks it) |
| `STRATA_Q8_PACKED`, `STRATA_Q6_PACKED`, `STRATA_MMVF_ROWS`, `STRATA_ATTN_LANECELL` | decode layouts and kernels (packed Q8_0 / Q6_K weights, 4-row BF16 GEMV, one-cell-per-thread attention scores) |
| `STRATA_EXPERT_V2`, `STRATA_TSUM`, `STRATA_LFUSE` | the grouped decode experts (IQ3_S / IQ4_NL), single-butterfly warp sums, fewer launches around the shared expert and the KV append |
| `STRATA_GDN_SPLIT`, `STRATA_QFUSE`, `STRATA_PLE_BATCH` | the GDN step over four blocks per head, activation images written by their producers, the PLE key / value projections of a verify window at once |
| `STRATA_SH_STREAM=1` | the shared expert on a second stream beside the routed experts. It is off by default on other AMD cards (#816), but it pays on gfx1151: decode +1.8% (UD-IQ4_XS) and +6.7% (UD-Q4_K_XL) at 8K |

## 5. What is not on by default (it changes bits)

**Five of the switches below (`STRATA_PF_GEMM`, `STRATA_HC_UPMIX`, `STRATA_PA_FAST`, `STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA`) are one opt-in on an integrated Radeon: `STRATA_IGPU_PROMPT_SET=1` ([AMD_HIP.md](AMD_HIP.md), "Integrated GPUs"). It was measured on a Radeon 780M only, where it is the default.** `STRATA_PF_FUSED`, `STRATA_PF_SWITCH_MIN_T` and `STRATA_PREFILL_STREAM_MIN` stay opt-in as well. `STRATA_HC_Q8` is on by default on gfx1151 (below); `STRATA_HC_Q8=0` turns it off.

These are faster on gfx1151 and **round differently** from the default path (a different summation order or a half-precision
intermediate), so they are opt-in and the maintainers gate them with KL against the default (`STRATA_PF_SWITCH_MIN_T=4096` keeps
the short prompts on the default numerics, since these pay on long ones). The maintainers' fast configuration sets all of
them:

```sh
export STRATA_PF_FUSED=1        # the prompt experts on the matrix cores (IQ2 / IQ3 / IQ4_XS gate/up, Q2_0 / IQ4_NL / Q8_0 down)
export STRATA_PF_GEMM=1         # the prompt's FP16 projections on the 128 x 256 WMMA GEMM
export STRATA_HC_UPMIX=1        # the hyper-connection up projection with its epilogue
export STRATA_PA_FAST=1         # the prompt attention with single FP16 q and p
export STRATA_HIP_WMMA=1        # the prompt attention on matrix cores
export STRATA_SELECT_WMMA=1     # the block scorer on matrix cores
export STRATA_PF_SWITCH_MIN_T=4096
export STRATA_PREFILL_STREAM_MIN=128   # prompt reads of 128-1,023 tokens on the fused experts too (below)
```

`STRATA_HC_Q8` (on by default on gfx1151 since 0.2.2; `STRATA_HC_Q8=0` restores the bf16 read) measured on the Strix Halo (IQ3_S, gfx1151, 2026-10-21, MTP 4 drafts / min-p 0.5): **+8.4%**
on `json_extract` at fixed windows (53.55 -> 58.03 tok/s), at the cost of **94 of 250 output tokens differing** from
the BF16 read (first at 60). It halves the hyper-connection weight bytes (13.29 -> 7.15 MB per read; cold read
93.8 -> 20.3 us, 142 -> 352 GB/s). The answers are not bit-identical to the bf16 read; how far they move is in docs/AMD_HIP.md ("Strix Halo: 8-bit hyper-connection weights").

`STRATA_PF_FUSED=1` reaches only chunks of `STRATA_PREFILL_STREAM_MIN` tokens or more (1,024 by default, the floor measured on
a discrete card, where a smaller chunk does not pay for streaming every expert); a smaller read runs its experts through MMQ.
With every expert in the unified memory nothing streams, and MMQ is the slow part: an agent's turn - a tool result of ~1,000
tokens on a cached conversation - read in 3.1-6.0 s at 1,024 and 0.9-2.5 s at 128 (C2T8, `--no-prefill-borrow`; 12-turn
replay 3.4 -> 2.1 s per turn). Its output is the fused path's, which reads of 1,024 tokens or more already take; it is not
under `STRATA_PF_SWITCH_MIN_T`, which gates only the hyper-connection and padding switches. A UD-Q4_K_XL or other K-quant
pack takes the fused experts only with `STRATA_PF_FUSED_KQ=1` as well.

`STRATA_PF_F16=1` (gfx11 only; add it to the list above) runs the covered prompt experts on the F16 matrix cores instead of
the int8 ones: K-quant gate/up and Q5_1 down (with `STRATA_PF_FUSED_KQ=1`), IQ2 / IQ3 / IQ4_XS gate/up, Q2_0 / IQ4_NL / Q8_0
down. One layer, 16,384 tokens: Q4_K / Q5_1 85.3 -> 53.5 ms, IQ3_S / IQ4_NL gate/up 11.65 -> 9.87 and down 6.11 -> 4.71 ms.
It rounds differently from the int8 kernels. First-token KL against a run with no switches, 12 natural-text prompts of
1.5K-24K tokens (Strata's own docs and sources), top-1 the same in 12 of 12: UD-Q4_K_XL mean 0.0068 (median 0.0021) against
0.0031 (0.0024) for `STRATA_PF_FUSED=1 STRATA_PF_FUSED_KQ=1` without it; UD-IQ4_XS 0.0085 (0.0036) against 0.0068 (0.0032). One
1.5K prompt carries most of the Q4_K_XL mean (0.049).

`STRATA_REFILL_PARTS=1` (opt-in, exact; HIP and CUDA, a GGUF read in place) refills the expert-cache slots a prompt borrowed straight from the GGUF mapping (an expert's gate, up and down tensors) instead of through an assembled 4 MB copy per expert: the same bytes in the slot (checked against the assembled blob on 162 slots), 0.1 to 0.3 s less at the end of a prompt; gfx1151, UD-Q4_K_XL and UD-IQ4_XS: a 4K-8K prompt +2.3% to +3.5%, 32K +1.1%. It is not in the table above because only gfx1151 has measured it.

Two engine flags help on long contexts and are not Strix-specific: `--mtp-window 8192` (the draft layer attends to the last 8,192
cells: 64K output +3.5%, 32K -0.4%) and `--spec`, `--lookup-chain`, `--mtp-q4` (see [DETAILS.md](DETAILS.md)).

## 5b. The desktop, the RAM, and a discrete card beside the Strix Halo (Linux)

The Strix Halo chip usually draws your desktop, and its graphics memory is your RAM. Two things follow, and the engine now takes both into account.

- **One memory, counted once.** The expert cache of an integrated GPU lives in the same RAM as the expert arena. The automatic cache size is the smaller of
  what the GPU reports free and what the OS reports available, less a reserve for the desktop (the larger of 8 GiB and 8% of the RAM; the start-up line says so
  when it changes the size). `STRATA_UMA_RESERVE_GIB=N` sets the reserve (`0` = the old sizing, nothing kept back); `--expert-cache N` is never changed.
- **A card of its own beside the chip** (for example a Radeon AI PRO R9700). A card of its own is the recommended Main card and the chip stays Off in the card roles.
  If you name the chip in a layer split anyway (`--gpus 0,1`, or the config's `"gpu": [0, 1]`), the split runs as asked, the chip is placed on the first layers and
  never on the last stage (the output head, the draft layer and the check of its guesses), the picture reader is started on the card of its own, and the
  start prints one plain line about it. `"gpu_order": "as_given"` and a manual `layer_split` keep your order. `setup --gpus all` leaves the chip out
  when two cards of their own are present; name it by number to use it.
- **The CPU.** On Linux with built-in Radeon graphics the engine starts at nice +5, so the desktop keeps its share of the CPU while the engine's workers spin.
  `STRATA_ENGINE_NICE=0` turns it off, `STRATA_ENGINE_NICE=N` sets another value.

What is not known yet: how much of the screen freeze a long compute job on the same chip causes, and whether registering the whole expert arena with the driver
(the Linux default on gfx1151; Windows uses the unregistered arena since 0.2.1.2) hurts when the GTT is small. `STRATA_ARENA_PIN_GIB=none` runs the unregistered
arena on any OS; reports with `amdgpu_top` and `dmesg` are welcome.

## 6. Measured

Ryzen AI Max+ 395 (128 GB, gfx1151), the maintainers' fast configuration of section 5 on top of the defaults of section 4, `--spec 4`,
`--mtp`, `--lookup-chain 3`, `--mtp-q4 all`, `--prefill 16384`, int8 KV, all experts in the unified memory. Prompt tok/s is
(prompt tokens - 1) / prompt time, output tok/s is the tokens / decode time; medians of 3 runs, one fresh process per run, the two
engines interleaved. "Before" is the maintainers' earlier engine branch (the same switches set by hand), "now" is this release.
Every pair printed the same token ids.

| Model | Context | Prompt before -> now | Output before -> now |
|---|---|---|---|
| UD-IQ4_XS | 8K | 1,212 -> 1,293 (+6.7%) | 51.7 -> 53.8 (+4.0%) |
| UD-IQ4_XS | 128K | 1,241 -> 1,320 (+6.4%) | 49.3 -> 51.4 (+4.2%) |
| UD-Q4_K_XL | 8K | 1,136 -> 1,169 (+2.9%) | 50.6 -> 52.6 (+3.9%) |
| UD-Q4_K_XL | 128K | 1,220 -> 1,248 (+2.4%) | 41.6 -> 43.3 (+3.9%) |
| IQ3_S | 8K | 1,202 -> 1,242 (+3.4%) | 56.2 -> 59.7 (+6.3%) |
| IQ3_S | 128K | 1,254 -> 1,312 (+4.6%) | 40.6 -> 42.8 (+5.5%) |
| IQ3_XXS | 8K | 1,237 -> 1,238 (+0.1%) | 48.8 -> 51.5 (+5.6%) |
| IQ3_XXS | 128K | 1,259 -> 1,252 (-0.6%) | 48.4 -> 50.9 (+5.1%) |

UD-IQ4_XS at 4K / 32K / 64K (5-6 interleaved pairs): prompt +9.2% / +6.0% / +5.3%, output +5.0% / +3.9% / +4.6%.
IQ3_XXS at 8K was measured again with 6 interleaved pairs (an earlier 3-run set had read 5% low while other jobs shared the GPU): prompt equal, output +5.6%. With the defaults off the same build's prompt is 8% lower (1,137), so the table helps there too.

### Re-measured on the merged 0.1.40 code

The same box and method, no environment switches at all (the arch defaults only), on the final merged tree. 4 interleaved pairs per cell,
medians; "j" is the build of the table above (the Aurora port before the other 0.1.40 work was merged), "merged" is the release
code. Every run printed the same token ids, and the ids also match `strata-int29` (the v4.2 config) on UD-IQ4_XS and UD-Q4_K_XL at
a8 / 4K / 64K.

| Model | Context | Prompt j -> merged | Output j -> merged |
|---|---|---|---|
| UD-IQ4_XS | 8K | 1,260 -> 1,159 (-8.0%, spread 1,070-1,320) | 53.94 -> 53.14 (-1.5%) |
| UD-IQ4_XS | 128K | 1,317 -> 1,299 (-1.4%) | 51.38 -> 50.72 (-1.3%) |
| UD-Q4_K_XL | 8K | 1,163 -> 1,104 (-5.0%, spread 890-1,240) | 52.66 -> 50.47 (-4.2%) |
| UD-Q4_K_XL | 128K | 1,252 -> 1,211 (-3.3%) | 43.28 -> 42.64 (-1.5%) |
| IQ3_S | 8K | 1,283 -> 1,265 (-1.4%) | 59.81 -> 59.10 (-1.2%) |
| IQ3_S | 128K | 1,311 -> 1,310 (-0.1%) | 42.74 -> 42.09 (-1.5%) |
| IQ3_XXS | 8K | 1,127 -> 1,212 (+7.6%) | 51.64 -> 50.95 (-1.3%) |
| IQ3_XXS | 128K | 1,252 -> 1,250 (-0.1%) | 50.88 -> 50.22 (-1.3%) |

The prompt numbers at 8K swing by more than the differences (other jobs share this box), so read them as unchanged. The output
column moved by a steady -1.2% to -1.5% (-4% on UD-Q4_K_XL at 8K, where the default arm also had slow runs): since 0.1.39 the
shared-expert stream fork is off by default on HIP (#816). It overlaps the shared expert with the routed experts on a second
stream and does help on gfx1151, so 0.1.40 turns it on there by default (section 4). It changes no bits (same ids). 4 interleaved pairs at 8K, merged
code with the fork off -> on: UD-IQ4_XS output 53.1 -> 54.05 (+1.8%), UD-Q4_K_XL 49.5 -> 52.8 (+6.7%, the default arm ranged 46.0-52.1),
prompt unchanged. With the fork on, the merged code's output is at or above the j build's (IQ4_XS 54.05 vs 53.94).
