# Strata on a Mac (Apple Silicon, Metal)

Strata runs on Apple Silicon Macs through its own engine on Metal. This is the same engine as on a PC. The NVIDIA
kernels are replaced by Metal ones (`metal/`), and the host code and the server are the PC's. There are two ways in:

- **The Strata app (`.dmg`) and the Terminal (`strata launch`).** The Metal engine comes ready-made: the `.dmg` has it
  inside, and the Terminal fetches it as an engine component. Nothing is built on your Mac and you need no Command Line
  Tools ([below](#the-desktop-app-and-strata-launch-on-a-mac)). macOS 15 or newer, as for the source build. If macOS says the
  app is "damaged", see [Troubleshooting](TROUBLESHOOTING.md#the-desktop-app-and-the-terminal); how the builds are signed:
  [packaging/mac/README.md](../packaging/mac/README.md).
- **From source: `./setup.sh`.** It sets things up the way `START-HERE.bat` does on Windows and builds the engine on your Mac
  (about 1-15 minutes).

**Pictures:** on by default in the app, the Terminal and the source build. The reader (`strata-vision`) comes in the engine package
and runs on the Mac's GPU through Metal ([Limits](#limits-this-version)). The NVIDIA, AMD and Intel engines are unchanged.

**Status:** new. It is measured on three Macs: an M5 Max with 128 GB, where the model stays in memory; and an M4 Mac mini
with 32 GB and an M1 Max with 34 GB, where the experts stream from the SSD (the M1 Max: 7.9 tok/s decode, 186 tok/s on a
9,940-token prompt, Q2_0, the expert cache at 64 slots, as reported with the kernel fix #2348). Other chips are untested, and reports help
([COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)).

## What you need

- **An Apple Silicon Mac (M1 or newer).** Intel Macs cannot run it.
- **macOS 15 or newer.** The engine keeps its buffers resident through a residency set, which macOS 15 introduced.
- **The Command Line Tools:** `xcode-select --install`. That is the compiler. Xcode itself is not needed.
- **Enough unified memory for the pack.** The GPU maps the model's main file and keeps it in memory while the model runs. Metal lets one
  program keep about 3/4 of the memory resident (107.5 GB of 128 GB, measured on the M5 Max).

| Pack | The file the GPU maps | Needs about (with the prompt buffers, KV cache, draft layer) | Recommended from |
|---|---|---|---|
| Q2_0 | 37.6 GB | 46 GB | 64 GB |
| IQ3_XXS | 47.0 GB | 55 GB | 74 GB (where IQ3_S does not fit) |
| IQ3_S (recommended; the default from 96 GB) | 54.8 GB | 63 GB | 96 GB |
| UD-Q4_K_XL (Unsloth, recommended: higher quality) | 82.6 GB | 91 GB | 128 GB |
| UD-Q5_K_XL, UD-Q6_K_XL (Unsloth) | 103.9 / 114.8 GB | stream from the SSD on 128 GB | 128 GB, streamed |
| Gyro-S (agentionai, one 58.5 GB file) | 29.7 GB (the file's other 28.8 GB is the n-gram table) | 38 GB | 48 GB; streams below that (measured on 32 GB) |
| Gyro-M (agentionai, one 92.0 GB file) | 63.2 GB (+ the 28.8 GB table) | 71 GB | 96 GB; streams below that (not measured on a Mac) |

- **The n-gram table:** each pack also has a 28.8 GB n-gram table (the model's second file). It is read from the SSD a few rows at a time and does not count against the memory above.
- **Disk:** the download is 66-84 GB, plus about 7 GB for the pack and the draft layer. A Mac that streams also needs room for the experts' copy (below): 34-40 GB.
- **A Mac whose memory cannot hold the pack (32-48 GB):** the engine streams the experts from the SSD instead. It decides this by itself when the experts are more than 60% of Metal's share.
  - It keeps the most-used experts in an arena: Metal's share, or the memory macOS reports as available less what it keeps for macOS, whichever is smaller. It keeps 7 GB, or 15% of the RAM when that is more, and it also keeps the n-gram table's share. It counts as free only what macOS can give without swapping (free and purgeable memory and the clean file cache, not other programs' idle memory). It checks again just before it takes the arena, and fills the arena in 1.5 GB steps, checking again before each one; it stops early rather than push macOS below what it keeps. On a Mac with 64 GB or more the arena, the Metal buffers and the host memory together also stay under 60% of the RAM (see "The engine's whole anonymous footprint" below).
  - It reads the rest as each token needs them, and reads ahead the experts the next layer will probably use.
  - Setup writes a second copy of the experts for such a Mac (`<model>.experts`, 40 GB for IQ3_XXS). In it each expert's three matrices sit together, so the SSD reads an expert with one request instead of three. On the 32 GB M4 that is about 17% faster decoding and 30% faster prompts, with the same tokens. Setup writes it only when the disk has room for it plus 10 GB; otherwise it warns, and the engine reads the model file as before.
  - Setup configures such a Mac without the MTP draft layer: one token per step is faster there.
  - Measured at 32 GB (below). Smaller Macs are untested.
- **Setup recommends, and never forces.** It recommends the pack by memory; a bigger pack chosen with `--model` is kept, with a warning about what it risks.

## Install

```
git clone https://github.com/Niko1221/Strata && cd Strata
./setup.sh                 # asks the size; --yes takes the recommended answers
```

What the first run does (each step is skipped when it is already done):

1. **Python.** It needs Python 3.10 or newer, but macOS ships 3.9. If that is all your Mac has, `setup.sh` downloads a standalone Python 3.12 (25 MB, pinned by SHA-256) into `.python`. No sudo, no Homebrew.
2. **Checks your Mac:** the chip, the memory, the macOS version, the Command Line Tools and the free disk.
3. **Python packages** into `.venv`. CMake and Ninja come from pip too, installed just before the build in step 4.
4. **Builds the engine** (`build-metal/strata`). It uses Apple's metal-cpp headers (pinned by SHA-256) and llama.cpp's ggml at the commit the PC build pins.
5. **Downloads the model.** The download resumes if interrupted. `--gguf-dir` uses files you already have.
6. **Prepares the model:** the pack (seconds), and the MTP draft layer (about 5 GB of the original checkpoint). On a Mac that streams: the experts' copy instead of the draft layer (about 3 minutes on the M4, checked byte for byte against the model file).
7. **Writes `strata-<model>.json` and `run-<model>.sh`, then starts the server.**

The config uses the Mac defaults: `--spec 2` and the Q8_0 hyper-connection read (below). After that, `./setup.sh` or `./run-q2_0.sh` starts the model directly. The server is the same as on a PC:
- the OpenAI-compatible API on `http://127.0.0.1:8080/v1`;
- the Anthropic API on `/v1/messages`;
- the chat page on `http://127.0.0.1:8080/`.

Options: `--model Q2_0|IQ3_XXS|IQ3_S|UD-Q4_K_XL|UD-Q5_K_XL|UD-Q6_K_XL` (the UD files are Unsloth's), `--vision yes|no`, `--context N`, `--port`, `--host` / `--api-key`, `--draft-vocab`, `--no-browser`,
`--data-dir`, `--models-dir`, `--gguf-dir`, `--setup` (another model or new settings), `--no-start`, `--check` (only
check the Mac), `--build` (build the engine again).

### The desktop app and `strata launch` on a Mac

The Strata app (and the Terminal's `strata launch`) do not build anything on your Mac. The Metal engine comes ready-made, as an engine component: from the update feed, or inside the app's installer. The model install is `setup.py` in its Control mode (`STRATA_CONTROL_INSTALL=1`): it downloads (or uses with `--gguf-dir`) and prepares the model files and writes a run config whose engine is `component:strata-engine`. It does not run `metal/setup_mac.py`, needs no Command Line Tools, and writes the same Metal flags (`--mmap-experts`, `--spec 2`, no draft layer when the experts stream from the SSD).

The model list and the recommendation judge a Mac by its unified memory with the rules of `strata_control/mac_fit.py`, the module `setup_mac.py` uses too: Q2_0, IQ3_XXS and IQ3_S are offered (the Unsloth many-shard files need the repack of `./setup.sh` for now); a pack that does not stay in memory is listed as "read from the SSD", and a Mac under 16 GB of unified memory is told the model does not fit.

## Measured (M5 Max, 128 GB, macOS 27.2)

**Decode**, in tokens per second over 200 generated tokens, with greedy answers. The table gives two prompts: a short question
("prompt1") and a story.

| | Q2_0 | IQ3_XXS | IQ3_S |
|---|---|---|---|
| Setup's defaults (`--spec 2` with MTP, the suffix drafter on) | **119.5 / 108.2** | **108.9 / 97.0** | - |
| `--spec 2` with MTP, suffix drafter off | 114.0 / 107.9 | 103.5 / 96.9 | 98.2 / 94.1 |
| No MTP (one token per step) | 87.4 / 87.3 | 78.7 / 78.6 | 74.5 / 74.3 |

The output with MTP is the same as without it, token for token, in every run. MTP only proposes tokens, and the
full model checks every one of them.

**Prompt reading** (time to the first token), on the needle-test prompts below:

| | 4K tokens | 32K tokens |
|---|---|---|
| Q2_0 | ~1,430 tok/s | ~1,460 tok/s |
| IQ3_XXS | ~1,260 tok/s | ~1,240 tok/s |
| IQ3_S | ~1,320 tok/s | ~1,140-1,190 tok/s |

The first prompt after the Mac starts is slower, while macOS reads the model file (IQ3_S: 413 tok/s on the first 4K
prompt, cold).

**For comparison**, llama.cpp on the same Mac with Q2_0: 56.6 tok/s decode on the story, and 887 / 809 tok/s on 4K / 32K prompts.

### Unsloth's UD-Q4/Q5/Q6_K_XL on the M5 Max

IQ3_S and UD-Q4_K_XL are the recommended sizes. Setup offers IQ3_S as the default and names UD-Q4_K_XL as the
higher-quality choice where it fits. Decode is tokens per second over 200 tokens (greedy, MTP); prompts are the warm rate
of the second request on (the first request after the start pays a few seconds of first-use cost).

| | IQ3_S | UD-Q4_K_XL | UD-Q5_K_XL (streamed) | UD-Q6_K_XL (streamed) |
|---|---|---|---|---|
| Quality against Q8_0 (top-1 / KL) | 87.9% / 0.142 | 92.8% / 0.055 | not measured | not measured |
| Decode, short question, no MTP / MTP | 75.4 / 101.9 | 60.7 / 94.0 | - | - |
| Decode, story, no MTP / MTP | 74.8 / 94.7 | 59.7 / 73.7 | 28 / 46-51 (`--spec 3`) | not measured / 47-51 (`--spec 3`) |
| Prompt 4K / 32K | ~1,180 / ~1,290 | ~1,150 / ~1,290 | ~1,070 / ~1,110 | ~1,020 / ~1,020 |
| llama.cpp Metal, decode (story / code) | 51.1 / 44.1 | 48.4 / 47.3 | 45.1 / 43.2 | does not run (Insufficient Memory) |
| llama.cpp Metal, prompt 4K / 32K | 791 / 730 | 821 / 739 | 729 / 693 | - |

- **UD-Q5 and UD-Q6 cannot stay resident:** their repacked file (103.9 and 114.8 GB) is bigger than the largest buffer Metal gives (86.6 GB), so the experts stream from the SSD into an arena of about 65-67 GB on the 128 GB M5 Max (about 68% of Q5's experts, 62% of Q6's; it was 80-86 GB before the footprint cap below, and the decode and prompt speeds did not change: Q6 story 51 tok/s, 15K prompt 1,020 tok/s at both sizes).
- **The cost is a fixed hand-shake per layer** (the engine reads the router's choice and makes sure the experts are in the arena), not the SSD. A window of 2 tokens costs 38 ms against ~17 resident, so a longer window pays: Q5 story 28 tok/s without MTP, 41.8 with `--spec 2`, 50.6 with `--spec 3`, 46 with `--spec 4`. Setup writes `--spec 3` and keeps the MTP draft layer for a streaming Mac with 96 GB or more.
- **The engine's whole anonymous footprint is limited to 60% of the RAM on a Mac with 64 GB or more (82.4 GB of the M5 Max's 137.4 GB): the arena, every Metal buffer and the host memory the process holds, not only the Metal buffers.** macOS keeps a floor of file cache (`vm.vm_page_filecache_min`: 37.2 GB here) and compresses anonymous memory instead of dropping cache below it. The arena is anonymous, and the GPU wires it only while a command buffer runs, so while it is filled it is ordinary pageable memory. Two kernel panics on the M5 Max (the watchdog daemon starved for 92 s) had the engine at 95 and 90 GiB resident and the compressor at 2-4 million pages. Measured again with a memory watchdog every second: an old-build arena of 100 GB compressed 19 GB of itself in 7 seconds and swapped 24,000 pages; the old automatic arena (69 GB) with a 40 GB file write and a 16 GB program beside it took the compressor from 21,000 to 4.07 million pages in 10 seconds (pressure "warn"). With 30% left (footprint 94.5 GB) and the file write alone, 12,000 pages were swapped out; with 35% left, 316; with 40% left and the write and the 16 GB program, none and the compressor at 0.3 million pages. The old rule (arena + Metal buffers under 67%) counted 7.5 GB of buffers where the process held 11-15 GB, so its real total was about 100 GB. A resident pack is different (clean file pages that macOS drops, never compresses): UD-Q4_K_XL resident (wired 91.7 GB) took the same 40 GB write and 16 GB program with no swap-out and nothing compressed, so resident packs are limited to 70% of the RAM as before. The arena is also wired (`mlock`) once it is filled: unwired it is ordinary pageable memory whenever the GPU is idle (the GPU's own wiring ends a few seconds after its last command buffer; a Metal residency set did not hold it), and a 20 GB file written beside the idle engine took the arena into the compressor and froze the Mac (a third watchdog panic, in a control run of this work). Wired, the same write at full speed raised the compressor by 0.7 million pages (the engine's non-arena memory), swapped out 408 pages and changed nothing else. `STRATA_METAL_WIRE_ARENA=0` turns it off; it is on for Macs with 64 GB or more. The engine prints the planned total (`planned resident footprint ... cap ...`) and the footprint after the fill. `STRATA_METAL_ARENA_GB` still sets the arena as you say, with a warning when it is over the cap.
- **The experts' copy** (`<model>.experts`, 98 GB for Q5) did not pay on the M5 Max: decode 50.3 against 50.2 tok/s without it, 4K prompt 741 against 685 (noisy). Setup does not write it on a Mac with 96 GB or more.
- **Prompts stream in chunks of 4096** on a streaming Mac with 96 GB or more (the engine's own default there; 1024 on a smaller one): every chunk makes one pass over the experts that are not in the arena, so a longer chunk reads them fewer times. UD-Q5, warm, 2 interleaved rounds: chunks of 1024 gave 700 tok/s on the 4K prompt, 2048 gave 920 and 4096 gave 1,100 (32K: 710, 970, 1,100); the decode did not change. `STRATA_METAL_PROMPT_CHUNK=N` sets it. On a resident model 4096 instead of 2048 gained only 2-4% (Q2_0), so that default stays.

### M4 Mac mini, 32 GB, experts streamed from the SSD

The arena was 11.5-14.7 GB. Tokens per second, one token per step (no MTP), medians of 5 interleaved runs:

| | Q2_0 | IQ3_XXS | llama.cpp, CPU only (Q2_0 / IQ3_XXS) |
|---|---|---|---|
| Decode, short question (200 tokens) | **14.1** | **10.0** | 6.3 / 4.1 |
| Decode, story (200 tokens) | **13.5** | **9.2** | 6.3 / 4.1 |
| Decode after a 6K-token document | **13.1** | **7.7** | 5.1 / 2.8 |
| Prompt, 4K tokens | **93.6** | **65.8** | 21.2 / 8.7 |

- The greedy answers equal the M5 Max's (model in memory) on 10 of 10 test prompts, for both packs.
- No memory was swapped out during the runs; the least memory left to macOS was 7.1 GB.
- MTP was slower here (Q2_0 story 12.1-12.3 with it against 12.3-12.8 without). The draft layer's 1.2 GB do more as expert cache.

**The experts' copy** (`<model>.experts`, one SSD request per expert): IQ3_XXS after the 6K-token document, the arena held at 12 GB, medians of 5 interleaved runs each:

| | Decode | The 6K prompt | SSD wait per token |
|---|---|---|---|
| The model file (three reads per expert) | 7.37 tok/s | 64.7 tok/s | 46.6 ms |
| The experts' copy (one read per expert) | **8.62 tok/s** | **84.1 tok/s** | 31.5 ms |

- The tokens are identical in all 10 runs.
- The 4K-token prompt (3 interleaved runs each, default arena): 72.9 tok/s with the copy against 62.5 with the model file (+17%), the same tokens.
- 32K-token prompts through the server, with the copy: 29,768 tokens at 89.2 tok/s (5.6 minutes to the first token), 33,803 tokens at 84.4 tok/s. The code word was found both times.
- A 34-minute server run with the copy (47 requests: 32K and 6K needles, short questions, a story, code; context 34,816) got every answer right and did not slow down over time. Decode was 7.2-7.7 tok/s on short answers and 9.9-10.4 on the story; the 6K prompts ran at 80-90 tok/s. macOS's memory pressure was never critical: one 2-second "warn" sample while the arena filled at the start. About 90 MB were swapped out in those same 2 seconds and nothing after; three restarts afterwards (smaller arenas, less memory free) swapped nothing.
- The SSD alone measures the same: 2.97 GB/s for one 1.75 MB read per expert against 2.30 GB/s for three 0.58 MB reads (6 threads, uncached).
- The arena's size matters most. With the model file, a 7.5 GB arena gave 5.5 tok/s, 10.5 GB gave 6.7 and 12 GB gave 7.5. The read-ahead depth, the I/O threads and the eviction order each moved decode by less than 3%.
- The Q2_0 copy is not measured yet. The numbers in the table above are with the model file.
- **Measured again on the release build, with less memory free:** the M4 had 13-17 GB available when the engine started (another user's programs held the rest), so the arena was 5.7 GB (Q2_0) and 9.3 GB (IQ3_XXS, with the experts' copy). One token per step, medians of 3: Q2_0 story 8.3 tok/s, 4K prompt 80; IQ3_XXS story 8.3, 4K prompt 89. That is the arena's size effect above, not a regression: with a 6.6 GB arena the engine used to stop at the first prompt ("no free slot"); a small arena now gives the prompt's 1.6 GB of buffers their own allocation, also when no chunk of the arena is big enough for them (on a 24 GB M2 whose arena shrank below the prompt's needs, every prompt stopped with "no arena region for the prompt buffers"), and a 26-minute server run with a 5.4 GB arena (29 requests, 6K and 31K needles found) was clean.
- With the default arena (12.1-12.2 GB here), the experts' copy gives 8.35-8.48 tok/s decode and 82.5-82.8 tok/s on the prompt. Nothing was swapped out. macOS's memory pressure stayed "normal", except for one 2-second "warn" sample while the arena fills; keeping 7 GB for macOS shows the same sample.
- Those runs and the server run below kept 5 GB for macOS. The default is now 7 GB with the stepped fill: after the 6K document, 7.3 and 6.7 tok/s decode (arenas 9.3 and 7.6 GB, as much as was free), the same tokens, the memory pressure "normal" throughout and nothing swapped out.

**Why `--spec 2`:** measured with MTP on the same two prompts.

| Window | Q2_0 (prompt1 / story) | IQ3_XXS (prompt1 / story) |
|---|---|---|
| `--spec 2` | 111 / 106 | 102 / 95 |
| `--spec 3` | 116 / 92 | 108 / 95 |
| `--spec 4` | 111 / 81 | 108 / 83 |
| `--spec 5-6` | slower | slower |

`--spec 2` is the best default on mixed text. An explicit `--spec N` in the config's `args` is kept.

### Gyro-S and Gyro-M (agentionai's trellis experts)

From 0.2.2 the Mac engine runs [Gyro](GYRO.md): the routed experts are trellis-coded blocks (ggml types TQ2_T, TQK6, TQK7) in a
Hadamard-rotated basis, read from the published file unchanged. The kernels are ours (`metal/kernels/quant.metal`, `moe.metal`,
`moe_mm.metal`); the codebook and the decoder are the CPU engine's, the rotation is applied to the gate/up and down inputs as on CUDA.
Setup offers Gyro-S and Gyro-M next to the other sizes (`./setup.sh --family gyro --model Gyro-S`), never as the default, and says
what the choice risks on your Mac's memory. The 28.8 GB n-gram table is inside the file and is read from the SSD like every other
pack's, so Gyro-S needs 58.5 GB of disk and about 38 GB of memory to stay resident (29.7 GB of weights).

**Measured on the M4 Mac mini, 32 GB** (the experts stream from the SSD; no experts' copy, because the copy would need 26 GB more
disk than this Mac had; one token per step, no draft layer; the arena was 11-12 GB of the 25.8 GB of experts):

| | Gyro-S |
|---|---|
| Decode, story, 120 tokens (5 interleaved pairs, medians) | **9.14 tok/s** (9.07-9.15) |
| Decode, with the dense-weight tiles turned off (`STRATA_METAL_GYRO_TILES=0`) | 8.70 tok/s (8.69-8.73); same tokens in all 10 runs |
| Prompt, 4,065 tokens (3 runs) | **109 tok/s** (108-110) |
| Memory the engine held | 18.6 GB of the 32 GB |

Where the time goes: a decode window of two tokens takes 111 ms. The routed experts are about 40 ms of it, the dense weights
(Gyro keeps its attention, state-space and shared-expert matrices in Q5_K and Q6_K: 3.6 GB read a token)
about 66 ms before the tile change (a run with those kernels skipped). The kernel for one expert layer (ten experts, one token) takes
278 us on the M4, 37.5 GB/s; IQ3_XXS takes 333 us and Q2_0 166 us. The trellis decode is limited by latency and ALU, not by memory:
32-bit funnel shifts for the stream bits, fetching the next unit while decoding the current one, one lane a row with the unit index
folded away and the codebook in constant memory were all tried and none beat the plain kernel, so it is the plain kernel with the
tile that measured fastest. The prompt kernels (fp16 matrix units) are on par with IQ3_XXS: 68 ms against 62 ms a layer for
1,024 tokens.

**Checked against the CPU engine's decoder, on the real file:** the decoder equals the format's definition written out bit by bit
(614,400 sampled blocks a layer, none differ); the Hadamard rotation equals the CPU's bit for bit (f32 and f16); the expert
kernels reproduce a double-precision reference to 1e-7 relative on real experts of layers 0 and 47 of Gyro-S and of layers 0, 23
and 47 of Gyro-M (the 92 GB file was read by range requests, 16 experts a layer: it does not fit this Mac's disk). Run it with
`metal/tests/gyro_parity.sh`.

**Against the CUDA engine, same file** (the pinned Gyro-S; `tools/quant_quality.py`, teacher-forced over the 5 text segments of
`data/quant_quality/v1`, 4,995 positions; CUDA = an RTX 3060 with the engine of this release, run twice: experts on the GPU, experts on the CPU):

| Reference -> run | top-1 agreement | mean KL | KL p99 | perplexity of the text |
|---|---|---|---|---|
| CUDA GPU experts -> CUDA CPU experts (the noise floor: two numerics of one engine) | 95.80 % | 0.0186 | 0.195 | 4.462 / 4.477 |
| CUDA GPU experts -> Mac | 95.46 % | 0.0207 | 0.241 | 4.486 / 4.477 |
| CUDA CPU experts -> Mac | 95.38 % | 0.0187 | 0.272 | 4.486 / 4.462 |

The Mac is as far from either CUDA run as the two CUDA runs are from each other (the CUDA GPU path quantizes the activations to q8_1 and
reads an int8 copy of the codebook, the CPU path an int16 one; the Mac multiplies fp16 codebook values with the float activations).

### Quality

- **The Q8_0 hyper-connection read is the default on Mac.** It keeps a Q8_0 copy of the hyper-connection matrices instead of BF16. `STRATA_HC_Q8=0` in the config's `env` gives the exact BF16 read.
- **How the KL was measured:** teacher-forced over 28 prompts (code, reasoning, chat, six languages, documents), about 6,000 positions per set. The reference is BF16.
  - On the model's own text: mean KL 0.0007, top-1 agreement 98.9%.
  - On another model's text: mean KL 0.0007, top-1 agreement 99.1%.
  - Perplexity of the text is unchanged (×0.9995 and ×1.0006).
  - For scale, IQ3_XXS against Q2_0 is a KL of 0.094.
  - The Mac engine is deterministic: two runs give identical numbers.
- **The fast prompt path** (the prompt kernels) against reading the prompt through decode windows: mean KL 0.0004, top-1 99.4-99.6%.
- **Needles:** a code word hidden in 8K, 16K and 32K tokens of text, at 10/50/90% depth. Found 9 of 9 with Q2_0 and with IQ3_XXS, both with Q8_0 and with BF16. With the fast prompt path (4K-32K) the answers are the same.
- **Same answers as on a PC:**
  - against the CUDA engine on a Tesla P100 (IQ3_XXS and IQ3_S), the first 200 tokens of a reply are identical on the short question;
  - on the story, the first IQ3_S tokens differ at token 1, a near-tie (margin 0.12). Following the CUDA engine's tokens, 4 of 200 picks differ, all at margins 0.06-0.21;
  - against the HIP engine on a Radeon AI PRO R9700 (Q2_0), 36 of 39 first tokens are identical.
  - The differences are near-ties between the two best tokens, from a different floating-point order.

## Limits (this version)

- **Images (vision):** on by default. The app and the Terminal use the `strata-vision` that sits beside `strata` in the engine package;
  `./setup.sh` builds it itself and asks "Should the model also read images?" (default yes). One program does both jobs: with `--gpu` it
  reads on the Mac's GPU (ggml's Metal backend, the shaders are built into the program), without it on the CPU. A Mac always gets the GPU
  reader, whatever its memory. The 24 GB rule of the PC cards does not apply: the Mac has no separate card memory to keep free, and the
  reader's footprint is about the same on either side (1.18 GB on the GPU, 1.05 GB on the CPU, mostly the 0.9 GB projector file).
  Measured on a Mac mini M4 (32 GB), Qwen3.8-Flash-Next's BF16 projector:

  | Picture | GPU reader | CPU reader |
  |---|---|---|
  | 260 tokens (640x400) | 0.45 s | 50.7 s |
  | 972 tokens (1600x1200) | 2.4 s | 194 s |

  `--vision cpu` is still allowed and prints that note. The CPU reader is slow because ggml has no vector kernel for BF16 on ARM (a build
  for newer chips, with the BF16 and int8 instructions, took the same 50.7 s). If the GPU reader cannot start, the server prints one line
  and starts the same program on the CPU. With Q2_0 streaming from the SSD on that M4, the STRATA 42 test picture (red circle, blue square,
  green triangle, the text "STRATA 42") was described correctly with both readers. Checked on one Mac only; the M5 Max is untested for pictures.
- **Context:** measured up to 32K (needles, KL). Setup recommends 32K below 96 GB and 64K from 96 GB. Up to the model's 262K can be set, but it is not measured.
- **One request at a time.** `--parallel` (batch slots) is not ported. Several clients queue.
- **Not ported:** the control vector (`--control-vector-scaled`) and the PC tuning (`--calibrate`).
- **Streaming:** measured at 32 GB only. The model's file must be on the internal SSD; an external disk is untested and likely much slower.
- **Families:** the original Qwen3.8-Flash-Next, Unsloth's UD-Q4/Q5/Q6_K_XL and agentionai's Gyro-S are measured (above). Setup repacks Unsloth's shards into one file for the GPU; a Gyro file is used as it is. Gyro-M runs on the same kernels (checked tensor by tensor) but no Mac here could hold it. The Swift and Coder releases are not set up on Mac yet, and Qwen3.6 does not run on the Mac engine.
- **The ready-made Mac engine is in the app and in the Terminal's engine component, not in the release zips.** The zips
  (for `START-HERE.bat` and `setup.sh`) have no Mac engine: `./setup.sh` builds it.

## Switches (for testing; all off unless set, the defaults are above)

| Variable | Effect |
|---|---|
| `STRATA_HC_Q8=0` | The BF16 hyper-connection read (exact) instead of Q8_0 |
| `STRATA_METAL_PROMPT=window` | Read the prompt through decode windows (the slow, exact fallback; ~190 tok/s) |
| `STRATA_METAL_PROMPT_CHUNK=N` | Prompt chunk size (default 2048) |
| `STRATA_METAL_MTP_Q8=0` | The draft layer's matrices in BF16 instead of Q8_0 |
| `STRATA_METAL_MTP_CHAIN=0` | One GPU submission per draft step instead of one per round |
| `STRATA_METAL_FUSE=0`, `STRATA_METAL_LAZY_FOLD=0` | Turn off two decode fusions (same tokens, slower) |
| `STRATA_METAL_PARTIAL=1`, `STRATA_METAL_ROUTE=1` | Two decode variants that measured no faster here (same tokens) |
| `STRATA_METAL_STREAM=1` / `=0` | Stream the experts from the SSD even when the pack fits / never (the default decides by size) |
| `STRATA_METAL_CONTIG=0` | Streaming: read the model file even when the experts' copy is there |
| `STRATA_METAL_ARENA_GB=N`, `STRATA_METAL_KEEP_FREE_GB=N` | Streaming: the arena's size, or the memory left to macOS when the arena is sized |
| `STRATA_METAL_LEAVE_SHARE=0.4` | Streaming, Macs with 64 GB or more: the share of the RAM left to macOS and its file cache when the footprint cap is set (for measuring) |
| `STRATA_METAL_WIRE_ARENA=0` | Streaming, Macs with 64 GB or more: leave the filled arena pageable (default: wired with mlock) |
| `STRATA_METAL_IGNORE_REPACK=1` | Start the engine although a model repack or setup's preparation is running beside it (the engine refuses by default: it froze this Mac) |
| `STRATA_METAL_KEEP_HOST_WEIGHTS=1` | Streaming: keep the host code's own copies of the weights (2-3 GB) instead of giving them back to macOS |
| `STRATA_METAL_TRACE=1`, `STRATA_METAL_PROFILE=1`, `STRATA_METAL_KPROF=T` | Per-window traces, a time breakdown, per-kernel GPU times |

**Two opt-ins an M1 Max cannot run (measured on an M1 Max, macOS 26.6.2, Q2_0):** `STRATA_METAL_PROMPT_EXPERTS=v3` - the
prompt's expert GEMMs with the M5's layout, `mm_gu3` - loses the threadgroup once a prompt spreads over more than 40
threadgroups of it (35 pass, 40 fail; 190 tiles and the engine's tile both fail, one tile passes), so it is off there (the
default is v2). `STRATA_METAL_ROUTE=1` - the router GEMV and `moe_route` fused into one dispatch - routes with stale
logits and gives a different top-10 per run (two identical runs differ in ids, weights and plan; the logits themselves
stay bitwise): the fusion hands the routing to the last threadgroup of the GEMV through a device counter, and this
Metal's device atomics are relaxed-only, with no acquire to pair with the writers' fence. Both are off by default. The
`no faster` measurement in the table is from the Macs above; on an M1 Max `ROUTE` is also wrong.

## How it works

The code is in `metal/` (the engine's side in `metal/src/engine`, the kernels in `metal/kernels`). In short:
- **Host code:** the PC host code compiles unchanged for macOS. `metal/include/metal_compat` maps the CUDA runtime calls it makes onto a small Metal runtime (`metal/src/gpu`).
- **The decode and prompt work** run in Metal kernels (`metal/kernels`, `metal/src/engine`):
  - one command buffer per verify window;
  - the experts are read in place from the mapped model file;
  - the hyper-connections, the GDN state and the QSA attention stay on the GPU;
  - a whole MTP draft round runs in one command buffer.
- **The PC builds are byte-identical** with this code present. Every shared file's change is behind `STRATA_GPU_METAL` or `__APPLE__`, and the CUDA and HIP objects are compared after each change.
