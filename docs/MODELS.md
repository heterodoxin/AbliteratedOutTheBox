# Which model? Sizes, versions and what fits

Strata's main model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), in several **sizes**
(the same model, compressed more or less) and several **versions** (the original, a coding version, a fine-tune, other
makers' compressions). It also runs a smaller model, [Qwen3.6-35B-A3B](#qwen36-35b-a3b). The desktop app, the Terminal
(`strata launch`) and the installer recommend one for your PC; this page explains the choice. Back to the
[README](../README.md).

> **On this page:** [Pick by memory](#pick-by-memory) · [Speed](#how-fast-is-each-size) · [The sizes](#the-sizes) ·
> [Will it fit?](#will-it-fit) · [The versions](#the-versions) · [Qwen3.6](#qwen36-35b-a3b) ·
> [Adding another model](#adding-or-switching-models)

## Pick by memory

The tiers count RAM and VRAM (the graphics card's memory) together, the same as the [README](../README.md#the-other-sizes).
`strata launch`, setup and the install tool of the MCP server pre-select by them; the desktop app does from 0.2.1. Every
other size stays a choice. Strata leaves 5 GB for Windows and the system, so the sum is RAM + VRAM minus 5 GB: a PC with a card of 12 GB and 32 GB of RAM is in the 40 GB tier (32 + 12 - 5 = 39), and a PC with a 16 GB card and 32 GB of RAM is too, not in the 48 GB tier (#3425).

| Your memory (RAM + VRAM) | Pre-selected | Notes |
| --- | --- | --- |
| **32 GB** | **Gyro-S**, or the **Coder** for code | The Coder needs 32 GB of RAM and keeps half of the experts, so it is weaker outside code ([why](#coder)). Gyro-S has lower quality than the 2-3-bit sizes, and the 2-bit Q2_0 decodes faster |
| **40 GB** | **Swift 1.5** IQ2_XS (39.2 GB) | Gyro-M (35 GB) is the other choice: close to IQ3_XXS in quality, but it writes answers 16-21% slower than Q2_0. Gyro runs on NVIDIA, AMD and Intel Arc cards and, from 0.2.2, on Macs (Gyro-S measured on a 32 GB Mac mini) |
| **48 GB** | **IQ3_XXS** (47.0 GB) | Slower, better |
| **64 GB** | **IQ3_S** (54.8 GB), the original. Swift 1.5's IQ3_S is an equal choice | The slowest, the best: it matches the full model on the published tests |
| **96 GB or more** | [Unsloth's UD-IQ4_XS to Q8_0](#unsloths-files-ud-iq4_xs-to-q8_0), 4 to 8 bit | UD-IQ4_XS keeps every expert in RAM from about 80 GB of RAM, Q8_0 from about 136 GB. Up to 192 GB, the 5-8-bit files fit in RAM. Setup still pre-selects IQ3_S here, except on a Strix Halo |

With less memory than a row says, Strata's [low-RAM mode](#a-big-graphics-card-and-little-ram) starts by itself and the
model runs slower, much slower with a small card. The figures above are RAM + VRAM; [Will it fit?](#will-it-fit) counts
RAM alone, for the sizes that keep their experts in RAM.

Not sure? Take what the installer pre-selects. The Coder fits a 32 GB PC, but it keeps only 256 of the 512 experts,
chosen on code data, so it is weaker outside code and in languages other than English. For general use, take a full
model (Q2_0, IQ2_XS or IQ3_S) whenever your memory allows.

## How fast is each size

These are older numbers, kept because they compare the sizes with each other. Newer engines are faster on some cards
(each change is listed with its own measurement in the [details](DETAILS.md)); the 0.2.0 engine (0.1.45.3) has not been
re-measured on the whole table. Measured on an RTX 5070 (12 GB), a Ryzen 5 7600 and 64 GB of RAM, engine 0.1.26 (the
Q2_0 row: engine 0.1.36, which made Q2_0 prompts faster):

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt |
| --- | ---: | ---: | ---: |
| **Q2_0** | 94 tokens/s | 76 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 63 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 49 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 46 tokens/s | 1,620 tokens/s |
| **Coder** (IQ1_M) | 55 tokens/s | 43 tokens/s | 2,180 tokens/s |

Measured on an AMD RX 9070 XT (16 GB), a Ryzen 9 3900X and 47 GB of RAM (Linux, setup's own install, 2026-10-02, an engine
from before 0.1.40; Q2_0 is above setup's RAM estimate for 47 GB and was installed with `--model Q2_0 --yes`):

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt |
| --- | ---: | ---: | ---: |
| **Q2_0** | 60 tokens/s | 48 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 36 tokens/s | 1,110 tokens/s |
| **Coder** (IQ1_M) | 44 tokens/s | 33 tokens/s | 1,420 tokens/s |

- **Writes answers** = how fast the reply appears (tokens per second; a token is about ¾ of a word).
- **Reads your prompt** = how fast it takes in what you send (long documents, code, chat history), measured on a
  32K-token prompt; a 4K prompt reads at 910-1,580 tokens/s. A 32K prompt takes about 15 seconds with Q2_0.

A card with more VRAM is faster, because more of the model fits on the GPU. For an RTX 3090 (24 GB) the
[details](DETAILS.md#other-gpus-estimated) give an estimate, not a measurement. All measurements, long-context numbers
and estimates for other cards are in the [details](DETAILS.md#speed-measured). The AMD measurements per card (RX 9070 XT,
Radeon AI PRO R9700, RX 7800 XT, RX 9060 XT, RX 6900 XT) are in [AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201).

Every PC is different: `START-HERE.bat --calibrate` measures a few engine settings on yours and keeps the fastest
(about 15-30 minutes, longer on a slow card; on the PC above it made the Coder 7% faster; NVIDIA cards for now).
`strata launch` and the app have their own tuning step (quick: about 20 minutes). Measured Strata on your own
PC? See [Community benchmark results](COMMUNITY_BENCHMARKS.md) for a report template and how to share your results
in a pull request.

## The sizes

| Model | RAM+VRAM Requirements | Speed | Quality |
| --- | ---: | --- | --- |
| **Q2_0** | 37.6 GB | fastest | good |
| **IQ2_XS** | 39.2 GB | fast | better (**recommended** at 40 GB, as Swift 1.5) |
| **IQ3_XXS** | 47.0 GB | slower | great |
| **IQ3_S** | 54.8 GB | slowest | best: matches the full model on the published tests |

The download is 66-76 GB for the three smaller sizes ([details](DETAILS.md#which-model)); the first start also
fetches the MTP draft layer (~6 GB, +1 GB with images).

## Will it fit?

Shard 1 is the part of the model that gets loaded when it starts: its experts go into your **RAM**, the rest onto
your graphics card (the second shard, a 29 GB lookup table, stays on the SSD). So it fits when your **RAM is at least
the experts + about 10 GB** for Windows and your other programs - the experts are most of shard 1: 34 GB for Q2_0,
35.5 for IQ2_XS, 43 for IQ3_XXS, 50 for IQ3_S, 23 for the Coder (the dense weights in shard 1 go to the graphics
card). With 64 GB of RAM every size fits (IQ3_S with little else open); with 48 GB, Q2_0 and IQ2_XS. A bigger graphics card makes it faster, but it doesn't lower the RAM needed
- except in the low-RAM mode below.

### A big graphics card and little RAM

On a PC whose RAM cannot hold the experts beside the system, setup maps them from the model's files instead and keeps
in RAM only what the graphics card does not hold (the [low-RAM mode](DETAILS.md#speed-measured), chosen by setup). For
example, a 32 GB PC with a 24 GB GPU runs Q2_0, IQ2_XS and the Coder this way, and a 32 GB PC with a 12-16 GB GPU the
Coder. With a small card most experts then come from the SSD and it is much slower (setup says so).
`START-HERE.bat --setup --low-ram on|off` overrides the choice.

## The versions

**Pictures.** The versions in setup's menu read pictures, and pictures are on by default wherever a model can
read them. The picture reader runs on the CPU when the graphics card has less than 24 GB (a few seconds per
picture, nothing taken from the card) and on the graphics card from 24 GB (about 1-1.4 GB of VRAM; 0.1-0.5 s per
picture). The rule is the same on NVIDIA and AMD cards. `--vision off` (or "No" in the app and in `strata launch`) turns
pictures off; `--vision cpu` or `--vision gpu` picks the reader. On AMD, `strata-vision-hip` is the card's reader; if it
cannot start, Strata uses the CPU reader by itself ([AMD_HIP.md](AMD_HIP.md)). The Mac reads pictures on its GPU
([MAC.md](MAC.md)). On an Intel Arc the rule is the same, with two differences: an Arc of 24 GB or more keeps the
CPU reader unless you ask for `--vision gpu` (the Intel GPU reader has only run on an 8 GB A750), and a Battlemage card (B580, B70)
has pictures off until you pass `--vision yes` ([INTEL.md](INTEL.md#pictures-on-an-arc-2026-10-19-branch-f-021-arcvis)). More: [Images](DETAILS.md#images-vision).

### Qwen3.8-Flash-Next

The original, in all four sizes. With images, and with the
[experimental speed projection](DETAILS.md#experimental-speed-projection-experimental-off-by-default) as an option.

### Coder

**[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** - ISTA-DASLab's coding version:
it keeps 256 of each layer's 512 experts, chosen on code (with agentic and image) data, and drops the rest (91% of
the full model's SWE-bench Verified score, 99% of LiveCodeBench, by its authors). One size (IQ1_M: its experts stored
like IQ3_S): shard 1 is **29.6 GB**, so it fits a PC with **32 GB of RAM**, runs 262K context on 64 GB, and reads long
prompts the fastest of all. With half the experts gone it is weaker outside code and in languages other than
English, Chinese and other CJK text included (#438: Chinese answers came out wrong or looping where English was
fine). For general use, or whenever your RAM allows, take a size that keeps every expert: Q2_0, IQ2_XS or IQ3_S.
More: [details](DETAILS.md#or-the-coder-half-the-experts-for-code).

```
START-HERE.bat --setup --family coder
```

### Swift 1.5

**[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** - a fine-tune by UkisAI that
thinks much shorter before answering, so you get the answer sooner, with about the same quality. Same speed per
token, and about the same RAM as the same size of the original. Its own license applies (see its page).
More: [details](DETAILS.md#or-swift-15-a-fine-tune-that-thinks-shorter).

```
START-HERE.bat --setup --family swift --model IQ2_XS
```

<a id="unsloth-ud-iq4_xs"></a><a id="unsloth-ud-q4_k_xl-experimental"></a>

### Unsloth's files: UD-IQ4_XS to Q8_0

The fourth version in setup's menu (`--family unsloth`) is **Unsloth's quantizations** of the same model, from ~4 to 8
bits. They keep the original's experts at more bits than the 2-3-bit sizes above, so they need more RAM. Setup keeps
every expert in RAM where they and 16 GB more fit (the fastest way); with less RAM it keeps as many as fit and reads
the rest from the SSD while it answers (much slower), and for the bigger files it asks first. Pictures are on by default
(the CPU reader under 24 GB of graphics memory, the card's reader from 24 GB).

| Size | Download | Experts | RAM for all of them in RAM | Same top token as Q8_0 | AMD R9700 (32 GB): story / code / 32K prompt |
| --- | ---: | ---: | ---: | ---: | --- |
| **UD-IQ4_XS** (setup's first) | 94 GB | 55 GiB | ~72 GB | 88.9% | - |
| **UD-Q4_K_XL** | 111 GB | 72 GiB | ~88 GB | 93.1% | 63 / 87 tok/s / 1,110-1,320 tok/s |
| **UD-Q5_K_XL** | 158 GB | 92 GiB | ~108 GB | 94.6% | 57-60 / 82 / 810-1,260 |
| **UD-Q6_K_XL** | 169 GB | 102 GiB | ~118 GB | 94.6% | 54 / 76 / 1,020 |
| **Q8_0** | 188 GB | 120 GiB | ~136 GB | 100% (the reference) | 53-57 / 66-74 / 1,230-1,300 |

For comparison, IQ3_S on the same card: 75 / 95 tok/s, 87.7% the same top token as Q8_0. The speed figures come from
engines older than 0.1.45.3 (the conditions are in [QUANTS.md](QUANTS.md)). The quality numbers, the
speed bound and NVIDIA/AMD/Intel details: [QUANTS.md](QUANTS.md); the files: [UD-Q4_K_XL](UNSLOTH_Q4.md),
[UD-Q6_K_XL](UNSLOTH_Q6.md).

```
START-HERE.bat --setup --family unsloth --model UD-Q5_K_XL
```

<a id="gyro"></a>

### Gyro-S and Gyro-M (agentionai)

The fifth version in setup's menu (`--family gyro`) is **agentionai's Gyro quantizations** of the same model: trellis-coded
experts in a rotated basis. They are the smallest options and never setup's default. The download includes the
28 GB n-gram table; the memory figure is VRAM and RAM together.

| Size | Download | Memory it needs (VRAM + RAM) | In plain words |
| --- | ---: | ---: | --- |
| **Gyro-S** | 58.5 GB | 27.6 GB | the smallest option, for PCs with little memory; lower quality |
| **Gyro-M** | 92.0 GB | 35 GB | about IQ3_XXS quality with about 20% less GPU memory; decodes 16-21% slower than Q2_0 (RTX 3060 12 GB, R9700) |

Gyro runs on NVIDIA and AMD (experimental) cards and on Intel Arc cards on Linux (tested on an Arc A750; the Arc engine is built from this release's source, see [INTEL.md](INTEL.md)) and, from 0.2.2, on Apple Silicon Macs ([MAC.md](MAC.md#gyro-s-and-gyro-m-agentionais-trellis-experts): Gyro-S is measured on a 32 GB Mac mini, Gyro-M only tensor by tensor). Gyro reads pictures (the same rule as above). On NVIDIA and AMD it needs
engine 0.1.45.1 or newer; if the engine is older, the app says "This model needs a newer engine" and offers Update
engine. Gyro gives correct output on RTX 50 cards with engine 0.1.45.3. Details, measurements and credit:
[GYRO.md](GYRO.md).

```
START-HERE.bat --setup --family gyro --model Gyro-S
```

<a id="nvfp4"></a>

### NVFP4: three versions (NVIDIA cards)

The sixth version in setup's menu (`--family nvfp4`) keeps the experts in NVIDIA's 4-bit NVFP4 format. There are three
versions of it. The app shows them by who made the weights:

| App name | Setup name (`--model`) | What it is | Download |
| --- | --- | --- | ---: |
| **NVFP4 (Strata's version)** | `NVFP4` | Strata's own conversion (GPTQ; the ready-made pack is sergqwer's strata-nvfp4). The best quality of the three | 128 GB |
| **NVFP4 (NVIDIA's version)** | `NVFP4-NVIDIA` | NVIDIA's own NVFP4 checkpoint (plain rounding), converted for Strata. The same size and speed as the first | 128 GB |
| **NVFP4 (NVIDIA, trained for FP4)** | `NVFP4-QAD` | NVIDIA's retrained 4-bit model, slightly faster on short prompts ([below](#nvfp4-qad)) | 102 GB (+110 GB of disk to prepare) |

All three need about 80 GB of RAM to keep every expert in RAM, run on NVIDIA cards only, and read pictures (the
same rule as above). RTX 50 cards have FP4 tensor cores; other NVIDIA cards run NVFP4 at 8-bit precision. The first two
need engine 0.1.44 or newer, the third 0.1.45 or newer. Details and measurements: [NVFP4.md](NVFP4.md).

```
START-HERE.bat --setup --family nvfp4 --model NVFP4
```

<a id="nvfp4-qad"></a>

#### NVIDIA's retrained 4-bit model (NVFP4-QAD)

NVIDIA retrained Qwen3.8-Flash-Next for 4-bit experts (quantization-aware distillation) and published it as
[Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local). In setup's NVFP4 family it is
the size **NVFP4-QAD**. It is not a ready-made pack: setup downloads NVIDIA's files as they are (102 GB) and **turns them
into a Strata pack on your PC** (once, 20-40 minutes; about 110 GB more disk while it runs, and the 102 GB can be deleted
afterwards). Nothing of NVIDIA's weights is redistributed by Strata.

| Size | Download | Memory it needs | In plain words |
| --- | ---: | ---: | --- |
| **NVFP4-QAD** (app: NVFP4 (NVIDIA, trained for FP4)) | 102 GB (+110 GB of disk to prepare) | like NVFP4: ~80 GB of RAM keeps every expert in RAM | NVIDIA's FP4, trained for FP4 - slightly faster on short prompts than the other two NVFP4 versions (Strata's version has the best quality) |

NVIDIA cards only. It reads pictures (through the original model's picture reader file, which setup downloads with it). Its licenses are the NVIDIA Open Model Agreement and Qwen's Community License.
Technical details and measurements: [NVFP4.md](NVFP4.md#nvidias-fp4-local-checkpoint).

```
START-HERE.bat --setup --family nvfp4 --model NVFP4-QAD
```

### OrcaRouter Uncensored IQ3_XXS

For **OrcaRouter's Flash-Next Uncensored IQ3_XXS**, see the [manual compatibility setup](ORCA.md). It needs an
explicit packing conversion and is not an installer menu option.

## Qwen3.6-35B-A3B

A smaller model from the same maker: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35 billion
parameters, 3 billion active per token, 256 experts, Apache 2.0 license). Strata runs [Unsloth's GGUFs with the MTP
draft layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) on its own engine path, chosen by the file. It is
in the app's model list and in setup (`--family qwen36`). It runs on NVIDIA and AMD cards; the Intel Arc engine and the
Mac engine do not run it. It reads pictures and serves several requests at once.

| Size | File | RAM setup plans for | In plain words |
| --- | ---: | ---: | --- |
| **UD-Q4_K_XL** | 22.9 GB | 32 GB | 4-bit (Unsloth Dynamic), the smallest |
| **UD-Q6_K_XL** | 32.6 GB | 44 GB | 6-bit, closer to the full model |
| **Q8_0** | 37.8 GB | 52 GB | 8-bit, the closest to the full model |

The picture reader file adds 0.9 GB. The engine keeps every expert it can on the graphics card and computes the rest on
the CPU. Measured with engine 0.1.45 and UD-Q4_K_XL: decode 117.8 tokens/s on a Radeon AI PRO R9700 (32 GB) and 59.7
tokens/s on an RTX 3060 (12 GB, most experts on the CPU). Pictures, requests at once, the full measurements and the
limits: [QWEN36.md](QWEN36.md).

```
START-HERE.bat --setup --family qwen36 --model UD-Q4_K_XL
```

## Adding or switching models

- **In the app:** the Models page lists every model with its size and whether it fits your computer; pick one and press
  the install button.
- **In the Terminal:** `strata launch` has a Model step with the same list (and "Use model files I already have").
  `strata models list` shows what is installed; `strata models import <folder or .gguf> --plan` shows what a set of files
  you already have would be, without changing anything.
- **With the installer:** `SETUP.bat` (the same as `START-HERE.bat --setup`; on Linux `./setup.sh --setup`) adds another
  model any time.

Files another model shares are not downloaded again (the Coder uses the original's shard 2 and vision encoder). With more
than one model installed, `START-HERE.bat` asks which one to start; `run-<model>.bat` (Linux: `run-<model>.sh`) starts one
directly.
