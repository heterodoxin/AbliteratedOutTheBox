# AMD Radeon: the HIP backend (gfx1100, gfx1101, gfx1200, gfx1201, gfx1030)

Strata runs on AMD Radeon cards through its HIP backend, the same engine as on NVIDIA compiled for AMD. This page
covers the build on Linux (on Windows a ready-made engine, see [Windows](#windows)) for the RX 7900 XT / XTX (RDNA3, gfx1100) and the
RX 9070 / 9070 XT / Radeon AI PRO R9700 (RDNA4, gfx1201; see [RDNA4](#rdna4-gfx1201)). The RX 7800 XT / 7700 XT
(gfx1101) and the RX 9060 XT (gfx1200) were validated by their owners (see [Community-validated
cards](#community-validated-cards)); the RX 6800 / 6900 series (RDNA2, gfx1030) builds and runs too, reported by a community machine and not yet validated by the maintainers (see [RDNA2](#rdna2-gfx1030)). Setup chooses it by itself on a PC with no NVIDIA card Strata can use (`--backend hip` on a PC with both); the
install steps for users are in [INSTALL.md](INSTALL.md#amd-cards). gfx906 (Instinct MI50 / MI60, Radeon VII; wave64) has a separate
opt-in build, see [gfx906](#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source). The Ryzen AI Max "Strix Halo" APU (gfx1151, RDNA3.5, unified memory) has its own page: [STRIX_HALO.md](STRIX_HALO.md). The ready-made 0.2.0 installers and engine packages for AMD cards are listed in [Ready-made packages](#ready-made-packages-020). Other AMD architectures and mixed
AMD/NVIDIA execution in one run are not supported.

The backend maps the CUDA-shaped runtime and BLAS calls to HIP/hipBLAS, uses
RDNA2/RDNA3/RDNA4's signed integer dot instruction for quantized kernels, and supplies
wave32 shuffle/packed-byte operations. CUDA-only QSA matrix instructions have
an ordered FP32 fallback. Prefill supports both dequantization plus hipBLAS GEMM and opt-in HIP ggml MMQ.
An optional, calibrated hipBLASLt path accelerates dense projections (per-architecture tables in `tools/hip`).
This does not claim bit-identical model answers across backends. See
[performance settings and evidence](AMD_HIP_PERFORMANCE.md).

## Install with setup (recommended)

On Linux with an RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT or Radeon AI PRO R9700 and
the kernel's amdgpu driver (no ROCm install needed):

```sh
./setup.sh --backend hip
```

- **Detection:** setup finds the card through the kernel's KFD topology. Integrated Radeon GPUs are listed as not
  supported. On a PC without an NVIDIA card Strata can use, `--backend hip` is chosen automatically.
- **ROCm:** a system ROCm 7 in `/opt/rocm` (or `$ROCM_PATH`) with hipcc and hipBLAS is used when present. Otherwise
  (or when it is older than 7.0) ROCm is installed into `.venv` from AMD's TheRock wheels (~10 GB, no sudo), from the
  card family's index: `gfx110X-dgpu` for gfx1100 / gfx1101, `gfx120X-all` for gfx1200 / gfx1201, `gfx103X-all` for
  gfx1030, `gfx1151` for Strix Halo. The version is chosen per family: 7.10.0a20251120 where that is on the index,
  7.13.0a20260515 for `gfx103X-all` (its index starts at 7.13, #1103) and 7.14.0a20260608 for `gfx1151` (the 7.10 wheel
  segfaults on kernel 7.2.8, #1267). When the index no longer offers the version, setup warns and takes the newest of
  the same 7.x line, else of the same major. `STRATA_ROCM_VERSION` / `STRATA_ROCM_INDEX` override them; the gfx1030
  wheels are unvalidated (a system ROCm 7 is the tested path there).
- **Engine:** compiled on your PC for the card's architecture (10-20 minutes, once; again after a `git pull` that
  changes it, or when you pick a card of another architecture). This needs a C++ compiler and git
  (`sudo apt install build-essential git`).
- **hipBLASLt tuning table:** setup uses `tools/hip/<arch>-hipblaslt-<version>.txt` only when it matches both the
  card's architecture and the installed hipBLASLt version (read from `hipblaslt-version.h`; 1.2.0 is `100200`).
  Otherwise it says so and the prompt's dense matrix products use plain hipBLAS (slower prompts, same answers).
  A table's solution ids are valid only for that pair, and the engine refuses any other table. On gfx103x (RDNA2,
  no hipBLASLt kernels) the same role falls to a rocBLAS table, `tools/hip/<arch>-rocblas-<version>.txt`, which the
  engine picks for its own rocBLAS build when setup points it at the directory ([RDNA2](#rdna2-gfx1030)).
- **Several cards:** setup takes one card (the one with the most VRAM, or `--gpu N`) unless you name more:
  `./setup.sh --backend hip --gpus 1,0` splits the model's layers across them, the faster card goes last and is the Main card (`--gpu-roles 1=main,0=helper` says it yourself; docs/MULTI_GPU.md) (numbers
  as setup lists them; `--gpus all` = every supported card, the most VRAM first). Every chosen card must be one of the
  architectures above; the engine is compiled for each of them (cards of two families, e.g. gfx1100 + gfx1201, need
  a system ROCm 7: AMD's wheels hold one family). A split pays only when no single card holds the model's experts
  (see RDNA4 below).
- **Pictures:** on by default; the reader is the CPU one under 24 GB of VRAM and the GPU one from 24 GB (the same rule on NVIDIA and AMD, `strata_control/vision_reader.py`); `--vision gpu` / `--vision cpu` switch it. The engine package carries two picture readers: `strata-vision-hip`, which reads a picture on the
  card (llama.cpp's HIP backend; about 1.4 GiB of VRAM, measured on an R9700: 0.06 s for a 260-token picture and 0.4 s for 972
  tokens, against 3.1 s and 14 s on the CPU), and `strata-vision`, the CPU reader. If the GPU reader cannot start (the card is
  not one it was built for, the HIP runtime does not load, the card is busy) the server starts the CPU reader by itself and
  prints one line; nothing is asked. `--vision cpu` picks the CPU reader on purpose.
- **Limits for now:** Setup does not offer the tuning
  (calibration) on AMD yet: its controls are being checked on HIP one at a time (#566). Since 0.1.39 a tuning run by
  hand (`./setup.sh --calibrate`) is saved for the AMD card it ran on and reused when setup runs again. The Monitor
  shows the card's load, VRAM, temperature and power from Linux sysfs (0.1.32).

The rest of setup is the same as on NVIDIA: the model download, the start script, the server.

## Ready-made packages (0.2.3)

The 0.2.3 release carries one AMD engine package per card family. The Terminal one-liner and `strata update` pick the package whose
GPU code covers the card; the full installers carry one package each. Any app installer also works: install it, press **Update engine**
in Settings (or run `strata update`) and the matching package downloads.

| Package (variant) | OS | GPU code inside | ROCm | Installer |
| --- | --- | --- | --- | --- |
| `rx9070` | Windows | gfx1200, gfx1201 | 10.2.0a20260930 (TheRock) | `Strata-0.2.3-windows-hip.exe` |
| `rdna3` | Windows | gfx1100, gfx1101, gfx1102, gfx1151 | 10.2.0a20260930 | `Strata-0.2.3-windows-hip-rdna3.exe` |
| `rdna2` (0.2.2) | Windows | gfx1030, gfx1031 | 10.2.0a20260930 | no installer of its own: any app installer, then Update engine |
| `rx9070` | Linux | gfx1200, gfx1201 | 7.10.0a20251120 (`gfx120X-all` wheels) | `Strata-0.2.3-linux-amd-rx9070.AppImage` |
| `rx7900` | Linux | gfx1100, gfx1101, gfx1102 | 7.10.0a20251120 (`gfx110X-dgpu` wheels) | `Strata-0.2.3-linux-amd-rx7900.AppImage` |
| `halo` | Linux | gfx1151 | 7.14.1 (TheRock) | `Strata-0.2.3-linux-amd-strixhalo.AppImage` |
| `rx6000` (0.2.2) | Linux | gfx1030, gfx1031 | 7.13.0a20260515 (`gfx103X-all` wheels) | no installer of its own: any app installer, then Update engine |

`rdna3` and `halo` were added to the release after the first upload (feed sequence 101). Why the split: on Windows one ROCm 10.2 library
set holds the kernels of every listed architecture, so RX 7000 and Strix Halo share a package. The Linux 7.10 wheels hold one chip family
per library set, and 7.10 crashes in the HSA runtime on a Strix Halo with a new kernel (#1267), so Linux Strix Halo has its own package on 7.14.1.

Each package carries the engine, the CPU picture reader and the GPU picture reader (`strata-vision-hip`) for the same architectures, and
the rocBLAS and hipBLASLt kernel data for them only. A package whose GPU code does not cover the card is refused with one line that
names the card family ("This build of Strata is for AMD RX 9000 cards. This computer has an AMD RX 7000 card ...").
**The RX 6000 series (gfx1030 / gfx1031, RX 6800 / 6900 / 6700 XT) has its own package from 0.2.2** (below). The 0.2.1 packages have no code for it,
and the app said so plainly instead of naming a build to ask for.

**RX 6000 package (0.2.2): built, not tested on a real RX 6000 card.** `rx6000` on Linux (ROCm 7.13.0a20260515, the `gfx103X-all` wheels
setup already uses for these cards) and `rdna2` on Windows (ROCm 10.2.0a20260930), each with the code for gfx1030 and gfx1031 only. It is a niche,
opt-in build: the feed offers it to a PC whose card reports gfx1030 or gfx1031 and to no other (a PC with an RX 7000 / 9000 card beside an RX 6000
one keeps the package it always had), and the RX 7000 / 9000 / Strix Halo packages are unchanged. It runs the plain
kernels: RDNA2 has no matrix cores, so none of the WMMA paths, the gfx12 prompt defaults or the 780M prompt set apply (they are chosen by the card's
architecture, `src/core/arch_defaults.cpp`), and hipBLASLt ships no gfx1030 kernels, so there is no tuning table and the dense matrix products use plain
hipBLAS. Linux build: `VARIANT=rx6000 packaging/build_hip_linux.sh`; Windows: `STRATA_HIP_ARCHS="gfx1030;gfx1031" tools\hip\build_windows.bat`, then
`package_engine.py --variant rdna2`. The [RDNA2 section](#rdna2-gfx1030) has what community owners measured with engines built from source. Please
report a run on an RX 6000 card with the end of `strata-<model>.log`.

**Automatic card (0.2.1):** with no card named in the model's settings, the engine start asks the package's `strata-device --list-devices`
and skips cards the package has no code for and integrated chips (a mini PC's Radeon 760M / 890M is HIP's device 0) while a card of its own
can run. Device 0 stays the choice when it is already such a card. When none of the computer's cards can run, the start stops
with one line that names the cards found and what the engine is built for. `HIP_VISIBLE_DEVICES`, a `"gpu"` in the settings file or `hip_ordinal` is never overridden.

What was run on real hardware:

- **Checked on every package** (no RX 7000 or Strix Halo card involved): it builds and links; every library the binaries import is in the
  package or belongs to the OS (`packaging/binary_deps.py`); `package_engine.py --verify` passes; the GPU code inside each binary and
  library lists exactly the architectures above (`llvm-objdump --offloading`); `strata --version` runs in a clean environment (on Windows with the package's `rocm\bin` folder on PATH, which Control sets); on a
  gfx1201 card the Linux `halo` and `rx7900` builds refuse to start with the plain message (the Windows `rdna3` build was not run on an RX 9000 card: there is no Windows AMD machine); the feed picks the right package for each
  architecture (`tests/terminal/test_release_matrix.py`).
- **Not run by the maintainers:** any model on an RX 7000 card (gfx1100-1102) or a Strix Halo (gfx1151), on either OS. The maintainers have
  no such card.
- **Run by other people:** an RX 7900 XTX on Windows 11 ran the earlier ready-made Windows engine (0.1.40; [report #1173](../bench/results/2026-10-06-community-rx-7900-xtx/README.md));
  RX 7900 XTX, XT and GRE owners ran Linux builds ([COMMUNITY.md](../bench/results/COMMUNITY.md)); Strix Halo owners ran Linux builds made from source
  (ROCm 7.14 line). Nobody has reported a Strix Halo on Windows, or the packaged `halo` and `rdna3` engines.
- **Tuning tables (0.2.1):** each AMD package carries the hipBLASLt tuning tables of `tools/hip` that match the hipBLASLt it bundles, as
  `tuning/<arch>-hipblaslt-<version>.txt` (listed and hashed in the package manifest), and records that version as `hipblaslt_version` in its
  BUILD.json. Windows `rx9070` has the gfx1201 table, Windows `rdna3` the gfx1100 and gfx1151 ones (all hipBLASLt 1.5.0), Linux `rx7900` the
  gfx1100 one (1.2.0), Linux `halo` the gfx1151 one (1.4.1). The RX 6000 packages (`rx6000`, `rdna2`, 0.2.2) have none, because hipBLASLt has no gfx1030 / gfx1031 kernels. Linux `rx9070` has none: its wheels carry hipBLASLt 1.2.0 and no table in `tools/hip`
  is for that version (a table calibrated for it on an R9700 read prompts about 1% faster and decode 2% slower, so none is shipped).
  The engine start picks the table for the card and uses it only when the package's hipBLASLt version is the table's; otherwise it logs
  why and the dense matrix products stay on plain hipBLAS. **A table changes which GEMM algorithm runs the prompt, so the rounding differs: the
  answers are not bit-identical to plain hipBLAS** (first-token KL 0.016 on an R9700, see below). Packaged engines therefore use the table by
  default only on Strix Halo (gfx1151), where a community A/B measured prompts 2.11x faster with it (251 to 529 tokens/s); every other card
  uses it only when you ask: `STRATA_HIPBLASLT_TABLE=on` (or `"hipblaslt_table": true` in the run config) uses it, `off` never does (on Strix
  Halo too), and the start log
  says which table was taken or why none was (`hipBLASLt tuning table: ...`). A `STRATA_HIPBLASLT_TUNING` you set yourself is always kept.
  A source build is unchanged (see [Tuning table](#tuning-table)). Testers on Strix Halo and RX 7000 cards: please send that log line and a
  prompt-speed run with `on` and `off`.

Please report a run on an RX 7000 card or a Strix Halo (card, driver, OS, the end of `strata-<model>.log`).

## Windows

Since 0.1.34 an AMD card on Windows is set up like an NVIDIA one: download Strata, double-click `START-HERE.bat`.
On a PC with no NVIDIA card Strata can use, the AMD card is chosen by itself; with both, setup asks
(`START-HERE.bat --backend hip` picks AMD directly).

- **You need:** Windows 10 or 11 (64-bit), one of the cards above, and a current AMD driver ([AMD Software:
  Adrenalin Edition](https://www.amd.com/en/support/download/drivers.html)). Nothing else: no ROCm or HIP SDK
  install, no compiler, no admin rights.
- **Detection:** setup reads the display adapters Windows lists (their PCI ids; the VRAM size from the display
  driver's registry entry). An integrated Radeon is listed as not supported, except the Ryzen AI Max "Strix Halo" (Radeon 8060S / 8050S / 8040S, gfx1151: [STRIX_HALO.md](STRIX_HALO.md)); a Strix Point (890M / 880M, gfx1150), Krackan (860M / 840M, gfx1152) or Phoenix / Hawk Point (780M / 760M, gfx1103) Radeon is named as what it is and not supported (on Linux the 780M / 760M has an opt-in build: [RADEON_780M.md](RADEON_780M.md)).
- **Engine (0.2.0 and later):** the installers and `strata update` use the engine packages in [Ready-made packages](#ready-made-packages-020); the zip described next is the START-HERE route of 0.1.x.
- **Engine (START-HERE, 0.1.x):** the ready-made engine package for your card, fetched from the signed update feed (releases from 0.2 carry no flat `strata-windows-x64-hip.zip`; that zip is what `build_windows.bat` makes for a self-build) (built by `tools\hip\build_windows.bat`
  for gfx1100, gfx1101, gfx1102, gfx1200, gfx1201, gfx1030 and, from 0.1.40, gfx1151 (Strix Halo, unvalidated on Windows)) goes into `engine\`. It carries the ROCm libraries the
  engine loads (`engine\rocm\bin`: the HIP runtime, hipBLAS / rocBLAS / hipBLASLt with their kernels for these cards,
  amd_comgr and the Microsoft C++ runtime; ROCm 10.2.0a20260930 from AMD's TheRock builds, licenses in
  `engine\rocm\licenses`). The HIP runtime works through the AMD driver's own components, so the driver is the one
  thing it needs from the PC.
- **The HIP runtime next to `strata.exe` (0.1.35, #468 #461):** `amdhip64_7.dll` and `amd_comgr.dll` are also put in
  `engine\` (setup copies them there on every start). Windows looks in the program's folder before System32, where
  some AMD drivers install their own `amdhip64_7.dll`; with that one, the bundled libraries crashed on the first
  prompt (an access violation, or `hipErrorInvalidDeviceFunction`). The engine's log names the runtime it loaded
  (`strata generate: HIP runtime ...`).
- **Before the ~60 GB model download** setup runs `engine\strata-device.exe --list-devices` (with `engine\rocm\bin` on
  the PATH): if the HIP runtime does not see the card, setup stops there and points to the driver. It also gives the
  card's HIP number: with an integrated Radeon that is device 1, not 0 (#325). From then on setup lists the AMD cards
  as HIP numbers them, so `--gpu N` and the config's `"gpu"` are HIP numbers.
- **Differences from Windows-on-NVIDIA and Linux-on-AMD:** one card per model (`--gpus` is Linux-only for now), no calibration.
- Two Windows-only engine details (#247, #325): hipBLAS can return success and still leave `hipErrorInvalidValue`
  set after some BF16/FP16 GEMMs (seen on gfx1201); the engine clears that one stale error after a GEMM that
  succeeded, on Windows only. `hipHostGetDevicePointer` returns the host pointer itself on Windows: kernels read
  mapped memory through it correctly, but a device-to-device copy into it does not land, which is why
  `tests/hip/handoff` times out there (the engine does not use that copy; `tests/hip/mapped_alias` reports it).
- Two more (#380, #377, by BlueKingMuch, measured on an RX 6800 that drives the desktop): `hipMemGetInfo` on Windows
  does not subtract what the desktop and other programs hold on the card, so `--expert-cache auto` filled the card
  past what Windows keeps in VRAM and decode fell from 41 to 30 tok/s. The engine now lowers that free figure by
  what Windows' video memory budget for the process withholds (logged once: `strata: Windows budgets N of this
  card's M MiB ...`; `STRATA_WDDM_BUDGET=0` turns it off). And the PCIe probe times its copies on the host clock
  there, since HIP's events read impossible speeds (3,300-26,000 GB/s), so a slow link now gets a smaller
  `pcie_frac` as on NVIDIA.

**What is validated (0.1.34):** #325's author ran the engine of this port on an RX 9070 XT (Windows 11, ROCm
10.2.0a20260930 in `.venv`, compiled on the PC): Coder IQ1_M at 32K, 29.2 tok/s decode, ~181 tok/s prefill,
correct answers; ctest 42 of 46. The maintainers have no Windows AMD card: the release zip was built on an NVIDIA PC,
and checked there on the Ryzen CPU's integrated Radeon (gfx1036, a test build of the same tree): with only the zip's
libraries on the PATH, `strata-device` lists the card and a hipBLAS BF16 GEMM matches the CPU; the HIP ctest passes
52 of 56, the 4 failures the same as on the RX 9070 XT (`hip_handoff`, and three tests that need a pack fixture).
The ready-made zip itself has not run a model on a discrete card yet - please report.

**Reporting a Windows AMD run** (an issue, or on #325): your card and driver version (AMD Software > System), then

```bat
engine\strata-device.exe --list-devices
engine\strata-device.exe --selftest
```

(from the Strata folder, after `set PATH=%CD%\engine\rocm\bin;%PATH%`), the end of `strata-<model>.log`, and the
speed lines the server window prints for a first answer.

**Building it yourself:** `tools\hip\build_windows.bat` (Visual Studio 2022 Build Tools with the C++ workload, Python,
git; no admin, no AMD GPU) installs ROCm from AMD's TheRock wheels into `.rocm-win`, builds, and packages
`dist\strata-windows-x64-hip.zip`; `START-HERE.bat --backend hip --prebuilt dist\` installs that one.
`tools\hip\build_windows.bat tests` also builds the HIP tests (`ctest` in `build-hip-win`, with
`.rocm-win\Lib\site-packages\_rocm_sdk_devel\bin` on the PATH). `STRATA_HIP_ARCHS` picks other architectures.

## Build

Requirements: a working ROCm driver/runtime, HIP development headers and
compiler, hipBLAS and (for tuned dense prefill) hipBLASLt development files, CMake 3.24+, a C++20 host compiler, and Git.
The model still needs sufficient system RAM and fast SSD storage for its PLE
table. VRAM occupancy alone is not a throughput measurement.

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

`CMAKE_HIP_ARCHITECTURES` is `gfx1100`, `gfx1101`, `gfx1200`, `gfx1201`, or a list such as `"gfx1100;gfx1201"`
(one binary for both). gfx1102 (the same wave32, 64 KiB LDS and dot4 instruction) builds with a warning: it passed
ctest (#192), and one RX 7600 XT (16 GB, i7-13700K, 64 GB) ran Swift IQ3_XXS over 146 requests up to 51K context at about 43 tok/s decode (28-50) and 250-300 tok/s prompt (#942); setup accepts it (unvalidated, #938); so does gfx1030 (RDNA2: the older `v_dot4_i32_i8`, a community run in #311). At startup the engine and `strata-device` compare each GPU they use
(`gcnArchName` up to the `:` feature suffix) with the architectures the binary was compiled for, and require
wave32. A binary carried to another card stops with the card's name, its architecture and the build's list,
instead of failing later with "invalid device function".

If CMake cannot find the HIP compiler, add
`-DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang++`. On the Fedora-family test
host this was `/usr/lib64/rocm/llvm/bin/clang++`. Use the compiler and libraries
from the same ROCm installation. Do not enable both GPU backends in one build.

Native IQ experts are enabled by default. CMake fetches the llama.cpp revision
pinned by this repository. For an offline build, point `STRATA_GGML_DIR` at a
checkout of that exact revision; using an arbitrary newer checkout changes the
dependency being tested.

## Opt-in chunked prompt GDN on gfx11

`STRATA_GDN_CHUNKED=1` runs the prompt's gated DeltaNet recurrence in 16-token chunks on the gfx11 matrix cores,
with FP16 operands and FP32 accumulation, and fuses the output norm. It is off by default: rounding differs from
the token-by-token FP32 recurrence. The CUDA switch of the same name uses a separate FP32 implementation.

The HIP path accepts gfx1100, gfx1101, gfx1102, gfx1150 and gfx1151, and prompt chunks of at least 64 tokens.
Other cards, shorter chunks, an output row stride not divisible by eight, and `STRATA_GDN_REC_HEADS` use the
existing recurrence. Runtime measurements and numerical checks were run on gfx1151 only; building for another
target does not validate its speed or answers. The synthetic test compares the FP16 output and carried FP32 state
against the existing recurrence, and short cases against a double-precision CPU calculation. It covers partial
chunks, padded rows, weak decay and a split request; the off/default tests require identical output and state bits.
It does not establish model answer equivalence.

`STRATA_GDN_CHUNKED=2`, which on CUDA skips the SM-count test, is the same as 1 here.

Measured on a Ryzen AI Max+ 395 (gfx1151, ROCm 7.14.1), UD-IQ4_XS. One layer's recurrence and norm take 1.60x less time than
gfx1151's default kernels at 16K and 32K tokens (`hip_gdn_chunked --bench` against `--bench-default`, three runs each: 20.92 ->
13.11 ms at 16K, 42.14 -> 26.30 ms at 32K). A whole prompt: the same binary with only this switch changed, a fresh engine per
side, 11 pairs in alternating order after a discarded one, the engine's `prompt_ms`, every prompt read in full (no reused
tokens). Each engine reads a discarded 4K prompt first, because the first prompt after a start varies more (on those first
prompts the same pairs gave 95% intervals of [1.004, 1.032] and [0.981, 1.015]); the prompts follow each other once
`/proc/vmstat`'s page-migration and direct-reclaim counters have stood still for 3 s, and a pair in which either prompt still
met page migration or direct reclaim was dropped and measured again (3 of the 69 pairs, all 32K with the switch on; kept in, 32K
gives 1.020 and 1.015). "Section 5" is the nine variables of `STRIX_HALO.md` section 5 (with `STRATA_PF_SWITCH_MIN_T=2048`);
"defaults" is the engine without them (section 4's gfx1151 defaults stay). Times are medians; the ratio is the geometric mean of
the 11 off / on ratios with a 95% bootstrap interval; "faster" counts the pairs in which the switch was faster:

| tokens | arm | off -> on | off / on | faster |
|---:|---|---:|---:|---:|
| 4,096 | section 5 | 3.703 -> 3.641 s | 1.018 [1.016, 1.020] | 11 of 11 |
| 4,096 | defaults | 5.914 -> 5.829 s | 1.014 [1.010, 1.017] | 10 of 11 |
| 32,768 | section 5 | 26.157 -> 25.589 s | 1.021 [1.020, 1.023] | 11 of 11 |
| 32,768 | defaults | 44.082 -> 43.479 s | 1.015 [1.012, 1.017] | 11 of 11 |
| 128,000 | section 5 | 107.149 -> 104.866 s | 1.021 [1.020, 1.023] | 11 of 11 |
| 128,000 | defaults | 193.439 -> 191.055 s | 1.014 [1.012, 1.016] | 11 of 11 |

So about 2% on top of section 5 and about 1.5% on the defaults. At 36 gated-DeltaNet layers the layer figures predict about
0.56 s for a 32K prompt; 0.57 s (section 5) and 0.60 s (defaults) were measured.

First-token KL against the same run without the switch, 12 prompts of 1.5K-24K tokens: UD-IQ4_XS 0.0067 mean (0.044 max) on the
defaults and 0.0039 (0.027) on top of section 5; UD-Q4_K_XL 0.0063 (0.043) and 0.0040 (0.014); the same top token in all of
them (across the 12 prompts the first token takes only two distinct values). Section 5's switches alone are 0.0068 (IQ4_XS) and
0.0031 (Q4_K_XL) against the defaults, a different change that is listed for scale and not as a limit for this one. Only the
first generated token was compared: a long generation and perplexity were not measured.

Build and run these checks in the HIP build directory:

```sh
cmake --build build-hip --target hip_gdn_chunked hip_gdn_rec_head
ctest --test-dir build-hip -R 'hip_gdn_(chunked|rec_head)' --output-on-failure
for i in 1 2 3; do ./build-hip/hip_gdn_chunked --bench-default; ./build-hip/hip_gdn_chunked --bench; done
```

Each benchmark run reports the GPU-event median of seven runs after two warmups, the same nonzero initial state every run:
`--bench` this path, `--bench-default` gfx1151's default kernels (`STRATA_GDN_PP=2`, `STRATA_GDN_NOY=1`). The switch is read
once per process, so the two are separate runs. It times one layer's recurrence and norm, not a whole prompt.

## Model and serving configuration

Prepare a supported model pack and its MTP runtime using the existing tools.
For OrcaRouter IQ3_XXS, follow [the explicit compatibility conversion](ORCA.md);
this backend does not change quantization, model licenses, or tokenizers.
Add `--experts-bin` to the `tools/iq_pack.py` command: mmap requires the pack's
`experts.bin`, which the default native packing command does not emit. This
consumes additional disk space. Original GSQ-RCO IQ3_XXS uses shard 2 for PLE;
Orca uses shard 1. Keep each model's own pack and tokenizer together.

Use `build-hip/strata` as the executable in the server JSON. Select the AMD
device with `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` if necessary. Start with
a modest context and prefill chunk size before measuring larger workloads.
Keep the PLE file on an SSD. Do not assume a configured context length proves
successful full-window inference.

Large pinned host allocations can fail on ROCm even when ordinary RAM is
available. The existing `--mmap-experts` path avoids allocating the full pinned
expert arena; it still depends on OS file-cache residency and may stall on
storage reads. It does not make SSD access equivalent to RAM.

Starting args for the original GSQ-RCO IQ3_XXS model (replace the paths):

```sh
build-hip/strata --serve \
  --pack packs/iq3xxs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache auto \
  --prefill 512 --spec 4 --spec-min-p 0.5 --mtp mtp/rt \
  --max-context 4096 --kv int8 --pool-workers 15 \
  --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
```

`--serve` is the engine's internal token protocol. For a browser or OpenAI API,
put these args in the `args` array of the [server JSON example](ORCA.md#local-server),
set `exe` to `build-hip/strata`, and use the matching pack's `tokenizer` directory.
Launch with `.venv/bin/python -m serve.server --engine strata --config strata-hip.json --port 8080`.
The worker count above was used on a 16-core CPU; measure it for your CPU.
The 4K context is a smoke-test starting point, not a model limit. The expert cache
sizes itself automatically and leaves 1 GiB of VRAM headroom.

**The card also drives a Linux desktop (#560 #516):** keep more VRAM free than the default 700 MiB, e.g.
`./setup.sh --vram-reserve-mib 3072`. With the cache filling the card, the desktop's next VRAM need makes amdgpu move
GPU memory to system RAM (GTT), the OOM killer then ends KWin/plasmashell or `systemd-oomd` ends apps, or the
compositor fails with "Failed to pin framebuffer with error -12".

The installer supports this backend (see "Install with setup" above). Pictures are on by default. They are read on the CPU (`strata-vision`) under 24 GB of VRAM and on the card (`strata-vision-hip`) from 24 GB, the same rule as on NVIDIA; if the engine has no card reader, or it cannot start, the CPU reader is used.
Setup installs one AMD card, or several with `--gpus` (the engine's layer split; see RDNA4 below).

**Opt-in switches for RDNA2 prompts and hangs (0.1.40, off by default):** `STRATA_DENSE_MMQ=1` (#820, HIP) runs the
prompt's dense GGUF projections through the int8 MMQ kernels instead of dequantize + hipBLAS: on the RX 6900 XT a 4K
prompt read 436 -> 757 tok/s (greedy text diverges from the default after a few dozen tokens: the q8_1 activation
rounding). `STRATA_HIP_ADAPT_KERNEL_COPY=1` (#884) copies the adaptive tier's swaps with a kernel instead of the SDMA
engine, a workaround for the gfx1030 hang seen with the MMQ prompt path and adaptive swaps (untested on the reporter's
machine). Windows HIP: the doorbell kernels fence their store (#697), and the shared-expert fork is off on HIP (#816).

**The shared-expert stream fork (`STRATA_SH_STREAM`):** it is off by default on HIP, except for the gfx1151 table, and
that default stays. The measurements disagree by platform. 2x RX 6900 XT on Linux (#1272): turning it off costs 5-13%
of decode (one card 47.5 / 46.9 / 48.3 -> 50.5 / 48.5 / 51.3 tok/s with it on). RX 6800 on Windows (#816,
`bench/results/2026-10-04-community-rx-6800-windows`): on was 12-22% slower. 4x R9700: fork off was 13-45% faster.
Linux RX 6000 users can try `STRATA_SH_STREAM=1`; the output is identical either way. Since the same card goes
both ways on different PCs, `./setup.sh --calibrate` measures it on yours (off against on, a restart each) and writes
`STRATA_SH_STREAM=1` into the config's `"env"` only when it wins by more than 3%; a later calibration that finds it no
longer faster removes it again.

**The draft's side stream (`STRATA_MTP_SIDE_STREAM`):** HIP does not use the MTP drafter's branch stream, and since
0.1.40.2 it does not create it (#1240: an R9700 lost prompt speed with it). gfx103x is the exception and creates it: on
an RX 6800 under Windows (`STRATA_HIP_PROMPT_F16=1`, IQ3_XXS, two runs each) a 7.9K-token prompt reads at 798-808 tok/s
with the stream and 584-596 without, and a 30.4K-token one at 855-863 against 553-600; `STRATA_PREFILL_TIMING=1` puts
the whole difference in the "wait copy" phase (3,081 -> 75 ms and 12,642 -> 288 ms). Decode and the output are the same.
`STRATA_MTP_SIDE_STREAM=0` turns it off on gfx103x, `=1` creates it on another HIP card. Measured on one card and one
OS; a Linux gfx103x number is welcome.

**Long prompts on a big card: check the lend-coverage line (#1389, measured by the reporter on an RX 7900 XTX, gfx1100,
v0.1.40.2, 50 GB RAM).** With `--resident-experts` the prompt path borrows cache slots, and the experts of those slots
must also sit in RAM. After the first request of a start the engine prints
`FileExpertSource: N of M of the prompt path's lendable slots keep their experts in RAM too`. When N is not M, the
experts not covered are read from the pack during the prompt (now also a `WARNING` line). A 74K prompt routes through
nearly all experts, so the misses repeat: the reporter saw `wait copy` at 0.2% and 2,583 tok/s with 6201 of 6201, and
36.8% and 1,810 tok/s with 5093 of 6202 (same binary, only the start differed). Short prompts (25K) read the same either
way. The resident set is sized from `MemAvailable` at start, minus `STRATA_RESIDENT_HEADROOM_GIB` (default 4, in GiB; the
`--resident-experts` switch and `--resident-budget-gib` read it). If the line is short, lower the headroom a little and
restart; if the machine then swaps or stalls, raise it again. Re-check the line after a ROCm update: the reporter's newer
runtime kept about 1 GiB more host RAM, which moved the best value from 8 to 7. Other levers from the same report (his
numbers, not re-measured here): `STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096` 925 -> 1,092 tok/s,
`--resident-experts` 1,092 -> 2,503 tok/s, `--prefill auto:32768` +23% at 74K. The engine uses a hipBLASLt tuning table
only when its version matches the installed library; otherwise it prefills on plain hipBLAS (about 45% slower for a 131K
prompt), so keep the table and the library paired. Not worth trying (measured flat): `--kv-resident` changes, a prefill
chunk above 32768, `--spec` above 4.

## Exact long-prompt QSA scorer (gfx1100 opt-in, gfx12 default)

`STRATA_SCORES_THREAD=1` at engine startup enables a HIP port of #1825's score kernel on gfx1100, for batches of
at least 64 queries with at least 16,385 live index blocks. On gfx12 (RDNA4) it is on by default for those batches and
`STRATA_SCORES_THREAD=0` turns it off (0.1.45): on an R9700 `qsa_scores_gfx1100_parity --full` compares 223,664,777 scores and
11,895,800 ids bitwise equal to the warp scorer, the scorer stage with 256 queries is 3.6x faster at 131K and 4.0x at 240K, and a
fresh prompt of 131K tokens reads 8.7% faster (4 of 4 pairs) and one of 230K tokens 16.4% faster (3 of 3), with the same answer tokens and byte-identical `exact_pp` residual dumps at 70K and 131K tokens (prompts under 65K cells never reach the bound). The default, short-prompt and decode paths keep the
existing scorer. On one RX 7900 XTX / 7950X3D / ROCm 10.0.0, three paired runs of a tuned 0.1.40.3 engine reduced
fresh one-token request time by 10.45% at 130,978 prompt tokens and 22.64% at 239,906; one same-binary current-main
pair measured 12.85% and 21.71%. Cached follow-ups did not show a repeatable large gain, and a two-task coding pilot
was 3.27% slower overall. This is opt-in, not a general coding-speed recommendation. The
[report, per-run data and standalone score/ID check](../bench/results/2026-10-11-gfx1100-qsa-scores/README.md)
give the settings, exact-parity checks and limits. `STRATA_SCORES_THREAD_TRACE=1` prints an engagement line.

## Deep contexts: the prompt's top-k selection (0.1.45, #1749)

Past ~135K cells (and from ~90K cells on a prompt read) the prompt's top-k selection runs the wide kernel, as on CUDA, instead of the four-pass
reference radix that every deep context on HIP paid; the same ids (`qsa_select_bench` compares them with the reference). On an R9700 the
selection of a 256-query batch is 0.29 -> 0.21 ms at 98K, 1.16 -> 0.49 ms at 262K and 22.0 -> 0.93 ms at 524K; a fresh prompt reads 1.3% faster at
230K tokens and 34% faster at 500K (716 -> 962 tok/s; 1,284 tok/s with the exact scorer above, 79% in all), with the same answer tokens.
`STRATA_TOPK_OLD=1` runs the original kernel, `STRATA_TOPK_STREAM=0` the register kernel in the 90K-135K band.

## Linux: verify timeouts while the kernel reclaims host memory (experimental workarounds)

A `verify: timed out at layer N` or "no progress for 60 s" message does not by itself mean a kernel or handshake bug. Two
community reports found the same mechanism: the GPU's queues are suspended while the kernel reclaims host pages the GPU
has pinned through a KFD userptr, and a restore that keeps returning `-EAGAIN` leaves them suspended for tens of seconds.
Each report is one machine, one workaround, and the cause is not confirmed on either; neither is a default or a general
speed claim, and neither is known to matter on Windows or on other ROCm versions.

- **Paged host allocations (#750, two Radeon AI PRO R9700, ROCm 7.2):** tracing correlated a timeout with about 31 s of
  USERPTR queue suspension on both cards. ROCr uses USERPTR for paged host allocations unless `HSA_USERPTR_FOR_PAGED_MEM=0`.
  With it in the server JSON `env`, five paired runs gave the same outputs and the same median (7 requests: 16.65 s against
  16.63 s), without the 29.8 s and 42.2 s outliers; solo requests were a little slower (3.55 s to 3.69 s).
- **`--mmap-experts` (#920, RX 6800 gfx1030, ROCm 7.2.4, 31 GiB RAM, IQ3_XXS):** every run stalled until
  `GPU_PINNED_MIN_XFER_SIZE=1048576` was in the `env`, and none has since. HIP pins the pageable source pages of large
  copies, here the mapped expert file. (The engine already sets this variable for `STRATA_ARENA_MMAP=1`.)

```json
"env": { "GPU_PINNED_MIN_XFER_SIZE": "1048576" }
```

Change one setting at a time with the rest of the configuration the same, restart the engine so the runtime reads it, and
remove it to go back to the default. Check free RAM and the driver's GTT limit before large-context tests. If the engine
runs in a systemd unit, limit it with `MemoryMax`, never `MemoryHigh`: `MemoryHigh` counts the page cache that
`--mmap-experts` reads from, and a 16K prompt stalled in `pread` for 8 minutes under it. In a report, give the runtime and
kernel versions, whether reclaim coincides with KFD queue eviction, and paired timings.

## RDNA4 (gfx1201)

The RX 9070 / 9070 XT and the Radeon AI PRO R9700 run the same kernels as gfx1100: wave32, 64 KiB of LDS per
workgroup, the signed dot4 instruction (`v_dot4_i32_iu8` through `__builtin_amdgcn_sudot4`) and a 100 MHz wall
clock. The first report and patch came from doplxyz (#178). Validated on 2026-09-30 on doplxyz's test machine:
an RX 9070 XT 16 GB and a Radeon AI PRO R9700 32 GB (both gfx1201), a Ryzen 9 3900X (16 threads, no AVX-512),
47 GB RAM, Ubuntu 24.04 in a KVM/VFIO guest.

- **Build:** complete HIP build with tests (`-DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_BUILD_TESTS=ON
  -DSTRATA_PREFILL_MMQ=ON`), with the system ROCm 7.14 (hipBLASLt 1.4.1) and with setup's own path: TheRock
  7.10.0a20251120 wheels from `gfx120X-all` (hipBLASLt 1.2.0) and `build_engine_hip`.
- **ctest** (all 45 registered tests, R9700): 42 pass, `hip_prefill_hipblaslt_gemm` skips (no table), and two
  fail for reasons outside the GPU: `ple_parity` needs the Q2_0 model fixture, `expert_multi_test` needs an
  AVX-512 CPU. `hip_handoff` needed the volatile ring store: on gfx1201 a plain store to mapped pinned memory
  stays in the GPU's L2 until the stream is synchronized.
- **Arch check:** a gfx1100-only build stops on the gfx1201 card at startup with the message above (engine and
  `strata-device`).
- **End to end** (Coder IQ1_M, setup's arguments: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`,
  MTP, `--kv int8`, `--max-context 32768`; 128 greedy tokens, system ROCm 7.14, no hipBLASLt table). The answers
  are coherent and the same on both cards. The live server smoke (`serve/server.py`: web page, models, props,
  chat, prefix reuse, streaming, the Anthropic endpoint, a sampled code answer) passed 9/9 on each card.

  | card | expert cache | peak VRAM | 4K prompt | 4K decode | 16K prompt | 16K decode |
  |---|---|---|---|---|---|---|
  | Radeon AI PRO R9700 32 GB | 12,288 slots, 23.4 GiB | 29.7 GiB | 982 tok/s | 45.5 tok/s | 1,402 tok/s | 48.3 tok/s |
  | RX 9070 XT 16 GB | 4,931 slots, 9.4 GiB | 15.6 GiB | 782 tok/s | 30.8 tok/s | 1,235 tok/s | 35.4 tok/s |
  | R9700, TheRock 7.10 wheels | 12,288 slots | | 957 tok/s | 44.2 tok/s | 1,284 tok/s | 43.9 tok/s |

  The engine's resident memory was about 26 GB in every run. Since 0.1.31 `__byte_perm` is one `v_perm_b32` and the
  packed byte subtracts/compare work on four lanes at once (#262, ttio2tech): decode +15% on the R9700 (46.0 -> 53.0
  tok/s on a 4K prompt, 52.0 -> 60.5 warm) and +5-7% on the 9070 XT, prompts unchanged, the same tokens.
- **hipBLASLt:** the validation above used hipBLASLt 1.4.1. A table calibrated on the R9700 at the engine's shapes
  with that version (0.98-1.76x per GEMM over hipBLAS) changed the end-to-end prompt speed by 0-3%, within noise,
  so none was shipped for it: there the plain hipBLAS path is already close (a table I calibrated on my own R9700
  with 1.4.1 measured the same: 1.15x per GEMM, +1% prompt - see "Tuning table" below). For hipBLASLt 1.5.0 (ROCm
  10.2.0a nightly) `tools/hip/gfx1201-hipblaslt-100500.txt` is shipped (see "Tuning table" below); on one R9700 it
  measured +3.9% prompt speed on 4,210-token prompts (1,590 vs 1,531 tok/s), a modest gain.
  With the hipBLASLt 1.2.2 of a system ROCm 7.2.4 the plain path is far off, and `tools/hip/gfx1201-hipblaslt-100202.txt`
  is shipped for it (24 of the gfx1100 table's 26 shapes; setup uses it only with that exact version): on an R9700
  with the full IQ3_XXS, `--kv int8 --kv-resident 32768`, a fresh 32K prompt read at 638 -> 1,177 tok/s and a 7K one
  at 656 -> 1,164 tok/s with it, decode unchanged.  `hip_prefill_hipblaslt_gemm` passes with it.
- **Both cards in one run (layer split, engine 0.1.30):** the config's `"backend": "hip", "gpu": [1, 0]` (R9700
  first) runs through `serve/server.py` (setup writes it with `--gpus 1,0` since 0.1.31). Auto split put layers 0-27 on the R9700 and
  28-47 on the 9070 XT. With every expert on the GPUs the split gives exactly the tokens of the R9700 alone (4K and
  16K prompts); checkpoints on the split (second turn, rewind, a prompt sharing a prefix, a cancelled prompt
  retried) give exactly the tokens of a fresh read. The conversation cache refuses a split at start (exit 2).
  On this pair the split does not pay: the R9700 alone already holds all 12,288 expert pairs.

  | run (the same session, warm) | 4K prompt | 16K prompt | decode |
  |---|---|---|---|
  | R9700 alone | 1,794 tok/s | 1,804 tok/s | 51-52 tok/s |
  | R9700 + 9070 XT, auto split | 1,384 tok/s | 1,906 tok/s | 42-43 tok/s |
  | RX 9070 XT alone | 1,016 tok/s | 1,519 tok/s | 38-40 tok/s |

  A split is worth it when no single card holds the model's experts. Since 0.1.31 a split pins the whole expert
  arena on Linux (the 8 GiB cap is for Windows/WSL2 only, #253): with an expert cache of 1,500 on the R9700 (most
  experts streamed) the split read a 16K prompt at 1,803 tok/s instead of 1,287, with the same tokens in 5 + 5
  starts. `"split_skip_if_fits": true` in the config (0.1.31, opt-in) runs such a pair on the first card alone when it
  holds every profiled expert: with the R9700 first, 4K prompts 1,776 tok/s (split: 1,244) and decode ~60 tok/s
  (split: ~51), the tokens of the R9700 alone (docs/MULTI_GPU.md).
- **Speed switches (engine 0.1.32, measured on the R9700 / 9070 XT with the Coder IQ1_M pack):**
  - the MoE router (`router_top10`) runs a HIP kernel without its serial FP64 sum and block barriers by default: the
    same ids and weights bit for bit (`hip_router_fast` checks 65,536 rows), 39 -> 9-12 us per call, the same greedy
    tokens (5 + 5 starts). Decode: 62.4 -> 70.0 tok/s on the R9700 and +4% on the 9070 XT in one A/B here (ROCm 10.2
    nightly); a user's repeated A/B/A/B with setup's TheRock 7.10 wheels and an i5-12600K measured +1-4%, inside a
    12-20% run-to-run spread (#432) - how much it gains depends on how much of decode the router is on that setup.
    `STRATA_HIP_ROUTER_OLD=1` runs the portable kernel.
  - `STRATA_HIP_WMMA` (on by default on gfx12 with an int8 KV since 148-amd, `STRATA_HIP_WMMA=0` turns it off; gfx11: opt-in with `=1`): the prompt path's QSA attention on RDNA4 matrix cores (the engine says so once, at the first prompt)
    (`v_wmma_f32_16x16x16_f16`, FP16 hi + lo halves like the CUDA tensor-core kernel; `hip_prompt_attn_wmma` bounds it
    against the FP32 kernel and FP64). 7.2-7.5x the portable kernel; R9700 prompts 4K 1,784 -> 2,427 tok/s, 16K
    1,797 -> 2,700 (a user with TheRock 7.10: +29-34%, #432). Not bitwise: greedy text differs from token ~50 on, as with the CUDA tensor-core attention. With it
    the prompt path's expert ring is 96 slots (as STRATA_PREFILL_RING=96): with the default 384 the 9070 XT's 4K
    prompts fell to 718 tok/s; with 96 they gain (1,017 -> 1,211; 16K 1,518 -> 2,032). PR #329
    (bsorensen110) contributed an equivalent gfx12 WMMA kernel of the same speed (within 1%); this one also masks KV
    pages that KV streaming has not made resident, as the decode kernel does. On my R9700 (engine 0.1.40,
    IQ3_S at 431072 ctx, `--kv int8`, the gfx1201 hipBLASLt 1.4.1 table below, which names the machine, GPU idle):
    fresh prompts of 4,210 and 8,830 tokens - the longest prompt I measured there is 8,830 tokens, the 431072 ctx is
    what the model runs at, not what these prompts reach - three matched trials each, the first fresh prompt after
    each start left out: 1,571 -> 2,064 and 1,568 -> 2,065 tok/s at 8,830 tokens (+31.4 and +31.7%) and 1,699 ->
    1,990 at 4,210 (+17.1%), medians 1,571 -> 2,064 and a 1.27x geometric mean over those three. decode
    89.8 -> 86.9 tok/s, unchanged. `STRATA_SELECT_WMMA=1` on top changed nothing at those lengths (2,026 tok/s);
    its +1.5% is a 16K/262K-ctx number. `hip_prompt_attn_wmma` passes on that card.
    **Default on gfx12 on Linux (148-amd).** Windows: opt-in until tested on Windows (`STRATA_HIP_WMMA=1` turns it on; the engine says so in its log at the first prompt). Quality on an R9700 (Q2_0, int8 KV, fixed expert cache so that the default against itself is bit-identical):
    teacher-forced KL against the old kernel through the real serve decode path (STRATA_FOLLOW, 28 prompts, 6,024 positions) 0.0003 (p99 0.006, top-1
    99.8%, perplexity of the forced text x1.0003), and 0.0014 over 1,760 positions of eight 6K-token documents (top-1 98.9%, x1.0009); HumanEval 157/164
    (default 156 and 156), reasoning 25/30 (24 and 26), agent 6/6, needles at 32K and 125K x three depths 6/6, CJK 8/8, tool calls 5/5. Prompt speed,
    interleaved pairs on a quiet R9700, Q2_0 32 GB: see the 148-amd table below. The first-token KL against the FP16 prompt path cannot rank it (0.0296 against 0.0280 for the old
    kernel, a paired difference of +0.0016 +- 0.0026): on a 2-bit pack any change of the prompt numerics, even `--prefill 4096`, moves the first token by about 0.025.
  - `STRATA_HIP_WMMA=auto`: the same kernel, chosen per call: on a gfx12 card with int8 KV and a prompt batch of
    `STRATA_HIP_WMMA_MIN_Q` (64) or more queries it runs the matrix-core kernel, everything else the FP32 one (`=1` forces it
    on every call and on gfx11, `=0` keeps the FP32 kernel; unset is the gfx12 default above: the matrix-core kernel on every call). A launch error switches the matrix-core
    kernel off for the rest of the process and the FP32 kernel runs (before, the engine exited). Kernel timing on an R9700
    (`hip_prompt_attn_wmma`, context 1.5K-32K, 64-2,048 queries): 6.9-8.4x the FP32 kernel at every shape, error against FP64
    3e-6 of the output scale (FP32: 2e-6). Engine, Q2_0 pack, 6 interleaved pairs, equal slots, fixed flags
    (`--pcie-frac 0 --adapt-every 0`, `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_ROUTE_TAIL_SKIP=0`): 20K prompt x1.46 (`=1`) /
    x1.54 (`auto`), 4K x1.33 / x1.34 (6/6 pairs; the noise arm of that run read +6.4-6.9%, so read them as +40-50% and
    +30%); the 8-token answers were the default's in every run.
  - The first prompt after a start pays two one-time hipBLAS library loads (the first BF16 product 1.2 s, the first FP16
    product 1.0 s on an R9700 with the gfx120X wheel's hipBLASLt 1.2.0; `STRATA_COLD_TRACE=1` times the first 80 dense
    projections to show them). The engine now runs one tiny BF16 and one tiny FP16 product on a thread of its own while the
    expert arena loads, so the first request does not carry them (`STRATA_PREFILL_WARM=0` turns it off; the outputs go to
    scratch buffers, nothing changes; start-up time is the same, the arena load takes longer than the thread). First
    721-token prompt after the 33-token warm request, 5 interleaved pairs: 154 -> 673 tok/s. The next 4K prompt reads
    ~1,090 tok/s either way (the "cold" first-prompt numbers above are about this).
  - **Fast and slow starts (the "bimodal" prompt speed, found in 0.2.1).** On an R9700 / RX 9000 under Linux about half of
    the fresh starts with MTP read prompts 10-23% slower than the others (p620, Q2_0: 4K 1,980 vs 2,290 tok/s, 20K 2,480 vs
    2,770; decode was never affected). The cause is where the kernel's scheduler puts the process's GPU queues. The prompt
    path has two busy queues, the compute stream and the stream that copies experts to the card (about 31,000 copies and
    event hand-offs per 4K prompt). The runtime opens its queues in the order the loader threads reach them, which changes
    every start; when both busy queues land on the same compute pipe of the card (the card has two for this process), every
    hand-off waits for a queue switch. The engine now opens two queues at normal priority (more queues cost decode: five cost 2 to 3.6%) one
    after the other on the main thread before any other thread starts (`STRATA_HIP_QUEUE_PRIME=0` turns it off; any other
    value is a list of `n` / `h` letters for tests). gfx1200 and gfx1201 on Linux only; nothing else changes (the answers
    are identical). Not clocks, not memory placement (read speed over the expert cache was 637-640 GB/s in both kinds of
    start), not CPU placement, not the MTP stream itself.
  - Ship status of `tools/hip/gfx1201-hipblaslt-*.txt`: the gfx120X wheel's hipBLASLt is 1.2.0 (version number 100200); the
    tables here are 100202 and 100500, so the engine says "version mismatch ... using hipBLASEx" and runs plain hipBLAS. A
    table tuned on the R9700 for 100200 (32 rows, `tune_hipblaslt`) measured within noise end to end (20K prompt x1.035 vs
    a noise arm x1.031), so none is shipped: the dense projections are ~10% of a 20K prompt here.
  - The prompt path's QSA top-k picks its kernel by the blocks a query actually has, not the cache's capacity (#337,
    bsorensen110): the same ids, on by default with AMD (NVIDIA keeps its capacity rule: there the 64K prompts read
    1-3% slower with it). `STRATA_SELECT_WMMA=1` (opt-in, gfx12) adds #337's
    matrix-core block scorer; it selects slightly differently (254 of 256 queries the same) and gained +1.5% on 16K
    prompts at a 262K context on the R9700.
  - `STRATA_PF_FUSED=1` (opt-in, gfx1200 / gfx1201, the native IQ packs, #1277): the prompt's experts run on int8
    matrix-core kernels (`v_wmma_i32_16x16x16_iu8`: the gfx11 kernels with gfx12's lane layout) instead of MMQ. Earlier
    engines ignored the flag on RDNA4 (MMQ stayed, no banner); now the engine prints `strata: prompt experts on the fused
    int8 kernels (STRATA_PF_FUSED=1, #136)` once, at the first prompt. It is not an arch default: it rounds differently
    from MMQ, and those defaults hold only bit-identical switches. `STRATA_PF_FUSED_NATIVE=0` keeps MMQ with the flag set
    (for an A/B). R9700 (gfx1201, ROCm 7.14, Coder IQ1_M pack), prompt time as the server reports it, alternating
    starts per arm: 4K 1,612 -> 1,353 ms (+19.1%), 16K 5,579 -> 5,147 ms (+8.4%), 32K 11,890 -> 10,962 ms (+8.5%); a
    second run of 3 starts per arm gave +12% to +16% at 4K (it depends on the start) and +8.3% at 16K; the
    kernels alone are 1.48x / 1.20x / 1.14x MMQ's at 4K / 16K / 32K. Quality: first-token KL against the FP16 prompt path
    at 4K (48 prompts) is 1.10x MMQ's, which that sample cannot tell apart from MMQ's; top-1 agreed on 48 of 48 (every
    prompt's top token was the same one, so that check says little); 4 of 4 long-context retrieval checks passed.
    0.1.42 re-check on an R9700 (Linux, ROCm 7.14), 12 prompts of 1.5K-24K tokens, first-token KL against the FP16 prompt path (MMQ / fused): IQ3_S 0.00332 / 0.00284, top-1 12 of 12 for both; IQ3_XXS 0.00335 / 0.00452 on one sample and 0.00344 / 0.00323 on a second (24 prompts pooled: 0.00340 / 0.00388, top-1 24 of 24 for both); the Q2_0 pack does not take these kernels on gfx12 (output bit-identical to MMQ). Because the IQ3_XXS pool is not at or below MMQ's, the flag stays opt-in. Measured on Linux only.
    gfx1200 builds the same kernels and was not run; Windows and K-quant packs (`STRATA_PF_FUSED_KQ=1`) were not measured.
    **Q2_0 gate/up (148-amd).** Windows: opt-in until tested on Windows (`STRATA_PF_FUSED=1`); the default below is Linux only. The native Q2_0 pack's gate/up (64-value blocks) take the same kernels and are in the gfx12 default set with IQ3_S and
    IQ3_XXS (`STRATA_PF_FUSED=0` keeps MMQ). One layer, 2,048 tokens x top 10 over 512 experts: MMQ 24.5 ms, fused 4.4 ms; 8,192 tokens 31.7 vs 13.2 ms
    (`hip_prefill_fused_iq`, fused against MMQ rel. RMS 1.6e-5). Engine, R9700 Q2_0 32 GB, 5 interleaved pairs: prompt 4K +16.7%, 20K +19.0% (noise arm 1.001 / 0.999).
    Quality: teacher-forced KL against MMQ 0.0004 (6,022 positions, top-1 99.5%) and 0.0016 (1,760 long-document positions, top-1 98.2%), the same tasks as
    above unchanged (156/164, 24/30, 6/6, 6/6, 8/8, 5/5). IQ2_XXS / IQ2_XS / IQ2_S gate/up stay opt-in on gfx12.
    **Both defaults together (148-amd), R9700, interleaved starts of one engine, prompt tok/s** (base = the build before them; the new paths change
    bits, so decode texts differ from the first differing token on and decode is compared with both switched off instead: 95.23 vs 95.28, same ids):
    Q2_0 32 GB, 5 pairs: 4K 1,151 -> 2,286 (a slower start mode gives 1,988), 20K 1,384 -> 2,751 (2,481); with one lever each (3 pairs): WMMA 4K x1.35 / 20K x1.46,
    Q2_0 fused experts x1.28 / x1.19, both x1.73 / x1.79. IQ3_XXS 32 GB, 5 pairs (only the attention is new): 4K 1,369 -> 1,436 (the base starts in a 1,170 or a
    1,370 mode), 20K 1,576 -> 1,997 (x1.27). The identity gate with both switched off (STRATA_HIP_WMMA=0, STRATA_PF_FUSED=0): 10 of 10 prompts and the
    4K / 20K residual dumps identical on both packs.
  - **`STRATA_PF_GEMM` / `STRATA_PF_PAD` are gfx11 only** (0.1.46): their product is `strata_pf_gemm_f16_ld`,
    built into the gfx11 matrix-core block of `wmma_gemm.cu`, so an RDNA4 card runs neither. Until 0.1.46 the padded
    half was worse than inert: `STRATA_PF_PAD=1 STRATA_PF_GEMM=1` let the two K 6144 producers write row stride 6,208
    into a GEMM that only the gfx11 kernel takes, and `Gemm::native` ended the engine (`prefill gemm: a padded X (ldx
    6208, K 6144) needs STRATA_PF_PAD's path`, exit code 1 - under `serve/` an engine restart loop per request).
    Measured on an RX 9070 (gfx1201, 16 GB, system ROCm 7.2.4, kernel 7.2.11, 31 GiB RAM, Coder IQ1_M): 3 of 3 starts
    exited that way at `--vram-reserve-mib` 700 and 1,536, so it was not the VRAM guard. Now the producer asks
    `strata_pf_padded_gemm_available()` and does not pad where no kernel reads the stride (one line at the first prompt),
    and `Gemm::native` copies a padded X's rows unpadded and runs the ordinary paths instead of exiting - checked with
    the gate forced off on that card: 4K prompt answers byte-identical to the unpadded run, no exit. `STRATA_PF_GEMM=1`
    alone is inert there too (4K 1,247 against 1,243 tok/s, 8K 1,330 against 1,321, medians of 3 starts, i.e. noise).
- **Known:** rarely (about 1 start in 10) a HIP run's greedy output differs from another start's at some token, on
  one card or two and on engine 0.1.29 as well; not yet explained.
- **Pictures:** the card's reader (`strata-vision-hip`) is measured on an R9700 (gfx1201): 57 ms for a 260-token picture and 0.40 s for a
  972-token one, against 3.13 s and 14.1 s on the CPU reader (EPYC, 32 threads); the embeddings agree with the CPU reader's (cosine mean 0.99985). It keeps
  about 1.4 GiB of VRAM (+1,470 MiB on the R9700). A test picture (red circle, blue square, green triangle, the text "STRATA 42") was answered correctly with both readers, and the
  fallback to the CPU reader was tested with a reader built for another card. The same reader built for gfx1103 ran on a Radeon 780M (0.40 s for 260 tokens
  against 2.42 s on its CPU). It has not run on an RDNA3 card (gfx1100-gfx1102) or on Windows with an AMD card.
- **Not validated:** long contexts beyond 16K, answer-quality benchmarks.

## Integrated GPUs (mini PCs)

On a Radeon that shares the computer's memory (the Ryzen mini PCs: Radeon 780M / 760M `gfx1103`, 890M / 880M `gfx1150`,
Strix Halo `gfx1151`) the matrix-core prompt set below is on by default on the Radeon 780M (`gfx1103`, Linux) and opt-in
(`STRATA_IGPU_PROMPT_SET=1`) on the others: it changes bits, and only the 780M was measured. The engine picks the set from the device's own "integrated" flag and
the architecture, not from the name: a discrete card (an RX 7900 XTX `gfx1100`, an R9700 `gfx1201`, any other) never gets it,
and neither does a computer that also holds a discrete card next to the iGPU. The engine prints one line at start-up that names
what it set.

| Switch | What it does |
|---|---|
| `STRATA_HIP_WMMA=1` | the prompt attention on the matrix cores (needs `--kv int8`) |
| `STRATA_PF_GEMM=1` | the prompt's FP16 projections on the 128 x 256 WMMA GEMM |
| `STRATA_PA_FAST=1` | the prompt attention with single FP16 q and p |
| `STRATA_SELECT_WMMA=1` | the block scorer on the matrix cores |
| `STRATA_HC_UPMIX=1` | the hyper-connection up projection with its epilogue |

**Why.** (148-amd first measurement; 021-minipc repeated it with 6 interleaved pairs: prompt x1.60 at 1K, x2.03 at 4K, decode x0.999, identical greedy ids.) The dense projections are 63% of a prompt's GPU time on a 12-CU iGPU (the experts only 16%), and they are what these
switches speed up. Radeon 780M (P2006, Ryzen 7 255, Q2_0, `--kv int8`, 4K prompt, one run each; 148-amd): 109.5 tok/s with
none of them, 214.7 with the first four (+96%), 233 with `STRATA_HC_UPMIX` as well (+111%); decode is the same (16.6 against 17-19
tok/s, inside the run-to-run noise). The prompt switches do not touch the decode kernels.

**What it costs.** They round differently from the old path (another summation order, half-precision intermediates), so the answers
differ in the last bits. Measured on an R9700 with the same kernels (Q2_0): teacher-forced KL 0.0003 (28 prompts) / 0.0014 (eight
~6K-token documents), and the task sets inside their run-to-run noise.

**Turning it on and off.** `STRATA_IGPU_PROMPT_SET=0` turns the whole set off, `=1` turns it on where it is not the default; each switch keeps `=0` as its off switch
(`STRATA_HC_UPMIX=0`); `STRATA_IGPU_DEFAULTS=0` turns the set off whatever else is set. A switch you set yourself, to anything, is never overridden. Put them in the server config's `env`, or in the shell.

**Windows.** The same code applies, but it has not been tested on Windows (there was no Windows iGPU machine to run it on).

**Strix Halo (`gfx1151`).** Its exact speed switches are separate and unchanged ([STRIX_HALO.md](STRIX_HALO.md)); the five above
were already its "maintainers' fast configuration". The set was measured on the 780M only.

**The 780M resets its GPU.** On the 780M's current Ubuntu firmware (gc 11.0.1 MES) a GPU reset can follow when the kernel evicts the
process's GPU queues while a kernel with barriers is running ([RADEON_780M.md](RADEON_780M.md), `notes-minipc.md`). It is
not caused by the prompt set: the default build without it resets the same way (forced compaction every 10 s killed 2 of 2 runs).
`STRATA_ARENA_PIN_GIB=none` (experimental, exact) leaves the expert arena unregistered, so that the kernel's page migration has
no big GPU mapping to invalidate; the experts then reach the GPU by copy. Measured on the 780M (30 interleaved runs, 021-minipc): the
eviction time drops from about 6 s to about 0.13 s per run, but resets still happened (4 of 20 runs against 1 of 10), and prompts/decode
were 1-4% slower, so it is not the default. A newer `linux-firmware` (MES) is the likely cure and is a host change the engine does not make.

## Community-validated cards

Run by their owners, not on the maintainers' machines; setup accepts them like gfx1100 / gfx1201. setup ships a
hipBLASLt table for gfx1200 (the numbers below); gfx1101 has none (make one with [Tuning table](#tuning-table)
and compare the prompt speed with and without it).

- **gfx1101, RX 7800 XT 16 GB** (jhohertz, #254; engine 0.1.29, Ryzen 9 5950X, 121 GiB RAM, system ROCm with
  hipBLASLt 1.4.1): `./setup.sh --backend hip` detected the card and compiled the engine; `strata-device --selftest`
  passed; ctest 30/32 (`ple_parity` needs the Q2_0 fixture, `platform_memory_test` the memlock limit). Coder IQ1_M,
  64K context, MTP, with a table the owner calibrated: fresh prompts of 4K-9K tokens at 898-953 tok/s, decode
  38-44 tok/s (128 tokens).
- **gfx1200, RX 9060 XT 16 GB** (Efeisot, #256 after #176; engine 0.1.29, Ryzen 9 7950X, 64 GiB RAM, ROCm 7.2 with
  hipBLASLt 1.2.2): ctest 32/32 (without `ple_parity` and `platform_memory_test`). Coder IQ1_M, greedy, MTP
  `--spec 4 --spec-min-p 0.5`, with a table the owner calibrated:

  | prompt | prompt speed | decode |
  |---|---|---|
  | 2,374 tokens | 540 tok/s | 27.1 tok/s (0.69 draft acceptance; 31.0 at 1.00) |
  | 65K (`--kv int8 --kv-resident 65536 --prefill 16384`) | 748-753 tok/s | 26-40 tok/s by acceptance |
  | 130K (the same flags) | 725 tok/s | - |

  Greedy output was the same across runs. For comparison, llama.cpp's HIP build measured 20 tok/s decode and
  450 tok/s prompt on that card.

  Since 0.1.38 setup ships that table (`tools/hip/gfx1200-hipblaslt-100202.txt`). On engine 0.1.31, one binary (ctest
  37/37, without `ple_parity` and `platform_memory_test`), prompts of repeated code blocks, prefill 2560 for
  the short prompt and 16384 with `--kv int8 --kv-resident 65536` otherwise, two runs each - prompt speed
  with the table vs plain hipBLAS (no table):

  | prompt | with the table | plain hipBLAS | gain |
  |---|---|---|---|
  | 2,374 tokens | 597-599 tok/s | 396-398 tok/s | 1.50x |
  | 65,045 tokens | 754-757 tok/s | 380 tok/s | 1.99x |
  | 130,091 tokens (the same flags) | 719 tok/s | 363-370 tok/s | 1.98x |

## RDNA2 (gfx1030)

The RX 6800 / 6800 XT / 6900 XT / 6950 XT run the same kernels: wave32 and 64 KiB of LDS per workgroup. The one
difference is the dot instruction: RDNA2 has no `v_dot4_i32_iu8` (gfx11 and newer), so
`dp4a` uses the plain signed `v_dot4_i32_i8` through `__builtin_amdgcn_sdot4`, which compiles to a single
`v_dot4c_i32_i8` (same signed x signed byte products, modulo 2^32). There is no WMMA; the QSA scorer takes the
same ordered FP32 fallback as gfx1100. CMake lists gfx1030 as unvalidated (the build warns) until a maintainer has
run it; the report below is from a community machine: an RX 6900 XT 16 GB (gfx1030), an i7-13700KF (8 P-cores and
8 E-cores, AVX2, no AVX-512), 63 GB RAM, NixOS, ROCm 7.2.3 from nixpkgs (clang 22, hipBLAS 3.2).

- **Build and tests** (engine 0.1.26): a complete HIP build for gfx1030, made by hand with cmake and ROCm's own
  `clang++` (the nixpkgs ROCm is not an `/opt/rocm` tree, so setup's `build_engine_hip` was not exercised; the
  binary was placed in `engine/` for setup to use). All 28 registered ctest tests pass on the card, including
  `hip_device_selftest`. An engine 0.1.30 build of this branch configures, builds and runs clean on the same card;
  the speeds below are measured on it.
- **End to end** (Swift 1.5 IQ3_XXS, `--context 131072` with `--kv-resident 32768 --adapt-every 1
  --vram-reserve-mib 1024`, 200 greedy tokens): 38-42 tok/s decode with the default 15 CPU pool workers (one per
  physical core except the host thread; 8 workers: 36; 24 = every logical core: 27), consistent even with the
  131,072-token context full.
- **Prefill** (the same 15-worker configuration): 246 tok/s on a 2,000-token prompt, 330-339 tok/s at the auto
  8,192-token chunk (7,997 and 15,967 tokens; time to first token 8.9 and 48.3 s), 133 tok/s on a 522-token
  prompt - short prompts are fixed overhead (19-25 tok/s on 34 tokens). Decode after a 16K prefill holds at
  45.5 tok/s.
- **16 GB card:** the expert cache holds 6,310 slots (10.2 GiB) at 16K context and 5,247 slots at 131,072, where
  setup keeps the KV cache in VRAM because IQ3_XXS needs about 60 GB of RAM plus the cache to stream it. The
  `--pcie-frac 0` and `--adapt-every 0` of the 7900 XTX configuration in
  [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md) cost 8.6 and 13.6 tok/s here (30 with the defaults): keep the
  defaults on a 16 GB card.
- **hipBLASLt:** ROCm's hipBLASLt ships no gfx1030 kernels, so there is no table and the plain hipBLAS path runs.
- **Prompt GEMMs in FP16 (#835):** rocBLAS on gfx1030 has tuned kernels for FP16-in / FP16-out GEMMs only; the prompt
  path's FP16-in / FP32-out and BF16 products ran generic kernels about 6.6x slower (5.6 and ~5.3 TFLOPS against 37.7 at
  N = 10240, T = 7313, K = 2560). `STRATA_HIP_PROMPT_F16=1` (opt-in, off by default; the engine prints a tip on gfx103x) runs the 16-bit prompt GEMMs FP16 in and out;
  unset is the old path. `STRATA_DBG_NAN=1` also counts the non-finite FP16 outputs (a sum past 65504). The output is not bit-identical to it (an FP16 rounding of each GEMM's output replaces the BF16 rounding of
  the BF16 GEMMs' activations); what the distribution check showed and did not show is in
  [bench/results/2026-10-04-rdna2-fp16-prompt](../bench/results/2026-10-04-rdna2-fp16-prompt/README.md). FP16 ends at
  65504: `STRATA_F16_RANGE=1` prints, per prompt, the largest value that entered or left FP16 and how many were beyond
  the range (466K tokens there: outputs peak at 410, activations at 112, none beyond; the image writers saturate, a GEMM
  output beyond the range would be Inf and is what the counter is for). Measured on a
  second community machine - 2x RX 6900 XT 16 GB (one card for these numbers), Ryzen 5 5600X (6 cores, AVX2), 128 GB,
  Ubuntu 26.04, ROCm 10.0.0, engine 0.1.39 built by hand with `-DSTRATA_PREFILL_MMQ=ON` as setup does, GSQ-RCO IQ3_S:
  a 9.4K / 34.7K / 105.8K-token prompt reads at 744 / 915 / 926 tok/s instead of 439 / 466 / 461; decode is unchanged.
  On that machine the HIP ctest passes 60 of 65 with and without this change: 2 skipped (`hip_prompt_attn_wmma`, no
  matrix cores; `hip_prefill_hipblaslt_gemm`, no hipBLASLt table) and 3 that fail for reasons outside the engine
  (`ple_parity` needs a Q2_0 PLE file that is not on that machine, `expert_multi_test` refuses the CPU without AVX-512,
  `platform_memory_test` cannot `mlock` at the shell's default `ulimit -l`).
- **The verifier's hyper-connection read as one persistent kernel:** `STRATA_HC_PERSIST=1` (opt-in, HIP, 32-lane
  waves) runs the read of 1-8 tokens as one kernel of 80 blocks with two grid barriers instead of three launches; a
  start-up run per token count with a barrier timeout decides where it is resident, and one `strata hc:` line per card
  says so. On 2x RX 6900 XT (ROCm 10.0, layer split): the read 1.35-2.10x, ms per verify window -4.2 % / -2.8 % at
  4K / 32K (6 rounds), not bit-identical (as close to a double-accumulation reference as the default read):
  [bench/results/2026-10-12-hc-persist-gfx1030](../bench/results/2026-10-12-hc-persist-gfx1030/README.md). Other
  RDNA cards are untested; the `strata hc:` line and a `STRATA_DECODE_TIMING` run from one are welcome. Not offered on gfx11 and
  gfx12 (RDNA3, RDNA4): the barrier timeout needs `s_memrealtime`, which gfx1100 and gfx1200 / gfx1201 do not have, and an R9700 (#2226 thread) measured the
  read +0.7 % / +1.0 % per window slower than the default one; the start-up line says so. A barrier that times out at run time
  (never seen in a working run) leaves the partial outputs of that read in place and every later persistent read of the
  process aborts at once, without a log line; the opt-in is for measurement on gfx103x until that is closed.
- **Prompt GEMMs as SGEMMs (engine 0.1.39 + `Gemm::rdna2_sgemm`):** the rocBLAS in setup's ROCm (10.2.0a20260930)
  has tuned gfx1030 kernels for FP16 -> FP16, int8 and FP32 -> FP32 only. The prompt path's dense GEMMs take FP16 or
  BF16 inputs and write FP32 (rocBLAS's HS / BS types), which run its fallback kernels at about 5 TFLOPS. With
  `STRATA_RDNA2_SGEMM=1` (opt-in; `=all` forces it on any AMD card) the engine widens both inputs to FP32 (exact) and runs SGEMM,
  which accumulates in FP32 like the native call (`src/prefill/gemm.cu`; N < 64 keeps the native call). The author ran it
  as the gfx103x default; here it is a switch because it is not bit-identical to the native call (below). Measured on an RX 6800 16 GB over OCuLink (PCIe 4.0 x4, 7.1 GB/s), Ryzen 7 8845HS, 28.8 GB RAM, Windows 11,
  Coder IQ1_M with setup's arguments (64K context, MTP), fresh 12K-token prompts, 4 per arm, A/B in one binary:

  | | native GEMMs | SGEMM route |
  |---|---|---|
  | prompt | 312 tok/s | 452 tok/s |
  | prompt GPU timeline (`STRATA_PREFILL_TIMING`) | 37.2 s | 25.5 s |
  | of which gdn / qsa proj / hc read | 9.4 / 3.6 / 3.85 s | 2.8 / 1.4 / 2.84 s |
  | decode | 29-34 tok/s | 29-35 tok/s |

  The exact calls on the card (opA = T, opB = N, FP32 out): N 10240 x T 8192 x K 2560 86.7 -> 27.7 ms including
  the widening; N 320 x T 8192 x K 10240 (BF16) 10.9 -> 7.5 ms. Against a float64 reference on random inputs in
  [-1, 1], the SGEMM route is as close as or closer than the native call on all 10 FP16 / BF16 shapes checked; the
  two differ by at most 4.3e-6 of the largest output. A 9-task code-repair benchmark (the model patches seeded bugs
  in a browser game until 38 browser checks pass) solved 9/9 on the first try with both, model time 458 -> 374 s.
  With the switch on and `STRATA_HIP_PROMPT_F16` unset the SGEMM route runs; with `STRATA_HIP_PROMPT_F16=1`
  the FP16 route above runs first and the SGEMM route takes the products it leaves (`beta != 0`). Same RX 6800, one
  session, fresh 12K-token prompts: native 310, SGEMM 439, FP16 (#835) 494 tok/s; `STRATA_F16_RANGE=1` on the Coder
  IQ1_M peaked at 105.3 (activations) and 424.8 (FP16 outputs) with nothing beyond 65504, and the repair benchmark
  solved 9/9 on the first try with both (model time 374 s SGEMM, 307 s FP16). The SGEMM route is the slower of the
  two; no value passes through FP16, so it has no 65504 limit to stay under. It is not bit-identical to the native
  call (at most 4.3e-6 of the largest output apart), which can flip a near-tied greedy token like any such change.
- **rocBLAS solution table** (for the `STRATA_HIP_PROMPT_F16=1` route above; with it off, nothing runs through the
  table): rocBLAS picks the kernel for each of those FP16 GEMMs from its own table by shape, and
  on gfx1030 that pick is far from the best kernel the library holds for some shapes: the GDN input projection
  (N = 10240, K = 2560) below about 1,150 tokens runs at 5 TFLOPS where the best of rocBLAS's own 237 solutions for the
  same shape runs at 32-38 (7x; the kernel choice flips at T = 1152). `tools/hip/tune_rocblas` enumerates the library's
  solutions per shape and token bucket, checks each against the default kernel's output and writes the winners as
  `tools/hip/<arch>-rocblas-<version>.txt`; the engine reads the table through `STRATA_ROCBLAS_TUNING` (a file, or
  the directory, from which it takes the file of its own rocBLAS build) and runs those solutions in place of rocBLAS's
  choice, the default kernel for every shape without a row. Solution indices are valid for one architecture and one
  rocBLAS build (the full version string, `5.6.0.8d1ae90e` for ROCm 10.0.0's), and the engine refuses any other
  table. Shipped: `gfx1030-rocblas-5.6.0.8d1ae90e.txt` (ROCm 10.0.0, the `/opt/rocm` setup installs on Ubuntu), measured
  on an RX 6900 XT. Other gfx1030 cards (RX 6800 / 6800 XT / 6950 XT, Radeon PRO W6800 / V620) run it as well - the
  indices are valid on every gfx1030 - but have fewer CUs or other clocks, so their own table may pick differently;
  `tune_rocblas` makes one in about 5 minutes ([Tuning table](#tuning-table)), and a report of what it changed is welcome. Setup
  points the engine at the directory when a table for the card's architecture exists, and the engine's log says
  `rocBLAS tuning enabled (... rows ...)` or why not. Measured on the 2x RX 6900 XT machine above (one card, #835 +
  this, `STRATA_PREFILL_TIMING=1`, temperature 0, the same answers with and without): a 799-token prompt's GPU time
  3,329 -> 2,985 ms (the GDN projections 443 -> 177 ms, `hc read` 270 -> 77), 495 tokens 2,477 -> 2,139 ms (365 ->
  146), 8,874 tokens 11,978 -> 11,459 ms (the 682-token tail chunk's projections, `hc read` 1,200 -> 898); a
  1,108-token prompt did not change (3,391 -> 3,395 ms): its GDN time fell 598 -> 201 ms and the whole saving
  reappeared as `wait copy`, the experts' PCIe transfer, which bounds that prompt. Per-run lines and the tuner's
  summary: [bench/results/2026-10-05-rdna2-rocblas-table](../bench/results/2026-10-05-rdna2-rocblas-table/README.md). Decode does not use these GEMMs and is unchanged. See [Tuning
  table](#tuning-table) for making a table for another rocBLAS build.
- **QSA block scores on the FP32 tiled scorer (`STRATA_SELECT_SIMT=1`, opt-in, as on CUDA below sm_80):** without
  matrix cores the prompt path's QSA selection scores every (query, KV block) pair on the warp kernel, the one prompt
  term that grows with the context. The switch runs the FP32 tiled scorer instead (FP32 in another summation order:
  not bitwise the warp kernel, so off by default), tiled 32/128/16/8 on gfx103x. `qsa_select_bench`, 256 queries:
  2.69 -> 0.587 ms at 128K (the original 32/64/32/4 tiling: 0.695), 0.67 -> 0.169 ms at 32K; FP64 gate passed,
  selections identical 256/256. Every tiling gives the same bytes (each score is one fmaf chain over k in order), and
  the fastest tiling was the same with 60 and 72 of the card's 80 CUs (`HSA_CU_MASK`). On the 2x RX 6900 XT split
  (0.1.40.3, IQ3_S, `--kv int8 --kv-resident 32768`, two cold starts per arm, ABBA) a 128K-token prompt reads 1,735 ->
  1,861 and 1,727 -> 1,858 tok/s (+7.4%), 32K +0.8%; decode is not affected (the text changes with the scores' order,
  so its speed is not compared). Measurements:
  [bench/results/2026-10-07-rdna2-qsa-select-simt](../bench/results/2026-10-07-rdna2-qsa-select-simt/README.md).
- **File-tier cache rule on gfx906 (`STRATA_FILE_CACHE_OLD_RULE`):** on an MI50 ROCm queues are suspended while the kernel reclaims around the process's HMM-tracked arena (#1752, KevinX8: stalls, prompt 46 -> 360 tok/s once the file tier stops churning the page cache). The gfx906 backend therefore keeps the old rule (the file cache is used only when the room left is at least the bytes read from the files); other HIP cards keep the current rule. `STRATA_FILE_CACHE_OLD_RULE=1` forces the old rule on any Linux build, `=0` the current one.
- **Windows** (the report above): setup's ready-made Windows engine 0.1.39 runs the RX 6800 beside a 780M iGPU
  (HIP numbers the RX 6800 1; setup points the engine at it), and `tools/hip/build_windows.bat` with
  `STRATA_HIP_ARCHS=gfx1030` builds a working engine from the `gfx103X-all` 10.2.0a20260930 wheels.
- **gfx1031** (RX 6700 XT, #524): setup knows it (the `gfx103X-all` wheels, unvalidated); its reporter runs it daily
  on one card. More reports: an RX 6700 XT 12 GB run as gfx1030 on ROCm 7.2.4 (#1027: IQ2_XS, decode 30-32 tok/s, prompt about 300 tok/s, 6 of 6
  needles), and an RX 6800M 12 GB on Windows with a self-built engine (#915, #1078: Q2_0, decode 9-27 tok/s, prompt 50-114 tok/s; the
  experts stream from disk with 31 GB of RAM). The ready-made Windows zip has no gfx1031 code unless it is built with it: `STRATA_HIP_ARCHS=gfx1031`
  in `tools\hip\build_windows.bat` (the default list has it from 0.1.40.2 on).
- **gfx1150** (Radeon 890M, Ryzen AI 9 HX PRO 370, Strix Point, #1217): builds and runs from 0.1.40.2 as an unvalidated target (CMake warns; the
  device code already covers it: the WMMA guards and `gfx_arch_is_gfx11_wmma()` include it). One community machine: MINISFORUM N5 PRO, 96 GB DDR5-5600, GTT
  raised to 64 GiB (`ttm.pages_limit=16777216 ttm.page_pool_size=16777216`), TheRock ROCm 7.14.1 for gfx1150, an unprivileged LXC container, IQ3_XXS: chat and
  tool calls work, decode 15-19 tok/s, the HIP ctest passes apart from tests that need files a public checkout lacks.
  Prompts read at about 125-130 tok/s on plain hipBLAS and 1.5-1.7x faster with `tools/hip/gfx1150-hipblaslt-100401.txt` (ROCm 7.14.1's
  hipBLASLt; 3.6K tokens 124 -> 216 tok/s, 7K 130 -> 226 tok/s, decode unchanged):
  [bench/results/2026-10-07-community-gfx1150](../bench/results/2026-10-07-community-gfx1150/README.md).
  Setup does not install for it yet (an integrated Radeon other than Strix Halo is named and not supported): build by hand with `-DCMAKE_HIP_ARCHITECTURES=gfx1150`
  and point `STRATA_HIPBLASLT_TUNING` at the table yourself.
- **gfx1152** (Radeon 860M / 840M, Ryzen AI 300 "Krackan", #1625): in CMake's unvalidated list like gfx1150, so
  `-DCMAKE_HIP_ARCHITECTURES=gfx1152` builds with a warning. It runs the portable kernels (the matrix-core guards name
  gfx1150 and gfx1151 only), no card has reported on it, and setup does not install for it (an integrated Radeon other
  than Strix Halo is named and not supported). A report from a real 860M is welcome.
- **gfx1032** (RX 6600/6600 XT): unvalidated; built with `STRATA_HIP_ARCHS=gfx1030;gfx1032` and ran under Windows as
  the *helper* expert GPU of an RX 6800 ([SECOND_GPU.md](SECOND_GPU.md)), 3.4-3.8k expert slots in its 8 GB and
  +8-18% decode over the 6800 alone. Its 14 WGP make a helper row about 2-3x the 6800's cost, so it pays as a
  cache tier for the CPU pool, not as a compute peer. No report of it as a primary card.
- **Not validated:** setup's own build path and the
  `gfx103X-all` wheels on gfx1030, images, answer-quality benchmarks. RDNA1 (gfx1012, RX 5500 XT) builds by hand:
  [OLDER_GPUS.md](OLDER_GPUS.md#amd-building-gfx906-and-gfx1012).

## gfx906 (Instinct MI50 / MI60, Radeon VII): wave64, built from source

gfx906 is wave64 and has no WMMA and no packed byte arithmetic, so the wave32 backend above refuses it. A separate opt-in build compiles the CUDA sources as HIP through a small compat layer
(`include/strata/platform/hip_compat/`), with a CUDA warp mapped to a logical half of the 64-lane wavefront
(32-wide shuffles, a ballot of its own half). The hot kernels have wave64 layouts of their own (below). Setup does
not build it yet: build by hand, and run `serve/server.py` with a config as on any other card.

**Community-tested, opt-in (0.1.42).** The gfx906 build is maintained with its users: we have no gfx906 card, so what is here was measured by community members on their MI50s (the attention switches `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` and `STRATA_GFX906_ATTN_REDUCE12=1`, PRs #1661 and #1718 by 0FL01, are off by default and compiled only into this build) and the engine here is checked to compile for gfx906, nothing more. Nothing in it changes the builds for other cards. A wave32 card seen by this build is refused at start with a message (#1728). Please report how it runs on your card.

**ROCm.** AMD's current ROCm releases no longer ship gfx906 libraries. The build and the measurements below used
HIP 7.14 from the community image [`mixa3607/rocm-gfx906:7.14-complete`](https://hub.docker.com/r/mixa3607/rocm-gfx906)
(rocBLAS/hipBLAS with gfx906 kernels), on the kernel's amdgpu driver (Ubuntu 24.04, kernel 6.8). hipBLASLt is not
used.

```sh
# inside the image, with this checkout at /src and llama.cpp at the pinned commit in third_party/llama.cpp
apt-get update && apt-get install -y cmake ninja-build git python3 build-essential
cmake -S . -B build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build-906 strata
```

`STRATA_HIP_GFX906` turns on the CUDA targets and refuses `STRATA_ENABLE_HIP` (one HIP build at a time). Run the
container with `--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit memlock=-1`; on some boards
`HSA_OVERRIDE_GFX_VERSION=9.0.6` is needed for the runtime to accept the card.

0.2.1 builds `ninja -C build-906 strata` again (0.1.44 to 0.2.0 stopped at `cublasSgemm` in `src/qwen35/q36_engine.cu`, at
`cuda.h` in `strata_mmq_w4a4` and at three link errors, #3172 #3271; compiled for gfx906 on ROCm 7.10 with clang 22, all 161 targets,
not run on a card). The Qwen3.6 family (`qwen35moe`) is not part of the gfx906 build: its engine calls cuBLAS entry points the compat
layer does not map and its kernels assume 32-lane warps. `strata --families` lists what the binary runs (`qwen4exp`: Qwen3.8-Flash-Next).

**What differs from the CUDA build** (all under `STRATA_HIP_GFX906`; each A/B switch restores the CUDA layout):

| | gfx906 | switch |
|---|---|---|
| grouped native experts | mode 7: signs as `dp4a(g ^ m, u) - dp4a(m, u)` (no byte SIMD), grids and activations in LDS, one weight load per group; the IQ formats only (Unsloth's K-quant / Q5_1 / Q8_0 experts take the CUDA layout) | `STRATA_EXP_MODE=2` (previous AMD layout) |
| MMVQ | one wavefront per row, 64-lane butterfly, no LDS | `STRATA_MMVQ_WAVE=0` |
| router top-10 | one wavefront, register argmax | - |
| verify-window routing (256-expert router) | the window's tokens in one multi-column BF16 projection + one top-k, ~3 ms of a ~55 ms window | `STRATA_ROUTE_PER_TOKEN=1` |
| hyper-connection read | norm loads in flight, 8 lanes per row for `up` | `STRATA_GR_FAST=0` |
| `gr_down_multi`, opt-in | split along K (8 rows x 5 slices of 2048), ~3% per window; sums in another order than the single-token read, so off by default | `STRATA_GR_SPLIT=1` |
| IQ4 table lookup | llama.cpp's `v_perm_b32` sequence (HIP's `__byte_perm` is a scratch array) | - |
| QSA prompt attention | the FP32 kernel (no tensor cores / WMMA) | - |

Every switch above was checked bitwise against the layout it replaces: the parity tests on synthetic data,
`native_expert_bench` on real GGUF rows (one mode against another), and the greedy text of a fixed request.

**Measured** on 2x Instinct MI50 16 GB (85 W power limit each, PCIe 3.0 x16 both, peer access between them),
Xeon E5-2666 v3 (10 cores, AVX2), 32 GB DDR4, SATA SSD; Coder IQ1_M (`--native` pack), 131,072-token context,
`--kv int8 --kv-resident 32768`, `--layer-split 27`, `--prefill 4096`, MTP `--spec 4 --spec-min-p 0.5`,
`--pcie-frac 0`, `STRATA_ARENA_MMAP=1`. Measured on the port this build was cut from (builds on the 0.1.30 and 0.1.33 bases, the same
kernels, plus the layer-split weight trim and the mapped arena, which are separate pull requests):

- all 12,288 experts held in VRAM across the two cards;
- a 17,043-token agent prompt (Claude Code's first request, tools included) read in 43.5 s (392 tok/s), decode
  on it 57.7-59.3 tok/s;
- 25 GB of the 32 GB RAM available while serving;
- six repeats of that request with a 6,000-token output cap and a 900-second two-client load test run six times
  (407 requests, 202,940 tokens): no stalls, no errors, no GPU faults.

For comparison on the same machine: llama.cpp (3cf0325, ROCm) on the same Coder IQ1_M measured 26.9 tok/s `tg128`.

**Not done:** setup.py detection and an automatic build, Windows, images (`--vision`), MI60 and Radeon VII (the
same gfx906 ISA; not run), a single-card run, the tensor-split experiment (the halves of every layer on two cards;
it works but is not part of this build).

**Two cards: try `--pipeline-windows 2` (community measurement, #3173; off by default on gfx906).** On 2x Instinct MI50 16 GB
(Ryzen 7 5700X, 64 GB DDR4-3600, ROCm 7.14 from the community image, engine 0.1.43.1) with the original IQ3_XXS at 131,072 tokens
of context, `--layer-split 27`, `--spec 3 --kv int8 --kv-resident 32768`, median of 7 runs (1.6K-token prompt, 128 greedy tokens):
decode 49.2 tok/s without it, 66.9 with `--pipeline-windows 2` (+36%); the prompt speed stayed 353 tok/s. The first card starts the
next verify window while the second still verifies this one ([MULTI_GPU.md](MULTI_GPU.md)). On RDNA4 the tokens are the serial
loop's up to CPU rounding; nobody has compared them on gfx906 yet. Add `--trim-stage-weights` (with the explicit `--layer-split`): prompt reading 353 -> 371 tok/s, decode +2%.
In the config: `"args": [ ..., "--pipeline-windows", "2", "--trim-stage-weights" ]`. One machine, one model, one run protocol: measure it
on yours (`--pipeline-windows 0` is the plain loop) before you keep it. `--batch` did not help one user there (aggregate 20 -> 23 tok/s
at two requests, each request slower), so leave it off for a single interactive user.

## Tuning table

A hipBLASLt table holds solution ids that are valid only for one GPU architecture and one hipBLASLt version, so it
is calibrated on the card with the ROCm the engine runs with. The shipped gfx1100 table's rows are the engine's
dense GEMM shapes; to calibrate them for another card or version (a few minutes):

```sh
cmake --build build-hip --target tune_hipblaslt
CASES=$(awk 'NR>2 {printf " --case %s,%s,%s,%s,%s", $1, $5, $2, $3, $4}' tools/hip/gfx1100-hipblaslt-100200.txt)
./build-hip/tune_hipblaslt $CASES --tuning-out table.txt
```

The file's second line names the architecture and version (`STRATA_HIPBLASLT_TUNING_V1 gfx1201 100401`); save it
as `tools/hip/<arch>-hipblaslt-<version>.txt` for setup, or point `STRATA_HIPBLASLT_TUNING` at it. Compare the
prompt speed with and without it before keeping it. For an engine package, `packaging/stage_tuning.py` (Linux builds) and
`tools/hip/package_windows.py` copy the tables that match the bundled hipBLASLt into the package's `tuning/` folder.

**Measured for 0.2.1 on a Radeon AI PRO R9700 (gfx1201), the Linux `rx9070` package 0.1.45.3, Q2_0, hipBLASLt 1.2.0, one card, the
box to itself.** A 1.2.0 table (32 rows, `tune_hipblaslt` over the gfx1201 100500 table's cases) against plain hipBLAS, 6 interleaved pairs
with a fresh engine per run and a second plain arm as the noise arm: prompt 4K 2,302 vs 2,119 tok/s and 20K 2,671 vs 2,507 (5 and 6 of 6 pairs
faster, but a plain run alone lands on either of two speeds, 1,970 or 2,290 at 4K, so the medians mix modes; inside one mode the table is
+0.5% at 4K and +1% at 20K), decode 91.2 vs 93.2 tok/s (-2.1%, 0 of 6 pairs faster). Not bit-identical: the final residual of a 4K and a
20K prompt differs, a 48-token greedy answer after the 20K prompt differs, and the first-token KL against plain is 0.0157 mean, 0.0716 max over
24 prompts of 1.5K-24K tokens (plain against plain: 0 and 0.00004; top-1 token the same 24 of 24; 24-token answers equal 1 of 6). That is why
no gfx1201 table is shipped for hipBLASLt 1.2.0 and why packaged engines use a table only when asked.

Shipped tables:

- `gfx1100-hipblaslt-100100.txt`, `gfx1100-hipblaslt-100200.txt`: RX 7900 XTX.
- `gfx1100-hipblaslt-100401.txt`: RX 7900 XTX, calibrated with the packaged ROCm 10.0.0
  (`rocm/dev-ubuntu-24.04:10.0.0-full`, hipBLASLt 1.4.1) on a Ryzen 7 9800X3D, over the 26 dense GEMM
  geometries of the shipped gfx1100 table. Without it, that stack reads a prompt through plain hipBLAS: on
  Swift 1.5 IQ3_XXS at 262144 ctx, fresh-prompt prefill measured 682 -> 1,038 tok/s at 1.7K tokens and
  857 -> 1,524 tok/s at 6.5K (medians of 3, decode unchanged), with `hip_prefill_hipblaslt_gemm` reporting
  `fallbacks=0`. setup uses it only when the installed hipBLASLt reports 1.4.1.
  The 43-row revision adds the two geometries the 26 dense ones do not reach, calibrated with
  `tune_hipblaslt` from the same tree and the engine's 32 MiB workspace: `f16 1280 2560 1280` (the
  small-T expert GEMM, T=8..999) and `bf16 10240 2560 10240`. A prompt on Qwen3.8-Flash-Next GSQ-RCO
  IQ2_XS at 192000 ctx otherwise falls back on the expert GEMM for nearly every launch:
  `STRATA_HIPBLASLT_VERBOSE=1` reported 642 fallbacks over 642 distinct T values (1..1873) with the
  26-row table. Adding the rows measured 711 -> 938 tok/s (+31.9 %, mean of 3 interleaved pairs:
  +36.0 / +32.4 / +27.6 %) at a 3151-token prompt, decode unchanged (58.6 -> 60.6 tok/s, inside this
  box's 2.3 % run drift), and `fallbacks=0`. The lookup takes the nearest T bucket for a matching
  geometry, so the buckets cover the whole small-T range rather than one row per T.
- `gfx1100-hipblaslt-100500.txt`: RX 7900 XTX (gfx1100, 24 GB), calibrated with a ROCm 10.2.0a20261003 nightly
  SDK (`libamdhip64.so.7.17.26392`, hipBLASLt 1.5.0, version number 100500). Same 16 dense GEMM geometries at
  T=4096 and T=8192 as the other gfx1100 tables (32 rows), calibrated with `tune_hipblaslt` run against that
  library. On this ROCm the engine refuses the 100100/100200 tables (version mismatch) and prefills on plain
  hipBLAS; with this table a 0.1.38-lineage nightly engine prefills a 131071-token prompt at 1687 tok/s vs 926
  without it (median of 3 clean runs each, greedy; decode unchanged). Per-run data and the calibration command:
  the `2026-10-04-community-rx7900xtx-hipblaslt-100500` folder of PR #745.
- `gfx1201-hipblaslt-100500.txt`: Radeon AI PRO R9700 (gfx1201, 32 GB), calibrated with ROCm 10.2.0a20260914
  (AMD's `gfx120X-all` nightly, hipBLASLt 1.5.0, library build `d3164197`). 16 dense GEMM geometries at T=4096 and
  T=8192, 32 rows. setup uses it only when the installed hipBLASLt reports 1.5.0 (it is found in `/opt/rocm`
  when that is a system ROCm 7 or newer). The version number is the only thing the engine can check, so another
  1.5.0 build could number its solutions differently. `hip_prefill_hipblaslt_gemm` (with `STRATA_HIPBLASLT_TUNING`
  set) is a smoke test: it refuses a table for another architecture or version and checks that one BF16 and one
  F16 row exist and agree with hipBLASEx, which covers 2 of the 32 rows. It does not prove that a solution id is
  valid: the engine falls back to hipBLASEx for an id the library rejects, and the test still passes. Run it with
  `STRATA_HIPBLASLT_VERBOSE=1` and look for `fallbacks=0` in its summary line, and recalibrate with
  `tune_hipblaslt` before using this table with a different 1.5.0 build.
- `gfx1151-hipblaslt-100400.txt`: Radeon 8060S (Ryzen AI Max+ 395, gfx1151, 128 GB; Fedora 44, kernel 7.2.8),
  calibrated with the ROCm 7.14.0a20260608 wheels that setup installs for gfx1151 since 0.1.40.2 (#1267): their
  hipBLASLt is 1.4.0 (version number 100400), so neither `gfx1151-hipblaslt-100401.txt` nor `-100500.txt` applied and
  setup's engine read the prompt through plain hipBLAS. The same 90 rows as the 100401 table (its geometries, the
  prompt shapes at `--prefill 16384` included), calibrated with `tune_hipblaslt --workspace-mib 32` and that table's
  cases: 3.1x to 14.1x faster than plain hipBLAS per shape (median 6.2x). With `STRATA_HIPBLASLT_TUNING` set,
  `hip_prefill_hipblaslt_gemm` passes (`launches=4 fallbacks=0`), `hip_prefill_hcd_exact_parity` passes (solutions
  1176 / 1177 at 9 chunk sizes, 0 outputs differ, `fallbacks=0`), and the HIP ctest passes 61 of 62 (`ple_parity`
  needs the Q2_0 model fixture). setup uses it only when the installed hipBLASLt reports 1.4.0. The prompt speed
  of a whole model with this table is not measured yet.

- `gfx1150-hipblaslt-100401.txt`: Radeon 890M (gfx1150, Strix Point APU, 16 CUs), calibrated with AMD's TheRock ROCm 7.14.1 for
  gfx1150 (hipBLASLt 1.4.1 `cd957402`, version number 100401) and the engine's 32 MiB workspace. The 16 dense GEMM geometries a
  Qwen3.8-Flash-Next prompt logs on that card (the same 16 as the gfx1201 100500 table) at T = 64, 128, ..., 16384: 144 rows. Two
  `tune_hipblaslt` passes back to back; each row keeps the solution with the lowest median of all six repetitions, because one
  repetition stalled now and then (a 100 ms call read up to 1.6 s) and the tool's mean-of-three picks differed in 26 rows between
  the passes. Every row beats plain hipBLAS, 1.55-22.6x per GEMM (geometric mean 4.5x), and `STRATA_HIPBLASLT_VERBOSE=1` reported
  no fallbacks over prompts of 28-7,880 tokens. Setup does not use it (it does not install for gfx1150): point
  `STRATA_HIPBLASLT_TUNING` at it. Measurements and the per-row timings:
  [bench/results/2026-10-07-community-gfx1150](../bench/results/2026-10-07-community-gfx1150/README.md).
- `gfx1103-hipblaslt-100401.txt`: Radeon 780M (gfx1103, RDNA3, 12 CUs, unified memory), calibrated with ROCm 10.1
  (`rocm/dev-ubuntu-24.04:10.1.0-full`, hipBLASLt 1.4.1, version number 100401) and the engine's 32 MiB workspace. The
  16 dense GEMM geometries a Qwen3.8-Flash-Next prompt logs on the card at T = 64, 128, ..., 16384: 144 cases, two
  `tune_hipblaslt` passes back to back, each row the solution with the lowest median of all six repetitions. One row
  is left out (`bf16 N=1 K=2560 T=64`: no candidate beat plain hipBLAS at 0.04 ms), so the table has **143 rows**,
  1.05-23.30x per GEMM, geometric mean 4.95x. Without it the engine prefills on plain hipBLAS: a 4K prompt on Q2_0
  read 90 tok/s, 187 with the table and 277 with the WMMA opt-ins (one 780M; 32K context, MTP). The merged table
  prefilled the 4K prompt at 187.2 tok/s. A context ladder to 131K (shipping configuration, fresh prompts) puts the
  full 128K read at 198-221 tok/s, its last ~14K at ~206, and decode at ~10 tok/s at that depth; the numbers are in
  the report folder. `hip_prefill_hipblaslt_gemm` reports `fallbacks=0`;
  `hip_prefill_hcd_exact_parity` skips, because the table picks no 1176 / 1177 for the HC down GEMM (the skip the
  gfx1150 table gets too). Setup installs for gfx1103 only with `STRATA_EXPERIMENTAL_GFX1103=1` and then uses the
  table when the installed hipBLASLt reports 1.4.1. Measurements:
  [bench/results/2026-10-12-community-gfx1103](../bench/results/2026-10-12-community-gfx1103/README.md).

Calibrated on my PC and kept in this checkout, not shipped: `gfx1201-hipblaslt-100401.txt`, a Radeon AI PRO
R9700 (gfx1201, 32 GB, 1002:7551) on a Ryzen 9 9950X3D with 62 GiB RAM, CachyOS (kernel 7.3.0-rc6-1-cachyos-rc,
`linux-firmware-amdgpu 20260916`, the kernel's amdgpu driver), system ROCm `rocm-gfx120x-bin 10.1.0` (hipBLASLt
1.4.1, HIP 7.16.26385, clang 24.0.0), Python 3.14.7 in setup's own venv, engine 0.1.40 compiled there for gfx1201,
the model files on a 4 TB NVMe (btrfs), IQ3_S at 131072 ctx with `--kv int8`, the GPU idle. `tune_hipblaslt` over
the 32 geometries of `gfx1201-hipblaslt-100500.txt`: 1.15x geometric mean per GEMM over hipBLAS - 1.55-1.68x at
N=512/640 K=2560, 0.99-1.13x at N=10240/12288 K=2560. End to end with `tools/hip/bench_prefill.py` over two
server starts, fresh prompts of 4,210 and 8,830 tokens, three matched trials each - the first fresh prompt after
each start is left out, it is cold (949 and 1,423 tok/s there): 1,606 -> 1,613, 1,701 -> 1,726 and 1,605 -> 1,610
tok/s (+0.5, +1.5, +0.4%), medians 1,606 -> 1,613 and a 1.008x geometric mean over those three, decode
87.1 -> 85.1 tok/s - within noise, as the R9700 paragraph above found. `hip_prefill_hipblaslt_gemm` passes with it: `tuning enabled (30 rows, gfx1201,
version 100401)`, `launches=4 fallbacks=0`.

Two rows were dropped from that table: bf16 N=10240 K=320 ldy=10240 at T=4096 and at T=8192. At T=8192 hipBLAS was
faster in five separate `tune_hipblaslt` runs - 0.609-0.619 ms against 0.648-0.677 ms for the best of the 16
hipBLASLt candidates (0.90-0.95x), with four different winning solution ids, all slower than hipBLAS. Dropping only
the T=8192 row does not send that call back to hipBLAS: `closest()` matches dtype/N/K/ldy and takes the nearest T
bucket, so T=8192 would run the T=4096 row; both rows go or neither.

Two calibration notes from my machine. The first case of a fresh `tune_hipblaslt` process times its hipBLAS
baseline while the clocks are still ramping: that T=4096 shape read 0.31-0.63 ms across five runs while the case
timed right after it held 0.606-0.619 ms every time, so time a heavy case first and read the later ones. And the
first fresh prompt after the model loads is not a measurement: it gave 949 and 1,423 tok/s in the two arms of that
A/B, while the three trials after it differed by 0.4-1.5%.

### rocBLAS table (gfx103x)

On RDNA2 the prompt GEMMs run through rocBLAS in FP16 (#835), and hipBLASLt has no gfx1030 kernels; the rocBLAS table
does for rocBLAS what the one above does for hipBLASLt (the [RDNA2 section](#rdna2-gfx1030) has the measurements). A
table is valid for one architecture and one rocBLAS build - the full version string, tweak hash included - so it is
calibrated on the card with the rocBLAS the engine loads (`lib_dirs` in the run configuration; `/opt/rocm/lib` is
rocBLAS 5.6.0.8d1ae90e on ROCm 10.0.0, while Ubuntu's own `librocblas5` package is a 5.1 build):

```sh
cmake --build build-hip --target tune_rocblas
LD_LIBRARY_PATH=/opt/rocm/lib ./build-hip/tune_rocblas --tuning-out table.txt   # 14 shapes x 8 token buckets, ~5 min
```

The tool times rocBLAS's default kernel and every solution the library lists for each shape at T = 256 ... 8192 (the
engine's dense GEMM shapes; `--case T,N,K,ldy` adds one, `--tokens` changes the buckets), checks the fastest ones
against the default kernel's output (within FP16 rounding, no write outside the N x T result) and that they also run at
smaller, odd token counts, and writes one row per shape and bucket: the winning solution index where it beats the
default by `--min-gain` (1.05), `default` otherwise - a `default` row is what keeps a neighbouring bucket's solution from
reaching token counts where rocBLAS's own choice measured best. The file's header names the architecture and version
(`STRATA_ROCBLAS_TUNING_V1 gfx1030 5.6.0.8d1ae90e`); save it as `tools/hip/<arch>-rocblas-<version>.txt` (the version
with trailing dots dropped) for setup, or point `STRATA_ROCBLAS_TUNING` at the file. `STRATA_ROCBLAS_VERBOSE=1` prints
the solution picked for each shape and a `tuned_launches=... fallbacks=...` summary at exit; a solution rocBLAS refuses
at run time (status 11 once while it sets up its workspace is normal) falls back to the default kernel for that call, and
after three refusals for good. `hip_prefill_rocblas_gemm` (with `STRATA_ROCBLAS_TUNING` set; skips otherwise) is the
smoke test: it refuses a table for another architecture or build and checks Gemm's FP16 route against hipBLASEx on the
table's first two shapes.

## Original backend validation (PR #94)

The following is historical validation of the original backend, not a fresh
test count for this replacement. Current build/test evidence and performance
limits are recorded in [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md).

Tested against upstream `c1e903310f211e6630780c3bd2038778c071c68d` (0.1.20),
with pinned llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
The host has an RX 7900 XTX (24 GiB), Ryzen 9 7950X3D, 64 GiB system RAM,
and Fedora-family Linux with ROCm 7.1.52802 / Clang 20. The original model used
for the smoke check was on mechanical RAID0; cold page faults were slow. Keep
latency-sensitive data on SSD and measure warm residency separately.

- Complete HIP Release build, including the executable and parity targets.
- All 25 tests selected by the command below passed on 2026-09-29.
- Intrinsic parity includes signed packed-byte dot products and overflow.
- CPU/GPU mapped-memory handoff covers delayed publication, changing payloads,
  and repeated graph replay in both directions.
- GR coverage includes the maximum eight-token fused batch and graph replay
  after inputs change. Native QSA checks the scalar HIP fallback.
- Q8_K quantization is byte-exact against the existing reference. HIP's
  `__fmul_rn` can become ordinary multiplication; disabling contraction in
  `quantize_act.cu` preserves the separately rounded product before the magic
  rounding bias is added. CUDA's compile flags remain unchanged.
- The mmap regression covers canonical and differing native layer sizes,
  truncated/oversized files, lookup bounds, and reopening after errors.
- CUDA 13 / GCC 15 / sm_89 executable build passed; GR, sampler, quantization,
  and mmap tests passed on the same tree. This is a regression check, not a
  claim of complete CUDA inference validation.
- CPU-only `strata-plan` build passed without either GPU backend.
- Real GSQ-RCO IQ3_XXS server smoke with mmap, the shipped expert profile,
  automatic GPU cache, and MTP passed arithmetic, executable Python addition,
  system-marker recall, and 1,170-token batched-prefill recall. All four requests
  ended normally with correct answers; one engine process served them all.
  The cache held 11,142 experts (18.08 GiB), with 874 MiB of VRAM free after
  startup. These short checks do not establish a throughput or quality benchmark.

Build all targets before running the registered focused checks:

```sh
cmake --build build-hip -j2
ctest --test-dir build-hip --output-on-failure --timeout 60 \
  -E '^(ple_parity|platform_memory_test)$'
```

`ple_parity` requires an external model fixture. `platform_memory_test` requests
256 MiB of locked memory; the test shell's 8 MiB memlock limit prevented that
test from passing. These are explicit exclusions, not skipped tests counted as
passes. Additional unpublished upstream fixture suites are not covered.

The published performance table explicitly identifies the measured development
snapshot; it must not be read as a benchmark of every subsequent rebase. Vision, long-context stress, broad answer-quality
equivalence, other AMD cards, and mixed-vendor inference are not validated here.

## Strix Halo: 8-bit hyper-connection weights (default on gfx1151)

On a Strix Halo (gfx1151) the hyper-connection down / up projections are read as Q8_0 by default (quantized at load from the pack, or the GGUF's own Q8_0),
which halves the 13.3 MB each read moves. Decode is about 9% faster (5 interleaved pairs per prompt on IQ3_S, MTP 4 drafts / min-p 0.5; json / code / tool
call / chat / agent between +7.9% and +10.7%). `STRATA_HC_Q8=0` restores the bf16 read. Other cards are unchanged (gfx12 has had it on since 0.1.4x).
Not bit-identical to the bf16 read: a greedy answer can pick another token at a near tie and then go its own way.

How far it moves the model, measured with the quant-quality yardstick (teacher forced, 5 texts of 1,000 tokens, reference = bf16 read; RTX A4000, IQ3_XXS,
2026-10-22 — the Q8_0 weights are the same bytes on every card):

| text | top-1 same | KL |
| --- | ---: | ---: |
| code | 97.9% | 0.0077 |
| documents | 93.5% | 0.0229 |
| English | 94.7% | 0.0107 |
| other languages | 98.0% | 0.0060 |
| agent transcript | 93.0% | 0.0779 |
| all 4,995 positions | 95.4% | 0.0251 |

For scale, the same yardstick with the bf16 read on both sides and only the experts computed on the CPU instead of the GPU (a rounding-only difference the
engine takes whenever an expert is not resident) gives code 0.0062, documents 0.0195, English 0.0088, other languages 0.0048, agent 0.0745, all 0.0228
(top-1 95.4%). The 8-bit read is at that level. The agent text's number rests on a few near-tie positions (the top 20 of 999 carry half of it); finer blocks
(16 or 8 values) did not lower it, and 8-bit hyper-connection weights in the first six layers alone give the full figure, in the last 24 layers a third of it.
