# Engine component manifest (schema 2)

One format for every engine package: Windows CUDA 13 and 12, Windows HIP, Linux CUDA, Linux HIP, Linux SYCL and macOS
Metal. Control (`strata_control/engine_component.py`) reads it, checks it, and only then starts the engine.
`packaging/package_engine.py` writes it. A package you make locally is unsigned and loads. A package in the release feed is signed with the
feed key (an Ed25519 key the owner keeps offline), and `strata update` / "Update engine" install it only when that signature verifies
([UPDATES.md](UPDATES.md)). The installers carry an engine of their own, so a new install needs no download to start a model.

## Folder

    <components>/engine/<version>-<platform>-<backend>[-<variant>]/
        engine-component.json
        strata | strata.exe              the entrypoint
        lib*.so* | *.dll | *.dylib       the GPU runtime libraries it loads from beside itself
        BUILD.json  LICENSE*             optional records
        rocm/**  strata-vision.exe  ...  anything else the build needs, listed in the manifest like the rest

`<components>` is `%LOCALAPPDATA%\Strata\components`, `$XDG_DATA_HOME/strata/components` or `STRATA_COMPONENTS_DIR`.
A package folder is immutable: a new build is a new folder.

## `engine-component.json`

| field | type | meaning |
|---|---|---|
| `schema_version` | 2 | schema 1 packages (no `archs`, `min_driver`, `build_id`, `signature`; `cuda_archs` instead of `archs`) still load |
| `id`, `name` | `"strata-engine"` | the component name; anything else is refused |
| `version` | string | the engine release, e.g. `0.1.45`; the newest compatible version wins |
| `platform` | string | `windows-x86_64`, `linux-x86_64`, `macos-arm64` (must match this PC) |
| `backend` | string | `cuda`, `hip`, `sycl`, `metal`, `cpu`; must be a combination `BACKEND_MATRIX` commits to, and the one this PC's GPU needs |
| `archs` | list of strings | what the build contains. CUDA: `sm_86` (machine code for that card), `compute_89` (PTX, runs on that generation or newer). HIP: `gfx1201`. SYCL and Metal: a descriptive tag (`intel-arc`, `apple-gpu`). Required for every non-`cpu` backend. Enforced for CUDA against `nvidia-smi` compute capability; recorded for the others |
| `min_driver` | `{vendor: version}` | the oldest driver the build runs on, e.g. `{"nvidia": "580.65"}`. Enforced for NVIDIA where `nvidia-smi` answers; an unreadable driver is not a refusal |
| `build_id` | string | which engine build this is: source commit and toolkit, e.g. `be885026f96a-cuda13.0`. Required |
| `variant` | string, optional | tag when one backend has several builds on a platform (`cuda13`, `cuda12`); goes in the folder name. With equal versions the build that needs the newer driver is preferred (CUDA 13 over CUDA 12) when both fit the card |
| `min_runtime` | string, optional | the oldest Control that may start it |
| `entrypoint` | string | relative path of the binary; must be in `files` |
| `files` | list | `{path, sha256, bytes}` for **every** file of the package except the manifest; paths are plain relative POSIX paths that stay inside the folder; no duplicates |
| `notes` | string, optional | free text |
| `signature` | `null` or `{alg, key_id, value}` | see below |

Example (shortened):

```json
{"schema_version": 2, "id": "strata-engine", "name": "strata-engine", "version": "0.1.45",
 "platform": "linux-x86_64", "backend": "cuda", "archs": ["sm_86"], "min_driver": {"nvidia": "580.65"},
 "build_id": "be885026f96a-cuda13.0", "entrypoint": "strata",
 "files": [{"path": "strata", "sha256": "<64 hex>", "bytes": 191000000}], "signature": null}
```

## What Control checks before a start

In this order, each refusal names the cause; none of them starts anything and none of them downloads anything:

1. the manifest parses strictly (types, schema, paths, archs, driver, signature shape);
2. the platform and backend are this PC's;
3. `min_runtime`, CUDA archs vs. the card, `min_driver` vs. the driver;
4. the **signature policy** (below);
5. **every file** exists, is a regular file inside the folder (no symlinks), has the listed size, **and matches its SHA-256**
   (a start hashes the whole package; the listing commands `strata components engine` hash only the entrypoint unless given `--full`);
6. on POSIX the entrypoint is executable.

If a package fails (5) or (4) and no other package fits this PC, the start ends with `EngineIntegrityError`
(exit code 5): "The engine package failed its integrity check, so it was not started: 0.1.45 linux-x86_64-cuda:
libcudart.so.13 does not match its recorded SHA-256. Repair or reinstall the engine component with Update engine in the app's
Settings or `strata update` in the Terminal". A package
for another GPU or an older driver is `EngineMissing` with its reason. A tampered newer package never falls back silently
to nothing: a good older package is used and the tampered one is listed under `problems` in `strata components engine`.

## Signature

`signature` is `null` in a package made locally. The release feed's packages carry one; the updater registers the verifier
(`strata_control/signing.py`) and requires it for every package it installs:

    engine_component.register_signature_verifier(fn)       # fn(payload: bytes, signature: dict, component) -> None, raises if bad
    engine_component.signing_payload(manifest_dict)         # canonical bytes the signature covers

The payload is the manifest without its `signature` member, as canonical JSON (sorted keys, `,` and `:` separators,
UTF-8). It contains every file hash, so one signature covers the whole package. Policy:

* unsigned package: loads, reported `unsigned`;
* signed, no verifier registered: loads, reported `unverified` (never "verified");
* signed, verifier registered: must verify, otherwise refused;
* `STRATA_REQUIRE_SIGNED_COMPONENTS=1`: unsigned and unverifiable packages are refused (the updater turns it on while it checks a downloaded package).

## Building a package

    python packaging/package_engine.py --src <build folder> --backend cuda --version 0.1.45 \
        --out <components>/engine --arch sm_86 --arch compute_89 --min-driver nvidia=580.65 \
        [--build-id <id>] [--variant cuda13] [--platform linux-x86_64] [--include 'rocm/**' --include strata-vision.exe | --tree]
    python packaging/package_engine.py --verify <package folder>       # full hash check

Without `--include`/`--tree` it takes the entrypoint, the libraries beside it, `BUILD.json` and `LICENSE*`. The build id
defaults to the checkout's commit (`-dirty` when the tree has changes). It prints the size, file count and the SHA-256 of
the manifest file (which holds every file's hash), used as the package's digest in notes.

## What each target's package holds

| Package | Built on | Contents | Notes |
|---|---|---|---|
| `windows-x86_64` cuda, variant `cuda13` | the owner PC, the same cmake line as the release zips (CUDA 13.0, `75;86;89;120` real + `120` PTX, `STRATA_PORTABLE=ON`) | `strata.exe`, `strata_kq.dll`, `strata-vision.exe` (built for this variant's archs by `packaging/build_vision_windows.bat cuda13 <stage>`, which writes `vision_archs` into BUILD.json), `strata-dsv41.exe` (DeepSeek-V4.1, `packaging/build_dsv41_windows.bat`, sm_86 + sm_120; `--require-dsv41`), `BUILD.json` | `min_driver` nvidia 580.65 |
| `windows-x86_64` cuda, variant `cuda12` | the owner PC (CUDA 12.9, `60;61;70;75;80;86;89` real + `89` PTX, `STRATA_EXPERIMENTAL_SM60=ON`) | the same four files, with its own `strata-vision.exe` (`packaging/build_vision_windows.bat cuda12 <stage>`: sm_60 .. sm_89; the CUDA 13 reader has no Pascal code, #3669) | experimental, `min_driver` nvidia 575.51; the CUDA 13 build wins when both fit |
| `windows-x86_64` hip | the owner PC, `tools/hip/build_windows.bat` (ROCm 10.2 nightly wheels) then `tools/hip/package_windows.py` | `strata.exe`, `strata-device.exe`, the ROCm DLLs and the rocBLAS/hipBLASLt kernels of the zip (`--tree`) | no Windows AMD test machine yet; with `STRATA_NO_VISION` unset it also builds the two picture readers (`strata-vision.exe` on the CPU, `strata-vision-hip.exe` on the GPU) |
| `linux-x86_64` cuda, variant `cuda13` | a Linux box with the CUDA 13 toolkit, `VARIANT=cuda13 packaging/build_cuda_linux.sh` | `strata`, `libstrata_kq.so`, `strata-vision` (built by the same script for the same archs, `BUILD_VISION=0` for a text-only package), `libcudart/libcublas/libcublasLt` (13.x) | the GPU driver's `libcuda` stays the PC's |
| `linux-x86_64` cuda, variant `cuda12` | a Linux box with CUDA 12.9, `VARIANT=cuda12 packaging/build_cuda_linux.sh` (`60;61;70;75;80;86;89` real + `89` PTX, `STRATA_EXPERIMENTAL_SM60=ON`) | the same files with the 12.x libraries (the reader has sm_60 / 61 / 70 too) | experimental, `min_driver` nvidia 575.51; the Pascal and Volta package (the 0.2.1 one lacked sm_60 / 61 / 70, #3503); `package_engine.py` refuses a cuda12 package without them |
| `linux-x86_64` hip | a box with ROCm (the Python wheels are enough) | `strata`, the HIP runtime, hipBLAS/hipBLASLt/rocBLAS, comgr + libLLVM, `rocblas/library` and `hipblaslt/library` for the listed gfx archs; `packaging/build_hip_linux.sh` builds it with both picture readers |
| `linux-x86_64` hip, variant `rx6000` (RX 6000, gfx1030 / gfx1031) | `VARIANT=rx6000 packaging/build_hip_linux.sh` in the jammy chroot, ROCm 7.13.0a20260515 `gfx103X-all` wheels | the same set as the other HIP packages, with `rocblas/library` for gfx1030 / gfx1031; no `hipblaslt/library` kernels exist for them and no `tuning/` table | built, not tested on a real RX 6000 card (the manifest notes say so). `windows-x86_64` hip, variant `rdna2`: the same archs from `tools/hip/build_windows.bat` with `STRATA_HIP_ARCHS=gfx1030;gfx1031`, packaged with `--variant rdna2`. A PC with an RX 7000 / 9000 card beside an RX 6000 one never takes it (`engine_component.hip_preference`) |
| `linux-x86_64` hip, gfx1103 (Radeon 780M / Phoenix iGPU) | `packaging/build_hip_gfx1103.sh` on the mini PC (TheRock gfx110X-all SDK) | the same set as the gfx1200/1201 package, with `rocblas/library` and `hipblaslt/library` for gfx1103 (600 MiB) | experimental, unvalidated: the iGPU resets now and then (MES REMOVE_QUEUE); never kill the engine mid-request; run configs set `STRATA_NO_ARENA_THP=1`. `engine_component.resolve` picks a HIP package only when its `archs` list a gfx target the Linux kfd topology reports |
| `linux-x86_64` sycl, `archs` `intel-arc` | a Linux box with oneAPI DPC++ 2026.1 and `patchelf`, `packaging/build_sycl_linux.sh` (JIT from SPIR-V; ocloc cannot build the Alchemist FP64 kernels ahead of time, so one JIT package serves A-series and B-series) | `strata`, libsycl, the Unified Runtime adapters, the oneMKL SYCL libraries, libimf / libsvml / libirng / libintlc (about 650 MiB); every ELF file has RUNPATH `$ORIGIN` (Intel's libraries ship without it), so `env -i ./strata --version` runs | the Level Zero loader and compute runtime are the PC's driver; the user must be in the `render` group (`strata doctor` warns). No Windows package: `BACKEND_MATRIX` allows sycl on `linux-x86_64` only |
| `macos-arm64` metal | an Apple Silicon Mac | `strata`, `strata-vision` (`packaging/build_vision_mac.sh`; `package_engine.py` refuses a macOS package without it, #3697), `metal/kernels/*.metal`, `metal/engine-msl/*.metal`, `metal/ggml-common.h`, `metal/data/*.bin` | the Metal engine compiles its kernel sources when it starts; since S1 it finds them beside the executable (`metal/src/resource_path.h`: environment variable, then `<exe dir>/metal/`, then the compiled-in source path) |

`packaging/collect_libs.py` gathers the Linux libraries (an `ldd` closure with the dlopen'd extras named, kernel data filtered by
GPU architecture) into the folder `package_engine.py` packages; libraries the distribution provides are listed in
`SYSTEM-LIBS.txt` inside the package so a missing one can be named on a target PC.

## hipBLASLt tuning tables in an AMD package

An AMD package records the hipBLASLt it bundles (`hipblaslt_version` in BUILD.json, 1.5.0 is `100500`) and carries the tables of `tools/hip`
for exactly that version and its `archs` as `tuning/<arch>-hipblaslt-<version>.txt`. They are ordinary files of the manifest (hashed and signed
with the package). `packaging/stage_tuning.py` does it for the Linux build script, `tools/hip/package_windows.py` for the Windows zip, and
`package_engine.py` refuses an AMD stage without `hipblaslt_version` or with a table for another version or architecture. The server's start
(`serve/server.py with_component_engine`, rule in `strata_control/hipblaslt_tuning.py`) picks the table of the card and sets
`STRATA_HIPBLASLT_TUNING` for the engine; a version mismatch, an unreadable card or a missing version are one log line and plain hipBLAS.
`STRATA_HIPBLASLT_TABLE=on|off` (or `"hipblaslt_table"` in the run config) overrides; unset it follows `AUTO_USES_TABLE`, which is off until the
owner decides, because a table changes the rounding of the prompt's GEMMs (docs/AMD_HIP.md, Tuning table). Tests: `tests/terminal/test_hipblaslt_tuning.py`.

## The picture reader must match the package

`package_engine.py` (the command line, not the library call) checks the reader before it packages:

* a CUDA package with `strata-vision` needs `vision_archs` in BUILD.json, and every `sm_NN` of the package must be in it. A reader reused from an
  older build was the cause of #3669 (the cuda12 package on a Pascal card: the engine ran, the reader stopped with "device kernel image is
  invalid"). `--allow-stale-vision` packages it anyway and prints which archs are missing;
* a macOS package without `strata-vision` is refused (`--no-picture-reader` ships a text-only Mac package on purpose);
* a CUDA package with no reader is allowed: it answers text only, the app says "the engine installed here does not include the picture reader
  yet" and keeps Read pictures off.

`python packaging/package_engine.py --stamp-vision <stage> --vision-archs "60;61;70;75;80;86;89"` writes the record for a reader built by hand.
At run time a reader that still cannot start tells why (stderr tail, exit status, and for a kernel-image error the card hint).

## Picture readers in an AMD package

A HIP package carries two programs for pictures next to `strata` (`vision` in its BUILD.json): `strata-vision` is the CPU reader, which
needs nothing of the GPU runtime and starts on every PC, and `strata-vision-hip` is the same reader on the card (llama.cpp's HIP
backend, built for the package's `archs`; BUILD.json `vision_archs`). Pictures are on by default. The reader runs on the CPU for a card under 24 GB and on the card from 24 GB, on every make of card (the rule is
`strata_control/vision_reader.py`; `--vision gpu` or `--vision cpu` picks the other one for a model). When the run config names the GPU reader first
and the CPU reader as `vision.cpu_exe`, and the GPU reader does not start (the card is not in its build, the HIP runtime does not load, the card is
busy) the server starts the CPU reader by itself and prints one line. Both programs have RUNPATH `$ORIGIN` on Linux; on Windows the
ROCm DLLs they import sit beside the programs or in `rocm/bin` (the engine's PATH entry, `lib_dirs`). `strata_control/vision_reader.py`
holds the choice; `tools/test_vision_reader.py` the tests.
