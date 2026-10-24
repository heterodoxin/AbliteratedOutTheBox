# Strata packaging: targets, recipes and locks

Revision: Strata 0.2.0. This directory holds the per-target build recipes and the machine-readable
manifests. **It describes what can be built where, and it is explicit about what has *not* been
built or tested.** Nothing here is a support claim.

The installers carry the engine: the desktop installers hold an engine package for their platform and graphics card, and the first
start copies it into the per-user component folder after checking every file against its manifest. A new install therefore starts a
model without a download; later engines come from the signed feed ([UPDATES.md](UPDATES.md)). The Terminal install is the other
route: one line, and the first model brings the engine for that PC from the same release.

## The target matrix

A backend name alone is not a target. Each row is an OS x CPU ABI x backend tuple, and a row is
`qualified` only after the acceptance checklist has actually run on it.

| Target tuple | App artifact | Engine package | Status |
| --- | --- | --- | --- |
| `macos-arm64-metal` | `.app` + `.dmg` (Tauri) | Metal `strata`, bundled | **built.** The app's self-test passed from inside the mounted DMG and the engine's 17 kernel-parity suites pass on an Apple M4 GPU. A large model does not fit the 16 GiB test Mac. Carries `strata-vision` (Metal), built by `packaging/build_vision_mac.sh`. Signing and notarising: `packaging/mac/sign-and-notarize.sh` (see [mac/README.md](mac/README.md)); it has been run with stand-ins for Apple's service, not yet against the real one. |
| Terminal `darwin-arm64` | self-contained `.tar.gz` | any | **built and verified.** Runs from a moved directory in a clean environment with no usable system Python. |
| `windows-x86_64-cuda` | NSIS `.exe` (Tauri) | `cuda13` and `cuda12` packages | **built and tested** on a Windows PC with an NVIDIA card: silent install with a PATH of System32 only, a real model, an agent chat. |
| `windows-x86_64-hip` | NSIS `.exe` (Tauri) | `tools/hip/build_windows.bat` | **built; never run on a real AMD card.** |
| `linux-x86_64-cuda` | AppImage (Tauri) | `setup.py`; built in a glibc 2.35 chroot | **cuda13 built and tested** (RTX 3060 and RTX A4000: app in the browser, a real model, a picture). cuda12 built. |
| `linux-x86_64-hip` | AppImage (Tauri) | `packaging/build_hip_linux.sh` | **RX 9000 (gfx1200 / gfx1201) built and tested** on a Radeon AI PRO R9700. **RX 7000 (gfx1100 / 1101 / 1102) built; not run on an RX 7000 card.** **RX 6000 (gfx1030 / 1031, 0.2.2, engine package `rx6000` only) built; not tested on a real RX 6000 card.** |
| `linux-x86_64-sycl` | | `packaging/build_sycl_linux.sh` (one JIT package for Alchemist and Battlemage) | **engine package built; tested on an Arc A750** (Intel Arc is an official platform since 0.1.40.2, docs/INTEL_ARC.md). Arc B-series (Battlemage) cards are supported and untested by us: no B-series card was available for the package (the B70 tests of 2026-10 were with the source build), and the Terminal says so once and tells the person how to report a problem. `sycl/setup_intel.py` (Docker) remains the source-build route. |
| `windows-x86_64-sycl` | none | none | **gated.** No native Windows SYCL path exists upstream; it needs a separate feasibility gate, not a recipe. |
| `macos-x86_64-*` | none | none | **unsupported.** The macOS engine is Apple-Silicon only (`metal/CMakeLists.txt` requires Apple Silicon for the Metal runtime). |

**Cross-compilation is not attempted for the desktop app.** Tauri bundles are per-OS and the
Windows installer must be produced on Windows. What *is* portable is
everything below the carrier: the Python contracts, the owner transport, the checks, and the engine
build recipes.

## The private Node runtime

The desktop bundle ships a pinned standalone Node beside the App Host, because the packaging
requirement is that an end user supplies no runtime. It is **not committed**: it is a ~117 MB
platform binary, it cannot be reviewed in a diff, and it is an input rather than source. Fetch and
verify it instead:

```sh
apps/desktop/scripts/fetch-node.sh                 # host platform
apps/desktop/scripts/fetch-node.sh --target linux-x64
apps/desktop/scripts/fetch-node.sh --check         # verify what is installed
```

The script refuses an archive whose SHA-256 does not match the digest recorded for that target. The
digests are taken from nodejs.org's own `SHASUMS256.txt` for the pinned version, so they are not
derived from the same response as the payload. **To bump the Node version: change
`STRATA_NODE_VERSION`, add the new digests, and re-run `--check`** — the script will refuse an
unknown target's archive rather than installing it unverified.

## Desktop app

```sh
cd apps/desktop
./scripts/fetch-node.sh                                                      # the sidecar (once)
npm install --prefix ../host && npm --prefix ../host run build                # the UI
cargo tauri build --bundles app                                              # -> Strata.app

# macOS DMG (headless: Tauri's default step needs a Finder session)
./scripts/make-dmg.sh

# Windows (on Windows): produces an NSIS per-user installer
cargo tauri build --bundles nsis

# Linux (on Linux): AppImage plus a deb
cargo tauri build --bundles appimage deb
```

Windows installer policy is set in `tauri.conf.json`: `installMode: currentUser`, so the installer
is **per-user and needs no elevation**. Uninstall retention (checklist P10) is a property of *where
the app keeps its data*, not of an installer flag: models, profiles and sessions live outside the
install directory (`~/.strata` and the platform state directories), so removing the application
cannot remove them. The packaging config was validated against this Tauri version's schema — an
earlier draft used `perMachine`/`oneClick`/`deleteAppDataOnUninstall`, which `tauri-utils` 2.10 does
not declare and which are rejected outright rather than ignored.

**The `dmg`/`app` targets were rebuilt and the packaged self-test re-run (8/8) after the Windows and
Linux targets were added**, so the added targets do not affect the macOS artifact.

## Strata Control and the engine in the installed app

The installer carries two things that start a model, and neither fetches anything when it runs.

**Control** (`<install>/control`, built by `apps/desktop/scripts/stage-control.py`): the same bundle as the Terminal
archive, the pinned CPython 3.12.15, the hash-locked wheels, `strata_control`, `strata_cli` and `serve`, plus a
`BUNDLE.json` with the interpreter and lock digests. The host finds it beside itself (or through `STRATA_CONTROL_DIR`),
runs it on its own interpreter (`PYTHONNOUSERSITE=1`, no `PYTHONHOME`/`PYTHONPATH`) and starts a model with
`strata profile create|use-config` + `strata runtime start`, so the engine runs under the owner runtime (loopback TCP and a
256-bit token on Windows, a private socket elsewhere; a Job Object / process group; leases) and a second client (the
Terminal) attaches to the same instance.

The app starts each engine with `runtime start --die-with-pid <host>`, so the engine ends with the host, and the host ends
with the shell (a kill-on-close Job Object around the app on Windows; the host watching its parent everywhere). The app is a
single instance: a second launch focuses the open window. One app can run several engines (profiles `app-1`, `app-2`, ...).
docs/CLOSING.md has every close path and how each is tested.

**The engine** comes with the installer (`<install>/engine-bundle/<package folder>`, the same layout as below); the first start checks it against its own manifest and copies it into the per-user component folder, where Control finds it like any engine that `strata update` installed. It is a component in the per-user component folder
(`%LOCALAPPDATA%\Strata\components`, `$XDG_DATA_HOME/strata/components`, or `STRATA_COMPONENTS_DIR`):

    components/engine/<version>-<platform>-<backend>/
        engine-component.json     schema 2: id/name "strata-engine", version, platform, backend, archs, min_driver,
                                  build_id, min_runtime (optional), entrypoint, files [{path, sha256, bytes}],
                                  signature (null for a local package, set by the release feed); full table and the checks in ENGINE-COMPONENT.md
        strata[.exe]  *.dll | *.so   the binary and the GPU runtime libraries beside it

A run config names it as `"exe": "component:strata-engine"`; at start `strata_control/engine_component.py` picks the package
for this PC (platform, then the backend its GPU needs: nvidia-smi / rocminfo / sycl-ls, or `STRATA_ENGINE_BACKEND`; CUDA
architectures, `min_driver` and `min_runtime` are checked; the newest version wins), checks every file's size and
SHA-256 (a start hashes the whole package; a mismatch is `EngineIntegrityError` and nothing starts) and puts the package
folder first on the library path.
Nothing matching is `EngineMissing`: `strata serve` ends with exit 5 and "No engine for this PC was found in ... The app never downloads an engine
while it starts or serves", and no fetch. That is the start itself. Everything before it is a warning, not a refusal: `strata launch` says in plain
words that it could not find an engine for this computer, asks whether to go on (Enter means yes), downloads the model and ends with the next step
(exit code 0, nothing started), and the app shows the same warning with an "Update engine" button in Settings. `packaging/package_engine.py` makes a local package from a build folder.

**No acquisition at run time.** `STRATA_NO_ACQUIRE=1` (set by the Tauri shell for the host, and for Control by the host)
makes `strata_control/acquisition.py` refuse pip, downloads and compiles; on the host side `server/acquisition.mjs`
refuses the llama.cpp and cloudflared fetches (a connector placed in `components/cloudflared/` is used as it is). Models the
user chooses to download are a separate, explicit action. `apps/host/tests/no-acquire.test.mjs` runs the host under
`tests/helpers/acquire-spy.mjs` and fails on any outbound request or installer process;
`apps/desktop/scripts/installed-flow.mjs` drives an installed app end to end (stub engine anywhere, real engine with `--components`).

## What every platform must produce before it can be called qualified

1. The app launches and the window serves the supplied UI from the host's origin.
2. The owner transport works: one owner per installation, a second client attaches, a stale owner
   can be recovered, and no client can stop a process it does not own (R03/R04).
3. Real inference on a supported pack, with the engine build hash and the model hashes recorded.
4. `docs/work/QUALIFICATION-KIT.md` run end to end, with raw evidence retained.

## Honest gaps in this directory

* There is no CI. Every recipe here is written to be run by hand; making them automated is separate
  work.
* Signing needs the owner's credentials, which are never in this tree. The update feed and every engine package are signed with the feed key
  (`feed_tool.py`, [UPDATES.md](UPDATES.md)); the app installers with the Tauri updater key. The Mac app and Terminal archive are signed and
  notarised on a Mac that holds the Developer ID certificate, with `packaging/mac/sign-and-notarize.sh` ([mac/README.md](mac/README.md)); for a release, `--updater-archive` also packs the notarised app as `Strata.app.tar.gz` for the in-app Mac update (the one `tauri build` makes holds the unsigned app), signed with the Tauri key on the PC and added with `feed_tool.py set-app-platform`. No
  Windows code-signing certificate is configured in this tree; Windows may warn about an installer that is not signed.
* No font or asset redistribution evidence has been gathered (D05). The app currently ships the
  brand assets from the supplied UI ZIP, which is fine for an internal candidate and not sufficient
  for redistribution.
* The terminal distribution bundles a private Python runtime and its dependencies for `windows-x86_64`,
  `linux-x86_64` and `darwin-arm64` (`package_terminal.py --target ...`, built on the target OS). The interpreter is
  the pinned python-build-standalone CPython 3.12.15 (digest-checked at build time); the packages come from
  `packaging/locks/runtime-<target>-cp312.txt`, installed with `pip install --require-hashes --no-deps
  --only-binary=:all:` at BUILD time only (`lock_wheels.py` makes the lock; review its diff when a pin changes).
  Wrappers: `bin/strata.cmd` / `bin/strata.ps1` on Windows, `bin/strata` on POSIX; START-HERE.bat, SETUP.bat,
  UPDATE.bat, setup.sh and update.sh use a `python/` folder beside them when there is one.
  `smoke_terminal.py <archive>` extracts an archive into a clean environment (a path with a space, a PATH with no
  Python) and runs `--version`, `doctor`, `profile list`, `install` and `serve --engine mock`. The macOS archive
  predates the lock: build it again on a Mac to get the hash-pinned install (the lock is already here).
* `.gitignore` coverage for build output was added late. `apps/desktop/src-tauri/target` and the
  Node sidecar were briefly committed, which left ~324 MB in the git object store. The files are
  untracked now and the ignore rules are in place, but the history is not rewritten: that is a
  deliberate choice, since rewriting published history is worse than a large local repository. A
  developer cloning this branch gets the current tree, and the sidecar arrives via
  `fetch-node.sh`.
