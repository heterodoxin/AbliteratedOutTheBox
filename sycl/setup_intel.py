"""setup.py for an Intel Arc: upstream's installer, steered onto the SYCL port from outside.

    ./setup.sh --backend sycl [setup.py's options]       (this file, run through setup.sh's virtual environment)

The SYCL port keeps out of the shared files (upstream merges stay clean), so this does not edit setup.py: it imports
it and replaces the few steps that are NVIDIA/AMD-specific, then runs setup's own main().  Everything else - the
model choice, the download, the pack, the tokenizer, the MTP draft layer, the context and KV questions - is
setup.py's, unchanged.  What is replaced:

  - the GPU check: the Intel Arc is offered through setup's AMD path (the one that compiles locally), named and
    sized from sysfs;
  - the engine step: the SYCL build (build-sycl-aot/strata, run in the strata-sycl-dev image by
    sycl/serve/strata-sycl.sh; docs/INTEL.md) instead of a CUDA/HIP build;
  - pictures (on by default on every Arc card, see pictures_on_by_default): read by the portable CPU reader (`strata-vision`, the program the
    AMD packages carry beside their engine, compiled here by setup's own step) on the host; it hands the image embeddings to the engine through
    /dev/shm, which strata-sycl.sh shares with the container.  The reader choice is strata_control/vision_reader.py's one rule (CPU reader under
    24 GB of VRAM); an Arc of 24 GB or more also keeps the CPU reader unless --vision gpu asks for the SYCL one (`strata-vision-sycl`, compiled
    on the host when the oneAPI compiler is there).  --vision no is the off switch;
  - the RAM rule: the CUDA engine keeps every expert in RAM, the SYCL port streams them from the GGUF into VRAM
    (--stream-experts), so RAM only bounds the KV streaming;
  - the config: the container's paths, the SYCL flags, backend "sycl"; the run script starts
    sycl/serve/server_intel.py (serve/server.py plus the Intel Monitor readings and the model menu).

When setup.py changes the steps this relies on, this stops with a message rather than writing a wrong config.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402
from strata_control import vram_fit  # noqa: E402

SYCL_WRAPPER = ROOT / "sycl" / "serve" / "strata-sycl.sh"
SYCL_IMAGE = "strata-sycl-dev"
SERVER = ROOT / "sycl" / "serve" / "server_intel.py"
MOUNT = Path(os.environ.get("STRATA_SYCL_ROOT") or ROOT.parent)   # what strata-sycl.sh mounts at /work

# Battlemage / Alchemist PCI device ids -> (name, VRAM GB). lspci's database lags new cards (an Arc Pro B70 reads
# "Intel Corporation Device [8086:e223]"), so the sysfs id is the reliable signal and the name table is ours.
INTEL_ARC = {"e223": ("Arc Pro B70", 32.0), "e221": ("Arc Pro B60", 24.0), "e211": ("Arc Pro B60", 24.0),
             "e20b": ("Arc B580", 12.0),
             "e20c": ("Arc B570", 10.0), "e212": ("Arc B50", 16.0),
             "56a0": ("Arc A770", 16.0), "56a1": ("Arc A750", 8.0), "56a2": ("Arc A580", 8.0),
             "56a5": ("Arc A380", 6.0), "56a6": ("Arc A310", 4.0), "5690": ("Arc A770M", 16.0)}

# Alchemist (DG2, the A-series) has no FP64 hardware and carries the driver's FP64 emulation. On i915 (the A750) a
# single sycl::malloc_host past ~3 GiB also fails, so the experts load into a CPU-computed RAM arena instead of the
# pinned mirror (--mmap-experts + STRATA_VERIFY_NO_HOST=0 - docs/INTEL.md, "Arc A750"). On xe (the A770) the mirror is
# built in chunks past that limit and the A-series uses the ordinary streamed path; see to_sycl().
ALCHEMIST_IDS = frozenset({"56a0", "56a1", "56a2", "56a5", "56a6", "5690"})


def intel_gpus():
    """Intel discrete GPUs from sysfs: vendor 0x8086 under the xe or i915 driver, named by PCI device id. VRAM comes
    from the id table, else from the size of the card's VRAM BAR."""
    found = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:                            # connectors (card0-DP-1) share the card's device
            continue
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
            devid = (dev / "device").read_text().strip().lower().replace("0x", "")
            driver = os.path.basename(os.path.realpath(dev / "driver")) if (dev / "driver").exists() else ""
        except OSError:
            continue
        if vendor != "0x8086" or driver not in ("xe", "i915"):
            continue
        name, vram = INTEL_ARC.get(devid, (None, 0.0))
        if name is None:
            if driver != "xe":                          # i915 without a known Arc id is an integrated GPU
                continue
            name = f"Intel GPU {devid} (xe)"
        if not vram:                                    # resource line 2 (BAR 2) is the VRAM aperture on xe cards
            try:
                start, end, _ = (dev / "resource").read_text().splitlines()[2].split()
                vram = (int(end, 16) - int(start, 16) + 1) / 2**30
            except (OSError, IndexError, ValueError):
                vram = 0.0
        found.append({"index": len(found), "name": f"Intel {name}", "vram_gb": vram, "arch": driver, "driver": driver,
                      "devid": devid})
    return found


def real_mem_have(mem_have, real_ram: float, fake_ram: float):
    """setup.mem_have for the Intel path: the RAM it is given is the 1024 GB stand-in of setup's RAM rule, so a model that counts VRAM + RAM
    (Gyro) would never warn; it gets the real RAM."""
    return lambda model, ram, vram=0.0: mem_have(model, real_ram if ram >= fake_ram else ram, vram)


def sycl_engine():
    """The SYCL build: (binary, why-not)."""
    if S.WIN:
        return None, "the SYCL port runs on Linux only"
    exe = next((b for b in (ROOT / "build-sycl-aot" / "strata", ROOT / "build-sycl" / "strata") if b.exists()), None)
    if exe is None:
        return None, "it is not built (docs/INTEL.md, \"How to build it\": Docker, then sycl/tools/build.sh in the image)"
    return exe, None


def sycl_path(path) -> str:
    """A host path as the engine's container sees it: the folder above the Strata checkout (or STRATA_SYCL_ROOT) is
    mounted at /work."""
    p, root = Path(path).resolve(), MOUNT.resolve()
    try:
        return "/work/" + p.relative_to(root).as_posix()
    except ValueError:
        S.fail(f"{p} is outside {root}, which the SYCL engine's container mounts",
               f"keep the models and the data folder under {root} (--models-dir / --data-dir)")


def sycl_version() -> str:
    return S.source_version()


def flag(args, name):
    return args[args.index(name) + 1] if name in args else None


def drop(args, name, value=False):
    while name in args:
        i = args.index(name)
        del args[i:i + 1 + int(value)]


# Pictures on an Intel Arc: the CPU reader encodes on the host and the embeddings reach the engine through /dev/shm.  Tested end to end
# on the Arc A750 (i915, Alchemist; 2026-10-19, docs/INTEL.md): 47 picture requests (shapes, text, two pictures in one request, a picture in the
# history, text after pictures) through the CPU reader and the same through the SYCL GPU reader, in the container and outside it, no GPU fault.
# Owner decision 2026-10-22: an Arc B-series (Battlemage, xe) card is allowed to run like any Arc, pictures included.  The first picture request on the
# Arc Pro B70 (engine 0.1.45 SYCL, IQ3_S, 2026-10-19) hung the GPU with the CPU reader: an xe engine reset (ccs + bcs) and "Fault response:
# Unsuccessful -EINVAL", the request ended 503.  That fault has not been looked at again (the B70 is gone), so it is a known risk, not a refusal:
# pictures are on, the CPU reader is kept, and the person is told once that the card is new to Strata and how to report a problem.
PICTURES_ON_BY_DEFAULT_IDS = ALCHEMIST_IDS | frozenset(INTEL_ARC)      # every Arc card setup knows


def pictures_on_by_default(driver: str, devid: str = "") -> bool:
    """Pictures are on in setup's default for every Arc card (they were opt-in on a B-series card until 2026-10-22)."""
    return True


def is_b_series(driver: str, devid: str = "") -> bool:
    """A Battlemage (B-series) card: on xe and not an Alchemist id (an unknown id on xe counts as one: it is newer than the A-series)."""
    return driver != "i915" and devid not in ALCHEMIST_IDS


B_SERIES_LINE = ("This Intel Arc B-series card is new to Strata and we have not tested it ourselves, so pictures and long answers may fail on it. "
                 "If something fails, run again with --vision no to turn pictures off, and tell us at https://github.com/Niko1221/Strata/issues "
                 "(say it is a B-series card and attach the engine log).")


READER_DIR = ROOT / "engine-sycl"    # the folder setup treats as the engine's: BUILD.json and, for pictures, the CPU reader strata-vision


def intel_vision(asked, vram_gb: float = 0.0) -> str:
    """The picture reader wanted on an Intel Arc, by strata_control/vision_reader.py's rule: "none" when switched off, "cpu" by default (under 24 GB
    of VRAM the CPU reader is the rule, and an Arc of 24 GB or more keeps it too until the SYCL reader has run on such a card), "gpu" only when
    asked for.  Takes setup.hip_vision's arguments."""
    from strata_control import vision_reader as VR
    return VR.wanted(asked, vram_gb, "intel")


def oneapi_compiler():
    """The oneAPI icx / icpx on this PC (PATH, else /opt/intel/oneapi), or None."""
    import glob
    import shutil
    cc, cxx = shutil.which("icx"), shutil.which("icpx")
    if not (cc and cxx):
        found = sorted(glob.glob("/opt/intel/oneapi/compiler/*/bin/icpx"))
        cxx = found[-1] if found else None
        cc = str(Path(cxx).with_name("icx")) if cxx else None
    return (cc, cxx) if cc and cxx and Path(cc).exists() else None


def oneapi_environment() -> dict:
    """The environment setvars.sh makes (the SYCL reader's build needs MKL and the compiler's libraries); {} when there is no setvars.sh."""
    setvars = Path("/opt/intel/oneapi/setvars.sh")
    if not setvars.is_file():
        return {}
    out = subprocess.run(["/bin/bash", "-c", f". {setvars} >/dev/null 2>&1; env -0"], capture_output=True, check=False).stdout
    return dict(item.split("=", 1) for item in out.decode("utf-8", "replace").split("\0") if "=" in item)


def build_vision_sycl_reader(stub: Path, llama) -> bool:
    """Compile tools/vision with STRATA_VISION_SYCL into `stub`/strata-vision-sycl.  False (and the reason said) when the oneAPI compiler is
    not on this PC or the build fails: the CPU reader is there, pictures still work."""
    import shutil
    compiler = oneapi_compiler()
    if compiler is None:
        S.warn("the GPU picture reader needs the oneAPI compiler (icx / icpx), which is not on this PC: pictures are read on the CPU "
               "(a few seconds each); install intel-oneapi-compiler-dpcpp-cpp and intel-oneapi-mkl-devel to build it")
        return False
    S.say("  Compiling the picture reader for your Intel GPU (a few minutes, once) ...")
    saved = dict(os.environ)
    try:
        os.environ.update(oneapi_environment())
        S.cmake_build(ROOT / "tools" / "vision", ROOT / "build-vision-sycl", "strata-vision",
                      [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_SYCL=ON", "-DSTRATA_PORTABLE=OFF",
                       f"-DCMAKE_C_COMPILER={compiler[0]}", f"-DCMAKE_CXX_COMPILER={compiler[1]}"], None, "")
        shutil.copy2(ROOT / "build-vision-sycl" / "bin" / "strata-vision-sycl", stub / "strata-vision-sycl")
    except SystemExit:
        raise
    except Exception as e:                               # noqa: BLE001 - the CPU reader is there; pictures work without the GPU one
        S.warn(f"the GPU picture reader did not compile ({e}): pictures are read on the CPU")
        return False
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return True


def build_vision_sycl(stub: Path, stamp: Path, llama, vsrc) -> None:
    meta = json.loads(stamp.read_text(encoding="utf-8"))
    if (stub / "strata-vision-sycl").exists() and meta.get("vision_gpu_src") == vsrc:
        return
    if build_vision_sycl_reader(stub, llama):
        stamp.write_text(json.dumps({**meta, "vision": "gpu", "vision_gpu": "strata-vision-sycl", "vision_gpu_src": vsrc}, indent=1))
        S.ok(f"picture reader (GPU): {stub / 'strata-vision-sycl'}, (CPU fallback): {stub / 'strata-vision'}")


SYCL_PROFILE = "expert-profile-sycl.bin"   # data/: the xe cards' expert ranking (to_sycl)
SMALL_CARD_GB = 12.0       # under this the engine's own advice is --vram-reserve-mib 300 (a 1024 MiB reserve leaves an
                           # 8 GB card no room for even one expert slot) and the English draft vocabulary
SMALL_RESERVE_MIB = 300


def small_card(vram_gb: float, driver: str) -> bool:
    return 0 < vram_gb < SMALL_CARD_GB or driver == "i915"


# #1980: on a layer split the host mirror covers the first stage only, so with STRATA_VERIFY_NO_HOST=1 (strata-sycl.sh's
# default) a later stage that cannot hold all of its layers' experts is REFUSED at the start.  Measured on 2x Arc Pro B60
# 24 GB with the trim and a 1024 MiB reserve: IQ3_XXS (42.9 GB of experts, 89% of the 48 GB) stage 1 2230-3731 experts
# short and refused; NO_HOST=0 runs it at 39-50 tok/s with 0 faults.  IQ2_XS (35.5 GB, 74%) loads 100% resident.
SPLIT_SHORT_FRACTION = 0.80


def split_likely_short(arena_gb: float, vram_gbs: list) -> bool:
    """True when the model's experts take so much of the split cards' VRAM that a later stage will not hold all of its
    layers' experts (the dense weights, the KV, the prompt buffers, the drafter and the reserve take the rest)."""
    total = sum(v for v in vram_gbs if v and v > 0)
    return arena_gb > 0 and total > 0 and arena_gb > SPLIT_SHORT_FRACTION * total


PREFILL_BMG_MIN_GB = 24.0   # a Battlemage card with this much VRAM runs the prompt in 8,192-token chunks (to_sycl)


def recommended_prefill(vram_gb: float, ctx: int, driver: str = "xe", devid: str = "") -> tuple:
    """(chunk size, one plain line saying why or "").  4096 everywhere (to_sycl's old rule), except a Battlemage card (xe
    driver, not an A-series) with 24 GB or more and a context over 32K, where the prompt borrows cache slots: 8192.
    Measured on the Arc Pro B70 (IQ3_S, 128K context, setup's flags, 2 interleaved rounds): a 16K prompt 1,222 -> 1,347-1,362
    tok/s (+10-11%), a 32K prompt 1,187 -> 1,361-1,383 (+15-16%), the same 32 greedy tokens, the same 11,795 cache slots (the
    lent slots are refilled after the prompt), decode after it unchanged (38 tok/s).  With the 32K context (own prompt
    buffers) a bigger chunk takes 1,400 cache slots from the decode (-9..-16% decode for +20% prompt), so that stays 4096."""
    bmg = driver != "i915" and devid not in ALCHEMIST_IDS and devid in INTEL_ARC
    if bmg and vram_gb >= PREFILL_BMG_MIN_GB and ctx > 32768:
        return 8192, ("prompt reading: chunks of 8,192 tokens on this Arc (the prompt borrows cache slots, so decode keeps all "
                      "of them): 10-16% faster on 16K-32K prompts than 4,096 (change --prefill in the config to pick another)")
    return 4096, ""


def _model_title(model: str, args: list) -> str:
    """The plain name for the fit line: Gyro-M / Gyro-S from the catalog name or the GGUF's file name, else the catalog name."""
    low = (model + " " + (flag(args, "--native") or "")).lower()
    for key, title in (("gyro-m", "Gyro-M"), ("gyro-s", "Gyro-S")):
        if key in low:
            return title
    return model if model and "/" not in model and chr(92) not in model else ""


def to_sycl(cfg: dict, exe: Path, ram: float, keep: dict, vram_gb: float = 0.0, driver: str = "xe",
            devid: str = "", fit_notes: list | None = None, model: str = "", ctx_asked: bool = False,
            auto_reserve: bool = False, fit: bool = True) -> dict:
    """setup's config (written for its HIP path) -> the SYCL port's: the container's paths, experts streamed from the
    GGUF into VRAM (--stream-experts: the engine reads them from the GGUF and keeps what does not fit in a pinned RAM
    mirror; the VRAM reserve is the smallest that leaves the KV and the prompt buffers room - docs/INTEL.md), KV
    streaming from 64K up when the RAM holds the KV (the B70 at 256K decodes at 40+ tok/s with it, 4-9 without).
    A reserve the user asked for (--vram-reserve-mib) is kept.

    0.2.2: a model that does not fit the card at the default context (Gyro-M on the 8 GB A750 is ~750 MiB short of one cache
    slot) gets a context down the ladder 32K -> 16K -> 8K, a smaller prompt chunk and a lower reserve until it does
    (strata_control/vram_fit.py); one plain line per change goes to `fit_notes`.  A config that fits comes back as it was.
    `ctx_asked` / a reserve without `auto_reserve`: the user chose it, it is kept.  `fit=False`: a layer split.

    An Alchemist card (the A-series, i915 or xe) has no FP64 hardware, so its config carries the driver's FP64
    emulation (a kernel on the sampled path declares double). On i915 (the A750) the experts load into a RAM arena
    and the CPU computes the ones the card does not hold (--ple-io ram, STRATA_VERIFY_NO_HOST=0 - docs/INTEL.md,
    "Arc A750"). On xe (the A770) the experts are streamed into VRAM like any other Arc; the mirror is built in
    chunks so DG2's >3 GiB single pinned-host allocation limit is not hit, and the build passes
    -ze-opt-greater-than-4GB-buffer-required so the prompt path is correct past the cache's 4 GiB (docs/INTEL.md,
    "Kernels misread memory more than 4 GiB into an allocation")."""
    args = list(cfg["args"])
    for f in ("--resident-experts", "--mmap-experts"):  # setup's low-RAM mode is the CUDA engine's
        drop(args, f)
    drop(args, "--kv-resident", True)                   # decided below on the real RAM
    if driver != "i915" and "--expert-profile" in args and (ROOT / "data" / SYCL_PROFILE).is_file():
        # 0.1.43: a card on xe runs every expert on the GPU (the ones the cache does not hold are read from the pinned mirror
        # by the same kernels), so WHICH experts the cache holds changes speed and no token: data/expert-profile-sycl.bin
        # ranks them from the routing of 45 varied prompts (chat, code, maths, six languages) and eight document summaries,
        # where the shipped profile holds 60% of a chat prompt's experts and 85% of a summary's (B70: +9..25% decode on prompts left
        # out of it, same tokens).  Another profile the user chose is kept; i915 keeps the shipped one (there the CPU
        # computes the misses, so the residency moves roundings).
        i = args.index("--expert-profile") + 1
        if args[i].replace("\\", "/").rsplit("/", 1)[-1] == "expert-profile.bin":
            args[i] = args[i][:-len("expert-profile.bin")] + SYCL_PROFILE
    for f in ("--pack", "--native", "--ple-gguf", "--expert-profile", "--mtp", "--control-vector-scaled"):
        if f in args:
            v = args[args.index(f) + 1]
            path, _, scale = v.rpartition(":") if f == "--control-vector-scaled" else (v, "", "")
            args[args.index(f) + 1] = sycl_path(path) + (f":{scale}" if scale else "")
    ctx = int(flag(args, "--max-context") or 32768)
    kv = flag(args, "--kv") or "int8"
    asked_reserve = flag(args, "--vram-reserve-mib")    # setup writes it only when the user gave it
    drop(args, "--vram-reserve-mib", True)
    reserve = asked_reserve or (str(SMALL_RESERVE_MIB) if small_card(vram_gb, driver) else
                                "1024" if ctx <= 32768 else "2048")
    new_prefill = None
    if fit and vram_gb:
        pf = flag(args, "--prefill")
        fr = vram_fit.plan(model or (flag(args, "--native") or ""), vram_gb, ctx, kv, int(reserve), pf or "auto", "intel",
                           title=_model_title(model, args), ctx_fixed=ctx_asked,
                           reserve_fixed=bool(asked_reserve) and not auto_reserve,
                           prefill_fixed=pf not in (None, "auto"))
        if fr.changed:
            if fr.ctx != ctx:
                if "--max-context" in args:
                    args[args.index("--max-context") + 1] = str(fr.ctx)
                else:
                    args += ["--max-context", str(fr.ctx)]
                ctx = fr.ctx
            reserve = str(fr.reserve_mib)
            new_prefill = fr.prefill
            if fit_notes is not None:
                fit_notes.append(fr.line)
    kv_ram = ctx * (13 * (576 if kv == "q4_0" else 1056)) / 1e9
    if kv != "k8v4" and ctx >= 65536 and ram >= kv_ram + 6:
        args += ["--kv-resident", "32768"]
    alchemist = driver == "i915" or devid in ALCHEMIST_IDS
    if driver == "i915":
        # Arc A-series on i915 (the A750): the experts load into an ordinary RAM arena and the CPU computes the ones
        # the card does not hold; the n-gram table is kept in RAM (setup's rotational-disk rule) and no mirror is built.
        # Gyro: its n-gram table is half of the 58-92 GB file and stays on the SSD (read row by row); the RAM holds the
        # experts' arena (24 GB for Gyro-S) and the CPU's work, so a table in RAM would not fit beside them on a 64 GB PC
        gyro = "gyro" in (flag(args, "--native") or "").lower()
        if "--ple-io" not in args and not gyro:
            args += ["--ple-io", "ram"]
        args += ["--vram-reserve-mib", reserve]
    else:
        args += ["--stream-experts", "--vram-reserve-mib", reserve]
    if ctx > 32768 or vram_gb >= 24:                    # long contexts: 4096-token chunks keep the prompt buffers small; a
        # 24 GB+ card with part of the experts in the RAM mirror streams those over PCIe once per chunk, so fewer, bigger
        # chunks read the prompt faster (Arc Pro B70, IQ3_S, a 4,095-token prompt: 2,048-token chunks 618 tok/s, 4,096: 1,002;
        # docs/INTEL.md); it costs ~700 cache slots.  A Battlemage card with the prompt borrowing cache slots reads 8,192.
        # A chunk size the user already put in the config is kept: this is a recommendation, not a rule.
        if flag(args, "--prefill") in (None, "auto", "2048", "4096", "8192"):
            drop(args, "--prefill", True)
            args += ["--prefill", str(recommended_prefill(vram_gb, ctx, driver, devid)[0])]
    if new_prefill:                                     # the fit rule's smaller prompt chunk (the card cannot spare 2048-token buffers)
        if "--prefill" in args:
            args[args.index("--prefill") + 1] = new_prefill
        else:
            args += ["--prefill", new_prefill]
    out = {k: v for k, v in cfg.items() if k not in ("lib_dirs", "env", "gpus")}   # "vision" stays: the CPU reader runs on the host
    out.update({"backend": "sycl", "exe": str(SYCL_WRAPPER), "args": args, "sycl_root": str(MOUNT)})
    env = {}
    if alchemist:
        if driver == "i915":
            # on i915 the CPU computes the misses and the GPU waits for them at every layer (NO_HOST=0). On xe the
            # experts are streamed and the mirror is used, so strata-sycl.sh's NO_HOST=1 default stands.
            env["STRATA_VERIFY_NO_HOST"] = "0"
        # an Alchemist has no FP64 hardware: a kernel that declares double (one is on the sampled path) is refused at its
        # first launch - "'double' is not supported in ... device", the engine dies on the first request with a
        # temperature - unless the driver emulates it (docs/INTEL.md)
        env.update({"IGC_EnableDPEmulation": "1", "OverrideDefaultFP64Settings": "1", "NEOReadDebugKeys": "1"})
    if exe != ROOT / "build-sycl-aot" / "strata":
        env["STRATA_SYCL_BIN"] = exe.relative_to(ROOT).as_posix()
    if MOUNT.resolve() != ROOT.parent.resolve():
        env["STRATA_SYCL_ROOT"] = str(MOUNT)
    if env:
        out["env"] = env
    out.update(keep)
    return out


def install(argv) -> None:
    intel = [] if S.WIN else intel_gpus()
    if not intel:
        S.fail("no Intel Arc found (an xe or i915 card in /sys/class/drm)", "on an NVIDIA or AMD card, run ./setup.sh")
    exe, why = sycl_engine()
    if exe is None:
        S.fail(f"Strata's SYCL engine cannot be used: {why}", "docs/INTEL.md: build it, then run this again")
    for name in ("gpus", "amd_gpus", "amd_problem", "hip_vision", "build_engine_hip", "hipblaslt_table", "ram_gb",
                 "write_run_script", "start", "say", "main"):
        if not callable(getattr(S, name, None)):
            S.fail(f"setup.py has no {name}() any more: sycl/setup_intel.py needs updating for this setup.py")

    real_ram = S.ram_gb()
    # what the person chose on the command line is kept by the fit rule (vram_fit); the small-card reserve added below is not theirs
    asked_ctx = any(a == "--context" or a.startswith("--context=") or a == "--max-context" for a in argv)
    asked_reserve = any(a == "--vram-reserve-mib" or a.startswith("--vram-reserve-mib=") for a in argv)
    keep = {}                                          # hand-set keys setup does not write: kept across a rerun
    for p in ROOT.glob("strata-*.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8-sig"))
            keep[p.name] = {k: c[k] for k in ("model_switcher", "sampling") if k in c}
        except (OSError, ValueError):
            pass
    stub = READER_DIR                                   # setup reads the engine's BUILD.json; the SYCL build has none.  A folder that
    stub.mkdir(exist_ok=True)                           # stays: the picture reader in it is what the run config points at
    try:
        stamped = json.loads((stub / "BUILD.json").read_text(encoding="utf-8"))   # keeps what the reader's build recorded
    except (OSError, ValueError):
        stamped = {}
    (stub / "BUILD.json").write_text(json.dumps({**stamped, "version": sycl_version(), "source": "local", "lib_dirs": []}))

    say = S.say
    fake_ram = max(real_ram, 1024.0)
    alch = intel[0]["driver"] == "i915" or intel[0].get("devid") in ALCHEMIST_IDS

    def say_intel(msg=""):
        """setup's words for its AMD path and its RAM rule, said for the Intel card."""
        msg = str(msg).replace("(AMD, experimental: docs/AMD_HIP.md)", "(Intel Arc: the SYCL port, docs/INTEL.md)")
        msg = msg.replace("Your AMD GPUs:", "Your Intel GPUs:").replace("just run ./setup.sh", "just run ./setup.sh --backend sycl")
        msg = re.sub(r"\b(xe|i915) \(AMD: docs/AMD_HIP\.md\)", r"\1 driver (Intel Arc: docs/INTEL.md)", msg)
        where = ("the experts the card does not hold are computed on the CPU" if alch
                 else "the experts are streamed into VRAM")
        msg = msg.replace(f"RAM: {fake_ram:.0f} GB", f"RAM: {real_ram:.0f} GB ({where})")
        msg = msg.replace(f"({fake_ram:.0f} GB RAM +", f"({real_ram:.0f} GB RAM +").replace(f"you have {fake_ram:.0f} GB RAM", f"you have {real_ram:.0f} GB RAM")
        if re.match(r"\s+\S+\s+needs ~\d+ GB RAM:", msg):   # --check's CUDA verdicts: replaced by the Intel one
            m = msg.split()[0]
            d = S.MODELS.get(m, {})
            shard1 = d.get("download_gb", 0) - 28.8          # all but the per-layer lookup table (read from disk)
            room = intel[0]["vram_gb"] - 3 + max(0.0, real_ram - 10)   # VRAM, plus a pinned host mirror (or the CPU)
            msg = (f"  {m:8s} ~{shard1:.0f} GB of weights: " +
                   ("fits in VRAM" if shard1 <= intel[0]["vram_gb"] - 1.5 else
                    "fits, the experts the card does not hold are computed on the CPU (slower)" if alch and shard1 <= room else
                    "fits with part of its experts mirrored in RAM (slower)" if shard1 <= room else "does not fit"))
        say(msg)
    S.say = say_intel
    S.gpus = lambda *a, **k: []
    S.amd_gpus = lambda *a, **k: intel
    S.amd_problem = lambda g: None
    S.hip_vision = intel_vision                         # pictures on by default, read on the CPU (setup's hip_reader settles it)

    def build_engine_hip(gpu, llama, vision="none"):
        """No engine to compile here (the SYCL build is separate); the picture readers are: the portable CPU reader, once, and for
        --vision gpu the SYCL GPU reader (a few minutes, once; it needs the oneAPI compiler on this PC, else the CPU reader stays)."""
        if vision != "none":
            stamp = stub / "BUILD.json"
            vsrc = S.source_hash(S.VISION_SOURCES)
            S.build_vision_cpu(stub, stamp, json.loads(stamp.read_text(encoding="utf-8")), llama, vsrc)
            if vision == "gpu":
                build_vision_sycl(stub, stamp, llama, vsrc)
        return stub
    S.build_engine_hip = build_engine_hip
    S.hipblaslt_table = lambda *a, **k: None
    S.ram_gb = lambda: fake_ram                         # the experts are in VRAM: setup's RAM rule does not apply
    # Gyro (mem_total) counts VRAM + RAM together and is the one family whose experts are NOT streamed from the file into VRAM: on
    # an Arc they sit in the RAM arena and the CPU computes what the card does not hold.  Its memory warning ("needs 27.6 GB of VRAM +
    # RAM together, you have ...", never a refusal) therefore reads the real RAM, not the 1024 GB above.
    S.mem_have = real_mem_have(S.mem_have, real_ram, fake_ram)

    bench_tips = S.bench_tips

    def bench_tips_intel(args, env, ram, *a, **k):
        """The tips as written for the CUDA engine, said with the real RAM (the 1024 GB above is only for the RAM rule);
        the --prefill auto:32768 one was measured on the CUDA engine and is left out."""
        return [t for t in bench_tips(args, env, real_ram, *a, **k) if "auto:32768" not in t]
    S.bench_tips = bench_tips_intel

    write = S.write_run_script

    def write_run_script(model, cfg_path, port, open_browser=True):   # setup.write_run_script's signature (#870)
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        # the card setup selected (--gpu / the first usable), not intel[0]: on a PC with two Arcs the config must match
        # the chosen one (docs/INTEL.md, "Two cards"). intel_gpus() lists them in sysfs card order and cfg["gpu"] is
        # that index (setup.py's amd path); a layer split writes a list, and then every listed card is used.
        gsel = cfg.get("gpu")
        split = isinstance(gsel, list)
        idx = gsel if isinstance(gsel, int) and 0 <= gsel < len(intel) else 0
        sel = intel[idx]
        fit_notes = []
        cfg = to_sycl(cfg, exe, real_ram, keep.get(Path(cfg_path).name, {}), sel["vram_gb"], sel["driver"],
                      sel.get("devid", ""), fit_notes=fit_notes, model=model, ctx_asked=asked_ctx,
                      auto_reserve=not asked_reserve, fit=not split)
        for line in fit_notes:                              # one plain line: what was chosen and why (never a refusal)
            S.say("  " + line)
        if len(intel) > 1:                                  # pin the SYCL device(s)
            # the Level Zero order followed the sysfs card order on the tested machines; a machine where it does not
            # needs an explicit ONEAPI_DEVICE_SELECTOR (kept if the user set one)
            selector = "level_zero:*" if split else f"level_zero:{idx}"
            cfg.setdefault("env", {}).setdefault("ONEAPI_DEVICE_SELECTOR", selector)
        if split:
            # #1980: without it every card keeps a full copy of the dense weights (1.4 GiB each on Flash-Next IQ2_XS) and the
            # load stops at 96-97% resident; the later stage's missing experts have no host mirror (STRATA_VERIFY_NO_HOST)
            cfg.setdefault("env", {}).setdefault("STRATA_STAGE_TRIM", "1")
            # #1980: a later stage short of VRAM is refused under NO_HOST=1; with the host path on it runs (the host
            # computes those experts).  Only when the experts will likely not fit; a value the user set is kept.
            cards = [intel[i]["vram_gb"] for i in gsel if isinstance(i, int) and 0 <= i < len(intel)]
            arena = S.MODELS.get(model, {}).get("arena_gb", 0)
            if split_likely_short(arena, cards) and "STRATA_VERIFY_NO_HOST" not in cfg["env"]:
                cfg["env"]["STRATA_VERIFY_NO_HOST"] = "0"
                S.say(f"  {model}'s {arena:.0f} GB of experts will likely not all fit the {len(cards)} cards' "
                      f"{sum(cards):.0f} GB of VRAM: the config sets STRATA_VERIFY_NO_HOST=0, so the CPU computes the "
                      "experts the later card does not hold instead of the start being refused (#1980). Set it to 1 "
                      "in the config's env if your cards hold everything; a smaller model fits fully (docs/INTEL_ARC.md)")
        need = S.MODELS.get(model, {}).get("ram_gb", 0)
        sel_alch = sel["driver"] == "i915" or sel.get("devid") in ALCHEMIST_IDS
        if sel_alch and need and real_ram < need:
            S.warn(f"{model} on this Arc computes the experts the card does not hold on the CPU (about {need} GB of "
                   f"weights; this PC has {real_ram:.0f} GB of RAM): the start will be slow or fail - a smaller model "
                   "(--model) fits better (docs/INTEL.md)")
        if flag(cfg["args"], "--prefill") == "8192":
            why = recommended_prefill(sel["vram_gb"], int(flag(cfg["args"], "--max-context") or 32768), sel["driver"],
                                      sel.get("devid", ""))[1]
            if why:
                S.say("  " + why)
        Path(cfg_path).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        script = write(model, cfg_path, port, open_browser)
        script.write_text(script.read_text().replace(str(ROOT / "serve" / "server.py"), str(SERVER)))
        return script
    S.write_run_script = write_run_script

    start = S.start

    def start_sycl(cfg_path, *a, **k):
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8-sig"))
        if cfg.get("backend") != "sycl":
            return start(cfg_path, *a, **k)
        script = ROOT / f"run-{Path(cfg_path).stem[len('strata-'):]}.sh"
        S.say(f"\nstarting {script.name} ...")
        os.execv("/bin/sh", ["/bin/sh", str(script)])
    S.start = start_sycl

    argv = list(argv)
    if small_card(intel[0]["vram_gb"], intel[0]["driver"]):   # the engine's own advice for a card this size
        if "--draft-vocab" not in argv:
            argv += ["--draft-vocab", "en"]             # ~110 MiB less VRAM than the default CJK subset
        if "--vram-reserve-mib" not in argv:
            argv += ["--vram-reserve-mib", str(SMALL_RESERVE_MIB)]
    sys.argv = [str(ROOT / "setup.py"), *argv]
    if "--backend" not in argv:
        sys.argv += ["--backend", "hip"]
    if is_b_series(intel[0]["driver"], intel[0].get("devid", "")):
        S.say("  " + B_SERIES_LINE)
    if "--low-ram" not in argv:
        sys.argv += ["--low-ram", "off"]
    sys.exit(S.main())


if __name__ == "__main__":
    try:
        install(sys.argv[1:])
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
