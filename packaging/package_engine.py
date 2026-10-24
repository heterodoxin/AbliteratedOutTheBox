#!/usr/bin/env python3
"""Make a local, unpublished engine package for the component folder (strata_control/engine_component.py).

    python packaging/package_engine.py --src <folder with strata[.exe] and its libraries> --backend cuda \
        --version 0.1.45 --out <components>/engine --arch sm_86 --arch compute_89 \
        --min-driver nvidia=580.65 [--build-id <id>] [--platform linux-x86_64] [--tree] [--include GLOB ...]

    python packaging/package_engine.py --verify <components>/engine/<package folder>     # full hash check, exit 0 / 1

The result is one folder `<out>/<version>-<platform>-<backend>/` holding the engine binary, the libraries it loads,
BUILD.json when the source has one, and `engine-component.json` (schema 2: every file's SHA-256 and size, the arch
list, the oldest driver, the engine build id, and an empty `signature` for S6). It is the thing Control resolves,
checks and starts; it is not signed, not published, and nothing here uploads it. Signing is stream S6
(`engine_component.register_signature_verifier`); `--sign-with` is deliberately absent until then.

Files taken from --src, by default: the entrypoint, every `*.dll` / `*.so` / `*.so.*` / `*.dylib` / `*.metallib`
beside it, `BUILD.json`, `LICENSE*`, the picture reader `strata-vision[.exe]` when the build has one (without it the
engine answers text only: setup.py Control mode keeps the person's picture choice only when it is there) and the DeepSeek-V4.1
engine `strata-dsv41[.exe]` (setup.py dsv41_exe() looks for it beside the engine; `--require-dsv41` refuses a package without it). `--include GLOB` (repeatable, relative to --src, `**` allowed) adds more, for
example `strata-vision.exe`, `strata-device.exe` or `rocm/**`. `--tree` takes every regular file under --src (the HIP
package ships a `rocm/` tree of kernels). Symlinks are never followed or copied.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import binary_deps  # noqa: E402
from strata_control import engine_component as ec  # noqa: E402
from strata_control.components import BACKEND_MATRIX, host_platform  # noqa: E402


DSV41_EXE = "strata-dsv41"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wanted(path: Path) -> bool:
    name = path.name.lower()
    return (name.endswith((".dll", ".so", ".dylib", ".metallib")) or ".so." in name or name == "build.json"
            or name.startswith("license") or name in ("strata-vision", "strata-vision.exe", "strata-vision-hip", "strata-vision-hip.exe",
                                                       DSV41_EXE, DSV41_EXE + ".exe"))   # the picture readers (CPU, AMD GPU) and the DeepSeek-V4.1 engine beside the engine


READER_NAMES = ("strata-vision", "strata-vision.exe")


def _arch_numbers(value) -> set[int]:
    """The CUDA architectures in a BUILD.json `archs` / `vision_archs` value ("60;61;70", [60, "sm_61", "70-real"], ...), as numbers."""
    if isinstance(value, str):
        value = re.split(r"[;,\s]+", value)
    found: set[int] = set()
    for item in value or []:
        m = re.fullmatch(r"(?:sm_|compute_)?(\d+)(?:a|f)?(?:-real|-virtual)?", str(item).strip().lower())
        if m:
            found.add(int(m.group(1)))
    return found


def check_picture_reader(src: Path, backend: str, target: str, archs: tuple[str, ...], *, variant: str = "",
                         allow_stale: bool = False, allow_none: bool = False) -> list[str]:
    """The picture reader of a package must run on every card and Mac the package does (tools tests: I1, I2, #3669, #3697).

    * CUDA package with a reader: BUILD.json (beside the engine) must list `vision_archs`, and every `sm_NN` the package carries
      code for must be there. A reader reused from an older build (the Windows zips' strata-vision.exe, unchanged since 0.1.44, had no
      sm_60) stops with "device kernel image is invalid" on a Pascal card while the engine runs. Refused, unless allow_stale.
    * macOS package without a reader: refused (a Mac with this package cannot read a picture), unless allow_none.
    * CUDA package without a reader: allowed (text-only), reported in the returned notes so the build log says it.
    Returns notes to print."""
    reader = next((n for n in READER_NAMES if (src / n).is_file()), None)
    notes: list[str] = []
    if target.startswith("macos"):
        if reader is None and not allow_none:
            raise SystemExit("refusing to package the macOS engine without its picture reader: a Mac with this package could not read "
                             "a picture (#3697). Build it with packaging/build_vision_mac.sh and copy strata-vision beside strata "
                             "(or give --no-picture-reader to ship a text-only package on purpose).")
        if reader is None:
            notes.append("no picture reader in this macOS package: pictures are off for it")
        return notes
    if backend != "cuda":
        return notes
    if reader is None:
        notes.append("no picture reader in this CUDA package: the engine answers text only (the app says pictures need the reader)")
        return notes
    needed = {n for n in _arch_numbers([a for a in archs if str(a).lower().startswith("sm_")])}
    have: set[int] = set()
    build_json = src / "BUILD.json"
    try:
        have = _arch_numbers(json.loads(build_json.read_text(encoding="utf-8-sig")).get("vision_archs"))
    except (OSError, ValueError, AttributeError):
        pass
    missing = sorted(needed - have)
    if missing and not allow_stale:
        what = ("BUILD.json has no vision_archs: the reader's build is not recorded" if not have
                else "the reader has no code for " + ", ".join(f"sm_{n}" for n in missing))
        raise SystemExit(f"refusing to package the {variant or backend} engine with this picture reader: {what}. A reader reused from an "
                         "older build stops with 'device kernel image is invalid' on a card it was not built for (#3669). Build it for the "
                         "package's architectures (packaging/build_vision_windows.bat, packaging/build_cuda_linux.sh) and record them as "
                         "vision_archs in BUILD.json (or give --allow-stale-vision).")
    if missing:
        notes.append("picture reader NOT built for " + ", ".join(f"sm_{n}" for n in missing) + " (allowed by --allow-stale-vision)")
    return notes


def stamp_vision_archs(stage: Path, archs: str) -> None:
    """Record the architectures the picture reader in `stage` was built for as BUILD.json `vision_archs` (the build recipes for Windows
    call this after building strata-vision.exe; check_picture_reader compares it with the package's archs)."""
    path = stage / "BUILD.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    data["vision_archs"] = ";".join(str(n) for n in sorted(_arch_numbers(archs)))
    path.write_text(json.dumps(data, indent=1) + chr(10), encoding="utf-8")


def git_build_id(repo: Path = ROOT) -> str:
    """The source commit of the engine build, `-dirty` when the tree had changes. `unknown` outside a checkout."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (head + ("-dirty" if dirty else "")) if head else "unknown"


def select(src: Path, entrypoint: str, *, tree: bool, include: list[str]) -> list[str]:
    """Relative POSIX paths to package, entrypoint first, the rest sorted."""
    chosen: set[str] = set()
    for path in sorted(src.rglob("*") if (tree or include) else src.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        if rel == ec.MANIFEST:
            continue
        top_level = "/" not in rel
        if tree or (top_level and wanted(path)) or any(fnmatch.fnmatch(rel, pattern) for pattern in include):
            chosen.add(rel)
    for path in sorted((src / "tuning").glob("*.txt")) if (src / "tuning").is_dir() else []:
        if path.is_file() and not path.is_symlink():      # the hipBLASLt tuning tables of an AMD package (strata_control/hipblaslt_tuning.py)
            chosen.add(path.relative_to(src).as_posix())
    chosen.discard(entrypoint)
    return [entrypoint] + sorted(chosen)


def build(src: Path, backend: str, version: str, target: str, out: Path, *, entry: str | None = None,
          archs: tuple[str, ...] = (), min_driver: dict | None = None, build_id: str | None = None,
          min_runtime: str | None = None, notes: str = "", tree: bool = False, include: list[str] | None = None,
          variant: str = "", search: list[Path] | None = None, allow_libs: list[str] | None = None,
          check_reader: bool = False, allow_stale_vision: bool = False, allow_no_reader: bool = False,
          require_dsv41: bool = False) -> Path:
    if backend not in BACKEND_MATRIX:
        raise SystemExit(f"unknown backend {backend!r}")
    if backend != "cpu" and not archs:
        raise SystemExit(f"a {backend} package needs at least one --arch (sm_86, gfx1201, ...)")
    if backend == "cuda" and variant == "cuda12" and not any(a in ("sm_60", "sm_61", "sm_70") for a in archs):
        # #3503: the cuda12 package is what the feed and docs/DOCKER.md offer to GTX 10 / P100 / V100; one built for
        # sm_75 and up starts a restart loop on those cards ("this PC's GPU is sm_60")
        raise SystemExit("refusing to package the cuda12 engine without sm_60, sm_61 or sm_70: it is the Pascal and Volta build "
                         "(CUDA 12.9, -DSTRATA_EXPERIMENTAL_SM60=ON, --arch sm_60 --arch sm_61 --arch sm_70 ...; see packaging/build_cuda_linux.sh)")
    if backend == "hip":
        from strata_control import hipblaslt_tuning
        problems = hipblaslt_tuning.check_stage(src, archs)
        if problems:
            raise SystemExit("refusing to package the AMD engine:\n  " + "\n  ".join(problems))
    entrypoint = entry or ("strata.exe" if target.startswith("windows") else "strata")
    if not (src / entrypoint).is_file():
        raise SystemExit(f"{src / entrypoint} does not exist")
    if check_reader:                       # the release recipes and the CLI ask for it; the library call keeps its old behaviour
        for note in check_picture_reader(src, backend, target, archs, variant=variant, allow_stale=allow_stale_vision,
                                         allow_none=allow_no_reader):
            print("note: " + note)
    if require_dsv41 and not any((src / n).is_file() for n in (DSV41_EXE, DSV41_EXE + ".exe")):
        raise SystemExit(f"refusing to package: {src} has no {DSV41_EXE}[.exe] (--require-dsv41); build it with -DSTRATA_BUILD_DSV41=ON "
                         "(--target strata-dsv41) and copy it beside the engine")
    folder = out / (f"{version}-{target}-{backend}" + (f"-{variant}" if variant else ""))
    if folder.exists():
        raise SystemExit(f"{folder} already exists; remove it first (packages are immutable)")
    staging = out / f".{folder.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        files = []
        for rel in select(src, entrypoint, tree=tree, include=include or []):
            destination = staging / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, destination)
            files.append({"path": rel, "sha256": sha256(destination), "bytes": destination.stat().st_size})
        files += _bundle_missing_libraries(staging, target, [f["path"] for f in files], search or [], allow_libs or [])
        manifest = {"schema_version": ec.SCHEMA_VERSION, "id": ec.COMPONENT_ID, "name": ec.COMPONENT_ID,
                    "version": version, "platform": target, "backend": backend, "archs": list(archs),
                    "min_driver": dict(min_driver or {}), "build_id": build_id or git_build_id(),
                    "entrypoint": entrypoint, "files": files, "signature": None}
        if variant:
            manifest["variant"] = variant
        if min_runtime:
            manifest["min_runtime"] = min_runtime
        if notes:
            manifest["notes"] = notes
        (staging / ec.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        if os.name != "nt":
            for exe_name in (entrypoint, DSV41_EXE, "strata-vision", "strata-vision-hip"):      # the programs setup starts: executable in the tar
                if (staging / exe_name).is_file():
                    (staging / exe_name).chmod(0o755)
        ec.verify(ec.read_manifest(staging), full=True)
        staging.rename(folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return folder


def _default_search(target: str) -> list[Path]:
    """Where a Windows build finds the redistributable DLLs its engine imports: the CUDA/HIP toolkits and PATH."""
    if not target.startswith("windows"):
        return []
    found: list[Path] = []
    for var in ("CUDA_PATH", "HIP_PATH"):
        root = os.environ.get(var)
        if root:
            found += [Path(root) / "bin" / "x64", Path(root) / "bin"]
    found += [Path(x) for x in os.environ.get("PATH", "").split(os.pathsep) if x]
    return [d for d in found if d.is_dir()]


def _bundle_missing_libraries(staging: Path, target: str, rels: list[str], search: list[Path], allow: list[str]) -> list[dict]:
    """Every library the package's binaries import must be in the package, or belong to the OS or the GPU driver
    (binary_deps.py). Windows builds copy a missing one from the search folders (the CUDA toolkit's bin/x64, PATH): the packager
    runs on a PC that has the toolkit, the person's PC does not. Anything still missing refuses the package."""
    added: list[dict] = []
    roots = search or _default_search(target)
    for _ in range(8):                                   # a copied DLL can import more DLLs
        missing = [(f, n) for f, n in binary_deps.uncovered(staging) if n.lower() not in {a.lower() for a in allow}]
        if not missing:
            return added
        progressed = False
        for _file, name in sorted(set(missing), key=lambda x: x[1].lower()):
            if not target.startswith("windows"):
                continue
            hit = next((d / name for d in roots if (d / name).is_file()), None)
            if hit is None:
                hit = next((c for d in roots for c in d.glob("*") if c.name.lower() == name.lower() and c.is_file()), None)
            if hit is None or (staging / hit.name).exists():
                continue
            shutil.copy2(hit, staging / hit.name)
            added.append({"path": hit.name, "sha256": sha256(staging / hit.name), "bytes": (staging / hit.name).stat().st_size})
            progressed = True
            print(f"bundled {hit.name} (imported by the engine; from {hit.parent})")
        if not progressed:
            break
    left = [(f, n) for f, n in binary_deps.uncovered(staging) if n.lower() not in {a.lower() for a in allow}]
    if left:
        lines = "\n".join(f"  {f} needs {n}" for f, n in sorted(set(left)))
        raise SystemExit("refusing to package: these libraries are neither bundled nor part of the OS or the GPU driver, so a clean PC could not "
                         f"start the engine (0xC0000135 / 'library not found'):\n{lines}\nGive --search <folder holding them> or fix the build.")
    return added


def package_sha256(folder: Path) -> str:
    """One digest for the whole package: SHA-256 of the manifest file (it holds every file's hash)."""
    return sha256(folder / ec.MANIFEST)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", type=Path, metavar="FOLDER", help="hash-check an existing package and exit")
    ap.add_argument("--src", type=Path)
    ap.add_argument("--backend", choices=sorted(BACKEND_MATRIX))
    ap.add_argument("--version")
    ap.add_argument("--platform", default=None, help="default: this machine")
    ap.add_argument("--out", type=Path, help="the component folder's engine/ directory")
    ap.add_argument("--entrypoint", default=None)
    ap.add_argument("--arch", "--cuda-arch", dest="arch", action="append", default=[],
                    help="sm_86 (code for that card), compute_89 (PTX), gfx1201, ...; repeat for several")
    ap.add_argument("--min-driver", action="append", default=[], metavar="VENDOR=VERSION",
                    help="the oldest driver, e.g. nvidia=580.65; repeat for several vendors")
    ap.add_argument("--variant", default="", help="tag when one backend has several builds (cuda13, cuda12); goes in the folder name")
    ap.add_argument("--build-id", default=None, help="default: the source commit of this checkout")
    ap.add_argument("--min-runtime", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--tree", action="store_true", help="every regular file under --src, recursively")
    ap.add_argument("--search", action="append", default=[], type=Path, metavar="DIR",
                    help="folder to copy a missing imported DLL from (default on Windows: CUDA_PATH/bin/x64, HIP_PATH/bin, PATH)")
    ap.add_argument("--allow-lib", action="append", default=[], metavar="NAME",
                    help="an imported library that really is provided by the target system (rare; named so the check passes)")
    ap.add_argument("--require-dsv41", action="store_true",
                    help="refuse the package when --src has no strata-dsv41[.exe] (the DeepSeek-V4.1 engine that setup.py looks for beside the engine)")
    ap.add_argument("--include", action="append", default=[], metavar="GLOB", help="more files, relative to --src")
    ap.add_argument("--stamp-vision", type=Path, metavar="STAGE", help="write --vision-archs into STAGE/BUILD.json and exit")
    ap.add_argument("--vision-archs", default="", help="with --stamp-vision: the architectures the reader was built for, e.g. '60;61;70;75;80;86;89'")
    ap.add_argument("--allow-stale-vision", action="store_true",
                    help="package a CUDA picture reader whose BUILD.json vision_archs do not cover the package's archs (it will fail on those cards)")
    ap.add_argument("--no-picture-reader", action="store_true", help="ship a macOS package without strata-vision on purpose (pictures stay off)")
    a = ap.parse_args(argv)
    if a.verify:
        try:
            component = ec.read_manifest(a.verify)
            ec.verify(component, full=True)
        except ec.ComponentProblem as exc:
            print(f"{a.verify}: REFUSED: {exc}")
            return 1
        if a.require_dsv41 and not any(f[0] in (DSV41_EXE, DSV41_EXE + ".exe") for f in component.files):
            print(f"{a.verify}: REFUSED: no {DSV41_EXE}[.exe] in the manifest (--require-dsv41)")
            return 1
        bad = binary_deps.uncovered(a.verify)
        if bad:
            for rel, name in bad:
                print(f"{a.verify}: REFUSED: {rel} needs {name}, which is neither bundled nor an OS / GPU driver library")
            return 1
        print(f"{a.verify}: ok ({len(component.files)} files, manifest sha256 {package_sha256(a.verify)})")
        return 0
    if a.stamp_vision:
        if not _arch_numbers(a.vision_archs):
            ap.error("--stamp-vision needs --vision-archs, for example '60;61;70;75;80;86;89'")
        stamp_vision_archs(a.stamp_vision, a.vision_archs)
        print(f"{a.stamp_vision / 'BUILD.json'}: vision_archs {a.vision_archs}")
        return 0
    for needed in ("src", "backend", "version", "out"):
        if getattr(a, needed) is None:
            ap.error(f"--{needed} is required")
    drivers = {}
    for item in a.min_driver:
        vendor, _, version = item.partition("=")
        if not vendor or not version:
            ap.error(f"--min-driver {item!r} is not VENDOR=VERSION")
        drivers[vendor] = version
    a.out.mkdir(parents=True, exist_ok=True)
    folder = build(a.src, a.backend, a.version, a.platform or host_platform(), a.out, entry=a.entrypoint,
                   archs=tuple(a.arch), min_driver=drivers, build_id=a.build_id, min_runtime=a.min_runtime,
                   notes=a.notes, tree=a.tree, include=a.include, variant=a.variant,
                   search=a.search, allow_libs=a.allow_lib, check_reader=True, allow_stale_vision=a.allow_stale_vision,
                   allow_no_reader=a.no_picture_reader, require_dsv41=a.require_dsv41)
    entries = [p for p in folder.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in entries)
    print(f"{folder}  ({total / (1 << 20):.1f} MiB, {len(entries)} files, manifest sha256 {package_sha256(folder)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
