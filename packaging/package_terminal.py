#!/usr/bin/env python3
"""Build the self-contained Strata Terminal distribution for Windows x64, Linux x86_64 or macOS arm64.

The requirement this exists for (checklist P04, and the owner's rule that starting or serving never fetches Python
packages from the internet): a user installs the terminal without Desktop, and needs no development checkout, no
system Python, no pip run, no compiler. So the archive carries its own Python runtime and its own dependencies, and
the `strata` launchers prefer them.

What goes in
------------
* a pinned **standalone CPython 3.12.15** (python-build-standalone 20261009, `install_only`), downloaded at BUILD
  time and SHA-256-verified against the pin below (the digests are the ones in that release's SHA256SUMS, the Mac one
  is also what `setup.sh` pins), so the terminal and the installer cannot drift apart;
* the runtime wheels from the target's lock, `packaging/locks/runtime-<target>-cp312.txt` (made by
  `packaging/lock_wheels.py`), installed into that interpreter with
  `pip install --require-hashes --no-deps --only-binary=:all:` at BUILD time only: every package, direct or
  transitive, is named with an exact version and the SHA-256 of the wheel(s) accepted, so nothing else can land in
  the bundle. pip itself is the one the interpreter shipped with (not upgraded);
* the application: `strata.py`, `strata_control`, `strata_cli`, `serve`, a `VERSION` file and the docs the CLI
  references;
* the read-only tooling under `tools/` that the terminal paths import. Everything else under `tools/` is engine
  installation and preparation work and is deliberately excluded: the serving import graph must not pull it in
  (tests/terminal/test_no_acquisition.py checks that);
* launchers: `bin/strata.cmd` and `bin/strata.ps1` (Windows), `bin/strata` (Linux, macOS).

What does NOT go in
-------------------
* the engine binary. That is a separate, independently versioned component (`packaging/README.md`);
* model weights or prepared data;
* `node_modules` or the UI. The terminal does not need a browser.

Usage
-----
    python packaging/package_terminal.py --out release/artifacts                    # the host's target
    python packaging/package_terminal.py --out /tmp/x --target linux-x86_64        # on a Linux x86_64 host
    python packaging/package_terminal.py --out /tmp/x --skip-deps                  # layout check only

An archive for a target must be built on that target: the lock is installed with the bundled interpreter, which runs
only there. `packaging/smoke_terminal.py <archive>` extracts one into a clean environment and runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "packaging" / "locks"

PBS_RELEASE = "20261009"
PYTHON_VERSION = "3.12.15"
PBS_BASE = f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_RELEASE}/"

#: The standalone CPython pins, one per target, from the release's SHA256SUMS. `darwin-arm64` is also what
#: `setup.sh` pins: a terminal built from one and an installer built from the other are the same interpreter.
PYTHON_PINS = {
    "windows-x86_64": {
        "file": f"cpython-{PYTHON_VERSION}+{PBS_RELEASE}-x86_64-pc-windows-msvc-install_only.tar.gz",
        "sha256": "194c8f0d9e0de6f1c4576e5dfa67819a48c764be87608b74b8ad43387a57b93b",
        "python": "python/python.exe", "archive": "zip",
    },
    "linux-x86_64": {
        "file": f"cpython-{PYTHON_VERSION}+{PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz",
        "sha256": "5864001bf969db329fadc2b654c4878c6027d170c6bd77e28f88666260bcaf63",
        "python": "python/bin/python3", "archive": "tar.gz",
    },
    "darwin-arm64": {
        "file": f"cpython-{PYTHON_VERSION}+{PBS_RELEASE}-aarch64-apple-darwin-install_only.tar.gz",
        "sha256": "83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8",
        "python": "python/bin/python3", "archive": "tar.gz",
    },
}
for _pin in PYTHON_PINS.values():
    _pin["url"] = PBS_BASE + _pin["file"].replace("+", "%2B")

#: Kept for callers of the Mac-only script: the macOS pin.
PYTHON_PIN = {**PYTHON_PINS["darwin-arm64"], "target": "darwin-arm64"}

#: The application, as paths relative to ROOT (copied to the same place in the archive).
PAYLOAD = (
    "strata.py",
    "strata_control",
    "strata_cli",
    "serve",
    "requirements.txt",
    "requirements-monitor.txt",
    "docs/MANAGEMENT_PROTOCOL.md",
    "docs/TERMINAL_LAUNCH.md",         # the README below points at it (`strata launch`); it used to be missing from the archive
    "data/gguf_fingerprints.json",     # `strata models import --plan` names a known file set from this table
    "bench/prompts/standard-v1",       # `strata bench run` (the standard benchmark) sends these frozen requests; hashed byte for byte
)

#: `tools/` modules the terminal paths import. Adding one here is a deliberate act: this list is what keeps
#: installation/preparation code out of the serving import graph.
TOOLS_SUBSET = (
    "strata_tokenizer.py",
    "gguf_reader.py",
    "dflash2_pack.py",     # the DFlash2 drafter: strata_control/drafting_fetch.py packs the download with it (numpy only)
)

#: What `strata launch` and the desktop app's installer need on top of the read-only terminal: the model catalog and GPU probe
#: (tools/strata_mcp.py reads setup.py's tables) and the install job, which downloads the model the user chose, prepares it with
#: the bundled tools and writes a run config naming "component:strata-engine". setup.py runs in its CONTROL mode there
#: (STRATA_CONTROL_INSTALL=1: no pip, no engine fetch, no compiler). Entry points; their local imports under tools/ come with them.
#: The same list the app's Control component stages (apps/desktop/scripts/stage-control.py uses these names).
INSTALLER_FILES = ("setup.py", "ref/load.py")   # ref/load.py: strata_pack.py (the Q2_0 AVX-512 repack) imports it
#: calibrate: setup.py imports it after a tune (calibrate_config, and the earlier tuning that a rerun applies); it was missing from this
#: list in 0.1.47, so every launch after a tune failed (#3613, #3467). setup_tool_imports() below also reads setup.py itself, so a tools/
#: module it starts to import is never forgotten again; tests/terminal/test_bundle_imports.py checks the staged bundle.
INSTALLER_TOOLS = ("strata_mcp", "iq_pack", "strata_pack", "pack_index", "mtp_fetch", "mtp_pack", "mtp_rt", "draft_vocab",
                   "strata_tokenizer", "gguf_reader", "calibrate")
#: tools run BY PATH (`python tools/.../x.py`) by setup.py, strata_cli or strata_control: no import names them, so setup_tool_imports() cannot find
#: them.  DeepSeek's setup and launch run dsv41/pack_gguf.py (the model folder), dsv41/smoke.py (the engine's first answer) and dsv41/tune.py
#: (the short tune); setup.py's other by-path tools (nvfp4_local_pack, strata_bench, strata_inspect) were missing from the bundle the same way.
#: tests/terminal/test_bundle_tools_paths.py reads the by-path references out of the source and fails when one is missing from the bundle.
INSTALLER_SCRIPTS = ("dsv41/pack_gguf", "dsv41/smoke", "dsv41/tune", "nvfp4_local_pack", "strata_bench", "strata_inspect")
INSTALLER_DATA = "data"           # draft vocabularies, expert profiles: 2 MB

#: packages the bundle must import after the install (the API server's and the CLI's).
PROBE = "import jinja2, yaml, regex, psutil, requests, numpy, PIL, tqdm; print('deps ok')"


def host_target() -> str:
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "windows-x86_64"
    if sys.platform.startswith("linux") and machine in ("x86_64", "amd64"):
        return "linux-x86_64"
    if sys.platform == "darwin" and machine in ("arm64", "aarch64"):
        return "darwin-arm64"
    raise SystemExit(f"no terminal target for {sys.platform}/{machine}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_python(target: str, cache: Path) -> Path:
    """Download the pinned interpreter once (build time), verifying the digest before it is used."""
    pin = PYTHON_PINS[target]
    archive = cache / pin["file"].replace("+", "_")
    if not archive.exists():
        print(f"fetching the pinned CPython from {pin['url']}")
        part = archive.with_suffix(archive.suffix + ".part")
        urllib.request.urlretrieve(pin["url"], part)
        part.replace(archive)
    got = sha256(archive)
    if got != pin["sha256"]:
        archive.unlink(missing_ok=True)
        raise SystemExit(
            f"the CPython download does not match its pin\n  expected {pin['sha256']}\n"
            f"  got      {got}\nRefusing to package an unverified interpreter.")
    return archive


def ignore(_directory: str, names: list[str]) -> set[str]:
    return {n for n in names if n in ("__pycache__", ".DS_Store") or n.endswith(".pyc")}


def lock_for(target: str) -> Path:
    path = LOCKS / f"runtime-{target}-cp312.txt"
    if not path.is_file():
        raise SystemExit(f"no lock for {target}: {path} (run packaging/lock_wheels.py --target {target})")
    return path


def lock_names(lock: Path) -> dict[str, str]:
    out = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
        if m:
            out[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return out


def install_locked(py: Path, lock: Path, root: Path, wheelhouse: Path | None = None) -> dict[str, str]:
    """pip install --require-hashes into the bundled interpreter, then check what is there is exactly the lock.
    With a wheelhouse (lock_wheels.py --download) nothing is fetched at all: pip reads the wheels from the folder and
    still checks every hash against the lock."""
    env = {**os.environ, "PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONNOUSERSITE": "1",
           "PIP_CONFIG_FILE": os.devnull}
    for name in ("PYTHONHOME", "PYTHONPATH", "PIP_REQUIRE_VIRTUALENV", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
        env.pop(name, None)
    source = ["--no-index", "--find-links", str(wheelhouse)] if wheelhouse else []
    subprocess.run([str(py), "-m", "pip", "install", "--isolated", "--no-cache-dir", "--no-warn-script-location",
                    "--require-hashes", "--no-deps", "--only-binary=:all:", *source, "-r", str(lock)],
                   check=True, cwd=str(root), env=env)
    subprocess.run([str(py), "-m", "pip", "check"], check=True, cwd=str(root), env=env)
    listed = subprocess.run([str(py), "-m", "pip", "list", "--format=json", "--isolated"], check=True,
                            cwd=str(root), env=env, capture_output=True, text=True).stdout
    have = {re.sub(r"[-_.]+", "-", d["name"]).lower(): d["version"] for d in json.loads(listed)}
    want = lock_names(lock)
    extra = {k: v for k, v in have.items() if k not in want and k not in ("pip", "setuptools", "wheel")}
    wrong = {k: (want[k], have.get(k)) for k in want if have.get(k) != want[k]}
    if extra or wrong:
        raise SystemExit(f"the bundle does not match its lock: extra {extra}, wrong/missing {wrong}")
    return have


def drop_build_path_launchers(root: Path, staging: Path, target: str) -> list[str]:
    """pip writes console-script launchers that name the build-time interpreter by absolute path: they break the
    moment the archive is extracted elsewhere. `python -m <module>` and the `strata` launchers replace them."""
    gone = []
    if target.startswith("windows"):
        scripts = root / "python" / "Scripts"
        if scripts.is_dir():
            for f in sorted(scripts.iterdir()):
                gone.append(f.name)
            shutil.rmtree(scripts)
        return gone
    needle = str(staging).encode()
    bindir = root / "python" / "bin"
    for f in sorted(bindir.iterdir()):
        if f.is_symlink() or not f.is_file():
            continue
        with f.open("rb") as fh:
            head = fh.read(512)
        if head.startswith(b"#!") and needle in head.split(b"\n", 1)[0]:
            gone.append(f.name)
            f.unlink()
    return gone


def local_imports(folder: Path, entry: tuple[str, ...]) -> list[str]:
    """The tools/ modules `entry` imports, transitively (static: import and from-import statements)."""
    import ast
    seen: set[str] = set()
    todo = list(entry)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        source = folder / "tools" / f"{name}.py"
        if not source.exists():
            raise SystemExit(f"installer tool missing: tools/{name}.py")
        sub = name.rsplit("/", 1)[0] + "/" if "/" in name else ""     # (tools/dsv41/tune.py imports its neighbour smoke.py)
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else                 [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            for m in names:
                for rel in ((sub + m,) if sub else ()) + (m,):
                    if (folder / "tools" / f"{rel}.py").exists():
                        if rel not in seen:
                            todo.append(rel)
                        break
    return sorted(seen)


def setup_tool_imports(source: Path = ROOT) -> list[str]:
    """The `tools/<name>.py` modules that the installer files import (any import statement, at any depth: setup.py imports most of them
    inside functions after putting tools/ on sys.path). tools/hip and tools/image hold engine-install code that Control's mode never runs."""
    import ast
    found: set[str] = set()
    for name in INSTALLER_FILES:
        for node in ast.walk(ast.parse((source / name).read_text(encoding="utf-8-sig"))):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else \
                [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            found.update(m for m in names if (source / "tools" / f"{m}.py").exists())
    return sorted(found)


def add_installer(folder: Path, source: Path = ROOT) -> dict:
    """Copy the installer's files from the source tree into a staged terminal folder (after the terminal payload is in place)."""
    tools = local_imports(source, tuple(sorted(set(INSTALLER_TOOLS) | set(INSTALLER_SCRIPTS) | set(setup_tool_imports(source)))))
    copied = []
    for name in INSTALLER_FILES:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, folder / name)
        copied.append(name)
    (folder / "tools").mkdir(exist_ok=True)
    for name in tools:
        (folder / "tools" / f"{name}.py").parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "tools" / f"{name}.py", folder / "tools" / f"{name}.py")
        copied.append(f"tools/{name}.py")
    if (folder / INSTALLER_DATA).exists():
        shutil.rmtree(folder / INSTALLER_DATA)
    shutil.copytree(source / INSTALLER_DATA, folder / INSTALLER_DATA, ignore=shutil.ignore_patterns("__pycache__"))
    return {"files": copied, "data": INSTALLER_DATA, "setup_sha256": sha256(folder / "setup.py"),
            "mode": "STRATA_CONTROL_INSTALL=1: downloads the chosen model, prepares it, installs no package and no engine"}


def write_archive(root: Path, name: str, out_dir: Path, kind: str) -> Path:
    if kind == "zip":
        path = out_dir / f"{name}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(root.rglob("*")):
                if p.is_dir():
                    continue
                zf.write(p, Path(name, p.relative_to(root)).as_posix())
        return path
    path = out_dir / f"{name}.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        tf.add(root, arcname=name)
    return path


def build_commit(explicit: str | None = None) -> str | None:
    """The full commit this archive is built from: --commit, the STRATA_COMMIT variable, STRATA_BUILD_COMMIT (a Docker build
    copies the files without .git and without git), or `git rev-parse HEAD` of the source tree. None when none of them gives
    40 hex digits (a tree copy without .git)."""
    for candidate in (explicit, os.environ.get("STRATA_COMMIT"), os.environ.get("STRATA_BUILD_COMMIT")):
        if candidate and re.fullmatch(r"[0-9a-fA-F]{40}", candidate.strip()):
            return candidate.strip().lower()
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip().lower()
    except OSError:                     # no git installed
        return None
    return out if re.fullmatch(r"[0-9a-f]{40}", out) else None


def source_commit() -> str:
    """`build_commit()` without an explicit value, as a string (empty when unknown)."""
    return build_commit() or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory to write the archive into")
    ap.add_argument("--target", choices=sorted(PYTHON_PINS), default=None,
                    help="windows-x86_64, linux-x86_64 or darwin-arm64 (default: this machine's)")
    ap.add_argument("--version", default=None, help="version label (default: read from the source)")
    ap.add_argument("--skip-deps", action="store_true",
                    help="skip the locked install (layout checks only; the result is NOT distributable)")
    ap.add_argument("--commit", default=None, metavar="SHA",
                    help="the full commit the sources come from, for a tree copy that has no .git (or set STRATA_COMMIT); "
                         "`strata bench run` records it, and a standard benchmark without it is nonconforming")
    ap.add_argument("--keep-staging", action="store_true")
    ap.add_argument("--no-installer", action="store_true",
                    help="leave out setup.py, tools/strata_mcp.py and data/ (the read-only terminal of earlier releases; "
                         "`strata launch` then cannot list or install models)")
    ap.add_argument("--cache", default=None, help="where the pinned interpreter archive is cached")
    ap.add_argument("--wheelhouse", default=None,
                    help="install the locked wheels from this folder (lock_wheels.py --download) and fetch nothing")
    args = ap.parse_args()

    target = args.target or host_target()
    if not args.skip_deps and target != host_target():
        raise SystemExit(f"{target} must be built on {target} (this is {host_target()}): the locked install runs the "
                         "bundled interpreter. Use --skip-deps for a layout check.")
    commit = build_commit(args.commit)
    if commit is None:
        message = ("the commit of these sources is unknown (no .git here, no --commit, no STRATA_COMMIT): the archive would carry "
                   "no COMMIT file, and every `strata bench run` on it would be nonconforming")
        if not args.skip_deps:
            raise SystemExit(message)
        print("warning: " + message)
    pin = PYTHON_PINS[target]
    version = args.version or _source_version()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"strata-terminal-{version}-{target}"

    cache = Path(args.cache) if args.cache else Path(tempfile.gettempdir()) / "strata-packaging-cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = fetch_python(target, cache)
    lock = lock_for(target)

    staging = Path(tempfile.mkdtemp(prefix="strata-terminal-build-"))
    root = staging / name
    root.mkdir(parents=True)
    try:
        print("unpacking the interpreter")
        with tarfile.open(archive) as tf:
            tf.extractall(root)
        py = root / pin["python"]
        if not py.exists():
            raise SystemExit(f"the archive did not produce {py}")

        print("copying the application")
        for item in PAYLOAD:
            src = ROOT / item
            dst = root / item
            if not src.exists():
                raise SystemExit(f"missing payload item: {item}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=ignore)
            else:
                shutil.copy2(src, dst)
        tools_dir = root / "tools"
        tools_dir.mkdir(exist_ok=True)
        for item in TOOLS_SUBSET:
            src = ROOT / "tools" / item
            if not src.exists():
                raise SystemExit(f"missing tools module: {item}")
            shutil.copy2(src, tools_dir / item)
        installer = None if args.no_installer else add_installer(root)
        (root / "VERSION").write_text(version + "\n", encoding="utf-8")
        # the commit the archive was built from: `strata bench run` records it (an archive has no .git to ask)
        if commit:
            (root / "COMMIT").write_text(commit + "\n", encoding="utf-8")

        bin_dir = root / "bin"
        bin_dir.mkdir(exist_ok=True)
        launchers = ["strata.cmd", "strata.ps1"] if target.startswith("windows") else ["strata"]
        for launcher in launchers:
            shutil.copy2(ROOT / "bin" / launcher, bin_dir / launcher)
            if launcher == "strata":
                (bin_dir / launcher).chmod(0o755)

        installed: dict[str, str] = {}
        removed: list[str] = []
        if not args.skip_deps:
            print("installing the locked runtime wheels into the bundled interpreter (build time only)")
            installed = install_locked(py, lock, root, Path(args.wheelhouse).resolve() if args.wheelhouse else None)
            removed = drop_build_path_launchers(root, staging, target)
            # nothing the bundle writes while it is built may stay: compiled files are rebuilt on first use
            for cache_dir in list(root.rglob("__pycache__")):
                shutil.rmtree(cache_dir, ignore_errors=True)

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        for k in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(k, None)
        probe = subprocess.run([str(py), "-c", PROBE], capture_output=True, text=True, cwd=str(root), env=env)
        print("  " + (probe.stdout.strip() or probe.stderr.strip()[:200]))
        if not args.skip_deps and probe.returncode != 0:
            raise SystemExit("the bundled interpreter cannot import the runtime packages")

        (root / "README.txt").write_text(_readme(version, target), encoding="utf-8")

        archive_path = write_archive(root, name, out_dir, pin["archive"])
        digest = sha256(archive_path)
        size = archive_path.stat().st_size
        print(f"\n{archive_path}")
        print(f"  bytes  {size}")
        print(f"  sha256 {digest}")

        manifest = out_dir / f"{name}.manifest.json"
        manifest.write_text(json.dumps({
            "name": name,
            "version": version,
            "target": target,
            "archive": archive_path.name,
            "bytes": size,
            "sha256": digest,
            "python": {"version": PYTHON_VERSION, "build": PBS_RELEASE, "url": pin["url"], "sha256": pin["sha256"],
                       "target": target},
            "lock": {"file": lock.relative_to(ROOT).as_posix(), "sha256": sha256(lock),
                     "install": "pip install --require-hashes --no-deps --only-binary=:all: (build time only)"},
            "installed": installed,
            "launchers_removed": removed,
            "payload": list(PAYLOAD),
            "tools_subset": list(TOOLS_SUBSET),
            "installer": installer,
            "excludes": ["engine binary", "model weights", "prepared data", "UI", "node_modules"],
            "deps_installed": not args.skip_deps,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"  manifest {manifest}")
        return 0
    finally:
        if args.keep_staging:
            print(f"staging kept at {staging}")
        else:
            shutil.rmtree(staging, ignore_errors=True)


def _source_version() -> str:
    """The version the source reports, so the artifact name cannot disagree with the code."""
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
    # `project(strata VERSION 0.1.44.1 LANGUAGES C CXX ...)` - the version is a dotted run of numbers and must be
    # read as one. Taking "everything after VERSION" yields "0.1.44.1.LANGUAGES.CXX".
    m = re.search(r"^\s*project\([^)]*?VERSION\s+([0-9]+(?:\.[0-9]+)+)", text, re.MULTILINE)
    return m.group(1) if m else "0.0.0"


def _readme(version: str, target: str) -> str:
    win = target.startswith("windows")
    ext = "zip" if win else "tar.gz"
    stem = f"strata-terminal-{version}-{target}"
    if win:
        install = f"""    Expand-Archive {stem}.zip .
    cd {stem}
    .\\bin\\strata.cmd --help

Put `bin\\` on your PATH to get the `strata` command everywhere (cmd.exe and PowerShell):

    setx PATH "%PATH%;%CD%\\bin"
"""
    else:
        install = f"""    tar xzf {stem}.tar.gz
    cd {stem}
    ./bin/strata --help

Put `bin/` on your PATH to get the `strata` command everywhere:

    echo "export PATH=\\"$PWD/bin:$PATH\\"" >> ~/.profile
"""
    ram = ("    Task Manager > Performance > Memory > Available, or in PowerShell:\n"
           "    [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)   # GB free" if win else
           "    free -g" if not target.startswith("darwin") else
           "    Activity Monitor > Memory, or:  vm_stat   (free + inactive pages x 16 KB)")
    return f"""Strata Terminal {version} ({target})
{'=' * (len(version) + len(target) + 18)}

This is the terminal distribution. It carries its own Python {PYTHON_VERSION} and its own packages, so you do not
need Python, pip, Node, a compiler or a development checkout. Serving and watching (`strata serve`, `strata runtime`,
`strata monitor`) never install or download anything; if something is missing, the error names the command that
installs it. Only two commands fetch files, and only when you run them: `strata launch` (the engine and the model you
choose, after it tells you the size) and `strata update` (the engine and the Strata components).

Install
-------
{install}
Before you start a big model
----------------------------
A model of tens of GB needs free memory. Check how much is free first:

{ram}

Close other programs (browsers use a lot) if it is less than the model needs; `strata launch` shows what each model needs.

What it does NOT include
------------------------
The **engine** is a separate, independently versioned component and is not bundled: the right engine package
depends on your OS, CPU and GPU. `strata launch` (or `strata update`) fetches the one for your card from the signed
release feed; or point Strata at one with STRATA_ENGINE. Model weights are likewise not included: `strata launch`
downloads the one you choose (the installer files setup.py and tools/ are in this archive for that).

Useful commands
---------------
    strata launch                     choose a model, tune it, serve it and watch it (guided screens; docs/TERMINAL_LAUNCH.md)
    strata connect opencode           the setup for a coding app on the running server (also codex, claude-code, omp, pi, hermes, openclaw)
    strata --version                  the version and the interpreter it runs on
    strata doctor                     read-only component inventory
    strata profile list               saved serving profiles
    strata status                     one read-only telemetry snapshot
    strata monitor                    the terminal monitor
    strata serve --profile NAME       start the inference server in the foreground
    strata serve --engine mock ...    a scripted engine for client tests (no GPU, no model)
    strata runtime start|status|stop  run a profile as a managed background instance

Existing installs are found in place: the run configs (strata-<model>.json) and the data folder remembered in
settings.json are used where they are; nothing is copied or moved.

Verification
------------
The manifest next to the archive records the SHA-256 of the archive, of the pinned interpreter and of the package
lock it was built from. The launcher prefers the bundled interpreter and ignores the user's site-packages.
"""


if __name__ == "__main__":
    sys.exit(main())
