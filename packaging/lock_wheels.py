#!/usr/bin/env python3
"""Write the hash-pinned runtime lock of one terminal target.

The terminal archive installs its Python packages from `packaging/locks/runtime-<target>-cp312.txt` with
`pip install --require-hashes --no-deps --only-binary=:all:`: every package, direct or not, is named with an exact
version and the SHA-256 of the wheel(s) accepted for it, so a build can only ever install the files this lock lists.
The lock is made here, from `requirements.txt` + `requirements-monitor.txt` (the direct, version-pinned runtime
packages) by letting pip resolve and download the wheels for the target's tags and hashing what it fetched.

    python packaging/lock_wheels.py --target windows-x86_64          # writes packaging/locks/...
    python packaging/lock_wheels.py --target linux-x86_64 --check    # fails when the committed lock differs
    python packaging/lock_wheels.py --target linux-x86_64 --download wheelhouse/   # exactly the locked wheels

Run it when a pin in requirements.txt changes, review the diff, commit. Building an archive never runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "packaging" / "locks"
INPUTS = ("requirements.txt", "requirements-monitor.txt")

#: pip's tags for each target, and the environment markers evaluate under for it.
TARGETS = {
    "windows-x86_64": {"platforms": ["win_amd64"], "env": {"sys_platform": "win32", "platform_system": "Windows",
                                                            "os_name": "nt"}},
    "linux-x86_64": {"platforms": ["manylinux2014_x86_64", "manylinux_2_17_x86_64", "manylinux_2_28_x86_64",
                                   "manylinux_2_34_x86_64", "linux_x86_64"],
                     "env": {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix"}},
    "darwin-arm64": {"platforms": ["macosx_11_0_arm64", "macosx_12_0_arm64", "macosx_13_0_arm64",
                                   "macosx_14_0_arm64"],
                     "env": {"sys_platform": "darwin", "platform_system": "Darwin", "os_name": "posix"}},
}
PYTHON = "3.12"
ABI = "cp312"


def lock_path(target: str) -> Path:
    return LOCKS / f"runtime-{target}-{ABI}.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dropped_for_target(target: str) -> set[str]:
    """Direct requirements whose environment marker is false for the target (pip evaluated markers for the host that
    runs this script: a Windows-only package must not land in the Linux lock, nor the reverse)."""
    from packaging.requirements import Requirement
    env = {"python_version": PYTHON, "python_full_version": PYTHON + ".15", "implementation_name": "cpython",
           "platform_python_implementation": "CPython", **TARGETS[target]["env"]}
    dropped = set()
    for name in INPUTS:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.split(" #")[0].strip()
            if not line or line.startswith("#"):
                continue
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate(env):
                dropped.add(norm(req.name))
    # a requirement listed twice with different markers (numpy) is kept when any line applies
    kept = set()
    for name in INPUTS:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.split(" #")[0].strip()
            if not line or line.startswith("#"):
                continue
            req = Requirement(line)
            if req.marker is None or req.marker.evaluate(env):
                kept.add(norm(req.name))
    return dropped - kept


def build_lock(target: str) -> str:
    spec = TARGETS[target]
    with tempfile.TemporaryDirectory(prefix="strata-lock-") as tmp:
        cmd = [sys.executable, "-m", "pip", "download", "--quiet", "--only-binary=:all:",
               "--python-version", PYTHON, "--implementation", "cp", "--abi", ABI, "-d", tmp]
        for p in spec["platforms"]:
            cmd += ["--platform", p]
        for name in INPUTS:
            cmd += ["-r", str(ROOT / name)]
        subprocess.run(cmd, check=True)
        wheels: dict[str, tuple[str, list[str]]] = {}
        for whl in sorted(Path(tmp).glob("*.whl")):
            name, version = whl.name.split("-")[:2]
            wheels.setdefault(norm(name), (version, []))[1].append(sha256(whl))
    drop = _dropped_for_target(target)
    out = [
        "# Hash-pinned runtime packages for the " + target + " terminal archive (CPython 3.12, " + ABI + ").",
        "# Made by packaging/lock_wheels.py from requirements.txt + requirements-monitor.txt; installed at BUILD time with",
        "#   pip install --require-hashes --no-deps --only-binary=:all: -r <this file>",
        "# and never on the user's PC. To change a package: edit requirements.txt, run lock_wheels.py, review the diff.",
    ]
    for name in sorted(wheels):
        if name in drop:
            continue
        version, hashes = wheels[name]
        out.append(f"{name}=={version} \\")
        out.append(" \\\n".join(f"    --hash=sha256:{h}" for h in sorted(set(hashes))))
    return "\n".join(out) + "\n"


def download_locked(target: str, dest: Path) -> int:
    """Fetch exactly the wheels the committed lock names (pip verifies every hash) into `dest`, for a build on a
    machine with no internet: package_terminal.py --wheelhouse dest. Works from any host for any target."""
    spec = TARGETS[target]
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "--quiet", "--only-binary=:all:", "--require-hashes", "--no-deps",
           "--python-version", PYTHON, "--implementation", "cp", "--abi", ABI, "-d", str(dest),
           "-r", str(lock_path(target))]
    for p in spec["platforms"]:
        cmd += ["--platform", p]
    subprocess.run(cmd, check=True)
    count = len(list(dest.glob("*.whl")))
    print(f"{count} wheels in {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--check", action="store_true", help="compare with the committed lock instead of writing")
    ap.add_argument("--download", metavar="DIR", help="fetch the locked wheels into DIR (hash-checked) and stop")
    a = ap.parse_args()
    if a.download:
        return download_locked(a.target, Path(a.download))
    text = build_lock(a.target)
    path = lock_path(a.target)
    if a.check:
        same = path.exists() and path.read_text(encoding="utf-8") == text
        print(("lock is current: " if same else "lock DIFFERS: ") + str(path))
        return 0 if same else 1
    LOCKS.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path} ({text.count('==')} packages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
