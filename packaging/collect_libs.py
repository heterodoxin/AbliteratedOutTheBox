#!/usr/bin/env python3
"""Collect the shared libraries an engine binary needs into one folder, ready for packaging/package_engine.py (Linux).

    python packaging/collect_libs.py --binary build/strata --search /usr/local/cuda/lib64 --out stage/ \
        [--also 'libcublasLt.so*'] [--data /opt/rocm/lib/rocblas/library=rocblas/library --data-glob '*gfx1201*']

What it does: runs `ldd` on the binary with LD_LIBRARY_PATH set to --search, copies every dependency that is NOT
supplied by the distribution (anything resolved outside /lib, /lib64, /usr/lib, /usr/lib64 and their x86_64-linux-gnu
folders) into --out under its soname (symlinks are resolved, the real file is copied once per name), repeats for each
copied library (their own dependencies), and adds libraries that are loaded with dlopen and so never show up in ldd
(--also, a glob searched in --search). `--data SRC=DEST` copies a data folder (rocBLAS / hipBLASLt kernel libraries)
to DEST under --out, optionally filtered by --data-glob.

Distribution libraries (libc, libm, libstdc++, libdrm, libnuma, ...) are not copied: they are listed in
`SYSTEM-LIBS.txt` beside the output so a missing one on a target PC can be named. The GPU driver's own libraries
(libcuda.so.1, libnvidia-*, libze_intel_gpu, libOpenCL ICDs) are never bundled: they belong to the driver.
Nothing is downloaded; this only reads files on the build machine.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SYSTEM_DIRS = ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")
DRIVER_PATTERNS = ("libcuda.so*", "libnvidia-*", "libnvcuvid*", "libze_intel_gpu*", "libigdrcl*", "libOpenCL.so*",
                   "libGLX*", "libEGL*", "libvulkan*", "libdrm_amdgpu*", "libdrm.so*", "libamdgpu*")
LDD_LINE = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(0x")


def is_system(path: str) -> bool:
    return path.startswith(SYSTEM_DIRS)


def is_driver(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in DRIVER_PATTERNS)


GFX = re.compile(r"gfx[0-9a-f]+", re.I)


def arch_wanted(name: str, wanted: str) -> bool:
    """Data files for other GPU architectures are left out; shared ones (no gfx name) stay."""
    if not wanted:
        return True
    named = {m.group(0).lower() for m in GFX.finditer(name)}
    return not named or bool(named & {w.strip().lower() for w in wanted.split(",") if w.strip()})


ELF_MAGIC = bytes([0x7F]) + b"ELF"


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == ELF_MAGIC
    except OSError:
        return False


def soname(path: Path) -> str | None:
    """The library's own name (DT_SONAME), which is the name its users ask the loader for."""
    done = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True)
    match = re.search(r"\(SONAME\)\s+Library soname: \[([^\]]+)\]", done.stdout)
    return match.group(1) if match else None


def dependencies(binary: Path, search: list[str]) -> tuple[dict[str, str], list[str]]:
    """{soname: resolved path} and the sonames ldd could not find."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(search + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    done = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, env=env, timeout=120)
    found, missing = {}, []
    for line in done.stdout.splitlines():
        if "not found" in line:
            missing.append(line.split("=>")[0].strip())
            continue
        match = LDD_LINE.match(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--search", action="append", default=[], help="a library folder to resolve against (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--also", action="append", default=[], help="glob of dlopen'd libraries to add, searched in --search")
    ap.add_argument("--data", action="append", default=[], metavar="SRC=DEST")
    ap.add_argument("--data-glob", action="append", default=[], help="keep only data files matching one of these globs")
    ap.add_argument("--data-arch", default="", help="comma list (gfx1200,gfx1201): of the data files that name a gfx arch keep "
                                                    "only these; files naming none (shared tables) are kept")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    system: set[str] = set()
    queue = [a.binary]
    seen = set()
    while queue:
        item = queue.pop()
        if item in seen:
            continue
        seen.add(item)
        found, missing = dependencies(item, a.search)
        for name in missing:
            if not is_driver(name):
                print(f"WARNING: {item.name} needs {name}, which ldd cannot find", file=sys.stderr)
        for name, path in found.items():
            if is_driver(name):
                continue
            if is_system(path):
                system.add(name)
                continue
            if name in copied:
                continue
            real = Path(path).resolve()
            shutil.copy2(real, a.out / name)
            copied[name] = str(real)
            queue.append(a.out / name)
    for pattern in a.also:
        hits = [Path(d) / n for d in a.search if os.path.isdir(d) for n in sorted(os.listdir(d)) if fnmatch.fnmatch(n, pattern)]
        if not hits:
            print(f"WARNING: --also {pattern!r} matched nothing in {a.search}", file=sys.stderr)
        for hit in hits:
            if not hit.is_file() or not is_elf(hit):
                continue
            real = hit.resolve()
            if str(real) in copied.values():          # libX.so -> libX.so.1 -> libX.so.1.2.3: one file, one name (the soname)
                continue
            name = soname(real) or hit.name
            if name in copied:
                continue
            shutil.copy2(real, a.out / name)
            copied[name] = str(real)
            found, missing = dependencies(a.out / name, a.search)
            for dep in missing:
                if not is_driver(dep):
                    print(f"WARNING: {name} needs {dep}, which ldd cannot find", file=sys.stderr)
            for dep, path in found.items():
                if is_driver(dep) or dep in copied:
                    continue
                if is_system(path):
                    system.add(dep)
                else:
                    shutil.copy2(Path(path).resolve(), a.out / dep)
                    copied[dep] = str(Path(path).resolve())
    for spec in a.data:
        src, _, dest = spec.partition("=")
        if not dest:
            ap.error(f"--data {spec!r} is not SRC=DEST")
        for path in sorted(Path(src).rglob("*")):
            if path.is_file() and (not a.data_glob or any(fnmatch.fnmatch(path.name, g) for g in a.data_glob))                     and arch_wanted(path.name, a.data_arch):
                target = a.out / dest / path.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path.resolve(), target)
    (a.out / "SYSTEM-LIBS.txt").write_text("\n".join(sorted(system)) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in a.out.rglob("*") if p.is_file())
    print(f"copied {len(copied)} libraries ({total / (1 << 20):.1f} MiB incl. data) to {a.out}; {len(system)} system libraries left to the OS")
    for name in sorted(copied):
        print(f"  {name} <- {copied[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
