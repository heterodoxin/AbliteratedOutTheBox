#!/usr/bin/env python3
"""Put the hipBLASLt tuning tables into a staged AMD engine package and record the hipBLASLt version in its BUILD.json.

    python packaging/stage_tuning.py --stage <stage folder> --archs "gfx1200 gfx1201" --rocm <ROCm devel root> [--tables tools/hip]
                                     [--expect gfx1201]

The stage folder is the one packaging/package_engine.py is pointed at. The hipBLASLt version is read from the ROCm tree the package's
libraries come from (include/hipblaslt/hipblaslt-version.h; 1.5.0 is 100500), written to BUILD.json as ``hipblaslt_version``, and every
table of tools/hip for one of --archs and exactly that version is copied to ``<stage>/tuning/``. An architecture without a table keeps plain
hipBLAS: it is listed on stdout, and --expect makes it an error for the ones that must have one. The tables are measured per architecture and
version (docs/AMD_HIP.md, Tuning table); nothing is calibrated or invented here. Windows packages do the same inside tools/hip/package_windows.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata_control import hipblaslt_tuning as ht  # noqa: E402


def stage(stage_dir: Path, archs: list[str], version: int | None, tables: Path, expect: list[str]) -> list[str]:
    if version is None:
        raise SystemExit("cannot read the hipBLASLt version of this ROCm tree (include/hipblaslt/hipblaslt-version.h): "
                         "an AMD package must record hipblaslt_version")
    info = stage_dir / "BUILD.json"
    meta = json.loads(info.read_text(encoding="utf-8-sig")) if info.is_file() else {}
    meta["hipblaslt_version"] = version
    info.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    copied = ht.stage_tables(stage_dir, archs, version, tables)
    left = ht.missing_report(archs, version, tables)
    for arch, have in left.items():
        print(f"no hipBLASLt table for {arch} with hipBLASLt {version} (the repo has: {', '.join(map(str, have)) or 'none'}): plain hipBLAS there")
    bad = [a for a in expect if a in left]
    if bad:
        raise SystemExit(f"required tables are missing for {', '.join(bad)} at hipBLASLt {version}")
    return copied


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, type=Path)
    ap.add_argument("--archs", required=True, help="space, comma or semicolon separated")
    ap.add_argument("--rocm", type=Path, help="the ROCm devel root (its include/hipblaslt/hipblaslt-version.h)")
    ap.add_argument("--version", type=int, help="the hipBLASLt version number (100500), instead of --rocm")
    ap.add_argument("--tables", type=Path, default=ROOT / "tools" / "hip")
    ap.add_argument("--expect", default="", help="architectures that must have a table (an error when one is missing)")
    a = ap.parse_args(argv)
    archs = [x for x in re.split(r"[;, ]+", a.archs) if x]
    version = a.version if a.version else ht.version_from_header(a.rocm) if a.rocm else None
    copied = stage(a.stage, archs, version, a.tables, [x for x in re.split(r"[;, ]+", a.expect) if x])
    print(f"hipBLASLt {version}: " + (", ".join(copied) if copied else "no tuning tables in this package"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
