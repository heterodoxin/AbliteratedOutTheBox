#!/usr/bin/env python3
"""Build a LOCAL, TEST-SIGNED release folder (a feed) from a Terminal archive and engine packages, to try `install.sh`, `install.ps1`,
`strata update` and `strata launch` end to end without any hosting. Nothing is uploaded.

    python packaging/make_local_feed.py --out FEED --terminal strata-terminal-0.1.45-linux-x86_64.tar.gz \
        --engine /path/to/components/engine/0.1.45-linux-x86_64-cuda [--engine ...] \
        [--keys KEYDIR] [--url http://127.0.0.1:8099] [--sequence 1]
    python -m http.server 8099 --bind 127.0.0.1 --directory FEED         # serve it (foreground; stop it with Ctrl+C)

What it writes into FEED (the folder a person would host):
    strata-feed.json + .sig        the signed feed (Ed25519, the key in KEYDIR, made once, never inside a git checkout)
    control/*, engine/*            the Terminal (as a "control" package) and each engine package, archived
    install-manifest.txt           one line per platform for install.sh / install.ps1: platform version url sha256 size (written by feed_tool build-feed)
    install.sh, install.ps1        the one-line installers with this feed's address filled in (--url)
    test-trust.json                the PUBLIC test key: `STRATA_UPDATE_TRUSTED_KEYS=<this file>` lets this build trust the feed
                                   (the real key goes into strata_control/update_keys.json at release; this one never does)

The engine package folders are copied (never changed in place) before they are signed with the test key.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_FEED = "https://github.com/Niko1221/Strata/releases/latest/download"      # the address built into install.sh / install.ps1; a local feed bakes its own in its place
sys.path.insert(0, str(ROOT / "packaging"))
sys.path.insert(0, str(ROOT))

import feed_tool as ft  # noqa: E402


def extract_terminal(archive: Path, into: Path) -> Path:
    """Unpack a terminal archive; returns the folder that holds python/ and strata.py."""
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    else:
        with tarfile.open(archive) as tf:
            try:
                tf.extractall(into, filter="data")
            except TypeError:                       # Python before 3.12 (a Mac's system 3.9): no extraction filters; the archive is our own
                tf.extractall(into)
    (inner,) = [p for p in into.iterdir() if p.is_dir()]
    return inner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--terminal", type=Path, action="append", default=[], help="a terminal archive (.tar.gz or .zip); repeat per platform")
    ap.add_argument("--engine", type=Path, action="append", default=[], help="an engine package folder (with engine-component.json)")
    ap.add_argument("--keys", type=Path, default=None, help="folder for the test key (default: <out>/../feed-test-keys)")
    ap.add_argument("--url", default="http://127.0.0.1:8099", help="the address this feed will be served at (baked into install.sh/.ps1)")
    ap.add_argument("--sequence", type=int, default=1)
    ap.add_argument("--version", default=None, help="control version label (default: from the first terminal archive's VERSION)")
    ap.add_argument("--channel", default="stable")
    a = ap.parse_args(argv)

    out = a.out.resolve()
    keys = (a.keys or out.parent / "feed-test-keys").resolve()
    if not (keys / "feed-key.json").exists():
        ft.keygen(keys, allow_git=True)
    key = ft._load_key(keys / "feed-key.json")
    shutil.rmtree(out, ignore_errors=True)
    (out / "engine").mkdir(parents=True)
    (out / "control").mkdir(parents=True)
    version = a.version
    with tempfile.TemporaryDirectory(prefix="strata-feed-") as tmp:
        tmpdir = Path(tmp)
        for n, term in enumerate(a.terminal):
            work = tmpdir / f"t{n}"
            work.mkdir()
            folder = extract_terminal(term.resolve(), work)
            v = version or (folder / "VERSION").read_text(encoding="utf-8").strip()
            version = v
            platform = ("windows-x86_64" if (folder / "python" / "python.exe").exists() else
                        "macos-arm64" if ("darwin" in term.name or "macos" in term.name) else "linux-x86_64")      # the feed's own platform names
            fmt = "zip" if platform.startswith("windows") else "tar.gz"
            entry = ft.pack_control(folder, out / "control", version=v, platform=platform, fmt=fmt)
            print(f"control {platform} {v}: {entry['size'] // 1048576} MB")
        for n, pkg in enumerate(a.engine):
            copy = tmpdir / f"e{n}" / pkg.name
            shutil.copytree(pkg, copy, symlinks=True)
            ft.sign_package(copy, key)
            entry = ft.pack_engine(copy, out / "engine")
            print(f"engine {entry['platform']}-{entry['backend']} {entry['version']}: {entry['size'] // 1048576} MB")
            shutil.rmtree(copy.parent, ignore_errors=True)
    ft.build_feed(out, key=key, sequence=a.sequence, channel=a.channel, engine_dir=out / "engine" if a.engine else None,
                  control_dir=out / "control" if a.terminal else None)
    base = a.url.rstrip("/")
    for name in ("install.sh", "install.ps1"):
        text = (ROOT / "packaging" / name).read_text(encoding="utf-8").replace(RELEASE_FEED, base)
        (out / name).write_text(text, encoding="utf-8", newline="\n" if name.endswith(".sh") else None)
    (out / "test-trust.json").write_text((keys / "feed-key.pub.json").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"feed written to {out}\n  serve it:  python -m http.server {base.rsplit(':', 1)[-1]} --bind 127.0.0.1 --directory {out}\n"
          f"  trust it:  STRATA_UPDATE_TRUSTED_KEYS={out / 'test-trust.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
