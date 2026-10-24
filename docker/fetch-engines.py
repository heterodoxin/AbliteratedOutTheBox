#!/usr/bin/env python3
"""Fetch the engine packages for the Docker images from a release, checked against the signed update feed.

    python3 docker/fetch-engines.py --release-tag v0.2.3 --flavors cuda13,cuda12,rocm-gfx120x --out ./pkgs
    python3 docker/fetch-engines.py --source /path/to/release-folder --flavors cuda13 --out ./pkgs
    python3 docker/fetch-engines.py --release-tag v0.2.3 --list

What it does, in the same order and with the same code as `strata update`:

1. reads strata-feed.json and strata-feed.json.sig from the release (GitHub release assets, or a folder / URL given with --source)
   and checks the Ed25519 signature against the public key in strata_control/update_keys.json;
2. for each flavor picks the linux-x86_64 engine package the feed lists for it (see FLAVORS);
3. downloads the archive and checks its size and SHA-256 against the signed feed (a mismatch deletes the file and stops);
4. unpacks it (the archive's names and links are checked, its size may not exceed what the feed declares), checks that the
   manifest is the one the feed names and says what the feed says, then the package signature and every file's hash.

Stdout gets one line per flavor, tab separated: `<flavor> <build.sh --flavor value> <package folder>`; everything else goes to
stderr. Nothing is installed anywhere and nothing is pushed. Needs only the Python standard library and this repository.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from strata_control import engine_component as ec, feed as feedmod, signing, transfer, updater  # noqa: E402

#: image flavor (the tag suffix) -> (feed backend, feed variant or None for any, docker/build.sh --flavor)
FLAVORS = {
    "cuda13": ("cuda", "cuda13", "cuda"),
    "cuda12": ("cuda", "cuda12", "cuda"),
    "rocm-gfx120x": ("hip", "rx9070", "hip"),       # RX 9000, Radeon AI PRO R9700 (gfx1200 / gfx1201)
    "rocm-gfx110x": ("hip", "rx7900", "hip"),       # RX 7000 (gfx1100 / gfx1101 / gfx1102)
    "rocm-gfx1151": ("hip", "halo", "hip"),         # Strix Halo (gfx1151)
    "sycl": ("sycl", None, "sycl"),                 # Intel Arc; only when the release has a Linux SYCL package
}
PLATFORM = "linux-x86_64"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def pick(feed: feedmod.Feed, flavor: str) -> feedmod.EnginePackage | None:
    backend, variant, _ = FLAVORS[flavor]
    found = [p for p in feed.engine if p.platform == PLATFORM and p.backend == backend and (variant is None or p.variant == variant)]
    if not found:
        return None
    return max(found, key=lambda p: feedmod.version_key(p.version))


def fetch_one(feed: feedmod.Feed, pkg: feedmod.EnginePackage, out: Path, trusted: dict) -> Path:
    folder = out / updater.folder_name(pkg)
    archive = out / ".partial" / pkg.url.rsplit("/", 1)[-1]
    if folder.is_dir() and (folder / ec.MANIFEST).is_file() and transfer.sha256_file(folder / ec.MANIFEST) == pkg.manifest_sha256:
        log(f"  {folder.name}: already unpacked, manifest matches the feed")
    else:
        log(f"  downloading {pkg.url} ({pkg.size / (1 << 20):.0f} MiB)")
        try:
            transfer.download(feed.url(pkg.url), archive, size=pkg.size, sha256=pkg.sha256)
        except transfer.TransferError as exc:
            raise SystemExit(f"fetch-engines: {exc}")
        log("  size and SHA-256 match the signed feed; unpacking")
        try:
            updater.extract(archive, folder, pkg.format, pkg.installed_bytes)
        except updater.UpdateError as exc:
            raise SystemExit(f"fetch-engines: {exc}")
        finally:
            archive.unlink(missing_ok=True)
    manifest = folder / ec.MANIFEST
    if transfer.sha256_file(manifest) != pkg.manifest_sha256:
        raise SystemExit(f"fetch-engines: {folder.name} has a manifest the signed feed does not name")
    try:
        component = ec.read_manifest(folder)
    except ec.ComponentProblem as exc:
        raise SystemExit(f"fetch-engines: {folder.name}: the package manifest is invalid: {exc}")
    for name, a, b in (("version", component.version, pkg.version), ("platform", component.platform, pkg.platform),
                       ("backend", component.backend, pkg.backend), ("variant", component.variant, pkg.variant),
                       ("build_id", component.build_id, pkg.build_id), ("archs", set(component.archs), set(pkg.archs))):
        if a != b:
            raise SystemExit(f"fetch-engines: {folder.name}: the manifest says a different {name} than the signed feed")
    outer = ec._verifier
    signing.install_component_verifier(trusted)
    try:
        ec.verify_signature(component, {ec.ENV_REQUIRE_SIGNED: "1"})
        ec.verify(component, full=True)
    except ec.ComponentProblem as exc:
        raise SystemExit(f"fetch-engines: {folder.name} failed its check: {exc}")
    finally:
        ec.register_signature_verifier(outer)
    log(f"  {folder.name}: signature and every file verified (build {pkg.build_id})")
    return folder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release-tag", help="the GitHub release tag, for example v0.2.3")
    ap.add_argument("--repo", default="Niko1221/Strata", help="owner/name of the GitHub repository (default %(default)s)")
    ap.add_argument("--source", help="a release folder or base URL holding strata-feed.json(.sig) and the archives (instead of --release-tag)")
    ap.add_argument("--flavors", default=",".join(FLAVORS), help="comma separated; default: all (%s)" % ", ".join(FLAVORS))
    ap.add_argument("--out", type=Path, help="where the unpacked package folders go")
    ap.add_argument("--skip-missing", action="store_true", help="a flavor the feed has no package for is skipped (default: an error)")
    ap.add_argument("--list", action="store_true", help="only print which package each flavor would use")
    a = ap.parse_args()
    if bool(a.release_tag) == bool(a.source):
        ap.error("give exactly one of --release-tag and --source")
    if not a.list and not a.out:
        ap.error("--out is required")
    flavors = [f.strip() for f in a.flavors.split(",") if f.strip()]
    unknown = [f for f in flavors if f not in FLAVORS]
    if unknown:
        ap.error(f"unknown flavor {', '.join(unknown)} (known: {', '.join(FLAVORS)})")
    source = a.source or f"https://github.com/{a.repo}/releases/download/{a.release_tag}"
    trusted = signing.load_trusted_keys()
    log(f"feed: {source}")
    try:
        feed = feedmod.fetch(source, trusted)
    except feedmod.FeedError as exc:
        raise SystemExit(f"fetch-engines: {exc}")
    log(f"feed signature OK (key {feed.key_id}), channel {feed.channel}, sequence {feed.sequence}")
    status = 0
    for flavor in flavors:
        pkg = pick(feed, flavor)
        if pkg is None:
            log(f"{flavor}: the feed has no {PLATFORM} package for it")
            if not a.skip_missing:
                status = 3
            else:
                print(f"{flavor}\t{FLAVORS[flavor][2]}\tSKIP")
            continue
        log(f"{flavor}: {pkg.label} ({pkg.url})")
        if a.list:
            print(f"{flavor}\t{FLAVORS[flavor][2]}\t{pkg.url}\t{pkg.size}\t{pkg.sha256}")
            continue
        a.out.mkdir(parents=True, exist_ok=True)
        folder = fetch_one(feed, pkg, a.out, trusted)
        print(f"{flavor}\t{FLAVORS[flavor][2]}\t{folder}", flush=True)
    if a.out and (a.out / ".partial").is_dir():
        shutil.rmtree(a.out / ".partial", ignore_errors=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
