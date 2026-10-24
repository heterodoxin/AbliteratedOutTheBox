"""DeepSeek-V4.1's engine as its own package (feed member ``dsv41``), downloaded when DeepSeek is chosen.

It goes in ``<data>/engine-dsv41/<version>-<platform>-<backend>[-<variant>]/``, beside the Strata engine and never inside it, so the Qwen
engine's files are not touched.  The package is picked, downloaded (resumable), unpacked and checked with the same code as an engine update
(strata_control/updater.py: the signed feed, the archive's SHA-256, the manifest hash the feed names, the package signature, every file's hash);
there is no test start here (setup's own 16-token question follows the pack).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

NAME = "strata-dsv41"
DIR = "engine-dsv41"


def exe_name() -> str:
    return NAME + (".exe" if os.name == "nt" else "")


def find(dest: Path) -> Path | None:
    """The newest installed package's strata-dsv41 in `dest` (engine-dsv41/), or None."""
    try:
        folders = sorted((d for d in Path(dest).iterdir() if d.is_dir() and (d / exe_name()).is_file()), key=lambda d: d.name)
    except OSError:
        return None
    return folders[-1] / exe_name() if folders else None


ARCHIVE_SUFFIXES = (".tar.gz", ".tar.xz", ".zip")
MAX_UNPACKED = 8 << 30                  # a DeepSeek engine package is under a GB; this only stops an archive that unpacks into something huge


def pending(dest: Path) -> list[Path]:
    """Packages the person put in `dest` (engine-dsv41/) by hand: strata-dsv41-*.tar.gz / .zip, newest first."""
    try:
        files = [f for f in Path(dest).iterdir() if f.is_file() and f.name.startswith("strata-dsv41") and f.name.endswith(ARCHIVE_SUFFIXES)]
    except OSError:
        return []
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def install_package(dest: Path, package: str | os.PathLike, *, emit: Callable[[str], None] = print) -> Path | None:
    """Install a DeepSeek engine package the person downloaded from the release page (a .tar.gz / .zip, or the unpacked folder) into `dest`.
    Checked before anything is installed: it is a DeepSeek engine package for this system, its manifest is signed by a key Strata trusts, and every
    file matches the hash in the manifest.  Returns the new strata-dsv41, or None with a plain line through `emit`."""
    from strata_control import components, engine_component as ec, signing, updater
    from strata_control.targets import host_platform

    src, dest = Path(package).expanduser(), Path(dest)
    if not src.exists():
        emit(f"{src} is not there. Download the DeepSeek engine package for your system from the release page and give its path again.")
        return None
    work = dest.parent / (DIR + ".manual")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        if src.is_dir():
            root = work / "pkg"
            shutil.copytree(src, root)
        else:
            fmt = "zip" if src.name.endswith(".zip") else "tar.xz" if src.name.endswith(".tar.xz") else "tar.gz"
            if not src.name.endswith(ARCHIVE_SUFFIXES):
                emit(f"{src.name} is not a package file (.tar.gz or .zip). Download the DeepSeek engine package for your system from the release page.")
                return None
            root = work / "pkg"
            updater.extract(src, root, fmt, MAX_UNPACKED)
        if not (root / ec.MANIFEST).is_file():
            inner = next((d for d in root.iterdir() if d.is_dir() and (d / ec.MANIFEST).is_file()), None) if root.is_dir() else None
            if inner is None:
                emit(f"{src.name} is not an engine package (it has no {ec.MANIFEST}).")
                return None
            root = inner
        component = ec.read_manifest(root)
        if component.entrypoint != exe_name():
            emit(f"{src.name} is not the DeepSeek engine (its program is {component.entrypoint}, not {exe_name()}). "
                 "The Strata engine package is a different download.")
            return None
        if component.platform != host_platform():
            emit(f"{src.name} is for {component.platform}, and this computer is {host_platform()}. Download the one for your system.")
            return None
        outer = ec._verifier
        signing.install_component_verifier(signing.load_trusted_keys())
        try:
            ec.verify_signature(component, {**os.environ, ec.ENV_REQUIRE_SIGNED: "1"})
            ec.verify(component, full=True)
        except ec.ComponentProblem as exc:
            emit(f"{src.name} did not pass its check ({exc}). It may be damaged or not from the Strata release: download it again from the release page.")
            return None
        finally:
            ec.register_signature_verifier(outer)
        name = f"{component.version}-{component.platform}-{component.backend}" + (f"-{component.variant}" if component.variant else "")
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / name
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(root), str(target))
        if os.name != "nt":
            (target / exe_name()).chmod(0o755)
        for old in dest.iterdir():
            if old.is_dir() and old.name != target.name:
                shutil.rmtree(old, ignore_errors=True)
        emit(f"DeepSeek engine {component.version} ({component.backend}) checked (signed, every file matches) and installed: {target}")
        return target / exe_name()
    except (updater.UpdateError, ec.ComponentProblem, OSError) as exc:
        emit(f"{src.name} could not be installed: {exc}")
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def install(dest: Path, source: str, *, hardware: dict | None = None, emit: Callable[[str], None] = print, cancel=None) -> Path | None:
    """Fetch the DeepSeek engine package for this PC from the signed feed at `source` (a URL or a folder) into `dest`.
    Returns the new strata-dsv41, or None with a plain line through `emit` when there is none (not in the feed, nothing for this PC)."""
    from strata_control import feed as feedmod, signing, updater

    dest = Path(dest)
    work = dest.parent / (DIR + ".work")
    hw = hardware or updater.detect_hardware()
    state_dir = work / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    ctx = updater.Context(components=work, source=source, trusted=signing.load_trusted_keys(), hardware=hw, cancel=cancel,
                          emit=lambda event: emit(event["message"]) if event.get("event") == "phase" and event.get("message") else None)
    state = updater.State(work)
    try:
        feed = updater.open_feed(ctx, state)
        pkg, reasons = feedmod.select_engine(list(feed.dsv41), hw)
        if pkg is None:
            emit("There is no DeepSeek engine package for this computer in the update feed"
                 + (": " + "; ".join(reasons[:2]) if reasons else "") + ".")
            return None
        archive = updater._download(ctx, "dsv41", pkg.url, pkg.size, pkg.sha256, "the DeepSeek engine", feed)
        stage_root, component = updater.stage_engine(ctx, feed, pkg, archive)
        try:
            folder = stage_root / "engine" / updater.folder_name(pkg)
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / folder.name
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(folder), str(target))
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
        if os.name != "nt":
            (target / exe_name()).chmod(0o755)
        for old in dest.iterdir():                              # one generation: the older folders of this engine go
            if old.is_dir() and old.name != target.name:
                shutil.rmtree(old, ignore_errors=True)
        emit(f"DeepSeek engine {pkg.version} installed: {target}")
        shutil.rmtree(work, ignore_errors=True)               # (on a failure the partial download stays in `work`: the next try resumes it)
        return target / exe_name()
    except (updater.UpdateError, feedmod.FeedError, signing.SignatureError) as e:
        emit(f"The DeepSeek engine could not be fetched: {e}")
        return None
