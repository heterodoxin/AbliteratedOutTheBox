#!/usr/bin/env python3
"""Build and manage the thin, read-only Strata monitor zipapp.

The bundle contains no engine, model, installer or terminal framework. One-shot
JSON status uses stdlib only; interactive mode needs the explicit Textual add-on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipapp
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES_V2 = (
    "strata_cli/__init__.py",
    "strata_cli/client.py",
    "strata_cli/connections.py",
    "strata_cli/render.py",
    "strata_cli/monitor.py",
    "strata_cli/monitor_commands.py",
    "strata_cli/logs.py",
)
SOURCES_V3 = (*SOURCES_V2, "strata_cli/dashboard.py")
SOURCES_V4 = (*SOURCES_V3, "strata_cli/detail_dashboard.py")
SOURCES_V5 = (*SOURCES_V4, "strata_cli/nav.py")      # schema 5: the view tabs every screen shares (strata_cli/nav.py)
# schema 6: the two modules the interactive Monitor imports at run time (strata_cli/monitor.py, render.py) that earlier lists missed.
# tests/terminal/test_monitor_bundle.py derives the import closure of the bundle's entry point and fails if this list falls behind.
SOURCES = (*SOURCES_V5, "strata_cli/consoleinfo.py", "strata_cli/overview.py")
BOOTSTRAP = b"from strata_cli.monitor_commands import main\nraise SystemExit(main())\n"
MEMBERS_V1 = ("__main__.py", *SOURCES_V2)
LICENSE_NOTICE = "LICENSE"
MEMBERS_V2 = (*MEMBERS_V1, LICENSE_NOTICE)
MEMBERS_V3 = (*MEMBERS_V2, "strata_cli/dashboard.py")
MEMBERS_V4 = (*MEMBERS_V3, "strata_cli/detail_dashboard.py")
MEMBERS_V5 = (*MEMBERS_V4, "strata_cli/nav.py")
MEMBERS = (*MEMBERS_V5, "strata_cli/consoleinfo.py", "strata_cli/overview.py")
COMPATIBILITY = {"python_min": "3.10", "telemetry_schema": [1]}
MAX_BUNDLE_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_MEMBER_BYTES = 256 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_path(bundle: Path) -> Path:
    return bundle.with_name(bundle.name + ".manifest.json")


def build(bundle: Path) -> dict:
    if bundle.exists() or manifest_path(bundle).exists():
        raise FileExistsError("bundle or manifest already exists; choose a new output path")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".strata-monitor-build-", dir=bundle.parent) as directory:
        stage = Path(directory) / "source"
        stage.mkdir()
        (stage / "__main__.py").write_bytes(BOOTSTRAP)
        for name in SOURCES:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copyfile(ROOT / LICENSE_NOTICE, stage / LICENSE_NOTICE)
        temp_bundle = Path(directory) / bundle.name
        zipapp.create_archive(stage, target=temp_bundle, interpreter="/usr/bin/env python3",
                              compressed=True)
        with zipfile.ZipFile(temp_bundle) as archive:
            names = archive.namelist()
            if set(names) != set((*MEMBERS, "strata_cli/")) or len(names) != len(MEMBERS) + 1:
                raise ValueError("bundle member list changed during build")
            files = {name: digest(archive.read(name)) for name in MEMBERS}
        report = {"schema_version": 6, "kind": "strata-monitor-bundle",
                  "bundle": bundle.name, "sha256": digest(temp_bundle.read_bytes()),
                  "files": files, "interactive_dependency": "textual==8.2.8",
                  "compatibility": COMPATIBILITY, "license": "MIT", "license_file": LICENSE_NOTICE,
                  "publisher_signature": None}
        temp_manifest = Path(directory) / manifest_path(bundle).name
        temp_manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temp_bundle, bundle)
        try:
            os.replace(temp_manifest, manifest_path(bundle))
        except OSError:
            bundle.unlink(missing_ok=True)
            raise
    return report


def verify(bundle: Path) -> dict:
    for path, limit in ((bundle, MAX_BUNDLE_BYTES), (manifest_path(bundle), MAX_MANIFEST_BYTES)):
        if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > limit:
            raise ValueError("bundle and manifest must be bounded regular files")
    report = json.loads(manifest_path(bundle).read_text())
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int or
            report["schema_version"] not in (1, 2, 3, 4, 5, 6) or
            report.get("kind") != "strata-monitor-bundle" or
            report.get("publisher_signature") is not None or
            report.get("bundle") != bundle.name or report.get("sha256") != digest(bundle.read_bytes())):
        raise ValueError("bundle manifest or SHA-256 mismatch")
    members = {1: MEMBERS_V1, 2: MEMBERS_V2, 3: MEMBERS_V3, 4: MEMBERS_V4,
               5: MEMBERS_V5, 6: MEMBERS}[report["schema_version"]]
    if report["schema_version"] >= 2:
        if (report.get("compatibility") != COMPATIBILITY or report.get("license") != "MIT" or
                report.get("license_file") != LICENSE_NOTICE):
            raise ValueError("bundle compatibility or license notice mismatch")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(members) + 1 or set(names) != set((*members, "strata_cli/")):
            raise ValueError("bundle contains missing, duplicate or unexpected files")
        if any(info.flag_bits & 1 for info in archive.infolist()):
            raise ValueError("encrypted bundle members are unsupported")
        if any(info.file_size > MAX_MEMBER_BYTES for info in archive.infolist()):
            raise ValueError("bundle member exceeds the size limit")
        if archive.testzip() is not None:
            raise ValueError("bundle CRC failure")
        files = {name: digest(archive.read(name)) for name in members}
        if report["schema_version"] >= 2 and not archive.read(LICENSE_NOTICE).replace(b"\r\n", b"\n").startswith(b"MIT License\n"):
            raise ValueError("bundle license notice is invalid")
    if report.get("files") != files:
        raise ValueError("bundle member checksum mismatch")
    return report


def _target(target: Path) -> Path:
    target = target.expanduser()
    if not target.is_absolute() or target.name in ("", ".", ".."):
        raise ValueError("install target must be an absolute new directory")
    parent = target.parent
    parent_info = parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or \
            (os.name == "posix" and (parent_info.st_uid != os.getuid() or parent_info.st_mode & 0o022)):
        raise ValueError("install target parent must be an existing owner-controlled directory")
    return target


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _previous(target: Path) -> Path:
    return target.with_name(target.name + ".previous")


def _installed(target: Path) -> tuple[Path, dict]:
    if not stat.S_ISDIR(target.lstat().st_mode):
        raise ValueError("install target must be a regular directory")
    candidates = [path for path in target.iterdir() if path.name.endswith(".pyz")]
    if len(candidates) != 1:
        raise ValueError("install target has no unique monitor bundle; inspect it manually")
    bundle = candidates[0]
    expected = {bundle.name, manifest_path(bundle).name}
    if {path.name for path in target.iterdir()} != expected:
        raise ValueError("install target has unexpected or missing files; inspect it manually")
    return bundle, verify(bundle)


def _stage_copy(bundle: Path, report: dict, target: Path) -> Path:
    stage = Path(tempfile.mkdtemp(prefix=".strata-monitor-install-", dir=target.parent))
    try:
        staged_bundle = stage / bundle.name
        shutil.copyfile(bundle, staged_bundle, follow_symlinks=False)
        shutil.copyfile(manifest_path(bundle), manifest_path(staged_bundle), follow_symlinks=False)
        os.chmod(staged_bundle, 0o600)
        os.chmod(manifest_path(staged_bundle), 0o600)
        if verify(staged_bundle)["sha256"] != report["sha256"]:
            raise ValueError("bundle changed during copy")
        return stage
    except Exception:
        shutil.rmtree(stage)
        raise


def _remove_install(bundle: Path) -> None:
    manifest_path(bundle).unlink()
    bundle.unlink()
    bundle.parent.rmdir()


def install(bundle: Path, target: Path, *, plan: bool = False,
            if_candidate_sha256: str | None = None) -> dict:
    """Copy a verified client into a new directory without touching the server."""
    report = verify(bundle)
    target = _target(target)
    if _present(target) or _present(_previous(target)):
        raise FileExistsError("install target already exists; choose a new directory")
    result = {"target": str(target), "sha256": report["sha256"],
              "bundle": bundle.name, "would_install": plan}
    if plan:
        return result
    if if_candidate_sha256 != report["sha256"]:
        raise ValueError("monitor install digest changed or is missing; preview again")
    stage = _stage_copy(bundle, report, target)
    try:
        if _present(target) or _present(_previous(target)):
            raise FileExistsError("install target appeared during copy; inspect it manually")
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {**result, "installed": True, "would_install": False}


def update(bundle: Path, target: Path, *, plan: bool = False,
           if_current_sha256: str | None = None, if_candidate_sha256: str | None = None) -> dict:
    """Promote a verified candidate and retain one verified rollback copy."""
    target = _target(target)
    current_bundle, current = _installed(target)
    candidate = verify(bundle)
    previous = _previous(target)
    if _present(previous):
        raise FileExistsError("previous monitor version exists; rollback or commit it before another update")
    changed = candidate["sha256"] != current["sha256"]
    result = {"target": str(target), "previous": str(previous),
              "current_sha256": current["sha256"], "candidate_sha256": candidate["sha256"],
              "current_bundle": current_bundle.name, "candidate_bundle": bundle.name,
              "would_update": plan and changed}
    if plan:
        return result
    if if_current_sha256 != current["sha256"] or if_candidate_sha256 != candidate["sha256"]:
        raise ValueError("monitor update digests changed or are missing; preview again")
    if not changed:
        return result
    stage = _stage_copy(bundle, candidate, target)
    try:
        if _present(previous):
            raise FileExistsError("previous monitor version appeared during copy")
        if _installed(target)[1]["sha256"] != current["sha256"]:
            raise ValueError("active monitor changed during copy; preview again")
        os.rename(target, previous)
        try:
            os.rename(stage, target)
        except OSError as exc:
            try:
                os.rename(previous, target)
            except OSError:
                raise OSError(f"monitor update interrupted; inspect {previous} and {target}") from exc
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {**result, "updated": True, "would_update": False}


def rollback(target: Path, *, plan: bool = False,
             if_active_sha256: str | None = None, if_rollback_sha256: str | None = None) -> dict:
    """Exchange the active and previous verified clients without deleting either."""
    target = _target(target)
    active_bundle, active = _installed(target)
    previous = _previous(target)
    prior_bundle, prior = _installed(previous)
    result = {"target": str(target), "previous": str(previous),
              "active_sha256": active["sha256"], "rollback_sha256": prior["sha256"],
              "active_bundle": active_bundle.name, "rollback_bundle": prior_bundle.name,
              "would_rollback": plan}
    if plan:
        return result
    if if_active_sha256 != active["sha256"] or if_rollback_sha256 != prior["sha256"]:
        raise ValueError("monitor rollback digests changed or are missing; preview again")
    swap = Path(tempfile.mkdtemp(prefix=".strata-monitor-swap-", dir=target.parent))
    if os.name == "nt":
        # os.rename onto an existing directory fails on Windows (POSIX replaces an empty one):
        # keep only the unique name mkdtemp reserved.
        swap.rmdir()
    try:
        os.rename(target, swap)
        try:
            os.rename(previous, target)
        except OSError as exc:
            try:
                os.rename(swap, target)
            except OSError:
                raise OSError(f"monitor rollback interrupted; inspect {swap} and {previous}") from exc
            raise
        try:
            os.rename(swap, previous)
        except OSError as exc:
            try:
                os.rename(target, previous)
                os.rename(swap, target)
            except OSError:
                raise OSError(f"monitor rollback interrupted; inspect {target}, {previous} and {swap}") from exc
            raise
    finally:
        if swap.is_dir() and not any(swap.iterdir()):
            swap.rmdir()
    return {**result, "rolled_back": True, "would_rollback": False}


def commit(target: Path, *, plan: bool = False,
           if_active_sha256: str | None = None, if_previous_sha256: str | None = None) -> dict:
    """Accept the active client and remove its verified rollback copy."""
    target = _target(target)
    _, active = _installed(target)
    previous = _previous(target)
    old_bundle, old = _installed(previous)
    result = {"target": str(target), "previous": str(previous),
              "active_sha256": active["sha256"], "previous_sha256": old["sha256"],
              "would_commit": plan}
    if plan:
        return result
    if if_active_sha256 != active["sha256"] or if_previous_sha256 != old["sha256"]:
        raise ValueError("monitor commit digests changed or are missing; preview again")
    _remove_install(old_bundle)
    return {**result, "committed": True, "would_commit": False}


def uninstall(target: Path, *, plan: bool = False,
              if_active_sha256: str | None = None) -> dict:
    """Remove only the two verified files installed by this command."""
    target = _target(target)
    bundle, report = _installed(target)
    if _present(_previous(target)):
        raise ValueError("previous monitor version exists; commit or rollback it before uninstall")
    result = {"target": str(target), "sha256": report["sha256"],
              "bundle": bundle.name, "would_remove": plan}
    if plan:
        return result
    if if_active_sha256 != report["sha256"]:
        raise ValueError("monitor uninstall digest changed or is missing; preview again")
    _remove_install(bundle)
    return {**result, "removed": True, "would_remove": False}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build or locally manage a thin Strata monitor bundle")
    sp = ap.add_subparsers(dest="action", required=True)
    sp.add_parser("build").add_argument("bundle", type=Path)
    sp.add_parser("verify").add_argument("bundle", type=Path)
    install_parser = sp.add_parser("install", help="copy a verified bundle into a new local directory")
    install_parser.add_argument("bundle", type=Path)
    install_parser.add_argument("--target", type=Path, required=True)
    install_parser.add_argument("--plan", action="store_true")
    install_parser.add_argument("--if-candidate-sha256")
    update_parser = sp.add_parser("update", help="replace a verified monitor and retain one rollback copy")
    update_parser.add_argument("bundle", type=Path)
    update_parser.add_argument("--target", type=Path, required=True)
    update_parser.add_argument("--plan", action="store_true")
    update_parser.add_argument("--if-current-sha256")
    update_parser.add_argument("--if-candidate-sha256")
    rollback_parser = sp.add_parser("rollback", help="exchange active and previous monitor clients")
    rollback_parser.add_argument("--target", type=Path, required=True)
    rollback_parser.add_argument("--plan", action="store_true")
    rollback_parser.add_argument("--if-active-sha256")
    rollback_parser.add_argument("--if-rollback-sha256")
    commit_parser = sp.add_parser("commit", help="keep active client and discard previous")
    commit_parser.add_argument("--target", type=Path, required=True)
    commit_parser.add_argument("--plan", action="store_true")
    commit_parser.add_argument("--if-active-sha256")
    commit_parser.add_argument("--if-previous-sha256")
    uninstall_parser = sp.add_parser("uninstall", help="remove a verified local monitor installation")
    uninstall_parser.add_argument("--target", type=Path, required=True)
    uninstall_parser.add_argument("--plan", action="store_true")
    uninstall_parser.add_argument("--if-active-sha256")
    args = ap.parse_args(argv)
    try:
        if args.action == "build":
            result = build(args.bundle)
        elif args.action == "verify":
            result = verify(args.bundle)
        elif args.action == "install":
            result = install(args.bundle, args.target, plan=args.plan,
                             if_candidate_sha256=args.if_candidate_sha256)
        elif args.action == "update":
            result = update(args.bundle, args.target, plan=args.plan,
                            if_current_sha256=args.if_current_sha256,
                            if_candidate_sha256=args.if_candidate_sha256)
        elif args.action == "rollback":
            result = rollback(args.target, plan=args.plan,
                              if_active_sha256=args.if_active_sha256,
                              if_rollback_sha256=args.if_rollback_sha256)
        elif args.action == "commit":
            result = commit(args.target, plan=args.plan,
                            if_active_sha256=args.if_active_sha256,
                            if_previous_sha256=args.if_previous_sha256)
        else:
            result = uninstall(args.target, plan=args.plan,
                               if_active_sha256=args.if_active_sha256)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        ap.exit(1, f"monitor bundle {args.action} failed: {exc}\n")
    print(json.dumps({"ok": True, "action": args.action,
                      **({"bundle": str(args.bundle), "files": len(result["files"]),
                          "sha256": result["sha256"]}
                         if args.action in ("build", "verify") else result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
