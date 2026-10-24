#!/usr/bin/env python3
"""The release-side tool for the update feed: keys, signed packages, archives, the signed feed.

Nothing here runs on a user's PC. It runs on the owner's release machine, with the owner's private key, and writes files
that are then uploaded (to a local folder or a local HTTP server while testing; to the real host only when the owner
decides). Nothing is uploaded by this tool.

Typical use (the owner's offline machine; replace ``KEYDIR`` with a folder that is NOT in a git checkout):

    python packaging/feed_tool.py keygen --out KEYDIR                      # once: feed-key.json (SECRET) + feed-key.pub.json
    python packaging/feed_tool.py sign-package <components>/engine/0.1.46-linux-x86_64-cuda --key KEYDIR/feed-key.json
    python packaging/feed_tool.py pack-engine  <that folder> --out RELEASE/engine           # archive + <name>.entry.json
    python packaging/feed_tool.py pack-control <control bundle folder> --version 0.1.46 --platform linux-x86_64 --out RELEASE/control
    python packaging/feed_tool.py build-feed --out RELEASE --sequence 12 --key KEYDIR/feed-key.json \
        --engine RELEASE/engine --control RELEASE/control [--app-json tauri-latest.json] [--channel stable] [--expires-days 365]
    python packaging/feed_tool.py verify-feed RELEASE --pub KEYDIR/feed-key.pub.json

``keygen`` never prints the secret and refuses to write inside a git checkout or over an existing file. The public file's
entry goes into ``strata_control/update_keys.json`` (``keys``) once, in the commit that ships the first signed release.
The Tauri updater key is a different key (minisign, made by ``tauri signer generate``); its public half goes in
``tauri.conf.json`` and in ``update_keys.json`` as ``app_minisign_public``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata_control import engine_component as ec, feed as feedmod, signing  # noqa: E402


def _inside_git(path: Path) -> bool:
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / ".git").exists():
            return True
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_key(path: Path) -> dict:
    key = json.loads(path.read_text(encoding="utf-8"))
    if key.get("alg") != signing.ALG or "seed" not in key:
        raise SystemExit(f"{path} is not a secret key file made by `feed_tool.py keygen`")
    return key


def keygen(out: Path, *, allow_git: bool = False) -> dict:
    out = Path(out)
    if _inside_git(out) and not allow_git:
        raise SystemExit(f"{out} is inside a git checkout; a private key must not be where it can be committed. Choose another folder.")
    secret, public = out / "feed-key.json", out / "feed-key.pub.json"
    if secret.exists() or public.exists():
        raise SystemExit(f"{secret} or {public} already exists; refusing to overwrite a key")
    out.mkdir(parents=True, exist_ok=True)
    key = signing.generate_key()
    fd = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(key, stream, indent=1)
        stream.write("\n")
    from strata_control import private_files
    private_files.restrict_to_current_user(secret, directory=False)          # 0600, or a protected ACL on Windows
    public.write_text(json.dumps({"keys": [signing.public_part(key)]}, indent=1) + "\n", encoding="utf-8")
    return signing.public_part(key)


def sign_package(folder: Path, key: dict) -> str:
    """Put the signature into the package's manifest. Returns the manifest's SHA-256 afterwards (the feed names it)."""
    path = Path(folder) / ec.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    manifest["signature"] = None
    payload = ec.signing_payload(manifest)
    manifest["signature"] = signing.sign_bytes(key, payload)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    ec.read_manifest(Path(folder))                      # still a valid manifest
    signing.verify_bytes({key["key_id"]: __import__("base64").b64decode(key["public"])}, payload, manifest["signature"])
    return _sha256(path)


_NATIVE_MAGIC = (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf", b"#!")      # ELF, Mach-O (64-bit, fat), script


def _is_native_executable(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            head = stream.read(4)
    except OSError:
        return False
    return any(head.startswith(m) for m in _NATIVE_MAGIC)


def _mode(path: Path, rel: str, entrypoint: str | None) -> int:
    if rel == entrypoint:
        return 0o755
    try:
        if os.name == "nt":                       # a Linux / Mac package packed on Windows has no execute bits: take them from the file's own header
            return 0o755 if _is_native_executable(path) else 0o644
        return 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
    except OSError:
        return 0o644


def _write_archive(files: list[tuple[str, Path]], archive: Path, fmt: str, entrypoint: str | None) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "zip":
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel, path in files:
                zf.write(path, rel)
        return
    mode = {"tar.gz": "w:gz", "tar.xz": "w:xz"}[fmt]
    with tarfile.open(archive, mode) as tar:
        for rel, path in files:
            info = tar.gettarinfo(str(path), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if info.issym():                              # a Control bundle's Python has relative symlinks; engines have none
                tar.addfile(info)
                continue
            info.mode = _mode(path, rel, entrypoint)
            with path.open("rb") as stream:
                tar.addfile(info, stream)


def _tree(folder: Path, *, links: bool = False) -> list[tuple[str, Path]]:
    """Regular files under `folder`; with `links` also the symbolic links, which must be relative and stay inside it."""
    found = []
    root = folder.resolve()
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            if not links:
                continue
            target = os.readlink(path)
            if os.path.isabs(target) or not (path.parent / target).resolve().is_relative_to(root):
                raise SystemExit(f"{path} is a symbolic link that leaves the package ({target}); a package may only link inside itself")
            found.append((path.relative_to(folder).as_posix(), path))
        elif path.is_file():
            found.append((path.relative_to(folder).as_posix(), path))
    return found


def pack_engine(folder: Path, out: Path, fmt: str = "tar.gz", *, url_prefix: str = "engine/", prefix: str = "strata-engine") -> dict:
    """Archive a (signed) engine package folder and return its feed entry. The archive holds the folder's files at its root."""
    folder = Path(folder)
    component = ec.read_manifest(folder)
    files = _tree(folder)
    installed = sum(p.stat().st_size for _, p in files)
    name = f"{prefix}-{folder.name}.{fmt}"        # prefix strata-dsv41 for the separate DeepSeek engine packages
    archive = Path(out) / name
    _write_archive(files, archive, fmt, component.entrypoint)
    entry = {"version": component.version, "platform": component.platform, "backend": component.backend,
             "variant": component.variant, "archs": list(component.archs), "min_driver": dict(component.min_driver),
             "min_runtime": component.min_runtime, "build_id": component.build_id, "format": fmt,
             "url": f"{url_prefix}{name}", "size": archive.stat().st_size, "sha256": _sha256(archive),
             "manifest_sha256": _sha256(folder / ec.MANIFEST), "installed_bytes": installed, "notes": component.notes}
    (Path(out) / (name + ".entry.json")).write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
    return entry


def pack_control(folder: Path, out: Path, *, version: str, platform: str, fmt: str = "tar.gz",
                 entry_file: str = "strata.py", url_prefix: str = "control/") -> dict:
    folder = Path(folder)
    files = _tree(folder, links=True)
    installed = sum(p.stat().st_size for _, p in files if not p.is_symlink())
    name = f"strata-control-{version}-{platform}.{fmt}"
    archive = Path(out) / name
    _write_archive(files, archive, fmt, entry_file)
    entry = {"platform": platform, "format": fmt, "url": f"{url_prefix}{name}", "size": archive.stat().st_size,
             "sha256": _sha256(archive), "installed_bytes": installed, "entry": entry_file, "version": version}
    (Path(out) / (name + ".entry.json")).write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
    return entry


def build_feed(out: Path, *, key: dict, sequence: int, channel: str = "stable", engine_dir: Path | None = None,
               control_dir: Path | None = None, app_json: Path | None = None, expires_days: int | None = None,
               now: dt.datetime | None = None, dsv41_dir: Path | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    install_manifest = ""
    document: dict = {"schema_version": feedmod.SCHEMA_VERSION, "feed": "strata-updates", "channel": channel,
                      "sequence": sequence, "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "expires_at": (now + dt.timedelta(days=expires_days)).strftime("%Y-%m-%dT%H:%M:%SZ") if expires_days else None}
    if engine_dir:
        document["engine"] = {"packages": [json.loads(p.read_text(encoding="utf-8"))
                                           for p in sorted(Path(engine_dir).glob("*.entry.json"))]}
    if dsv41_dir:
        document["dsv41"] = {"packages": [json.loads(p.read_text(encoding="utf-8"))
                                          for p in sorted(Path(dsv41_dir).glob("*.entry.json"))]}
    if control_dir:
        entries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(control_dir).glob("*.entry.json"))]
        versions = {e["version"] for e in entries}
        if len(versions) > 1:
            raise SystemExit(f"the control packages are for different versions: {sorted(versions)}")
        if entries:
            document["control"] = {"version": versions.pop(), "packages": [{k: v for k, v in e.items() if k != "version"} for e in entries]}
            for e in entries:     # install.sh unpacks with tar -xzf, install.ps1 with Expand-Archive: a pack in the other format cannot be installed that way
                want = "zip" if e["platform"].startswith("windows") else "tar.gz"
                if e.get("format") != want:
                    print(f"note: the {e['platform']} control package is {e.get('format')}, so it is left out of install-manifest.txt "
                          f"(the one-line installer needs {want}; pack-control --format {want})", file=sys.stderr)
                    continue
                install_manifest += f"{e['platform']} {e['version']} {e['url']} {e['sha256']} {e['size']}\n"
    if app_json:
        document["app"] = json.loads(Path(app_json).read_text(encoding="utf-8"))
    payload = (json.dumps(document, indent=1, sort_keys=True) + "\n").encode("utf-8")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / feedmod.FEED_NAME).write_bytes(payload)
    signature = signing.sign_bytes(key, payload)
    (Path(out) / (feedmod.FEED_NAME + feedmod.SIGNATURE_SUFFIX)).write_text(json.dumps(signature) + "\n", encoding="utf-8")
    if install_manifest:                        # what install.sh / install.ps1 read: platform version url sha256 size, one line per Terminal (Control) package
        (Path(out) / "install-manifest.txt").write_text(install_manifest, encoding="utf-8", newline="\n")
    if "app" in document:                       # the file the Tauri updater plugin reads
        latest = Path(out) / "app" / "latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps({k: v for k, v in document["app"].items()}, indent=1) + "\n", encoding="utf-8")
    # a parse of what was written is the last check: a feed this tool makes must be a feed Control accepts
    feedmod.verify_and_parse(payload, json.dumps(signature).encode(), source=str(Path(out) / feedmod.FEED_NAME),
                             trusted={key["key_id"]: __import__("base64").b64decode(key["public"])}, now=now)
    return document


TAURI_TARGET_NAMES = tuple(feedmod.TAURI_TARGETS.values()) + ("linux-aarch64",)


def _default_app_pubkey() -> str:
    conf = Path(__file__).resolve().parents[1] / "apps" / "desktop" / "src-tauri" / "tauri.conf.json"
    return json.loads(conf.read_text(encoding="utf-8"))["plugins"]["updater"]["pubkey"]


def _check_mac_archive(archive: Path) -> None:
    """A Mac updater archive is a gzip tar with exactly one top-level ``<Name>.app`` folder (what Tauri's bundler makes)."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tops = {m.name.lstrip("./").split("/", 1)[0] for m in tar.getmembers() if m.name.lstrip("./")}
            names = set(tar.getnames())
    except (tarfile.TarError, OSError) as exc:
        raise SystemExit(f"{archive.name} is not a readable .tar.gz ({exc})")
    if len(tops) != 1 or not next(iter(tops)).endswith(".app"):
        raise SystemExit(f"{archive.name} must hold one <Name>.app folder at its root; found {sorted(tops)[:5]}")
    app = next(iter(tops))
    if not any(n.lstrip("./") == f"{app}/Contents/Info.plist" for n in names):
        raise SystemExit(f"{archive.name} has no {app}/Contents/Info.plist")


def set_app_platform(app_json: Path, *, target: str, archive: Path, signature_file: Path, url: str, version: str | None = None,
                     notes: str | None = None, pub_date: str | None = None, pubkey: str | None = None,
                     now: dt.datetime | None = None) -> dict:
    """Add or replace one platform in the Tauri updater JSON (the file ``build-feed --app-json`` takes; it becomes ``latest.json``).

    The ``.sig`` is made by ``npx tauri signer sign`` (or by ``tauri build`` for the Windows / Linux installers) and is only READ here;
    this tool never touches the Tauri private key. It is checked against the public key in ``tauri.conf.json`` (or ``pubkey``) over the
    exact archive bytes, so a signature of an older or unsigned file, or one from another key, is refused before it can be published.
    ``sha256`` and ``size`` are taken from the archive."""
    if target not in TAURI_TARGET_NAMES:
        raise SystemExit(f"unknown target {target!r}; use one of {', '.join(TAURI_TARGET_NAMES)}")
    archive, signature_file = Path(archive), Path(signature_file)
    if target.startswith("darwin-"):
        _check_mac_archive(archive)
    signature = signature_file.read_text(encoding="utf-8").strip()
    try:
        with archive.open("rb") as stream:
            signing.minisign_verify(pubkey or _default_app_pubkey(), signature, iter(lambda: stream.read(1 << 20), b""))
    except signing.SignatureError as exc:
        raise SystemExit(f"{signature_file.name} does not verify for {archive.name}: {exc}")
    document: dict = {}
    if Path(app_json).exists():
        document = json.loads(Path(app_json).read_text(encoding="utf-8"))
    if version:
        if document.get("version") not in (None, version):
            raise SystemExit(f"{app_json} is for version {document['version']}, not {version}")
        document["version"] = version
    if not document.get("version"):
        raise SystemExit("a new app json needs --version")
    document.setdefault("notes", notes or "")
    if notes:
        document["notes"] = notes
    document["pub_date"] = pub_date or document.get("pub_date") or (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    document.setdefault("platforms", {})[target] = {"url": url, "signature": signature, "sha256": _sha256(archive),
                                                    "size": archive.stat().st_size}
    Path(app_json).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return document["platforms"][target]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("sign-package")
    sp.add_argument("folder", type=Path)
    sp.add_argument("--key", type=Path, required=True)
    pe = sub.add_parser("pack-engine")
    pe.add_argument("folder", type=Path)
    pe.add_argument("--out", type=Path, required=True)
    pe.add_argument("--format", choices=feedmod.FORMATS, default="tar.gz")
    pe.add_argument("--url-prefix", default="engine/", help='where the archive sits relative to the feed ("" for flat GitHub release assets)')
    pe.add_argument("--prefix", default="strata-engine", help="archive name prefix (strata-dsv41 for the DeepSeek engine packages)")
    pc = sub.add_parser("pack-control")
    pc.add_argument("folder", type=Path)
    pc.add_argument("--out", type=Path, required=True)
    pc.add_argument("--version", required=True)
    pc.add_argument("--platform", required=True, choices=feedmod.PLATFORMS)
    pc.add_argument("--format", choices=feedmod.FORMATS, default="tar.gz")
    pc.add_argument("--url-prefix", default="control/")
    bf = sub.add_parser("build-feed")
    bf.add_argument("--out", type=Path, required=True)
    bf.add_argument("--key", type=Path, required=True)
    bf.add_argument("--sequence", type=int, required=True)
    bf.add_argument("--channel", default="stable")
    bf.add_argument("--engine", type=Path)
    bf.add_argument("--dsv41", type=Path, help="folder of the DeepSeek engine packages' .entry.json files (feed member dsv41)")
    bf.add_argument("--control", type=Path)
    bf.add_argument("--app-json", type=Path)
    bf.add_argument("--expires-days", type=int)
    sa = sub.add_parser("set-app-platform", help="add one signed app installer to the Tauri updater json")
    sa.add_argument("--app-json", type=Path, required=True, help="the file to create or patch (what build-feed --app-json takes)")
    sa.add_argument("--target", required=True, help="darwin-aarch64, darwin-x86_64, windows-x86_64 or linux-x86_64")
    sa.add_argument("--archive", type=Path, required=True)
    sa.add_argument("--sig", type=Path, required=True, help="the .sig file made by tauri signer sign")
    sa.add_argument("--url", help="where the archive is published (default: the latest GitHub release)")
    sa.add_argument("--version")
    sa.add_argument("--notes")
    sa.add_argument("--pub-date")
    sa.add_argument("--pubkey", help="the Tauri public key (default: tauri.conf.json plugins.updater.pubkey)")
    vf = sub.add_parser("verify-feed")
    vf.add_argument("source")
    vf.add_argument("--pub", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "keygen":
        public = keygen(a.out)
        print(f"wrote {a.out / 'feed-key.json'} (SECRET, keep it offline and backed up) and {a.out / 'feed-key.pub.json'}")
        print(f"key_id {public['key_id']}")
    elif a.cmd == "sign-package":
        print("manifest sha256", sign_package(a.folder, _load_key(a.key)))
    elif a.cmd == "pack-engine":
        entry = pack_engine(a.folder, a.out, a.format, url_prefix=a.url_prefix, prefix=a.prefix)
        print(json.dumps({k: entry[k] for k in ("version", "backend", "variant", "size", "sha256", "manifest_sha256")}, indent=1))
    elif a.cmd == "pack-control":
        entry = pack_control(a.folder, a.out, version=a.version, platform=a.platform, fmt=a.format, url_prefix=a.url_prefix)
        print(json.dumps({k: entry[k] for k in ("version", "platform", "size", "sha256")}, indent=1))
    elif a.cmd == "build-feed":
        doc = build_feed(a.out, key=_load_key(a.key), sequence=a.sequence, channel=a.channel, engine_dir=a.engine,
                         control_dir=a.control, app_json=a.app_json, expires_days=a.expires_days, dsv41_dir=a.dsv41)
        print(f"wrote {a.out / feedmod.FEED_NAME} (+ .sig); engines {len((doc.get('engine') or {}).get('packages', []))}, "
              f"deepseek engines {len((doc.get('dsv41') or {}).get('packages', []))}, control {'yes' if 'control' in doc else 'no'}, app {'yes' if 'app' in doc else 'no'}")
    elif a.cmd == "set-app-platform":
        url = a.url or f"https://github.com/Niko1221/Strata/releases/latest/download/{a.archive.name}"
        entry = set_app_platform(a.app_json, target=a.target, archive=a.archive, signature_file=a.sig, url=url, version=a.version,
                                 notes=a.notes, pub_date=a.pub_date, pubkey=a.pubkey)
        print(f"{a.app_json}: {a.target} -> {entry['url']} ({entry['size']} bytes, sha256 {entry['sha256'][:16]}..., signature verified)")
    elif a.cmd == "verify-feed":
        trusted = signing.load_trusted_keys(a.pub, environ={})
        loaded = feedmod.fetch(a.source, trusted)
        print(f"ok: sequence {loaded.sequence}, channel {loaded.channel}, signed by {loaded.key_id}, "
              f"{len(loaded.engine)} engine packages, control {len(loaded.control)}, app {'yes' if loaded.app else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
