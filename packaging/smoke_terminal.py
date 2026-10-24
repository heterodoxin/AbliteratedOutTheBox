#!/usr/bin/env python3
"""Smoke-test a terminal archive in a clean environment.

    python packaging/smoke_terminal.py release/artifacts/strata-terminal-0.1.45-windows-x86_64.zip [--json out.json]

The archive is extracted into a fresh folder whose name has a space in it (so a moved or awkward path is part of
the test), and every command runs with a PATH that holds no Python (Windows: System32 only; POSIX: a folder with
dirname/readlink/cat and nothing else), no PYTHON* variables, a private user profile, and a dead HTTP proxy so that
anything reaching for the internet fails loudly. It then checks, through the bundled launcher:

    strata --version              must name the interpreter inside the extracted folder
    strata doctor                 read-only inventory
    strata profile list           saved profiles (none in a clean profile)
    strata serve --engine mock    the server on the scripted engine: /health, /v1/models and a chat answer
    strata install               must refuse (no installer in the archive) with exit 5, not fetch anything

The script is one foreground run: the mock server it starts is a child that is stopped (whole tree) before it exits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def extract(archive: Path, dest: Path) -> Path:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    roots = [p for p in dest.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise SystemExit(f"expected one folder in the archive, found {[p.name for p in roots]}")
    return roots[0]


def clean_env(sandbox: Path) -> tuple[dict, list[str]]:
    home = sandbox / "home"
    home.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        system32 = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")
        env = {"PATH": system32, "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
               "SystemDrive": os.environ.get("SystemDrive", "C:"), "COMSPEC": os.environ.get("COMSPEC", ""),
               "TEMP": str(sandbox / "tmp"), "TMP": str(sandbox / "tmp"), "USERPROFILE": str(home),
               "APPDATA": str(home / "AppData" / "Roaming"), "LOCALAPPDATA": str(home / "AppData" / "Local"),
               "HOMEDRIVE": "C:", "HOMEPATH": str(home)[2:], "PATHEXT": ".COM;.EXE;.BAT;.CMD"}
        shell = ["cmd", "/c"]
    else:
        minimal = sandbox / "minbin"
        minimal.mkdir(parents=True, exist_ok=True)
        for tool in ("dirname", "readlink", "cat", "uname", "basename"):
            found = shutil.which(tool)
            if found and not (minimal / tool).exists():
                (minimal / tool).symlink_to(found)
        env = {"PATH": str(minimal), "HOME": str(home), "TMPDIR": str(sandbox / "tmp"),
               "XDG_CONFIG_HOME": str(home / ".config"), "LANG": "C.UTF-8"}
        shell = []
    (sandbox / "tmp").mkdir(parents=True, exist_ok=True)
    # anything that reaches for the internet fails loudly; the loopback server is unaffected
    env.update({"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "127.0.0.1,localhost", "PIP_NO_INDEX": "1"})
    return env, shell


def launcher(root: Path) -> list[str]:
    return [str(root / "bin" / "strata.cmd")] if os.name == "nt" else [str(root / "bin" / "strata")]


def as_command(args: list[str]):
    """What to hand to subprocess. A .cmd launcher in a folder with a space needs cmd's /s rule (strip only the outer
    pair of quotes), the way a user's shell does it."""
    if os.name != "nt":
        return args
    return 'cmd.exe /d /s /c "' + subprocess.list2cmdline(args) + '"'


def run(cmd: list[str], env: dict, cwd: Path, timeout: int = 120) -> dict:
    t = time.time()
    r = subprocess.run(as_command(cmd), env=env, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return {"cmd": " ".join(cmd), "rc": r.returncode, "seconds": round(time.time() - t, 2),
            "stdout": r.stdout, "stderr": r.stderr}


def inside_archive(root: Path, text: str) -> bool:
    """The interpreter the launcher reported is the one in the extracted folder (8.3 and long forms of a path both)."""
    tail = (root.name + os.sep + ("python" + os.sep + "python.exe" if os.name == "nt" else "python/bin/python3")).lower()
    return tail in text.lower().replace("/", os.sep)


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def http(url: str, body: dict | None = None, timeout: float = 20) -> tuple[int, str]:
    # the dead proxy is for the child processes; this client talks to loopback directly
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("--json", help="write the full results here")
    ap.add_argument("--keep", action="store_true", help="keep the extracted folder")
    a = ap.parse_args()
    archive = Path(a.archive).resolve()
    sandbox = Path(tempfile.mkdtemp(prefix="strata smoke "))
    results: dict = {"archive": archive.name, "sandbox_has_space": " " in str(sandbox), "checks": {}}
    ok = True
    proc = None
    try:
        root = extract(archive, sandbox / "extract")
        env, _ = clean_env(sandbox)
        results["root"] = str(root)
        strata = launcher(root)

        # no other Python can be found from this environment
        if os.name == "nt":
            found = {"rc": 1, "stdout": "", "stderr": ""}
            for name in ("python", "py", "python3"):
                w = subprocess.run(["where", name], env=env, capture_output=True, text=True)
                if w.returncode == 0 and w.stdout.strip():
                    found = {"rc": 0, "stdout": w.stdout, "stderr": ""}
        else:
            found = {"rc": 1, "stdout": "", "stderr": ""}
            for name in ("python", "python3", "py"):
                if shutil.which(name, path=env["PATH"]):
                    found = {"rc": 0, "stdout": name, "stderr": ""}
        results["other_python_on_path"] = found["rc"] == 0 and bool(found["stdout"].strip())
        ok &= not results["other_python_on_path"]

        def check(name: str, cond: bool, **info) -> None:
            nonlocal ok
            results["checks"][name] = {"ok": bool(cond), **info}
            ok &= bool(cond)
            print(("PASS " if cond else "FAIL ") + name)

        r = run([*strata, "--version"], env, root)
        inside = inside_archive(root, r["stdout"])
        check("version", r["rc"] == 0 and r["stdout"].startswith("strata ") and inside,
              rc=r["rc"], output=r["stdout"].strip(), interpreter_inside_archive=inside)

        if os.name == "nt":
            ps = Path(env["SystemRoot"]) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
            r = run([str(ps), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "bin" / "strata.ps1"),
                     "--version"], env, root)
            check("version (strata.ps1)", r["rc"] == 0 and inside_archive(root, r["stdout"]),
                  rc=r["rc"], output=r["stdout"].strip(), stderr=r["stderr"].strip()[:200])

        r = run([*strata, "doctor"], env, root)
        check("doctor", r["rc"] == 0, rc=r["rc"], output=r["stdout"].strip()[:600])

        r = run([*strata, "profile", "list"], env, root)
        check("profile list", r["rc"] == 0, rc=r["rc"], output=r["stdout"].strip()[:300])

        r = run([*strata, "install"], env, root)
        check("install refuses without an installer", r["rc"] == 5 and "no installer" in r["stderr"],
              rc=r["rc"], stderr=r["stderr"].strip()[:300])

        # serve on the mock engine: a fake run config (the mock never loads it), a free port, a chat answer
        cfg_dir = sandbox / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / "engine.exe").write_bytes(b"not an engine")
        (cfg_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        cfg = cfg_dir / "strata-mock.json"
        cfg.write_text(json.dumps({"exe": str(cfg_dir / "engine.exe"), "tokenizer": str(cfg_dir / "tokenizer.json"),
                                   "args": [], "model_name": "mock"}), encoding="utf-8")
        port = free_port()
        log = sandbox / "serve.log"
        with log.open("wb") as lf:
            proc = subprocess.Popen(as_command([*strata, "serve", "--profile", str(cfg), "--engine", "mock",
                                                "--port", str(port), "--host", "127.0.0.1"]),
                                    env=env, cwd=str(root), stdout=lf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)
        health = None
        deadline = time.time() + 60
        while time.time() < deadline and proc.poll() is None:
            try:
                status, body = http(f"http://127.0.0.1:{port}/health", timeout=3)
                if status == 200:
                    health = body
                    break
            except OSError:
                time.sleep(0.5)
        check("serve: /health", health is not None, port=port, body=(health or "")[:200],
              exited=proc.poll(), log=log.read_text(errors="replace")[-600:] if health is None else "")
        if health is not None:
            status, models = http(f"http://127.0.0.1:{port}/v1/models")
            check("serve: /v1/models", status == 200, body=models[:200])
            status, chat = http(f"http://127.0.0.1:{port}/v1/chat/completions",
                                {"model": "mock", "messages": [{"role": "user", "content": "hello"}],
                                 "max_tokens": 16})
            text = ""
            try:
                msg = json.loads(chat)["choices"][0]["message"]
                text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
            except (ValueError, KeyError, IndexError, TypeError):
                pass
            check("serve: chat answer", status == 200 and bool(text.strip()), answer=text[:200], body=chat[:400])
    finally:
        if proc is not None:
            kill_tree(proc)
        if not a.keep:
            shutil.rmtree(sandbox, ignore_errors=True)
    results["ok"] = ok
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("SMOKE " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
