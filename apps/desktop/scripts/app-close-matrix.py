#!/usr/bin/env python3
"""The installed Windows app, closed every way, with a stub engine: nothing may be left running.

    python apps/desktop/scripts/app-close-matrix.py --install <folder the installer wrote> [--scenarios all] [--report out.json]

For each case it starts the REAL installed `strata-desktop.exe` (its window opens), loads a stub engine through the host's API
(profile + `runtime start` under Control's owner runtime, exactly the app's path), records every process of the tree (the app,
the host, Control, the owner host, the server, the engine), ends the app one way, and asserts:
  * no process of the tree is left, the engine's port is free, the runtime lease can be taken again;
  * the engine got the clean stop where the way of closing allows one (normal close, window X);
  * the next load of the same slot works (a lost owner's state is cleaned, not a blocker).
It also proves a second launch focuses the open window and starts nothing, and that one app runs two engines and closes both.
Everything per-user lives under a scratch folder; the stub engine is a script. No real engine ever runs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

from strata_control import procs, runtime  # noqa: E402
from tests.terminal.lifecycle_matrix import install_fake_engine, port_open  # noqa: E402

if os.name != "nt":
    raise SystemExit("this matrix drives the Windows installer's app")

user32 = ctypes.WinDLL("user32", use_last_error=True)
WM_CLOSE = 0x0010


def main_window(pid: int) -> int | None:
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if buf.value.strip() == "Strata":
                found.append(hwnd)
        return True
    user32.EnumWindows(visit, 0)
    return found[0] if found else None


class Scratch:
    def __init__(self, root: Path, install: Path):
        self.root, self.install = root, install
        self.components = root / "components"
        self.models = root / "models"
        self.mark = root / "mark"
        for d in (self.components, self.models, self.mark, root / "local", root / "home", root / "appdata", root / "webview"):
            d.mkdir(parents=True, exist_ok=True)
        install_fake_engine(self.components)
        from tools.strata_tokenizer import BYTE_TO_UNICODE
        tok = self.models / "tokenizer"
        tok.mkdir()
        tokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        (tok / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}))
        (tok / "merges.txt").write_text("a b")
        (tok / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]))
        for name in ("stub", "stub2"):
            (self.models / f"strata-{name}.json").write_text(json.dumps({
                "exe": "component:strata-engine", "backend": "cuda", "tokenizer": str(tok), "args": [], "cwd": str(root),
                "model_name": name, "host": "127.0.0.1", "port": 0}))
        self.env = {**os.environ, "STRATA_COMPONENTS_DIR": str(self.components), "STRATA_PROFILE_DIR": str(root / "profiles"),
                    "STRATA_RUNTIME_DIR": str(root / "runtime"), "STRATA_INSTALL_DIR": str(self.models),
                    "STRATA_MODELS_DIR": str(root / "models-unused"), "STRATA_ENGINE_BACKEND": "cuda",
                    "STRATA_HOME": str(root / "home"), "APPDATA": str(root / "appdata"), "LOCALAPPDATA": str(root / "local"),
                    "WEBVIEW2_USER_DATA_FOLDER": str(root / "webview"), "FAKE_ENGINE_MARK": str(self.mark),
                    "FAKE_ENGINE_SLOW_STEPS": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        self.env.pop("STRATA_API_KEY", None)
        for key in ("STRATA_COMPONENTS_DIR", "STRATA_PROFILE_DIR", "STRATA_RUNTIME_DIR", "LOCALAPPDATA", "STRATA_ENGINE_BACKEND",
                    "STRATA_NO_ACQUIRE", "FAKE_ENGINE_MARK"):
            os.environ[key] = self.env.get(key, "1")          # this process reads the same runtime and records folders


class App:
    def __init__(self, scratch: Scratch):
        self.s = scratch
        self.proc = subprocess.Popen([str(scratch.install / "strata-desktop.exe")], cwd=scratch.install, env=scratch.env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.base = None

    def host(self):
        for child in psutil.Process(self.proc.pid).children(recursive=True):
            if child.name().lower() == "node.exe":
                return child
        return None

    def wait_up(self, seconds=90):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the app exited with {self.proc.returncode}")
            try:
                host = self.host()
                if host:
                    for c in host.net_connections(kind="tcp"):
                        if c.status == "LISTEN" and c.laddr.ip == "127.0.0.1":
                            base = f"http://127.0.0.1:{c.laddr.port}"
                            try:
                                if urllib.request.urlopen(base + "/api/health", timeout=2).status == 200 and main_window(self.proc.pid):
                                    self.base = base
                                    return
                            except OSError:
                                pass
            except psutil.Error:
                pass
            time.sleep(0.5)
        raise RuntimeError("the app did not come up")

    def call(self, path, method="GET", body=None):
        req = urllib.request.Request(self.base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{method} {path} -> {exc.code} {exc.read()[:300]!r}") from None

    def load(self, model, multiple=False):
        self.call("/api/native?refresh=1")
        started = self.call("/api/native/operations", "POST", {"kind": "load", "args": {"model": model, **({"multiple": True} if multiple else {})}})
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            state = self.call("/api/native")
            job = next((j for j in state["jobs"] if j["id"] == started["id"]), None)
            if job and job["state"] != "running":
                if job["state"] != "done":
                    raise RuntimeError(f"load {model}: {job['state']} {job.get('error')}")
                return state
            time.sleep(0.3)
        raise RuntimeError("load did not finish")


def tree_of(app: App, slots=("app-1",)) -> tuple[list[procs.Ident], list[int]]:
    idents, ports = [], []
    members = [psutil.Process(app.proc.pid), *psutil.Process(app.proc.pid).children(recursive=True)]
    pids = {p.pid for p in members}
    for slot in slots:
        state = runtime._read_state(slot)
        if state:
            pids.add(state["host_pid"])
    for r in procs.records():
        if r.get("profile") in slots:                 # the app's own slots; a Terminal service is not part of the app's tree
            pids.add(r["pid"])
    for pid in pids:
        token = procs.start_token(pid)
        if token:
            idents.append(procs.Ident(pid, token, tuple(procs.command_line(pid) or [])))
    return idents, ports


def survivors(idents):
    return [f"{i.pid} {' '.join(i.cmd)[:80]}" for i in idents if procs.start_token(i.pid) == i.token]


def wait_gone(idents, seconds):
    deadline = time.monotonic() + seconds
    left = survivors(idents)
    while left and time.monotonic() < deadline:
        time.sleep(0.25)
        left = survivors(idents)
    return left


def lease_free() -> bool:
    try:
        fd = runtime.acquire_activity_lease(shared=False)
    except Exception:  # noqa: BLE001
        return False
    runtime._unlock_file(fd)
    os.close(fd)
    return True


def control(scratch: Scratch, *argv: str, timeout: int = 300) -> dict:
    py = scratch.install / "control" / "python" / "python.exe"
    out = subprocess.run([str(py), str(scratch.install / "control" / "strata.py"), *argv, "--output", "json"],
                         env={**scratch.env, "PYTHONNOUSERSITE": "1", "STRATA_NO_ACQUIRE": "1"},
                         capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(out.stdout)
    except ValueError:
        return {"ok": False, "raw": out.stdout[-300:] + out.stderr[-300:]}


def stop_cli(scratch: Scratch, slot: str) -> dict:
    py = scratch.install / "control" / "python" / "python.exe"
    out = subprocess.run([str(py), str(scratch.install / "control" / "strata.py"), "runtime", "stop", "--profile", slot, "--wait", "30",
                          "--output", "json"], env={**scratch.env, "PYTHONNOUSERSITE": "1", "STRATA_NO_ACQUIRE": "1"},
                         capture_output=True, text=True, timeout=120)
    try:
        return json.loads(out.stdout)
    except ValueError:
        return {"state": "?", "raw": out.stdout[-200:] + out.stderr[-200:]}


def case(name: str, install: Path, how: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="strata-app-matrix-", ignore_cleanup_errors=True) as tmp:
        saved = dict(os.environ)
        scratch = Scratch(Path(tmp), install)
        app = App(scratch)
        problems, note = [], ""
        try:
            app.wait_up()
            slots = ("app-1", "app-2") if how == "two-engines" else ("app-1",)
            terminal = None
            if how == "window-x-terminal-engine":
                # an engine the Terminal started (its own profile, no tie to the app) must survive the app's close
                control(scratch, "profile", "create", "terminal-1", "--from-config", str(scratch.models / "strata-stub2.json"))
                terminal = control(scratch, "runtime", "start", "--profile", "terminal-1", "--wait", "60")
                if terminal.get("state") != "ready":
                    raise RuntimeError(f"the Terminal's service did not start: {terminal}")
                terminal_pids = {r["pid"] for r in procs.records(profile="terminal-1")} | {runtime._read_state("terminal-1")["host_pid"]}
            state = app.load("stub")
            if how == "two-engines":
                state = app.load("stub2", multiple=True)
            engine_ports = [i["port"] for i in state["instances"]]
            idents, _ = tree_of(app, slots)
            kinds = sorted({("engine" if "engine" in " ".join(i.cmd) else "server" if "serve.server" in " ".join(i.cmd) else
                             "owner" if "strata_control.runtime" in " ".join(i.cmd) else "app" if "strata-desktop" in i.cmd[0].lower()
                             else "host" if "node" in i.cmd[0].lower() else "other") for i in idents})
            note = f"tree {len(idents)} processes: {', '.join(kinds)}; engine ports {engine_ports}"
            t0 = time.monotonic()
            if how == "normal-close":
                subprocess.run(["taskkill", "/PID", str(app.proc.pid)], capture_output=True)
            elif how in ("window-x", "two-engines", "window-x-terminal-engine"):
                hwnd = main_window(app.proc.pid)
                if not hwnd:
                    raise RuntimeError("no main window")
                user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            elif how == "kill-app":
                subprocess.run(["taskkill", "/F", "/PID", str(app.proc.pid)], capture_output=True)
            elif how == "host-crash":
                host = app.host()
                subprocess.run(["taskkill", "/F", "/PID", str(host.pid)], capture_output=True)
            elif how == "second-launch":
                second = subprocess.Popen([str(install / "strata-desktop.exe")], cwd=install, env=scratch.env,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    code = second.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    second.kill()
                    problems.append("the second launch did not exit")
                    code = None
                time.sleep(2)
                hosts = [p for p in psutil.process_iter(["name", "pid"]) if p.info["name"].lower() == "node.exe"
                         and (p.parent() and p.parent().pid == app.proc.pid)]
                apps = [p for p in psutil.process_iter(["name"]) if p.info["name"].lower() == "strata-desktop.exe"]
                if app.proc.poll() is not None:
                    problems.append("the first app went away")
                if len(apps) != 1:
                    problems.append(f"{len(apps)} app processes (want 1)")
                if len(hosts) != 1:
                    problems.append(f"{len(hosts)} hosts under the first app (want 1)")
                after, _ = tree_of(app, slots)
                new = [i for i in after if i.pid not in {x.pid for x in idents}]
                if new:
                    problems.append(f"the second launch started: {[' '.join(i.cmd)[:60] for i in new]}")
                note += f"; second launch exit code {code}, app processes {len(apps)}, hosts {len(hosts)}"
                subprocess.run(["taskkill", "/PID", str(app.proc.pid)], capture_output=True)
            if how == "host-crash":
                # the shell and its window stay (nothing here restarts a dead host); the host, Control's owner, the server and
                # the engine must not
                idents = [i for i in idents if any(k in " ".join(i.cmd) for k in ("strata_control.runtime", "serve.server", "engine", "node.exe"))]
            left = wait_gone(idents, 90)
            if left:
                problems.append(f"processes left: {left}")
            for port in engine_ports:
                deadline = time.monotonic() + 10
                while port_open(port) and time.monotonic() < deadline:
                    time.sleep(0.2)
                if port_open(port):
                    problems.append(f"engine port {port} still open")
            if terminal is None and not lease_free():
                problems.append("lease still held")
            marks = sorted(p.name for p in scratch.mark.iterdir())
            if terminal is not None:
                status = control(scratch, "runtime", "status", "--profile", "terminal-1")
                alive_pids = [p for p in terminal_pids if procs.start_token(p)]
                if status.get("state") != "ready" or len(alive_pids) != len(terminal_pids):
                    problems.append(f"closing the app disturbed the Terminal's service: {status.get('state')}, alive {alive_pids} of {sorted(terminal_pids)}")
                note += f"; the Terminal's service is {status.get('state')} after the app closed"
                stopped = control(scratch, "runtime", "stop", "--profile", "terminal-1", "--wait", "30")
                if stopped.get("state") != "stopped" or [p for p in terminal_pids if procs.start_token(p)]:
                    problems.append(f"the Terminal's service did not stop on request: {stopped}")
                time.sleep(1)
                if not lease_free():
                    problems.append("lease still held after the Terminal's service stopped")
            if how in ("normal-close", "window-x", "two-engines", "second-launch", "window-x-terminal-engine") and not ({"quit", "eof"} & set(marks)):
                problems.append(f"the engine was not stopped the clean way (marks {marks})")
            seconds = round(time.monotonic() - t0, 1)
            if how == "host-crash":
                subprocess.run(["taskkill", "/PID", str(app.proc.pid)], capture_output=True)
                note += "; the app shell stays open after a host crash (by design today)"
            # the slot must be usable again: stopping a lost owner's slot cleans its state; nothing blocks the next load
            for slot in slots:
                res = stop_cli(scratch, slot)
                already = "not managed or is stopped" in json.dumps(res)       # nothing was left to stop: the clean outcome
                if res.get("state") not in ("stopped",) and not already:
                    problems.append(f"runtime stop {slot} after the close: {res}")
                if runtime._read_state(slot) is not None:
                    problems.append(f"state of {slot} left behind")
            return {"name": f"app-{how}", "ok": not problems, "problems": problems, "seconds": seconds, "marks": marks, "note": note}
        except Exception as exc:  # noqa: BLE001
            return {"name": f"app-{how}", "ok": False, "problems": [f"{type(exc).__name__}: {exc}"] + problems, "note": note}
        finally:
            for p in psutil.process_iter(["name", "exe"]):
                try:
                    if (p.info["exe"] or "").lower().startswith(str(install).lower()):
                        p.kill()
                except psutil.Error:
                    pass
            for r in procs.records():
                try:
                    psutil.Process(r["pid"]).kill()
                except psutil.Error:
                    pass
            os.environ.clear(); os.environ.update(saved)


SCENARIOS = ["normal-close", "window-x", "window-x-terminal-engine", "kill-app", "host-crash", "second-launch", "two-engines"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", required=True)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--report")
    a = ap.parse_args()
    names = SCENARIOS if a.scenarios == "all" else a.scenarios.split(",")
    results = []
    for how in names:
        r = case(how, Path(a.install), how)
        results.append(r)
        print(("PASS " if r["ok"] else "FAIL ") + r["name"], r.get("seconds", ""), r.get("problems") or "", r.get("note", ""), flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps({"platform": "windows", "engine": "fake", "results": results}, indent=1), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
