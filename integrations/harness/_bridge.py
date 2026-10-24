#!/usr/bin/env python3
"""Agent bridge tests: Strata API worker -> DeepSeek Harness -> durable session.

This exercises the integration that the handoff calls the difficult part, without needing model
weights. It proves the *bridge*: that the pinned Harness boots with the Strata profile, that an
agent turn reaches the local Strata API worker, that the assistant message records the Strata
provider and model as its source, and that the turn reaches a terminal state in a durable session.

It does NOT prove inference quality. The worker runs its built-in `MockEngine`, because no Strata
pack fits this development Mac (see docs/work/IMPLEMENTATION_STATUS.md section 3.2). Every reported
fact is about the path, not the model; the qualification kit replaces the mock engine with a real
one on a machine that can hold a pack.

Usage:
    python integrations/harness/bridge_e2e.py [--dsh /path/to/dsh] [--port 8095] [--keep]

Exit codes:
    0  the turn completed and the assistant message names the Strata provider and model
    1  the bridge failed; the durable session events are printed to help diagnose it
    2  a prerequisite was missing (the dsh CLI, the worker, or the profile)
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PROFILE_PATCH = HERE / "profile" / "cordis.patch.yml"

DEFAULT_DSH = "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"
DEFAULT_PYTHON = REPO.parent / "python311" / "bin" / "python"

WORKSPACE_MARKER = "strata-bridge-e2e"
PROMPT = "Say hello in exactly five words."


class BridgeError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def http_json(method: str, host: str, port: int, path: str, body=None, cookie: str | None = None):
    conn = http.client.HTTPConnection(host, port, timeout=30)
    headers = {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    if cookie:
        headers["cookie"] = cookie
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", "replace")
    set_cookie = resp.getheader("set-cookie") or ""
    conn.close()
    return resp.status, raw, set_cookie


def wait_for_worker(port: int, timeout=60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, raw, _ = http_json("GET", "127.0.0.1", port, "/health")
            if status == 200:
                return json.loads(raw)
        except OSError:
            pass
        time.sleep(0.4)
    raise BridgeError(f"the Strata worker did not answer /health on 127.0.0.1:{port} within {timeout}s")


def start_worker(python: Path, port: int, tmp: Path) -> tuple[subprocess.Popen, Path]:
    logfile = tmp / "worker.log"
    handle = logfile.open("w")
    proc = subprocess.Popen(
        [str(python), "-m", "serve.server", "--engine", "mock", "--port", str(port),
         "--script", "I am a local Strata model."],
        cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT,
    )
    return proc, logfile


def start_agent(dsh: Path, port: int, tmp: Path) -> tuple[subprocess.Popen, Path]:
    logfile = tmp / "dsh.log"
    handle = logfile.open("w")
    proc = subprocess.Popen(
        [str(dsh), "--profile", "web", "--patch", str(PROFILE_PATCH),
         "--port", str(port), "--no-open"],
        cwd=str(tmp / "ws"), stdout=handle, stderr=subprocess.STDOUT,
    )
    return proc, logfile


def wait_for_agent_url(logfile: Path, proc: subprocess.Popen, timeout=120.0) -> str:
    deadline = time.time() + timeout
    pattern = re.compile(r"http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9_-]+)")
    while time.time() < deadline:
        if proc.poll() is not None:
            raise BridgeError(f"the Harness exited early (code {proc.returncode}); log:\n"
                              + logfile.read_text(errors="replace")[-2000:])
        text = logfile.read_text(errors="replace") if logfile.exists() else ""
        m = pattern.search(text)
        if m:
            return text
        time.sleep(0.5)
    raise BridgeError(f"the Harness did not print its URL within {timeout}s; log:\n"
                      + (logfile.read_text(errors="replace")[-2000:] if logfile.exists() else "(none)"))


def gateway_call(host: str, port: int, cookie: str, rpc_id: str, method: str, args: dict):
    body = {"type": "client-request", "rpcId": rpc_id, "method": method, "payload": {"args": args}}
    status, raw, _ = http_json("POST", host, port, f"/api/{method}", body, cookie=cookie)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise BridgeError(f"{method}: the gateway returned non-JSON ({status}): {raw[:300]}") from e
    result = parsed.get("result") or {}
    if not result.get("ok"):
        raise BridgeError(f"{method} was refused: {json.dumps(result.get('error'))[:400]}")
    return result.get("value")


def read_session_events(session_id: str, home: Path | None = None) -> list[dict]:
    """Read the durable session file, across every concatenated zstd frame.

    `home` names the Harness home explicitly. A per-attempt revision runs with its own DSH_HOME, so
    relying on this process's environment would read the wrong sessions directory.
    """
    base = Path(home) if home is not None else Path(os.environ.get("DSH_HOME", Path.home() / ".dsh"))
    root = base / "sessions"
    matches = list(root.glob(f"*/{session_id}/session.v*.jsonl*"))
    if not matches:
        return []
    path = matches[0]
    if not path.name.endswith(".zstd"):
        return [json.loads(l) for l in path.read_text(errors="replace").splitlines() if l.strip()]
    # zstd frames are concatenated as the session is appended to; `zstd -d` reads them all.
    try:
        raw = subprocess.run(["zstd", "-d", "-c", "--long=27", str(path)],
                             capture_output=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    out = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


