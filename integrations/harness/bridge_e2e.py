#!/usr/bin/env python3
"""End-to-end agent bridge test: Strata API worker -> DeepSeek Harness -> durable session.

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


def read_session_events(session_id: str) -> list[dict]:
    """Read the durable session file, across every concatenated zstd frame."""
    root = Path(os.environ.get("DSH_HOME", Path.home() / ".dsh")) / "sessions"
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsh", default=os.environ.get("DSH_CLI", DEFAULT_DSH))
    ap.add_argument("--python", default=str(DEFAULT_PYTHON))
    ap.add_argument("--worker-port", type=int, default=8095)
    ap.add_argument("--agent-port", type=int, default=0, help="0 lets the OS choose")
    ap.add_argument("--keep", action="store_true", help="leave the processes and temp dir in place")
    args = ap.parse_args()

    dsh = Path(args.dsh)
    python = Path(args.python)
    if not dsh.exists():
        print(f"prerequisite missing: the dsh CLI is not at {dsh}", file=sys.stderr)
        return 2
    if not python.exists():
        print(f"prerequisite missing: no interpreter at {python}", file=sys.stderr)
        return 2
    if not PROFILE_PATCH.exists():
        print(f"prerequisite missing: the Strata profile patch is not at {PROFILE_PATCH}", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix=f"{WORKSPACE_MARKER}-"))
    (tmp / "ws").mkdir()
    worker = agent = None
    failure: str | None = None
    session_id = None

    print("strata agent bridge — end to end")
    print(f"  profile : {PROFILE_PATCH}")
    print(f"  dsh     : {dsh}")
    print(f"  temp    : {tmp}")
    print()

    try:
        print("[1/5] Strata API worker")
        worker, worker_log = start_worker(python, args.worker_port, tmp)
        health = wait_for_worker(args.worker_port)
        log(f"ready: model={health.get('model')} context={health.get('max_context')}")

        print("[2/5] DeepSeek Harness with the Strata profile")
        agent, dsh_log = start_agent(dsh, args.agent_port, tmp)
        text = wait_for_agent_url(dsh_log, agent)
        m = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9_-]+)", text)
        assert m, "URL was matched by the caller but not re-matched here"
        agent_port, token = int(m.group(1)), m.group(2)
        log(f"ready on 127.0.0.1:{agent_port}")

        print("[3/5] Operator session with approval answering enabled")
        # `interactions=answer` is required: it defaults to 'none', and a peer without it cannot
        # answer approvals, so every ask would fail closed as 'unavailable'.
        _, _, set_cookie = http_json("GET", "127.0.0.1", agent_port,
                                     f"/?token={token}&interactions=answer")
        cookie = set_cookie.split(";", 1)[0]
        if not cookie.startswith("dsh-auth-"):
            raise BridgeError(f"no operator cookie was issued (got {set_cookie[:80]!r})")
        log("authenticated as owner with interactions=answer")

        print("[4/5] Session, then one agent turn")
        created = gateway_call("127.0.0.1", agent_port, cookie, "c1", "session/create",
                               {"request": {"cwd": str(tmp / "ws")}})
        session_id = created["sessionId"]
        log(f"session {session_id}")
        gateway_call("127.0.0.1", agent_port, cookie, "p1", "session/prompt",
                     {"request": {"requestId": "req-1", "sessionId": session_id,
                                  "mode": "queue",
                                  "content": [{"type": "text", "text": PROMPT}]}})
        log("prompt accepted")

        print("[5/5] Waiting for the turn to reach a terminal state")
        events = []
        deadline = time.time() + 120
        while time.time() < deadline:
            events = read_session_events(session_id)
            if any(e.get("type") == "turn/end" for e in events):
                break
            time.sleep(1.0)

        turn_end = next((e for e in events if e.get("type") == "turn/end"), None)
        if turn_end is None:
            raise BridgeError("the turn never reached turn/end within 120s")
        reason = (turn_end.get("data") or {}).get("reason") or {}
        if reason.get("kind") != "completed":
            raise BridgeError(f"the turn ended as {json.dumps(reason)[:400]}")

        assistant = next((e for e in events if e.get("type") == "assistant/message"), None)
        if assistant is None:
            raise BridgeError("the turn completed but committed no assistant message")
        source = ((assistant.get("data") or {}).get("message") or {}).get("source") or {}
        if source.get("provider") != "strata-local":
            raise BridgeError(f"the assistant message names provider {source.get('provider')!r}, "
                              f"not 'strata-local'")

        # Independent confirmation that the request actually reached the worker, not just that the
        # session recorded a plausible-looking source.
        _, raw, _ = http_json("GET", "127.0.0.1", args.worker_port, "/v1/status")
        activity = (json.loads(raw).get("activity") or {})
        requests = activity.get("requests") or 0
        if requests < 1:
            raise BridgeError("the session completed but the Strata worker recorded no requests")

        print()
        print("PASS")
        print(f"  turn            : {reason.get('kind')}")
        print(f"  provider/model  : {source.get('provider')} / {source.get('model')}")
        print(f"  worker requests : {requests}")
        print(f"  turn statements : {', '.join(e.get('type','?') for e in events if e.get('type','').startswith('turn/'))}")
        return 0

    except BridgeError as e:
        failure = str(e)
    except Exception as e:  # noqa: BLE001 - a test harness reports, it does not crash
        failure = f"{type(e).__name__}: {e}"
    finally:
        if not args.keep:
            for proc in (agent, worker):
                if proc and proc.poll() is None:
                    try:
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=8)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
        else:
            print(f"\nkeeping processes; temp dir: {tmp}")

    print()
    print("FAIL")
    print(f"  {failure}")
    if session_id:
        evs = read_session_events(session_id)
        print(f"  durable session events ({len(evs)}):")
        for e in evs[-18:]:
            t = e.get("type")
            d = e.get("data") or {}
            extra = ""
            if t == "turn/end":
                extra = json.dumps(d.get("reason"))[:220]
            elif t == "assistant/message":
                extra = json.dumps((d.get("message") or {}).get("source"))[:160]
            print(f"    {e.get('seq')} {t} {extra}")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"  logs left in {tmp}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
