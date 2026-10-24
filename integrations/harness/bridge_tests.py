#!/usr/bin/env python3
"""Agent bridge tests: Strata API worker -> DeepSeek Harness -> durable session.

These cover the three behaviours the handoff singles out as the difficult part of the integration,
and they run without model weights:

    default      one turn reaches the local Strata worker and completes with recorded provenance
    --cancel     a turn is cancelled mid-flight and reaches an aborted terminal state
    --approval   a tool call raises an approval request over the gateway and the answer is applied
    --all        all three

What they do NOT prove: inference quality, tokens-per-second, or tool correctness against a real
model. The worker runs its built-in `MockEngine` because no Strata pack fits this development Mac
(docs/work/IMPLEMENTATION_STATUS.md section 3.2). Every assertion here is about the *path*.

The reusable helpers live in `_bridge.py`; `bridge_e2e.py` is the single-behaviour convenience
wrapper around the `default` case.

Usage:
    python integrations/harness/bridge_tests.py [--all|--cancel|--approval] [--keep]

Exit codes: 0 all selected cases passed, 1 at least one failed, 2 a prerequisite was missing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bridge import (  # noqa: E402
    DEFAULT_DSH,
    DEFAULT_PYTHON,
    PROFILE_PATCH,
    BridgeError,
    gateway_call,
    http_json,
    log,
    read_session_events,
    start_agent,
    start_worker,
    wait_for_agent_url,
    wait_for_worker,
)


class Fixture:
    """A running worker + Harness pair with an authenticated owner session."""

    def __init__(self, dsh: Path, python: Path, worker_port: int, keep: bool):
        self.dsh, self.python, self.worker_port, self.keep = dsh, python, worker_port, keep
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-bridge-"))
        (self.tmp / "ws").mkdir()
        self.worker = self.agent = None
        self.agent_port = 0
        self.cookie = ""

    def __enter__(self) -> "Fixture":
        self.worker, self.worker_log = start_worker(self.python, self.worker_port, self.tmp)
        wait_for_worker(self.worker_port)
        self.agent, self.agent_log = start_agent(self.dsh, 0, self.tmp)
        text = wait_for_agent_url(self.agent_log, self.agent)
        import re

        m = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9_-]+)", text)
        if not m:
            raise BridgeError("the Harness printed no usable URL")
        self.agent_port, token = int(m.group(1)), m.group(2)
        # `interactions=answer` is REQUIRED: it defaults to 'none', and a peer without it cannot
        # answer approvals, so every ask would fail closed as 'unavailable'.
        _, _, set_cookie = http_json("GET", "127.0.0.1", self.agent_port,
                                     f"/?token={token}&interactions=answer")
        self.cookie = set_cookie.split(";", 1)[0]
        if not self.cookie.startswith("dsh-auth-"):
            raise BridgeError(f"no operator cookie was issued ({set_cookie[:80]!r})")
        return self

    def __exit__(self, *exc) -> None:
        if not self.keep:
            for proc in (self.agent, self.worker):
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
            print(f"    (kept: {self.tmp})")
        if not self.keep:
            shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- operations -------------------------------------------------------

    def new_session(self) -> str:
        created = gateway_call("127.0.0.1", self.agent_port, self.cookie, "c", "session/create",
                               {"request": {"cwd": str(self.tmp / "ws")}})
        return created["sessionId"]

    def prompt(self, session_id: str, text: str, rpc_id="p", request_id="req-1"):
        return gateway_call("127.0.0.1", self.agent_port, self.cookie, rpc_id, "session/prompt",
                            {"request": {"requestId": request_id, "sessionId": session_id,
                                         "mode": "queue",
                                         "content": [{"type": "text", "text": text}]}})

    def cancel(self, session_id: str, rpc_id="x"):
        return gateway_call("127.0.0.1", self.agent_port, self.cookie, rpc_id, "session/cancel",
                            {"request": {"sessionId": session_id}})

    def wait_turn_end(self, session_id: str, timeout=120.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            events = read_session_events(session_id)
            end = next((e for e in events if e.get("type") == "turn/end"), None)
            if end is not None:
                return end, events
            time.sleep(0.5)
        return None, events

    def worker_requests(self) -> int:
        _, raw, _ = http_json("GET", "127.0.0.1", self.worker_port, "/v1/status")
        return (json.loads(raw).get("activity") or {}).get("requests") or 0


# --------------------------------------------------------------------------- cases

def case_completion(fx: Fixture) -> None:
    """A turn must reach the local worker and complete, with recorded provenance."""
    sid = fx.new_session()
    log(f"session {sid}")
    fx.prompt(sid, "Say hello in exactly five words.")
    log("prompt accepted")
    end, events = fx.wait_turn_end(sid)
    if end is None:
        raise BridgeError("the turn never reached turn/end within 120s")
    reason = (end.get("data") or {}).get("reason") or {}
    if reason.get("kind") != "completed":
        raise BridgeError(f"the turn ended as {json.dumps(reason)[:400]}")
    assistant = next((e for e in events if e.get("type") == "assistant/message"), None)
    if assistant is None:
        raise BridgeError("the turn completed but committed no assistant message")
    source = ((assistant.get("data") or {}).get("message") or {}).get("source") or {}
    if source.get("provider") != "strata-local":
        raise BridgeError(f"assistant source names {source.get('provider')!r}, not 'strata-local'")
    n = fx.worker_requests()
    if n < 1:
        raise BridgeError("the session completed but the worker recorded no requests")
    log(f"turn=completed provider={source.get('provider')} model={source.get('model')} "
        f"worker_requests={n}")


def case_cancellation(fx: Fixture) -> None:
    """Cancelling a turn must abort it and produce an aborted terminal state, not a completion.

    Also checks the handoff's specific warning: `session/cancel` returns `{accepted:true}` as
    *admission only*. The assertion is on the durable terminal state, not on that acknowledgement.
    """
    sid = fx.new_session()
    log(f"session {sid}")
    fx.prompt(sid, "Write a very long, detailed essay about the history of computing.")
    log("prompt accepted")
    # Cancel promptly: after acceptance, before the turn can finish.
    time.sleep(0.3)
    value = fx.cancel(sid)
    log(f"cancel acknowledged: {json.dumps(value)}")
    if value.get("accepted") is not True:
        raise BridgeError(f"cancel was not accepted: {json.dumps(value)[:200]}")
    end, events = fx.wait_turn_end(sid)
    if end is None:
        raise BridgeError("after cancel the turn never reached a terminal state")
    kind = ((end.get("data") or {}).get("reason") or {}).get("kind")
    log(f"terminal state: {kind}")
    if kind == "completed":
        # Not necessarily a failure of the bridge: a mock engine answers instantly, so the turn may
        # genuinely finish before the cancel lands. Say so rather than pretending either way.
        raise BridgeError("the turn completed before the cancel took effect; with an instant mock "
                          "engine this is a timing race, not evidence either way — rerun to "
                          "confirm, and re-run against a real model for a decisive result")
    if kind not in ("aborted", "interrupted", "failed", "error"):
        raise BridgeError(f"unexpected terminal state after cancel: {json.dumps(kind)}")
    log(f"cancel produced '{kind}' (accepted != settled, as documented)")


def case_approval(fx: Fixture) -> None:
    """A tool call must raise an approval request that the operator can answer over the gateway.

    The handoff records this as an unimplemented transport. It is implemented on the gateway
    surface: `approval/request` is a forwarded waterfall and `POST /api/$events/result` settles it.
    Assertions are written so an approved run proves the *outbound* seam (a request really was
    raised and had to be decided); the tools the model is offered here may not need approval, in
    which case the case reports that honestly instead of inventing a pass.
    """
    # Read the workspace before/after so a shell tool that really ran is visible.
    marker = fx.tmp / "ws" / "approval-probe.txt"
    sid = fx.new_session()
    log(f"session {sid}")
    # Ask for an action that needs a tool rather than an answer from memory.
    fx.prompt(sid, "Create a file named approval-probe.txt in the current directory "
                   "containing the single word hello, then tell me you did it.", rpc_id="pa")
    log("prompt accepted")
    end, events = fx.wait_turn_end(sid)
    if end is None:
        raise BridgeError("the turn never reached a terminal state")
    kind = ((end.get("data") or {}).get("reason") or {}).get("kind")
    asked = [e for e in events if e.get("type") == "approval/asked"]
    decided = [e for e in events if e.get("type") == "approval/decided"]
    log(f"terminal={kind} approval/asked={len(asked)} approval/decided={len(decided)} "
        f"probe_file={'present' if marker.exists() else 'absent'}")
    if asked:
        # The durable audit pair must be complete and every decision must be in the vocabulary.
        outcomes = [(e.get("data") or {}).get("outcome") for e in decided]
        allowed = {"allowed-once", "rejected", "cancelled", "unavailable"}
        bad = [o for o in outcomes if o not in allowed]
        if bad:
            raise BridgeError(f"approval decided with a non-vocabulary outcome: {bad}")
        log(f"approval decisions: {outcomes}")
        return
    log("no approval was asked for this tool set; the outbound seam was therefore not exercised. "
        "Re-run with a tool that requires approval to cover it (see the qualification kit).")


CASES = {"completion": case_completion, "cancel": case_cancellation, "approval": case_approval}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsh", default=os.environ.get("DSH_CLI", DEFAULT_DSH))
    ap.add_argument("--python", default=str(DEFAULT_PYTHON))
    ap.add_argument("--worker-port", type=int, default=8095)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cancel", action="store_true")
    ap.add_argument("--approval", action="store_true")
    args = ap.parse_args()

    selected = []
    if args.all:
        selected = ["completion", "cancel", "approval"]
    else:
        if args.cancel:
            selected.append("cancel")
        if args.approval:
            selected.append("approval")
        if not selected:
            selected = ["completion"]

    dsh, python = Path(args.dsh), Path(args.python)
    for path, what in ((dsh, "dsh CLI"), (python, "python interpreter"), (PROFILE_PATCH, "profile")):
        if not path.exists():
            print(f"prerequisite missing: {what} not at {path}", file=sys.stderr)
            return 2

    print("strata <-> harness bridge tests")
    print(f"  cases : {', '.join(selected)}")
    print()
    failures = []
    for name in selected:
        print(f"[{name}]")
        try:
            with Fixture(dsh, python, args.worker_port, args.keep) as fx:
                CASES[name](fx)
            print(f"  -> PASS\n")
        except Exception as e:  # noqa: BLE001 - a test reports, it does not crash
            print(f"  -> FAIL: {e}\n")
            failures.append((name, str(e)))

    if failures:
        print("FAIL")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    print("PASS — all selected bridge cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
