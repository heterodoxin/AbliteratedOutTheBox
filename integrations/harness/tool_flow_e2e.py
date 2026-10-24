#!/usr/bin/env python3
"""Tool execution, approval and cancellation, end to end (checklist A03, A04, A05).

This drives the real path — a scripted Strata worker, the Harness agent loop with its real bash tool,
and a durable session — so what it proves is the *integration*, not a mock of it.

The model is scripted. That is a deliberate and limited choice: writing a file via bash is
deterministic, and a scripted tool call exercises exactly the same code as a model-chosen one —
the worker's parser, the Harness tool executor, the approval gate and the durable record. What it
does **not** prove is that a real model chooses tools well, and this script never claims that.

Three flows, each with one assertion that matters:

**A03 — the tool runs once, in the Harness.**
    The scripted turn calls bash to append a line to a file. The assertions are that the file has
    exactly one line (so the tool ran exactly once), that the durable session records a tool call and
    a tool result, and that the Strata worker was given the transcript rather than executing anything
    itself — the worker has no tool-execution path at all, and the call count is the evidence. The
    call escalates the sandbox, so it cannot run until a peer *grants* it: the flow connects that peer
    and asserts an approval was asked, answered, and recorded as the grant.

**A04 — the denial test: a tool that could escalate without an answerer would be the whole failure.**
    The same scripted call and the same peer connection; the only difference is the decision. The peer
    delegates (`next`) instead of granting, so the composed answerer chain has no terminal answerer
    and the request resolves the fail-closed `unavailable`. The assertions are that the escalation
    reached the gate, that its recorded outcome is exactly `unavailable`, that the turn settled on
    that denial, and that the file does not exist afterwards. Connecting the peer is what makes the
    outcome observable — with no peer at all the gateway keeps a forwarded waterfall pending until a
    client connects, so the request would be neither granted nor decided.

**A05 — Stop settles the turn.**
    A prompt is sent and cancelled while the turn is live. The assertions are that the turn reaches a
    terminal state without the tool having run, and that cancellation is acknowledged only as
    admission (`accepted`), with the durable turn state being what actually settles it.

Run: python integrations/harness/tool_flow_e2e.py --dsh <checkout> [--keep]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import http.client
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRATA = HERE.parents[1]
PROFILE_PATCH = HERE / "profile" / "cordis.patch.yml"
#: A TEST-ONLY overlay, applied after the product patch. It disables the session-title model
#: request, which a scripted worker cannot absorb: the worker replays one script per request, so
#: an extra request shifts every later reply and the run looks like a model that chose not to
#: call a tool. The product profile must keep answering title requests, so the row lives here.
SCRIPTED_OVERLAY = HERE / "profile" / "scripted-worker.cordis.patch.yml"
PYTHON = os.environ.get("STRATA_PYTHON", sys.executable)

#: The bash call the scripted worker emits. It appends exactly one line, so counting the lines counts
#: the executions -- a replayed tool would leave two.
# The `</think>` is not decoration: without it the server treats the *whole* reply as reasoning, so
# the tool-call block lands in the reasoning field and is never executed. The default mock script ends
# its thinking the same way, which is where the convention comes from. The failure is silent -- the
# turn completes normally with the tool call sitting in the wrong field.
#
# Every parameter is load-bearing:
#   * `command` is the effect the assertions count;
#   * `description` is REQUIRED by this runtime's bash schema -- a call without it is rejected by
#     argument validation ("invalid arguments: missing required property \"description\"") before the
#     tool body runs, which looks exactly like a tool that never executed;
#   * `sandbox_permissions` + `justification` are what make the call ask for approval at all. The
#     approval policy is `ask` and the sandbox is `workspace-write`, but a command the sandbox would
#     allow runs *without* an approval prompt: only a strictly wider sandbox mode is decided by
#     `ctx.approval`. So a call that never escalates cannot prove anything about the approval gate --
#     it runs with no answerer present, which is how A04's denial test can pass while testing nothing.
TOOL_CALL = (
    "I will write the marker file.</think>\n\n"
    "<tool_call>\n<function=bash>\n"
    "<parameter=command>\n"
    "echo run >> {marker}\n"
    "</parameter>\n"
    "<parameter=description>\n"
    "Append one line to the marker file\n"
    "</parameter>\n"
    "<parameter=sandbox_permissions>\n"
    "danger-full-access\n"
    "</parameter>\n"
    "<parameter=justification>\n"
    "The marker write must not be refused by the file sandbox.\n"
    "</parameter>\n"
    "</function>\n</tool_call>"
)
FINAL_ANSWER = "The marker file has been written."

#: The worker is started with `--script-cycle`, so the script is chosen by the request's position
#: rather than by a counter that every extra request advances. That matters because the Harness makes
#: requests the script author did not plan for -- a session title, and a re-ask after a tool result --
#: and with the default behaviour one of those shifts every later reply. The symptom is a run that
#: looks like a model which simply chose not to use a tool.


class FlowError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def http_json(method: str, host: str, port: int, path: str, body=None, cookie: str | None = None,
              timeout: float = 30.0):
    """A request that does not follow redirects, because the redirect carries the credential.

    `urllib` follows the agent's 303 and discards the `set-cookie` on it, which surfaces as a 401 on
    the next request and looks like an authentication bug rather than a client one. `http.client`
    returns the response as sent, which is what this needs.
    """
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    headers = {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    if cookie:
        headers["cookie"] = cookie
    try:
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", "replace")
        return response.status, raw, response.getheader("set-cookie") or ""
    finally:
        connection.close()


def rpc(host: str, port: int, cookie: str, rpc_id: str, method: str, args: dict):
    """One gateway call.

    The envelope is the gateway's own: `{"type": "client-request", "rpcId", "method", "payload":
    {"args"}}`. Posting `args` directly is answered with `invalid client-request message` and a list
    of schema issues, which reads like a malformed session rather than a malformed envelope.
    """
    body = {"type": "client-request", "rpcId": rpc_id, "method": method, "payload": {"args": args}}
    status, raw, _ = http_json("POST", host, port, f"/api/{method}", body, cookie=cookie)
    if status != 200:
        raise FlowError(f"{method} returned HTTP {status}: {raw[:300]}")
    payload = json.loads(raw)
    result = payload.get("result") or {}
    if not result.get("ok"):
        raise FlowError(f"{method} was refused: {json.dumps(result.get('error'))[:400]}")
    return result.get("value")


#: The gateway's event stream, and the HTTP endpoint that answers one forwarded waterfall. Both are
#: gateway-internal endpoint names, not HTTP routes: the stream is a logical stream carried by the
#: WebSocket mux, and only the answer travels as an ordinary `/api/<endpoint>` call.
EVENT_STREAM_ENDPOINT = "$events"
EVENT_RESULT_ENDPOINT = "$events/result"
REMOTE_STREAM_MUX_PATH = "/api/remote.mux"


class ApprovalAnswerer:
    """A real peer, because an approval cannot be answered without one.

    `approval/request` is not an RPC that a client can simply call: it is a Cordis *waterfall* the
    gateway forwards to connected peers, and it is delivered over the gateway's event stream. A peer
    that wants to decide one must receive the frame (a WebSocket on `/api/remote.mux`, opened as the
    `$events` logical stream) and then send its outcome back (`POST /api/$events/result`).

    `decision` is the outcome this peer returns for every request:
      * `'allowed-once'` — the grant A03 needs;
      * `'next'` — do not decide, delegate to the listeners behind this peer. It is what an
        answerer that is *present but not available* does, so the request resolves `unavailable` and
        the action fails closed, which is the outcome A04 asserts.

    A peer is also the only thing that can make an approval observable at all: with no client holding
    the stream open the gateway keeps the forwarded waterfall pending until one connects (it has a
    test named "delivers a pending waterfall to the first Client that connects"), so the request is
    never decided and nothing is `unavailable` -- the turn simply waits. That is why A04 connects a
    peer and delegates instead of asserting on an answerer that does not exist.

    No WebSocket client is available in the standard library, so the handshake and the four frame
    types this needs (text, ping, pong, close) are implemented here. Only frames the Host sends are
    parsed; the one frame this peer sends is the stream `open`.
    """

    def __init__(self, host: str, port: int, cookie: str, decision: str = "allowed-once"):
        self.host, self.port, self.cookie, self.decision = host, port, cookie, decision
        self.requests: list[dict] = []      # every approval request this peer was asked to decide
        self.outcomes: list[str] = []       # the decision actually sent for each of them
        self.error: str | None = None
        self.ready = threading.Event()      # set once the Host proved the event stream is live
        self.client_id: str | None = None
        self._sock: socket.socket | None = None
        self._buffer = b""
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------------------------------

    def start(self, timeout: float = 30.0) -> "ApprovalAnswerer":
        """Connect and wait until the event stream is live.

        The wait is the point: an approval raised before the stream exists is delivered to nobody and
        fails closed, so a peer that connects late turns "no answerer" into a tool that never ran.
        """
        self._thread = threading.Thread(target=self._run, name="approval-answerer", daemon=True)
        self._thread.start()
        if not self.ready.wait(timeout):
            self.close()
            raise FlowError(f"the approval answerer never opened the event stream: {self.error or 'no reply'}")
        return self

    def close(self) -> None:
        sock = self._sock
        if sock is None:
            return
        try:
            self._send_frame(0x8, b"")       # close
        except Exception:                    # noqa: BLE001 - the peer is going away either way
            pass
        self._sock = None
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    # -- WebSocket carrier -------------------------------------------------------------------------

    def _connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=15)
        sock.settimeout(30)
        key = base64.b64encode(os.urandom(16)).decode()
        handshake = (
            f"GET {REMOTE_STREAM_MUX_PATH} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            # The operator credential from the launch URL: the same cookie the RPC calls use, and
            # the only thing that makes this a peer rather than an anonymous socket.
            f"Cookie: {self.cookie}\r\n\r\n"
        )
        sock.sendall(handshake.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = sock.recv(4096)
            if not chunk:
                raise FlowError("the event stream socket closed during the WebSocket handshake")
            head += chunk
        header, self._buffer = head.split(b"\r\n\r\n", 1)
        status = header.split(b"\r\n", 1)[0].decode("latin-1")
        if " 101 " not in status:
            raise FlowError(f"the event stream upgrade was refused: {status!r}")
        # The stream is idle between events, so a read deadline would end the peer during a quiet
        # stretch and silently turn A03 back into A04. Blocking reads are released by close().
        sock.settimeout(None)
        self._sock = sock

    def _read_exactly(self, count: int) -> bytes:
        while len(self._buffer) < count:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise FlowError("the event stream closed")
            self._buffer += chunk
        taken, self._buffer = self._buffer[:count], self._buffer[count:]
        return taken

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        """One client frame. Client frames must be masked (RFC 6455), and the Host rejects one that is not."""
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = bytes([0x80 | opcode, 0x80 | length])
        elif length < (1 << 16):
            header = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack("!H", length)
        else:
            header = bytes([0x80 | opcode, 0x80 | 127]) + struct.pack("!Q", length)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._sock.sendall(header + mask + masked)

    def _send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode())

    def _read_frame(self) -> tuple[int, bytes]:
        first, second = self._read_exactly(2)
        opcode, length = first & 0x0F, second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exactly(8))[0]
        masked = bool(second & 0x80)
        mask = self._read_exactly(4) if masked else b""
        payload = self._read_exactly(length)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, payload

    # -- protocol ----------------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._connect()
            stream_id = "approvals"
            self._send_text(json.dumps({
                "type": "open", "streamId": stream_id,
                "endpoint": EVENT_STREAM_ENDPOINT,
                # The endpoint's own envelope: exactly one `args` field, empty for this stream.
                "payload": {"args": {}},
            }))
            while True:
                opcode, payload = self._read_frame()
                if opcode == 0x8:                                  # close
                    return
                if opcode == 0x9:                                  # ping: the Host's liveness probe
                    self._send_frame(0xA, payload)
                    continue
                if opcode != 0x1:                                  # pong and anything else: ignore
                    continue
                self._handle(json.loads(payload.decode("utf-8", "replace")))
        except Exception as error:                                 # noqa: BLE001 - reported, never raised here
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self.ready.set()                                       # a failed start must not hang start()

    def _handle(self, message: dict) -> None:
        if message.get("type") != "item":
            return
        value = message.get("value")
        if not isinstance(value, dict):
            return
        if value.get("type") == "ready":
            self.client_id = value.get("clientId")
            self.ready.set()
            return
        if value.get("type") != "waterfall" or value.get("event") != "approval/request":
            return                                          # emitted notifications are not questions
        self._answer(value)

    def _answer(self, frame: dict) -> None:
        self.requests.append(frame.get("request") or {})
        # `next` is the DSH word for "this answerer does not decide": the waterfall continues to the
        # listeners behind this peer, and with no terminal answerer composed it settles on the
        # approval service's fail-closed default, `unavailable`. It is not a rejection -- `rejected`
        # would be a human saying no, which is a different outcome and a different test.
        outcome = ({"kind": "next"} if self.decision == "next"
                   else {"kind": "result", "value": self.decision})
        body = {
            "type": "client-request", "rpcId": f"approval-{len(self.requests)}",
            "method": EVENT_RESULT_ENDPOINT,
            "payload": {"args": {
                "clientId": self.client_id, "eventId": frame.get("eventId"),
                "outcome": outcome,
            }},
        }
        status, raw, _ = http_json("POST", self.host, self.port, f"/api/{EVENT_RESULT_ENDPOINT}",
                                   body, cookie=self.cookie)
        if status != 200:
            raise FlowError(f"answering the approval returned HTTP {status}: {raw[:200]}")
        answer = (json.loads(raw).get("result") or {})
        if not answer.get("ok"):
            raise FlowError(f"the approval answer was refused: {json.dumps(answer.get('error'))[:300]}")
        self.outcomes.append(self.decision)


def wait_for_worker(port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, _, _ = http_json("GET", "127.0.0.1", port, "/health", timeout=2)
            if status == 200:
                return
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.3)
    raise FlowError(f"the worker on {port} never became healthy")


def write_profile(port: int, tmp: Path) -> Path:
    """The Strata profile patch with this run's worker port.

    The shipped patch names 8095, and a worker left over from an earlier session on that port will
    happily answer every request -- producing a green run against the *wrong* worker, which is
    exactly how a stale process hid a failure here once already. Substituting the port makes the
    agent and the worker this test started the only pair that can talk.
    """
    text = PROFILE_PATCH.read_text(encoding="utf-8")
    if "http://127.0.0.1:8095/v1" not in text:
        raise FlowError(f"{PROFILE_PATCH} no longer names http://127.0.0.1:8095/v1; the substitution "
                        "below would silently do nothing")
    patched = tmp / "cordis.patch.yml"
    patched.write_text(text.replace("http://127.0.0.1:8095/v1", f"http://127.0.0.1:{port}/v1"),
                       encoding="utf-8")
    return patched


def start_worker(scripts: list[str], port: int, log_path: Path,
                 delay_s: float = 0.0) -> subprocess.Popen:
    if worker_port_in_use(port):
        raise FlowError(
            f"port {port} is already in use; a worker left over from an earlier session would answer "
            "this test's requests and its results would look plausible")
    argv = [PYTHON, "-m", "serve.server", "--engine", "mock", "--port", str(port), "--script-cycle"]
    if delay_s:
        # A turn that finishes before Stop can arrive proves nothing about Stop. The mock honours
        # cancellation between tokens, so a delay makes the window real.
        argv += ["--script-delay", str(delay_s)]
    for script in scripts:
        argv += ["--script", script]
    handle = log_path.open("wb")
    return subprocess.Popen(argv, cwd=str(STRATA), stdout=handle, stderr=subprocess.STDOUT)


def worker_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def start_agent(dsh: Path, port: int, home: Path, profile: str, log_path: Path,
                cwd: Path, patch: Path) -> subprocess.Popen:
    """Boot the agent with the Strata profile.

    The profile is selected the way the working bridge does it: `--profile <base> --patch <file>`, a
    patch layer over a shipped bundle. `dsh web --profile` does not exist, and passing it produces
    `unknown option '--profile'` before anything starts.
    """
    env = dict(os.environ)
    env["DSH_HOME"] = str(home)
    env.pop("ELECTRON_RUN_AS_NODE", None)
    argv = [str(dsh), "--profile", profile, "--patch", str(patch),
            "--patch", str(SCRIPTED_OVERLAY),
            "--port", str(port), "--no-open"]
    handle = log_path.open("wb")
    return subprocess.Popen(argv, env=env, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)


def wait_for_agent(log_path: Path, proc: subprocess.Popen, timeout: float = 150.0) -> tuple[int, str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise FlowError(f"the agent exited with code {proc.returncode}:\n"
                            f"{log_path.read_text(errors='replace')[-800:]}")
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        match = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9_-]+)", text)
        if match:
            return int(match.group(1)), match.group(2)
        time.sleep(0.5)
    raise FlowError(f"the agent never printed a URL:\n{log_path.read_text(errors='replace')[-800:]}")


def read_events(session_id: str, home: Path) -> list[dict]:
    root = home / "sessions"
    matches = list(root.glob(f"*/{session_id}/session.v*.jsonl*"))
    if not matches:
        return []
    newest = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        out = subprocess.run(["zstd", "-d", "-c", "--long=27", str(newest)],
                             capture_output=True, timeout=30)
        text = out.stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return []
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def wait_for_turn(session_id: str, home: Path, timeout: float = 180.0) -> list[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        events = read_events(session_id, home)
        if any(e.get("type") == "turn/end" for e in events):
            return events
        time.sleep(1.0)
    return read_events(session_id, home)


def summarise(events: list[dict]) -> dict:
    kinds: dict[str, int] = {}
    for event in events:
        kinds[event.get("type", "?")] = kinds.get(event.get("type", "?"), 0) + 1
    turn_end = next((e for e in events if e.get("type") == "turn/end"), None)
    reason = ((turn_end or {}).get("data") or {}).get("reason") or {}
    return {"kinds": kinds, "end": reason.get("kind"), "reason": reason}


class Agent:
    """One agent process with a session, so each flow gets a clean session."""

    def __init__(self, dsh: Path, port: int, home: Path, profile: str, answerer: bool, tmp: Path,
                 cwd: Path, worker_port: int):
        self.home, self.port, self.tmp, self.cwd = home, port, tmp, cwd
        cwd.mkdir(parents=True, exist_ok=True)
        # A profile naming this run's worker port, so no other process can answer for it.
        self.patch = write_profile(worker_port, tmp)
        self.log_path = tmp / f"agent-{port}.log"
        self.proc = start_agent(dsh, port, home, profile, self.log_path, self.cwd, self.patch)
        self.agent_port, token = wait_for_agent(self.log_path, self.proc)
        query = f"/?token={token}" + ("&interactions=answer" if answerer else "")
        _, _, cookie = http_json("GET", "127.0.0.1", self.agent_port, query)
        self.cookie = cookie.split(";", 1)[0]
        if not self.cookie.startswith("dsh-auth-"):
            raise FlowError(f"no operator cookie (got {cookie[:80]!r})")
        self.answerer = answerer
        self.peer: ApprovalAnswerer | None = None

    def answer_approvals(self, decision: str = "allowed-once") -> ApprovalAnswerer:
        """Connect the peer that decides approvals, and return it once its stream is live.

        This is what makes an approval answerable, and `interactions=answer` on the launch URL is
        not: that parameter is read only on `/connect`
        (`packages/client/connection/src/browser-auth.ts`), while `GET /?token=...` issues the
        browser cookie whose shared peer already permits the event stream
        (`packages/client/connection/src/peer.ts`: `options.interactions ?? 'answer'`). So
        `answerer=False` never removed an answerer; the `decision` a flow passes here is the real
        difference between A03 and A04.
        """
        peer = ApprovalAnswerer("127.0.0.1", self.agent_port, self.cookie, decision).start()
        self.peer = peer
        return peer

    def session(self, cwd: Path) -> str:
        result = rpc("127.0.0.1", self.agent_port, self.cookie, "c1", "session/create",
                     {"request": {"cwd": str(cwd)}})
        return result["sessionId"]

    def prompt(self, session_id: str, text: str, request_id: str = "req-1") -> None:
        rpc("127.0.0.1", self.agent_port, self.cookie, f"p-{request_id}", "session/prompt",
            {"request": {"requestId": request_id, "sessionId": session_id, "mode": "queue",
                         "content": [{"type": "text", "text": text}]}})

    def cancel(self, session_id: str) -> dict:
        return rpc("127.0.0.1", self.agent_port, self.cookie, "x1", "session/cancel",
                   {"request": {"sessionId": session_id}})

    def close(self):
        # Teardown never raises and never skips a step: a failure closing the peer once left the agent
        # process alive on its port, and the next run died at startup with EADDRINUSE instead of
        # reporting the real problem. The peer is best-effort; killing the process is not.
        if self.peer is not None:
            try:
                self.peer.close()
            except Exception as error:                     # noqa: BLE001 - cleanup must not mask the run
                log(f"closing the approval peer failed: {type(error).__name__}: {error}")
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def flow_a03(dsh: Path, tmp: Path, profile: str, base_port: int, results: dict) -> None:
    print("[A03] a read/edit/test workflow: the tool runs once, in the Harness")
    ws = tmp / "ws-a03"
    ws.mkdir(parents=True, exist_ok=True)
    marker = ws / "marker.txt"
    worker = start_worker([TOOL_CALL.format(marker=marker), FINAL_ANSWER],
                          base_port, tmp / "worker-a03.log")
    agent = None
    try:
        wait_for_worker(base_port)
        agent = Agent(dsh, base_port + 1, tmp / "home-a03", profile, answerer=True, tmp=tmp, cwd=ws, worker_port=base_port)
        # The scripted call escalates the sandbox, so the tool cannot run until a peer grants it. The
        # peer is connected (and its stream proved live) BEFORE the prompt, because an approval raised
        # while no peer holds the stream open fails closed and is gone for good.
        peer = agent.answer_approvals("allowed-once")
        session = agent.session(ws)
        agent.prompt(session, "Write the marker file using bash, then report that you did.")
        events = wait_for_turn(session, agent.home)
        summary = summarise(events)

        calls = [e for e in events if "tool" in str(e.get("type", ""))]
        asked = [e for e in events if e.get("type") == "approval/asked"]
        decided = [e for e in events if e.get("type") == "approval/decided"]
        executed = marker.exists()
        lines = marker.read_text().splitlines() if executed else []

        results["A03"] = {
            "turn_end": summary["end"],
            "tool_events": len(calls),
            "marker_lines": len(lines),
            "executions": len(lines),
            "approvals_asked": len(asked),
            "approvals_answered": len(peer.outcomes),
            "approval_outcomes": [((e.get("data") or {}).get("outcome")) for e in decided],
        }
        log(f"turn ended as {summary['end']!r}; {len(calls)} tool-related event(s)")
        log(f"the workspace marker has {len(lines)} line(s)")
        log(f"the peer answered {len(peer.outcomes)} of {len(asked)} approval(s): "
            f"{[((e.get('data') or {}).get('outcome')) for e in decided]}")

        if summary["end"] != "completed":
            raise FlowError(f"the turn did not complete: {json.dumps(summary)[:300]}")
        if not executed:
            raise FlowError("the bash tool did not run: no marker file was written")
        if len(lines) != 1:
            raise FlowError(f"the tool ran {len(lines)} times, not once: a completed tool effect was "
                            "replayed")
        # The tool ran because the approval was granted, not because it was never asked: without this
        # the flow would still pass if the escalation stopped reaching the approval gate, which is the
        # state A04's denial test exists to rule out. The grant must also be the recorded outcome.
        if len(peer.outcomes) < 1 or not asked:
            raise FlowError("the tool ran without an answered approval: the call did not reach the "
                            f"approval gate (asked={len(asked)}, answered={len(peer.outcomes)}, "
                            f"peer_error={peer.error!r})")
        if [e for e in decided if (e.get("data") or {}).get("outcome") != "allowed-once"]:
            raise FlowError(f"the session recorded an approval outcome other than the granted one: "
                            f"{[((e.get('data') or {}).get('outcome')) for e in decided]}")
        # The worker must have been given the transcript rather than executing the tool itself: it has
        # no tool-execution path, so a healthy worker is one that only ever answered.
        status, raw, _ = http_json("GET", "127.0.0.1", base_port, "/v1/status")
        activity = json.loads(raw).get("activity") or {}
        results["A03"]["worker_requests"] = activity.get("requests")
        log(f"the worker recorded {activity.get('requests')} request(s) and executed nothing itself")
        print("  PASS A03")
    finally:
        if agent:
            agent.close()
        worker.terminate()
        worker.wait(timeout=20)


def flow_a04(dsh: Path, tmp: Path, profile: str, base_port: int, results: dict) -> None:
    print("[A04] with no answerer that decides, the approval fails closed and the tool does not run")
    ws = tmp / "ws-a04"
    ws.mkdir(parents=True, exist_ok=True)
    marker = ws / "must-not-exist.txt"
    worker = start_worker([TOOL_CALL.format(marker=marker), FINAL_ANSWER],
                          base_port, tmp / "worker-a04.log")
    agent = None
    try:
        wait_for_worker(base_port)
        # `answerer=False` keeps `interactions=answer` off the launch URL. That flag is not what
        # decides this flow -- it is read only on `/connect`, so on `/?token=...` it does nothing, and
        # the peer below can open the event stream either way. What decides the flow is that no peer
        # *grants* the escalation: the one connected peer delegates (`next`), which leaves the
        # composed answerer chain without a terminal answerer and the request resolves the
        # fail-closed `unavailable`.
        agent = Agent(dsh, base_port + 1, tmp / "home-a04", profile, answerer=False, tmp=tmp, cwd=ws, worker_port=base_port)
        # The peer must exist for the outcome to be observable at all. With no peer holding the
        # event stream, the gateway keeps the forwarded waterfall pending until one connects -- the
        # request is then never decided, nothing is `unavailable`, and the turn simply waits, which is
        # a timeout, not a denial.
        peer = agent.answer_approvals("next")
        session = agent.session(ws)
        agent.prompt(session, "Write the marker file using bash.", request_id="a04")
        events = wait_for_turn(session, agent.home)
        summary = summarise(events)

        created = marker.exists()
        asked = [e for e in events if e.get("type") == "approval/asked"]
        outcomes = [((e.get("data") or {}).get("outcome")) for e in events
                    if e.get("type") == "approval/decided"]
        results["A04"] = {"turn_end": summary["end"], "marker_created": created,
                          "answerer": agent.answerer,
                          "approvals_asked": len(asked), "approval_outcomes": outcomes,
                          "grants": outcomes.count("allowed-once")}
        log(f"turn ended as {summary['end']!r}")
        log(f"the marker file exists: {created}")
        log(f"the escalation was asked {len(asked)} time(s) and decided {outcomes}")

        if created:
            raise FlowError("the tool ran even though no answerer granted the escalation: an "
                            "unanswerable approval must fail closed")
        # The denial has to be observed, not assumed. Each check below fails the run for a different
        # way the flow could otherwise pass while proving nothing: the tool never reaching the gate,
        # the gate deciding to grant, or the turn never settling because the request was left pending.
        if not asked:
            raise FlowError("the bash call never reached the approval gate, so this run proves nothing "
                            "about an unanswerable approval: no approval/asked was recorded")
        if not outcomes or any(outcome != "unavailable" for outcome in outcomes):
            raise FlowError(f"the unanswerable escalation did not fail closed as 'unavailable': "
                            f"{outcomes}")
        if summary["end"] != "completed":
            raise FlowError(f"the turn did not settle after the denial (ended as {summary['end']!r}): a "
                            "request left pending is a hang, not a closed decision")
        print("  PASS A04 (the escalation was decided 'unavailable' and the tool did not run)")
    finally:
        if agent:
            agent.close()
        worker.terminate()
        worker.wait(timeout=20)


def flow_a05(dsh: Path, tmp: Path, profile: str, base_port: int, results: dict) -> None:
    print("[A05] Stop settles the turn without the tool having run")
    ws = tmp / "ws-a05"
    ws.mkdir(parents=True, exist_ok=True)
    marker = ws / "cancelled.txt"
    # 0.05 s per token: the tool call alone is ~40 tokens, so the turn is live for seconds.
    worker = start_worker([TOOL_CALL.format(marker=marker), FINAL_ANSWER],
                          base_port, tmp / "worker-a05.log", delay_s=0.05)
    agent = None
    try:
        wait_for_worker(base_port)
        agent = Agent(dsh, base_port + 1, tmp / "home-a05", profile, answerer=True, tmp=tmp, cwd=ws, worker_port=base_port)
        session = agent.session(ws)
        agent.prompt(session, "Write the marker file using bash.", request_id="a05")

        # Wait until the turn is demonstrably live before sending Stop. Without this the scripted turn
        # finishes in milliseconds, Stop arrives after `turn/end`, and the run passes while proving
        # nothing -- which is how this flow passed on its first attempt.
        live = False
        deadline = time.time() + 60
        while time.time() < deadline:
            events = read_events(session, agent.home)
            if any(e.get("type") == "turn/start" for e in events):
                live = True
                break
            time.sleep(0.2)
        if not live:
            raise FlowError("the turn never started, so Stop could not be tested")

        acknowledgement = agent.cancel(session)
        events = wait_for_turn(session, agent.home, timeout=120)
        summary = summarise(events)
        results["A05"] = {"turn_end": summary["end"], "acknowledgement": acknowledgement,
                          "marker_created": marker.exists(), "turn_was_live": live}
        log(f"cancel acknowledged as {json.dumps(acknowledgement)[:120]}")
        log(f"turn ended as {summary['end']!r}")
        if summary["end"] is None:
            raise FlowError("the turn never reached a terminal state after Stop")
        # `aborted` is the Harness's terminal reason for a turn a peer stopped; `cancelled` is the
        # other spelling seen in the vocabulary. Anything else means Stop did not settle the turn.
        if summary["end"] not in ("aborted", "cancelled"):
            raise FlowError(f"the turn ended as {summary['end']!r} after Stop, which is not a "
                            "cancellation")
        if marker.exists():
            raise FlowError("Stop was acknowledged but the tool ran anyway: the turn was not really "
                            "cancelled")
        print(f"  PASS A05 (a live turn was stopped and settled as {summary['end']!r})")
    finally:
        if agent:
            agent.close()
        worker.terminate()
        worker.wait(timeout=20)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsh", required=True, help="path to the dsh executable")
    parser.add_argument("--profile", default="web",
                        help="the base bundle profile the Strata patch layers over")
    parser.add_argument("--base-port", type=int, default=8300)
    parser.add_argument("--only", default=None, help="run one flow: A03, A04 or A05")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args()

    dsh = Path(args.dsh)
    if not dsh.exists():
        raise SystemExit(f"no dsh at {dsh}")
    if not SCRIPTED_OVERLAY.exists():
        raise SystemExit(f"the test-only profile overlay is missing: {SCRIPTED_OVERLAY}")
    for tool in ("zstd",):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is required to read the durable session")

    tmp = Path(tempfile.mkdtemp(prefix="strata-tool-flow-"))
    print(f"working in {tmp}")
    results: dict = {}
    try:
        flows = {"A03": flow_a03, "A04": flow_a04, "A05": flow_a05}
        chosen = [args.only] if args.only else list(flows)
        for index, name in enumerate(chosen):
            flows[name](dsh, tmp, args.profile, args.base_port + index * 10, results)
            print()
        print("PASS")
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    except FlowError as e:
        print(f"\nFAIL {args.only or ''}: {e}")
        print(json.dumps(results, indent=2, sort_keys=True))
        return 1
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
