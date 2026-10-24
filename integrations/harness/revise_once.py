#!/usr/bin/env python3
"""One bounded Harness revision attempt, for Time-skip's `--revise-command`.

Time-skip owns the job: it checkpoints, isolates a candidate directory, and decides from checks.
This script owns only the *revision step* — it asks the pinned DeepSeek Harness for one bounded
agent turn inside the candidate directory and returns when the turn reaches a terminal state.

```
strata timeskip run --workspace DIR --prompt "..." \
    --revise-command "python integrations/harness/revise_once.py"
```

The contract Time-skip depends on:

* the process runs with the **candidate directory** as its cwd, and only ever writes there;
* the prompt arrives in `STRATA_TIMESKIP_PROMPT` and the attempt number in `STRATA_TIMESKIP_ATTEMPT`;
* exit 0 means "an attempt was made" — *not* "the attempt succeeded". Whether it improved anything is
  decided by the checks, never by this script. A non-zero exit reports that the attempt itself could
  not be made (for example the Harness is not installed), which Time-skip records as a failed run
  rather than a rejected candidate.

A bounded turn means all of: a wall-clock deadline, a per-attempt session, and a Harness that is
stopped when the deadline passes — so a stuck agent cannot hold a Time-skip run open forever.
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE))

from _bridge import (  # noqa: E402
    DEFAULT_DSH,
    PROFILE_PATCH,
    BridgeError,
    gateway_call,
    http_json,
    read_session_events,
    wait_for_agent_url,
)

#: The whole attempt's wall-clock bound. Time-skip's own phases are unbounded by design; the bound
#: belongs to the thing that can run away, which is the agent turn.
DEFAULT_DEADLINE_S = 240.0

INSTRUCTIONS = """You are revising a copy of a user's work. Follow these rules exactly.

1. Work only inside the current directory. It is an isolated copy: the user's original is safe and
   you cannot see it. Never try to read or write outside this directory.
2. Make the smallest change that achieves the goal. Do not reorganise, rename or reformat anything
   that is not needed for the goal.
3. Do not add commentary files, notes, README files or explanations of what you did. Change the
   work itself.
4. If the goal cannot be achieved from what is here, change nothing and say so in one sentence.

When you are finished, reply with a single short sentence describing what you changed."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsh", default=os.environ.get("DSH_CLI", DEFAULT_DSH))
    ap.add_argument("--workspace", default=None,
                    help="the candidate directory (default: the current directory)")
    ap.add_argument("--deadline", type=float,
                    default=float(os.environ.get("STRATA_TIMESKIP_DEADLINE_S", DEFAULT_DEADLINE_S)))
    ap.add_argument("--keep", action="store_true",
                    help="keep the per-attempt Harness home for inspection")
    ap.add_argument("--print-session", action="store_true",
                    help="print the session id and its event types, for diagnosis")
    args = ap.parse_args()

    candidate = Path(args.workspace or os.getcwd()).resolve()
    prompt = os.environ.get("STRATA_TIMESKIP_PROMPT", "").strip()
    attempt = os.environ.get("STRATA_TIMESKIP_ATTEMPT", "1")
    if not prompt:
        print("revise_once: STRATA_TIMESKIP_PROMPT is empty; nothing to ask for", file=sys.stderr)
        return 2
    if not candidate.is_dir():
        print(f"revise_once: {candidate} is not a directory", file=sys.stderr)
        return 2

    dsh = Path(args.dsh)
    if not dsh.exists():
        print(f"revise_once: the Harness CLI is not at {dsh}; set DSH_CLI or --dsh", file=sys.stderr)
        return 3
    if not PROFILE_PATCH.exists():
        print(f"revise_once: the Strata profile is not at {PROFILE_PATCH}", file=sys.stderr)
        return 3

    # Each attempt gets its own Harness home. That keeps sessions and approvals scoped to the
    # attempt, and means a crashed attempt cannot corrupt the next one's state.
    home = Path(tempfile.mkdtemp(prefix=f"strata-revise-a{attempt}-"))
    (home / "sessions").mkdir(parents=True, exist_ok=True)
    agent = None
    deadline = time.time() + args.deadline
    try:
        env = dict(os.environ)
        env["DSH_HOME"] = str(home)
        log = (home / "dsh.log").open("w")
        agent = subprocess.Popen(
            [str(dsh), "--profile", "web", "--patch", str(PROFILE_PATCH), "--port", "0", "--no-open"],
            cwd=str(candidate), env=env, stdout=log, stderr=subprocess.STDOUT)
        text = wait_for_agent_url(home / "dsh.log", agent, timeout=min(120.0, args.deadline))
        m = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9_-]+)", text)
        if not m:
            raise BridgeError("the Harness printed no usable URL")
        port, token = int(m.group(1)), m.group(2)

        # `interactions=answer` is required, or every approval request fails closed as
        # 'unavailable'. A revision attempt that needs to run a command must be able to be approved.
        _, _, set_cookie = http_json("GET", "127.0.0.1", port,
                                     f"/?token={token}&interactions=answer")
        cookie = set_cookie.split(";", 1)[0]
        if not cookie.startswith("dsh-auth-"):
            raise BridgeError("no operator cookie was issued")

        created = gateway_call("127.0.0.1", port, cookie, "c1", "session/create",
                               {"request": {"cwd": str(candidate)}})
        session_id = created["sessionId"]
        gateway_call("127.0.0.1", port, cookie, "p1", "session/prompt",
                     {"request": {"requestId": f"timeskip-{attempt}", "sessionId": session_id,
                                  "mode": "queue",
                                  "content": [{"type": "text",
                                               "text": f"{INSTRUCTIONS}\n\nGoal:\n{prompt}"}]}})

        # Wait for a terminal state, or the deadline, whichever comes first.
        events: list[dict] = []
        note = ""
        while time.time() < deadline:
            events = read_session_events(session_id, home=home)
            end = next((e for e in events if e.get("type") == "turn/end"), None)
            if end is not None:
                reason = (end.get("data") or {}).get("reason") or {}
                if reason.get("kind") != "completed":
                    print(f"revise_once: the turn ended as {json.dumps(reason)[:200]}", file=sys.stderr)
                    return 1
                assistant = next((e for e in events if e.get("type") == "assistant/message"), None)
                if assistant:
                    message = (assistant.get("data") or {}).get("message") or {}
                    parts = message.get("content") or []
                    note = " ".join(p.get("text", "") for p in parts
                                    if isinstance(p, dict) and p.get("type") in ("text", "reasoning"))
                break
            time.sleep(0.5)
        else:
            print(f"revise_once: the attempt did not finish within {args.deadline:.0f}s; "
                  "stopping the Harness and reporting a made attempt", file=sys.stderr)
            return 1

        if args.print_session:
            print(f"session {session_id}: {[e.get('type') for e in events]}", file=sys.stderr)
        # Exit 0 means an attempt was made. What it achieved is the checks' business.
        print((note or f"attempt {attempt} completed").strip()[:500])
        return 0

    except BridgeError as e:
        print(f"revise_once: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - report, and let Time-skip record a failed run
        print(f"revise_once: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if agent is not None and agent.poll() is None:
            try:
                agent.send_signal(signal.SIGTERM)
                agent.wait(timeout=8)
            except Exception:
                try:
                    agent.kill()
                except Exception:
                    pass
        if not args.keep:
            shutil.rmtree(home, ignore_errors=True)
        else:
            print(f"revise_once: kept {home}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
