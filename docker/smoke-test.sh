#!/bin/bash
# Smoke test of one Strata image on a real card (docs/DOCKER.md, "Build it yourself"). Run it under your own GPU lock if the box is shared.
#
#   IMAGE=strata:0.1.45-sycl NAME=strata-t PORT=18080 BASE=http://127.0.0.1:18080 \
#   RUN_FLAGS="--device /dev/dri -p 18080:8080" ENVS="-e MODEL=IQ3_S -e MODEL_PATH=/models -e API_KEY=..." VOLS="-v $HOME/dt/data:/data -v /m:/models:ro" \
#   GPU_USED_CMD='nvidia-smi --query-gpu=memory.used --format=csv,noheader' OUT=$HOME/dt/out docker/smoke-test.sh
#
# Steps: run -> wait for the model (health check) -> GET /v1/models -> pong chat -> tool call -> docker exec strata monitor --once ->
# strata connect opencode --check (from inside) -> docker stop (time, exit code, log) -> GPU memory back, no strata process left.
set -u
: "${IMAGE:?}" "${NAME:?}" "${BASE:?}" "${OUT:?}"
DOCKER=${DOCKER:-docker}
WAIT=${WAIT:-2400}
KEY=${KEY:-test-key-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')}
mkdir -p "$OUT"
LOG="$OUT/test.log"
exec > >(tee -a "$LOG") 2>&1
ts() { date +%H:%M:%S; }
step() { echo; echo "### $(ts) $*"; }
fail=0
ok() { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; fail=1; }
gpu_used() { if [ -n "${GPU_USED_CMD:-}" ]; then bash -c "$GPU_USED_CMD" 2>&1 | head -3 | tr '\n' ' '; else echo n/a; fi; }
procs() { pgrep -af "strata --serve|strata.py (serve|runtime)" | grep -v -E "pgrep|test_container|bash -c" | cut -c1-120; }

step "before: GPU used = $(gpu_used); strata-like host processes:"; procs
$DOCKER rm -f "$NAME" >/dev/null 2>&1
step "docker run"
# shellcheck disable=SC2086
$DOCKER run -d --name "$NAME" --stop-timeout 150 --shm-size 2g --ulimit memlock=-1:-1 -e API_KEY="$KEY" ${ENVS:-} ${VOLS:-} ${RUN_FLAGS:-} "$IMAGE" || { bad "docker run"; exit 1; }
t0=$(date +%s)

step "wait for the model (health check = GET /v1/models with the key), up to ${WAIT}s"
state=""
while :; do
  now=$(date +%s); el=$((now - t0))
  st=$($DOCKER inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' "$NAME" 2>/dev/null)
  if [ "$st" != "$state" ]; then echo "$(ts) +${el}s container: $st"; state=$st; fi
  case "$st" in exited*|dead*) bad "container ended before serving"; $DOCKER logs --tail 60 "$NAME"; exit 1 ;; esac
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $KEY" "$BASE/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && break
  [ "$el" -gt "$WAIT" ] && { bad "not serving after ${WAIT}s"; $DOCKER logs --tail 60 "$NAME"; exit 1; }
  sleep 10
done
echo "serving after $((($(date +%s) - t0)))s"
step "first log lines"; $DOCKER logs "$NAME" 2>&1 | grep -v "^\s*$" | head -40 | cut -c1-200

step "GET /v1/models"
models=$(curl -s -m 10 -H "Authorization: Bearer $KEY" "$BASE/v1/models"); echo "$models" | cut -c1-300
echo "$models" | grep -q '"data"' && ok "GET /v1/models" || bad "GET /v1/models"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/v1/models"); [ "$code" = "401" ] && ok "no key -> 401" || bad "no key gave $code (expected 401)"
MODEL=$(echo "$models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo strata)

step "pong chat"
r=$(curl -s -m 300 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$BASE/v1/chat/completions" \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: pong\"}]}")
echo "$r" | cut -c1-400
echo "$r" | python3 -c 'import sys,json; t=json.load(sys.stdin)["choices"][0]["message"].get("content") or ""; print("answer:",repr(t)); sys.exit(0 if "pong" in t.lower() else 1)' && ok "pong chat" || bad "pong chat"

step "tool call"
r=$(curl -s -m 300 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$BASE/v1/chat/completions" \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":1536,\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the get_weather tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Weather of a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]}")
echo "$r" | cut -c1-500
echo "$r" | python3 -c '
import sys,json
m=json.load(sys.stdin)["choices"][0]["message"]; tc=m.get("tool_calls") or []
print("tool_calls:",[(t["function"]["name"],t["function"]["arguments"]) for t in tc])
sys.exit(0 if tc and tc[0]["function"]["name"]=="get_weather" and "paris" in tc[0]["function"]["arguments"].lower() else 1)' && ok "tool call get_weather(Paris)" || bad "tool call"

step "docker exec strata monitor --once (no key typed: the wrapper supplies it)"
$DOCKER exec "$NAME" strata monitor --once 2>&1 | head -40 && ok "monitor --once ran" || bad "monitor --once"
step "docker exec strata status"
$DOCKER exec "$NAME" strata status 2>&1 | head -12
step "docker exec strata connect opencode --check (inside the container)"
chk=$($DOCKER exec "$NAME" strata connect opencode --check 2>&1); echo "$chk" | tail -8
echo "$chk" | grep -q "^FAIL" && bad "connect opencode --check reported a failure (model answered in words instead of calling the tool? see above)" || ok "connect opencode --check: prompt and tool call"
step "GPU used while serving: $(gpu_used)"
step "docker stats / health"
$DOCKER inspect -f 'health={{.State.Health.Status}}' "$NAME"
$DOCKER stats --no-stream --format 'mem={{.MemUsage}} cpu={{.CPUPerc}}' "$NAME"

step "docker stop (clean stop)"
s0=$(date +%s)
$DOCKER stop "$NAME"
s1=$(date +%s)
ec=$($DOCKER inspect -f '{{.State.ExitCode}}' "$NAME")
echo "stop took $((s1 - s0))s, exit code $ec"
[ "$ec" = "0" ] && ok "exit code 0 after docker stop" || bad "exit code $ec after docker stop"
$DOCKER logs --tail 12 "$NAME" 2>&1 | cut -c1-200
sleep 3
echo "GPU used after stop: $(gpu_used)"
leftover=$(procs); [ -z "$leftover" ] && ok "no strata process left on the host" || { bad "processes left:"; echo "$leftover"; }

if [ "${RESTART:-1}" = "1" ]; then
  step "docker start (second start: the preset in /data is reused, no download)"
  t1=$(date +%s); $DOCKER start "$NAME" >/dev/null
  while :; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $KEY" "$BASE/v1/models" 2>/dev/null)
    [ "$code" = "200" ] && break
    [ $(($(date +%s) - t1)) -gt "$WAIT" ] && { bad "second start not serving"; break; }
    sleep 10
  done
  echo "second start serving after $(($(date +%s) - t1))s"; [ "$code" = "200" ] && ok "restart reuses /data"
  $DOCKER logs --tail 25 "$NAME" 2>&1 | grep -E "Config|preset|Serving|Starting|installed|tuned" | cut -c1-200
  $DOCKER stop "$NAME" >/dev/null; echo "second stop exit code $($DOCKER inspect -f '{{.State.ExitCode}}' "$NAME"); GPU used $(gpu_used)"
fi
step "RESULT: $([ $fail = 0 ] && echo ALL PASS || echo SOME FAILED)"
exit $fail
