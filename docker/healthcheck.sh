#!/bin/sh
# Container health: GET /v1/models with the API key. Healthy only when the server answers and has a model loaded.
# While the model downloads or loads (can take hours the first time) the container is "starting", not "unhealthy".
KEY_FILE="${STRATA_KEY_FILE:-/data/state/api-key}"
KEY="${API_KEY:-}"
[ -z "$KEY" ] && [ -r "$KEY_FILE" ] && KEY=$(cat "$KEY_FILE")
exec /opt/strata/terminal/python/bin/python3 - "http://127.0.0.1:${PORT:-8080}/v1/models" "$KEY" <<'PY'
import sys, urllib.request, json
url, key = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key} if key else {})
try:
    with urllib.request.urlopen(req, timeout=4) as r:
        data = json.load(r)
        sys.exit(0 if r.status == 200 and data.get("data") else 1)
except Exception as e:
    print("not healthy: %s" % e, file=sys.stderr)
    sys.exit(1)
PY
