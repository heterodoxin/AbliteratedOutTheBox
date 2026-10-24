# Running Strata behind llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) puts several model servers behind one address and starts the one a
request names, stopping the others to free the GPU. Strata works as one of its models. This page is what we needed to
get it running; it was tested with llama-swap's default stop behaviour, Strata 0.1.39 (Coder IQ1_M on a Tesla V100) and
clients reaching the machine by its host name.

## The entry

Set Strata up as usual first (`./setup.sh --no-start ...`), then point llama-swap at the server directly. The start script
setup writes (`run-<model>.sh`) fixes the port at 8080, so call `serve/server.py` with llama-swap's port instead:

```yaml
healthCheckTimeout: 300            # Strata takes ~1.5-2 min to load; give it room

models:
  strata:
    cmd: >-
      sh -c 'cd /path/to/Strata && exec .venv/bin/python serve/server.py
      --engine strata --config strata-coder-iq1_m.json --port ${PORT}'
    env: ["STRATA_ALLOWED_HOSTS=my-server"]     # the host name your clients use -- see below
    checkEndpoint: /health
    ttl: 1800
```

- `exec` makes the Python server the process llama-swap stops. On llama-swap's stop (SIGTERM) the server stops its engine:
  in our test the engine was gone and the V100's memory freed within 2 s.
- `checkEndpoint: /health` -- Strata's `/health` answers only once the model is loaded. Measured load time on our machine
  (model files on a SATA SSD): 80 s, 105 s while another download was using the disk.
- If other models share the GPU, put them in a llama-swap group or let swapping unload them; Strata needs the card's
  memory to itself.

## The trap: "Host ... is not allowed"

The first request through llama-swap failed at once with:

```text
403 Host 'my-server:8040' is not allowed (DNS rebinding protection)
```

Strata only answers to the host names it knows, so a web page cannot reach it by pointing a DNS name at 127.0.0.1.
llama-swap passes the client's Host header through, so a client that reaches the proxy by name (`http://my-server:8040`)
is refused. Requests by IP address and to `localhost` always pass, which is why it can work in a quick test and fail
from another machine. Any one of these fixes it:

- `STRATA_ALLOWED_HOSTS=my-server` in the entry's `env` (comma-separated names; ports are ignored), or
- `"allowed_hosts": ["my-server"]` in `strata-<model>.json`, or
- an `"api_key"`, which turns the check off -- then every client must send the key.

## Strata's own pages through llama-swap

llama-swap forwards `/upstream/<model>/...` unchanged, so Strata's status and health pages stay reachable, and calling one
also loads the model without sending a chat request:

```bash
curl http://my-server:8040/upstream/strata/health
curl http://my-server:8040/upstream/strata/v1/status
```

## Claude Code and other Anthropic-API clients: the thinking level

Claude Code (and the Claude Agent SDK) send `"output_config": {"effort": "high"}` with every request unless told
otherwise, so Strata thinks at its highest level (seen by logging the requests). `CLAUDE_CODE_EFFORT_LEVEL=medium` (or
`low`) in the client's environment changes it. One measurement, for scale only: on one real coding task in our setup,
`medium` finished in 6.5 min against 9.7 min at `high`, both correct; `low` took 10.1 min (more trial-and-fix steps).
One run each -- not a general result.
