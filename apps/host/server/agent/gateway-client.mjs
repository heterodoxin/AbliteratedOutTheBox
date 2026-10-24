/**
 * A client for the Harness Gateway (the `dsh web` HTTP + WebSocket surface).
 *
 * Two carriers, both authenticated by the cookie the launch URL's token buys:
 *   - HTTP unary RPC:   POST /api/<method>  with the gateway's envelope
 *   - WebSocket mux:    /api/remote.mux     carrying many independent logical streams
 *
 * The streams used here are `$events` (Host notifications and the approval waterfall) and
 * `session/follow` (a session's durable event log plus live assistant frames).
 *
 * Node 24's global WebSocket accepts a `headers` option, which is how the cookie reaches the upgrade.
 * No dependency is added to the host.
 */

export class GatewayError extends Error {
  constructor(message, { code = 'gateway/error', details } = {}) {
    super(message);
    this.name = 'GatewayError';
    this.code = code;
    this.details = details;
  }
}

export class GatewayClient {
  #base; #token; #cookie = null; #rpcId = 0; #ws = null; #wsReady = null; #streams = new Map(); #nextStream = 0;
  #closed = false; #reconnectTimer = null; #onState;

  /**
   * @param {{baseUrl: string, token: string, onState?: (state: 'open'|'closed') => void}} options
   */
  constructor({ baseUrl, token, onState = () => {} }) {
    this.#base = baseUrl.replace(/\/+$/, '');
    this.#token = token;
    this.#onState = onState;
  }

  /** Exchange the launch token for the API cookie. */
  async connect() {
    const res = await fetch(`${this.#base}/?token=${encodeURIComponent(this.#token)}`, { redirect: 'manual' });
    const cookie = (res.headers.get('set-cookie') ?? '').split(';')[0];
    if (!cookie.startsWith('dsh-auth-')) throw new GatewayError(`the agent refused the launch token (HTTP ${res.status})`, { code: 'gateway/auth' });
    this.#cookie = cookie;
  }

  /** One unary call. Resolves with the value, rejects with a GatewayError carrying the gateway's code. */
  async rpc(method, args = {}, { signal } = {}) {
    if (!this.#cookie) throw new GatewayError('not connected', { code: 'gateway/not-connected' });
    const rpcId = `h${++this.#rpcId}`;
    const res = await fetch(`${this.#base}/api/${method}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', cookie: this.#cookie },
      body: JSON.stringify({ type: 'client-request', rpcId, method, payload: { args } }),
      signal,
    });
    let body;
    try { body = await res.json(); } catch { throw new GatewayError(`${method}: the agent returned a non-JSON answer (HTTP ${res.status})`, { code: 'gateway/bad-response' }); }
    const result = body?.result;
    if (!result?.ok) {
      const e = result?.error ?? {};
      throw new GatewayError(`${method}: ${e.message ?? `HTTP ${res.status}`}`, { code: e.code ?? 'gateway/error', details: e.details });
    }
    return result.value;
  }

  /**
   * Open a logical stream on the mux. Handlers: `onItem(value)`, `onEnd()`, `onError(error)`.
   * The stream is re-opened after a socket reconnect and `onItem` receives it from the start again
   * (a `session/follow` opens with a fresh snapshot, so callers de-duplicate by sequence number).
   */
  openStream(endpoint, args, handlers) {
    const streamId = `s${++this.#nextStream}`;
    const entry = { streamId, endpoint, args, ...handlers, closed: false };
    this.#streams.set(streamId, entry);
    void this.#ensureSocket().then(() => this.#sendOpen(entry), () => {});
    return {
      streamId,
      close: () => {
        entry.closed = true;
        this.#streams.delete(streamId);
        if (this.#ws?.readyState === 1) {
          try { this.#ws.send(JSON.stringify({ type: 'cancel', streamId })); } catch { /* socket going away */ }
        }
      },
    };
  }

  /** Answer one forwarded approval waterfall. `outcome` is `allowed-once`, `rejected`, or `next`. */
  async answerEvent({ clientId, eventId, outcome }) {
    const wire = outcome === 'next' ? { kind: 'next' } : { kind: 'result', value: outcome };
    return this.rpc('$events/result', { clientId, eventId, outcome: wire });
  }

  close() {
    this.#closed = true;
    clearTimeout(this.#reconnectTimer);
    for (const s of this.#streams.values()) s.closed = true;
    this.#streams.clear();
    try { this.#ws?.close(); } catch { /* already closed */ }
    this.#ws = null;
  }

  // -- socket ---------------------------------------------------------------------------------------

  #ensureSocket() {
    if (this.#closed) return Promise.reject(new GatewayError('closed', { code: 'gateway/closed' }));
    if (this.#ws && this.#ws.readyState === 1) return Promise.resolve();
    if (this.#wsReady) return this.#wsReady;
    this.#wsReady = new Promise((resolve, reject) => {
      const url = `${this.#base.replace(/^http/, 'ws')}/api/remote.mux`;
      const ws = new WebSocket(url, { headers: { cookie: this.#cookie } });
      this.#ws = ws;
      ws.onopen = () => { this.#wsReady = null; this.#onState('open'); resolve(); };
      ws.onmessage = e => this.#onMessage(String(e.data));
      ws.onerror = () => { /* the close handler reports it */ };
      ws.onclose = () => {
        const wasReady = this.#wsReady;
        this.#wsReady = null;
        if (this.#ws === ws) this.#ws = null;
        this.#onState('closed');
        if (wasReady) reject(new GatewayError('the agent socket closed before it opened', { code: 'gateway/socket' }));
        if (this.#closed) return;
        // Streams are re-opened by a new socket; a dropped socket must not strand a follower.
        clearTimeout(this.#reconnectTimer);
        if (this.#streams.size) this.#reconnectTimer = setTimeout(() => {
          this.#ensureSocket().then(() => { for (const s of this.#streams.values()) this.#sendOpen(s); }).catch(() => {
            for (const s of [...this.#streams.values()]) s.onError?.(new GatewayError('the agent socket could not be re-opened', { code: 'gateway/socket' }));
          });
        }, 500);
      };
    });
    return this.#wsReady;
  }

  #sendOpen(entry) {
    if (entry.closed || this.#ws?.readyState !== 1) return;
    this.#ws.send(JSON.stringify({ type: 'open', streamId: entry.streamId, endpoint: entry.endpoint, payload: { args: entry.args } }));
  }

  #onMessage(text) {
    let m; try { m = JSON.parse(text); } catch { return; }
    const s = this.#streams.get(m.streamId);
    if (!s) return;
    if (m.type === 'item') s.onItem?.(m.value);
    else if (m.type === 'end') { this.#streams.delete(s.streamId); s.onEnd?.(); }
    else if (m.type === 'error') s.onError?.(new GatewayError(m.error?.message ?? 'stream failed', { code: m.error?.code, details: m.error?.details }));
  }
}
