import { userError } from './format';
/** Thin HTTP client for the Strata host. Zero deps, works in browser and Electron renderer. */

const BASE = (globalThis as { __STRATA_HOST__?: string }).__STRATA_HOST__ ?? '';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(userError(message));
    this.name = 'ApiError';
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body === undefined ? undefined : { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  const data = text ? (JSON.parse(text) as unknown) : undefined;
  if (!res.ok) {
    const message =
      (data as { error?: string } | undefined)?.error ?? `Request failed (${res.status} ${res.statusText})`;
    throw new ApiError(message, res.status, data);
  }
  return data as T;
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
  patch: <T>(path: string, body?: unknown) => request<T>('PATCH', path, body),
  put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body),
  del: <T>(path: string) => request<T>('DELETE', path),
};

export interface StreamEvent {
  event: string;
  data: Record<string, unknown>;
}

/** Subscribe to the host event stream. Returns an unsubscribe function. */
export function onEvent(handler: (e: StreamEvent) => void, onStatus?: (s: 'live' | 'offline') => void) {
  let source: EventSource;
  let seenAt = Date.now();
  const names = ['engine', 'monitor', 'models', 'native', 'image-models', 'runtime', 'providers', 'serving', 'download', 'log', 'setup', 'skills', 'task', 'goal', 'timeskip', 'plan', 'queue'];
  const connect = () => {
    source?.close();
    seenAt = Date.now();
    source = new EventSource(`${BASE}/api/events`);
    for (const name of names) {
      source.addEventListener(name, ((ev: MessageEvent) => {
        seenAt = Date.now();
        try { handler({ event: name, data: JSON.parse(ev.data) as Record<string, unknown> }); }
        catch { /* Ignore malformed frames without losing valid state. */ }
      }) as EventListener);
    }
    source.addEventListener('open', () => { seenAt = Date.now(); onStatus?.('live'); });
    source.addEventListener('error', () => onStatus?.('offline'));
    source.addEventListener('message', ev => {
      seenAt = Date.now();
      try { handler({ event: 'message', data: JSON.parse((ev as MessageEvent).data) as Record<string, unknown> }); }
      catch { /* Ignore malformed frames. */ }
    });
  };
  connect();
  // The host sends monitor frames each second. Some proxies leave a dead SSE
  // socket open, so silence must also invalidate readings and retry the stream.
  const watchdog = setInterval(() => {
    if (Date.now() - seenAt <= 10000) return;
    onStatus?.('offline');
    connect();
  }, 5000);
  return () => { clearInterval(watchdog); source.close(); };
}

/** POST a JSON body and consume an NDJSON/SSE-ish stream of JSON frames. */
export async function streamPost(
  path: string,
  body: unknown,
  onFrame: (frame: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => '');
    throw new ApiError(text || `Stream failed (${res.status})`, res.status);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      try {
        onFrame(JSON.parse(line) as Record<string, unknown>);
      } catch {
        /* ignore partial frame */
      }
    }
  }
  if (buffer.trim()) {
    try {
      onFrame(JSON.parse(buffer.trim()) as Record<string, unknown>);
    } catch {
      /* ignore */
    }
  }
}
