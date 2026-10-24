/**
 * The engine's own answer speed for one assistant turn: every token its requests produced (thinking, answer and tool calls; with
 * MTP drafting only the tokens it kept) over the time they spent decoding. These are the numbers the Monitor and `/metrics` show,
 * so the chat footer reads the same. Nothing here is a guess from the stream: it returns null when the engine does not say.
 */

/** Sum the engine's finished requests that began at or after `since` (ms). `attempts` is how many model requests the turn made:
 *  when the engine lists more (another chat ran at the same time), only the newest `attempts` are the turn's. */
export function sumEngineRequests(requests, { since, attempts = Infinity } = {}) {
  if (!Array.isArray(requests)) return null;
  const mine = requests
    .filter(r => Number.isFinite(r?.time) && r.time * 1000 >= since - 1500)
    .sort((a, b) => a.time - b.time)
    .slice(-Math.max(1, attempts));
  let tokens = 0, decodeMs = 0, promptRead = 0, promptMs = 0;
  for (const r of mine) {
    const made = Number.isFinite(r.engine_generated) ? r.engine_generated : r.output_tokens;
    if (!(made > 0) || !(r.decode_ms > 0)) continue;
    tokens += made; decodeMs += r.decode_ms;
    const read = Number.isFinite(r.prompt_read) ? r.prompt_read : (r.prompt_tokens ?? 0) - (r.reused ?? 0);
    if (read > 0 && r.prompt_ms > 0) { promptRead += read; promptMs += r.prompt_ms; }
  }
  if (!(tokens > 0 && decodeMs > 0)) return null;
  return {
    tokens, decodeMs, tps: Number((tokens / (decodeMs / 1000)).toFixed(1)),
    promptTps: promptRead > 0 && promptMs > 0 ? Number((promptRead / (promptMs / 1000)).toFixed(1)) : null,
  };
}

/** Ask a local engine for its request history. Any failure is null: the caller keeps the stream's own estimate. */
export async function turnEngineSpeed({ baseUrl, key, since, attempts, fetcher = fetch, timeoutMs = 1500 }) {
  try {
    const origin = new URL(baseUrl).origin;
    const res = await fetcher(`${origin}/metrics?requests=all`, {
      signal: AbortSignal.timeout(timeoutMs),
      headers: key ? { authorization: `Bearer ${key}` } : undefined,
    });
    if (!res.ok) return null;
    return sumEngineRequests((await res.json())?.requests, { since, attempts });
  } catch { return null; }
}
