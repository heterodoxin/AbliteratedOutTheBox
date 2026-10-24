/**
 * Does a model server read pictures? Asked of the server itself, cached for a few minutes:
 *  - its model list (`/models`): `architecture.input_modalities`, `input_modalities`, `modalities`, or a `capabilities` list that names image / vision / multimodal;
 *  - a llama-server's `/props` (`modalities.vision`), which is how `llama-server --mmproj` says it can read pictures (its `/v1/models` does not).
 * Unknown or unreachable counts as "no" (a picture sent to a server without a vision encoder makes the whole request fail), but the person can
 * always tick "This model reads pictures" in the model's settings: that answer wins over anything asked here.
 */
const TTL_MS = 5 * 60_000;

/** True / false / null (the row does not say) from one entry of a server's model list. */
export function modelReadsPictures(row) {
  if (!row || typeof row !== 'object') return null;
  const lists = [row.architecture?.input_modalities, row.input_modalities, row.modalities?.input, Array.isArray(row.modalities) ? row.modalities : null];
  for (const list of lists) if (Array.isArray(list)) return list.some(x => /^(image|vision|images)$/i.test(String(x)));
  if (row.modalities && typeof row.modalities === 'object' && typeof row.modalities.vision === 'boolean') return row.modalities.vision;
  const caps = row.capabilities ?? row.meta?.capabilities;
  if (Array.isArray(caps)) return caps.some(x => /vision|image|multimodal/i.test(String(x)));
  if (caps && typeof caps === 'object' && typeof caps.vision === 'boolean') return caps.vision;
  return null;
}

/**
 * Does a model list row say the model can think (reasoning)? OpenRouter lists `supported_parameters` per model: `reasoning` or
 * `include_reasoning` in it means the model takes a reasoning effort. True / false / null (the row does not say).
 */
export function modelSupportsReasoning(row) {
  const list = row?.supported_parameters ?? row?.supportedParameters;
  if (!Array.isArray(list)) return null;
  return list.some(x => /^(include_)?reasoning$/i.test(String(x)));
}

/**
 * Pictures are ON by default (owner rule): a model whose server does not say either way is sent pictures; only an explicit "no" (the person's
 * own switch, or the server's own list or /props) turns them off.
 */
export function readsPicturesByDefault(model, asked) {
  return effectiveVision(model, asked) ?? true;
}

/** The effective answer for a saved model: what the person ticked, else what the server said (undefined: never asked). */
export function effectiveVision(model, asked) {
  if (model?.vision === true) return true;
  if (model?.vision === false) return false;
  return asked;
}

export function createVisionProbe({ fetcher = fetch, now = Date.now } = {}) {
  const cache = new Map();   // key -> { at, value }
  const keyOf = (conn, modelId) => `${conn.id}|${modelId}`;
  return {
    /** Last known answer for a model (true / false), or undefined when never asked. */
    known: (conn, modelId) => { const e = cache.get(keyOf(conn, modelId)); return e ? e.value : undefined; },
    /** Whether the model takes a reasoning effort, as its server's list said (true / false), null when it did not say, undefined when never asked. */
    knownThinks: (conn, modelId) => { const e = cache.get(keyOf(conn, modelId)); return e ? e.thinks : undefined; },
    /** Ask now (at most once per TTL per model). Answers true / false / null (the server did not say). */
    async check(conn, modelId) {
      const k = keyOf(conn, modelId); const hit = cache.get(k);
      if (hit && now() - hit.at < TTL_MS) return hit.value;
      let value = null, thinks = null;
      const base = String(conn.baseUrl).replace(/\/+$/, '');
      const headers = conn.apiKey ? { authorization: `Bearer ${conn.apiKey}` } : {};
      const get = async path => { const res = await fetcher(base + path, { headers, signal: AbortSignal.timeout(4000) }); return res.ok ? res.json() : null; };
      try {
        const list = (await get('/models'))?.data ?? [];
        const row = list.find(m => m?.id === modelId) ?? (list.length === 1 ? list[0] : null);
        value = modelReadsPictures(row);
        thinks = modelSupportsReasoning(row);
        if (value === null) {            // the list did not say: a llama-server tells in /props (next to /v1, not under it)
          try {
            const props = await (async () => { const res = await fetcher(base.replace(/\/v1$/, '') + '/props', { headers, signal: AbortSignal.timeout(4000) }); return res.ok ? res.json() : null; })();
            value = props?.modalities?.vision === true ? true : props?.modalities?.vision === false ? false : null;
          } catch { /* no /props: not a llama-server */ }
        }
      } catch { value = null; }
      cache.set(k, { at: now(), value, thinks });
      return value;
    },
  };
}
