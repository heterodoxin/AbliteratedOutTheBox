/**
 * Response settings for agent turns.
 *
 * The chat's own path sends the Expert sampling (temperature, top-p, top-k, repeat penalty, seed) and the composer's thinking budget with
 * every request. The agent's requests are made by the Harness, which has no place for either, so they never reached the model. When a
 * person has changed one of them, the agent's model route goes through the small forwarding endpoint below (on this computer only), which
 * adds exactly the fields that differ from the defaults and passes everything else through untouched. With every setting at its default the
 * agent talks to the model directly, byte for byte as before.
 *
 * Only a local model, or an OpenAI-style server on this computer or this network, is shaped: a cloud service would reject fields it does not know.
 */

import http from 'node:http';
import https from 'node:https';

/** The thinking budgets the composer offers (0 = no limit is not a budget). */
export const BUDGETS = Object.freeze([2048, 8192, 32768]);
/** The offered budget nearest to what a chat asks for (0 stays 0). */
export function nearestBudget(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 0;
  return BUDGETS.reduce((best, b) => (Math.abs(b - n) < Math.abs(best - n) ? b : best), BUDGETS[0]);
}

/** Expert setting id -> request field, and the default each one has (the catalog's recommended value). */
const SAMPLING = Object.freeze({
  'sample.temp': ['temperature', 0.6],
  'sample.topP': ['top_p', 0.95],
  'sample.topK': ['top_k', 40],
  'sample.repeat': ['repeat_penalty', 1.05],
  'sample.seed': ['seed', -1],
});

/** The request fields for the Expert settings that are not at their default, or null when all are. */
export function samplingFromExpert(expert) {
  const out = {};
  for (const [id, [field, recommended]] of Object.entries(SAMPLING)) {
    const v = expert?.[id];
    if (typeof v === 'number' && Number.isFinite(v) && v !== recommended) out[field] = v;
  }
  if (expert?.['sample.speedProjection'] === false) out.experimental_speed_projection = false;
  const cap = Number(expert?.['sample.maxTokens']);
  if (Number.isFinite(cap) && cap > 0) out.max_tokens = Math.floor(cap);
  return Object.keys(out).length ? out : null;
}

/**
 * The request body with the person's settings added. A field the request already carries is left alone (the Harness's own wins), the
 * thinking budget is only added while thinking is on, and an output cap only ever lowers the request's own.
 */
export function shapeBody(body, { sampling = null, budget = 0 } = {}) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return body;
  const out = { ...body };
  for (const [field, value] of Object.entries(sampling ?? {})) {
    if (field === 'max_tokens' || field === 'max_completion_tokens') continue;
    if (out[field] === undefined) out[field] = value;
  }
  if (sampling?.max_tokens) {
    for (const key of ['max_tokens', 'max_completion_tokens']) if (Number.isFinite(out[key])) out[key] = Math.min(out[key], sampling.max_tokens);
    if (out.max_tokens === undefined && out.max_completion_tokens === undefined) out.max_tokens = sampling.max_tokens;
  }
  const thinkingOff = out.chat_template_kwargs?.enable_thinking === false;
  if (budget > 0 && !thinkingOff && out.reasoning_budget_tokens === undefined) out.reasoning_budget_tokens = budget;
  return out;
}

/** An address on this computer or on a private network: where the person's own model servers live. */
export function isPrivateHost(url) {
  let host;
  try { host = new URL(url).hostname.toLowerCase(); } catch { return false; }
  host = host.replace(/^\[|\]$/g, '');
  if (host === 'localhost' || host === '::1' || host.endsWith('.local') || host.endsWith('.lan')) return true;
  const m = /^(\d+)\.(\d+)\.(\d+)\.(\d+)$/.exec(host);
  if (!m) return false;
  const [a, b] = [Number(m[1]), Number(m[2])];
  return a === 127 || a === 10 || (a === 192 && b === 168) || (a === 172 && b >= 16 && b <= 31) || (a === 169 && b === 254) || (a === 100 && b >= 64 && b <= 127);
}

const HOP = new Set(['host', 'connection', 'content-length', 'transfer-encoding', 'keep-alive', 'upgrade', 'proxy-connection']);

/**
 * The forwarding endpoint. `targetOf(key)` gives the real base address of a shaped route (or null), `samplingNow()` the Expert fields that
 * differ from the defaults right now. Called for `/api/internal/agent-llm/<key>/<path>`; this computer's own address only.
 */
export function createLlmProxy({ targetOf, samplingNow }) {
  return function handle(req, res, key, rest) {
    const remote = req.socket?.remoteAddress ?? '';
    if (!/^(::1|127\.|::ffff:127\.)/.test(remote)) { res.writeHead(403, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: 'This endpoint is for the assistant on this computer only.' })); return; }
    const base = targetOf(key);
    if (!base) { res.writeHead(404, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: 'Unknown model route.' })); return; }
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('error', () => res.destroy());
    req.on('end', () => {
      let body = Buffer.concat(chunks);
      const budget = nearestBudget(req.headers['x-strata-budget']);
      if (req.method === 'POST' && /\/chat\/completions\/?$/.test(rest) && body.length) {
        try { body = Buffer.from(JSON.stringify(shapeBody(JSON.parse(body.toString('utf8')), { sampling: samplingNow(), budget }))); } catch { /* not JSON: pass it on untouched */ }
      }
      // The route's address is the model's own base (it ends in /v1): the assistant's /v1 is that same prefix.
      const target = new URL(base.replace(/\/+$/, '') + rest.replace(/^\/v1(?=\/|$|\?)/, ''));
      const headers = {};
      for (const [k, v] of Object.entries(req.headers)) if (!HOP.has(k) && !k.startsWith('x-strata-')) headers[k] = v;
      headers['content-length'] = String(body.length);
      const lib = target.protocol === 'https:' ? https : http;
      const upstream = lib.request(target, { method: req.method, headers }, reply => {
        const out = { ...reply.headers }; delete out['transfer-encoding']; delete out.connection;
        res.writeHead(reply.statusCode ?? 502, out);
        reply.pipe(res);
        reply.on('error', () => res.destroy());
      });
      upstream.on('error', e => { if (!res.headersSent) { res.writeHead(502, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: `The model server could not be reached: ${e.message}` })); } else res.destroy(); });
      // The assistant stopped waiting (Stop): stop the model too.
      res.on('close', () => { if (!res.writableEnded) upstream.destroy(); });
      upstream.end(body);
    });
  };
}
