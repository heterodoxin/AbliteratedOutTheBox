/** Saved API connections. Secrets never leave the host's private credentials file. */
import { modelReadsPictures, modelSupportsReasoning } from './vision-probe.mjs';
import { existsSync, readFileSync, writeFileSync, renameSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { makePrivate } from './private-file.mjs';

export const PROVIDER_CATALOG = [
  { id: 'openai', name: 'OpenAI', baseUrl: 'https://api.openai.com/v1', protocol: 'openai-responses' },
  { id: 'anthropic', name: 'Anthropic', baseUrl: 'https://api.anthropic.com/v1', protocol: 'anthropic' },
  { id: 'deepseek', name: 'DeepSeek', baseUrl: 'https://api.deepseek.com', protocol: 'openai-chat' },
  { id: 'openrouter', name: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', protocol: 'openai-chat' },
  { id: 'baseten', name: 'Baseten', baseUrl: 'https://inference.baseten.co/v1', protocol: 'openai-chat' },
  { id: 'cerebras', name: 'Cerebras', baseUrl: 'https://api.cerebras.ai/v1', protocol: 'openai-chat' },
  { id: 'fireworks', name: 'Fireworks', baseUrl: 'https://api.fireworks.ai/inference/v1', protocol: 'openai-chat' },
  { id: 'google', name: 'Google Gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', protocol: 'openai-chat' },
  { id: 'groq', name: 'Groq', baseUrl: 'https://api.groq.com/openai/v1', protocol: 'openai-chat' },
  { id: 'huggingface', name: 'Hugging Face', baseUrl: 'https://router.huggingface.co/v1', protocol: 'openai-chat' },
  { id: 'minimax', name: 'MiniMax', baseUrl: 'https://api.minimax.io/v1', protocol: 'openai-chat' },
  { id: 'mistral', name: 'Mistral', baseUrl: 'https://api.mistral.ai/v1', protocol: 'openai-chat' },
  { id: 'moonshotai', name: 'Moonshot AI · Kimi', baseUrl: 'https://api.moonshot.ai/v1', protocol: 'openai-chat' },
  { id: 'nvidia', name: 'NVIDIA', baseUrl: 'https://integrate.api.nvidia.com/v1', protocol: 'openai-chat' },
  { id: 'qwen', name: 'Qwen · Model Studio', baseUrl: '', protocol: 'openai-chat', baseUrlHint: 'Paste the OpenAI compatible base URL for your workspace and region from Model Studio.' },
  { id: 'qwen-us', name: 'Qwen · US', baseUrl: 'https://dashscope-us.aliyuncs.com/compatible-mode/v1', protocol: 'openai-chat' },
  { id: 'together', name: 'Together AI', baseUrl: 'https://api.together.ai/v1', protocol: 'openai-chat' },
  { id: 'vercel-ai-gateway', name: 'Vercel AI Gateway', baseUrl: 'https://ai-gateway.vercel.sh/v1', protocol: 'openai-chat' },
  { id: 'xai', name: 'xAI', baseUrl: 'https://api.x.ai/v1', protocol: 'openai-responses' },
  { id: 'zai', name: 'Z.AI', baseUrl: 'https://api.z.ai/api/paas/v4', protocol: 'openai-chat' },
];
const PROTOCOLS = ['openai-chat', 'openai-responses', 'anthropic'];
const fail = message => { throw new Error(message); };
const text = (value, max = 160) => typeof value === 'string' ? value.trim().slice(0, max) : '';
/** The answer speed of a model API's reply: the tokens it reports over the time from its first streamed token to the end, never over the whole
 *  wait (that would put the time to the first token, often seconds of prompt reading, into the speed). null when it reported no tokens;
 *  with no usable first-token time (or under a quarter of a second) the whole time is the best there is. */
export function answerSpeed(tokensOut, durationMs, firstTokenMs) {
  if (!Number.isFinite(tokensOut) || tokensOut <= 0) return null;
  const decodeMs = Number.isFinite(firstTokenMs) ? durationMs - firstTokenMs : NaN;
  const ms = decodeMs >= 250 ? decodeMs : durationMs;
  return Number((tokensOut / Math.max(0.001, ms / 1000)).toFixed(1));
}

export function normalizeBaseUrl(value) {
  let url; try { url = new URL(text(value, 2048)); } catch { fail('Enter a full http:// or https:// API address.'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) fail('Use an HTTP or HTTPS base URL without credentials, a query, or a fragment.');
  if (/\/(chat\/completions|responses|messages|models)\/?$/.test(url.pathname)) fail('Enter the API base URL, without /chat/completions, /responses, /messages, or /models.');
  return url.href.replace(/\/+$/, '');
}
const publicConnection = ({ apiKey, ...connection }) => ({ ...connection, apiKeySet: !!apiKey });
const numeric = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;

export function createModelProviders({ directory, changed = () => {}, fetchImpl = fetch }) {
  mkdirSync(directory, { recursive: true });
  const file = join(directory, 'model-providers.json');
  let connections = [];
  if (existsSync(file)) {
    // Do not silently erase connections if a credentials file needs repair.
    const saved = JSON.parse(readFileSync(file, 'utf8'));
    if (!Array.isArray(saved)) fail('The model connections file is invalid.');
    connections = saved; makePrivate(file);
  }
  function list() { return { catalog: PROVIDER_CATALOG, connections: connections.map(publicConnection) }; }
  function persist(next) {
    const temp = `${file}.${randomUUID()}.tmp`;
    writeFileSync(temp, JSON.stringify(next, null, 2), { mode: 0o600 });
    makePrivate(temp); renameSync(temp, file); makePrivate(file);
    connections = next; changed(list()); return list();
  }
  function draft(body) {
    const existing = body.id ? connections.find(c => c.id === body.id) : null;
    if (body.id && !existing) fail('This connection no longer exists. Close the editor and try again.');
    const kind = body.kind === 'custom' ? 'custom' : 'provider';
    const catalog = kind === 'provider' ? PROVIDER_CATALOG.find(p => p.id === body.provider) : null;
    if (kind === 'provider' && !catalog) fail('Choose a model provider.');
    const protocol = body.protocol ?? catalog?.protocol;
    if (!PROTOCOLS.includes(protocol)) fail('Choose a supported API protocol.');
    const name = text(body.name) || catalog?.name;
    if (!name) fail('Give this connection a display name.');
    const baseUrl = normalizeBaseUrl(body.baseUrl || catalog?.baseUrl);
    const sameDestination = existing && existing.baseUrl === baseUrl && existing.protocol === protocol;
    // An unchanged blank key means keep it; changing the destination requires a new key.
    const apiKey = body.clearKey ? '' : text(body.apiKey, 4096) || (sameDestination ? existing.apiKey : '') || '';
    if (/[\r\n]/.test(apiKey)) fail('The API key contains an invalid character.');
    if (kind === 'provider' && !apiKey) fail('Enter this provider’s API key.');
    const rows = Array.isArray(body.models) ? body.models : [];
    if (rows.length > 100) fail('Add up to 100 models to one connection.');
    const models = rows.map(row => {
      const id = text(row.id, 256); if (!id || /[\r\n]/.test(id)) fail('Every model needs a model ID.');
      return { id, name: text(row.name) || id, contextMax: numeric(row.contextMax), maxOutput: numeric(row.maxOutput), vision: row.vision === true ? true : row.vision === false ? false : null, reasoning: row.reasoning === true ? true : row.reasoning === false ? false : null };
    });
    if (new Set(models.map(m => m.id)).size !== models.length) fail('Each model ID must be unique within this connection.');
    return { id: existing?.id ?? randomUUID(), kind, provider: catalog?.id ?? null, name, baseUrl, protocol, apiKey, models };
  }
  function save(body) {
    const connection = draft(body);
    if (!connection.models.length) fail('Choose at least one model or add its ID.');
    return persist(connections.some(c => c.id === connection.id) ? connections.map(c => c.id === connection.id ? connection : c) : [...connections, connection]);
  }
  function remove(id) { return persist(connections.filter(c => c.id !== id)); }
  function resolve(selection) {
    const connection = connections.find(c => c.id === selection?.connectionId);
    const model = connection?.models.find(m => m.id === selection?.modelId);
    if (!connection || !model) fail('This API model is no longer available. Choose another model in the chat picker.');
    return { connection, model };
  }
  function headers(connection) {
    return {
      'content-type': 'application/json',
      ...(connection.protocol === 'anthropic' ? { 'anthropic-version': '2023-06-01', ...(connection.apiKey ? { 'x-api-key': connection.apiKey } : {}) } : connection.apiKey ? { authorization: `Bearer ${connection.apiKey}` } : {}),
    };
  }
  async function request(connection, path, { signal, ...options } = {}) {
    let res;
    try {
      res = await fetchImpl(`${connection.baseUrl}/${path}`, { ...options, headers: headers(connection), redirect: 'error', signal });
    } catch (err) {
      if (signal?.aborted) throw signal.reason;
      fail('Could not reach the model API. Check the address and whether the server is running.');
    }
    if (!res.ok) {
      // Provider errors can echo credentials/request bodies. Report status, not their raw body.
      await res.body?.cancel();
      fail(res.status === 401 || res.status === 403 ? 'The model API rejected the credentials. Check the API key.' : res.status === 404 ? 'The model API could not find this endpoint or model. Check the base URL and model ID.' : res.status === 429 ? 'The model API is rate limited or out of credit. Try again later.' : `The model API returned HTTP ${res.status}. Check the connection and try again.`);
    }
    return res;
  }
  async function discover(body, signal) {
    const connection = draft(body);
    const timeout = AbortSignal.timeout(20000);
    const requestSignal = signal ? AbortSignal.any([signal, timeout]) : timeout;
    let data, rows = [], after = null;
    // Anthropic paginates /models. Most compatible APIs return the list in one page.
    for (let page = 0; page < 20; page++) {
      const path = connection.protocol === 'anthropic' ? `models?limit=100${after ? `&after_id=${encodeURIComponent(after)}` : ''}` : 'models';
      const res = await request(connection, path, { signal: requestSignal });
      try { data = await res.json(); } catch { fail('The API did not return a model list. You can add a model ID manually.'); }
      if (!Array.isArray(data.data)) fail('The API did not return a model list. You can add a model ID manually.');
      rows.push(...data.data);
      if (connection.protocol !== 'anthropic' || !data.has_more) break;
      after = data.last_id; if (!after) break;
    }
    const models = [...new Map(rows.filter(m => text(m.id, 256)).map(m => [m.id, { id: text(m.id, 256), name: text(m.display_name ?? m.name) || text(m.id, 256), contextMax: numeric(m.context_length ?? m.max_context_length ?? m.context_window), maxOutput: numeric(m.max_output_tokens ?? m.max_completion_tokens ?? m.top_provider?.max_completion_tokens), vision: modelReadsPictures(m), reasoning: modelSupportsReasoning(m) }])).values()];
    return { models: models.sort((a, b) => a.name.localeCompare(b.name)) };
  }
  async function stream({ selection, message, history = [], systemPrompt = '', images = [], signal, onFrame }) {
    const { connection, model } = resolve(selection);
    const started = Date.now();
    const messages = history.filter(m => ['user', 'assistant'].includes(m.role) && typeof m.content === 'string').slice(-40).map(m => ({ role: m.role, content: m.content }));
    // Pictures only go to OpenAI-style servers (the one protocol every local server shares); other protocols keep the text.
    const pics = connection.protocol === 'openai-chat' ? images : [];
    messages.push({ role: 'user', content: pics.length ? [{ type: 'text', text: message }, ...pics.map(i => ({ type: 'image_url', image_url: { url: i.url } }))] : message });
    const protocol = connection.protocol;
    const payload = protocol === 'anthropic'
      ? { model: model.id, messages, system: systemPrompt, max_tokens: 8192, stream: true }
      : protocol === 'openai-responses'
        ? { model: model.id, input: messages, instructions: systemPrompt, stream: true, store: false }
        : { model: model.id, messages: [...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []), ...messages], stream: true, ...(['openai', 'deepseek', 'openrouter', 'google', 'moonshotai'].includes(connection.provider) ? { stream_options: { include_usage: true } } : {}) };
    const requestSignal = signal ? AbortSignal.any([signal, AbortSignal.timeout(10 * 60 * 1000)]) : AbortSignal.timeout(10 * 60 * 1000);
    const res = await request(connection, protocol === 'anthropic' ? 'messages' : protocol === 'openai-responses' ? 'responses' : 'chat/completions', { method: 'POST', body: JSON.stringify(payload), signal: requestSignal });
    if (!res.body) fail('The model API returned an empty response.');
    onFrame({ type: 'start', model: model.name, contextMax: model.contextMax, demo: false });
    let tokensIn = null, tokensOut = null, textLength = 0, firstToken = null, finished = false;
    function emit(type, value) {
      if (typeof value !== 'string' || !value) return;
      if (firstToken === null) firstToken = Date.now() - started;
      if (type === 'delta') textLength += value.length;
      onFrame({ type, text: value });
    }
    function usage(value) {
      if (!value) return;
      tokensIn = numeric(value.prompt_tokens ?? value.input_tokens) ?? tokensIn;
      tokensOut = numeric(value.completion_tokens ?? value.output_tokens) ?? tokensOut;
    }
    function consume(data) {
      if (data.error || data.type === 'error' || data.type === 'response.failed') fail('The model API could not complete this response. Check the provider and try again.');
      if (protocol === 'anthropic') {
        if (data.type === 'message_start') usage(data.message?.usage);
        if (data.type === 'content_block_delta') { emit('delta', data.delta?.text); emit('reasoning', data.delta?.thinking); }
        if (data.type === 'message_delta') usage(data.usage);
        if (data.type === 'message_stop') finished = true;
      } else if (protocol === 'openai-responses') {
        if (['response.output_text.delta', 'response.refusal.delta'].includes(data.type)) emit('delta', data.delta);
        if (data.type === 'response.reasoning_summary_text.delta') emit('reasoning', data.delta);
        if (data.type === 'response.completed') { usage(data.response?.usage); finished = true; }
        if (data.type === 'response.incomplete') fail('The API stopped before completing the response. Partial output is preserved.');
      } else {
        const choice = data.choices?.find(c => c.index === 0) ?? data.choices?.[0];
        emit('delta', choice?.delta?.content ?? choice?.delta?.refusal);
        emit('reasoning', choice?.delta?.reasoning_content ?? choice?.delta?.reasoning);
        usage(data.usage);
        if (choice?.finish_reason) finished = true;
      }
    }
    const reader = res.body.getReader(), decoder = new TextDecoder();
    let buffer = '', eventData = [];
    const line = value => {
      if (!value) {
        if (!eventData.length) return;
        const raw = eventData.join('\n'); eventData = [];
        if (raw === '[DONE]') { finished = true; return; }
        let data; try { data = JSON.parse(raw); } catch { fail('The model API sent an invalid stream event. Partial output is preserved.'); }
        consume(data);
      } else if (value.startsWith('data:')) eventData.push(value.slice(5).trimStart());
    };
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        if (buffer.length > 4 * 1024 * 1024) fail('The model API sent an oversized stream event.');
        let end;
        while ((end = buffer.indexOf('\n')) >= 0) { line(buffer.slice(0, end).replace(/\r$/, '')); buffer = buffer.slice(end + 1); }
      }
      if (buffer) line(buffer.replace(/\r$/, '')); line('');
      requestSignal.throwIfAborted();
      if (!finished) fail('The API connection ended early. Partial output is preserved; try again.');
      if (!textLength) fail('The model returned no text. Choose a model that supports text chat.');
    } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
    const durationMs = Date.now() - started;
    const result = { tokensIn, tokensOut, usageSource: tokensIn !== null && tokensOut !== null ? 'reported' : 'unavailable', durationMs, ttftMs: firstToken, tps: answerSpeed(tokensOut, durationMs, firstToken) };
    onFrame({ type: 'done', ...result }); return result;
  }
  /** Connections WITH their keys, for the host's own agent bridge only; never serialised to a client. */
  function withSecrets() { return connections.map(c => ({ ...c, models: c.models.map(m => ({ ...m })) })); }
  return { list, save, remove, discover, resolve, stream, withSecrets };
}
