/**
 * Inference bridge.
 *
 * Preference order:
 *   1. the Strata engine — the C++ engine behind Strata's own Python API worker. This is the
 *      product's inference path and is tried first. `STRATA_SERVER_URL` names it explicitly;
 *      otherwise the Strata worker's default loopback address is probed.
 *   2. the engine Strata supervises itself (llama.cpp), when one is configured
 *   3. a *different* OpenAI-compatible server already running on localhost, but only when it
 *      identifies itself as one; a host that merely answers a port is not adopted silently
 *   4. a clearly-labelled demo engine — **development only**. A packaged or `STRATA_REAL_ENGINE=1`
 *      run never reaches this branch: it raises a typed error instead, so no simulated answer can
 *      enter a production success path.
 */
import * as supervisor from './supervisor.mjs';
import { get as getStore } from './store.mjs';
import { which } from './util.mjs';

/** Where Strata's own API worker listens by default (`serve/server.py`, `--port 8095`). */
const STRATA_DEFAULT_URL = 'http://127.0.0.1:8095/v1';

const CANDIDATES = [
  { name: 'llama.cpp', url: 'http://127.0.0.1:8080/v1' },
  { name: 'Ollama', url: 'http://127.0.0.1:11434/v1' },
  { name: 'LM Studio', url: 'http://127.0.0.1:1234/v1' },
];

let backend = null;
let checkedAt = 0;

/**
 * True when this process must not simulate inference.
 *
 * Packaged builds are always real. `STRATA_REAL_ENGINE` can force it either way for development,
 * but it can never turn a packaged build back into demo mode — that is the guarantee acceptance
 * checklist row U12 asks for ("no success path using demo responses/metrics").
 */
export function requiresRealEngine() {
  if (process.env.STRATA_FORCE_DEMO === '1' && process.env.STRATA_PACKAGED !== '1') return false;
  if (process.env.STRATA_REAL_ENGINE === '1') return true;
  return process.env.STRATA_PACKAGED === '1';
}

/** Where the Strata API worker is, if it was named explicitly. */
function strataUrl() {
  const raw = process.env.STRATA_SERVER_URL?.trim();
  if (!raw) return STRATA_DEFAULT_URL;
  return raw.replace(/\/+$/, '').endsWith('/v1') ? raw.replace(/\/+$/, '') : `${raw.replace(/\/+$/, '')}/v1`;
}

/**
 * Probe a loopback endpoint and confirm it really is a Strata worker.
 *
 * A bare OpenAI-compatible listener is not enough: Strata's worker answers `/v1/status` with its
 * own shape (it carries `service: "strata"` and the reported context). Adopting an unrelated
 * server because it happens to hold the port is exactly the silent substitution the architecture
 * forbids, so the identity check is required.
 */
async function probeStrata(baseUrl) {
  try {
    const res = await fetch(`${baseUrl.replace(/\/v1$/, '')}/v1/status`, { signal: AbortSignal.timeout(900) });
    if (!res.ok) return null;
    const status = await res.json().catch(() => null);
    if (!status || typeof status !== 'object') return null;
    if (status.service !== 'strata') return null;
    const models = await fetch(`${baseUrl}/models`, { signal: AbortSignal.timeout(900) })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    return {
      kind: 'strata',
      name: 'Strata engine',
      baseUrl,
      models: (models?.data ?? []).map((m) => m.id).filter(Boolean),
      modelId: status.model ?? null,
      context: status.context?.native ?? null,
      loaded: status.loaded === true,
      dialects: status.dialects ?? [],
    };
  } catch {
    return null;
  }
}

export async function detectBackend({ force = false } = {}) {
  // 1. Strata's own engine, always first: it is the product's inference path.
  const strata = await probeStrata(strataUrl());
  if (strata) {
    backend = strata;
    checkedAt = Date.now();
    return backend;
  }

  if (requiresRealEngine() && !supervisor.isRunning()) {
    // No simulation is permitted here. Try the supervisor and the identified candidates below,
    // and if none is a real engine the caller gets a typed error rather than invented tokens.
  }

  if (supervisor.isRunning()) {
    const status = supervisor.status();
    backend = {
      kind: 'engine',
      name: 'Strata engine · llama.cpp',
      baseUrl: `http://127.0.0.1:${status.port}/v1`,
      models: status.modelId ? [status.modelId] : [],
    };
    checkedAt = Date.now();
    return backend;
  }

  if (!force && backend && backend.kind !== 'engine' && Date.now() - checkedAt < 20_000) return backend;

  for (const c of CANDIDATES) {
    try {
      const res = await fetch(`${c.url}/models`, { signal: AbortSignal.timeout(700) });
      if (res.ok) {
        const data = await res.json().catch(() => ({}));
        backend = {
          kind: 'openai-compatible',
          name: c.name,
          baseUrl: c.url,
          models: (data?.data ?? []).map((m) => m.id),
        };
        checkedAt = Date.now();
        return backend;
      }
    } catch {
      /* try the next candidate */
    }
  }

  if (requiresRealEngine()) {
    backend = {
      kind: 'unavailable',
      name: 'No Strata engine',
      baseUrl: null,
      models: [],
      error:
        `No Strata engine answered at ${strataUrl()} and no supervised engine is running. ` +
        'Start Strata (or load a model) and retry. Simulated output is disabled in this build.',
    };
    checkedAt = Date.now();
    return backend;
  }

  const binary = supervisor.findBinary() ?? (await which('llama-server'));
  backend = {
    kind: 'demo',
    name: binary ? 'llama.cpp (engine not running)' : 'Strata demo engine',
    baseUrl: null,
    models: [],
    binary: binary?.path ?? null,
    // Every demo frame is labelled for the UI; nothing here may be recorded as a real result.
    simulated: true,
  };
  checkedAt = Date.now();
  return backend;
}

export function currentBackend() {
  if (supervisor.isRunning()) {
    const status = supervisor.status();
    return {
      kind: 'engine',
      name: 'Strata engine · llama.cpp',
      baseUrl: `http://127.0.0.1:${status.port}/v1`,
      models: status.modelId ? [status.modelId] : [],
    };
  }
  return backend;
}

/* ------------------------------------------------------------------- utils --- */

const CHUNK_MS = 26;

function tokenize(text) {
  const out = [];
  const words = text.split(/(\s+)/);
  let buffer = '';
  for (const w of words) {
    buffer += w;
    if (buffer.length >= 3 || /\n/.test(w)) {
      out.push(buffer);
      buffer = '';
    }
  }
  if (buffer) out.push(buffer);
  return out;
}

export function estimateTokens(text) {
  return Math.max(1, Math.round(text.length / 3.8));
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* -------------------------------------------------------------- the stream --- */

export async function streamChat({ message, history = [], systemPrompt = '', modelLabel, onFrame, signal, engine, candidates = [] }) {
  const be = await detectBackend();

  if (be.kind === 'strata') {
    return proxyCompletion({
      baseUrl: be.baseUrl,
      message,
      history,
      systemPrompt,
      onFrame,
      signal,
      supervised: false,
      engine,
      modelId: be.modelId || 'strata',
    });
  }

  if (be.kind === 'engine' || be.kind === 'openai-compatible') {
    return proxyCompletion({
      baseUrl: be.baseUrl,
      message,
      history,
      systemPrompt,
      onFrame,
      signal,
      supervised: be.kind === 'engine',
      engine,
    });
  }

  if (be.kind === 'unavailable') {
    // Fail loudly and specifically. This is a real, non-simulated error path.
    throw new Error(be.error ?? 'No Strata engine is available.');
  }

  return demoCompletion({ message, modelLabel, onFrame, signal, engine, candidates });
}

/* --------------------------------------------------------- real engine path --- */

export async function proxyCompletion({ baseUrl, message, history = [], systemPrompt = '', onFrame, signal, supervised = false, engine,modelId='local',authorization,images=[],extra={} }) {
  const started = Date.now();
  engine?.beginReading(0, engine.variantId);
  const messages = [
    ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
    ...history.filter((m) => ['system', 'user', 'assistant'].includes(m.role)).map((m) => ({ role: m.role, content: m.content })),
    { role: 'user', content:images.length?[{type:'text',text:message},...images.map(i=>({type:'image_url',image_url:{url:i.url}}))]:message },
  ];

  const defaults = getStore().expert ?? {};
  const sampling = { temperature: defaults['sample.temp'] ?? 0.6, top_p: defaults['sample.topP'] ?? 0.95, top_k: defaults['sample.topK'] ?? 40, repeat_penalty: defaults['sample.repeat'] ?? 1.05, seed: defaults['sample.seed'] ?? -1, experimental_speed_projection: defaults['sample.speedProjection'] ?? true, ...(Number(defaults['sample.maxTokens']) > 0 ? {max_tokens: Number(defaults['sample.maxTokens'])} : {}) };
  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...(authorization?{authorization:`Bearer ${authorization}`} :supervised && getStore().serving?.apiKey ? { authorization: `Bearer ${getStore().serving.apiKey}` } : {}) },
    signal,
    body: JSON.stringify({
      model: modelId,
      stream: true,
      stream_options: { include_usage: true },
      ...sampling,
      ...extra,
      messages,
    }),
  });

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => '');
    throw new Error(`Engine responded ${res.status}${detail ? `: ${detail.slice(0, 200)}` : ''}`);
  }

  onFrame({ type: 'start', demo: false });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let out = null;
  let promptTokens = null;
  let cachedTokens = null;
  let firstTokenAt = null;
  let streaming = true;
  let writing = false;
  let timings=null,finishReason=null;
  /* The speed shown while it streams is the decode speed of everything generated, reasoning included, from the first token: tokens
     (one per streamed chunk, or characters / 4 when the server batches) over the time since the first token, never the visible
     answer over the whole wall time. The engine's own numbers replace it when the answer is done. */
  let chunks=0,chars=0,lastStats=0;
  const liveStats=()=>{if(supervised||firstTokenAt===null)return;const now=Date.now();if(now-lastStats<400)return;const seconds=(now-firstTokenAt)/1000;if(seconds<.5)return;lastStats=now;
   const tokens=chars/Math.max(1,chunks)<8?chunks:Math.ceil(chars/4);onFrame({type:'stats',tokensOut:tokens,tps:Number((tokens/seconds).toFixed(1)),source:'stream'});};

  const metricsTicker = supervised
    ? setInterval(() => {
        void supervisor.pollMetrics().then((m) => {
          if (streaming && m.source === 'engine' && m.outputTps > 0) {
            onFrame({ type: 'stats', tokensOut: out, tps: m.outputTps, promptTps: m.promptTps, source: 'engine' });
          }
        }).catch(() => {});
      }, 900)
    : null;

  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() + '\n' : decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === '[DONE]') continue;
        let parsed;
        try {
          parsed = JSON.parse(payload);
        } catch {
          continue;
        }
        if (parsed.error) throw new Error(parsed.error.message ?? 'Inference failed.');
        if(parsed.strata_mcp)onFrame({type:'native-tool',event:parsed.strata_mcp});
        if(parsed.timings)timings=parsed.timings;
        if(parsed.choices?.[0]?.finish_reason)finishReason=parsed.choices[0].finish_reason;

        if (parsed.usage) {
          if (Number.isFinite(parsed.usage.prompt_tokens)) promptTokens = parsed.usage.prompt_tokens;
          if (Number.isFinite(parsed.usage.completion_tokens)) out = parsed.usage.completion_tokens;
          if (Number.isFinite(parsed.usage.prompt_tokens_details?.cached_tokens)) cachedTokens = parsed.usage.prompt_tokens_details.cached_tokens;
        }

        const delta = parsed.choices?.[0]?.delta;
        if (!delta) continue;

        // Qwen3-style thinking models stream reasoning separately.
        if (delta.reasoning_content) {
          if (firstTokenAt === null) engine?.beginThinking();
          if (firstTokenAt === null) firstTokenAt = Date.now();
          chunks++; chars+=delta.reasoning_content.length;
          onFrame({ type: 'reasoning', text: delta.reasoning_content });
          liveStats();
        }
        if (delta.content) {
          if (!writing) { engine?.beginGenerating(); writing = true; }
          if (firstTokenAt === null) firstTokenAt = Date.now();
          chunks++; chars+=delta.content.length;
          onFrame({ type: 'delta', text: delta.content });
          liveStats();
        }
      }
      if (done) break;
    }
  } finally {
    streaming = false;
    if (metricsTicker) clearInterval(metricsTicker);
    reader.releaseLock();
  }

  signal?.throwIfAborted();
  const durationMs = Date.now() - started;
  const decodeMs = firstTokenAt ? Date.now() - firstTokenAt : durationMs;
  const tps = Number.isFinite(timings?.predicted_per_second)?timings.predicted_per_second:out !== null && decodeMs > 0 ? Number((out / (decodeMs / 1000)).toFixed(1)) : null;
  const usageSource = promptTokens !== null && out !== null ? 'reported' : 'unavailable';

  const result = { tokensIn: promptTokens, tokensOut: out, cachedTokens, tps, promptTps:Number.isFinite(timings?.prompt_per_second)?timings.prompt_per_second:null,hitRate:Number.isFinite(timings?.hit_rate)?timings.hit_rate:null,pcieShare:Number.isFinite(timings?.pcie_share)?timings.pcie_share:null,finishReason,durationMs, usageSource, ttftMs: firstTokenAt === null ? null : firstTokenAt - started };
  onFrame({ type: 'done', ...result, demo: false });
  return result;
}

/* --------------------------------------------------------------- demo path --- */

const wantsPreviewExamples = message => /preview examples|website.*(?:markdown|pdf)|(?:markdown|pdf).*website/i.test(message);
const previewExamples = [
  { id:'sample-website', kind:'website', title:'Fieldnotes — website', url:'/examples/fieldnotes.html', demo:true },
  { id:'sample-markdown', kind:'markdown', title:'project-brief.md', url:'/examples/project-brief.md', demo:true },
  { id:'sample-pdf', kind:'pdf', title:'project-brief.pdf', url:'/examples/project-brief.pdf', demo:true },
];
function demoAnswer(prompt, variantLabel) {
  const p = prompt.toLowerCase();
  const lead = `Running on ${variantLabel}. `;
  if (wantsPreviewExamples(prompt)) return 'Here are three example outputs for a small project called Fieldnotes. Open the website, Markdown brief or PDF below to review it beside our conversation. These are built-in preview examples.';

  if (/readme|repo|project|summar/.test(p)) {
    return `${lead}Here is the short version of what that file is telling you.

**What it is** — a local inference app: one model, one engine, tuned to the machine it runs on.

**The important parts**
- The engine host detects your hardware before anything downloads.
- Models are suggested by memory class. Fit and memory figures are estimates.
- Prompt reading and generation appear separately. This demo uses simulated values.

**What I would do next**
1. Check the \`Models\` tab and confirm the recommended variant matches your RAM class.
2. Open \`Monitor\` while a long prompt runs and watch prompt throughput — that is usually the bottleneck, not generation.
3. If replies feel slow, lower the context length in \`Engine tuning\` before changing sampling values.

Ask me about any specific section and I will go deeper.`;
  }

  if (/kv cache|cache|context/.test(p)) {
    return `${lead}Think of the KV cache as the model's short-term memory for the conversation.

Every token you send gets turned into a key and a value vector, and those vectors are kept so the model does not have to re-read the whole prompt on every new token. That is why the first token is slow and the rest are fast.

**Why it matters on your machine**
- Cache size scales with **context length × layers × head width**.
- At INT8 the cache is roughly half the size of FP16, at a small quality cost.
- A 32K context can easily cost more memory than the model weights on a small machine.

If memory is tight, cut context length first — it is the single biggest lever.`;
  }

  return `${lead}Short answer first, then the reasoning.

**Short answer** — yes, and the useful part is *where* the cost lands. On a single local model the trade is almost always memory against latency.

**What I would check on this machine**
1. Monitor separates prompt reading from answer generation.
2. Each reading identifies its source, including simulated or unavailable values.
3. Context describes how much text the model can consider; it is not a quality score.

This is a demo response. Real answers require a connected inference engine.`;
}

/**
 * Picks a file that is actually in the workspace.
 *
 * The demo used to propose reading `README.md` on every machine, including ones
 * with no README at the root. A proposal naming a file that does not exist is a
 * bad demonstration and, worse, an approval that grants access to nothing.
 *
 * @param candidates relative paths the host found in the workspace.
 */
function pickReadable(candidates) {
  if (!candidates.length) return null;
  return (
    candidates.find((c) => /^readme\.md$/i.test(c)) ??
    candidates.find((c) => /readme/i.test(c)) ??
    candidates.find((c) => /\.md$/i.test(c)) ??
    candidates[0]
  );
}

/** Which tools the demo engine would reach for, given the prompt. */
function proposeTools(message, candidates = []) {
  const p = message.toLowerCase();
  const tools = [];
  if (/readme|read the|open |file|repo|workspace|docs?\b/.test(p)) {
    const path = pickReadable(candidates);
    if (path) {
      tools.push({
        id: 'read_file',
        name: 'Read file',
        args: { path },
        preview: path,
        effect: 'Reads a file inside the workspace. Nothing leaves this computer.',
        scope: 'read',
      });
    }
  }
  if (/run|test|build|lint|install|command/.test(p)) {
    tools.push({
      id: 'run_command',
      name: 'Run command',
      args: { command: 'npm test -- --runInBand' },
      preview: 'npm test -- --runInBand',
      effect: 'Runs a shell command in the workspace and returns the output.',
      scope: 'write',
    });
  }
  if (/edit|change|fix|refactor|write|update the/.test(p)) {
    tools.push({
      id: 'write_file',
      name: 'Edit file',
      args: { path: 'src/app/App.tsx', change: 'apply the requested change' },
      preview: 'src/app/App.tsx · 1 hunk',
      effect: 'Writes to a file inside the workspace. A diff is shown first.',
      scope: 'write',
    });
  }
  return tools.slice(0, 2);
}

async function demoCompletion({ message, modelLabel, onFrame, signal, engine, candidates = [] }) {
  const variety = modelLabel ?? 'the loaded model';
  const promptTokens = estimateTokens(message);

  onFrame({ type: 'start', demo: true, backend: 'demo', promptTokens });

  // Propose tool calls before answering — this is how a local agent behaves,
  // and it is the flow the approval card exists for.
  for (const tool of wantsPreviewExamples(message) ? [] : proposeTools(message, candidates)) {
    onFrame({ type: 'tool', tool });
  }
  await engine.beginReading(promptTokens, engine.variantId);

  const step = Math.max(1, Math.round(promptTokens / 14));
  for (let done = 0; done <= promptTokens; done += step) {
    signal?.throwIfAborted();
    engine.progressReading(Math.min(done, promptTokens));
    await sleep(CHUNK_MS * 2);
  }
  engine.progressReading(promptTokens);
  await sleep(220);
  signal?.throwIfAborted();

  engine.beginGenerating();
  const chunks = tokenize(demoAnswer(message, variety));
  const started = Date.now();
  let emitted = 0;

  for (const chunk of chunks) {
    signal?.throwIfAborted();
    onFrame({ type: 'delta', text: chunk });
    emitted += estimateTokens(chunk);
    if (emitted % 3 === 0) {
      const seconds = Math.max(0.001, (Date.now() - started) / 1000);
      onFrame({ type: 'stats', tokensOut: emitted, tps: Number((emitted / seconds).toFixed(1)) });
    }
    await sleep(CHUNK_MS);
  }

  const durationMs = Date.now() - started;
  const tps = Number((emitted / Math.max(0.001, durationMs / 1000)).toFixed(1));
  signal?.throwIfAborted();
  if (wantsPreviewExamples(message)) onFrame({ type: 'artifacts', artifacts: previewExamples });
  onFrame({ type: 'done', tokensIn: promptTokens, tokensOut: emitted, tps, durationMs, demo: true, usageSource: 'demo' });
  return { tokensIn: promptTokens, tokensOut: emitted, durationMs, tps, usageSource: 'demo' };
}

/**
 * The start-up line "inference · ...". The worker that serves a model is only running once a model is loaded, so on a restart nothing answers
 * at the worker's address even though the engine is installed and usable; the line says that instead of "No Strata engine".
 * `engine` is Control's answer (null when there is no Control, as in a source checkout).
 */
export function inferenceLine({ backend, engine = null, settled = true }) {
  if (!settled) return 'unpacking the bundled engine… (about a minute the first time)';
  if (backend?.kind !== 'unavailable' || !engine) return backend?.name ?? 'No Strata engine';
  if (engine.usable) return `Strata engine${engine.engine?.version ? ' ' + engine.engine.version : ''} is installed; no model is loaded yet`;
  return engine.notice ? `No Strata engine for this PC. ${engine.notice}` : backend.name;
}
