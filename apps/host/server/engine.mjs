/** Engine configuration: beginner goals, expert parameters, predictions, restart. */
import { CONTEXT_TOKENS, expertParams, footprint, variant } from './catalog.mjs';
import { cachedHardware, detect } from './hardware.mjs';
import { get as getStore, update } from './store.mjs';
import * as supervisor from './supervisor.mjs';
import { nowIso } from './util.mjs';

const engineState = {
  running: false,
  startedAt: null,
  uptimeSec: 0,
  version: 'strata-engine 0.1.0',
  backend: 'metal',
  timer: null,
};

let hubRef = null;

export function init(hub) {
  hubRef = hub;
  supervisor.init(hub);
  if (engineState.timer) clearInterval(engineState.timer);
  engineState.timer = setInterval(() => {
    if (!engineState.running || !engineState.startedAt) return;
    engineState.uptimeSec = Math.round((Date.now() - new Date(engineState.startedAt).getTime()) / 1000);
  }, 1000);
}

export function markRunning(backend) {
  engineState.running = true;
  engineState.startedAt = nowIso();
  engineState.uptimeSec = 0;
  if (backend) engineState.backend = backend;
}

export function markStopped() {
  engineState.running = false;
  engineState.startedAt = null;
  engineState.uptimeSec = 0;
}

/** Merge stored overrides on top of the catalog defaults. */
export function params() {
  const overrides = getStore().expert ?? {};
  const catalog = expertParams(cachedHardware()?.cpu?.cores ?? null);
  const known = new Set(catalog.map(p => p.id));
  const mapped = new Set(['gpu.layers','ctx.length','ctx.kvMode','mem.mmap','mem.lock','perf.batch','perf.threads','perf.flashAttn','server.port','server.parallel']);
  return [...catalog.map((p) => ({
    ...p,
    applies: p.id.startsWith('sample.') ? 'Next request' : mapped.has(p.id) ? 'Next model load' : 'Preserved only · adapter does not apply this value',
    value: Object.hasOwn(overrides, p.id) ? overrides[p.id] : p.value,
  })), ...Object.entries(overrides).filter(([id]) => !known.has(id)).map(([id, value]) => ({
    id, value, recommended: value, group: 'Additional settings', label: id,
    effect: 'Preserved; no verified engine mapping in this build.', applies: 'Preserved only',
    type: typeof value === 'boolean' ? 'toggle' : typeof value === 'number' ? 'number' : 'text',
  }))];
}

function loadArgs(args) {
  const sampling = new Set(['--temp', '--top-p', '--top-k', '--repeat-penalty', '--seed']);
  const result = [];
  for (let i = 0; i < args.length; i++) {
    if (sampling.has(args[i])) { i++; continue; }
    result.push(args[i]);
  }
  return result;
}

export async function config() {
  const s = getStore();
  const id = supervisor.status().modelId ?? s.loaded ?? s.current;
  const v = id ? variant(id) : null;
  const hw = await detect();
  const sup = supervisor.status();
  const ctxLabel = String(s.beginner.contextLength).toUpperCase();
  const ctxTokens = CONTEXT_TOKENS[ctxLabel.toLowerCase()] ?? 32768;

  const { pathFor } = await import('./models.mjs');
  const previewArgs = v ? supervisor.buildArgs({
    modelPath: sup.running ? pathFor(v) : `<models>/${v.id}/${v.source.file}`,
    config: { beginner: s.beginner, expert: params(), serving: s.serving, layerCount: v.layerCount },
    hardware: hw, port: Number(s.expert?.['server.port'] ?? 8080),
  }) : [];

  return {
    extra: s.configExtra ?? {},
    variantId: v?.id ?? null,
    beginner: s.beginner,
    expert: params(),
    predicted: predict(s.beginner, hw, v),
    footprint: v ? footprint(v, ctxTokens) : null,
    dirty: sup.running && JSON.stringify(loadArgs(previewArgs)) !== JSON.stringify(loadArgs(sup.args)),
    appliedAt: sup.startedAt ?? undefined,
    engine: {
      running: sup.running,
      status: sup.status,
      pid: sup.pid ?? undefined,
      uptimeSec: sup.startedAt ? Math.round((Date.now() - new Date(sup.startedAt).getTime()) / 1000) : 0,
      version: sup.binary ? `llama.cpp · ${sup.binary.source}` : engineState.version,
      backend: hw.accelerator.backend,
      startedAt: sup.startedAt ?? undefined,
      port: sup.port,
      command: v ? [sup.binary?.path ?? 'llama-server', ...previewArgs].join(' ') : '',
      args: previewArgs,
      error: sup.error,
    },
  };
}

export function predict(beginner, hw, v, expert = getStore().expert ?? {}) {
  const notes = [];
  const ctx = CONTEXT_TOKENS[beginner.contextLength] ?? 32768;

  const speed = beginner.priority === 'speed' ? 'Larger prompt batches' : beginner.priority === 'quality' ? 'Lower randomness' : beginner.priority === 'coding' ? 'Focused sampling' : 'Balanced';
  let memory = 'Recommended';
  let restartNeeded = true;

  if (beginner.priority === 'speed' && ctx <= 16384) notes.push('Short context keeps the KV cache small.');
  if (beginner.priority === 'quality') notes.push('Lower temperature makes responses less varied; it does not guarantee accuracy.');
  if (beginner.priority === 'coding') notes.push('Lower temperature favors repeatable responses.');
  if (beginner.readImages) {
    memory = 'Higher — vision layer adds ~1.8 GB';
    notes.push('Vision needs the multimodal projector loaded alongside the model.');
  }
  if (ctx >= 65536) {
    memory = 'Much higher — long context dominates memory';
    notes.push('Longer context allocates a larger KV cache. Actual usage varies by architecture and cache format.');
  }
  if (beginner.gpuIdle !== 'never') {
    notes.push(`Freeing the GPU after ${beginner.gpuIdle.replace('m', ' minutes').replace('1h', 'an hour')} means the next prompt reloads the model.`);
  } else {
    notes.push('Keeping the model loaded avoids a model reload between requests. Prompt reading still takes time.');
  }
  if (v && hw) {
    const headroom = hw.accelerator.unified ? hw.memory.totalGb * 0.72 : hw.accelerator.usableVramGb;
    const fp = { ...footprint(v, ctx) };
    const kvFactor = ['FP16'].includes(expert['ctx.kvMode']) ? 2 : expert['ctx.kvMode'] === 'Q4_0' ? .5 : 1;
    fp.kvGb *= kvFactor; fp.totalGb = fp.weightsGb + fp.kvGb + fp.overheadGb;
    if (headroom) {
      const verdict = fp.totalGb <= headroom * 0.85 ? 'Comfortable' : fp.totalGb <= headroom ? 'Tight' : 'Above estimated memory budget';
      /* Non-breaking spaces inside the measurement: in the narrow effect
         panel this wrapped after "of", leaving an orphaned "12 GB" on a line
         of its own. The break now falls before the figure, so the second line
         is the whole measurement rather than its tail. */
      const nb = '\u00a0';
      memory = `${verdict} · about${nb}${fp.totalGb.toFixed(1)}${nb}GB${nb}of${nb}${headroom.toFixed(0)}${nb}GB`;
      notes.push(
        `${v.label} at ${(ctx / 1024).toFixed(0)}K: ${fp.weightsGb.toFixed(1)} GB weights + ${fp.kvGb.toFixed(2)} GB KV cache + ${fp.overheadGb.toFixed(1)} GB runtime.`,
      );
      if (fp.totalGb > headroom) notes.push('This estimate exceeds the selected memory budget. Use less context or a smaller variant.');
    }
  }
  if (ctx > (v?.contextMax ?? 32768)) {
    notes.push(`Beyond the model's advertised context. Output reliability is not verified past ${((v?.contextMax ?? 32768) / 1024).toFixed(0)}K.`);
  }

  return { responseSpeed: speed, memory, restartNeeded, notes };
}

export async function patch(body) {
  const s = getStore();
  const next = { ...s };

  if (body.beginner) {
    next.beginner = { ...s.beginner, ...body.beginner };
    if (body.beginner.contextLength) next.expert = { ...(s.expert ?? {}), 'ctx.length': body.beginner.contextLength === 'recommended' ? '128K' : body.beginner.contextLength.toUpperCase() };
  }
  if (Array.isArray(body.expert)) {
    const overrides = body.replaceExpert ? {} : { ...(next.expert ?? {}) };
    for (const item of body.expert) {
      if (!item || typeof item.id !== 'string') continue;
      overrides[item.id] = item.value;
    }
    next.expert = overrides;
  }
  if (body.expertReset) next.expert = {};
  if (body.extra && typeof body.extra === 'object' && !Array.isArray(body.extra)) next.configExtra = body.extra;

  update(next);
  const cfg = await config();
  hubRef?.send('engine', cfg);
  return cfg;
}

export async function reset() {
  update({
    beginner: { priority: 'balanced', contextLength: 'recommended', readImages: false, gpuIdle: 'never' },
    expert: {},
  });
  const cfg = await config();
  hubRef?.send('engine', cfg);
  return cfg;
}

export async function restart() {
  if (process.env.STRATA_REAL_ENGINE !== '1') throw new Error('Restart is unavailable with the example engine. Save settings for the next load.');
  const s = getStore();
  const id = supervisor.status().modelId ?? s.loaded;
  const v = id ? variant(id) : null;
  const hw = await detect();

  if (!v) {
    hubRef?.send('log', { level: 'warn', message: 'No model loaded — nothing to restart', at: nowIso() });
    const cfg = await config();
    hubRef?.send('engine', cfg);
    return cfg;
  }

  hubRef?.send('log', { level: 'info', message: `Restarting engine with ${v.label}`, at: nowIso() });
  try {
    const { pathFor } = await import('./models.mjs');
    await supervisor.restart({
      modelId: v.id,
      modelPath: pathFor(v),
      port: Number(s.expert?.['server.port'] ?? 8080),
      config: { beginner: s.beginner, expert: params(), serving: s.serving, layerCount: v.layerCount },
      hardware: hw,
    });
    markRunning(hw.accelerator.backend);
  } catch (err) {
    hubRef?.send('log', { level: 'error', message: `Restart failed: ${err.message}`, at: nowIso() });
    markStopped();
  }

  const cfg = await config();
  hubRef?.send('engine', cfg);
  return cfg;
}

/**
 * @param draft optional beginner config to predict *instead of* the saved one.
 *   Without it the screen can only show the consequence of settings that have
 *   already been applied, which makes an "impact" line a restatement of the
 *   status quo rather than a preview of the choice in front of you.
 */
export async function validate(draft = {}) {
  const hw = await detect();
  const s = getStore();
  const beginner = { ...s.beginner, ...(draft?.beginner ?? {}) };
  const v = variant(draft?.variantId ?? s.loaded ?? s.current);
  const issues = [];
  const values = Object.fromEntries(params().map(p => [p.id, p.value]));
  for (const p of draft?.expert ?? []) values[p.id] = p.value;
  const addError = (key, message) => issues.push({ level: 'error', key, message: `${key}: ${message}` });
  const enums = { priority: ['speed', 'balanced', 'quality', 'coding'], contextLength: Object.keys(CONTEXT_TOKENS), gpuIdle: ['never'] };
  for (const [key, options] of Object.entries(enums)) if (!options.includes(beginner[key])) addError(key, 'choose a supported value.');
  if (typeof beginner.readImages !== 'boolean' || beginner.readImages) addError('readImages', 'image input is not supported by this host.');
  const limits = {
    'sample.temp': [0, 2], 'sample.topP': [0, 1], 'sample.topK': [0, 10000], 'sample.repeat': [0.5, 3],
    'sample.seed': [-1, 2147483647], 'sample.maxTokens': [0, 1048576], 'perf.threads': [1, 1024], 'perf.batch': [1, 65536],
    'perf.draftDepth': [1, 64], 'mem.expertCache': [0, 10000], 'server.port': [1, 65535],
    'server.parallel': [1, 64], 'server.queue': [0, 10000], 'server.timeout': [1, 86400],
    'adv.rope': [0.01, 100], 'adv.ropeBase': [1, 1000000000],
  };
  const fractional = new Set(['sample.temp', 'sample.topP', 'sample.repeat', 'adv.rope']);
  for (const p of expertParams(hw.cpu.cores)) {
    const value = values[p.id];
    if (p.type === 'number') {
      const [min, max] = limits[p.id] ?? [-Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER];
      if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (!fractional.has(p.id) && !Number.isInteger(value))) addError(p.id, `enter ${fractional.has(p.id) ? 'a number' : 'a whole number'} from ${min} to ${max}.`);
    } else if (p.type === 'toggle' && typeof value !== 'boolean') addError(p.id, 'use true or false.');
    else if (p.type === 'select' && !p.options?.includes(value)) addError(p.id, 'choose a supported value.');
    else if (p.type === 'text' && typeof value !== 'string') addError(p.id, 'enter text.');
  }
  if (!CONTEXT_TOKENS[String(values['ctx.length']).toLowerCase()] || String(values['ctx.length']).toLowerCase() === 'recommended') addError('ctx.length', 'enter a whole context size from 4K to 1024K.');
  const kv = values['ctx.kvMode'];
  if (draft?.beginner && draft?.expert?.some(p => p.id === 'ctx.length')) {
    const effective = beginner.contextLength === 'recommended' ? '128K' : String(beginner.contextLength).toUpperCase();
    if (values['ctx.length'] !== effective) addError('ctx.length', 'match beginner.contextLength (Recommended means 128K).');
  }
  const ctxLabel = beginner.contextLength;
  const ctx = CONTEXT_TOKENS[String(ctxLabel).toLowerCase()] ?? 32768;
  const headroom = hw.accelerator.unified ? hw.memory.totalGb * 0.72 : hw.accelerator.usableVramGb;

  if (v && ctx > v.contextMax) {
    issues.push({ level: 'warn', message: `${ctx / 1024}K exceeds ${v.label}'s trained context — quality beyond that limit is not verified.` });
  }
  if (v && headroom) {
    const fp = footprint(v, ctx); const factor = kv === 'FP16' ? 2 : kv === 'Q4_0' ? .5 : 1;
    const estimate = fp.weightsGb + fp.kvGb * factor + fp.overheadGb;
    if (estimate > hw.memory.totalGb) issues.push({ level: 'warn', message: `Estimated working memory (${estimate.toFixed(1)} GB) exceeds physical memory. Choose a smaller model or shorter context.` });
  }
  if (v && headroom && v.diskGb > headroom) {
    issues.push({ level: 'warn', message: `${v.label} needs ${v.diskGb} GB; about ${headroom.toFixed(0)} GB is fast memory. Actual fit and offload behavior depend on the adapter.` });
  }
  const threads = Number(values['perf.threads'] ?? 0);
  if (threads > hw.cpu.cores) {
    issues.push({ level: 'warn', message: `CPU threads (${threads}) exceed physical cores (${hw.cpu.cores}).` });
  }
  const parallel = Number(values['server.parallel'] ?? 1);
  if (parallel > 1) {
    issues.push({ level: 'warn', message: 'Parallel requests multiply KV cache memory — keep it at 1 on unified memory.' });
  }
  // The prediction for the config being asked about, not the one in force.
  return {
    ok: issues.every((i) => i.level !== 'error'),
    issues,
    predicted: predict(beginner, hw, v, values),
  };
}

export async function exportConfig() {
  const s = getStore();
  return {
    ...(s.configExtra ?? {}),
    schema: 'strata.engine/1',
    exportedAt: nowIso(),
    model: s.loaded ?? s.current,
    beginner: s.beginner,
    expert: Object.fromEntries(params().map((p) => [p.id, p.value])),
  };
}

export function engineInfo() {
  return { ...engineState, timer: undefined };
}
