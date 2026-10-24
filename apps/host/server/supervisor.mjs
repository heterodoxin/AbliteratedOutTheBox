/**
 * Engine supervisor.
 *
 * Owns the inference process: resolves a binary, translates the tuning
 * configuration into real flags, spawns it, waits for health, keeps a log ring
 * buffer, and polls the engine's own Prometheus metrics so the Monitor shows
 * measured numbers rather than guesses.
 *
 * Supported backends, in preference order:
 *   1. llama.cpp `llama-server` (bundled under $STRATA_HOME/engine, or on PATH)
 *   2. any already-running OpenAI-compatible server on localhost
 */
import { CONTEXT_TOKENS } from './catalog.mjs';
import { spawn } from 'node:child_process';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { MODELS_DIR } from './store.mjs';
import { which } from './util.mjs';

const LOG_LIMIT = 400;

const state = {
  status: 'stopped', // stopped | starting | running | stopping | error
  binary: null,
  binarySource: null,
  pid: null,
  port: 8080,
  modelId: null,
  modelPath: null,
  args: [],
  command: '',
  startedAt: null,
  error: null,
  logs: [],
  metrics: {
    source: 'none',
    promptTokens: 0,
    predictedTokens: 0,
    promptTps: 0,
    outputTps: 0,
    kvUsage: 0,
    kvTokens: 0,
    requestsProcessing: 0,
    requestsDeferred: 0,
    sampledAt: null,
  },
};

let child = null;
let pollTimer = null;
let hubRef = null;
let mcpAt=0,mcpStatus=null,mcpPending=false;

export function init(hub) {
  hubRef = hub;
}

/* ------------------------------------------------------------------ binary --- */

const BINARY_NAMES = ['llama-server'];

export function findBinary() {
  const candidates = [];

  if (process.env.STRATA_LLAMA_SERVER) candidates.push({ path: process.env.STRATA_LLAMA_SERVER, source: 'STRATA_LLAMA_SERVER' });

  // A build downloaded into the app's own folder wins over anything on PATH:
  // it is the version Strata was tested against.
  try {
    const engineDir = join(MODELS_DIR, '..', 'engine');
    if (existsSync(engineDir)) {
      for (const entry of readdirSync(engineDir)) {
        const dir = join(engineDir, entry);
        if (!statSync(dir).isDirectory()) continue;
        for (const name of BINARY_NAMES) {
          const full = join(dir, name);
          if (existsSync(full)) candidates.push({ path: full, source: 'bundled' });
        }
      }
    }
  } catch {
    /* ignore unreadable engine dir */
  }

  for (const name of BINARY_NAMES) {
    candidates.push({ path: `/opt/homebrew/bin/${name}`, source: 'homebrew' });
    candidates.push({ path: `/usr/local/bin/${name}`, source: 'homebrew' });
  }

  return candidates.find((c) => existsSync(c.path)) ?? null;
}

export async function resolveBinary() {
  const found = findBinary();
  if (found) {
    state.binary = found.path;
    state.binarySource = found.source;
    return found;
  }
  const onPath = await which('llama-server');
  if (onPath) {
    state.binary = onPath;
    state.binarySource = 'PATH';
    return { path: onPath, source: 'PATH' };
  }
  return null;
}

/* --------------------------------------------------------------- arguments --- */

const KV_TYPES = { INT8: 'q8_0', FP16: 'f16', Q8_0: 'q8_0', Q4_0: 'q4_0' };

/**
 * Turn the tuning configuration into a real llama-server command line.
 * Exported so the UI can show exactly what will run.
 */
export function buildArgs({ modelPath, config, hardware, port = state.port }) {
  const beginner = config.beginner ?? {};
  const params = new Map((config.expert ?? []).map((p) => [p.id, p.value]));
  const num = (id, fallback) => {
    const v = Number(params.get(id));
    return Number.isFinite(v) ? v : fallback;
  };
  const ctxLabel = beginner.contextLength === 'recommended' ? '128K' : String(beginner.contextLength).toUpperCase();
  const ctx = CONTEXT_TOKENS[ctxLabel.toLowerCase()] ?? 32768;

  const args = [
    '-m', modelPath,
    '--host', '127.0.0.1',
    '--port', String(port),
    '-c', String(ctx),
    '--metrics',
    '--slots',
    '--jinja',
    '--no-webui',
  ];

  // GPU offload — on unified memory "all layers" is the right default.
  const split = String(params.get('gpu.layers') ?? 'Automatic');
  const ngl = split === 'CPU only' ? 0 : split === 'Quarter' ? Math.max(1, Math.round(Number(config.layerCount ?? 48) / 4)) : split === 'Half' ? Math.max(1, Math.round(Number(config.layerCount ?? 48) / 2)) : 999;
  args.push('-ngl', String(ngl));

  const threads = num('perf.threads', Math.max(4, (hardware?.cpu?.cores ?? 8) - 2));
  args.push('-t', String(threads));

  const kv = KV_TYPES[String(params.get('ctx.kvMode') ?? 'INT8')] ?? 'q8_0';
  args.push('-ctk', kv, '-ctv', kv);

  const batch = num('perf.batch', 2048);
  args.push('-b', String(batch), '-ub', String(Math.max(128, Math.min(512, Math.round(batch / 4)))));

  if (params.get('perf.flashAttn') === false) args.push('-fa', 'off');
  else args.push('-fa', 'on');

  if (params.get('mem.mmap') === false) args.push('--no-mmap');
  if (params.get('mem.lock') === true) args.push('--mlock');

  const parallel = Math.max(1, num('server.parallel', 1));
  args.push('-np', String(parallel));

  // Sampling defaults applied to every request by the server.
  args.push(
    '--temp', String(params.get('sample.temp') ?? 0.6),
    '--top-p', String(params.get('sample.topP') ?? 0.95),
    '--top-k', String(params.get('sample.topK') ?? 40),
    '--repeat-penalty', String(params.get('sample.repeat') ?? 1.05),
  );
  const seed = Number(params.get('sample.seed') ?? -1);
  if (seed >= 0) args.push('--seed', String(seed));

  if (config.serving?.apiKey) args.push('--api-key', config.serving.apiKey);

  return args;
}

export function describeCommand() {
  return [state.binary, ...state.args].filter(Boolean).join(' ');
}

/* ---------------------------------------------------------------- lifecycle --- */

function log(stream, line) {
  const entry = { at: new Date().toISOString(), stream, line };
  state.logs.push(entry);
  if (state.logs.length > LOG_LIMIT) state.logs.shift();
  hubRef?.send('engine-log', entry);
}

export function logs(limit = 200) {
  return state.logs.slice(-limit);
}

async function waitForHealth(port, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1500) });
      const body = await res.json().catch(() => ({}));
      if (res.ok && body.status === 'ok') return true;
      if (body.status) log('supervisor', `engine reports: ${body.status}`);
    } catch {
      /* not up yet */
    }
    if (child?.exitCode !== null && child?.exitCode !== undefined) {
      throw new Error(`Engine exited early with code ${child.exitCode}`);
    }
    await new Promise((r) => setTimeout(r, 700));
  }
  throw new Error('Engine did not become healthy in time');
}

export async function start({ modelId, modelPath, port = 8080, config, hardware, timeoutMs = 120_000 }) {
  if (state.status === 'running' && state.modelId === modelId && state.port === port) {
    return status();
  }
  if (child) await stop();

  const binary = await resolveBinary();
  if (!binary) {
    state.status = 'error';
    state.error = 'No llama-server binary found. Download the engine from Settings → Engine.';
    throw new Error(state.error);
  }
  if (!existsSync(modelPath)) {
    state.status = 'error';
    state.error = `Model file missing: ${modelPath}`;
    throw new Error(state.error);
  }

  state.status = 'starting';
  state.port = port;
  state.modelId = modelId;
  state.modelPath = modelPath;
  state.error = null;
  state.logs = [];
  state.args = buildArgs({ modelPath, config, hardware, port });
  state.command = describeCommand();
  hubRef?.send('engine-status', status());

  log('supervisor', `$ ${state.command}`);

  child = spawn(binary.path, state.args, {
    cwd: join(MODELS_DIR, '..'),
    env: { ...process.env, LLAMA_CACHE: join(MODELS_DIR, '..', 'cache') },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  state.pid = child.pid;
  state.startedAt = new Date().toISOString();

  const onData = (stream) => (chunk) => {
    for (const line of chunk.toString().split('\n')) {
      if (line.trim()) log(stream, line.replace(/\s+$/, ''));
    }
  };
  child.stdout.on('data', onData('out'));
  child.stderr.on('data', onData('err'));

  child.on('exit', (code, signal) => {
    log('supervisor', `engine exited (${signal ?? code})`);
    child = null;
    state.pid = null;
    if (state.status !== 'stopping') {
      state.status = code === 0 ? 'stopped' : 'error';
      if (code !== 0 && code !== null) state.error = `Engine exited with code ${code}`;
    } else {
      state.status = 'stopped';
    }
    stopPolling();
    hubRef?.send('engine-status', status());
  });

  try {
    await waitForHealth(port, timeoutMs);
  } catch (err) {
    state.status = 'error';
    state.error = err.message;
    hubRef?.send('engine-status', status());
    await stop();
    throw err;
  }

  state.status = 'running';
  state.startedAt = new Date().toISOString();
  startPolling();
  hubRef?.send('engine-status', status());
  log('supervisor', 'engine ready');
  return status();
}

export async function stop() {
  if (!child) {
    state.status = 'stopped';
    state.pid = null;
    stopPolling();
    hubRef?.send('engine-status', status());
    return status();
  }
  state.status = 'stopping';
  hubRef?.send('engine-status', status());
  const proc = child;
  proc.kill('SIGTERM');
  const exited = await Promise.race([
    new Promise((r) => proc.once('exit', () => r(true))),
    new Promise((r) => setTimeout(() => r(false), 4000)),
  ]);
  if (!exited) {
    log('supervisor', 'engine ignored SIGTERM — killing');
    proc.kill('SIGKILL');
  }
  child = null;
  state.status = 'stopped';
  state.pid = null;
  state.startedAt = null;
  stopPolling();
  hubRef?.send('engine-status', status());
  return status();
}

export async function restart(opts) {
  await stop();
  await new Promise((r) => setTimeout(r, 400));
  return start(opts);
}

/* ------------------------------------------------------------------ metrics --- */

function startPolling() {
  stopPolling();
  pollTimer = setInterval(() => void pollMetrics(), 1000);
  pollTimer.unref?.();
  void pollMetrics();
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

function parsePrometheus(text) {
  const out = {};
  for (const line of text.split('\n')) {
    if (!line || line.startsWith('#')) continue;
    const match = /^([a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$/.exec(line.trim());
    if (!match) continue;
    out[match[1]] = Number(match[3]);
  }
  return out;
}

/** Original Strata serves JSON; llama.cpp serves Prometheus. Preserve the original dashboard facts. */
export function parseEngineMetrics(text) {
  if(text.trim().startsWith('{')) {
    const dashboard=JSON.parse(text);
    if(!dashboard.engine||!dashboard.live)throw new Error('Unrecognized engine metrics');
    const live=dashboard.live;
    return {source:'engine',dashboard,promptTokens:live.prompt_tokens??0,predictedTokens:live.generated??0,promptTps:live.prefill_tok_s_mean??0,outputTps:live.tok_s??0,kvUsage:0,kvTokens:0,requestsProcessing:live.running??(['reading','generating'].includes(live.state)?1:0),requestsDeferred:live.queued??0};
  }
  const m=parsePrometheus(text);
  if(!Object.keys(m).some(key=>/prompt_tokens|tokens_predicted|predicted_tokens|kv_cache|requests_processing/.test(key)))throw new Error('Unrecognized engine metrics');
  const p=name=>m[`llamacpp:${name}`]??m[name]??0;
  return {source:'engine',promptTokens:p('prompt_tokens_total'),predictedTokens:p('tokens_predicted_total'),promptTps:p('prompt_tokens_seconds'),outputTps:p('predicted_tokens_seconds'),kvUsage:p('kv_cache_usage_ratio'),kvTokens:p('kv_cache_tokens'),requestsProcessing:p('requests_processing'),requestsDeferred:p('requests_deferred')};
}

export async function pollMetrics() {
  if (state.status !== 'running') return state.metrics;
  try {
    const res = await fetch(`http://127.0.0.1:${state.port}/metrics${state.metrics.dashboard?'?requests=all':''}`, {
      signal: AbortSignal.timeout(1500),
      headers: state.apiKey ? { authorization: `Bearer ${state.apiKey}` } : undefined,
    });
    if (!res.ok) throw new Error(String(res.status));
    state.metrics = { ...parseEngineMetrics(await res.text()), sampledAt: new Date().toISOString() };
    if(state.metrics.dashboard) {
      state.metrics.dashboard.mcp=mcpStatus;
      if(!mcpPending&&Date.now()-mcpAt>10000) {
        mcpPending=true;mcpAt=Date.now();
        void fetch(`http://127.0.0.1:${state.port}/mcp`,{signal:AbortSignal.timeout(1500),headers:state.apiKey?{authorization:`Bearer ${state.apiKey}`}:undefined})
          .then(async response=>{mcpStatus=response.ok?await response.json():null;})
          .catch(()=>{mcpStatus=null;}).finally(()=>{mcpPending=false;});
      }
    }
  } catch {
    state.metrics = { ...state.metrics, source: 'unavailable', sampledAt: new Date().toISOString() };
  }
  return state.metrics;
}

/* ------------------------------------------------------------------- status --- */

export function status() {
  const binary = state.binary ? { path: state.binary, source: state.binarySource } : null;
  return {
    status: state.status,
    running: state.status === 'running',
    binary,
    pid: state.pid,
    port: state.port,
    modelId: state.modelId,
    modelPath: state.modelPath,
    startedAt: state.startedAt,
    error: state.error,
    args: state.args,
    command: state.command,
    metrics: state.metrics,
    logCount: state.logs.length,
  };
}

export function currentPort() {
  return state.port;
}

export function isRunning() {
  return state.status === 'running';
}
