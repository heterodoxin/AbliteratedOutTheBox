/** Live monitor: one state machine for phase, throughput, memory and thermals. */
import { variant as catalogVariant, footprint } from './catalog.mjs';
import { detect, cachedHardware } from './hardware.mjs';
import { readGpuTelemetry, gpuMemoryTotals } from './device-probes.mjs';
import { createSystemSampler } from './system-telemetry.mjs';
import { state as servingState } from './serving.mjs';
import { originalDashboard } from './monitor-dashboard.mjs';
import { get as getStore } from './store.mjs';
import * as supervisor from './supervisor.mjs';
import { clamp, round } from './util.mjs';

const HISTORY = 60;

const state = {
  phase: 'idle',
  phaseLabel: 'Waiting for a request',
  variantId: null,
  prompt: { done: 0, total: 0, pct: 0 },
  generation: { done:null,total:null },
  queue: { running: 0, waiting: 0 },
  context: { usedTokens: 0, maxTokens: 32768, pct: 0 },
  expertCache: { used: 48, total: 64 },
  hardware: {
    gpus: [],
    gpuLoad: 3,
    vramUsedGb: 2.1,
    vramTotalGb: 24,
    ramUsedGb: 14,
    ramTotalGb: 32,
    gpuTempC: 44,
    tempLimitC: 100,
  },
  history: [],
  totals: {
    tokensIn: 0,
    tokensOut: 0,
    requests: 0,
    avgTps: 0,
    since: new Date().toISOString(),
    /** Time to first token, averaged over the session. */
    ttftMs: 0,
    /** Prompt-cache accounting. Everything the engine read is either reused
     *  from the previous request's KV state or processed fresh. */
    cachedTokens: 0,
    uncachedTokens: 0,
    usageRequests: 0,
    latencyRequests: 0,
  },
  /** Where the context window actually went, so the breakdown can be shown. */
  contextParts: { system: 0, tools: 0, messages: 0 },
  target: { gpuLoad: 3, outputTps: 0, promptTps: 0 },
  /** Which numbers come from the engine and which Strata has to estimate. */
  telemetry: { throughput: 'estimated', thermals: 'estimated', memory: 'estimated', ttftMs: null },
};

let hubRef = null;
let hwCache = null;
let ticker = null;
let memoryWarningShown = false;
let sensorsPending = false, sensorsAt = 0;
let systemReading=null;
const sampleSystem=createSystemSampler();
const finite = value => typeof value==='number'&&Number.isFinite(value);
function aggregate(readings,key,mode='sum') {
  const values=readings.map(g=>g[key]);
  if(!values.length||values.some(v=>!finite(v)))return null;
  return mode==='max'?Math.max(...values):values.reduce((n,v)=>n+v,0)/(mode==='mean'?values.length:1);
}

function hardwareSnapshot() {
  const demo=process.env.STRATA_REAL_ENGINE!=='1',hw=state.hardware;
  const devices=cachedHardware()?.gpus??hwCache?.gpus??[];
  const memoryPools=Math.max(1,devices.filter(g=>!g.unified).length+Number(devices.some(g=>g.unified)));
  const readings=demo?devices.map((gpu,i)=>({id:gpu.id??`${gpu.vendor}:${gpu.index}`,source:'demo',sampledAt:new Date().toISOString(),utilizationPct:Math.round(clamp(hw.gpuLoad+i*2,0,100)),temperatureC:Math.round(hw.gpuTempC+i),memoryUsedGb:round(Math.min(gpu.vramGb,hw.vramUsedGb/memoryPools),2),powerW:gpu.unified?null:round(35+hw.gpuLoad*2,0),powerLimitW:gpu.unified?null:300,pcieGen:gpu.unified?null:4,pcieGenMax:gpu.unified?null:4,pcieWidth:gpu.unified?null:16,pcieRxMbs:gpu.unified?null:round(hw.gpuLoad*1.2,1)})):hw.gpus.map(g=>({...g}));
  const memory = gpuMemoryTotals(devices,readings);
  return {
    gpus:readings,cpus:systemReading?.cpus??[],cpuLoad:systemReading?.cpuLoad??null,
    gpuLoad:aggregate(readings,'utilizationPct','mean'),
    vramUsedGb:memory.used,
    vramTotalGb:memory.total,
    ramUsedGb:systemReading?.ramUsedGb??null,ramTotalGb:systemReading?.ramTotalGb??hw.ramTotalGb,
    gpuTempC:aggregate(readings,'temperatureC','max'),tempLimitC:hw.tempLimitC,
    powerW:aggregate(readings,'powerW'),powerLimitW:aggregate(readings,'powerLimitW'),
    pcieRxMbs:aggregate(readings,'pcieRxMbs'),diskReadGbs:systemReading?.diskReadGbs??null,diskWriteMbs:systemReading?.diskWriteMbs??null,
  };
}

export function init(hub, hardware) {
  hubRef = hub;
  hwCache = hardware;
  supervisor.init(hub);
  const total = hardware.memory.totalGb;
  state.hardware.ramTotalGb = Math.round(total);
  state.hardware.ramUsedGb = Math.round(hardware.memory.usedGb);
  state.hardware.vramTotalGb = Math.round(hardware.accelerator.usableVramGb);
  state.hardware.tempLimitC = hardware.accelerator.unified ? 105 : 90;
  state.expertCache = { used: 0, total: 64 };
  state.history = [];
  state.totals.since = new Date().toISOString();

  if (ticker) clearInterval(ticker);
  ticker = setInterval(tick, 1000);
  ticker.unref?.();
}

function approach(current, target, factor = 0.22) {
  return current + (target - current) * factor;
}

function tick() {
  if(!sensorsPending&&Date.now()-sensorsAt>=3000){
    sensorsPending=true;sensorsAt=Date.now();
    const hardware=cachedHardware()??hwCache;
    void Promise.all([readGpuTelemetry(hardware?.gpus??[]),sampleSystem(hardware?.processors??[])])
      .then(([readings,system])=>{state.hardware.gpus=readings;systemReading=system;})
      .catch(()=>{state.hardware.gpus=[];systemReading=null;})
      .finally(()=>{sensorsPending=false;});
  }
  const hw = state.hardware;
  const busy = state.phase === 'generating' || state.phase === 'reading' || state.phase === 'thinking';

  // Real numbers whenever a real engine is running.
  const engine = supervisor.status();
  const engineMetrics = engine.metrics;
  if(state.hardware.ramUsedGb > state.hardware.ramTotalGb && !memoryWarningShown){
    console.warn('Memory reading exceeded physical capacity; clamping the monitor snapshot.');
    memoryWarningShown = true;
  }
  if(state.hardware.ramUsedGb <= state.hardware.ramTotalGb) memoryWarningShown = false;
  if (engine.running && engineMetrics.source === 'engine') {
    state.telemetry.throughput = 'engine';
    state.telemetry.memory = 'estimated';
    if (state.phase === 'generating' && engineMetrics.outputTps > 0) state.target.outputTps = engineMetrics.outputTps;
    if (engineMetrics.promptTps > 0) state.target.promptTps = engineMetrics.promptTps;
    if (engineMetrics.kvUsage > 0) {
      state.context.usedTokens = Math.round(engineMetrics.kvUsage * state.context.maxTokens);
    }
    const v = engine.modelId ? catalogVariant(engine.modelId) : null;
    if (v) {
      const kvUsed = (state.context.usedTokens / 1024) * (v.kvKbPerToken / 1024);
      const estimate = footprint(v, state.context.usedTokens || 1024).weightsGb + kvUsed;
      state.hardware.vramUsedGb = clamp(estimate, 0.2, state.hardware.vramTotalGb);
    }
    state.queue.running = engineMetrics.requestsProcessing || (busy ? 1 : 0);
    state.queue.waiting = engineMetrics.requestsDeferred || 0;
  } else {
    state.telemetry.throughput = process.env.STRATA_REAL_ENGINE !== '1' ? 'demo' : 'none';
  }

  state.target.gpuLoad = busy ? (state.phase === 'reading' ? 88 : 82) : state.phase === 'loading' ? 64 : 3;
  hw.gpuLoad = clamp(approach(hw.gpuLoad, state.target.gpuLoad) + (Math.random() - 0.5) * 3, 1, 100);

  const modelVram = state.variantId ? 0 : 0;
  const targetVram = busy
    ? Math.min(hw.vramTotalGb, hw.vramTotalGb * 0.86)
    : state.phase === 'loading'
      ? hw.vramTotalGb * 0.55
      : 2.4 + modelVram;
  if (process.env.STRATA_REAL_ENGINE !== '1') hw.vramUsedGb = clamp(approach(hw.vramUsedGb, targetVram, 0.14), 0.4, hw.vramTotalGb);

  const ramTarget = busy ? hw.ramTotalGb * 0.66 : hw.ramTotalGb * 0.44;
  if (process.env.STRATA_REAL_ENGINE !== '1') hw.ramUsedGb = clamp(approach(hw.ramUsedGb, ramTarget, 0.08), 1, hw.ramTotalGb);

  const tempTarget = busy ? hw.tempLimitC - 18 : 44;
  hw.gpuTempC = clamp(approach(hw.gpuTempC, tempTarget, 0.05) + (Math.random() - 0.5), 32, hw.tempLimitC);

  const demo = process.env.STRATA_REAL_ENGINE !== '1';
  state.telemetry.thermals = demo ? 'demo' : 'none';
  state.telemetry.memory = demo ? 'demo' : state.variantId ? 'estimated' : 'none';
  const outNow = state.phase === 'generating' ? state.target.outputTps : 0;
  const promptNow = state.phase === 'reading' ? state.target.promptTps : 0;
  const measuredHardware=hardwareSnapshot();

  state.history = [
    ...state.history.slice(-(HISTORY - 1)),
    {
      t: Date.now(),
      outputTps: round(outNow * (demo ? 0.9 + Math.random() * 0.2 : 1), 1),
      promptTps: round(promptNow * (demo ? 0.9 + Math.random() * 0.2 : 1), 0),
      ...measuredHardware,
    },
  ];

  if (!engine.running) state.queue.running = busy ? 1 : 0;
  state.context.pct = clamp((state.context.usedTokens / Math.max(state.context.maxTokens, 1)) * 100, 0, 100);
  hubRef?.send('monitor', snapshot());
}

export function snapshot() {
  const last = state.history.at(-1);
  const engine = supervisor.status();
  const engineMetrics = engine.metrics;
  if(state.hardware.ramUsedGb > state.hardware.ramTotalGb && !memoryWarningShown){
    console.warn('Memory reading exceeded physical capacity; clamping the monitor snapshot.');
    memoryWarningShown = true;
  }
  if(state.hardware.ramUsedGb <= state.hardware.ramTotalGb) memoryWarningShown = false;
  const result = {
    phase: state.phase,
    phaseLabel: state.phaseLabel,
    variantId: state.variantId ?? getStore().loaded,
    prompt: { ...state.prompt },
    generation: { ...state.generation },
    outputTps: state.phase === 'generating' ? round(state.target.outputTps, 1) : (last?.outputTps ?? 0),
    promptTps: ['reading','thinking','generating'].includes(state.phase) ? Math.round(state.target.promptTps) : (last?.promptTps ?? 0),
    queue: { ...state.queue },
    context: { ...state.context, parts: { ...state.contextParts } },
    expertCache: { ...state.expertCache },
    hardware: hardwareSnapshot(),
    requests: servingState().requests,
    conversationCache: { source: 'unavailable' },
    history: state.history,
    totals: { ...state.totals },
    telemetry: { ...state.telemetry, system:systemReading?'sensor':'none' },
    engine: {
      status: engine.status,
      running: engine.running,
      pid: engine.pid,
      port: engine.port,
      modelId: engine.modelId,
      metricsSource: engineMetrics.source,
      command: engine.command,
      binary: engine.binary,
      startedAt: engine.startedAt,
      logCount: engine.logCount,
    },
  };
  return engine.running&&engineMetrics.source==='engine'&&engineMetrics.dashboard?originalDashboard(engineMetrics.dashboard,cachedHardware()??hwCache,result):result;
}

export function setContextMax(max) {
  state.context.maxTokens = max;
}

export function setRecordedContext({ tokensIn, tokensOut, usageSource, requestId, sessionId }) {
  const available = Number.isFinite(tokensIn) && Number.isFinite(tokensOut);
  state.context = { ...state.context, usedTokens: available ? tokensIn + tokensOut : 0, requestId, sessionId };
  state.context.pct = available ? clamp(state.context.usedTokens / Math.max(1, state.context.maxTokens) * 100, 0, 100) : 0;
  state.contextParts = {};
  state.telemetry.context = available ? usageSource : 'unavailable';
}

/**
 * Records how full the context window is *now*.
 *
 * This used to **accumulate** — `usedTokens += tokensIn + tokensOut` — while the
 * per-part split beside it was computed as an absolute size. The two disagreed,
 * and the accumulating one is simply the wrong quantity: a context window is a
 * sliding space, so what matters is what has to fit in it at this moment, not
 * the sum of everything ever sent through it.
 *
 * The visible result was a meter that climbed past its own maximum within one
 * conversation — "40k / 32K tokens, 100%", clamped, with no way back down — on
 * the one indicator that tells a reader why a local model's answers are getting
 * worse.
 */
export function setContext(parts) {
  state.contextParts = {
    system: parts.system ?? 0,
    tools: parts.tools ?? 0,
    messages: parts.messages ?? 0,
  };
  state.context.usedTokens =
    state.contextParts.system + state.contextParts.tools + state.contextParts.messages;
}

export function resetContext() {
  state.context.usedTokens = 0;
  state.contextParts = { system: 0, tools: 0, messages: 0 };
}

export function setExpertCache(used, total = 64) {
  state.expertCache = { used, total };
}

export async function setModel(id) {
  state.variantId = id;
  if (!id) {
    state.phase = 'idle';
    state.phaseLabel = 'Waiting for a request';
    state.expertCache = { used: 0, total: 64 };
    return;
  }
  const hw = hwCache ?? (await detect());
  const v = getStore();
  const total = 64;
  const demo=process.env.STRATA_REAL_ENGINE!=='1';
  const used = v.loaded&&demo ? Math.round(total * 0.75) : 0;
  state.expertCache = { used, total, source:demo?'demo':'none' };
  state.hardware.vramTotalGb = Math.round(hw.accelerator.usableVramGb);
  state.hardware.ramTotalGb = Math.round(hw.memory.totalGb);
}

export async function beginLoading(id) {
  state.variantId = id;
  state.phase = 'loading';
  state.phaseLabel = 'Loading model into memory';
  state.prompt = { done: 0, total: 0, pct: 0 };
  state.queue.waiting = 0;
  hubRef?.send('monitor', snapshot());
}

export function beginReading(tokens, id) {
  state.variantId = id;
  state.phase = 'reading';
  state.phaseLabel = 'Reading prompt';
  state.prompt = { done: 0, total: tokens, pct: 0 };
  state.generation={done:null,total:Number(getStore().expert?.['sample.maxTokens'])||null};
  state.target.promptTps = process.env.STRATA_REAL_ENGINE !== '1' ? 1400 + Math.random() * 500 : 0;
  hubRef?.send('monitor', snapshot());
}

export function progressReading(done) {
  state.prompt.done = Math.min(done, state.prompt.total);
  state.prompt.pct = state.prompt.total ? (state.prompt.done / state.prompt.total) * 100 : 100;
  if (state.prompt.pct >= 100 && state.phase === 'reading') {
    state.phase = 'thinking';
    state.phaseLabel = 'Thinking';
  }
  hubRef?.send('monitor', snapshot());
}

export function beginGenerating() {
  state.phase = 'generating';
  state.phaseLabel = 'Generating response';
  state.target.outputTps = process.env.STRATA_REAL_ENGINE !== '1' ? 34 + Math.random() * 14 : 0;
  hubRef?.send('monitor', snapshot());
}

export function progressOutput(tokens) {
  if(Number.isFinite(tokens)&&tokens>=0)state.generation.done=tokens;
}

export function beginThinking() {
  state.phase = 'thinking';
  state.phaseLabel = 'Thinking';
  hubRef?.send('monitor', snapshot());
}

export function idle(label = 'Waiting for a request') {
  state.phase = 'idle';
  state.phaseLabel = label;
  state.prompt = { done: 0, total: 0, pct: 0 };
  state.target.outputTps = 0;
  state.target.promptTps = 0;
  state.queue = { running: 0, waiting: state.queue.waiting };
  hubRef?.send('monitor', snapshot());
}

export function record({ tokensIn, tokensOut, durationMs = 0, ttftMs = null, cachedTokens = null, tps: answerTps = null }) {
  if (Number.isFinite(tokensIn)) state.totals.tokensIn += tokensIn;
  if (Number.isFinite(tokensOut)) state.totals.tokensOut += tokensOut;
  state.totals.requests += 1;
  // The average is of answer speeds (tokens over decode time). A request with no decode figure is left out: dividing by the whole request would
  // put the prompt reading into it and read lower than the engine and the chat do.
  if (Number.isFinite(answerTps) && answerTps > 0) {
    const tps = answerTps;
    const n = ++state.totals.usageRequests;
    state.totals.avgTps = round((state.totals.avgTps * (n - 1) + tps) / n, 1);
  }

  // Time to first token, averaged the same way as throughput.
  if (typeof ttftMs === 'number' && ttftMs >= 0) {
    const n = ++state.totals.latencyRequests;
    state.totals.ttftMs = Math.round(((state.totals.ttftMs * (n - 1)) + ttftMs) / n);
  }

  // Include cache reuse only when the inference server reports it.
  if (Number.isFinite(cachedTokens) && Number.isFinite(tokensIn)) {
    const cached = Math.min(cachedTokens, tokensIn);
    state.totals.cachedTokens += cached;
    state.totals.uncachedTokens += Math.max(0, tokensIn - cached);
  }
}

export function queueWaiting(n) {
  state.queue.waiting = n;
}

export function currentOutputTps() {
  return state.target.outputTps;
}
