/**
 * The model ladder Strata ships with, and every engine parameter Expert can reach.
 *
 * Every variant points at a real GGUF on Hugging Face, so "Choose" performs a
 * real download and "Load" starts a real engine process. Sizes are the real
 * file sizes; `kvKbPerToken` is derived from each architecture's layer/KV-head
 * geometry so memory estimates are honest rather than guessed.
 */

import { REAL_VARIANTS } from './real-model-catalog.mjs';
import { DATA_DIR } from './store.mjs';
import { createLocalModels } from './local-models.mjs';
// Design model versions are isolated from real downloadable artifacts.
const DESIGN_VERSIONS = [
 ['Q2_K','Smallest',2.9], ['Q3_K_M','Compact',3.9], ['Q4_K_M','Balanced',4.9],
 ['Q5_K_M','High quality',5.7], ['Q6_K','Very high',6.6], ['Q8_0','Near-original',8.5], ['BF16','Original',16],
];
const DESIGN_FAMILIES = [
 {id:'qwen38',family:'Qwen3.8-Flash-Next',params:30.5,contextMax:1048576,layerCount:48,experts:128,kvKbPerToken:24},
 // Architecture metadata follows Qwen's config; file sizes and speeds remain illustrative.
 {id:'qwen36',family:'Qwen3.6-35B-A3B',params:35,contextMax:262144,layerCount:40,experts:256,kvKbPerToken:20},
];
const BUILT_IN_VARIANTS = process.env.STRATA_REAL_ENGINE === '1' ? REAL_VARIANTS : DESIGN_FAMILIES.flatMap(f=>DESIGN_VERSIONS.map(([quant,name,bits])=>{
 const diskGb=Math.round(f.params*bits/8*10)/10;
 return {id:`${f.id}-${quant.toLowerCase()}`,name,family:f.family,quant,label:`${f.family} · ${name} (${quant})`,params:`${f.params}B`,
 blurb:'The same model at a different precision.',role:quant==='Q4_K_M'?'best-fit':'option',tier:quant==='Q2_K'?'starter':'balanced',
 ramClass:{minGb:diskGb+3,maxGb:diskGb+6,label:`${Math.ceil(diskGb+6)} GB memory`},diskGb,downloadGb:diskGb,contextMax:f.contextMax,
 layerCount:f.layerCount,experts:f.experts,kvKbPerToken:f.kvKbPerToken,hasDraft:true,kvModes:['INT8','FP16'],tags:[],minMemoryGb:diskGb+3,
 source:{repo:'design-preview',file:`${quant}.gguf`,bytes:Math.round(diskGb*1024**3)},designOnly:true};
}));
export const localModels = createLocalModels(DATA_DIR);
export const VARIANTS = [...BUILT_IN_VARIANTS, ...localModels.list()];
export function refreshLocalCatalog() { VARIANTS.splice(BUILT_IN_VARIANTS.length, VARIANTS.length, ...localModels.list()); }

export function variant(id) {
  return VARIANTS.find((v) => v.id === id) ?? null;
}

/** Direct download URL for a variant's weights. */
export function downloadUrl(v) {
  if (v.designOnly || v.local) return undefined;
  return `https://huggingface.co/${v.source.repo}/resolve/main/${encodeURIComponent(v.source.file)}?download=true`;
}

/** Local path a variant's weights live at. */
export function modelFile(v) {
  return `${v.id}/${v.source.file}`;
}

/**
 * Memory footprint at a given context: weights + KV cache + runtime overhead.
 * KV grows linearly with context, which is why context length is the first
 * lever Strata offers in Engine tuning.
 */
export function footprint(v, contextTokens) {
  const kvGb = (v.kvKbPerToken * contextTokens) / 1024 / 1024;
  const overheadGb = v.experts > 0 ? 2.5 : 1.2;
  return { weightsGb: v.diskGb, kvGb, overheadGb, totalGb: v.diskGb + kvGb + overheadGb };
}

/** Expert parameters: grouped, searchable, none hidden. */
/**
 * @param cores physical cores detected on this machine, when known.
 *
 * CPU threads used to be a hardcoded 8 with a hardcoded recommendation of 8,
 * whatever the machine. On a 4-core laptop that produced a warning on a fresh
 * install — before the reader had touched anything — which is exactly the kind
 * of noise that teaches people to ignore warnings.
 */
export function expertParams(cores = null) {
  const threads = cores ? Math.max(4, cores - 2) : 8;
  return [
    // Context & memory
    { id: 'gpu.layers', group: 'Context & memory', label: 'GPU / layer split', effect: 'Which GPUs hold the model', type: 'select', options: ['Automatic', 'All layers', 'Half', 'Quarter', 'CPU only'], value: 'Automatic', recommended: 'Automatic' },
    { id: 'ctx.length', group: 'Context & memory', label: 'Context length', effect: 'Maximum prompt length', type: 'text', value: '128K', recommended: '128K' },
    { id: 'ctx.kvMode', group: 'Context & memory', label: 'KV mode', effect: 'Context memory format', type: 'select', options: ['INT8', 'FP16', 'Q8_0', 'Q4_0'], value: 'INT8', recommended: 'INT8' },
    { id: 'mem.mode', group: 'Context & memory', label: 'Expert cache / RAM mode', effect: 'RAM, VRAM, disk balance', type: 'select', options: ['Automatic', 'GPU first', 'Balanced', 'Memory saver'], value: 'Automatic', recommended: 'Automatic' },
    { id: 'mem.expertCache', group: 'Context & memory', label: 'Expert cache slots', effect: 'How many experts stay resident', type: 'number', value: 64, recommended: 64 },
    { id: 'mem.mmap', group: 'Context & memory', label: 'Memory-map model files', effect: 'Loads from disk on demand', type: 'toggle', value: true, recommended: true },
    { id: 'mem.lock', group: 'Context & memory', label: 'Lock model in memory', effect: 'Prevents the OS paging it out', type: 'toggle', value: false, recommended: false },

    // Performance
    { id: 'perf.draft', group: 'Performance', label: 'Draft / speculation', effect: 'Output acceleration', type: 'select', options: ['On', 'Off', 'Aggressive'], value: 'On', recommended: 'On' },
    { id: 'perf.draftDepth', group: 'Performance', label: 'Draft depth', effect: 'Tokens guessed per step', type: 'number', value: 4, recommended: 4 },
    { id: 'perf.batch', group: 'Performance', label: 'Batch size', effect: 'Prompt reading throughput', type: 'number', value: 2048, recommended: 2048 },
    { id: 'perf.threads', group: 'Performance', label: 'CPU threads', effect: `Threads used off the GPU${cores ? ` — ${cores} cores detected` : ''}`, type: 'number', value: threads, recommended: threads },
    { id: 'perf.mlock', group: 'Performance', label: 'Keep engine warm', effect: 'Avoids reload between turns', type: 'toggle', value: true, recommended: true },
    { id: 'perf.flashAttn', group: 'Performance', label: 'Flash attention', effect: 'Faster attention, less memory', type: 'toggle', value: true, recommended: true },

    // Sampling
    { id: 'sample.temp', group: 'Sampling', label: 'Temperature', effect: 'Randomness of the answer', type: 'number', value: 0.6, recommended: 0.6 },
    { id: 'sample.topP', group: 'Sampling', label: 'Top-p', effect: 'Nucleus sampling cutoff', type: 'number', value: 0.95, recommended: 0.95 },
    { id: 'sample.topK', group: 'Sampling', label: 'Top-k', effect: 'Candidate pool size', type: 'number', value: 40, recommended: 40 },
    { id: 'sample.repeat', group: 'Sampling', label: 'Repeat penalty', effect: 'Discourages loops', type: 'number', value: 1.05, recommended: 1.05 },
    { id: 'sample.seed', group: 'Sampling', label: 'Seed', effect: 'Fixes output for repeatability', type: 'number', value: -1, recommended: -1 },

    { id: 'sample.speedProjection', group: 'Sampling', label: 'Speed projection', effect: 'Experimental control vector; requires engine support', type: 'toggle', value: true, recommended: true },
    { id: 'sample.maxTokens', group: 'Sampling', label: 'Max output tokens', effect: 'Zero uses the engine default', type: 'number', value: 0, recommended: 0 },

    // Server
    { id: 'server.host', group: 'Server', label: 'Server access', effect: 'Preserved setting; host access is configured separately', type: 'select', options: ['Local only', 'Local + key', 'Network', 'Network + key'], value: 'Local only', recommended: 'Local only' },
    { id: 'server.port', group: 'Server', label: 'Internal engine port', effect: 'Public app API uses the host address shown in Serving', type: 'number', value: 8080, recommended: 8080 },
    { id: 'server.parallel', group: 'Server', label: 'Engine request slots', effect: 'This UI host dispatches one request at a time', type: 'number', value: 1, recommended: 1 },
    { id: 'server.queue', group: 'Server', label: 'Queue depth', effect: 'Requests allowed to wait', type: 'number', value: 8, recommended: 8 },
    { id: 'server.timeout', group: 'Server', label: 'Request timeout', effect: 'Seconds before a call is dropped', type: 'number', value: 600, recommended: 600 },
    { id: 'server.cors', group: 'Server', label: 'Allow browser origins', effect: 'This switch does nothing in the app. Web pages from any site can already call the API shown under Serving. To allow only some sites, or to use the engine’s own server (port 8080), add "cors_origins": ["https://your.site"] to the model’s config file.', type: 'toggle', value: false, recommended: false },

    // Advanced
    { id: 'adv.rope', group: 'Advanced', label: 'RoPE scaling', effect: 'Extends context beyond training', type: 'number', value: 1, recommended: 1, advanced: true },
    { id: 'adv.ropeBase', group: 'Advanced', label: 'RoPE base', effect: 'Frequency base for scaling', type: 'number', value: 1000000, recommended: 1000000, advanced: true },
    { id: 'adv.tensorSplit', group: 'Advanced', label: 'Tensor split', effect: 'Layer distribution across GPUs', type: 'text', value: '', recommended: '', advanced: true },
    { id: 'adv.numa', group: 'Advanced', label: 'NUMA policy', effect: 'Memory locality on multi-socket hosts', type: 'select', options: ['Default', 'Distribute', 'Isolate', 'Number of cores'], value: 'Default', recommended: 'Default', advanced: true },
    { id: 'adv.grammar', group: 'Advanced', label: 'Grammar', effect: 'Forces structured output', type: 'text', value: '', recommended: '', advanced: true },
    { id: 'adv.logLevel', group: 'Advanced', label: 'Engine log level', effect: 'Verbosity of engine output', type: 'select', options: ['Error', 'Warn', 'Info', 'Debug'], value: 'Info', recommended: 'Info', advanced: true },
  ];
}

export const CONTEXT_TOKENS = { recommended: 131072, ...Object.fromEntries(Array.from({length:1021}, (_, i) => [`${i+4}k`, (i+4)*1024])) };
