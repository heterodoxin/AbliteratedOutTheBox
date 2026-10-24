/**
 * Model library.
 *
 * With the real engine on, the app downloads models from the Models screen of the native engine, not from this list (see install). Loading
 * starts a real engine process through the supervisor — no model files, no process, no load.
 */
import { existsSync, rmSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { CONTEXT_TOKENS, VARIANTS, downloadUrl, footprint, localModels, refreshLocalCatalog, variant } from './catalog.mjs';

/**
 * Design-iteration mode.
 *
 * Strata is being iterated on as a product surface, so by default the model
 * library is simulated: downloads animate, loads stage, nothing is fetched and
 * no process is spawned. Set STRATA_REAL_ENGINE=1 to use the real path
 * (Hugging Face download + supervised llama.cpp engine).
 */
export const REAL = process.env.STRATA_REAL_ENGINE === '1';
import { MODELS_DIR, diskFreeGb, get as getStore, update } from './store.mjs';
import * as supervisor from './supervisor.mjs';
import { EventHub, nowIso, round, uid } from './util.mjs';
import { locateModelFile } from './model-files.mjs';

/** In-memory runtime state that never needs persisting. */
const runtime = new Map(); // id -> { state, progress, error }
const downloads = new Map(); // id -> AbortController

let hubRef = null;

export function pathFor(v) {
  if (v.local) return v.local.path;
  return locateModelFile(MODELS_DIR, v) ?? join(MODELS_DIR, v.id, v.source.file);
}

export function isOnDisk(id) {
  const v = variant(id);
  if (!v) return false;
  if (v.local) return localModels.ready(v);
  // In design mode "installed" is a state file entry, not a file on disk.
  if (!REAL && getStore().installed?.[id]?.simulated) return true;
  return !!locateModelFile(MODELS_DIR, v);
}

export function diskUsage(id) {
  const v = variant(id);
  if (!v) return 0;
  if (v.local) return v.local.files.reduce((n, f) => n + f.bytes, 0) / 1024 ** 3;
  if (!existsSync(pathFor(v))) return REAL ? 0 : v.diskGb; // simulated installs keep the catalog size
  return statSync(pathFor(v)).size / 1024 ** 3;
}

export function init(hub) {
  hubRef = hub;
  supervisor.init(hub);
  if (!REAL) {
    const previous = getStore();
    const old = Object.keys(previous.installed ?? {}).filter(id => !id.startsWith('local-') && !VARIANTS.some(v=>v.id===id));
    if (old.length) {
      const balanced=VARIANTS.find(v=>v.quant==='Q4_K_M').id;
      update({installed:{...previous.installed,[balanced]:{installedAt:nowIso(),simulated:true}},
        ...(old.includes(previous.loaded)?{loaded:balanced}:{}),...(old.includes(previous.current)?{current:balanced}: {})});
    }
  }

  runtime.clear();
  for (const v of VARIANTS) {
    runtime.set(v.id, { state: isOnDisk(v.id) ? 'installed' : 'available' });
  }

  // Trust the filesystem over the JSON state file.
  const s = getStore();
  const installed = { ...(s.installed ?? {}) };
  for (const id of Object.keys(installed)) if (!VARIANTS.some(v => v.id === id)) delete installed[id];
  for (const v of VARIANTS) {
    if (isOnDisk(v.id) && !installed[v.id]) installed[v.id] = { installedAt: nowIso() };
    if (!isOnDisk(v.id) && installed[v.id]) delete installed[v.id];
  }
  const patch = { installed };
  if (s.loaded && !isOnDisk(s.loaded)) patch.loaded = null;
  if (s.current && !isOnDisk(s.current)) patch.current = null;

  // Files on disk are the source of truth for "current", so a state file that
  // disagrees with the filesystem heals itself.
  const onDisk = VARIANTS.filter((v) => isOnDisk(v.id));
  if (!patch.current && !s.current && onDisk.length) {
    patch.current = (onDisk.find((v) => v.tier === 'starter') ?? onDisk[0]).id;
  }
  update(patch);
}

export function library() {
  const s = getStore();
  const usedGb = VARIANTS.reduce((sum, v) => sum + (isOnDisk(v.id) ? diskUsage(v.id) : 0), 0);
  const engine = supervisor.status();
  const activeDownloads = [...runtime.values()].filter((r) => r.state === 'downloading').length;

  return {
    mode: REAL ? 'real' : 'simulated',
    current: s.current,
    loaded: REAL ? (engine.running ? engine.modelId : null) : s.loaded,
    variants: VARIANTS.map((v) => {
      const entry = runtime.get(v.id) ?? { state: 'available' };
      const onDisk = isOnDisk(v.id);
      const loading = REAL ? engine.status === 'starting' && engine.modelId === v.id : entry.state === 'loading';
      const isLoaded = REAL ? engine.running && engine.modelId === v.id : s.loaded === v.id;
      const state = loading ? 'loading' : entry.state === 'error' ? 'error' : onDisk ? (isLoaded ? 'loaded' : 'installed') : v.local ? 'error' : entry.state;
      return {
        ...v,
        catalog: {
          family: v.family.startsWith('Qwen3-Coder') ? 'Qwen Coder' : 'Qwen',
          version: v.family,
          purpose: v.family.includes('Coder') ? 'Coding-focused instruction model' : 'General text and conversation',
          capabilities: v.family.includes('Coder') ? ['Text', 'Code'] : ['Text'],
          agentApproved: false,
          balancedRank: v.role === 'best-fit' ? 1 : v.tier === 'starter' ? 10 : 5,
          review: 'Capability review pending',
        },
        diskGb: onDisk ? round(diskUsage(v.id), 2) : v.diskGb,
        state,
        progress: entry.progress,
        error: v.local && !onDisk ? 'The local file is missing or has changed. Use a local model again to reconnect it.' : entry.error,
        installedAt: v.local ? onDisk ? v.importedAt : undefined : s.installed?.[v.id]?.installedAt,
        downloadUrl: downloadUrl(v),
      };
    }),
    downloads: { active: activeDownloads, queued: 0, totalGbToday: round(usedGb, 2) },
    disk: { modelsDir: MODELS_DIR, usedGb: round(usedGb, 2), freeGb: round(diskFreeGb(MODELS_DIR), 0) },
    engine: { status: engine.status, binary: engine.binary, port: engine.port, modelId: engine.modelId },
  };
}

function emit() {
  hubRef?.send('models', library());
}

/* ---------------------------------------------------------------- download --- */

/**
 * With the real engine on, models come from the Models screen of the native engine (its installer pins every file by hash). This
 * starter list names files by name only and has no hash, so it is never downloaded: the route refuses before any request is made.
 */
export const LIST_DOWNLOAD_REFUSED = "The app downloads models from the Models screen of the native engine, not from this list.";

export async function install(id) {
  const v = variant(id);
  if (!v) throw new Error(`Unknown model ${id}`);
  if (v.local) throw new Error('Local models are not downloaded. Choose their GGUF file again to reconnect them.');
  if (isOnDisk(id)) {
    const s = getStore();
    update({ installed: { ...(s.installed ?? {}), [id]: s.installed?.[id] ?? { installedAt: nowIso() } }, current: s.current ?? id });
    emit();
    return library();
  }
  if (downloads.has(id)) return library();

  if (!REAL) return simulateInstall(v);
  throw Object.assign(new Error(LIST_DOWNLOAD_REFUSED), { status: 409 });
}

export function cancel(id) {
  const download = downloads.get(id);
  download?.abort();
  // Real jobs retain their guard until their finally block has settled.
  if (download) return library();
  runtime.set(id, { state: 'available' });
  emit();
  return library();
}

export function remove(id) {
  const v = variant(id);
  if (!v) return library();
  const current = library();
  if (current.loaded === id || current.variants.find(row => row.id === id)?.state === 'loading') throw new Error('Unload this model before deleting its files.');
  if (v.local) {
    localModels.remove(id); refreshLocalCatalog(); runtime.delete(id);
    const s = getStore(), installed = { ...s.installed }; delete installed[id];
    update({ installed, current: s.current === id ? null : s.current });
    emit(); return library();
  }
  cancel(id);
  try {
    const file = pathFor(v);
    // Existing folders may contain unrelated weights. Delete only this variant's file.
    if (existsSync(file)) rmSync(file, { force: true });
  } catch (err) {
    hubRef?.send('log', { level: 'warn', message: `Could not delete ${v.family} ${v.quant}: ${err.message}`, at: nowIso() });
    throw new Error(`Could not delete model files: ${err.message}`);
  }
  const s = getStore();
  const installed = { ...(s.installed ?? {}) };
  delete installed[id];
  update({
    installed,
    loaded: s.loaded === id ? null : s.loaded,
    current: s.current === id ? null : s.current,
  });
  runtime.set(id, { state: 'available' });
  emit();
  return library();
}

/* -------------------------------------------------------------------- load --- */

export async function load(id, hooks = {}) {
  const v = variant(id);
  if (!v) throw new Error(`Unknown model ${id}`);
  if (!isOnDisk(id)) throw new Error(`${v.label} is not downloaded yet`);

  const s = getStore();
  const port = Number(s.expert?.['server.port'] ?? 8080);
  const hardware = hooks.hardware;
  const config = hooks.config;

  if (!REAL) return simulateLoad(v, hooks);

  runtime.set(id, { state: 'loading', progress: { pct: 10, mbps: 0, etaSec: 0, stage: 'Starting engine' } });
  emit();
  await hooks.onPhase?.('loading', id);

  try {
    await supervisor.restart({
      modelId: id,
      modelPath: pathFor(v),
      port,
      config: { ...config, layerCount: v.layerCount },
      hardware,
    });
  } catch (err) {
    runtime.set(id, { state: 'error', error: err.message });
    update({ loaded: null });
    emit();
    throw err;
  }

  update({ loaded: id, current: id });
  runtime.set(id, { state: 'loaded' });
  emit();
  await hooks.onPhase?.('ready', id);

  return library();
}

export async function unload(hooks = {}) {
  if (!REAL) {
    update({ loaded: null });
    const s = getStore();
    if (s.current && runtime.has(s.current)) runtime.set(s.current, { state: 'installed' });
    emit();
    await hooks.onPhase?.('idle', null);
    return library();
  }
  await supervisor.stop();
  update({ loaded: null });
  const s = getStore();
  if (s.current && runtime.has(s.current)) runtime.set(s.current, { state: 'installed' });
  emit();
  await hooks.onPhase?.('idle', null);
  return library();
}

/* ------------------------------------------------------- simulated (design) --- */

function simulateInstall(v) {
  const totalSec = 10; // Preview timing is independent of fictional file sizes.
  const startedAt = Date.now();
  runtime.set(v.id, {
    state: 'downloading',
    progress: { pct: 0, mbps: 140, etaSec: totalSec, stage: `Fetching ${v.source.file}` },
  });
  emit();

  const tick = setInterval(() => {
    const elapsed = (Date.now() - startedAt) / 1000;
    const pct = Math.min(100, (elapsed / totalSec) * 100);
    const entry = runtime.get(v.id);
    if (!entry || entry.state !== 'downloading') {
      clearInterval(tick);
      return;
    }
    runtime.set(v.id, {
      state: 'downloading',
      progress: {
        pct,
        mbps: 120 + Math.sin(elapsed * 2.2) * 28,
        etaSec: Math.max(0, totalSec - elapsed),
        stage: pct < 55 ? `Fetching ${v.source.file}` : pct < 88 ? 'Verifying shard hashes' : 'Finalising files',
      },
    });
    emit();

    if (pct >= 100) {
      clearInterval(tick);
      const s = getStore();
      update({
        installed: { ...(s.installed ?? {}), [v.id]: { installedAt: nowIso(), simulated: true } },
        current: s.current ?? v.id,
      });
      runtime.set(v.id, { state: 'installed' });
      hubRef?.send('log', { level: 'info', message: `${v.label} ready (simulated)`, at: nowIso() });
      emit();
    }
  }, 280);

  hubRef?.send('log', { level: 'info', message: `Downloading ${v.label} (simulated)`, at: nowIso() });
  return library();
}

async function simulateLoad(v, hooks) {
  runtime.set(v.id, { state: 'loading', progress: { pct: 12, mbps: 0, etaSec: 3, stage: 'Mapping weights into memory' } });
  emit();
  await hooks.onPhase?.('loading', v.id);

  for (const pct of [38, 64, 86, 100]) {
    await new Promise((r) => setTimeout(r, 420));
    runtime.set(v.id, {
      state: 'loading',
      progress: { pct, mbps: 0, etaSec: Math.max(0, (100 - pct) / 30), stage: pct < 70 ? 'Mapping weights into memory' : 'Warming expert cache' },
    });
    emit();
  }

  update({ loaded: v.id, current: v.id });
  runtime.set(v.id, { state: 'loaded' });
  emit();
  await hooks.onPhase?.('ready', v.id);
  return library();
}

export function loadedVariant() {
  const engine = supervisor.status();
  const id = REAL ? (engine.running ? engine.modelId : null) : getStore().loaded;
  return id ? variant(id) : null;
}

export function runtimeState(id) {
  return runtime.get(id)?.state ?? 'available';
}

export function directoryBusy() {
  return downloads.size > 0 || [...runtime.values()].some(r => ['loading', 'downloading'].includes(r.state)) || supervisor.status().status === 'starting';
}

export function newDownloadId() {
  return uid('dl');
}

export { footprint, CONTEXT_TOKENS };

export function localModelAdded() {
  refreshLocalCatalog();
  emit();
  return library();
}
