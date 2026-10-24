/** Optional diffusion weights have their own library; they never replace the chat model. */
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, statfsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { matchesGGUF } from './model-files.mjs';
import { setTimeout as delay } from 'node:timers/promises';

export const IMAGE_CATALOG = JSON.parse(readFileSync(new URL('./image-catalog.json', import.meta.url), 'utf8'));

export function createImageModels({ dataDir, modelsDir, real = false, changed = () => {}, fetcher = fetch, catalog = IMAGE_CATALOG, demoDuration = 2800 }) {
  mkdirSync(dataDir, { recursive: true });
  const stateFile = join(dataDir, 'image-models.json');
  const choices = new Set(['off', ...catalog.variants.map(v => v.id)]);
  const directory = () => typeof modelsDir === 'function' ? modelsDir() : modelsDir;
  let saved = { schema: 2, selected: 'off', applied: 'off', installed: {}, installedByDirectory: {} };
  try {
    const previous = JSON.parse(readFileSync(stateFile, 'utf8'));
    saved = { ...saved, selected: choices.has(previous.selected) ? previous.selected : 'off',
      applied: choices.has(previous.applied) ? previous.applied : 'off', installedByDirectory: previous.installedByDirectory ?? { [directory()]: previous.installed ?? {} } };
  } catch { /* A new library starts off without downloading anything. */ }
  const runtime = new Map();
  const jobs = new Map();
  const entries = () => saved.installedByDirectory[directory()] ?? {};
  const fileFor = v => entries()[v.id]?.file && matchesGGUF(entries()[v.id].file, v.bytes, directory()) ? entries()[v.id].file : join(directory(), 'image', v.id, v.file);
  const persist = () => writeFileSync(stateFile, JSON.stringify(saved, null, 2));
  function installed(v) {
    const entry = entries()[v.id];
    if (!real) return entry?.simulated === true;
    return entry?.simulated === false && entry.sha256 === v.sha256 && matchesGGUF(fileFor(v), v.bytes, directory());
  }
  // A demo's applied state cannot carry over to the real engine.
  if (real || !catalog.variants.some(v => v.id === saved.applied && installed(v))) saved.applied = 'off';
  function library() {
    return {
      mode: real ? 'real' : 'simulated', family: catalog.family,
      sourceUrl: `https://huggingface.co/${catalog.repo}`, revision: catalog.revision,
      selected: saved.selected, applied: saved.applied,
      // Real image inference needs a dedicated adapter, text encoder and VAE.
      generationAvailable: !real,
      variants: catalog.variants.map(v => ({ ...v, downloadGb: v.bytes / 1e9,
        state: runtime.get(v.id)?.state ?? (installed(v) ? 'installed' : 'available'),
        progress: runtime.get(v.id)?.progress, error: runtime.get(v.id)?.error,
        downloadUrl: `https://huggingface.co/${catalog.repo}/resolve/${catalog.revision}/${v.file}` })),
    };
  }
  const emit = () => changed(library());
  function select(id) {
    if (!choices.has(id)) throw new Error('Choose Off, Q4, Q5, Q6 or Q8.');
    saved = { ...saved, selected: id };
    persist(); emit(); return library();
  }
  function apply() {
    const id = saved.selected;
    if (id !== 'off') {
      const v = catalog.variants.find(v => v.id === id);
      if (!installed(v)) throw new Error('Download the selected image model version first.');
      if (real) throw new Error('Image generation requires a compatible image engine, text encoder and VAE.');
    }
    saved = { ...saved, applied: id };
    persist(); emit(); return library();
  }
  function install(id) {
    const v = catalog.variants.find(v => v.id === id);
    if (!v) throw new Error('Choose an image version to download.');
    if (installed(v) || jobs.has(id)) return library();
    if (jobs.size) throw new Error('Finish or cancel the current image model download first.');
    if (real) {
      mkdirSync(directory(), { recursive: true });
      const disk = statfsSync(directory());
      if (disk.bavail * disk.bsize < v.bytes + 512 * 1024 ** 2) throw new Error('There is not enough free storage for this image model.');
    }
    const controller = new AbortController();
    runtime.set(id, { state: 'downloading', progress: { pct: 0, stage: 'Downloading' } });
    emit();
    const job = Promise.resolve().then(async () => {
      const part = fileFor(v) + '.part';
      try {
        if (!real) {
          for (let step = 1; step <= 10; step++) {
            await delay(demoDuration / 10, undefined, { signal: controller.signal });
            runtime.set(id, { state: 'downloading', progress: { pct: step * 10, stage: step === 10 ? 'Verifying' : 'Downloading' } });
            emit();
          }
        } else {
          mkdirSync(join(directory(), 'image', v.id), { recursive: true });
          const url = library().variants.find(row => row.id === id).downloadUrl;
          const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(30 * 60 * 1000)]);
          const response = await fetcher(url, { signal, redirect: 'follow' });
          if (!response.ok || !response.body) throw new Error(`Image download failed (${response.status}).`);
          const hash = createHash('sha256');
          let received = 0, magic = Buffer.alloc(0), lastEmit = 0;
          const verify = new Transform({ transform(chunk, encoding, callback) {
            received += chunk.length;
            if (magic.length < 4) magic = Buffer.concat([magic, chunk.subarray(0, 4 - magic.length)]);
            if (received > v.bytes) return callback(new Error('The image model file is larger than expected.'));
            hash.update(chunk);
            if (Date.now() - lastEmit > 250) {
              lastEmit = Date.now();
              runtime.set(id, { state: 'downloading', progress: { pct: Math.min(99, received / v.bytes * 100), stage: 'Downloading' } });
              emit();
            }
            callback(null, chunk);
          } });
          await pipeline(Readable.fromWeb(response.body), verify, createWriteStream(part), { signal });
          if (received !== v.bytes || magic.toString() !== 'GGUF' || hash.digest('hex') !== v.sha256) throw new Error('The downloaded image model did not pass verification. Retry the download.');
          signal.throwIfAborted();
          renameSync(part, fileFor(v));
        }
        controller.signal.throwIfAborted();
        saved = { ...saved, installedByDirectory: { ...saved.installedByDirectory, [directory()]: { ...entries(), [id]: { simulated: !real, sha256: v.sha256, installedAt: new Date().toISOString() } } } };
        persist(); runtime.delete(id);
      } catch (error) {
        if (real) rmSync(part, { force: true });
        runtime.set(id, controller.signal.aborted ? { state: 'available' } : { state: 'error', error: error.message });
      } finally {
        jobs.delete(id); emit();
      }
    });
    jobs.set(id, { controller, job });
    return library();
  }
  async function cancel(id) {
    const task = jobs.get(id);
    if (task) { task.controller.abort(); await task.job; }
    return library();
  }
  async function rescan(resetApplied = false) {
    if (jobs.size) throw new Error('Finish or cancel the image download first.');
    runtime.clear();
    const found = { ...entries() };
    if (real) for (const v of catalog.variants) {
      const file = [join(directory(), 'image', v.id, v.file), join(directory(), v.file)].find(path => matchesGGUF(path, v.bytes, directory()));
      if (!file) { delete found[v.id]; continue; }
      const stat = statSync(file), cached = found[v.id];
      if (!(cached?.file === file && cached?.mtimeMs === stat.mtimeMs && cached?.sha256 === v.sha256)) {
        const hash = createHash('sha256');
        for await (const chunk of createReadStream(file)) hash.update(chunk);
        if (hash.digest('hex') !== v.sha256) { delete found[v.id]; continue; }
      }
      found[v.id] = { simulated: false, file, mtimeMs: stat.mtimeMs, sha256: v.sha256, installedAt: cached?.installedAt ?? new Date().toISOString() };
    }
    saved = { ...saved, installedByDirectory: { ...saved.installedByDirectory, [directory()]: found } };
    if (resetApplied || !catalog.variants.some(v => v.id === saved.applied && installed(v))) saved.applied = 'off';
    persist(); emit(); return library();
  }
  return { library, select, apply, install, cancel, rescan, busy: () => jobs.size > 0, whenIdle: () => Promise.all([...jobs.values()].map(v => v.job)) };
}
