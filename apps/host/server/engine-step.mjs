/**
 * The engine comes with the model.
 *
 * A model install job (packaged app) also makes sure this PC has a usable engine: if none is installed, or the signed feed offers
 * a newer compatible one, the job runs Control's component updater for the engine (same feed, signature and hash checks, staged
 * test start) next to the model files. It reports through the job's own progress fields, so the UI shows one combined total.
 *
 * It never throws: a failed engine download leaves the model files alone and becomes `engine.state === 'failed'` with a retry
 * message. Loading, starting and serving never download (STRATA_NO_ACQUIRE stays on for them); this runs only inside the install
 * job the user started. Contract: see phaseA-tests/notes-split-s6.md ("Job contract").
 */

import { plainUpdateError, PHASE_TEXT } from './update-messages.mjs';

const GB = 1e9;
const round = (value, digits = 2) => Math.round(value * 10 ** digits) / 10 ** digits;

export const RETRY = 'Update engine';

export function createEngineStep({ control, signal = null, onChange = () => {} } = {}) {
  const part = { kind: 'engine', name: 'Engine', version: null, state: 'pending', percent: 0, done_gb: 0, total_gb: 0, message: 'Waiting for the model download to start.' };
  let note = null;
  const set = patch => { Object.assign(part, patch); onChange({ ...part }); };

  async function run() {
    if (!control || typeof control.update !== 'function') { set({ state: 'skipped', message: 'This build has no component updater.' }); return { ...part }; }
    try {
      set({ state: 'checking', message: 'Checking the engine for this PC…' });
      let usable = false;
      try { usable = !!(await control.engine()).usable; } catch { /* treated as not usable */ }
      let report = null;
      try {
        report = await control.updateCheck();
      } catch (error) {
        if (usable) { set({ state: 'skipped', message: 'The engine for this PC is installed.' }); return { ...part }; }
        throw error;
      }
      const engine = report.engine ?? {};
      const wanted = engine.status === 'install' || engine.status === 'available';
      if (!wanted) {
        if (!usable && engine.status !== 'current') throw new Error(engine.reasons?.[0] ?? 'The update feed has no engine for this PC.');
        set({ state: 'current', version: engine.installed?.version ?? null, message: 'The engine for this PC is ready.', percent: 100 });
        return { ...part };
      }
      const version = engine.offered?.version ?? null;
      set({ state: 'downloading', version, name: `Engine ${version ?? ''}`.trim(), total_gb: round((engine.download_bytes ?? 0) / GB), message: 'Downloading the engine for this PC…' });
      const result = await control.update({
        engine: true, control: false, signal,
        onEvent: event => {
          if (event.event === 'progress') {
            set({ state: 'downloading', percent: event.percent ?? 0, done_gb: round((event.done ?? 0) / GB), total_gb: round((event.total ?? 0) / GB), message: `Downloading the engine for this PC: ${event.percent ?? 0}%` });
          } else if (event.event === 'phase') {
            const state = { verify: 'verifying', test: 'testing', switch: 'switching' }[event.phase];
            if (state) set({ state, message: PHASE_TEXT[event.phase] ?? part.message });
          } else if (event.event === 'waiting') {
            set({ message: 'Waiting for the model to stop before finishing the engine…' });
          }
        },
      });
      const done = (result.results ?? []).find(r => r.component === 'engine');
      set({ state: 'done', percent: 100, done_gb: part.total_gb, version: done?.installed_after?.version ?? version, message: 'The engine for this PC is ready.' });
    } catch (error) {
      const plain = plainUpdateError(error, 'engine');
      note = `Your model is ready. The engine for this PC couldn't be downloaded. ${plain.next}`;
      set({ state: 'failed', error: plain.headline, details: plain.details, retry: RETRY, message: note });
    }
    return { ...part };
  }

  /** The install progress with the engine part added and one combined download total. Pure: returns a new object. */
  function merge(install) {
    if (!install || typeof install !== 'object') return install;
    const model = install.download ?? null;
    const modelDone = Number.isFinite(model?.done_gb) ? model.done_gb : null;
    const modelRemaining = Number.isFinite(model?.remaining_gb) ? model.remaining_gb : null;
    const active = !['done', 'current', 'skipped', 'failed'].includes(part.state);
    const engineRemaining = active || part.state === 'done' ? Math.max(0, part.total_gb - part.done_gb) : 0;
    /* The install knows its files (strata_mcp files_progress): the model's parts and the picture reader, each counted. Without that, one
       model part from the combined figures. The engine is added either way. */
    const own = Array.isArray(install.parts) && install.parts.some(p => p?.kind === 'model') ? install.parts.filter(p => p?.kind !== 'engine') : null;
    const parts = [
      ...(own ?? [{ kind: 'model', name: model?.file ?? install.model ?? 'Model', state: install.running === false ? 'done' : 'downloading', done_gb: modelDone ?? 0, total_gb: round((modelDone ?? 0) + (modelRemaining ?? 0)) }]),
      { ...part },
    ];
    const merged = { ...install, engine: { ...part }, parts };
    if (model) {
      const done = round((modelDone ?? 0) + (part.state === 'pending' || part.state === 'checking' ? 0 : part.done_gb));
      const remaining = round((modelRemaining ?? 0) + (part.state === 'pending' || part.state === 'checking' ? 0 : engineRemaining));
      const total = done + remaining;
      merged.download = { ...model, done_gb: done, remaining_gb: remaining, percent: total > 0 ? Math.min(100, Math.round(100 * done / total)) : model.percent,
        file: active && part.state === 'downloading' && (modelRemaining ?? 0) <= 0 ? part.name : model.file };
    }
    return merged;
  }

  /** What to tell the user when the model is ready but the engine is not. Null when all is well. */
  function summaryNote() {
    return part.state === 'failed' ? note : null;
  }

  return { run, merge, summaryNote, part: () => ({ ...part }) };
}
