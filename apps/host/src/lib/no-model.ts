import type { NativeState } from './native';

/**
 * Why a message cannot be sent right now, in plain words, or null when a model can answer. One answer for the composer's Send and for the
 * store, so a run never starts without a model. A model that is still downloading shows its progress instead of "none".
 */
export function noModelNotice(native: NativeState | null, opts: { remote?: boolean; libraryLoaded?: boolean } = {}): { title: string; message: string } | null {
  if (opts.remote || opts.libraryLoaded) return null;
  if (native?.instances.some(i => i.status === 'ready')) return null;
  const downloading = native?.jobs?.find(j => j.state === 'running' && (j.result?.install?.download || /install|prepare|download|pull|fetch/i.test(j.kind)));
  if (downloading) {
    const percent = Number(downloading.result?.install?.download?.percent);
    return { title: 'A model is downloading', message: Number.isFinite(percent) ? `Your model is downloading (${Math.round(percent)}%). You can chat when it has finished.` : 'Your model is downloading. You can chat when it has finished.' };
  }
  if (native?.instances.some(i => i.status === 'loading')) return { title: 'The model is loading', message: 'Your model is still loading. You can chat in a moment.' };
  return { title: 'No model is loaded', message: 'No model is loaded. Load one in Models first.' };
}
