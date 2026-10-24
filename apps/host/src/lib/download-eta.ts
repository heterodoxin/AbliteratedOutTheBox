import { useEffect, useRef, useState } from 'react';
import { tr } from './i18n';

/** Minutes left from recent samples of "GB done": the rate over the last minute and a half, so a stall or a burst does not
 *  swing the estimate. Null until there is enough to say (about 10 s of samples) or when nothing is moving. */
export function etaMinutes(samples: { at: number; done: number }[], remainingGb: number | undefined): number | null {
  if (!Number.isFinite(remainingGb) || samples.length < 2) return null;
  const first = samples[0], last = samples[samples.length - 1];
  const seconds = (last.at - first.at) / 1000, gained = last.done - first.done;
  if (seconds < 10 || gained <= 0) return null;
  return Math.max(1, Math.ceil((remainingGb! / (gained / seconds)) / 60));
}

export function etaLabel(minutes: number | null): string {
  if (minutes === null) return '';
  if (minutes >= 90) {
    const h = Math.floor(minutes / 60), m = minutes % 60;
    return m ? tr('About {0} h {1} min left', [h, m]) : tr('About {0} h left', [h]);
  }
  return tr('About {0} min left', [minutes]);
}

/** The approximate time left of a running download, for the progress bar. `key` restarts the estimate for another job. */
export function useDownloadEta(download: { done_gb?: number; remaining_gb?: number } | undefined, key: string | undefined): string {
  const samples = useRef<{ at: number; done: number }[]>([]);
  const owner = useRef<string | undefined>(undefined);
  const [label, setLabel] = useState('');
  const done = download?.done_gb, remaining = download?.remaining_gb;
  useEffect(() => {
    if (owner.current !== key) { owner.current = key; samples.current = []; }
    if (!Number.isFinite(done)) { setLabel(''); return; }
    const now = Date.now();
    const list = samples.current;
    if (!list.length || list[list.length - 1].done !== done) list.push({ at: now, done: done! });
    while (list.length > 2 && now - list[0].at > 90_000) list.shift();
    setLabel(etaLabel(etaMinutes(list, remaining)));
  }, [done, remaining, key]);
  return label;
}

/** What the progress UI reads from an install job's `install` object: the parts it is downloading (the model's files and,
 *  when the job adds it, the engine) summed into ONE count. A part is {kind, name, done_gb, total_gb, state}. Without `parts`
 *  the job's combined `download` fields ({done_gb, total_gb, remaining_gb, percent}) are used as they are. */
export interface InstallParts { parts?: { done_gb?: number; total_gb?: number; state?: string }[]; download?: { done_gb?: number; total_gb?: number; remaining_gb?: number; percent?: number; file?: string } }
export function installProgress(install: InstallParts | undefined): { done_gb: number; total_gb: number; remaining_gb: number; percent: number; file?: string } | undefined {
  const parts = (install?.parts ?? []).filter(p => Number.isFinite(p.total_gb) && (p.total_gb ?? 0) > 0 && p.state !== 'skipped' && p.state !== 'current');
  if (parts.length) {
    const total = parts.reduce((n, p) => n + (p.total_gb ?? 0), 0);
    const done = Math.min(total, parts.reduce((n, p) => n + Math.min(p.done_gb ?? 0, p.total_gb ?? 0), 0));
    return { done_gb: done, total_gb: total, remaining_gb: Math.max(0, total - done), percent: Math.min(99, Math.floor(100 * done / total)), file: install?.download?.file };
  }
  const d = install?.download;
  if (d && Number.isFinite(d.done_gb) && Number.isFinite(d.total_gb)) return { done_gb: d.done_gb!, total_gb: d.total_gb!, remaining_gb: Number.isFinite(d.remaining_gb) ? d.remaining_gb! : Math.max(0, d.total_gb! - d.done_gb!), percent: d.percent ?? 0, file: d.file };
  return undefined;
}

/** Which part of a multi-part install is being fetched now: the model's parts (with "part 2 of 4"), the picture reader or the engine. */
export interface InstallStage { kind: 'model' | 'vision' | 'engine' | 'draft' | 'other'; index?: number; count?: number }
export function installStage(install: { parts?: { kind?: string; state?: string; files_done?: number; files_total?: number }[]; running?: boolean } | undefined): InstallStage | undefined {
  const parts = install?.parts ?? [];
  const now = parts.find(p => p.state === 'downloading') ?? parts.find(p => !['done', 'current', 'skipped', 'failed'].includes(p.state ?? ''));
  if (!now) return undefined;
  const kind = now.kind === 'model' || now.kind === 'vision' || now.kind === 'engine' || now.kind === 'draft' ? now.kind : 'other';
  const count = Number(now.files_total) > 1 ? Number(now.files_total) : undefined;
  return { kind, ...(count ? { index: Math.min(count, Number(now.files_done ?? 0) + 1), count } : {}) };
}
/** The sentence for it, in plain words. */
export function stageText(stage: InstallStage | undefined): string {
  if (!stage) return tr('Downloading model files…');
  if (stage.kind === 'vision') return tr('Downloading the picture reader…');
  if (stage.kind === 'draft') return tr('Preparing your model…');   // the speed-up layer's own download (setup step 6)
  if (stage.kind === 'engine') return tr('Downloading the engine for this PC…');
  if (stage.count && stage.index) return tr('Downloading model files · part {0} of {1}', [stage.index, stage.count]);
  return tr('Downloading model files…');
}
