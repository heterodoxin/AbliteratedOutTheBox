import type { TuneProgress } from './native-tune';

/**
 * Which of the five steps the tuner is in, from the stage its status file names (strata_control/autotune.py):
 * start and waiting are the PC check, base the first measurement, plan / rounds / drift / per_request the settings tried,
 * confirm the repeat test, and done the last moment before the result is saved and shown.
 */
export function tunePhase(stage: string | undefined | null): number {
  switch (stage) {
    case 'base': return 1;
    case 'plan': case 'rounds': case 'drift': case 'per_request': return 2;
    case 'confirm': return 3;
    case 'done': return 4;
    default: return 0;
  }
}

/** Every measurement so far: "base" first, then each setting tried, as decode tokens per second. */
export function tuneSpeeds(progress: TuneProgress | undefined | null): { label: string; tok_s: number }[] {
  const list = (progress as (TuneProgress & { speeds?: { label?: unknown; tok_s?: unknown }[] }) | undefined | null)?.speeds;
  if (!Array.isArray(list)) return [];
  return list.filter(x => typeof x?.tok_s === 'number' && Number.isFinite(x.tok_s) && x.tok_s > 0).map(x => ({ label: String(x.label ?? ''), tok_s: x.tok_s as number }));
}
