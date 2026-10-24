import { tr } from './i18n';
import type { TimeSkipJob } from './time-skip';

/** What the working view says it is doing: the host's own words (activity.code / text / args.step, notes-split-agent3.md), else from the phase. */
export function activityWords(job: Pick<TimeSkipJob, 'phase' | 'activity'> | undefined): string {
  const a = job?.activity;
  if (a?.code === 'line' && a.text) return a.text;
  if (a?.code === 'trying-version') {
    const step = typeof a.args?.step === 'string' && a.args.step ? String(a.args.step) : '';
    return step || tr('Trying an improved version…');
  }
  switch (a?.code) {
    case 'reading-files': return tr('Reading your files…');
    case 'checking-result': return tr('Checking the result…');
    case 'comparing': return tr('Comparing the two versions…');
    default: break;
  }
  switch (job?.phase) {
    case 'generating': return tr('Trying an improved version…');
    case 'checking': case 'decision': return tr('Checking the result…');
    default: return tr('Reading your files…');
  }
}

/** A rough fraction for the progress line from the phase (the host reports phases, not percentages). */
export function phaseFraction(phase: string | undefined): number {
  switch (phase) {
    case 'queued': return 0.06;
    case 'checkpointing': return 0.18;
    case 'generating': return 0.6;
    case 'checking': return 0.85;
    case 'decision': return 0.95;
    default: return 0.06;
  }
}
