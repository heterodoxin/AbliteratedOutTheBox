import type { NativeJob } from './native';

/** What the host says about a job a restart cut short. */
export const INTERRUPTED = 'The host restarted before completion. Check diagnostics, then retry; completed files are kept.';
const updateKinds = new Set(['update-strata', 'update-engine']);

/**
 * A later finished update of the same component settles the restart interruption of an earlier one (ported from #3578 by medking82).
 * The records are never changed: a running retry, another component or an ordinary failure stays a plain failure.
 */
export function recoveredInterruptions(jobs: readonly NativeJob[]): Set<string> {
  const time = (job: NativeJob) => Date.parse(job.startedAt);
  return new Set(jobs.filter(job => job.state === 'error' && updateKinds.has(job.kind) && job.error === INTERRUPTED && Number.isFinite(time(job))
    && jobs.some(later => later.state === 'done' && later.kind === job.kind && Number.isFinite(time(later)) && time(later) > time(job))).map(job => job.id));
}

/** The text of a failed job's alert. The alert stays red; an interruption a later update has settled says so in plain words. */
export function jobErrorText(job: NativeJob, jobs: readonly NativeJob[]): string {
  return recoveredInterruptions(jobs).has(job.id)
    ? 'This update was cut short when the app restarted. A later update finished, so nothing needs doing.'
    : job.error ?? job.summary;
}
