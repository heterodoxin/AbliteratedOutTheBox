import { tr, localeTag } from '../lib/i18n';
import { Icon } from './ui';
import { Markdown } from './Markdown';
import { ErrorNotice } from './ui';
import type { Task } from '../lib/tasks';

const FREQUENCY: Record<string, string> = { hourly: 'Every hour', daily: 'Every day', weekdays: 'Weekdays', weekly: 'Every week' };
const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const MODE: Record<string, string> = { plan: 'Look only, change nothing', manual: 'Stop and tell me before changing anything', bypass: 'Bypass permissions' };

const when = (iso?: string) => (iso ? new Date(iso).toLocaleString(localeTag(), { dateStyle: 'medium', timeStyle: 'short' }) : '');

/** The schedule in one line: "Weekdays · 08:00", "Every week · Friday · 16:00". */
export function scheduleText(t: Task): string {
  const base = tr(FREQUENCY[t.frequency] ?? t.frequency);
  if (t.frequency === 'hourly') return base;
  return [base, t.frequency === 'weekly' ? tr(WEEKDAYS[t.weekday] ?? '') : '', t.at].filter(Boolean).join(' · ');
}

export type TaskState = 'running' | 'active' | 'paused';
export const taskState = (t: Task): TaskState => (t.status === 'running' ? 'running' : t.enabled ? 'active' : 'paused');

/**
 * An opened task, skimmable: when it runs, what it may do, what it did last, then the instructions and the last answer.
 * The title, the state and the actions (pause or resume, delete, Edit, Run now) are in the header the screen draws.
 */
export function TaskDetail({ task }: { task: Task }) {
  const state = taskState(task);
  const result = task.status === 'running' ? { label: 'Running…', tone: '' }
    : task.error ? { label: 'Last run failed', tone: 'is-bad' }
      : task.lastRunAt || task.output ? { label: 'Done', tone: 'is-good' } : { label: 'Not run yet', tone: '' };
  return <div className="task-detail">
    <h2 className="task-name">{task.title}</h2>
    <section className="st-card task-facts" aria-label={tr("Task summary")}>
      <div><span className="label">{tr("Schedule")}</span><strong><Icon name="clock" size={16}/>{scheduleText(task)}</strong>
        <span className="small muted">{tr(state === 'paused' ? 'Paused' : task.nextRunAt ? tr("Next: {0}", [when(task.nextRunAt)]) : 'Calculating the next run…')}</span></div>
      <div><span className="label">{tr("What it may do")}</span><strong><Icon name="shield" size={16}/>{tr(MODE[task.mode ?? 'plan'])}</strong></div>
      <div><span className="label">{tr("Last result")}</span><strong className={result.tone}>{result.tone === 'is-good' ? <Icon name="check" size={16}/> : null}{tr(result.label)}</strong>
        <span className="small muted">{when(task.lastRunAt)}</span></div>
    </section>
    <section className="st-card"><h2 className="label">{tr("What it does")}</h2><p className="task-instructions">{task.prompt}</p></section>
    <section className="st-card"><h2 className="label">{tr("Last answer")}{task.demo ? tr(' · Example') : ''}</h2>
      {task.error ? <ErrorNotice message={task.error} next={tr("Try running this task again.")}/> : task.output ? <div className="task-answer"><Markdown text={task.output}/></div> : <p className="small muted">{tr("Run this task to see its first result here.")}</p>}</section>
  </div>;
}
