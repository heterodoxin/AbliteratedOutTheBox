/**
 * Small host-owned schedules. No catch-up bursts. A task runs as the agent (a Harness session of its own) under the mode it was
 * given when it was created: `plan` (looks, changes nothing), `manual` (asks first: nobody is there to answer, so it stops at the
 * first thing that needs a yes and says what that was) or `bypass` (full access). The default is `plan`.
 */
import { randomUUID } from 'node:crypto';
import { lastLine } from './task-report.mjs';
const frequencies = ['hourly', 'daily', 'weekdays', 'weekly'];
export const TASK_MODES = ['plan', 'manual', 'bypass'];
export function validateTask(input) {
  const title = String(input.title ?? '').trim(), prompt = String(input.prompt ?? '').trim();
  if (!title || title.length > 120) throw new Error('Use a task name of 1–120 characters.');
  if (!prompt || prompt.length > 12000) throw new Error('Use instructions of 1–12,000 characters.');
  if (!frequencies.includes(input.frequency)) throw new Error('Choose a supported schedule.');
  if (!/^([01]\d|2[0-3]):[0-5]\d$/.test(input.at ?? '')) throw new Error('Choose a valid time.');
  const weekday = Number(input.weekday ?? 1);
  if (!Number.isInteger(weekday) || weekday < 0 || weekday > 6) throw new Error('Choose a weekday.');
  const timeZone = String(input.timeZone ?? 'UTC');
  new Intl.DateTimeFormat('en', { timeZone }).format();
  const mode = input.mode ?? 'plan';
  if (!TASK_MODES.includes(mode)) throw new Error('Choose what this task may do.');
  return { title, prompt, frequency: input.frequency, at: input.at, weekday, timeZone, mode, enabled: input.enabled !== false };
}
export function nextRun(task, after = Date.now()) {
  const format = new Intl.DateTimeFormat('en-GB', { timeZone: task.timeZone, year: 'numeric', month: '2-digit', day: '2-digit', weekday: 'short', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
  const previous = Object.fromEntries(format.formatToParts(after).map(v => [v.type, v.value]));
  const previousDate = `${previous.year}-${previous.month}-${previous.day}`;
  const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  // Search real minutes, so DST gaps are skipped and offsets remain correct.
  const start = Math.floor(after / 60000) * 60000 + 60000;
  for (let n = 0; n < 8 * 24 * 60; n++) {
    const time = start + n * 60000;
    const p = Object.fromEntries(format.formatToParts(time).map(v => [v.type, v.value]));
    const clock = `${p.hour}:${p.minute}`;
    // A daily run is once per calendar day, even when the clock repeats an hour.
    if (task.frequency !== 'hourly' && `${p.year}-${p.month}-${p.day}` === previousDate && `${previous.hour}:${previous.minute}` >= task.at) continue;
    if (task.frequency === 'hourly' ? p.minute !== '00' : clock !== task.at) continue;
    if (task.frequency === 'weekdays' && ['Sat','Sun'].includes(p.weekday)) continue;
    if (task.frequency === 'weekly' && p.weekday !== days[task.weekday]) continue;
    return new Date(time).toISOString();
  }
  throw new Error('Could not calculate the next run.');
}
export function createTaskService({ read, write, run, notify = () => {}, now = Date.now, timeoutMs = 15 * 60_000 }) {
  let records = read().map(t => ({ ...t, status: t.status === 'running' ? 'interrupted' : t.status, nextRunAt: t.enabled && !t.archived ? nextRun(t, now()) : null }));
  const controllers = new Map();
  const save = () => write(records);
  if (records.length) save();
  const list = () => records;
  function find(id) { const t = records.find(t => t.id === id); if (!t) throw new Error('Task not found.'); return t; }
  function put(input, id) {
    const old = id ? find(id) : null;
    if (old?.status === 'running') throw new Error('Stop the current run before editing.');
    const fields = validateTask({ ...old, ...input });
    const task = { ...old, ...fields, id: old?.id ?? randomUUID(), createdAt: old?.createdAt ?? new Date(now()).toISOString(), status: old?.status ?? 'idle', archived: input.archived ?? old?.archived ?? false };
    // "I have seen this": the board hides a finished or failed run the person dismissed. It is only a mark; the result stays in the task.
    if ('ackedAt' in input) task.ackedAt = input.ackedAt ? String(input.ackedAt) : null;
    task.nextRunAt = task.enabled && !task.archived ? nextRun(task, now()) : null;
    records = old ? records.map(t => t.id === id ? task : t) : [...records, task]; save(); return task;
  }
  async function execute(id) {
    const t = find(id); if (t.status === 'running') return;
    if (t.archived) throw new Error('Restore this task first.');
    const controller = new AbortController(); controllers.set(id, controller);
    // A new run replaces the last one's problem (error and the note that something needed a yes) and its dismissal, so the board only ever shows the newest.
    t.status = 'running'; t.lastRunAt = new Date(now()).toISOString(); t.error = null; t.attention = null; t.ackedAt = null; t.lastLine = null; t.finishedAt = null; t.files = [];
    t.nextRunAt = t.enabled ? nextRun(t, now()) : null; save();
    const timeout = setTimeout(() => controller.abort(), timeoutMs); timeout.unref?.();
    try { const result = await run(t, controller.signal); t.output = result.text.slice(0, 100000); t.demo = result.demo; t.attention = result.attention ?? null; t.lastLine = lastLine(result.text); t.files = Array.isArray(result.files) ? result.files.slice(0, 30) : []; t.status = 'done'; }
    catch (e) { t.status = controller.signal.aborted ? 'cancelled' : 'error'; t.error = controller.signal.aborted ? 'Run stopped.' : String(e.message ?? e); }
    finally { t.finishedAt = new Date(now()).toISOString(); clearTimeout(timeout); controllers.delete(id); save(); try { notify(t); } catch { /* a notification is a courtesy */ } }
  }
  function tick() { for (const t of records) if (t.enabled && !t.archived && t.status !== 'running' && t.nextRunAt && Date.parse(t.nextRunAt) <= now()) void execute(t.id); }
  return { list, put, execute, tick, stop: id => controllers.get(id)?.abort(), close: () => controllers.forEach(c => c.abort()) };
}
