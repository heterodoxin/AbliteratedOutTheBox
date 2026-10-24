/**
 * Tasks as the agent: a scheduled task runs in a Harness session of its own, under the mode it was given, and its result and a
 * notification reach the person. Pure tests always run; the rest need a built Harness (STRATA_HARNESS_DIR).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { createAgentTaskRunner } from '../server/agent/task-run.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { createTaskService, validateTask, nextRun, TASK_MODES } from '../server/tasks.mjs';
import { taskReport, cardLine } from '../server/task-report.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };
const fwd = p => p.replaceAll('\\', '/');
const base = { title: 'Open downloads', prompt: 'open my downloads folder', frequency: 'daily', at: '09:00', timeZone: 'UTC' };

test('a task carries the mode it was created with: plan by default, manual and bypass on request, nothing else', () => {
  assert.deepEqual(TASK_MODES, ['plan', 'manual', 'bypass']);
  assert.equal(validateTask(base).mode, 'plan');
  assert.equal(validateTask({ ...base, mode: 'bypass' }).mode, 'bypass');
  assert.throws(() => validateTask({ ...base, mode: 'auto' }), /what this task may do/);
});

test('the result lists what it did in plain words and says exactly what needed a yes', () => {
  const report = taskReport({ text: 'Done.', cards: [
    { name: 'pwsh', state: 'done', effect: 'List the files', preview: 'Get-ChildItem' },
    { name: 'pwsh', state: 'denied', effect: 'Run a command that may change things: open the folder', preview: 'explorer.exe C:/Users/x/Downloads' },
  ] });
  assert.match(report.text, /^Done\./);
  assert.match(report.text, /\*\*What it did\*\*/);
  assert.match(report.text, /- List the files \(Get-ChildItem\)/);
  assert.match(report.text, /Could not do this by itself: Run a command that may change things: open the folder/);
  assert.match(report.attention, /needed a yes, which nobody could give while it ran by itself/);
  assert.match(report.attention, /Nothing was changed/);
  assert.equal(taskReport({ text: '', cards: [] }).text, 'The task ran and had nothing to report.');
  assert.equal(taskReport({ text: 'ok', cards: [] }).attention, null);
  assert.match(cardLine({ state: 'error', effect: 'Opens a web page.', preview: 'https://x' }), /It did not work/);
});

test('a task scheduled two minutes ahead runs when its minute comes (fake clock), is reported through notify, and the schedule moves on', async () => {
  let clock = Date.parse('2026-10-16T10:00:20Z');
  const stored = [];
  const notified = [];
  const at = new Date(clock + 2 * 60_000).toISOString().slice(11, 16);   // 10:02
  const service = createTaskService({
    read: () => stored, write: records => { stored.length = 0; stored.push(...records); }, now: () => clock,
    notify: task => notified.push({ title: task.title, status: task.status, attention: task.attention }),
    run: async () => ({ text: 'Opened.', attention: null, demo: false }),
  });
  const task = service.put({ ...base, at, mode: 'bypass' });
  assert.equal(task.nextRunAt, `2026-10-16T${at}:00.000Z`);
  service.tick(); await new Promise(r => setTimeout(r, 20));
  assert.equal(notified.length, 0, 'not yet');
  clock = Date.parse(`2026-10-16T${at}:05Z`);
  service.tick(); await new Promise(r => setTimeout(r, 50));
  assert.deepEqual(notified, [{ title: 'Open downloads', status: 'done', attention: null }]);
  assert.equal(service.list()[0].output, 'Opened.');
  assert.equal(service.list()[0].nextRunAt, nextRun({ ...base, at }, clock), 'tomorrow, not again now');
  service.close();
});

async function rig(t, { slots = 2 } = {}) {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-task-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots }];
  const agent = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await agent.close(); await worker.close(); cleanup(dir); });
  const runner = createAgentTaskRunner({ agent, workspace: ws, choice: () => ({ instanceId: 'inst-1' }) });
  return { worker, agent, ws, runner, dir };
}

test('a task runs as the agent in each mode: Plan changes nothing, Manual stops at the first change and says so, Bypass does it', { skip }, async (t) => {
  const r = await rig(t);
  const file = join(r.ws, 'made-by-task.txt');
  const prompt = `SCRIPT:call write ${JSON.stringify({ file_path: fwd(file), content: 'task was here' })}`;

  const plan = await r.runner({ id: 'tp', prompt, mode: 'plan' });
  assert.equal(existsSync(file), false); assert.equal(plan.attention, null, 'Plan refuses without asking anybody');

  const manual = await r.runner({ id: 'tm', prompt, mode: 'manual' });
  assert.equal(existsSync(file), false, 'nobody was there to say yes');
  assert.match(manual.attention, /needed a yes/); assert.match(manual.text, /Could not do this by itself: Create or replace the file/);
  assert.equal(r.agent.status().pendingApprovals, 0, 'nothing is left waiting for an answer');

  const bypass = await r.runner({ id: 'tb', prompt, mode: 'bypass' });
  assert.equal(existsSync(file), true); assert.equal(bypass.attention, null);
  assert.match(bypass.text, /What it did/);
});

test('every run of a task starts a session of its own, and a model that is not ready falls back to the caller', { skip }, async (t) => {
  const r = await rig(t);
  await r.runner({ id: 'ts', prompt: 'SCRIPT:echo one', mode: 'plan' });
  assert.equal(r.agent.sessionOf('task:ts'), null, 'the mapping is dropped after the run');
  const first = r.worker.requests.length;
  await r.runner({ id: 'ts', prompt: 'SCRIPT:echo two', mode: 'plan' });
  assert.ok(r.worker.requests.length > first);
  const none = createAgentTaskRunner({ agent: r.agent, workspace: r.ws, choice: () => null });
  assert.equal(await none({ id: 'x', prompt: 'hi' }), null);
});

test('Stop on the task ends its agent run', { skip }, async (t) => {
  const r = await rig(t);
  const controller = new AbortController();
  const run = r.runner({ id: 'tstop', prompt: 'SCRIPT:slow 300', mode: 'plan' }, controller.signal);
  await new Promise(res => setTimeout(res, 3500));
  controller.abort();
  await assert.rejects(run);
  assert.equal(r.agent.status().activeTurns, 0);
});
