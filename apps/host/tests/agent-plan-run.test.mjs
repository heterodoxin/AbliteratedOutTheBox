// /plan, plan goals, goal points and queued messages through the real Harness with a scripted model.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const wait = ms => new Promise(r => setTimeout(r, ms));
const words = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');

async function rig(t) {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-planrun-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const events = { plan: [], queue: [], goal: [] };
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000,
    onPlan: (chatId, event, plan) => events.plan.push({ chatId, event, plan }), onQueue: (chatId, queue) => events.queue.push({ chatId, queue }), onGoal: (chatId, goal) => events.goal.push({ chatId, goal }) });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const base = (chatId, onFrame = () => {}) => ({ chatId, workspacePath: ws, mode: 'bypass', onFrame });
  return { worker, bridge, ws, events, base };
}

test('/plan: the assistant looks (Plan mode), the host sends the plan, and the person can edit, approve or reject it', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.planCommand({ ...r.base('p1', f => out.push(f)), mode: 'manual', line: 'SCRIPT:planreply' });
  const start = out.find(f => f.type === 'start');
  assert.equal(start.mode, 'plan', 'a plan turn is read-only whatever mode the chat is in'); assert.equal(start.planning, true);
  const plan = out.find(f => f.type === 'plan');
  assert.equal(plan.event, 'proposed'); assert.equal(plan.plan.state, 'proposed'); assert.equal(plan.plan.steps.length, 3);
  assert.ok(out.findIndex(f => f.type === 'plan') < out.findIndex(f => f.type === 'done'), 'the plan frame comes before the end of the turn');
  assert.equal(r.events.plan.at(-1).event, 'proposed');
  assert.equal(r.bridge.planState('p1').id, plan.plan.id);
  const edited = r.bridge.planControl({ chatId: 'p1', planId: plan.plan.id, action: 'edit', steps: ['Only step SCRIPT:goalstep', { text: 'Second SCRIPT:goalstep', detail: 'why' }], title: 'My plan' });
  assert.equal(edited.plan.steps.length, 2); assert.equal(edited.plan.state, 'proposed'); assert.equal(edited.plan.title, 'My plan');
  const approved = r.bridge.planControl({ chatId: 'p1', planId: plan.plan.id, action: 'approve' });
  assert.equal(approved.plan.state, 'approved'); assert.match(approved.prompt, /1\. Only step/);
  assert.throws(() => r.bridge.planControl({ chatId: 'p1', planId: 'other', action: 'approve' }), /no plan waiting/);
  const rej = r.bridge.planControl({ chatId: 'p1', planId: plan.plan.id, action: 'reject' });
  assert.equal(rej.plan.state, 'rejected');
  assert.throws(() => r.bridge.planControl({ chatId: 'p1', planId: plan.plan.id, action: 'edit', steps: ['x'] }), /already been used/);
  assert.deepEqual(r.events.plan.map(e => e.event), ['proposed', 'edited', 'approved', 'rejected']);
});

test('Start as goal: each plan step is a goal step, reported one by one, queued messages are delivered between steps, and the plan ends done', { skip }, async (t) => {
  const r = await rig(t);
  await r.bridge.planCommand({ ...r.base('p2'), line: 'SCRIPT:planreply' });
  const plan = r.bridge.planState('p2');
  assert.equal(plan.steps.length, 3);
  const out = [];
  let queued = null;
  await r.bridge.planGoal({ ...r.base('p2', f => {
    out.push(f);
    if (f.type === 'plan-step' && f.index === 1 && f.state === 'running' && !queued) { try { queued = r.bridge.queueMessage({ chatId: 'p2', text: 'Also add a moon SCRIPT:echo noted' }); } catch (e) { out.push({ type: 'test-error', message: e.message }); } }
  }), planId: plan.id });
  assert.deepEqual(out.filter(f => f.type === 'test-error'), []);
  const stepFrames = out.filter(f => f.type === 'plan-step').map(f => `${f.index}:${f.state}`);
  assert.deepEqual(stepFrames, ['1:running', '1:done', '2:running', '2:done', '3:running', '3:done']);
  const lines = words(out);
  assert.match(lines, /First thing SCRIPT:goalstep — done/); assert.match(lines, /Third thing SCRIPT:goalstep — done/);
  assert.ok(out.find(f => f.type === 'plan-step' && f.state === 'done').summary, 'a done step carries a short summary from its closing words');
  const goals = out.filter(f => f.type === 'goal' && f.goal);
  assert.ok(goals.some(f => f.goal.step?.index === 2 && f.goal.step.total === 3 && f.goal.objective === plan.title), 'the goal bar knows the step and shows the plan title');
  assert.equal(r.bridge.planState('p2').state, 'done'); assert.ok(r.bridge.planState('p2').steps.every(s => s.state === 'done'));
  assert.equal(out.filter(f => f.type === 'plan').at(-1).event, 'done');
  assert.equal(queued.message.deliverAt, 'step');
  assert.deepEqual(out.filter(f => f.type === 'queued-message').map(f => f.state), ['queued', 'delivered']);
  assert.equal(out.find(f => f.type === 'queued-message' && f.state === 'delivered').afterStep, 1);
  assert.equal(out.filter(f => f.type === 'queued-release').length, 0);
  assert.match(words(out), /noted/, 'the assistant answered the delivered message');
  assert.equal(out.at(-1).type, 'done');
  assert.deepEqual(r.bridge.takeQueue('p2'), { queue: [], released: [] });
});

test('a goal with numbered points runs point by point and reports each one', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.goalCommand({ ...r.base('p3', f => out.push(f)), line: 'Make the page: 1. draw the base SCRIPT:goalstep 2. add the roof SCRIPT:goalstep' });
  const plan = r.bridge.planState('p3');
  assert.equal(plan.source, 'goal'); assert.equal(plan.title, 'Make the page'); assert.equal(plan.state, 'done');
  assert.match(words(out), /draw the base SCRIPT:goalstep — done/); assert.match(words(out), /add the roof SCRIPT:goalstep — done/);
});

test('a message typed while a goal without points runs is queued, can be taken back, and is handed back when the goal is stopped', { skip }, async (t) => {
  const r = await rig(t);
  assert.throws(() => r.bridge.queueMessage({ chatId: 'q1', text: 'hello' }), /No goal is running/);
  const out = [];
  const run = r.bridge.runTurn({ ...r.base('q1', f => out.push(f)), goal: { action: 'create', objective: 'keep going', maxRounds: 50 } });
  // the goal turn may take a while to start on a busy machine: queue as soon as it runs
  let a = null;
  for (let i = 0; i < 150 && !a; i++) { try { a = r.bridge.queueMessage({ chatId: 'q1', text: 'first thing' }); } catch { await wait(100); } }
  assert.ok(a, 'the goal was running');
  await wait(500);
  const b = r.bridge.queueMessage({ chatId: 'q1', text: 'second thing' });
  assert.equal(a.message.deliverAt, 'end'); assert.deepEqual(b.queue.map(m => m.text), ['first thing', 'second thing']);
  const back = r.bridge.cancelQueued({ chatId: 'q1', id: b.message.id });
  assert.deepEqual(back.queue.map(m => m.text), ['first thing']);
  assert.throws(() => r.bridge.cancelQueued({ chatId: 'q1', id: b.message.id }), /already sent or removed/);
  // Stop may answer 'still stopping' on a busy machine: ask again until the run has ended
  for (let i = 0; i < 5; i++) { const st = await r.bridge.stop('q1'); if (st.stopped || !r.bridge.isRunning('q1')) break; }
  await run;
  const rel = out.find(f => f.type === 'queued-release');
  assert.deepEqual(rel.messages.map(m => m.text), ['first thing']);
  assert.ok(out.findIndex(f => f.type === 'queued-release') < out.findIndex(f => f.type === 'done'));
  assert.deepEqual(out.filter(f => f.type === 'queued-message').map(f => f.state), ['queued', 'queued', 'cancelled']);
  assert.deepEqual(r.bridge.takeQueue('q1'), { queue: [], released: [] });
});

test('Stop in the middle of a plan pauses it with the finished steps kept; Resume goes on from the unfinished step', { skip }, async (t) => {
  const r = await rig(t);
  await r.bridge.planCommand({ ...r.base('p4'), line: 'SCRIPT:planreply' });
  const plan = r.bridge.planState('p4');
  r.bridge.planControl({ chatId: 'p4', planId: plan.id, action: 'edit', steps: ['Quick one SCRIPT:goalstep', 'Slow one SCRIPT:slow 400', 'Last one SCRIPT:goalstep'] });
  const out = [];
  const run = r.bridge.planGoal({ ...r.base('p4', f => out.push(f)), planId: plan.id });
  for (let i = 0; i < 100 && !out.some(f => f.type === 'plan-step' && f.index === 2 && f.state === 'running'); i++) await wait(100);
  await wait(1500);
  assert.equal((await r.bridge.stop('p4')).stopped, true);
  await run;
  const paused = r.bridge.planState('p4');
  assert.equal(paused.state, 'paused'); assert.deepEqual(paused.steps.map(s => s.state), ['done', 'pending', 'pending']);
  assert.equal(out.filter(f => f.type === 'plan').at(-1).event, 'paused');
  const goal = await r.bridge.goalState('p4');
  assert.equal(goal.phase, 'paused'); assert.equal(goal.step.index, 2);
  const again = [];
  const second = r.bridge.goalCommand({ ...r.base('p4', f => again.push(f)), line: 'resume' });
  for (let i = 0; i < 100 && !again.some(f => f.type === 'plan' && f.event === 'step'); i++) await wait(100);
  assert.equal(r.bridge.planState('p4').state, 'running'); assert.equal(r.bridge.planState('p4').steps[1].state, 'running');
  await wait(1000);
  await r.bridge.stop('p4'); await second;
  assert.equal(r.bridge.planState('p4').state, 'paused');
});

test('/research: gathering is not shown, a run that opened no pages is sent back once, and the answer is written in a second step with its sources listed', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.researchCommand({ ...r.base('rs1', f => out.push(f)), line: 'SCRIPT:echo Fact one [1]. https://a.example/x' });
  const shown = words(out);
  assert.equal(shown.split('Fact one').length - 1, 1, 'only the final answer is shown, not the notes of the first two replies: ' + JSON.stringify(shown));
  const asked = r.worker.requests.filter(q => q.raw.includes('Research task:') || q.raw.includes('You have not opened enough pages') || q.raw.includes('Now write the final answer'));
  assert.ok(asked.length >= 3, 'gather, nudge (no pages were opened), write');
  const last = r.worker.requests.at(-1).raw;
  assert.ok(last.includes('Now write the final answer to the research task') && last.includes('## Summary'), 'the last request carries the writing prompt');
  const research = out.find(f => f.type === 'research');
  assert.deepEqual(research.sources, [{ n: 1, title: 'a.example', url: 'https://a.example/x' }]);
  assert.ok(out.findIndex(f => f.type === 'research') < out.findIndex(f => f.type === 'done'));
  assert.equal(out.at(-1).type, 'done');
  // without a topic, or with search turned off, nothing runs
  const help = [];
  await r.bridge.researchCommand({ ...r.base('rs2', f => help.push(f)), line: '' });
  assert.match(words(help), /Type \/research followed by/);
});
