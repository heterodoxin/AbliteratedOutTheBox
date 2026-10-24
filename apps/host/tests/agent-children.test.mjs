/**
 * Helper agents (the Harness's subagents) in the chat: rows, concurrency, one approval card, Stop, and /goal.
 *
 * Pure tests always run; the rest need a built Harness (STRATA_HARNESS_DIR) and skip loudly without it.
 * The model is scripted; the Harness, its subagent runtime, its approval service, the gate and the bridge are real.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge, plainBlocker, goalEnding, plainGoalError } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { ChildProjector, activityOf } from '../server/agent/children.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };
const fwd = p => p.replaceAll('\\', '/');

// ---------------------------------------------------------------------------------------------------------- pure

const ev = (type, data, extra = {}) => ({ type: 'event', event: { seq: 1, time: extra.time ?? 1000, type, data } });

test('a helper row follows its helper: queued, working, completed with its closing words, tokens counted', () => {
  let t = 0;
  const cp = new ChildProjector({ id: 'c1', name: 'Check the tests', task: 'Run the unit tests and report', now: () => (t += 100) });
  assert.equal(cp.agent().status, 'queued');
  assert.equal(cp.agent().description, 'Waiting for its turn. Run the unit tests and report');
  cp.consume(ev('step/start', { turn: 1, step: 1 }));
  assert.equal(cp.agent().status, 'queued', 'a request that was sent is still waiting until the engine says its first word');
  cp.consume({ type: 'assistant-stream', frame: { type: 'chunk', chunk: { type: 'text-delta', text: 'Running' } } });
  assert.equal(cp.agent().status, 'working');
  assert.equal(cp.agent().description, 'Run the unit tests and report');
  cp.consume({ type: 'assistant-stream', frame: { type: 'chunk', chunk: { type: 'usage', usage: { inputTokens: 100, outputTokens: 20 } } } });
  assert.equal(cp.agent().tokens, 120);
  cp.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: 'All 12 tests pass.' }] } }));
  cp.consume(ev('turn/end', { turn: 1, reason: { kind: 'completed' } }));
  const row = cp.agent();
  assert.equal(row.status, 'completed'); assert.equal(row.description, 'All 12 tests pass.');
  assert.ok(row.durationMs > 0);
  assert.ok(['id', 'name', 'description', 'status', 'durationMs', 'tokens', 'updatedAt'].every(k => k in row), 'exactly the fields the existing rows read');
});

test('a helper that failed or was stopped says so in plain words', () => {
  const failed = new ChildProjector({ id: 'f', name: 'A' });
  failed.consume(ev('step/start', {}));
  failed.consume(ev('turn/end', { reason: { kind: 'error', error: { message: 'fetch failed: ECONNREFUSED' } } }));
  assert.equal(failed.agent().status, 'failed'); assert.match(failed.agent().description, /not reachable/);
  const stopped = new ChildProjector({ id: 's', name: 'B' });
  stopped.consume(ev('turn/end', { reason: { kind: 'aborted' } }));
  assert.deepEqual([stopped.agent().status, stopped.agent().description], ['cancelled', 'Stopped.']);
});

test('only a helper request that needs the person becomes a card, and its id names the helper', () => {
  const cp = new ChildProjector({ id: 'child-1', name: 'Writer' });
  assert.deepEqual(cp.consume(ev('tool/call', { callId: 'k1', name: 'read', arguments: '{"file_path":"a"}' })).frames, [], 'a read is not a card');
  cp.consume(ev('tool/result', { message: { toolCallId: 'k1', content: [{ type: 'text', text: 'x' }] } }));
  cp.consume(ev('tool/call', { callId: 'k2', name: 'write', arguments: '{"file_path":"C:/a.txt","content":"x"}' }));
  const asked = cp.consume(ev('approval/asked', { id: 'a1', callId: 'k2', reason: 'Create or replace the file C:/a.txt' })).frames[0].tool;
  assert.equal(asked.id, 'child-1:k2'); assert.equal(asked.state, 'proposed'); assert.match(asked.effect, /^Writer asks: Create or replace the file/);
  assert.equal(cp.consume(ev('approval/decided', { id: 'a1', outcome: 'rejected' })).frames[0].tool.result, 'Denied. Nothing ran.');
  const done = cp.consume(ev('tool/result', { message: { toolCallId: 'k2', isError: true, content: [{ type: 'text', text: 'Error: the user rejected tool "write"' } ] } })).frames[0].tool;
  assert.equal(done.state, 'denied'); assert.equal(done.result, 'Denied. Nothing ran.');
  assert.equal(activityOf('pwsh'), 'Running a command');
});

test('goal wording: plain, never a code', () => {
  assert.match(plainBlocker({ code: 'round-limit' }), /most steps/);
  assert.match(goalEnding({ phase: 'complete' }), /complete/);
  assert.match(plainGoalError(new Error('goal "x" is not finished; an unfinished goal must be edited')), /already has a goal/);
  assert.doesNotMatch(plainGoalError(new Error('boom')), /boom/);
});

// ------------------------------------------------------------------------------------------------- real Harness

async function rig(t, { slots = 4, serial = false, onGoal } = {}) {
  const worker = await startScriptedWorker({ serial });
  const dir = mkdtempSync(join(tmpdir(), 'strata-kids-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000, ...(onGoal ? { onGoal } : {}) });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const run = async (chatId, message, { mode = 'bypass', answer, onFrame, goal } = {}) => {
    const frames = [];
    const on = f => {
      frames.push({ ...f, at: Date.now() }); onFrame?.(f);
      if (answer && f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId, callId: f.tool.id, decision: answer }).catch(e => frames.push({ type: 'test-error', message: e.message }));
    };
    const result = goal ? await bridge.runTurn({ chatId, workspacePath: ws, mode, goal, onFrame: on }) : await bridge.runTurn({ chatId, message, workspacePath: ws, mode, onFrame: on });
    return { frames, result };
  };
  return { worker, bridge, dir, ws, run };
}
const rows = frames => { const m = new Map(); for (const f of frames.filter(x => x.type === 'agent')) m.set(f.agent.id, [...(m.get(f.agent.id) ?? []), f.agent]); return m; };
const text = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');

test('two helpers run at the same time as two rows, the run ends when both have reported, and the delegation tool shows no raw card', { skip }, async (t) => {
  const r = await rig(t);
  const { frames } = await r.run('kids', 'SCRIPT:spawn 2 slow 40');
  const byId = rows(frames);
  assert.equal(byId.size, 2);
  const names = [...byId.values()].map(history => history.at(-1).name).sort();
  assert.deepEqual(names, ['Helper 1', 'Helper 2']);
  for (const history of byId.values()) {
    assert.ok(history.some(a => a.status === 'working'), 'it worked');
    assert.equal(history.at(-1).status, 'completed');
    assert.ok(history.at(-1).tokens > 0 && history.at(-1).durationMs > 0);
    assert.match(history[1]?.description ?? history[0].description, /SCRIPT:slow 40/, 'the row names its task');
  }
  // concurrent: both were working before either finished (by the order the frames arrived in, not by the clock: a loaded computer delays a frame, it does not reorder it)
  const agentFrames = frames.filter(f => f.type === 'agent');
  const firstDone = agentFrames.findIndex(f => f.agent.status === 'completed');
  for (const id of byId.keys()) {
    const working = agentFrames.findIndex(f => f.agent.id === id && f.agent.status === 'working');
    assert.ok(working >= 0 && working < firstDone, `helper ${id} was working before the first helper finished`);
  }
  assert.equal(frames.filter(f => f.type === 'agent-tool').length, 0, 'no card for the delegation tool itself');
  assert.match(text(frames), /Noted/, 'the parent answered after the helpers reported back');
  assert.equal(frames.at(-1).type, 'done');
  assert.ok(frames.at(-1).at >= Math.max(...frames.filter(f => f.type === 'agent' && f.agent.status === 'completed').map(f => f.at)), 'done comes after the last helper');
  assert.equal(r.bridge.isRunning('kids'), false);
  assert.equal(r.worker.requests.filter(q => q.directive === 'slow').length, 2);
});

test('helpers are limited to the Harness cap (4, or the slot count if larger); the fifth is refused with a message the model reads', { skip }, async (t) => {
  const r = await rig(t, { slots: 2 });
  const { frames } = await r.run('few', 'SCRIPT:spawn 5 slow 30');
  assert.equal(rows(frames).size, 4);
});

test('Manual: a helper that wants to change a file reaches the person as the same single Approve card, and the answer is honoured', { skip }, async (t) => {
  const r = await rig(t);
  const approved = join(r.ws, 'from-helper.txt'), denied = join(r.ws, 'denied.txt');
  const ok = await r.run('kid-ok', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(approved), content: 'by helper' })}`, { mode: 'manual', answer: 'approve' });
  const okCards = ok.frames.filter(f => f.type === 'agent-tool').map(f => f.tool);
  assert.deepEqual([...new Set(okCards.map(c => c.state))], ['proposed', 'running', 'done']);
  assert.equal(new Set(okCards.map(c => c.id)).size, 1); assert.match(okCards[0].id, /^[0-9a-f-]{36}:/);
  assert.match(okCards[0].effect, /^Helper 1 asks: Create or replace the file/);
  assert.equal(readFileSync(approved, 'utf8').trim(), 'by helper');
  const no = await r.run('kid-no', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(denied), content: 'x' })}`, { mode: 'manual', answer: 'deny' });
  assert.equal(existsSync(denied), false);
  assert.equal(no.frames.filter(f => f.type === 'agent-tool').at(-1).tool.state, 'denied');
  assert.equal(no.frames.filter(f => f.type === 'agent-tool').at(-1).tool.result, 'Denied. Nothing ran.');
  assert.equal(rows(no.frames).size, 1);
});

test("a chat cannot answer another chat's helper question", { skip }, async (t) => {
  const r = await rig(t);
  let asked = null;
  const file = join(r.ws, 'x.txt');
  const run = r.run('owner', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(file), content: 'x' })}`, { mode: 'manual', onFrame: f => { if (f.type === 'agent-tool' && f.tool.state === 'proposed') asked = f.tool; } });
  for (let i = 0; i < 200 && !asked; i++) await new Promise(res => setTimeout(res, 50));
  assert.ok(asked, 'the question reached the person');
  await assert.rejects(r.bridge.approve({ chatId: 'someone-else', callId: asked.id, decision: 'approve' }), /no longer waiting/);
  assert.equal(r.bridge.pendingFor('owner').length, 1);
  await r.bridge.approve({ chatId: 'owner', callId: asked.id, decision: 'deny' });
  await run;
  assert.equal(existsSync(file), false);
});

test('Stop on the chat stops its helpers, ends the run quickly and starts no further turn', { skip }, async (t) => {
  const r = await rig(t);
  const started = Date.now();
  let stopped = null;
  const { frames } = await r.run('stopper', 'SCRIPT:spawn 2 slow 2000', { onFrame: f => {
    if (f.type === 'agent' && f.agent.status === 'working' && !stopped) stopped = new Promise(res => setTimeout(res, 300)).then(() => r.bridge.stop('stopper'));
  } });
  const result = await stopped;
  assert.equal(result.stopped, true);
  const final = [...rows(frames).values()].map(h => h.at(-1).status);
  assert.deepEqual(final, ['cancelled', 'cancelled']);
  assert.ok(Date.now() - started < 40_000, 'seconds, not the 80 s the helpers would have taken (a generous bound: a loaded computer is slow, never that slow)');
  const done = frames.at(-1);
  assert.equal(done.type, 'done'); assert.equal(done.cancelled, true);
  assert.equal(r.bridge.isRunning('stopper'), false);
  assert.ok(!/Noted/.test(text(frames)), 'the parent was not woken to comment on stopped helpers');
});

test('/goal: sets a goal, runs its rounds in the chat, says how it ended, refuses a second unfinished goal, and Stop pauses it', { skip }, async (t) => {
  const r = await rig(t);
  const frames = [];
  const say = line => { const out = []; return r.bridge.goalCommand({ chatId: 'g', line, workspacePath: r.ws, mode: 'bypass', onFrame: f => { out.push(f); frames.push(f); } }).then(() => out); };
  assert.match(text(await say('')), /No goal is set/);
  const ran = await r.run('g', null, { goal: { action: 'create', objective: 'say hello a few times', maxRounds: 3 } });
  // The round is a live status now (it used to be an italic line of text that stayed on screen after the model moved on).
  assert.ok(!/Working on the goal/.test(text(ran.frames)), 'no goal-step text in the message');
  assert.ok(ran.frames.some(f => f.type === 'status' && f.phase === 'goal' && f.code === 'goal-step' && /^Working on the goal…$/.test(f.text) && f.args.round >= 1), 'a goal-step status was sent');
  const goalFrames = ran.frames.filter(f => f.type === 'goal');
  assert.ok(goalFrames.length >= 2 && goalFrames[0].goal.objective === 'say hello a few times' && goalFrames[0].goal.maxRounds === 3, 'the goal frame carries the objective and the round limit');
  assert.equal(goalFrames.at(-1).goal.phase, 'blocked'); assert.match(goalFrames.at(-1).goal.blocked, /most steps/);
  assert.equal(ran.frames.filter(f => f.type === 'status').at(-1).phase, 'idle', 'the status line is cleared before the end');
  assert.ok(ran.frames.findIndex(f => f.type === 'status' && f.phase === 'idle') < ran.frames.findIndex(f => f.type === 'done'));
  assert.match(text(ran.frames), /The goal stopped: it worked for the most steps/);
  assert.equal(ran.frames.at(-1).type, 'done');
  assert.match(text(await say('')), /stopped/);
  assert.match((await say('a different goal')).find(f => f.type === 'error').message, /already has a goal/);
  assert.match(text(await say('clear')), /cleared/);
  // a long goal is paused by Stop, and can be resumed
  const slowGoal = r.run('g2', null, { goal: { action: 'create', objective: 'keep going', maxRounds: 50 } });
  r.worker.requests.length = 0;
  await new Promise(res => setTimeout(res, 3500));
  const stop = await r.bridge.stop('g2');
  assert.equal(stop.stopped, true);
  await slowGoal;
  const before = r.worker.requests.length;
  await new Promise(res => setTimeout(res, 1500));
  assert.equal(r.worker.requests.length, before, 'no further round starts after Stop');
  assert.match(text(await r.bridge.goalCommand({ chatId: 'g2', line: '', workspacePath: r.ws, mode: 'bypass', onFrame: f => frames.push(f) }).then(() => frames.filter(f => f.type === 'delta').slice(-1))), /paused/);
});

test('one request slot: the second helper shows as waiting for its turn, then works, and the run still ends cleanly', { skip }, async (t) => {
  const r = await rig(t, { slots: 1, serial: true });
  const { frames } = await r.run('queue', 'SCRIPT:spawn 2 slow 25');
  const byId = rows(frames);
  assert.equal(byId.size, 2, 'both helpers exist (they are not refused)');
  const histories = [...byId.values()];
  // walk the frames in order: at some moment one helper was working while the other was waiting for its turn
  const now = new Map(); let overlap = null;
  for (const f of frames.filter(x => x.type === 'agent')) {
    now.set(f.agent.id, f.agent);
    const all = [...now.values()];
    const waiting = all.find(a => a.status === 'queued' && a.description.startsWith('Waiting for its turn.'));
    if (waiting && all.some(a => a.status === 'working')) overlap ??= waiting;
  }
  assert.ok(overlap, 'one helper visibly waited for its turn while the other worked');
  for (const h of histories) assert.equal(h.at(-1).status, 'completed');
  assert.equal(frames.at(-1).type, 'done');
});

test('goal bar: the state, pause, edit and remove work from outside a turn, and every change is announced', { skip }, async (t) => {
  const announced = [];
  const r = await rig(t, { onGoal: (chatId, goal) => announced.push({ chatId, goal }) });
  assert.equal(await r.bridge.goalState('bar'), null, 'no goal in a chat that never had one');
  await assert.rejects(r.bridge.goalControl({ chatId: 'bar', action: 'pause' }), /no goal/i);
  await r.run('bar', null, { goal: { action: 'create', objective: 'say hello a few times', maxRounds: 1 } });
  const shown = await r.bridge.goalState('bar');
  assert.equal(shown.objective, 'say hello a few times'); assert.equal(shown.maxRounds, 1); assert.ok(['blocked', 'complete', 'paused', 'active'].includes(shown.phase));
  await assert.rejects(r.bridge.goalControl({ chatId: 'bar', action: 'edit', text: '  ' }), /few words/);
  const edited = await r.bridge.goalControl({ chatId: 'bar', action: 'edit', text: 'say goodbye instead' });
  assert.equal(edited.goal.objective, 'say goodbye instead');
  assert.equal(announced.at(-1).goal.objective, 'say goodbye instead', 'the edit was announced for the bar');
  const cleared = await r.bridge.goalControl({ chatId: 'bar', action: 'clear' });
  assert.deepEqual(cleared, { goal: null }); assert.equal(announced.at(-1).goal, null);
  assert.equal(await r.bridge.goalState('bar'), null);
  await assert.rejects(r.bridge.goalControl({ chatId: 'bar', action: 'explode' }), /no goal|not known/i);
});

test('mode change mid-run: Manual to Bypass settles the question already waiting and the very next tool call does not ask again', { skip }, async (t) => {
  const r = await rig(t);
  const first = join(r.ws, 'before.txt');
  let asked = null; const cards = [];
  const run = r.run('mid', `SCRIPT:spawn 1 call write ${JSON.stringify({ file_path: fwd(first), content: 'one' })}`, { mode: 'manual', onFrame: f => { if (f.type === 'agent-tool') { cards.push(f.tool); if (f.tool.state === 'proposed') asked = f.tool; } } });
  for (let i = 0; i < 200 && !asked; i++) await new Promise(res => setTimeout(res, 50));
  assert.ok(asked, 'Manual asked first');
  const switched = await r.bridge.setMode('mid', 'bypass');
  assert.deepEqual({ applied: switched.applied, mode: switched.mode }, { applied: true, mode: 'bypass' });
  assert.equal(switched.settled, 1, 'the waiting question was settled by the new mode');
  await run;
  assert.equal(readFileSync(first, 'utf8').trim(), 'one');
  assert.equal(r.bridge.pendingFor('mid').length, 0);
  // Bypass then Plan: nothing waiting, and the chat now reports Plan to the next call.
  assert.equal((await r.bridge.setMode('mid', 'plan')).mode, 'plan');
  assert.deepEqual(await r.bridge.setMode('nobody-yet', 'manual'), { applied: false, mode: 'manual', reason: 'no session yet' });
});

test('the gate stops a helper asking once the chat has left Manual', async () => {
  const { apply, MANUAL_PRESET } = await import('../server/agent/plugins/strata-gate.mjs');
  let handler; const appended = [];
  const parent = { header: { origin: 'user' } };
  const child = { header: { origin: 'subagent', parentSession: 'p' }, append: (type, data) => appended.push([type, data.policy]) };
  let preset = MANUAL_PRESET; let override = 'ask';
  apply({ on: (_e, h) => { handler = h; }, inject: () => {}, sessions: new Map([['p', parent]]), permissionPresets: { current: () => preset }, approval: { overrideOf: () => override } });
  const exec = { name: 'write', agent: { session: child }, input: { file_path: 'x' } };
  await handler(exec, async () => ({ kind: 'allow' }));
  assert.deepEqual(appended, [], 'in Manual the helper keeps asking');
  preset = 'strata-bypass-ish';
  await handler(exec, async () => ({ kind: 'allow' }));
  assert.deepEqual(appended, [['approval/policy', 'never']], 'the first call after the switch un-pins the helper');
  override = 'never';
  await handler(exec, async () => ({ kind: 'allow' }));
  assert.equal(appended.length, 1, 'and does it once');
});
