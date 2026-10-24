/**
 * The chat Mode menu (Manual / Plan / Bypass) as real Harness behaviour.
 *
 * The Harness is real: its process, its sandbox, its approval gate, its tools, its `/permission` command.
 * Only the model is scripted (tests/helpers/scripted-worker.mjs), and it behaves like a careful model: when the
 * sandbox denies a change and offers an escalation, it retries the exact call once with `sandbox_permissions`.
 * Every test checks the file on disk, not only the frames, because the frames are what the Harness reports.
 *
 * Needs STRATA_HARNESS_DIR (a built Harness); without it these tests skip, loudly.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { MODE_PRESETS, modeOf, buildProfile } from '../server/agent/profile.mjs';
import { NOT_IN_MODE, TurnProjector } from '../server/agent/events.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* the Harness sandbox leaves access entries; best effort */ } };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const lines = file => readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).length;
const tools = frames => frames.filter(f => f.type === 'agent-tool');
const states = frames => tools(frames).map(f => f.tool.state);
const text = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');

async function rig(t) {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-modes-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  /** One turn in `mode`; `answer` decides what the person says to an approval card (approve | deny | undefined = nobody answers). */
  const run = async (chatId, message, { mode, answer, onFrame } = {}) => {
    const frames = [];
    await bridge.runTurn({
      chatId, message, workspacePath: ws, mode,
      onFrame: f => {
        frames.push(f); onFrame?.(f);
        if (answer && f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId, callId: f.tool.id, decision: answer }).catch(e => frames.push({ type: 'test-error', message: e.message }));
      },
    });
    return frames;
  };
  /** The preset the Harness itself reports for the chat's session. */
  const preset = async chatId => {
    const { rt } = bridge._debug();
    const r = await rt.gw.rpc('session/projections', { request: { sessionId: bridge.sessionOf(chatId), keys: ['permissions'] } });
    return r.values.permissions.currentValue;
  };
  return { worker, bridge, dir, ws, run, preset };
}

test('mode mapping: three presets, Manual by default, unknown values fall back to Manual', () => {
  assert.deepEqual(Object.keys(MODE_PRESETS), ['manual', 'plan', 'bypass']);
  // Manual leaves the sandbox open and asks BEFORE a change through the Strata gate (plugins/strata-gate.mjs).
  assert.deepEqual([MODE_PRESETS.manual.sandbox, MODE_PRESETS.manual.approval], ['danger-full-access', 'ask']);
  assert.deepEqual([MODE_PRESETS.plan.sandbox, MODE_PRESETS.plan.approval], ['read-only', 'never']);
  assert.deepEqual([MODE_PRESETS.bypass.sandbox, MODE_PRESETS.bypass.approval], ['danger-full-access', 'never']);
  for (const v of ['manual', 'plan', 'bypass']) assert.equal(modeOf(v), v);
  for (const v of [undefined, null, '', 'auto', 'accept-edits', 'danger-full-access', 'constructor']) assert.equal(modeOf(v), 'manual', String(v));
  // Only Bypass gives full access WITHOUT asking, and the profile names our three and nothing else.
  const entry = JSON.parse(buildProfile({ local: [] }).patch.split('\n').slice(1).join('\n')).find(e => e.id === 'permission');
  assert.deepEqual(Object.keys(entry.config.presets), ['strata-manual', 'strata-plan', 'strata-bypass']);
  assert.equal(entry.config.defaultPreset, 'strata-manual');
  assert.deepEqual(Object.entries(entry.config.presets).filter(([, p]) => p.sandbox === 'danger-full-access' && p.approval === 'never').map(([k]) => k), ['strata-bypass']);
  assert.deepEqual(Object.entries(entry.config.presets).filter(([, p]) => p.sandbox !== 'danger-full-access').map(([k]) => k), ['strata-plan']);
});

test('projection: under "never" no card ever offers Approve; "ask" marks the approved card', () => {
  const call = { type: 'event', event: { seq: 1, time: 1, type: 'tool/call', data: { callId: 'c1', name: 'write', arguments: { file_path: 'x.txt', content: 'x' } } } };
  const asked = { type: 'event', event: { seq: 2, time: 2, type: 'approval/asked', data: { id: 'a1', callId: 'c1', reason: 'escalate sandbox to workspace-write: because' } } };
  const rejected = { type: 'event', event: { seq: 3, time: 3, type: 'approval/decided', data: { id: 'a1', outcome: 'rejected' } } };
  const failed = { type: 'event', event: { seq: 4, time: 4, type: 'tool/result', data: { message: { toolCallId: 'c1', isError: true, content: [{ type: 'text', text: 'Error: the user rejected escalating' }] } } } };
  const never = new TurnProjector({ policy: 'never' });
  const frames = [call, asked, rejected, failed].flatMap(i => never.consume(i));
  assert.deepEqual(frames.map(f => f.tool.state), ['running', 'denied', 'denied']);
  assert.ok(frames.every(f => !f.tool.approved));
  assert.equal(frames.at(-1).tool.result, NOT_IN_MODE);

  const ask = new TurnProjector({ policy: 'ask' });
  const ok = { type: 'event', event: { seq: 3, time: 3, type: 'approval/decided', data: { id: 'a1', outcome: 'allowed-once' } } };
  const done = { type: 'event', event: { seq: 4, time: 4, type: 'tool/result', data: { message: { toolCallId: 'c1', isError: false, content: [{ type: 'text', text: 'Created file' }] } } } };
  const f2 = [call, asked, ok, done].flatMap(i => ask.consume(i));
  assert.deepEqual(f2.map(f => f.tool.state), ['running', 'proposed', 'running', 'done']);
  assert.equal(f2.at(-1).tool.approved, true);
  // A call that never asked finishes without the flag.
  const plain = new TurnProjector({ policy: 'ask' });
  const f3 = [call, done].flatMap(i => plain.consume(i));
  assert.equal(f3.at(-1).tool.state, 'done'); assert.equal(f3.at(-1).tool.approved, undefined);
});

test('MODE Manual: an in-workspace write is held for the person, and runs once after Approve', { skip }, async t => {
  const r = await rig(t);
  let fileWhileAsking = null;
  const frames = await r.run('m-manual-write', 'SCRIPT:retrywrite approved.txt hello', {
    mode: 'manual', answer: 'approve',
    onFrame: f => { if (f.type === 'agent-tool' && f.tool.state === 'proposed') fileWhileAsking = existsSync(join(r.ws, 'approved.txt')); },
  });
  assert.equal(fileWhileAsking, false, 'the file must not exist while the card is asking');
  const final = tools(frames).at(-1).tool;
  assert.equal(final.state, 'done'); assert.equal(final.approved, true); assert.equal(final.scope, 'write');
  assert.ok(states(frames).includes('proposed'));
  assert.equal(lines(join(r.ws, 'approved.txt')), 1);
  assert.equal(frames.find(f => f.type === 'start').mode, 'manual');
  assert.equal(await r.preset('m-manual-write'), MODE_PRESETS.manual.key);
  assert.equal(frames.at(-1).type, 'done');
});

test('MODE Manual: Deny leaves the file absent, and a shell command that writes is held too', { skip }, async t => {
  const r = await rig(t);
  const denied = await r.run('m-manual-deny', 'SCRIPT:retrywrite denied.txt nope', { mode: 'manual', answer: 'deny' });
  assert.equal(existsSync(join(r.ws, 'denied.txt')), false);
  assert.equal(states(denied).at(-1), 'denied');
  assert.equal(denied.at(-1).type, 'done');

  const shell = await r.run('m-manual-shell', 'SCRIPT:retryshell shell.txt via-shell', { mode: 'manual', answer: 'approve' });
  assert.ok(states(shell).includes('proposed'), 'the shell write asked first');
  const card = tools(shell).at(-1).tool;
  assert.equal(card.scope, 'shell'); assert.equal(card.state, 'done'); assert.equal(card.approved, true);
  assert.equal(lines(join(r.ws, 'shell.txt')), 1);
});

test('MODE Manual: reading needs no question', { skip }, async t => {
  const r = await rig(t);
  writeFileSync(join(r.ws, 'notes.txt'), 'plain notes\n');
  const frames = await r.run('m-manual-read', 'SCRIPT:read notes.txt', { mode: 'manual' });
  assert.deepEqual([...new Set(states(frames))], ['running', 'done']);
  assert.equal(tools(frames).at(-1).tool.approved, undefined);
  assert.match(tools(frames).at(-1).tool.result, /plain notes/);
});

test('MODE Plan: a write is refused with no question, the file never appears, reading still works', { skip }, async t => {
  const r = await rig(t);
  const frames = await r.run('m-plan', 'SCRIPT:retrywrite plan.txt nope', { mode: 'plan' });
  assert.equal(existsSync(join(r.ws, 'plan.txt')), false);
  assert.ok(!states(frames).includes('proposed'), 'Plan never shows an Approve card');
  assert.equal(states(frames).at(-1), 'denied');
  assert.equal(tools(frames).at(-1).tool.result, NOT_IN_MODE);
  assert.equal(frames.at(-1).type, 'done', 'the turn still settles; the agent answers');
  assert.equal(await r.preset('m-plan'), MODE_PRESETS.plan.key);
  assert.equal(r.bridge.status().pendingApprovals, 0);

  const shell = await r.run('m-plan-shell', 'SCRIPT:retryshell plan-shell.txt nope', { mode: 'plan' });
  assert.equal(existsSync(join(r.ws, 'plan-shell.txt')), false);
  assert.ok(!states(shell).includes('proposed'));

  writeFileSync(join(r.ws, 'readme.txt'), 'readable in plan mode\n');
  const read = await r.run('m-plan-read', 'SCRIPT:read readme.txt', { mode: 'plan' });
  assert.equal(tools(read).at(-1).tool.state, 'done');
  assert.match(tools(read).at(-1).tool.result, /readable in plan mode/);
});

test('MODE Bypass is full access: in-workspace and outside writes run without asking', { skip }, async t => {
  const r = await rig(t);
  const frames = await r.run('m-bypass', 'SCRIPT:retrywrite free.txt hello', { mode: 'bypass' });
  assert.ok(!states(frames).includes('proposed'), 'Bypass asks nothing');
  assert.equal(lines(join(r.ws, 'free.txt')), 1);
  const card = tools(frames).at(-1).tool;
  assert.equal(card.state, 'done'); assert.equal(card.approved, undefined, 'it ran; nobody approved it');
  assert.equal(await r.preset('m-bypass'), MODE_PRESETS.bypass.key);

  const shell = await r.run('m-bypass-shell', 'SCRIPT:retryshell free-shell.txt hello', { mode: 'bypass' });
  assert.ok(!states(shell).includes('proposed'));
  assert.equal(lines(join(r.ws, 'free-shell.txt')), 1);

  // Bypass is the Harness's full access with no prompts: a call that asks for danger-full-access runs, in the workspace or outside it.
  const wide = await r.run('m-bypass-wide', 'SCRIPT:tool wide.txt x', { mode: 'bypass' });
  assert.ok(!states(wide).includes('proposed'), 'full access asks nothing');
  assert.equal(lines(join(r.ws, 'wide.txt')), 1);
  assert.equal(states(wide).at(-1), 'done');
  const outside = join(r.dir, 'outside-workspace.txt');
  const out = await r.run('m-bypass-outside', `SCRIPT:tool ${outside} y`, { mode: 'bypass' });
  assert.ok(!states(out).includes('proposed'));
  assert.equal(lines(outside), 1, 'a path outside the workspace was written');
  assert.equal(states(out).at(-1), 'done');
});

test('MODE the choice follows the chat: it changes on the next turn, keeps the session, and an unknown value is Manual', { skip }, async t => {
  const r = await rig(t);
  await r.run('m-switch', 'SCRIPT:echo one', { mode: 'manual' });
  const session = r.bridge.sessionOf('m-switch');
  assert.equal(await r.preset('m-switch'), MODE_PRESETS.manual.key);
  await r.run('m-switch', 'SCRIPT:echo two', { mode: 'plan' });
  assert.equal(await r.preset('m-switch'), MODE_PRESETS.plan.key);
  const bypass = await r.run('m-switch', 'SCRIPT:retrywrite switched.txt hi', { mode: 'bypass' });
  assert.equal(lines(join(r.ws, 'switched.txt')), 1);
  assert.ok(!states(bypass).includes('proposed'));
  // Back to Manual: asks again, on the same session, and the earlier turns are still its history.
  const manual = await r.run('m-switch', 'SCRIPT:retrywrite switched-again.txt hi', { mode: 'manual', answer: 'approve' });
  assert.ok(states(manual).includes('proposed'));
  assert.equal(r.bridge.sessionOf('m-switch'), session);
  // A mode the app does not offer (older stored value) or none at all is Manual, never something looser.
  for (const [chat, mode] of [['m-odd-1', 'auto'], ['m-odd-2', undefined], ['m-odd-3', 'danger-full-access']]) {
    const f = await r.run(chat, 'SCRIPT:retrywrite odd.txt x', { mode, answer: 'deny' });
    assert.ok(states(f).includes('proposed'), `${mode} asks like Manual`);
  }
  assert.equal(existsSync(join(r.ws, 'odd.txt')), false);
});

test('MODE changing the mode while a turn runs does not touch that turn', { skip }, async t => {
  const r = await rig(t);
  const frames = [];
  const slow = r.bridge.runTurn({ chatId: 'm-mid', message: 'SCRIPT:slow 120', workspacePath: r.ws, mode: 'manual', onFrame: f => frames.push(f) });
  for (let i = 0; i < 200 && frames.filter(f => f.type === 'delta').length < 3; i++) await sleep(50);
  assert.ok(frames.filter(f => f.type === 'delta').length >= 3, 'the turn must be live first');
  // The person flips the menu to Bypass and sends again. The second send is refused (a chat runs one turn at a time) and,
  // because it is refused before anything is applied, the running turn's permissions do not move.
  await assert.rejects(r.bridge.runTurn({ chatId: 'm-mid', message: 'SCRIPT:echo x', workspacePath: r.ws, mode: 'bypass', onFrame: () => {} }), /still working/);
  assert.equal(await r.preset('m-mid'), MODE_PRESETS.manual.key);
  // Another chat in another mode is its own session and does not move this one either.
  await r.run('m-mid-other', 'SCRIPT:echo y', { mode: 'bypass' });
  assert.equal(await r.preset('m-mid'), MODE_PRESETS.manual.key);
  await slow;
  assert.equal(frames.filter(f => f.type === 'delta').length, 120, 'the running turn completed untouched');
  assert.equal(frames.at(-1).finishReason, 'stop');
  // And the new choice is what the next turn uses.
  await r.run('m-mid', 'SCRIPT:echo z', { mode: 'bypass' });
  assert.equal(await r.preset('m-mid'), MODE_PRESETS.bypass.key);
});

test('MODE the mode survives a restart of the agent process: the chat resumes in the mode it is asked for', { skip }, async t => {
  const r = await rig(t);
  await r.run('m-restart', 'SCRIPT:echo one', { mode: 'plan' });
  const session = r.bridge.sessionOf('m-restart');
  await r.bridge.close();
  // A new bridge over the same data directory: the persisted session is resumed, then put in the requested mode.
  const worker2 = r.worker;
  const bridge2 = createAgentBridge({ dataDir: r.dir, localInstances: () => [{ id: 'inst-1', label: 'Scripted', baseUrl: worker2.url, modelId: 'scripted-model', context: 32768 }], connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(() => bridge2.close());
  const frames = [];
  await bridge2.runTurn({ chatId: 'm-restart', message: 'SCRIPT:retrywrite resumed.txt hi', workspacePath: r.ws, mode: 'bypass', onFrame: f => frames.push(f) });
  assert.equal(bridge2.sessionOf('m-restart'), session, 'the same session');
  assert.equal(lines(join(r.ws, 'resumed.txt')), 1);
  const { rt } = bridge2._debug();
  const now = await rt.gw.rpc('session/projections', { request: { sessionId: session, keys: ['permissions'] } });
  assert.equal(now.values.permissions.currentValue, MODE_PRESETS.bypass.key);
});
