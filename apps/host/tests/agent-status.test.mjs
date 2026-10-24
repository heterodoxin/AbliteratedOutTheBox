/** Live status of a running turn: the sentences, the throttling, and what the projector says as the model moves from thinking to writing a file. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { StatusReporter, toolStatusText, TokenCount } from '../server/agent/status.mjs';
import { TurnProjector } from '../server/agent/events.mjs';

test('a tool is described by what it does to what, in the person\'s words', () => {
  assert.equal(toolStatusText('write', { file_path: 'C:\\work\\pagoda.html' }), 'Writing pagoda.html…');
  assert.equal(toolStatusText('write', { file_path: '/a/b/pagoda.html' }, { bytes: 12000 }), 'Writing pagoda.html… (12 KB)');
  assert.equal(toolStatusText('write', {}), 'Writing a file…');
  assert.equal(toolStatusText('edit', { path: 'notes.md' }), 'Changing notes.md…');
  assert.equal(toolStatusText('pwsh', { command: 'Get-ChildItem' }), 'Running a command…');
  assert.equal(toolStatusText('web_search', {}), 'Searching the web…');
  assert.equal(toolStatusText('update_goal', {}), 'Updating the goal…');
  assert.equal(toolStatusText('web_fetch', { url: 'https://example.com/a' }), 'Opening example.com…');
  assert.equal(toolStatusText('mcp__browser__browser_navigate', { url: 'https://example.com' }), 'Opening https://example.com…');
  assert.equal(toolStatusText('mcp__strata-commands__x', {}), 'Running your tool…');
  assert.equal(toolStatusText('mcp__other__thing', {}), 'Using a tool…');
});

function rig() {
  let clock = 1000;
  const timers = [];
  const frames = [];
  const reporter = new StatusReporter({
    emit: f => frames.push(f), now: () => clock,
    timers: { set: (fn, ms) => { const t = { fn, at: clock + ms, unref() {} }; timers.push(t); return t; }, clear: t => { const i = timers.indexOf(t); if (i >= 0) timers.splice(i, 1); } },
  });
  const advance = ms => { clock += ms; for (const t of [...timers].sort((a, b) => a.at - b.at)) if (t.at <= clock) { timers.splice(timers.indexOf(t), 1); t.fn(); } };
  return { reporter, frames, advance, now: () => clock };
}

test('status: a change is sent at once, the same words are never sent twice, and a quick burst keeps only the newest', () => {
  const { reporter, frames, advance } = rig();
  reporter.set({ phase: 'starting', code: 'starting', text: 'Starting…' });
  reporter.set({ phase: 'starting', code: 'starting', text: 'Starting…' });
  assert.equal(frames.length, 1); assert.deepEqual([frames[0].type, frames[0].seq, frames[0].phase, frames[0].text], ['status', 1, 'starting', 'Starting…']);
  advance(50);
  reporter.set({ phase: 'waiting', code: 'reading-prompt', text: 'Reading your message…' });   // 50 ms after the last: held
  reporter.set({ phase: 'preparing', code: 'preparing-tool', text: 'Writing a.html…', args: { tool: 'write', target: 'a.html', bytes: 10 } });
  assert.equal(frames.length, 1, 'held back, not dropped');
  advance(300);
  assert.equal(frames.length, 2); assert.equal(frames[1].text, 'Writing a.html…', 'the newest change wins; the one in between was never true for long enough to show');
});

test('status: a counter (reasoning tokens, bytes) is refreshed every 2 s, not on every chunk', () => {
  const { reporter, frames, advance } = rig();
  reporter.set({ phase: 'thinking', code: 'thinking', text: 'Thinking… 10 tokens', tokens: 10, args: { tokens: 10 } });
  for (let i = 1; i <= 15; i++) { advance(100); reporter.set({ phase: 'thinking', code: 'thinking', text: `Thinking… ${10 + i * 10} tokens`, tokens: 10 + i * 10, args: { tokens: 10 + i * 10 } }); }
  assert.ok(frames.length <= 2, `1.5 s of chunks made ${frames.length} frames`);
  advance(600);
  assert.equal(frames.at(-1).tokens, 160, 'the refresh carries the latest count');
  assert.ok(frames.length <= 3);
  assert.ok(frames.every(f => f.phase === 'thinking'));
});

test('status: clear says idle once and then nothing more is sent', () => {
  const { reporter, frames, advance } = rig();
  reporter.set({ phase: 'tool', code: 'tool', text: 'Running a command…' });
  advance(400);
  reporter.clear();
  assert.deepEqual(frames.map(f => f.phase), ['tool', 'idle']);
  reporter.set({ phase: 'tool', code: 'tool', text: 'x' }); advance(5000);
  assert.equal(frames.length, 2);
});

test('projector: thinking counts tokens; composing a file call is "Writing pagoda.html…" while no text arrives; running, result and next step follow', () => {
  const p = new TurnProjector();
  const seen = [];
  p.onStatus = u => seen.push(u);
  const chunk = (c, attemptId = 'a1') => p.consume({ type: 'assistant-stream', frame: { type: 'chunk', attemptId, chunk: c } });
  for (const t of ['Let ', 'me ', 'think ', 'about ', 'this']) chunk({ type: 'reasoning-delta', text: t });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).tokens, seen.at(-1).text], ['thinking', 5, 'Thinking… 5 tokens']);
  chunk({ type: 'text-delta', text: 'I will build the pagoda HTML file now.' });
  assert.equal(seen.at(-1).phase, 'answering');
  chunk({ type: 'reasoning-delta', text: 'again' }, 'a2');
  assert.equal(seen.at(-1).tokens, 1, 'a new model request counts its own thinking from zero');
  chunk({ type: 'tool-call-delta', index: 0, id: 'call-1', name: 'write', argumentsDelta: '{"file_path":"C:\\\\work\\\\pagoda.html","content":"<html>' });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).code, seen.at(-1).text], ['preparing', 'preparing-tool', 'Writing pagoda.html…'], 'the stale sentence is replaced the moment the call starts');
  assert.equal(seen.at(-1).args.target, 'C:\\work\\pagoda.html');
  chunk({ type: 'tool-call-delta', index: 0, id: 'call-1', argumentsDelta: 'x'.repeat(9000) });
  assert.match(seen.at(-1).text, /^Writing pagoda\.html… \(\d+(\.\d)? KB\)$/);
  p.consume({ type: 'event', event: { type: 'tool/call', time: 1, data: { callId: 'call-1', name: 'write', arguments: JSON.stringify({ file_path: 'C:/work/pagoda.html', content: 'x' }) } } });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).text], ['tool', 'Writing pagoda.html…']);
  p.consume({ type: 'event', event: { type: 'tool/result', time: 2, data: { message: { toolCallId: 'call-1', content: [{ type: 'text', text: 'ok' }] } } } });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).code], ['waiting', 'model-working']);
});

test('projector: a tool that asks shows "waiting for your answer", and a helper call is not announced as a tool', () => {
  const p = new TurnProjector();
  const seen = [];
  p.onStatus = u => seen.push(u);
  p.consume({ type: 'event', event: { type: 'tool/call', time: 1, data: { callId: 'c1', name: 'pwsh', arguments: JSON.stringify({ command: 'rm x' }) } } });
  p.consume({ type: 'event', event: { type: 'approval/asked', time: 2, data: { id: 'ap1', callId: 'c1', reason: 'Run a command' } } });
  assert.deepEqual([seen.at(-1).phase, seen.at(-1).text], ['approval', 'Waiting for your answer…']);
  p.consume({ type: 'event', event: { type: 'approval/decided', time: 3, data: { id: 'ap1', outcome: 'allowed-once' } } });
  assert.equal(seen.at(-1).phase, 'tool', 'back to the running tool after the answer');
  const before = seen.length;
  p.consume({ type: 'assistant-stream', frame: { type: 'chunk', attemptId: 'a', chunk: { type: 'tool-call-delta', index: 0, id: 'h1', name: 'subagent', argumentsDelta: '{"prompt":"x"}' } } });
  assert.equal(seen.length, before, 'helpers have their own status');
});

test('token count: one per chunk, or characters / 4 when the server batches', () => {
  const a = new TokenCount(); for (const t of ['a', 'b', 'c']) a.add(t); assert.equal(a.tokens, 3);
  const b = new TokenCount(); b.add('x'.repeat(400)); assert.equal(b.tokens, 100);
});

test('plain chat says what it is doing too: reading the message, thinking with a token count, writing, then the line is cleared before done', async () => {
  const { withPlainStatus } = await import('../server/agent/status.mjs');
  let clock = 1000; const written = [];
  const frame = withPlainStatus(f => written.push(f), { now: () => clock, timers: { set: () => ({ unref() {} }), clear() {} } });
  const feed = f => { clock += 2100; frame(f); };
  feed({ type: 'start', demo: false });
  for (const t of ['a', 'b', 'c']) feed({ type: 'reasoning', text: t });
  feed({ type: 'delta', text: 'Hello' });
  feed({ type: 'done', tokensOut: 4 });
  const phases = written.filter(f => f.type === 'status').map(f => f.phase);
  assert.deepEqual(phases.filter((p, i, a) => a[i - 1] !== p), ['starting', 'waiting', 'thinking', 'answering', 'idle']);
  const think = written.filter(f => f.type === 'status' && f.phase === 'thinking').at(-1);
  assert.equal(think.tokens, 3); assert.equal(think.text, 'Thinking… 3 tokens');
  assert.equal(written.at(-1).type, 'done');
  assert.equal(written.at(-2).phase, 'idle');
  frame.dispose();
});

// ------------------------------------------------------------------------------------------------- real Harness

import { existsSync, mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const skip = !process.env.STRATA_HARNESS_DIR || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('real Harness: a turn that writes a file says what it is doing, in order, and ends with the status cleared before done', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-status-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const frames = [];
  await bridge.runTurn({ chatId: 's1', message: `SCRIPT:call write ${JSON.stringify({ file_path: join(ws, 'pagoda.html'), content: '<html></html>' })}`, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
  const statuses = frames.filter(f => f.type === 'status');
  assert.ok(statuses.length >= 2, 'statuses were sent');
  assert.equal(statuses[0].phase, 'starting');
  assert.deepEqual(statuses.map(s => s.seq), statuses.map((_, i) => i + 1), 'sequence numbers count up');
  assert.equal(statuses.at(-1).phase, 'idle');
  assert.ok(frames.indexOf(statuses.at(-1)) < frames.findIndex(f => f.type === 'done'), 'cleared before done');
  assert.ok(existsSync(join(ws, 'pagoda.html')));
});

test('a sentence with a name or a number also carries its catalog template and values, and every status sentence is in the catalog in every language', async () => {
  const { readFileSync } = await import('node:fs');
  const catalog = Object.fromEntries(['en', 'de', 'es', 'ja', 'ko', 'pt', 'ru', 'zh'].map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/${l}.json`, import.meta.url), 'utf8'))]));
  const idOf = text => Object.entries(catalog.en).find(([, v]) => v === text)?.[0];
  const p = new TurnProjector(); const seen = []; p.onStatus = u => seen.push(u);
  p.consume({ type: 'assistant-stream', frame: { type: 'chunk', attemptId: 'a', chunk: { type: 'tool-call-delta', index: 0, id: 'c', name: 'write', argumentsDelta: '{"file_path":"/w/pagoda.html","content":"' + 'x'.repeat(4000) } } });
  assert.deepEqual([seen.at(-1).key, seen.at(-1).values], ['Writing {0}… ({1} KB)', ['pagoda.html', '3.9']]);
  assert.equal(seen.at(-1).text, 'Writing pagoda.html… (3.9 KB)');
  const keys = ['Starting the assistant…', 'Starting…', 'Reading your message…', 'Working out the next step…', 'Thinking… {0} tokens', 'Writing the answer…', 'Waiting for your answer…', 'Helpers are working ({0})…', 'Working on the goal…',
    'Writing {0}…', 'Writing a file…', 'Writing {0}… ({1} KB)', 'Writing a file… ({0} KB)', 'Changing {0}…', 'Changing a file…', 'Reading {0}…', 'Reading a file…', 'Running a command…', 'Searching your files…', 'Searching the web…',
    'Opening {0}…', 'Opening a web page…', 'Using the browser…', 'Setting the goal…', 'Updating the goal…', 'Checking the goal…', 'Using a tool…', 'Reading your files…', 'Trying an improved version…', 'Checking the result…', 'Comparing the two versions…'];
  for (const key of keys) { const id = idOf(key); assert.ok(id, `${key} is in the catalog`); for (const l of ['de', 'es', 'ja', 'ko', 'pt', 'ru', 'zh']) { assert.ok(catalog[l][id]?.trim(), `${l}: ${key}`); assert.deepEqual((catalog[l][id].match(/\{\d\}/g) ?? []).sort(), (key.match(/\{\d\}/g) ?? []).sort(), `${l} keeps the slots of ${key}`); } }
});

test('a goal step is always sent, even right after another status (it used to be swallowed by the 300 ms gap, which made a test flaky)', () => {
  const { reporter, frames, advance } = rig();
  reporter.set({ phase: 'waiting', code: 'reading-prompt', text: 'Reading your message…' });
  advance(20);
  reporter.set({ phase: 'goal', code: 'goal-step', text: 'Working on the goal…', args: { round: 1, max: 3 } });
  advance(10);
  reporter.set({ phase: 'thinking', code: 'thinking', text: 'Thinking… 1 tokens', tokens: 1 });
  advance(400);
  assert.ok(frames.some(f => f.phase === 'goal' && f.args.round === 1), frames.map(f => f.phase).join(','));
});

import { plainTurnFailure } from '../server/agent/agent-bridge.mjs';
test('a start failure from the agent reads in plain words and keeps the raw text for Details', () => {
  const raw = new Error('session/create: tool-bash (@deepseek-ai/dsh-tool-bash): waiting for shell\ntool-workflow: waiting for workflowEngine');
  raw.code = 'agent-preset/invalid';
  const f = plainTurnFailure(raw);
  assert.doesNotMatch(f.message, /session\/create|waiting for|@deepseek-ai|workflowEngine/);
  assert.match(f.message, /reinstall|reopen/i);
  assert.match(f.details, /waiting for workflowEngine/);
  assert.doesNotMatch(plainTurnFailure(new Error('turn/start: boom')).message, /turn\/start/);
});

import { createSessionWhenReady, AgentUnavailable } from '../server/agent/agent-bridge.mjs';
test('first chat on a fresh start waits for the Harness plugins instead of failing', async () => {
  let calls = 0; let waits = 0;
  const create = async () => {
    calls++;
    if (calls < 3) throw new Error('session/create: tool-bash (@deepseek-ai/dsh-tool-bash): waiting for shell\ntool-workflow: waiting for workflowEngine');
    return { sessionId: 's1' };
  };
  const r = await createSessionWhenReady(create, { readyMs: 5000, retryMs: 5, onWait: () => { waits++; } });
  assert.deepEqual(r, { sessionId: 's1' });
  assert.equal(calls, 3);
  assert.equal(waits, 2, 'the page is told twice that the assistant is starting');
});
test('a start that never finishes gives the plain message with the raw text for Details', async () => {
  const raw = 'session/create: tool-bash: waiting for shell';
  let clock = 0;
  await assert.rejects(
    createSessionWhenReady(async () => { throw new Error(raw); }, { readyMs: 100, retryMs: 1, now: () => (clock += 60) }),
    e => {
      assert.ok(e instanceof AgentUnavailable);
      const f = plainTurnFailure(e);
      assert.equal(f.message, 'The assistant could not start. Try again in a moment.');
      assert.equal(f.details, raw);
      assert.equal(f.code, 'AGENT_NOT_READY');
      return true;
    });
});
test('other session errors are not retried, and Stop while waiting ends the wait', async () => {
  let calls = 0;
  await assert.rejects(createSessionWhenReady(async () => { calls++; throw new Error('session/create: disk full'); }, { retryMs: 1 }), /disk full/);
  assert.equal(calls, 1);
  const r = await createSessionWhenReady(async () => { throw new Error('x: waiting for shell'); }, { retryMs: 1, cancelled: () => true });
  assert.equal(r, null);
});
