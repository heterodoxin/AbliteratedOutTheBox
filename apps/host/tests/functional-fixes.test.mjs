/**
 * Fixes from the real-engine functional test of the integrated build (notes-split-uitest.md, "Functional (real engine)"):
 * tok/s and context in the agent path, plain-language tool rows, a launch that reports success, and the size and wording of
 * what the agent sends to the model. The model is scripted; the Harness, the bridge and the projector are real.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TurnProjector, outcomeText, toolVerb } from '../server/agent/events.mjs';
import { AGENT_PROMPT } from '../server/agent/preset.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };
const ev = (type, data, seq = 1, time = 1000) => ({ type: 'event', event: { type, seq, time, data } });
const chunk = (c, attemptId = 'a:1') => ({ type: 'assistant-stream', frame: { type: 'chunk', attemptId, chunk: c } });

/** Run `fn` with a clock the test moves by hand. */
function withClock(fn) {
  const real = Date.now; let now = 1_000_000;
  Date.now = () => now;
  try { return fn(ms => { now += ms; }); } finally { Date.now = real; }
}

test('item 1: the agent path reports live tok/s while it streams and a final tok/s, not null', () => withClock(tick => {
  const p = new TurnProjector({ startedAt: Date.now() });
  const frames = [];
  // 60 chunks of one token each, 25 ms apart: 40 tokens per second.
  for (let i = 0; i < 60; i++) { tick(25); frames.push(...p.consume(chunk({ type: 'text-delta', index: 0, text: 'ab' }))); }
  const stats = frames.filter(f => f.type === 'stats');
  assert.ok(stats.length >= 2, 'live speed frames while the answer streams');
  const last = stats.at(-1);
  assert.equal(last.source, 'stream');
  assert.ok(last.tps > 30 && last.tps < 50, `live tps ${last.tps}`);
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 900, outputTokens: 60 } }));
  p.consume(ev('turn/end', { turn: 1, reason: { kind: 'completed' } }));
  const done = p.finish({ model: 'M', contextMax: 65536 }).find(f => f.type === 'done');
  assert.ok(done.tps > 30 && done.tps < 50, `final tps ${done.tps}`);
  assert.equal(done.tokensOut, 60);
}));

test('item 1: reasoning tokens count towards the speed too (a thinking model streams reasoning first)', () => withClock(tick => {
  const p = new TurnProjector({ startedAt: Date.now() });
  for (let i = 0; i < 40; i++) { tick(20); p.consume(chunk({ type: 'reasoning-delta', index: 0, text: 'x' })); }
  p.consume(ev('turn/end', { turn: 1, reason: { kind: 'completed' } }));
  const done = p.finish({}).find(f => f.type === 'done');
  assert.ok(done.tps > 40 && done.tps < 60, `tps ${done.tps}`);
}));

test('item 2: the context is the LAST request (prompt + reply), not the sum of every step', () => {
  const p = new TurnProjector({ startedAt: 0 });
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 5000, outputTokens: 100 } }, 'a:1'));
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 5200, outputTokens: 300 } }, 'a:2'));
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 5600, outputTokens: 50 } }, 'a:3'));
  p.consume(ev('turn/end', { turn: 1, reason: { kind: 'completed' } }));
  const done = p.finish({ model: 'M', contextMax: 65536 }).find(f => f.type === 'done');
  assert.equal(done.contextTokens, 5650, 'what the context holds after the turn');
  assert.equal(done.tokensIn, 15800, 'the billed sum is still reported separately');
});

test('item 3: a prompt the server already held counts towards the context (the Harness reports it apart from the input)', () => {
  const p = new TurnProjector({ startedAt: 0 });
  // The shape of a real Swift 64K run: the last request had a prompt of 8,042 tokens of which 7,317 came from the conversation cache.
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 5800, outputTokens: 40, cacheReadTokens: 0 } }, 'a:1'));
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 725, outputTokens: 110, cacheReadTokens: 7317 } }, 'a:2'));
  p.consume(ev('turn/end', { turn: 1, reason: { kind: 'completed' } }));
  const done = p.finish({ model: 'M', contextMax: 65536 }).find(f => f.type === 'done');
  assert.equal(done.contextTokens, 8152, 'prompt (input + cache) + reply of the last request, which is what the engine held');
  assert.equal(done.tokensIn, 5800 + 725 + 7317);
});

test('item 7: tool rows say what happened in plain words, with what it was about', () => {
  const p = new TurnProjector({ startedAt: 0 });
  const call = p.consume(ev('tool/call', { callId: 'w1', name: 'write', arguments: JSON.stringify({ file_path: 'C:/work/notes.md', content: 'x' }) }))[0].tool;
  assert.equal(call.verb, 'Writing a file'); assert.equal(call.target, 'C:/work/notes.md');
  const asked = p.consume(ev('approval/asked', { id: 'a1', callId: 'w1', reason: 'Create or replace the file C:/work/notes.md' }))[0].tool;
  assert.equal(asked.state, 'proposed'); assert.equal(asked.target, 'C:/work/notes.md', 'the approval card shows the file');
  const ok = p.consume(ev('tool/result', { message: { toolCallId: 'w1', content: [{ type: 'text', text: 'written' }] } }))[0].tool;
  assert.equal(ok.verb, 'Wrote a file');
  p.consume(ev('tool/call', { callId: 'w2', name: 'write', arguments: JSON.stringify({ file_path: 'D:/x.txt' }) }));
  const bad = p.consume(ev('tool/result', { message: { toolCallId: 'w2', isError: true, content: [{ type: 'text', text: 'denied' }] } }))[0].tool;
  assert.equal(bad.state, 'error'); assert.equal(bad.verb, "Couldn't write the file");
  const sh = p.consume(ev('tool/call', { callId: 's1', name: 'pwsh', arguments: JSON.stringify({ command: 'Get-ChildItem C:/work' }) }))[0].tool;
  assert.equal(sh.verb, 'Running a command'); assert.equal(sh.target, 'Get-ChildItem C:/work');
  const g = p.consume(ev('tool/call', { callId: 'g1', name: 'glob', arguments: JSON.stringify({ pattern: '**/*.md' }) }))[0].tool;
  assert.equal(g.verb, 'Searching for files'); assert.equal(g.target, '**/*.md');
  assert.equal(toolVerb('glob', 'done'), 'Searched for files');
  assert.equal(toolVerb('some_connected_tool', 'running'), 'Using a tool');
  for (const name of ['write', 'edit', 'read', 'glob', 'grep', 'pwsh', 'bash', 'web_fetch', 'todo_write']) {
    for (const state of ['running', 'done', 'error']) assert.ok(!/^(write|edit|read|glob|grep|pwsh|bash|web_fetch|todo_write)\b/.test(toolVerb(name, state)), `${name} ${state}: no raw tool name`);
  }
});

test('item 8 / T9: starting a program that prints nothing is reported as started, never as opened', () => {
  for (const command of ['Start-Process explorer.exe C:\\work', 'explorer.exe C:\\work', 'start notepad', 'Invoke-Item C:\\work\\a.pdf', 'ii .', 'xdg-open .', 'open .']) {
    const said = outcomeText('pwsh', '(no output)', { command });
    assert.match(said, /^The command finished without an error, so the program was started\./, command);
    assert.match(said, /cannot confirm that a window or page actually opened/, command);
    assert.ok(!/was opened|is open/i.test(said), command);
  }
  assert.match(outcomeText('pwsh', '(no output)', { command: 'New-Item x.txt | Out-Null' }), /finished without any error/, 'other silent commands keep their wording');
  assert.equal(outcomeText('pwsh', 'hello', { command: 'echo hello' }), 'hello', 'a command that printed something shows it');
  const p = new TurnProjector({ startedAt: 0 });
  p.consume(ev('tool/call', { callId: 'e1', name: 'pwsh', arguments: JSON.stringify({ command: 'Start-Process explorer.exe C:\\work' }) }));
  const res = p.consume(ev('tool/result', { message: { toolCallId: 'e1', content: [{ type: 'text', text: '(no output)' }] } }))[0].tool;
  assert.match(res.result, /cannot confirm that a window or page actually opened/); assert.equal(res.verb, 'Ran a command');
});

test('item 7/10: our prompt has no internal terms and is short', () => {
  assert.ok(AGENT_PROMPT.length < 1600, `${AGENT_PROMPT.length} chars`);
  assert.doesNotMatch(AGENT_PROMPT, /DSH|DeepSeek|sandbox|isolated session|file policy|Harness/i);
  // O6 / #3576: the prompt points at the present tool for local files and never tells the model to claim a window opened
  assert.match(AGENT_PROMPT, /use the present tool/);
  assert.match(AGENT_PROMPT, /Never say a window opened, a page rendered/);
  assert.doesNotMatch(AGENT_PROMPT, /say plainly that it is open|do not say you could not confirm/i);
});

test('items 3, 7, 10: what the agent sends the model is small, plain, and carries Thinking Off', { skip }, async t => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-fx-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'i', baseUrl: worker.url, modelId: 'scripted-model', context: 65536 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  worker.requests.length = 0;
  await bridge.runTurn({ chatId: 'fx-1', message: 'SCRIPT:echo hi', workspacePath: ws, mode: 'plan', thinking: 'Off', onFrame: () => {} });
  const req = worker.requests.find(r => r.toolNames.length);
  assert.ok(req, 'a request with tools');
  assert.equal(req.params.chat_template_kwargs.enable_thinking, false, 'Thinking Off reaches the model request');
  for (const kept of ['pwsh', 'read', 'write', 'edit', 'glob', 'grep', 'workflow', 'create_goal', 'update_goal', 'get_goal', 'skill', 'subagent_fork']) assert.ok(req.toolNames.includes(kept), `${kept} is offered (owner decision: the goal, skill, workflow and fork tools stay)`);
});

// The Reasoning menu in the agent: every level reaches the model server as chat_template_kwargs, and an unset menu is Medium (owner decision).
for (const [level, thinking, effort] of [['Off', 'Off', undefined], ['Low', 'Low', 'low'], ['Medium', 'Medium', 'medium'], ['High', 'High', 'high'], ['unset (Medium)', undefined, 'medium']]) {
  test(`agent Reasoning ${level}: the request carries chat_template_kwargs ${effort ? `enable_thinking true, reasoning_effort ${effort}` : 'enable_thinking false and no effort'}`, { skip }, async t => {
    const worker = await startScriptedWorker({});
    const dir = mkdtempSync(join(tmpdir(), 'strata-eff-'));
    const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
    const local = [{ id: 'i', baseUrl: worker.url, modelId: 'scripted-model', context: 65536 }];
    const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
    t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
    worker.requests.length = 0;
    await bridge.runTurn({ chatId: `eff-${level}`, message: 'SCRIPT:echo hi', workspacePath: ws, mode: 'plan', ...(thinking ? { thinking } : {}), onFrame: () => {} });
    const req = worker.requests.find(r => r.toolNames.length);
    assert.ok(req, 'a request with tools');
    const kwargs = req.params.chat_template_kwargs;
    assert.equal(kwargs.enable_thinking, effort !== undefined);
    assert.equal(kwargs.reasoning_effort, effort);
  });
}
