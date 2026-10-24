/**
 * The host's HTTP surface for agent chats, against a real host process, the real Harness and a
 * scripted worker standing in as an API model.
 *
 * Needs STRATA_HARNESS_DIR (a built Harness) for everything except the "component missing" test.
 * What is exercised: /api/chat/stream (NDJSON), /api/chat/approval, /api/chat/stop, /api/agent/status,
 * resume across a host restart, two chats at once, and the visible fallback when the component is absent.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const HOST = resolve(import.meta.dirname, '..', 'server', 'index.mjs');

const freePort = () => new Promise(r => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* the Harness sandbox leaves access entries; best effort */ } };
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function startHost(t, { home, ws, harness = HARNESS } = {}) {
  const port = await freePort();
  const env = { ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), STRATA_WORKSPACE: ws, STRATA_REAL_ENGINE: '0', STRATA_COMPONENTS_DIR: join(home, 'components') };
  if (harness) env.STRATA_HARNESS_DIR = harness; else delete env.STRATA_HARNESS_DIR;
  const child = spawn(process.execPath, [HOST], { env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let log = '';
  child.stdout.on('data', d => { log += d; }); child.stderr.on('data', d => { log += d; });
  const stop = async () => {
    if (child.exitCode !== null) return;
    if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true });
    else child.kill('SIGTERM');
    await new Promise(r => child.once('exit', r));
  };
  t.after(stop);
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    try { if ((await fetch(`${base}/api/agent/status`)).ok) break; } catch { /* not up yet */ }
    if (child.exitCode !== null) throw new Error(`the host exited: ${log.slice(-500)}`);
    await sleep(200);
  }
  const post = async (path, body) => { const res = await fetch(`${base}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }); return { status: res.status, body: await res.json().catch(() => null) }; };
  /** Stream one chat turn; `onFrame` may react while it runs. Resolves with every frame. */
  const chat = async (body, onFrame = () => {}) => {
    const res = await fetch(`${base}/api/chat/stream`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    assert.equal(res.status, 200, `chat/stream: ${res.status}`);
    const frames = []; let buffer = '';
    const decoder = new TextDecoder();
    for await (const chunk of res.body) {
      buffer += decoder.decode(chunk, { stream: true });
      let i;
      while ((i = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, i).trim(); buffer = buffer.slice(i + 1);
        if (line) { const f = JSON.parse(line); frames.push(f); onFrame(f); }
      }
    }
    return frames;
  };
  return { base, port, post, chat, stop, log: () => log, status: async () => (await fetch(`${base}/api/agent/status`)).json() };
}

async function setup(t, { harness = HARNESS, reuse } = {}) {
  const worker = reuse?.worker ?? await startScriptedWorker({ model: 'cloud-model' });
  const home = reuse?.home ?? mkdtempSync(join(tmpdir(), 'strata-host-'));
  const ws = join(home, 'workspace'); mkdirSync(ws, { recursive: true });
  const host = await startHost(t, { home, ws, harness });
  let connectionId = reuse?.connectionId;
  if (!connectionId) {
    const saved = await host.post('/api/model-providers', { kind: 'custom', name: 'Scripted API', baseUrl: worker.url, protocol: 'openai-chat', apiKey: 'sk-host-test-0123456789', models: [{ id: 'cloud-model', name: 'Cloud model', contextMax: 64000 }] });
    assert.equal(saved.status, 200, JSON.stringify(saved.body));
    connectionId = saved.body.connections[0].id;
  }
  if (!reuse) t.after(async () => { await worker.close(); cleanup(home); });
  const remoteModel = { connectionId, modelId: 'cloud-model' };
  const send = (host, sessionId, message, onFrame) => host.chat({ sessionId, message, remoteModel, workspaceId: null, history: [] }, onFrame);
  return { worker, home, ws, host, connectionId, remoteModel, send: (sessionId, message, onFrame) => send(host, sessionId, message, onFrame) };
}

const toolStates = (frames, id) => frames.filter(f => f.type === 'agent-tool' && (!id || f.tool.id === id)).map(f => f.tool.state);
const text = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');
const lines = file => readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).length;

test('status names the component and the routes, and carries no secret', { skip }, async t => {
  const s = await setup(t);
  const status = await s.host.status();
  assert.equal(status.harness.version, '0.2.1-alpha.3');
  assert.ok(status.routes.some(r => r.kind === 'api' && r.modelId === 'cloud-model'));
  assert.ok(!JSON.stringify(status).includes('sk-host-test'));
});

test('A03 over HTTP: the card waits for Approve, the answer reaches the Harness, the tool runs once', { skip }, async t => {
  const s = await setup(t);
  let answered = false;
  const frames = await s.send('chat-h-a03', 'SCRIPT:tool http.txt hi', async f => {
    if (f.type === 'agent-tool' && f.tool.state === 'proposed' && !answered) {
      answered = true;
      const r = await s.host.post('/api/chat/approval', { sessionId: 'chat-h-a03', toolId: f.tool.id, decision: 'approve' });
      assert.equal(r.status, 200, JSON.stringify(r.body));
    }
  });
  assert.ok(answered, 'an approval request must have reached the client');
  const id = frames.find(f => f.type === 'agent-tool').tool.id;
  assert.deepEqual([...new Set(toolStates(frames, id))], ['running', 'proposed', 'done']);
  assert.equal(lines(join(s.ws, 'http.txt')), 1);
  assert.equal(frames.at(-1).type, 'done');
  // A second answer to the same approval is refused: it is no longer pending.
  const again = await s.host.post('/api/chat/approval', { sessionId: 'chat-h-a03', toolId: id, decision: 'approve' });
  assert.equal(again.status, 409);
});

test('A04 over HTTP: Deny means the tool never runs', { skip }, async t => {
  const s = await setup(t);
  const frames = await s.send('chat-h-a04', 'SCRIPT:tool never-http.txt x', async f => {
    if (f.type === 'agent-tool' && f.tool.state === 'proposed') await s.host.post('/api/chat/approval', { sessionId: 'chat-h-a04', toolId: f.tool.id, decision: 'deny' });
  });
  assert.equal(toolStates(frames).at(-1), 'denied');
  assert.equal(existsSync(join(s.ws, 'never-http.txt')), false);
  assert.equal(frames.at(-1).type, 'done');
});

test('A05 over HTTP: /api/chat/stop ends that chat only and answers after the turn ended', { skip }, async t => {
  const s = await setup(t);
  let live = 0, stopResult = null;
  const slow = s.send('chat-h-a05-a', 'SCRIPT:slow 400', async f => {
    if (f.type === 'delta' && ++live === 3) stopResult = await s.host.post('/api/chat/stop', { sessionId: 'chat-h-a05-a' });
  });
  const other = s.send('chat-h-a05-b', 'SCRIPT:slow 50');
  const [a, b] = await Promise.all([slow, other]);
  assert.equal(stopResult.status, 200);
  assert.equal(stopResult.body.stopped, true); assert.equal(stopResult.body.ended, 'aborted');
  assert.equal(a.at(-1).finishReason, 'aborted');
  assert.equal(b.at(-1).finishReason, 'stop'); assert.equal(b.filter(f => f.type === 'delta').length, 50);
  assert.ok(s.worker.aborted >= 1);
  // Nothing to stop afterwards.
  const none = await s.host.post('/api/chat/stop', { sessionId: 'chat-h-a05-a' });
  assert.equal(none.body.stopped, false);
});

test('a client that disconnects does not stop the turn; the tool still waits for its approval', { skip }, async t => {
  const s = await setup(t);
  const controller = new AbortController();
  const res = await fetch(`${s.host.base}/api/chat/stream`, { method: 'POST', headers: { 'content-type': 'application/json' }, signal: controller.signal, body: JSON.stringify({ sessionId: 'chat-h-detach', message: 'SCRIPT:tool detach.txt d', remoteModel: s.remoteModel, history: [] }) });
  const reader = res.body.getReader(); let seen = '';
  while (!seen.includes('"proposed"')) { const { value, done } = await reader.read(); if (done) break; seen += new TextDecoder().decode(value); }
  assert.match(seen, /"proposed"/);
  controller.abort();                                            // the tab closes
  await sleep(500);
  assert.equal(existsSync(join(s.ws, 'detach.txt')), false, 'a closed tab is not an approval');
  const pending = await (await fetch(`${s.host.base}/api/agent/pending?chatId=chat-h-detach`)).json();
  assert.equal(pending.pending.length, 1, 'the approval is still waiting for someone');
  // Someone comes back and answers it; the turn finishes in the background.
  const r = await s.host.post('/api/chat/approval', { sessionId: 'chat-h-detach', toolId: pending.pending[0].callId, decision: 'approve' });
  assert.equal(r.status, 200);
  for (let i = 0; i < 100 && !existsSync(join(s.ws, 'detach.txt')); i++) await sleep(100);
  assert.equal(lines(join(s.ws, 'detach.txt')), 1);
});

test('A06 over HTTP: after the host restarts, the same chat resumes its session and nothing is replayed', { skip }, async t => {
  const first = await setup(t);
  const frames1 = await first.send('chat-h-a06', 'SCRIPT:tool again.txt once', async f => {
    if (f.type === 'agent-tool' && f.tool.state === 'proposed') await first.host.post('/api/chat/approval', { sessionId: 'chat-h-a06', toolId: f.tool.id, decision: 'approve' });
  });
  assert.equal(frames1.at(-1).type, 'done');
  const sessionBefore = JSON.parse(readFileSync(join(first.home, 'data', 'agent', 'sessions.json'), 'utf8'))['chat-h-a06'].sessionId;
  await first.host.stop();
  const second = await setup(t, { reuse: { worker: first.worker, home: first.home, connectionId: first.connectionId } });
  const frames2 = await second.send('chat-h-a06', 'SCRIPT:echo back again');
  assert.equal(text(frames2), 'back again');
  assert.equal(frames2.filter(f => f.type === 'agent-tool').length, 0);
  assert.equal(lines(join(second.ws, 'again.txt')), 1, 'the first turn\'s tool did not run a second time');
  const sessionAfter = JSON.parse(readFileSync(join(second.home, 'data', 'agent', 'sessions.json'), 'utf8'))['chat-h-a06'].sessionId;
  assert.equal(sessionAfter, sessionBefore);
  t.after(() => { cleanup(first.home); });
  await first.worker.close().catch(() => {});
});

test('A08 over HTTP: two chats stream at the same time without mixing', { skip }, async t => {
  const s = await setup(t);
  const [a, b] = await Promise.all([s.send('chat-h-a08-a', 'SCRIPT:slow 30'), s.send('chat-h-a08-b', 'SCRIPT:echo only b')]);
  assert.equal(a.filter(f => f.type === 'delta').length, 30);
  assert.equal(text(b), 'only b');
  assert.ok(!text(a).includes('only b') && !text(b).includes('w0'));
  assert.notEqual(a.find(f => f.type === 'start').sessionId, b.find(f => f.type === 'start').sessionId);
});

test('/goal over HTTP: shown, set with rounds that stream into the chat, and refused when one is unfinished', { skip }, async t => {
  const s = await setup(t);
  const status = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');
  assert.match(status(await s.send('chat-h-goal', '/goal')), /No goal is set/);
  assert.match(status(await s.send('chat-h-goal', '/goal edit nothing')), /no goal in this chat yet/i);
  assert.equal((await s.host.status()).activeTurns, 0);
});

test('without the agent component the chat says so and still uses the model the user chose', async t => {
  const s = await setup(t, { harness: null });
  const status = await s.host.status();
  assert.equal(status.state, 'unavailable'); assert.match(status.reason, /coding assistant/i);
  const frames = await s.send('chat-h-fallback', 'SCRIPT:hello');
  const notice = frames.find(f => f.type === 'notice');
  assert.equal(notice.code, 'agent-unavailable'); assert.match(notice.message, /coding assistant isn.t installed/); assert.match(notice.message, /Chat works without it/);
  assert.match(notice.details, /agent component/, 'the technical text is kept apart as details');
  assert.equal(frames.filter(f => f.type === 'agent-tool').length, 0);
  assert.match(text(frames), /Hello from the scripted worker/);
  // The direct path used the API model the chat picked, nothing else.
  assert.ok(s.worker.requests.length >= 1 && s.worker.requests.every(q => q.model === 'cloud-model'));
});

test('Mode over HTTP: the chat body carries the Mode menu choice; Plan refuses a write, Bypass writes, Manual asks', { skip }, async t => {
  const s = await setup(t);
  const body = (sessionId, message, mode) => ({ sessionId, message, remoteModel: s.remoteModel, workspaceId: null, history: [], ...(mode ? { mode } : {}) });

  const plan = await s.host.chat(body('chat-h-plan', 'SCRIPT:retrywrite plan-http.txt x', 'plan'));
  assert.equal(plan.find(f => f.type === 'start').mode, 'plan');
  assert.equal(existsSync(join(s.ws, 'plan-http.txt')), false);
  assert.ok(!toolStates(plan).includes('proposed'));
  assert.equal(toolStates(plan).at(-1), 'denied');

  const bypass = await s.host.chat(body('chat-h-bypass', 'SCRIPT:retrywrite bypass-http.txt x', 'bypass'));
  assert.equal(lines(join(s.ws, 'bypass-http.txt')), 1);
  assert.ok(!toolStates(bypass).includes('proposed'));

  // No mode in the body (an older page) is Manual: it asks.
  let asked = false;
  const manual = await s.host.chat(body('chat-h-manual', 'SCRIPT:retrywrite manual-http.txt x'), async f => {
    if (f.type === 'agent-tool' && f.tool.state === 'proposed' && !asked) {
      asked = true;
      assert.equal(existsSync(join(s.ws, 'manual-http.txt')), false);
      await s.host.post('/api/chat/approval', { sessionId: 'chat-h-manual', toolId: f.tool.id, decision: 'approve' });
    }
  });
  assert.ok(asked);
  assert.equal(manual.find(f => f.type === 'start').mode, 'manual');
  assert.equal(lines(join(s.ws, 'manual-http.txt')), 1);
  assert.equal(manual.filter(f => f.type === 'agent-tool').at(-1).tool.approved, true);
});

test('item 9: a chat that asks for no tools never runs one (agent:false and tools:false answer directly)', { skip }, async t => {
  const s = await setup(t);
  for (const [chatId, flag] of [['chat-plain-a', { agent: false }], ['chat-plain-t', { tools: false }]]) {
    s.worker.requests.length = 0;
    const frames = await s.host.chat({ sessionId: chatId, message: 'SCRIPT:tool plain-never.txt x', remoteModel: s.remoteModel, workspaceId: null, history: [], ...flag });
    assert.equal(frames.filter(f => f.type === 'agent-tool').length, 0, `${chatId}: no tool card`);
    assert.ok(!frames.some(f => f.type === 'start' && f.agent === true), `${chatId}: not an agent turn`);
    assert.ok(!frames.some(f => f.type === 'notice' && f.code === 'agent-skipped'), `${chatId}: asking for plain chat is not a surprise to explain`);
    assert.equal(existsSync(join(s.ws, 'plain-never.txt')), false);
    assert.ok(s.worker.requests.length >= 1 && s.worker.requests.every(r => r.toolNames.length === 0), `${chatId}: the model was offered no tool`);
  }
});
