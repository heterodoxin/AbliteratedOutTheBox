/**
 * Agent bridge, end to end against the real pinned Harness and a scripted worker.
 *
 * Needs the built Harness: set STRATA_HARNESS_DIR to a checkout built with `pnpm install` and
 * `pnpm run build:artifacts`. Without it these tests skip, loudly, rather than pass.
 *
 * The model is scripted (tests/helpers/scripted-worker.mjs). What is real: the Harness process, its
 * agent loop, its tool executor and approval gate, the gateway, the durable session, and the bridge.
 * A scripted model proves the path; it does not prove a model chooses tools well.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, mkdirSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

function cleanup(dir) {
  // The Harness's Windows sandbox adds access entries under the directories it works in, which can make a
  // plain recursive delete fail with EPERM. Cleanup is best effort; a leftover temp dir is not a failure.
  try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ }
}

const lines = file => readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).length;

async function rig(t, { connections = [], dataDir } = {}) {
  const worker = await startScriptedWorker({});
  const dir = dataDir ?? mkdtempSync(join(tmpdir(), 'strata-agent-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => connections, env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); if (!dataDir) cleanup(dir); });
  const run = (chatId, message, opts = {}) => {
    const frames = [];
    const answerWith = opts.answer; // 'approve' | 'deny' | undefined
    const promise = bridge.runTurn({
      chatId, message, workspacePath: ws, choice: opts.choice ?? null, mode: opts.mode ?? null, history: opts.history ?? [],
      onFrame: f => {
        frames.push(f);
        opts.onFrame?.(f);
        if (answerWith && f.type === 'agent-tool' && f.tool.state === 'proposed') {
          void bridge.approve({ chatId, callId: f.tool.id, decision: answerWith }).catch(e => frames.push({ type: 'test-error', message: e.message }));
        }
      },
      signal: opts.signal,
    });
    return { frames, promise };
  };
  return { worker, bridge, dir, ws, run, local };
}

const states = (frames, id) => frames.filter(f => f.type === 'agent-tool' && (!id || f.tool.id === id)).map(f => f.tool.state);
const text = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');

test('A01/A02 a turn streams reasoning and text, then finishes with usage', { skip }, async t => {
  const r = await rig(t);
  const { frames, promise } = r.run('chat-a01', 'SCRIPT:hello');
  await promise;
  assert.equal(frames[0].type, 'queued');
  const start = frames.find(f => f.type === 'start');
  assert.equal(start.agent, true); assert.equal(start.model, 'Scripted'); assert.equal(start.contextMax, 32768);
  assert.match(frames.filter(f => f.type === 'reasoning').map(f => f.text).join(''), /Considering a greeting/);
  assert.equal(text(frames), 'Hello from the scripted worker.');
  const done = frames.at(-1);
  assert.equal(done.type, 'done'); assert.equal(done.finishReason, 'stop'); assert.equal(done.usageSource, 'reported');
  assert.ok(done.tokensIn > 0 && done.tokensOut > 0);
  assert.equal(frames.filter(f => f.type === 'error').length, 0);
  // The route is the declared local one, not a default the catalog would have chosen.
  assert.ok(r.worker.requests.every(q => q.model === 'scripted-model'));
});

test('A03 a tool runs exactly once, in the Harness, after the user approves', { skip }, async t => {
  const r = await rig(t);
  const { frames, promise } = r.run('chat-a03', 'SCRIPT:tool marker.txt hi', { answer: 'approve' });
  await promise;
  const calls = frames.filter(f => f.type === 'agent-tool' && f.tool.state === 'running');
  assert.ok(calls.length >= 1);
  const id = frames.find(f => f.type === 'agent-tool').tool.id;
  assert.deepEqual([...new Set(states(frames, id))], ['running', 'proposed', 'done']);
  assert.equal(frames.filter(f => f.type === 'test-error').length, 0);
  assert.equal(lines(join(r.ws, 'marker.txt')), 1, 'the marker must have exactly one line: the tool ran once');
  const card = frames.filter(f => f.type === 'agent-tool').at(-1).tool;
  assert.equal(card.scope, 'shell'); assert.match(card.preview, /marker\.txt/);
  assert.equal(frames.at(-1).type, 'done');
  // The worker only saw a transcript: it has no execution path, and it was asked twice (call, then result).
  assert.equal(r.worker.requests.filter(q => q.toolNames.length).length, 2);
  assert.match(text(frames), /Tool finished/);
});

test('A04 a denied tool never runs, and the agent is told', { skip }, async t => {
  const r = await rig(t);
  const { frames, promise } = r.run('chat-a04', 'SCRIPT:tool denied.txt nope', { answer: 'deny' });
  await promise;
  const id = frames.find(f => f.type === 'agent-tool').tool.id;
  assert.deepEqual([...new Set(states(frames, id))], ['running', 'proposed', 'denied']);
  assert.equal(existsSync(join(r.ws, 'denied.txt')), false, 'a denied tool must not run');
  assert.equal(frames.at(-1).type, 'done', 'the turn still settles');
  // The model got the denial back as a tool result it can read.
  assert.match(text(frames), /reject|denied|Tool finished/i);
});

test('A05 Stop cancels exactly that chat, waits for the turn to end, and leaves other chats running', { skip }, async t => {
  const r = await rig(t);
  const slowA = r.run('chat-a05-a', 'SCRIPT:slow 400');
  const slowB = r.run('chat-a05-b', 'SCRIPT:slow 60');
  // wait until A is demonstrably live
  for (let i = 0; i < 200 && slowA.frames.filter(f => f.type === 'delta').length < 3; i++) await new Promise(x => setTimeout(x, 50));
  assert.ok(slowA.frames.filter(f => f.type === 'delta').length >= 3, 'the turn must be live before Stop, or the test proves nothing');
  const stopped = await r.bridge.stop('chat-a05-a');
  assert.equal(stopped.stopped, true); assert.equal(stopped.ended, 'aborted');
  await slowA.promise;
  assert.equal(slowA.frames.at(-1).finishReason, 'aborted'); assert.equal(slowA.frames.at(-1).cancelled, true);
  assert.ok(slowA.frames.filter(f => f.type === 'delta').length < 400, 'the stream was cut short');
  assert.ok(r.worker.aborted >= 1, 'the model request was cancelled at the worker');
  assert.equal(r.bridge.isRunning('chat-a05-a'), false);
  // B was not touched and runs to its end.
  await slowB.promise;
  assert.equal(slowB.frames.at(-1).finishReason, 'stop');
  assert.equal(slowB.frames.filter(f => f.type === 'delta').length, 60);
  // And A can be used again.
  const again = r.run('chat-a05-a', 'SCRIPT:echo after stop'); await again.promise;
  assert.equal(text(again.frames), 'after stop');
});

test('A05b Stop while an approval is pending denies it and ends the turn', { skip }, async t => {
  const r = await rig(t);
  const { frames, promise } = r.run('chat-a05b', 'SCRIPT:tool never.txt x');
  for (let i = 0; i < 200 && !states(frames).includes('proposed'); i++) await new Promise(x => setTimeout(x, 50));
  assert.ok(states(frames).includes('proposed'));
  const stopped = await r.bridge.stop('chat-a05b');
  assert.equal(stopped.stopped, true);
  await promise;
  assert.equal(existsSync(join(r.ws, 'never.txt')), false);
  assert.equal(states(frames).at(-1), 'denied');
  assert.equal(r.bridge.pendingFor('chat-a05b').length, 0);
});

test('A06 reopening a chat resumes its Harness session without replaying a tool', { skip }, async t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-agent-resume-'));
  t.after(() => cleanup(dir));
  const first = await rig(t, { dataDir: dir });
  const turn1 = first.run('chat-a06', 'SCRIPT:tool resume.txt once', { answer: 'approve' });
  await turn1.promise;
  const sessionId = first.bridge.sessionOf('chat-a06');
  assert.ok(sessionId);
  assert.equal(lines(join(first.ws, 'resume.txt')), 1);
  await first.bridge.close();          // the host goes away; the Harness process with it

  // A new host process over the same data directory.
  const second = await rig(t, { dataDir: dir });
  assert.equal(second.bridge.sessionOf('chat-a06'), sessionId, 'the chat keeps its session id');
  const turn2 = second.run('chat-a06', 'SCRIPT:echo second turn');
  await turn2.promise;
  assert.equal(second.bridge.sessionOf('chat-a06'), sessionId);
  assert.equal(text(turn2.frames), 'second turn');
  assert.equal(turn2.frames.filter(f => f.type === 'agent-tool').length, 0, 'no tool card appears on resume');
  assert.equal(lines(join(second.ws, 'resume.txt')), 1, 'the earlier tool effect was not replayed');
  // The agent really carried the earlier conversation: its request holds the earlier turn too.
  const resumed = second.worker.requests.find(q => q.toolNames.length);
  assert.ok(resumed.messageCount > 5, `expected earlier messages in the request, saw ${resumed.messageCount}`);
});

test('A08 two chats run independently, each with its own session and approval', { skip }, async t => {
  const r = await rig(t);
  const a = r.run('chat-a08-a', 'SCRIPT:tool a.txt A');        // will wait on approval
  for (let i = 0; i < 200 && !states(a.frames).includes('proposed'); i++) await new Promise(x => setTimeout(x, 50));
  assert.ok(states(a.frames).includes('proposed'));
  const b = r.run('chat-a08-b', 'SCRIPT:hello');
  await b.promise;                                             // B finishes while A is parked on an approval
  assert.equal(text(b.frames), 'Hello from the scripted worker.');
  assert.equal(states(a.frames).at(-1), 'proposed', 'A is still waiting');
  assert.notEqual(r.bridge.sessionOf('chat-a08-a'), r.bridge.sessionOf('chat-a08-b'));
  assert.equal(r.bridge.pendingFor('chat-a08-b').length, 0);
  assert.equal(r.bridge.pendingFor('chat-a08-a').length, 1);
  assert.deepEqual(r.bridge.pendingAll().map(c => c.chatId), ['chat-a08-a'], 'the board lists every chat that waits, and only those');
  const callId = a.frames.find(f => f.type === 'agent-tool').tool.id;
  // An approval cannot be answered through the wrong chat.
  await assert.rejects(r.bridge.approve({ chatId: 'chat-a08-b', callId, decision: 'approve' }), /no longer waiting/);
  await r.bridge.approve({ chatId: 'chat-a08-a', callId, decision: 'approve' });
  await a.promise;
  assert.equal(states(a.frames).at(-1), 'done');
  assert.equal(lines(join(r.ws, 'a.txt')), 1);
});

test('a second send to a chat that is mid-turn is refused', { skip }, async t => {
  const r = await rig(t);
  const first = r.run('chat-busy', 'SCRIPT:slow 80');
  for (let i = 0; i < 100 && !first.frames.some(f => f.type === 'delta'); i++) await new Promise(x => setTimeout(x, 50));
  await assert.rejects(r.run('chat-busy', 'SCRIPT:hello').promise, /still working/);
  await first.promise;
});

test('an API connection becomes a route; its key reaches the model by reference and nowhere else', { skip }, async t => {
  const secret = 'sk-test-SECRET-0123456789abcdef';
  const dir = mkdtempSync(join(tmpdir(), 'strata-agent-keys-'));
  t.after(() => cleanup(dir));
  const apiWorker = await startScriptedWorker({ model: 'cloud-model' });
  t.after(() => apiWorker.close());
  const connections = [{ id: 'abcd1234-conn', name: 'Cloud test', baseUrl: apiWorker.url, protocol: 'openai-chat', apiKey: secret, models: [{ id: 'cloud-model', name: 'Cloud model', contextMax: 64000 }] }];
  const r = await rig(t, { dataDir: dir, connections });
  const frames = [];
  await r.bridge.runTurn({ chatId: 'chat-api', message: 'SCRIPT:hello', workspacePath: r.ws, choice: { connectionId: 'abcd1234-conn', modelId: 'cloud-model' }, onFrame: f => frames.push(f) });
  assert.equal(frames.find(f => f.type === 'start').remoteModel, true);
  assert.equal(frames.find(f => f.type === 'start').model, 'Cloud model');
  assert.equal(text(frames), 'Hello from the scripted worker.');
  // The chat used the API worker, the local worker was never asked.
  assert.ok(apiWorker.requests.length >= 1); assert.equal(r.worker.requests.length, 0);
  const expected = createHash('sha256').update(`Bearer ${secret}`).digest('hex');
  assert.ok(apiWorker.requests.every(q => q.authSha === expected), 'the key arrived as the bearer credential');
  // A different chat on the local default is unaffected by the first chat's choice.
  const local = []; await r.bridge.runTurn({ chatId: 'chat-local', message: 'SCRIPT:hello', workspacePath: r.ws, onFrame: f => local.push(f) });
  assert.equal(local.find(f => f.type === 'start').remoteModel, false);
  assert.ok(r.worker.requests.length >= 1);

  // The shell tool the agent runs must not see the key.
  // Bypass: the dump is a plain in-workspace write, which Manual would (rightly) make the person approve first.
  const env = r.run('chat-api', 'SCRIPT:envdump envdump.txt', { choice: { connectionId: 'abcd1234-conn', modelId: 'cloud-model' }, mode: 'bypass' });
  await env.promise;
  assert.ok(existsSync(join(r.ws, 'envdump.txt')), 'the env dump tool must have run');
  const dump = readFileSync(join(r.ws, 'envdump.txt'), 'utf8');
  assert.match(dump, /^PATH=/im, 'the dump must really list the environment, or its silence about the key proves nothing');
  assert.ok(!dump.includes(secret), 'the agent shell must not inherit the key');
  // Nowhere on disk the host wrote: profile, logs, session map. (The compressed session store is checked by absence in frames.)
  const stack = [join(dir, 'agent')];
  while (stack.length) {
    const p = stack.pop();
    if (statSync(p).isDirectory()) { if (p.endsWith('home')) continue; for (const n of readdirSync(p)) stack.push(join(p, n)); continue; }
    assert.ok(!readFileSync(p).includes(secret), `${p} must not contain the key`);
  }
  assert.ok(!JSON.stringify(frames).includes(secret) && !JSON.stringify(r.bridge.status()).includes(secret));
});

test('a missing or removed connection is a typed error, never another model', { skip }, async t => {
  const r = await rig(t);
  await assert.rejects(r.bridge.runTurn({ chatId: 'chat-gone', message: 'SCRIPT:hello', workspacePath: r.ws, choice: { connectionId: 'nope', modelId: 'm' }, onFrame: () => {} }), err => err.code === 'model-unavailable');
  assert.equal(r.worker.requests.length, 0, 'nothing was sent to the local model instead');
});

test('A05c Stop pressed while the agent is still starting cancels the start, and nothing reaches the model', { skip }, async t => {
  const r = await rig(t);
  const { frames, promise } = r.run('chat-a05c', 'SCRIPT:slow 200');
  await new Promise(x => setTimeout(x, 150));          // the Harness process takes about two seconds to boot
  const stopped = await r.bridge.stop('chat-a05c');
  assert.equal(stopped.stopped, true);
  const result = await promise;
  assert.equal(result.cancelled, true);
  assert.equal(frames.at(-1).finishReason, 'aborted');
  assert.equal(r.worker.requests.length, 0, 'the model was never asked');
  // The chat is usable afterwards.
  const again = r.run('chat-a05c', 'SCRIPT:echo fine'); await again.promise;
  assert.equal(text(again.frames), 'fine');
});

test('a leftover agent from a host that died is stopped at the next start, and an unrelated PID is never touched', { skip }, async t => {
  const { HarnessProcess, stopStaleAgent } = await import('../server/agent/harness-runtime.mjs');
  const { buildProfile } = await import('../server/agent/profile.mjs');
  const { writeFileSync } = await import('node:fs');
  const { spawn } = await import('node:child_process');
  const dir = mkdtempSync(join(tmpdir(), 'strata-stale-'));
  t.after(() => cleanup(dir));
  const profile = buildProfile({ local: [{ id: 'x', baseUrl: 'http://127.0.0.1:9/v1', modelId: 'm', context: 8192 }] });
  const patchFile = join(dir, 'p.patch.yml'); writeFileSync(patchFile, profile.patch);
  const pidFile = join(dir, 'harness.pid');
  const hp = new HarnessProcess({ harness: resolveHarness({ env: process.env }), home: join(dir, 'home'), patchFile, cwd: join(dir, 'run'), env: profile.env, pidFile });
  await hp.start({ timeoutMs: 90_000 });
  const pid = hp.pid;
  assert.ok(existsSync(pidFile), 'the agent records its PID');
  // The host "dies": nothing calls stop(). The next start finds the record.
  assert.equal(stopStaleAgent(pidFile), pid);
  for (let i = 0; i < 50 && hp.exited === null; i++) await new Promise(x => setTimeout(x, 100));
  assert.notEqual(hp.exited, null, 'the leftover agent was stopped');
  assert.equal(existsSync(pidFile), false);

  // A record that points at some other process must not kill it.
  const bystander = spawn(process.execPath, ['-e', 'setTimeout(() => {}, 60000)'], { stdio: 'ignore' });
  t.after(() => bystander.kill());
  writeFileSync(pidFile, JSON.stringify({ pid: bystander.pid }));
  assert.equal(stopStaleAgent(pidFile), null);
  await new Promise(x => setTimeout(x, 300));
  assert.equal(bystander.exitCode, null, 'an unrelated process is left alone');
});

test('a question the agent asks the person does not hang the turn: it is declared unanswerable and the model carries on', { skip }, async t => {
  const r = await rig(t);
  const { frames, promise } = r.run('chat-ask', 'SCRIPT:ask');
  const result = await Promise.race([promise, new Promise(x => setTimeout(() => x('hung'), 30_000))]);
  assert.notEqual(result, 'hung', 'the turn must not wait forever for an answerer that does not exist');
  assert.equal(frames.at(-1).type, 'done');
  const card = frames.filter(f => f.type === 'agent-tool').at(-1).tool;
  assert.equal(card.name, 'ask_user_question');
  assert.equal(card.state, 'error', 'the tool reports that no one could answer');
  assert.match(text(frames), /Tool finished/, 'the model got the result back and finished');
});
