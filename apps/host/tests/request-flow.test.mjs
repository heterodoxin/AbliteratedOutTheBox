import assert from 'node:assert/strict';
import { test } from 'node:test';
import http from 'node:http';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { transform } from 'esbuild';
import { RequestQueue } from '../server/request-queue.mjs';
import { selectSkills, skillInstructions } from '../server/skill-selection.mjs';

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const listen = server => new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server.address().port)));
const close = server => new Promise(resolve => server.close(resolve));

test('a cancelled queued request never executes or interrupts the active task', async () => {
  const queue = new RequestQueue();
  const active = new AbortController(), queued = new AbortController();
  let finish, ran = false;
  const first = queue.run(active.signal, () => new Promise(resolve => { finish = resolve; }));
  const second = queue.run(queued.signal, () => { ran = true; });
  assert.equal(queue.waiting, 1);
  queued.abort();
  await assert.rejects(second, { name: 'AbortError' });
  assert.equal(ran, false); assert.equal(active.signal.aborted, false);
  finish('done'); assert.equal(await first, 'done');
  assert.equal(await queue.run(new AbortController().signal, () => 'next'), 'next');
});

test('skill instructions require both enabled state and a matching workspace', () => {
  const skills = [
    { id: 'global', name: 'Global review', enabled: true, triggers: ['review'], body: 'Check correctness.' },
    { id: 'scoped', name: 'Local review', enabled: true, onlyIn: 'a', triggers: ['review'], body: 'Check project rules.' },
    { id: 'disabled', name: 'Disabled', enabled: false, onlyIn: 'a', triggers: ['review'], body: 'Never inject me.' },
  ];
  assert.deepEqual(selectSkills(skills, 'review this', 'a').map(s => s.id), ['global', 'scoped']);
  assert.deepEqual(selectSkills(skills, 'review this', 'b').map(s => s.id), ['global']);
  assert.deepEqual(selectSkills(skills, 'hello', 'a'), []);
  assert.match(skillInstructions(selectSkills(skills, 'review this', 'a')), /Check project rules/);
});

test('stream usage comes from the server, includes reasoning latency and preserves instructions', async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-stream-test-'));
  process.env.STRATA_HOME = home;
  process.env.STRATA_MODELS_DIR = join(home, 'models');
  const { proxyCompletion } = await import('../server/inference.mjs');
  let request;
  const fixture = http.createServer(async (req, res) => {
    let body = ''; for await (const chunk of req) body += chunk;
    request = JSON.parse(body);
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    // Deliberately split one SSE line and finish without a trailing newline.
    res.write('data: {"choices":[{"delta":{"reasoning_content":"Check it"}}]}\n\n');
    await wait(20);
    res.write('data: {"choices":[{"delta":{"content":"Several words in a single');
    res.write(' chunk"}}]}\n\n');
    res.end('data: {"choices":[],"usage":{"prompt_tokens":31,"completion_tokens":9,"prompt_tokens_details":{"cached_tokens":7}}}');
  });
  const port = await listen(fixture);
  try {
    const frames = [];
    const result = await proxyCompletion({ baseUrl: `http://127.0.0.1:${port}`, message: 'review', history: [{ role: 'system', content: 'Client instruction' }], systemPrompt: 'Skill: review\nCheck correctness.', onFrame: f => frames.push(f), signal: new AbortController().signal });
    assert.equal(result.tokensOut, 9); assert.equal(result.tokensIn, 31); assert.equal(result.cachedTokens, 7);
    assert.equal(result.usageSource, 'reported'); assert.ok(result.ttftMs < result.durationMs);
    assert.match(request.messages[0].content, /Check correctness/); assert.equal(request.messages[1].content, 'Client instruction');
    assert.equal(frames.filter(f => f.type === 'delta').length, 1);
  } finally { await close(fixture); await rm(home, { recursive: true, force: true }); }
});

test('a stream without usage never presents chunks as tokens', async () => {
  const { proxyCompletion } = await import('../server/inference.mjs');
  const fixture = http.createServer((req, res) => { req.resume(); res.writeHead(200, { 'content-type': 'text/event-stream' }); res.end('data: {"choices":[{"delta":{"content":"many words"}}]}\n\ndata: [DONE]\n\n'); });
  const port = await listen(fixture);
  try {
    const result = await proxyCompletion({ baseUrl: `http://127.0.0.1:${port}`, message: 'hello', onFrame: () => {}, signal: new AbortController().signal });
    assert.equal(result.tokensIn, null); assert.equal(result.tokensOut, null); assert.equal(result.tps, null); assert.equal(result.usageSource, 'unavailable');
  } finally { await close(fixture); }
});

test('Serving cancellation stops active and queued demo requests; test keys enforce access', { timeout: 90000 }, async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-request-host-'));
  await mkdir(join(home, 'data'));
  const id = 'qwen38-q4_k_m';
  await writeFile(join(home, 'data/state.json'), JSON.stringify({ installed: { [id]: { installedAt: new Date().toISOString(), bytes: 491400032, simulated:true } }, beginner:{priority:'balanced',contextLength:'32k',readImages:false,gpuIdle:'never'}, current: id, loaded: id }));
  const socket = http.createServer(); const port = await listen(socket); await close(socket);
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: join(import.meta.dirname, '..'), env: { ...process.env, STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'), STRATA_PORT: String(port), STRATA_REAL_ENGINE: '0' }, stdio: 'ignore' });
  const base = `http://127.0.0.1:${port}`;
  const call = async (path, method = 'GET', body, key) => {
    const res = await fetch(base + path, { method, headers: { 'content-type': 'application/json', ...(key ? { authorization: `Bearer ${key}` } : {}) }, body: body ? JSON.stringify(body) : undefined });
    return { status: res.status, data: await res.json() };
  };
  try {
    // A host started on a loaded computer can take many seconds to listen; wait for it, and say so if it never does (a 4 s wait made this test flaky).
    let up = false;
    for (let i = 0; i < 600 && !up; i++) { try { up = (await call('/api/health')).status === 200; } catch {} if (!up) await wait(50); }
    assert.ok(up, 'the host did not start listening within 30 s');
    const stream = () => fetch(base + '/api/chat/stream', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ message: 'Explain this test in detail' }) }).then(r => r.text());
    const first = stream(); await wait(100); const second = stream(); await wait(100);
    let rows = (await call('/api/serving')).data.requests;
    const active = rows.find(r => r.state === 'generating'), queued = rows.find(r => r.state === 'queued');
    assert.ok(active); assert.ok(queued);
    await call(`/api/serving/requests/${queued.id}/cancel`, 'POST');
    const secondFrames = (await second).trim().split('\n').map(JSON.parse);
    assert.equal(secondFrames.some(f => f.type === 'done'), false); assert.ok(secondFrames.some(f => f.type === 'error'));
    rows = (await call('/api/serving')).data.requests;
    assert.equal(rows.find(r => r.id === active.id).state, 'generating');
    await call(`/api/serving/requests/${active.id}/cancel`, 'POST');
    const firstFrames = (await first).trim().split('\n').map(JSON.parse);
    assert.equal(firstFrames.some(f => f.type === 'done'), false); assert.ok(firstFrames.some(f => f.type === 'error'));
    rows = (await call('/api/serving')).data.requests;
    assert.ok(rows.every(r => r.state === 'error' && !r.cancellable && r.durationMs > 0));
    await call('/api/serving', 'PATCH', { apiKey: 'disposable-test-key' });
    assert.equal((await call('/v1/models')).status, 401);
    assert.equal((await call('/v1/models', 'GET', undefined, 'disposable-test-key')).status, 200);
    await call('/api/serving', 'PATCH', { apiKey: null });
    assert.equal((await call('/v1/models')).status, 200);
  } finally {
    const ended = new Promise(resolve => host.once('exit', resolve)); host.kill('SIGTERM'); await ended;
    await rm(home, { recursive: true, force: true });
  }
});

// Reproduces the browser-observed race: final SSE event arrives before the
// slower HTTP response carrying the earlier Cancelling state.
test('a late cancel response cannot overwrite a completed request event', async () => {
  const source = await readFile(new URL('../src/lib/serving-state.ts', import.meta.url), 'utf8');
  const compiled = await transform(source, { loader: 'ts', format: 'esm' });
  const { newestServingState } = await import(`data:text/javascript;base64,${Buffer.from(compiled.code).toString('base64')}`);
  const cancelling = { revision: 100, requests: [{ state: 'generating', cancelRequested: true }] };
  const completed = { revision: 101, requests: [{ state: 'error', cancellable: false }] };
  assert.equal(newestServingState(completed, cancelling), completed);
  assert.equal(newestServingState(cancelling, completed), completed);
  assert.equal(newestServingState(null, completed), completed);
});
