// Response settings (thinking effort, thinking budget, Expert sampling) reach the agent's model request, and nothing changes while they are at their defaults.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { BUDGETS, createLlmProxy, isPrivateHost, nearestBudget, samplingFromExpert, shapeBody } from '../server/agent/llm-shape.mjs';
import { buildProfile } from '../server/agent/profile.mjs';
import { createVisionProbe } from '../server/vision-probe.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('only the Expert settings that differ from their defaults are sent; the thinking budget only while thinking is on', () => {
  assert.equal(samplingFromExpert({}), null);
  assert.equal(samplingFromExpert({ 'sample.temp': 0.6, 'sample.topP': 0.95, 'sample.topK': 40, 'sample.repeat': 1.05, 'sample.seed': -1, 'sample.speedProjection': true, 'sample.maxTokens': 0 }), null, 'all defaults: nothing');
  assert.deepEqual(samplingFromExpert({ 'sample.temp': 0.2, 'sample.topK': 10, 'sample.seed': 7, 'sample.maxTokens': 500, 'sample.speedProjection': false }),
    { temperature: 0.2, top_k: 10, seed: 7, max_tokens: 500, experimental_speed_projection: false });
  const body = { model: 'm', messages: [], max_tokens: 4096, chat_template_kwargs: { enable_thinking: true, reasoning_effort: 'high' } };
  assert.deepEqual(shapeBody(body, {}), body, 'no settings: the request is the same');
  const shaped = shapeBody(body, { sampling: { temperature: 0.2, max_tokens: 500 }, budget: 8192 });
  assert.equal(shaped.temperature, 0.2); assert.equal(shaped.max_tokens, 500, 'an output cap only lowers the request'); assert.equal(shaped.reasoning_budget_tokens, 8192);
  assert.equal(shapeBody({ ...body, max_tokens: 100 }, { sampling: { max_tokens: 500 } }).max_tokens, 100);
  assert.equal(shapeBody({ ...body, temperature: 1 }, { sampling: { temperature: 0.2 } }).temperature, 1, 'a field the request already has wins');
  assert.equal(shapeBody({ ...body, chat_template_kwargs: { enable_thinking: false } }, { budget: 8192 }).reasoning_budget_tokens, undefined, 'thinking off: no budget');
  assert.deepEqual(BUDGETS, [2048, 8192, 32768]); assert.equal(nearestBudget(0), 0); assert.equal(nearestBudget(5000), 2048); assert.equal(nearestBudget(7000), 8192); assert.equal(nearestBudget('32768'), 32768);
});

test('only this computer and private networks are shaped, never a cloud address', () => {
  for (const ok of ['http://127.0.0.1:8080/v1', 'http://localhost:11434/v1', 'http://192.168.1.20:8080', 'http://10.0.0.5/v1', 'http://172.20.1.1/v1', 'http://z440.local:8080', 'http://[::1]:9/v1']) assert.equal(isPrivateHost(ok), true, ok);
  for (const no of ['https://api.openai.com/v1', 'http://8.8.8.8/v1', 'http://172.32.0.1/v1', 'http://example.com', 'nonsense']) assert.equal(isPrivateHost(no), false, no);
});

test('the profile adds a response-settings copy of each shapeable route (local, or an OpenAI-style server on this network) and never of a cloud route', () => {
  const profile = buildProfile({
    local: [{ id: 'a', modelId: 'm', baseUrl: 'http://127.0.0.1:9001', context: 65536 }],
    connections: [
      { id: 'lan', name: 'LAN', baseUrl: 'http://192.168.1.9:8080/v1', protocol: 'openai-chat', apiKey: '', models: [{ id: 'x' }] },
      { id: 'cloud', name: 'Cloud', baseUrl: 'https://api.deepseek.com', protocol: 'openai-chat', apiKey: 'k', models: [{ id: 'y' }] },
    ],
    proxyBase: 'http://127.0.0.1:8791/api/internal/agent-llm',
  });
  const local = profile.routes.find(r => r.kind === 'local');
  assert.deepEqual(Object.keys(local.shaped), ['0', '2048', '8192', '32768']);
  assert.equal(local.shaped[8192], 'strata-local~s8192');
  assert.deepEqual(Object.keys(profile.shapeTargets).sort(), [...Object.values(local.shaped), ...Object.values(profile.routes.find(r => r.connectionId === 'lan').shaped)].sort());
  assert.equal(profile.shapeTargets['strata-local~s0'], 'http://127.0.0.1:9001');
  assert.equal(profile.routes.find(r => r.connectionId === 'cloud').shaped, undefined);
  const doc = JSON.parse(profile.patch.slice(profile.patch.indexOf('[')));
  const providers = doc.find(e => e.id === 'llm-pi-ai').config.providers;
  assert.equal(providers['api-lan'].compat.thinkingFormat, 'chat-template', 'a model server on this network gets the thinking vocabulary of a local model');
  assert.equal(providers['api-lan'].models[0].reasoning, true);
  assert.equal(providers['api-cloud'].compat, undefined, 'a cloud service does not');
  const small = buildProfile({ local: [], connections: [{ id: 'own', name: 'Own', baseUrl: 'http://127.0.0.1:18798/v1', protocol: 'openai-chat', apiKey: '', models: [{ id: 'm', contextMax: 32768 }] }] });
  const own = JSON.parse(small.patch.slice(small.patch.indexOf('['))).find(e => e.id === 'llm-pi-ai').config.providers['api-own'];
  assert.equal(own.models[0].maxTokens, 9830, 'a server on this network keeps 30 percent of its window for the reply so long tasks can be compacted (it was 28672 of 32768: no room, no compaction)');
  assert.match(profile.patch, /"baseURL": "http:\/\/127\.0\.0\.1:8791\/api\/internal\/agent-llm\/strata-local~s8192\/v1"/);
  assert.match(profile.patch, /"x-strata-budget": "8192"/);
  assert.match(profile.patch, /"provider": "strata-local~s8192"/, 'compaction keeps its timing on the copy too');
  const plain = buildProfile({ local: [{ id: 'a', modelId: 'm', baseUrl: 'http://127.0.0.1:9001', context: 65536 }], connections: [] });
  assert.equal(plain.routes[0].shaped, undefined); assert.ok(!/~s/.test(plain.patch), 'without a forwarding address nothing changes');
});

test('the forwarding endpoint adds the settings, streams the reply back and passes the key through', async (t) => {
  const seen = [];
  const upstream = http.createServer((req, res) => {
    let raw = ''; req.on('data', c => { raw += c; }); req.on('end', () => {
      seen.push({ url: req.url, auth: req.headers.authorization, internal: req.headers['x-strata-budget'], body: raw ? JSON.parse(raw) : null });
      res.writeHead(200, { 'content-type': 'text/event-stream' }); res.write('data: one\n\n'); setTimeout(() => { res.write('data: two\n\n'); res.end(); }, 30);
    });
  });
  await new Promise(r => upstream.listen(0, '127.0.0.1', r));
  let expert = null;
  const proxy = createLlmProxy({ targetOf: key => (key === 'k1' ? `http://127.0.0.1:${upstream.address().port}/v1` : null), samplingNow: () => expert });
  const front = http.createServer((req, res) => { const m = /^\/api\/internal\/agent-llm\/([^/]+)(\/.*)$/.exec(req.url); proxy(req, res, m[1], m[2]); });
  await new Promise(r => front.listen(0, '127.0.0.1', r));
  t.after(() => { front.close(); upstream.close(); });
  const call = (key, headers = {}) => new Promise((resolve, reject) => {
    const r = http.request({ host: '127.0.0.1', port: front.address().port, path: `/api/internal/agent-llm/${key}/v1/chat/completions`, method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer abc', ...headers } }, res => { let t2 = ''; res.on('data', c => { t2 += c; }); res.on('end', () => resolve({ status: res.statusCode, text: t2 })); });
    r.on('error', reject); r.end(JSON.stringify({ model: 'm', stream: true, messages: [], max_tokens: 100 }));
  });
  const a = await call('k1');
  assert.equal(a.status, 200); assert.match(a.text, /one[\s\S]*two/);
  assert.deepEqual(seen[0].body, { model: 'm', stream: true, messages: [], max_tokens: 100 }, 'defaults: the body is untouched'); assert.equal(seen[0].auth, 'Bearer abc'); assert.equal(seen[0].url, '/v1/chat/completions'); assert.equal(seen[0].internal, undefined, 'our own header is not forwarded');
  expert = { temperature: 0.1 };
  await call('k1', { 'x-strata-budget': '8192' });
  assert.equal(seen[1].body.temperature, 0.1); assert.equal(seen[1].body.reasoning_budget_tokens, 8192);
  assert.equal((await call('nope')).status, 404);
});

test('a turn uses the response-settings route only when a setting is changed: the model receives the sampling and the budget; defaults go straight to the model', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-shape-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  let expert = null; let front; let bridge;
  const proxy = createLlmProxy({ targetOf: key => bridge.llmTarget(key), samplingNow: () => expert });
  front = http.createServer((req, res) => { const m = /^\/api\/internal\/agent-llm\/([^/]+)(\/.*)$/.exec(req.url); proxy(req, res, m[1], m[2]); });
  await new Promise(r => front.listen(0, '127.0.0.1', r));
  bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }], connections: () => [], env: process.env, startTimeoutMs: 90_000,
    proxyBase: `http://127.0.0.1:${front.address().port}/api/internal/agent-llm`, expertSampling: () => expert });
  t.after(async () => { await bridge.close(); await worker.close(); front.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const run = (chatId, extra = {}) => bridge.runTurn({ chatId, message: 'SCRIPT:echo hello', workspacePath: ws, mode: 'bypass', onFrame: () => {}, ...extra });
  const last = () => worker.requests.filter(r => r.directive === 'echo').at(-1).params;

  await run('s1', { thinking: 'High' });
  assert.equal(last().temperature, undefined, 'defaults: no sampling sent'); assert.equal(last().reasoning_budget_tokens, undefined);
  assert.equal(last().chat_template_kwargs.reasoning_effort, 'high', 'the reasoning level reaches the request');
  await run('s2', { thinking: 'Low' });
  assert.equal(last().chat_template_kwargs.reasoning_effort, 'low');
  await run('s3', { thinking: 'Off' });
  assert.equal(last().chat_template_kwargs.enable_thinking, false, 'Off turns thinking off');
  await run('s4', { thinking: 'High', thinkingBudget: 8192 });
  assert.equal(last().reasoning_budget_tokens, 8192, 'the composer budget reaches the request');
  assert.equal(last().chat_template_kwargs.reasoning_effort, 'high');
  await run('s5', { thinking: 'Off', thinkingBudget: 8192 });
  assert.equal(last().reasoning_budget_tokens, undefined, 'a budget is not sent while thinking is off');
  expert = { temperature: 0.2, top_k: 12, seed: 5 };
  await run('s6', { thinking: 'Medium' });
  assert.equal(last().temperature, 0.2); assert.equal(last().top_k, 12); assert.equal(last().seed, 5); assert.equal(last().reasoning_budget_tokens, undefined);
  await run('s7', { thinking: 'Medium', thinkingBudget: 2048 });
  assert.equal(last().temperature, 0.2); assert.equal(last().reasoning_budget_tokens, 2048);
  expert = null;
  await run('s8', { thinking: 'Medium' });
  assert.equal(last().temperature, undefined, 'back to defaults: straight to the model again');
});

test('a model server on this network gets picture input unless its own model list or the person says it does not read pictures', async () => {
  const lists = { yes: { data: [{ id: 'm', architecture: { input_modalities: ['text', 'image'] } }] }, no: { data: [{ id: 'm', architecture: { input_modalities: ['text'] } }] } };
  let which = 'yes';
  const probe = createVisionProbe({ fetcher: async () => ({ ok: true, json: async () => lists[which] }) });
  const conn = { id: 'c', baseUrl: 'http://127.0.0.1:1/v1', apiKey: '' };
  assert.equal(probe.known(conn, 'm'), undefined);
  assert.equal(await probe.check(conn, 'm'), true); assert.equal(probe.known(conn, 'm'), true);
  which = 'no'; assert.equal(await probe.check(conn, 'm'), true, 'cached for a few minutes');
  assert.equal(await createVisionProbe({ fetcher: async () => { throw new Error('down'); } }).check(conn, 'm'), null, 'unreachable is unknown: pictures stay on by default');
  const build = vision => JSON.parse(buildProfile({ local: [], connections: [{ id: 'own', name: 'Own', baseUrl: 'http://127.0.0.1:18798/v1', protocol: 'openai-chat', apiKey: '', models: [{ id: 'm', contextMax: 32768, vision }] }] }).patch.slice(buildProfile({ local: [], connections: [] }).patch.indexOf('['))).find(e => e.id === 'llm-pi-ai').config.providers['api-own'].models[0].input;
  assert.deepEqual(build(true), ['text', 'image']); assert.deepEqual(build(false), ['text']); assert.deepEqual(build(undefined), ['text', 'image'], 'pictures are on by default');
});
