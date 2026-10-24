import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createModelProviders, normalizeBaseUrl, PROVIDER_CATALOG } from '../server/model-providers.mjs';
import { createMockModelApi } from './mock-model-api.mjs';
import { assertPrivate } from './helpers/assert-private.mjs';

test('connections persist, redact keys, retain unchanged keys, and validate endpoints/models', async t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-providers-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const service = createModelProviders({ directory });
  const draft = { kind: 'custom', name: 'Other PC', baseUrl: 'http://192.168.1.20:1234/v1/', protocol: 'openai-chat', apiKey: 'test-key', models: [{ id: 'my-model', name: 'My model' }] };
  const saved = service.save(draft).connections[0];
  assert.equal(saved.baseUrl, 'http://192.168.1.20:1234/v1');
  assert.equal(saved.apiKeySet, true); assert.equal('apiKey' in saved, false);
  assertPrivate(join(directory, 'model-providers.json'));
  service.save({ ...saved, apiKey: '' });
  assert.equal(JSON.parse(readFileSync(join(directory, 'model-providers.json')))[0].apiKey, 'test-key');
  const reopened = createModelProviders({ directory });
  assert.deepEqual(reopened.list(), service.list());
  reopened.save({ ...saved, apiKey: '', baseUrl: 'http://192.168.1.21:1234/v1' });
  assert.equal(reopened.list().connections[0].apiKeySet, false);
  assert.throws(() => service.save({ ...draft, models: [] }), /at least one/);
  assert.throws(() => service.save({ ...draft, models: [{ id: 'a' }, { id: 'a' }] }), /unique/);
  for (const url of ['file:///tmp', 'https://user:pass@example.com/v1', 'https://a/v1?key=secret', 'https://a/v1/chat/completions']) assert.throws(() => normalizeBaseUrl(url));
  reopened.remove(saved.id); assert.equal(reopened.list().connections.length, 0);
  assert.throws(() => reopened.resolve({ connectionId: saved.id, modelId: 'my-model' }), /no longer/);
});

test('discovery uses provider auth, pagination, and avoids redirect/error credential leaks', async t => {
  const mock = await createMockModelApi(), directory = mkdtempSync(join(tmpdir(), 'strata-providers-'));
  t.after(async () => { await mock.close(); rmSync(directory, { recursive: true, force: true }); });
  const service = createModelProviders({ directory });
  const draft = { kind: 'custom', name: 'Mock', baseUrl: mock.baseUrl, protocol: 'openai-chat', apiKey: 'test-key', models: [{ id: 'mock-chat' }] };
  assert.equal((await service.discover(draft)).models.length, 2);
  assert.equal(mock.requests[0].headers.authorization, 'Bearer test-key');
  assert.equal((await service.discover({ ...draft, protocol: 'anthropic' })).models.length, 3);
  assert.equal(mock.requests[1].headers['x-api-key'], 'test-key');
  const before = mock.requests.length;
  await assert.rejects(service.discover({ ...draft, baseUrl: mock.baseUrl.replace('/v1', '/redirect') }), /Could not reach/);
  assert.equal(mock.requests.length, before + 1);
  await assert.rejects(service.discover({ ...draft, baseUrl: mock.baseUrl.replace('/v1', '/bad') }), err => !err.message.includes('test-key') && /credentials/.test(err.message));
});

for (const protocol of ['openai-chat', 'openai-responses', 'anthropic']) test(`${protocol} forwards history, streams text, and records reported usage`, async t => {
  const mock = await createMockModelApi(), directory = mkdtempSync(join(tmpdir(), 'strata-providers-'));
  t.after(async () => { await mock.close(); rmSync(directory, { recursive: true, force: true }); });
  const service = createModelProviders({ directory });
  const connection = service.save({ kind: 'custom', name: 'Mock', baseUrl: mock.baseUrl, protocol, apiKey: 'test-key', models: [{ id: 'mock-chat' }] }).connections[0];
  const frames = [];
  const result = await service.stream({ selection: { connectionId: connection.id, modelId: 'mock-chat' }, message: 'New prompt', history: [{ role: 'user', content: 'Earlier prompt' }, { role: 'assistant', content: 'Earlier answer' }], systemPrompt: 'Be clear.', onFrame: frame => frames.push(frame) });
  assert.match(frames.filter(f => f.type === 'delta').map(f => f.text).join(''), /Hello.*test API/);
  assert.equal(result.tokensIn, 14); assert.equal(result.tokensOut, 7); assert.equal(result.usageSource, 'reported');
  assert.equal(frames.at(-1).type, 'done');
  const body = mock.requests[0].body;
  assert.deepEqual((body.messages ?? body.input).slice(-3).map(m => m.content), ['Earlier prompt', 'Earlier answer', 'New prompt']);
  if (protocol === 'openai-responses') assert.equal(body.store, false);
});

test('truncated streams, provider stream errors, and cancellation preserve output without a false done', async t => {
  const mock = await createMockModelApi(), directory = mkdtempSync(join(tmpdir(), 'strata-providers-'));
  t.after(async () => { await mock.close(); rmSync(directory, { recursive: true, force: true }); });
  const service = createModelProviders({ directory });
  const connection = service.save({ kind: 'custom', name: 'Mock', baseUrl: mock.baseUrl, protocol: 'openai-chat', apiKey: 'test-key', models: ['truncated', 'stream-error', 'slow'].map(id => ({ id })) }).connections[0];
  for (const modelId of ['truncated', 'stream-error']) {
    const frames = [];
    await assert.rejects(service.stream({ selection: { connectionId: connection.id, modelId }, message: 'Test', onFrame: frame => frames.push(frame) }), err => !err.message.includes('test-key'));
    assert.equal(frames.some(f => f.type === 'done'), false);
  }
  const controller = new AbortController(), frames = [];
  await assert.rejects(service.stream({ selection: { connectionId: connection.id, modelId: 'slow' }, message: 'Test', signal: controller.signal, onFrame: frame => { frames.push(frame); if (frame.type === 'delta') controller.abort(); } }), { name: 'AbortError' });
  assert.equal(frames.find(f => f.type === 'delta').text, 'Partial'); assert.equal(frames.some(f => f.type === 'done'), false);
});


test('every API key preset discovers and streams with its configured protocol', async t => {
  const mock = await createMockModelApi(), directory = mkdtempSync(join(tmpdir(), 'strata-presets-'));
  t.after(async () => { await mock.close(); rmSync(directory, { recursive: true, force: true }); });
  const service = createModelProviders({ directory });
  assert.ok(PROVIDER_CATALOG.length >= 20);
  assert.equal(new Set(PROVIDER_CATALOG.map(p => p.id)).size, PROVIDER_CATALOG.length);
  for (const preset of PROVIDER_CATALOG) {
    if (preset.baseUrl) assert.equal(normalizeBaseUrl(preset.baseUrl), preset.baseUrl);
    const draft = { kind: 'provider', provider: preset.id, baseUrl: mock.baseUrl, apiKey: 'test-key', models: [{ id: 'mock-chat' }] };
    assert.ok((await service.discover(draft)).models.length > 0, preset.name);
    const connection = service.save(draft).connections.at(-1);
    assert.equal(connection.protocol, preset.protocol);
    const frames = [];
    await service.stream({ selection: { connectionId: connection.id, modelId: 'mock-chat' }, message: 'Test', onFrame: f => frames.push(f) });
    assert.equal(frames.at(-1).type, 'done', preset.name);
    const request = mock.requests.at(-1);
    if (preset.protocol === 'anthropic') assert.equal(request.headers['x-api-key'], 'test-key');
    else assert.equal(request.headers.authorization, 'Bearer test-key');
    if (preset.protocol === 'openai-chat' && !['deepseek','openrouter','google','moonshotai'].includes(preset.id)) assert.equal(request.body.stream_options, undefined, 'optional usage parameter is not assumed for newly added providers');
  }
  assert.throws(() => service.save({ kind: 'provider', provider: 'qwen', apiKey: 'test-key', models: [{ id: 'test' }] }), /full http/);
});
