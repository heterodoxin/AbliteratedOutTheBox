import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVisionProbe, effectiveVision, modelReadsPictures } from '../server/vision-probe.mjs';
import { createModelProviders } from '../server/model-providers.mjs';

const answer = (routes, seen = []) => async (url, init) => {
  const path = new URL(url).pathname;
  seen.push(path);
  const body = routes[path];
  return body === undefined ? { ok: false, status: 404, json: async () => ({}) } : { ok: true, status: 200, json: async () => body };
};

test('a model list row says whether the model reads pictures in every common shape', () => {
  assert.equal(modelReadsPictures({ architecture: { input_modalities: ['text', 'image'] } }), true);
  assert.equal(modelReadsPictures({ architecture: { input_modalities: ['text'] } }), false);
  assert.equal(modelReadsPictures({ modalities: { input: ['text', 'image'] } }), true);
  assert.equal(modelReadsPictures({ capabilities: ['completion', 'vision'] }), true);
  assert.equal(modelReadsPictures({ capabilities: { vision: false } }), false);
  assert.equal(modelReadsPictures({ id: 'x' }), null);
  assert.equal(modelReadsPictures(null), null);
});

test('what the person ticked wins over what the server said', () => {
  assert.equal(effectiveVision({ vision: true }, false), true);
  assert.equal(effectiveVision({ vision: false }, true), false);
  assert.equal(effectiveVision({ vision: null }, true), true);
  assert.equal(effectiveVision({}, undefined), undefined);
});

test('a llama-server started with --mmproj says so in /props, next to /v1, and the answer is cached', async () => {
  const seen = [];
  const probe = createVisionProbe({ fetcher: answer({ '/v1/models': { data: [{ id: 'smol' }] }, '/props': { modalities: { vision: true } } }, seen) });
  const conn = { id: 'c1', baseUrl: 'http://127.0.0.1:8080/v1' };
  assert.equal(probe.known(conn, 'smol'), undefined);
  assert.equal(await probe.check(conn, 'smol'), true);
  assert.equal(await probe.check(conn, 'smol'), true);
  assert.equal(probe.known(conn, 'smol'), true);
  assert.deepEqual(seen, ['/v1/models', '/props']);
});

test('a text-only server counts as not reading pictures; an unreachable one is unknown (pictures stay on by default)', async () => {
  const text = createVisionProbe({ fetcher: answer({ '/v1/models': { data: [{ id: 'm' }] }, '/props': { modalities: { vision: false } } }) });
  assert.equal(await text.check({ id: 'c', baseUrl: 'http://h/v1' }, 'm'), false);
  const down = createVisionProbe({ fetcher: async () => { throw new Error('refused'); } });
  assert.equal(await down.check({ id: 'c', baseUrl: 'http://h/v1' }, 'm'), null);
});

test('a provider list that names image input is believed without asking /props', async () => {
  const seen = [];
  const probe = createVisionProbe({ fetcher: answer({ '/v1/models': { data: [{ id: 'gpt', architecture: { input_modalities: ['text', 'image'] } }] } }, seen) });
  assert.equal(await probe.check({ id: 'c', baseUrl: 'https://api.example.com/v1' }, 'gpt'), true);
  assert.deepEqual(seen, ['/v1/models']);
});

test('the tick "This model reads pictures" is saved with the model and survives a reload', t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-vision-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const service = createModelProviders({ directory });
  const saved = service.save({ kind: 'custom', name: 'Other PC', baseUrl: 'http://192.168.1.20:8080/v1', protocol: 'openai-chat', apiKey: '', models: [{ id: 'smol', name: 'SmolVLM', vision: true }, { id: 'text', name: 'Text only' }] }).connections[0];
  assert.deepEqual(saved.models.map(m => m.vision), [true, null]);
  const again = createModelProviders({ directory }).list().connections[0];
  assert.deepEqual(again.models.map(m => m.vision), [true, null]);
});
