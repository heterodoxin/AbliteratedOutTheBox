/**
 * OpenRouter models: the Reasoning menu works when OpenRouter's own model list says the model can think, and the chosen effort reaches
 * OpenRouter in its own words (`reasoning: { effort }`). The server is a mock; no key exists and none is stored.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { buildProfile, OPENROUTER_EFFORTS } from '../server/agent/profile.mjs';
import { createModelProviders } from '../server/model-providers.mjs';
import { createVisionProbe, modelSupportsReasoning, readsPicturesByDefault } from '../server/vision-probe.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

const LIST = { data: [
  { id: 'deepseek/deepseek-v3.2', name: 'DeepSeek: DeepSeek V3.2', context_length: 163840, architecture: { input_modalities: ['text'] }, supported_parameters: ['tools', 'reasoning', 'include_reasoning', 'temperature'] },
  { id: 'meta/plain', name: 'Plain', context_length: 8192, architecture: { input_modalities: ['text', 'image'] }, supported_parameters: ['tools', 'temperature'] },
  { id: 'unlisted/x', name: 'No parameters listed' },
] };

test('a model list row says whether the model can think', () => {
  assert.equal(modelSupportsReasoning(LIST.data[0]), true);
  assert.equal(modelSupportsReasoning({ supported_parameters: ['include_reasoning'] }), true);
  assert.equal(modelSupportsReasoning(LIST.data[1]), false);
  assert.equal(modelSupportsReasoning(LIST.data[2]), null);
});

test('fetching the model list keeps the reasoning answer, and a saved connection keeps it', async t => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-or-')); t.after(() => rmSync(directory, { recursive: true, force: true }));
  const fetchImpl = async () => ({ ok: true, status: 200, json: async () => LIST, body: { cancel() {} } });
  const service = createModelProviders({ directory, fetchImpl });
  const found = await service.discover({ kind: 'provider', provider: 'openrouter', apiKey: 'mock-key', models: [] });
  assert.deepEqual(found.models.map(m => [m.id, m.reasoning]), [['meta/plain', false], ['deepseek/deepseek-v3.2', true], ['unlisted/x', null]].sort((a, b) => a[0].localeCompare(b[0])).sort((x, y) => found.models.findIndex(m => m.id === x[0]) - found.models.findIndex(m => m.id === y[0])));
  const saved = service.save({ kind: 'provider', provider: 'openrouter', apiKey: 'mock-key', models: found.models }).connections[0];
  assert.equal(saved.models.find(m => m.id === 'deepseek/deepseek-v3.2').reasoning, true);
  assert.equal(createModelProviders({ directory }).list().connections[0].models.find(m => m.id === 'meta/plain').reasoning, false);
});

test('the probe asks the list once for pictures and thinking, and pictures are on unless something says no', async () => {
  const probe = createVisionProbe({ fetcher: async () => ({ ok: true, json: async () => LIST }) });
  const conn = { id: 'c', baseUrl: 'https://openrouter.ai/api/v1', apiKey: 'k' };
  assert.equal(await probe.check(conn, 'deepseek/deepseek-v3.2'), false);
  assert.equal(probe.knownThinks(conn, 'deepseek/deepseek-v3.2'), true);
  assert.equal(await probe.check(conn, 'unlisted/x'), null);
  assert.equal(readsPicturesByDefault({}, null), true, 'unknown: on');
  assert.equal(readsPicturesByDefault({ vision: false }, true), false, 'the person switched it off');
  assert.equal(readsPicturesByDefault({}, false), false, 'the server says no');
});

test('the profile gives a thinking model the OpenRouter efforts and the OpenRouter format, even at another address', () => {
  const conn = { id: 'or', name: 'OpenRouter', provider: 'openrouter', baseUrl: 'http://127.0.0.1:1/v1', protocol: 'openai-chat', apiKey: 'k', models: [{ id: 'a/think', reasoning: true }, { id: 'a/plain', reasoning: false }, { id: 'a/unknown' }] };
  const entries = JSON.parse(buildProfile({ local: [], connections: [conn] }).patch.split('\n').slice(1).join('\n'));
  const provider = Object.values(entries.find(e => e.id === 'llm-pi-ai').config.providers)[0];
  assert.equal(provider.compat.thinkingFormat, 'openrouter');
  const [think, plain, unknown] = provider.models;
  assert.deepEqual(think.reasoningEfforts, OPENROUTER_EFFORTS); assert.equal(think.reasoning, true);
  assert.equal(plain.reasoning, undefined); assert.equal(unknown.reasoning, undefined);
  assert.deepEqual(unknown.input, ['text', 'image'], 'pictures on by default');
});

test('the real Harness: the chosen effort is sent to OpenRouter as reasoning.effort', { skip }, async t => {
  const worker = await startScriptedWorker({ model: 'a/think' });
  const dir = mkdtempSync(join(tmpdir(), 'strata-or-run-')); const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({ dataDir: dir, env: process.env, startTimeoutMs: 90_000, localInstances: () => [], connections: () => [{ id: 'or-conn-1', name: 'OpenRouter', provider: 'openrouter', baseUrl: worker.url, protocol: 'openai-chat', apiKey: 'mock', models: [{ id: 'a/think', name: 'Think', contextMax: 32768, reasoning: true }] }] });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const seen = {};
  for (const level of ['High', 'Max', 'Off']) {
    await bridge.runTurn({ chatId: `or-${level}`, message: 'SCRIPT:echo hi', workspacePath: ws, mode: 'bypass', thinking: level, choice: { connectionId: 'or-conn-1', modelId: 'a/think' }, onFrame: () => {} });
    seen[level] = worker.requests.at(-1).params.reasoning;
  }
  assert.deepEqual(seen, { High: { effort: 'high' }, Max: { effort: 'xhigh' }, Off: { effort: 'none' } });
});
