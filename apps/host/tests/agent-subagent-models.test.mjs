/**
 * Helper agents on a model the person names (server/agent/model-routes.mjs and plugins/strata-models.mjs).
 * Pure tests always run; the real-Harness tests need STRATA_HARNESS_DIR and skip loudly without it.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { resolveModel, modelProblem, routeList } from '../server/agent/model-routes.mjs';
import { AGENT_PROMPT } from '../server/agent/preset.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

const routes = [
  { provider: 'strata-local', model: 'gyro-gyro-s', name: 'qwen3.8-flash-next-gyro-gyro-s', connection: 'this computer', local: true },
  { provider: 'api-aaaa', model: 'deepseek/deepseek-v3.2', name: 'DeepSeek: DeepSeek V3.2', connection: 'OpenRouter', local: false },
  { provider: 'api-aaaa', model: 'anthropic/claude-sonnet-4.5', name: 'Anthropic: Claude Sonnet 4.5', connection: 'OpenRouter', local: false },
  { provider: 'api-bbbb', model: 'deepseek-chat', name: 'DeepSeek Chat', connection: 'DeepSeek', local: false },
];

test('nothing named: the helper uses the chat model', () => {
  assert.equal(resolveModel(routes, {}).status, 'inherit');
  assert.equal(modelProblem(routes, {}), null);
});

test('exact route, exact id, provider/id, connection and name, in any letters', () => {
  const want = (request, id) => { const r = resolveModel(routes, request); assert.equal(r.status, 'exact', JSON.stringify(request)); assert.equal(r.route.model, id); };
  want({ provider: 'api-aaaa', model: 'deepseek/deepseek-v3.2' }, 'deepseek/deepseek-v3.2');
  want({ model: 'deepseek/deepseek-v3.2' }, 'deepseek/deepseek-v3.2');
  want({ model: 'DeepSeek/DeepSeek-V3.2' }, 'deepseek/deepseek-v3.2');
  want({ provider: 'openrouter', model: 'deepseek/deepseek-v3.2' }, 'deepseek/deepseek-v3.2');
  want({ model: 'openrouter/deepseek/deepseek-v3.2' }, 'deepseek/deepseek-v3.2');
  want({ model: 'DeepSeek: DeepSeek V3.2' }, 'deepseek/deepseek-v3.2');
  want({ model: 'deepseek chat' }, 'deepseek-chat');
  want({ model: 'Local' }, 'gyro-gyro-s');
  want({ provider: 'strata-local', model: 'gyro-gyro-s' }, 'gyro-gyro-s');
});

test('a close guess is checked with the person, several fits are listed, nothing fitting lists the models', () => {
  const guess = resolveModel(routes, { model: 'claude sonnet' });
  assert.equal(guess.status, 'confirm'); assert.equal(guess.route.model, 'anthropic/claude-sonnet-4.5');
  assert.match(modelProblem(routes, { model: 'claude sonnet' }), /Did you mean Anthropic: Claude Sonnet 4\.5/);
  const several = resolveModel(routes, { model: 'deepseek' });
  assert.equal(several.status, 'ambiguous'); assert.ok(several.options.length >= 2);
  const none = modelProblem(routes, { model: 'llama banana' });
  assert.match(none, /No model is called "llama banana"/); assert.match(none, /DeepSeek Chat/); assert.match(none, /Nothing was started/);
  assert.match(modelProblem([], { model: 'x' }), /No other model is set up/);
  assert.match(routeList(routes), /running on this computer/);
});

test('the assistant is told to use the subagent tool when asked for a subagent or a model', () => {
  assert.match(AGENT_PROMPT, /ask a subagent/); assert.match(AGENT_PROMPT, /Never look for a model with shell commands/);
});

async function rig(t) {
  const local = await startScriptedWorker();
  const remote = await startScriptedWorker({ model: 'deepseek/deepseek-v3.2' });
  const dir = mkdtempSync(join(tmpdir(), 'strata-sub-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({
    dataDir: dir, env: process.env, startTimeoutMs: 90_000,
    localInstances: () => [{ id: 'inst-1', label: 'Local Scripted', baseUrl: local.url, modelId: 'scripted-model', context: 32768, slots: 2 }],
    connections: () => [{ id: 'conn-1', name: 'OpenRouter', baseUrl: remote.url, protocol: 'openai-chat', apiKey: 'k', models: [{ id: 'deepseek/deepseek-v3.2', name: 'DeepSeek V3.2', contextMax: 32768 }] }],
  });
  t.after(async () => { await bridge.close(); await local.close(); await remote.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const call = async (chatId, args) => {
    const frames = [];
    await bridge.runTurn({ chatId, message: `SCRIPT:call subagent ${JSON.stringify(args)}`, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
    return { frames, text: frames.filter(f => f.type === 'delta').map(f => f.text).join('') };
  };
  return { local, remote, bridge, call, ws };
}

test('the real Harness: the model sees guidance and the discovery list, and a name starts the helper on that model', { skip }, async (t) => {
  const r = await rig(t);
  const first = await r.call('m1', { description: 'Weather', prompt: 'SCRIPT:echo grey', model: 'DeepSeek V3.2' });
  const tools = JSON.parse([...r.local.requests].reverse().find(q => q.toolNames.length).raw).tools;
  const sub = tools.find(x => x.function.name === 'subagent').function;
  assert.match(sub.description, /ask a subagent/); assert.match(sub.description, /do not look for the model yourself/);
  assert.match(sub.parameters.properties.model.description, /"local"/);
  assert.ok(tools.some(x => x.function.name === 'list_subagent_models'));
  // the helper ran on the remote route, not on the chat's model
  for (let i = 0; i < 60 && !r.remote.requests.length; i++) await new Promise(res => setTimeout(res, 250));
  assert.ok(r.remote.requests.length >= 1, `the helper reached the OpenRouter mock (${first.text})`);
});

test('the real Harness: a close guess and an unknown model are answered in plain words and start nothing', { skip }, async (t) => {
  const r = await rig(t);
  const guess = await r.call('m2', { description: 'Weather', prompt: 'x', model: 'deepseek v3' });
  assert.match(guess.text, /Did you mean DeepSeek V3\.2/);
  const none = await r.call('m3', { description: 'Weather', prompt: 'x', model: 'llama banana' });
  assert.match(none.text, /No model is called "llama banana"/);
  const seen = [...r.local.requests].reverse().find(q => q.lastRole === 'tool').raw;
  assert.match(seen, /DeepSeek V3.2/, 'the list the assistant read names the models');
  await new Promise(res => setTimeout(res, 1500));
  assert.equal(r.remote.requests.length, 0, 'nothing was started');
});

test('the real Harness: naming the local model, and naming no model, both work', { skip }, async (t) => {
  const r = await rig(t);
  const before = r.local.requests.length;
  await r.call('m4', { description: 'Local', prompt: 'SCRIPT:echo hi', model: 'the local model' });
  await r.call('m5', { description: 'Plain', prompt: 'SCRIPT:echo hi' });
  for (let i = 0; i < 40 && r.local.requests.filter(q => q.directive === 'echo').length < 2; i++) await new Promise(res => setTimeout(res, 250));
  assert.ok(r.local.requests.length > before);
  assert.equal(r.remote.requests.length, 0);
});

test('the real Harness: a helper that finishes while the chat model is still answering does not end the run early ("interrupted before it finished")', { skip }, async (t) => {
  // The chat model answers slowly, the helper's model (a second server) answers at once: the helper reports back while the parent is mid-answer.
  const slow = await startScriptedWorker({ chunkDelayMs: 900 });
  const fast = await startScriptedWorker({ model: 'deepseek/deepseek-v3.2', chunkDelayMs: 5 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-race-')); const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({ dataDir: dir, env: process.env, startTimeoutMs: 90_000,
    localInstances: () => [{ id: 'inst-1', label: 'Slow chat model', baseUrl: slow.url, modelId: 'scripted-model', context: 32768, slots: 2 }],
    connections: () => [{ id: 'conn-1', name: 'OpenRouter', baseUrl: fast.url, protocol: 'openai-chat', apiKey: 'k', models: [{ id: 'deepseek/deepseek-v3.2', name: 'DeepSeek V3.2', contextMax: 32768 }] }] });
  t.after(async () => { await bridge.close(); await slow.close(); await fast.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const frames = [];
  await bridge.runTurn({ chatId: 'race', message: `SCRIPT:call subagent ${JSON.stringify({ description: 'Quick', prompt: 'SCRIPT:echo hi', model: 'DeepSeek V3.2' })}`, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
  assert.ok(!frames.some(f => f.type === 'error'), `no error: ${JSON.stringify(frames.filter(f => f.type === 'error'))}`);
  assert.equal(frames.at(-1).type, 'done');
  assert.match(frames.filter(f => f.type === 'delta').map(f => f.text).join(''), /Tool finished/, 'the chat model got to answer');
});
