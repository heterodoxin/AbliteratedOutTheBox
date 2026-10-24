// A turn always ends: when the model goes silent the watchdog stops it with a plain message.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('a silent model ends the turn with a plain message, and the chat works again afterwards', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-wd-')); const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }], connections: () => [], env: process.env, startTimeoutMs: 90_000, stallMs: 2500, stallToolMs: 6000 });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const out = [];
  const t0 = Date.now();
  await bridge.runTurn({ chatId: 'w1', message: 'SCRIPT:hang', workspacePath: ws, mode: 'bypass', onFrame: f => out.push(f) });
  assert.ok(Date.now() - t0 < 30_000, 'it ended by itself');
  const err = out.find(f => f.type === 'error');
  assert.match(err.message, /stopped answering/); assert.match(err.message, /Send your message again/);
  assert.equal(out.at(-1).type, 'error');
  const again = [];
  await bridge.runTurn({ chatId: 'w1', message: 'SCRIPT:echo fine', workspacePath: ws, mode: 'bypass', onFrame: f => again.push(f) });
  assert.match(again.filter(f => f.type === 'delta').map(f => f.text).join(''), /fine/);
});

test('engine progress keeps a long prompt read alive past the stall limit, then the normal limit applies (#3623)', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-wd-')); const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const started = Date.now();
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }], connections: () => [], env: process.env, startTimeoutMs: 90_000, stallMs: 2500, stallToolMs: 6000, engineGraceMs: 60_000,
    // The engine moves for 7 s (a prompt being read), then goes quiet.
    engineProgress: () => (Date.now() - started < 7_000 ? `reading:${Date.now()}` : null) });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const out = [];
  const t0 = Date.now();
  await bridge.runTurn({ chatId: 'w2', message: 'SCRIPT:hang', workspacePath: ws, mode: 'bypass', onFrame: f => out.push(f) });
  assert.ok(Date.now() - started > 7_000, 'the 2.5 s stall limit did not fire while the engine was reading');
  assert.ok(Date.now() - t0 < 40_000, 'and it still ended by itself once the engine went quiet');
  assert.match(out.find(f => f.type === 'error').message, /stopped answering/);
});
