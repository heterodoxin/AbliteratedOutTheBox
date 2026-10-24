// QA report F04: a skill created with "Skill enabled" off stays off, and it keeps its trigger phrases.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');

async function startHost(home) {
  const socket = createServer(); await new Promise(r => socket.listen(0, '127.0.0.1', r));
  const port = socket.address().port; await new Promise(r => socket.close(r));
  const child = spawn(process.execPath, ['server/index.mjs'], { cwd: root, env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_REAL_ENGINE: '0', STRATA_WORKSPACE: join(home, 'work') }, stdio: 'ignore', windowsHide: true });
  const base = `http://127.0.0.1:${port}`;
  const call = async (path, method = 'GET', body) => {
    const response = await fetch(base + path, { method, headers: body ? { 'Content-Type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined });
    return { status: response.status, data: await response.json() };
  };
  for (let i = 0; i < 100; i++) { try { if ((await call('/api/skills')).status === 200) break; } catch { /* not up yet */ } await new Promise(r => setTimeout(r, 100)); }
  const stopped = new Promise(r => child.once('exit', r));
  return { call, stop: async () => { child.kill(); await stopped; } };
}

test('a skill created with Skill enabled off is saved off, kept after a restart, and an enabled one stays on', { timeout: 60000 }, async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-qa2-skill-'));
  let host = await startHost(home);
  try {
    const off = await host.call('/api/skills', 'POST', { name: 'QA disabled creation check', description: 'a', body: 'Harmless test instructions.', enabled: false, triggers: ['qa check', ' ', 'qa check', 'disabled'] });
    assert.equal(off.status, 201);
    const row = off.data.skills.find(s => s.id === off.data.id);
    assert.equal(row.enabled, false, 'the answer to the save already says off');
    assert.deepEqual(row.triggers, ['qa check', 'disabled']);
    const on = await host.call('/api/skills', 'POST', { name: 'QA enabled one', description: '', body: 'More instructions.', enabled: true });
    assert.equal(on.data.skills.find(s => s.id === on.data.id).enabled, true);
    const plain = await host.call('/api/skills', 'POST', { name: 'QA no flag', body: 'Older callers send no flag.' });
    assert.equal(plain.data.skills.find(s => s.id === plain.data.id).enabled, true, 'no flag keeps the old behaviour: on');
    assert.equal((await host.call('/api/skills', 'POST', { name: 'Bad', body: 'x', triggers: 'not a list' })).status, 400);
    await host.stop();
    host = await startHost(home);
    const again = (await host.call('/api/skills')).data;
    assert.equal(again.find(s => s.id === off.data.id).enabled, false, 'still off after a restart');
    assert.deepEqual(again.find(s => s.id === off.data.id).triggers, ['qa check', 'disabled']);
    assert.equal(again.find(s => s.id === on.data.id).enabled, true);
  } finally { await host.stop(); await rm(home, { recursive: true, force: true, maxRetries: 3 }).catch(() => {}); }
});

test('F05: the host refuses an output limit below 0 or above its maximum, and keeps the saved one', { timeout: 60000 }, async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-qa2-tokens-'));
  const host = await startHost(home);
  try {
    const value = async () => (await host.call('/api/engine')).data.expert.find(p => p.id === 'sample.maxTokens').value;
    assert.equal(await value(), 0);
    assert.equal((await host.call('/api/engine', 'PATCH', { expert: [{ id: 'sample.maxTokens', value: -1 }] })).status, 400);
    assert.equal((await host.call('/api/engine', 'PATCH', { expert: [{ id: 'sample.maxTokens', value: 1048577 }] })).status, 400);
    assert.equal((await host.call('/api/engine', 'PATCH', { expert: [{ id: 'sample.maxTokens', value: 2.5 }] })).status, 400);
    assert.equal(await value(), 0, 'nothing was saved');
    assert.equal((await host.call('/api/engine', 'PATCH', { expert: [{ id: 'sample.maxTokens', value: 4096 }] })).status, 200);
    assert.equal(await value(), 4096);
  } finally { await host.stop(); await rm(home, { recursive: true, force: true, maxRetries: 3 }).catch(() => {}); }
});
