import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { transform } from 'esbuild';
import { ROOMY_HOST } from './helpers/hardware.mjs';
import { plainNoModel } from '../server/no-model-text.mjs';

/** Sending a chat while no model can answer: say so in plain words, with the one next step, and start no run. */
const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

test('the engine\'s own "no model" wording is turned into plain words with the next step', () => {
  assert.equal(plainNoModel('Choose a loaded model ID.'), 'No model is loaded. Load one in Models first.');
  assert.equal(plainNoModel('This model is not ready.'), 'Your model is still loading. You can chat in a moment.');
  assert.equal(plainNoModel('Something else happened.'), 'Something else happened.');
});

test('the composer\'s notice covers every state: downloading, loading, none, and a model that can answer', async () => {
  const { noModelNotice } = await import(`data:text/javascript;base64,${Buffer.from((await transform(await readFile(new URL('../src/lib/no-model.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' })).code).toString('base64')}`);
  const none = noModelNotice({ instances: [], jobs: [] });
  assert.deepEqual(none, { title: 'No model is loaded', message: 'No model is loaded. Load one in Models first.' });
  assert.match(noModelNotice({ instances: [], jobs: [{ kind: 'install', state: 'running', result: { install: { download: { percent: 76 } } } }] }).message, /downloading \(76%\)\. You can chat when it has finished/);
  assert.match(noModelNotice({ instances: [{ status: 'loading' }], jobs: [] }).message, /still loading/);
  assert.match(noModelNotice({ instances: [{ status: 'error' }, { status: 'stopped' }], jobs: [] }).title, /No model is loaded/, 'a model that failed or stopped cannot answer');
  assert.equal(noModelNotice({ instances: [{ status: 'loading' }, { status: 'ready' }], jobs: [] }), null);
});

test('sendMessage checks for a model before it adds the message, picks a chat or calls the host, and waits for the first read of the engine', async () => {
  const source = (await readFile(new URL('../src/lib/store.ts', import.meta.url), 'utf8')).replace(/\r\n/g, '\n');
  const start = source.indexOf('export async function sendMessage');
  const body = source.slice(start, source.indexOf('\n}\n', start));
  const guard = body.indexOf('noModelNotice(');
  assert.ok(guard > 0, 'the guard is in sendMessage, the one entry for typed messages, loops, plans, goals and Resume');
  for (const later of ['newSession()', 'api.', 'fetch(', 'streamReal', 'appendMessage', 'takeChatPhotos']) {
    const at = body.indexOf(later);
    assert.ok(at < 0 || at > guard, `${later} comes after the guard`);
  }
  assert.match(body, /!nativeState\(\) && !state\.remoteModel && !state\.library\?\.loaded\) \{ pushToast\('info', 'Still starting up'/, 'before the first read of the engine nothing is claimed and nothing is sent');
});

test('over HTTP, a chat with no model loaded gets one plain error and no reply', async t => {
  const home = await mkdtemp(join(tmpdir(), 'strata-nomodel-'));
  const port = await freePort();
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, stdio: 'ignore', env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_REAL_ENGINE: '0', STRATA_TEST_HARDWARE: ROOMY_HOST } });
  t.after(async () => { host.kill(); await new Promise(r => host.once('exit', r)); await rm(home, { recursive: true, force: true }).catch(() => {}); });
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
  assert.ok(up, 'host started');
  const response = await fetch(`http://127.0.0.1:${port}/api/chat/stream`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ message: 'hello', sessionId: 'chat-1', mode: 'manual' }) });
  const frames = (await response.text()).split('\n').filter(Boolean).map(line => JSON.parse(line));
  assert.ok(!frames.some(f => f.type === 'start' || f.type === 'delta'), `no reply started: ${JSON.stringify(frames)}`);
  const failure = frames.find(f => f.type === 'error');
  assert.ok(failure, `an error frame: ${JSON.stringify(frames)}`);
  assert.match(failure.message, /No model is loaded|still loading/);
  assert.doesNotMatch(failure.message, /model ID|Choose a loaded/);
});
