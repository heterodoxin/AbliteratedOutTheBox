import test from 'node:test';
import { ROOMY_HOST } from './helpers/hardware.mjs';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

/**
 * Strata works without internet. The host is started with every non-loopback fetch failing (and logged), the way a PC with the cable
 * pulled behaves. Starting, reading the screens' data and loading a model make no outside call. The update check is the one
 * thing that reaches out, and only when it is asked for: opening Settings does not ask.
 */
const root = fileURLToPath(new URL('..', import.meta.url));
const preload = pathToFileURL(fileURLToPath(new URL('./helpers/offline-preload.mjs', import.meta.url))).href;
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

async function startOfflineHost() {
  const home = await mkdtemp(join(tmpdir(), 'strata-offline-'));
  const log = join(home, 'net.log');
  const port = await freePort();
  let output = '';
  const host = spawn(process.execPath, ['--import', preload, 'server/index.mjs'], {
    cwd: root, stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_PACKAGED: '1', STRATA_NET_LOG: log, STRATA_NO_ACQUIRE: '1', STRATA_REAL_ENGINE: '0', STRATA_TEST_HARDWARE: ROOMY_HOST },
  });
  host.stdout.on('data', d => { output += d; }); host.stderr.on('data', d => { output += d; });
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
  assert.ok(up, `host started\n${output}`);
  const calls = () => (existsSync(log) ? readFileSync(log, 'utf8').split('\n').filter(Boolean) : []);
  const get = async path => { const r = await fetch(`http://127.0.0.1:${port}${path}`); return { status: r.status, data: await r.json().catch(() => null) }; };
  const post = async path => { const r = await fetch(`http://127.0.0.1:${port}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' }); return { status: r.status, data: await r.json().catch(() => null) }; };
  return { host, get, post, calls, stop: async () => { host.kill(); await new Promise(r => host.once('exit', r)); await rm(home, { recursive: true, force: true }).catch(() => {}); } };
}

test('with no internet the host starts, serves its screens and loads a model without one outside call', async t => {
  const h = await startOfflineHost();
  t.after(h.stop);
  assert.deepEqual(h.calls(), [], 'nothing reached out while starting');
  for (const path of ['/api/native', '/api/setup', '/api/models', '/api/engine', '/api/health']) {
    const r = await h.get(path);
    assert.equal(r.status, 200, `${path} answers`);
  }
  const install = await h.post('/api/models/qwen38-q4_k_m/install');
  assert.equal(install.status, 200);
  let installed = false;
  for (let i = 0; i < 150 && !installed; i++) { installed = (await h.get('/api/models')).data.variants.find(v => v.id === 'qwen38-q4_k_m').state === 'installed'; if (!installed) await delay(100); }
  assert.ok(installed, 'the model is on this computer');
  const load = await h.post('/api/models/qwen38-q4_k_m/load');
  assert.equal(load.status, 200, JSON.stringify(load.data));
  assert.equal(load.data.loaded, 'qwen38-q4_k_m');
  assert.deepEqual(h.calls(), [], 'opening the screens and loading a model made no outside call');
});

test('the update check reaches out only when it is asked for, and offline it answers in plain words', async t => {
  const h = await startOfflineHost();
  t.after(h.stop);
  await h.get('/api/setup'); await h.get('/api/native'); await h.get('/api/models');
  assert.equal(h.calls().length, 0, 'opening the screens does not check');
  const checked = await h.get('/api/updates/check');
  assert.equal(checked.status, 200);
  assert.equal(h.calls().length, 1, 'one check, from the click');
  assert.equal(checked.data.strata.status, 'error');
  assert.equal(checked.data.strata.message, "No internet connection. Try again when you're online.");
  assert.doesNotMatch(JSON.stringify(checked.data), /fetch failed/);
});
