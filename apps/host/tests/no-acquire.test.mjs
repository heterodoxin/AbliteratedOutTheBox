import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:net';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { AcquisitionRefused, forbidden, require as requireAcquisition } from '../server/acquisition.mjs';
import * as engineInstall from '../server/engine-install.mjs';
import { createConnectorInstaller } from '../server/cloudflared-install.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const spy = pathToFileURL(fileURLToPath(new URL('./helpers/acquire-spy.mjs', import.meta.url))).href;
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

test('the host-side switch: STRATA_NO_ACQUIRE or the packaged app forbids acquiring, nothing else does', () => {
  assert.equal(forbidden({}), false);
  for (const v of ['1', 'true', 'YES', 'on']) assert.equal(forbidden({ STRATA_NO_ACQUIRE: v }), true, v);
  for (const v of ['0', 'false', '', 'no']) assert.equal(forbidden({ STRATA_NO_ACQUIRE: v }), false, v);
  assert.equal(forbidden({ STRATA_PACKAGED: '1' }), true);
  assert.throws(() => requireAcquisition('the llama.cpp engine', undefined, { STRATA_NO_ACQUIRE: '1' }), AcquisitionRefused);
  assert.doesNotThrow(() => requireAcquisition('x', undefined, {}));
});

test('the llama.cpp and cloudflared fetches refuse before any request is made', async () => {
  const previous = process.env.STRATA_NO_ACQUIRE;
  process.env.STRATA_NO_ACQUIRE = '1';
  const realFetch = globalThis.fetch;
  let requests = 0;
  globalThis.fetch = async () => { requests++; throw new Error('network'); };
  const directory = await mkdtemp(join(tmpdir(), 'strata-no-acquire-'));
  try {
    await assert.rejects(engineInstall.install({ force: true }), err => err instanceof AcquisitionRefused && /llama\.cpp/.test(err.message) && err.status === 409);
    await assert.rejects(engineInstall.findRelease(), AcquisitionRefused);
    let connectorRequests = 0;
    const installer = createConnectorInstaller({ directory, platform: 'linux', arch: 'x64', env: { STRATA_CLOUDFLARED_SEARCH: 'own' }, fetchImpl: async () => { connectorRequests++; return { ok: false }; } });
    await assert.rejects(installer.ensure(), err => err instanceof AcquisitionRefused && /optional connector component/.test(err.message));
    assert.equal(requests + connectorRequests, 0, 'nothing was requested');
    assert.equal(existsSync(join(directory, '.install-')), false);
  } finally {
    globalThis.fetch = realFetch;
    if (previous === undefined) delete process.env.STRATA_NO_ACQUIRE; else process.env.STRATA_NO_ACQUIRE = previous;
    await rm(directory, { recursive: true, force: true });
  }
});

test('a connector placed in the component folder is used as it is; it is never fetched', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'strata-connector-component-'));
  const components = join(directory, 'components', 'cloudflared');
  await mkdir(components, { recursive: true });
  const name = process.platform === 'win32' ? 'cloudflared.exe' : 'cloudflared';
  await writeFile(join(components, name), 'binary', { mode: 0o755 });
  try {
    const installer = createConnectorInstaller({ directory: join(directory, 'own'), componentDirectory: components, env: { STRATA_CLOUDFLARED_SEARCH: 'own' }, run: async () => 'cloudflared version 2026.5.0', fetchImpl: async () => { throw new Error('no network'); } });
    assert.equal(await installer.locate(), join(components, name));
    assert.equal(await installer.ensure(), join(components, name));
  } finally { await rm(directory, { recursive: true, force: true }); }
});

async function runHost({ env, calls }) {
  const home = await mkdtemp(join(tmpdir(), 'strata-no-acquire-host-'));
  const log = join(home, 'spy.log');
  await writeFile(log, '');
  const port = await freePort(), base = `http://127.0.0.1:${port}`;
  const host = spawn(process.execPath, ['--import', spy, 'server/index.mjs'], {
    cwd: root, stdio: 'ignore',
    env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_REAL_ENGINE: '1', STRATA_CLOUDFLARED_SEARCH: 'own', ACQUIRE_SPY_LOG: log, ...env },
  });
  const call = async (path, method = 'GET', body) => {
    const response = await fetch(base + path, { method, headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
    return { status: response.status, body: await response.json().catch(() => null) };
  };
  try {
    let up = false;
    for (let i = 0; i < 100 && !up; i++) { try { up = (await call('/api/health')).status === 200; } catch { await delay(100); } }
    assert.ok(up, 'host started');
    const results = await calls(call);
    await delay(300);
    const lines = (await readFile(log, 'utf8')).split('\n').filter(Boolean).map(l => JSON.parse(l));
    return { results, attempts: lines };
  } finally {
    host.kill();
    await new Promise(r => host.once('exit', r));
    await rm(home, { recursive: true, force: true });
  }
}

const probes = async call => ({
  health: await call('/api/health'),
  binary: await call('/api/engine/binary', 'POST'),
  tunnel: await call('/api/serving/tunnel', 'POST', { action: 'prepare' }),
  control: await call('/api/control/status'),
});

test('the packaged host serves, and every download/install route is refused without one attempt', { timeout: 60000 }, async () => {
  const { results, attempts } = await runHost({ env: { STRATA_PACKAGED: '1', STRATA_NO_ACQUIRE: '1' }, calls: probes });
  assert.equal(results.health.status, 200);
  assert.equal(results.binary.status, 409);
  assert.match(results.binary.body.error ?? results.binary.body.message ?? JSON.stringify(results.binary.body), /never downloads/);
  assert.ok(results.tunnel.status >= 400, 'preparing the connector is a refusal');
  assert.equal(results.control.body.acquisition.forbidden, true);
  assert.deepEqual(attempts, [], 'nothing tried to reach the internet or run an installer');
});

test('the spy is live: without the switch the same routes DO attempt to download (and are blocked)', { timeout: 60000 }, async () => {
  const { attempts } = await runHost({ env: { STRATA_PACKAGED: '', STRATA_NO_ACQUIRE: '' }, calls: async call => { await call('/api/engine/binary', 'POST'); await call('/api/serving/tunnel', 'POST', { action: 'prepare' }); } });
  assert.ok(attempts.some(a => a.kind === 'fetch' && /api\.github\.com\/repos\/ggml-org\/llama\.cpp/.test(a.detail)), JSON.stringify(attempts));
});
