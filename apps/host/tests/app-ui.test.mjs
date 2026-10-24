import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';
import { EventEmitter } from 'node:events';
import { createAppUi, openInBrowser, startOpener, APP_UPDATE_UNAVAILABLE, BROWSER_UPDATE_MESSAGE } from '../server/app-ui.mjs';
import { checkComponentUpdates } from '../server/updates.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
const bridgeEnv = { STRATA_SHELL_BRIDGE: 'http://127.0.0.1:5555', STRATA_SHELL_TOKEN: 'a'.repeat(32) };

test('a host nobody started has no window mode, no Quit and no updater', () => {
  const ui = createAppUi({ env: {} });
  assert.deepEqual(ui.state(), { mode: null, reason: null, canQuit: false, appUpdate: 'unavailable' });
  assert.equal(ui.updater, null);
  assert.equal(ui.quit().ok, false);
});

test('window mode: no Quit entry (closing the window ends everything), the updater waits for the window process', async () => {
  const ui = createAppUi({ env: { STRATA_UI_MODE: 'window' } });
  assert.equal(ui.state().canQuit, false);
  assert.equal(ui.state().appUpdate, 'available');
  assert.equal(ui.quit().status, 409);
  await assert.rejects(ui.updater.check(), error => error.code === APP_UPDATE_UNAVAILABLE);
  ui.setBridge({ url: 'http://127.0.0.1:6001', token: 'b'.repeat(32) });
  assert.ok(ui.updater);
  assert.throws(() => ui.setBridge({ url: 'http://example.com:1', token: 'b'.repeat(32) }), /not a bridge/);
  assert.throws(() => ui.setBridge({ url: 'http://127.0.0.1:6001', token: 'short' }), /not a bridge/);
});

test('browser mode (chosen): Quit works once and the app updater says why it is not available', async () => {
  let stopped = 0;
  const ui = createAppUi({ env: { STRATA_UI_MODE: 'browser' }, shutdown: () => { stopped++; } });
  assert.deepEqual(ui.state(), { mode: 'browser', reason: 'chosen', canQuit: true, appUpdate: 'unavailable' });
  assert.equal(ui.updater.unavailable, BROWSER_UPDATE_MESSAGE);
  await assert.rejects(ui.updater.check(), error => error.code === APP_UPDATE_UNAVAILABLE && error.message === BROWSER_UPDATE_MESSAGE);
  assert.equal(ui.quit().status, 202);
  await delay(20);
  assert.equal(stopped, 1);
});

test('the shell moves a window host to a browser tab: opens the page, sends the notice, drops the window updater', async () => {
  const opened = [], notices = [];
  const ui = createAppUi({ env: { STRATA_UI_MODE: 'window' }, open: async url => { opened.push(url); return true; }, notify: async text => { notices.push(text); } });
  ui.setBridge({ url: 'http://127.0.0.1:6001', token: 'b'.repeat(32) });
  const out = await ui.switchToBrowser({ url: 'http://127.0.0.1:47831/', notice: 'Strata opened in your browser because the app window couldn\'t start on this computer.' });
  assert.deepEqual(out, { mode: 'browser', reason: 'window-failed', opened: true });
  assert.deepEqual(opened, ['http://127.0.0.1:47831/']);
  assert.equal(notices.length, 1);
  assert.equal(ui.state().canQuit, true);
  await assert.rejects(ui.updater.check(), error => error.code === APP_UPDATE_UNAVAILABLE);
  // no browser could be opened: the shell is told, it prints the address
  const none = createAppUi({ env: { STRATA_UI_MODE: 'window' }, open: async () => false, notify: async () => false });
  assert.equal((await none.switchToBrowser({ url: 'http://127.0.0.1:1/' })).opened, false);
});

test('openers are tried in order until one takes the address', async () => {
  const tried = [];
  const ok = await openInBrowser('http://127.0.0.1:1/', { openers: [['xdg-open'], ['gio', 'open'], ['firefox']], start: async (cmd, args) => { tried.push([cmd, ...args].join(' ')); return cmd === 'gio'; } });
  assert.equal(ok, true);
  assert.deepEqual(tried, ['xdg-open http://127.0.0.1:1/', 'gio open http://127.0.0.1:1/']);
  assert.equal(await openInBrowser('x', { openers: [['a'], ['b']], start: async () => false }), false);
});

test('an opener that is missing or fails fast is a failure; one still running is a success', async () => {
  const fake = behaviour => () => { const child = new EventEmitter(); child.unref = () => {}; queueMicrotask(() => behaviour(child)); return child; };
  assert.equal(await startOpener('x', [], { env: {}, settleMs: 200, spawnProcess: fake(c => c.emit('error', new Error('ENOENT'))) }), false);
  assert.equal(await startOpener('x', [], { env: {}, settleMs: 200, spawnProcess: fake(c => c.emit('exit', 3)) }), false);
  assert.equal(await startOpener('x', [], { env: {}, settleMs: 200, spawnProcess: fake(c => c.emit('exit', 0)) }), true);
  assert.equal(await startOpener('x', [], { env: {}, settleMs: 50, spawnProcess: fake(() => {}) }), true);
});

test('update check in browser mode: the app part says it cannot be done here, engine/component updates stay possible', async () => {
  const control = { version: '0.1.0', updateCheck: async () => ({ app: { status: 'available', offered: '0.2.0', notes: 'Faster.' }, engine: { status: 'current', installed: { version: '1' } }, control: { status: 'current' } }) };
  const browser = createAppUi({ env: { STRATA_UI_MODE: 'browser' } });
  const inBrowser = await checkComponentUpdates({ control, appUpdater: browser.updater, appVersion: '0.1.0' });
  assert.equal(inBrowser.strata.status, 'available');
  assert.equal(inBrowser.strata.canInstall, false);
  assert.match(inBrowser.strata.message, /can't be updated in browser mode/);
  // the same check with the window's updater in place offers the update as before
  const windowed = createAppUi({ env: { STRATA_UI_MODE: 'window', ...bridgeEnv }, createUpdater: ({ env }) => env.STRATA_SHELL_BRIDGE ? { check: async () => ({ available: true, version: '0.2.0', notes: 'Faster.' }), install: async () => {} } : null });
  const inWindow = await checkComponentUpdates({ control, appUpdater: windowed.updater, appVersion: '0.1.0' });
  assert.equal(inWindow.strata.canInstall, true);
  assert.doesNotMatch(inWindow.strata.message, /browser mode/);
});

async function freePort() { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

async function startHost(extraEnv) {
  const home = await mkdtemp(join(tmpdir(), 'strata-host-appui-'));
  const port = await freePort();
  const host = spawn(process.execPath, ['server/index.mjs'], {
    cwd: root, stdio: 'ignore',
    env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_NO_ACQUIRE: '1', STRATA_NO_BUNDLED_ENGINE: '1', ...extraEnv },
  });
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
  assert.ok(up, 'the host answers');
  const exited = new Promise(resolve => host.on('exit', code => resolve(code)));
  return { port, host, exited, home, base: `http://127.0.0.1:${port}` };
}

test('browser-mode host over HTTP: mode, Quit needs the client header, then the host stops cleanly and only listens on 127.0.0.1', async () => {
  const { port, host, exited, home, base } = await startHost({ STRATA_UI_MODE: 'browser' });
  try {
    const mode = await (await fetch(`${base}/api/app/mode`)).json();
    assert.equal(mode.mode, 'browser'); assert.equal(mode.canQuit, true);
    // a plain POST (what a web page could send without asking first) is refused, and so is a page from another site
    assert.equal((await fetch(`${base}/api/app/quit`, { method: 'POST' })).status, 403);
    assert.equal((await fetch(`${base}/api/app/quit`, { method: 'POST', headers: { 'x-strata-client': 'ui', origin: 'https://example.com' } })).status, 403);
    // the shell's own routes need the secret
    assert.equal((await fetch(`${base}/api/control/ui`, { method: 'POST', body: '{}' })).status, 403);
    // it is listening on the loopback address only
    const { networkInterfaces } = await import('node:os');
    const other = Object.values(networkInterfaces()).flat().find(i => i && !i.internal && i.family === 'IPv4');
    if (other) await assert.rejects(fetch(`http://${other.address}:${port}/api/health`, { signal: AbortSignal.timeout(2000) }));
    assert.equal((await fetch(`${base}/api/app/quit`, { method: 'POST', headers: { 'x-strata-client': 'ui' } })).status, 202);
    assert.equal(await Promise.race([exited, delay(30000).then(() => 'still running')]), 0);
  } finally { host.kill(); await rm(home, { recursive: true, force: true }); }
});

test('window-mode host over HTTP: Quit is refused; the shell (with its secret) registers the bridge and can move the host to browser mode', async () => {
  const secret = 'c'.repeat(64);
  const { host, home, base } = await startHost({ STRATA_UI_MODE: 'window', STRATA_SHUTDOWN_TOKEN: secret, STRATA_OPENER: 'record' });
  try {
    assert.equal((await fetch(`${base}/api/app/quit`, { method: 'POST', headers: { 'x-strata-client': 'ui' } })).status, 409);
    const as = (body, token = secret) => fetch(`${base}/api/control/ui`, { method: 'POST', headers: { 'x-strata-shutdown': token, 'content-type': 'application/json' }, body: JSON.stringify(body) });
    assert.equal((await as({}, 'd'.repeat(64))).status, 403);
    assert.equal((await as({ bridge: { url: 'http://evil.example:1', token: 'e'.repeat(32) } })).status, 400);
    assert.equal((await as({ bridge: { url: 'http://127.0.0.1:9', token: 'e'.repeat(32) } })).status, 200);
    const moved = await (await as({ mode: 'browser', reason: 'window-failed', open: false })).json();
    assert.equal(moved.mode, 'browser');
    assert.equal((await (await fetch(`${base}/api/app/mode`)).json()).reason, 'window-failed');
  } finally { host.kill(); await rm(home, { recursive: true, force: true }); }
});
