/**
 * The first workspace is never the app's own folder, and any workspace can be renamed (a display name; the path stays).
 * A real host process with a scratch user profile; no STRATA_WORKSPACE, started inside the host folder like the installed app.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const HOST_DIR = resolve(import.meta.dirname, '..');
const freePort = () => new Promise(r => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function start(t, { user, home }) {
  const port = await freePort();
  const env = { ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), USERPROFILE: user, HOME: user, STRATA_REAL_ENGINE: '0', STRATA_COMPONENTS_DIR: join(home, 'components') };
  delete env.STRATA_WORKSPACE;
  const child = spawn(process.execPath, [join(HOST_DIR, 'server', 'index.mjs')], { cwd: HOST_DIR, env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let log = ''; child.stdout.on('data', d => { log += d; }); child.stderr.on('data', d => { log += d; });
  t.after(async () => {
    if (child.exitCode !== null) return;
    if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }); else child.kill('SIGTERM');
    await new Promise(r => child.once('exit', r));
  });
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    try { if ((await fetch(`${base}/api/workspaces`)).ok) break; } catch { /* not up yet */ }
    if (child.exitCode !== null) throw new Error(`the host exited: ${log.slice(-400)}`);
    await sleep(200);
  }
  const call = async (method, path, body) => { const r = await fetch(base + path, { method, headers: { 'content-type': 'application/json' }, body: body ? JSON.stringify(body) : undefined }); return { status: r.status, body: await r.json().catch(() => null) }; };
  return { call };
}

const scratch = () => { const dir = mkdtempSync(join(tmpdir(), 'strata-ws-')); const user = join(dir, 'user'); const home = join(dir, 'home'); mkdirSync(user, { recursive: true }); mkdirSync(home, { recursive: true }); return { dir, user, home }; };

test('item 6: with nothing chosen the first workspace is Documents\\Strata, never the app folder', async t => {
  const s = scratch(); t.after(() => rmSync(s.dir, { recursive: true, force: true }));
  const host = await start(t, s);
  const { body } = await host.call('GET', '/api/workspaces');
  const first = body.workspaces[0];
  assert.equal(first.id, 'root');
  assert.equal(resolve(first.path), join(s.user, 'Documents', 'Strata', 'Workspace'));
  assert.equal(first.name, 'Strata');
  assert.ok(existsSync(first.path), 'the folder is created');
  assert.notEqual(resolve(first.path), HOST_DIR);
});

test('item 6: a saved first workspace that is the app folder moves to the default; its chats follow through an alias', async t => {
  const s = scratch(); t.after(() => rmSync(s.dir, { recursive: true, force: true }));
  mkdirSync(join(s.home, 'data'), { recursive: true });
  writeFileSync(join(s.home, 'data', 'workspaces.json'), JSON.stringify([{ id: 'root', name: 'host', path: HOST_DIR }, { id: 'ws_a', name: 'Mine', path: s.user }]));
  const host = await start(t, s);
  const { body } = await host.call('GET', '/api/workspaces');
  assert.equal(resolve(body.workspaces[0].path), join(s.user, 'Documents', 'Strata', 'Workspace'));
  assert.equal(body.workspaces[0].name, 'Strata');
  assert.deepEqual(body.workspaces[0].aliases, [HOST_DIR]);
  assert.equal(body.workspaces[1].path, s.user, 'a folder the person added is left alone');
});

test('item 6: renaming changes the display name only', async t => {
  const s = scratch(); t.after(() => rmSync(s.dir, { recursive: true, force: true }));
  const host = await start(t, s);
  const before = (await host.call('GET', '/api/workspaces')).body.workspaces[0];
  const done = await host.call('PATCH', '/api/workspaces/root', { name: '  My projects  ' });
  assert.equal(done.status, 200);
  assert.equal(done.body.name, 'My projects'); assert.equal(done.body.path, before.path);
  assert.equal((await host.call('GET', '/api/workspaces')).body.workspaces[0].name, 'My projects', 'it is saved');
  assert.equal((await host.call('PATCH', '/api/workspaces/root', { name: '   ' })).status, 400);
  assert.equal((await host.call('PATCH', '/api/workspaces/nope', { name: 'x' })).status, 404);
});
