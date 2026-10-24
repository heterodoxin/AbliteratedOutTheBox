import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';
import { createControl } from '../server/control-runtime.mjs';

/**
 * The host's side of "closing always ends what this app started":
 *  - it asks Control to tie each engine's life to the host's own (--die-with-pid),
 *  - it takes every ordinary stop signal down the clean path,
 *  - started by the desktop shell, it leaves when the shell is gone (a kill or a crash is not announced).
 * The owner runtime's side (the engine ends when the host's pid is gone) is tested in tests/terminal.
 */
const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }
const alive = pid => { try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; } };
async function until(fn, ms = 15000) { const end = Date.now() + ms; while (Date.now() < end) { if (await fn()) return true; await delay(100); } return false; }

test('start asks Control to end the engine with this host; cleanup asks it to end what a lost owner left', async () => {
  const seen = [];
  const control = createControl({
    control: { dir: '/c', python: '/p', cli: '/c/strata.py', bundle: { version: '9' } },
    run: async args => { seen.push(args.join(' ')); return { code: 0, json: { ok: true }, text: '', stderr: '' }; },
  });
  await control.start('app-1');
  await control.start('app-2', { dieWithPid: 4242 });
  await control.start('app-3', { dieWithPid: null });
  await control.cleanup();
  assert.deepEqual(seen, [
    `runtime start --profile app-1 --wait 300 --shutdown-timeout 15 --die-with-pid ${process.pid} --output json`,
    'runtime start --profile app-2 --wait 300 --shutdown-timeout 15 --die-with-pid 4242 --output json',
    'runtime start --profile app-3 --wait 300 --shutdown-timeout 15 --output json',
    'runtime cleanup --output json',
  ]);
});

async function startHost(extraEnv = {}, spawnOptions = {}) {
  const home = await mkdtemp(join(tmpdir(), 'strata-host-life-'));
  const port = await freePort();
  const host = spawn(process.execPath, ['server/index.mjs'], {
    cwd: root, stdio: 'ignore', ...spawnOptions,
    env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_REAL_ENGINE: '1', ...extraEnv },
  });
  let up = false;
  for (let i = 0; i < 120 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
  assert.ok(up, 'host started');
  return { host, port, home, done: () => rm(home, { recursive: true, force: true }) };
}

test('SIGTERM and SIGHUP end the host through the clean shutdown (exit 0)', { skip: process.platform === 'win32' && 'Windows has no SIGTERM/SIGHUP to send' }, async t => {
  for (const signal of ['SIGTERM', 'SIGHUP']) {
    const h = await startHost();
    t.after(h.done);
    const exited = new Promise(r => h.host.once('exit', (code, sig) => r({ code, sig })));
    h.host.kill(signal);
    const result = await Promise.race([exited, delay(20000).then(() => 'timeout')]);
    assert.notEqual(result, 'timeout', `${signal}: the host did not end`);
    assert.deepEqual(result, { code: 0, sig: null }, signal);
  }
});

test('Started by the shell (STRATA_EXIT_WITH_PARENT=1), the host leaves when its parent is killed', async t => {
  // a stand-in shell: it starts the host and then waits to be killed hard
  const home = await mkdtemp(join(tmpdir(), 'strata-host-parent-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const port = await freePort();
  const shell = spawn(process.execPath, ['-e', `
    const {spawn}=require('node:child_process');
    const h=spawn(process.execPath,['server/index.mjs'],{cwd:${JSON.stringify(root)},stdio:'ignore',env:{...process.env,STRATA_EXIT_WITH_PARENT:'1',STRATA_HOME:${JSON.stringify(home)},STRATA_MODELS_DIR:${JSON.stringify(join(home, 'models'))},STRATA_PORT:'${port}',STRATA_SHARE_PORT:'${await freePort()}',STRATA_TUNNEL_PORT:'${await freePort()}',STRATA_REAL_ENGINE:'1'}});
    console.log(h.pid);
    setInterval(()=>{},1000);
  `], { cwd: root, stdio: ['ignore', 'pipe', 'inherit'] });
  const hostPid = await new Promise(r => shell.stdout.once('data', c => r(Number(String(c).trim()))));
  t.after(() => { try { process.kill(hostPid, 'SIGKILL'); } catch { /* gone */ } shell.kill('SIGKILL'); });
  let up = false;
  for (let i = 0; i < 120 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
  assert.ok(up, 'host started');
  assert.ok(alive(hostPid));
  shell.kill('SIGKILL');                         // TerminateProcess on Windows: nothing is announced
  assert.ok(await until(() => !alive(hostPid), 20000), 'the host outlived its killed shell');
  let reachable = true;
  try { await fetch(`http://127.0.0.1:${port}/api/health`, { signal: AbortSignal.timeout(1500) }); } catch { reachable = false; }
  assert.equal(reachable, false, 'the port is free again');
});

test('Without STRATA_EXIT_WITH_PARENT a host started from a source checkout does not follow its parent', async t => {
  const h = await startHost();
  t.after(h.done);
  t.after(() => h.host.kill('SIGKILL'));
  await delay(2500);
  assert.equal(alive(h.host.pid), true);
});
