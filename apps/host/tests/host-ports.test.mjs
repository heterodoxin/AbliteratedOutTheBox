import test from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';
import { spawn } from 'node:child_process';
import { mkdtempSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { findFreePort, listenProblem } from '../server/ports.mjs';

const cwd = fileURLToPath(new URL('..', import.meta.url));
const hold = port => new Promise(resolve => { const s = net.createServer(); s.listen(port, '127.0.0.1', () => resolve(s)); });

test('a taken preferred port gives the next free one, never the taken one', async () => {
  const first = await hold(0); const taken = first.address().port;
  try {
    const got = await findFreePort(taken, '127.0.0.1');
    assert.notEqual(got, taken);
    assert.ok(got > 0);
    assert.notEqual(await findFreePort(got, '127.0.0.1', { skip: [got] }), got);
  } finally { first.close(); }
});

test('the plain sentences name the cause and the one next step', () => {
  const use = listenProblem({ code: 'EADDRINUSE' }, 8791);
  assert.match(use, /Port 8791 is already in use/); assert.match(use, /STRATA_PORT/);
  const denied = listenProblem({ code: 'EACCES' }, 8792);
  assert.match(denied, /Hyper-V, WSL and Docker/); assert.match(denied, /excludedportrange/); assert.match(denied, /STRATA_PORT/);
  assert.doesNotMatch(denied, /EACCES|permission denied/i);
});

test('the host on a taken port prints the plain sentence and exits, with no stack trace', async t => {
  const home = realpathSync(mkdtempSync(join(tmpdir(), 'strata-port-'))); t.after(() => rmSync(home, { recursive: true, force: true }));
  const blocker = await hold(0); t.after(() => blocker.close());
  const port = blocker.address().port;
  const child = spawn(process.execPath, ['server/index.mjs'], { cwd, env: { ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), STRATA_MODELS_DIR: join(home, 'm') } });
  let err = ''; child.stderr.on('data', d => { err += d; });
  const code = await new Promise(resolve => { const timer = setTimeout(() => { child.kill(); resolve('timeout'); }, 30000); child.once('exit', c => { clearTimeout(timer); resolve(c); }); });
  assert.equal(code, 1, err);
  assert.match(err, new RegExp(`Port ${port} is already in use`));
  assert.doesNotMatch(err, /at Server\.setupListenHandle/);
});
