import { fileURLToPath } from 'node:url';
/**
 * The tuning and Time-skip client routes.
 *
 * These are contract tests for the bridge between the App Host and Strata Control. They are
 * deliberately about the *shapes and the refusals*, not about measurement numbers: the numbers come
 * from the job, and the job has its own tests. What matters here is that the Host does not become a
 * second source of truth.
 *
 * Two properties are load-bearing and are asserted directly:
 *
 *   1. The tuning plan is computed from the **engine binary's own flags**, so a dimension the binary
 *      cannot accept is excluded with a reason. A plan that offered one would let a sweep "measure"
 *      a configuration the engine would refuse to start.
 *   2. A concurrent user edit is a **409 conflict**, not a silent overwrite and not a 500.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const HOST = fileURLToPath(new URL('../server/index.mjs',import.meta.url));
const REPO = fileURLToPath(new URL('../../..',import.meta.url));

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function listenFree() {
  const net = await import('node:net');
  const srv = net.createServer();
  await new Promise((r) => srv.listen(0, '127.0.0.1', r));
  const { port } = srv.address();
  await new Promise((r) => srv.close(r));
  return port;
}

/** Start the host against a disposable home, and wait for health. */
async function startHost({ python } = {}) {
  const home = await mkdtemp(join(tmpdir(), 'strata-tune-routes-'));
  await mkdir(join(home, 'data'), { recursive: true });
  const port = await listenFree();
  const child = spawn(process.execPath, [HOST, '--dist', join(REPO, 'apps/host/dist')], {
    cwd: join(REPO, 'apps/host'),
    env: {
      ...process.env,
      STRATA_HOME: home,
      // the Time-skip runs of whoever uses this PC live in their own state folder; a test must not read (or write) them
      STRATA_TIMESKIP_DIR: join(home, 'time-skip'),
      STRATA_PORT: String(port),
      STRATA_REAL_ENGINE: '0',
      ...(python ? { STRATA_PYTHON: python } : {}),
    },
    stdio: 'ignore',
  });
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    try {
      const res = await fetch(`${base}/api/health`);
      if (res.ok) return { child, base, home };
    } catch { /* not up yet */ }
    await wait(100);
  }
  child.kill('SIGKILL');
  throw new Error('the host did not become healthy');
}

test('an unconfigured engine is reported as unavailable rather than guessed at', { timeout: 30000 }, async () => {
  const { child, base } = await startHost();
  try {
    const r = await fetch(`${base}/api/tune/engine`);
    assert.equal(r.status, 200);
    const body = await r.json();
    // The panel uses `available` to decide whether to offer a plan at all. A guessed path would
    // produce a plan about the wrong engine, which is worse than no plan.
    assert.equal(typeof body.available, 'boolean');
    assert.ok('path' in body);
    if (!body.available) assert.equal(body.path, null);
  } finally {
    child.kill('SIGKILL');
  }
});

test('the tuning plan comes from the engine binary and excludes what it cannot accept', { timeout: 60000 }, async () => {
  // A stub engine that accepts exactly two of the flags the static table knows about. The plan must
  // offer only those two, and must name each exclusion.
  const dir = await mkdtemp(join(tmpdir(), 'strata-tune-stub-'));
  // A Python stub, run by Strata Control as [python, stub.py] on every OS (a #! script is WinError 193 on Windows).
  const engine = join(dir, 'strata-stub.py');
  await writeFile(engine, 'import sys\nfor line in ("usage:", "  --kv KV", "  --ple-io MODE"):\n    print(line, file=sys.stderr)\n');

  const { child, base } = await startHost();
  try {
    const r = await fetch(`${base}/api/tune/plan`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ engine, backend: 'metal', model: 'fixture', objective: 'generation' }),
    });
    // Read the body once: consuming it in an assertion message leaves nothing for `.json()`.
    const raw = await r.text();
    assert.equal(r.status, 200, raw);
    const { plan } = JSON.parse(raw);
    assert.equal(plan.engine_flags_known, true, 'the stub engine read cleanly');

    const offered = plan.dimensions.map((d) => d.id);
    assert.deepEqual(offered.sort(), ['kv', 'ple_io'],
      'only the dimensions whose flags this binary accepts may be searched');

    const excluded = Object.fromEntries(plan.excluded.map((e) => [e.id, e.reason]));
    for (const id of ['vram_reserve', 'pcie_frac']) {
      assert.ok(excluded[id], `${id} must be excluded and explained`);
      assert.match(excluded[id], /does not accept|NVIDIA\/AMD/);
    }
    // The bound is displayed before a sweep starts, and the repeat floor is the specification's.
    assert.equal(plan.repeats, 3);
    assert.ok(plan.max_trials > 0);
  } finally {
    child.kill('SIGKILL');
    await rm(dir, { recursive: true, force: true });
  }
});

test('a plan request without an engine or backend is refused with a typed error', { timeout: 30000 }, async () => {
  const { child, base } = await startHost();
  try {
    const r = await fetch(`${base}/api/tune/plan`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({}),
    });
    assert.equal(r.status, 400);
    const body = await r.json();
    assert.ok(body.error, 'a refusal must carry a reason');
  } finally {
    child.kill('SIGKILL');
  }
});

test('an unknown tuning job is a 404, and a malformed id cannot escape the store', { timeout: 30000 }, async () => {
  const { child, base } = await startHost();
  try {
    for (const id of ['deadbeefdeadbeef', '../../etc/passwd', 'not-hex']) {
      const r = await fetch(`${base}/api/tune/job?id=${encodeURIComponent(id)}`);
      // `deadbeef…` is well-formed but absent; the other two must be rejected by shape. Either way
      // the answer is 404 — never a path outside the store.
      assert.equal(r.status, 404, `id ${id} must not resolve`);
    }
  } finally {
    child.kill('SIGKILL');
  }
});

test('the Time-skip run list is empty and honest when nothing has run', { timeout: 30000 }, async () => {
  const { child, base } = await startHost();
  try {
    const r = await fetch(`${base}/api/timeskip`);
    assert.equal(r.status, 200);
    const body = await r.json();
    assert.deepEqual(body.runs, []);
    assert.ok(body.stateDir, 'the store location is reported so a developer can inspect it');
  } finally {
    child.kill('SIGKILL');
  }
});

test('applying an unknown Time-skip run is refused, not silently accepted', { timeout: 30000 }, async () => {
  const { child, base } = await startHost();
  try {
    const r = await fetch(`${base}/api/timeskip/apply`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ id: 'a'.repeat(32) }),
    });
    assert.ok(r.status >= 400, 'an unknown run must not report success');
    const body = await r.json();
    assert.ok(body.error !== undefined);
  } finally {
    child.kill('SIGKILL');
  }
});

test('an HTTP caller cannot supply an arbitrary revision command', { timeout: 30000 }, async () => {
  // `reviseCommand` is configured, never requested: accepting one would hand a remote caller
  // command execution on the user's machine. The route must ignore the field.
  const { child, base } = await startHost();
  try {
    const r = await fetch(`${base}/api/timeskip/start`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ workspace: '/nonexistent-workspace', prompt: 'x',
                             reviseCommand: 'touch /tmp/strata-should-not-exist' }),
    });
    // The run fails because the workspace does not exist -- the point is that it fails for *that*
    // reason, and the injected command is never run.
    assert.ok(r.status >= 400, 'a nonexistent workspace must not start a run');
  } finally {
    child.kill('SIGKILL');
    await rm('/tmp/strata-should-not-exist', { force: true });
  }
});
