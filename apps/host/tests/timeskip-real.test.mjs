/**
 * Time-skip in the app, for real: the host starts the job without waiting for it, follows its record, lists the changed files and
 * applies a revision the person chose. The revision step here is a small script (the assistant's turn is covered by the box runs).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createTimeSkip, diffRun, measureFolder, timeskipRevisionMessage } from '../server/timeskip.mjs';

const REPO = resolve(fileURLToPath(new URL('.', import.meta.url)), '..', '..', '..');
const PYTHON = process.env.STRATA_TEST_PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3');
const hasPython = spawnSync(PYTHON, ['--version'], { windowsHide: true }).status === 0;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

test('the revision request tells the assistant to stay in the copy and to say what it did', () => {
  const text = timeskipRevisionMessage('make a launch plan');
  assert.match(text, /current folder is that copy/);
  assert.match(text, /Never read or write outside it/);
  assert.match(text, /make a launch plan$/);
});

test('measureFolder stops early on a folder that is too big to copy', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ts-measure-'));
  try {
    for (let i = 0; i < 30; i++) writeFileSync(join(dir, `f${i}.txt`), 'x');
    assert.equal(measureFolder(dir).tooBig, false);
    assert.equal(measureFolder(dir, { maxFiles: 10 }).tooBig, true);
    mkdirSync(join(dir, 'node_modules'));
    writeFileSync(join(dir, 'node_modules', 'big.txt'), 'y'.repeat(1000));
    assert.equal(measureFolder(dir, { maxBytes: 100 }).tooBig, false, 'node_modules is not copied for a trial, so it does not count');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('diffRun lists added, changed and removed files and says which revision to apply', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ts-diff-'));
  try {
    const base = join(dir, 'base'), cand = join(dir, 'cand');
    mkdirSync(base); mkdirSync(cand);
    writeFileSync(join(base, 'same.txt'), 'same'); writeFileSync(join(cand, 'same.txt'), 'same');
    writeFileSync(join(base, 'edit.txt'), 'one'); writeFileSync(join(cand, 'edit.txt'), 'one two');
    writeFileSync(join(base, 'gone.txt'), 'bye');
    writeFileSync(join(cand, 'new.md'), '# new');
    const found = diffRun({ checkpoints: [{ id: 'cp1-baseline', label: 'baseline', path: base }], selected_checkpoint: 'cp1-baseline', attempts: [{ index: 1, path: cand }] });
    assert.deepEqual(found.files.map(f => [f.path, f.status]), [['edit.txt', 'changed'], ['gone.txt', 'removed'], ['new.md', 'added']]);
    assert.equal(found.files[0].original, 'one'); assert.equal(found.files[0].updated, 'one two');
    assert.equal(found.applyAttempt, 1, 'nothing was kept by the checks, so the person applies the last revision');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a run starts without the caller waiting, shows its changed files, and applies the chosen revision with a rollback copy', { skip: !hasPython && 'no python on this machine', timeout: 120000 }, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'ts-run-'));
  try {
    const workspace = join(dir, 'work'); mkdirSync(workspace);
    writeFileSync(join(workspace, 'plan.md'), 'first draft\n');
    const script = join(dir, 'revise.mjs');
    writeFileSync(script, "import { writeFileSync, appendFileSync } from 'node:fs'; await new Promise(r => setTimeout(r, 1500)); appendFileSync('plan.md', 'second line\\n'); writeFileSync('extra.md', process.env.STRATA_TIMESKIP_PROMPT);");
    const state = join(dir, 'state');
    const ts = createTimeSkip({ repoRoot: REPO, python: PYTHON, env: { ...process.env, STRATA_TIMESKIP_DIR: state } });
    const started = Date.now();
    const result = await ts.startDetached({ workspace, prompt: 'extend the plan', attempts: 1, reviseCommand: `"${process.execPath}" "${script}"` });
    assert.ok(result.ok, JSON.stringify(result));
    assert.ok(Date.now() - started < 1400, 'the call returned while the revision was still running');
    const id = result.run.id;
    let run;
    for (let i = 0; i < 100; i++) { run = ts.status(id); if (['completed', 'failed'].includes(run.phase)) break; await sleep(300); }
    assert.equal(run.phase, 'completed', JSON.stringify(run));
    assert.equal(run.needsUserReview, true, 'no checks were declared, so the person decides');
    const files = ts.files(id);
    assert.deepEqual(files.files.map(f => [f.path, f.status]), [['extra.md', 'added'], ['plan.md', 'changed']]);
    assert.equal(files.applyAttempt, 1);
    assert.equal(readFileSync(join(workspace, 'plan.md'), 'utf8'), 'first draft\n', 'the person\'s files are untouched until they choose');
    const applied = await ts.apply({ id, attempt: files.applyAttempt });
    assert.ok(applied.ok, JSON.stringify(applied));
    assert.match(readFileSync(join(workspace, 'plan.md'), 'utf8'), /second line/);
    assert.equal(readFileSync(join(workspace, 'extra.md'), 'utf8'), 'extend the plan');
    const restored = await ts.apply({ id, restore: true });
    assert.ok(restored.ok, JSON.stringify(restored));
    assert.equal(readFileSync(join(workspace, 'plan.md'), 'utf8'), 'first draft\n');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a run whose host is gone reads as interrupted, not as still working', { skip: !hasPython && 'no python on this machine', timeout: 60000 }, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'ts-gone-'));
  try {
    const state = join(dir, 'state'), id = 'a'.repeat(32);
    mkdirSync(join(state, id), { recursive: true });
    writeFileSync(join(state, id, 'run.json'), JSON.stringify({ id, state: 'generating', workspace: dir, prompt: 'x', checkpoints: [], attempts: [] }));
    const ts = createTimeSkip({ repoRoot: REPO, python: PYTHON, env: { ...process.env, STRATA_TIMESKIP_DIR: state } });
    assert.equal(ts.status(id).phase, 'interrupted');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('Stop ends the job and everything it started, deletes the copies, leaves the person\'s files alone, and the run then reads as cancelled', { skip: !hasPython && 'no python on this machine', timeout: 120000 }, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'ts-stop-'));
  try {
    const workspace = join(dir, 'work'); mkdirSync(workspace);
    writeFileSync(join(workspace, 'plan.md'), 'first draft\n');
    const pidFile = join(dir, 'revise.pid');
    const script = join(dir, 'slow.mjs');
    writeFileSync(script, `import { writeFileSync } from 'node:fs'; writeFileSync(${JSON.stringify(pidFile)}, String(process.pid)); writeFileSync('half-done.md', 'x'); await new Promise(r => setTimeout(r, 120000));`);
    const state = join(dir, 'state');
    const ts = createTimeSkip({ repoRoot: REPO, python: PYTHON, env: { ...process.env, STRATA_TIMESKIP_DIR: state } });
    const started = await ts.startDetached({ workspace, prompt: 'slow one', attempts: 1, reviseCommand: `"${process.execPath}" "${script}"` });
    assert.ok(started.ok, JSON.stringify(started));
    const id = started.run.id;
    for (let i = 0; i < 100 && !existsSync(pidFile); i++) await sleep(200);
    assert.ok(existsSync(pidFile), 'the revision step is running');
    const pid = Number(readFileSync(pidFile, 'utf8'));
    assert.equal(ts.status(id).phase, 'generating');
    assert.equal(await ts.stop(id), true);
    assert.equal(existsSync(join(state, id)), false, 'the copies are deleted');
    assert.equal(ts.status(id).phase, 'cancelled');
    let alive = true; for (let i = 0; i < 40 && alive; i++) { try { process.kill(pid, 0); await sleep(150); } catch { alive = false; } }
    assert.equal(alive, false, 'the revision command (a grandchild of the job) was ended too');
    assert.equal(readFileSync(join(workspace, 'plan.md'), 'utf8'), 'first draft\n');
    assert.equal(existsSync(join(workspace, 'half-done.md')), false, 'nothing reached the person\'s folder');
    assert.equal(await ts.stop('b'.repeat(32)), false, 'an id that is not a run is not "stopped"');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('the revision step reads the host\'s stream: a long silent turn is fine, a failure is told in the host\'s words, a vanished host is "could not be reached"', { timeout: 60000 }, async () => {
  const http = await import('node:http');
  const { spawn } = await import('node:child_process');
  const script = join(REPO, 'apps', 'host', 'server', 'timeskip-revise.mjs');
  const mode = { current: 'ok' };
  const server = http.createServer((req, res) => {
    if (req.headers['x-strata-internal'] !== 's3cret') { res.writeHead(403, { 'content-type': 'application/json' }); res.end('{"error":"Not allowed."}'); return; }
    if (mode.current === 'refuse') { res.writeHead(503, { 'content-type': 'application/json' }); res.end('{"error":"Load a model first, then try Time-skip again."}'); return; }
    res.writeHead(200, { 'content-type': 'application/x-ndjson' });
    res.write('{"type":"beat"}\n');
    setTimeout(() => res.write('{"type":"beat"}\n'), 300);
    setTimeout(() => {
      if (mode.current === 'ok') res.end('{"type":"result","text":"I added a second line."}\n');
      else if (mode.current === 'error') res.end('{"type":"error","message":"The model stopped answering."}\n');
      else res.end();                                                           // the stream ends with no result
    }, 900);
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const run = (env = {}) => new Promise(resolve => {
    const child = spawn(process.execPath, [script], { env: { ...process.env, STRATA_HOST_URL: `http://127.0.0.1:${server.address().port}`, STRATA_INTERNAL_SECRET: 's3cret', STRATA_TIMESKIP_PROMPT: 'x', ...env } });
    let out = '', err = ''; child.stdout.on('data', d => { out += d; }); child.stderr.on('data', d => { err += d; });
    child.on('close', code => resolve({ code, out: out.trim(), err: err.trim() }));
  });
  try {
    mode.current = 'ok'; assert.deepEqual(await run(), { code: 0, out: 'I added a second line.', err: '' });
    mode.current = 'error'; const bad = await run(); assert.equal(bad.code, 1); assert.equal(bad.err, 'The model stopped answering.');
    mode.current = 'cut'; const cut = await run(); assert.equal(cut.code, 1); assert.match(cut.err, /stopped before it finished/);
    mode.current = 'refuse'; assert.match((await run()).err, /Load a model first/);
    assert.match((await run({ STRATA_INTERNAL_SECRET: 'wrong' })).err, /Not allowed/);
    assert.match((await run({ STRATA_INTERNAL_SECRET: '' })).err, /needs the Strata app/);
  } finally { server.close(); }
  const gone = await new Promise(resolve => { const child = spawn(process.execPath, [script], { env: { ...process.env, STRATA_HOST_URL: 'http://127.0.0.1:9', STRATA_INTERNAL_SECRET: 's', STRATA_TIMESKIP_PROMPT: 'x' } }); let err = ''; child.stderr.on('data', d => { err += d; }); child.on('close', code => resolve({ code, err })); });
  assert.equal(gone.code, 1); assert.match(gone.err, /could not be reached/);
});
