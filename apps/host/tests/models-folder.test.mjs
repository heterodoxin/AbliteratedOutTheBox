import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { createControl } from '../server/control-runtime.mjs';

/** The host in the packaged app with a stand-in for Strata Control: the move itself is Control's (tests/terminal/test_models_move.py). */
function fixture(t, { plan = {}, move = null, unfinished = null } = {}) {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'strata-mfolder-')));
  const install = join(root, 'install'), data = join(root, 'host'), oldFolder = join(root, 'old'), newFolder = join(root, 'new');
  for (const d of [install, data, oldFolder, newFolder]) mkdirSync(d, { recursive: true });
  const earlier = { install: process.env.STRATA_INSTALL_DIR, terminal: process.env.STRATA_TERMINAL_SETTINGS_FILE, settings: process.env.STRATA_SETTINGS_FILE, comp: process.env.STRATA_COMPONENTS_DIR };
  process.env.STRATA_INSTALL_DIR = install;
  process.env.STRATA_TERMINAL_SETTINGS_FILE = join(root, 'terminal-settings.json');
  process.env.STRATA_COMPONENTS_DIR = join(root, 'components');
  delete process.env.STRATA_SETTINGS_FILE;
  const calls = [];
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python.exe'), cli: join(root, 'control', 'strata.py') }, version: '0.2.2',
    async installedModels() { return { install_dir: install, models: [] }; },
    async engine() { return { usable: true, engine: { version: '0.2.2', root: root } }; },
    async importPlan() { return {}; },
    async planModelsMove(from, to) {
      calls.push(['plan', from, to]);
      return { ok: true, issue: null, old: from, new: to, same_drive: false, free_bytes: 900e9, models_bytes: 74e9, needed_bytes: 77.7e9, models: [{ group: 'gyro-s', name: 'gyro-s', bytes: 58e9 }, { group: 'coder', name: 'coder', bytes: 16e9 }],
        choices: { move: { possible: true, reason: '' }, keep: { possible: true, reason: '' }, empty: { possible: true, reason: '' } }, ...plan };
    },
    async moveModels(from, to, options) {
      calls.push(['move', from, to, options.installDir]);
      if (move) return move(from, to, options);
      options.onEvent({ event: 'progress', done: 5e9, total: 74e9, file: 'models/gyro-s/a.gguf', group: 'gyro-s', index: 1, count: 3, seconds_left: 300, same_drive: false });
      return { ok: true, event: 'result', done: true, stopped: false, moved: ['gyro-s', 'coder'], remaining: [], bytes_moved: 74e9, total_bytes: 74e9, files_moved: 3, same_drive: false };
    },
    async interruptedModelsMove(from) { calls.push(['status', from]); return unfinished; },
    async resumeModelsMove(from, options) {
      calls.push(['resume', from, options.installDir]);
      options.onEvent({ event: 'progress', done: 20e9, total: 74e9, file: 'models/coder/c.gguf', group: 'coder', index: 2, count: 3, seconds_left: 120, same_drive: false });
      return { ok: true, event: 'result', done: true, stopped: false, moved: ['coder'], remaining: [], bytes_moved: 74e9, total_bytes: 74e9, files_moved: 2, same_drive: false };
    },
    async leaveModelsMove(from, options) { calls.push(['leave', from, options.installDir]); return { ok: true, removed: 1, restored: 1, bytes_removed: 5 }; },
    async stop() { return { ok: true }; },
  };
  const spawnProcess = () => { throw new Error('the host starts no process here'); };
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher: async () => Response.json({}), requests: () => [] });
  t.after(() => {
    for (const [key, name] of [['install', 'STRATA_INSTALL_DIR'], ['terminal', 'STRATA_TERMINAL_SETTINGS_FILE'], ['settings', 'STRATA_SETTINGS_FILE'], ['comp', 'STRATA_COMPONENTS_DIR']]) {
      if (earlier[key] === undefined) delete process.env[name]; else process.env[name] = earlier[key];
    }
    rmSync(root, { recursive: true, force: true });
  });
  native.setDownloadDirectory(oldFolder);
  return { native, control, calls, root, install, oldFolder, newFolder, terminalFile: process.env.STRATA_TERMINAL_SETTINGS_FILE };
}
const finish = async (native, id) => { for (let i = 0; i < 400; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('the plan says what the picked folder means and keeps the person\'s spelling of it', async t => {
  const f = fixture(t);
  const plan = await f.native.planModelsFolder(f.newFolder);
  assert.equal(plan.current, f.oldFolder);
  assert.equal(plan.folder, f.newFolder);
  assert.equal(plan.wholeDrive, false);
  assert.equal(plan.shown, f.newFolder);
  assert.equal(plan.models.length, 2);
  assert.deepEqual(f.calls[0], ['plan', f.oldFolder, f.newFolder]);
});

test('a whole drive is planned as <drive>\\Strata, shown as <drive>\\Strata\\Models', async t => {
  const f = fixture(t);
  const root = f.root.slice(0, f.root.indexOf(process.platform === 'win32' ? '\\' : '/') + 1) || '/';
  const plan = await f.native.planModelsFolder(root);
  assert.equal(plan.wholeDrive, true);
  assert.equal(plan.folder, join(root, 'Strata'));
  assert.equal(plan.shown, join(root, 'Strata', 'Models'));
  rmSync(join(root, 'Strata'), { recursive: true, force: true });
});

test('the card is blocked while a model runs or a job runs: a running model must be stopped on purpose', async t => {
  const f = fixture(t);
  await assert.rejects(f.native.startModelsFolder({ folder: f.newFolder, choice: 'bogus' }), /Choose what to do/);
  const refused = fixture(t, { plan: { choices: { move: { possible: false, reason: 'Your models need 77 GB, this drive has 38 GB free.' }, keep: { possible: true, reason: '' }, empty: { possible: true, reason: '' } } } });
  await assert.rejects(refused.native.startModelsFolder({ folder: refused.newFolder, choice: 'move' }), /need 77 GB/);
  assert.equal(refused.calls.some(c => c[0] === 'move'), false);
});

test('Keep: the new folder is used for new downloads, nothing is moved, both settings files follow', async t => {
  const f = fixture(t);
  writeFileSync(f.terminalFile, JSON.stringify({ data_dir: f.oldFolder, data_dir_at: 1 }));   // the Terminal had chosen the old folder
  const result = await f.native.startModelsFolder({ folder: f.newFolder, choice: 'keep' });
  assert.equal(result.done, true);
  assert.equal(f.native.state().downloadDirectory, f.newFolder);
  assert.equal(f.calls.some(c => c[0] === 'move'), false);
  assert.equal(JSON.parse(readFileSync(f.terminalFile, 'utf8')).data_dir, f.newFolder);
});

test('Start empty: the old models are no longer listed (their run configs are set aside, not deleted), the model files stay', async t => {
  const f = fixture(t);
  const oldFile = join(f.oldFolder, 'models', 'gyro-s', 'a.gguf');
  mkdirSync(join(f.oldFolder, 'models', 'gyro-s'), { recursive: true });
  writeFileSync(oldFile, 'x');
  writeFileSync(join(f.install, 'strata-gyro-s.json'), JSON.stringify({ args: ['--native', oldFile] }));
  writeFileSync(join(f.install, 'strata-other.json'), JSON.stringify({ args: ['--native', join(f.root, 'elsewhere', 'b.gguf')] }));
  const result = await f.native.startModelsFolder({ folder: f.newFolder, choice: 'empty' });
  assert.deepEqual(result.setAside, ['strata-gyro-s.json']);
  assert.equal(existsSync(join(f.install, 'strata-gyro-s.json')), false);
  assert.equal(existsSync(join(f.install, 'old-models', 'strata-gyro-s.json')), true);
  assert.equal(existsSync(join(f.install, 'strata-other.json')), true, 'a model elsewhere is not touched');
  assert.equal(existsSync(oldFile), true);
  assert.equal(f.native.state().downloadDirectory, f.newFolder);
});

test('Move: a job with progress; the folder changes only when everything moved', async t => {
  const f = fixture(t);
  const job = await f.native.startModelsFolder({ folder: f.newFolder, choice: 'move' });
  assert.equal(job.kind, 'models-move');
  const done = await finish(f.native, job.id);
  assert.equal(done.state, 'done');
  assert.equal(done.result.done, true);
  assert.equal(done.result.sameDrive, false);
  assert.equal(f.native.state().downloadDirectory, f.newFolder);
  assert.deepEqual(f.calls.find(c => c[0] === 'move').slice(1), [f.oldFolder, f.newFolder, f.install]);
});

test('Move stopped: the folder stays, what moved is reported', async t => {
  const f = fixture(t, { move: async () => ({ ok: true, event: 'result', done: false, stopped: true, moved: ['coder'], remaining: ['gyro-s'], bytes_moved: 16e9, total_bytes: 74e9, files_moved: 1, same_drive: false }) });
  const job = await f.native.startModelsFolder({ folder: f.newFolder, choice: 'move' });
  const done = await finish(f.native, job.id);
  assert.equal(done.result.stopped, true);
  assert.deepEqual(done.result.moved, ['coder']);
  assert.equal(f.native.state().downloadDirectory, f.oldFolder, 'a partial move never points the app at an incomplete folder');
});

test('Move failed (disk full): the folder stays and the job says what happened', async t => {
  const f = fixture(t, { move: async () => { throw Object.assign(new Error('The drive ran out of space.'), { code: 'disk_full', details: 'ENOSPC: no space left on device', output: { moved: ['coder'], remaining: ['gyro-s'], bytes_moved: 52e9, total_bytes: 74e9 } }); } });
  const job = await f.native.startModelsFolder({ folder: f.newFolder, choice: 'move' });
  const done = await finish(f.native, job.id);
  assert.equal(done.state, 'error');
  assert.equal(done.result.code, 'disk_full');
  assert.equal(done.result.bytes, 52e9);
  assert.match(done.details, /ENOSPC/);
  assert.equal(f.native.state().downloadDirectory, f.oldFolder);
});

test('Move: Stop writes the cancel file and the run ends by itself', async t => {
  let release;
  const f = fixture(t, { move: (from, to, options) => new Promise(resolve => { release = () => resolve({ ok: true, event: 'result', done: false, stopped: existsSync(options.cancelFile), moved: [], remaining: ['gyro-s', 'coder'], bytes_moved: 0, total_bytes: 74e9, files_moved: 0, same_drive: false }); }) });
  const job = await f.native.startModelsFolder({ folder: f.newFolder, choice: 'move' });
  await f.native.cancel(job.id);
  assert.equal(f.native.state().jobs.find(j => j.id === job.id).state, 'running', 'it ends when Control has cleaned up, not before');
  release();
  const done = await finish(f.native, job.id);
  assert.equal(done.result.stopped, true);
  assert.equal(f.native.state().downloadDirectory, f.oldFolder);
});

test('a move needs the app: the Control command line is built from the plan and the cancel file', async () => {
  const seen = [];
  const control = createControl({ control: { dir: '/c', python: '/c/py', cli: '/c/strata.py' }, run: async (args, options) => { seen.push([args, options]); return args.includes('--plan') ? { code: 0, json: { ok: true, plan: { ok: true } }, text: '', stderr: '' } : { code: 0, json: { ok: true, event: 'result', done: true }, text: '', stderr: '' }; } });
  assert.deepEqual(await control.planModelsMove('/old', '/new'), { ok: true });
  assert.deepEqual(seen[0][0], ['models', 'move', '/new', '--from', '/old', '--plan', '--output', 'json']);
  const result = await control.moveModels('/old', '/new', { installDir: '/i', cancelFile: '/c.cancel' });
  assert.equal(result.done, true);
  assert.deepEqual(seen[1][0], ['models', 'move', '/new', '--from', '/old', '--install-dir', '/i', '--cancel-file', '/c.cancel', '--output', 'jsonl']);
  const failing = createControl({ control: { dir: '/c', python: '/c/py', cli: '/c/strata.py' }, run: async () => ({ code: 5, json: { ok: false, error: { code: 'disk_full', message: 'The drive ran out of space.', details: 'ENOSPC' }, moved: [] }, text: '', stderr: '' }) });
  await assert.rejects(failing.moveModels('/o', '/n'), e => e.code === 'disk_full' && e.details === 'ENOSPC' && /ran out of space/.test(e.message));
});

test('an unfinished move is found at the next start, and Finish resumes it as a job that changes the folder only when done', async t => {
  const f = fixture(t);
  f.control.interruptedModelsMove = async from => { f.calls.push(['status', from]); return { state: 'failed', from: f.oldFolder, to: f.newFolder, bytes: 74e9 }; };
  assert.equal((await f.native.interruptedMove()).to, f.newFolder);
  assert.deepEqual(f.calls[0], ['status', f.oldFolder]);
  const job = await f.native.finishInterruptedMove({ stopRunning: true });
  const done = await finish(f.native, job.id);
  assert.equal(done.state, 'done');
  assert.equal(f.calls.find(c => c[0] === 'resume')[1], f.oldFolder);
  assert.equal(f.native.state().downloadDirectory, f.newFolder);
});

test('Leave it keeps the old folder and asks Control to remove the partial copies; nothing interrupted means nothing to do', async t => {
  const f = fixture(t);
  assert.equal(await f.native.interruptedMove(), null);
  await assert.rejects(f.native.leaveInterruptedMove(), /No move was interrupted/);
  f.control.interruptedModelsMove = async () => ({ state: 'running', from: f.oldFolder, to: f.newFolder, bytes: 1 });
  const result = await f.native.leaveInterruptedMove();
  assert.equal(result.folder, f.oldFolder);
  assert.deepEqual(f.calls.find(c => c[0] === 'leave').slice(0, 2), ['leave', f.oldFolder]);
  assert.equal(f.native.state().downloadDirectory, f.oldFolder);
});

test('the Control command lines for status, resume and leave', async () => {
  const seen = [];
  const control = createControl({ control: { dir: '/c', python: '/c/py', cli: '/c/strata.py' }, run: async args => { seen.push(args); return args.includes('--status') ? { code: 0, json: { ok: true, interrupted: { from: '/o', to: '/n' } }, text: '', stderr: '' } : args.includes('--leave') ? { code: 0, json: { ok: true, removed: 0 }, text: '', stderr: '' } : { code: 0, json: { ok: true, event: 'result', done: true }, text: '', stderr: '' }; } });
  assert.equal((await control.interruptedModelsMove('/o')).to, '/n');
  await control.resumeModelsMove('/o', { installDir: '/i', cancelFile: '/c.cancel' });
  await control.leaveModelsMove('/o', { installDir: '/i' });
  assert.deepEqual(seen[0], ['models', 'move', '--status', '--from', '/o', '--output', 'json']);
  assert.deepEqual(seen[1], ['models', 'move', '--resume', '--from', '/o', '--install-dir', '/i', '--cancel-file', '/c.cancel', '--output', 'jsonl']);
  assert.deepEqual(seen[2], ['models', 'move', '--leave', '--from', '/o', '--install-dir', '/i', '--output', 'json']);
});

test('the models folder strings exist in all 8 languages with the same placeholders', () => {
  const read = lang => JSON.parse(readFileSync(new URL(`../src/locales/${lang}.json`, import.meta.url), 'utf8'));
  const en = read('en');
  const ours = Object.keys(en).filter(k => Number(k.slice(1)) >= 8339 && Number(k.slice(1)) <= 8422);
  assert.equal(ours.length, 84);
  for (const lang of ['de', 'es', 'ja', 'ko', 'pt', 'ru', 'zh']) {
    const text = read(lang);
    for (const key of ours) {
      assert.ok(text[key], `${lang} ${key}`);
      assert.deepEqual((text[key].match(/\{\d\}/g) ?? []).sort(), (en[key].match(/\{\d\}/g) ?? []).sort(), `${lang} ${key} placeholders`);
    }
  }
  // every string the dialog shows through tr() is in the catalogue
  const source = readFileSync(new URL('../src/components/ModelsFolderDialog.tsx', import.meta.url), 'utf8');
  const values = new Set(Object.values(en));
  for (const text of ['Models folder', 'Change folder', 'Move my models', 'Use both folders', 'Start empty', 'Stop moving', 'Your models are moved', 'Moving stopped']) {
    assert.ok(source.includes(text) && values.has(text), text);
  }
});
