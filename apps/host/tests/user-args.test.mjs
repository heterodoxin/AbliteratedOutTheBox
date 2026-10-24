/**
 * The person's own engine settings survive a load (#3215 #3270 #3216): a --vram-reserve-mib of 0, the Expert setting "GPU memory for other
 * apps", and engine arguments the app does not know (the control vector options). Nothing here starts an engine; Control is a fake that
 * records the run file the host saved and asked it to start.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { resourceSettings, explicitReserve } from '../server/native-resources.mjs';
import { capture, merge, groups } from '../server/native-user-args.mjs';

const ARGS = ['--pack', '/p', '--expert-cache', 'auto', '--spec', '4', '--max-context', '40000', '--kv', 'int8'];
const CVEC = ['--control-vector-scaled', 'C:/vectors/mine.gguf:1.0', '--control-vector-layer-range', '4', '44', '--cvec-mode', 'project', '--cvec-dir', 'per-layer'];

function fixture(config = {}, { gpus = [{ index: 0, vram_gb: 12 }] } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-userargs-test-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const path = join(install, 'strata-q2_0.json');
  writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ARGS, model_name: 'Qwen q2_0', port: 8080, ...config }));
  const saved = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.2.0',
    async installedModels() { return { install_dir: install, models: [{ id: 'q2_0', name: 'Qwen q2_0', source_config: path, component_status: 'present', context_tokens: 40000 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.2.0', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const native = createNativeControl({ control, hardwareProbe: () => gpus, directory: data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [] });
  return { native, saved, path, read: () => JSON.parse(readFileSync(path, 'utf8')), write: c => writeFileSync(path, JSON.stringify(c)), cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
const reserveOf = args => args[args.indexOf('--vram-reserve-mib') + 1];
const load = async (f, args = {}) => { const job = await finish(f.native, f.native.operation('load', { model: 'q2_0', vision: 'no', ...args }).id); assert.equal(job.state, 'done', job.error); return f.saved.at(-1); };

test('the rule: groups, merge and capture', () => {
  assert.deepEqual(groups(['--a', '1', '--b', '--c', 'x', 'y']), [['--a', '1'], ['--b'], ['--c', 'x', 'y']]);
  assert.deepEqual(merge(['--a', '1', '--b', '2'], ['--a', '9']), ['--b', '2', '--a', '9']);
  const written = [...ARGS, '--vram-reserve-mib', '1536'];
  assert.deepEqual(capture({ args: [...written, ...CVEC], args_written: written }), CVEC);
  assert.deepEqual(capture({ args: written, args_written: written }), []);
  assert.deepEqual(capture({ args: written.map(a => (a === '1536' ? '0' : a)), args_written: written }), ['--vram-reserve-mib', '0']);
  assert.deepEqual(capture({ args: ['--max-context', '1'], args_written: ['--max-context', '2'] }), [], 'the context is the app\'s own choice');
});

test('explicitReserve: any number the person set counts (0 too); the engine default, the app\'s own value and beta leftovers do not', () => {
  assert.equal(explicitReserve({ args: ['--vram-reserve-mib', '0'] }), 0);
  assert.equal(explicitReserve({ args: [], vram_reserve_mib: 0 }), 0);
  assert.equal(explicitReserve({ args: [], user_args: ['--vram-reserve-mib', '256'], vram_reserve_mib: 3000 }), 256);
  assert.equal(explicitReserve({ args: [], vram_reserve_mib: 3000 }, { vram_reserve_mib: 128 }), 128, 'the Expert setting of this load');
  assert.equal(explicitReserve({ args: ['--vram-reserve-mib', '700'] }), null, 'setup writes 700 for pictures: not a choice');
  assert.equal(explicitReserve({ args: [] }), null);
  // what the app wrote last is not the person's: the value and the note that it was the app's
  assert.equal(explicitReserve({ args: ['--vram-reserve-mib', '3276'], vram_reserve_mib: 3276, ui_reserve_applied: 3276 }), null);
  assert.equal(explicitReserve({ args: ['--vram-reserve-mib', '1024'], vram_reserve_mib: 1024, ui_reserve_applied: 3276 }), 1024, 'changed since: the person\'s');
  // a 0.2.0-beta file: the app wrote the same number in both places, without a note
  const beta = { args: ['--vram-reserve-mib', '1536'], vram_reserve_mib: 1536, ui_resource_baseline: { workers: 0, reserve: 700 } };
  assert.equal(explicitReserve(beta), null);
  assert.equal(explicitReserve({ ...beta, args: ['--vram-reserve-mib', '0'] }), 0, 'a different number in args is the person\'s');
});

test('3215: a reserve of 0 is kept on both resource profiles; the safety default stays where nothing was asked', () => {
  const display = { cpu: { cores: 12 }, ram_gb: 64, gpus: [{ index: 0, vram_gb: 12, drives_display: true }] };
  const headless = { cpu: { cores: 12 }, ram_gb: 64, gpus: [{ index: 0, vram_gb: 12, drives_display: false }] };
  const unknown = { cpu: { cores: 12 }, ram_gb: 64, gpus: [{ index: 0, vram_gb: 12 }] };
  const zero = { gpu: 0, args: ['--vram-reserve-mib', '0'] };
  for (const profile of ['work', 'performance']) for (const hw of [display, headless, unknown]) {
    const r = resourceSettings(zero, hw, profile);
    assert.equal(reserveOf(r.args), '0', `${profile}`);
    assert.equal(r.vram_reserve_mib, 0);
    assert.equal(r.ui_reserve_applied, undefined, 'the app did not choose it');
  }
  assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: true } }, display, 'performance').vram_reserve_mib, 2048, 'a display card with pictures: the safety default');
  assert.equal(resourceSettings({ gpu: 0, args: [] }, display, 'performance').vram_reserve_mib, 1536);
  assert.equal(resourceSettings({ gpu: 0, args: [] }, unknown, 'performance').vram_reserve_mib, 1536, 'not known to be headless: the safety default');
  assert.equal(resourceSettings({ gpu: 0, args: [] }, display, 'work').vram_reserve_mib, 1536, 'a fifth of 12 GB (2457) is capped at 1536 on a card of 16 GB or less (0.2.3 tuning)');
  // a card that drives no display: 512 MiB (700 with pictures), and "Leave 20% for my work" has no desktop to keep room for
  for (const profile of ['work', 'performance']) {
    assert.equal(resourceSettings({ gpu: 0, args: [] }, headless, profile).vram_reserve_mib, 512, profile);
    assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: true } }, headless, profile).vram_reserve_mib, 700, `${profile} with pictures`);
  }
  // two cards, one of them on the desktop
  const mixed = { ...display, gpus: [{ index: 0, vram_gb: 12, drives_display: false }, { index: 1, vram_gb: 12, drives_display: true }] };
  assert.equal(resourceSettings({ gpu: [0, 1], args: [] }, mixed, 'performance').vram_reserve_mib, 1536);
});

test('3215: the app load keeps --vram-reserve-mib 0 typed into the file, on every load', async () => {
  const f = fixture({ args: [...ARGS, '--vram-reserve-mib', '0'] });
  try {
    await f.native.refresh(true);
    for (let i = 0; i < 3; i++) {
      const run = await load(f, { resourceProfile: i === 1 ? 'performance' : 'work' });
      assert.equal(reserveOf(run.args), '0', `load ${i + 1}`);
      assert.equal(run.args.filter(a => a === '--vram-reserve-mib').length, 1);
      const file = f.read();
      assert.equal(reserveOf(file.args), '0', 'the file keeps it too');
      assert.deepEqual(file.user_args, ['--vram-reserve-mib', '0']);
    }
  } finally { await f.native.close(); f.cleanup(); }
});

test('3270: the Expert setting "GPU memory for other apps" is kept across loads, a lowered value is not reset, and clearing it brings the default back', async () => {
  const f = fixture({ args: [...ARGS, '--vram-reserve-mib', '2048'], args_written: [...ARGS, '--vram-reserve-mib', '2048'], vram_reserve_mib: 2048 });
  try {
    await f.native.refresh(true);
    await load(f, { resourceProfile: 'work' });                                    // the app's own load first: its numbers are in the file now
    f.native.save('q2_0', { values: { vram_reserve_mib: 1024 } });
    for (let i = 0; i < 2; i++) {
      const run = await load(f, { resourceProfile: 'work' });
      assert.equal(reserveOf(run.args), '1024', `load ${i + 1}`);
      assert.equal(f.read().vram_reserve_mib, 1024);
      assert.equal(reserveOf(f.read().args), '1024');
    }
    f.native.save('q2_0', { values: { vram_reserve_mib: 300 } });
    assert.equal(reserveOf((await load(f, { resourceProfile: 'performance' })).args), '300');
    // edited by hand in both places of the file
    const hand = f.read(); hand.vram_reserve_mib = 900; hand.args = hand.args.map(a => (a === '300' ? '900' : a)); f.write(hand);
    assert.equal(reserveOf((await load(f, { resourceProfile: 'work' })).args), '900');
    // the value of this load alone
    assert.equal(reserveOf((await load(f, { resourceProfile: 'work', values: { vram_reserve_mib: 128 } })).args), '128');
    f.native.save('q2_0', { values: { vram_reserve_mib: null } });
    const cleared = f.read();
    assert.equal('vram_reserve_mib' in cleared, false);
    assert.equal(cleared.args.includes('--vram-reserve-mib'), false);
    assert.equal(cleared.user_args, undefined);
    assert.equal(reserveOf((await load(f, { resourceProfile: 'performance' })).args), '1536', 'nothing asked: the safety default');
  } finally { await f.native.close(); f.cleanup(); }
});

test('3270: a 0.2.0-beta file (the app wrote 1536 itself) is not taken for the person\'s choice, but a number they changed is', async () => {
  const f = fixture({ args: [...ARGS, '--vram-reserve-mib', '1536'], vram_reserve_mib: 1536, ui_resource_profile: 'work', ui_resource_baseline: { workers: 0, reserve: 700, ramBudget: false, ramHeadroom: false } });
  try {
    await f.native.refresh(true);
    assert.equal(reserveOf((await load(f, { resourceProfile: 'performance' })).args), '1536');
    const edited = f.read(); edited.args = edited.args.map(a => (a === '1536' ? '1024' : a)); f.write(edited);
    assert.equal(reserveOf((await load(f, { resourceProfile: 'performance' })).args), '1024');
  } finally { await f.native.close(); f.cleanup(); }
});

test('3216: engine arguments the app does not know (control vectors) survive a load and a setup run that rewrites "args"', async () => {
  // setup wrote ARGS (and a copy of them); the person typed the control vector options into the JSON editor
  const f = fixture({ args: [...ARGS, ...CVEC], args_written: ARGS });
  try {
    await f.native.refresh(true);
    const run = await load(f, { resourceProfile: 'work' });
    assert.deepEqual(run.args.slice(run.args.indexOf('--control-vector-scaled'), run.args.indexOf('--control-vector-scaled') + 9), CVEC);
    const file = f.read();
    assert.deepEqual(file.user_args, CVEC);
    assert.ok(file.args.join(' ').includes(CVEC.join(' ')), 'the file keeps them in "args" too');
    // a setup run rewrites "args" from its own choices and carries user_args along (setup.py carry_user_args)
    f.write({ ...file, args: [...ARGS, '--max-context', '65536'], args_written: undefined });
    const again = await load(f, { resourceProfile: 'work' });
    assert.ok(again.args.join(' ').includes(CVEC.join(' ')), 'still there after the rewrite');
    assert.equal(again.args.filter(a => a === '--cvec-mode').length, 1);
    assert.deepEqual(f.read().user_args, CVEC);
    // removing them from "user_args" and "args" by hand takes them off for good
    const bare = f.read(); bare.user_args = undefined; bare.args = [...ARGS]; bare.args_written = [...ARGS]; f.write(bare);
    assert.ok(!(await load(f, { resourceProfile: 'work' })).args.includes('--cvec-mode'));
  } finally { await f.native.close(); f.cleanup(); }
});

test('3216: without any resource preference the load keeps them too, and the person\'s value beats a generated one for the same option', async () => {
  const f = fixture({ args: ARGS.map(a => (a === '4' ? '2' : a)), args_written: ARGS });
  try {
    await f.native.refresh(true);
    const run = await load(f);                                                      // no resourceProfile: the plain path
    assert.equal(run.args.filter(a => a === '--spec').length, 1);
    assert.equal(run.args[run.args.indexOf('--spec') + 1], '2', 'a hand-changed --spec is the person\'s');
    assert.deepEqual(f.read().user_args, ['--spec', '2']);
  } finally { await f.native.close(); f.cleanup(); }
});
