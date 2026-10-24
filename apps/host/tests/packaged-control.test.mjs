import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { existsSync } from 'node:fs';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { bundledPython, componentsDirectory, controlEnvironment, createControl, findControl, runCli } from '../server/control-runtime.mjs';
import { resolvePython } from '../server/python-resolver.mjs';
import { createNativeControl } from '../server/native-control.mjs';

const isWin = process.platform === 'win32';

function fakeControlDir() {
  const dir = mkdtempSync(join(tmpdir(), 'strata-control-dir-'));
  mkdirSync(join(dir, 'python', ...(isWin ? [] : ['bin'])), { recursive: true });
  mkdirSync(join(dir, 'strata_control'), { recursive: true });
  mkdirSync(join(dir, 'serve'), { recursive: true });
  writeFileSync(bundledPython(dir), 'x');
  writeFileSync(join(dir, 'strata.py'), '# cli');
  writeFileSync(join(dir, 'serve', 'server.py'), 'port\nparallel\n/v1/responses');
  writeFileSync(join(dir, 'BUNDLE.json'), JSON.stringify({ version: '0.1.45', python: '3.12.15' }));
  return dir;
}

test('the Control component is found by STRATA_CONTROL_DIR or beside the host, and only if it is whole', () => {
  const dir = fakeControlDir();
  try {
    const found = findControl({ env: { STRATA_CONTROL_DIR: dir }, hostDir: join(dir, 'nowhere') });
    assert.equal(found.dir, dir);
    assert.equal(found.python, bundledPython(dir));
    assert.equal(found.bundle.python, '3.12.15');
    // beside the host: <install>/host + <install>/control
    const install = mkdtempSync(join(tmpdir(), 'strata-install-'));
    try {
      mkdirSync(join(install, 'host', 'server'), { recursive: true });
      assert.equal(findControl({ env: {}, hostDir: join(install, 'host') }), null, 'nothing beside the host yet');
      mkdirSync(join(install, 'control', 'python', ...(isWin ? [] : ['bin'])), { recursive: true });
      mkdirSync(join(install, 'control', 'strata_control'));
      writeFileSync(bundledPython(join(install, 'control')), 'x');
      writeFileSync(join(install, 'control', 'strata.py'), '#');
      assert.equal(findControl({ env: {}, hostDir: join(install, 'host') }).dir, join(install, 'control'));
    } finally { rmSync(install, { recursive: true, force: true }); }
    rmSync(join(dir, 'strata.py'));
    assert.equal(findControl({ env: { STRATA_CONTROL_DIR: dir }, hostDir: join(dir, 'x') }), null, 'a half component is not used');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("Control runs on its own interpreter with nothing from the user's Python and acquiring forbidden", () => {
  const control = { dir: '/c', python: '/c/python/python3', cli: '/c/strata.py' };
  const env = controlEnvironment(control, { env: { PYTHONHOME: 'x', PYTHONPATH: 'y', VIRTUAL_ENV: 'z', STRATA_PYTHON: 'q', PATH: 'p', STRATA_API_KEY: 'secret' }, components: '/components' });
  assert.equal(env.PYTHONHOME, undefined);
  assert.equal(env.PYTHONPATH, undefined);
  assert.equal(env.VIRTUAL_ENV, undefined);
  assert.equal(env.STRATA_PYTHON, undefined);
  assert.equal(env.PYTHONNOUSERSITE, '1');
  assert.equal(env.STRATA_NO_ACQUIRE, '1');
  assert.equal(env.STRATA_COMPONENTS_DIR, '/components');
  assert.equal(env.STRATA_API_KEY, undefined, 'an inherited inference key never reaches the engine (the run config carries its own)');
  assert.equal(env.PATH, 'p');
  // the resolver prefers the bundled interpreter over py -3 / PATH / a venv
  const python = resolvePython({ repo: '/repo', control, env: {}, exists: () => true, starts: () => { throw new Error('must not probe'); }, useCache: false });
  assert.deepEqual([python.command, python.source], ['/c/python/python3', 'bundled']);
  assert.equal(resolvePython({ control, env: { STRATA_PYTHON: '/mine' }, useCache: false }).command, '/mine', 'an explicit choice still wins');
  assert.equal(componentsDirectory({ env: { STRATA_COMPONENTS_DIR: '/x' } }), '/x');
  assert.match(componentsDirectory({ env: { LOCALAPPDATA: 'C:\\L' }, platform: 'win32', home: 'C:\\H' }), /Strata[\\/]components$/);
});

test('runCli passes the arguments untouched (no shell) and reads JSON', async () => {
  const seen = [];
  const spawnProcess = (command, args, options) => {
    seen.push({ command, args, options });
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.kill = () => {};
    queueMicrotask(() => { child.stdout.emit('data', '{"ok": true, "n": 1}\n'); child.emit('close', 0); });
    return child;
  };
  const control = { dir: '/c', python: '/c/python/python3', cli: '/c/strata.py' };
  const result = await runCli(control, ['models', 'list', '--installed', '--output', 'json'], { spawnProcess, env: {} });
  assert.deepEqual(result.json, { ok: true, n: 1 });
  assert.deepEqual(seen[0].args, ['/c/strata.py', 'models', 'list', '--installed', '--output', 'json']);
  assert.equal(seen[0].options.shell, undefined);
  assert.equal(seen[0].options.windowsHide, true);
  assert.equal(seen[0].options.env.STRATA_NO_ACQUIRE, '1');
});

// ---------------------------------------------------------------------------------- native-control in packaged mode

function packagedFixture({ engineUsable = true, updater = null, repair = false } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-packaged-test-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  const earlierInstallEnv = process.env.STRATA_INSTALL_DIR;
  process.env.STRATA_INSTALL_DIR = install;      // the app's own install folder (the host always tells Control which one; it never falls back to an older CLI folder)
  const configs = ['iq2_xs', 'iq3_s'].map(id => {
    const path = join(install, `strata-${id}.json`);
    writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--max-context', '32768'], model_name: `Qwen ${id}`, port: 8080 }));
    return { id, path };
  });
  const calls = [];
  const saved = [];
  const spawned = [];
  const rpc = [];
  let startFailure = null, startFailureLeft = Infinity;
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', 'python.exe') }, version: '0.1.45',
    async installedModels(options) { calls.push(['installedModels', options?.installDir ?? null]); return { install_dir: install, models: configs.map(c => ({ id: c.id, name: `Qwen ${c.id}`, source_config: c.path, component_status: 'present', context_tokens: 32768, backend: 'cuda' })) }; },
    async engine() {
      calls.push(['engine']);
      return engineUsable
        ? { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.1.45', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\engine\\strata.exe', root: 'E:\\engine' } }
        : { usable: false, notice: 'No engine for this PC was found in E:\\c: the component folder holds no engine package. The app never downloads an engine while it starts or serves.', backend: 'cuda', engine: null, components_dir: 'E:\\c' };
    },
    async importPlan(path) { calls.push(['importPlan', path]); return { file_set_status: 'plausible', candidate_selection: { family: 'qwen', model: 'IQ3_S' }, file_count: 2, total_bytes: 1e9, issues: [] }; },
    async saveProfile(name, path) { calls.push(['saveProfile', name, basename(path)]); saved.push(JSON.parse(readFileSync(path, 'utf8'))); return {}; },
    async start(name, options) { calls.push(['start', name, options?.waitSeconds]); if (startFailure && startFailureLeft-- > 0) { const e = new Error(startFailure.message); e.output = startFailure.output; throw e; } return { ok: true, state: 'ready' }; },
    async stop(name) { calls.push(['stop', name]); return { ok: true, state: 'stopped' }; },
    ...(repair ? { async repairAccess() { calls.push(['repairAccess']); return { ok: true }; } } : {}),
    ...(updater ? { async updateCheck() { calls.push(['updateCheck']); return updater.report; }, async update(options) { calls.push(['update', options.engine, options.control]); return updater.update(options); } } : {}),
  };
  const requests = [];
  const fetcher = async url => { calls.push(['fetch', url.replace(/:\d+/, ':PORT')]); return Response.json({ status: 'ok' }); };
  // The only process the host may start is Control's installer (its bundled tools/strata_mcp.py on Control's Python); the fake
  // answers like it: the catalog, the hardware with the GPUs, and an install plan or job.
  const catalog = { families: [{ family: 'qwen', title: 'Qwen3.8-Flash-Next', about: 'x', images: true, experimental: false, sizes: [
    { id: 'q2_0', model: 'Q2_0', download_gb: 66.4, ram_needed_gb: 48, about: 'fast', on_this_pc: 'fits', installed: false, images: true, experimental: false },
    { id: 'iq3_xxs', model: 'IQ3_XXS', download_gb: 75.8, ram_needed_gb: 60, about: 'balanced', on_this_pc: 'fits', installed: false, images: true, experimental: false }] }],
    contexts: [8192, 32768], recommendation: { family: 'qwen', model: 'IQ3_XXS', context: 32768 }, context_note: '' };
  const hardware = { ram_gb: 63, cpu: { name: 'cpu', cores: 12 }, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5070', vram_gb: 11.9, usable: true }] };
  let installJob = null;
  const spawnProcess = (command, args, options) => {
    spawned.push({ command, args, options });
    if (!String(args[0]).endsWith('strata_mcp.py')) throw new Error('no process but the installer may be started by the host in packaged mode: ' + command + ' ' + args.join(' '));
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.exitCode = null;
    child.kill = () => { child.exitCode = 0; queueMicrotask(() => child.emit('exit', 0)); return true; };
    child.stdin = { write(line) {
      const message = JSON.parse(line); if (!message.id) return;
      if (message.method === 'tools/call') rpc.push(message.params);
      let result;
      if (message.method === 'initialize') result = { protocolVersion: '2024-11-05' };
      else if (message.method === 'tools/list') result = { tools: ['strata_models', 'strata_status', 'strata_install', 'strata_remove'].map(name => ({ name })) };
      else if (message.params.name === 'strata_models') result = { structuredContent: catalog };
      else if (message.params.name === 'strata_remove') result = { structuredContent: { summary: 'Deleted', plan: { gb: 1.2 }, failed: [] } };
      else if (message.params.name === 'strata_status') result = { structuredContent: { hardware, recommendation: catalog.recommendation, install: installJob, running: [], summary: 'x' } };
      else if (message.params.arguments?.confirm) { installJob = { running: false, exit_code: 0 }; result = { structuredContent: { summary: 'Install started', plan: { download_gb: 66.4 }, install: { running: true, step: 'downloading' } } }; }
      else if (message.params.arguments?.cancel === undefined && message.params.arguments?.confirm === undefined) result = { structuredContent: { summary: 'Install finished', install: { running: false, exit_code: 0 } } };
      else result = { structuredContent: { summary: 'Plan', plan: { title: 'Qwen3.8-Flash-Next Q2_0', download_gb: 66.4, disk_short: null } } };
      queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }) + String.fromCharCode(10))));
    } };
    return child;
  };
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher, requests: () => requests });
  const others = () => spawned.filter(s => !String(s.args[0]).endsWith('strata_mcp.py'));
  return { native, calls, saved, spawned, others, rpc, failStart: (v, times = Infinity) => { startFailure = v; startFailureLeft = times; }, catalog, hardware, install, root, cleanup: () => { if (earlierInstallEnv === undefined) delete process.env.STRATA_INSTALL_DIR; else process.env.STRATA_INSTALL_DIR = earlierInstallEnv; rmSync(root, { recursive: true, force: true }); } };
}
const finish = async (native, id) => { for (let i = 0; i < 400; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('packaged: models come from Control, no python, git or engine process is started by the host', async () => {
  const f = packagedFixture();
  try {
    const state = await f.native.refresh(true);
    assert.equal(state.available, true, state.error);
    assert.deepEqual(state.installed.models.map(m => m.model), ['iq2_xs', 'iq3_s']);
    assert.equal(state.installed.models[0].ready, true);
    assert.equal(state.packaged, true);
    assert.match(state.summary, /engine 0\.1\.45/);
    assert.equal(state.version, '0.1.45');
    assert.equal(f.others().length, 0, 'nothing but the installer is started');
    // the catalog, the GPUs and the recommendation come from the bundled installer, run on Control's own Python
    assert.equal(state.catalog.families[0].sizes.length, 2);
    assert.equal(state.catalog.recommendation.model, 'IQ3_XXS');
    assert.equal(state.hardware.gpus[0].name, 'NVIDIA GeForce RTX 5070');
    assert.equal(state.hardware.ram_gb, 63);
    const installer = f.spawned[0];
    assert.equal(installer.command, join(f.root, 'control', 'python', 'python.exe'));
    assert.deepEqual(installer.args.slice(1), ['--root', f.install, '--src', join(f.root, 'control')]);
    assert.equal(installer.options.env.STRATA_CONTROL_INSTALL, '1');
    assert.equal(installer.options.env.STRATA_NO_ACQUIRE, '1', 'the installer process itself may not acquire; only its explicit install job does');
    assert.equal(installer.options.env.STRATA_INSTALL_DIR, f.install);
    assert.equal(installer.options.env.PYTHONNOUSERSITE, '1');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: load = saved profile + runtime start under Control; unload = runtime stop; two models get two profiles', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const load = async args => { const job = await finish(f.native, f.native.operation('load', args).id); assert.equal(job.state, 'done', job.error); return job; };
    await load({ model: 'iq2_xs' });
    const first = f.native.state().instances[0];
    assert.equal(first.status, 'ready');
    const startCalls = f.calls.filter(c => ['saveProfile', 'start'].includes(c[0]));
    assert.deepEqual(startCalls.map(c => c.slice(0, 2)), [['saveProfile', 'app-1'], ['start', 'app-1']]);
    assert.equal(startCalls[1][2], 300);
    await load({ model: 'iq3_s', multiple: true });
    assert.deepEqual(f.calls.filter(c => c[0] === 'start').map(c => c[1]), ['app-1', 'app-2']);
    assert.equal(f.native.state().instances.length, 2);
    await f.native.unload(first.id);
    assert.deepEqual(f.calls.filter(c => c[0] === 'stop').at(-1), ['stop', 'app-1']);
    assert.equal(f.native.state().instances.length, 1);
    assert.equal(f.others().length, 0, 'the host never spawned anything itself');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: no engine for this PC is the notice, and nothing is started', async () => {
  const f = packagedFixture({ engineUsable: false });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /No engine for this PC/);
    assert.match(job.error, /never downloads an engine/);
    assert.equal(f.calls.some(c => ['saveProfile', 'start'].includes(c[0])), false);
    assert.equal(f.others().length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the user can download and prepare a model: the plan and the install job go to the bundled installer, the engine is not touched', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const plan = await f.native.plan({ family: 'qwen', model: 'Q2_0' });
    assert.equal(plan.plan.title, 'Qwen3.8-Flash-Next Q2_0');
    const planned = f.rpc.at(-1);
    assert.equal(planned.name, 'strata_install');
    assert.equal(planned.arguments.confirm, false, 'a plan changes nothing');
    const job = await finish(f.native, f.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id);
    assert.equal(job.state, 'done', job.error);
    const started = f.rpc.filter(c => c.name === 'strata_install' && c.arguments.confirm);
    assert.equal(started.length, 1);
    assert.equal(started[0].arguments.family, 'qwen');
    assert.equal(started[0].arguments.model, 'Q2_0');
    assert.equal(f.others().length, 0, 'the host starts no pip, git, python of its own or engine');
    assert.equal(f.calls.filter(c => ['saveProfile', 'start'].includes(c[0])).length, 0, 'installing does not start an engine');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the installer works in a folder the app can write, never in the read-only bundle', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    assert.equal(existsSync(f.install), true);
    assert.notEqual(f.native.state().installDir, join(f.root, 'control'));
    assert.equal(f.spawned[0].args.at(-3), f.install);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: update, rollback and calibration are refused with the Terminal notice and start nothing', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    for (const [kind, args] of [['update-engine', {}], ['update-strata', {}], ['rollback', {}], ['calibrate', { model: 'iq3_s' }]]) {
      const job = await finish(f.native, f.native.operation(kind, args).id);
      assert.equal(job.state, 'error', kind);
      assert.match(job.error, /Terminal|calibrates NVIDIA|No saved previous engine/, `${kind}: ${job.error}`);
    }
    assert.equal(f.others().length, 0);
    assert.equal(f.calls.filter(c => ['saveProfile', 'start'].includes(c[0])).length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the install folder can be chosen, and it is what Control is asked about', async () => {
  const f = packagedFixture();
  try {
    const other = join(f.root, 'other');
    mkdirSync(other);
    await f.native.configure({ repository: other });
    const saved = JSON.parse(readFileSync(join(f.root, 'data', 'native-host.json'), 'utf8'));
    assert.equal(saved.installDir.toLowerCase(), other.toLowerCase());
    assert.equal(f.calls.filter(c => c[0] === 'installedModels').at(-1)[1].toLowerCase(), other.toLowerCase());
  } finally { await f.native.close(); f.cleanup(); }
});

test('createControl asks the CLI for the engine, models and runtime and reads the notice of a missing engine', async () => {
  const seen = [];
  const control = createControl({
    control: { dir: '/c', python: '/p', cli: '/c/strata.py', bundle: { version: '9' } },
    run: async args => { seen.push(args.join(' ')); return args[0] === 'components' ? { code: 5, json: { ok: false, usable: false, notice: 'No engine for this PC' }, text: '', stderr: '' } : { code: 0, json: { ok: true, profiles: [] }, text: '', stderr: '' }; },
  });
  assert.equal((await control.engine()).notice, 'No engine for this PC');
  await control.installedModels({ installDir: 'D:\\Strata' });
  await control.saveProfile('app-1', 'C:\\x.json');
  await control.start('app-1');
  await control.stop('app-1');
  assert.deepEqual(seen, [
    'components engine --output json',
    'models list --installed --install-dir D:\\Strata --output json',
    'profile list --output json',
    'profile create app-1 --from-config C:\\x.json --output json',
    `runtime start --profile app-1 --wait 300 --shutdown-timeout 15 --die-with-pid ${process.pid} --output json`,
    'runtime stop --profile app-1 --wait 120 --output json',
  ]);
  assert.equal(createControl({ control: null }), null);
});

test('packaged: the catalog offers no picture reading, an install asks for none by default and for it only when Read pictures was ticked', async () => {
  const f = packagedFixture();
  try {
    const state = await f.native.refresh(true);
    assert.ok(state.catalog.families[0].sizes.every(s => s.images === false));
    assert.ok(state.catalog.families.every(fam => fam.images === false));
    await f.native.plan({ family: 'qwen', model: 'Q2_0' });
    assert.equal(f.rpc.at(-1).arguments.vision, 'no');
    await f.native.plan({ family: 'qwen', model: 'Q2_0', vision: 'yes' });
    assert.equal(f.rpc.at(-1).arguments.vision, 'yes');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: an engine that stops while loading keeps the code and the last engine line for the Details disclosure', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'memory guard: 5.1 GiB free is under the 6 GiB limit' } });
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /^server_exited: memory guard: 5\.1 GiB free/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: a start that died because Windows ran out of locked memory is tried once more with a smaller pin, said in plain words, and the saved settings stay as they were', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'verify: mapped staging allocation failed [pinned-memory-exhausted] the OS gave no more page-locked memory; start again with STRATA_ARENA_PIN_GIB=26 (pin less of the expert arena)' } }, 1);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 2, 'one try and exactly one more');
    assert.match(job.result.summary, /ready, after a second try: Windows would not lock as much memory/);
    assert.match(job.details[0], /pinned-memory-exhausted/);
    assert.match(job.details[1], /STRATA_ARENA_PIN_GIB=26/);
    const instance = f.native.state().instances[0];
    assert.equal(instance.status, 'ready');
    const saved = JSON.parse(readFileSync(join(f.install, 'strata-iq2_xs.json'), 'utf8'));
    assert.equal(saved.env, undefined, 'the safer setting is for this start only; the model config is not changed');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the second try is the last one: a second failure is reported with both, and nothing loops', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'mapped staging allocation failed [pinned-memory-exhausted]' } });
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 2);
    assert.doesNotMatch(job.error, /first stop was/, 'the same stop twice is said once, not repeated');
    assert.match(job.details.join(' '), /Second try/, 'the second try stays in the details');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: retrySafer false keeps the single try', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'mapped staging allocation failed [pinned-memory-exhausted]' } });
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs', retrySafer: false }).id);
    assert.equal(job.state, 'error');
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 1);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: a run config that points into an old Control folder is repaired to this Control (a reinstall or an update moved it)', async () => {
  const f = packagedFixture();
  try {
    const here = join(f.root, 'control');
    mkdirSync(join(here, 'data'), { recursive: true });
    writeFileSync(join(here, 'data', 'expert-profile.bin'), 'profile');
    const path = join(f.install, 'strata-iq2_xs.json');
    const old = 'C:/Users/Alex/AppData/Local/Temp/strata-life-install/control';
    writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', cwd: old, args: ['--pack', 'D:/Strata/packs/x', '--expert-profile', old + '/data/expert-profile.bin', '--max-context', '32768'], model_name: 'x', port: 8080 }));
    await f.native.refresh(true);
    const repaired = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(repaired.cwd.toLowerCase(), here.toLowerCase());
    assert.equal(repaired.args[repaired.args.indexOf('--expert-profile') + 1].toLowerCase(), join(here, 'data', 'expert-profile.bin').toLowerCase());
    assert.equal(repaired.args[1], 'D:/Strata/packs/x', 'paths outside Control are left alone');
    assert.ok(existsSync(path + '.bak'), 'the old config is kept');
    // a config that is already right, or whose new target is not there, is not touched
    const before = readFileSync(path, 'utf8');
    await f.native.refresh(true);
    assert.equal(readFileSync(path, 'utf8'), before);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: deleting a downloaded model unloads it first, then the installer removes its files', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const load = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(load.state, 'done', load.error);
    assert.equal(f.native.state().instances.length, 1);
    const job = await finish(f.native, f.native.operation('remove', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.native.state().instances.length, 0, 'the loaded model was unloaded first');
    assert.deepEqual(f.calls.filter(c => c[0] === 'stop').at(-1), ['stop', 'app-1']);
    const removed = f.rpc.filter(c => c.name === 'strata_remove');
    assert.equal(removed.length, 1);
    assert.deepEqual(removed[0].arguments, { model: 'iq2_xs', confirm: true });
    assert.equal(f.others().length, 0);
    const bad = await finish(f.native, f.native.operation('remove', { model: '../x' }).id);
    assert.equal(bad.state, 'error', 'an id that is not an installed model is refused');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: a local GGUF folder is checked by Control, then installed from where it is (no download)', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const preview = await f.native.localPreview('D:/models/iq3s');
    assert.equal(preview.candidate_selection.model, 'IQ3_S');
    assert.deepEqual(f.calls.find(c => c[0] === 'importPlan'), ['importPlan', 'D:/models/iq3s']);
    await assert.rejects(() => f.native.localPreview('  '), /Choose a folder/);
    await f.native.plan({ family: 'qwen', model: 'IQ3_S', gguf_dir: 'D:/models/iq3s' });
    assert.equal(f.rpc.at(-1).arguments.gguf_dir, 'D:/models/iq3s');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the install job also fetches the engine through Control (merge of the model progress and the engine step); a failed engine keeps the model and says so', async () => {
  const report = { engine: { status: 'install', offered: { version: '0.1.46' }, download_bytes: 5e8 } };
  const f = packagedFixture({ engineUsable: false, updater: { report, async update(options) {
    options.onEvent({ event: 'progress', phase: 'download', percent: 50, done: 2.5e8, total: 5e8 });
    return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.46' } }] };
  } } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.result.install.engine.state, 'done');
    assert.deepEqual(job.result.install.parts.map(p => p.kind), ['model', 'engine']);
    assert.deepEqual(f.calls.filter(c => c[0] === 'update'), [['update', true, false]], 'the engine only, from this install job');
    assert.equal(f.calls.filter(c => ['saveProfile', 'start'].includes(c[0])).length, 0);
  } finally { await f.native.close(); f.cleanup(); }
  const g = packagedFixture({ engineUsable: false, updater: { report, async update() { throw new Error('the download of http://x/e.tar.gz was interrupted after 100 of 200 bytes (ConnectionResetError)'); } } });
  try {
    await g.native.refresh(true);
    const job = await finish(g.native, g.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.match(job.summary, /Your model is ready\. The engine for this PC couldn't be downloaded/);
    assert.equal(job.result.install.engine.state, 'failed');
    assert.doesNotMatch(job.summary, /http|ConnectionReset|bytes/);
  } finally { await g.native.close(); g.cleanup(); }
});

test('packaged: the app default of several requests at once and the memory retry share ONE second try, never two', async () => {
  const f = packagedFixture(); f.hardware.gpus[0].vram_gb = 48;
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'something else went wrong' } }, 1);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 2, 'the failed first start and one second try');
    assert.match(job.result.summary, /after a second try/);
    assert.equal(f.native.state().instances[0].parallel, 1, 'the second try runs one request at a time');
  } finally { await f.native.close(); f.cleanup(); }
  const g = packagedFixture(); g.hardware.gpus[0].vram_gb = 48;
  try {
    await g.native.refresh(true);
    g.failStart({ message: 'server_exited', output: { log_tail: 'mapped staging allocation failed [pinned-memory-exhausted]' } });
    const job = await finish(g.native, g.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.equal(g.calls.filter(c => c[0] === 'start').length, 2, 'memory change and parallel drop ride on the same single try');
  } finally { await g.native.close(); g.cleanup(); }
});

test('packaged: a tune measures the engine a load starts: same start args, slots included, for the same model and profile', async () => {
  for (const vram of [11.9, 48]) {
    const f = packagedFixture(); f.hardware.gpus[0].vram_gb = vram;
    try {
      await f.native.refresh(true);
      const args = { model: 'iq2_xs', resourceProfile: 'work', vision: 'no' };
      await f.native.tuneStatus(args).catch(() => {});      // the fake host cannot run Control's tuner; the config it would measure is written first
      const tuned = JSON.parse(readFileSync(join(f.root, 'data', 'native-tune', 'iq2_xs.json'), 'utf8'));
      const job = await finish(f.native, f.native.operation('load', args).id);
      assert.equal(job.state, 'done', job.error);
      const loaded = { ...f.saved.at(-1) }; delete loaded.port; delete loaded.api_key;
      assert.deepEqual(tuned.args, loaded.args, 'the engine arguments are identical');
      for (const [key, value] of Object.entries(tuned)) assert.deepEqual(loaded[key], value, `${key} is the same in the tune and the load`);   // a load adds only its own identity (aliases, model name, key, port)
      if (vram > 40) assert.ok(tuned.parallel > 1, 'a big card gets the app default of several slots in the tune too');
    } finally { await f.native.close(); f.cleanup(); }
  }
});

test('packaged: a profile an elevated run locked (PermissionError) is repaired once and the start is tried again; without repair it is a plain message, not a memory retry', async () => {
  const f = packagedFixture({ repair: true });
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'cannot read profile (PermissionError)' }, 1);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.filter(c => c[0] === 'repairAccess').length, 1);
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 2);
  } finally { await f.native.close(); f.cleanup(); }
  const g = packagedFixture({ repair: true });
  try {
    await g.native.refresh(true);
    g.failStart({ message: 'cannot read profile (PermissionError)' });
    const job = await finish(g.native, g.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.equal(g.calls.filter(c => c[0] === 'start').length, 2, 'one repair try, no memory retry on top');
    assert.match(job.error, /PermissionError/);
  } finally { await g.native.close(); g.cleanup(); }
});

test('a GGUF folder Control refuses (exit 5, ok:false) still comes back as a checked plan with its reasons, so the page can say it in words', async () => {
  const plan = { file_set_status: 'invalid_or_incomplete', candidate_selection: null, issues: ['x.gguf: GGUF tensor or metadata count exceeds the inspection limit', 'unrecognized_or_unsupported_model_variant'] };
  const control = createControl({ control: { dir: '/c', python: '/p', cli: '/c/strata.py', bundle: { version: '9' } }, run: async () => ({ code: 5, json: { ok: false, plan }, text: '', stderr: '' }) });
  assert.deepEqual(await control.importPlan('/some/folder'), plan);
  const broken = createControl({ control: { dir: '/c', python: '/p', cli: '/c/strata.py', bundle: { version: '9' } }, run: async () => ({ code: 1, json: null, text: '', stderr: 'boom' }) });
  await assert.rejects(() => broken.importPlan('/some/folder'), /boom/);
});

test("preparing an adopted GGUF folder again never takes the folder's parent for the data folder (it is the person's, not Strata's)", async () => {
  const f = packagedFixture();
  try {
    // The model is already adopted: its run config points at a GGUF in someone's own ".../models/IQ3_S" folder.
    const own = join(f.root, 'own');
    const configPath = join(f.install, 'strata-iq3_s.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8'));
    writeFileSync(configPath, JSON.stringify({ ...config, args: [...config.args, '--native', join(own, 'models', 'IQ3_S', 'a-00001-of-00002.gguf')] }));
    f.catalog.families[0].sizes.push({ id: 'iq3_s', model: 'IQ3_S', download_gb: 83, ram_needed_gb: 62, about: 'x', on_this_pc: 'fits', installed: true, images: true, experimental: false });
    await f.native.refresh(true);
    await f.native.plan({ family: 'qwen', model: 'IQ3_S', gguf_dir: join(own, 'models', 'IQ3_S') });
    const sent = f.rpc.at(-1).arguments;
    assert.notEqual(sent.data_dir, own, "the person's own folder is not used for the packs");
    assert.equal(sent.gguf_dir, join(own, 'models', 'IQ3_S'));
  } finally { await f.native.close(); f.cleanup(); }
});

test('the app never adopts an older command-line install: Control is always told the app\'s own folder, and has its own settings file', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const asked = f.calls.filter(c => c[0] === 'installedModels').map(c => c[1]);
    assert.ok(asked.length > 0 && asked.every(dir => dir === f.install || (dir && dir.toLowerCase() === f.install.toLowerCase())), `Control was asked about ${JSON.stringify(asked)}`);
  } finally { f.cleanup(); }
  const env = controlEnvironment({ dir: '/c' }, { env: { APPDATA: '/old/appdata' }, components: join(tmpdir(), 'Strata', 'components') });
  assert.equal(env.STRATA_SETTINGS_FILE, join(tmpdir(), 'Strata', 'app-settings.json'));
});

test("onboarding 'use your own GGUF' (#3507, #3424): the models folder the person picked holds the GGUF files, so preparing them never takes it for the data folder", async () => {
  const f = packagedFixture();
  try {
    const own = join(f.root, 'Strata-data', 'models', 'unsloth-UD-Q4_K_XL');
    mkdirSync(own, { recursive: true });
    await f.native.refresh(true);
    f.native.setDownloadDirectory(own);                        // what the first-run folder chooser wrote
    await f.native.plan({ family: 'qwen', model: 'IQ3_S', gguf_dir: own });
    const sent = f.rpc.at(-1).arguments;
    assert.equal(sent.gguf_dir, own);
    assert.notEqual(sent.data_dir, own, 'the folder of GGUF files is not the data folder: the packs would be downloaded into it');
    // a folder of GGUF files named like a data folder's models/<tag> hands its Strata folder to the installer
    assert.ok(sent.data_dir === undefined || sent.data_dir === join(f.root, 'Strata-data'), `data_dir was ${sent.data_dir}`);
  } finally { await f.native.close(); f.cleanup(); }
});
