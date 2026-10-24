/** 0.2.1, group 2: choosing graphics cards (#3203 #3195 #3256 #3221 #3211). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { build, transform } from 'esbuild';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

const fit = await transform(await readFile(new URL('../src/lib/native-fit.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { runCards } = await import(`data:text/javascript;base64,${Buffer.from(fit.code).toString('base64')}`);
const native = await build({ entryPoints: ['src/lib/native.ts'], bundle: true, platform: 'node', format: 'esm', write: false, alias: { react: fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url)) }, logLevel: 'silent' });

const card = (index, name, vram, extra = {}) => ({ index, name, vram_gb: vram, usable: true, ...extra });

test('#3195: the cards to choose from are the ones the engine drives; the other make and the iGPU are not listed', () => {
  // two RTX 3090 and an AMD chip that the report numbers 0 again (the same number as the first 3090)
  const gpus = [card(0, 'NVIDIA GeForce RTX 3090', 24, { vendor: 'nvidia' }), card(1, 'NVIDIA GeForce RTX 3090', 24, { vendor: 'nvidia' }),
    card(0, 'AMD Radeon (gfx1036)', 2, { vendor: 'amd', usable: false })];
  const shown = runCards(gpus, 'nvidia');
  assert.deepEqual(shown.map(g => g.index), [0, 1], 'no second card 0');
  assert.equal(new Set(shown.map(g => g.index)).size, shown.length, 'every number once, so one tick never moves another');
  // even when the chip is flagged usable, the engine's make decides
  assert.deepEqual(runCards([...gpus.slice(0, 2), card(0, 'AMD Radeon 760M', 8, { vendor: 'amd' })], 'nvidia').map(g => g.name), ['NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3090']);
});

test('#3195: a chip that shares the system memory is left out while a card of its own exists, and is the only choice when it is alone', () => {
  const igpu = card(0, 'AMD Radeon 8060S', 64, { vendor: 'amd', uma: true });
  const real = card(1, 'AMD Radeon RX 9070 XT', 16, { vendor: 'amd' });
  assert.deepEqual(runCards([igpu, real], 'amd').map(g => g.index), [1]);
  assert.deepEqual(runCards([igpu], 'amd').map(g => g.index), [0]);
});

test('#3203: a card the engine cannot use (usable false) is never offered', () => {
  const gpus = [card(0, 'AMD Radeon 760M', 8, { vendor: 'amd', usable: false }), card(1, 'AMD Radeon RX 6800 XT', 16, { vendor: 'amd', usable: false })];
  assert.deepEqual(runCards(gpus, 'amd'), []);
  assert.deepEqual(runCards(null, 'amd'), []);
});

test('#3256: the model settings file keeps its GPU order when the app only suggests cards; a card picked in the screen wins', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const load = async args => { const done = await finishJob(f.control, f.control.operation('load', args).id); assert.equal(done.state, 'done', done.error); return f.control.resolveInstance(done.result.instanceId); };
    // the fixture's file says "gpu": 0; the screen suggests card 1
    const suggested = await load({ model: 'iq2_xs', gpus: [1], gpusFromRecommendation: true });
    assert.deepEqual(suggested.gpus, [0], 'the file names its card, so the suggestion does not replace it');
    await f.control.unload(suggested.id);
    const picked = await load({ model: 'iq2_xs', gpus: [1] });
    assert.deepEqual(picked.gpus, [1], 'a pick in the screen is the person\'s word');
  } finally { await f.cleanup(); }
});

test('#3256: with no GPU in the file the suggested cards apply, in the order given', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const { readFileSync, writeFileSync } = await import('node:fs');
    const { join } = await import('node:path');
    const path = join(f.repo, 'strata-iq2_xs.json');
    const config = JSON.parse(readFileSync(path, 'utf8')); delete config.gpu; writeFileSync(path, JSON.stringify(config));
    await f.control.refresh(true);
    const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs', gpus: [1, 0], gpusFromRecommendation: true }).id);
    assert.equal(done.state, 'done', done.error);
    assert.deepEqual(f.control.resolveInstance(done.result.instanceId).gpus, [1, 0]);
  } finally { await f.cleanup(); }
});

test('#3203: one chosen card is sent to the installer as "gpu", never as "gpus" (it asks for two or more there)', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    await f.control.plan({ family: 'qwen', model: 'IQ2_XS', context: 131072, gpus: '0' });
    const call = f.rpcCalls.filter(c => c.name === 'strata_install').pop();
    assert.equal(call.arguments.gpu, 0);
    assert.equal('gpus' in call.arguments && call.arguments.gpus !== undefined, false);
    await f.control.plan({ family: 'qwen', model: 'IQ2_XS', context: 131072, gpus: '0,1' });
    const two = f.rpcCalls.filter(c => c.name === 'strata_install').pop();
    assert.equal(two.arguments.gpus, '0,1');
  } finally { await f.cleanup(); }
});

test('#3211: automatic placement names no card in the config; the cards the engine reports are the model\'s assigned cards', async () => {
  const mod = await import(`data:text/javascript;base64,${Buffer.from(native.outputFiles[0].text).toString('base64')}`);
  const hardware = { gpus: [{ id: 'nvidia:0', index: 0, vendor: 'nvidia' }, { id: 'nvidia:1', index: 1, vendor: 'nvidia' }] };
  const instance = (gpus, reported) => ({ id: 'a', apiModelId: 'a', model: 'm', label: 'M', status: 'ready', loadedAt: 'x', phase: 'idle', gpus, metrics: reported ? { hardware: { gpus: reported } } : null });
  const routed = (inst) => mod.nativeRouting({ instances: [inst], defaultId: 'a' }, hardware).instances[0].gpuIds;
  assert.deepEqual(routed(instance([1], null)), ['nvidia:1'], 'a named card stays as it was');
  assert.deepEqual(routed(instance([], [{ index: 0 }])), ['nvidia:0'], 'Automatic: the card the engine reports');
  assert.deepEqual(routed(instance([], null)), [], 'nothing known: no assigned cards, so the Monitor shows all');
});

test('#3211 #3221: the Monitor opens on the model\'s assigned cards and the dropdown rows grow with their text', async () => {
  const monitor = await readFile(new URL('../src/screens/Monitor.tsx', import.meta.url), 'utf8');
  assert.match(monitor, /useKept<string \| null>\('monitor\.gpu', null\)/, 'unchosen until the person picks, and the pick is kept');
  assert.match(monitor, /gpuSelection === 'all' \|\| !assignedDevices\.length \? 'all' : 'assigned'/, 'assigned by default when the model has cards');
  const css = await readFile(new URL('../src/styles/annotation-fixes.css', import.meta.url), 'utf8');
  assert.match(css, /\.select__opt \{ height: auto; min-height: 36px;/, 'a wrapped name no longer runs over the next row');
});

test('a card picked in the screen replaces an old AMD config\'s "hip_ordinal" (it would send the engine to the old card)', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const { readFileSync, writeFileSync } = await import('node:fs');
    const { join } = await import('node:path');
    const path = join(f.repo, 'strata-iq2_xs.json');
    const config = JSON.parse(readFileSync(path, 'utf8')); config.hip_ordinal = 0; writeFileSync(path, JSON.stringify(config));
    await f.control.refresh(true);
    const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs', gpus: [1] }).id);
    assert.equal(done.state, 'done', done.error);
    const instance = f.control.resolveInstance(done.result.instanceId);
    assert.deepEqual(instance.gpus, [1]);
    const written = JSON.parse(readFileSync(instance.configPath, 'utf8'));
    assert.equal(written.gpu, 1);
    assert.equal('hip_ordinal' in written, false);
  } finally { await f.cleanup(); }
});
