import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { engineRefusal, engineTooOld, markEngine, parseEngineVersion } from '../server/engine-version.mjs';
import { nativeFixture } from './helpers/native-fixture.mjs';

test('engine versions compare as numbers, a missing part counting as zero', () => {
  assert.deepEqual(parseEngineVersion('0.1.45.3'), [0, 1, 45, 3]);
  assert.deepEqual(parseEngineVersion('v0.1.45'), [0, 1, 45]);
  assert.equal(parseEngineVersion('dev'), null);
  assert.equal(engineTooOld('0.1.45', '0.1.45.1'), true, 'the released 0.1.45 stops at v4 files: too old for Gyro');
  assert.equal(engineTooOld('0.1.45.1', '0.1.45.1'), false);
  assert.equal(engineTooOld('0.1.45.3', '0.1.45.1'), false);
  assert.equal(engineTooOld('0.1.46', '0.1.45.1'), false);
  assert.equal(engineTooOld('0.1.44.1', '0.1.45.1'), true);
  assert.equal(engineTooOld('0.1.45', null), false, 'no minimum: never too old');
  assert.equal(engineTooOld(null, '0.1.45.1'), false, 'an unreadable version is not called too old');
  assert.equal(engineTooOld('0.1.40', '0.1.45.1', 'local'), false, 'an engine built from this checkout has it whatever its number says');
});

const catalog = () => ({ families: [{ family: 'qwen', sizes: [{ id: 'q2_0', model: 'Q2_0', engine_min: null }] }, { family: 'gyro', sizes: [{ id: 'gyro-gyro-s', model: 'Gyro-S', engine_min: '0.1.45.1' }, { id: 'gyro-gyro-m', model: 'Gyro-M', engine_min: '0.1.45.1' }] }] });

test('the catalog marks the sizes an older engine cannot read; the refusal says what to do', () => {
  const marked = markEngine(catalog(), { version: '0.1.45' });
  assert.deepEqual(marked.families.flatMap(f => f.sizes.map(s => [s.id, s.needsNewerEngine])), [['q2_0', false], ['gyro-gyro-s', true], ['gyro-gyro-m', true]]);
  const text = engineRefusal(marked, 'gyro-gyro-s', { version: '0.1.45' });
  assert.match(text, /^This model needs a newer engine: Gyro-S needs engine 0\.1\.45\.1 or newer, and this one is 0\.1\.45\. Update the engine in Settings, then try again\.$/);
  assert.equal(engineRefusal(marked, 'q2_0', { version: '0.1.45' }), null);
  const fresh = markEngine(catalog(), { version: '0.1.45.3' });
  assert.ok(fresh.families.every(f => f.sizes.every(s => !s.needsNewerEngine)));
  assert.equal(markEngine(null, {}), null);
});

test('with an older engine installed the app refuses to download or load Gyro, with the plain sentence; a new enough engine goes on', async () => {
  const old = nativeFixture({ engineStatus: { version: '0.1.45', backend: 'cuda' } });
  try {
    old.catalog.families.push({ family: 'gyro', title: 'Qwen3.8-Flash-Next (Gyro)', about: '', images: false, sizes: [{ id: 'iq2_xs', model: 'Gyro-S', download_gb: 58, ram_needed_gb: 28, about: '', on_this_pc: 'fits', installed: true, engine_min: '0.1.45.1' }] });
    old.catalog.families[0].sizes[0].engine_min = '0.1.45.1';
    const state = await old.control.refresh(true);
    const sizes = state.catalog.families.flatMap(f => f.sizes);
    assert.ok(sizes.find(s => s.engine_min).needsNewerEngine);
    const id = sizes.find(s => s.engine_min).id;
    assert.throws(() => old.control.operation('load', { model: id }), /This model needs a newer engine: .* needs engine 0\.1\.45\.1 or newer, and this one is 0\.1\.45\./);
    assert.throws(() => old.control.operation('install', { model: id }), /needs a newer engine/);
    assert.equal(old.control.state().jobs.length, 0, 'nothing was started');
  } finally { await old.cleanup(); }
  const ok = nativeFixture({ engineStatus: { version: '0.1.45.3', backend: 'cuda' } });
  try {
    ok.catalog.families[0].sizes[0].engine_min = '0.1.45.1';
    const state = await ok.control.refresh(true);
    assert.ok(state.catalog.families[0].sizes.every(s => !s.needsNewerEngine));
  } finally { await ok.cleanup(); }
});

test('the screens use the existing "needs a newer engine" notice for a size, and the buttons are held back', async () => {
  const models = await readFile(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  assert.match(models, /selectedFamily && \(!selectedSize \|\| selectedSize\.needsNewerEngine\) \? <><h2 className="h3">\{tr\("This model needs a newer engine\."\)\}/);
  assert.match(models, /!!selectedSize\.needsNewerEngine\}/);
  const guided = await readFile(new URL('../src/components/GuidedSetup.tsx', import.meta.url), 'utf8');
  assert.equal((guided.match(/needsNewerEngine/g) ?? []).length, 3);
});
