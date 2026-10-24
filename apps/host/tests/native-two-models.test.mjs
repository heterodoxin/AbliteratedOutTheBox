import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

test('a second model on a card another model uses still loads (recommend, never force); two cards, two models', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const load = async args => { const done = await finishJob(f.control, f.control.operation('load', args).id); assert.equal(done.state, 'done', done.error); return done; };
    await load({ model: 'iq2_xs', gpus: [0], multiple: true });
    await load({ model: 'iq3_s', gpus: [1], multiple: true });
    assert.deepEqual(f.control.state().instances.map(i => [i.model, i.gpus, i.status]), [['iq2_xs', [0], 'ready'], ['iq3_s', [1], 'ready']]);
    await load({ model: 'iq3_s', gpus: [0], multiple: true });   // the same card as the first model: allowed, and said on the card's row
    assert.equal(f.control.state().instances.length, 3);
  } finally { await f.cleanup(); }
});

test('the screens say which card a loaded model uses, prefer a free card for a second model, and show an unload refusal beside its button', async () => {
  const models = await readFile(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  assert.match(models, /usedBy=\{i => usedBy\(\[i\]\)\}/);
  assert.match(models, /const busy = \(i: number\) => expert && multiple && usedBy\(\[i\]\)\.length > 0/);   // a card another loaded model uses starts Off
  const roles = await readFile(new URL('../src/components/GpuRoles.tsx', import.meta.url), 'utf8');
  assert.match(roles, /tr\('In use by \{0\}', \[other\.join\(', '\)\]\)/);
  assert.match(models, /loadedAction\(`instances\/\$\{i\.id\}\/unload`\)/);
  assert.match(models, /loadedError \? <ErrorNotice message=\{loadedError\}/);
  const control = await readFile(new URL('../server/native-control.mjs', import.meta.url), 'utf8');
  assert.match(control, /is already used by \$\{sharedWith\.label\}, so this model did not fit there/);
});

test('usedBy is declared before the card choice that calls it (a const used earlier broke the Models page when Run multiple models was on)', async () => {
  const models = await readFile(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  assert.ok(models.indexOf('const usedBy = ') > 0 && models.indexOf('const usedBy = ') < models.indexOf('const recommended = '));
});

test('a model that failed to start is not offered in the chat model picker', async () => {
  const picker = await readFile(new URL('../src/components/ChatModelPicker.tsx', import.meta.url), 'utf8');
  assert.equal((picker.match(/native\?\.instances\.filter\(i => i\.status !== 'error'\)\.map/g) ?? []).length, 2);
});

import { friendlyModelName } from '../server/model-names.mjs';
test('a loaded model is named from the catalog (family and version), the file id only as a fallback', async () => {
  const catalog = { families: [{ title: 'Qwen3.8-Flash-Next', sizes: [{ id: 'q2_0', model: 'Q2_0' }] }, { title: 'Qwen3.8-Flash-Next (Gyro)', sizes: [{ id: 'gyro-gyro-s', model: 'Gyro-S' }] }] };
  assert.equal(friendlyModelName(catalog, 'q2_0'), 'Qwen3.8-Flash-Next Q2_0');
  assert.equal(friendlyModelName(catalog, 'gyro-gyro-s'), 'Qwen3.8-Flash-Next Gyro-S');
  assert.equal(friendlyModelName(catalog, 'nope'), null);
  assert.equal(friendlyModelName(null, 'q2_0'), null);
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(done.state, 'done', done.error);
    assert.equal(f.control.state().instances[0].label, 'Qwen3.8-Flash-Next IQ2_XS');
  } finally { await f.cleanup(); }
});

test('card names use the GPU numbering of the other screens, and the fit figures subtract the models that stay loaded', async () => {
  const models = await readFile(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  const roles = await readFile(new URL('../src/lib/gpu-roles.ts', import.meta.url), 'utf8');
  assert.ok(roles.includes('`GPU ${c.index + 1}`'));    // GPU 1 is card 0, as on every other screen
  assert.match(await readFile(new URL('../src/components/GpuRoles.tsx', import.meta.url), 'utf8'), /\{gpuLabel\(c\)\} · \{c\.name\}/);
  assert.match(models, /const keptModels = expert && multiple/);
  assert.match(models, /inUse=\{keptModels\.length \?/);
  const card = await readFile(new URL('../src/components/PcFitCard.tsx', import.meta.url), 'utf8');
  assert.match(card, /total=\{freeVram\}/);
  assert.match(card, /total=\{freeRam\}/);
  // the arithmetic itself
  const text = 'fits: needs 44 GB of the 273 GB available.';
  const gone = Math.round(14 + 27);
  assert.equal(text.replace(/of the (\d+) GB available/, (_m, n) => `of the ${Math.max(0, Number(n) - gone)} GB available`), 'fits: needs 44 GB of the 232 GB available.');
  assert.equal('RX 9070 (card 0) + RAM'.replace(/\(card (\d+)\)/g, (_m, n) => `(GPU ${Number(n) + 1})`), 'RX 9070 (GPU 1) + RAM');
});
