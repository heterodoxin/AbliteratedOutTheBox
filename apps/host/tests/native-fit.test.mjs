/** Items 10 and 11: what a version needs of RAM + VRAM (without the n-gram table) and how much would be read from the SSD. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/native-fit.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { fitOn, memoryNeeded, RECOMMENDED, VERSION_GROUPS, groupOpen, recommendedVersions, PICK_NOTE, OTHER_NOTE, MTP_VRAM_GB } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

const swift = { ram_needed_gb: 48, experts_gb: 35.5 };
const gyroS = { ram_needed_gb: 27.6, experts_gb: 25.8, mem_total: true };
const q4 = { ram_needed_gb: 48, experts_gb: 77 };

test('the memory a version needs is RAM + VRAM without the n-gram table', () => {
  assert.equal(memoryNeeded(swift), 36.5, 'experts plus about 1 GB for the dense part');
  assert.equal(memoryNeeded(gyroS), 27.6, 'Gyro states its total');
  assert.equal(memoryNeeded({ ...swift, memory_gb: 36.5 }), 36.5, 'the catalog figure wins when it is sent');
});

test('a PC with room reads nothing from the SSD', () => {
  const f = fitOn(swift, { ram_gb: 64, vram_gb: 12 });
  assert.equal(f.known, true); assert.equal(f.spillGb, 0);
  assert.ok(f.vramUseGb > 0 && f.vramUseGb <= 12 && f.ramUseGb > 0 && f.ramUseGb <= 54);
});

test('a PC without room reads the rest from the SSD, and the amount is real', () => {
  const f = fitOn(q4, { ram_gb: 64, vram_gb: 12 });          // 77 GB of experts: 6 on the card (the draft layer takes 1 GB), 54 in RAM, 17 from the SSD
  assert.equal(f.spillGb, 17);
  assert.equal(fitOn(q4, { ram_gb: 64, vram_gb: 12 }, false).spillGb, 16, 'with the draft layer off the card holds 1 GB more');
  assert.equal(fitOn(q4, { ram_gb: 192, vram_gb: 24 }).spillGb, 0);
  assert.ok(fitOn(gyroS, { ram_gb: 16, vram_gb: 8 }).spillGb > 10, 'a small PC spills a lot');
});

test('unknown hardware gives no verdict instead of a made-up one', () => {
  const f = fitOn(swift, null);
  assert.equal(f.known, false); assert.equal(f.spillGb, 0); assert.equal(f.memoryGb, 36.5);
  assert.equal(fitOn(swift, { ram_gb: 0 }).known, false);
});

test('Recommended is the fixed four, in the owner\'s order; the groups are Swift, GSQ-RCO, NVFP4, Gyro (Unsloth for the rest)', () => {
  assert.deepEqual(RECOMMENDED.map(r => `${r.family}:${r.model}`), ['gyro:Gyro-S', 'swift:IQ2_XS', 'qwen:IQ3_S', 'unsloth:UD-Q4_K_XL']);
  assert.deepEqual(VERSION_GROUPS.map(g => g.title), ['Swift models', 'GSQ-RCO models', 'NVFP4 models', 'Gyro models', 'Unsloth models']);
  assert.ok(RECOMMENDED.every(r => r.note.length > 0 && r.note.length < 40), 'one short note each');
});

test('the group of the running version can be collapsed (Gyro-M running, the Gyro group stayed open)', () => {
  let opened = {};
  assert.equal(groupOpen(opened, 'gyro', true), true, 'it starts open: it holds the selected version');
  assert.equal(groupOpen(opened, 'swift', false), false, 'the others start closed');
  opened = { ...opened, gyro: !groupOpen(opened, 'gyro', true) };   // the person clicks the Gyro header
  assert.equal(groupOpen(opened, 'gyro', true), false, 'one click closes it');
  opened = { ...opened, gyro: !groupOpen(opened, 'gyro', true) };
  assert.equal(groupOpen(opened, 'gyro', true), true, 'and one more opens it again');
});

test('the draft layer (MTP) counts about 1 GB of the card', () => {
  assert.equal(MTP_VRAM_GB, 1);
  const on = fitOn(swift, { ram_gb: 40, vram_gb: 12 }), off = fitOn(swift, { ram_gb: 40, vram_gb: 12 }, false);
  assert.equal(Math.round((on.vramUseGb - off.vramUseGb) * 10) / 10, 0, 'the card is full either way here; the experts it cannot hold move to RAM');
  assert.equal(Math.round((on.ramUseGb - off.ramUseGb) * 10) / 10, 1);
  assert.equal(fitOn(swift, { ram_gb: 40, vram_gb: 0 }).vramUseGb, 0, 'no card, no draft layer on it');
});

const v = (sourceFamily, model) => ({ sourceFamily, model, id: sourceFamily + '-' + model });
const all = [v('qwen', 'Q2_0'), v('gyro', 'Gyro-S'), v('gyro', 'Gyro-M'), v('swift', 'IQ2_XS'), v('qwen', 'IQ3_XXS'), v('qwen', 'IQ3_S'), v('unsloth', 'UD-Q4_K_XL')];

test('the page shows the own pick of the engine for this PC first (the same one strata launch prints), per memory tier', () => {
  const names = rows => rows.map(x => x.v.model + (x.r.note === PICK_NOTE ? '*' : x.r.note === OTHER_NOTE ? '+' : ''));
  assert.deepEqual(names(recommendedVersions(all)), ['Gyro-S', 'IQ2_XS', 'IQ3_S', 'UD-Q4_K_XL'], 'no pick: the fixed list');
  const t40 = { family: 'swift', model: 'IQ2_XS', why: '40 GB tier: Swift 1.5 IQ2_XS; Gyro-M (35 GB) is the other choice there' };
  assert.deepEqual(names(recommendedVersions(all, t40)), ['IQ2_XS*', 'Gyro-M+', 'Gyro-S', 'IQ3_S', 'UD-Q4_K_XL']);
  assert.deepEqual(names(recommendedVersions(all, { family: 'swift', model: 'IQ2_XS', why: 'the 40 GB tier' })), ['IQ2_XS*', 'Gyro-S', 'IQ3_S', 'UD-Q4_K_XL'], 'no Gyro-M when the engine does not name it (Intel, Mac)');
  assert.deepEqual(names(recommendedVersions(all, { family: 'qwen', model: 'IQ3_XXS', why: '48 GB tier' })), ['IQ3_XXS*', 'Gyro-S', 'IQ2_XS', 'IQ3_S', 'UD-Q4_K_XL']);
  assert.deepEqual(names(recommendedVersions(all, { family: 'qwen', model: 'IQ3_S', why: '64 GB tier' })), ['IQ3_S*', 'Gyro-S', 'IQ2_XS', 'UD-Q4_K_XL']);
  assert.deepEqual(names(recommendedVersions(all, { family: null, model: null })), ['Gyro-S', 'IQ2_XS', 'IQ3_S', 'UD-Q4_K_XL']);
});

test('the tiers of the engine are 40 GB Swift IQ2_XS, 48 GB IQ3_XXS, 64 GB IQ3_S in both Python places the page and strata launch read', async () => {
  for (const file of ['../../../setup.py', '../../../tools/strata_mcp.py']) {
    const text = await readFile(new URL(file, import.meta.url), 'utf8');
    const m = /^MEMORY_TIERS\s*=\s*(.*)$/m.exec(text);
    assert.ok(m, file);
    assert.deepEqual([...m[1].matchAll(/\(([\d.]+),\s*"(\w+)",\s*"(\w+)"\)/g)].map(x => [Number(x[1]), x[2], x[3]]), [[62, 'qwen', 'IQ3_S'], [46, 'qwen', 'IQ3_XXS'], [38, 'swift', 'IQ2_XS']], file);
  }
});
