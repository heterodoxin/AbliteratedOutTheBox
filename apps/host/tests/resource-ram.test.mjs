import test from 'node:test';
import assert from 'node:assert/strict';
import { resourceSettings, saferStart } from '../server/native-resources.mjs';

const pc64 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5070', vram_gb: 11.9 }] };
const iq3s = { arenaGb: 50.3 };
const config = { gpu: 0, args: ['--native', 'x.gguf', '--max-context', '32768', '--pool-workers', '12'] };
const flag = (args, name) => args[args.indexOf(name) + 1];

test('"leave 20% for my work" keeps the model within 80% of the RAM: a RAM budget for the experts', () => {
  const work = resourceSettings(config, pc64, 'work', iq3s);
  assert.ok(work.args.includes('--resident-budget-gib'), 'a model whose experts do not fit 80% of 63 GB gets a RAM budget');
  const gib = Number(flag(work.args, '--resident-budget-gib'));
  assert.ok(gib + 10 <= 63 * 0.8, `${gib} GiB plus the 10 GB the rest needs stays inside 80% of the RAM`);
  assert.equal(work.ui_resource_baseline.ramBudget, true);
});

test('full performance takes that budget out again and keeps what the installation had', () => {
  const work = resourceSettings(config, pc64, 'work', iq3s);
  const back = resourceSettings({ ...config, args: work.args, ui_resource_profile: 'work', ui_resource_baseline: work.ui_resource_baseline }, pc64, 'performance', iq3s);
  assert.equal(back.args.includes('--resident-budget-gib'), false);
  assert.equal(flag(back.args, '--pool-workers'), '12');
});

test('a model that fits, or one that already has a RAM mode, is left alone', () => {
  assert.equal(resourceSettings(config, { ...pc64, ram_gb: 128 }, 'work', iq3s).args.includes('--resident-budget-gib'), false);
  assert.equal(resourceSettings(config, pc64, 'work', { arenaGb: 34 }).args.includes('--resident-budget-gib'), false, 'Q2_0 fits 80% of 63 GB with 10 to spare');
  const own = { ...config, args: [...config.args, '--mmap-experts'] };
  assert.equal(resourceSettings(own, pc64, 'work', iq3s).args.includes('--resident-budget-gib'), false);
  assert.equal(resourceSettings(config, pc64, 'work').args.includes('--resident-budget-gib'), false, 'no model size, no change');
});

test('neither profile reserves less VRAM than the engine\'s safe floor (the driver pages GPU memory below it and decoding collapses)', () => {
  const card12 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, vram_gb: 11.9 }] };
  const card24 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, vram_gb: 24 }] };
  for (const profile of ['work', 'performance']) {
    const lowConfig = { gpu: 0, args: ['--max-context', '32768', '--vram-reserve-mib', '700'] };
    assert.ok(resourceSettings(lowConfig, card12, profile).vram_reserve_mib >= 1536, `${profile} on 12 GB`);
    assert.ok(resourceSettings({ gpu: 0, args: [] }, card12, profile).vram_reserve_mib >= 1536, `${profile}, nothing configured`);
    assert.ok(resourceSettings(lowConfig, card24, profile).vram_reserve_mib >= 1024, `${profile} on 24 GB`);
  }
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card12, 'performance').vram_reserve_mib, 1536);
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card24, 'performance').vram_reserve_mib, 1024);
  assert.equal(resourceSettings({ gpu: 0, args: [], vram_reserve_mib: 2437 }, card12, 'performance').vram_reserve_mib, 2437, 'a larger reserve of the installation is kept');
});

test('"leave 20% for my work" also asks the engine to leave 20% of the RAM free at its start, and full performance takes that out again', () => {
  const work = resourceSettings(config, pc64, 'work', iq3s);
  assert.equal(work.env.STRATA_RESIDENT_HEADROOM_GIB, '13', '20% of 63 GB, rounded up');
  assert.equal(work.ui_resource_baseline.ramHeadroom, true);
  const back = resourceSettings({ ...config, args: work.args, env: work.env, ui_resource_profile: 'work', ui_resource_baseline: work.ui_resource_baseline }, pc64, 'performance', iq3s);
  assert.equal(back.env.STRATA_RESIDENT_HEADROOM_GIB, undefined);
  const own = resourceSettings({ ...config, env: { STRATA_RESIDENT_HEADROOM_GIB: '6' } }, pc64, 'work', iq3s);
  assert.equal(own.env, undefined, 'a value the user set is not overwritten');
  assert.equal(resourceSettings(config, { ...pc64, ram_gb: 128 }, 'work', iq3s).env, undefined, 'no budget, no headroom setting');
});

test('a start that died gets ONE safer second try: pin less when the locked memory ran out, else a RAM budget with the rest on the SSD', () => {
  const pinned = saferStart(config, pc64, iq3s, 'server_exited: mapped staging allocation failed [pinned-memory-exhausted] start again with STRATA_ARENA_PIN_GIB=26 (pin less)');
  assert.equal(pinned.config.env.STRATA_ARENA_PIN_GIB, '26');
  assert.match(pinned.words, /would not lock as much memory/);
  assert.equal(flag(saferStart(pinned.config, pc64, iq3s, 'server_exited: mapped staging allocation failed [pinned-memory-exhausted]').config.args, '--resident-budget-gib'), String(Math.floor(63 * .5) - 4), 'already pinned less: the next safer setting is the RAM budget');
  const noHint = saferStart(config, pc64, iq3s, 'mapped staging allocation failed');
  assert.equal(noHint.config.env.STRATA_ARENA_PIN_GIB, String(Math.floor(63 / 2) - 8));
  const low = saferStart(config, pc64, iq3s, 'server_exited: memory guard: 1.9 GiB free');
  assert.equal(flag(low.config.args, '--resident-budget-gib'), String(Math.floor(63 * .5) - 4));
  assert.match(low.words, /low-memory mode/);
  const already = saferStart({ ...config, args: [...config.args, '--resident-budget-gib', '40'] }, pc64, iq3s, 'server_exited');
  assert.equal(flag(already.config.args, '--resident-budget-gib'), '28', '70% of the budget it had');
  assert.equal(saferStart({ ...config, args: [...config.args, '--resident-budget-gib', '10'] }, pc64, iq3s, 'server_exited'), null, 'below 8 GB there is nothing left to give');
  assert.equal(saferStart({ ...config, args: [...config.args, '--resident-experts'] }, pc64, iq3s, 'server_exited'), null, 'already the SSD mode');
  assert.equal(saferStart(config, pc64, { arenaGb: 20 }, 'server_exited'), null, 'a model that fits RAM gains nothing from the SSD mode');
  assert.equal(saferStart(config, pc64, undefined, 'server_exited'), null, 'unknown model size: no guess');
  assert.equal(saferStart(config, pc64, iq3s, 'Some model files are missing'), null, 'not a memory failure: no second try');
  assert.equal(config.args.includes('--resident-budget-gib'), false, 'the input config is not changed');
});

test('an engine that stopped because the GPU expert cache got no room is started again with less VRAM kept free, then a shorter context (#3192)', () => {
  const tenGb = { cpu: { cores: 16 }, ram_gb: 64, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 3080', vram_gb: 10 }] };
  const cfg = { gpu: 0, args: ['--native', 'x.gguf', '--max-context', '131072', '--vision', '--vram-reserve-mib', '2048'] };
  const log = "server_exited (exit code 1): strata generate: no VRAM is left for the expert cache: it needs at least 288 slots (639 MiB), about 727 MiB more than this card has free. To make room: a smaller --max-context (the session, mostly its KV cache, takes 1931 MiB at 131072 tokens), --kv q4_0 | strata serve: verify: needs the profile-filled VRAM expert tier";
  const safer = saferStart(cfg, tenGb, iq3s, log);
  assert.equal(flag(safer.config.args, '--vram-reserve-mib'), '1065', '2048 - (727 + 256 margin)');
  assert.equal(flag(safer.config.args, '--max-context'), '131072', 'the reserve alone was enough: the context stays');
  assert.match(safer.words, /no room left/);
  const tight = saferStart({ ...cfg, args: ['--max-context', '131072', '--vram-reserve-mib', '1100'] }, tenGb, iq3s, log);
  assert.equal(flag(tight.config.args, '--vram-reserve-mib'), '1024', 'never below 1024 MiB (#2065)');
  assert.ok(Number(flag(tight.config.args, '--max-context')) < 131072, 'the rest comes from the context');
  assert.equal(saferStart({ ...cfg, args: ['--max-context', '16384', '--vram-reserve-mib', '1024'] }, tenGb, iq3s, log), null, 'nothing left to give');
  assert.equal(flag(saferStart(cfg, tenGb, undefined, 'server_exited: strata serve: verify: needs the profile-filled VRAM expert tier').config.args, '--vram-reserve-mib'), '1092', 'without the numbers: a fixed step, no model size needed');
  assert.equal(cfg.args[cfg.args.indexOf('--vram-reserve-mib') + 1], '2048', 'the input config is not changed');
});

test('0.2.3 tuning: the Work profile takes 80% of the PHYSICAL cores rounded up, never more than the cores less the host thread', () => {
  const w = cores => Number(flag(resourceSettings({ gpu: 0, args: [] }, { cpu: { cores }, ram_gb: 64, gpus: [{ index: 0, vram_gb: 12 }] }, 'work').args, '--pool-workers'));
  assert.equal(w(6), 5, 'Ryzen 5 7600: the engine\'s own 5 (floor gave 4, 8-10% slower; 12 logical threads gave 9 in 0.2.1)');
  assert.equal(w(4), 3);
  assert.equal(w(8), 7);
  assert.equal(w(10), 8);
  assert.equal(w(12), 10);
  assert.equal(w(16), 13);
  assert.equal(w(2), 1);
  const own = { gpu: 0, args: ['--pool-workers', '3'] };
  assert.equal(flag(resourceSettings(own, { cpu: { cores: 6 }, ram_gb: 64, gpus: [] }, 'work').args, '--pool-workers'), '3', 'a count the person set stays when it is lower');
});

test('0.2.3 tuning: Work keeps a fifth of the card but no more than 1536 MiB on a card of 16 GB or less; a CPU picture reader needs no 2048 floor', () => {
  const card = gb => ({ cpu: { cores: 6 }, ram_gb: 64, gpus: [{ index: 0, vram_gb: gb, drives_display: true }] });
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card(11.9), 'work').vram_reserve_mib, 1536, 'RTX 5070: 2437 gave 38 tok/s, 1536 gave 46 (Gyro-S 64K), with 1.1 GiB still free');
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card(16), 'work').vram_reserve_mib, 1536);
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card(24), 'work').vram_reserve_mib, 4915, 'above 16 GB the fifth stays');
  assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: false } }, card(12), 'performance').vram_reserve_mib, 1536, 'the CPU reader takes no VRAM');
  assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: false } }, card(12), 'work').vram_reserve_mib, 1536);
  assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: true } }, card(12), 'performance').vram_reserve_mib, 2048, 'the GPU reader keeps its 2048');
  assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: true } }, card(12), 'work').vram_reserve_mib, 2048);
  assert.equal(resourceSettings({ gpu: 0, args: ['--vram-reserve-mib', '3000'], user_args: ['--vram-reserve-mib', '3000'] }, card(12), 'work').vram_reserve_mib, 3000, 'the person\'s own number is kept');
});
