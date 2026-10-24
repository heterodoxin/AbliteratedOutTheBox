/**
 * Several requests at once in the installed app: the default, how it is sized, what a person's own number does, and the
 * retry with one when a load with several slots fails. Nothing here starts an engine; Control is a fake that records what
 * the host saved and asked it to start.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { chooseParallel, applyParallel, withoutParallel, slotGb, DEFAULT_SLOTS, PARALLEL_POLICY, BATCH_MTP_POLICY, FALLBACK_NOTE } from '../server/native-parallel.mjs';
import { buildProfile } from '../server/agent/profile.mjs';

const hw = (gb, n = 1) => ({ gpus: Array.from({ length: n }, (_, i) => ({ index: i, vram_gb: gb })) });
const cfg = (context, extra = {}) => ({ args: ['--max-context', String(context), '--mtp', '/m', '--spec', '4'], gpu: 0, ...extra });

test('the default aims for 4 request slots; the policy is one switch; with no hardware read it is one at a time', () => {
  assert.equal(DEFAULT_SLOTS, 4);
  assert.ok(['fit', 'fixed'].includes(PARALLEL_POLICY));
  const unknown = chooseParallel({ config: cfg(32768) });
  assert.equal(unknown.slots, PARALLEL_POLICY === 'fixed' ? 4 : 1, 'nothing to size against: the safe default');
});

test('sizing: the slots may take at most a fifth of the expert cache the card would have; said in plain words', () => {
  // measured on the 0.1.45 engine: 0.92 GiB per slot at 32K; a 12 GB card ends with 5.1 GiB of expert cache
  assert.ok(Math.abs(slotGb({ context: 32768 }) * 0.931 - 0.92) < 0.03);
  if (PARALLEL_POLICY !== 'fit') return;
  const slots = (gb, context, extra = {}, cards = 1) => chooseParallel({ config: cfg(context, extra), hardware: hw(gb, cards) });
  assert.deepEqual([12, 16, 20, 24, 32].map(gb => slots(gb, 32768).slots), [1, 1, 2, 3, 4]);
  assert.deepEqual([12, 24, 32, 48].map(gb => slots(gb, 65536).slots), [1, 1, 2, 4], 'the context is per slot: doubling it halves the slots');
  assert.equal(slots(24, 65536, { gpu: [0, 1] }, 2).slots, 4, 'two cards share the load');
  const small = slots(12, 32768);
  assert.equal(small.slots, 1); assert.match(small.note, /one request at a time/); assert.match(small.note, /Helper agents and a second chat take their turn/);
  const some = slots(20, 32768);
  assert.match(some.note, /room for 2 requests at once/); assert.match(some.note, /smaller context size leaves room for more/);
  assert.equal(slots(32, 32768).note, null, 'no note when the default fits');
  assert.doesNotMatch(small.note + some.note, /--|batch|VRAM|GiB|tok|cache/i, 'no engine words');
});

test('batch-mtp is automatic on big single cards only: 24 GB and up with room for the drafters, never on small cards or a layer split; Expert mode overrides either way', () => {
  assert.equal(BATCH_MTP_POLICY, 'auto');
  const mtp = (gb, extra = {}, cards = 1, values = {}, context = 32768) => chooseParallel({ config: cfg(context, extra), hardware: hw(gb, cards), values });
  assert.deepEqual([12, 16, 20].map(gb => mtp(gb).batchMtp), [false, false, false], 'small cards: it did not pay (3060: -6% for one chat, 400 MiB less cache)');
  assert.deepEqual([24, 32, 48, 72].map(gb => mtp(gb).batchMtp), [true, true, true, true]);
  assert.equal(mtp(72).slots, 4);
  assert.equal(mtp(24, {}, 1, {}, 131072).batchMtp, false, 'one slot only at this context: nothing to batch');
  assert.equal(mtp(72, { gpu: [0, 1] }, 2).batchMtp, false, 'a layer split: -22% with all experts in VRAM');
  assert.equal(chooseParallel({ config: { args: ['--max-context', '32768'], gpu: 0 }, hardware: hw(72) }).batchMtp, false, 'no draft layer, nothing to batch');
  assert.equal(chooseParallel({ config: cfg(32768, { args: ['--max-context', '32768', '--mtp', '/m', '--batch-mtp'] }), hardware: hw(72) }).batchMtp, false, 'already there: not added twice');
  const applied = applyParallel(cfg(32768), mtp(72));
  assert.ok(applied.cfg.args.includes('--batch-mtp')); assert.equal(applied.cfg.parallel, 4);
  // Expert mode: off wins on a big card; on wins on a small one-card setup; an explicit slot count can still ask for it
  assert.equal(mtp(72, {}, 1, { batchMtp: false }).batchMtp, false);
  assert.equal(mtp(72, { batchMtp: false }).batchMtp, false);
  assert.equal(mtp(20, {}, 1, { batchMtp: true }).batchMtp, true, '20 GB: 2 slots, asked for by hand');
  assert.equal(mtp(72, { gpu: [0, 1] }, 2, { batchMtp: true }).batchMtp, false, 'never forced on a layer split');
  const manual = chooseParallel({ config: cfg(32768), values: { parallel: 2, batchMtp: true }, hardware: hw(12) });
  assert.deepEqual([manual.slots, manual.explicit, manual.batchMtp], [2, true, true]);
  assert.ok(applyParallel(cfg(32768), manual).cfg.args.includes('--batch-mtp'));
  assert.ok(!applyParallel(cfg(32768, { parallel: 2 }), chooseParallel({ config: cfg(32768, { parallel: 2 }), hardware: hw(72) })).cfg.args.includes('--batch-mtp'), 'a number the person chose alone does not add it');
  // the retry with one request at a time takes it away again
  assert.ok(!withoutParallel(applied.cfg).args.includes('--batch-mtp'));
});

test("a number the person chose is kept exactly: Expert mode's value, the saved settings, an explicit --batch", () => {
  assert.deepEqual(applyParallel(cfg(32768, { parallel: 1 }), chooseParallel({ config: cfg(32768, { parallel: 1 }), hardware: hw(24) })).cfg.parallel, 1);
  const six = chooseParallel({ config: cfg(32768), values: { parallel: 6 }, hardware: hw(8) });
  assert.deepEqual([six.slots, six.explicit, six.batchMtp], [6, true, false], 'recommend, never force: 6 on an 8 GB card is allowed');
  const one = chooseParallel({ config: cfg(32768), values: { parallel: 1 }, hardware: hw(24) });
  assert.deepEqual([one.slots, one.explicit], [1, true]);
  assert.equal(chooseParallel({ config: cfg(32768, { args: ['--max-context', '32768', '--batch', '3'] }), hardware: hw(24) }).explicit, true);
  const saved = cfg(32768, { parallel: 3 });
  assert.equal(applyParallel(saved, chooseParallel({ config: saved })).cfg, saved, 'untouched');
});

test('the retry configuration has no slots and no batch-mtp', () => {
  const next = withoutParallel({ ...cfg(32768), parallel: 4, args: [...cfg(32768).args, '--batch-mtp'] });
  assert.equal(next.parallel, undefined); assert.ok(!next.args.includes('--batch-mtp'));
});

test('helper agents: at least as many as the engine slots, and extra ones wait for their turn instead of being refused', () => {
  const profile = slots => buildProfile({ local: [{ id: 'a', baseUrl: 'http://127.0.0.1:1/v1', modelId: 'm', slots }] });
  const cap = slots => JSON.parse(profile(slots).patch.split('\n').slice(1).join('\n')).find(e => e.id === 'subagent').config.maxActiveSubagents;
  assert.deepEqual([cap(1), cap(2), cap(4), cap(6)], [4, 4, 4, 6]);
  assert.notEqual(profile(6).signature, profile(2).signature, 'a changed helper limit restarts the agent when idle');
});

// ---------------------------------------------------------------------------------------- the installed app's load

function fixture({ failStarts = 0, config = {} } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-par-test-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;      // the app's own install folder: the host always names it to Control
  const path = join(install, 'strata-iq3_s.json');
  writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--max-context', '32768', '--mtp', '/m', '--spec', '4'], model_name: 'Qwen iq3_s', port: 8080, ...config }));
  const saved = []; const calls = [];
  let starts = 0;
  const control = {
    control: { dir: join(root, 'control') }, version: '0.1.45',
    async installedModels() { return { install_dir: install, models: [{ id: 'iq3_s', name: 'Qwen iq3_s', source_config: path, component_status: 'present', context_tokens: 32768 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.1.45', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); calls.push(['saveProfile', name, basename(file)]); return {}; },
    async start(name) { calls.push(['start', name]); starts++; if (starts <= failStarts) throw new Error('server_exited: out of memory'); return { ok: true, state: 'ready' }; },
    async stop(name) { calls.push(['stop', name]); return { ok: true }; },
  };
  const fetcher = async () => Response.json({ status: 'ok' });
  const native = createNativeControl({ control, hardwareProbe: () => [{ index: 0, vram_gb: 32 }], directory: data, spawnProcess: () => { throw new Error('the host starts no process'); }, fetcher, requests: () => [] });
  return { native, saved, calls, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('the installed app loads with 4 request slots by default', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq3_s' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.saved.length, 1);
    assert.equal(f.saved[0].parallel, 4);
    assert.ok(f.saved[0].args.includes('--batch-mtp'), 'a 32 GB card: MTP per request is on automatically');
    assert.equal(f.native.state().instances[0].parallel, 4, 'the agent reads the slot count from here');
  } finally { await f.native.close(); f.cleanup(); }
});

test("the person's own number wins: saved settings and Expert mode's value", async () => {
  for (const [config, values, expected] of [[{ parallel: 1 }, {}, 1], [{}, { parallel: 3 }, 3], [{ parallel: 2 }, { parallel: 6 }, 6]]) {
    const f = fixture({ config });
    try {
      await f.native.refresh(true);
      const job = await finish(f.native, f.native.operation('load', { model: 'iq3_s', values }).id);
      assert.equal(job.state, 'done', job.error);
      assert.equal(f.saved[0].parallel, expected);
      assert.ok(!f.saved[0].args.includes('--batch-mtp'));
    } finally { await f.native.close(); f.cleanup(); }
  }
});

test('a load that fails with several slots is tried again with one, and the person is told in plain words', async () => {
  const f = fixture({ failStarts: 1 });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq3_s' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.saved.length, 2);
    assert.equal(f.saved[0].parallel, 4); assert.equal(f.saved[1].parallel, undefined);
    assert.ok(!f.saved[1].args.includes('--batch-mtp'));
    assert.match(job.result.summary, /one request at a time/);
    assert.equal(job.result.summary.includes(FALLBACK_NOTE), true);
    assert.equal(f.native.state().instances[0].parallel, 1);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a load that fails even with one slot reports the failure, and a chosen number is never silently changed', async () => {
  const both = fixture({ failStarts: 2 });
  try {
    await both.native.refresh(true);
    const job = await finish(both.native, both.native.operation('load', { model: 'iq3_s' }).id);
    assert.equal(job.state, 'error');
    assert.equal(both.saved.length, 2);
  } finally { await both.native.close(); both.cleanup(); }
  const chosen = fixture({ failStarts: 1, config: { parallel: 3 } });
  try {
    await chosen.native.refresh(true);
    const job = await finish(chosen.native, chosen.native.operation('load', { model: 'iq3_s' }).id);
    assert.equal(job.state, 'error', 'the person asked for 3: no quiet fallback');
    assert.equal(chosen.saved.length, 1);
  } finally { await chosen.native.close(); chosen.cleanup(); }
});

test('the draft layer (MTP) is part of what the card is sized for; switched off, its 1 GB goes to the slots', () => {
  if (PARALLEL_POLICY !== 'fit') return;
  const slots = (gb, extra = {}, args) => chooseParallel({ config: { ...cfg(32768, extra), ...(args ? { args } : {}) }, hardware: hw(gb) }).slots;
  assert.equal(slots(26.5), 3, 'MTP on (the default): the measured sizing stands');
  assert.equal(slots(26.5, { mtp: false }), 4, 'a card with the draft layer off has 1 GB more cache');
  assert.equal(slots(26.5, {}, ['--max-context', '32768', '--spec', '4']), 4, 'no draft layer in the arguments');
  assert.equal(slots(12, { mtp: false }), 1, 'a 12 GB card still gets one request at a time');
});
