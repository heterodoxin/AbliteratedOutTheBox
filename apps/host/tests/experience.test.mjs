import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
const result=await build({entryPoints:[fileURLToPath(new URL('../src/lib/experience.ts',import.meta.url))],bundle:true,write:false,platform:'node',format:'esm'});
const code=result.outputFiles[0].text;
const { estimate, recommend, configKey, hardwareKey, groupVariants, DEFAULT_EXECUTION: config } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const hw = { cpu: { model: 'Fixture CPU' }, gpus: [{ name: 'Fixture GPU' }], memory: { totalGb: 32 }, accelerator: { unified: true, backend: 'metal', usableVramGb: 24 } };
const base = { id: 'fixture-q4', name: 'Fixture', family: 'Fixture-8B', params: '8B', quant: 'Q4', diskGb: 4, kvKbPerToken: 24, experts: 0, contextMax: 32768, catalog: { family: 'Fixture', version: '8B', purpose: 'Test', capabilities: ['Text'], agentApproved: false, balancedRank: 2, review: 'Not reviewed' } };
function measured(v, reading, answer, extra = {}) { return { ...v, performance: [{ source: 'measured', hardware: hardwareKey(hw), backend: 'metal', config: configKey(v, config), workload: 'document', cache: 'cold-prompt', reading: { low: reading, high: reading + 10 }, answer: { low: answer, high: answer + 5 }, date: '2026-10-05', attribution: 'Test fixture', ...extra }] }; }
test('unknown hardware and invalid drafts do not manufacture fit or performance', () => {
  const unknown = estimate(base, null, config, 'document', false);
  assert.equal(unknown.fit, 'unknown'); assert.equal(unknown.reading, null); assert.equal(unknown.memory.budget, null);
  const invalid = estimate(base, hw, { ...config, batch: NaN }, 'document', true);
  assert.equal(invalid.memory, null); assert.equal(invalid.answer, null);
  assert.ok(recommend([base], null, config, 'document', true).every(r => !r.variant));
});
test('context changes memory; unified pools are never doubled; offload remains unknown', () => {
  const a = estimate(base, hw, config), b = estimate(base, hw, { ...config, context: 8192 });
  assert.ok(a.memory.cache > b.memory.cache); assert.equal(a.memory.total, a.memory.weights + a.memory.cache + a.memory.runtime);
  assert.equal(a.memory.systemBudget, null); assert.equal(a.memory.budget, 32 * .72);
  const discrete = { ...hw, accelerator: { unified: false, backend: 'cuda', usableVramGb: 8 } };
  assert.equal(estimate(base, discrete, config).memory.budget, 8);
  assert.equal(estimate(base, discrete, { ...config, placement: 'Half' }).fit, 'unknown');
  assert.equal(estimate(base, hw, { ...config, context: 4096 }, 'document').fitsWorkload, false);
  assert.equal(estimate(base, hw, { ...config, context: 4096 }, 'document', true).reading, null);
  assert.equal(estimate(base, hw, { ...config, context: 65536 }).fit, 'unsupported');
});
test('agent capability gating beats raw reading speed; output speed selects Performance', () => {
  const fast = measured({ ...base, id: 'fast' }, 900, 150);
  const agent = measured({ ...base, id: 'agent', catalog: { ...base.catalog, agentApproved: true, balancedRank: 1, review: 'Reviewed fixture' } }, 400, 40);
  const r = recommend([fast, agent], hw, config, 'document', false);
  assert.equal(r.find(x => x.profile === 'agentic').variant.id, 'agent');
  assert.equal(r.find(x => x.profile === 'performance').variant.id, 'fast');
  assert.equal(r.find(x => x.profile === 'balanced').variant.id, 'agent');
  assert.equal(recommend([fast], hw, config, 'document', false)[0].variant, undefined);
});
test('unmatched evidence cannot rank a changed configuration or be presented as measured', () => {
  const v = measured(base, 300, 40);
  assert.equal(estimate(v, hw, config).source, 'measured');
  assert.equal(estimate(v, hw, { ...config, batch: 4096 }).source, 'unavailable');
  assert.equal(estimate(v, hw, config, 'short').source, 'unavailable');
  const wrongBackend = measured(base, 300, 40, { backend: 'cuda' });
  assert.equal(estimate(wrongBackend, hw, config).source, 'unavailable');
  const demo = estimate(wrongBackend, hw, config, 'document', true);
  assert.equal(demo.source, 'demo'); assert.match(demo.provenance, /not a measurement/);
  assert.equal(wrongBackend.performance.length, 1);
});
test('missing speed is provisional; the same approved variant may win all three profiles', () => {
  assert.ok(recommend([base], hw, config, 'document', false).every(x => x.provisional));
  const approved = measured({ ...base, catalog: { ...base.catalog, agentApproved: true, review: 'Reviewed fixture' } }, 300, 80);
  assert.deepEqual(recommend([approved], hw, config, 'document', false).map(x => x.variant.id), [base.id, base.id, base.id]);
  assert.ok(recommend([{ ...base, diskGb: 100 }], hw, config, 'document', false).every(x => !x.variant));
});
test('large catalogs preserve every exact variant within stable family/version groups', () => {
  const variants = Array.from({ length: 3000 }, (_, i) => ({ ...base, id: `variant-${i}`, quant: `quant-${i % 20}`, catalog: { ...base.catalog, family: `Family ${Math.floor(i / 100)}`, version: `Release ${Math.floor(i / 20)}` } }));
  const groups = groupVariants(variants);
  assert.equal(groups.length, 30); assert.equal(groups[0].versions.length, 5);
  assert.equal(groups.flatMap(g => g.versions.flatMap(v => v.variants)).length, 3000);
});

test('shared winners merge by variant AND full execution configuration', async () => {
  const { groupRecommendations } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  const r = recommend([base], hw, config, 'document', false);
  assert.equal(groupRecommendations(r).length, 2); // One absent Agentic, one shared winner.
  const changed = r.map(x => x.profile === 'performance' ? { ...x, experience: estimate(base, hw, { ...config, batch: 1024 }) } : x);
  assert.equal(groupRecommendations(changed).length, 3);
  for (const patch of [{context:8192}, {kv:'FP16'}, {placement:'CPU only'}, {threads:4}]) {
    assert.equal(groupRecommendations(r.map(x => x.profile === 'performance' ? { ...x, experience: estimate(base, hw, { ...config, ...patch }) } : x)).length, 3);
  }
});
