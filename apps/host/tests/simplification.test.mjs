import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';
import { migrateSetup } from '../server/setup.mjs';
const { code } = await transform(await readFile(new URL('../src/lib/lifecycle.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { lifecyclePolicy } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('old setup steps map once and preserve selections and completed work', () => {
  const choices = { variantId:'fixture', profile:'performance', contextLength:'8k', batch:1024, batchAuto:false };
  const history = [{step:3, at:'fixture'}];
  for (const [oldStep, expected] of [[1,1],[2,2],[3,2],[4,3],[5,4]]) {
    const original = { step:oldStep, choices, history, dismissed:true, complete:false };
    const migrated = migrateSetup(original);
    assert.equal(migrated.step, expected); assert.equal(migrated.flowVersion, 2);
    assert.deepEqual(migrated.choices, choices); assert.deepEqual(migrated.history, history);
    assert.equal(migrated.dismissed, true); assert.equal(migrated.complete, false);
    assert.deepEqual(migrateSetup(migrated), migrated);
    assert.equal(original.step, oldStep);
  }
});

test('shared lifecycle policy blocks queued, active, disconnected and unknown status', () => {
  const idle = {connected:true, streaming:false, stats:{running:0, waiting:0}, busy:{}};
  assert.equal(lifecyclePolicy(idle).allowed, true);
  for (const patch of [
    {stats:{running:0, waiting:2}}, {stats:{running:1, waiting:0}}, {streaming:true},
    {connected:false}, {stats:null}, {stats:{running:0}}, {stats:{running:0,waiting:NaN}},
    {busy:{restart:true}}, {busy:{unload:true}}, {busy:{'load:fixture':true}},
  ]) assert.equal(lifecyclePolicy({...idle, ...patch}).allowed, false, JSON.stringify(patch));
  assert.match(lifecyclePolicy({...idle, stats:{running:0,waiting:2}}).reason, /2 queued/);
  assert.match(lifecyclePolicy({...idle, stats:null}).reason, /unavailable/);
  assert.equal(lifecyclePolicy({...idle, busy:{'install:fixture':true}}).allowed, true);
});

test('blocked lifecycle writes do not change config; save-only still works', async () => {
  const { build } = await import('esbuild');
  const { createRequire } = await import('node:module');
  const { outputFiles } = await build({entryPoints:[fileURLToPath(new URL('../src/lib/store.ts',import.meta.url))],bundle:true,write:false,platform:'node',format:'cjs'});
  const mod={exports:{}}; new Function('require','module','exports','matchMedia',outputFiles[0].text)(createRequire(import.meta.url),mod,mod.exports,()=>({addEventListener(){},matches:false}));
  const store=mod.exports; const originalFetch=globalThis.fetch; const requests=[];
  globalThis.fetch=async (url, options) => {requests.push([url,options]); return new Response(JSON.stringify({expert:[],beginner:{contextLength:'recommended'},engine:{running:false}}),{status:200});};
  try {
    for (const stats of [null,{running:0,waiting:2}]) {
      Object.assign(store.getState(),{connection:'live',serving:stats ? {stats} : null,library:{mode:'real',variants:[],loaded:'fixture'},busy:{}});
      await assert.rejects(store.saveEngine({expert:[]},true,true),/unavailable|queued/);
      assert.equal(await store.loadVariant('fixture'),false);
      assert.equal(await store.unloadModel(),false);
      assert.equal(await store.restartEngine(),false);
      assert.equal(requests.length,0);
      for(const toast of store.getState().toasts) store.holdToast(toast.id);
    }
    await store.saveEngine({expert:[]},true,false);
    assert.equal(requests.length,1); assert.equal(requests[0][1].method,'PATCH');
  } finally {globalThis.fetch=originalFetch; for(const toast of store.getState().toasts)store.holdToast(toast.id);}
});
