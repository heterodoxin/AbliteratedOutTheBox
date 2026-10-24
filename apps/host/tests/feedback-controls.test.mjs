import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';
async function moduleAt(path) { const {code}=await transform(await readFile(new URL(path,import.meta.url),'utf8'),{loader:'ts',format:'esm'}); return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`); }
const {dialAngle, dialIndex, unwrapAngle, closestStop}=await moduleAt('../src/lib/tuning-controls.ts');
const {memoryInventory}=await moduleAt('../src/lib/hardware-summary.ts');
test('circular dial follows an arc, clamps its gap, and accepts approximate entry',()=>{
  assert.equal(dialAngle(0,-100),0); assert.equal(dialAngle(100,0),90);
  assert.equal(dialIndex(-135,4),0); assert.equal(dialIndex(135,4),3);
  const overGap=unwrapAngle(-170,170); assert.equal(overGap,190); assert.equal(dialIndex(overGap,4),3);
  assert.equal(dialIndex(unwrapAngle(170,-170),4),0);
  assert.equal(closestStop(5000,[4096,8192,16384,32768]),4096);
  assert.equal(closestStop(30000,[4096,8192,16384,32768]),32768);
});
test('memory inventory counts shared memory once and includes each dedicated GPU',()=>{
  assert.deepEqual(memoryInventory({memory:{totalGb:16},accelerator:{unified:true},gpus:[{vramGb:16,unified:true,backend:'metal'}]}),{ram:16,vram:0,total:16,shared:true});
  assert.deepEqual(memoryInventory({memory:{totalGb:32},accelerator:{unified:false},gpus:[{vramGb:24,unified:false,backend:'cuda'},{vramGb:12,unified:false,backend:'cuda'}]}),{ram:32,vram:36,total:68,shared:false});
  assert.equal(memoryInventory({memory:{totalGb:32},accelerator:{unified:false},gpus:[]}).total,32);
});
