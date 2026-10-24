import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {transform} from 'esbuild';
const {code}=await transform(await readFile(new URL('../src/lib/hardware-summary.ts',import.meta.url),'utf8'),{loader:'ts',format:'esm'});
const {memoryInventory}=await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
test('RAM and every dedicated GPU contribute to the aggregate inventory',()=>{
 const hardware={memory:{totalGb:64},accelerator:{unified:false},gpus:[{unified:false,backend:'cuda',vramGb:24},{unified:false,backend:'cuda',vramGb:16}]};
 assert.deepEqual(memoryInventory(hardware),{ram:64,vram:40,total:104,shared:false});
});
test('unified GPU memory is counted once, even with multiple reported shared GPU pools',()=>{
 const hardware={memory:{totalGb:16},accelerator:{unified:true},gpus:[{unified:true,backend:'metal',vramGb:16},{unified:true,backend:'metal',vramGb:16}]};
 assert.deepEqual(memoryInventory(hardware),{ram:16,vram:0,total:16,shared:true});
});
test('shared GPU memory is excluded when dedicated GPUs are also connected',()=>{
 const hardware={memory:{totalGb:32},accelerator:{unified:true},gpus:[{unified:true,backend:'metal',vramGb:32},{unified:false,backend:'cuda',vramGb:24},{unified:false,backend:'cuda',vramGb:16}]};
 assert.deepEqual(memoryInventory(hardware),{ram:32,vram:40,total:72,shared:false});
});
