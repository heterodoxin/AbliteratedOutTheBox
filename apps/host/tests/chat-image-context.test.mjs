import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {transform} from 'esbuild';
test('repeated photo messages keep the newest pixels inside the request budget, including duplicate references',async()=>{
 const source=await readFile(new URL('../src/lib/chat-photos.ts',import.meta.url),'utf8');const {code}=await transform(source,{loader:'ts',format:'esm'});const module=await import('data:text/javascript;base64,'+Buffer.from(code.replace("import { useSyncExternalStore } from \"react\";",'const useSyncExternalStore=()=>{};')).toString('base64'));
 const messages=[{photos:[{id:'old',name:'old.png'}]},{photos:[{id:'same',name:'x.png'},{id:'same',name:'x.png'}]},{photos:[{id:'new',name:'new.png'}]}];
 assert.deepEqual([...module.photoHistorySlots(messages,1)],['2:0','1:1']);assert.equal(module.photoHistorySlots(messages,3).size,0);assert.equal(module.photoHistorySlots(messages,0).size,3);
});
