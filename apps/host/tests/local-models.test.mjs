import { ROOMY_HOST } from './helpers/hardware.mjs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createServer } from 'node:net';
import { readGGUF } from '../server/gguf.mjs';
import { browseLocalModels, inspectLocalModel, createLocalModels } from '../server/local-models.mjs';
import { ggufFixture } from './helpers/local-gguf.mjs';
const cwd=fileURLToPath(new URL('..',import.meta.url)),pause=ms=>new Promise(r=>setTimeout(r,ms));
function fixture(t){const root=mkdtempSync(join(tmpdir(),'strata-local-model-'));t.after(()=>rmSync(root,{recursive:true,force:true}));return root;}

test('bounded reader gets real metadata and rejects fake extensions, partial tensors, unsupported versions and duplicate metadata',t=>{
 const root=fixture(t),file=join(root,'huihui.gguf');ggufFixture(file,{version:2});const info=readGGUF(file);assert.equal(info.metadata['general.architecture'],'qwen3next');assert.equal(info.metadata['tokenizer.ggml.tokens'].count,1);assert.equal(info.parameters,16);
 const preview=inspectLocalModel(file);assert.equal(preview.variant.quant,'Q4_K_M');assert.equal(preview.variant.contextMax,32768);assert.equal(preview.variant.layerCount,48);assert.equal(preview.variant.experts,128);assert.equal(preview.variant.kvKbPerToken,12);assert.equal(preview.variant.family,'Local models');
 assert.throws(()=>inspectLocalModel(root),/\.gguf|folder/);writeFileSync(file,'not a model');assert.throws(()=>inspectLocalModel(file),/truncated|not a GGUF/);
 ggufFixture(file,{truncate:10});assert.throws(()=>readGGUF(file),/incomplete/);
 ggufFixture(file,{version:1});assert.throws(()=>readGGUF(file),/version/);
 ggufFixture(file,{extra:[['general.name',8,'duplicate']]});assert.throws(()=>readGGUF(file),/duplicate/);
 ggufFixture(file,{arch:'clip'});assert.throws(()=>inspectLocalModel(file),/architecture.*not supported/);
 ggufFixture(file,{tensorCount:0});assert.throws(()=>inspectLocalModel(file),/missing tensors/);
 ggufFixture(file,{context:1024});assert.throws(()=>inspectLocalModel(file),/context_length/);
 const oversized=Buffer.alloc(24);oversized.write('GGUF');oversized.writeUInt32LE(3,4);oversized.writeBigUInt64LE(2n**63n,8);writeFileSync(file,oversized);assert.throws(()=>readGGUF(file),/unsupported integer/);
});

test('choosing any shard finds the first, validates every shard and catches missing or mismatched sets',t=>{
 const root=fixture(t),first=join(root,'Orca-00001-of-00002.gguf'),second=join(root,'Orca-00002-of-00002.gguf');
 ggufFixture(first,{splitNo:0,splitCount:2,tensorName:'a.weight'});ggufFixture(second,{splitNo:1,splitCount:2,tensorName:'b.weight'});
 const model=inspectLocalModel(second);assert.equal(model.variant.local.shards,2);assert.ok(model.variant.local.path.endsWith('00001-of-00002.gguf'));assert.equal(model.variant.source.bytes,statSync(first).size+statSync(second).size);
 rmSync(second);assert.throws(()=>inspectLocalModel(first),/Keep every shard/);
 ggufFixture(second,{splitNo:0,splitCount:2});assert.throws(()=>inspectLocalModel(first),/matching set/);
 ggufFixture(second,{splitNo:1,splitCount:2,arch:'llama'});assert.throws(()=>inspectLocalModel(first),/different architectures/);
 rmSync(first);assert.throws(()=>inspectLocalModel(second),/first shard/);
});

test('folder browsing shows only folders/GGUFs; reference persistence, deduplication and removal preserve original weights',t=>{
 const root=fixture(t),file=join(root,'Orca.gguf');mkdirSync(join(root,'subfolder'));writeFileSync(join(root,'private.txt'),'keep');writeFileSync(join(root,'.hidden.gguf'),'ignore');const bytes=ggufFixture(file);
 const listing=browseLocalModels(root);assert.deepEqual(listing.entries.map(e=>e.name),['subfolder','Orca.gguf']);
 const service=createLocalModels(join(root,'data')),preview=inspectLocalModel(file);service.add(preview);service.add(preview);assert.equal(service.list().length,1);assert.equal(service.ready(service.list()[0]),true);
 const restarted=createLocalModels(join(root,'data'));assert.equal(restarted.list()[0].id,preview.variant.id);
 renameSync(file,join(root,'moved.gguf'));assert.equal(restarted.ready(restarted.list()[0]),false);renameSync(join(root,'moved.gguf'),file);
 restarted.remove(preview.variant.id);assert.deepEqual(readFileSync(file),bytes);assert.equal(readFileSync(join(root,'private.txt'),'utf8'),'keep');
});

test('native library uses the exact imported path and keeps its reference across directory changes',t=>{
 const root=fixture(t),file=join(root,'real-path.gguf');ggufFixture(file);const next=join(root,'next');mkdirSync(next);
 const result=spawnSync(process.execPath,['--input-type=module','-e',`
 import assert from 'node:assert/strict';import fs from 'node:fs';
 const catalog=await import('./server/catalog.mjs'),local=await import('./server/local-models.mjs'),models=await import('./server/models.mjs'),store=await import('./server/store.mjs');
 const preview=local.inspectLocalModel(${JSON.stringify(file)});catalog.localModels.add(preview);models.localModelAdded();
 const v=catalog.variant(preview.variant.id);assert.equal(models.pathFor(v),v.local.path);assert.ok(models.isOnDisk(v.id));assert.equal(models.library().variants.find(row=>row.id===v.id).state,'installed');
 store.changeModelsDirectory(${JSON.stringify(next)});models.init({send(){}});assert.ok(models.isOnDisk(v.id));
 models.remove(v.id);assert.equal(catalog.variant(v.id),null);assert.ok(fs.existsSync(${JSON.stringify(file)}));
 `],{cwd,env:{...process.env,STRATA_HOME:join(root,'host'),STRATA_MODELS_DIR:join(root,'host','models'),STRATA_REAL_ENGINE:'1'},encoding:'utf8'});assert.equal(result.status,0,result.stderr);
});

async function freePort(){const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;await new Promise(r=>server.close(r));return port;}
test('HTTP import leaves the running model intact, loads through normal controls, protects active references and preserves files on removal',{timeout:30000},async t=>{
 const root=fixture(t),home=join(root,'host'),file=join(root,'huihui-Orca.gguf');mkdirSync(join(home,'data'),{recursive:true});ggufFixture(file);
 writeFileSync(join(home,'data','state.json'),JSON.stringify({setup:{complete:true,step:4,flowVersion:2},installed:{'qwen38-q4_k_m':{simulated:true}},loaded:'qwen38-q4_k_m'}));
 const port=await freePort(),base=`http://127.0.0.1:${port}`,host=spawn(process.execPath,['server/index.mjs'],{cwd,env:{...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'),STRATA_REAL_ENGINE:'0',STRATA_TEST_HARDWARE:ROOMY_HOST,STRATA_PORT:String(port),STRATA_BIND:'127.0.0.1'},stdio:'ignore'});
 async function call(path,body,method){const response=await fetch(base+path,{method:method??(body===undefined?'GET':'POST'),headers:{'content-type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});return {status:response.status,body:await response.json()};}
 try {
 let ready=false;for(let i=0;i<100;i++){try{ready=(await call('/api/runtime')).status===200;if(ready)break;}catch{}await pause(100);}assert.ok(ready);
 const original=(await call('/api/runtime')).body.defaultId;
 assert.equal((await call('/api/local-models/inspect',{path:join(root,'missing.gguf')})).status,400);
 const added=await call('/api/local-models',{path:file,name:'My huihui version'});assert.equal(added.status,200,JSON.stringify(added.body));const id=added.body.variantId;
 assert.equal((await call('/api/runtime')).body.defaultId,original);assert.equal((await call('/api/local-models',{path:file})).body.variantId,id);
 const engine=(await call('/api/engine')).body,hardware=(await call('/api/hardware')).body;
 const config={beginner:{...engine.beginner,contextLength:'32k'},expert:engine.expert.map(p=>({id:p.id,value:p.id==='ctx.length'?'32K':p.value})),extra:{headroom:'shared'}};
 const loaded=await call('/api/runtime/load',{variantId:id,gpuIds:hardware.gpus.map(g=>g.id??`${g.vendor}:${g.index}`),config});assert.equal(loaded.status,200,JSON.stringify(loaded.body));assert.equal(loaded.body.state.instances.length,1);
 assert.equal((await call(`/api/models/${id}`,undefined,'DELETE')).status,409);
 ggufFixture(file,{name:'Changed metadata'});assert.equal((await call('/api/local-models',{path:file})).status,409);
 await call(`/api/runtime/${loaded.body.instance.id}/unload`,{});
 const reconnected=await call('/api/local-models',{path:file});assert.equal(reconnected.status,200);assert.equal(reconnected.body.variantId,id);
 assert.equal((await call(`/api/models/${id}`,undefined,'DELETE')).status,200);assert.ok(readFileSync(file).length>0);assert.equal((await call('/api/models')).body.variants.some(v=>v.id===id),false);
 assert.equal((await call('/api/runtime')).body.instances.length,0);
 } finally {host.kill('SIGTERM');await new Promise(r=>host.once('exit',r));}
});
