import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:net';
import { locateModelFile } from '../server/model-files.mjs';
import { createImageModels, IMAGE_CATALOG } from '../server/image-models.mjs';
const cwd = fileURLToPath(new URL('..',import.meta.url));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function fixture(t) { const home=realpathSync(mkdtempSync(join(tmpdir(),'strata-directory-'))); t.after(()=>rmSync(home,{recursive:true,force:true})); return home; }

test('existing flat and Strata-layout weights are found; partial, wrong-header and outside symlinks are rejected', t => {
 const home=fixture(t), folder=join(home,'models'); mkdirSync(folder);
 const bytes=Buffer.from('GGUFsmall-known-fixture'), model={id:'known',source:{file:'known.gguf',bytes:bytes.length}};
 const flat=join(folder,model.source.file); writeFileSync(flat,bytes);
 assert.equal(locateModelFile(folder,model),flat);
 writeFileSync(flat,bytes.subarray(0,5)); assert.equal(locateModelFile(folder,model),null);
 writeFileSync(flat,Buffer.from('bad!small-known-fixture')); assert.equal(locateModelFile(folder,model),null);
 rmSync(flat); const outside=join(home,'external.gguf');writeFileSync(outside,bytes);symlinkSync(outside,flat);
 assert.equal(locateModelFile(folder,model),null);rmSync(flat);
 mkdirSync(join(folder,model.id));const nested=join(folder,model.id,model.source.file);writeFileSync(nested,bytes);
 assert.equal(locateModelFile(folder,model),nested);
});

test('directory preferences survive restart, scope demo installs, and reject bad paths without changing app data', t => {
 const home=fixture(t), next=join(home,'existing models');mkdirSync(next);
 const result=spawnSync(process.execPath,['--input-type=module','-e',`
  import assert from 'node:assert/strict';import {join} from 'node:path';import {existsSync} from 'node:fs';
  const store=await import('./server/store.mjs');const models=await import('./server/models.mjs');
  const original=store.MODELS_DIR,id='qwen38-q4_k_m';store.update({installed:{[id]:{simulated:true}},setup:{complete:true,step:4},serving:{apiKey:'fixture-only-key'}});
  for(const path of ['relative/path',join(store.HOME,'missing'),store.DATA_DIR,store.HOME])assert.throws(()=>store.validateModelsDirectory(path));
  assert.equal(store.MODELS_DIR,original);assert.equal(existsSync(join(store.HOME,'missing')),false);
  store.changeModelsDirectory(${JSON.stringify(next)});models.init({send(){}});
  assert.equal(models.isOnDisk(id),false);assert.equal(store.get().setup.complete,true);assert.equal(store.get().serving.apiKey,'fixture-only-key');
  store.changeModelsDirectory(original);models.init({send(){}});assert.equal(models.isOnDisk(id),true);
  store.changeModelsDirectory(${JSON.stringify(next)});
 `],{cwd,env:{...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'),STRATA_REAL_ENGINE:'0'},encoding:'utf8'});
 assert.equal(result.status,0,result.stderr);
 const restarted=spawnSync(process.execPath,['--input-type=module','-e',`const store=await import('./server/store.mjs');console.log(store.MODELS_DIR);`],{cwd,env:{...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models')},encoding:'utf8'});
 assert.equal(restarted.stdout.trim(),next);
});

test('real library reuses a known flat GGUF and deletion preserves other files', t => {
 const home=fixture(t), next=join(home,'ggufs');mkdirSync(next);
 const result=spawnSync(process.execPath,['--input-type=module','-e',`
  import assert from 'node:assert/strict';import fs from 'node:fs';import {join} from 'node:path';
  const store=await import('./server/store.mjs');const {VARIANTS}=await import('./server/catalog.mjs');const models=await import('./server/models.mjs');
  const v=VARIANTS[0],bytes=Buffer.from('GGUFsmall-catalog-fixture');v.source.bytes=bytes.length;
  const file=join(${JSON.stringify(next)},v.source.file),other=join(${JSON.stringify(next)},'unrelated.gguf');fs.writeFileSync(file,bytes);fs.writeFileSync(other,'keep');
  store.changeModelsDirectory(${JSON.stringify(next)});models.init({send(){}});
  assert.equal(models.isOnDisk(v.id),true);assert.equal(models.pathFor(v),file);assert.equal(models.library().variants[0].state,'installed');
  models.remove(v.id);assert.equal(fs.existsSync(file),false);assert.equal(fs.readFileSync(other,'utf8'),'keep');
 `],{cwd,env:{...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'),STRATA_REAL_ENGINE:'1'},encoding:'utf8'});
 assert.equal(result.status,0,result.stderr);
});

test('image downloads are scoped per directory and existing real image files require the pinned hash', async t => {
 const home=fixture(t), dataDir=join(home,'data'), first=join(home,'first'),second=join(home,'second');mkdirSync(first);mkdirSync(second);
 let directory=first;const demo=createImageModels({dataDir,modelsDir:()=>directory,demoDuration:10});demo.install('q4');await demo.whenIdle();demo.select('q4');demo.apply();
 directory=second;await demo.rescan(true);assert.equal(demo.library().variants[0].state,'available');assert.equal(demo.library().applied,'off');
 directory=first;await demo.rescan(true);assert.equal(demo.library().variants[0].state,'installed');
 const bytes=Buffer.from('GGUFverified-small-image'), catalog={...IMAGE_CATALOG,variants:[{...IMAGE_CATALOG.variants[0],bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')}]};
 writeFileSync(join(second,catalog.variants[0].file),bytes);
 const real=createImageModels({dataDir:join(home,'real-data'),modelsDir:second,real:true,catalog});await real.rescan();assert.equal(real.library().variants[0].state,'installed');
 writeFileSync(join(second,catalog.variants[0].file),Buffer.from('GGUFmodified-small-image'));await real.rescan();assert.equal(real.library().variants[0].state,'available');
});

async function freePort(){const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;await new Promise(r=>server.close(r));return port;}
test('directory API protects loaded models, active requests and downloads, then rescans without moving files',{timeout:30000},async t => {
 const home=fixture(t), next=join(home,'existing');mkdirSync(next);mkdirSync(join(home,'data'));mkdirSync(join(home,'models'));
 const original=join(home,'models'),keep=join(original,'keep.txt');writeFileSync(keep,'preserve');
 writeFileSync(join(home,'data','state.json'),JSON.stringify({installed:{'qwen38-q4_k_m':{simulated:true}},loaded:'qwen38-q4_k_m',setup:{complete:true,flowVersion:2,step:4}}));
 const port=await freePort(),base=`http://127.0.0.1:${port}`;
 const host=spawn(process.execPath,['server/index.mjs'],{cwd,env:{...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'),STRATA_REAL_ENGINE:'0',STRATA_PORT:String(port),STRATA_BIND:'127.0.0.1'},stdio:'ignore'});
 async function call(path,body){const response=await fetch(base+path,{method:body===undefined?'GET':'POST',headers:{'content-type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});return {status:response.status,body:await response.json()};}
 try {
  let ready=false;for(let i=0;i<100;i++){try{ready=(await call('/api/runtime')).status===200;if(ready)break;}catch{}await pause(100);}assert.ok(ready);
  assert.equal((await call('/api/storage/models-directory',{directory:next})).status,409);
  assert.equal((await call('/api/models')).body.disk.modelsDir,original);
  assert.equal((await call('/api/storage/models-directory',{directory:join(home,'missing'),unload:true})).status,400);
  assert.equal((await call('/api/runtime')).body.instances.length,1);
  const same=await call('/api/storage/models-directory',{directory:original});assert.equal(same.status,200);assert.equal(same.body.runtime.instances.length,1);
  const controller=new AbortController();const response=await fetch(base+'/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({stream:true,messages:[{role:'user',content:'hello'}]}),signal:controller.signal});assert.equal(response.status,200);
  assert.equal((await call('/api/storage/models-directory',{directory:next,unload:true})).status,409);
  for(const r of (await call('/api/serving')).body.requests)await call(`/api/serving/requests/${r.id}/cancel`,{});controller.abort();
  for(let i=0;i<30;i++){if(!(await call('/api/serving')).body.stats.running)break;await pause(100);}
  await call('/api/image-models/q4/install',{});assert.equal((await call('/api/storage/models-directory',{directory:next,unload:true})).status,409);await call('/api/image-models/q4/cancel',{});
  await call('/api/models/qwen36-q2_k/install',{});assert.equal((await call('/api/storage/models-directory',{directory:next,unload:true})).status,409);await call('/api/models/qwen36-q2_k/cancel',{});
  const changed=await call('/api/storage/models-directory',{directory:next,unload:true});assert.equal(changed.status,200,JSON.stringify(changed.body));assert.equal(changed.body.runtime.instances.length,0);assert.equal(changed.body.library.disk.modelsDir,next);assert.equal(changed.body.hardware.disk.modelsDir,next);assert.equal(changed.body.setup.app.modelsDir,next);assert.equal(changed.body.setup.complete,true);assert.equal(changed.body.library.variants.find(v=>v.id==='qwen38-q4_k_m').state,'available');assert.equal(readFileSync(keep,'utf8'),'preserve');
  const back=await call('/api/storage/models-directory',{directory:original});assert.equal(back.status,200);assert.equal(back.body.library.variants.find(v=>v.id==='qwen38-q4_k_m').state,'installed');assert.equal(back.body.runtime.instances.length,0);
 } finally {host.kill('SIGTERM');await new Promise(r=>host.once('exit',r));}
});
