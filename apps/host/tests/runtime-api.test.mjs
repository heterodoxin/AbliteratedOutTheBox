import { ROOMY_HOST } from './helpers/hardware.mjs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createServer} from 'node:net';
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function freePort(){const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;await new Promise(r=>server.close(r));return port;}
test('HTTP gateway lists all models, routes both protocols exactly, and isolates lifecycle requests',{timeout:30000},async()=>{
 const home=await mkdtemp(join(tmpdir(),'strata-multiple-model-test-'));await mkdir(join(home,'data'));
 await writeFile(join(home,'data','state.json'),JSON.stringify({installed:{'qwen38-q4_k_m':{installedAt:'2026-10-14',simulated:true},'qwen36-q4_k_m':{installedAt:'2026-10-14',simulated:true}}}));
 const port=await freePort(),base=`http://127.0.0.1:${port}`;
 const host=spawn(process.execPath,['server/index.mjs'],{cwd:fileURLToPath(new URL('..',import.meta.url)),env:{...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'),STRATA_REAL_ENGINE:'0',STRATA_TEST_HARDWARE:ROOMY_HOST,STRATA_PORT:String(port),STRATA_BIND:'127.0.0.1'},stdio:'ignore'});
 async function call(path,body){const response=await fetch(base+path,{method:body===undefined?'GET':'POST',headers:{'content-type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});return {status:response.status,body:await response.json()};}
 try{
  let ready=false;for(let i=0;i<80;i++){try{ready=(await call('/api/runtime')).status===200;if(ready)break;}catch{}await delay(100);}assert.ok(ready,'test host started');
  const engine=(await call('/api/engine')).body;
  const config={beginner:{...engine.beginner,contextLength:'32k'},expert:engine.expert.map(p=>({id:p.id,value:p.id==='ctx.length'?'32K':p.value})),extra:{headroom:'shared'}};
  const hardware=(await call('/api/hardware')).body;
  const a=await call('/api/runtime/load',{variantId:'qwen38-q4_k_m',gpuIds:hardware.gpus.map(g=>g.id??`${g.vendor}:${g.index}`),config,executionMode:'multiple'});assert.equal(a.status,200,JSON.stringify(a.body));
  const b=await call('/api/runtime/load',{variantId:'qwen36-q4_k_m',gpuIds:[],config,executionMode:'multiple'});assert.equal(b.status,200,JSON.stringify(b.body));
  const list=(await call('/v1/models')).body.data;assert.equal(list.length,2);assert.deepEqual(new Set(list.map(i=>i.id)),new Set([a.body.instance.apiModelId,b.body.instance.apiModelId]));
  assert.equal((await call('/v1/chat/completions',{model:'missing',messages:[{role:'user',content:'hello'}]})).status,409);
  assert.equal((await call('/v1/messages',{messages:[{role:'user',content:'hello'}]})).status,409);
  assert.equal((await call('/v1/v1/messages',{messages:[{role:'user',content:'hello'}]})).status,409,'an Anthropic client given the /v1 base reaches the same route');
  assert.notEqual((await call('/v1/v1/chat/completions',{model:'missing',messages:[{role:'user',content:'hello'}]})).status,409,'only the messages path is aliased');
  const controllers=[new AbortController(),new AbortController()];
  const responseA=await fetch(base+'/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({stream:true,model:a.body.instance.apiModelId,messages:[{role:'user',content:'hello'}]}),signal:controllers[0].signal});
  const responseB=await fetch(base+'/v1/messages',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({stream:true,model:b.body.instance.apiModelId,messages:[{role:'user',content:'hello'}]}),signal:controllers[1].signal});
  assert.equal(responseA.status,200);assert.equal(responseB.status,200);
  const state=(await call('/api/serving')).body;assert.equal(state.stats.running,2);assert.deepEqual(new Set(state.requests.map(r=>r.instanceId)),new Set([a.body.instance.id,b.body.instance.id]));
  assert.equal((await call(`/api/runtime/${a.body.instance.id}/unload`,{})).status,409);
  for(const request of state.requests)await call(`/api/serving/requests/${request.id}/cancel`,{});
  controllers.forEach(c=>c.abort());
  for(let i=0;i<30;i++){if(!(await call('/api/serving')).body.stats.running)break;await delay(100);}
  await call(`/api/runtime/${a.body.instance.id}/unload`,{});
  assert.equal((await call('/v1/models')).body.data[0].id,b.body.instance.apiModelId);
  assert.equal((await call('/api/runtime')).body.instances.length,1);
  const switched=await call('/api/runtime/load',{variantId:'qwen38-q4_k_m',gpuIds:[],config});assert.equal(switched.status,200,JSON.stringify(switched.body));
  assert.equal(switched.body.state.instances.length,1);assert.equal(switched.body.state.instances[0].variantId,'qwen38-q4_k_m');assert.equal(switched.body.state.defaultId,switched.body.instance.id);
  const legacy=await call('/api/models/qwen36-q4_k_m/load',{});assert.equal(legacy.status,200,JSON.stringify(legacy.body));
  assert.equal((await call('/v1/models')).body.data.length,1);assert.equal((await call('/api/runtime')).body.instances[0].variantId,'qwen36-q4_k_m');

 }finally{host.kill('SIGTERM');await new Promise(r=>host.once('exit',r));await rm(home,{recursive:true,force:true});}
});
