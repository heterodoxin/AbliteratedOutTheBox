import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import {createRuntimeModels,reservationBudget,rebalanceReservations} from '../server/runtime-models.mjs';
const hw={memory:{totalGb:64},gpus:[{id:'gpu-a',index:0,name:'GPU A',vramGb:24,unified:false},{id:'gpu-b',index:1,name:'GPU B',vramGb:24,unified:false}]};
const model={id:'chat-q4',family:'Chat',name:'Balanced',quant:'Q4',state:'installed',diskGb:14,kvKbPerToken:24,contextMax:262144,experts:0};
const config={beginner:{contextLength:'32k'},expert:[{id:'gpu.layers',value:'Automatic'},{id:'ctx.kvMode',value:'Q8_0'},{id:'mem.mmap',value:true}],extra:{headroom:'shared'}};
function fixture(hardware=hw,real=false){let log=[];const service=createRuntimeModels({hardware:()=>hardware,library:()=>({variants:[model]}),requests:()=>log,real});return {service,setLog:value=>{log=value;},load:(gpuIds,patch={})=>service.load({variantId:model.id,config,gpuIds,executionMode:'multiple',...patch})};}
test('separate GPU instances retain identity, context and assignments; identical load is idempotent',()=>{
 const {service,load}=fixture();const a=load(['gpu-a']).instance,b=load(['gpu-b']).instance;
 assert.notEqual(a.id,b.id);assert.notEqual(a.apiModelId,b.apiModelId);assert.equal(service.state().instances.length,2);
 assert.deepEqual(service.resolve(a.apiModelId).gpuIds,['gpu-a']);assert.deepEqual(service.resolve(b.apiModelId).gpuIds,['gpu-b']);
 assert.equal(load(['gpu-a']).instance.id,a.id);assert.throws(()=>service.resolve(),/Choose a model ID/);assert.throws(()=>service.resolve('unloaded'),/not loaded/);
 assert.equal(service.resolve(undefined,{chatDefault:true}).id,a.id);
 service.setDefault(b.id);assert.equal(service.resolve(undefined,{chatDefault:true}).id,b.id);
});
test('per-instance queues execute across GPUs concurrently and serialize within an instance',async()=>{
 const {service,load}=fixture();const a=load(['gpu-a']).instance,b=load(['gpu-b']).instance;
 const signal=new AbortController().signal;let release;const gate=new Promise(r=>release=r);const started=[];
 const first=service.queue(a.id).run(signal,async()=>{started.push('a1');await gate;});
 const second=service.queue(a.id).run(signal,async()=>started.push('a2'));
 await service.queue(b.id).run(signal,async()=>started.push('b1'));
 assert.deepEqual(started,['a1','b1']);release();await Promise.all([first,second]);assert.deepEqual(started,['a1','b1','a2']);
});
test('unload blocks only the target instance and preserves the other model',()=>{
 const {service,load,setLog}=fixture();const a=load(['gpu-a']).instance,b=load(['gpu-b']).instance;
 setLog([{instanceId:a.id,state:'generating'}]);assert.throws(()=>service.unload(a.id),/requests/);
 service.unload(b.id);assert.equal(service.state().instances.length,1);assert.equal(service.resolve(a.apiModelId).id,a.id);
 setLog([]);service.unload(a.id);assert.equal(service.state().defaultId,null);assert.throws(()=>service.resolve(a.apiModelId),/not loaded/);
});
test('dedicated VRAM budgets cannot be borrowed from an unselected GPU',()=>{
 const {load}=fixture();load(['gpu-a']);assert.throws(()=>load(['gpu-a','gpu-b'],{config:{...config,beginner:{contextLength:'256k'}}}),/reserved this memory/);
 assert.doesNotThrow(()=>load(['gpu-b']));assert.throws(()=>load(['missing']),/detected GPUs/);assert.throws(()=>load(['gpu-a','gpu-a']),/detected GPUs/);
});
test('unified memory is one RAM pool; pageable weights share space and fixed allocations cannot spill',()=>{
 const unified={memory:{totalGb:16},gpus:[{id:'metal',name:'Apple',vramGb:16,unified:true}]};
 const {service,load}=fixture(unified);load(['metal']);load([]);
 const state=service.state();assert.equal(state.memory.totalGb,16);assert.equal(state.memory.budgetGb,12);assert.equal(state.memory.devices.length,0);assert.ok(state.memory.ramGb<=12.01);assert.ok(state.instances.every(i=>i.memory.spillGb>0));
 const rows=[{memory:{reserveGb:4,ramGb:10,minimumRamGb:10,pageable:true,devices:[]}} , {memory:{reserveGb:4,ramGb:10,minimumRamGb:10,pageable:true,devices:[]}}];
 assert.equal(rebalanceReservations(rows,unified),null);assert.equal(reservationBudget(rows,unified).fits,false);
});
test('load is rejected when unavailable, unsupported, or native multi-instance capability is absent',()=>{
 const {load}=fixture(hw,true);assert.throws(()=>load(['gpu-a']),/not supported/);
 const {load:demoLoad}=fixture();assert.throws(()=>demoLoad(['gpu-a'],{config:{...config,beginner:{contextLength:'1024k'}}}),/supported conversation context/);
 const service=createRuntimeModels({hardware:()=>hw,library:()=>({variants:[{...model,state:'available'}]})});assert.throws(()=>service.load({variantId:model.id,config,gpuIds:['gpu-a']}),/Download/);
});

// Monitor scoping must never carry one model's context or cache into another.
import {build} from 'esbuild';
const bundled=await build({entryPoints:[fileURLToPath(new URL('../src/lib/runtime-models.ts',import.meta.url))],bundle:true,write:false,platform:'node',format:'esm'});
const {scopedMonitor}=await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);
test('Monitor keeps per-model requests/context separate and host RAM readings shared',()=>{
 const {service,load}=fixture();const a=load(['gpu-a']).instance,b=load(['gpu-b']).instance;
 service.update(a.id,{context:{usedTokens:100,maxTokens:32768,pct:.3}});
 const live={phase:'idle',telemetry:{context:'reported'},context:{usedTokens:999},hardware:{ramUsedGb:8},history:[],totals:{tokensIn:999,tokensOut:999,requests:8},conversationCache:{source:'engine'},expertCache:{used:999}};
 const requests=[{instanceId:a.id,state:'done',tokensIn:100,tokensOut:20},{instanceId:b.id,state:'done',tokensIn:7,tokensOut:3}];
 const scoped=scopedMonitor(live,service.state(),b.id,requests);
 assert.equal(scoped.context.usedTokens,0);assert.equal(scoped.requests.length,1);assert.equal(scoped.totals.tokensIn,7);assert.equal(scoped.hardware.ramUsedGb,8);assert.equal(scoped.conversationCache,undefined);
 const all=scopedMonitor(live,service.state(),'all',requests);assert.equal(all.context.maxTokens,0);assert.equal(all.totals.requests,2);assert.equal(all.totals.tokensOut,23);
});

test('changed settings reload only the matching instance and keep its routing identity',()=>{
 const {service,load,setLog}=fixture();const a=load(['gpu-a']).instance,b=load(['gpu-b']).instance;
 const changed={...config,beginner:{contextLength:'64k'}};
 setLog([{instanceId:a.id,state:'generating'}]);assert.throws(()=>load(['gpu-a'],{config:changed}),/before reloading/);
 setLog([{instanceId:b.id,state:'generating'}]);const reloaded=load(['gpu-a'],{config:changed}).instance;
 assert.equal(reloaded.id,a.id);assert.equal(reloaded.apiModelId,a.apiModelId);assert.equal(reloaded.memory.context,65536);assert.equal(service.state().instances.length,2);assert.equal(service.resolve(b.apiModelId).memory.context,32768);
});
test('native single-engine projection retains CPU placement and reload identity',()=>{
 const {service}=fixture(hw,true);
 const cpu={...config,expert:config.expert.map(p=>p.id==='gpu.layers'?{...p,value:'CPU only'}:p)};
 service.adopt(model.id,cpu);const first=service.state().instances[0];assert.deepEqual(first.gpuIds,[]);assert.equal(service.state().capabilities.multipleModels,false);
 service.adopt(model.id,{...cpu,beginner:{contextLength:'64k'}});assert.equal(service.state().instances[0].id,first.id);assert.equal(service.state().instances[0].memory.context,65536);
 service.adopt(null,cpu);assert.equal(service.state().instances.length,0);
});

test('single is the default and switching models releases previous instances before routing',()=>{
 const {service,load}=fixture();const a=load(['gpu-a']).instance,b=load(['gpu-b']).instance;
 const body={variantId:model.id,config,gpuIds:['gpu-a']};
 const preview=service.plan(body);assert.equal(preview.executionMode,'single');assert.equal(preview.budget.devices[1].usedGb,0);assert.equal(service.state().instances.length,2);
 const selected=service.load(body).instance;assert.equal(selected.id,a.id);assert.equal(service.state().instances.length,1);assert.equal(service.state().defaultId,a.id);
 assert.throws(()=>service.resolve(b.apiModelId),/not loaded/);assert.equal(service.resolve().id,a.id);
 const switched=service.load({...body,gpuIds:['gpu-b']}).instance;assert.equal(service.state().instances.length,1);assert.deepEqual(switched.gpuIds,['gpu-b']);assert.equal(service.resolve().id,switched.id);
});
test('single-mode validation and busy requests leave every existing model intact',()=>{
 const {service,load,setLog}=fixture();const a=load(['gpu-a']).instance,b=load(['gpu-b']).instance;
 const body={variantId:model.id,config,gpuIds:['gpu-b']};
 setLog([{instanceId:a.id,state:'queued'}]);assert.throws(()=>service.load(body),/before switching/);assert.equal(service.state().instances.length,2);assert.equal(service.state().defaultId,a.id);
 setLog([]);assert.throws(()=>service.load({...body,config:{...config,beginner:{contextLength:'1024k'}}}),/supported conversation context/);assert.equal(service.resolve(b.apiModelId).id,b.id);
 assert.throws(()=>service.load({...body,executionMode:'invalid'}),/execution/);
});
test('single-mode planning fits the selected model independently of siblings and explicit opt-in still adds models',()=>{
 const unified={memory:{totalGb:16},gpus:[{id:'metal',name:'Apple',vramGb:16,unified:true}]};
 const {service}=fixture(unified);
 const large={...config,beginner:{contextLength:'256k'}};
 const body={variantId:model.id,config:large,gpuIds:['metal']};service.load(body);
 assert.equal(service.plan({...body,gpuIds:[]}).allowed,true);assert.equal(service.plan({...body,gpuIds:[],executionMode:'multiple'}).allowed,false);
 assert.throws(()=>service.load({...body,gpuIds:[],executionMode:'multiple'}),/reserved this memory/);assert.equal(service.state().instances.length,1);
 service.load({...body,gpuIds:[]});assert.equal(service.state().instances.length,1);assert.deepEqual(service.state().instances[0].gpuIds,[]);
});
