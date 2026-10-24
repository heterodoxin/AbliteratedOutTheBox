import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,readdirSync,mkdirSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {nativeFixture,finishJob} from './helpers/native-fixture.mjs';
test('native instances isolate devices, ports, IDs and private keys; default loading switches one model',async()=>{
 const f=nativeFixture();try{
  await f.control.refresh(true);const load=async args=>{const job=f.control.operation('load',args);const done=await finishJob(f.control,job.id);assert.equal(done.state,'done',done.error);return f.control.resolveInstance(done.result.instanceId);};
  const a=await load({model:'iq2_xs',gpus:[0]}),b=await load({model:'iq3_s',gpus:[1],multiple:true});assert.equal(f.control.state().instances.length,2);assert.notEqual(a.port,b.port);assert.notEqual(a.key,b.key);assert.deepEqual(b.gpus,[1]);assert.throws(()=>f.control.resolveInstance('chat'),/more than one/);assert.throws(()=>f.control.resolveInstance(),/Choose/);
  assert.equal(JSON.stringify(f.control.state()).includes(a.key),false);
  const failed=await finishJob(f.control,f.control.operation('load',{model:'iq3_s',gpus:[99]}).id);assert.equal(failed.state,'error');assert.equal(f.control.state().instances.length,2);
  f.requests.push({instanceId:a.id,state:'generating'});await assert.rejects(f.control.unload(a.id),/Finish/);f.requests.length=0;
  await f.control.unload(b.id);const c=await load({model:'iq2_xs',gpus:[1],multiple:true});assert.equal(c.apiModelId,'iq2_xs--2');const same=await load({model:'iq2_xs',gpus:[1],multiple:true});assert.equal(same.id,c.id);const beforeCount=f.control.state().instances.length;const reloaded=await load({model:'iq2_xs',gpus:[1],multiple:true,values:{parallel:2}});assert.equal(reloaded.id,c.id);assert.equal(reloaded.apiModelId,c.apiModelId);assert.equal(f.control.state().instances.length,beforeCount);assert.deepEqual(f.control.resolveInstance(a.apiModelId).gpus,[0]);
  await load({model:'iq3_s',gpus:[0]});assert.equal(f.control.state().instances.length,1);assert.equal(readdirSync(join(f.directory,'native-instances')).length,1);
 }finally{await f.cleanup();}
});
test('native save preserves unknown fields, credentials and a backup; null restores native defaults',async()=>{
 const f=nativeFixture();try{await f.control.refresh();const original=f.control.settings('iq2_xs');assert.equal(original.values.env.STRATA_TOKEN,'••••••••');f.control.save('iq2_xs',{values:{...original.values,parallel:2,reasoning_budget_tokens:null},mcp:{files:{command:'node',args:['file-server.mjs'],env:{TOKEN:'secret'}}}});
 const saved=JSON.parse(readFileSync(join(f.repo,'strata-iq2_xs.json')));assert.deepEqual(saved.unknown_native_key,{preserve:true});assert.equal(saved.env.PATH,'/native/bin');assert.equal(saved.env.STRATA_TOKEN,'private');assert.equal(saved.parallel,2);assert.equal('reasoning_budget_tokens' in saved,false);assert.ok(readFileSync(join(f.repo,'strata-iq2_xs.json.bak')));
 assert.equal(f.control.settings('iq2_xs').mcp.files.env.TOKEN,'••••••••');
 }finally{await f.cleanup();}
});
test('benchmarks retain reported rates and cancellation aborts measurement',async()=>{
 const f=nativeFixture();try{await f.control.refresh();const loaded=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);assert.equal(loaded.state,'done');const measured=await finishJob(f.control,f.control.operation('benchmark',{instanceId:loaded.result.instanceId}).id);assert.equal(measured.result.outputTps,41);assert.equal(measured.result.promptTps,1000);assert.equal(f.control.benchmarkReport().source,'engine');
 f.setBenchmarkWait(true);const job=f.control.operation('benchmark',{instanceId:loaded.result.instanceId});await new Promise(r=>setTimeout(r,10));await f.control.cancel(job.id);assert.equal(f.control.state().jobs.find(j=>j.id===job.id).state,'cancelled');assert.equal(f.control.benchmarkReport().outputTps,41);
 }finally{await f.cleanup();}
});
test('native update refuses local edits; calibration targets the chosen installed configuration',async()=>{
 const f=nativeFixture();try{await f.control.refresh();const update=await finishJob(f.control,f.control.operation('update-strata').id);assert.equal(update.state,'error');assert.match(update.error,/local edits/);assert.equal(f.calls.some(c=>c.command==='git'&&c.args[0]==='pull'),false);
 const calibrated=await finishJob(f.control,f.control.operation('calibrate',{model:'iq3_s'}).id);assert.equal(calibrated.state,'done',calibrated.error);assert.ok(f.calls.find(c=>c.args[0]==='-c'&&c.args[2].endsWith('strata-iq3_s.json')));
 f.setExternal([{port:8080,model:'other'}]);await f.control.refresh(true);const load=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs'}).id);assert.equal(load.state,'error');assert.match(load.error,/outside this UI/);
 }finally{await f.cleanup();}
});

test('cancelling a native load before dispatch preserves the running instance',async()=>{const f=nativeFixture();try{await f.control.refresh();const first=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',gpus:[0]}).id);const job=f.control.operation('load',{model:'iq3_s',gpus:[1]});await f.control.cancel(job.id);await new Promise(r=>setTimeout(r,20));assert.equal(f.control.state().instances.length,1);assert.equal(f.control.state().instances[0].id,first.result.instanceId);assert.equal(f.control.state().jobs.find(j=>j.id===job.id).state,'cancelled');}finally{await f.cleanup();}});

test('retuning or repairing an installed native model reuses its original data folder',async()=>{const f=nativeFixture();try{await f.control.refresh();await f.control.plan({family:'qwen',model:'IQ2_XS',context:131072});const call=f.rpcCalls.find(c=>c.name==='strata_install');assert.equal(call.arguments.data_dir,f.installed.data_folder);assert.equal(call.arguments.confirm,false);}finally{await f.cleanup();}});

test('a model the person pointed at their own GGUF folder keeps that folder for the next load: setup is not asked to download it again (#3199)',async()=>{const f=nativeFixture();try{
 const own=join(f.directory,'my-ggufs');mkdirSync(own,{recursive:true});const shard=join(own,'Qwen3.8-Flash-Next-IQ2_XS-00001-of-00002.gguf');writeFileSync(shard,'x');
 const path=join(f.repo,'strata-iq2_xs.json'),config=JSON.parse(readFileSync(path));config.args=['--native',shard,'--max-context','32768'];writeFileSync(path,JSON.stringify(config));
 await f.control.refresh(true);await f.control.plan({family:'qwen',model:'IQ2_XS',context:131072});
 const call=f.rpcCalls.filter(c=>c.name==='strata_install').pop();assert.equal(call.arguments.gguf_dir,own);assert.equal(call.arguments.data_dir,undefined,'the person\'s folder is not the data folder');
 await f.control.plan({family:'qwen',model:'IQ3_S',context:131072});const other=f.rpcCalls.filter(c=>c.name==='strata_install').pop();assert.equal(other.arguments.gguf_dir,undefined,'a model Strata downloaded itself keeps its normal folders');
}finally{await f.cleanup();}});
