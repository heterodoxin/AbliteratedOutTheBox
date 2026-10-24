import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,realpathSync} from 'node:fs';
import {join} from 'node:path';
import {resourceSettings} from '../server/native-resources.mjs';
import {runtimeVision} from '../server/native-vision.mjs';
import {nativeFixture,finishJob} from './helpers/native-fixture.mjs';

test('new native downloads use the selected model root without adding a legacy subfolder',async()=>{
 const directory='/fixture/Documents/Strata/Models',f=nativeFixture({modelsDirectory:()=>directory});try{
  await f.control.refresh();f.installed.models.length=0;await f.control.refresh(true);
  assert.equal(f.control.state().downloadDirectory,directory);
  await f.control.plan({family:'qwen',model:'IQ2_XS'});
  assert.equal(f.rpcCalls.find(c=>c.name==='strata_install').arguments.data_dir,directory);
 }finally{await f.cleanup();}
});

test('work profile reserves GPU memory and CPU workers while preserving model arguments',()=>{
 const hardware={cpu:{cores:10},gpus:[{index:0,vram_gb:24}]};
 const original={gpu:0,args:['--max-context','32768','--pool-workers','10','--vram-reserve-mib','700'],vision:{gpu:true}};
 const work=resourceSettings(original,hardware,'work');
 assert.deepEqual(work.args.slice(0,2),['--max-context','32768']);
 assert.equal(work.args[work.args.indexOf('--pool-workers')+1],'8');
 assert.equal(work.vram_reserve_mib,4915);
 assert.equal(original.args[3],'10');
 const performance=resourceSettings({...original,...work,ui_resource_profile:'work'},hardware,'performance');
 assert.equal(performance.args[performance.args.indexOf('--pool-workers')+1],'10');
 assert.equal(performance.vram_reserve_mib,2048); // The vision encoder still needs its safety reserve.
 assert.throws(()=>resourceSettings(original,hardware,'invalid'),/preference/);
});

test('guided tuning targets the prepared model, persists headroom and opens one measured instance',async()=>{
 const f=nativeFixture();try{
  f.setHardware({cpu:{cores:10},ram_gb:128,gpus:[{index:0,name:'NVIDIA RTX 4090',vram_gb:24}]});
  await f.control.refresh(true);
  await finishJob(f.control,f.control.operation('load',{model:'iq3_s'}).id);
  const tuned=await finishJob(f.control,f.control.operation('tune',{model:'iq2_xs',profile:'work',calibrate:true}).id);
  assert.equal(tuned.state,'done',tuned.error);
  assert.ok(f.calls.find(call=>call.args[0]==='-c'&&call.args[2].endsWith('strata-iq2_xs.json')));
  assert.equal(f.control.state().instances.length,1);
  assert.equal(f.control.state().instances[0].model,'iq2_xs');
  assert.equal(tuned.result.outputTps,41);
  const config=JSON.parse(readFileSync(join(f.repo,'strata-iq2_xs.json')));
  assert.equal(config.ui_resource_profile,'work');assert.equal(config.vram_reserve_mib,4915);
  assert.deepEqual(config.unknown_native_key,{preserve:true});
  assert.equal(config.args[config.args.indexOf('--pool-workers')+1],'8');
  assert.ok(readFileSync(join(f.repo,'strata-iq2_xs.json.bak')));
 }finally{await f.cleanup();}
});

test('quick setup applies resource settings without invoking long calibration',async()=>{
 const f=nativeFixture();try{
  await f.control.refresh();
  const tuned=await finishJob(f.control,f.control.operation('tune',{model:'iq2_xs',profile:'performance',calibrate:false}).id);
  assert.equal(tuned.state,'done',tuned.error);
  assert.equal(f.calls.some(c=>c.args[0]==='-c'),false);
  assert.equal(f.control.benchmarkReport().source,'engine');
 }finally{await f.cleanup();}
});

test('an existing native model folder is persisted and used by the installer without moving files',async()=>{
 const f=nativeFixture();try{
  await f.control.refresh();f.installed.models.length=0;await f.control.refresh(true);
  f.control.setDownloadDirectory(f.directory);
  assert.equal(f.control.state().downloadDirectory,realpathSync(f.directory));
  await f.control.plan({family:'qwen',model:'IQ2_XS'});
  assert.equal(f.rpcCalls.find(c=>c.name==='strata_install').arguments.data_dir,realpathSync(f.directory));
  assert.throws(()=>f.control.setDownloadDirectory(join(f.repo,'setup.py')),/folder/);
 }finally{await f.cleanup();}
});

test('preparation completes without loading until the user chooses a resource preference',async()=>{
 const f=nativeFixture();try{
  await f.control.refresh();
  const prepared=await finishJob(f.control,f.control.operation('install',{family:'qwen',model:'IQ2_XS',loadAfter:false}).id);
  assert.equal(prepared.state,'done',prepared.error);assert.equal(f.control.state().instances.length,0);
  assert.equal(f.calls.some(c=>c.args[0]?.replaceAll('\\','/').endsWith('serve/server.py')),false);
 }finally{await f.cleanup();}
});

test('stopping long tuning retains files and does not continue into loading',async()=>{
 const f=nativeFixture();try{
  await f.control.refresh();f.setProgramWait(true);
  const job=f.control.operation('tune',{model:'iq2_xs',profile:'work'});
  while(!f.calls.some(c=>c.args[0]==='-c'))await new Promise(r=>setTimeout(r,5));
  await f.control.cancel(job.id);await new Promise(r=>setTimeout(r,10));
  assert.equal(f.control.state().jobs.find(j=>j.id===job.id).state,'cancelled');
  assert.equal(f.control.state().instances.length,0);
  assert.ok(readFileSync(join(f.repo,'strata-iq2_xs.json')));
  assert.equal(f.calls.some(c=>c.args[0]?.replaceAll('\\','/').endsWith('serve/server.py')),false);
 }finally{await f.cleanup();}
});

test('the consumer can change resource preference later and restore full performance',async()=>{
 const f=nativeFixture();try{
  f.setHardware({cpu:{cores:10},gpus:[{index:0,name:'NVIDIA RTX 4090',vram_gb:24}]});await f.control.refresh();
  const work=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',resourceProfile:'work'}).id);
  assert.equal(work.state,'done',work.error);
  assert.equal(f.control.state().installed.models[0].resourceProfile,'work');
  const performance=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',resourceProfile:'performance'}).id);
  assert.equal(performance.state,'done',performance.error);assert.equal(f.control.state().instances.length,1);
  const config=JSON.parse(readFileSync(join(f.repo,'strata-iq2_xs.json')));
  assert.equal(config.ui_resource_profile,'performance');assert.equal(config.vram_reserve_mib,1024); // the fixture's picture reader is the CPU one (gpu:0): no VRAM, so the 24 GB card's 1024 floor
  assert.equal(config.args.includes('--pool-workers'),false);
 }finally{await f.cleanup();}
});

test('stopping during the health check cannot leave a cancelled model loaded',async()=>{
 const f=nativeFixture();try{
  await f.control.refresh();f.setHealthWait(true);
  const job=f.control.operation('load',{model:'iq2_xs',resourceProfile:'work'});
  while(!f.hasPendingHealth())await new Promise(r=>setTimeout(r,5));
  await f.control.cancel(job.id);f.releaseHealth();await new Promise(r=>setTimeout(r,20));
  assert.equal(f.control.state().instances.length,0);
  assert.equal(f.control.state().jobs.find(j=>j.id===job.id).state,'cancelled');
 }finally{f.releaseHealth();await f.cleanup();}
});

test('vision can be disabled at load without losing the prepared encoder config or triggering install',async()=>{
 const f=nativeFixture();try{
  await f.control.refresh();
  const off=await finishJob(f.control,f.control.operation('tune',{model:'iq2_xs',profile:'work',calibrate:false,vision:'no'}).id);
  assert.equal(off.state,'done',off.error);assert.equal(f.control.state().instances[0].vision,false);
  const source=JSON.parse(readFileSync(join(f.repo,'strata-iq2_xs.json')));
  assert.deepEqual(source.vision,{gpu:0});assert.equal(source.ui_vision_enabled,false);
  assert.equal(f.control.state().installed.models[0].visionEnabled,false);
  const proc=f.calls.findLast(c=>c.args[0]?.replaceAll('\\','/').endsWith('serve/server.py'));
  assert.equal(JSON.parse(readFileSync(proc.args[proc.args.indexOf('--config')+1])).vision,undefined);
  const again=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',vision:'no'}).id);
  assert.equal(again.state,'done',again.error);assert.equal(f.calls.filter(c=>c.args[0]?.replaceAll('\\','/').endsWith('serve/server.py')).length,1);
  const on=await finishJob(f.control,f.control.operation('load',{model:'iq2_xs',vision:'cpu'}).id);
  assert.equal(on.state,'done',on.error);assert.equal(f.control.state().instances[0].vision,true);
  const runtime=f.calls.findLast(c=>c.args[0]?.replaceAll('\\','/').endsWith('serve/server.py'));
  assert.equal(JSON.parse(readFileSync(runtime.args[runtime.args.indexOf('--config')+1])).vision.gpu,false);
  assert.equal(f.control.state().installed.models[0].visionEnabled,true);
  assert.equal(f.control.state().installed.models[0].visionDevice,'cpu');
  assert.equal(f.rpcCalls.some(c=>c.name==='strata_install'),false);
 }finally{await f.cleanup();}
});

test('runtime vision checks prepared files, validates mode, and leaves the source config intact',()=>{
 const config={vision:{mmproj:'/models/mmproj.gguf',gpu:true,max_tokens:8192},ui_vision_enabled:false};
 const before=structuredClone(config);
 assert.equal(runtimeVision(config),undefined);
 assert.equal(runtimeVision(config,'cpu').gpu,false);
 assert.equal(runtimeVision(config,'yes').mmproj,config.vision.mmproj);
 assert.deepEqual(config,before);
 const amd={vision:{exe:'/e/strata-vision-hip',cpu_exe:'/e/strata-vision',mmproj:'/m.gguf',gpu:true}};
 assert.deepEqual(runtimeVision(amd,'cpu'),{exe:'/e/strata-vision',mmproj:'/m.gguf',gpu:false});
 assert.equal(runtimeVision(amd,'gpu').exe,'/e/strata-vision-hip');assert.equal(runtimeVision(amd,'gpu').cpu_exe,'/e/strata-vision');
 assert.equal(runtimeVision(amd,'yes').gpu,true);
 assert.throws(()=>runtimeVision({},'yes'),/Prepare/);
 assert.throws(()=>runtimeVision(config,'invalid'),/picture reading/);
});

test('download-and-load prepares vision automatically while loading text only when requested',async()=>{
 const f=nativeFixture();try{
  await f.control.refresh();
  const job=await finishJob(f.control,f.control.operation('install',{family:'qwen',model:'IQ2_XS',vision:'yes',loadVision:'no',loadAfter:true}).id);
  assert.equal(job.state,'done',job.error);
  const install=f.rpcCalls.find(c=>c.name==='strata_install'&&c.arguments.confirm);
  assert.equal(install.arguments.vision,'yes');assert.equal(install.arguments.loadVision,undefined);
  assert.equal(f.control.state().instances[0].vision,false);
  assert.deepEqual(JSON.parse(readFileSync(join(f.repo,'strata-iq2_xs.json'))).vision,{gpu:0});
 }finally{await f.cleanup();}
});

test('a card that shares the PC memory does not hold back a fifth of the shared pool as graphics memory (#3510)',()=>{
 const original={gpu:0,args:['--max-context','131072','--pool-workers','10'],vision:{gpu:true}};
 const shared={cpu:{cores:16},ram_gb:128,gpus:[{index:0,vram_gb:82,uma:true,name:'AMD Radeon 8060S'}]};
 const apu=resourceSettings(original,shared,'work');
 assert.equal(apu.vram_reserve_mib,2048,'the display floor with pictures, not 20% of 82 GB (16 GB)');
 const discrete=resourceSettings(original,{cpu:{cores:16},ram_gb:128,gpus:[{index:0,vram_gb:82,name:'a card'}]},'work');
 assert.equal(discrete.vram_reserve_mib,Math.floor(82*1024*.2),'a card with its own memory keeps its fifth');
});
