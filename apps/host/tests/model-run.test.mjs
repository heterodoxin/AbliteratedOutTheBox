import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';
async function loadTs(path) {
  const {code}=await transform(await readFile(new URL(path,import.meta.url),'utf8'),{loader:'ts',format:'esm'});
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}
const profiles=await loadTs('../src/lib/run-profiles.ts');
const memory=await loadTs('../src/lib/run-memory.ts');
const engine={expert:[{id:'ctx.length',recommended:'128K'},{id:'perf.threads',recommended:8},{id:'mem.mmap',recommended:true}]};
const hardware={host:{name:'Test PC',platform:'darwin',arch:'arm64'},cpu:{model:'Test CPU',cores:10},memory:{totalGb:16},gpus:[{id:'a',name:'Unified GPU',vramGb:16,backend:'metal',unified:true}],accelerator:{backend:'metal',unified:true}};
const experience={source:'demo',config:{placement:'Automatic'},reading:{low:100,high:120},answer:{low:20,high:30},memory:{weights:19,cache:2.7,runtime:2.5,total:24.2}};

test('saved settings retain the optional image choice and older profiles restore Off',()=>{
  const config=profiles.recommendedConfig(engine,hardware);
  const profile={id:'images',name:'With images',variantId:'q4',hardware:profiles.computerIdentity(hardware),engineVersion:'1',config,imageModel:'q6',updatedAt:'2026-10-12'};
  const saved=profiles.saveProfile(profiles.emptyProfiles(),profile,true);
  assert.equal(profiles.readProfiles(JSON.stringify(saved)).profiles[0].imageModel,'q6');
  const older=structuredClone(saved);delete older.profiles[0].imageModel;
  assert.equal(profiles.readProfiles(JSON.stringify(older)).profiles[0].imageModel,'off');
  older.profiles[0].imageModel='unknown-version';
  assert.equal(profiles.readProfiles(JSON.stringify(older)).profiles[0].imageModel,'off');
});

test('profiles retain custom expert values and defaults belong to an exact version on a specific computer',()=>{
  const config=profiles.recommendedConfig(engine,hardware,'256k');config.expert['sample.temp']=.37;
  const computer=profiles.computerIdentity(hardware);
  const profile={id:'daily',name:'Everyday',variantId:'q4',hardware:computer,engineVersion:'1',config,updatedAt:'2026-10-10'};
  const store=profiles.saveProfile(profiles.emptyProfiles(),profile,true);
  config.expert['sample.temp']=1; // A later draft must not alter the saved profile.
  const restored=profiles.readProfiles(JSON.stringify(store));
  assert.equal(restored.profiles[0].config.expert['sample.temp'],.37);
  assert.equal(restored.profiles[0].config.expert['ctx.length'],'256K');
  assert.equal(restored.defaults[profiles.defaultProfileKey(computer,'q4')],'daily');
  assert.equal(restored.defaults[profiles.defaultProfileKey(computer,'q2')],undefined);
  const different=profiles.computerIdentity({...hardware,memory:{totalGb:32}});
  assert.equal(restored.defaults[profiles.defaultProfileKey(different,'q4')],undefined);
  assert.equal(profiles.saveProfile(restored,restored.profiles[0],false).defaults[profiles.defaultProfileKey(computer,'q4')],undefined);
});
test('resource choices preserve expert tuning and reset returns to 128K with hardware-aware CPU headroom',()=>{
  const config=profiles.recommendedConfig(engine,hardware,'512k');config.expert['sample.temp']=.3;
  const full=profiles.resourceConfig(config,'full',hardware);
  assert.equal(full.expert['perf.threads'],10);assert.equal(full.expert['ctx.length'],'512K');assert.equal(full.expert['sample.temp'],.3);
  const recommended=profiles.recommendedConfig(engine,hardware);
  assert.equal(recommended.beginner.contextLength,'recommended');assert.equal(recommended.expert['ctx.length'],'128K');assert.equal(recommended.expert['perf.threads'],8);
  const dual=profiles.resourceConfig(config,'shared',{...hardware,cpu:{...hardware.cpu,cores:48}});
  assert.equal(dual.expert['perf.threads'],38);
});
test('unified memory is counted once; SSD plans change with context and computer use',()=>{
  const shared=memory.planMemory(experience,hardware,'shared');const full=memory.planMemory(experience,hardware,'full');
  assert.equal(shared.gpu,0);assert.equal(shared.budget,12);assert.equal(full.budget,14);
  assert.ok(Math.abs(shared.ram+shared.ssd-shared.total)<.001);
  assert.ok(full.ssd<shared.ssd);
  const shorter=memory.planMemory({...experience,memory:{...experience.memory,cache:.7,total:22.2}},hardware,'shared');
  assert.ok(shorter.ssd<shared.ssd);
  const tooMuchCache=memory.planMemory({...experience,memory:{...experience.memory,cache:20}},hardware,'shared');
  assert.equal(tooMuchCache.cacheFits,false);assert.equal(tooMuchCache.ssd,0,'KV cache overflow must not be represented as disk-backed weights');
  const parallel=memory.planMemory(experience,hardware,'shared',4);assert.equal(parallel.cacheFits,false);
  const locked=memory.planMemory(experience,hardware,'shared',1,false);assert.equal(locked.ssd,0);assert.ok(locked.total>locked.budget,'Locked model memory must not be presented as an SSD spill plan');
});
test('multi-GPU planning retains individual budgets and respects CPU-only and unknown partial placement',()=>{
  const discrete={...hardware,memory:{totalGb:32},accelerator:{backend:'cuda',unified:false},gpus:[{id:'a',name:'GPU A',vramGb:24,backend:'cuda'},{id:'b',name:'GPU B',vramGb:12,backend:'cuda'}]};
  const plan=memory.planMemory(experience,discrete,'shared');
  assert.equal(plan.devices.length,2);assert.equal(plan.budget,24+21.6+10.8);assert.equal(plan.ssd,0);
  const cpu=memory.planMemory({...experience,config:{placement:'CPU only'}},discrete,'shared');assert.equal(cpu.gpu,0);assert.equal(cpu.devices.length,0);assert.equal(cpu.budget,24);
  assert.equal(memory.planMemory({...experience,config:{placement:'Half'}},discrete,'shared'),null);
});
test('illustrative spill penalties never modify or impersonate measured performance',()=>{
  const plan=memory.planMemory(experience,hardware,'shared');
  assert.ok(memory.demoRunExperience(experience,plan,'shared').answer.low<experience.answer.low);
  const measured={...experience,source:'measured'};
  assert.strictEqual(memory.demoRunExperience(measured,plan,'full'),measured);
  assert.deepEqual(experience.answer,{low:20,high:30});
});
