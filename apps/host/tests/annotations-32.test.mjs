import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { build } from 'esbuild';
import { parseCpuTopology, parseWindowsProcessors, parseNvidiaDevices, parseNvidiaReadings, readGpuTelemetry } from '../server/device-probes.mjs';
import { CONTEXT_TOKENS, expertParams } from '../server/catalog.mjs';
import { buildArgs } from '../server/supervisor.mjs';
async function loadTs(path) {
 const result=await build({entryPoints:[fileURLToPath(new URL(path,import.meta.url))],bundle:true,write:false,platform:'node',format:'esm'}); const code=result.outputFiles[0].text;
 return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}
const {usageDashboard,calendarDays}=await loadTs('../src/lib/usage-dashboard.ts');
const {modelFamilies,modelVersions}=await loadTs('../src/lib/model-versions.ts');
const {pickerVersionState,pickerVersions}=await loadTs('../src/lib/model-picker.ts');
const {contextTokens}=await loadTs('../src/lib/experience.ts');
const {configDocument}=await loadTs('../src/lib/config.ts');

test('128K recommendation reaches saved config and launch arguments while explicit context is preserved',()=>{
 const engine={beginner:{contextLength:'recommended'},expert:expertParams(8)};
 const recommended=configDocument(engine);
 assert.equal(contextTokens('recommended'),131072);
 assert.equal(CONTEXT_TOKENS.recommended,131072);
 assert.equal(recommended.expert['ctx.length'],'128K');
 assert.equal(engine.expert.find(p=>p.id==='ctx.length').recommended,'128K');
 for(const [contextLength,expected] of [['recommended',131072],['32k',32768],['256k',262144]]) {
  const doc=configDocument({...engine,beginner:{contextLength}});
  const args=buildArgs({modelPath:'/test.gguf',config:{beginner:doc.beginner,expert:[]}});
  assert.equal(args[args.indexOf('-c')+1],String(expected));
  assert.equal(contextTokens(contextLength),expected);
 }
});

test('all integer K context sizes reach the engine without a 32K fallback',()=>{
 for(const k of [4,5,31,32,33,127,128,129,511,1024]) {
  assert.equal(CONTEXT_TOKENS[`${k}k`],k*1024);
  const args=buildArgs({modelPath:'/test.gguf',config:{beginner:{contextLength:`${k}k`},expert:[]}});
  assert.equal(args[args.indexOf('-c')+1],String(k*1024));
 }
 assert.equal(CONTEXT_TOKENS['3k'],undefined);
 assert.equal(CONTEXT_TOKENS['33.5k'],undefined);
 assert.equal(CONTEXT_TOKENS['1025k'],undefined);
});
test('picker covers available, downloaded, running, transitions and retry paths',()=>{
 const version={id:'v',state:'available',diskGb:1};
 assert.equal(pickerVersionState(version,null).label,'Download');
 assert.equal(pickerVersionState({...version,state:'installed'},null).label,'Downloaded');
 assert.equal(pickerVersionState({...version,state:'loaded'},'v').label,'Running');
 for(const state of ['downloading','loading','unloading'])assert.equal(pickerVersionState({...version,state},'v').busy,true);
 assert.equal(pickerVersionState({...version,state:'error'},null).action,'Retry download');
 assert.equal(pickerVersionState({...version,state:'error',installedAt:'2026-10-09'},null).action,'Retry loading');
 assert.deepEqual(pickerVersions({loaded:'large'},[{...version,id:'small'},{...version,id:'large',diskGb:12}]).map(v=>v.id),['large','small']);
});
test('physical processor sockets retain per-socket cores and logical threads',()=>{
 const raw='# CPU,Core,Socket\n0,0,0\n1,0,0\n2,1,0\n3,1,0\n4,0,1\n5,0,1\n6,1,1\n7,1,1';
 const cpus=parseCpuTopology(raw,'Xeon','x64');
 assert.equal(cpus.length,2);
 assert.deepEqual(cpus.map(c=>[c.cores,c.threads]),[[2,4],[2,4]]);
 assert.deepEqual(parseCpuTopology('0,-,-','Unknown','arm64'),[]);
 const windows=parseWindowsProcessors(JSON.stringify([{DeviceID:'CPU0',Name:'Xeon A',NumberOfCores:12,NumberOfLogicalProcessors:24},{DeviceID:'CPU1',Name:'Xeon B',NumberOfCores:16,NumberOfLogicalProcessors:32}]),'x64');
 assert.deepEqual(windows.map(c=>[c.id,c.cores,c.threads]),[['CPU0',12,24],['CPU1',16,32]]);
});
test('GPU readings use UUIDs and never turn unavailable sensors into zero',()=>{
 const gpus=parseNvidiaDevices('0, GPU-a, 00000000:01:00.0, RTX A, 24576, 555\n1, GPU-b, 00000000:02:00.0, RTX B, 49152, 555');
 assert.deepEqual(gpus.map(g=>[g.id,g.vramGb]),[['GPU-a',24],['GPU-b',48]]);
 const readings=parseNvidiaReadings('GPU-b, 71, 94, 10240\nGPU-a, N/A, 0, 0');
 assert.equal(readings.find(g=>g.id==='GPU-b').temperatureC,71);
 assert.equal(readings.find(g=>g.id==='GPU-a').temperatureC,null);
 assert.equal(readings.find(g=>g.id==='GPU-a').utilizationPct,0);
 assert.equal(parseNvidiaReadings('GPU-a, 999, -1, N/A')[0].temperatureC,null);
});
test('multiple Linux sensor paths stay separate and lost sensors become unavailable',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'strata-gpu-'));
 try {
  const gpus=[];
  for(const [id,temp] of [['a',42000],['b',73000]]) {
   const device=join(dir,id);await mkdir(join(device,'hwmon','hwmon0'),{recursive:true});
   await writeFile(join(device,'hwmon','hwmon0','temp1_input'),String(temp));
   await writeFile(join(device,'hwmon','hwmon0','temp1_label'),'edge');
   gpus.push({id,vendor:'amd',sensorPath:device});
  }
  assert.deepEqual((await readGpuTelemetry(gpus)).map(g=>g.temperatureC),[42,73]);
  await rm(join(dir,'a'),{recursive:true});
  const readings=await readGpuTelemetry(gpus);
  assert.equal(readings[0].temperatureC,null);assert.equal(readings[0].source,'none');assert.equal(readings[1].temperatureC,73);
 }finally{await rm(dir,{recursive:true,force:true});}
});
test('usage totals ignore demo, estimated and copied branch usage',()=>{
 const now=new Date('2026-10-09T12:00:00Z');
 const m={id:'m',role:'assistant',at:'2026-10-08T10:00:00Z',content:'Hello',requestId:'r1',model:'model-a',usageSource:'reported',inputTokens:100,tokens:25};
 const sessions=[{id:'one',messages:[{id:'u',role:'user',at:m.at,content:'Hi'},m]}, {id:'branch',messages:[{...m,id:'copy'}]}, {id:'demo',messages:[{...m,id:'d',requestId:'d',demo:true,tokens:9999}]},{id:'old',messages:[{...m,id:'old',requestId:'old',at:'2026-08-01T00:00:00Z'}]},{id:'show',showcase:true,messages:[{...m,requestId:'show'}]}];
 const usage=usageDashboard(sessions,'7d',now);
 assert.equal(usage.total,125);assert.equal(usage.messages,3);assert.equal(usage.unreported,1);assert.equal(usage.activeDays,1);assert.equal(usage.models[0].requests,1);
 assert.equal(usageDashboard(sessions,'all',now).total,250);
 assert.equal(usageDashboard([{messages:[{...m,usageSource:'estimated',requestId:'e'}]}],'all',now).total,0);
});
test('empty usage and calendar ranges are stable',()=>{
 const empty=usageDashboard([],'all',new Date('2026-10-09T12:00:00Z'));
 assert.equal(empty.messages,0);assert.equal(empty.peakHour,null);assert.equal(empty.favorite,null);
 assert.deepEqual(calendarDays(new Date(2026,9,7),new Date(2026,9,9)),['2026-10-07','2026-10-08','2026-10-09']);
});

test('family choices follow the catalog and version filtering follows the selection',()=>{
 const library={mode:'simulated',loaded:'a',variants:[{id:'a',family:'Alpha',diskGb:4},{id:'b',family:'Beta',diskGb:8},{id:'c',family:'Beta',diskGb:3}]};
 assert.deepEqual(modelFamilies(library),['Alpha','Beta']);
 assert.deepEqual(modelVersions(library).map(v=>v.id),['a']);
 assert.deepEqual(modelVersions(library,'Beta').map(v=>v.id),['c','b']);
});
