import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
async function moduleAt(path) { const result=await build({entryPoints:[fileURLToPath(new URL(path,import.meta.url))],bundle:true,write:false,platform:'node',format:'esm'}); const code=result.outputFiles[0].text; return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`); }
const {formatModelName,userError}=await moduleAt('../src/lib/format.ts');
const {versionFit,recommendedVersion}=await moduleAt('../src/lib/model-versions.ts');
const {memoryReading}=await moduleAt('../src/lib/hardware-summary.ts');
const {chatMetrics}=await moduleAt('../src/lib/chat-metrics.ts');
test('model labels preserve real family and remove fixture prefixes from visible failures',()=>{
 assert.equal(formatModelName(null),'No model loaded');
 assert.match(formatModelName({family:'Qwen3.8-Flash-Next',quant:'Q4_K_M'}),/Qwen3.8-Flash-Next.*Balanced.*Q4_K_M/);
 assert.match(formatModelName({family:'Qwen2.5-0.5B',quant:'Q4_K_M'}),/^Qwen2.5-0.5B/);
 assert.equal(userError('Fixture: model load failed.'),'Model load failed.');
});
test('memory readings never exceed capacity or become negative',()=>{
 assert.deepEqual(memoryReading(19,16),{used:16,total:16,free:0,over:3});
 assert.deepEqual(memoryReading(-1,16),{used:0,total:16,free:16,over:0});
 assert.deepEqual(memoryReading(NaN,Infinity),{used:0,total:0,free:0,over:0});
});
test('fit distinguishes unified and dedicated budgets and only recommends a fast version',()=>{
 const hardware={memory:{totalGb:32},accelerator:{usableVramGb:32,unified:true,backend:'metal'}};
 const versions=[4,12,24,48].map((diskGb,i)=>({id:String(i),diskGb,contextMax:1048576,kvKbPerToken:24,experts:128}));
 assert.equal(versionFit(versions[0],null).zone,'unknown');
 assert.equal(recommendedVersion(versions,hardware).id,'1');
 assert.equal(versionFit(versions[2],hardware).zone,'slower');
 assert.equal(versionFit(versions[3],hardware).zone,'too-big');
 assert.equal(versionFit(versions[1],hardware,1048576).zone,'too-big');
 assert.equal(recommendedVersion(versions,{memory:{totalGb:16},accelerator:{usableVramGb:0,unified:false,backend:'cpu'}}),undefined);
 const discrete={memory:{totalGb:32},accelerator:{usableVramGb:8,unified:false,backend:'cuda'}};
 assert.equal(versionFit(versions[0],discrete).zone,'fast');
 assert.equal(versionFit(versions[1],discrete).zone,'slower');
});
test('live usage, configured context and empty time-skip telemetry remain honest',()=>{
 const s={id:'s',tokensIn:100,tokensOut:20,statsAvailable:true,messages:[{role:'assistant',streaming:true,usageSource:'estimated',tokens:8,inputTokens:100,contextMax:128000,tps:12}]};
 const m=chatMetrics(s,null,32768);assert.equal(m.max,32768);assert.equal(m.total,228);assert.equal(m.streaming,true);assert.equal(m.estimated,true);
 assert.equal(chatMetrics({...s,messages:[{role:'assistant',timeSkipId:'demo'}]},null).hasReplies,false);
 assert.equal(chatMetrics({...s,statsAvailable:false},null).total,null);
});
const {readConfig,jsonErrorOffset}=await moduleAt('../src/lib/config.ts');
test('raw JSON errors identify their line and column across malformed structures',()=>{
 for(const [text,offset] of [['{\n  "bad": }',11],['{"a": [1,]}',9],['{"a" 1}',5],['{"a":true} extra',11],['{"a": "bad\\q"}',11]])assert.equal(jsonErrorOffset(text),offset,text);
 assert.throws(()=>readConfig('{\n  "bad": }'),/Line 2, column 10/);
 assert.doesNotThrow(()=>readConfig('{"beginner":{},"expert":{}}'));
});
