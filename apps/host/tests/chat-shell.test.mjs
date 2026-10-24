import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, symlink, rm, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { transform } from 'esbuild';
import { openArtifact } from '../server/artifacts.mjs';
const code=await readFile(new URL('../src/lib/chat-metrics.ts',import.meta.url),'utf8');
const {code:js}=await transform(code,{loader:'ts',format:'esm'});
const {chatMetrics}=await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
const {code:previewJs}=await transform(await readFile(new URL('../src/lib/preview-tabs.ts',import.meta.url),'utf8'),{loader:'ts',format:'esm'});
const {closePreviewTab}=await import(`data:text/javascript;base64,${Buffer.from(previewJs).toString('base64')}`);
test('closing a preview preserves selection or selects its nearest remaining neighbor',()=>{
 const tabs=[{id:'a'},{id:'b'},{id:'c'}];
 assert.deepEqual(closePreviewTab(tabs,'b','a'),{items:[{id:'b'},{id:'c'}],selected:'b'});
 assert.equal(closePreviewTab(tabs,'a','a').selected,'b');
 assert.equal(closePreviewTab(tabs,'b','b').selected,'c');
 assert.equal(closePreviewTab(tabs,'c','c').selected,'b');
 assert.deepEqual(closePreviewTab([{id:'a'}],'a','a'),{items:[],selected:null});
 assert.deepEqual(tabs,[{id:'a'},{id:'b'},{id:'c'}]);
});
const session={id:'chat-a',statsAvailable:true,tokensIn:1000,tokensOut:100,messages:[{role:'user',content:'hi'},{role:'assistant',inputTokens:1000,tokens:100,contextMax:10000,tps:50,ttftMs:200,durationMs:2000,tools:[{durationMs:100}]}]};
test('chat metrics ignore another session and all-request totals',()=>{
 const m=chatMetrics(session,{phase:'generating',outputTps:900,totals:{tokensIn:80000000},context:{sessionId:'chat-b',usedTokens:9500,maxTokens:10000}});
 assert.equal(m.total,1100);assert.equal(m.speed,50);assert.equal(m.pct,11);assert.equal(m.ttft,200);assert.equal(m.toolMs,100);
 const active=chatMetrics(session,{phase:'generating',outputTps:60,context:{sessionId:'chat-a',usedTokens:2000,maxTokens:10000}});
 assert.equal(active.speed,60);assert.equal(active.pct,20);
 assert.equal(chatMetrics({...session,statsAvailable:false},null).total,null);
 assert.equal(chatMetrics(null,null).pct,null);
});
test('native file actions use explicit argument arrays and reject traversal and symlinks',async()=>{
 const root=await mkdtemp(join(tmpdir(),'strata-artifact-test-'));
 try {
  const examplesDir=join(root,'examples'),previewsDir=join(root,'copies');await mkdir(examplesDir);
  await writeFile(join(examplesDir,'a file.md'),'Example');await writeFile(join(root,'private.md'),'Outside');await symlink(join(root,'private.md'),join(examplesDir,'outside.md'));
  const calls=[];const deps={examplesDir,previewsDir,platform:'darwin',launch:async(...args)=>calls.push(args)};
  await openArtifact({action:'reveal',path:'/examples/a file.md'},deps);
  assert.deepEqual(calls[0],['open',['-R',await realpath(join(examplesDir,'a file.md'))]]);
  await openArtifact({action:'open',name:'../notes.md',content:'Exported preview'},deps);
  assert.equal(await readFile(calls[1][1][0],'utf8'),'Exported preview');
  await assert.rejects(openArtifact({action:'open',path:'/examples/../private.md'},deps));
  await assert.rejects(openArtifact({action:'open',path:'/examples/outside.md'},deps));
  await assert.rejects(openArtifact({action:'open',path:'/etc/passwd'},deps));
  await assert.rejects(openArtifact({action:'open',name:'bad.app',content:'No'},deps));
  assert.equal(calls.length,2);
 } finally {await rm(root,{recursive:true,force:true});}
});

test('API replies use their own context limit and never inherit local engine readings',()=>{
 const remote={...session,messages:[...session.messages,{role:'assistant',remoteModel:true,inputTokens:50,tokens:10,contextMax:64000,tps:12}]};
 const monitor={phase:'generating',outputTps:900,context:{sessionId:'chat-a',usedTokens:9000,maxTokens:10000}};
 const metrics=chatMetrics(remote,monitor,32768);
 assert.equal(metrics.max,64000);assert.equal(metrics.used,60);assert.equal(metrics.speed,12);
 assert.equal(chatMetrics({...remote,messages:[{role:'assistant',remoteModel:true,inputTokens:50,tokens:10}]},monitor,32768).max,null);
});
