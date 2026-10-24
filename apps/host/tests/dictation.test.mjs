import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/dictation.ts',import.meta.url),'utf8'),{loader:'ts',format:'esm'});
const { createDictation, DICTATION_SILENCE_MS } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
function harness(){
  let now=0,id=0;const timers=new Map(),states=[],texts=[];
  const recognition={start(){this.onstart?.();},stop(){this.stops=(this.stops??0)+1;},abort(){this.aborts=(this.aborts??0)+1;},stops:0,aborts:0};
  const clock={set(fn,ms){timers.set(++id,{fn,at:now+ms});return id;},clear(id){timers.delete(id);}};
  const session=createDictation(recognition,{lang:'en-US',onState:(state,error)=>states.push({state,error}),onText:text=>texts.push(text)},clock);
  return {session,recognition,states,texts,timers,tick(ms){now+=ms;for(const [id,timer] of [...timers])if(timer.at<=now){timers.delete(id);timer.fn();}}};
}
const result=(text,final=false)=>({isFinal:final,0:{transcript:text}});
test('interim corrections replace text and final results do not duplicate earlier words',()=>{
  const h=harness();h.session.start();
  h.recognition.onresult({results:[result('Hello wor')]});
  h.recognition.onresult({results:[result('Hello world',true)]});
  h.recognition.onresult({results:[result('Hello world',true),result('How are you',true)]});
  assert.deepEqual(h.texts,['Hello wor','Hello world','Hello world How are you']);
  h.session.stop();h.recognition.onresult({results:[result('Hello world',true),result('How are you?',true)]});h.recognition.onend();
  assert.equal(h.texts.at(-1),'Hello world How are you?');assert.equal(h.states.at(-1).state,'finished');assert.equal(h.timers.size,0);
});
test('speech resuming cancels the pause timer; 3.5 seconds of silence stops capture',()=>{
  const h=harness();h.session.start();h.recognition.onspeechstart();h.recognition.onspeechend();h.tick(2000);
  h.recognition.onspeechstart();h.tick(2000);assert.equal(h.recognition.stops,0);
  h.recognition.onspeechend();h.tick(DICTATION_SILENCE_MS-1);assert.equal(h.recognition.stops,0);
  h.tick(1);assert.equal(h.recognition.stops,1);assert.equal(h.states.at(-1).state,'finishing');h.recognition.onend();assert.equal(h.states.at(-1).state,'finished');
});
test('permission failures are clear and late results cannot change the draft',()=>{
  const h=harness();h.session.start();h.recognition.onerror({error:'not-allowed'});h.recognition.onresult({results:[result('late')]});h.recognition.onend();
  assert.equal(h.states.at(-1).state,'error');assert.match(h.states.at(-1).error,/denied/);assert.equal(h.texts.length,0);assert.equal(h.timers.size,0);
});
test('disposing on navigation releases capture and all timers',()=>{
  const h=harness();h.session.start();h.recognition.onresult({results:[result('Draft')]});h.session.dispose();h.tick(20000);
  assert.equal(h.recognition.onresult,null);assert.equal(h.recognition.aborts,1);assert.equal(h.recognition.stops,0);assert.equal(h.timers.size,0);
});
test('a service that never ends is aborted after a stop, and an idle mic times out',()=>{
  const h=harness();h.session.start();h.tick(10000);assert.equal(h.recognition.stops,1);h.tick(2000);
  assert.equal(h.recognition.aborts,1);assert.equal(h.states.at(-1).state,'finished');assert.equal(h.timers.size,0);
});
test('a stop failure still aborts pending capture',()=>{
  const h=harness();h.session.start();h.recognition.stop=()=>{throw new Error('Not yet started');};h.session.stop();
  assert.equal(h.recognition.aborts,1);assert.equal(h.states.at(-1).state,'finished');assert.equal(h.timers.size,0);
});
