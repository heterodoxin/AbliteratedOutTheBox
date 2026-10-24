import test from 'node:test';
import assert from 'node:assert/strict';
import { createTaskService, nextRun, validateTask } from '../server/tasks.mjs';
import { createLocalShare } from '../server/local-share.mjs';
const draft = { title:'Review', prompt:'Review my notes.', frequency:'daily', at:'09:00', weekday:1, timeZone:'Europe/Zagreb', enabled:true };
test('schedules respect local weekdays and DST, and reject invalid input', () => {
  assert.equal(nextRun(draft, Date.parse('2026-10-06T06:59:00Z')), '2026-10-06T07:00:00.000Z');
  assert.equal(nextRun({...draft,frequency:'weekdays'},Date.parse('2026-10-09T07:00:00Z')), '2026-10-12T07:00:00.000Z');
  assert.equal(nextRun(draft,Date.parse('2026-10-24T07:01:00Z')), '2026-10-25T08:00:00.000Z');
  assert.equal(nextRun({...draft,at:'02:30'},Date.parse('2026-10-25T00:30:00Z')), '2026-10-26T01:30:00.000Z');
  assert.throws(()=>validateTask({...draft,at:'25:60'}));
  assert.throws(()=>validateTask({...draft,timeZone:'Not/A_Zone'}));
  assert.throws(()=>validateTask({...draft,prompt:''}));
});
test('tasks persist, prevent overlapping runs, pause and resume without catch-up bursts', async () => {
  let now=Date.parse('2026-10-06T06:59:00Z'), saved=[], count=0, release;
  const service=createTaskService({read:()=>saved,write:t=>{saved=structuredClone(t);},now:()=>now,run:()=>{count++;return new Promise(resolve=>{release=resolve;});}});
  const task=service.put(draft); now=Date.parse(task.nextRunAt);service.tick();service.tick();
  assert.equal(count,1);assert.equal(saved[0].status,'running');assert.throws(()=>service.put({title:'New'},task.id),/Stop/);
  release({text:'Checked notes.',demo:true});await new Promise(resolve=>setImmediate(resolve));
  assert.equal(saved[0].status,'done');assert.equal(saved[0].output,'Checked notes.');assert.equal(saved[0].demo,true);
  service.put({enabled:false},task.id);assert.equal(saved[0].nextRunAt,null);
  now+=7*24*60*60*1000;
  const restarted=createTaskService({read:()=>saved,write:t=>{saved=structuredClone(t);},now:()=>now,run:async()=>({text:'new',demo:true})});
  restarted.put({enabled:true},task.id);assert.ok(Date.parse(saved[0].nextRunAt)>now);restarted.close();service.close();
});
test('sharing never exposes host controls or files, and a key is checked when one is set', async () => {
  let key=false;
  const share=createLocalShare({host:'127.0.0.1',port:0,hasKey:()=>key,handle:(req,res)=>{res.writeHead(req.headers.authorization==='Bearer test-only-key'?200:401);res.end('API');}});
  key=true;
  try {
    const {url}=await share.start();const origin=new URL(url).origin;
    assert.equal((await fetch(origin+'/api/tasks')).status,404);
    assert.equal((await fetch(origin+'/examples/project-brief.pdf')).status,404);
    assert.equal((await fetch(url+'/models')).status,401);
    assert.equal((await fetch(url+'/models',{headers:{authorization:'Bearer test-only-key'}})).status,200);
  } finally {await share.stop();}
  assert.equal(share.state().active,false);
});
