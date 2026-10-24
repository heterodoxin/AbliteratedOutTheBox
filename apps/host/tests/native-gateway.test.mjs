import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {nativeGateway} from '../server/native-gateway.mjs';
const listen=server=>new Promise(r=>server.listen(0,'127.0.0.1',()=>r(server.address().port)));
const close=server=>new Promise(r=>{server.closeAllConnections();server.close(r);});
test('native gateway preserves all three protocols and routes exact IDs, usage, errors and cancellation',{timeout:10000},async()=>{
 const received=[],records=[];let cancelled=false;
 const engine=http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;const body=JSON.parse(text);received.push({path:req.url,body,key:req.headers.authorization});
  if(body.input==='fail'){res.writeHead(422,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:'Invalid native input'}}));return;}
  if(body.stream){res.writeHead(200,{'content-type':'text/event-stream'});res.write('event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hello"}\n\n');if(body.input==='slow'){res.on('close',()=>{cancelled=true;});return;}res.end('event: response.completed\ndata: {"response":{"usage":{"input_tokens":12,"output_tokens":3}}}\n\n');return;}
  res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify(req.url==='/v1/messages'?{type:'message',content:[{type:'text',text:'Hello'}],usage:{input_tokens:12,output_tokens:3}}:{object:'response',output:[{type:'function_call',name:'lookup',arguments:'{}'}],usage:{input_tokens:12,output_tokens:3}}));
 });const port=await listen(engine);
 const instances=[{id:'first',apiModelId:'native-a',model:'iq2_xs',port,key:'private-a',gpus:[0]},{id:'second',apiModelId:'native-b',model:'iq3_s',port,key:'private-b',gpus:[1]}];
 const serving={open:value=>{const record={...value,id:String(records.length)};records.push(record);return record;},mark:(id,value)=>Object.assign(records[Number(id)],value),finish:(id,value)=>Object.assign(records[Number(id)],value)};
 const control={resolveInstance:id=>{const found=instances.find(i=>i.apiModelId===id);if(!found)throw new Error('Choose an exact loaded ID');return found;}};
 const gateway=http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;await nativeGateway({req,res,path:req.url,body:JSON.parse(text),control,serving});});const publicPort=await listen(gateway);
 const call=async(path,body)=>fetch(`http://127.0.0.1:${publicPort}${path}`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
 try{
  const tools=[{type:'function',name:'lookup',parameters:{type:'object'}}];const input=[{type:'message',role:'user',content:[{type:'input_text',text:'Hello'}]},{type:'function_call_output',call_id:'c1',output:'Found'}];
  let result=await call('/v1/responses',{model:'native-b',input,tools,strata_mcp:true,reasoning:{effort:'high'}});assert.equal(result.status,200);assert.equal((await result.json()).output[0].name,'lookup');assert.deepEqual(received[0].body.input,input);assert.deepEqual(received[0].body.tools,tools);assert.equal(received[0].body.model,'native-b');assert.equal(received[0].body.strata_mcp,false);assert.equal(received[0].key,'Bearer private-b');assert.equal(records[0].tokensIn,12);assert.equal(records[0].tokensOut,3);
  result=await call('/v1/messages',{model:'native-a',messages:[{role:'user',content:'Hello'}],max_tokens:50});assert.equal((await result.json()).type,'message');assert.equal(received[1].path,'/v1/messages');assert.equal(received[1].key,'Bearer private-a');
  result=await call('/v1/chat/completions',{model:'native-a',messages:[{role:'user',content:'Hello'}]});assert.equal(result.status,200);await result.text();assert.equal(received[2].path,'/v1/chat/completions');
  result=await call('/v1/responses',{model:'native-b',stream:true,input:'Hello'});assert.match(await result.text(),/event: response.completed/);assert.equal(records[3].tokensOut,3);
  result=await call('/v1/responses',{model:'missing',input:'Hello'});assert.equal(result.status,400);assert.equal((await result.json()).error.param,'model');
  result=await call('/v1/responses',{model:'native-b',input:'fail'});assert.equal(result.status,422);assert.equal((await result.json()).error.message,'Invalid native input');
  result=await call('/v1/responses',{model:'native-a',input:'slow',stream:true});const read=result.body.getReader();await read.read();await read.cancel();for(let n=0;n<30&&!cancelled;n++)await new Promise(r=>setTimeout(r,20));assert.equal(cancelled,true);
 }finally{await close(gateway);await close(engine);}
});
