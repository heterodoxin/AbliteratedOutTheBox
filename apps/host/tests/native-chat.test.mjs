import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {proxyCompletion} from '../server/inference.mjs';
test('native chat sends image content, auth and tool opt-in and exposes reported MCP events',async()=>{
 let received;
 const server=http.createServer(async(req,res)=>{let text='';for await(const c of req)text+=c;received={body:JSON.parse(text),key:req.headers.authorization};res.writeHead(200,{'content-type':'text/event-stream'});res.end('data: {"strata_mcp":{"event":"call","id":"c1","name":"read_file","arguments":{"path":"example.txt"}}}\n\ndata: {"strata_mcp":{"event":"result","id":"c1","ok":true,"text":"Contents","ms":5}}\n\ndata: {"choices":[{"delta":{"content":"A picture"}}]}\n\ndata: {"usage":{"prompt_tokens":44,"completion_tokens":2},"timings":{"predicted_per_second":20,"prompt_per_second":800}}\n\ndata: [DONE]\n\n');});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 try{const events=[];const result=await proxyCompletion({baseUrl:`http://127.0.0.1:${server.address().port}/v1`,modelId:'iq2_xs',authorization:'private-test-key',message:'Describe',images:[{url:'data:image/png;base64,aGVsbG8='}],extra:{strata_mcp:true,reasoning_effort:'none'},onFrame:f=>events.push(f),signal:new AbortController().signal});
 assert.equal(received.key,'Bearer private-test-key');assert.equal(received.body.strata_mcp,true);assert.equal(received.body.messages.at(-1).content[1].image_url.url,'data:image/png;base64,aGVsbG8=');assert.equal(received.body.model,'iq2_xs');assert.deepEqual(events.filter(e=>e.type==='native-tool').map(e=>e.event.event),['call','result']);assert.equal(result.tokensIn,44);assert.equal(result.tokensOut,2);assert.equal(result.usageSource,'reported');
 }finally{server.closeAllConnections();await new Promise(r=>server.close(r));}
});
