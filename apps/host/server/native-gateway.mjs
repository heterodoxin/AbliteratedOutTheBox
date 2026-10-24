/** Preserve native protocol events and tool payloads; the gateway owns identity and cancellation. */
import { once } from 'node:events';
export async function nativeGateway({req,res,path,body,control,serving,fetcher=fetch}){
 let instance;try{instance=control.resolveInstance(body.model);}catch(e){res.writeHead(400,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:e.message,type:'invalid_request_error',param:'model',code:null}}));return;}
 const abort=new AbortController(),started=Date.now();res.on('close',()=>{if(!res.writableEnded)abort.abort();});
 const entry=serving.open({client:req.headers['user-agent']?.slice(0,100)??'API client',endpoint:path.slice(3),model:instance.apiModelId,instanceId:instance.id,gpuIds:instance.gpus.map(String),cancel:()=>abort.abort()});
 let tokensIn=null,tokensOut=null,tps=null,promptTps=null;
 const observe=value=>{const usage=value.usage??value.response?.usage;if(usage){tokensIn=usage.prompt_tokens??usage.input_tokens??tokensIn;tokensOut=usage.completion_tokens??usage.output_tokens??tokensOut;}if(value.timings){tps=value.timings.predicted_per_second??tps;promptTps=value.timings.prompt_per_second??promptTps;}};
 try{
  const response=await fetcher(`http://127.0.0.1:${instance.port}${path}`,{method:'POST',headers:{'content-type':'application/json',authorization:`Bearer ${instance.key}`},signal:abort.signal,body:JSON.stringify({...body,model:instance.apiModelId,strata_mcp:false})});
  if(!response.ok){const content=await response.text();serving.finish(entry.id,{durationMs:Date.now()-started,error:`Engine returned ${response.status}`});res.writeHead(response.status,{'content-type':response.headers.get('content-type')??'application/json'});res.end(content);return;}
  serving.mark(entry.id,{state:'generating'});
  res.writeHead(response.status,{'content-type':response.headers.get('content-type')??'application/json','cache-control':'no-cache','x-strata-request-id':entry.id,'x-accel-buffering':'no'});
  if(body.stream===true){let buffer='';const decoder=new TextDecoder();for await(const chunk of response.body){buffer+=decoder.decode(chunk,{stream:true});let end;while((end=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,end);buffer=buffer.slice(end+1);if(line.startsWith('data:'))try{observe(JSON.parse(line.slice(5)));}catch{}}if(!res.write(chunk))await once(res,'drain',{signal:abort.signal});}}
  else{const text=await response.text();try{observe(JSON.parse(text));}catch{}res.write(text);}
  serving.mark(entry.id,{tokensIn,tokensOut,tps,promptTps,usageSource:tokensIn===null||tokensOut===null?'unavailable':'reported'});serving.finish(entry.id,{tokensOut,durationMs:Date.now()-started});res.end();
 }catch(e){serving.finish(entry.id,{tokensOut,durationMs:Date.now()-started,error:abort.signal.aborted?'Request cancelled.':e.message});if(!res.headersSent){res.writeHead(502,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:e.message,type:'server_error'}}));}else res.end();}
}
