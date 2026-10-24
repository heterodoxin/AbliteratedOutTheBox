import {EventEmitter} from 'node:events';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {NATIVE_FIELDS} from '../../server/native-settings.mjs';
import {createNativeControl} from '../../server/native-control.mjs';
export function nativeFixture({modelsDirectory,tuneHandler,engineStatus}={}){
 const root=mkdtempSync(join(tmpdir(),'strata-native-test-')),repo=join(root,'repo'),directory=join(root,'data');
 const venvBin=process.platform==='win32'?'.venv/Scripts':'.venv/bin',venvPython=process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python';
 for(const path of ['tools','serve',venvBin,'engine/.previous'])mkdirSync(join(repo,path),{recursive:true});
 writeFileSync(join(repo,'setup.py'),'# --rollback-engine\n');for(const path of ['tools/strata_mcp.py',venvPython])writeFileSync(join(repo,path),'fixture');
 writeFileSync(join(repo,'serve/server.py'),NATIVE_FIELDS.map(f=>f.id).join('\n')+'\n/v1/responses');
 const catalog={families:[{family:'qwen',title:'Qwen3.8-Flash-Next',about:'Native fixture',images:true,sizes:[{id:'iq2_xs',model:'IQ2_XS',download_gb:68,ram_needed_gb:48,about:'Compact',on_this_pc:'fits',installed:true},{id:'iq3_s',model:'IQ3_S',download_gb:84,ram_needed_gb:62,about:'Quality',on_this_pc:'fits',installed:true}]}],contexts:[32768,131072],recommendation:{family:'qwen',model:'IQ2_XS',context:32768}};
 const installed={models:['iq2_xs','iq3_s'].map(model=>({model,config:`strata-${model}.json`,model_name:`Qwen ${model}`,context:32768,images:'gpu',ready:true,kv:'int8'})),data_folder:join(root,'Strata-data'),engine:{version:'v0.1.40'}};
 for(const model of installed.models)writeFileSync(join(repo,model.config),JSON.stringify({model:model.model,gpu:0,args:['--max-context','32768'],env:{PATH:'/native/bin',STRATA_TOKEN:'private'},vision:{gpu:0},aliases:['chat'],unknown_native_key:{preserve:true}}));
 const calls=[],children=[],requests=[],rpcCalls=[];let programWait=false,hardware={ram_gb:128,gpus:[{index:0,name:'NVIDIA RTX 4090',vram_gb:24},{index:1,name:'NVIDIA RTX 3090',vram_gb:24}]},external=[];
 const spawnProcess=(command,args,options)=>{
  calls.push({command,args,options});const child=new EventEmitter();child.stdout=new EventEmitter();child.stderr=new EventEmitter();child.exitCode=null;child.kill=()=>{child.exitCode=0;queueMicrotask(()=>child.emit('exit',0));return true;};
  if(tuneHandler&&args.some(a=>String(a).endsWith('strata.py'))){child.kill=()=>{child.exitCode=0;queueMicrotask(()=>{child.emit('exit',0);child.emit('close',null);});return true;};queueMicrotask(()=>tuneHandler(child,args,options));return child;}children.push(child);
  if(args[0]?.endsWith('strata_mcp.py'))child.stdin={write(line){const message=JSON.parse(line);if(!message.id)return;if(message.method==='tools/call')rpcCalls.push(message.params);let result;
   if(message.method==='initialize')result={protocolVersion:'2024-11-05'};
   else if(message.method==='tools/list')result={tools:['strata_models','strata_status','strata_install'].map(name=>({name}))};
   else if(message.params.name==='strata_models')result={structuredContent:catalog};
   else if(message.params.name==='strata_status')result={structuredContent:{installed,hardware,running:external,summary:'Fixture installation',...(engineStatus?{engine:engineStatus}:{})}};
   else result={structuredContent:{summary:'Ready to prepare',plan:{download_gb:68},install:{running:false,exit_code:0}}};
   queueMicrotask(()=>child.stdout.emit('data',Buffer.from(JSON.stringify({jsonrpc:'2.0',id:message.id,result})+'\n')));
  }};
  else queueMicrotask(()=>{if(command==='git')child.stdout.emit('data',Buffer.from(args[0]==='describe'?'v0.1.40\n':args[0]==='status'?' M local.py\n':'https://github.com/Niko1221/Strata\n'));if(command==='git'||args[0]==='-c'&&!programWait){child.exitCode=0;child.emit('exit',0);}});
  return child;
 };
 let benchmarkWait=false,healthWait=false,releaseHealth=null;
 const fetcher=async(url,options)=>{
  if(url.endsWith('/health')&&healthWait)await new Promise(resolve=>releaseHealth=resolve);
  if(url.includes('/metrics'))return Response.json({engine:{max_context:32768},live:{state:'idle'},hardware:{cpu:12,ram_used:48*1024**3,ram_total:128*1024**3},requests:[{time:Date.now()/1000,decode_tok_s:41,prompt_tokens:120,reused:20,prompt_ms:100,output_tokens:80,duration_s:2}],history:{tok_s:[40,41]}});
  if(url.endsWith('/mcp'))return Response.json({tools:1,servers:[{name:'files',status:'ready',tools:[{name:'read_file'}]}]});
  if(url.endsWith('/v1/chat/completions')){if(benchmarkWait)await new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('Stopped','AbortError')),{once:true}));return Response.json({usage:{prompt_tokens:30,completion_tokens:200},timings:{predicted_per_second:41,prompt_per_second:1000}});}
  return Response.json({status:'ok'});
 };
 const control=createNativeControl({directory,defaultRepo:repo,spawnProcess,fetcher,modelsDirectory,requests:()=>requests});
 return {control,repo,directory,catalog,installed,requests,calls,rpcCalls,children,setExternal:value=>external=value,setHardware:value=>hardware=value,setBenchmarkWait:value=>benchmarkWait=value,setProgramWait:value=>programWait=value,setHealthWait:value=>healthWait=value,hasPendingHealth:()=>!!releaseHealth,releaseHealth:()=>{healthWait=false;releaseHealth?.();releaseHealth=null;},cleanup:async()=>{await control.close();rmSync(root,{recursive:true,force:true});}};
}
export async function finishJob(control,id){for(let n=0;n<400;n++){const job=control.state().jobs.find(j=>j.id===id);if(job.state!=='running')return job;await new Promise(r=>setTimeout(r,10));}throw new Error('Fixture operation timed out');}
