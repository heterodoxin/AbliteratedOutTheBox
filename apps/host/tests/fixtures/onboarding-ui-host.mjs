/** Screenshot-only first-launch fixture. No weights, subprocesses or production writes. */
import http from 'node:http';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,extname} from 'node:path';

const root=resolve(import.meta.dirname,'../../dist'),upstream=process.env.STRATA_ONBOARDING_UPSTREAM??'http://127.0.0.1:5161';
const native=process.env.STRATA_UI_PREVIEW==='1' ? JSON.parse(await readFile(new URL('./ui-preview-native.json',import.meta.url),'utf8')) : await (await fetch(upstream+'/api/native')).json();
native.hardware={cpu:{cores:16},ram_gb:128,gpus:[{index:0,name:'NVIDIA RTX 4090',vram_gb:24,usable:true}],disk_free_gb:250};
if(process.env.STRATA_ONBOARDING_SCENARIO==='multi-gpu')native.hardware.gpus.push({index:1,name:'NVIDIA RTX 3090',vram_gb:24,usable:true});
if(process.env.STRATA_ONBOARDING_GPUS)native.hardware.gpus=JSON.parse(process.env.STRATA_ONBOARDING_GPUS);
native.catalog.this_pc=native.hardware;
native.catalog.recommendation={family:'qwen',model:'Q2_0',context:32768};
native.catalog.summary='Supported hardware fixture';
for(const family of native.catalog.families)for(const size of family.sizes){size.on_this_pc='fits';size.installed=false;}
if(process.env.STRATA_ONBOARDING_SCENARIO==='limited'){
 for(const family of native.catalog.families)for(const size of family.sizes)size.on_this_pc='does not fit';
 native.catalog.families.find(f=>f.family==='swift').sizes[0].on_this_pc='fits';
 native.catalog.recommendation.why='Selected model is incompatible in this fixture; Swift Compact is available.';
}
native.repo='/Applications/Strata/engine';
native.downloadDirectory='/Users/example/Documents/Strata/Models';
native.installed={models:[],data_folder:'/Models/Strata',engine:{version:native.version}};
native.instances=[];native.jobs=[];native.defaultId=null;native.external=[];
native.summary='Choose a version to get started';native.error=null;native.capabilities.calibration=true;
let setup={complete:false,dismissed:false,choices:{},step:1,flowVersion:2,history:[],app:{version:'0.1.0',modelsDir:'/Models',dataDir:'/Strata/data'}};
let selected=null;
let loadVision='no';
const events=new Set();
const modelSettings=new Map();
const automatic=process.env.STRATA_ONBOARDING_PREVIEW==='1';
function later(id,delay,action){const timer=setTimeout(()=>{if(job()?.id===id&&job().state==='running')action();},delay);timer.unref();}
const job=()=>native.jobs.at(-1);
function stage(name){
 if(!job())throw new Error('Begin Download & load in the UI first.');
 if(name==='download')Object.assign(job(),{state:'running',kind:'install',summary:'Downloading model files',result:{install:{running:true,step:'Downloading model files',download:{file:`Qwen3.8-Flash-Next-${selected?.model??'Q2_0'}-00001-of-00003.gguf`,percent:38,done_gb:8.4,total_gb:22.1}}}});
 else if(name==='prepare')Object.assign(job(),{state:'running',kind:'install',summary:'Preparing model files',result:{install:{running:true,step:'Packing model files for your computer'}}});
 else if(name==='loading')Object.assign(job(),{state:'running',kind:'install',summary:'Loading model',result:{install:{running:true,step:'Loading model into memory'}}});
 else if(name==='tuning')Object.assign(job(),{state:'running',kind:'tune',summary:'Measuring the best engine settings for this computer…'});
 else if(name==='prepared'||name==='ready'){
  const family=native.catalog.families.find(f=>f.family===selected.family),size=family.sizes.find(s=>s.model===selected.model);
  size.installed=true;
  native.installed.models=[{model:size.id,config:`strata-${size.id}.json`,model_name:family.title,ready:true,context:selected.context,kv:selected.kv,images:selected.vision==='no'?'off':'gpu',lowRamMode:selected.low_ram,dataFolder:'/Models/Strata'}];
  if(name==='ready'){native.installed.models[0].visionEnabled=loadVision!=='no';native.instances=[{id:'onboarding-model',apiModelId:size.id,model:size.id,label:family.title,aliases:[],vision:loadVision!=='no',context:selected.context,port:5262,gpus:[0],status:'ready',phase:'idle',loadedAt:new Date().toISOString(),error:null,metrics:null}];}
  if(name==='ready')native.defaultId='onboarding-model';native.summary=name==='ready'?'Model ready':'Model prepared';
  Object.assign(job(),{state:'done',summary:'Model ready',result:{install:{running:false,exit_code:0}}});
 }else throw new Error('Unknown fixture stage');
}
// Open the final setup stage directly for design review, without simulating a download first.
if(['make-it-yours','ready'].includes(process.env.STRATA_ONBOARDING_START)){
 const family=native.catalog.families.find(f=>f.family===native.catalog.recommendation.family)??native.catalog.families[0];
 const size=family.sizes.find(s=>s.model===native.catalog.recommendation.model)??family.sizes[0];
 selected={family:family.family,model:size.model,context:32768,vision:family.images?'yes':'no',kv:'int8',low_ram:'auto'};
 native.jobs.push({id:'preview-prepared',kind:'install',guided:true,model:size.model,state:'running',startedAt:new Date().toISOString(),error:null,result:null});
 stage('prepared');
 setup={...setup,step:4,choices:{onboarding:{step:4,family:family.family,size:size.id,context:32768,vision:false,profile:'work',calibrate:true,installJobId:'preview-prepared'}}};
 if(process.env.STRATA_ONBOARDING_START==='ready'){
  stage('ready');setup.complete=true;
  if(process.env.STRATA_ONBOARDING_SCENARIO==='telemetry'){
   const series=(base,amplitude)=>Array.from({length:60},(_,i)=>Math.round((base+Math.sin(i/6)*amplitude)*100)/100);
   native.instances[0].phase='generating';native.instances[0].metrics={time:Date.now()/1000,engine:{max_context:32768,expert_slots:64,experts:256,expert_cache_mib:8192},live:{state:'generating',running:1,queued:0,prompt_tokens:4096,generated:128,max_tokens:512,tok_s:24.8,prefill_tok_s_mean:642},requests:[{id:'fixture-request',time:Date.now()/1000-40,prompt_tokens:2048,reused:1024,output_tokens:360,decode_tok_s:24,duration_s:15,prompt_ms:1600}],totals:{requests:1,prompt_tokens:2048,reused:1024,output_tokens:360,prompt_ms:1600,decode_ms:15000},hardware:{cpu:34,ram_used:24*1024**3,ram_total:128*1024**3,gpu_util:78,gpu_mem_used:18*1024**3,gpu_mem_total:24*1024**3,gpu_temp:63,gpu_power:185,gpu_power_limit:450,gpu_pcie_gen:4,gpu_pcie_gen_max:4,gpu_pcie_width:16,gpu_pcie_rx_mb:240,disk_read_mb:180,disk_write_mb:12},history:{tok_s:series(24,3),prefill_tok_s_mean:series(640,85),gpu_util:series(78,9),gpu_mem_used:series(18*1024**3,0.3*1024**3),gpu_temp:series(62,2),gpu_power:series(185,18),gpu_pcie_rx_mb:series(240,40),cpu:series(34,7),ram_used:series(24*1024**3,0.3*1024**3),disk_read_mb:series(180,40)}};
  }
 }
}
if(process.env.STRATA_ONBOARDING_GPUS){native.instances=[];native.defaultId=null;}/* GPU roles previews: the model is prepared, not loaded, so the roles shown are the recommendation */
if(process.env.STRATA_PREVIEW_STATE_FILE){const saved=JSON.parse(await readFile(process.env.STRATA_PREVIEW_STATE_FILE,'utf8'));Object.assign(native,saved.native);setup=saved.setup;const running=native.instances.find(i=>i.id===native.defaultId),entry=native.installed.models.find(m=>m.model===running?.model),family=native.catalog.families.find(f=>f.sizes.some(s=>s.id===entry?.model)),size=family?.sizes.find(s=>s.id===entry?.model);if(family&&size){selected={family:family.family,model:size.model,context:entry.context??32768,vision:entry.images==='off'?'no':'yes',kv:entry.kv??'int8',low_ram:entry.lowRamMode??'auto'};loadVision=entry.visionEnabled===false?'no':selected.vision;}}
const server=http.createServer(async(req,res)=>{try{
 const path=new URL(req.url,'http://fixture').pathname;let payload='';if(req.method!=='GET')for await(const chunk of req)payload+=chunk;const body=payload?JSON.parse(payload):{};
 const json=(value,status=200)=>{res.writeHead(status,{'content-type':'application/json'});res.end(JSON.stringify(value));};
 if(process.env.STRATA_CAPTURE_OUTPUT&&path==='/__fixture/capture'&&req.method==='POST'){
  if(typeof body.dataUrl!=='string'||!body.dataUrl.startsWith('data:image/png;base64,'))throw new Error('A PNG export is required');
  await writeFile(process.env.STRATA_CAPTURE_OUTPUT,Buffer.from(body.dataUrl.slice('data:image/png;base64,'.length),'base64'));
  return json({saved:true,width:body.width,height:body.height});
 }
 if(process.env.STRATA_CAPTURE_OUTPUT&&path==='/__fixture/html-to-image.js'){
  res.writeHead(200,{'content-type':'text/javascript'});res.end(await readFile(resolve(import.meta.dirname,'../../node_modules/html-to-image/dist/html-to-image.js')));return;
 }
 if(process.env.STRATA_CAPTURE_OUTPUT&&path==='/__fixture/export.js'){
  res.writeHead(200,{'content-type':'text/javascript'});res.end(await readFile(resolve(import.meta.dirname,'onboarding-export.js')));return;
 }
 if(path==='/__fixture/stage'&&req.method==='POST'){stage(body.stage);return json({stage:body.stage});}
 if(path==='/api/native'&&req.method==='GET')return json(native);
 if(/^\/api\/native\/models\/[^/]+\/settings$/.test(path)){const model=path.split('/')[4];if(req.method==='PATCH')modelSettings.set(model,{model,values:body.values??{},mcp:body.mcp??{}});return json(modelSettings.get(model)??{model,values:{sampling:{},args:[],env:{}},mcp:{},restartNeeded:true});}
 if(path==='/api/native/directory'){native.downloadDirectory=body.directory;return json(native);}
 if(path==='/api/local-models/browse')return json({directory:new URL(req.url,'http://fixture').searchParams.get('directory')??'/Models/Strata',parent:'/Models',entries:[{name:'downloads',path:'/Models/Strata/downloads',kind:'folder'},{name:'models',path:'/Models/Strata/models',kind:'folder'}],shortcuts:[],truncated:false});
 if(path==='/api/native/plan'){
  const family=native.catalog.families.find(f=>f.family===body.family),size=family.sizes.find(s=>s.model===body.model);
  return json({plan:{title:family.title,download_gb:size.download_gb,disk_short:null}});
 }
 if(path==='/api/native/operations'&&req.method==='POST'){
  if(!['install','tune'].includes(body.kind))throw new Error('Only fixture preparation and tuning are supported');
  if(body.kind==='install')selected=body.args;
  if(body.kind==='tune')loadVision=body.args.vision??'no';
  native.jobs.push({id:`onboarding-${body.kind}-${native.jobs.length}`,kind:body.kind,guided:true,model:body.args.model,state:'running',startedAt:new Date().toISOString(),summary:'Starting…',error:null,result:null});
  stage(body.kind==='install'?'download':'tuning');
  if(automatic){const id=job().id;if(body.kind==='install'){later(id,4000,()=>{job().result.install.download.percent=72;job().result.install.download.done_gb=15.9;});later(id,8000,()=>stage('prepare'));later(id,12000,()=>stage('prepared'));}else{later(id,4000,()=>{job().summary='Applying your preference and loading the model…';});later(id,8000,()=>stage('ready'));}}
  return json(job());
 }
 if(/^\/api\/native\/operations\/[^/]+\/cancel$/.test(path)){Object.assign(job(),{state:'cancelled',summary:'Preparation stopped'});return json(job());}
 if(path==='/api/setup'&&req.method==='GET')return json(setup);
 if(path==='/api/setup/step'&&req.method==='POST'){setup={...setup,...body};return json(setup);}
 if(path==='/api/setup/complete'&&req.method==='POST'){
  if(!native.instances.some(i=>i.status==='ready'))throw new Error('Load a model first');
  setup={...setup,complete:true,step:4};return json(setup);
 }
 if(path==='/api/workspaces')return json({workspaces:[]});
 if(path==='/api/events'){
  res.writeHead(200,{'content-type':'text/event-stream','cache-control':'no-cache'});res.write('data: {}\n\n');events.add(res);
  req.on('close',()=>events.delete(res));return;
 }
 if(path.startsWith('/api/')){
  if(req.method!=='GET')throw new Error('Production mutations are blocked by this fixture');
  const response=await fetch(upstream+req.url),value=await response.json();
  if(path==='/api/hardware')Object.assign(value,{gpus:native.hardware.gpus.map(g=>({index:g.index,id:`gpu-${g.index}`,vendor:'nvidia',name:g.name,vramGb:g.vram_gb,backend:'cuda',unified:false})),memory:{totalGb:128,usedGb:16,freeGb:112,pressure:'low'},accelerator:{backend:'cuda',unified:false,usableVramGb:native.hardware.gpus.reduce((sum,g)=>sum+g.vram_gb,0),label:'Fixture GPUs'}});
  if(path==='/api/models')value.loaded=null;
  if(path==='/api/runtime')Object.assign(value,{instances:[],defaultId:null});
  return json(value,response.status);
 }
 const file=path==='/'?'/index.html':path;if(file.includes('..'))throw new Error('Invalid fixture path');
 let content=await readFile(root+file);
 if(process.env.STRATA_CAPTURE_OUTPUT&&file==='/index.html')content=Buffer.from(content.toString().replace('</body>','<script src="/__fixture/html-to-image.js"></script><script src="/__fixture/export.js"></script></body>'));
 res.writeHead(200,{'content-type':({'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.woff2':'font/woff2','.png':'image/png'})[extname(file)]??'application/octet-stream'});res.end(content);
 }catch(e){res.writeHead(400,{'content-type':'application/json'});res.end(JSON.stringify({error:e.message}));}});
const heartbeat=setInterval(()=>{for(const response of events)response.write('data: {}\n\n');},1000);
const port=Number(process.env.STRATA_ONBOARDING_PORT??5271);
server.listen(port,'127.0.0.1',()=>console.log(`Disposable onboarding screenshots: http://127.0.0.1:${port}`));
function stop(){clearInterval(heartbeat);server.closeAllConnections();server.close(()=>process.exit(0));}
process.on('SIGTERM',stop);process.on('SIGINT',stop);
