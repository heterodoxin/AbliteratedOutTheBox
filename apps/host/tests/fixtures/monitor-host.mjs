/** Disposable Monitor UI fixtures. GET-only API; the user's host and saved state are untouched. */
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
const root=resolve(import.meta.dirname,'../../dist'),upstream='http://127.0.0.1:5161';
const originalHardware=await (await fetch(upstream+'/api/hardware')).json();
const originalMonitor=await (await fetch(upstream+'/api/monitor')).json();
const hardware={...originalHardware,cpu:{model:'Intel Xeon Gold 6338',cores:64,threads:128,arch:'x64',ghz:2,topologyKnown:true},processors:[{id:'socket-0',socket:'0',model:'Intel Xeon Gold 6338',cores:32,threads:64,arch:'x64',ghz:2},{id:'socket-1',socket:'1',model:'Intel Xeon Gold 6338',cores:32,threads:64,arch:'x64',ghz:2}],gpus:[{id:'gpu-a',index:0,vendor:'nvidia',name:'NVIDIA RTX 3090',vramGb:24,unified:false,backend:'cuda'},{id:'gpu-b',index:1,vendor:'nvidia',name:'NVIDIA RTX 4090',vramGb:24,unified:false,backend:'cuda'}],memory:{totalGb:128,usedGb:48,freeGb:80,pressure:'low'},accelerator:{backend:'cuda',label:'NVIDIA RTX 4090',usableVramGb:24,unified:false}};
function monitor(unavailable=false, unloaded=false, fullUse=false) {
  const gpus=[{id:'gpu-a',utilizationPct:35,temperatureC:45,memoryUsedGb:4,powerW:70,powerLimitW:350,pcieGen:3,pcieGenMax:4,pcieWidth:16,pcieRxMbs:120,source:'engine',sampledAt:new Date().toISOString()},{id:'gpu-b',utilizationPct:85,temperatureC:78,memoryUsedGb:10,powerW:180,powerLimitW:450,pcieGen:4,pcieGenMax:4,pcieWidth:16,pcieRxMbs:320,source:'engine',sampledAt:new Date().toISOString()}];
  if(unavailable)for(const key of ['utilizationPct','temperatureC','memoryUsedGb','powerW','pcieGen','pcieGenMax','pcieWidth','pcieRxMbs'])gpus[1][key]=null;
  if(fullUse){
    Object.assign(gpus[0],{utilizationPct:96,temperatureC:78,memoryUsedGb:22,powerW:310,pcieGen:4,pcieRxMbs:820});
    Object.assign(gpus[1],{utilizationPct:98,temperatureC:82,memoryUsedGb:23,powerW:420,pcieRxMbs:1140});
  }
  if(unloaded)for(const gpu of gpus)Object.assign(gpu,{utilizationPct:2,temperatureC:34,memoryUsedGb:.3,powerW:18,pcieRxMbs:0});
  const requests=Array.from({length:18},(_,i)=>({id:`fixture-${i}`,at:new Date(Date.now()-i*60000).toISOString(),client:'Fixture client',endpoint:'/chat/completions',model:'Qwen3.8-Flash-Next',state:i===2?'error':'done',tokensIn:2100,cachedTokens:i===3?null:1800,tokensOut:1200,tps:43.2,promptTps:1450,hitRate:.85,pcieShare:.1,durationMs:28200,usageSource:'reported',finishReason:i===1?'length':'stop'}));
  const data={...originalMonitor,phase:'generating',phaseLabel:'Generating response',prompt:{done:2100,total:2100,pct:100},generation:{done:120,total:600},outputTps:43.2,promptTps:1450,queue:{running:1,waiting:2},context:{usedTokens:8192,maxTokens:32768,pct:25},expertCache:{used:48,total:64,memoryGb:8,source:'engine'},hardware:{gpus,cpus:[{id:'socket-0',utilizationPct:32},{id:'socket-1',utilizationPct:78}],cpuLoad:55,vramUsedGb:14,vramTotalGb:48,gpuLoad:60,gpuTempC:78,ramUsedGb:48,ramTotalGb:128,tempLimitC:90,powerW:250,powerLimitW:800,pcieRxMbs:440,diskReadGbs:.4,diskWriteMbs:3.2},history:Array.from({length:60},(_,i)=>({t:Date.now()-(59-i)*1000,outputTps:25+Math.sin(i/4)*12+i*.1,promptTps:i<18?900+Math.sin(i/3)*500:0,gpus:gpus.map((g,index)=>({...g,utilizationPct:g.utilizationPct===null?null:g.utilizationPct+Math.sin(i/4+index)*8,powerW:g.powerW===null?null:g.powerW+Math.sin(i/5)*12})),cpus:[{id:'socket-0',utilizationPct:25+Math.sin(i/4)*10},{id:'socket-1',utilizationPct:70+Math.sin(i/4)*15}],gpuLoad:60,vramUsedGb:14,ramUsedGb:48,gpuTempC:78,cpuLoad:50+Math.sin(i/4)*15,powerW:250,pcieRxMbs:100+Math.sin(i/4)*90,diskReadGbs:.3+Math.sin(i/5)*.1})),requests,totals:{...originalMonitor.totals,requests:18,tokensIn:37800,tokensOut:21600,cachedTokens:32400,uncachedTokens:5400,promptMs:5400/1450*1000,decodeMs:21600/43.2*1000},conversationCache:{source:'engine',enabled:true,parked:2,slots:4,memoryGb:2,budgetGb:8,parks:3,restores:1,evictions:0,lastEvent:'restored',lastTokens:2048,lastAt:Date.now()/1000-20},mcp:{tools:2,servers:[{name:'Workspace files',state:'ready',transport:'stdio',tools:[{name:'read_file'},{name:'list_directory'}]}]},telemetry:{throughput:'engine',memory:'engine',thermals:'engine',system:'sensor',context:'reported'}};
  if(fullUse){
    Object.assign(data.hardware,{gpuLoad:97,vramUsedGb:45,gpuTempC:82,powerW:730,pcieRxMbs:1960,cpuLoad:84,ramUsedGb:92});
    data.hardware.cpus=[{id:'socket-0',utilizationPct:80},{id:'socket-1',utilizationPct:88}];
    data.context={usedTokens:24576,maxTokens:32768,pct:75};
    data.generation={done:1800,total:2400};
    data.expertCache={used:64,total:64,memoryGb:20,source:'engine'};
    data.history=data.history.map((sample,i)=>({...sample,cpuLoad:84+Math.sin(i/4)*6,ramUsedGb:92,gpus:sample.gpus.map((g,index)=>({...g,utilizationPct:Math.max(0,Math.min(100,90+index*2+Math.sin(i/4)*7))})),cpus:[{id:'socket-0',utilizationPct:80+Math.sin(i/4)*8},{id:'socket-1',utilizationPct:88+Math.sin(i/4)*8}]}));
  }
  if(unloaded){
    Object.assign(data,{variantId:null,phase:'idle',phaseLabel:'No model loaded',prompt:{done:0,total:0,pct:0},generation:{done:null,total:null},outputTps:0,promptTps:0,queue:{running:0,waiting:0},context:{usedTokens:0,maxTokens:32768,pct:0},expertCache:{used:0,total:64,source:'none'},requests:[],conversationCache:{source:'unavailable'}});
    Object.assign(data.hardware,{gpus,cpuLoad:5,cpus:[{id:'socket-0',utilizationPct:4},{id:'socket-1',utilizationPct:6}],ramUsedGb:8,gpuLoad:2,vramUsedGb:.6,gpuTempC:34,powerW:36,pcieRxMbs:0,diskReadGbs:.0001,diskWriteMbs:.2});
    data.history=data.history.map((sample,i)=>({...sample,outputTps:0,promptTps:0,gpus:gpus.map(g=>({...g,utilizationPct:2+Math.sin(i/4)})),cpus:[{id:'socket-0',utilizationPct:4},{id:'socket-1',utilizationPct:6}],cpuLoad:5+Math.sin(i/4),ramUsedGb:8,diskReadGbs:.0001}));
    Object.assign(data.totals,{requests:0,tokensIn:0,tokensOut:0,cachedTokens:0,uncachedTokens:0,promptMs:0,decodeMs:0});
    data.telemetry={...data.telemetry,throughput:'none',context:'none'};
  }
  return data;
}
/** Full-app screenshot scenarios, with only the devices present in each example. */
function captureScenario(mode) {
  const fullUse=mode==='capture-full-use',unloaded=mode==='capture-no-use';
  const data=monitor(false,unloaded,fullUse),inventory=structuredClone(hardware);
  if(mode!=='capture-multi-gpu')inventory.gpus=inventory.gpus.filter(g=>g.id==='gpu-b');
  if(mode!=='capture-dual-cpu')inventory.processors=inventory.processors.slice(0,1);
  const gpuIds=inventory.gpus.map(g=>g.id),cpuIds=inventory.processors.map(p=>p.id);
  const filterReadings=sample=>{
    sample.gpus=sample.gpus.filter(g=>gpuIds.includes(g.id));
    sample.cpus=sample.cpus.filter(p=>cpuIds.includes(p.id));
    sample.gpuLoad=sample.gpus.reduce((n,g)=>n+g.utilizationPct,0)/sample.gpus.length;
    sample.vramUsedGb=sample.gpus.reduce((n,g)=>n+g.memoryUsedGb,0);
    sample.gpuTempC=Math.max(...sample.gpus.map(g=>g.temperatureC));
    sample.powerW=sample.gpus.reduce((n,g)=>n+g.powerW,0);
    sample.powerLimitW=sample.gpus.reduce((n,g)=>n+g.powerLimitW,0);
    sample.pcieRxMbs=sample.gpus.reduce((n,g)=>n+g.pcieRxMbs,0);
    sample.cpuLoad=sample.cpus.reduce((n,p)=>n+p.utilizationPct,0)/sample.cpus.length;
  };
  filterReadings(data.hardware);
  data.history.forEach(filterReadings);
  inventory.cpu={...inventory.cpu,cores:inventory.processors.reduce((n,p)=>n+p.cores,0),threads:inventory.processors.reduce((n,p)=>n+p.threads,0)};
  inventory.memory.totalGb=inventory.processors.length===1?64:128;
  inventory.memory.usedGb=unloaded?8:fullUse?48:32;
  inventory.memory.freeGb=inventory.memory.totalGb-inventory.memory.usedGb;
  data.hardware.ramUsedGb=inventory.memory.usedGb;
  data.hardware.ramTotalGb=inventory.memory.totalGb;
  data.hardware.vramTotalGb=inventory.gpus.reduce((n,g)=>n+g.vramGb,0);
  data.history.forEach(sample=>{sample.ramUsedGb=inventory.memory.usedGb;});
  return {inventory,data};
}
http.createServer(async(req,res)=>{
  try {
    const url=new URL(req.url,'http://localhost'),path=url.pathname;
    if(req.method!=='GET'){res.writeHead(405);res.end();return;}
    const mode=req.headers.cookie?.match(/(?:^|;\s*)fixture=([^;]+)/)?.[1]??'normal';
    const capture=mode.startsWith('capture-')?captureScenario(mode):null;
    const unavailable=mode==='unavailable',unloaded=mode==='unloaded'||mode==='capture-no-use',fullUse=mode==='full-use';
    if(['/capture-full-use','/capture-no-use','/capture-multi-gpu','/capture-single-gpu','/capture-dual-cpu'].includes(path)){res.writeHead(302,{'set-cookie':`fixture=${path.slice(1)}; Path=/`,'location':'/#/monitor'});res.end();return;}
    if(['/unloaded','/full-use'].includes(path)){res.writeHead(302,{'set-cookie':`fixture=${path.slice(1)}; Path=/`,'location':'/#/monitor'});res.end();return;}
    if(path==='/unavailable'){res.writeHead(302,{'set-cookie':'fixture=unavailable; Path=/','location':'/#/monitor'});res.end();return;}
    if(path==='/normal'){res.writeHead(302,{'set-cookie':'fixture=normal; Path=/','location':'/#/monitor'});res.end();return;}
    if(path==='/api/events'){
      res.writeHead(200,{'content-type':'text/event-stream','cache-control':'no-cache'});res.write(': ready\n\n');
      const timer=setInterval(()=>res.write(`event: monitor\ndata: ${JSON.stringify(capture?captureScenario(mode).data:monitor(unavailable,unloaded,fullUse))}\n\n`),1000);
      res.on('close',()=>clearInterval(timer));return;
    }
    if(path.startsWith('/api/')){
      let data=path==='/api/hardware'?(capture?.inventory??hardware):path==='/api/monitor'?(capture?.data??monitor(unavailable,unloaded,fullUse)):await(await fetch(upstream+req.url)).json();
      if(path==='/api/models'&&unloaded)data={...data,loaded:null,variants:data.variants.map(v=>({...v,state:v.state==='loaded'?'installed':v.state}))};
      if(path==='/api/setup')data={...data,complete:true,dismissed:true};
      res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify(data));return;
    }
    const file=resolve(root,path==='/'?'index.html':'.'+path);
    if(!file.startsWith(root+'/')){res.writeHead(403);res.end();return;}
    const body=await readFile(file);
    res.writeHead(200,{'content-type':{'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.woff2':'font/woff2'}[extname(file)]??'application/octet-stream'});res.end(body);
  }catch(error){res.writeHead(500);res.end(error.message);}
}).listen(8796,'127.0.0.1',()=>console.log('Monitor fixture at http://127.0.0.1:8796/#/monitor'));
