import type {NativeState} from './native';
import type {Hardware,MonitorState,MonitorSample,GpuReading,ServeRequest} from './types';
import {gpuId,gpuCapacity,hasReading} from './monitor-hardware';
const value=(n:unknown)=>hasReading(n)?n:null;
const bytes=(n:unknown)=>hasReading(n)?n/1024**3:null;
const reading=(n:unknown)=>value(n)??NaN;
const total=(list:(number|null)[])=>list.length&&list.every(hasReading)?list.reduce((sum,n)=>sum+n,0):NaN;

/** Native engine facts projected onto the existing visual dashboard. No demo fallback. */
export function nativeMonitorData(native:NativeState,inventory:Hardware|null,selection='all'):MonitorState{
 const instances=native.instances,selected=instances.find(instance=>instance.id===selection)??(instances.length===1?instances[0]:undefined);
 const scoped=selected?[selected]:instances;
 const raw=selected?.metrics??scoped.filter(instance=>instance.metrics).sort((a,b)=>(b.metrics?.time??0)-(a.metrics?.time??0))[0]?.metrics;
 const live=selected?.metrics?.live,engine=selected?.metrics?.engine;
 const hardware=instances.filter(instance=>instance.metrics?.hardware).sort((a,b)=>(b.metrics?.time??0)-(a.metrics?.time??0))[0]?.metrics?.hardware;
 const devices=new Map<number,Record<string,any>>();
 for(const instance of [...instances].sort((a,b)=>(a.metrics?.time??0)-(b.metrics?.time??0))){
  const h=instance.metrics?.hardware;if(!h)continue;
  if(h.gpus?.length)for(const gpu of h.gpus)devices.set(gpu.index,gpu);
  else{const index=instance.gpus.length===1?instance.gpus[0]:native.hardware?.gpus.length===1?native.hardware.gpus[0].index:null;if(index!==null)devices.set(index,{util:h.gpu_util,mem_used:h.gpu_mem_used,temp:h.gpu_temp,power:h.gpu_power,power_limit:h.gpu_power_limit,pcie_rx_mb:h.gpu_pcie_rx_mb,pcie_gen:h.gpu_pcie_gen,pcie_gen_max:h.gpu_pcie_gen_max,pcie_width:h.gpu_pcie_width});}
 }
 const gpus:GpuReading[]=(inventory?.gpus??[]).map(gpu=>{const r=devices.get(gpu.index);return{id:gpuId(gpu),utilizationPct:value(r?.util),memoryUsedGb:bytes(r?.mem_used),temperatureC:value(r?.temp),powerW:value(r?.power),powerLimitW:value(r?.power_limit),pcieRxMbs:value(r?.pcie_rx_mb),pcieGen:value(r?.pcie_gen),pcieGenMax:value(r?.pcie_gen_max),pcieWidth:value(r?.pcie_width),source:r?'sensor':'unavailable',sampledAt:new Date((raw?.time??Date.now()/1000)*1000).toISOString()};});
 const requests:ServeRequest[]=scoped.flatMap(instance=>(instance.metrics?.requests??[]).map((r:Record<string,any>,index:number)=>({id:String(r.id??`${instance.id}:${r.time}:${index}`),instanceId:instance.id,at:new Date((r.time??0)*1000).toISOString(),client:'Native engine',endpoint:'/v1/chat/completions',state:r.error?'error' as const:'done' as const,tokensIn:value(r.prompt_tokens),tokensOut:value(r.output_tokens),cachedTokens:value(r.reused),durationMs:reading(r.duration_s)*1000,model:instance.label,tps:value(r.decode_tok_s),promptTps:hasReading(r.prompt_ms)&&r.prompt_ms>0&&hasReading(r.prompt_tokens)&&hasReading(r.reused)?(r.prompt_tokens-r.reused)/(r.prompt_ms/1000):null,usageSource:'reported' as const,finishReason:r.finish_reason}))).sort((a,b)=>Date.parse(b.at)-Date.parse(a.at));
 const last=requests[0],phase=selected?.phase??'idle',active=['reading','thinking','generating'].includes(phase);
 const used=active?hasReading(live?.prompt_tokens)&&hasReading(live?.generated)?live.prompt_tokens+live.generated:NaN:last&&hasReading(last.tokensIn)&&hasReading(last.tokensOut)?last.tokensIn+last.tokensOut:NaN;
 const maxContext=reading(engine?.max_context??selected?.context);
 const historyRaw=raw?.history??{},count=Math.max(0,...Object.values(historyRaw).filter(Array.isArray).map(series=>series.length));
 const at=(key:string,index:number)=>{const series=historyRaw[key];return Array.isArray(series)?value(series[index-(count-series.length)]):null;};
 const history:MonitorSample[]=Array.from({length:count},(_,index)=>({t:((raw?.time??Date.now()/1000)-(count-index-1))*1000,outputTps:selected?at('tok_s',index):null,promptTps:selected?at('prefill_tok_s_mean',index):null,cpuLoad:at('cpu',index),ramUsedGb:bytes(at('ram_used',index)),gpuLoad:at('gpu_util',index),vramUsedGb:bytes(at('gpu_mem_used',index)),gpuTempC:at('gpu_temp',index),powerW:at('gpu_power',index),pcieRxMbs:at('gpu_pcie_rx_mb',index),diskReadGbs:hasReading(at('disk_read_mb',index))?at('disk_read_mb',index)!/1024:null}));
 const metricInstance=selected??scoped.filter(instance=>instance.metrics).sort((a,b)=>(b.metrics?.time??0)-(a.metrics?.time??0))[0];
 const historyIndices=raw?.hardware?.gpus?.map((gpu:Record<string,any>)=>gpu.index)??(metricInstance?.gpus.length?metricInstance.gpus:native.hardware?.gpus.length===1?[native.hardware.gpus[0].index]:[]);
 const historyDevices=(inventory?.gpus??[]).filter(gpu=>historyIndices.includes(gpu.index));
 for(const sample of history){
  sample.gpuIds=historyDevices.map(gpuId);
  if(historyDevices.length===1)sample.gpus=[{id:gpuId(historyDevices[0]),source:'sensor',sampledAt:new Date(sample.t).toISOString(),utilizationPct:sample.gpuLoad,memoryUsedGb:sample.vramUsedGb,temperatureC:sample.gpuTempC,powerW:sample.powerW,pcieRxMbs:sample.pcieRxMbs}];
 }
 const totals=scoped.map(instance=>instance.metrics?.totals);
 const tokenTotal=(key:string)=>total(totals.map(item=>value(item?.[key])));
 return{phase:['idle','reading','thinking','generating','loading','error'].includes(phase)?phase as MonitorState['phase']:'error',phaseLabel:selected?.error??(phase==='idle'?'Waiting for a request':phase==='reading'?'Reading your message':phase==='generating'?'Generating a reply':phase==='loading'?'Loading model':'Model unavailable'),variantId:selected?.model??null,
  telemetry:{throughput:raw?'engine':'unavailable',thermals:hardware?'sensor':'unavailable',memory:hardware?'sensor':'unavailable',context:selected&&hasReading(used)?'reported':'unavailable'},
  prompt:{done:reading(live?.prompt_read),total:reading(live?.prompt_total),pct:hasReading(live?.prompt_read)&&live.prompt_total>0?live.prompt_read/live.prompt_total*100:0},generation:{done:value(live?.generated),total:value(live?.max_tokens)},outputTps:reading(live?.tok_s),promptTps:reading(live?.prefill_tok_s_mean),
  queue:{running:total(scoped.map(instance=>instance.metrics?.live?value(instance.metrics.live.running)??(['reading','generating'].includes(instance.metrics.live.state)?1:0):null)),waiting:total(scoped.map(instance=>value(instance.metrics?.live?.queued)))},
  context:{usedTokens:used,maxTokens:maxContext,pct:hasReading(used)&&maxContext>0?used/maxContext*100:NaN},expertCache:{used:reading(engine?.expert_slots),total:reading(engine?.experts),memoryGb:hasReading(engine?.expert_cache_mib)?engine.expert_cache_mib/1024:null,source:engine&&hasReading(engine.expert_slots)?'engine':'unavailable'},
  hardware:{gpus,cpuLoad:value(hardware?.cpu_pct??hardware?.cpu),gpuLoad:value(hardware?.gpu_util),vramUsedGb:bytes(hardware?.gpu_mem_used),vramTotalGb:gpuCapacity(inventory?.gpus??[]),ramUsedGb:bytes(hardware?.ram_used),ramTotalGb:bytes(hardware?.ram_total)??inventory?.memory.totalGb??NaN,gpuTempC:value(hardware?.gpu_temp),tempLimitC:90,powerW:value(hardware?.gpu_power),powerLimitW:value(hardware?.gpu_power_limit),pcieRxMbs:value(hardware?.gpu_pcie_rx_mb),diskReadGbs:hasReading(hardware?.disk_read_mb)?hardware.disk_read_mb/1024:null,diskWriteMbs:value(hardware?.disk_write_mb)},history,requests,
  mcp:raw?.mcp?{servers:raw.mcp.servers??[],tools:raw.mcp.tools??0}:undefined,
  totals:{tokensIn:tokenTotal('prompt_tokens'),tokensOut:tokenTotal('output_tokens'),requests:tokenTotal('requests'),avgTps:NaN,since:selected?.loadedAt??instances[0]?.loadedAt??new Date().toISOString(),ttftMs:NaN,cachedTokens:tokenTotal('reused'),uncachedTokens:tokenTotal('prompt_tokens')-tokenTotal('reused'),promptMs:hasReading(tokenTotal('prompt_ms'))?tokenTotal('prompt_ms'):undefined,decodeMs:hasReading(tokenTotal('decode_ms'))?tokenTotal('decode_ms'):undefined}
 };
}

/** GPU temperature, load, VRAM, power and PCIe come from the engine's metrics while a model runs. With no model loaded (or an engine that
 *  reports none of them) the host's own sensors fill in: nvidia-smi, or the AMD/Intel sysfs readings the host probes itself. */
export function withHostSensors(live:MonitorState,host:MonitorState|null):MonitorState{
 const s=host?.hardware,h=live.hardware;if(!s||!h)return live;
 const empty=(g:GpuReading)=>![g.utilizationPct,g.temperatureC,g.memoryUsedGb,g.powerW].some(hasReading);
 const pick=<T,>(a:T,b:T)=>hasReading(a as unknown)?a:b;
 return {...live,hardware:{...h,
  gpus:(h.gpus??[]).map(g=>{const sensor=s.gpus?.find(x=>x.id===g.id);return sensor&&empty(g)?{...g,...sensor}:g;}),
  gpuLoad:pick(h.gpuLoad,s.gpuLoad),vramUsedGb:pick(h.vramUsedGb,s.vramUsedGb),vramTotalGb:pick(h.vramTotalGb,s.vramTotalGb),gpuTempC:pick(h.gpuTempC,s.gpuTempC),
  powerW:pick(h.powerW,s.powerW),powerLimitW:pick(h.powerLimitW,s.powerLimitW),pcieRxMbs:pick(h.pcieRxMbs,s.pcieRxMbs)}};
}
