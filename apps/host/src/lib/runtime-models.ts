import type { Hardware, MonitorState, RuntimeModel, RuntimeModels, ServeRequest } from './types';
import { gpuId } from './monitor-hardware';
export function deviceLabel(ids:string[],hardware:Hardware|null){
  if(!ids.length)return 'CPU only';
  return ids.map(id=>{const index=hardware?.gpus.findIndex(g=>gpuId(g)===id)??-1;return index>=0?`GPU ${index+1}`:'GPU unavailable';}).join(' + ');
}
export function scopedMonitor(live:MonitorState|null,runtime:RuntimeModels|null,id:string,requests:ServeRequest[]):MonitorState|null{
  if(!live || !runtime)return live;
  const selected=runtime.instances.find(i=>i.id===id);
  const entries=selected?[selected]:runtime.instances;
  const ids=new Set(entries.map(i=>i.id));
  const log=selected?requests.filter(r=>r.instanceId===selected.id):requests;
  if(runtime.mode==='real')return {...live,requests:log};
  const current=entries.find(i=>['generating','thinking'].includes(i.phase))??entries.find(i=>i.phase==='reading');
  const done=log.filter(r=>r.state==='done');
  const phase=current?.phase??'idle';
  const instance=selected??(entries.length===1?entries[0]:undefined);
  const prompt=instance?.prompt??{done:0,total:0,pct:0};
  const count=(key:'tokensIn'|'tokensOut')=>done.reduce((n,r)=>n+(r[key]??0),0);
  return {...live,variantId:instance?.variantId??null,phase,phaseLabel:current?`${current.family} · ${phase==='reading'?'Reading':'Generating'}`:'Waiting for a request',prompt,generation:{done:null,total:null},outputTps:0,promptTps:0,
    queue:{running:requests.filter(r=>ids.has(r.instanceId??'')&&r.state==='generating').length,waiting:requests.filter(r=>ids.has(r.instanceId??'')&&r.state==='queued').length},
    context:instance?.context??{usedTokens:0,maxTokens:0,pct:0},
    telemetry:{thermals:'unavailable',memory:'unavailable',...live.telemetry,throughput:current?'unavailable':done.length?'demo':'unavailable',context:instance&&instance.context.usedTokens>0?'demo':'unavailable'},
    expertCache:{used:0,total:0,source:'unavailable'},conversationCache:undefined,
    history:live.history.map(s=>({...s,outputTps:null,promptTps:null})),requests:log,
    totals:{...live.totals,tokensIn:count('tokensIn'),tokensOut:count('tokensOut'),requests:done.length,cachedTokens:0,uncachedTokens:count('tokensIn'),promptMs:undefined,decodeMs:undefined,avgTps:0}
  };
}
export const instanceLabel=(i:RuntimeModel)=>`${i.family}${i.family.toLowerCase().endsWith(i.name.toLowerCase())?"":` · ${i.name}`}${i.memory.context>0?` · ${i.memory.context/1024}K`:""}`;
