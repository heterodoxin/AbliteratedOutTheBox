/** Translate the original Strata /metrics dashboard without inventing missing sensors. */
export function originalDashboard(raw, hardware, fallback) {
  const live=raw.live,engine=raw.engine,hw=raw.hardware??{},history=raw.history??{},cache=raw.conversation_cache,totals=raw.totals;
  const finite=n=>typeof n==='number'&&Number.isFinite(n);
  const gb=n=>finite(n)?n/1024**3:null;
  const devices=hardware?.gpus??[];
  const readings=(hw.gpus??(devices.length===1?[{index:devices[0].index,util:hw.gpu_util,mem_used:hw.gpu_mem_used,temp:hw.gpu_temp,power:hw.gpu_power,power_limit:hw.gpu_power_limit,pcie_gen:hw.gpu_pcie_gen,pcie_gen_max:hw.gpu_pcie_gen_max,pcie_width:hw.gpu_pcie_width,pcie_rx_mb:hw.gpu_pcie_rx_mb}]:[])).map(g=>{
    const device=devices.find(d=>d.index===g.index);
    return {id:device?.id??`${device?.vendor??'unknown'}:${g.index}`,source:'engine',sampledAt:new Date().toISOString(),temperatureC:g.temp??null,utilizationPct:g.util??null,memoryUsedGb:gb(g.mem_used),powerW:g.power??null,powerLimitW:g.power_limit??null,pcieGen:g.pcie_gen??null,pcieGenMax:g.pcie_gen_max??null,pcieWidth:g.pcie_width??null,pcieRxMbs:g.pcie_rx_mb??null};
  });
  const last=raw.requests?.[0],used=live.state!=='idle'&&live.state!=='unloaded'?(live.prompt_tokens??0)+(live.generated??0):(last?.prompt_tokens??0)+(last?.output_tokens??0);
  const maximum=engine.max_context??fallback.context.maxTokens;
  const sampleCount=Math.max(0,...Object.values(history).filter(Array.isArray).map(v=>v.length));
  const samples=Array.from({length:sampleCount},(_,i)=>({t:Date.now()-(sampleCount-1-i)*1000,outputTps:history.tok_s?.[i]??null,promptTps:history.prefill_tok_s_mean?.[i]??null,gpuLoad:history.gpu_util?.[i]??null,vramUsedGb:gb(history.gpu_mem_used?.[i]),gpuTempC:history.gpu_temp?.[i]??null,powerW:history.gpu_power?.[i]??null,pcieRxMbs:history.gpu_pcie_rx_mb?.[i]??null,cpuLoad:history.cpu?.[i]??null,ramUsedGb:gb(history.ram_used?.[i]),diskReadGbs:finite(history.disk_read_mb?.[i])?history.disk_read_mb[i]/1024:null}));
  const maxPct=(used,total)=>total?Math.min(100,used/total*100):0;
  const requests=(raw.requests??[]).map((r,i)=>({id:r.id??`engine-${r.time}-${i}`,at:new Date(r.time*1000).toISOString(),client:'Local engine',endpoint:'/chat/completions',state:r.finish==='error'?'error':'done',tokensIn:r.prompt_tokens??null,tokensOut:r.output_tokens??null,cachedTokens:r.reused??null,tps:r.decode_tok_s??null,promptTps:r.prompt_ms>0?Math.max(0,(r.prompt_tokens??0)-(r.reused??0))/(r.prompt_ms/1000):null,hitRate:r.hit_rate??null,pcieShare:r.pcie_share??null,durationMs:finite(r.duration_s)?r.duration_s*1000:0,finishReason:r.finish??null,usageSource:'reported',model:engine.model}));
  return {
    ...fallback,phase:live.state==='unloaded'?'idle':live.state,phaseLabel:live.state==='generating'?(live.phase??'Generating response'):live.state==='reading'?'Reading prompt':'Waiting for a request',
    prompt:{done:live.prompt_read??0,total:live.prompt_total??0,pct:maxPct(live.prompt_read??0,live.prompt_total)},outputTps:live.tok_s??0,promptTps:live.prefill_tok_s_mean??0,
    generation:{done:live.generated??null,total:live.max_tokens??null},
    queue:{running:live.running??(['reading','generating'].includes(live.state)?1:0),waiting:live.queued??0},context:{usedTokens:used,maxTokens:maximum,pct:maxPct(used,maximum)},
    expertCache:{used:engine.expert_slots??0,total:engine.experts??0,memoryGb:finite(engine.expert_cache_mib)?engine.expert_cache_mib/1024:null,source:finite(engine.expert_slots)?'engine':'none'},
    hardware:{...fallback.hardware,gpus:readings.length?readings:fallback.hardware.gpus,gpuLoad:hw.gpu_util??null,vramUsedGb:gb(hw.gpu_mem_used),vramTotalGb:gb(hw.gpu_mem_total)??fallback.hardware.vramTotalGb,gpuTempC:hw.gpu_temp??null,powerW:hw.gpu_power??null,powerLimitW:hw.gpu_power_limit??null,pcieRxMbs:hw.gpu_pcie_rx_mb??null,cpuLoad:hw.cpu??fallback.hardware.cpuLoad,ramUsedGb:gb(hw.ram_used)??fallback.hardware.ramUsedGb,ramTotalGb:gb(hw.ram_total)??fallback.hardware.ramTotalGb,diskReadGbs:finite(hw.disk_read_mb)?hw.disk_read_mb/1024:null,diskWriteMbs:hw.disk_write_mb??null},
    history:samples.length?samples:fallback.history,requests,
    totals:totals?{...fallback.totals,requests:totals.requests??0,tokensIn:totals.prompt_tokens??0,tokensOut:totals.output_tokens??0,cachedTokens:totals.reused??0,uncachedTokens:(totals.prompt_tokens??0)-(totals.reused??0),promptMs:totals.prompt_ms,decodeMs:totals.decode_ms,since:totals.since?new Date(totals.since*1000).toISOString():fallback.totals.since}:fallback.totals,
    conversationCache:cache?{source:'engine',enabled:cache.enabled,parked:cache.parked,slots:cache.slots,memoryGb:gb(cache.bytes),budgetGb:finite(cache.budget_mib)?cache.budget_mib/1024:null,parks:cache.parks,restores:cache.restores,evictions:cache.evictions,lastEvent:cache.last_event,lastAt:cache.last_at,lastTokens:cache.last_tokens,requests:cache.requests,requestsReused:cache.requests_reused,lastPrompt:cache.last_prompt,lastReused:cache.last_reused}:fallback.conversationCache,
    mcp:raw.mcp?{tools:raw.mcp.tools,servers:(raw.mcp.servers??[]).map(server=>({name:server.name,state:server.status,transport:server.transport,info:server.info,error:server.error,tools:(server.tools??[]).map(tool=>({name:tool.tool??tool.name,description:tool.description}))}))}:undefined,
    telemetry:{...fallback.telemetry,throughput:'engine',context:'reported',memory:'engine',thermals:'engine'},
  };
}
