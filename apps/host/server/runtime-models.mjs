/** Independent model identities for the design host. Native engines remain capability gated. */
import { randomUUID } from 'node:crypto';
import { RequestQueue } from './request-queue.mjs';
const deviceId = gpu => gpu.id ?? `${gpu.vendor}:${gpu.index}`;
const active = r => ['queued','generating'].includes(r.state);

export function modelReservation(model, config, hardware, gpuIds, checkCapacity=true) {
  const expert = Object.fromEntries((config.expert ?? []).map(p => [p.id,p.value]));
  const label = config.beginner?.contextLength === 'recommended' ? '128k' : config.beginner?.contextLength;
  const context = Number.parseInt(label ?? '128k') * 1024;
  if (!Number.isFinite(context) || context < 4096 || checkCapacity && context > model.contextMax) throw new Error('Choose a supported conversation context.');
  const chosen = hardware.gpus.filter(g => gpuIds.includes(deviceId(g)));
  const reserve = config.extra?.headroom === 'full' ? Math.max(2,hardware.memory.totalGb*.125) : Math.max(4,hardware.memory.totalGb*.25);
  const cache = (model.kvKbPerToken ?? 24) * context / 1024**2 * (expert['ctx.kvMode']==='FP16'?2:expert['ctx.kvMode']==='Q4_0'?.5:1) * Math.max(1,Number(expert['server.parallel'] ?? 1));
  const fixed = cache + (model.experts>0?2.5:1.2);
  const weights = model.diskGb;
  // Equal placement is a planning assumption, never a claim about engine allocation.
  const dedicated = chosen.filter(g=>!g.unified);
  const gpuBudget = dedicated.reduce((n,g)=>n+Math.max(0,g.vramGb-1),0);
  const gpu = Math.min(weights+fixed,gpuBudget);
  const ramBudget = Math.max(0,hardware.memory.totalGb-reserve);
  const ram = Math.min(ramBudget,Math.max(0,weights+fixed-gpu));
  const spill = Math.max(0,weights+fixed-gpu-ram);
  if (checkCapacity && (spill>weights || (expert['mem.mmap']===false || expert['mem.lock']===true) && spill>0)) throw new Error('This setup cannot fit its context and runtime in available memory. Reduce context or use a smaller version.');
  return { minimumRamGb:Math.max(0,fixed-gpu), pageable:expert['mem.mmap']!==false && expert['mem.lock']!==true, context, ramGb:ram, gpuGb:gpu, spillGb:spill, reserveGb:reserve, totalGb:weights+fixed, devices:dedicated.map(g=>({id:deviceId(g),gb:gpu/dedicated.length})) };
}

/** RAM is one host pool. Dedicated VRAM is checked separately on each selected GPU. */
export function rebalanceReservations(instances, hardware) {
  const rows=structuredClone(instances);
  const reserve=Math.min(hardware.memory.totalGb,Math.max(0,...rows.map(i=>i.memory.reserveGb)));
  const budget=Math.max(0,hardware.memory.totalGb-reserve);
  let excess=Math.max(0,rows.reduce((n,i)=>n+i.memory.ramGb,0)-budget);
  const flexible=rows.filter(i=>i.memory.pageable);
  const spare=flexible.reduce((n,i)=>n+Math.max(0,i.memory.ramGb-i.memory.minimumRamGb),0);
  if(excess>spare+.01)return null;
  for(const row of flexible){const reduction=spare?excess*Math.max(0,row.memory.ramGb-row.memory.minimumRamGb)/spare:0;row.memory.ramGb-=reduction;row.memory.spillGb+=reduction;}
  return rows;
}

export function reservationBudget(instances, hardware) {
  const reserveGb = Math.min(hardware.memory.totalGb,Math.max(0,...instances.map(i=>i.memory.reserveGb)));
  const ramGb = instances.reduce((n,i)=>n+i.memory.ramGb,0);
  const devices = hardware.gpus.filter(g=>!g.unified).map(g=>({id:deviceId(g),name:g.name,usedGb:instances.reduce((n,i)=>n+(i.memory.devices.find(d=>d.id===deviceId(g))?.gb??0),0),budgetGb:Math.max(0,g.vramGb-1)}));
  return { ramGb, budgetGb:Math.max(0,hardware.memory.totalGb-reserveGb), reserveGb, totalGb:hardware.memory.totalGb, devices, fits:ramGb<=hardware.memory.totalGb-reserveGb+.01 && devices.every(d=>d.usedGb<=d.budgetGb+.01) };
}

export function createRuntimeModels({hardware, library, requests=()=>[], changed=()=>{}, real=false}) {
  const records = new Map(), queues = new Map();
  let defaultId = null, revision = 0;
  const publish = () => { revision++; changed(state()); };
  function devices(ids) {
    if (!Array.isArray(ids) || new Set(ids).size!==ids.length || ids.some(id=>!hardware().gpus.some(g=>deviceId(g)===id))) throw new Error('Choose detected GPUs, or CPU only.');
    return hardware().gpus.filter(g=>ids.includes(deviceId(g))).map(deviceId);
  }
  function prepare(body) {
    const model = library().variants.find(v=>v.id===body.variantId);
    if (!model) throw new Error('This model version is unavailable.');
    const gpuIds = devices(body.gpuIds);
    const config = structuredClone(body.config);
    if (!config?.beginner || !Array.isArray(config.expert)) throw new Error('A validated engine configuration is required.');
    if (gpuIds.length===0) {
      const layers=config.expert.find(p=>p.id==='gpu.layers');
      if(layers)layers.value='CPU only'; else config.expert.push({id:'gpu.layers',value:'CPU only'});
    } else if(config.expert.find(p=>p.id==='gpu.layers')?.value==='CPU only') throw new Error('Choose CPU only as the execution device, or enable GPU layers.');
    if(!real && gpuIds.length && !['Automatic','All layers',undefined].includes(config.expert.find(p=>p.id==='gpu.layers')?.value))throw new Error('Partial GPU layer placement needs a verified allocation plan. Choose Automatic, All layers, or CPU only.');
    const memory = modelReservation(model,config,hardware(),gpuIds,!real);
    return {variantId:model.id,family:model.family,name:model.name,quant:model.quant,gpuIds,config,memory};
  }
  function state() {
    const entries=[...records.values()].map(i=>{
      const log=requests().filter(r=>r.instanceId===i.id);
      return {...structuredClone(i),active:log.filter(r=>r.state==='generating').length,queued:log.filter(r=>r.state==='queued').length};
    });
    return {revision,mode:real?'real':'simulated',capabilities:{multipleModels:!real,deviceAssignment:!real},defaultId,instances:entries,memory:reservationBudget(entries,hardware())};
  }
  function add(prepared, apiModelId) {
    const id=randomUUID();
    const instance={...prepared,id,apiModelId,loadedAt:new Date().toISOString(),status:'ready',phase:'idle',prompt:{done:0,total:0,pct:0},outputTps:0,promptTps:0,context:{usedTokens:0,maxTokens:prepared.memory.context,pct:0}};
    records.set(id,instance);
    queues.set(id,new RequestQueue(()=>publish()));
    if(!defaultId)defaultId=id;
    return instance;
  }
  function adopt(variantId,config) {
    if (!variantId) { if(real){records.clear();queues.clear();defaultId=null;} return; }
    const existing=[...records.values()].find(i=>i.variantId===variantId);
    if(real) { for(const i of records.values())if(i.variantId!==variantId){records.delete(i.id);queues.delete(i.id);} }
    if(existing){if(real)try{const prepared=prepare({variantId,config,gpuIds:config.expert?.find(p=>p.id==='gpu.layers')?.value==='CPU only'?[]:hardware().gpus.map(deviceId)});Object.assign(existing,prepared,{context:{...existing.context,maxTokens:prepared.memory.context}});}catch{}defaultId=existing.id;return;}
    const model=library().variants.find(v=>v.id===variantId);if(!model)return;
    const ids=real&&config.expert?.find(p=>p.id==='gpu.layers')?.value==='CPU only'?[]:hardware().gpus.map(deviceId);
    try { const prepared=prepare({variantId,config,gpuIds:ids});const row=add(prepared,variantId);defaultId=row.id; }
    catch { /* A legacy configuration may be invalid; never fabricate a ready instance. */ }
  }
  function executionMode(body) {
    const mode=body.executionMode??'single';
    if(!['single','multiple'].includes(mode))throw new Error('Choose single or multiple model execution.');
    return mode;
  }
  function samePlacement(prepared) {
    return [...records.values()].find(i=>i.variantId===prepared.variantId && JSON.stringify(i.gpuIds)===JSON.stringify(prepared.gpuIds));
  }
  function plan(body) {
    const mode=executionMode(body),prepared=prepare(body),matching=samePlacement(prepared);
    const identical=matching && JSON.stringify(matching.config)===JSON.stringify(prepared.config);
    const siblings=mode==='multiple'?[...records.values()].filter(i=>i.id!==matching?.id):[];
    const planned=rebalanceReservations([...siblings,prepared],hardware());
    const budget=reservationBudget(planned??[...siblings,prepared],hardware());
    const candidate=planned?.at(-1)??prepared;
    return {memory:candidate.memory,budget,allowed:!!planned&&budget.fits,alreadyLoaded:!!identical,reloads:!!matching,executionMode:mode};
  }
  function load(body) {
    if(real)throw new Error('Multiple native engines and GPU assignment are not supported by this host yet.');
    const mode=executionMode(body),prepared=prepare(body),matching=samePlacement(prepared);
    const model=library().variants.find(v=>v.id===prepared.variantId);
    if(!model.installedAt && !['loaded','installed'].includes(model.state))throw new Error('Download this model version before loading.');
    const identical=matching && JSON.stringify(matching.config)===JSON.stringify(prepared.config);
    if(identical && (mode==='multiple'||records.size===1))return {state:state(),instance:structuredClone(matching)};
    const affected=mode==='single'?[...records.values()]:matching?[matching]:[];
    if(affected.some(i=>requests().some(r=>r.instanceId===i.id&&active(r))))throw new Error(mode==='single'?'Finish or cancel model requests before switching models.':'Finish or cancel this model’s requests before reloading it.');
    if(mode==='multiple'&&!matching&&records.size>=8)throw new Error('Unload a model before adding another. This preview supports eight instances.');
    const siblings=mode==='multiple'?[...records.values()].filter(i=>i.id!==matching?.id):[];
    const planned=rebalanceReservations([...siblings,prepared],hardware());
    if(!planned || !reservationBudget(planned,hardware()).fits)throw new Error(mode==='single'?'This model cannot fit its context and runtime in available memory. Reduce context or use a smaller version.':'Other loaded models have reserved this memory. Unload one, choose another GPU, or reduce context.');
    // Validation and all request/memory checks precede the switch. Failed loads keep the current model.
    if(mode==='single')for(const row of records.values())if(row.id!==matching?.id){records.delete(row.id);queues.delete(row.id);}
    let suffix=1,apiModelId=matching?.apiModelId??prepared.variantId;
    while(!matching&&[...records.values()].some(i=>i.apiModelId===apiModelId))apiModelId=`${prepared.variantId}--${++suffix}`;
    for(const row of planned.slice(0,-1))records.get(row.id).memory=row.memory;
    prepared.memory=planned.at(-1).memory;
    const instance=matching?Object.assign(matching,prepared,{loadedAt:new Date().toISOString(),phase:'idle',prompt:{done:0,total:0,pct:0},outputTps:0,promptTps:0,context:{usedTokens:0,maxTokens:prepared.memory.context,pct:0}}):add(prepared,apiModelId);
    if(mode==='single')defaultId=instance.id;
    publish();return {state:state(),instance:structuredClone(instance)};
  }
  function resolve(modelId,{chatDefault=false}={}) {
    if(modelId!==undefined && typeof modelId!=='string')throw new Error('Use a model ID from /v1/models.');
    if(modelId){const row=[...records.values()].find(i=>i.apiModelId===modelId);if(!row)throw new Error('That model ID is not loaded. Choose one from /v1/models.');return row;}
    if(!records.size)throw new Error('No model is loaded.');
    if(!chatDefault && records.size>1)throw new Error('Choose a model ID in your request when multiple models are loaded.');
    return records.get(defaultId);
  }
  function unload(id) {
    if(!records.has(id))throw new Error('This model is no longer loaded.');
    if(requests().some(r=>r.instanceId===id && active(r)))throw new Error('Finish or cancel this model’s requests before unloading it.');
    records.delete(id);queues.delete(id);if(defaultId===id)defaultId=records.keys().next().value??null;publish();return state();
  }
  function setDefault(id) {if(!records.has(id))throw new Error('This model is no longer loaded.');defaultId=id;publish();return state();}
  function update(id,patch) {const row=records.get(id);if(row){Object.assign(row,patch);publish();}}
  function libraryState(value) {const ids=new Set([...records.values()].map(i=>i.variantId));return {...value,loaded:records.get(defaultId)?.variantId??value.loaded,variants:value.variants.map(v=>ids.has(v.id)?{...v,state:'loaded'}:v)};}
  return {state,prepare,plan,adopt,load,resolve,unload,setDefault,update,libraryState,queue:id=>queues.get(id),hasVariant:id=>[...records.values()].some(i=>i.variantId===id)};
}
