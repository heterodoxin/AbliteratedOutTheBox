import {USER_KEY,flagValue} from './native-user-args.mjs';

/** The engine's own default reserve, and the one setup writes for pictures read on the GPU: neither is a choice of the person. */
const ENGINE_DEFAULT_RESERVE_MIB=700;
/** A card that drives no display needs little beside the engine; the engine keeps the last few hundred MiB for allocations it makes after the cache. */
const COMPUTE_ONLY_RESERVE_MIB=512;

const wholeMib=v=>v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v))&&Number(v)>=0?Math.trunc(Number(v)):null;

/** The graphics memory the person asked to keep free, or null when they asked for nothing (#3215 #3270): the Expert setting of this load,
 *  "user_args", the setting saved for the model ("vram_reserve_mib", unless it is the value the app itself wrote last), or a value written by hand into "args"
 *  (not the engine's default 700, and not the one the app wrote). Any number counts, 0 too. */
function candidateReserve(config,values={},legacyDefaults=new Set()){
 if(values&&typeof values==='object'&&'vram_reserve_mib' in values){const v=wholeMib(values.vram_reserve_mib);if(v!==null)return v;}
 const user=wholeMib(flagValue(config?.[USER_KEY],'--vram-reserve-mib'));if(user!==null)return user;
 const applied=config?.ui_reserve_applied,legacy=applied===undefined&&config?.ui_resource_baseline!==undefined;   // 0.2.0-beta wrote the key itself, without saying so
 const key=wholeMib(config?.vram_reserve_mib),arg=wholeMib(flagValue(config?.args,'--vram-reserve-mib'));
 const keyOk=key!==null&&key!==applied,argOk=arg!==null&&arg!==applied&&arg!==ENGINE_DEFAULT_RESERVE_MIB;
 if(legacy)return [arg,key].map((n,i)=>(i===0?argOk:keyOk)?n:null).find(n=>n!==null&&!(n===1536||n===2048||legacyDefaults.has(n)))??null;   // a 0.2.0-beta file: the app wrote both, so what is not one of its own numbers is the person's
 return keyOk?key:argOk?arg:null;
}

/** What the app keeps free when the person asked for nothing: [work, performance]. A card that drives a display (or is not known to be headless) gets no
 *  less than 1536 MiB on a card of 16 GB or less, 1024 above, 2048 with pictures (the engine warns below ~1-1.5 GiB of free VRAM: the driver then pages GPU memory to
 *  system RAM and decoding falls to a few tokens a second, #2065); "work" also keeps a fifth of the card. A card the probe says drives no display keeps 512 MiB
 *  (700 with pictures) on either profile. */
export function defaultReserves(config,hardware,baselineReserve=0){
 const placement=Array.isArray(config.gpu)?config.gpu:[config.gpu??0];
 const placed=(hardware?.gpus??[]).filter(g=>placement.includes(g.index)),gpuMemory=placed.map(g=>g.vram_gb).filter(n=>Number.isFinite(n)&&n>0);
 const headless=placed.length>0&&placed.every(computeOnly);   // no display on the card (the desktop is on an integrated GPU): no desktop to keep room for
 /* Only a picture reader that runs on the GPU needs VRAM of its own: the CPU reader (vision.gpu false) took the same 2048 floor and cost a 12 GB card a quarter of its expert cache (0.2.3 tuning, notes-023-tune.md). */
 const pictures=config.vision?(typeof config.vision==='object'?Boolean(config.vision.gpu):true):false;
 const floor=headless?(pictures?ENGINE_DEFAULT_RESERVE_MIB:COMPUTE_ONLY_RESERVE_MIB):gpuMemory.length&&Math.min(...gpuMemory)>16?1024:1536;
 const safety=headless?floor:Math.max(pictures?2048:floor,baselineReserve);
 /* A card that shares the PC's memory (Strix Halo, other APUs) has no memory of its own to keep a fifth of: that fifth is RAM, and "Leave 20% for my work" already keeps it
  *  through the RAM budget. A fifth of the shared pool here held back 16 GB of expert cache for nothing (#3510); the display floor stays. */
 const shared=placed.length>0&&placed.some(g=>g.uma===true||g.unified===true);
 /* "work" keeps a fifth of the card, but on a card of 16 GB or less no more than 1536 MiB: a fifth of 12 GB (2457) cost 20% of the speed against 1536, which already leaves 1.1 GiB of VRAM free after the load (RTX 5070, Gyro-S 64K, 5 pairs; notes-023-tune.md). */
 const fifth=gpuMemory.length?Math.floor(Math.min(...gpuMemory)*1024*.2):0,cap=gpuMemory.length&&Math.min(...gpuMemory)<=16?1536:Infinity;
 return {work:gpuMemory.length&&!headless&&!shared?Math.max(safety,Math.min(fifth,cap)):safety,performance:safety};
}

/** The reserve the person asked for (null: nothing) and the default for each profile. A 0.2.0-beta file has the key and the same number in "args" with no note that the
 *  app wrote them: a number that is one of the app's two defaults is the app's, any other is the person's. */
export function reserveChoice(config,hardware,values={},baselineReserve=0){
 const defaults=defaultReserves(config,hardware,config.ui_resource_baseline?.reserve??baselineReserve);
 const asked=candidateReserve(config,values,new Set([defaults.work,defaults.performance]));
 return {asked,defaults};
}
export const explicitReserve=(config,values={},hardware=null)=>reserveChoice(config,hardware,values).asked;

/** True when the probe says this card drives no display: only compute runs on it. Unknown counts as a display card. */
const computeOnly=g=>g?.drives_display===false;

/** Headroom is a worker/memory budget, not a hard utilisation throttle. */
export function resourceSettings(config,hardware,profile,model={},values={}){
 if(!['work','performance'].includes(profile))throw new Error('Choose a resource preference.');
 const args=[...(config.args??[])];
 const remove=flag=>{for(let at=args.indexOf(flag);at>=0;at=args.indexOf(flag))args.splice(at,2);};
 const value=flag=>{const at=args.indexOf(flag);return at>=0?Number(args[at+1])||0:0;};
 const first=reserveChoice(config,hardware,values).asked;
 const baseline=config.ui_resource_baseline??{workers:value('--pool-workers'),reserve:first!==null?0:Math.max(Number(config.vram_reserve_mib)||0,value('--vram-reserve-mib'))};
 const {asked,defaults}=reserveChoice(config,hardware,values,baseline.reserve);
 if(profile==='performance'&&config.ui_resource_profile==='work'){
  remove('--pool-workers');if(baseline.workers>0)args.push('--pool-workers',String(baseline.workers));
 }
 const cores=Number(hardware?.cpu?.cores)||0;
 if(profile==='work'&&cores>1){
  const at=args.indexOf('--pool-workers'),previous=at>=0?Number(args[at+1]):0;
  /* 80% of the physical cores rounded UP, never more than the cores less the host thread (the engine's own count): floor(0.8 x 6) = 4 ran 8-10% slower than the engine's 5 on a 6-core Ryzen (notes-023-tune.md); 12 logical threads gave 9 in 0.2.1. */
  const workers=Math.max(1,Math.min(cores-1,Math.ceil(cores*.8)));remove('--pool-workers');
  args.push('--pool-workers',String(previous>0?Math.min(previous,workers):workers));
 }
 // Preserve calibrated worker counts and the installation's safety/vision reserve.
 /* A reserve the person set is theirs, 0 included: no floor and no share of the card (#3215 #3270). Without one the card gets the safety default (defaultReserves). */
 const reserve=asked!==null?asked:defaults[profile];
 remove('--vram-reserve-mib');args.push('--vram-reserve-mib',String(reserve));
 // RAM: "leave 20% for my work" keeps the model's RAM within 80% of the PC. A model whose experts do not fit there (the
 // setup rule: experts plus 10 GB) keeps only that many GiB of them in RAM and reads the rest from the model files
 // (--resident-budget-gib, which implies --mmap-experts). Full performance takes the budget out again.
 if(baseline.ramBudget)remove('--resident-budget-gib');
 let env=null;   // the engine's own "leave RAM free" setting travels in the run config's env
 if(baseline.ramHeadroom&&config.env){env={...config.env};delete env.STRATA_RESIDENT_HEADROOM_GIB;}
 const ram=Number(hardware?.ram_gb)||0,arena=Number(model?.arenaGb)||0;let ramBudget=false;
 if(profile==='work'&&ram>0&&arena>0&&!['--resident-experts','--mmap-experts','--resident-budget-gib'].some(f=>args.includes(f))&&ram*.8<arena+10){
  args.push('--resident-budget-gib',String(Math.max(8,Math.min(Math.floor(ram*.8)-10,Math.floor(arena/1.073741824)))));ramBudget=true;
 }
 // The budget caps what the engine keeps in RAM. "Leave 20%" is also asked of the engine itself: at its start it leaves 20% of the RAM
 // available for other programs (it keeps fewer experts if the PC is busy), and it said nothing about this before.
 let ramHeadroom=false;
 if(ramBudget&&!(config.env?.STRATA_RESIDENT_HEADROOM_GIB)){env={...(env??config.env??{}),STRATA_RESIDENT_HEADROOM_GIB:String(Math.ceil(ram*.2))};ramHeadroom=true;}
 return {args,vram_reserve_mib:reserve,...(asked!==null?{}:{ui_reserve_applied:reserve}),...(env?{env}:{}),ui_resource_baseline:{...baseline,ramBudget,ramHeadroom}};
}

/* One automatic second try after a start that died. Returns null when there is nothing safer to try (the caller then reports the first
   failure as it was) or {config, words, detail}: the same run config with ONE safer memory setting, a sentence for the user and the
   technical line for Details. The caller tries once and never again, and does not save this config. */
export function saferStart(config,hardware,model,failure){
 const text=String(failure??'');
 // A start that died for a reason that is not memory (a missing or bad DLL, a crash, a blocked file) is not helped by a memory setting, and must not be called a memory failure.
 if(/0xC0000135|0xC000007B|0xC0000142|0xC000001D|0xC0000005|0xC0000409|0xC0000022|error while loading shared libraries|cannot open shared object file|\bENOENT\b|exit code 127|exit code 139|illegal instruction/i.test(text))return null;
 const ranOut=/out of memory|memory guard|\boom\b|\bkilled\b|exit code 137|cannot allocate|not enough memory|bad_alloc|insufficient memory/i.test(text),first=ranOut?'The first start ran out of memory':'The first start did not finish';
 if(!/server_exited|exited before|died before|engine stopped|pinned-memory-exhausted|mapped staging|out of memory|memory guard|\bkilled\b/i.test(text))return null;
 // The engine stopped because the GPU's expert cache got no room (a 10 GB card with the app's 20% reserve and pictures on): the engine can shrink its own
 // reserve only when none was given, and the app always gives one. Ask for less kept free (never under 1024 MiB: below that the driver pages
 // graphics memory to RAM, #2065) and, when that is not enough, for a shorter context. Never saved.
 if(/no VRAM (?:is|was) left for the expert cache|needs the profile-filled VRAM expert tier/i.test(text)){
  const args=[...(config.args??[])],at=args.indexOf('--vram-reserve-mib'),cur=at>=0?Number(args[at+1])||0:Number(config.vram_reserve_mib)||0;
  const short=Number((text.match(/about (\d+) MiB more than this card has free/)??[])[1])||700,need=short+256;
  const notes=[];let left=need;
  if(cur>1024){const next=Math.max(1024,cur-need);left=need-(cur-next);if(next<cur){if(at>=0)args[at+1]=String(next);else args.push('--vram-reserve-mib',String(next));notes.push(`${next} MiB of graphics memory kept free for other programs (was ${cur})`);}}
  const cx=args.indexOf('--max-context'),ctx=cx>=0?Number(args[cx+1])||0:0,sess=text.match(/takes (\d+) MiB at (\d+) tokens/);
  if(left>0&&ctx>16384&&sess){
   const session=Number(sess[1]),tokens=Number(sess[2]);let next=ctx;
   while(next>16384&&session*(ctx-next)/tokens<left)next=Math.max(16384,Math.floor(next/2));
   if(next<ctx){args[cx+1]=String(next);notes.push(`a shorter conversation context (${next} tokens, was ${ctx})`);}
  }
  if(!notes.length)return null;
  return {config:{...config,args},words:`This graphics card had no room left for the model's expert cache, so Strata started again with ${notes.join(' and ')}. Nothing is saved; change it in the model's settings if you want it permanently.`,detail:notes.join('; ')};
 }
 const args=[...(config.args??[])],env={...(config.env??{})};
 const ram=Number(hardware?.ram_gb)||0,arena=Number(model?.arenaGb)||0;
 // Windows lets a program lock only about half the RAM for the graphics card; the engine pins the model's experts up to that.
 // When it ran out of locked memory, pin less of them: they stay in RAM, a little slower to reach.
 if(/pinned-memory-exhausted|mapped staging/i.test(text)&&!env.STRATA_ARENA_PIN_GIB){
  const hint=Number((text.match(/STRATA_ARENA_PIN_GIB=(\d+)/)??[])[1]);
  const gib=hint>0?hint:ram>0?Math.max(4,Math.floor(ram/2)-8):16;
  env.STRATA_ARENA_PIN_GIB=String(gib);
  return {config:{...config,env},words:`Windows would not lock as much memory as the model asked for, so Strata started again and locks ${gib} GB of the model instead of all of it. Replies may start a little slower.`,detail:`STRATA_ARENA_PIN_GIB=${gib}`};
 }
 if(['--resident-experts','--mmap-experts'].some(f=>args.includes(f)))return null;
 const at=args.indexOf('--resident-budget-gib');
 if(at>=0){
  const next=Math.floor((Number(args[at+1])||0)*.7);if(next<8)return null;
  args[at+1]=String(next);
  return {config:{...config,args},words:`${first}, so Strata started again and keeps less of the model in RAM (${next} GB); more of it is read from the SSD, which is slower.`,detail:`--resident-budget-gib ${next}`};
 }
 if(!(ram>0&&arena>0))return null;
 const budget=Math.max(8,Math.floor(ram*.5)-4);if(budget>=arena)return null;
 args.push('--resident-budget-gib',String(budget));
 return {config:{...config,args},words:`${first}, so Strata started again in low-memory mode: ${budget} GB of the model stays in RAM and the rest is read from the SSD. It is slower, but it should load.`,detail:`--resident-budget-gib ${budget}`};
}
