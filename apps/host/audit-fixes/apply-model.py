exec((__import__('pathlib').Path(__file__).parent/'apply-shared.py').read_text().split("append('src/lib/format.ts'")[0])
p=root/'server/catalog.mjs';s=p.read_text();a=s.index('export const VARIANTS = [');b=s.index('\nexport function variant',a);real=s[a:b].replace('export const VARIANTS','export const REAL_VARIANTS');(root/'server/real-model-catalog.mjs').write_text('/** Verified file identities for the optional real engine. Never relabel these as demo weights. */\n'+real+'\n');s=s[:a]+'''import { REAL_VARIANTS } from './real-model-catalog.mjs';
// Design model versions are isolated from real downloadable artifacts.
const DESIGN_VERSIONS = [
 ['Q2_K','Smallest',2.9], ['Q3_K_M','Compact',3.9], ['Q4_K_M','Balanced',4.9],
 ['Q5_K_M','High quality',5.7], ['Q6_K','Very high',6.6], ['Q8_0','Near-original',8.5], ['BF16','Original',16],
];
export const VARIANTS = process.env.STRATA_REAL_ENGINE === '1' ? REAL_VARIANTS : DESIGN_VERSIONS.map(([quant,name,bits])=>{
 const diskGb=Math.round(30.5*bits/8*10)/10;
 return {id:`qwen38-${quant.toLowerCase()}`,name,family:'Qwen3.8-Flash-Next',quant,label:`Qwen3.8-Flash-Next · ${name} (${quant})`,params:'30.5B',
 blurb:'The same model at a different precision.',role:quant==='Q4_K_M'?'best-fit':'option',tier:quant==='Q2_K'?'starter':'balanced',
 ramClass:{minGb:diskGb+3,maxGb:diskGb+6,label:`${Math.ceil(diskGb+6)} GB memory`},diskGb,downloadGb:diskGb,contextMax:1048576,
 layerCount:48,experts:128,kvKbPerToken:24,hasDraft:true,kvModes:['INT8','FP16'],tags:[],minMemoryGb:diskGb+3,
 source:{repo:'design-preview',file:`${quant}.gguf`,bytes:Math.round(diskGb*1024**3)},designOnly:true};
});
''' +s[b:];s=s.replace("return `https://huggingface.co/", "if (v.designOnly) return undefined;\n  return `https://huggingface.co/");s=s.replace("'64K', '128K']", "'64K', '128K', '256K', '512K', '1024K']");s=s.replace("  '128k': 131072,", "  '128k': 131072,\n  '256k': 262144,\n  '512k': 524288,\n  '1024k': 1048576,");p.write_text(s)
# Existing preview state is migrated without changing real-engine files.
p=root/'server/models.mjs';s=p.read_text();s=s.replace('  supervisor.init(hub);','''  supervisor.init(hub);
  if (!REAL) {
    const previous = getStore();
    const old = Object.keys(previous.installed ?? {}).filter(id => !VARIANTS.some(v=>v.id===id));
    if (old.length) {
      const balanced=VARIANTS.find(v=>v.quant==='Q4_K_M').id;
      update({installed:{...previous.installed,[balanced]:{installedAt:nowIso(),simulated:true}},
        ...(old.includes(previous.loaded)?{loaded:balanced}:{}),...(old.includes(previous.current)?{current:balanced}: {})});
    }
  }''');s=s.replace('const totalSec = 11 + v.downloadGb * 0.55;', 'const totalSec = 10; // Preview timing is independent of fictional file sizes.');p.write_text(s)
for p in (root/'tests').glob('*.test.mjs'):
 s=p.read_text().replace('qwen2.5-0.5b-q4km','qwen38-q4_k_m');p.write_text(s)
edit('src/lib/types.ts',"'64k' | '128k';", "'64k' | '128k' | '256k' | '512k' | '1024k';")
edit('src/components/EngineExperience.tsx','32768, 65536, 131072]', '32768, 65536, 131072, 262144, 524288, 1048576]')
(root/'src/lib/model-versions.ts').write_text('''import type { Hardware, ModelVariant, ModelLibrary } from './types';
export type VersionFit = {zone:'fast'|'slower'|'too-big'|'unknown';needed:number;fastBudget:number;budget:number;over:number};
/** Fit uses detected pools without double-counting unified memory. No speed benchmark is implied. */
export function versionFit(v:ModelVariant, hardware:Hardware|null, context=32768):VersionFit {
 const needed=v.diskGb+(v.kvKbPerToken??0)*context/1024**2+(v.experts?2.5:1.2);
 if(!hardware)return {zone:'unknown',needed,fastBudget:0,budget:0,over:0};
 const ram=Math.max(0,hardware.memory.totalGb),gpu=Math.max(0,hardware.accelerator.usableVramGb);
 const unified=hardware.accelerator.unified;
 const fastBudget=unified?ram*.72:hardware.accelerator.backend==='cpu'?0:gpu*.92;
 const budget=unified?ram*.9:gpu+ram*.6;
 const zone=context>v.contextMax||needed>budget?'too-big':needed<=fastBudget?'fast':'slower';
 return {zone,needed,fastBudget,budget,over:Math.max(0,needed-budget)};
}
export function modelVersions(library:ModelLibrary){const family=library.mode==='simulated'?'Qwen3.8-Flash-Next':library.variants.find(v=>v.id===(library.loaded??library.current))?.family??library.variants[0]?.family;return library.variants.filter(v=>v.family===family).sort((a,b)=>a.diskGb-b.diskGb);}
export function recommendedVersion(versions:ModelVariant[],hardware:Hardware|null,context=32768){return versions.filter(v=>versionFit(v,hardware,context).zone==='fast').at(-1)??versions.find(v=>versionFit(v,hardware,context).zone==='slower');}
export const VERSION_FIT_LABEL={fast:'Fast fit',slower:'Slower · system memory','too-big':'Too big','unknown':'Fit unknown'};
''')
(root/'src/components/VersionLadder.tsx').write_text('''import type { Hardware, ModelVariant } from '../lib/types';
import { versionFit, VERSION_FIT_LABEL } from '../lib/model-versions';
import { sizeGb } from '../lib/format';
import { Icon } from './ui';
export function VersionLadder({versions,hardware,context,selected,recommended,onSelect}:{versions:ModelVariant[];hardware:Hardware|null;context:number;selected?:string;recommended?:string;onSelect:(id:string)=>void}){
 return <div className="version-ladder" role="group" aria-label="Model versions">{versions.map(v=>{const fit=versionFit(v,hardware,context);return <button key={v.id} className="version-rung" data-fit={fit.zone} aria-pressed={selected===v.id} onClick={()=>onSelect(v.id)}><span className="version-rung__node"><Icon name={v.state==='loaded'?'check':'layers'} size={16}/></span><span><strong>{v.name}</strong><small>{v.quant} · {sizeGb(v.diskGb)}</small></span><span className="version-rung__state">{v.state==='loaded'?'Running':v.state==='downloading'?`${Math.round(v.progress?.pct??0)}%`:recommended===v.id?'Recommended':v.state==='installed'?'Downloaded':VERSION_FIT_LABEL[fit.zone]}</span></button>;})}</div>;
}
''')
(root/'src/screens/Models.tsx').write_text('''import { useState } from 'react';
import { ActionError, Bar, Btn, Dialog, ErrorNotice, Icon, Select } from '../components/ui';
import { Screen } from '../components/layout';
import { VersionLadder } from '../components/VersionLadder';
import { LifecycleNotice } from '../components/LifecycleNotice';
import { modelVersions, recommendedVersion, versionFit, VERSION_FIT_LABEL } from '../lib/model-versions';
import { formatModelName, sizeGb } from '../lib/format';
import { cancelInstall, go, installVariant, loadVariant, removeVariant, unloadModel, useStore, lifecycleStatus } from '../lib/store';
export function ModelsScreen(){
 const library=useStore(s=>s.library),hardware=useStore(s=>s.hardware),engine=useStore(s=>s.engine);const lifecycle=lifecycleStatus(useStore(s=>s));
 const [selected,setSelected]=useState(''),[context,setContext]=useState(32768),[confirm,setConfirm]=useState<'switch'|'delete'|null>(null),[pending,setPending]=useState(false),[error,setError]=useState('');
 if(!library)return <Screen title="Model"><p>Loading model versions…</p></Screen>;
 const versions=modelVersions(library),recommended=recommendedVersion(versions,hardware,context),running=library.variants.find(v=>v.id===library.loaded);
 const version=versions.find(v=>v.id===selected)??running??recommended??versions[0];
 if(!version)return <Screen title="Model"><p>No model versions are available.</p></Screen>;
 const fit=versionFit(version,hardware,context),loaded=version.id===library.loaded,installed=['loaded','installed'].includes(version.state),working=['loading','downloading','unloading'].includes(version.state);
 const contexts=[4096,8192,16384,32768,65536,131072,262144,524288,1048576].filter(n=>n<=version.contextMax);
 async function act(action:()=>Promise<boolean|void>){setPending(true);setError('');try{const ok=await action();if(ok===false)throw new Error('The operation could not finish. Check the engine connection and try again.');setConfirm(null);}catch(e){setError(e instanceof Error?e.message:String(e));}finally{setPending(false);}}
 return <Screen title="Model" sub="One model. A version that fits your computer." actions={<Btn icon="gauge" onClick={()=>go('tuning')}>Engine</Btn>}><div className="stack stack--lg model-page"><ActionError/>
 <section className="model-hero"><span className="model-hero__icon"><Icon name="layers" size={28}/></span><div><span className="eyebrow">{running?'RUNNING':'YOUR MODEL'}</span><h2>{running?formatModelName(running):versions[0].family}</h2><p className="small muted">{running?'Shared by all chats':'Pick a version below to get started.'}</p></div><span className="model-status">{running?'Ready':'Not loaded'}</span></section>
 {!hardware?<ErrorNotice message="Hardware has not been detected." next="Check this computer to see which versions fit." action="Run guided setup" onRetry={()=>go('setup')}/>:null}
 <div className="section-toolbar"><div><h2 className="h3">Choose a version</h2><p className="small muted">Smaller versions use less memory. Larger versions preserve more precision.</p></div><Select ariaLabel="Version context" value={String(context)} onChange={v=>setContext(Number(v))} options={contexts.map(n=>({value:String(n),label:`${n/1024}K context`}))}/></div>
 {!recommended&&hardware?<ErrorNotice message={`${context/1024}K context exceeds the available model budget.`} next="A shorter context may make room for a smaller version." action={context>4096?'Use shorter context':'Review computer'} onRetry={()=>context>4096?setContext(context/2):go('setup')}/>:null}
 <div className="model-version-layout"><VersionLadder versions={versions} hardware={hardware} context={context} selected={version.id} recommended={recommended?.id} onSelect={id=>{setSelected(id);setError('');}}/>
 <section className="version-detail" aria-label="Version details"><div className="section-toolbar"><span className="eyebrow">{version.quant}</span><span className="model-status">{loaded?'Running':installed?'Downloaded':VERSION_FIT_LABEL[fit.zone]}</span></div><h2>{version.name}</h2><p className="small muted">{formatModelName(version)}</p><div className="version-numbers"><div><strong>{sizeGb(version.diskGb)}</strong><span>Download size</span></div><div><strong>{sizeGb(fit.needed)}</strong><span>Estimated memory</span></div></div><p className="small">{fit.zone==='unknown'?'Check your computer to estimate fit.':fit.zone==='too-big'?`${sizeGb(fit.over)} over the estimated memory budget at this context.`:fit.zone==='fast'?'Fits the fast memory budget with room for other apps.':'Fits with less headroom or with weights in slower system memory.'}</p><p className="xs muted">Speed on this PC: {version.performance?.[0]?.source==='measured'?`${version.performance[0].answer.low}–${version.performance[0].answer.high} tok/s`:'Run a speed test in Engine'}</p>
 {working?<><Bar value={version.progress?.pct??0} label={version.state}/><p className="small" role="status">{version.progress?.stage??'Working…'}</p>{version.state==='downloading'?<Btn disabled={pending} onClick={()=>void act(()=>cancelInstall(version.id))}>Cancel download</Btn>:null}</>:loaded?<Btn disabled={pending||!lifecycle.allowed} onClick={()=>void act(unloadModel)}>Unload version</Btn>:installed?<div className="row"><Btn variant="primary" disabled={pending||!lifecycle.allowed||fit.zone==='too-big'} onClick={()=>setConfirm('switch')}>Switch to {version.name}</Btn><Btn icon="trash" disabled={pending} onClick={()=>setConfirm('delete')}>Delete</Btn></div>:<Btn variant="primary" icon="download" disabled={pending||fit.zone==='too-big'} onClick={()=>void act(()=>installVariant(version.id))}>{fit.zone==='too-big'?'Too big for this PC':`Download · ${sizeGb(version.downloadGb)}`}</Btn>}
 {version.error||error?<ErrorNotice message={error||version.error!} next="Your downloaded files are retained."/>:null}<LifecycleNotice/>
 </section></div>
 <section><h2 className="h3">On this computer</h2>{versions.filter(v=>['installed','loaded'].includes(v.state)).map(v=><div className="installed-version" key={v.id}><span><strong>{v.name}</strong><small>{v.quant} · {sizeGb(v.diskGb)}</small></span><Btn size="sm" onClick={()=>setSelected(v.id)}>{v.state==='loaded'?'View running version':'View version'}</Btn></div>)}{!versions.some(v=>['installed','loaded'].includes(v.state))?<p className="small muted">No versions downloaded yet. Choose one above to get started.</p>:null}<p className="small muted">{sizeGb(library.disk.usedGb)} model files · {sizeGb(library.disk.freeGb)} free</p></section>
 </div><Dialog open={!!confirm} title={confirm==='delete'?'Delete this version?':`Switch to ${version.name}?`} onClose={()=>!pending&&setConfirm(null)} actions={<><Btn disabled={pending} onClick={()=>setConfirm(null)}>Cancel</Btn><Btn variant={confirm==='delete'?'danger':'primary'} disabled={pending} onClick={()=>void act(()=>confirm==='delete'?removeVariant(version.id):loadVariant(version.id))}>{pending?'Working…':error?'Try again':confirm==='delete'?'Delete version':'Switch version'}</Btn></>}><p>{confirm==='delete'?`Remove ${sizeGb(version.diskGb)} of downloaded files. You can download this version again.`:`Switch all chats to ${version.name} (${sizeGb(version.diskGb)}). Loading may take a few moments.`}</p>{confirm==='switch'?<details className="disclosure"><summary>Engine settings</summary><p className="small muted">Uses your saved {engine?.beginner.contextLength??'recommended'} context. Change settings in Engine.</p></details>:null}{error?<ErrorNotice message={error}/>:null}</Dialog></Screen>;
}
''')
append('src/styles/app.css','''
.model-hero { display:flex; align-items:center; gap:20px; border:1px solid var(--st-line); border-radius:20px; padding:26px; background:var(--st-surface); }
.model-hero > div { flex:1; min-width:0; }.model-hero h2 { margin:6px 0; font-size:24px; letter-spacing:-.5px; }.model-hero p { margin:0; }
.model-hero__icon { color:var(--st-accent-text); }.model-status { font-size:12px; color:var(--st-ink-soft); white-space:nowrap; }
.model-version-layout { display:grid; grid-template-columns:minmax(260px,1fr) minmax(320px,1fr); gap:28px; align-items:start; }
.version-ladder { display:flex; flex-direction:column; gap:5px; }.version-rung { display:flex; gap:12px; align-items:center; width:100%; text-align:left; padding:13px; background:var(--st-surface); border:1px solid var(--st-line); border-radius:12px; font:inherit; color:var(--st-ink); cursor:pointer; }
.version-rung[aria-pressed='true'] { border-color:var(--st-accent-text); background:var(--st-accent-surface); }.version-rung small { display:block; color:var(--st-ink-muted); font-size:12px; margin-top:4px; }.version-rung__node { color:var(--st-accent-text); }.version-rung__state { margin-left:auto; font-size:11px; color:var(--st-ink-soft); text-align:right; }.version-rung[data-fit='too-big'] .version-rung__node { color:var(--st-ink-muted); }
.version-detail { border:1px solid var(--st-line); border-radius:18px; padding:24px; background:var(--st-surface); position:sticky; top:12px; }.version-detail h2 { font-size:28px; margin:20px 0 4px; }.version-numbers { display:flex; gap:28px; margin:24px 0; }.version-numbers strong { font-size:22px; }.version-numbers span { display:block; font-size:12px; color:var(--st-ink-muted); margin-top:6px; }.installed-version { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--st-line); padding:14px 0; }.installed-version small { display:block; color:var(--st-ink-muted); margin-top:4px; }
@media(max-width:850px) { .model-version-layout { grid-template-columns:1fr; }.model-hero { padding:18px; }.model-hero h2 { font-size:19px; }.version-detail { position:static; } }
''')
