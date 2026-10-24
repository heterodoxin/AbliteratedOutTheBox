import type { Hardware, ModelVariant, ModelLibrary } from './types';
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
export function modelFamilies(library:ModelLibrary){return [...new Set(library.variants.map(v=>v.family))];}
export function modelVersions(library:ModelLibrary, selectedFamily?:string){const family=selectedFamily??library.variants.find(v=>v.id===(library.loaded??library.current))?.family??library.variants[0]?.family;return library.variants.filter(v=>v.family===family).sort((a,b)=>a.diskGb-b.diskGb);}
/** Keep family choices within their exact catalog identity, preferring downloaded files. */
export function familyVersion(library:ModelLibrary,family:string,quant?:string,remembered?:string){
 const versions=modelVersions(library,family),installed=versions.filter(v=>['installed','loaded','loading','unloading'].includes(v.state)||!!v.installedAt);
 return versions.find(v=>v.id===remembered)??versions.find(v=>v.id===library.loaded)??installed.find(v=>v.quant===quant)??installed[0]??versions.find(v=>v.quant===quant)??versions.find(v=>v.quant==='Q4_K_M')??versions[0];
}
export function recommendedVersion(versions:ModelVariant[],hardware:Hardware|null,context=32768){return versions.filter(v=>versionFit(v,hardware,context).zone==='fast').at(-1);}
export const VERSION_FIT_LABEL={fast:'Fast fit',slower:'Slower · system memory','too-big':'Too big','unknown':'Fit unknown'};
