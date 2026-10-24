import type {NativeFamily,NativeSize} from './native';
import {tr} from './i18n';
import {nvfp4Version} from './native-names';
export interface CatalogVersion extends NativeSize{sourceFamily:string;edition:string;images:boolean}
export interface CatalogModel extends Omit<NativeFamily,'sizes'>{sizes:CatalogVersion[]}
export function nativeVersionLabel(model:string){const nv=nvfp4Version(model);if(nv)return tr(nv.name);return tr(({Q2_0:'Fastest',IQ2_XS:'Compact',IQ3_XXS:'Balanced',IQ3_S:'High quality',IQ1_M:'Coding','Gyro-S':'Smallest','Gyro-M':'Small memory'} as Record<string,string>)[model]??model);}
export const catalogVersionLabel=(version:CatalogVersion)=>{const label=nativeVersionLabel(version.model);return [tr(version.edition),label===version.model&&version.edition?'':label].filter(Boolean).join(' · ');};
/** The one line that says how a version differs from its neighbours (the three NVFP4 versions); empty for the rest. */
export const versionNote=(model:string)=>{const nv=nvfp4Version(model);return nv?tr(nv.note):'';};
/** Display groups are distinct from the engine's installation family IDs. */
export function nativeCatalogModels(families:NativeFamily[]):CatalogModel[]{
 if(!families.length)return[];
 const groups:CatalogModel[]=[{family:'qwen3.8',title:'Qwen3.8-Flash-Next',about:'',experimental:false,images:false,sizes:[]},{family:'qwen3.6',title:'Qwen3.6-35B-A3B',about:'',experimental:false,images:false,sizes:[]}];
 for(const family of families){
  const target=/qwen3[._-]?6|35b[-_]?a3b/i.test(family.title+' '+family.family)?groups[1]:['qwen','swift','coder','unsloth','gyro'].includes(family.family)||/qwen3[._-]?8|flash.next/i.test(family.title)?groups[0]:undefined;
  if(!target)continue;
  const edition=target===groups[0]?({swift:'Swift 1.5',coder:'Coder',unsloth:'Unsloth',gyro:'Gyro'} as Record<string,string>)[family.family]??'':'';
  target.images ||= family.images;
  target.sizes.push(...family.sizes.map(size=>({...size,sourceFamily:family.family,edition,images:family.images})));
 }
 return groups;
}
