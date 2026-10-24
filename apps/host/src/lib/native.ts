import { useEffect, useSyncExternalStore } from 'react';
import { api } from './client';
import type { Hardware, RuntimeModels } from './types';
export interface GpuChoiceOption {key:string;cards:number[];label:string;verdict:'fits'|'tight'|'spills'|'unknown';words:string;notes:string[];recommended:boolean}
export interface GpuChoice {options:GpuChoiceOption[];recommended:string|null;ignored:{name?:string;why_ignored:string}[];cards:number}
export interface NativeSize {/** The first engine that reads this size's files (the catalog), and whether the installed engine is older: the screen then says "needs a newer engine" and offers the update. */engine_min?:string|null;needsNewerEngine?:boolean;id:string;model:string;about:string;download_gb:number;ram_needed_gb:number;experts_gb:number;on_this_pc:string|null;installed:boolean;gpu_options?:GpuChoice|null;images?:boolean;imagesCapable?:boolean}
export interface NativeFamily {family:string;title:string;about:string;experimental:boolean;images:boolean;sizes:NativeSize[]}
export interface NativeField {id:string;label:string;group:string;type:'number'|'boolean'|'string'|'enum'|'array';min?:number;max?:number;choices?:string[];hint?:string;supported?:boolean}
export interface NativeJob {id:string;kind:string;guided?:boolean;model?:string;family?:string;state:'running'|'done'|'error'|'cancelled';startedAt:string;summary:string;error:string|null;details?:string;result:Record<string,any>|null;progress?:Record<string,any>|null}
export interface NativeInstance {id:string;apiModelId:string;model:string;label:string;aliases?:string[];vision?:boolean;resourceProfile?:'work'|'performance'|null;context?:number;port:number;gpus:number[];status:'loading'|'ready'|'unloading'|'error';phase:string;loadedAt:string;error:string|null;metrics:Record<string,any>|null}
export interface NativeState {
  /** The picture reader (strata-vision) is installed with the engine, so models marked for pictures can read them here. */
  /** Strata Control's view of the engine for this computer (the app): usable false = none found or recognised; `notice` says why in plain words. */
  engine?:{usable:boolean;notice?:string|null;wrongMake?:boolean;version?:string|null};
  visionReader?: boolean;available:boolean;repo:string;downloadDirectory?:string;/** Free bytes on the drive that holds the models folder. */downloadFreeBytes?:number|null;downloadTotalBytes?:number|null;version?:string|null;summary:string;error:string|null;catalog:{families:NativeFamily[];contexts:number[];recommendation:{family:string|null;model:string|null;context?:number;why?:string};context_note:string}|null;installed:{models:{model:string;config:string;model_name:string;ready:boolean;/** The picture-reading files are on this computer (notes-split-agent8.md, item 8). */visionPrepared?:boolean;context?:number;kv?:string;images?:string;visionEnabled?:boolean;visionDevice?:string;lowRamMode?:string;resourceProfile?:'work'|'performance';/** The Drafting choice saved with the model: auto | mtp | dflash2 | off. */drafting?:string;dataFolder?:string;diskGb?:number;diskBytes?:number;/** Files this model shares with another (the picture reader, the draft layer): counted once, in `usage`. */sharedBytes?:number;/** What Delete frees: files in Strata's own folders that no other model uses. */removableGb?:number;missing_files?:string[]}[];/** What all models take on disk, from their real files, shared files counted once. */usage?:{total_bytes:number;total_gb:number;shared_bytes:number;shared_gb:number;models:Record<string,{bytes:number;gb:number}>}|null;data_folder:string;engine:{version:string|null}}|null;hardware:{ram_gb:number;engine_vendor?:string;gpus:{index:number;name:string;vram_gb:number;usable:boolean;/** SMs x clock: only the order between cards matters (strata_control/gpu_roles.py). */speed?:number;vendor?:string;uma?:boolean;integrated?:boolean;unified_gb?:number;problem?:string|null}[]}|null;fields:NativeField[];instances:NativeInstance[];defaultId:string|null;jobs:NativeJob[];capabilities:{/** The engine serves DFlash2 (Control says): the Drafting choice then offers it. */dflash2?:boolean;responses?:boolean;updates:boolean;rollback:boolean;calibration:boolean;componentUpdates?:boolean}}
let state:NativeState|null=null;
const listeners=new Set<()=>void>();
function put(value:NativeState){if(state&&JSON.stringify(state)===JSON.stringify(value))return;state=value;listeners.forEach(fn=>fn());}
/** The host pushes its whole native state (jobs included) at every progress step: the page shows it at once, without waiting for the next poll. */
export function applyNativeEvent(value:unknown){if(value&&typeof value==='object'&&'available' in value&&'jobs' in value)put(value as NativeState);}
const READ_LIMIT_MS=60000;
function read(force:boolean):Promise<void>{
 const request=api.get<NativeState>(`/api/native${force?'?refresh=1':''}`);
 // A read that never answers must not freeze the page: after the limit the next poll starts a fresh one.
 let stop:ReturnType<typeof setTimeout>|undefined;
 const limit=new Promise<never>((_,reject)=>{stop=setTimeout(()=>reject(new Error('The host did not answer in time.')),READ_LIMIT_MS);});
 return Promise.race([request,limit]).then(put).catch(error=>{if(state){state={...state,error:error.message,instances:state.instances.map(i=>({...i,metrics:null,phase:'unavailable'}))};listeners.forEach(fn=>fn());}}).finally(()=>clearTimeout(stop));
}
/* Two kinds of read that never wait for each other: the light poll (the host answers at once with the jobs it holds) and the forced read after an action (the host
   re-reads the installation, which can take many seconds while a download is writing). A slow forced read used to hold every later poll, so a running download looked stuck. */
let light:Promise<void>|null=null,forced:Promise<void>|null=null,again=false;
/** A forced refresh (after an action) never joins a read that began before the action: that read cannot know about the new job. It waits for it, then reads again. */
export function refreshNative(force=false):Promise<void>{
 if(!force){if(light)return light;light=read(false).finally(()=>{light=null;});return light;}
 if(forced){if(again)return forced;again=true;return forced.then(()=>{again=false;return refreshNative(true);});}
 forced=read(true).finally(()=>{forced=null;});return forced;
}
const POLL_MS=2500,RUNNING_POLL_MS=1000;
let users=0,timeout:ReturnType<typeof setTimeout>|null=null,watching=false;
const hidden=()=>typeof document!=='undefined'&&document.visibilityState==='hidden';
/** How long until the next poll: quicker while a download or load is running, none while the window is hidden (it reads again the moment it is shown). */
export const nativePollDelay=(value:NativeState|null,isHidden=false):number|null=>isHidden?null:value?.jobs?.some(j=>j.state==='running')?RUNNING_POLL_MS:POLL_MS;
function schedule(){
 if(timeout){clearTimeout(timeout);timeout=null;}
 if(!users)return;
 const wait=nativePollDelay(state,hidden());
 if(wait===null)return;
 timeout=setTimeout(()=>{timeout=null;void refreshNative().finally(schedule);},wait);
}
const shown=()=>{if(!hidden()&&users){void refreshNative();schedule();}};
export function useNative(){
 const value=useSyncExternalStore(fn=>{listeners.add(fn);return()=>listeners.delete(fn);},()=>state);
 useEffect(()=>{
  users++;void refreshNative().finally(schedule);
  if(!watching&&typeof document!=='undefined'){watching=true;document.addEventListener('visibilitychange',shown);window.addEventListener('focus',shown);}
  return()=>{users--;if(!users){if(timeout){clearTimeout(timeout);timeout=null;}if(watching&&typeof document!=='undefined'){watching=false;document.removeEventListener('visibilitychange',shown);window.removeEventListener('focus',shown);}}};
 },[]);
 // a new job (or its end) changes how fast the next poll should come
 const running=value?.jobs?.some(j=>j.state==='running')??false;
 useEffect(()=>{schedule();},[running]);
 return value;
}
export const nativeState=()=>state;
/** Connection-list projection only; memory reservations are not inferred. */
export function nativeRouting(native:NativeState|null,hardware:Hardware|null):RuntimeModels|null{
 if(!native?.instances.length)return null;
 return {revision:Date.now(),mode:'real',capabilities:{multipleModels:true,deviceAssignment:true},defaultId:native.defaultId,instances:native.instances.filter(i=>i.status==='ready').map(i=>({id:i.id,apiModelId:i.apiModelId,variantId:i.model,family:i.label,name:i.model,quant:i.model,loadedAt:i.loadedAt,status:'ready',phase:i.phase as RuntimeModels['instances'][number]['phase'],active:i.metrics?.live?.running??0,queued:i.metrics?.live?.queued??0,prompt:{done:0,total:0,pct:0},outputTps:i.metrics?.live?.tok_s??0,promptTps:i.metrics?.live?.prefill_tok_s_mean??0,context:{usedTokens:0,maxTokens:i.metrics?.engine?.max_context??i.context??0,pct:0},gpuIds:/* Automatic placement names no card in the config: the cards the engine reports (or the only card) are the ones it runs on. */(i.gpus.length?i.gpus:(i.metrics?.hardware?.gpus??[]).map((g:{index:number})=>g.index).filter((n:unknown)=>typeof n==='number')).map((index:number)=>{const g=hardware?.gpus.find(g=>g.index===index);return g?.id??`${g?.vendor??'native'}:${index}`;}),memory:{context:i.metrics?.engine?.max_context??i.context??0,ramGb:0,gpuGb:0,spillGb:0,reserveGb:0,totalGb:0,devices:[]}})),memory:{ramGb:0,budgetGb:0,reserveGb:0,totalGb:0,fits:true,devices:[]}};
}
export async function nativeAction<T=unknown>(path:string,body:unknown={}){const result=await api.post<T>(`/api/native/${path}`,body);
 // A started operation is a job: show it at once instead of waiting for the next read (the owner saw download progress only after switching tabs).
 const job=result as unknown as NativeJob|undefined;
 if(state&&job&&typeof job.id==='string'&&path==='operations'&&!state.jobs.some(j=>j.id===job.id)){state={...state,jobs:[...state.jobs,job]};listeners.forEach(fn=>fn());}
 await refreshNative(true);return result;}

export {nativeVersionLabel} from './native-catalog';
