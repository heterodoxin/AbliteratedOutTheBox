export interface DownloadProgress {percent?:number;done_gb?:number;total_gb?:number;remaining_gb?:number;file?:string}
/** The native installer reports each shard, not whole-model progress. */
export function downloadCounter(size:number|undefined,progress:DownloadProgress|undefined,running:boolean){
 if(running&&progress){
  if(Number.isFinite(progress.remaining_gb))return {value:Math.max(0,progress.remaining_gb!),label:'GB left to download',key:'total'};
  if(Number.isFinite(progress.total_gb)&&Number.isFinite(progress.done_gb))return {value:Math.max(0,progress.total_gb!-progress.done_gb!),label:'GB left in this file',key:progress.file??'file'};
  if(Number.isFinite(progress.total_gb)&&Number.isFinite(progress.percent))return {value:Math.max(0,progress.total_gb!*(1-Math.min(100,Math.max(0,progress.percent!))/100)),label:'GB left in this file',key:progress.file??'file'};
 }
 return running?{value:undefined,label:progress?'Downloading…':'Preparing files…',key:'preparing'}:{value:size,label:'GB to download',key:'size'};
}
export function onboardingError(message:string){
 if(/(?:not enough.*(?:disk|storage)|disk.*(?:space|full)|ENOSPC)/i.test(message)){
  const needed=message.match(/(?:about\s+)?([\d.]+)\s*GB\s+(?:needed|required)/i)?.[1];
  const free=message.match(/([\d.]+)\s*GB\s+(?:free|available)/i)?.[1];
  return {title:'You need more storage.',message:needed?`This model needs about ${needed} GB.${free?` You have ${free} GB free.`:''}`:'There isn’t enough free space to download this model.',next:'Choose another folder or a smaller model.',storage:true};
 }
 return {title:'Something needs your attention.',message,next:undefined,storage:false};
}
export function modelFolderLabel(path:string){
 return /[\\/]Documents[\\/]Strata[\\/]Models[\\/]?$/i.test(path)?'Documents / Strata / Models':path;
}
