import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {utf8Script} from './powershell-utf8.mjs';
const execute=promisify(execFile);
let cached=null;
export async function systemFonts({platform=process.platform,run=execute}={}){
 if(cached&&platform===process.platform&&run===execute)return cached;
 const read=(cmd,args)=>run(cmd,args,{timeout:20000,maxBuffer:16*1024*1024,windowsHide:true});
 try{
  let names=[];
  if(platform==='darwin'){
   const {stdout}=await read('/usr/sbin/system_profiler',['SPFontsDataType','-json','-detailLevel','mini']);
   const collect=value=>{if(!value||typeof value!=='object')return;if(typeof value.family==='string')names.push(value.family);for(const nested of Object.values(value))if(typeof nested==='object')collect(nested);};collect(JSON.parse(stdout));
  }else if(platform==='win32'){
   const {stdout}=await read('powershell.exe',['-NoProfile','-NonInteractive','-Command',utf8Script('Add-Type -AssemblyName System.Drawing; (New-Object System.Drawing.Text.InstalledFontCollection).Families.Name | ConvertTo-Json -Compress')]);
   const result=JSON.parse(stdout.replace(/^\uFEFF/,''));names=Array.isArray(result)?result:[result];
  }else{
   const {stdout}=await read('fc-list',['-f','%{family}\n']);names=stdout.split(/[\n,]/);
  }
  const fonts=[...new Set(names.filter(name=>typeof name==='string').map(name=>name.trim()).filter(Boolean))].sort((a,b)=>a.localeCompare(b));
  const result={fonts,available:fonts.length>0};if(platform===process.platform&&run===execute)cached=result;return result;
 }catch{return{fonts:[],available:false};}
}
