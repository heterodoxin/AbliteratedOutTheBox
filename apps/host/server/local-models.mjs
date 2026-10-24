/** Local GGUF references. Adding/removing a reference never copies or deletes the user's weights. */
import { accessSync, constants, mkdirSync, readdirSync, readFileSync, realpathSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { basename, dirname, extname, isAbsolute, join } from 'node:path';
import { homedir } from 'node:os';
import { readGGUF } from './gguf.mjs';
const ARCHITECTURES=new Set(['qwen2','qwen2moe','qwen3','qwen3moe','qwen3next','qwen35','qwen35moe','llama','gemma','gemma2','gemma3','phi3']);
const QUANTS={0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',7:'Q8_0',8:'Q5_0',9:'Q5_1',10:'Q2_K',11:'Q3_K_S',12:'Q3_K_M',13:'Q3_K_L',14:'Q4_K_S',15:'Q4_K_M',16:'Q5_K_S',17:'Q5_K_M',18:'Q6_K',19:'IQ2_XXS',20:'IQ2_XS',21:'Q2_K_S',22:'IQ3_XS',23:'IQ3_XXS',24:'IQ1_S',25:'IQ4_NL',26:'IQ3_S',27:'IQ3_M',28:'IQ2_S',29:'IQ2_M',30:'IQ4_XS',31:'IQ1_M',32:'BF16'};
const clean=value=>String(value??'').replace(/[\x00-\x1f\x7f]/g,' ').trim().slice(0,160);
function fullPath(input){if(typeof input!=='string'||!input.trim()||input.includes('\0'))throw new Error('Choose a model file or enter its full path.');const path=input.trim().replace(/^~(?=$|[\\/])/,homedir());if(!isAbsolute(path))throw new Error('Enter a full path.');try{return realpathSync(path);}catch{throw new Error('This file or folder is unavailable. Check its location and permissions.');}}
export function browseLocalModels(input){
 const directory=fullPath(input);if(!statSync(directory).isDirectory())throw new Error('Choose a folder to browse.');
 const entries=readdirSync(directory,{withFileTypes:true}).filter(e=>!e.name.startsWith('.')).map(e=>{try{const path=join(directory,e.name),s=statSync(path);if(s.isDirectory())return {name:e.name,path,kind:'folder'};if(s.isFile()&&extname(e.name).toLowerCase()==='.gguf')return {name:e.name,path,kind:'model',bytes:s.size};}catch{}return null;}).filter(Boolean).sort((a,b)=>(a.kind==='folder'?0:1)-(b.kind==='folder'?0:1)||a.name.localeCompare(b.name));
 return {directory,parent:dirname(directory)===directory?null:dirname(directory),entries:entries.slice(0,1000),truncated:entries.length>1000};
}
export function inspectLocalModel(input){
 let file=fullPath(input);if(extname(file).toLowerCase()!=='.gguf')throw new Error('Select a .gguf model file. Folders and adapters cannot be loaded as a chat model.');
 if(/^mmproj/i.test(basename(file)))throw new Error('This is a picture reader (mmproj), not a model. Choose the model’s own .gguf file; a model that reads pictures needs its reader in the same folder.');
 if(!statSync(file).isFile())throw new Error('Select the model’s .gguf file, rather than its folder.');
 accessSync(file,constants.R_OK);
 let selected=readGGUF(file);const split=basename(file).match(/^(.*)-(\d{5})-of-(\d{5})\.gguf$/i);
 const declared=selected.metadata['split.count']??1;
 if(declared>1||split){
  if(!split||Number(split[3])!==declared)throw new Error('Split GGUF filenames must include 00001-of-000NN and match their metadata.');
  file=join(dirname(file),`${split[1]}-00001-of-${split[3]}.gguf`);
  try{selected=readGGUF(file);}catch{throw new Error('The first shard is missing or invalid. Keep all model shards in the same folder.');}
 }
 const m=selected.metadata,arch=m['general.architecture'];
 if(!ARCHITECTURES.has(arch))throw new Error(`The declared architecture ${clean(arch)||'(missing)'} is not supported by this importer. This model type isn’t supported by the Strata engine; add it as a Custom model API (llama-server) instead.`);
 if(m['general.type']&&m['general.type']!=='model')throw new Error('This is an adapter or auxiliary file. Choose the full model GGUF.');
 if(!m['tokenizer.ggml.model']||!m['tokenizer.ggml.tokens']?.count)throw new Error('This GGUF does not include its tokenizer. Choose the full model, not an adapter. A picture reader (mmproj) goes next to its model, not in its place.');
 const integer=(key,min,max)=>{const value=m[key];if(!Number.isInteger(value)||value<min||value>max)throw new Error(`Missing or invalid ${key} metadata. This model cannot be configured safely.`);return value;};
 const context=integer(`${arch}.context_length`,4096,100_000_000),layers=integer(`${arch}.block_count`,1,4096),embedding=integer(`${arch}.embedding_length`,1,1_000_000),heads=integer(`${arch}.attention.head_count`,1,8192);
 const kvHeads=m[`${arch}.attention.head_count_kv`]??heads,keyLength=m[`${arch}.attention.key_length`]??embedding/heads,valueLength=m[`${arch}.attention.value_length`]??keyLength,experts=m[`${arch}.expert_count`]??0;
 if(![kvHeads,keyLength,valueLength].every(n=>Number.isInteger(n)&&n>=1&&n<=1_000_000)||!Number.isInteger(experts)||experts<0||experts>100_000)throw new Error('This model has invalid attention or expert metadata.');
 const count=m['split.count']??1;if(!Number.isInteger(count)||count<1||count>256)throw new Error('This split model has an unsupported number of shards.');
 const files=[],names=[];let totalTensors=0,parameters=0;
 for(let i=0;i<count;i++){
  const path=count===1?file:join(dirname(file),`${split[1]}-${String(i+1).padStart(5,'0')}-of-${split[3]}.gguf`);
  let info;try{info=i===0?selected:readGGUF(path);}catch(e){throw new Error(`${basename(path)}: ${e.message} Keep every shard in the same folder.`);}
  if(count>1&&((info.metadata['split.no']!==i)||(info.metadata['split.count']!==count)||(info.metadata['split.tensors.count']!==m['split.tensors.count'])))throw new Error('The model shards do not form a matching set. Re-download the missing or mismatched shard.');
  if(info.metadata['general.architecture']&&info.metadata['general.architecture']!==arch)throw new Error('These shards declare different architectures.');
  files.push({path:realpathSync(path),bytes:info.size,mtimeMs:info.mtimeMs,ctimeMs:info.ctimeMs});names.push(basename(path));totalTensors+=info.tensorCount;parameters+=info.parameters;
 }
 if(totalTensors<1||count>1&&totalTensors!==m['split.tensors.count'])throw new Error('This model is missing tensors or model shards.');
 const bytes=files.reduce((sum,f)=>sum+f.bytes,0),name=clean(m['general.name'])||basename(file).replace(/-00001-of-\d{5}/,'').replace(/\.gguf$/i,''),quant=QUANTS[m['general.file_type']]??'GGUF';
 const id=`local-${createHash('sha256').update(file).digest('hex').slice(0,24)}`,gb=bytes/1024**3;
 const local={path:file,files,architecture:arch,hasChatTemplate:!!m['tokenizer.chat_template'],contextLength:context,shards:count,metadataSignature:createHash('sha256').update(JSON.stringify(m)).digest('hex')};
 const variant={id,name,family:'Local models',quant,label:`${name} · ${quant}`,params:parameters>=1e9?`${(parameters/1e9).toFixed(1)}B`:`${Math.round(parameters/1e6)}M`,blurb:'Local model. Engine compatibility is checked when loading.',role:'option',tier:'local',ramClass:{minGb:gb+2,maxGb:gb+6,label:'From GGUF metadata'},diskGb:gb,downloadGb:0,contextMax:Math.min(context,1048576),layerCount:layers,experts,kvKbPerToken:layers*kvHeads*(keyLength+valueLength)/1024,hasDraft:false,kvModes:['INT8','FP16'],tags:['local'],source:{repo:'local',file:basename(file),bytes},local};
 return {variant,files:names,warnings:[...(local.hasChatTemplate?[]:['No chat template was declared. Response formatting depends on the engine.']),'Memory is estimated from metadata; engine support and actual allocation are checked when loading.']};
}
export function createLocalModels(dataDir){
 mkdirSync(dataDir,{recursive:true});const file=join(dataDir,'local-models.json');let records=[];
 try{const saved=JSON.parse(readFileSync(file,'utf8'));if(Array.isArray(saved.models))records=saved.models.filter(v=>v?.id?.startsWith('local-')&&v.local?.files?.length&&isAbsolute(v.local.path));}catch{}
 const persist=next=>{const temp=file+'.tmp';writeFileSync(temp,JSON.stringify({schema:1,models:next},null,2));renameSync(temp,file);records=next;};
 return {list:()=>structuredClone(records),add:preview=>{const v=preview.variant;const previous=records.find(row=>row.id===v.id);persist([...records.filter(row=>row.id!==v.id),{...v,importedAt:previous?.importedAt??new Date().toISOString()}]);return v;},remove:id=>persist(records.filter(v=>v.id!==id)),ready:v=>v.local.files.every(f=>{try{const s=statSync(f.path);return s.isFile()&&s.size===f.bytes&&s.mtimeMs===f.mtimeMs&&s.ctimeMs===f.ctimeMs;}catch{return false;}})};
}
