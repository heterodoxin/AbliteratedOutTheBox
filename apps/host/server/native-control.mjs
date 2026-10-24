/** Native Strata control plane. All user actions are explicit; opening the UI only reads state. */
import {missingFilesError,missingNames} from './missing-files.mjs';
import { spawn } from 'node:child_process';
import {chooseParallel,applyParallel,withoutParallel,FALLBACK_NOTE} from './native-parallel.mjs';
import { randomBytes, randomUUID, createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, writeFileSync, mkdirSync, realpathSync, copyFileSync, renameSync, rmSync, statfsSync, unlinkSync, chmodSync, statSync, accessSync, constants } from 'node:fs';
import { join, resolve, dirname, basename, sep, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:net';
import { NATIVE_FIELDS, APP_OWNED_FIELDS, withoutDraftLayer, validateNativeSettings, validateMcpServers, publicMcpServers, publicNativeEnv, restoreNativeEnv } from './native-settings.mjs';
import { resourceSettings, saferStart, reserveChoice } from './native-resources.mjs';
import { USER_KEY, capture as captureUserArgs, merge as mergeUserArgs, setFlag as setUserFlag, stamped as stampedUserArgs } from './native-user-args.mjs';
import { resolvePython, venvPython } from './python-resolver.mjs';
import os from 'node:os';
import { runtimeVision } from './native-vision.mjs';
import { componentsDirectory, installerEnvironment, controlEnvironment } from './control-runtime.mjs';
import { createEngineStep } from './engine-step.mjs';
import { dataFolderFromPick, isDriveRoot, makeFolder, rememberDataFolder, syncDataFolder, terminalSettingsFile } from './model-folder.mjs';
import { createAutoTune, writeTuneConfig } from './native-autotune.mjs';
import { plainError, PHASE_TEXT } from './update-messages.mjs';
import { checkRunConfig, describeIssues } from './run-config-check.mjs';
import { openTextFile } from './open-text-file.mjs';
import { markEngine, engineRefusal } from './engine-version.mjs';
import { friendlyModelName } from './model-names.mjs';
import { parseSaved as parseRoles, mirror as mirrorRoles } from './gpu-roles.mjs';

const ROOT=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const safeId=value=>typeof value==='string'&&/^[a-z0-9_-]{1,80}$/i.test(value);
const busy=requests=>requests.some(r=>['queued','generating'].includes(r.state));
function readJson(path,fallback={}){try{return JSON.parse(readFileSync(path,'utf8').replace(/^\uFEFF/,''));}catch{return fallback;}}

/** The plain words for "no usable engine": the model still downloads, but loading needs the engine (owner decision 2026-10-18). */
export const ENGINE_NOT_RECOGNISED='Strata didn’t recognise an engine for this computer. The model is downloaded, but it can’t run until the right engine is installed. Open Settings → Engine.';
export function createNativeControl({directory,requests=()=>[],changed=()=>{},spawnProcess=spawn,fetcher=fetch,modelsDirectory,nativeDataDirectory=value=>value,beforeLoad=async()=>{},hardwareProbe=()=>[],control=null,ready=null,appUpdater=null,defaultRepo=process.env.STRATA_NATIVE_REPO??resolve(ROOT,'../../Strata')}={}){
 mkdirSync(directory,{recursive:true});
 const settingsPath=join(directory,'native-host.json');
 let preferences=readJson(settingsPath,{repo:defaultRepo});
 let child=null,seq=0,buffer='',pending=new Map(),definitions=[],refreshing=null,lastRead=0,connecting=null;
 let snapshot={available:false,repo:preferences.repo,summary:'Connect a local Strata installation.',catalog:null,installed:null,hardware:null,error:null};
 const instances=new Map(),processes=new Map(),jobs=new Map(),logs=[];
 const savedJobs=readJson(join(directory,'native-jobs.json'),[]);for(const j of (Array.isArray(savedJobs)?savedJobs:[]).slice(-12))jobs.set(j.id,{...j,...(j.state==='running'?{state:'error',error:'The host restarted before completion. Check diagnostics, then retry; completed files are kept.',summary:'Interrupted operation'}:{})});
 /* Packaged mode: the app ships Strata Control (control-runtime.mjs). Models are run configs in an install folder, the engine
    comes from the component folder, and a start is `strata runtime start` under the owner runtime. Nothing is installed,
    updated or fetched from here. Without `control` (a source checkout) every path below is the original one. */
 const packaged=!!control;
 /* The auto-tuner and the saved tunings are Strata Control's (`strata tune auto`, `strata presets`): the app runs them and follows their status file. */
 const autotuner=createAutoTune({spawnProcess,workDir:join(directory,'native-tune'),launcher:()=>{if(packaged)return {command:control.control.python,prefix:[control.control.cli],cwd:control.control.dir,env:controlEnvironment(control.control)};const py=python();return {command:py.command,prefix:[...py.args,join(repo(),'strata.py')],cwd:repo(),env:{...process.env,PYTHONUNBUFFERED:'1',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8'}};}});
 /* In the packaged app the model catalog, the GPU probe and the install job are Control's bundled tools/strata_mcp.py and
    setup.py, run on Control's own Python in the installer's mode (STRATA_CONTROL_INSTALL: no pip, no engine, no compiler).
    The install only runs when the user clicks Download; starting and serving (control.start) keep STRATA_NO_ACQUIRE. */
 const defaultInstall=()=>process.env.STRATA_INSTALL_DIR?resolve(process.env.STRATA_INSTALL_DIR):join(dirname(componentsDirectory()),'install');
 let packagedInstall=null;
 const NO_INSTALL='Installing, updating, tuning with calibration or rolling back model files and engines happens from the Terminal (strata install), never while the app runs.';
 let defaultId=null,timer=null,closing=false,polling=false;
 const emit=()=>{const value=state();writeFileSync(join(directory,'native-jobs.json'),JSON.stringify(value.jobs),{mode:0o600});changed(value);};
 const append=(instanceId,line)=>{logs.push({at:new Date().toISOString(),instanceId,line:String(line).slice(0,2000)});if(logs.length>400)logs.shift();};
 function diskSpace(path){try{const s=statfsSync(path);return {free:Number(s.bavail)*Number(s.bsize),total:Number(s.blocks)*Number(s.bsize)};}catch{return null;}}
 /* Does the engine in this release serve DFlash2? Control knows (strata_control/drafting.py); asked once, false until it answers. The Models screen offers the choice only when true. */
 let dflash2Served=null;
 function probeDrafting(){if(!packaged||dflash2Served!==null||typeof control.draftingPlan!=='function')return;dflash2Served=false;control.draftingPlan({mode:'auto'}).then(p=>{dflash2Served=p.serves===true;emit();}).catch(()=>{});}
 function state(){const folder=preferences.downloadDirectory??(modelsDirectory?modelsDirectory():snapshot.installed?.data_folder),space=folder?diskSpace(folder):null;return {...snapshot,downloadDirectory:folder,downloadFreeBytes:space?.free??null,downloadTotalBytes:space?.total??null,fields:nativeFields(),instances:[...instances.values()].map(({key,configPath,fingerprint,mcpServers,...i})=>structuredClone(i)),defaultId,jobs:[...jobs.values()].slice(-12).map(({process,abort,engineAbort,rawResult,...j})=>j),capabilities:{responses:sourceText().includes('/v1/responses'),updates:!!snapshot.available,rollback:previousEngine(),componentUpdates:packaged&&typeof control.update==='function',calibration:packaged?false:snapshot.hardware?.gpus?.some(g=>/nvidia/i.test(g.name??''))??false,dflash2:dflash2Served===true}};}
 let sourceCache={path:'',time:0,text:''};
 function sourceText(){try{const path=join(packaged?control.control.dir:preferences.repo,'serve/server.py'),time=statSync(path).mtimeMs;if(sourceCache.path===path&&sourceCache.time===time)return sourceCache.text;sourceCache={path,time,text:readFileSync(path,'utf8')};return sourceCache.text;}catch{return '';}}
 function nativeFields(){const source=sourceText();return NATIVE_FIELDS.map(field=>({...field,supported:APP_OWNED_FIELDS.includes(field.id)||field.id.startsWith('vision_')||field.id.startsWith('mcp_')||field.id==='vram_reserve_mib'||source.includes(field.id)}));}
 const modelArena=id=>({arenaGb:(snapshot.catalog?.families??[]).flatMap(f=>f.sizes??[]).find(s=>s.id===id)?.experts_gb});
 /* A start that ends with `server_exited` carries the engine's last log line with it; the app turns the code into words and
    keeps this text under "Details" (src/lib/plain-errors.ts). */
 function exitCodeText(code){const n=Number(code);if(!Number.isFinite(n)||n===0)return '';const u=n<0?n+2**32:n;return u>=0x80000000?`exit code 0x${u.toString(16).toUpperCase().padStart(8,'0')}`:`exit code ${u}`;}
 function humanStartError(error){const text=String(error?.message??error),tail=error?.output?.log_tail,code=exitCodeText(error?.output?.engine_exit_code);const head=code&&!text.includes(code)?`${text} (${code})`:text;return tail&&!head.includes(tail)?`${head}: ${tail}`:head;}
 /* A run config the app's install job wrote names files inside the app's own Control folder (`cwd`, `--expert-profile`). When the
    app is reinstalled to another folder, or an update moves Control, those absolute paths point at nothing and every installed
    model breaks. At every read of the installed models, a path that ends up inside a folder called `control` that no longer
    exists is pointed at the Control this app has now, when the file is there. The config is saved (with a .bak) only then. */
 function repairControlPaths(dir){if(!packaged||!dir||!existsSync(dir))return;let names=[];try{names=readdirSync(dir).filter(n=>/^strata-[a-z0-9_-]+\.json$/i.test(n));}catch{return;}
  const here=control.control.dir,inside=/^(.+?[\/]control)(?=[\/]|$)(.*)$/i;
  const fix=value=>{if(typeof value!=='string')return value;const m=value.match(inside);if(!m||existsSync(value)||resolve(m[1]).toLowerCase()===resolve(here).toLowerCase())return value;const candidate=join(here,m[2]);return existsSync(candidate)?candidate:value;};
  for(const name of names){const path=join(dir,name),config=readJson(path,null);if(!config||typeof config!=='object')continue;
   const next={...config,...(config.cwd!==undefined?{cwd:fix(config.cwd)}:{}),...(Array.isArray(config.args)?{args:config.args.map(fix)}:{})};
   if(JSON.stringify(next)!==JSON.stringify(config)){try{writeModelConfig(path,next);}catch{/* read-only folder: the load reports it */}}}}
 function installedInfo(info){if(!info)return info;return {...info,models:(info.models??[]).map(entry=>{const cfg=/^strata-[a-z0-9_-]+\.json$/i.test(entry.config??'')?readJson(entry.configPath??join(repo(),entry.config)):{};const at=cfg.args?.indexOf('--native')??-1,nativeFile=at>=0?cfg.args[at+1]:null;const dataFolder=typeof nativeFile==='string'&&basename(dirname(dirname(nativeFile)))==='models'?dirname(dirname(dirname(nativeFile))):info.data_folder;/* A model the person pointed at their own GGUF folder (not <data>/models/<tag>): remembered, so a later load runs setup with that folder again instead of downloading the files (#3199). */const ownGgufDir=typeof nativeFile==='string'&&basename(dirname(dirname(nativeFile)))!=='models'&&existsSync(nativeFile)?dirname(nativeFile):undefined;return {...entry,ownGgufDir,diskGb:entry.diskGb??entry.disk_gb,diskBytes:entry.diskBytes??entry.disk_bytes,sharedBytes:entry.sharedBytes??entry.shared_bytes,removableGb:entry.removableGb??entry.removable_gb,dataFolder,visionPrepared:!!cfg.vision,/* The picture files are prepared when the run config has a vision section: the Models screen then loads straight away instead of running setup again (it used to read 'off' for every model, which sent every load of a picture-capable model through "Apply settings and load", and setup writes the arguments, the draft layer and the reader back each time). */images:cfg.vision?'yes':'off',draftLayer:cfg.mtp!==false,drafting:cfg.drafting??'auto',/* Picture reading is ON unless the person turned it off: no saved choice and not loaded is undefined, not false (owner rule: pictures on by default). */visionEnabled:cfg.ui_vision_enabled??(()=>{const running=[...instances.values()].find(i=>i.model===entry.model&&i.status==='ready');return running?!!running.vision:undefined;})(),visionDevice:cfg.ui_vision_device,resourceProfile:cfg.ui_resource_profile,lowRamMode:cfg.args?.includes('--resident-experts')?'resident':cfg.args?.includes('--mmap-experts')?'mmap':'auto'};})};}
 function installRoot(){const found=[preferences.installDir].find(d=>d&&existsSync(d));if(found)return realpathSync(found);mkdirSync(defaultInstall(),{recursive:true});return realpathSync(defaultInstall());}
 /* The settings file Strata Control and the installer read (control-runtime.mjs sets STRATA_SETTINGS_FILE to it): the app's data folder is written there. */
 function controlSettingsFile(){return process.env.STRATA_SETTINGS_FILE||join(dirname(componentsDirectory()),'app-settings.json');}
 /* One models folder for the app and the Terminal: the one changed last wins and the other settings file is updated (nothing is moved). */
 function shareDataFolder(){if(!packaged)return;try{let at=0;if(preferences.downloadDirectory)try{at=statSync(settingsPath).mtimeMs/1000;}catch{/* no file yet */}const found=syncDataFolder({appFile:controlSettingsFile(),terminalFile:terminalSettingsFile(),app:preferences.downloadDirectory?{folder:preferences.downloadDirectory,at}:null});if(found.from==='terminal'&&found.folder&&found.folder!==preferences.downloadDirectory){preferences={...preferences,downloadDirectory:found.folder};writeFileSync(settingsPath,JSON.stringify(preferences,null,2),{mode:0o600});}}catch{/* the shared setting is a convenience: never stop the app */}}
 function repo(){if(packaged)return installRoot();const path=realpathSync(preferences.repo);if(!existsSync(join(path,'setup.py'))||!existsSync(join(path,'tools/strata_mcp.py'))||!existsSync(join(path,'serve/server.py')))throw new Error('Choose the Strata repository folder containing setup.py and tools/strata_mcp.py.');return path;}
 function python(){return resolvePython({repo:preferences.repo});}
 function send(method,params,timeout=30000){return new Promise((resolve,reject)=>{const id=++seq;const alarm=setTimeout(()=>{pending.delete(id);reject(new Error('The native engine did not respond in time. Check engine logs.'));},timeout);pending.set(id,{resolve,reject,alarm});child.stdin.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');});}
 function disconnect(){child?.kill();child=null;definitions=[];for(const p of pending.values()){clearTimeout(p.alarm);p.reject(new Error('Native control connection closed.'));}pending.clear();}
 async function connect(){if(closing)throw new Error("The host is shutting down.");if(child&&definitions.length)return;if(!connecting)connecting=startClient().finally(()=>{connecting=null;});return connecting;}
 async function startClient(){
  const folder=repo();
  buffer='';const py0=packaged?{command:control.control.python,args:[]}:python();
  const script=packaged?join(control.control.dir,'tools/strata_mcp.py'):join(folder,'tools/strata_mcp.py');
  const client=child=spawnProcess(py0.command,[...py0.args,script,...(packaged?['--root',folder,'--src',control.control.dir]:[])],{cwd:packaged?control.control.dir:folder,windowsHide:true,stdio:['pipe','pipe','pipe'],env:packaged?installerEnvironment(control.control,{installDir:folder}):{...process.env,PYTHONUNBUFFERED:'1',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8'}});
  client.stdout.on('data',chunk=>{if(child!==client)return;buffer+=chunk.toString();let end;while((end=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,end);buffer=buffer.slice(end+1);let message;try{message=JSON.parse(line);}catch{continue;}const p=pending.get(message.id);if(p){clearTimeout(p.alarm);pending.delete(message.id);message.error?p.reject(new Error(message.error.message)):p.resolve(message.result);}}});
  child.stderr.on('data',chunk=>append('control',chunk.toString()));
  client.on('error',error=>{if(child===client){snapshot.error=error.message;disconnect();}});
  client.on('exit',()=>{if(child===client){child=null;for(const p of pending.values()){clearTimeout(p.alarm);p.reject(new Error('Strata control stopped. Python 3.10 or newer is required.'));}pending.clear();}});
  await send('initialize',{protocolVersion:'2024-11-05',capabilities:{},clientInfo:{name:'Strata UI',version:'0.1.0'}});
  child.stdin.write(JSON.stringify({jsonrpc:'2.0',method:'notifications/initialized'})+'\n');
  definitions=(await send('tools/list',{})).tools??[];
 }
 async function controlTool(name){
  if(name==='strata_status'){
   for(const dir of [preferences.installDir,packagedInstall,installRoot()])repairControlPaths(dir);
   // Always the app's own install folder (or the one the person chose in Settings): Control is told explicitly, so it never falls
   // back to an older command-line Strata's folder remembered in %APPDATA% (tester report 2026-10-17: logs landed in T:\Strata).
   const list=await control.installedModels({installDir:installRoot()});
   packagedInstall=list.install_dir;
   const engine=await control.engine();
   const installer=await mcpCall('strata_status',{hardware:true}).catch(error=>({error:error.message}));
   const disk=Object.fromEntries((installer.installed?.models??[]).map(m=>[m.model,{diskGb:m.disk_gb,diskBytes:m.disk_bytes,sharedBytes:m.shared_bytes,removableGb:m.removable_gb,partsMissing:m.parts_missing??0,missingFiles:m.missing_files}]));
   const models=(list.models??[]).filter(m=>m.component_status!=='invalid'&&m.source_config).map(m=>({...(disk[m.id]??{}),model:m.id,config:basename(m.source_config),configPath:m.source_config,model_name:m.name,context:m.context_tokens??0,backend:m.backend,ready:m.component_status==='present'&&!(disk[m.id]?.partsMissing),missing:missingNames(m.missing,disk[m.id]?.missingFiles),images:'off',port:undefined,last_used:null}));
   const cores=os.cpus().length;
   return {summary:engine.usable?`Strata Control is ready; engine ${engine.engine.version} (${engine.engine.platform}-${engine.engine.backend}).`:(engine.notice??'No engine for this PC was found.'),
    installDir:list.install_dir,engine:{usable:engine.usable,notice:engine.notice,wrongMake:engine.wrongMake===true,version:engine.engine?.version??null,backend:engine.backend,componentsDir:engine.components_dir,previous:engine.previous??null,pinned:engine.pinned??null,updateNotice:engine.update_notice??null},
    installed:{strata_folder:list.install_dir,data_folder:null,engine:engine.engine?{version:engine.engine.version,backend:engine.engine.backend,folder:engine.engine.root}:{folder:null,version:null},models,usage:installer.installed?.usage??null},
    hardware:installer.hardware?{...installer.hardware,cpu:{...(installer.hardware.cpu??{}),cores:installer.hardware.cpu?.cores??cores},gpus:installer.hardware.gpus??[]}:{cpu:{cores},ram_gb:Math.round(os.totalmem()/1024**3),gpus:[]},
    recommendation:installer.recommendation??null,installerError:installer.error??null,install:installer.install??null,running:null};
  }
  throw new Error(`Strata Control has no ${name}.`);
 }
 async function tool(name,args={}){if(packaged&&name==='strata_status')return controlTool(name,args);return mcpCall(name,args);}
 async function mcpCall(name,args={}){await connect();if(!definitions.some(d=>d.name===name))throw new Error(`This Strata installation does not support ${name}. Update the installation first.`);const result=await send('tools/call',{name,arguments:args},120000);let value=result.structuredContent;if(!value){const text=result.content?.find(c=>c.type==='text')?.text;try{value=JSON.parse(text);}catch{value={summary:text};}}if(result.isError)throw new Error(value?.error??value?.summary??'Native operation failed.');return value;}
 async function refresh(force=false){
  if(ready)await ready.catch(()=>{});
  if(closing)return state();
  /* A read that is not forced never makes the window wait for the (slow) status of Control: it gets the last state at once, and a stale one
     is refreshed in the background. Only the first read, and a forced one after an operation, waits. */
  if(refreshing)return !force&&snapshot.available?state():refreshing;if(!force&&Date.now()-lastRead<45000)return state();
  probeDrafting();
  refreshing=(async()=>{try{const [catalog,status]=packaged?await (async()=>{const s=await tool('strata_status',{hardware:true});const c=await tool('strata_models').catch(error=>({families:[],contexts:[],recommendation:{family:null,model:null},context_note:'',error:error.message}));return [c,s];})():await Promise.all([tool('strata_models'),tool('strata_status',{hardware:true})]);let visionReader=false;if(packaged){try{const e=await control.engine();const root=e?.engine?.root??(e?.engine?.exe?dirname(e.engine.exe):'');visionReader=!!root&&['strata-vision','strata-vision.exe'].some(n=>existsSync(join(root,n)));}catch{visionReader=false;}}const markImages=c=>c&&Array.isArray(c.families)?{...c,families:c.families.map(f=>({...f,imagesCapable:!!f.images,images:!!f.images&&visionReader,sizes:(f.sizes??[]).map(s=>({...s,imagesCapable:!!s.images,images:!!s.images&&visionReader}))}))}:c;snapshot={available:true,repo:packaged?status.installDir:repo(),version:packaged?(control.version??null):await execute('git',['describe','--tags','--always'],repo()).then(s=>s.trim()).catch(()=>null),summary:status.summary,catalog:markEngine(packaged?markImages(catalog):catalog,status.engine),engineInfo:{version:status.engine?.version??null,source:status.engine?.source??null},visionReader:packaged?visionReader:true,installed:installedInfo(status.installed),hardware:status.hardware,error:null,install:status.install,external:status.running??null,...(packaged?{packaged:true,installDir:status.installDir,engine:status.engine,installerError:status.installerError}:{})};lastRead=Date.now();if(status.install?.running&&![...jobs.values()].some(j=>j.state==='running'))begin('install',j=>watchInstall(j,status.install));}catch(error){snapshot={...snapshot,available:false,error:error.message};lastRead=Date.now();}emit();return state();})().finally(()=>{refreshing=null;});const run=refreshing;if(!force&&snapshot.available){run.catch(()=>{});return state();}return run;
 }
 function configure({repository}){if(packaged){if(instances.size||[...jobs.values()].some(j=>j.state==='running'))throw new Error('Unload models and finish operations before changing the installation.');const next=realpathSync(String(repository));if(!statSync(next).isDirectory())throw new Error('Choose the folder that holds your Strata model configurations.');preferences={...preferences,installDir:next};packagedInstall=null;disconnect();writeFileSync(settingsPath,JSON.stringify(preferences,null,2),{mode:0o600});lastRead=0;return refresh(true);}if(instances.size||[...jobs.values()].some(j=>j.state==='running'))throw new Error('Unload models and finish operations before changing the installation.');const next=realpathSync(String(repository));const old=preferences;preferences={...preferences,repo:next};try{repo();}catch(e){preferences=old;throw e;}disconnect();writeFileSync(settingsPath,JSON.stringify(preferences,null,2),{mode:0o600});lastRead=0;return refresh(true);}
 function setDownloadDirectory(input,{insideJob=false}={}){if(!insideJob&&[...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish the current operation before changing the folder.');if(typeof input!=='string'||!input.trim()||/[\r\n\0]/.test(input))throw new Error('Choose a model folder.');let path;try{path=realpathSync(input.trim());}catch{throw new Error('That folder does not exist. Check the path, or that the drive is connected.');}if(!statSync(path).isDirectory())throw new Error('Choose a folder, not a file.');/* A whole drive (D:\) is not filled with models: D:\Strata is made and used. */if(isDriveRoot(path)){const own=dataFolderFromPick(path).path;makeFolder(own);path=realpathSync(own);}try{accessSync(path,constants.R_OK|constants.W_OK);}catch{throw new Error('Strata cannot write in that folder. Choose a folder you can save files in.');}preferences={...preferences,downloadDirectory:path};/* the same choice for the installer and the Terminal: the Control settings file */if(packaged)rememberDataFolder(controlSettingsFile(),path,Date.now()/1000,true);writeFileSync(settingsPath,JSON.stringify(preferences,null,2),{mode:0o600});shareDataFolder();emit();return state();}
 function recordFile(model){if(!safeId(model))throw new Error('Choose an installed model.');const entry=snapshot.installed?.models?.find(i=>i.model===model);if(!entry)throw new Error('This model is not installed. Download and prepare it first.');const path=realpathSync(entry.configPath??join(repo(),entry.config));if(dirname(path)!==repo()||!/^strata-[a-z0-9_-]+\.json$/i.test(basename(path)))throw new Error('The model configuration is outside the Strata installation.');return {entry,path};}
 function record(model){const {entry,path}=recordFile(model);return {entry,path,config:readJson(path)};}
 /* The text file a power user edits by hand: where it is and what is wrong in it (run-config-check.mjs). */
 function settingsFile(model){const {path}=recordFile(model);return {model,path,name:basename(path),...checkRunConfig(readFileSync(path,'utf8')),note:'Changes you make in the file apply the next time the model starts.'};}
 async function openSettingsFile(model,options={}){const info=settingsFile(model);await openTextFile(info.path,options);return info;}
 /* A mistake in the file holds the start back with the line, instead of an engine that dies on it. Warnings (an unknown key) never do. */
 function assertSettingsFile(path){const check=checkRunConfig(readFileSync(path,'utf8'));if(!check.ok)throw new Error(`The settings file for this model has a mistake, so the model was not started. ${describeIssues(check.issues,'error').join(' ')}`);}
 function settings(model){const {config}=record(model);const values=Object.fromEntries(NATIVE_FIELDS.map(f=>[f.id,config[f.id]??null]));if(typeof values.aliases==='string')values.aliases=values.aliases.split(',').map(v=>v.trim()).filter(Boolean);values.sampling=config.sampling??{};values.args=config.args??[];values.env=publicNativeEnv(config.env??{});return {model,values,mcp:publicMcpServers(config.mcp_servers??config.mcpServers??{}),restartNeeded:true};}
 function save(model,{values,mcp}){if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish the current operation before saving native settings.');const {path,config}=record(model);if(busy(requests().filter(r=>instances.get(r.instanceId)?.model===model)))throw new Error('Finish this model’s requests before saving its settings.');const patch=validateNativeSettings(values??{}),next={...config};for(const field of nativeFields())if(field.supported===false&&patch[field.id]!=null&&JSON.stringify(patch[field.id])!==JSON.stringify(config[field.id]))throw new Error(`${field.label} needs a newer native Strata installation. Update it in Settings → About.`);for(const [key,value] of Object.entries(patch)){if(key==='env')next.env=restoreNativeEnv(value,config.env??{});else if(value===null)delete next[key];else next[key]=value;}if('vram_reserve_mib' in patch){const base=captureUserArgs(next);const mine=setUserFlag(base,'--vram-reserve-mib',patch.vram_reserve_mib);if(mine.length)next[USER_KEY]=mine;else delete next[USER_KEY];next.args=mine.length&&patch.vram_reserve_mib!==null?mergeUserArgs(next.args??[],mine):setUserFlag(next.args??[],'--vram-reserve-mib',null);delete next.ui_reserve_applied;if(Array.isArray(next.args_written))next.args_written=[...next.args];}if(mcp!==undefined){next.mcp_servers=validateMcpServers(mcp,config.mcp_servers??config.mcpServers??{});delete next.mcpServers;}copyFileSync(path,path+'.bak');chmodSync(path+'.bak',0o600);const temp=path+'.ui-tmp';writeFileSync(temp,JSON.stringify(next,null,2)+'\n',{mode:0o600});renameSync(temp,path);return settings(model);}
 function previousEngine(){if(packaged)return !!snapshot.engine?.previous;try{const data=snapshot.installed?.data_dir??snapshot.installed?.data_folder;return [join(preferences.repo,'engine/.previous'),data&&join(data,'engine/.previous')].filter(Boolean).some(existsSync)&&readFileSync(join(preferences.repo,'setup.py'),'utf8').includes('--rollback-engine');}catch{return false;}}
 async function plan(args){await refresh();if(!snapshot.available)throw new Error(snapshot.error);const result=await tool('strata_install',installArgs(args,false));
  /* A chosen drafter that is not on this computer yet is part of the download: its size is listed beside the model's (Control decides; nothing is fetched here). */
  const choice=args.drafting;if(packaged&&typeof choice==='string'&&choice!=='auto'&&typeof control.draftingPlan==='function'){try{const d=await control.draftingPlan({mode:choice,family:args.family??null,dataDir:snapshot.installed?.data_folder??null});if(d.needed)return {...result,drafting:{needed:true,gb:d.gb,line:d.line}};}catch{/* the model plan still stands */}}
  return result;}
 /* The folder the person picked in the first-run chooser can be the one that holds their GGUF files (or a folder inside it). It is theirs, not a Strata data folder: packs are never made in it, and a Strata folder around it (<Strata>/models/<tag>) is used instead (#3507, #3424). */
 function ownFilesDataFolder(candidate,args){if(!candidate||args.data_dir!==undefined||!args.gguf_dir)return candidate;const same=(a,b)=>{const x=resolve(a),y=resolve(b);return process.platform==='win32'?x.toLowerCase()===y.toLowerCase():x===y;};const inside=(a,b)=>{const rel=relative(resolve(b),resolve(a));return rel===''||(!!rel&&!rel.startsWith('..')&&!isAbsolute(rel));};if(!same(candidate,args.gguf_dir)&&!inside(candidate,args.gguf_dir))return candidate;const models=dirname(resolve(args.gguf_dir)),top=dirname(models);return basename(models).toLowerCase()==='models'&&/strata/i.test(basename(top))?top:undefined;}
 function installArgs(args,confirm){const allowed=['family','model','context','vision','kv','gpu','gpus','backend','low_ram','port','gguf_dir'];const size=snapshot.catalog?.families?.find(f=>f.family===args.family)?.sizes.find(s=>s.model===args.model);const existing=snapshot.installed?.models?.find(m=>m.model===size?.id);/* A model adopted from the person's own GGUF folder reports that folder's parent as its data folder: it is theirs, not Strata's, so a second preparation never uses it for the packs. */if(args.gguf_dir===undefined&&existing?.ownGgufDir)args={...args,gguf_dir:existing.ownGgufDir};const data=nativeDataDirectory(ownFilesDataFolder(args.data_dir??(args.gguf_dir?undefined:existing?.dataFolder)??preferences.downloadDirectory??modelsDirectory?.(),args));if(data&&packaged&&!args.data_dir)rememberDataFolder(controlSettingsFile(),data);return {...(data?{data_dir:data}:{}),...Object.fromEntries(allowed.filter(k=>args[k]!==undefined).map(k=>[k,args[k]])),/* One card is "gpu", never "gpus" (the installer wants two or more there: "Gpus='0' is not valid", #3203). */...(/^\d{1,2}$/.test(String(args.gpus??''))?{gpu:Number(args.gpus),gpus:undefined}:{}),/* Picture reading is only downloaded when it was asked for (Read pictures); the default stays the smaller download. */...(packaged&&args.vision===undefined?{vision:'no'}:{}),confirm};}

 function begin(kind,run,metadata={}){if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish or cancel the current engine operation first.');const job={id:randomUUID(),kind,state:'running',startedAt:new Date().toISOString(),summary:'Starting…',result:null,error:null,...metadata};for(const [id,old] of jobs)if(jobs.size>=12&&old.state!=='running')jobs.delete(id);jobs.set(job.id,job);emit();void run(job).then(result=>{if(job.state==='running'){Object.assign(job,{state:'done',summary:result?.summary??'Complete',result,finishedAt:new Date().toISOString()});}}).catch(error=>{if(job.state==='running')Object.assign(job,{state:'error',error:error.message,summary:error.message,...(error.details?{details:error.details}:{}),finishedAt:new Date().toISOString()});}).finally(()=>{if(job.kind.startsWith('update')||job.kind==='rollback')disconnect();lastRead=0;emit();void refresh(true);});return state().jobs.find(j=>j.id===job.id);}
 async function watchInstall(job,current){job.summary=current.step??'Resuming preparation status…';job.result={install:current};while(job.state==='running'){await new Promise(r=>setTimeout(r,1500));const result=await tool('strata_install');job.summary=result.summary;job.result=result;emit();if(!result.install?.running){if(result.install?.exit_code!==0)throw new Error(result.install?.error??'Preparation did not finish. Retry to reuse completed files.');return {summary:'Model preparation complete. Load it from Models.'};}}}
 function externalRunning(){return Array.isArray(snapshot.external)?snapshot.external.filter(s=>![...instances.values()].some(i=>i.port===s.port)):[];}
 async function install(job,args){
  if(ready)await ready.catch(()=>{});
  if(job.state!=='running')return;
  if(externalRunning().length)throw new Error('Stop the native server running outside this UI before preparing files.');
  if(busy(requests().filter(r=>instances.has(r.instanceId))))throw new Error('Finish or cancel requests before preparing model files.');
  for(const id of [...instances.keys()])await unload(id);
  if(job.state!=='running')return;
  let result=await tool('strata_install',installArgs(args,true));
  /* The engine comes with the model: the same job makes sure this PC has a usable engine (none installed, or a newer compatible
     one in the signed feed) through Control's component updater, and shows it in the job's own progress (install.engine, install.parts,
     one combined download total). A failed engine download keeps the model files and ends with a retry message. */
  job.engineAbort=new AbortController();
  const engineStep=packaged&&typeof control.update==='function'?createEngineStep({control,signal:job.engineAbort.signal,onChange:()=>{if(job.rawResult)publish(job.rawResult);else emit();}}):null;
  const publish=r=>{job.rawResult=r;job.summary=r.summary;job.result=engineStep&&r.install?{...r,install:engineStep.merge(r.install)}:r;emit();};
  publish(result);
  const engineTask=engineStep?engineStep.run():null;
  try{
   for(;;){
    if(job.state!=='running')return;
    await new Promise(resolve=>setTimeout(resolve,1500));
    result=await tool('strata_install');
    publish(result);
    if(result.install&&!result.install.running){
     if(result.install.exit_code!==0)throw new Error(result.install.error??result.summary);
     if(engineTask){job.summary='Model files are ready. Finishing the engine for this PC…';emit();await engineTask;publish(result);}
     await refresh(true);
     const note=engineStep?.summaryNote();
     const finalResult=engineStep?job.result:result;
     if(note)return {...finalResult,summary:note};
     if(args.loadAfter){const model=(result.install.family??args.family??'qwen')==='qwen'?String(args.model).toLowerCase():`${args.family}-${String(args.model).toLowerCase()}`;return loadEngine(job,{model,multiple:!!args.multiple,resourceProfile:args.resourceProfile,vision:args.loadVision,...(typeof args.drafting==='string'?{values:{drafting:args.drafting}}:{})});}
     return finalResult;
    }
   }
  }catch(error){job.engineAbort?.abort();throw error;}
 }
 function terminate(proc,signal='SIGTERM'){if(!proc)return;if(process.platform!=='win32'&&proc.pid){try{process.kill(-proc.pid,signal);return;}catch{}}proc.kill(signal);}
 async function stopProcess(proc){if(!proc||proc.exitCode!=null)return;await new Promise(resolve=>{const timeout=setTimeout(()=>{terminate(proc,'SIGKILL');resolve();},8000);proc.once('exit',()=>{clearTimeout(timeout);resolve();});terminate(proc);});}
 async function unload(id,force=false){const instance=instances.get(id);if(!instance)throw new Error('This model is no longer loaded.');if(!force&&[...jobs.values()].some(j=>j.kind==='benchmark'&&j.state==='running'&&j.instanceId===id))throw new Error('Stop the benchmark before unloading this model.');if(!force&&busy(requests().filter(r=>r.instanceId===id)))throw new Error('Finish or cancel this model’s requests first.');instance.status='unloading';if(instance.profile){try{await control.stop(instance.profile);}catch(error){if(!force){instance.status='error';instance.error=error.message;emit();throw error;}}}else await stopProcess(processes.get(id));processes.delete(id);instances.delete(id);try{unlinkSync(instance.configPath);}catch{}if(defaultId===id)defaultId=instances.keys().next().value??null;emit();return state();}
 /* Delete a downloaded model: stop it first, then the installer removes the model's own files and its run config and keeps
    whatever another installed model still uses (shared MTP draft, shared PLE shard). */
 async function removeModel(job,args){if(!safeId(args.model))throw new Error('Choose an installed model.');
  if(externalRunning().length)throw new Error('Stop the native server running outside this UI before deleting a model.');
  for(const instance of [...instances.values()].filter(i=>i.model===args.model)){if(busy(requests().filter(r=>r.instanceId===instance.id)))throw new Error('Finish or cancel requests before deleting this model.');await unload(instance.id);}
  job.summary='Deleting downloaded files…';emit();
  const result=await tool('strata_remove',{model:args.model,confirm:true});
  if(result.failed?.length)throw new Error(`Some files could not be removed: ${result.failed.slice(0,3).join('; ')}`);
  await refresh(true);
  /* Verify it is really gone: the model must no longer be listed as installed, and the files the plan named must not exist. */
  const still=(result.plan?.remove??[]).map(i=>i.path).filter(p=>{try{return existsSync(p);}catch{return false;}});
  if(still.length)throw new Error(`Some files could not be removed: ${still.slice(0,3).map(p=>basename(p)).join('; ')}`);
return {summary:result.summary??'Deleted',removed:result.plan?.gb??null};}
 /* The configuration a load starts the engine with (before its port and key): the saved model config, the user's choices and the resource profile. The tuner measures exactly this, so a saved tuning belongs to the same choices a load makes. */
 function buildConfig(args,config){
  const vision=runtimeVision(config,args.vision);
  const visionPreferences=args.vision!==undefined?{ui_vision_enabled:args.vision!=='no',ui_vision_device:args.vision==='no'?config.ui_vision_device??'yes':args.vision}:{};
  /* The engine options the person set by hand (user_args, and what they wrote into "args" since the last write) go over everything the app generates, and an
     explicit GPU-memory reserve (the Expert setting, user_args, a value they wrote) is never replaced by the safety default (#3215 #3270 #3216). */
  const values0=args.values??{},cleared='vram_reserve_mib' in values0&&(values0.vram_reserve_mib===null||values0.vram_reserve_mib===''),base0=cleared?{...config,vram_reserve_mib:undefined,[USER_KEY]:setUserFlag(config[USER_KEY],'--vram-reserve-mib',null)}:config;
  let mine=captureUserArgs(base0);const asked=cleared?null:reserveChoice({...base0,[USER_KEY]:mine},snapshot.hardware,values0).asked;
  if(asked!==null)mine=setUserFlag(mine,'--vram-reserve-mib',asked);
  const resources=args.resourceProfile?resourceSettings({...base0,[USER_KEY]:mine,vision},snapshot.hardware,args.resourceProfile,modelArena(args.model),values0):null;
  const cfg={...config,...visionPreferences,...(resources?{...resources,ui_resource_profile:args.resourceProfile}:{}),host:'127.0.0.1',open_browser:false,...validateNativeSettings(args.values??{}),vision};
  /* The app's suggested cards only fill a gap: a model whose settings file already names its GPUs ("gpu": [1, 0], with "gpu_order": "as_given") keeps them (#3256). A card the person picked in the screen always wins. */
  const namedInFile=(Array.isArray(config.gpu)?config.gpu.length>0:config.gpu!==undefined&&config.gpu!==null&&config.gpu!=='')||!!config.gpu_roles;
  /* Roles chosen in the screen (Main / Helper / Off per card, gpu-roles.mjs): the card list is the roles' mirror - Helpers first, the Main card LAST, Off cards left out - with "gpu_roles" beside it. Every card Off is allowed: the server says plainly what the engine can do. */
  if(args.gpuRoles!==undefined){const make=snapshot.hardware?.engine_vendor,cards=(snapshot.hardware?.gpus??[]).filter(g=>!make||!g.vendor||g.vendor===make),named=parseRoles(args.gpuRoles);if(!named)throw new Error('Choose Main, Helper or Off for each graphics card, with one Main.');const known=new Set(cards.map(g=>g.index));if(Object.keys(named).some(k=>!known.has(Number(k))))throw new Error('Choose detected graphics cards.');const roles=Object.fromEntries(cards.map(g=>[g.index,named[g.index]??'off'])),m=mirrorRoles(roles,cards);for(const k of ['gpu','gpu_order','gpu_none','gpu_roles','gpu_cards','hip_ordinal'])delete cfg[k];Object.assign(cfg,m);const count=Array.isArray(m.gpu)?m.gpu.length:m.gpu_none?0:1;cfg.layer_split=count>1?validateNativeSettings({layer_split:args.layerSplit??'auto'}).layer_split:'auto';if(cfg.layer_split!=='auto'&&cfg.layer_split.split(',').length!==count-1)throw new Error('Choose one layer split point between each pair of GPUs.');}
  else   if(args.gpus!==undefined&&!(args.gpusFromRecommendation&&namedInFile)){const detected=new Set((snapshot.hardware?.gpus??[]).map(g=>g.index));if(!Array.isArray(args.gpus)||!args.gpus.length||new Set(args.gpus).size!==args.gpus.length||args.gpus.some(g=>!Number.isInteger(g)||!detected.has(g)))throw new Error('Choose distinct detected GPUs. CPU-only text inference is not supported by this native adapter.');cfg.gpu=args.gpus.length===1?args.gpus[0]:args.gpus;for(const k of ['gpu_roles','gpu_cards','gpu_none'])delete cfg[k];delete cfg.hip_ordinal;cfg.layer_split=args.gpus.length>1?validateNativeSettings({layer_split:args.layerSplit??'auto'}).layer_split:'auto';if(cfg.layer_split!=='auto'&&cfg.layer_split.split(',').length!==args.gpus.length-1)throw new Error('Choose one layer split point between each pair of GPUs.');}
  if(cfg.vram_reserve_mib!==undefined&&cfg.vram_reserve_mib!==null){cfg.args=[...(cfg.args??[])];const at=cfg.args.indexOf('--vram-reserve-mib');if(at>=0)cfg.args.splice(at,2);cfg.args.push('--vram-reserve-mib',String(cfg.vram_reserve_mib));}
  if(mine.length){cfg.args=mergeUserArgs(cfg.args??[],mine);cfg[USER_KEY]=mine;}else delete cfg[USER_KEY];
  if(cfg.vision){if(cfg.vision_tokens!=null)cfg.vision={...cfg.vision,max_tokens:cfg.vision_tokens};if(cfg.vision_min_tokens!=null)cfg.vision={...cfg.vision,min_tokens:cfg.vision_min_tokens};}cfg.mcp={...(cfg.mcp??{}),...(cfg.mcp_timeout_s!=null?{timeout_s:cfg.mcp_timeout_s}:{}),...(cfg.mcp_max_rounds!=null?{max_rounds:cfg.mcp_max_rounds}:{}),...(cfg.mcp_max_result_chars!=null?{max_result_chars:cfg.mcp_max_result_chars}:{})};
  /* The model's own choices win over what setup wrote into the arguments: "Draft layer (MTP)" off leaves --mtp out of every start, and a picture reading that is off leaves --vision out (setup writes both again whenever it runs; the saved settings are applied here, on every load). */
  if((cfg.mtp===false||cfg.drafting==='off')&&Array.isArray(cfg.args))cfg.args=withoutDraftLayer(cfg.args);
  if(!cfg.vision&&Array.isArray(cfg.args))cfg.args=cfg.args.filter(a=>a!=='--vision');
  for(const arg of cfg.args??[])if(arg.endsWith('.gguf')&&!existsSync(resolve(repo(),arg)))throw new Error('An engine argument names a missing GGUF. Restore the saved arguments or repair the files.');
  /* The app's own choice of request slots (4, fewer where the card has no room) is part of the configuration, so a tune measures the engine a load will start: slots take VRAM from the expert cache. */
  const parallelChoice=packaged?chooseParallel({config:cfg,values:args.values??{},hardware:snapshot.hardware?.gpus?.length?snapshot.hardware:{gpus:hardwareProbe()??[]}}):null;
  let parallelNote=null;
  /* The agent asks for long answers (a thinking model can use most of the window): let the engine clamp the output to what is left of the context instead of refusing the request. */
  if(packaged&&cfg.fit_max_tokens===undefined)cfg.fit_max_tokens=true;
  if(parallelChoice){const applied=applyParallel(cfg,parallelChoice);for(const k of Object.keys(cfg))if(!(k in applied.cfg))delete cfg[k];Object.assign(cfg,applied.cfg);parallelNote=applied.note;}
  return {cfg,resources,visionPreferences,parallelChoice,parallelNote};
 }
 /* The DFlash2 drafter, before a load that uses it: Control says whether the choice needs it (and the file is not here), downloads it pinned and checked, and packs it, with plain progress in the job.
    Never stops the load: a drafter that cannot be had leaves the MTP running, and the job says so. The choice itself is remembered in the model's settings file. */
 async function prepareDrafter(job,args,cfg,configFile){
  const asked=args.values?.drafting;
  if(asked!==undefined)rememberSetting(configFile,'drafting',asked===null||asked==='auto'?null:asked);
  if(!packaged||typeof control.draftingFetch!=='function'||!cfg.drafting||cfg.drafting==='auto'||cfg.drafting==='mtp'||cfg.drafting==='off')return;
  const work=join(directory,'native-instances');mkdirSync(work,{recursive:true});const probe=join(work,'drafting-'+randomUUID()+'.json');
  try{
   writeFileSync(probe,JSON.stringify(cfg),{mode:0o600});
   job.engineAbort??=new AbortController();
   await control.draftingFetch({configPath:probe,mode:cfg.drafting,signal:job.engineAbort.signal,onEvent:event=>{if(event.event==='progress'&&job.state==='running'){job.summary=event.step;job.details=event.total>1&&event.done<event.total?`${(event.done/1e9).toFixed(2)} of ${(event.total/1e9).toFixed(2)} GB`:undefined;emit();}}});
   if(job.state==='running'){job.summary='Loading the model…';job.details=undefined;emit();}
  }catch(error){if(job.state==='running'){job.drafterNote=error.message;job.summary='The DFlash2 drafter could not be downloaded, so the model starts with the MTP. Loading the model…';emit();}}
  finally{try{rmSync(probe,{force:true});}catch{/* best effort */}}
 }
 /* One key of a model's settings file, set or removed (the Drafting choice): written like the Settings screen writes it, to be there for the next load. */
 function rememberSetting(configFile,key,value){try{const current=readJson(configFile);if(value===null?!(key in current):current[key]===value)return;const next={...current};if(value===null)delete next[key];else next[key]=value;const temp=configFile+'.ui-tmp';writeFileSync(temp,JSON.stringify(next,null,2)+'\n',{mode:0o600});renameSync(temp,configFile);}catch{/* the load goes on with the choice for this start */}}
 /* Which drafter runs, when it is not the one that was asked for (the engine's own words, "Drafting: MTP (DFlash2 didn't fit)"): under Details of the load. */
 async function noteDrafting(job,instance,key){let note='';try{const r=await fetcher(`http://127.0.0.1:${instance.port}/metrics`,{headers:{authorization:`Bearer ${key}`},signal:AbortSignal.timeout(2000)});if(r.ok)note=String((await r.json())?.engine?.drafting_note??'');}catch{/* a convenience */}const lines=[job.drafterNote,note].filter(Boolean);if(lines.length)job.details=[job.details,...lines].filter(Boolean).join('\n');}
 async function loadEngine(job,args){
  await refresh(true);if(job.state!=='running')return;const {config,entry,path:configFile}=record(args.model);assertSettingsFile(configFile);if(packaged){const engine=await control.engine();if(!engine.usable)throw new Error(ENGINE_NOT_RECOGNISED+(engine.notice?` ${engine.notice}`:''));}if(!entry.ready){const error=missingFilesError(args.model,entry.missing);append(args.model,error.details[0]);throw error;}
  if(externalRunning().length)throw new Error('A Strata server is already running outside this UI. Stop it before loading here, or connect its API in Settings → Models.');
  if(args.multiple&&instances.size>=8)throw new Error('Unload an instance before adding another. Up to eight native instances can run here.');
  const {cfg,resources,visionPreferences,parallelChoice,parallelNote:chosenNote}=buildConfig(args,config);
  let parallelNote=chosenNote;
  await prepareDrafter(job,args,cfg,configFile);if(job.state!=='running')return;
  const fingerprint=createHash('sha256').update(JSON.stringify({...cfg,api_key:undefined,port:undefined})).digest('hex');
  const placement=Array.isArray(cfg.gpu)?cfg.gpu:cfg.gpu===undefined?[]:[cfg.gpu];const matching=[...instances.values()].find(i=>i.model===args.model&&JSON.stringify(i.gpus)===JSON.stringify(placement));
  if(matching?.status==='ready'&&matching.fingerprint===fingerprint&&(args.multiple||instances.size===1))return {summary:`${matching.label} is already ready.`,instanceId:matching.id};
  const wasDefault=matching?.id===defaultId;
  const id=matching?.id??randomUUID();let apiModelId=matching?.apiModelId??args.model,suffix=1;while(!matching&&[...instances.values()].some(i=>i.apiModelId===apiModelId))apiModelId=`${args.model}--${++suffix}`;
  cfg.aliases=[...new Set([...(Array.isArray(cfg.aliases)?cfg.aliases:typeof cfg.aliases==='string'?cfg.aliases.split(',').map(s=>s.trim()).filter(Boolean):[]),apiModelId])];
  const key=randomBytes(32).toString('hex');cfg.api_key=key;cfg.port=await freePort(args.port);
  const work=join(directory,'native-instances');mkdirSync(work,{recursive:true});const configPath=join(work,`${id}.json`);
  const py=packaged?null:venvPython(repo());if(!packaged&&!existsSync(py))throw new Error('The Strata Python environment is missing. Use Download / repair first.');
  if(job.state!=='running')return;
  await beforeLoad();if(job.state!=='running')return;
  if(!args.multiple){if(busy(requests().filter(r=>instances.has(r.instanceId))))throw new Error('Finish or cancel native requests before switching models.');for(const id of [...instances.keys()])await unload(id);}else if(matching)await unload(matching.id);
  if(job.state!=='running')return;
  writeFileSync(configPath,JSON.stringify(cfg,null,2),{mode:0o600});
  const instance={id,apiModelId,fingerprint,context:entry.context??0,aliases:cfg.aliases??[],vision:!!cfg.vision,resourceProfile:cfg.ui_resource_profile??null,model:args.model,label:friendlyModelName(snapshot.catalog,args.model)??entry.model_name??args.model,port:cfg.port,gpus:Array.isArray(cfg.gpu)?cfg.gpu:cfg.gpu===undefined?[]:[cfg.gpu],status:'loading',phase:'loading',parallel:cfg.parallel??1,mcpServers:cfg.mcp_servers??cfg.mcpServers??{},key,configPath,loadedAt:new Date().toISOString(),metrics:null,error:null};
  /* Another model on the same graphics card: the engine takes the card's free memory, so a second one usually finds none. Loading goes on (recommend, never force); if it fails, the reason is said. */
  const sharedWith=instance.gpus.length?[...instances.values()].find(i=>i.id!==id&&i.status!=='error'&&i.model!==args.model&&i.gpus.some(g=>instance.gpus.includes(g))):null;
  const sharedSentence=sharedWith?`GPU ${instance.gpus.find(g=>sharedWith.gpus.includes(g))+1} is already used by ${sharedWith.label}, so this model did not fit there. Load it on another GPU, or unload ${sharedWith.label} first.`:null;
  let proc=null,started=null,retried=null;
  if(packaged){
   /* The engine starts under Control's owner runtime, not as our child: a saved profile per slot, then `runtime start`. */
   const used=new Set([...instances.values()].map(i=>i.profile));let slot=1;while(used.has('app-'+slot))slot++;instance.profile='app-'+slot;
   instances.set(id,instance);if(!defaultId||!args.multiple||wasDefault)defaultId=id;emit();
   const startWith=value=>{writeFileSync(configPath,JSON.stringify(value,null,2),{mode:0o600});return control.stop(instance.profile).catch(()=>{}).then(()=>control.saveProfile(instance.profile,configPath)).then(()=>control.start(instance.profile,{waitSeconds:300}));};
   started=startWith(cfg);
   started.then(()=>append(id,'Strata Control reports the engine is ready.'),error=>{append(id,error.message);});
   try{await started;}catch(firstError){let error=firstError;
    /* A settings, profile or run file an elevated run made can be closed to this user: repair the access in Strata's own folders and start again (this is not the memory try below). */
    const isAccess=e=>/PermissionError|access is denied|EACCES/i.test(String(e?.message??'')+JSON.stringify(e?.output??''));
    if(isAccess(error)&&typeof control.repairAccess==='function'&&job.state==='running'){await control.repairAccess();append(id,'Strata repaired access to its own files; starting again.');try{await startWith(cfg);started=Promise.resolve();error=null;}catch(again){error=again;}}
    if(error&&sharedSentence&&!isAccess(error)){instance.status='error';instance.error=sharedSentence;emit();throw new Error(sharedSentence);}
    if(error){const said=humanStartError(error);
    if(isAccess(error)){instance.status='error';instance.error=said;emit();throw new Error(said);}   // access errors are not memory stops: no safer retry
    /* ONE automatic second try in all, never saved, never repeated (args.retrySafer false turns it off): with a safer memory setting when the
       failure looks like a memory stop, and without several-requests-at-once when that was the app's own default (not a number the user chose).
       Both changes ride on the same single try, with the plain sentence of the memory change when there is one. */
    const allowed=args.retrySafer!==false&&job.state==='running';
    const safer=allowed?saferStart(cfg,snapshot.hardware,modelArena(args.model),said):null;
    const dropParallel=allowed&&cfg.parallel>1&&!parallelChoice?.explicit;
    if(!safer&&!dropParallel){instance.status='error';instance.error=said;emit();throw new Error(said);}
    let retryConfig=safer?safer.config:cfg;if(dropParallel)retryConfig=withoutParallel(retryConfig);
    const words=safer?safer.words:FALLBACK_NOTE,detail=[safer?.detail,dropParallel?'one request at a time':null].filter(Boolean).join('; ');
    retried={words,detail};if(dropParallel){instance.parallel=1;if(!safer)parallelNote=FALLBACK_NOTE;}
    append(id,`The first start stopped (${said}). Starting once more: ${detail}`);job.summary=words;job.details=[said,`Second try: ${detail}`];instance.notice=words;emit();
    try{await startWith(retryConfig);append(id,'Strata Control reports the engine is ready (second try).');}
    catch(second){const said2=humanStartError(second);instance.status='error';instance.error=said2;/* the first stop and the second try stay in job.details; the pop-up shows the last stop once */emit();throw new Error(instance.error);}}}
  } else {
  proc=spawnProcess(py,[join(repo(),'serve/server.py'),'--engine','strata','--config',configPath,'--port',String(cfg.port),'--host','127.0.0.1'],{cwd:repo(),detached:process.platform!=='win32',stdio:['ignore','pipe','pipe'],env:{...process.env,PYTHONUNBUFFERED:'1',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8',STRATA_API_KEY:''}});
  instances.set(id,instance);processes.set(id,proc);if(!defaultId||!args.multiple||wasDefault)defaultId=id;
  for(const stream of [proc.stdout,proc.stderr])stream.on('data',chunk=>{const lines=chunk.toString().split('\n').filter(Boolean);lines.forEach(line=>append(id,line.replaceAll(key,'[redacted]')));job.summary=lines.at(-1)??job.summary;emit();});
  proc.on('error',error=>{instance.status='error';instance.error=error.message;emit();});proc.on('exit',code=>{if(instances.has(id)&&instance.status!=='unloading'){instance.status='error';instance.error=`Engine stopped${code===null?'':` (exit ${code})`}. Open diagnostics or reload.`;emit();}});emit();
  }
  const deadline=Date.now()+300000;while(Date.now()<deadline&&job.state==='running'){
   if(instance.status==='error')throw new Error(instance.error);
   try{const response=await fetcher(`http://127.0.0.1:${instance.port}/health`,{headers:{authorization:`Bearer ${key}`},signal:AbortSignal.timeout(2000)});if(job.state!=='running'){await unload(id);return;}if(response.ok){instance.status='ready';instance.phase='idle';await noteDrafting(job,instance,key);if(resources||args.vision!==undefined||JSON.stringify(cfg[USER_KEY]??null)!==JSON.stringify(config[USER_KEY]??null))writeModelConfig(record(args.model).path,{...config,...(resources?{...resources,ui_resource_profile:args.resourceProfile}:{}),...visionPreferences,...(cfg[USER_KEY]?{[USER_KEY]:cfg[USER_KEY]}:{})});emit();ensurePoll();instance.parallelNote=parallelNote;return {summary:retried?`${instance.label} is ready, after a second try: ${retried.words}`:`${instance.label} is ready.${parallelNote?' '+parallelNote:''}`,instanceId:id};}}catch{}
   await new Promise(resolve=>setTimeout(resolve,1200));
  }
  if(job.state!=='running'){await unload(id);return;}
  if(instance.profile)await control.stop(instance.profile).catch(()=>{});else await stopProcess(proc);instance.status='error';instance.error='Loading took more than five minutes. Check the engine log and free memory.';throw new Error(instance.error);
 }
 async function freePort(requested){for(let port=Number(requested??18500);port<=Number(requested??18500)+20;port++){if(!Number.isInteger(port)||port<1024||port>65535)throw new Error('Choose a port from 1024 to 65535.');if([...instances.values()].some(i=>i.port===port)){if(requested)throw new Error('This private engine port is already in use.');continue;}const available=await new Promise(resolve=>{const probe=createServer();probe.once('error',()=>resolve(false));probe.listen(port,'127.0.0.1',()=>probe.close(()=>resolve(true)));});if(available)return port;if(requested)throw new Error('This private engine port is already in use.');}throw new Error('No free private engine port was found.');}
 let pollCount=0;
 /* The engine's metrics carry every finished request with `?requests=all`: a long session makes that a large JSON that the host parses and
    pushes to the window every 2 s. The last dozen is what the app shows live; the full list is refreshed every tenth poll. */
 async function poll(){if(polling||closing)return;polling=true;const full=pollCount++%10===0;try{await Promise.all([...instances.values()].map(async instance=>{if(instance.status!=='ready')return;try{const response=await fetcher(`http://127.0.0.1:${instance.port}/metrics${full?'?requests=all':''}`,{headers:{authorization:`Bearer ${instance.key}`},signal:AbortSignal.timeout(1500)});if(!response.ok)throw new Error('Metrics unavailable');const fresh=await response.json();const older=instance.metrics?.requests,recent=fresh.requests;instance.metrics=full||!Array.isArray(older)||!Array.isArray(recent)||!recent.length?fresh:{...fresh,requests:[...recent,...older.filter(r=>Number(r.time)<Math.min(...recent.map(x=>Number(x.time))))]};const tools=await fetcher(`http://127.0.0.1:${instance.port}/mcp`,{headers:{authorization:`Bearer ${instance.key}`},signal:AbortSignal.timeout(1500)});if(tools.ok)instance.metrics.mcp=await tools.json();instance.phase=instance.metrics.live?.state??'idle';}catch{instance.metrics=null;instance.phase='unavailable';}}));emit();}finally{polling=false;}}
 function ensurePoll(){if(!timer){timer=setInterval(()=>void poll(),2000);timer.unref?.();}void poll();}
 async function benchmark(job,args){const instance=resolveInstance(args.instanceId);job.instanceId=instance.id;if(busy(requests().filter(r=>r.instanceId===instance.id)))throw new Error('Wait for this model to finish before benchmarking.');const started=Date.now();job.abort=new AbortController();const response=await fetcher(`http://127.0.0.1:${instance.port}/v1/chat/completions`,{method:'POST',headers:{'content-type':'application/json',authorization:`Bearer ${instance.key}`},signal:AbortSignal.any([job.abort.signal,AbortSignal.timeout(180000)]),body:JSON.stringify({model:instance.apiModelId,messages:[{role:'user',content:'Explain how a rainbow forms in clear, simple language.'}],temperature:0,reasoning_effort:'none',max_tokens:200,stream:false})});const body=await response.json();if(!response.ok)throw new Error(body.error?.message??'Benchmark failed.');if(job.state!=='running')return;const result={summary:'Measured benchmark complete.',at:new Date().toISOString(),model:instance.apiModelId,durationMs:Date.now()-started,tokensIn:body.usage?.prompt_tokens??null,tokensOut:body.usage?.completion_tokens??null,outputTps:body.timings?.predicted_per_second??null,promptTps:body.timings?.prompt_per_second??null,source:'engine',settings:'Greedy · thinking off · up to 200 output tokens'};writeFileSync(join(directory,'native-benchmark.json'),JSON.stringify(result,null,2),{mode:0o600});return result;}
 function resolveInstance(id){const aliases=[...instances.values()].filter(i=>i.aliases?.includes(id));if(aliases.length>1&&!instances.has(id)&&![...instances.values()].some(i=>i.apiModelId===id))throw new Error('This alias matches more than one instance. Choose its exact model ID.');const i=instances.get(id)??[...instances.values()].find(i=>i.apiModelId===id)??aliases[0]??(!id&&instances.size===1?[...instances.values()][0]:undefined);if(!i)throw new Error('Choose a loaded model ID.');if(i.status!=='ready')throw new Error('This model is not ready.');return i;}
 async function program(job,args){if(packaged)return componentProgram(job);if(externalRunning().length)throw new Error('Stop the native server running outside this UI before changing engine files.');if(instances.size)throw new Error('Unload native models before changing engine files.');const folder=repo();let {command,args:prefix}=python(),argv;
  if(job.kind==='calibrate'||job.kind==='tune'){if(!state().capabilities.calibration)throw new Error('This installation calibrates NVIDIA GPUs only. Use the short benchmark on other devices.');const model=args.model??snapshot.installed?.models?.[0]?.model;const chosen=record(model);if(!chosen.entry.ready)throw new Error('Prepare this model before calibrating.');command=venvPython(folder);prefix=[];if(!existsSync(command))throw new Error('Prepare the Python environment first.');argv=['-c','import sys; from pathlib import Path; import setup; sys.exit(0 if setup.calibrate_config(Path(sys.argv[1])) else 1)',chosen.path];}
  else if(job.kind==='rollback'){if(!previousEngine())throw new Error('No saved previous engine is available.');argv=[join(folder,'setup.py'),'--rollback-engine'];}
  else if(job.kind==='update-engine'){if(!snapshot.installed?.models?.length)throw new Error('Prepare a model first. Its installer downloads the matching engine.');argv=[join(folder,'setup.py'),'--update'];}
  else if(job.kind==='update-strata'){const status=await execute('git',['status','--porcelain'],folder);if(status.trim())throw new Error('This Strata checkout has local edits. Save or commit them before updating.');const remote=await execute('git',['remote','get-url','origin'],folder);if(!/^https:\/\/github\.com\/Niko1221\/Strata(?:\.git)?\s*$/.test(remote))throw new Error('The native installation must use the official Strata repository to update here.');await execute('git',['pull','--ff-only'],folder);argv=[join(folder,'setup.py'),'--update'];}
  else throw new Error('Unsupported native operation.');
  return new Promise((resolve,reject)=>{const proc=spawnProcess(command,[...prefix,...argv],{cwd:folder,windowsHide:true,detached:process.platform!=='win32',stdio:['ignore','pipe','pipe'],env:{...process.env,PYTHONUNBUFFERED:'1',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8'}});job.process=proc;for(const stream of [proc.stdout,proc.stderr])stream.on('data',chunk=>{chunk.toString().split('\n').filter(Boolean).forEach(line=>append(job.id,line));job.summary=chunk.toString().trim().slice(-500);emit();});proc.on('error',reject);proc.on('exit',code=>{delete job.process;code===0?resolve({summary:job.kind==='calibrate'?'Calibration complete. Native settings were saved.':'Native installation updated.'}):reject(new Error(`Operation stopped (exit ${code}). Open diagnostics for details.`));});});
 }
 /* Packaged app: engine and Control updates and the engine rollback go through Control's explicit update step (`strata update`,
    `strata components rollback`), never through git, setup.py or a download in this process. The app itself is updated by the
    shell's signed Tauri updater. Models are not part of any of it. */
 async function componentProgram(job){
  if(!['update-engine','update-strata','rollback'].includes(job.kind)||typeof control.update!=='function')throw new Error(NO_INSTALL);
  if(instances.size)throw new Error('Unload native models before changing engine files.');
  job.abort=new AbortController();
  const subject=job.kind==='update-strata'?'Strata':'engine';
  try{return await componentSteps(job,subject);}catch(error){throw error?.details!==undefined&&error.message?error:plainError(error,subject);}
 }
 async function componentSteps(job,subject){
  const onEvent=event=>{
   if(event.event==='progress'){job.summary=`Downloading: ${event.percent}%`;job.result={install:{step:job.summary,download:{file:event.label,percent:event.percent}}};}
   else if(event.event==='phase'&&PHASE_TEXT[event.phase])job.summary=PHASE_TEXT[event.phase];
   else if(event.event==='waiting')job.summary='Waiting for the running model to stop…';
   emit();
  };
  if(job.kind==='rollback'){const done=await control.rollbackEngine();return {summary:'The previous engine is back. Your models and settings are unchanged.',result:done};}
  if(job.kind==='update-engine'){const done=await control.update({engine:true,control:false,onEvent,signal:job.abort.signal});return {summary:updateSummary(done),result:done};}
  const done=await control.update({engine:false,control:true,onEvent,signal:job.abort.signal});
  const notes=[updateSummary(done)];
  if(appUpdater){
   job.summary='Looking for a new version of the app…';emit();
   let verdict;
   try{verdict=await appUpdater.check();}catch(error){if(error?.code!=='app-update-unavailable')throw error;verdict={available:false};notes.push(appUpdater.unavailable||error.message);}
   if(verdict.available){await appUpdater.install({signal:job.abort.signal,onProgress:p=>{if(p.total){job.summary=`Downloading the app update: ${Math.round(100*p.downloaded/p.total)}%`;job.result={install:{step:job.summary,download:{file:'Strata app',percent:Math.round(100*p.downloaded/p.total)}}};emit();}}});notes.push('The app update is installed. Strata will restart.');}
  }
  return {summary:notes.filter(Boolean).join(' '),result:done};
 }
 function updateSummary(done){const r=(done.results??[]).filter(x=>x.status==='updated').map(x=>x.component==='engine'?'The engine is updated and ready.':'Strata is updated. It takes effect the next time you open the app.');return r.length?r.join(' '):'Everything is already up to date.';}
 function execute(cmd,args,cwd){return new Promise((resolve,reject)=>{let output='';const process=spawnProcess(cmd,args,{cwd,stdio:['ignore','pipe','pipe']});process.stdout.on('data',c=>{output+=c;if(output.length>100000)process.kill();});process.stderr.on('data',c=>append('update',c.toString()));process.on('error',reject);process.on('exit',code=>code===0?resolve(output):reject(new Error('Git could not update this installation. Open diagnostics.')));});}
 /* Every write of a model's run config keeps the person's options: those they wrote into "args" or "user_args" since the file was last written are read from the file itself, go over what the caller built, and "args_written" is refreshed. */
 function writeModelConfig(path,next){const mine=mergeUserArgs(captureUserArgs(readJson(path)),next[USER_KEY]??[]);next=stampedUserArgs(next,mine);copyFileSync(path,path+'.bak');chmodSync(path+'.bak',0o600);const temp=path+'.ui-tmp';writeFileSync(temp,JSON.stringify(next,null,2)+'\n',{mode:0o600});renameSync(temp,path);}
 async function tune(job,args){
  await refresh(true);if(job.state!=='running')return;
  const chosen=record(args.model);resourceSettings(chosen.config,snapshot.hardware,args.profile,modelArena(args.model));
  if(!chosen.entry.ready)throw new Error('Prepare the model before tuning.');
  if(externalRunning().length)throw new Error('Stop the native server outside this UI before tuning.');
  if(busy(requests().filter(r=>instances.has(r.instanceId))))throw new Error('Finish requests before tuning.');
  for(const id of [...instances.keys()])await unload(id);
  if(job.state!=='running')return;
  const initialValues=resourceSettings(chosen.config,snapshot.hardware,args.profile,modelArena(args.model));writeModelConfig(chosen.path,{...chosen.config,...initialValues,ui_resource_profile:args.profile});
  if(args.calibrate!==false&&state().capabilities.calibration){job.summary='Measuring the best engine settings for this computer…';emit();await program(job,args);}
  if(job.state!=='running')return;
  const current=record(args.model),values=resourceSettings(current.config,snapshot.hardware,args.profile,modelArena(args.model));
  const next={...current.config,...values,ui_resource_profile:args.profile};
  writeModelConfig(current.path,next);
  job.summary='Applying your resource preference and loading the model…';emit();
  const loaded=await loadEngine(job,{model:args.model,multiple:false,vision:current.config.vision?args.vision:'no'});
  if(job.state!=='running'||!loaded?.instanceId)return;
  job.summary='Checking your model’s speed…';emit();
  const measured=await benchmark(job,{instanceId:loaded.instanceId});
  return {...loaded,...measured,summary:'Your model is tuned and ready.',profile:args.profile};
 }
 /* What the tuner measures is what a load would start: the saved model config with the user's choices and the resource profile. */
 function tuneConfig(args){const {config,entry}=record(args.model);if(!entry.ready)throw new Error('Prepare the model before tuning it.');const {cfg}=buildConfig({model:args.model,vision:args.vision,resourceProfile:args.resourceProfile,gpus:args.gpus,values:args.values},config);return {path:writeTuneConfig(directory,args.model,cfg),entry};}
 async function tuneStatus(args){await refresh();const {path}=tuneConfig(args);return autotuner.status(path);}
 async function tuneEffective(args){await refresh();const {path}=tuneConfig(args);return autotuner.effective(path);}
 async function tuneOff(args){await refresh();const {path}=tuneConfig(args);return autotuner.off(path);}
 async function autotune(job,args){
  await refresh(true);if(job.state!=='running')return;
  if(packaged){const engine=await control.engine();if(!engine.usable)throw new Error(ENGINE_NOT_RECOGNISED+(engine.notice?` ${engine.notice}`:''));}
  const {path}=tuneConfig(args);
  if(externalRunning().length)throw new Error('Stop the native server running outside this UI before tuning.');
  if(busy(requests().filter(r=>instances.has(r.instanceId))))throw new Error('Finish requests before tuning.');
  for(const id of [...instances.keys()])await unload(id);
  if(job.state!=='running')return;
  job.summary='Finding the fastest settings for your PC…';emit();
  const room=args.resourceProfile==='work';
  return autotuner.start(job,job.kind==='recheck'?'recheck':'tune',{configPath:path,presetId:args.presetId,mode:args.mode,use:args.use,goal:room?'room':'speed',leaveRam:room?20:0,allowSmallDifferences:args.allowSmallDifferences===true,name:args.name},{onProgress:p=>{job.summary=p.text??job.summary;emit();}});
 }
 function operation(kind,args={}){if(!['install','load','benchmark','tune','autotune','recheck','calibrate','update-engine','update-strata','rollback','remove'].includes(kind))throw new Error('Choose a supported native operation.');if(!snapshot.available)throw new Error(snapshot.error??'Connect the native Strata installation first.');if(kind==='install'||kind==='load'){const refusal=engineRefusal(snapshot.catalog,args.model,snapshot.engineInfo);if(refusal)throw new Error(refusal);}return begin(kind,job=>kind==='install'?install(job,args):kind==='load'?loadEngine(job,args):kind==='benchmark'?benchmark(job,args):kind==='tune'?tune(job,args):kind==='autotune'||kind==='recheck'?autotune(job,args):kind==='remove'?removeModel(job,args):program(job,args),{guided:args.guided===true,model:args.model,...(args.family?{family:args.family}:{})});}
 /* ---- The models folder card (0.2.2): what a picked folder means, and the three choices. The copying is Control's (strata models move). ---- */
 const currentModelsFolder=()=>preferences.downloadDirectory??(modelsDirectory?modelsDirectory():snapshot.installed?.data_folder)??null;
 const moveFile=()=>join(directory,'models-move.cancel');
 const folderWords=error=>({message:error.message,details:error.details||''});
 /** A picked folder as the data folder: a whole drive (D:\) becomes D:\Strata, and the person is told (the models then live in D:\Strata\Models). */
 function pickedModelsFolder(input){
  if(typeof input!=='string'||!input.trim()||/[\r\n\0]/.test(input))throw new Error('Choose a models folder.');
  const picked=resolve(input.trim()),target=dataFolderFromPick(picked);
  return {picked,folder:target.path,wholeDrive:!!target.note,shown:target.note?join(target.path,'Models'):target.path};
 }
 async function planModelsFolder(input){
  if(!packaged)throw new Error('Changing the models folder is part of the Strata app.');
  const now=currentModelsFolder();if(!now)throw new Error('Strata is still reading your models folder. Try again in a moment.');
  const pick=pickedModelsFolder(input),plan=await control.planModelsMove(now,pick.folder);
  return {...plan,...pick,current:now,running:instances.size,busy:[...jobs.values()].some(j=>j.state==='running')};
 }
 /* "Start empty": the old models are no longer listed. Their run configs are moved (not deleted) into old-models beside the others; the model files are not touched. */
 function setAsideConfigs(old){
  const folder=repo(),kept=join(folder,'old-models'),aside=[];
  const under=value=>typeof value==='string'&&[ 'models','packs','mtp' ].some(item=>{const base=join(old,item).toLowerCase();const v=value.toLowerCase();return v===base||v.startsWith(base+sep)||v.startsWith(base+'/');});
  const scan=value=>Array.isArray(value)?value.some(scan):value&&typeof value==='object'?Object.values(value).some(scan):under(value);
  for(const name of readdirSync(folder)){
   if(!/^strata-[a-z0-9_-]+\.json$/i.test(name)||name.endsWith('.shared-settings.json'))continue;
   if(!scan(readJson(join(folder,name))))continue;
   mkdirSync(kept,{recursive:true});renameSync(join(folder,name),join(kept,name));aside.push(name);
  }
  return aside;
 }
 async function startModelsFolder({folder:input,choice,stopRunning=false}){
  if(!packaged)throw new Error('Changing the models folder is part of the Strata app.');
  if(!['move','keep','empty'].includes(choice))throw new Error('Choose what to do with your models.');
  if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish or stop the download to change the folder.');
  const plan=await planModelsFolder(input);
  if(plan.issue)throw Object.assign(new Error(plan.issue.message),{details:plan.issue.details,code:plan.issue.code});
  if(!plan.choices[choice].possible)throw Object.assign(new Error(plan.choices[choice].reason||'That choice is not possible here.'),{code:'not_possible'});
  if(instances.size){
   if(!stopRunning)throw Object.assign(new Error('A model is running. Stop it first, then change the folder.'),{code:'running'});
   for(const id of [...instances.keys()])await unload(id);
  }
  const old=plan.current,target=plan.folder;
  if(choice!=='move'){
   makeFolder(target);
   const aside=choice==='empty'?setAsideConfigs(old):[];
   setDownloadDirectory(target);lastRead=0;await refresh(true);
   return {done:true,choice,folder:target,oldFolder:old,setAside:aside};
  }
  return runModelsMove(old,target,(from,to,options)=>control.moveModels(from,to,options));
 }
 /* The move as a job (the progress dialog reads it). `run` is moveModels for a new move, resumeModelsMove for "Finish it". */
 function runModelsMove(old,target,run){
  rmSync(moveFile(),{force:true});
  return begin('models-move',async job=>{
   job.cancelFile=moveFile();job.summary='Starting…';
   try{
    const result=await run(old,target,{installDir:repo(),cancelFile:moveFile(),onEvent:event=>{
     if(event.event!=='progress')return;
     job.progress={done:event.done,total:event.total,file:event.file,group:event.group,index:event.index,count:event.count,secondsLeft:event.seconds_left||0,sameDrive:!!event.same_drive};
     job.summary=`${Math.round(event.done/1e8)/10} of ${Math.round(event.total/1e8)/10} GB`;emit();}});
    const finished={done:!!result.done,stopped:!!result.stopped,moved:result.moved??[],remaining:result.remaining??[],bytes:result.bytes_moved??0,total:result.total_bytes??0,from:old,to:target,files:result.files_moved??0,sameDrive:!!result.same_drive};
    /* Only a whole move changes the folder the app uses: a stopped move leaves it where it was, and every model works where it is. */
    if(result.done){setDownloadDirectory(target,{insideJob:true});finished.folder=target;}
    job.progress=null;
    return {summary:result.done?'Your models are moved.':'Moving stopped. Nothing was lost.',...finished};
   }catch(error){
    const left=error.output;
    job.result={failed:true,code:error.code,moved:left?.moved??[],remaining:left?.remaining??[],bytes:left?.bytes_moved??0,total:left?.total_bytes??0,from:old,to:target};
    job.progress=null;
    throw error;
   }finally{rmSync(moveFile(),{force:true});}
  },{family:undefined});
 }
 /* "A move was interrupted. Finish it or leave it": the next start finds the unfinished move (Control keeps a note in the models folder in use). */
 async function interruptedMove(){
  if(!packaged)return null;
  const now=currentModelsFolder();if(!now)return null;
  if([...jobs.values()].some(j=>j.kind==='models-move'&&j.state==='running'))return null;
  try{return await control.interruptedModelsMove(now);}catch{return null;}
 }
 async function finishInterruptedMove({stopRunning=false}={}){
  const st=await interruptedMove();if(!st)throw new Error('No move was interrupted.');
  if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish or stop the download to change the folder.');
  if(instances.size){
   if(!stopRunning)throw Object.assign(new Error('A model is running. Stop it first, then change the folder.'),{code:'running'});
   for(const id of [...instances.keys()])await unload(id);
  }
  return runModelsMove(st.from,st.to,(from,to,options)=>control.resumeModelsMove(from,options));
 }
 async function leaveInterruptedMove(){
  const st=await interruptedMove();if(!st)throw new Error('No move was interrupted.');
  if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish or stop the download to change the folder.');
  const result=await control.leaveModelsMove(st.from,{installDir:repo()});
  lastRead=0;await refresh(true);
  return {done:true,folder:st.from,partial:st.to,removed:result.removed??0,restored:result.restored??0};
 }
 async function cancel(id){const job=jobs.get(id);if(!job||job.state!=='running')throw new Error('This operation is no longer running.');if(job.kind==='install'){job.engineAbort?.abort();await tool('strata_install',{cancel:true});}else if(job.kind==='models-move'){/* a safe stop: the copy in flight is dropped, files already moved stay */try{writeFileSync(job.cancelFile,'stop');}catch{/* the run ends anyway */}job.summary='Stopping…';emit();return state();}else if(job.abort)job.abort.abort();else if(job.process){if(!['tune','calibrate','autotune','recheck'].includes(job.kind))throw new Error('File updates cannot be cancelled safely while files are being replaced.');job.state='cancelled';await stopProcess(job.process);}job.state='cancelled';job.summary=job.kind==='install'?'Stopped. Completed downloads are kept.':'Stopped.';emit();return state();}
 async function close(){closing=true;clearInterval(timer);const installing=packaged&&child&&[...jobs.values()].some(j=>j.kind==='install'&&j.state==='running');if(installing)await send('tools/call',{name:'strata_install',arguments:{cancel:true}},15000).catch(()=>{});const operations=[];for(const job of jobs.values()){job.abort?.abort();if(job.process)operations.push(stopProcess(job.process));if(job.state==='running')job.state='cancelled';}await Promise.all(operations);await Promise.all([...instances.keys()].map(id=>unload(id,true).catch(()=>{})));disconnect();}
 // Remove nonserializable process references from public jobs.
 const publicState=()=>{const value=state();value.jobs=value.jobs.map(({process,engineAbort,rawResult,...job})=>job);return value;};
 async function localPreview(path){if(!packaged)throw new Error('Checking local model files is part of the Strata app.');if(typeof path!=='string'||!path.trim()||[...path].some(c=>c.charCodeAt(0)<32))throw new Error('Choose a folder.');return control.importPlan(path.trim());}
 return {state:publicState,tuneStatus,tuneEffective,tuneOff,tunePresets:{list:()=>autotuner.list(),action:(kind,id,extra)=>autotuner.action(kind,id,extra),undoFor:async args=>autotuner.undoFor(tuneConfig(args).path)},refresh,localPreview,configure,setDownloadDirectory,planModelsFolder,startModelsFolder,interruptedMove,finishInterruptedMove,leaveInterruptedMove,plan,settings,save,settingsFile,openSettingsFile,operation,cancel,unload,resolveInstance,poll,logs:()=>logs.slice(),benchmarkReport:()=>readJson(join(directory,'native-benchmark.json'),null),setDefault(id){defaultId=resolveInstance(id).id;emit();return publicState();},hasLive:()=>[...instances.values()].some(i=>i.status==='ready'),readyInstances:()=>[...instances.values()].filter(i=>i.status==='ready'),defaultInstanceId:()=>defaultId,close};
}
