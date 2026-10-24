import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,realpathSync,rmSync,symlinkSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {defaultModelDirectory} from '../server/model-directory-default.mjs';
test('macOS and Linux default to Documents/Strata/Models',()=>{
 for(const platform of ['darwin','linux'])assert.equal(defaultModelDirectory({platform,home:'/home/me',env:{},read:()=>{throw new Error('no configuration');}}),'/home/me/Documents/Strata/Models');
});
test('Windows uses the known Documents folder including OneDrive redirection',()=>{
 assert.equal(defaultModelDirectory({platform:'win32',home:'C:\\Users\\Me',env:{},windowsDocuments:()=> 'C:\\Users\\Me\\OneDrive\\Documents'}),'C:\\Users\\Me\\OneDrive\\Documents\\Strata\\Models');
 assert.equal(defaultModelDirectory({platform:'win32',home:'C:\\Users\\Me',env:{},windowsDocuments:()=>{throw new Error('unavailable');}}),'C:\\Users\\Me\\Documents\\Strata\\Models');
});
test('Linux respects localized XDG Documents and rejects shell expressions',()=>{
 assert.equal(defaultModelDirectory({platform:'linux',home:'/home/me',env:{XDG_CONFIG_HOME:'/custom'},read:path=>{assert.equal(path,'/custom/user-dirs.dirs');return 'XDG_DOCUMENTS_DIR="$HOME/Dokumente"';}}),'/home/me/Dokumente/Strata/Models');
 assert.equal(defaultModelDirectory({platform:'linux',home:'/home/me',env:{XDG_DOCUMENTS_DIR:'${HOME}/Work'},read:()=>''}),'/home/me/Work/Strata/Models');
 assert.equal(defaultModelDirectory({platform:'linux',home:'/home/me',env:{XDG_DOCUMENTS_DIR:'$(echo /private)'},read:()=>''}),'/home/me/Documents/Strata/Models');
});
test('new hosts create the default; custom folders and legacy weights remain in place',t=>{
 const root=realpathSync(mkdtempSync(join(tmpdir(),'strata-documents-')));t.after(()=>rmSync(root,{recursive:true,force:true}));
 function boot(name,state,weights=false){const home=join(root,name),documents=join(root,name+'-documents');mkdirSync(join(home,'data'),{recursive:true});mkdirSync(join(home,'models'),{recursive:true});if(state)writeFileSync(join(home,'data','state.json'),JSON.stringify(state));if(weights){mkdirSync(join(home,'models','native','qwen'),{recursive:true});writeFileSync(join(home,'models','native','qwen','model.gguf'),'GGUFfixture');}const env={...process.env,STRATA_HOME:home,STRATA_DOCUMENTS_DIR:documents,STRATA_TEST_NATIVE:'1',STRATA_TEST_DEFAULT:join(documents,'Strata','Models')};delete env.STRATA_MODELS_DIR;const result=spawnSync(process.execPath,['--input-type=module','-e',"const store=await import('./server/store.mjs');if(process.env.STRATA_TEST_NATIVE==='1'&&store.MODELS_DIR===process.env.STRATA_TEST_DEFAULT){const {statSync}=await import('node:fs');const {join}=await import('node:path');const native=store.nativeDataDirectory(store.MODELS_DIR);const a=statSync(join(native,'models')),b=statSync(store.MODELS_DIR);if(a.ino!==b.ino||a.dev!==b.dev)throw new Error('Native weights miss requested model folder');}console.log(store.MODELS_DIR);"],{cwd:fileURLToPath(new URL('..',import.meta.url)),env,encoding:'utf8',windowsHide:true});assert.equal(result.status,0,result.stderr);return {path:result.stdout.trim(),home,documents};}
 const fresh=boot('fresh');assert.equal(fresh.path,join(fresh.documents,'Strata','Models'));
 const target=join(root,'redirected-target');mkdirSync(target);symlinkSync(target,join(root,'redirected-documents'),process.platform==='win32'?'junction':'dir');assert.equal(boot('redirected').path,join(target,'Strata','Models'));
 const custom=join(root,'custom');mkdirSync(custom);assert.equal(boot('custom-host',{modelsDir:custom}).path,custom);
 const legacy=boot('legacy',{},true);assert.equal(legacy.path,join(legacy.home,'models'));
});


test('legacy model libraries resolve to a guarded native data root without moving weights',t=>{
 const root=realpathSync(mkdtempSync(join(tmpdir(),'strata-legacy-models-')));t.after(()=>rmSync(root,{recursive:true,force:true}));
 for(const choice of ['detected','saved','environment']){
  const home=join(root,choice),models=join(home,'models');mkdirSync(join(home,'data'),{recursive:true});mkdirSync(models);
  mkdirSync(join(models,'IQ3_XXS'));writeFileSync(join(models,'IQ3_XXS','model.gguf'),Buffer.from([71,71,85,70,0,255,128,1]));
  mkdirSync(join(root,'documents','Strata','Models'),{recursive:true});
  if(choice==='saved')writeFileSync(join(home,'data','state.json'),JSON.stringify({modelsDir:models}));
  const env={...process.env,STRATA_HOME:home,STRATA_DOCUMENTS_DIR:join(root,'documents')};delete env.STRATA_MODELS_DIR;
  if(choice==='environment')env.STRATA_MODELS_DIR=models;
  const result=spawnSync(process.execPath,['--input-type=module','-e',`
   import assert from 'node:assert/strict';import {join,basename} from 'node:path';
   import {readFileSync,realpathSync,readdirSync,statSync} from 'node:fs';
   const store=await import('./server/store.mjs');const models=realpathSync(join(store.HOME,'models'));
   assert.equal(store.MODELS_DIR,models);
   const data=store.nativeDataDirectory(models);
   assert.match(basename(data),/^strata/i,'a new native data root must satisfy the installer guard');
   assert.equal(realpathSync(join(data,'models')),models,'native setup must write to the existing library');
   const weight=join(models,'IQ3_XXS','model.gguf'),nativeWeight=join(data,'models','IQ3_XXS','model.gguf');
   assert.deepEqual(readFileSync(nativeWeight),Buffer.from([71,71,85,70,0,255,128,1]),'native setup sees the original weight bytes');
   assert.equal(statSync(nativeWeight).ino,statSync(weight).ino,'the alias reuses the original file');
   assert.equal(store.nativeDataDirectory(models),data,'resolution is idempotent');
   assert.deepEqual(readFileSync(weight),Buffer.from([71,71,85,70,0,255,128,1]));
   assert.deepEqual(readdirSync(models),['IQ3_XXS'],'weights stay in the original library');
  `],{cwd:fileURLToPath(new URL('..',import.meta.url)),env,encoding:'utf8',windowsHide:true});
  assert.equal(result.status,0,result.stderr);
 }
});


test('native data roots leave explicit custom libraries and new installer folders unchanged',t=>{
 const root=realpathSync(mkdtempSync(join(tmpdir(),'strata-custom-models-')));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const custom=join(root,'chosen-library');mkdirSync(custom);writeFileSync(join(custom,'keep.gguf'),'GGUFfixture');
 for(const choice of ['saved','environment']){
  const home=join(root,choice);mkdirSync(join(home,'data'),{recursive:true});
  if(choice==='saved')writeFileSync(join(home,'data','state.json'),JSON.stringify({modelsDir:custom}));
  const env={...process.env,STRATA_HOME:home,STRATA_DOCUMENTS_DIR:join(root,'documents'),STRATA_TEST_CUSTOM:custom};delete env.STRATA_MODELS_DIR;
  if(choice==='environment')env.STRATA_MODELS_DIR=custom;
  const result=spawnSync(process.execPath,['--input-type=module','-e',`
   import assert from 'node:assert/strict';import {join} from 'node:path';import {existsSync,readFileSync} from 'node:fs';
   const store=await import('./server/store.mjs'),custom=process.env.STRATA_TEST_CUSTOM;
   assert.equal(store.MODELS_DIR,custom);assert.equal(store.nativeDataDirectory(custom),custom);
   const requested=join(store.HOME,'Strata-new');assert.equal(store.nativeDataDirectory(requested),requested);
   assert.equal(existsSync(requested),false);assert.equal(existsSync(join(store.HOME,'Strata-data')),false);
   assert.equal(readFileSync(join(custom,'keep.gguf'),'utf8'),'GGUFfixture');
  `],{cwd:fileURLToPath(new URL('..',import.meta.url)),env,encoding:'utf8',windowsHide:true});
  assert.equal(result.status,0,result.stderr);
 }
});

test('legacy native aliases never overwrite a conflicting library',t=>{
 const root=realpathSync(mkdtempSync(join(tmpdir(),'strata-conflicting-models-')));t.after(()=>rmSync(root,{recursive:true,force:true}));
 for(const kind of ['directory','link']){
 const home=join(root,kind),original=join(home,'models'),other=join(home,'other-library'),alias=join(home,'Strata-data','models');mkdirSync(original,{recursive:true});mkdirSync(other);mkdirSync(join(home,'Strata-data'));
 if(kind==='link')symlinkSync(other,alias,process.platform==='win32'?'junction':'dir');else mkdirSync(alias);
 writeFileSync(join(original,'original.gguf'),'GGUForiginal');writeFileSync(join(other,'other.gguf'),'GGUFother');
 if(kind==='directory')writeFileSync(join(alias,'other.gguf'),'GGUFconflict');
 const env={...process.env,STRATA_HOME:home,STRATA_DOCUMENTS_DIR:join(home,'documents')};delete env.STRATA_MODELS_DIR;
 const result=spawnSync(process.execPath,['--input-type=module','-e',`
  import assert from 'node:assert/strict';import {join} from 'node:path';import {readFileSync,realpathSync} from 'node:fs';
  const store=await import('./server/store.mjs');
  assert.throws(()=>store.nativeDataDirectory(store.MODELS_DIR),/conflicts with another library/);
  assert.equal(realpathSync(store.MODELS_DIR),realpathSync(join(store.HOME,'models')));
  assert.equal(readFileSync(join(store.MODELS_DIR,'original.gguf'),'utf8'),'GGUForiginal');
  assert.equal(readFileSync(join(store.HOME,'other-library','other.gguf'),'utf8'),'GGUFother');
  assert.equal(readFileSync(join(store.HOME,'Strata-data','models','other.gguf'),'utf8'),${JSON.stringify(kind==='link'?'GGUFother':'GGUFconflict')});
 `],{cwd:fileURLToPath(new URL('..',import.meta.url)),env,encoding:'utf8',windowsHide:true});
 assert.equal(result.status,0,result.stderr);
 }
});

test('an unavailable Documents default retains the upstream fallback without creating a legacy alias',t=>{
 const root=realpathSync(mkdtempSync(join(tmpdir(),'strata-fallback-models-')));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const home=join(root,'home'),documents=join(root,'blocked-documents');mkdirSync(home);writeFileSync(documents,'not a directory');
 const env={...process.env,STRATA_HOME:home,STRATA_DOCUMENTS_DIR:documents};delete env.STRATA_MODELS_DIR;
 const result=spawnSync(process.execPath,['--input-type=module','-e',`
  import assert from 'node:assert/strict';import {join} from 'node:path';import {existsSync} from 'node:fs';
  const store=await import('./server/store.mjs');
  assert.match(store.MODELS_DIR_NOTICE,/Strata could not use/);
  assert.equal(store.nativeDataDirectory(store.MODELS_DIR),store.MODELS_DIR);
  assert.equal(existsSync(join(store.HOME,'Strata-data')),false);
 `],{cwd:fileURLToPath(new URL('..',import.meta.url)),env,encoding:'utf8',windowsHide:true});
 assert.equal(result.status,0,result.stderr);
});
