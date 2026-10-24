import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createRequire} from 'node:module';
import {build} from 'esbuild';

/* The installed app crashed on its first screen ("Cannot read properties of undefined (reading 'family')") when the
   catalog it was given had no `recommendation` and no families. Render the real GuidedSetup and NativeModels components
   (react-dom/server) with the state libraries replaced by fixed values, for every shape of an empty or failed catalog. */
const root=resolve(import.meta.dirname,'..');
/** Every export of the real module becomes a harmless no-op, except the few that carry the fixed state. */
function stubFor(file,overrides){
 const source=readFileSync(resolve(root,'src/lib',file),'utf8');
 const names=new Set([...source.matchAll(/export\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)/g)].map(m=>m[1]));
 for(const m of source.matchAll(/export\s*\{([^}]*)\}/g))for(const part of m[1].split(','))names.add(part.split(/\s+as\s+/).pop().trim());
 names.delete('');
 const lines=[...names].filter(n=>!(n in overrides)).map(n=>`export const ${n}=(...a)=>Promise.resolve({});`);
 for(const [n,v] of Object.entries(overrides))lines.push(`export const ${n}=${v};`);
 return lines.join(String.fromCharCode(10));
}
const stubs={
 native:stubFor('native.ts',{useNative:'()=>globalThis.__native',nativeState:'()=>globalThis.__native'}),
 store:stubFor('store.ts',{useStore:'sel=>sel(globalThis.__store)'}),
 client:stubFor('client.ts',{}),
 router:stubFor('router.ts',{}),
};

let dir,mod;
async function load(){
 if(mod)return mod;
 dir=mkdtempSync(join(tmpdir(),'strata-gs-'));
 const stubPlugin={name:'stubs',setup(b){
  b.onResolve({filter:/\/lib\/(native|store|client|router)$/},args=>({path:args.path.split('/').pop(),namespace:'stub'}));
  b.onLoad({filter:/.*/,namespace:'stub'},args=>({contents:stubs[args.path],loader:'js',resolveDir:root}));
 }};
 const entry=join(dir,'entry.tsx');
 const out=join(dir,'out.cjs');
 await build({
  stdin:{contents:`export {GuidedSetup} from ${JSON.stringify(resolve(root,'src/components/GuidedSetup.tsx'))};
   export {NativeModels} from ${JSON.stringify(resolve(root,'src/components/NativeModels.tsx'))};
   export {renderToString} from 'react-dom/server'; export {createElement} from 'react';`,resolveDir:root,sourcefile:'entry.tsx',loader:'tsx'},
  bundle:true,format:'cjs',platform:'node',outfile:out,jsx:'automatic',logLevel:'silent',plugins:[stubPlugin],
  loader:{'.svg':'dataurl','.png':'dataurl','.css':'empty'},
 });
 globalThis.window??=globalThis;globalThis.matchMedia??=()=>({matches:false,addEventListener(){},removeEventListener(){}});
 mod=createRequire(import.meta.url)(out);
 return mod;
}
test.after(()=>{if(dir)rmSync(dir,{recursive:true,force:true});});

const store=(complete=false)=>({setup:{complete,choices:{},app:{modelsDir:''}},hardware:null});
const base={available:true,repo:'',summary:'',error:null,installed:{models:[],data_folder:'',engine:{version:null}},hardware:null,instances:[],jobs:[],defaultId:null,fields:[],capabilities:{}};
const catalogs={
 'no catalog at all':null,
 'the packaged host before it had a catalog':{families:[],sizes:[]},
 'no recommendation':{families:[],contexts:[32768]},
 'recommendation with nothing in it':{families:[],contexts:[],recommendation:{family:null,model:null}},
 'no families key':{contexts:[]},
 'a family without sizes':{families:[{family:'qwen',title:'Qwen3.8-Flash-Next',sizes:[]}],contexts:[32768]},
};

for(const [name,catalog] of Object.entries(catalogs)){
 for(const component of ['GuidedSetup','NativeModels']){
  test(`${component} renders with ${name}`,async()=>{
   const m=await load();
   globalThis.__native={...base,catalog};globalThis.__store=store();
   let html;
   assert.doesNotThrow(()=>{html=m.renderToString(m.createElement(m[component]));});
   assert.equal(typeof html,'string');
  });
 }
}

test('GuidedSetup lists the real catalog when it has one',async()=>{
 const m=await load();
 globalThis.__native={...base,catalog:{families:[{family:'qwen',title:'Qwen3.8-Flash-Next',about:'',images:true,experimental:false,sizes:[
   {id:'iq3_xxs',model:'IQ3_XXS',download_gb:75.8,ram_gb:60,about:'',on_this_pc:'fits',installed:false}]}],contexts:[32768],
   recommendation:{family:'qwen',model:'IQ3_XXS',context:32768},context_note:''}};
 globalThis.__store={...store(),setup:{complete:false,choices:{onboarding:{step:2,manual:false}},app:{modelsDir:''}}};
 const html=m.renderToString(m.createElement(m.GuidedSetup));
 assert.match(html,/IQ3_XXS/);
});
