import {test} from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {createRequire} from 'node:module';
const bundle=await build({stdin:{contents:"export {getState,setLanguage,setFontFamily,resetAppearance} from './src/lib/store'; export {tr,currentLanguage} from './src/lib/i18n';",resolveDir:process.cwd()},bundle:true,write:false,platform:'node',format:'cjs'});
function open(saved){
 const data=new Map(Object.entries(saved)),css=new Map(),root={dataset:{},style:{setProperty:(key,value)=>css.set(key,value),removeProperty:key=>css.delete(key)}};
 const storage={getItem:key=>data.get(key)??null,setItem:(key,value)=>data.set(key,value),removeItem:key=>data.delete(key)};
 const mod={exports:{}};
 new Function('require','module','exports','localStorage','sessionStorage','matchMedia','document',bundle.outputFiles[0].text)(createRequire(import.meta.url),mod,mod.exports,storage,storage,()=>({matches:false,addEventListener(){}}),{documentElement:root,querySelector:()=>null});
 return{...mod.exports,data,css,root};
}
test('saved language applies on the first render; language and installed font persist together',()=>{
 const first=open({'strata.prefs.v2':JSON.stringify({language:'ko',fontFamily:''})});
 assert.equal(first.currentLanguage(),'ko');assert.notEqual(first.tr('Settings'),'Settings');
 first.setLanguage('pt');first.setFontFamily('Arial');
 assert.equal(first.root.lang,'pt');assert.match(first.css.get('--st-font'),/^"Arial",system-ui/);
 const second=open(Object.fromEntries(first.data));assert.equal(second.currentLanguage(),'pt');assert.equal(second.getState().fontFamily,'Arial');
 second.setFontFamily('');assert.equal(second.css.has('--st-font'),false);assert.equal(second.getState().language,'pt');
});
test('invalid saved language falls back to English and font names reject control characters',()=>{
 const app=open({'strata.prefs.v2':JSON.stringify({language:'unknown',fontFamily:'Arial\u0000'})});assert.equal(app.currentLanguage(),'en');assert.equal(app.getState().fontFamily,'Arial');
});
