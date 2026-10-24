import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
const { outputFiles } = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/commands.ts',import.meta.url))], bundle: true, write: false, platform: 'node', format: 'cjs', plugins: [{ name: 'test-store', setup(build) { build.onResolve({ filter: /^\.\/store$/ }, () => ({ path: 'store', namespace: 'test-store' })); build.onLoad({ filter: /.*/, namespace: 'test-store' }, () => ({ contents: `
export const getState = () => testStore.state;
export const selectSession = id => { testStore.state.activeSession = id; testStore.calls.push(['select',id]); };
export const go = (...args) => testStore.calls.push(['go',...args]);
export const exportConversation=()=>{},newSession=()=>({id:'new'}),pushToast=()=>{},restartEngine=()=>{},setFocus=()=>{},setNavOpen=()=>{},setPalette=()=>{},setShortcuts=()=>{},setTheme=()=>{},toggleFocus=()=>{},toggleRail=()=>{},unloadModel=()=>{};`, loader: 'js' })); } }] });
function harness(overrides={}) {
  const testStore={state:{theme:'light',sessions:Array.from({length:8},(_,i)=>({id:String(i),title:'Chat '+i,updatedAt:`2026-10-${String(10-i).padStart(2,'0')}`})),activeSession:'7'},calls:[]};
  const module={exports:{}};
  new Function('require','module','exports','testStore','localStorage','navigator','window',outputFiles[0].text)(createRequire(import.meta.url),module,module.exports,testStore,{getItem:()=>JSON.stringify(overrides)},{platform:'MacIntel'},{location:{hash:'#/chat/7'}});
  const press=(key, extra={})=> {const event={key,code:'',metaKey:true,ctrlKey:false,altKey:false,shiftKey:false,target:{matches:()=>false},preventDefault(){this.prevented=true;},...extra};module.exports.handleAppShortcut(event);return event;};
  return {...module.exports,testStore,press};
}
test('numbered shortcuts switch to the same current-plus-five slots shown in the sidebar',()=>{
  const h=harness();
  assert.equal(h.press('2').prevented,true);
  assert.deepEqual(h.testStore.calls,[['select','1'],['go','chat','1']]);
  h.testStore.calls=[];
  h.press('1',{metaKey:false,ctrlKey:true});
  assert.deepEqual(h.testStore.calls,[['select','0'],['go','chat','0']]);
});
test('Option digit navigation stays distinct from numbered chats on macOS',()=>{
  const h=harness();h.press('¡',{altKey:true,code:'Digit1'});
  assert.deepEqual(h.testStore.calls,[['go','chat']]);
});
test('saved custom shortcuts dispatch after reload and recording suppresses app actions',()=>{
  const h=harness({'go:chat':{key:'c',mod:true,alt:false,shift:true}});
  h.press('C',{shiftKey:true});assert.deepEqual(h.testStore.calls,[['go','chat']]);
  h.testStore.state.shortcutsOpen=true;h.testStore.calls=[];h.press('1');assert.deepEqual(h.testStore.calls,[]);
});
