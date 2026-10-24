/** User bindings are separate from commands, so the dispatcher and labels agree. */
import { useSyncExternalStore } from 'react';
export interface Binding { key: string; mod: boolean; shift: boolean; alt: boolean; }
const KEY='strata.shortcuts.v1';
let overrides: Record<string,Binding|null> = {};
try { const raw=JSON.parse(localStorage.getItem(KEY)??'{}'); for(const [id,b] of Object.entries(raw)) if(b===null || (typeof b==='object' && b && typeof (b as Binding).key==='string' && ['mod','shift','alt'].every(k=>typeof (b as Record<string,unknown>)[k]==='boolean'))) overrides[id]=b as Binding|null; } catch { /* Defaults remain available. */ }
const listeners=new Set<()=>void>();
export const bindingFor=(id:string,fallback:Binding|null):Binding|null => Object.hasOwn(overrides,id)?overrides[id]:fallback;
export function setBinding(id:string,value:Binding|null|undefined) { overrides={...overrides}; if(value===undefined) delete overrides[id]; else overrides[id]=value; localStorage.setItem(KEY,JSON.stringify(overrides)); listeners.forEach(fn=>fn()); }
export function resetBindings(){overrides={};localStorage.removeItem(KEY);listeners.forEach(fn=>fn());}
export const useBindings=()=>useSyncExternalStore(fn=>{listeners.add(fn);return()=>listeners.delete(fn);},()=>overrides);
export const bindingEqual=(a:Binding|null,b:Binding|null)=>!!a&&!!b&&a.key===b.key&&a.mod===b.mod&&a.shift===b.shift&&a.alt===b.alt;
export function eventBinding(e:Pick<KeyboardEvent,'key'|'metaKey'|'ctrlKey'|'shiftKey'|'altKey'> & Partial<Pick<KeyboardEvent,'code'>>):Binding {
  // Option changes digit characters on macOS; navigation still uses the physical digit.
  const key=e.altKey && /^Digit[0-9]$/.test(e.code??'') ? e.code!.slice(5) : e.key;
  return {key:key.toLowerCase(),mod:e.metaKey||e.ctrlKey,shift:e.shiftKey,alt:e.altKey};
}
export function bindingKeys(b:Binding|null,mod:string) { return b?[...(b.mod?[mod]:[]),...(b.alt?['Alt']:[]),...(b.shift?['Shift']:[]),b.key===' '?'Space':b.key.length===1?b.key.toUpperCase():b.key]:['Unassigned']; }
export function composerSends(e:Pick<KeyboardEvent,'key'|'metaKey'|'ctrlKey'|'shiftKey'|'altKey'>){return bindingEqual(eventBinding(e),bindingFor('chat:send',{key:'enter',mod:false,shift:false,alt:false}));}
