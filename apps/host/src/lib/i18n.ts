import baseEnglish from '../locales/en.json';
import tune_en from '../locales/tune-en.json';
import tune_zh from '../locales/tune-zh.json';
import tune_ja from '../locales/tune-ja.json';
import tune_de from '../locales/tune-de.json';
import tune_es from '../locales/tune-es.json';
import tune_ru from '../locales/tune-ru.json';
import tune_pt from '../locales/tune-pt.json';
import tune_ko from '../locales/tune-ko.json';
import tools_en from '../locales/tools-en.json';
import tools_zh from '../locales/tools-zh.json';
import tools_ja from '../locales/tools-ja.json';
import tools_de from '../locales/tools-de.json';
import tools_es from '../locales/tools-es.json';
import tools_ru from '../locales/tools-ru.json';
import tools_pt from '../locales/tools-pt.json';
import tools_ko from '../locales/tools-ko.json';
import base_zh from '../locales/zh.json';
import base_ja from '../locales/ja.json';
import base_de from '../locales/de.json';
import base_es from '../locales/es.json';
import base_ru from '../locales/ru.json';
import base_pt from '../locales/pt.json';
import base_ko from '../locales/ko.json';

export const LANGUAGES=[{value:'en',label:'English'},{value:'zh',label:'中文（简体）'},{value:'ja',label:'日本語'},{value:'de',label:'Deutsch'},{value:'es',label:'Español'},{value:'ru',label:'Русский'},{value:'pt',label:'Português'},{value:'ko',label:'한국어'}] as const;
export type Language=typeof LANGUAGES[number]['value'];
/* The Tune model dialog's strings live in tune-<lang>.json (apps/host/scripts/make-tune-locales.py) so other interface work does not collide with them. */
/* The tool-server, picture and agent-result strings of the tools and images fixes live in tools-<lang>.json for the same reason. */
const english:Record<string,string>={...baseEnglish,...tune_en,...tools_en};
const [zh,ja,de,es,ru,pt,ko]=[{...base_zh,...tune_zh,...tools_zh},{...base_ja,...tune_ja,...tools_ja},{...base_de,...tune_de,...tools_de},{...base_es,...tune_es,...tools_es},{...base_ru,...tune_ru,...tools_ru},{...base_pt,...tune_pt,...tools_pt},{...base_ko,...tune_ko,...tools_ko}] as Record<string,string>[];
const catalogs:Record<Language,Record<string,string>>={en:english,zh,ja,de,es,ru,pt,ko};
const entities:Record<string,string>={amp:'&',lt:'<',gt:'>',quot:'"',apos:"'",nbsp:'\u00a0'};
const decode=(text:string)=>text.replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos|nbsp);/gi,(whole,key:string)=>key[0]==='#'?String.fromCodePoint(key[1].toLowerCase()==='x'?parseInt(key.slice(2),16):parseInt(key.slice(1),10)):entities[key]??whole);
const ids=new Map(Object.entries(english).map(([id,value])=>[decode(value),id]));
let language:Language='en';
export const validLanguage=(value:unknown):value is Language=>LANGUAGES.some(item=>item.value===value);
export const currentLanguage=()=>language;
export const localeTag=()=>language==='zh'?'zh-CN':language;
export function setTranslationLanguage(value:Language){language=value;}
const escape=(value:string)=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const patterns=Object.entries(english).filter(([,value])=>/\{\d+\}/.test(value)).map(([id,value])=>{
 const slots:number[]=[];let expression='',at=0;
 for(const match of value.matchAll(/\{(\d+)\}/g)){expression+=escape(value.slice(at,match.index))+'([\\s\\S]*?)';slots.push(Number(match[1]));at=match.index!+match[0].length;}
 expression+=escape(value.slice(at));
 return{id,slots,pattern:new RegExp('^'+expression+'$'),specificity:value.replace(/\{\d+\}/g,'').length};
}).sort((a,b)=>b.specificity-a.specificity);
const localizedTemplateSlots:Record<string,number[]>={s2473:[1,2],s0825:[0,1],s2455:[0],s2456:[0],s7400:[1]};

/** Translate interface copy only. Non-string React children pass through unchanged. */
export function tr<T>(value:T,values?:readonly unknown[]):T{
 if(typeof value!=='string')return value;
 const leading=value.match(/^\s*/)?.[0]??'',trailing=value.match(/\s*$/)?.[0]??'',key=decode(value.trim());
 const dictionary=catalogs[language]??english;
 let id=ids.get(key),slots=values;
 if(!id&&language!=='en')for(const candidate of patterns){const match=candidate.pattern.exec(key);if(match){id=candidate.id;const found:unknown[]=[];candidate.slots.forEach((slot,index)=>{const captured=match[index+1];found[slot]=localizedTemplateSlots[candidate.id]?.includes(slot)?tr(captured):captured;});slots=found;break;}}
 let text=decode(id?(dictionary[id]??english[id as keyof typeof english]):key);
 if(slots)text=text.replace(/\{(\d+)\}/g,(_,index)=>String(slots![Number(index)]??''));
 if(!id&&!values)return decode(value) as T;
 return (leading+text+trailing) as T;
}
