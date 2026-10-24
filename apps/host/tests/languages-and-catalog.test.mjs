import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {build} from 'esbuild';
const languages=['en','zh','ja','de','es','ru','pt','ko'];
const catalogs=Object.fromEntries(await Promise.all(languages.map(async language=>[language,JSON.parse(await readFile(`src/locales/${language}.json`,'utf8'))])));
const english=catalogs.en;
const slots=text=>[...text.matchAll(/\{\d+\}/g)].map(match=>match[0]).sort();
const compiled=await build({stdin:{contents:"export * from './src/lib/i18n'; export * from './src/lib/native-catalog'; export * from './src/lib/session-title';",resolveDir:process.cwd()},bundle:true,write:false,platform:'node',format:'esm'});
const {tr,setTranslationLanguage,currentLanguage,localeTag,nativeCatalogModels,catalogVersionLabel,sessionTitle}=await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
test('every language covers every interface key and preserves interpolation slots',()=>{
 for(const language of languages){
  assert.deepEqual(Object.keys(catalogs[language]),Object.keys(english),language);
  for(const [id,source] of Object.entries(english)){
   const translated=catalogs[language][id];assert.equal(typeof translated,'string',`${language}:${id}`);
   assert(translated.trim()||language==='zh'&&['s2376','s2440'].includes(id),`${language}:${id} is empty`);
   const expected=slots(source).filter(slot=>!(language!=='en'&&id==='s2463'&&slot==='{1}'));
   assert.deepEqual(slots(translated),expected,`${language}:${id}`);
  }
 }
});
test('language switching translates static and formatted UI without changing supplied names or commands',()=>{
 for(const language of languages){
  setTranslationLanguage(language);assert.equal(currentLanguage(),language);assert.equal(localeTag(),language==='zh'?'zh-CN':language);
  const id=Object.keys(english).find(id=>english[id]==='Settings');assert.equal(tr('Settings'),catalogs[language][id]);
  const remove=Object.keys(english).find(id=>english[id]==='Edit {0}');
  for(const name of ['Ready','/goal','C:\\Users\\me\\Models','私のメモ'])assert.equal(tr('Edit {0}',[name]),catalogs[language][remove].replace('{0}',name));
  assert.equal(tr('Edit version'),catalogs[language][remove].replace('{0}','version'));
  assert.equal(tr('Qwen3.8-Flash-Next'),'Qwen3.8-Flash-Next');assert.equal(tr('/goal'),'/goal');
  assert.equal(tr('unrecognized user content Ω'),'unrecognized user content Ω');assert.equal(tr(42),42);assert.equal(tr(null),null);
  assert(!tr('Since {0}: {1} requests · {2} prompt tokens · {3} output tokens',['12:30',2,2048,100]).includes('{'));
 }
 setTranslationLanguage('en');assert.equal(tr('Privacy &amp; security'),'Privacy & security');
});
const size=(id,model)=>({id,model,download_gb:4,ram_needed_gb:8,about:'',installed:false});
const family=(id,title,images,sizes)=>({family:id,title,images,sizes,about:'',experimental:false});
test('only the two requested display families exist; engine IDs and per-version capabilities survive grouping',()=>{
 setTranslationLanguage('en');
 const raw=[family('qwen','Qwen3.8-Flash-Next',true,[size('q2_0','Q2_0')]),family('swift','Swift 1.5',false,[size('swift-iq2_xs','IQ2_XS')]),family('coder','Coder',false,[size('coder-iq1_m','IQ1_M')]),family('unsloth','Unsloth',true,[size('unsloth-ud-q4_k_xl','UD-Q4_K_XL')])];
 const groups=nativeCatalogModels(raw);assert.deepEqual(groups.map(item=>item.title),['Qwen3.8-Flash-Next','Qwen3.6-35B-A3B']);assert.equal(groups[1].sizes.length,0);
 assert.deepEqual(groups[0].sizes.map(item=>[item.sourceFamily,item.id,item.images]),[['qwen','q2_0',true],['swift','swift-iq2_xs',false],['coder','coder-iq1_m',false],['unsloth','unsloth-ud-q4_k_xl',true]]);
 assert.equal(catalogVersionLabel(groups[0].sizes[1]),'Swift 1.5 · Compact');
 const newer=nativeCatalogModels([...raw,family('qwen36','Qwen3.6-35B-A3B',true,[size('real-36-q4','Q4_K_M')])]);assert.equal(newer[1].sizes[0].id,'real-36-q4');assert.equal(newer[1].sizes[0].sourceFamily,'qwen36');
 assert.deepEqual(nativeCatalogModels([]),[]);
});

test('the Gyro sizes join the Qwen3.8 group as their own edition, without images',()=>{
 setTranslationLanguage('en');
 const groups=nativeCatalogModels([family('qwen','Qwen3.8-Flash-Next',true,[size('q2_0','Q2_0')]),family('gyro','Qwen3.8-Flash-Next (Gyro)',false,[size('gyro-gyro-s','Gyro-S'),size('gyro-gyro-m','Gyro-M')])]);
 assert.deepEqual(groups[0].sizes.map(item=>[item.sourceFamily,item.model,item.edition,item.images]),[['qwen','Q2_0','',true],['gyro','Gyro-S','Gyro',false],['gyro','Gyro-M','Gyro',false]]);
 assert.equal(catalogVersionLabel(groups[0].sizes[1]),'Gyro · Smallest');assert.equal(catalogVersionLabel(groups[0].sizes[2]),'Gyro · Small memory');
 assert.equal(groups[1].sizes.length,0);
});

test('empty default chat titles follow the interface language while chosen names stay verbatim',()=>{
 setTranslationLanguage('de');const empty={title:'New conversation',messages:[]};assert.notEqual(sessionTitle(empty),'New conversation');assert.equal(sessionTitle({...empty,defaultTitle:false}),'New conversation');assert.equal(sessionTitle({title:'Ready',messages:[]}), 'Ready');assert.equal(sessionTitle({...empty,messages:[{role:'user',content:'hello'}]}),'New conversation');setTranslationLanguage('en');
});
