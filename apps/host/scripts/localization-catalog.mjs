import ts from 'typescript';
import {readdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
// One-time migration: stable IDs must never be regenerated after translations exist.
if(await readFile('src/locales/en.json','utf8').catch(()=>''))throw new Error('The catalog is established. Use node scripts/append-localization.mjs to preserve translation IDs.');
const keys=new Set();
function add(value){value=value.replace(/\s+/g,' ').trim();if(value.length>1&&/[A-Za-z]{2}/.test(value)&&!/^([.#/]|https?:|data:|[\w.-]+\.(json|mjs|tsx?|py|html|svg|png|gguf)\b|\(max-width|\(prefers-)/.test(value)&&!/[{}<>;=]/.test(value)&&!/^var\(/.test(value)&&!/(?:\bst-|\bnative-|\bonboard-|\bmonitor-|\brun-|\bsettings-|\bts-|__)/.test(value))keys.add(value);}
function clean(text){const lines=text.replace(/\r/g,'').split('\n');let result='';for(let i=0;i<lines.length;i++){let line=lines[i].replace(/\t/g,' ');if(i)line=line.replace(/^ +/,'');if(i<lines.length-1)line=line.replace(/ +$/,'');if(line){result+=line;if(i<lines.length-1)result+=' ';}}return result;}
function visit(node){
 if(ts.isJsxText(node))add(clean(node.text));
 if(ts.isStringLiteral(node)||ts.isNoSubstitutionTemplateLiteral(node)){
  const p=node.parent;
  if(ts.isJsxAttribute(p)&&! /^(title|label|hint|description|placeholder|alt|ariaLabel|aria-label|aria-description)$/.test(p.name.getText()))return;
  if(ts.isPropertyAssignment(p)&&/^(className|id|key|aliases|path|url|href|args|command|value|type|style)$/.test(p.name.getText().replace(/['"]/g,'')))return;
  if(node.getSourceFile().fileName.startsWith('server/')&&! (ts.isPropertyAssignment(p)&&/^(label|hint|title|detail|description|about|message|phaseLabel)$/.test(p.name.getText().replace(/['"]/g,''))||ts.isNewExpression(p)&&/Error$/.test(p.expression.getText())||ts.isCallExpression(p)&&p.expression.getText()==='error'))return;
  if(ts.isJsxAttribute(p)&&/^(title|label|hint|description|placeholder|alt|ariaLabel|aria-label|aria-description)$/.test(p.name.getText()))add(node.text);
  else if(ts.isPropertyAssignment(p)&&/^(label|hint|title|detail|description|about|name|section|heading|action|emptyLabel|phaseLabel)$/.test(p.name.getText().replace(/['"]/g,'')))add(node.text);
  else if(/[A-Za-z][ ,!?…:–—][A-Za-z ]/.test(node.text)&&!/[\n\t]/.test(node.text))add(node.text);
  else if(ts.isReturnStatement(p)||ts.isConditionalExpression(p)||ts.isJsxExpression(p))add(node.text);
 }
 ts.forEachChild(node,visit);
}
async function scan(dir){for(const entry of await readdir(dir,{withFileTypes:true})){const path=join(dir,entry.name);if(entry.isDirectory())await scan(path);else if(/\.(tsx?|mjs)$/.test(path)&&!path.includes('/locales/')&&!path.includes('chat-showcase')&&!path.includes('localization-catalog')){const source=await readFile(path,'utf8');const sf=ts.createSourceFile(path,source,ts.ScriptTarget.Latest,true,path.endsWith('.tsx')?ts.ScriptKind.TSX:ts.ScriptKind.TS);visit(sf);}}}
await scan('src');
await scan('server');
const extra=['Strata font','Font','Choose an installed font for the interface and conversations.','Use a font from your computer','Choose a system font','Font family','Installed fonts','Enter a font family name','Apply font','Language','Choose the language used throughout Strata.','English','Chinese','Japanese','Deutsch','Spanish','Russian','Portuguese','Korean','Loading installed fonts…','Your browser cannot list installed fonts. Enter the name of a font installed on your computer.','This model needs a newer engine.','Update your engine or import a local GGUF for this model.','Update engine','Swift 1.5','Coder','Unsloth','No GPU readings yet','No recorded speed samples yet','Expert settings','Ready','Available','Downloaded','Fastest','Compact','Balanced','High quality','Coding','Automatic','System','Appearance','Models','Storage','Engine','About','General','Reading & focus','Update','Privacy','Time-skip','Goal','Loop','Subagents','Welcome','Your model','Prepare','Make it yours'];extra.forEach(add);
// Template keys retain numbered interpolation slots for grammar and word order.
async function templates(dir){for(const entry of await readdir(dir,{withFileTypes:true})){const path=join(dir,entry.name);if(entry.isDirectory()){if(entry.name!=='locales')await templates(path);}else if(/\.(tsx?|mjs)$/.test(path)){const source=await readFile(path,'utf8'),sf=ts.createSourceFile(path,source,ts.ScriptTarget.Latest,true,path.endsWith('.tsx')?ts.ScriptKind.TSX:ts.ScriptKind.TS);const walk=node=>{if(ts.isTemplateExpression(node)){let key=node.head.text;node.templateSpans.forEach((span,i)=>key+=`{${i}}${span.literal.text}`);const plain=key.replace(/\{\d+\}/g,'');if(/[A-Za-z]{2} /.test(plain)&&!/[<>\n\t]/.test(plain)&&!/^https?:|^[.#/]|^var\(/.test(plain))keys.add(key.trim());}ts.forEachChild(node,walk);};walk(sf);}}}
// Remove the temporary placeholders used by the broad literal pass.
for(const key of keys)if(key.includes('PLACEHOLDER'))keys.delete(key);
await templates('src');
const result=Object.fromEntries([...keys].sort().map((key,index)=>[`s${String(index+1).padStart(4,'0')}`,key]));
await writeFile('src/locales/en.json',JSON.stringify(result,null,2)+'\n');
console.log(`${keys.size} interface strings`);
