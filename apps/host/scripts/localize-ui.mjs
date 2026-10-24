// One-time migration retained for reference. Do not run it on localized source.
throw new Error("Localization migration already applied. Add explicit tr() calls for new interface copy; preserve user content.");
import ts from 'typescript';
import {readdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
const attrs=new Set(['title','label','hint','description','placeholder','alt','ariaLabel','aria-label','aria-description','emptyLabel','action']);
const printer=ts.createPrinter({newLine:ts.NewLineKind.LineFeed});
const f=ts.factory;
function text(value){const lines=value.replace(/\r/g,'').split('\n');return lines.map((line,i)=>{line=line.replace(/\t/g,' ');if(i)line=line.replace(/^ +/,'');if(i<lines.length-1)line=line.replace(/ +$/,'');return line;}).filter(Boolean).join(' ');}
function excluded(node){for(let p=node.parent;p;p=p.parent){if(ts.isJsxElement(p)){const name=p.openingElement.tagName.getText();if(['pre','code','textarea','Markdown','MarkdownContent','Code','Syntax','Mermaid'].includes(name))return true;}}return false;}
function translate(expression){
 if(ts.isCallExpression(expression)&&expression.expression.getText?.()==='tr')return expression;
 if(ts.isTemplateExpression(expression)){let key=expression.head.text;const values=expression.templateSpans.map((span,i)=>{key+=`{${i}}${span.literal.text}`;return span.expression;});return f.createCallExpression(f.createIdentifier('tr'),undefined,[f.createStringLiteral(key),f.createArrayLiteralExpression(values)]);}
 return f.createCallExpression(f.createIdentifier('tr'),undefined,[expression]);
}
async function scan(dir){for(const entry of await readdir(dir,{withFileTypes:true})){const path=join(dir,entry.name);if(entry.isDirectory())await scan(path);else if(path.endsWith('.tsx')&&!path.endsWith('/main.tsx')){
 const source=await readFile(path,'utf8'),sf=ts.createSourceFile(path,source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);let changed=false;
 const result=ts.transform(sf,[context=>root=>{const visit=node=>{
  if(ts.isJsxText(node)&&!excluded(node)&&/[A-Za-z]/.test(node.text)){changed=true;return f.createJsxExpression(undefined,translate(f.createStringLiteral(text(node.text))));}
  if(ts.isJsxAttribute(node)&&attrs.has(node.name.getText())&&node.initializer&&ts.isStringLiteral(node.initializer)){changed=true;return f.updateJsxAttribute(node,node.name,f.createJsxExpression(undefined,translate(node.initializer)));}
  if(ts.isJsxExpression(node)&&node.expression){const p=node.parent;const isTextAttribute=ts.isJsxAttribute(p)&&attrs.has(p.name.getText());const isChild=ts.isJsxElement(p)||ts.isJsxFragment(p);const visited=ts.visitEachChild(node,visit,context);if((isTextAttribute||isChild)&&!excluded(node)&&! /(?:message\.content|message\.reasoning|msg\.content|session\.title|\.path\b|\.code\b|\.markdown\b|\.src\b)/.test(node.expression.getText())){changed=true;return f.updateJsxExpression(visited,translate(visited.expression));}return visited;}
  return ts.visitEachChild(node,visit,context);
 };return ts.visitNode(root,visit);}]);
 if(changed){let output=printer.printFile(result.transformed[0]);if(!/import\s*\{[^}]*\btr\b[^}]*\}\s*from/.test(source)){const relative=path.startsWith('src/lib/')?'./i18n':'../lib/i18n';output=`import { tr } from '${relative}';\n`+output;}await writeFile(path,output);}
 result.dispose();
 }} }
await scan('src');
