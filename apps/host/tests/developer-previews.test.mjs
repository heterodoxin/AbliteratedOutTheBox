import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import { build, transform } from 'esbuild';
import { readFile } from 'node:fs/promises';
const {outputFiles}=await build({entryPoints:[fileURLToPath(new URL('../src/lib/syntax.ts',import.meta.url))],bundle:true,platform:'node',format:'esm',write:false});
const {codeLanguage,highlightedLines}=await import(`data:text/javascript;base64,${Buffer.from(outputFiles[0].text).toString('base64')}`);
const source=await transform(await readFile(new URL('../src/lib/preview-layout.ts',import.meta.url),'utf8'),{loader:'ts',format:'esm'});
const {previewWidth,pointerPreviewFraction,DEFAULT_PREVIEW_FRACTION}=await import(`data:text/javascript;base64,${Buffer.from(source.code).toString('base64')}`);
const reconstruct=lines=>lines.map(line=>line.map(token=>token.text).join('')).join('\n');

test('TS, Python and JSX/TSX grammars preserve source and produce language-aware tokens',()=>{
 for(const [language,code,expected] of [
  ['typescript','export interface Note { title: string; pinned: boolean; }\nconst x = "hello";',['keyword','string']],
  ['python','# a comment\ndef hello(name):\n    return f"Hi {name}"\n',['comment','keyword']],
  ['tsx','type Props = {name: string};\nconst Card = ({name}: Props) => <section className="card">{name}</section>;',['keyword','tag','attr-name']],
  ['jsx','const Card = () => <div title="Hello">{42}</div>;',['tag','number']],
  ['markdown','# Heading\n\n**Bold** and `code`\n\n```python\nprint("Hi")\n```',['title','bold']],
 ]){const lines=highlightedLines(code,language);assert.equal(reconstruct(lines),code);for(const kind of expected)assert.ok(lines.flat().some(t=>t.classes.includes(kind)),`${language} has ${kind}`);}
});
test('multiline comments, empty lines, indentation, unsafe markup and large/unknown sources retain exact text',()=>{
 const text='/* multiline\n * comment */\n\n\tconst value = `<script>alert("unsafe")</script>`;\n';
 const lines=highlightedLines(text,'typescript');assert.equal(reconstruct(lines),text);assert.ok(lines[0].some(t=>t.classes.includes('comment')));assert.ok(lines[1].some(t=>t.classes.includes('comment')));assert.equal(lines[2].length,0);
 assert.equal(reconstruct(highlightedLines('<img src=x onerror=alert(1)>','unknown')), '<img src=x onerror=alert(1)>');
 const large='let text = "keep";\n'.repeat(10000);const fallback=highlightedLines(large,'typescript');assert.equal(reconstruct(fallback),large);assert.ok(fallback.flat().every(t=>!t.classes.length));
});
test('language aliases use extensions and fenced labels without executing source',()=>{
 assert.equal(codeLanguage('note-card.ts',true),'typescript');assert.equal(codeLanguage('component.TSX?version=1',true),'tsx');assert.equal(codeLanguage('tool.py',true),'python');assert.equal(codeLanguage('jsx title="Card"'),'jsx');assert.equal(codeLanguage('file.unknown',true),'plain');
});
test('preview drag and keyboard widths preserve room for both columns and resize with the window',()=>{
 assert.equal(previewWidth(1400,0),320);assert.equal(previewWidth(1400,1),1040);assert.equal(previewWidth(1400,.5),700);
 const fraction=pointerPreviewFraction(1400,1600,900);assert.equal(fraction,.5);assert.equal(previewWidth(1200,fraction),600);
 assert.equal(previewWidth(1400,Infinity),previewWidth(1400,DEFAULT_PREVIEW_FRACTION));assert.equal(previewWidth(0,.5),0);
 assert.equal(pointerPreviewFraction(1400,1600,-999),1040/1400);assert.equal(pointerPreviewFraction(1400,1600,9999),320/1400);
});
