/** Bundled grammars: previews work offline and highlighted text stays plain React text. */
import Prism from 'prismjs';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-jsx';
import 'prismjs/components/prism-tsx';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-yaml';
import 'prismjs/components/prism-rust';
import 'prismjs/components/prism-go';
import 'prismjs/components/prism-java';
import 'prismjs/components/prism-c';
import 'prismjs/components/prism-cpp';
import 'prismjs/components/prism-csharp';
import 'prismjs/components/prism-sql';
Prism.manual = true;
const aliases:Record<string,string>={ts:'typescript',tsx:'tsx',mts:'typescript',cts:'typescript',js:'javascript',jsx:'jsx',mjs:'javascript',cjs:'javascript',py:'python',pyw:'python',md:'markdown',mdx:'markdown',markdown:'markdown',html:'markup',htm:'markup',xml:'markup',svg:'markup',json:'json',css:'css',sh:'bash',shell:'bash',bash:'bash',zsh:'bash',yml:'yaml',yaml:'yaml',rs:'rust',go:'go',java:'java',c:'c',h:'c',cc:'cpp',cpp:'cpp',cxx:'cpp',hpp:'cpp',cs:'csharp',csharp:'csharp',sql:'sql',typescript:'typescript',javascript:'javascript',python:'python',rust:'rust'};
export function codeLanguage(value:string,filename=false){
 const input=filename?(value.split(/[?#]/)[0].split('.').at(-1)??''):value.trim().split(/\s/)[0].toLowerCase();
 return aliases[input.toLowerCase()]??'plain';
}
export interface CodeToken {text:string;classes:string[]}
export function highlightedLines(text:string,language:string):CodeToken[][]{
 const lines:CodeToken[][]=[[]];
 function append(value:string,classes:string[]){const parts=value.split('\n');for(let i=0;i<parts.length;i++){if(i)lines.push([]);if(parts[i])lines.at(-1)!.push({text:parts[i],classes});}}
 function walk(tokens:Prism.TokenStream,classes:string[]=[]){
  if(typeof tokens==='string'){append(tokens,classes);return;}
  if(Array.isArray(tokens)){tokens.forEach(token=>walk(token,classes));return;}
  const aliases=typeof tokens.alias==='string'?[tokens.alias]:tokens.alias??[];
  walk(tokens.content,[...classes,tokens.type,...aliases]);
 }
 const grammar=Prism.languages[language];
 if(!grammar||text.length>120000){append(text,[]);return lines;}
 try{walk(Prism.tokenize(text,grammar));}catch{lines.splice(0,lines.length,[]);append(text,[]);}
 return lines;
}
