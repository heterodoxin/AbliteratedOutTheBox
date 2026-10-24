import { useMemo } from 'react';
import { codeLanguage, highlightedLines } from '../lib/syntax';
export function SyntaxCode({text,language='',filename,lineNumbers=false,className=''}:{text:string;language?:string;filename?:string;lineNumbers?:boolean;className?:string}){
 const resolved=filename?codeLanguage(filename,true):codeLanguage(language);
 const lines=useMemo(()=>highlightedLines(text,resolved),[text,resolved]);
 return <pre className={`syntax-code ${lineNumbers?'code-preview':''} ${className}`} data-language={resolved}><code>{lines.map((line,i)=><span className="syntax-line" key={i}>{lineNumbers?<span className="syntax-gutter" aria-hidden="true">{i+1}</span>:null}<span className="syntax-text">{line.map((part,j)=>part.classes.length?<span className={part.classes.map(c=>`syntax-${c}`).join(' ')} key={j}>{part.text}</span>:part.text)}{!lineNumbers&&i<lines.length-1?'\n':''}</span></span>)}</code></pre>;
}
