import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, mkdtemp, realpath, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, join, resolve, sep } from 'node:path';
import { externalEnvironment } from './external-env.mjs';
const run=(command,args)=>promisify(execFile)(command,args,{env:externalEnvironment()});   // xdg-open and the apps it starts: the environment the AppImage was started from
const extensions=new Set(['.md','.txt','.pdf','.html','.htm','.png','.jpg','.jpeg','.webp','.gif','.svg','.csv','.ts','.tsx','.js','.jsx','.py','.json','.css','.rs','.go','.sh','.mp4','.webm']);
export async function openArtifact(body,{examplesDir,previewsDir,platform=process.platform,launch=run}) {
  if(!['open','reveal'].includes(body.action))throw new Error('Choose Open or Show in file explorer.');
  let file;
  if(typeof body.path==='string') {
    // Only public example assets can be addressed by path. Never accept host paths.
    if(!body.path.startsWith('/examples/'))throw new Error('This file is not a local preview.');
    const base=await realpath(examplesDir);
    file=await realpath(resolve(base,body.path.slice('/examples/'.length)));
    if(!file.startsWith(base+sep)||!(await stat(file)).isFile())throw new Error('This file is not a local preview.');
  } else {
    const name=basename(String(body.name??'document.txt')).replace(/[^\w. -]/g,'_').slice(0,120);
    if(!extensions.has(extname(name).toLowerCase()))throw new Error('This file type cannot be opened from a preview.');
    if(typeof body.content!=='string'&&typeof body.base64!=='string')throw new Error('No file content was supplied.');
    const bytes=typeof body.base64==='string'?Buffer.from(body.base64,'base64'):Buffer.from(body.content,'utf8');
    if(bytes.length>20*1024*1024)throw new Error('Choose a file smaller than 20 MB.');
    await mkdir(previewsDir,{recursive:true});const folder=await mkdtemp(join(previewsDir,'preview-'));
    file=join(folder,name);await writeFile(file,bytes,{flag:'wx',mode:0o600});
  }
  if(!extensions.has(extname(file).toLowerCase()))throw new Error('This file type cannot be opened from a preview.');
  if(platform==='darwin')await launch('open',body.action==='reveal'?['-R',file]:[file]);
  else if(platform==='win32')await launch('explorer.exe',body.action==='reveal'?[`/select,${file}`]:[file]);
  else await launch('xdg-open',[body.action==='reveal'?dirname(file):file]);
  return {ok:true};
}
