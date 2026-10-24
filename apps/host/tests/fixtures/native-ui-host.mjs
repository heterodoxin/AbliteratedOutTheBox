/** Disposable UI fixture. All native operations are mocked; no weights or external programs run. */
import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve,extname} from 'node:path';
import {nativeFixture} from '../helpers/native-fixture.mjs';
const fixture=nativeFixture(),control=fixture.control,root=resolve(import.meta.dirname,'../../dist'),upstream='http://127.0.0.1:5161';
await control.refresh(true);
const server=http.createServer(async(req,res)=>{try{
 const path=new URL(req.url,'http://fixture').pathname;let text='';if(req.method!=='GET')for await(const chunk of req)text+=chunk;const body=text?JSON.parse(text):{};
 const json=value=>{res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify(value));};
 if(path==='/api/native'&&req.method==='GET')return json(await control.refresh());
 if(path==='/api/native/plan')return json(await control.plan(body));
 if(path==='/api/native/operations')return json(control.operation(body.kind,body.args));
 if(path==='/api/native/benchmark')return json(control.benchmarkReport());
 if(path==='/api/native/logs')return json({logs:control.logs()});
 let match=/^\/api\/native\/models\/([^/]+)\/settings$/.exec(path);if(match)return json(req.method==='GET'?control.settings(match[1]):control.save(match[1],body));
 match=/^\/api\/native\/instances\/([^/]+)\/(default|unload)$/.exec(path);if(match)return json(match[2]==='default'?control.setDefault(match[1]):await control.unload(match[1]));
 match=/^\/api\/native\/operations\/([^/]+)\/cancel$/.exec(path);if(match)return json(await control.cancel(match[1]));
 if(path==='/api/hardware'){const value=await(await fetch(upstream+path)).json();return json({...value,gpus:[{index:0,id:'gpu-0',vendor:'nvidia',name:'NVIDIA RTX 4090',vramGb:24,backend:'cuda',unified:false},{index:1,id:'gpu-1',vendor:'nvidia',name:'NVIDIA RTX 3090',vramGb:24,backend:'cuda',unified:false}],memory:{totalGb:128,usedGb:48,freeGb:80,pressure:'low'}});}
 if(path.startsWith('/api/')){const response=await fetch(upstream+req.url,{method:req.method,headers:text?{'content-type':'application/json'}:{},body:text||undefined});res.writeHead(response.status,{'content-type':response.headers.get('content-type')??'application/json'});for await(const chunk of response.body)res.write(chunk);res.end();return;}
 const file=path==='/'?'/index.html':path;if(file.includes('..'))throw new Error('Invalid fixture path');const content=await readFile(root+file);res.writeHead(200,{'content-type':({'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.woff2':'font/woff2','.png':'image/png'})[extname(file)]??'application/octet-stream'});res.end(content);
 }catch(e){res.writeHead(400,{'content-type':'application/json'});res.end(JSON.stringify({error:e.message}));}});
server.listen(5270,'127.0.0.1',()=>console.log('Disposable native UI fixture: http://127.0.0.1:5270'));
let stopping=false;async function stop(){if(stopping)return;stopping=true;server.closeAllConnections();server.close();await fixture.cleanup();process.exit(0);}process.on('SIGTERM',stop);process.on('SIGINT',stop);
