/** Disposable API-edge fixtures for manual CUA review. Never used by the app host.
 * STRATA_FIXTURE_FILE=/tmp/...json node tests/ui-fixture-host.mjs
 * Proxies the isolated demo host at 8794; serves the same built UI at 8796.
 */
import http from 'node:http';
import { readFileSync } from 'node:fs';
const file = process.env.STRATA_FIXTURE_FILE;
const upstreamUrl = process.env.STRATA_FIXTURE_UPSTREAM ?? 'http://127.0.0.1:8794';
const port = Number(process.env.STRATA_FIXTURE_PORT ?? 8796);
if (!file) throw new Error('Provide a disposable fixture JSON path.');
const read = () => JSON.parse(readFileSync(file, 'utf8'));
let failed = false;
function transform(path, value) {
  const { scenario, responses = {} } = read();
  if(Object.hasOwn(responses,path)) return responses[path];
  if (path === '/api/hardware' && scenario === 'unknown-hardware') return null;
  if (path === '/api/serving' && ['queued', 'unknown-status'].includes(scenario)) return { ...value, stats: { ...value.stats, running: scenario === 'queued' ? 1 : null, waiting: scenario === 'queued' ? 2 : null } };
  if (path === '/api/models') {
    if (scenario === 'missing-evidence') return { ...value, mode:'real' };
    if (scenario === 'large-catalog') {
      const original = value.variants[0];
      const variants = Array.from({length:1200}, (_,i) => ({...original, id:`fixture-${i}`, family:`A deliberately long model family name / multilingual research preview release ${Math.floor(i/20)} / 0.5B-Instruct`, quant:`Q${i%20}_Fixture_Quantization_With_A_Long_Exact_Name`, state:'available', installedAt:undefined, catalog:{...original.catalog, family:`Fixture family ${Math.floor(i/100)}`, version:`Release ${Math.floor(i/20)}`, balancedRank:i}}));
      return {...value, loaded:null, variants};
    }
    if (scenario === 'failed-load' && failed) return {...value, loaded:null, variants:value.variants.map(v => v.id === 'qwen38-q4_k_m' ? {...v,state:'error',error:'Fixture: model load failed; installed files retained.'} : v)};
  }
  return value;
}
http.createServer(async (req, res) => {
  try {
    const path = new URL(req.url,'http://localhost').pathname;
    const fixture = read();
    if(fixture.errors?.[path]) { res.writeHead(503,{'content-type':'application/json'});res.end(JSON.stringify({error:fixture.errors[path]}));return; }
    if(!['GET','HEAD'].includes(req.method) && fixture.readOnly) {res.writeHead(400,{'content-type':'application/json'});res.end(JSON.stringify({error:'Fixture: this review action is intentionally unavailable.'}));return;}
    if (path.endsWith('/load') && req.method === 'POST' && read().scenario === 'failed-load') {
      failed=true; const lib=await (await fetch(upstreamUrl+'/api/models')).json();
      res.writeHead(200,{'content-type':'application/json'}); res.end(JSON.stringify(transform('/api/models',lib))); return;
    }
    if (read().scenario !== 'failed-load') failed=false;
    const chunks=[]; for await (const chunk of req) chunks.push(chunk);
    const upstream = await fetch(upstreamUrl+req.url,{method:req.method,headers:req.headers['content-type'] ? {'content-type':req.headers['content-type']} : {},body:chunks.length ? Buffer.concat(chunks) : undefined});
    if (path === '/api/events') {
      res.writeHead(200,{'content-type':'text/event-stream','cache-control':'no-cache'});
      const reader=upstream.body.getReader(); const decoder=new TextDecoder(); let buffer='',event='';
      req.on('close',()=>void reader.cancel());
      while(true) {
        const {value,done}=await reader.read(); if(done)break;
        buffer+=decoder.decode(value,{stream:true});
        let pos; while((pos=buffer.indexOf('\n'))>=0) {
          let line=buffer.slice(0,pos); buffer=buffer.slice(pos+1);
          if(line.startsWith('event:'))event=line.slice(6).trim();
          if(line.startsWith('data:')) { try { line='data: '+JSON.stringify(transform('/api/'+event,JSON.parse(line.slice(5)))); } catch {} }
          res.write(line+'\n');
        }
      }
      res.end(); return;
    }
    const contentType=upstream.headers.get('content-type') ?? 'application/octet-stream';
    res.writeHead(upstream.status,{'content-type':contentType});
    if (contentType.includes('application/json')) res.end(JSON.stringify(transform(path,await upstream.json())));
    else res.end(Buffer.from(await upstream.arrayBuffer()));
  } catch(error) { if(!res.headersSent)res.writeHead(500);res.end(String(error)); }
}).listen(port,'127.0.0.1',()=>console.log(`Disposable CUA fixture host: http://127.0.0.1:${port}`));
