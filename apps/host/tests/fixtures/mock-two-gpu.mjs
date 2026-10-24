/** UI check fixture (not a test): the native fixture with one model loaded on GPU 2, every other API answers empty. Serves on 5271. */
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { nativeFixture, finishJob } from '../helpers/native-fixture.mjs';
const fixture = nativeFixture(), control = fixture.control, root = resolve(import.meta.dirname, '../../dist');
const opts = [{ key: 'c0', cards: [0], label: 'RTX 4090 + RAM', verdict: 'fits', words: 'fits: needs 40 GB of the 152 GB available.', notes: [], recommended: true }, { key: 'c1', cards: [1], label: 'RTX 3090 + RAM', verdict: 'fits', words: 'fits: needs 40 GB of the 152 GB available.', notes: [], recommended: false }];
for (const s of fixture.catalog.families.flatMap(f => f.sizes)) s.gpu_options = { options: opts, recommended: 'c0', ignored: [], cards: 2 };
await control.refresh(true);
await finishJob(control, control.operation('load', { model: 'iq2_xs', gpus: [1], multiple: true }).id);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png', '.woff2': 'font/woff2', '.json': 'application/json' };
http.createServer(async (req, res) => {
  const path = new URL(req.url, 'http://x').pathname;
  const json = v => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(v)); };
  try {
    if (path === '/api/native') return json(await control.refresh());
    if (path === '/api/hardware') return json({ gpus: [{ index: 0, id: 'g0', vendor: 'nvidia', name: 'RTX 4090', vramGb: 24, backend: 'cuda' }, { index: 1, id: 'g1', vendor: 'nvidia', name: 'RTX 3090', vramGb: 24, backend: 'cuda' }], memory: { totalGb: 128 } });
    if (path.startsWith('/api/')) return json({});
    const file = path === '/' ? '/index.html' : path;
    const body = await readFile(root + file); res.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream' }); res.end(body);
  } catch { res.writeHead(404); res.end(); }
}).listen(5271, '127.0.0.1', () => console.log('mock on 5271'));
