/** Disposable UI fixture for the Tune model dialog: the real native control over fake models, and a pretend `strata` command that
 * reports progress and a result after a few seconds. No weights, no engine. Needs the demo host on 5161 and a built dist. */
import http from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { writeFileSync } from 'node:fs';
import { resolve, extname } from 'node:path';
import { nativeFixture } from '../helpers/native-fixture.mjs';

const arg = (a, n) => { const i = a.indexOf(n); return i < 0 ? undefined : a[i + 1]; };
let saved = [];
const slot = 'fixture-slot';
function status() {
  const active = saved.find(p => p.active) ?? null;
  return { ok: true, slot, suggested_name: 'IQ2_XS 32K – Work', active, state: active ? 'soft' : null, reasons: active ? ['the graphics driver changed'] : [], presets: saved, can_undo: saved.length > 0 };
}
const stages = [[0, 'Measuring your current settings', 9], [2000, 'Trying: Run the control thread on the last CPU core (round 1 of 2)', 7], [4000, 'Trying: Pin memory only when needed (round 2 of 2)', 3]];
const handler = (child, args) => {
  const cmd = args.slice(args.findIndex(a => String(a).endsWith('strata.py')) + 1);
  const out = json => { child.stdout.emit('data', Buffer.from(JSON.stringify(json))); child.exitCode = 0; child.emit('exit', 0); child.emit('close', 0); };
  if (cmd[0] === 'presets') {
    if (cmd[1] === 'status') return out(status());
    if (cmd[1] === 'list') return out({ ok: true, presets: saved });
    const id = cmd[2];
    if (cmd[1] === 'apply') saved = saved.map(p => ({ ...p, active: p.id === id }));
    if (cmd[1] === 'delete') saved = saved.filter(p => p.id !== id);
    if (cmd[1] === 'rename') saved = saved.map(p => p.id === id ? { ...p, name: cmd[3] } : p);
    if (cmd[1] === 'undo') saved = saved.slice(1);
    return out({ ok: true });
  }
  if (cmd[0] === 'tune') {
    const sf = arg(cmd, '--status-file');
    stages.forEach(([at, step, left], i) => setTimeout(() => { try { writeFileSync(sf, JSON.stringify({ state: 'running', stage: 'rounds', text: `Finding the fastest settings for your PC… about ${left} min left`, minutes_left: left, runs_done: i * 2 + 1, runs_planned: 8, message: step })); } catch { /* gone */ } }, at));
    setTimeout(() => {
      const preset = { id: 'c'.repeat(32), name: 'IQ2_XS 32K – Work', active: true, gain_pct: 11.6, answers: { class: 'exact', fields: [] } };
      saved = [preset, ...saved.map(p => ({ ...p, active: false }))];
      out({ ok: true, result: { outcome: 'improved', gain_pct: 11.6, seconds: 1100, preset, answers: { class: 'exact', fields: [] }, base: { decode_tok_s: 54.2 }, chosen: { decode_tok_s: 60.5 }, pairs: 2, faster_pairs: 2 } });
    }, 7000);
  }
};
const fixture = nativeFixture({ tuneHandler: handler });
const control = fixture.control, root = resolve(import.meta.dirname, '../../dist'), upstream = 'http://127.0.0.1:5161';
await control.refresh(true);
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://fixture'), path = url.pathname;
    let text = ''; if (req.method !== 'GET') for await (const chunk of req) text += chunk;
    const body = text ? JSON.parse(text) : {};
    const json = value => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(value)); };
    if (path === '/__fixture/capture' && req.method === 'POST') { await writeFile(process.env.TUNE_SHOT_DIR + '/' + body.name + '.png', Buffer.from(body.data, 'base64')); return json({ ok: true }); }
    if (path === '/api/native' && req.method === 'GET') return json(await control.refresh());
    if (path === '/api/native/operations') return json(control.operation(body.kind, body.args));
    if (path === '/api/native/tune/status') return json(await control.tuneStatus({ model: url.searchParams.get('model'), resourceProfile: url.searchParams.get('resourceProfile') || undefined, vision: url.searchParams.get('vision') || undefined }));
    if (path === '/api/native/tune/undo') return json(await control.tunePresets.undoFor(body));
    let m = /^\/api\/native\/tune\/presets\/([0-9a-f]+)\/(apply|rename|delete|undo)$/.exec(path); if (m) return json(await control.tunePresets.action(m[2], m[1], body));
    m = /^\/api\/native\/operations\/([^/]+)\/cancel$/.exec(path); if (m) return json(await control.cancel(m[1]));
    if (path === '/api/native/benchmark') return json(control.benchmarkReport());
    if (path === '/api/hardware') { const v = await (await fetch(upstream + path)).json(); return json({ ...v, gpus: [{ index: 0, id: 'gpu-0', vendor: 'nvidia', name: 'NVIDIA RTX 4090', vramGb: 24, backend: 'cuda', unified: false }] }); }
    if (path.startsWith('/api/')) { const r = await fetch(upstream + req.url, { method: req.method, headers: text ? { 'content-type': 'application/json' } : {}, body: text || undefined }); res.writeHead(r.status, { 'content-type': r.headers.get('content-type') ?? 'application/json' }); res.end(Buffer.from(await r.arrayBuffer())); return; }
    const file = path === '/' ? '/index.html' : path; if (file.includes('..')) throw new Error('Invalid fixture path');
    const content = await readFile(root + file);
    res.writeHead(200, { 'content-type': ({ '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.png': 'image/png' })[extname(file)] ?? 'application/octet-stream' }); res.end(content);
  } catch (e) { res.writeHead(400, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: e.message })); }
});
server.listen(Number(process.env.TUNE_UI_PORT ?? 5270), '127.0.0.1', () => console.log('Tune UI fixture ready'));
let stopping = false; async function stop() { if (stopping) return; stopping = true; server.closeAllConnections(); server.close(); await fixture.cleanup(); process.exit(0); }
process.on('SIGTERM', stop); process.on('SIGINT', stop);
