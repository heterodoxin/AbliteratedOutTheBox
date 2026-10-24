#!/usr/bin/env node
/**
 * Attached pictures reaching a model, end to end on a test box, through the installed app layout (packaged host + bundled Control + an engine
 * component that carries strata-vision):
 *   1. the model catalog marks the picture-capable models (images: true) when the picture reader is installed;
 *   2. API models: a llama-server with a small picture model (--mmproj) is saved as a Custom model API; the host works out that it reads pictures,
 *      and a real picture goes out as image_url and is answered. A server without a picture reader, a model set "not to read pictures", and an API
 *      of another style each get a plain refusal (409) that says how to fix it; ticking "This model reads pictures" lets the picture through;
 *   3. a local model (unless --skip-local): Download and prepare with Read pictures ON, load, and a picture through the app's own chat.
 *
 *   node pics-e2e.mjs --install <node/ host/ control/> --components <engine components> --work <scratch> --image shapes.png \
 *   Also: --silent <url> (a server whose model list says nothing about pictures: ON by default), --agent (the same picture through the agent; needs a Harness)
 *        --vlm http://127.0.0.1:18090/v1 --vlm-model smolvlm --text-only http://127.0.0.1:18091/v1 [--skip-local --family qwen --model IQ3_XXS --data <folder> --device cpu]
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import { createServer } from 'node:net';
import { join, resolve } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]?.startsWith('--') || all[i + 1] === undefined ? true : all[i + 1]]] : acc), []));
for (const need of ['install', 'components', 'work', 'image']) if (!args[need]) { console.error('missing --' + need); process.exit(2); }
const install = resolve(args.install), work = resolve(args.work), components = resolve(args.components);
const want = String(args.match ?? args.model ?? 'IQ3_XXS').toLowerCase(), contextSize = Number(args.context ?? 32768), skipLocal = args['skip-local'] === true, device = String(args.device ?? 'cpu'), family = String(args.family ?? 'qwen'), modelId = String(args.model ?? 'IQ3_XXS');
const node = join(install, 'node', 'node'), hostEntry = join(install, 'host', 'server', 'index.mjs');
const dirs = { models: join(work, 'models'), profiles: join(work, 'profiles'), runtime: join(work, 'runtime'), home: join(work, 'home'), xdg: join(work, 'xdg'), appdata: join(work, 'appdata') };
for (const d of [dirs.models, dirs.home, dirs.xdg, dirs.appdata]) mkdirSync(d, { recursive: true });
const delay = ms => new Promise(r => setTimeout(r, ms));
let failed = false;
const record = (name, ok, detail = '') => { if (!ok) failed = true; console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + String(detail).slice(0, 600) : ''}`); };
const freePort = async () => { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; };
const env = { ...process.env, STRATA_COMPONENTS_DIR: components, STRATA_PROFILE_DIR: dirs.profiles, STRATA_RUNTIME_DIR: dirs.runtime, STRATA_INSTALL_DIR: dirs.models,
  STRATA_HOME: dirs.home, STRATA_MODELS_DIR: join(work, 'models-unused'), APPDATA: dirs.appdata, XDG_CONFIG_HOME: dirs.xdg, PYTHONDONTWRITEBYTECODE: '1' };
delete env.STRATA_API_KEY;
const port = await freePort();
const base = `http://127.0.0.1:${port}`;
const host = spawn(node, [hostEntry], { cwd: join(install, 'host'), stdio: ['ignore', 'pipe', 'pipe'],
  env: { ...env, STRATA_PORT: String(port), STRATA_DIST: join(install, 'host', 'dist'), NODE_ENV: 'production', STRATA_PACKAGED: '1', STRATA_REAL_ENGINE: '1', STRATA_NO_ACQUIRE: '1',
    STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()) } });
let hostLog = '';
host.stdout.on('data', c => { hostLog += c; }); host.stderr.on('data', c => { hostLog += c; });
const call = async (path, method = 'GET', body) => {
  const response = await fetch(base + path, { method, headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: response.status, body: await response.json().catch(() => null) };
};
const png = 'data:image/png;base64,' + readFileSync(resolve(args.image)).toString('base64');
const QUESTION = 'Name each shape in this picture with its colour, and read out any text.';

/** One chat turn through the app's own endpoint: the plain-text answer, or the refusal (status + message). */
async function chat(extra, message = QUESTION) {
  const response = await fetch(base + '/api/chat/stream', { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ message, agent: false, images: [{ url: png, name: 'shapes.png' }], ...extra }) });
  if (response.status !== 200) { const b = await response.json().catch(() => ({})); return { status: response.status, message: b.error ?? b.message ?? '' }; }
  let text = '', error = '';
  for (const line of (await response.text()).split('\n')) {
    if (!line.trim()) continue;
    let f; try { f = JSON.parse(line); } catch { continue; }
    if (f.type === 'error') error = f.message ?? 'error';
    else if (typeof f.text === 'string' && f.type === 'delta') text += f.text;
  }
  return { status: 200, text, error };
}
const saveApi = async (name, baseUrl, models, protocol = 'openai-chat') => {
  const existing = (await call('/api/model-providers')).body.connections.find(c => c.name === name);
  return (await call('/api/model-providers', 'POST', { ...(existing ? { id: existing.id } : {}), kind: 'custom', provider: null, name, baseUrl, protocol, apiKey: '', models })).body;
};
async function readsAfter(connectionName, id, want, seconds = 20) {
  for (let i = 0; i < seconds * 2; i++) {
    const c = (await call('/api/model-providers')).body.connections.find(x => x.name === connectionName);
    const m = c?.models.find(x => x.id === id);
    if (m && m.reads === want) return { c, m };
    await delay(500);
  }
  const c = (await call('/api/model-providers')).body.connections.find(x => x.name === connectionName);
  return { c, m: c?.models.find(x => x.id === id) };
}
async function waitJob(id, seconds) {
  let last = '';
  for (let i = 0; i < seconds * 2; i++) {
    const state = (await call('/api/native')).body;
    const job = state.jobs.find(j => j.id === id);
    if (job && job.summary !== last) { last = job.summary; console.log('    ... ' + String(last).slice(0, 120)); }
    if (job && job.state !== 'running') return { job, state };
    await delay(500);
  }
  throw new Error('operation did not finish');
}
const names = low => /(red|blue|green|circle|square|triangle|strata|42)/.test(low);

async function main() {
  let up = false;
  for (let i = 0; i < 200 && !up; i++) { try { up = (await fetch(base + '/api/health')).ok; } catch { /* not yet */ } if (!up) await delay(150); }
  record('host healthy', up, up ? base : hostLog.slice(-500));
  if (!up) return;

  // ---- 1. the catalog
  const packages = existsSync(join(components, 'engine')) ? readdirSync(join(components, 'engine')) : [];
  const reader = packages.some(p => existsSync(join(components, 'engine', p, 'strata-vision')));
  const native = (await call('/api/native?refresh=1')).body;
  record('the engine component carries the picture reader and the host knows it (visionReader)', reader && native.visionReader === true, `component=${reader} host=${native.visionReader}`);
  const rows = (native.catalog?.families ?? []).flatMap(f => (f.sizes ?? []).map(s => ({ family: f.title ?? f.family, size: s.id, images: s.images, capable: s.imagesCapable })));
  console.log('    catalog (family / size / images / imagesCapable):');
  for (const r of rows) console.log(`      ${String(r.family).padEnd(26)} ${String(r.size).padEnd(30)} images=${r.images} capable=${r.capable}`);
  record('some models in the catalog read pictures', rows.some(r => r.images === true), `${rows.filter(r => r.images).length} of ${rows.length}`);
  record('every picture-capable model reads pictures when the reader is installed', rows.every(r => r.capable === r.images), '');
  const flash = rows.filter(r => /flash-next/i.test(`${r.family} ${r.size}`) && !/gyro|qad/i.test(`${r.family} ${r.size}`));
  record('the Qwen3.8-Flash-Next models are marked images: true', flash.length > 0 && flash.every(r => r.images === true), flash.map(r => r.size + '=' + r.images).join(' '));

  // ---- 2. API models
  if (args.vlm) {
  const smol = args['vlm-model'] ?? 'smolvlm';
  await saveApi('Small picture model', args.vlm, [{ id: smol, name: 'SmolVLM' }]);
  const found = await readsAfter('Small picture model', smol, true);
  record('the host works out that the llama-server reads pictures (no tick needed)', found.m?.reads === true, JSON.stringify(found.m));
  const conn = found.c;
  const sent = await chat({ remoteModel: { connectionId: conn?.id, modelId: smol } });
  console.log(`    answer from the API model: ${JSON.stringify((sent.text ?? sent.message ?? '').slice(0, 300))}`);
  record('the picture goes to the API model and is answered', sent.status === 200 && !sent.error && (sent.text ?? '').trim().length > 0 && names((sent.text ?? '').toLowerCase()), `status=${sent.status} err=${sent.error ?? ''}`);

  if (args['text-only']) {
    await saveApi('Text only server', args['text-only'], [{ id: 'plain', name: 'Plain model' }]);
    const t = await readsAfter('Text only server', 'plain', false);
    record('a server without a picture reader is known not to read pictures', t.m?.reads === false, JSON.stringify(t.m));
    const refused = await chat({ remoteModel: { connectionId: t.c.id, modelId: 'plain' } });
    console.log(`    refusal: ${JSON.stringify(refused.message)}`);
    record('a picture for it is refused in plain words that say how to fix it', refused.status === 409 && /Read pictures with this model/.test(refused.message ?? '') && /Settings/.test(refused.message ?? ''), `${refused.status}`);
    const ticked = await saveApi('Text only server', args['text-only'], [{ id: 'plain', name: 'Plain model', vision: true }]);
    const after = (await call('/api/model-providers')).body.connections.find(x => x.name === 'Text only server');
    record('ticking "Read pictures with this model" is saved and makes the model read pictures', after?.models[0]?.vision === true && after?.models[0]?.reads === true, JSON.stringify(after?.models[0]));
    const through = await chat({ remoteModel: { connectionId: after.id, modelId: 'plain' } });
    record('with the tick the picture is sent on (the server itself decides what to do with it)', through.status === 200 && (through.error ?? '') !== '' ? /turn off/.test(through.error) : through.status === 200, `${through.status} ${through.message ?? through.error ?? ''}`);
  }

  await saveApi('Small picture model', args.vlm, [{ id: smol, name: 'SmolVLM', vision: false }]);
  const off = await readsAfter('Small picture model', smol, false);
  const notRead = await chat({ remoteModel: { connectionId: off.c.id, modelId: smol } });
  record('a model switched OFF refuses pictures and says where to turn them on again', notRead.status === 409 && /not to read pictures/.test(notRead.message ?? '') && /Settings/.test(notRead.message ?? ''), `${notRead.status} ${notRead.message ?? ''}`);
  await saveApi('Small picture model', args.vlm, [{ id: smol, name: 'SmolVLM', vision: null }]);

  // unknown is ON by default: a server whose list and /props say nothing is sent the picture
  if (args.silent) {
    await saveApi('Silent server', args.silent, [{ id: 'quiet', name: 'Quiet model' }]);
    const q = await readsAfter('Silent server', 'quiet', true, 8);
    record('a server that says nothing about pictures is ON by default', q.m?.reads === true && q.m?.vision === null, JSON.stringify(q.m));
    const sentQuiet = await chat({ remoteModel: { connectionId: q.c.id, modelId: 'quiet' } });
    record('and the picture is sent to it (not refused)', sentQuiet.status === 200, `${sentQuiet.status} ${sentQuiet.message ?? sentQuiet.error ?? ''}`);
  }
  await saveApi('Other style', 'https://api.anthropic.com/v1', [{ id: 'claude-x', name: 'Claude' }], 'anthropic');
  const other = (await call('/api/model-providers')).body.connections.find(x => x.name === 'Other style');
  const style = await chat({ remoteModel: { connectionId: other.id, modelId: 'claude-x' } });
  console.log(`    other style: ${JSON.stringify(style.message)}`);
  record('an API of another style says plainly why pictures stay in the preview', style.status === 409 && /OpenAI/.test(style.message ?? ''), `${style.status}`);

  }
  if (skipLocal) return;

  // ---- 3. a local model
  const values = { family, model: modelId, context: contextSize, vision: 'yes', loadVision: device, ...(args.data ? { data_dir: resolve(args.data) } : {}) };
  const started = await call('/api/native/operations', 'POST', { kind: 'install', args: values });
  record('the install operation starts', started.status === 200 || started.status === 202, JSON.stringify(started.body).slice(0, 200));
  const done = await waitJob(started.body.id, 3600);
  record('Download and prepare finishes', done.job.state === 'done', `${done.job.state} ${done.job.error ?? done.job.summary}`);
  const refreshed = (await call('/api/native?refresh=1')).body;
  const model = refreshed.installed?.models?.find(m => [m.id, m.model, m.tag].some(x => String(x ?? '').toLowerCase() === want));
  record('the model is prepared with its picture reader', !!model?.visionPrepared, JSON.stringify({ ready: model?.ready, visionPrepared: model?.visionPrepared }));
  const load = await call('/api/native/operations', 'POST', { kind: 'load', args: { model: model?.model ?? want } });
  const loaded = await waitJob(load.body.id, 900);
  const instance = loaded.state.instances[0];
  record('the model loads with the picture reader', loaded.job.state === 'done' && instance?.status === 'ready' && instance?.vision === true, `${loaded.job.state} ${loaded.job.error ?? ''} vision=${instance?.vision}`);
  record('the model loaded with pictures ON without being asked per load (default)', instance?.vision === true, '');
  const t0 = Date.now();
  let local = { status: 0 };
  for (let i = 0; i < 2 && !(local.text ?? '').trim(); i++) local = await chat({ nativeModelId: instance?.id, maxTokens: 4096 });
  console.log(`    answer from the local model (${Math.round((Date.now() - t0) / 1000)} s): ${JSON.stringify((local.text ?? local.message ?? '').slice(0, 300))}`);
  const low = (local.text ?? '').toLowerCase();
  record('the local model answers about the picture (names the shapes)', local.status === 200 && /red|circle/.test(low) && /blue|square/.test(low) && /green|triangle/.test(low), `status=${local.status} err=${local.error ?? ''}`);
  record('the local model reads STRATA 42', low.includes('42'));
  // the same picture through the agent (the Harness route must declare image input: the "does not declare image input" error of the tester)
  if (args.agent) {   // needs a Harness beside the host (the installed layout has one; a bare test box does not)
    const t1 = Date.now(); let a = { status: 0 };
    for (let i = 0; i < 2 && !/red|circle|square|triangle/.test((a.text ?? '').toLowerCase()); i++) a = await chat({ nativeModelId: instance?.id, agent: true, sessionId: 'pics-agent-' + i, mode: 'plan', maxTokens: 4096 });
    console.log(`    answer through the agent (${Math.round((Date.now() - t1) / 1000)} s): ${JSON.stringify((a.text ?? a.message ?? '').slice(0, 300))}`);
    record('the agent route reads the picture too', a.status === 200 && /red|circle|square|triangle/.test((a.text ?? '').toLowerCase()) && !/does not declare image input/.test(a.text ?? ''), `status=${a.status} err=${a.error ?? ''}`);
  }
  // OFF: loading with Read pictures off refuses a picture in plain words and says what to do
  await call(`/api/native/instances/${instance?.id}/unload`, 'POST', {});
  const loadOff = await call('/api/native/operations', 'POST', { kind: 'load', args: { model: model?.model ?? want, vision: 'no' } });
  const loadedOff = await waitJob(loadOff.body.id, 900);
  const off2 = loadedOff.state.instances[0];
  record('loaded with Read pictures OFF', loadedOff.job.state === 'done' && off2?.vision === false, `${loadedOff.job.state} vision=${off2?.vision}`);
  const refusedLocal = await chat({ nativeModelId: off2?.id, maxTokens: 64 });
  console.log(`    refusal: ${JSON.stringify(refusedLocal.message)}`);
  record('a picture for the OFF model is refused with a next step', refusedLocal.status === 409 && /Read pictures/.test(refusedLocal.message ?? '') && /Models/.test(refusedLocal.message ?? ''), `${refusedLocal.status}`);
  const unloaded = await call(`/api/native/instances/${off2?.id}/unload`, 'POST', {});
  record('unload', unloaded.status === 200, unloaded.status);
}
try { await main(); } catch (e) { record('run', false, e.stack ?? e); }
host.kill();
await delay(1500);
console.log(failed ? 'PICS-E2E FAIL' : 'PICS-E2E PASS');
process.exit(failed ? 1 : 0);
