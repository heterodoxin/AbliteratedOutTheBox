#!/usr/bin/env node
/**
 * "Use your own GGUF" end to end on a box, through the installed app layout (packaged host + bundled Control + a real engine component):
 *   preview a folder of a supported model, preview an unsupported GGUF (must be refused in plain terms), prepare the supported one
 *   (pack, from the folder, nothing copied), load it, get a chat answer, unload, shut down, and check nothing is left running.
 *
 *   node local-gguf-e2e.mjs --root /data/ui4-e2e --good /data/141-gyro/models/IQ2_XS --bad /data/ui4-e2e/bad [--context 8192] [--backend cuda|metal]
 *   (the app folder is <root>/app: node/, host/, control/ ...; <root>/components gets the engine, from <root>/app/engine-bundle when the app carries one)
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { join, resolve } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const root = resolve(args.root);
const good = args.good, bad = args.bad;
const context = Number(args.context ?? 8192);
const app = join(root, 'app');
const dirs = { components: join(root, 'components'), install: join(root, 'install'), profiles: join(root, 'profiles'), runtime: join(root, 'runtime'), home: join(root, 'home'), xdg: join(root, 'xdg'), appdata: join(root, 'appdata') };
for (const d of [dirs.install, dirs.home, dirs.xdg, dirs.appdata]) mkdirSync(d, { recursive: true });
const delay = ms => new Promise(r => setTimeout(r, ms));
const steps = [];
let failed = false;
const record = (name, ok, detail = '') => { steps.push({ name, ok: !!ok, detail: String(detail).slice(0, 400) }); console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + String(detail).slice(0, 300) : ''}`); if (!ok) failed = true; };
const freePort = async () => { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; };

const env = { ...process.env, STRATA_COMPONENTS_DIR: dirs.components, STRATA_PROFILE_DIR: dirs.profiles, STRATA_RUNTIME_DIR: dirs.runtime, STRATA_INSTALL_DIR: dirs.install, STRATA_HOME: dirs.home, STRATA_MODELS_DIR: join(root, 'Strata-data'), STRATA_ENGINE_BACKEND: args.backend ?? 'cuda', XDG_CONFIG_HOME: dirs.xdg, APPDATA: dirs.appdata, PYTHONDONTWRITEBYTECODE: '1' };
delete env.STRATA_API_KEY;
const port = await freePort();
const base = `http://127.0.0.1:${port}`;
const host = spawn(join(app, 'node', 'node'), [join(app, 'host', 'server', 'index.mjs')], { cwd: join(app, 'host'), stdio: ['ignore', 'pipe', 'pipe'], detached: false,
  env: { ...env, STRATA_PORT: String(port), STRATA_DIST: join(app, 'host', 'dist'), NODE_ENV: 'production', STRATA_PACKAGED: '1', STRATA_REAL_ENGINE: '1', STRATA_NO_ACQUIRE: '1', STRATA_EXIT_WITH_PARENT: '1',
    STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_WORKSPACE: join(root, 'workspace') } });
mkdirSync(join(root, 'workspace'), { recursive: true });
let hostLog = ''; host.stdout.on('data', c => { hostLog += c; }); host.stderr.on('data', c => { hostLog += c; });
const call = async (path, method = 'GET', body) => { const r = await fetch(base + path, { method, headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) }); return { status: r.status, body: await r.json().catch(() => null) }; };
async function waitJob(id, seconds) {
  const until = Date.now() + seconds * 1000;
  let last = '';
  while (Date.now() < until) {
    const state = (await call('/api/native')).body;
    const job = state.jobs.find(j => j.id === id);
    if (job && job.state !== 'running') return { job, state };
    if (job && job.summary !== last) { last = job.summary; console.log('     ...', String(job.summary).slice(0, 140)); }
    await delay(2000);
  }
  throw new Error('operation did not finish in time');
}
const engineProcs = () => spawnSync('ps', ['-eo', 'pid,args'], { encoding: 'utf8' }).stdout.split('\n').filter(l => l.includes(join(dirs.components, 'engine')) && /\/strata( |$)/.test(l));

async function main() {
  let up = false;
  for (let i = 0; i < 200 && !up; i++) { try { up = (await fetch(base + '/api/health')).ok; } catch { await delay(150); } }
  record('host healthy', up, up ? base : hostLog.slice(-500));
  if (!up) return;
  let status = (await call('/api/control/status')).body;
  for (let i = 0; i < 180 && status?.engineSetup; i++) { await delay(1000); status = (await call('/api/control/status')).body; }   // the engine that came with the app is copied on the first start
  record('Control status: bundled, engine usable', status?.engine?.usable === true, `engine=${status?.engine?.engine?.version} ${status?.engine?.notice ?? ''}`);
  await call('/api/native?refresh=1');

  // ---- the unsupported GGUF: a plain refusal
  const badPreview = await call('/api/native/local/preview', 'POST', { path: bad });
  const badIssues = badPreview.body?.preview?.issues ?? badPreview.body?.issues ?? [];
  const badStatus = badPreview.body?.preview?.file_set_status;
  console.log('     bad preview:', badPreview.status, JSON.stringify(badPreview.body).slice(0, 400));
  record('an unsupported GGUF is refused, nothing adopted', badPreview.status === 200 ? (badStatus !== 'plausible' || !badPreview.body.preview.candidate_selection) : badPreview.status >= 400, `status=${badPreview.status}; file_set_status=${badStatus}; issues=${JSON.stringify(badIssues)}`);

  // ---- the supported folder
  const goodPreview = await call('/api/native/local/preview', 'POST', { path: good });
  const selection = goodPreview.body?.preview?.candidate_selection;
  console.log('     good preview:', JSON.stringify(goodPreview.body).slice(0, 500));
  record('a folder of a supported model is recognised', goodPreview.status === 200 && goodPreview.body?.preview?.file_set_status === 'plausible' && !!selection, `family=${selection?.family}; model=${selection?.model}; files=${goodPreview.body?.preview?.file_count}`);
  if (!selection || args.only === 'preview') return;

  // ---- prepare (what the page's button does): install from the folder, nothing copied or downloaded
  const native0 = (await call('/api/native?refresh=1')).body;
  const catalogSize = native0.catalog?.families?.find(f => f.family === selection.family)?.sizes.find(s => s.model === selection.model);
  record('the catalog lists the recognised version', !!catalogSize, catalogSize ? `id=${catalogSize.id}` : `no ${selection.family}/${selection.model} in the catalog`);
  const started = await call('/api/native/operations', 'POST', { kind: 'install', args: { family: selection.family, model: selection.model, gguf_dir: good, context, vision: 'no' } });
  record('prepare starts', started.status < 300 && !!started.body?.id, `status=${started.status} ${JSON.stringify(started.body).slice(0, 200)}`);
  if (!started.body?.id) return;
  const prepared = await waitJob(started.body.id, 3600);
  record('prepare finishes', prepared.job.state === 'done', `${prepared.job.state}: ${prepared.job.error ?? prepared.job.summary}`);
  if (prepared.job.state !== 'done') return;
  const native1 = (await call('/api/native?refresh=1')).body;
  const modelId = catalogSize?.id ?? `${selection.family === 'qwen' ? '' : selection.family + '-'}${String(selection.model).toLowerCase()}`;
  const entry = native1.installed?.models?.find(m => m.model === modelId);
  record('the model is ready and points at the folder, not a copy', entry?.ready === true, `model=${modelId}; ready=${entry?.ready}; dataFolder=${entry?.dataFolder}`);
  if (!entry?.ready) return;

  // ---- load
  const load = await call('/api/native/operations', 'POST', { kind: 'load', args: { model: modelId } });
  const loaded = await waitJob(load.body.id, 1800);
  const instance = loaded.state.instances[0];
  record('load starts the engine from the component folder', loaded.job.state === 'done' && instance?.status === 'ready', `job=${loaded.job.state} ${loaded.job.error ?? ''}; instance=${instance?.status}`);
  if (instance?.status !== 'ready') return;
  record('one engine process is running', engineProcs().length === 1, engineProcs().map(l => l.trim().slice(0, 140)).join(' | '));

  // ---- a chat answer, checked
  const chat = await fetch(base + '/api/chat/stream', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ sessionId: 'e2e-chat', message: 'Reply with exactly one word: pong', history: [], nativeModelId: instance.id, thinking: 'Off', agent: false, workspaceId: null }) });
  let text = '', done = null;
  for await (const chunk of chat.body) {
    for (const line of Buffer.from(chunk).toString('utf8').split('\n')) {
      if (!line.trim()) continue;
      try { const f = JSON.parse(line); if (f.type === 'delta') text += f.text ?? ''; if (f.type === 'done') done = f; if (f.type === 'error') text += ` [error: ${f.message}]`; } catch { /* partial line */ }
    }
  }
  record('the model answers through the app (the answer text is checked)', chat.status === 200 && /pong/i.test(text), `status=${chat.status}; answer=${JSON.stringify(text.slice(0, 120))}; tokensOut=${done?.tokensOut}; tps=${done?.tps}`);

  // ---- unload, shutdown, nothing left
  const unload = await call(`/api/native/instances/${instance.id}/unload`, 'POST');
  await delay(3000);
  record('unload stops the engine', unload.status < 300 && engineProcs().length === 0, `status=${unload.status}; engines left=${engineProcs().length}`);
}

try { await main(); } catch (e) { record('the flow', false, e?.stack ?? e); }
host.kill();
for (let i = 0; i < 100 && host.exitCode === null; i++) await delay(100);
if (host.exitCode === null) host.kill('SIGKILL');
await delay(2000);
const left = spawnSync('ps', ['-eo', 'pid,args'], { encoding: 'utf8' }).stdout.split('\n').filter(l => (l.includes(dirs.install) || l.includes(join(app, 'host'))) && !l.includes('ps -eo'));
record('nothing of this run is left running', left.length === 0, left.join(' | ').slice(0, 300));
writeFileSync(join(root, 'e2e-result.json'), JSON.stringify({ steps, hostLogTail: hostLog.slice(-2000) }, null, 1));
console.log(failed ? 'FAILED' : 'ALL OK');
process.exit(failed ? 1 : 0);
