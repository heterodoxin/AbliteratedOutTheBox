#!/usr/bin/env node
/**
 * Picture reading in the APP, end to end on a test box, through the installed app layout (packaged host + bundled Control + a real engine component
 * that carries strata-vision):
 *   Download and prepare with "Read pictures" ON (the picture-reader file is fetched, the vision settings are written), load the model with the picture
 *   reader, send a real picture through the host's own /v1 endpoint and check that the answer names what is in it, unload, shut down.
 *
 *   node vision-e2e.mjs --install <folder with node/ host/ control/> --components <folder holding engine/<package with strata-vision>> \
 *        --work <scratch> --family qwen --model IQ3_XXS --data <data folder that has (or will get) the model files> --image shapes.png [--device cpu|gpu]
 *
 * The picture has a red circle, a blue square, a green triangle and the text "STRATA 42". The run exits 1 if any step fails.
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { createServer } from 'node:net';
import { join, resolve } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]?.startsWith('--') || all[i + 1] === undefined ? true : all[i + 1]]] : acc), []));
for (const need of ['install', 'components', 'work', 'model', 'image']) if (!args[need]) { console.error('missing --' + need); process.exit(2); }
const install = resolve(args.install), work = resolve(args.work), components = resolve(args.components), family = String(args.family ?? 'qwen'), modelId = String(args.model);
const device = String(args.device ?? 'cpu');
const node = join(install, 'node', 'node'), hostEntry = join(install, 'host', 'server', 'index.mjs'), controlDir = join(install, 'control');
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
async function main() {
  let up = false;
  for (let i = 0; i < 200 && !up; i++) { try { up = (await fetch(base + '/api/health')).ok; } catch { /* not yet */ } if (!up) await delay(150); }
  record('host healthy', up, up ? base : hostLog.slice(-500));
  if (!up) return;
  const packages = existsSync(join(components, 'engine')) ? readdirSync(join(components, 'engine')) : [];
  const reader = packages.some(p => existsSync(join(components, 'engine', p, 'strata-vision')));
  record('the engine component carries the picture reader (strata-vision)', reader, packages.join(', '));
  const status = (await call('/api/control/status')).body;
  record('Control: engine usable', status.engine?.usable === true, `engine=${status.engine?.engine?.version}`);

  // ---- Download and prepare, "Read pictures" on (the app sends vision "yes" for the download)
  const values = { family, model: modelId, context: 32768, vision: 'yes', loadVision: device, ...(args.data ? { data_dir: resolve(args.data) } : {}) };
  const plan = await call('/api/native/plan', 'POST', values);
  record('the plan for the download is made', plan.status === 200, JSON.stringify(plan.body).slice(0, 300));
  const started = await call('/api/native/operations', 'POST', { kind: 'install', args: values });
  record('the install operation starts', started.status === 200 || started.status === 202, JSON.stringify(started.body).slice(0, 200));
  const done = await waitJob(started.body.id, 3600);
  record('Download and prepare finishes', done.job.state === 'done', `${done.job.state} ${done.job.error ?? done.job.summary}`);
  const native = (await call('/api/native?refresh=1')).body;
  console.log('    installed models: ' + JSON.stringify((native.installed?.models ?? []).map(m => ({ id: m.id, model: m.model, ready: m.ready, visionPrepared: m.visionPrepared, images: m.images, config: m.config }))));
  const want = modelId.toLowerCase();
  const model = native.installed?.models?.find(m => [m.id, m.model, m.tag].some(x => String(x ?? '').toLowerCase() === want));
  record('the model is installed with the picture reader prepared (visionPrepared)', !!model?.visionPrepared, JSON.stringify({ ready: model?.ready, visionPrepared: model?.visionPrepared, images: model?.images }));
  const configFile = model?.configPath ?? (model?.config ? join(dirs.models, model.config) : null);
  let cfg = {};
  try { cfg = JSON.parse(readFileSync(configFile, 'utf8')); } catch { /* reported below */ }
  const mm = cfg.vision?.mmproj;
  record('the run settings carry the vision entry and --vision, and the mmproj file is on disk',
    !!cfg.vision && (cfg.args ?? []).includes('--vision') && !!mm && existsSync(mm) && statSync(mm).size > 800e6,
    `vision=${cfg.vision ? 'yes' : 'no'} --vision=${(cfg.args ?? []).includes('--vision')} mmproj=${mm ? Math.round(statSync(mm).size / 1048576) + ' MB' : 'none'}`);

  // ---- load with the picture reader
  const load = await call('/api/native/operations', 'POST', { kind: 'load', args: { model: model?.model ?? modelId.toLowerCase(), vision: device } });
  const loaded = await waitJob(load.body.id, 900);
  const instance = loaded.state.instances[0];
  record('load starts the engine with the picture reader', loaded.job.state === 'done' && instance?.status === 'ready', `${loaded.job.state} ${loaded.job.error ?? ''}; instance=${instance?.status}; vision=${instance?.vision}`);
  record('the host says the loaded model reads pictures', !!instance?.vision, `vision=${instance?.vision}`);

  // ---- a real picture through the host's own /v1 endpoint
  const png = readFileSync(resolve(args.image)).toString('base64');
  const t = Date.now();
  let answer = '', finish = '';
  for (const budget of [1536, 4096]) {
    const chat = await call('/v1/chat/completions', 'POST', { model: instance?.apiModelId ?? modelId, max_tokens: budget, stream: false, messages: [{ role: 'user', content: [
      { type: 'text', text: 'Describe this picture: name each shape with its colour and read out any text.' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,' + png } }] }] });
    const choice = chat.body?.choices?.[0] ?? {};
    answer = choice.message?.content ?? '';
    finish = choice.finish_reason ?? String(chat.status);
    if (answer.trim() || finish !== 'length') break;
  }
  console.log(`    answer (${Math.round((Date.now() - t) / 1000)} s, finish=${finish}): ${JSON.stringify(answer.slice(0, 300))}`);
  const low = answer.toLowerCase();
  record('the answer names the red circle', /red|circle/.test(low));
  record('the answer names the blue square', /blue|square/.test(low));
  record('the answer names the green triangle', /green|triangle/.test(low));
  record('the answer reads STRATA 42', low.includes('42'));

  // ---- unload
  const unloaded = await call(`/api/native/instances/${instance?.id}/unload`, 'POST', {});
  record('unload', unloaded.status === 200, unloaded.status);
}
try { await main(); } catch (e) { record('run', false, e.stack ?? e); }
host.kill();
await delay(1500);
console.log(failed ? 'VISION-E2E FAIL' : 'VISION-E2E PASS');
process.exit(failed ? 1 : 0);
