#!/usr/bin/env node
/**
 * The Graphics cards choice of the app, end to end on a box with two cards, through the installed app layout (packaged host + bundled Control + a real
 * engine component): the model list carries the options (one rule, strata_control/gpu_fit.py), a model that fits one card recommends that card, then
 *   load with ONE card -> answers, only that card holds the model;  load with BOTH cards (the layer split) -> answers, both cards hold the model.
 *
 *   node cards-e2e.mjs --install <folder with node/ host/ control/> --components <folder holding engine/<package>> --work <scratch> \
 *        --family qwen --model IQ3_XXS --data <data folder that has the model files> [--smi "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"]
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync } from 'node:fs';
import { createServer } from 'node:net';
import { join, resolve } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]?.startsWith('--') || all[i + 1] === undefined ? true : all[i + 1]]] : acc), []));
for (const need of ['install', 'components', 'work', 'model']) if (!args[need]) { console.error('missing --' + need); process.exit(2); }
const install = resolve(args.install), work = resolve(args.work), components = resolve(args.components), family = String(args.family ?? 'qwen'), modelId = String(args.model);
const node = join(install, 'node', 'node'), hostEntry = join(install, 'host', 'server', 'index.mjs');
const dirs = { models: join(work, 'models'), profiles: join(work, 'profiles'), runtime: join(work, 'runtime'), home: join(work, 'home'), xdg: join(work, 'xdg'), appdata: join(work, 'appdata') };
for (const d of [dirs.models, dirs.home, dirs.xdg, dirs.appdata]) mkdirSync(d, { recursive: true });
const delay = ms => new Promise(r => setTimeout(r, ms));
let failed = false;
const record = (name, ok, detail = '') => { if (!ok) failed = true; console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + String(detail).slice(0, 500) : ''}`); };
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
const smi = () => {
  const cmd = String(args.smi ?? 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').split(' ');
  const r = spawnSync(cmd[0], cmd.slice(1), { encoding: 'utf8' });
  return (r.stdout ?? '').split('\n').map(s => Number(s.trim())).filter(Number.isFinite);
};
async function chatPong(instance) {
  const chat = await call('/v1/chat/completions', 'POST', { model: instance?.apiModelId ?? modelId.toLowerCase(), max_tokens: 400, stream: false, temperature: 0, messages: [{ role: 'user', content: 'Reply with exactly one word: pong' }] });
  const m = chat.body?.choices?.[0]?.message ?? {};
  return { ok: chat.status === 200 && /pong/i.test(`${m.content ?? ''}`), text: `${m.content ?? ''}`.slice(0, 60), status: chat.status };
}
async function main() {
  let up = false;
  for (let i = 0; i < 200 && !up; i++) { try { up = (await fetch(base + '/api/health')).ok; } catch { /* not yet */ } if (!up) await delay(150); }
  record('host healthy', up, up ? base : hostLog.slice(-500));
  if (!up) return;
  const packages = existsSync(join(components, 'engine')) ? readdirSync(join(components, 'engine')) : [];
  record('an engine component is installed', packages.length > 0, packages.join(', '));

  // ---- the model list carries the choice
  const native = (await call('/api/native?refresh=1')).body;
  const gpus = native.hardware?.gpus ?? [];
  record('the host sees two or more graphics cards', gpus.filter(g => g.usable !== false).length >= 2, JSON.stringify(gpus.map(g => ({ i: g.index, n: g.name, gb: g.vram_gb }))));
  const size = native.catalog?.families?.find(f => f.family === family)?.sizes.find(s => s.model === modelId);
  const opt = size?.gpu_options;
  record('the model list offers the cards choice for this model', !!opt && opt.options.length >= 2, JSON.stringify(opt?.options?.map(o => ({ key: o.key, cards: o.cards, verdict: o.verdict, rec: o.recommended }))));
  record('a model that fits one card recommends the single largest card (the split is offered, not forced)', opt?.recommended === 'single' && opt.options.find(o => o.key === 'split'), `recommended=${opt?.recommended}`);
  const single = opt?.options.find(o => o.key === 'single')?.cards ?? [0];
  const both = opt?.options.find(o => o.key === 'split')?.cards ?? [0, 1];

  // ---- Download and prepare with the recommended card
  const values = { family, model: modelId, context: 32768, vision: 'no', gpu: single[0], ...(args.data ? { data_dir: resolve(args.data) } : {}) };
  const started = await call('/api/native/operations', 'POST', { kind: 'install', args: values });
  record('the install operation starts', started.status === 200 || started.status === 202, JSON.stringify(started.body).slice(0, 160));
  const done = await waitJob(started.body.id, 3600);
  record('Download and prepare finishes', done.job.state === 'done', `${done.job.state} ${done.job.error ?? done.job.summary}`);
  const installed = (await call('/api/native?refresh=1')).body.installed?.models?.find(m => String(m.model).toLowerCase() === modelId.toLowerCase());
  const model = installed?.model ?? modelId.toLowerCase();

  for (const [label, cards] of [['ONE card', single], ['BOTH cards (layer split)', both]]) {
    const before = smi();
    const load = await call('/api/native/operations', 'POST', { kind: 'load', args: { model, gpus: cards } });
    const loaded = await waitJob(load.body.id, 900);
    const instance = loaded.state.instances[0];
    record(`load with ${label} starts the engine`, loaded.job.state === 'done' && instance?.status === 'ready', `${loaded.job.state} ${loaded.job.error ?? ''}; instance=${instance?.status}; gpus=${JSON.stringify(instance?.gpus)}`);
    record(`the instance runs on exactly the chosen card(s)`, JSON.stringify([...(instance?.gpus ?? [])].sort()) === JSON.stringify([...cards].sort()), `want ${JSON.stringify(cards)} got ${JSON.stringify(instance?.gpus)}`);
    await delay(3000);
    const used = smi();
    const holding = used.map((mb, i) => (mb - (before[i] ?? 0)) > 1000 ? i : -1).filter(i => i >= 0);
    record(`only the chosen card(s) hold the model (graphics memory in use: ${used.join(', ')} MiB)`, JSON.stringify(holding) === JSON.stringify([...cards].sort()), `holding ${JSON.stringify(holding)}`);
    const pong = await chatPong(instance);
    record(`it answers (${label})`, pong.ok, `${pong.status} ${JSON.stringify(pong.text)}`);
    const unloaded = await call(`/api/native/instances/${instance?.id}/unload`, 'POST', {});
    record(`unload (${label})`, unloaded.status === 200, unloaded.status);
    await delay(4000);
  }
}
try { await main(); } catch (e) { record('run', false, e.stack ?? e); }
host.kill();
await delay(1500);
console.log(failed ? 'CARDS-E2E FAIL' : 'CARDS-E2E PASS');
process.exit(failed ? 1 : 0);
