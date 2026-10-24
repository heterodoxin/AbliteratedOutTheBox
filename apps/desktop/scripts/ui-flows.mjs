#!/usr/bin/env node
/**
 * Headless UI flows of the installed app against a packaged host with a FAKE installer and a FAKE engine (nothing is
 * downloaded, no real engine runs; the real Control starts the fake engine through the owner runtime):
 *
 *   node apps/desktop/scripts/ui-flows.mjs [--control <staged control>] [--node <node.exe>] [--shots <dir>] [--only download,models,tune,fail,theme]
 *
 *   download  the onboarding download: ONE combined bar over both files, an ETA in minutes left of Stop, Finish later closes the
 *             pop-up with the download still running, the Models screen shows the same progress, Stop stops it
 *   models    an installed model is the one Models selects and loads, without a download pop-up
 *   tune      the tuning step offers no picture reading, tunes, writes a RAM budget for "leave 20%", opens chat; reading size
 *   fail      an engine that stops while loading: the error is in words and the pop-up does not scroll
 *   theme     the app follows the system light/dark setting
 * Per-user folders are all under a scratch folder. Exit 1 if a check fails.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, rmdirSync, symlinkSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]]] : acc), []));
const here = dirname(fileURLToPath(import.meta.url));
const repoHost = resolve(here, '../../host');
const staged = resolve(here, '../src-tauri/resources');
const stagedControl = resolve(args.control ?? join(staged, 'control'));
const nodeExe = resolve(args.node ?? join(staged, 'node', process.platform === 'win32' ? 'node.exe' : 'node'));
const work = mkdtempSync(join(tmpdir(), 'strata-ui-flows-'));
const shots = resolve(args.shots ?? join(work, 'shots'));
mkdirSync(shots, { recursive: true });
const only = new Set(String(args.only ?? 'download,models,profile,ui2,ui2d,tune,fail,delete,theme,restart,perf,polish,stop,latest').split(','));
const isWin = process.platform === 'win32';
const { chromium } = await import(pathToFileURL(join(repoHost, 'node_modules/playwright-core/index.mjs')).href).catch(() => import('playwright-core'));
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);
const delay = ms => new Promise(r => setTimeout(r, ms));
const sha = buf => createHash('sha256').update(buf).digest('hex');
const freePort = async () => { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; };
let failed = false;
const check = (ok, label, detail = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? '  ' + detail : ''}`); if (!ok) failed = true; };

// ------------------------------------------------------------------ the app folder: the real Control with a fake installer
const control = join(work, 'control');
mkdirSync(control, { recursive: true });
for (const name of ['python', 'strata_control', 'strata_cli', 'serve']) symlinkSync(join(stagedControl, name), join(control, name), 'junction');
for (const name of ['strata.py', 'VERSION', 'BUNDLE.json', 'requirements.txt']) copyFileSync(join(stagedControl, name), join(control, name));
cpSync(join(stagedControl, 'tools'), join(control, 'tools'), { recursive: true });
const python = isWin ? join(control, 'python', 'python.exe') : join(control, 'python', 'bin', 'python3');

const FAKE_INSTALLER = String.raw`
import json, os, sys, time
from pathlib import Path
ROOT = Path(sys.argv[sys.argv.index('--root') + 1]) if '--root' in sys.argv else Path('.')
STATE = Path(os.environ['FAKE_STATE']); STATE.mkdir(parents=True, exist_ok=True)
RATE = float(os.environ.get('FAKE_RATE_GB_S', '0.5'))
def size(model, gb, ram, mid):
    return {'model': model, 'about': model, 'download_gb': gb, 'ram_needed_gb': ram, 'experts_gb': ram - 8, 'on_this_pc': 'fits', 'installed': False, 'images': True, 'experimental': False, 'id': mid}
CATALOG = {'summary': 'fake', 'families': [{'family': 'qwen', 'title': 'Qwen3.8-Flash-Next', 'about': 'x', 'experimental': False, 'images': True, 'sizes': [
    size('Q2_0', 66.4, 48, 'q2_0'), size('IQ3_XXS', 75.8, 60, 'iq3_xxs'), size('IQ3_S', 83.6, 62, 'iq3_s')]}],
    'contexts': [8192, 32768, 65536], 'recommendation': {'family': 'qwen', 'model': 'IQ3_XXS', 'title': 'Qwen3.8-Flash-Next IQ3_XXS', 'context': 32768}, 'context_note': ''}
HARDWARE = {'ram_gb': 63.1, 'cpu': {'name': 'cpu', 'cores': 12}, 'gpus': [{'index': 0, 'name': 'NVIDIA GeForce RTX 5070', 'vram_gb': 11.9, 'usable': True}]}
TOTAL = 83.6
def job():
    f = STATE / 'job.json'
    if not f.exists(): return None
    j = json.loads(f.read_text())
    if j.get('cancelled'):
        return {'running': False, 'exit_code': 1, 'error': 'stopped', 'result': 'failed'}
    done = min(TOTAL, RATE * (time.time() - j['start']))
    return {'running': True, 'step': '5/7: downloading Qwen3.8-Flash-Next IQ3_S', 'model': 'IQ3_S',
            'download': {'file': 'model files', 'done_gb': round(done, 3), 'total_gb': TOTAL, 'remaining_gb': round(TOTAL - done, 3), 'percent': int(100 * done / TOTAL)}}
def call(name, a):
    if name == 'strata_models': return CATALOG
    if name == 'strata_remove':
        cfg = ROOT / ('strata-%s.json' % a['model'])
        if a.get('confirm') and cfg.exists(): cfg.unlink()
        return {'summary': 'Deleted', 'plan': {'gb': 83.6}, 'failed': []}
    if name == 'strata_status': return {'summary': 'fake', 'installed': {'models': [{'model': 'iq3_s', 'disk_gb': 83.6, 'disk_bytes': 83600000000}]}, 'hardware': HARDWARE, 'recommendation': CATALOG['recommendation'], 'install': job(), 'running': []}
    if name == 'strata_install':
        if a.get('cancel'):
            f = STATE / 'job.json'
            if f.exists(): j = json.loads(f.read_text()); j['cancelled'] = True; f.write_text(json.dumps(j))
            return {'summary': 'stopped', 'install': job()}
        if a.get('confirm'):
            (STATE / 'job.json').write_text(json.dumps({'start': time.time()}))
            return {'summary': 'started', 'plan': {'download_gb': TOTAL}, 'install': job()}
        if not a: return {'summary': 'progress', 'install': job()}
        return {'summary': 'Plan', 'plan': {'title': 'Qwen3.8-Flash-Next IQ3_S', 'download_gb': TOTAL, 'disk_short': None}}
    raise Exception('unknown tool ' + name)
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    m = json.loads(line)
    if 'id' not in m: continue
    method = m.get('method')
    if method == 'initialize': res = {'protocolVersion': '2024-11-05'}
    elif method == 'tools/list': res = {'tools': [{'name': n} for n in ('strata_models', 'strata_status', 'strata_install', 'strata_remove')]}
    else:
        p = m['params']
        res = {'structuredContent': call(p['name'], p.get('arguments') or {})}
    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': m['id'], 'result': res}) + '\n'); sys.stdout.flush()
`;
writeFileSync(join(control, 'tools', 'strata_mcp.py'), FAKE_INSTALLER);

const FAKE_ENGINE = `import os, sys, json, time
from pathlib import Path
if os.environ.get('FAKE_ENGINE_FAIL'):
    print('memory guard: 5.1 GiB of RAM free is under the 6 GiB limit (limit 63.1 GiB)', flush=True)
    sys.exit(3)
ctx = '4096'
if '--max-context' in sys.argv: ctx = sys.argv[sys.argv.index('--max-context') + 1]
print('READY %s stop' % ctx, flush=True)
for line in sys.stdin:
    if line.strip() == 'QUIT': break
    if line.startswith(('GEN ', 'BGEN ')):
        slot = int(line.split()[1]) if line.startswith('BGEN ') else None
        t0 = time.time(); n = 0
        for ch in os.environ.get('FAKE_REPLY', 'a'):
            print('T %d' % ord(ch), flush=True); n += 1
            if os.environ.get('FAKE_TOKEN_DELAY'): time.sleep(float(os.environ['FAKE_TOKEN_DELAY']))
        print('DONE %d 12 300.0 %.1f length 0 0 0' % (n, max(1.0, (time.time() - t0) * 1000)), flush=True)
        if slot is not None: print(f'BADM {slot} 0', flush=True)
`;

// one scenario = one fresh host with its own per-user folders
async function scenario(name, { installed = false, env = {} } = {}, body) {
  if (!only.has(name)) return;
  console.log(`--- ${name}`);
  const dir = join(work, name);
  const dirs = { home: join(dir, 'home'), appdata: join(dir, 'appdata'), local: join(dir, 'local'), components: join(dir, 'local', 'Strata', 'components'), install: join(dir, 'local', 'Strata', 'install'), state: join(dir, 'fake-state'), profiles: join(dir, 'profiles'), runtime: join(dir, 'runtime') };
  for (const d of [dirs.home, dirs.appdata, dirs.local, dirs.components, dirs.install, dirs.state, join(dir, 'models')]) mkdirSync(d, { recursive: true });
  const baseEnv = { ...process.env, STRATA_COMPONENTS_DIR: dirs.components, STRATA_PROFILE_DIR: dirs.profiles, STRATA_RUNTIME_DIR: dirs.runtime, STRATA_HOME: dirs.home, STRATA_MODELS_DIR: join(dir, 'models'), APPDATA: dirs.appdata, LOCALAPPDATA: dirs.local, USERPROFILE: dirs.home, XDG_CONFIG_HOME: join(dir, 'xdg'), XDG_DATA_HOME: join(dir, 'xdgdata'), PYTHONDONTWRITEBYTECODE: '1', STRATA_ENGINE_BACKEND: 'cuda', FAKE_STATE: dirs.state, ...env };
  delete baseEnv.STRATA_API_KEY; delete baseEnv.STRATA_INSTALL_DIR;
  if (installed) {
    const platform = isWin ? 'windows-x86_64' : 'linux-x86_64';
    const folder = join(dirs.components, 'engine', `0.1.45-${platform}-cuda`);
    mkdirSync(folder, { recursive: true });
    writeFileSync(join(folder, 'engine_main.py'), FAKE_ENGINE);
    const entry = isWin ? 'engine.cmd' : 'engine';
    writeFileSync(join(folder, entry), isWin ? `@echo off\r\n"${python}" "%~dp0engine_main.py" %*\r\n` : `#!/bin/sh\nexec "${python}" "$(dirname "$0")/engine_main.py" "$@"\n`);
    const files = [entry, 'engine_main.py'].map(p => { const b = readFileSync(join(folder, p)); return { path: p, sha256: sha(b), bytes: b.length }; });
    writeFileSync(join(folder, 'engine-component.json'), JSON.stringify({ schema_version: 1, id: 'strata-engine', version: '0.1.45', platform, backend: 'cuda', entrypoint: entry, files }));
    const tok = join(dirs.install, 'tokenizer');
    mkdirSync(tok, { recursive: true });
    const script = `import json, sys\nfrom tools.strata_tokenizer import BYTE_TO_UNICODE\ntok = sys.argv[1]\ntokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ['ab', '<|im_start|>', '<|im_end|>', '<|endoftext|>']\njson.dump({t: i for i, t in enumerate(tokens)}, open(tok + '/vocab.json', 'w'))\nopen(tok + '/merges.txt', 'w').write('a b')\njson.dump([0] * 257 + [3, 3, 3], open(tok + '/token_type.json', 'w'))\n`;
    const made = spawnSync(python, ['-c', script, tok], { cwd: control, env: { ...baseEnv, PYTHONNOUSERSITE: '1' }, encoding: 'utf8' });
    if (made.status !== 0) throw new Error('tokenizer fixture: ' + made.stderr);
    writeFileSync(join(dirs.install, 'strata-iq3_s.json'), JSON.stringify({ exe: 'component:strata-engine', backend: 'cuda', tokenizer: tok, args: ['--max-context', '32768', '--pool-workers', '12'], cwd: dir, model_name: 'stub', host: '127.0.0.1', port: 0, kv: 'int8' }));
  }
  const port = env.FIXED_PORT ? Number(env.FIXED_PORT) : await freePort(), base = `http://127.0.0.1:${port}`;
  const host = spawn(nodeExe, [join(repoHost, 'server', 'index.mjs')], { cwd: repoHost, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
    env: { ...baseEnv, STRATA_PORT: String(port), STRATA_DIST: join(repoHost, 'dist'), STRATA_PACKAGED: '1', STRATA_REAL_ENGINE: '1', STRATA_NO_ACQUIRE: '1', STRATA_CONTROL_DIR: control, STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()) } });
  let log = ''; host.stdout.on('data', c => { log += c; }); host.stderr.on('data', c => { log += c; });
  const api = async (path, init) => { const r = await fetch(base + path, init); return { status: r.status, body: await r.json().catch(() => null) }; };
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  try {
    let up = false;
    for (let i = 0; i < 200 && !up; i++) { try { up = (await api('/api/health')).status === 200; } catch { await delay(150); } }
    check(up, `${name}: the packaged host starts`);
    await api('/api/native?refresh=1');
    await body({ base, api, browser, dirs, shot: (page, n) => page.screenshot({ path: join(shots, `${name}-${n}.png`) }), log: () => log });
  } catch (error) {
    check(false, `${name}: the flow`, String(error?.stack ?? error).slice(0, 700) + '\n' + log.slice(-600));
  } finally {
    await browser.close().catch(() => {});
    host.kill(); await new Promise(r => host.once('exit', r));
  }
}
const newPage = async (browser, base, options = {}) => {
  const context = await browser.newContext({ viewport: { width: 1400, height: 950 }, ...(process.env.SCHEME ? { colorScheme: process.env.SCHEME } : {}), ...options });
  const page = await context.newPage();
  const problems = [];
  page.on('pageerror', e => problems.push(e.message));
  page.problems = problems;
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  return page;
};
const clickText = async (page, re, timeout = 30000) => { const b = page.getByRole('button', { name: re }).first(); await b.waitFor({ timeout }); await b.click(); };
const noScroll = async page => page.evaluate(() => {
  const root = document.querySelector('.guided-onboarding');
  const bad = [];
  for (let el = document.querySelector('.onboard-tune-stage, .onboard-prepare-stage, .onboard-model-stage') ?? root; el && el !== document.body; el = el.parentElement) {
    const style = getComputedStyle(el);
    if (/(auto|scroll)/.test(style.overflowY) && el.scrollHeight > el.clientHeight + 1) bad.push(`${el.className || el.tagName}: ${el.scrollHeight} > ${el.clientHeight}`);
  }
  return bad;
});

// ------------------------------------------------------------------------------------------------ download
await scenario('download', {}, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /get started/i);
  await page.getByText(/Choose a version/i).waitFor({ timeout: 30000 });
  await page.locator('[role=radio]').filter({ hasText: 'IQ3_S' }).first().click();
  await clickText(page, /^Continue/);
  await clickText(page, /Download model/i, 60000);
  await page.locator('.onboard-progress progress').waitFor({ timeout: 30000 });
  // ONE bar: sample the percent while it runs, it only goes up
  const samples = [];
  let eta = '';
  for (let i = 0; i < 40 && !eta; i++) {
    const state = (await api('/api/native')).body;
    const d = state.jobs.at(-1)?.result?.install?.download;
    if (d) samples.push(d.done_gb);
    eta = (await page.locator('.onboard-progress-actions span').first().textContent({ timeout: 1000 }).catch(() => '')) ?? '';
    await delay(1000);
  }
  await shot(page, '1-wizard-progress');
  check(/About \d+ (min left|h)/.test(eta), 'the wizard shows about how many minutes are left', eta.trim());
  const order = await page.evaluate(() => { const a = document.querySelector('.onboard-progress-actions'); const kids = a ? [...a.children].map(k => k.tagName + ':' + Math.round(k.getBoundingClientRect().left)) : []; return kids; });
  check(order.length === 2 && order[0].startsWith('SPAN') && order[1].startsWith('BUTTON') && Number(order[0].split(':')[1]) < Number(order[1].split(':')[1]), 'the time left sits left of Stop', order.join(' '));
  check(samples.length >= 3 && samples.every((v, i) => i === 0 || v >= samples[i - 1]), 'one combined count that only goes up', samples.map(v => v.toFixed(1)).join(', '));
  const meta = await page.locator('.onboard-progress-meta span').first().textContent();
  check(/GB/.test(meta ?? '') && /83\.6/.test(meta ?? ''), 'the bar says done / total over the whole model', meta);
  // close the pop-up while it downloads
  await clickText(page, /Finish later/i);
  await page.waitForURL(/#\/models/, { timeout: 15000 }).catch(() => {});
  await page.getByText(/Preparing your model/i).waitFor({ timeout: 15000 }).catch(() => {});
  await delay(1500);
  await shot(page, '2-models-progress');
  await page.locator('.native-progress').first().screenshot({ path: join(shots, 'download-3-models-progress-card.png') }).catch(() => {});
  check(await page.locator('.guided-onboarding').count() === 0, 'the setup pop-up closed');
  check(await page.getByText(/Preparing your model/i).count() > 0, 'the Models screen shows the running download');
  const still = (await api('/api/native')).body.jobs.at(-1);
  check(still?.state === 'running', 'the download keeps running in the background', still?.state);
  const modelsEta = await page.locator('.native-progress .onboard-progress-actions span').first().textContent({ timeout: 15000 }).catch(() => '');
  check(/About \d+ (min left|h)/.test(modelsEta ?? ''), 'the Models screen shows the time left too', (modelsEta ?? '').trim());
  await clickText(page, /^Stop$/);
  await delay(2500);
  check((await api('/api/native')).body.jobs.at(-1)?.state !== 'running', 'Stop stops it');
  check(page.problems.length === 0, 'no page error', page.problems.join('; '));
});

// ------------------------------------------------------------------------------------------------ models
await scenario('models', { installed: true }, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /Go straight to the app/i);
  await page.waitForURL(/#\/models/, { timeout: 15000 }).catch(() => {});
  await page.getByText(/Conversation context/i).first().waitFor({ timeout: 30000 });
  await delay(1500);
  await shot(page, '1-models');
  await page.locator('.version-trigger').first().waitFor({ timeout: 20000 }).catch(async () => { await shot(page, '0-models-missing-summary'); console.log('info  page text:', JSON.stringify((await page.locator('body').innerText()).slice(0, 500))); });
  const summary = await page.locator('.version-trigger').first().textContent({ timeout: 5000 }).catch(() => '');
  // the Models tab fits the first view: nothing to scroll while no list is open and Expert mode is closed
  if (await page.getByRole('button', { name: /^Close expert mode$/ }).count()) { await page.getByRole('button', { name: /^Close expert mode$/ }).click(); await delay(500); }
  await shot(page, '1b-models-one-view');
  const overflow = await page.evaluate(() => { const el = document.querySelector('.main__body'); return el ? el.scrollHeight - el.clientHeight : 0; });
  check(overflow <= 2, 'the Models tab fits the first view without scrolling', String(overflow) + ' px over');
  check(await page.locator('.family-switch').count() === 1, 'the model family switch is in the navbar');
  check(await page.locator('h1').filter({ hasText: /^Model$/ }).count() === 0, 'and the navbar does not say "Model"');
  check(/IQ3_S|High quality/.test(summary ?? ''), 'Models selects the model that is on this PC, not the recommendation', (summary ?? '').replace(/\s+/g, ' ').trim());
  check(!/IQ3_XXS|Balanced/.test(summary ?? ''), 'the recommendation (IQ3_XXS) is not what is selected');
  const load = page.getByRole('button', { name: /^Load model$/ }).first();
  await load.click();
  await delay(1200);
  check(await page.getByRole('dialog').filter({ hasText: /Download and prepare/i }).count() === 0, 'loading an installed model opens no download pop-up');
  let ready = false;
  for (let i = 0; i < 90 && !ready; i++) { const s = (await api('/api/native')).body; ready = s.instances.some(x => x.status === 'ready'); const err = s.instances.find(x => x.status === 'error'); if (err) throw new Error('load failed: ' + err.error); await delay(1000); }
  if (!ready) { const st = (await api('/api/native')).body; console.log('info  jobs', JSON.stringify(st.jobs.map(j => [j.kind, j.state, j.error ?? j.summary])), 'instances', JSON.stringify(st.instances.map(i => [i.status, i.error]))); await shot(page, '2-models-after-click'); }
  check(ready, 'the model loads (a fake engine under the real Control)');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: /^Unload$/ }).first().waitFor({ timeout: 30000 }).catch(() => {});
  await shot(page, '3-models-loaded');
  check(await page.getByRole('button', { name: /^Unload$/ }).count() > 0, 'after a reload the Models screen shows the model as loaded, with Unload');
  check(await page.getByRole('button', { name: /^Load model$/ }).count() === 0, 'and no longer offers Load model');
  await page.getByRole('button', { name: /^Unload$/ }).first().click();
  let gone = false;
  for (let i = 0; i < 40 && !gone; i++) { gone = (await api('/api/native')).body.instances.length === 0; await delay(500); }
  check(gone, 'Unload stops the engine');
  await page.getByRole('button', { name: /^Load model$/ }).first().waitFor({ timeout: 15000 }).catch(() => {});
  check(await page.getByRole('button', { name: /^Load model$/ }).count() > 0, 'and Load model is back');
  check(page.problems.length === 0, 'no page error', page.problems.join('; '));
});

// ------------------------------------------------------------------------------------------------ the profile used, the popover, the pop-up
await scenario('profile', { installed: true }, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /Go straight to the app/i);
  await page.getByText(/Conversation context/i).first().waitFor({ timeout: 30000 });
  await delay(1500);
  const work = () => page.getByRole('button', { name: /Leave 20% for my work/ }).first();
  const full = () => page.getByRole('button', { name: /Full performance/ }).first();
  await work().click();
  check(await work().getAttribute('aria-pressed') === 'true', 'Leave 20% is chosen before loading');
  await page.getByRole('button', { name: /^Load model$/ }).first().click();
  let ready = false;
  for (let i = 0; i < 90 && !ready; i++) { const s = (await api('/api/native')).body; ready = s.instances.some(x => x.status === 'ready'); await delay(500); }
  check(ready, 'the model loads');
  const inst = (await api('/api/native')).body.instances.find(x => x.status === 'ready');
  check(inst?.resourceProfile === 'work', 'the running model reports the profile it was started with', String(inst?.resourceProfile));
  await delay(1500);
  check(await work().getAttribute('aria-pressed') === 'true' && await full().getAttribute('aria-pressed') === 'false', 'after the load the page still shows Leave 20%, not Full performance');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: /^Unload$/ }).first().waitFor({ timeout: 30000 });
  await delay(1000);
  check(await work().getAttribute('aria-pressed') === 'true', 'and after a reload it is still the profile in use');
  await shot(page, '1-profile-used');
  // the reasoning popover names the loaded model
  await page.evaluate(() => { location.hash = '#/chat'; });
  const effort = page.getByRole('button', { name: /^Reasoning effort/ }).first();
  await effort.waitFor({ timeout: 30000 });
  await effort.click();
  const header = await page.locator('.reasoning-menu-model strong').first().textContent({ timeout: 10000 });
  check(header && !/No model loaded/i.test(header), 'the Reasoning popover names the loaded model, not "No model loaded"', String(header));
  await shot(page, '2-reasoning-popover');
  await page.keyboard.press('Escape');
  // an installed model asked for with other settings: the pop-up never speaks of downloading
  await page.evaluate(() => { location.hash = '#/models'; });
  await page.getByRole('button', { name: /^Unload$/ }).first().waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: /^Unload$/ }).first().click();
  await page.getByRole('button', { name: /^Load model$/ }).first().waitFor({ timeout: 30000 });
  await page.getByRole('combobox', { name: 'Conversation context' }).first().click();
  const options = page.getByRole('option');
  const pick = (await options.allTextContents()).findIndex(t => /^8K$/.test(t.trim()));
  if (pick >= 0) await options.nth(pick).click();
  await page.getByRole('button', { name: /^Load model$|^Checking/ }).first().click();
  const dialog = page.getByRole('dialog').first();
  await dialog.waitFor({ timeout: 15000 }).catch(() => {});
  const text = (await dialog.textContent().catch(() => '')) ?? '';
  await shot(page, '3-apply-dialog');
  if (text) {
    check(!/Download and prepare|Download . load|approximately|[0-9] GB/i.test(text), 'for a downloaded model the pop-up has no download wording or size', text.replace(/\s+/g, ' ').slice(0, 600));
    check(/Apply/.test(text), 'it says what happens: apply settings and load');
  } else console.log('info  the settings were the saved ones, so loading needed no pop-up at all');
  check(page.problems.length === 0, 'no page error', page.problems.join('; '));
});

// ------------------------------------------------------------------------------------------------ version list, This computer, Tune, a task
for (const scheme of ['light', 'dark']) await scenario(scheme === 'light' ? 'ui2' : 'ui2d', { installed: true, env: { FAKE_TOKEN_DELAY: '0.02', FAKE_REPLY: 'I am thinking about it. </think> ' + 'Here is the answer, word by word. '.repeat(12), STRATA_TUNE_NO_QUIET_CHECK: '1' } }, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base, { colorScheme: scheme });
  await clickText(page, /Go straight to the app/i);
  await page.getByText(/Conversation context/i).first().waitFor({ timeout: 30000 });
  await delay(1500);
  // the version list: Recommended (the fixed four), then folded groups, one line per model
  console.log('info  triggers', await page.locator('.version-trigger').count(), await page.getByRole('button', { name: 'Model version' }).count());
  await page.locator('.version-trigger').first().click();
  await delay(300);
  console.log('info  expanded after click', await page.locator('.version-trigger').first().getAttribute('aria-expanded'));
  await delay(800); await shot(page, `0-after-click-${scheme}`);
  console.log('info  menu', await page.evaluate(() => { const m = document.querySelector('.version-menu'); const t = document.querySelector('.version-trigger'); return JSON.stringify({ menu: m ? m.getBoundingClientRect() : null, parent: m ? [m.parentElement.className, getComputedStyle(m.parentElement).display, getComputedStyle(m.parentElement).visibility, m.parentElement.style.cssText.slice(0, 120)] : null, expanded: t?.getAttribute('aria-expanded'), roles: [...document.querySelectorAll('[role=menu]')].length }); }));
  await page.locator('.version-menu').waitFor({ timeout: 10000 });
  const heads = await page.locator('.version-head span:first-child, .version-group__head').allTextContents();
  const names = await page.locator('.version-menu > .version-row .version-row__name strong').allTextContents();
  console.log('info  version list', JSON.stringify(heads.map(t => t.replace(/\s+/g, ' ').trim())), JSON.stringify(names));
  check(heads[0]?.trim() === 'Recommended', `${scheme}: the list starts with Recommended`);
  check(names.length >= 1 && names.length <= 4, `${scheme}: Recommended shows only the picks this catalog has (at most the fixed four)`, names.join(' | '));
  check(heads.some(t => /Swift models|GSQ-RCO models|Gyro models|NVFP4 models|Unsloth models/.test(t)), `${scheme}: the other models are in folded groups`);
  check(heads.at(-1)?.includes('Use your own GGUF'), `${scheme}: the last row is Use your own GGUF`);
  const tall = await page.locator('.version-row').evaluateAll(els => els.filter(e => e.getBoundingClientRect().height > 44).length);
  check(tall === 0, `${scheme}: every model is one line`, String(tall));
  await shot(page, `1-versions-${scheme}`);
  await page.keyboard.press('Escape');
  // This computer
  const pc = page.locator('.pc-card');
  await pc.waitFor({ timeout: 10000 });
  const pcText = ((await pc.textContent()) ?? '').replace(/\s+/g, ' ');
  check(/This computer/.test(pcText) && /(Fits fully in memory|read from the SSD)/.test(pcText), `${scheme}: This computer says in one line whether it fits`, pcText.slice(0, 200));
  check(await pc.locator('.pc-meter').count() >= 1, `${scheme}: it shows the memory bars`);
  check(await pc.locator('.chip, .st-pill, [class*="badge"]').count() === 0, `${scheme}: no chips or badges in it`);
  await shot(page, `2-computer-${scheme}`);
  // Tune: five steps, a live speed, a result card
  await page.getByRole('button', { name: /^Tune model$/ }).first().click({ timeout: 10000 });
  await delay(1500); await shot(page, `2b-tune-dialog-${scheme}`);
  // The fake engine has no other setting worth trying, so its tune passes through "trying" too fast to be seen. The page is shown the progress a
  // real tune reports (Control's status file: stage, speeds) for the running view; the start, the result and every other call are real.
  let fakePhase = 'run';
  await page.route('**/api/native*', async route => {
    const response = await route.fetch(); let json = null;
    try { json = await response.json(); } catch { return route.fulfill({ response }); }
    for (const job of json?.jobs ?? []) if (job.kind === 'autotune' && fakePhase === 'done') { job.state = 'done'; job.progress = null; job.result = { outcome: 'improved', gain_pct: 18, base: { decode_tok_s: 41.2 }, chosen: { decode_tok_s: 48.6 }, preset: { id: 'fast', name: 'Fast · 64K' }, answers: { class: 'exact', fields: [] } }; } else if (job.kind === 'autotune' && job.state === 'running') job.progress = { stage: 'rounds', text: null, minutesLeft: 9, runsDone: 3, runsPlanned: 6, elapsedS: 300, step: 'Trying: more experts kept in memory (round 1 of 2)', speeds: [{ label: 'base', tok_s: 41.2 }, { label: 'Fewer helper threads', tok_s: 40.1 }, { label: 'More experts kept in memory', tok_s: 46.8 }] };
    return route.fulfill({ response, json });
  });
  await page.getByRole('button', { name: /Start tuning|Tune again/ }).first().click({ timeout: 15000 });
  await page.locator('.tune-step').first().waitFor({ timeout: 60000 });
  check(await page.locator('.tune-step').count() === 5, `${scheme}: the tuner shows its five steps`);
  let sawRun = false, sawSpeed = false;
  for (let i = 0; i < 400; i++) {
    const done = await page.locator('.tune-result, .tune-progress').count();
    if (i % 4 === 0) { const jobs = (await api('/api/native')).body?.jobs ?? []; const j = jobs.find(x => x.kind === 'autotune'); if (j) console.log('info  tune job', j.state, JSON.stringify(j.progress ?? null).slice(0, 300)); }
    if (await page.locator('.tune-step[data-s=run]').count()) sawRun = true;
    const big = await page.locator('.tune-live__big').first().textContent().catch(() => '');
    if (/[0-9]/.test(big ?? '')) sawSpeed = true;
    if (sawRun && sawSpeed) { await delay(700); await shot(page, `3-tune-running-${scheme}`); break; }
    if (done) break;
    await delay(600);
  }
  check(sawRun, `${scheme}: a step is shown as running`);
  check(sawSpeed, `${scheme}: the live speed shows a number while it runs`);
  if (sawSpeed) { fakePhase = 'done'; await page.locator('.tune-outcome').waitFor({ timeout: 15000 }).catch(() => {}); check(/18% faster/.test((await page.locator('.tune-outcome__body h3').textContent().catch(() => '')) ?? ''), `${scheme}: the result card shows the gain as its headline`); check(/Fast/.test((await page.locator('.tune-outcome__facts').textContent().catch(() => '')) ?? ''), `${scheme}: and what it was saved as`); }
  await shot(page, `4-tune-result-${scheme}`);
  await page.getByRole('button', { name: /^Close$/ }).first().click({ timeout: 10000 }).catch(() => {});
  // a task: the opened task is skimmable, with pause / delete / Edit / Run now
  const made = await api('/api/tasks', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ title: 'Open Downloads folder', prompt: 'Open my Downloads folder and tell me what is new.', frequency: 'weekdays', at: '08:00', weekday: 1, timeZone: 'Europe/Ljubljana', mode: 'manual', enabled: true }) });
  const id = made.body?.id ?? made.body?.task?.id;
  check(!!id, `${scheme}: a task is created`, JSON.stringify(made.body).slice(0, 120));
  await page.evaluate(i => { location.hash = '#/tasks/' + i; }, id);
  await page.getByRole('button', { name: /^Edit$/ }).first().waitFor({ timeout: 15000 });
  console.log('info  header buttons', await page.getByRole('button', { name: /^Pause / }).count(), await page.getByRole('button', { name: /^Delete / }).count(), await page.getByRole('button', { name: /Run now/ }).count(), JSON.stringify(await page.locator('.task-header-actions button').evaluateAll(b => b.map(x => (x.getAttribute('aria-label') ?? x.textContent ?? '').trim()))));
  check(await page.getByRole('button', { name: /^Pause / }).count() === 1 && await page.getByRole('button', { name: /^Delete / }).count() >= 1 && await page.getByRole('button', { name: /Run now/ }).count() >= 1, `${scheme}: pause, delete, Edit and Run now are in the header`);
  check(await page.locator('.main__back').count() === 0 && await page.locator('.main__title-line .task-state').count() === 0, `${scheme}: nothing sits on the left of an opened task's navbar`);
  const factsText = ((await page.locator('.task-facts').textContent()) ?? '').replace(/\s+/g, ' ');
  check(/Weekdays/.test(factsText) && /Schedule/i.test(factsText) && /Last result/i.test(factsText), `${scheme}: schedule, what it may do and the last result are at the top`, factsText.slice(0, 160));
  await shot(page, `5-task-${scheme}`);
  await page.getByRole('button', { name: /^Pause / }).first().click();
  await page.getByRole('button', { name: /^Resume / }).first().waitFor({ timeout: 10000 });
  check(true, `${scheme}: Pause becomes Resume`);
  check(page.problems.length === 0, `${scheme}: no page error`, page.problems.join('; '));
});

// ------------------------------------------------------------------------------------------------ tune + reading size
await scenario('tune', { installed: true, env: { FAKE_TOKEN_DELAY: '0.02', FAKE_REPLY: ['I am thinking about it.', '</think>', '', 'Here is a list:', '', '- one', '- two', '', '## Heading', '', 'Some `code` and **bold** text.', '', '| a | b |', '|---|---|', '| 1 | 2 |', '', '```', 'code line', '```', ''].join(String.fromCharCode(10)) } }, async ({ base, api, browser, dirs, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /get started/i);
  await page.getByText(/Choose a version/i).waitFor({ timeout: 30000 });
  await page.locator('[role=radio]').filter({ hasText: 'IQ3_S' }).first().click();
  await clickText(page, /^Continue/);
  await page.getByText(/Leave 20% for my work/i).waitFor({ timeout: 30000 });
  await delay(800);
  await shot(page, '1-tune-step');
  check(await page.getByText(/Read pictures when this model loads/i).count() === 0, 'the tuning step offers no picture reading for an app install');
  await clickText(page, /Tune & start chatting/i);
  let done = false, job = null;
  for (let i = 0; i < 120 && !done; i++) { job = (await api('/api/native')).body.jobs.findLast?.(j => j.kind === 'tune') ?? null; done = !!job && job.state !== 'running'; await delay(1000); }
  check(job?.state === 'done', 'tuning finishes without a vision error', job?.error ?? job?.state);
  const cfg = JSON.parse(readFileSync(join(dirs.install, 'strata-iq3_s.json'), 'utf8'));
  check(cfg.args.includes('--resident-budget-gib'), '"leave 20%" wrote a RAM budget for the experts', cfg.args.join(' '));
  const gib = Number(cfg.args[cfg.args.indexOf('--resident-budget-gib') + 1]);
  check(gib + 10 <= 63.1 * 0.8, 'the budget keeps the model within 80% of the PC RAM', `${gib} GiB`);
  // chat + reading size
  await page.getByRole('textbox', { name: /Message Strata/i }).waitFor({ timeout: 60000 });
  await page.getByRole('textbox', { name: /Message Strata/i }).fill('hello');
  await page.keyboard.press('Enter');
  await page.locator('.st-msg--assistant .st-bubble').first().waitFor({ timeout: 60000 });
  await page.locator('.st-msg--assistant .st-bubble p').first().waitFor({ timeout: 60000 }).catch(async () => console.log('info  reply so far:', JSON.stringify(await page.locator('.st-msg--assistant').first().innerText()).slice(0, 300)));
  await page.waitForFunction(() => !document.querySelector('.st-msg--assistant .caret'), null, { timeout: 90000 }).catch(() => {});
  // the chat's telemetry chips: speed and context
  const chips = page.locator('.telemetry__chip');
  const chipTexts = await chips.allInnerTexts();
  console.log('info  chips', JSON.stringify(chipTexts));
  let speedText = '', contextText = '';
  for (let i = 0; i < await chips.count(); i++) {
    await chips.nth(i).click();
    await delay(400);
    const pop = await page.locator('.telepop').last().innerText().catch(() => '');
    if (/tok\/s|speed/i.test(pop) && !speedText) speedText = pop;
    if (/context/i.test(pop) && !contextText) contextText = pop;
    await page.keyboard.press('Escape');
  }
  const tps = Number(speedText.match(/([\d.]+)\s*tok(?:ens)?\/s/)?.[1] ?? NaN);
  check(tps > 15 && tps < 120, 'the speed shown is the decode speed (the engine paced at ~50 tokens a second)', speedText.replace(/\s+/g, ' ').slice(0, 160));
  check(/32\.8K|32K|32,768|32768/.test(contextText) && !/128K/.test(contextText), 'the context shown is the one the model runs with (32K), not 128K', contextText.replace(/\s+/g, ' ').slice(0, 160));
  const sizes = [];
  for (const px of [12, 14, 16, 22]) {
    await page.evaluate(v => { const k = 'strata.prefs.v2'; const cur = JSON.parse(localStorage.getItem(k) || '{}'); localStorage.setItem(k, JSON.stringify({ ...cur, contentFontSize: v })); }, px);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.locator('.st-msg--assistant .st-bubble').first().waitFor({ timeout: 30000 });
    await page.screenshot({ path: join(shots, `tune-3-chat-${px}px.png`) });
    sizes.push(await page.evaluate(() => { const b = document.querySelector('.st-msg--assistant .st-bubble'); const t = b.querySelector('p, .md-li, span') ?? b; const u = document.querySelector('.st-msg--user .st-bubble'); const all = {}; for (const el of b.querySelectorAll('*')) { if (!el.childNodes.length || ![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) continue; all[el.tagName.toLowerCase() + (el.className ? '.' + String(el.className).split(' ')[0] : '')] = parseFloat(getComputedStyle(el).fontSize); } const extra = {}; for (const [k, q] of Object.entries({ reasoning: '.st-msg--assistant .st-collapse__body', codeblock: '.st-msg--assistant .st-code pre', codehead: '.st-msg--assistant .st-code__head' })) { const e = document.querySelector(q); if (e) extra[k] = parseFloat(getComputedStyle(e).fontSize); } Object.assign(all, extra); const chain = []; for (let e = b.querySelector('p'); e && e !== b; e = e.parentElement) chain.push(e.tagName + '.' + String(e.className).split(' ')[0] + '=' + getComputedStyle(e).fontSize); all.chain = chain; globalThis.__all = all; return [parseFloat(getComputedStyle(b).fontSize), parseFloat(getComputedStyle(t).fontSize), parseFloat(getComputedStyle(u).fontSize), all]; }));
  }
  console.log('info  chain', JSON.stringify(await page.evaluate(() => { const el = document.querySelector('.st-msg--assistant .st-bubble p'); const out = []; for (let e = el; e && !e.classList.contains('st-msg'); e = e.parentElement) out.push(e.tagName + '.' + String(e.className).split(' ')[0] + '=' + getComputedStyle(e).fontSize); return out; })));
  // which visible text does NOT follow the reading size? (same elements at 12 and 22)
  const survey = async px => { await page.evaluate(v => { const k = 'strata.prefs.v2'; const cur = JSON.parse(localStorage.getItem(k) || '{}'); localStorage.setItem(k, JSON.stringify({ ...cur, contentFontSize: v })); }, px); await page.reload({ waitUntil: 'domcontentloaded' }); await page.locator('.st-msg--assistant .st-bubble').first().waitFor({ timeout: 30000 }); await delay(600);
    return page.evaluate(() => { const out = {}; for (const el of document.querySelectorAll('main *, .chat *, textarea')) { if (![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()) && el.tagName !== 'TEXTAREA') continue; const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue; const key = el.tagName.toLowerCase() + '.' + String(el.className).split(' ')[0] + ' <' + (el.parentElement?.className ?? '').toString().split(' ')[0] + '>'; out[key] = parseFloat(getComputedStyle(el).fontSize); } return out; }); };
  const a = await survey(12), b = await survey(22);
  // a fresh app (nothing saved) uses 14 px
  await page.evaluate(() => localStorage.removeItem('strata.prefs.v2'));
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.locator('.st-msg--assistant .st-bubble').first().waitFor({ timeout: 30000 });
  check(await page.evaluate(() => getComputedStyle(document.querySelector('.st-msg--assistant .st-bubble')).fontSize) === '14px', 'the default reading size is 14 px');
  const fixed = Object.keys(a).filter(k => k in b && a[k] === b[k]);
  console.log('info  text that does not follow the reading size:', JSON.stringify(fixed.map(k => k + '=' + a[k])));
  await shot(page, '2-chat-22px');
  console.log('info  sizes', JSON.stringify(sizes), JSON.stringify(await page.locator('.st-msg--assistant .st-bubble').first().innerHTML()).slice(0, 400));
  check(sizes[0][1] === 12 && sizes[1][1] === 14 && sizes[2][1] === 16 && sizes[3][1] === 22 && sizes[0][2] === 12 && sizes[3][2] === 22, 'the reading size changes the text size of the conversation', JSON.stringify(sizes));
  // the slider itself
  await page.evaluate(() => { location.hash = '#/chat?settings=1'; });
  const slider = page.getByLabel('Reading text size');
  if (await slider.count()) {
    await slider.first().focus();
    for (let i = 0; i < 4; i++) await page.keyboard.press('ArrowRight');
    await delay(500);
    const now = await page.evaluate(() => getComputedStyle(document.querySelector('.st-msg--assistant .st-bubble') ?? document.body).fontSize);
    check(parseFloat(now) === 18, 'moving the slider changes the conversation text at once', now);
    const box = await slider.first().boundingBox();
    await page.mouse.click(box.x + box.width * 0.9, box.y + box.height / 2);
    await delay(400);
    const dragged = await page.evaluate(() => [getComputedStyle(document.querySelector('.st-msg--assistant .st-bubble') ?? document.body).fontSize, getComputedStyle(document.documentElement).getPropertyValue('--st-content-fs'), JSON.parse(localStorage.getItem('strata.prefs.v2') || '{}').contentFontSize]);
    check(parseFloat(dragged[0]) >= 20 && parseFloat(dragged[2]) >= 20, 'clicking the slider with the mouse changes it too, and it is saved', dragged.join(' '));
  } else console.log('info  the settings slider was not reachable by hash here; the stored size was checked above');
  check(page.problems.length === 0, 'no page error', page.problems.join('; '));
});

// ------------------------------------------------------------------------------------------------ delete
await scenario('delete', { installed: true }, async ({ base, api, browser, dirs, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /Go straight to the app/i);
  await page.waitForURL(/#\/models/, { timeout: 15000 }).catch(() => {});
  await page.getByRole('button', { name: /^Load model$/ }).first().click();
  let ready = false;
  for (let i = 0; i < 90 && !ready; i++) { ready = (await api('/api/native')).body.instances.some(x => x.status === 'ready'); await delay(1000); }
  check(ready, 'a model is loaded');
  await clickText(page, /Manage models/i);
  await page.getByRole('button', { name: /^Delete$/ }).first().click();
  await delay(600);
  await shot(page, '1-delete-confirm');
  const dialog = await page.getByRole('dialog').last().innerText();
  check(/Delete this version\?/.test(dialog) && /Remove 83\.6 GB of downloaded files\. You can download this version again\./.test(dialog), 'the confirm dialog uses the existing words and the real size', dialog.replace(/\s+/g, ' ').slice(0, 600));
  check(/running model will be unloaded/i.test(dialog), 'and says the loaded model is unloaded first');
  await page.getByRole('button', { name: /^Delete version$/ }).click();
  let state = null;
  for (let i = 0; i < 40; i++) { state = (await api('/api/native')).body; if (state.instances.length === 0 && !state.installed?.models?.some(m => m.model === 'iq3_s')) break; await delay(500); }
  check(state.instances.length === 0, 'the loaded model was unloaded');
  check(!existsSync(join(dirs.install, 'strata-iq3_s.json')), 'the installer removed the run config of the model');
  check(!state.installed?.models?.some(m => m.model === 'iq3_s'), 'the model is no longer listed as downloaded');
  await shot(page, '2-after-delete');
});

// ------------------------------------------------------------------------------------------------ perf
await scenario('perf', { installed: true }, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /Go straight to the app/i);
  await page.waitForURL(/#\/models/, { timeout: 15000 }).catch(() => {});
  await page.getByRole('button', { name: /^Load model$/ }).first().click();
  let ready = false;
  for (let i = 0; i < 90 && !ready; i++) { ready = (await api('/api/native')).body.instances.some(x => x.status === 'ready'); await delay(1000); }
  check(ready, 'a model is loaded');
  await delay(2500);
  const requests = [];
  const started = new Map();
  page.on('request', r => started.set(r, Date.now()));
  page.on('requestfinished', r => requests.push([r.url().replace(base, '').slice(0, 60), Date.now() - (started.get(r) ?? Date.now())]));
  await page.evaluate(() => { window.__long = []; new PerformanceObserver(l => { for (const e of l.getEntries()) window.__long.push(Math.round(e.duration)); }).observe({ entryTypes: ['longtask'] }); });
  const t0 = Date.now();
  await page.evaluate(() => { location.hash = '#/chat'; });
  await page.getByRole('textbox', { name: /Message Strata/i }).waitFor({ timeout: 30000 });
  const open = Date.now() - t0;
  await delay(1500);
  const long = await page.evaluate(() => window.__long);
  console.log('info  chat opened in', open, 'ms; long tasks', JSON.stringify(long), 'requests', JSON.stringify(requests.filter(r => r[1] > 100)), 'all', requests.length);
  check(open < 1500 && Math.max(0, ...long) < 300, 'Models to Chat opens at once, with no long block', open + ' ms, longest task ' + Math.max(0, ...long) + ' ms');
});

// ------------------------------------------------------------------------------------------------ polish
await scenario('polish', { installed: true }, async ({ base, api, browser, dirs, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /Go straight to the app/i);
  await page.waitForURL(/#\/models/, { timeout: 15000 }).catch(() => {});
  await page.getByText(/Conversation context/i).first().waitFor({ timeout: 30000 });
  check(await page.getByRole('button', { name: /^Tune model$/ }).count() === 1, 'the Models tab has the Tune model button (the real tuner, not the old short benchmark)');
  await clickText(page, /Manage models/i);
  await delay(500);
  const rows = await page.locator('.native-manage-row').count();
  check(rows === 1, 'Manage models lists only the downloaded model, not the catalog', rows + ' row(s)');
  await shot(page, '1-manage-models');
  await page.getByRole('button', { name: /^Done$/ }).click();
  await page.getByRole('button', { name: /Other local models/i }).click().catch(async () => { await clickText(page, /Manage models/i); await clickText(page, /Use a local GGUF/i); });
  await page.getByText(/Folder with the GGUF files/i).first().waitFor({ timeout: 15000 }).catch(() => {});
  await shot(page, '2-local-gguf');
  check(await page.getByText(/Folder with the GGUF files/i).count() > 0, 'the local GGUF page opens in the same design as Models');
  await page.getByPlaceholder(/Full path to the folder/i).fill(join(dirs.install, 'does-not-exist'));
  await page.getByRole('button', { name: /^Check files$/ }).click();
  await delay(3000);
  check(await page.locator('.error-notice').count() > 0, 'a folder that cannot be used says so in words');
  await shot(page, '2b-local-gguf-problem');
  await page.evaluate(() => { location.hash = '#/chat?settings=1'; });
  await delay(1500);
  const text = await page.locator('body').innerText();
  check(!/Contrast/.test(text), 'Settings has no Contrast option');
  check(!/View request log/.test(text), 'and no "View request log" button');
  await page.getByText(/^Engine$/).first().click().catch(() => {});
  await delay(1500);
  await shot(page, '3-settings-engine');
  const engine = await page.locator('body').innerText();
  check(/Update component/.test(engine) && /Diagnostics/.test(engine), 'Settings > Engine has Update component and Diagnostics');
  check(!/Performance|Model settings|Calibrate|Engine installation/.test(engine), 'and nothing else');
  await page.getByRole('button', { name: /^View logs$/ }).click();      // the logs open in a pop-up that holds Save logs
  await page.getByRole('button', { name: /^Save logs$/ }).click();
  await page.getByText(/Saved to/).first().waitFor({ timeout: 15000 }).catch(() => {});
  const savedText = await page.getByText(/Saved to/).first().innerText().catch(() => '');
  const savedPath = savedText.replace(/^Saved to\s*/, '').trim();
  check(!!savedPath && existsSync(savedPath), 'Save logs writes a file to Downloads and says where', savedPath);
  await shot(page, '4-logs-saved');
  await page.evaluate(() => { location.hash = '#/monitor'; });
  await delay(6000);
  await shot(page, '5-monitor');
  const mon = await page.locator('body').innerText();
  check(/°C/.test(mon), 'the Monitor shows GPU data with no model loaded', (mon.match(/.{0,30}°C.{0,20}/)?.[0] ?? 'no temperature text').replace(/\s+/g, ' '));
});

// ------------------------------------------------------------------------------------------------ stop
await scenario('stop', { installed: true, env: { FAKE_TOKEN_DELAY: '0.1', FAKE_REPLY: 'thinking '.repeat(30) } }, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /Go straight to the app/i);
  await page.waitForURL(/#\/models/, { timeout: 15000 }).catch(() => {});
  await page.getByRole('button', { name: /^Load model$/ }).first().click();
  let ready = false;
  for (let i = 0; i < 90 && !ready; i++) { ready = (await api('/api/native')).body.instances.some(x => x.status === 'ready'); await delay(1000); }
  if (!ready) { const st = (await api('/api/native')).body; console.log('info  jobs', JSON.stringify(st.jobs.map(j => [j.kind, j.state, j.error ?? j.summary])), JSON.stringify(st.instances.map(i => [i.status, i.error]))); await shot(page, '0-stop-notloaded'); }
  check(ready, 'a model is loaded');
  await page.evaluate(() => { location.hash = '#/chat'; });
  const box = page.getByRole('textbox', { name: /Message Strata/i });
  await box.waitFor({ timeout: 30000 });
  await box.fill('hello');
  await page.keyboard.press('Enter');
  await page.locator('.typing').first().waitFor({ timeout: 30000 });
  await delay(1500);
  await page.getByRole('button', { name: /^Stop/i }).first().click().catch(async () => { await page.keyboard.press('Escape'); });
  await delay(4000);
  const dots = await page.locator('.typing').count();
  await shot(page, '1-after-stop');
  check(dots === 0, 'after Stop in the middle of thinking the three dots are gone', dots + ' left');
});

// ------------------------------------------------------------------------------------------------ latest
await scenario('latest', {}, async ({ base, browser, shot }) => {
  const page = await newPage(browser, base);
  const now = Date.now();
  await page.evaluate(now => {
    const messages = [];
    for (let i = 0; i < 40; i++) { messages.push({ id: 'u' + i, role: 'user', content: 'question ' + i, at: now }, { id: 'a' + i, role: 'assistant', content: 'answer number ' + i + ' with some text to fill the line and more of it.', at: now, tokens: 20, inputTokens: 5 }); }
    localStorage.setItem('strata.sessions.v1', JSON.stringify([{ id: 's_latest', title: 'long', updatedAt: now, createdAt: now, messages }]));
    localStorage.setItem('strata.active-session.v1', 's_latest');
  }, now);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await delay(1500);
  if (await page.getByRole('button', { name: /Go straight to the app/i }).count()) { await clickText(page, /Go straight to the app/i); await delay(1500); }
  await page.goto(base + '/#/chat', { waitUntil: 'domcontentloaded' });
  await page.getByRole('textbox', { name: /Message Strata/i }).waitFor({ timeout: 30000 });
  await delay(800);
  await page.locator('.main__body.chat').first().waitFor({ timeout: 15000 }).catch(async () => { await shot(page, '0-latest-missing'); console.log('info  body', JSON.stringify((await page.locator('body').innerText()).slice(0, 300))); });
  await page.locator('.main__body.chat').first().evaluate(el => { el.scrollTop = 3000; });
  await delay(600);
  await page.mouse.wheel(0, -400);
  await delay(600);
  await page.locator('.main__body.chat').first().evaluate(el => { el.scrollTop = 0; });
  await delay(800);
  const gaps = await page.evaluate(() => { const b = document.querySelector('.jumplatest'), c = document.querySelector('.main__body.chat'); if (!b || !c) return null; const r = b.getBoundingClientRect(), cr = c.getBoundingClientRect(); return { top: Math.round(r.top - cr.top), bottom: Math.round(cr.bottom - r.bottom) }; });
  console.log('info  chat dom', JSON.stringify(await page.evaluate(() => { const c = document.querySelector('.main__body.chat'); return { msgs: document.querySelectorAll('.st-msg').length, scrollH: c?.scrollHeight, clientH: c?.clientHeight, top: c?.scrollTop, jump: document.querySelectorAll('.jumplatest').length }; })));
  await shot(page, '1-latest-button');
  check(!!gaps && gaps.top >= 12 && gaps.bottom >= 12, 'the Latest button keeps a gap from the top navigation and from the bottom', JSON.stringify(gaps));
});

// ------------------------------------------------------------------------------------------------ fail
await scenario('fail', { installed: true, env: { FAKE_ENGINE_FAIL: '1' } }, async ({ base, api, browser, shot }) => {
  const page = await newPage(browser, base);
  await clickText(page, /get started/i);
  await page.getByText(/Choose a version/i).waitFor({ timeout: 30000 });
  await page.locator('[role=radio]').filter({ hasText: 'IQ3_S' }).first().click();
  await clickText(page, /^Continue/);
  await clickText(page, /Tune & start chatting/i, 30000);
  await page.getByText(/ran low on memory/i).first().waitFor({ timeout: 120000 });
  await delay(1000);
  await shot(page, '1-tune-error');
  const text = await page.locator('.guided-onboarding').innerText();
  check(/ran low on memory/.test(text), 'the error says in words that the PC ran low on memory');
  check(/Leave 20% for my work/.test(text) && /low-RAM/.test(text), 'and what to try');
  check(!/server_exited|memory guard/.test(text), 'no code or log line in the main text');
  check(!/vision files/i.test(text), 'no picture-reading error');
  const bad = await noScroll(page);
  check(bad.length === 0, 'the tuning pop-up does not scroll when an error shows', bad.join('; '));
  await page.locator('.error-notice summary').first().click();
  await delay(500);
  await shot(page, '2-tune-error-details');
  const details = await page.locator('.error-notice pre').first().innerText();
  check(/server_exited/.test(details) && /exited before/.test(details), 'Details holds the code and the reason the server logged', details.replace(/\s+/g, ' ').slice(0, 120));
  const badOpen = await noScroll(page);
  check(badOpen.length === 0, 'and the pop-up still does not scroll with Details open', badOpen.join('; '));
  await page.setViewportSize({ width: 1280, height: 700 });
  await delay(600);
  const small = await noScroll(page);
  check(small.length === 0, 'nor in a small window (1280x700), Details open', small.join('; '));
  await shot(page, '3-tune-error-small-window');
  void api;
});

// ------------------------------------------------------------------------------------------------ restart
// The app keeps its theme, text size, language and chat list in the browser's storage, which belongs to the window's origin
// (127.0.0.1:<port>). The shell now uses the same port every launch: stop the host, start it again on that port, reopen the
// window with the same browser profile, and everything is still there.
if (only.has('restart')) {
  console.log('--- restart');
  const fixed = await freePort();
  const profile = join(work, 'browser-profile');
  const run = async (body) => scenario('restart', { env: { FIXED_PORT: String(fixed) } }, async ctx => {
    const context = await chromium.launchPersistentContext(profile, { executablePath: CHROME, headless: true, viewport: { width: 1200, height: 800 } });
    try { await body(context, ctx); } finally { await context.close(); }
  });
  only.add('restart');
  await run(async (context, { base }) => {
    const page = context.pages()[0] ?? await context.newPage();
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { localStorage.setItem('strata.prefs.v2', JSON.stringify({ theme: 'dark', contentFontSize: 19, language: 'de' })); localStorage.setItem('strata.sessions.v1', JSON.stringify([{ id: 'chat-1', title: 'My chat' }])); });
  });
  await run(async (context, { base }) => {
    const page = context.pages()[0] ?? await context.newPage();
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    const saved = await page.evaluate(() => ({ prefs: JSON.parse(localStorage.getItem('strata.prefs.v2') || '{}'), chats: localStorage.getItem('strata.sessions.v1'), theme: document.documentElement.dataset.theme }));
    check(saved.prefs.contentFontSize === 19 && saved.prefs.language === 'de', 'text size and language survive a restart on the same port', JSON.stringify(saved.prefs));
    check(saved.theme === 'dark', 'the theme survives too', saved.theme);
    check(/My chat/.test(saved.chats ?? ''), 'and the chat list', saved.chats);
  });
}

// ------------------------------------------------------------------------------------------------ theme
await scenario('theme', {}, async ({ base, browser, shot }) => {
  for (const scheme of ['dark', 'light']) {
    const page = await newPage(browser, base, { colorScheme: scheme });
    await delay(1500);
    const theme = await page.evaluate(() => document.documentElement.dataset.theme);
    check(theme === scheme, `a fresh app follows the system (${scheme})`, theme);
    await shot(page, `1-${scheme}`);
    await page.context().close();
  }
});

// the junctions into the staged Control go first (rmdir on a junction removes only the link)
for (const name of ['python', 'strata_control', 'strata_cli', 'serve']) { try { rmdirSync(join(control, name)); } catch { /* already gone */ } }
try { rmSync(work, { recursive: true, force: true, maxRetries: 3 }); } catch (e) { console.log('info  scratch folder left behind:', work); }
console.log(`shots in ${shots}`);
process.exit(failed ? 1 : 0);
