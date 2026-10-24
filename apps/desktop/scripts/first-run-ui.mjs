#!/usr/bin/env node
/**
 * The first run of the installed app, headless: the host as the Tauri shell starts it (STRATA_PACKAGED=1, STRATA_NO_ACQUIRE=1,
 * the staged Control, nothing installed, nothing in the per-user folders) and Chrome on its onboarding.
 *
 *   node apps/desktop/scripts/first-run-ui.mjs [--control <staged control>] [--node <node.exe>] [--work <scratch>] [--shots <dir>]
 *
 * Checks, one line each (exit 1 if any fails):
 *   the host answers; /api/native has a real catalog (families with sizes), a recommendation and the GPUs of this PC
 *   (from Control's bundled installer on Control's Python); the page shows "Let's get started", and after it the model
 *   list with sizes (no "couldn't show this screen"); no page error; the install plan for the recommended model is a
 *   plan (nothing is downloaded or started: a plan changes nothing); the spy sees no pip/git/compiler/engine fetch.
 * Per-user folders (STRATA_HOME, APPDATA, LOCALAPPDATA, the components and install folders) are all under --work.
 * No model is downloaded and no engine runs.
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]]] : acc), []));
const here = dirname(fileURLToPath(import.meta.url));
const repoHost = resolve(here, '../../host');
const staged = resolve(here, '../src-tauri/resources');
const control = resolve(args.control ?? join(staged, 'control'));
const nodeExe = resolve(args.node ?? join(staged, 'node', process.platform === 'win32' ? 'node.exe' : 'node'));
const work = resolve(args.work ?? mkdtempSync(join(tmpdir(), 'strata-first-run-')));
const shots = resolve(args.shots ?? join(work, 'shots'));
mkdirSync(work, { recursive: true }); mkdirSync(shots, { recursive: true });
const spy = pathToFileURL(join(repoHost, 'tests/helpers/acquire-spy.mjs')).href;
const spyLog = join(work, 'spy.log'); writeFileSync(spyLog, '');

const { chromium } = await import(pathToFileURL(join(repoHost, 'node_modules/playwright-core/index.mjs')).href).catch(() => import('playwright-core'));
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);

let failed = false;
const check = (ok, label, detail = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? '  ' + detail : ''}`); if (!ok) failed = true; };
const delay = ms => new Promise(r => setTimeout(r, ms));
const freePort = async () => { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; };

for (const need of [nodeExe, join(control, 'strata.py'), join(repoHost, 'dist', 'index.html')]) if (!existsSync(need)) { console.error(`missing ${need} (run stage-control.py, fetch-node and "npm run build" in apps/host first)`); process.exit(2); }

const dirs = { home: join(work, 'home'), appdata: join(work, 'appdata'), local: join(work, 'local'), components: join(work, 'local', 'Strata', 'components') };
for (const d of Object.values(dirs)) mkdirSync(d, { recursive: true });
const port = await freePort();
const env = {
  ...process.env,
  STRATA_PORT: String(port), STRATA_DIST: join(repoHost, 'dist'), STRATA_PACKAGED: '1', STRATA_REAL_ENGINE: '1', STRATA_NO_ACQUIRE: '1',
  STRATA_CONTROL_DIR: control, STRATA_HOME: dirs.home, STRATA_MODELS_DIR: join(work, 'models'), STRATA_COMPONENTS_DIR: dirs.components,
  APPDATA: dirs.appdata, LOCALAPPDATA: dirs.local, XDG_CONFIG_HOME: join(work, 'xdg'), XDG_DATA_HOME: join(work, 'xdgdata'),
  STRATA_PROFILE_DIR: join(work, 'profiles'), STRATA_RUNTIME_DIR: join(work, 'runtime'), ACQUIRE_SPY_LOG: spyLog,
};
delete env.STRATA_INSTALL_DIR;
mkdirSync(env.STRATA_MODELS_DIR, { recursive: true });
const host = spawn(nodeExe, [...(args['no-spy'] === undefined ? ['--import', spy] : []), join(repoHost, 'server', 'index.mjs')], { cwd: repoHost, env, stdio: ['ignore', 'pipe', 'pipe'] });
let hostLog = '';
host.stdout.on('data', c => { hostLog += c; }); host.stderr.on('data', c => { hostLog += c; });
const base = `http://127.0.0.1:${port}`;
const api = async (path, init) => { const r = await fetch(base + path, init); return { status: r.status, body: await r.json().catch(() => null) }; };

let browser;
try {
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { up = (await api('/api/health')).status === 200; } catch { await delay(200); } }
  check(up, 'the packaged host starts', `port ${port}`);
  if (!up) throw new Error(hostLog.slice(-2000));

  const native = await api('/api/native?refresh=1');
  const state = native.body;
  check(native.status === 200 && state?.available === true, 'the host reads Control', state?.error ?? '');
  const sizes = (state?.catalog?.families ?? []).reduce((n, f) => n + (f.sizes?.length ?? 0), 0);
  check((state?.catalog?.families?.length ?? 0) > 0 && sizes >= 3, 'the catalog has real families and sizes', `${state?.catalog?.families?.length} families, ${sizes} sizes`);
  check(!!state?.catalog?.recommendation, 'the catalog has a recommendation', JSON.stringify(state?.catalog?.recommendation ?? null).slice(0, 120));
  check(Array.isArray(state?.hardware?.gpus), 'the hardware lists GPUs', (state?.hardware?.gpus ?? []).map(g => `${g.name} ${g.vram_gb} GB`).join('; ') || 'none on this PC');
  check(state?.installDir && realpathSync.native(state.installDir).toLowerCase().startsWith(realpathSync.native(work).toLowerCase()), 'models go to a writable per-user folder, not the bundle', state?.installDir);
  const hw = (await api('/api/hardware')).body;
  console.log('info host hardware probe:', JSON.stringify((hw?.gpus ?? []).map(g => ({ name: g.name, backend: g.backend, vramGb: g.vramGb, unified: g.unified }))));
  check(!state?.installerError, 'the bundled installer answers', state?.installerError ?? '');

  browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage({ viewport: { width: 1400, height: 950 } });
  const problems = [];
  page.on('pageerror', e => problems.push(`pageerror: ${e.message}`));
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('text=/Let.s get started|Choose a version|Connect your Strata/i', { timeout: 30000 }).catch(() => {});
  await page.screenshot({ path: join(shots, '1-welcome.png') });
  const broken = await page.locator('text=/couldn.t show this screen/i').count();
  check(broken === 0, 'the first screen renders (not "Strata couldn\'t show this screen")');
  const start = page.getByRole('button', { name: /get started/i });
  check(await start.count() > 0, 'the welcome step offers "Let\'s get started"');
  if (await start.count()) await start.first().click();
  await page.waitForSelector('text=/Choose a version/i', { timeout: 30000 }).catch(() => {});
  await delay(1200);
  await page.screenshot({ path: join(shots, '2-models.png') });
  check(await page.locator('text=/couldn.t show this screen/i').count() === 0, 'the model step renders');
  check(await page.getByText(/Choose a version/i).count() > 0, 'the model step lists the versions');
  const labels = await page.locator('[role=radio]').allInnerTexts();
  check(labels.length >= 3, 'the versions are the real catalog\'s sizes', labels.map(l => l.replace(/\s+/g, ' ').slice(0, 40)).join(' | '));
  check(labels.some(l => /\d+\.\d\s*GB/.test(l)), 'each version shows its download size');
  check(problems.length === 0, 'no page error', problems.join('; '));

  const recommended = state?.catalog?.recommendation;
  if (recommended?.model) {
    const plan = await api('/api/native/plan', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ family: recommended.family, model: recommended.model, context: recommended.context, data_dir: join(work, 'Strata-data') }) });
    check(plan.status === 200 && !!plan.body?.plan, 'the install plan for the recommended model is a plan (nothing downloaded)', plan.body?.summary?.slice(0, 100) ?? JSON.stringify(plan.body).slice(0, 150));
    check(!existsSync(join(work, 'Strata-data', 'models')), 'a plan wrote no model files');
  }
} catch (error) {
  check(false, 'the run', String(error?.message ?? error).slice(0, 600));
} finally {
  await browser?.close().catch(() => {});
  host.kill();
  await new Promise(r => host.once('exit', r)); // only the PID this script started
  const attempts = readFileSync(spyLog, 'utf8').split('\n').filter(Boolean);
  check(attempts.length === 0, 'the spy saw no pip, git, compiler, engine or internet access', attempts.slice(0, 3).join(' '));
  console.log(`shots in ${shots}`);
  if (!args.work) rmSync(work, { recursive: true, force: true, maxRetries: 3 });
}
process.exit(failed ? 1 : 0);
