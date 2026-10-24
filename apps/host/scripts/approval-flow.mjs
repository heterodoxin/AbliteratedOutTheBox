/**
 * Headless check of the approval card, light and dark, against a real host, the real Harness and a scripted model:
 *
 *   STRATA_HARNESS_DIR=... node scripts/approval-flow.mjs [shots-dir]
 *
 * For each command the scripted model asks to run in Manual mode it checks that the card asks in plain words, says nothing of
 * tool names or JSON in its headline, and keeps the exact command under Details; then it approves and checks the row that remains.
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { startScriptedWorker } from '../tests/helpers/scripted-worker.mjs';

const HOST_DIR = resolve(import.meta.dirname, '..');
const SHOTS = resolve(process.argv[2] ?? 'approval-shots');
mkdirSync(SHOTS, { recursive: true });
const { chromium } = await import(pathToFileURL(join(HOST_DIR, 'node_modules/playwright-core/index.mjs')).href);
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);
const freePort = () => new Promise(r => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let failed = false;
const check = (ok, label, extra = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${extra ? '  ' + extra : ''}`); if (!ok) failed = true; };

const worker = await startScriptedWorker({ model: 'cloud-model' });
const home = mkdtempSync(join(tmpdir(), 'strata-appr-'));
const ws = join(home, 'workspace'); mkdirSync(ws, { recursive: true });
const port = await freePort();
const env = { ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), STRATA_WORKSPACE: ws, STRATA_REAL_ENGINE: '0', STRATA_COMPONENTS_DIR: join(home, 'components') };
const child = spawn(process.execPath, [join(HOST_DIR, 'server', 'index.mjs')], { cwd: HOST_DIR, env, stdio: 'ignore', windowsHide: true });
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/agent/status`)).ok) break; } catch { /* starting */ } await sleep(200); }
const saved = await (await fetch(`${base}/api/model-providers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ kind: 'custom', name: 'Scripted API', baseUrl: worker.url, protocol: 'openai-chat', apiKey: 'sk-approval-test-0123456789', models: [{ id: 'cloud-model', name: 'Cloud model', contextMax: 64000 }] }) })).json();
check(!!saved.connections?.[0]?.id, 'a scripted API model is connected');

const CASES = [
  { call: { command: 'Start-Process explorer.exe -ArgumentList "C:\\Users\\Ana\\Documents"', description: 'Open Documents' }, tool: 'pwsh', question: /Open your Documents folder in File Explorer\?/, details: /Start-Process explorer\.exe/ },
  { call: { file_path: join(ws, 'notes.txt').replaceAll('\\', '/'), content: 'a\nb\nc\n' }, tool: 'write', question: /Create the file notes\.txt\?/, details: /notes\.txt/ },
  { call: { command: 'Start-Process "C:\\Users\\Ana\\Downloads\\setup-1.4.2.exe" -ArgumentList "/S"', description: 'Run the installer' }, tool: 'pwsh', question: /Run a program: setup-1\.4\.2\.exe\?/, details: /setup-1\.4\.2\.exe/ },
];

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
for (const scheme of ['light', 'dark']) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 860 }, colorScheme: scheme });
  const page = await context.newPage();
  const problems = []; page.on('pageerror', e => problems.push(e.message));
  await page.goto(`${base}/#/chat`);
  await page.waitForTimeout(2000);
  const skip = page.getByText('Go straight to the app');
  if (await skip.count()) { await skip.click(); await page.waitForTimeout(600); }
  await page.getByRole('button', { name: 'New chat' }).first().click();
  await page.waitForTimeout(600);
  await page.getByRole('button', { name: /Choose model/ }).first().click();
  await page.getByRole('menuitemradio', { name: /Cloud model/ }).first().click();
  let n = 0;
  for (const c of CASES) {
    n += 1;
    await page.locator('textarea').first().fill(`SCRIPT:call ${c.tool} ${JSON.stringify(c.call)}`);
    await page.getByRole('button', { name: 'Send' }).click();
    const card = page.locator('.askcard').first();
    await card.waitFor({ timeout: 120000 });
    const head = ((await card.locator('.askcard__q').textContent()) ?? '').trim();
    check(c.question.test(head), `${scheme}: the card asks "${head}"`);
    check(!/\b(pwsh|bash|write|sandbox|json|powershell)\b/i.test(head) || /Run a program/.test(head), `${scheme}: the headline has no tool name or jargon`);
    const box = await card.boundingBox();
    check(!!box && box.height < 110, `${scheme}: the card is compact`, String(Math.round(box?.height ?? 0)) + ' px');
    check(await card.locator('details').evaluate(d => !d.open), `${scheme}: Details starts closed`);
    await card.locator('summary').click();
    const exact = ((await card.locator('pre').textContent()) ?? '');
    check(c.details.test(exact), `${scheme}: the exact command is under Details`, exact.slice(0, 80));
    if (n === 1) await page.screenshot({ path: join(SHOTS, `approval-${scheme}.png`) });
    await page.getByRole('button', { name: 'Deny' }).click();
    await page.waitForFunction(() => document.documentElement.dataset.streaming !== 'true', null, { timeout: 120000 }).catch(() => {});
    await page.waitForTimeout(400);
  }
  check(problems.length === 0, `${scheme}: no page error`, problems.join('; '));
  await context.close();
}
await browser.close();
if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }); else child.kill('SIGTERM');
await worker.close();
try { rmSync(home, { recursive: true, force: true, maxRetries: 3 }); } catch { /* the Harness sandbox leaves entries; best effort */ }
process.exit(failed ? 1 : 0);
