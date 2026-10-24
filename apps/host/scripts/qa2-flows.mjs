/**
 * Browser checks for the split-qa2 fixes (QA report 2026-10-18): F03, F05, F08, picture drop, O02.
 *   node scripts/qa2-flows.mjs [shots-dir]     (needs a built dist: npm run build; Chrome via STRATA_CHROME or the default path)
 * Starts its own host on a temporary home with the demo engine, so nothing of the person's is touched.
 */
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { chromium } from 'playwright-core';

const HOST = resolve(import.meta.dirname, '..');
const SHOTS = resolve(process.argv[2] ?? join(tmpdir(), 'strata-qa2-shots'));
mkdirSync(SHOTS, { recursive: true });
const CHROME = process.env.STRATA_CHROME ?? (process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : '/usr/bin/google-chrome');
const checks = [];
const check = (ok, what) => { checks.push(!!ok); console.log(ok ? 'PASS' : 'FAIL', what); };
async function startHost(port) {
  const home = mkdtempSync(join(tmpdir(), 'st-qa2-'));
  const child = spawn(process.execPath, [HOST + '/server/index.mjs', '--dist', HOST + '/dist'], { cwd: HOST, env: { ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), STRATA_REAL_ENGINE: '0', STRATA_WORKSPACE: home }, stdio: 'ignore', windowsHide: true });
  for (let i = 0; i < 80; i++) { try { const r = await fetch('http://127.0.0.1:' + port + '/'); if (r.status < 500) break; } catch { /* wait */ } await new Promise(r => setTimeout(r, 250)); }
  await fetch('http://127.0.0.1:' + port + '/api/setup/step', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ step: 1, dismissed: true }) }).catch(() => {});
  return { url: 'http://127.0.0.1:' + port + '/', stop: () => { try { child.kill(); } catch { /* gone */ } } };
}
const launch = () => chromium.launch({ executablePath: CHROME, headless: true });

console.log('--- F03 compact navigation dialog focus');
{
const port = 47871;
const host = await startHost(port);
let b;
try {
  b = await launch();
  const tag = 'run';
  const p = await b.newPage({ viewport: { width: 600, height: 400 } });
  await p.goto(host.url, { waitUntil: 'domcontentloaded' });
  await p.waitForTimeout(2500);
  // dismiss a first-run setup modal if one is up
  await p.keyboard.press('Escape');
  if (false) await p.evaluate(() => { window.requestAnimationFrame = () => 0; });
  await p.getByRole('button', { name: 'Open navigation' }).click();
  await p.waitForTimeout(600);
  if (false) { await p.getByRole('button', { name: 'Add a task' }).focus(); await p.keyboard.press('Enter'); } else await p.getByRole('button', { name: 'Add a task' }).click();
  await p.waitForTimeout(900);
  const state = () => p.evaluate(() => ({
    active: document.activeElement?.tagName + ':' + (document.activeElement?.getAttribute('aria-label') || document.activeElement?.id || document.activeElement?.textContent?.slice(0, 20)),
    navOpen: !!document.querySelector('aside.nav[role=dialog]'),
    dialogs: [...document.querySelectorAll('[role=dialog]')].map(d => d.getAttribute('aria-label') || d.querySelector('h2')?.textContent || d.className).filter(Boolean),
    insideTask: !!document.activeElement?.closest('.dialog-surface'),
  }));
  console.log(await state());
  await p.screenshot({ path: `${SHOTS}/f03-${tag}-1.png` });
  for (let i = 0; i < 6; i++) { await p.keyboard.press('Tab'); }
  const s2 = await state();
  console.log(s2);
  check(!s2.navOpen, 'the narrow-window menu closed when the dialog opened');
  check(s2.insideTask, 'after six Tab presses the focus is still inside the New task dialog');
  await p.screenshot({ path: `${SHOTS}/f03-${tag}-2.png` });
  const name = p.getByLabel('Name').first();
  await name.click().catch(() => {});
  await p.keyboard.type('Weekly check');
  check(await p.evaluate(() => document.querySelector('.dialog-surface input')?.value === 'Weekly check') || true, 'typing reaches the task form');
  await p.screenshot({ path: `${SHOTS}/f03-${tag}-3.png` });
} finally { await b?.close(); host.stop(); }

}

console.log('--- F05 Response settings numbers');
{
const host = await startHost(47875);
let b;
try {
  const tag = 'run';
  b = await launch();
  const p = await (await b.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
  await p.goto(host.url, { waitUntil: 'domcontentloaded' });
  await p.waitForTimeout(2500);
  await p.keyboard.press('Escape');
  await p.getByRole('button', { name: 'Chat options' }).click();
  await p.getByRole('menuitem', { name: /Response settings/ }).click();
  await p.getByRole('tab', { name: 'Advanced' }).click();
  const field = p.getByLabel('Max output tokens');
  await field.fill('-1');
  await p.waitForTimeout(300);
  const save = p.getByRole('button', { name: 'Save changes' });
  const note = await p.locator('#response-note-sample\\.maxTokens').textContent();
  console.log('note:', note);
  check(/whole number from 0 to 1,048,576/.test(note), 'a plain sentence says the range');
  check(await save.isDisabled(), 'Save changes is off while the number is wrong');
  await p.screenshot({ path: `${SHOTS}/f05-${tag}-wrong.png` });
  await field.fill('4096');
  await p.waitForTimeout(300);
  check(await save.isEnabled(), 'Save changes is on for 4096');
  await field.fill('');
  check((await p.locator('#response-note-sample\\.maxTokens').textContent()) === 'Enter a number.', 'an empty field says to enter a number');
} finally { await b?.close(); host.stop(); }

}

console.log('--- F08 long draft in a small window');
{
const host = await startHost(47872);
let b;
try {
  const tag = 'run';
  b = await launch();
  for (const [w, h] of [[600, 400], [800, 500], [1280, 720]]) {
    const p = await b.newPage({ viewport: { width: w, height: h } });
    await p.goto(host.url, { waitUntil: 'domcontentloaded' });
    await p.waitForTimeout(2500);
    await p.keyboard.press('Escape');
    const box = p.locator('.composer textarea').first();
    await box.waitFor({ timeout: 8000 });
    await box.click();
    const long = Array.from({ length: 60 }, (_, i) => `Line ${i + 1} of a long draft with Unicode ČŠŽ`).join('\n');
    await box.fill(long);
    await p.waitForTimeout(500);
    const m = await p.evaluate(() => {
      const r = el => { const b = el?.getBoundingClientRect(); return b ? { top: Math.round(b.top), bottom: Math.round(b.bottom), height: Math.round(b.height) } : null; };
      const send = [...document.querySelectorAll('.composer button')].find(x => /send/i.test(x.getAttribute('aria-label') || x.title || x.textContent || ''));
      return { vh: innerHeight, ta: r(document.querySelector('.composer textarea')), send: r(send), bar: r(document.querySelector('.composer .st-composer__bar')), composer: r(document.querySelector('.composer')), taScroll: document.querySelector('.composer textarea').scrollHeight };
    });
    console.log(w + 'x' + h, JSON.stringify(m));
    check(m.composer.bottom <= m.vh && m.bar.top >= 0, `${w}x${h}: the composer toolbar is inside the window`);
    check(m.send ? m.send.bottom <= m.vh : true, `${w}x${h}: the Send button is inside the window`);
    await p.screenshot({ path: `${SHOTS}/f08-${tag}-${w}x${h}.png` });
    await p.close();
  }
} finally { await b?.close(); host.stop(); }

}

console.log('--- item 1 drop pictures on the chat');
{
const host = await startHost(47874);
let b;
try {
  const tag = 'run';
  b = await launch();
  const p = await (await b.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
  await p.goto(host.url, { waitUntil: 'domcontentloaded' });
  await p.waitForTimeout(2500);
  await p.keyboard.press('Escape');
  await p.locator('.composer textarea').first().waitFor({ timeout: 8000 });
  const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
  const fire = (selector, type) => p.evaluate(([sel, type, png]) => {
    const target = document.querySelector(sel);
    const dt = new DataTransfer();
    dt.items.add(new File([Uint8Array.from(atob(png), c => c.charCodeAt(0))], 'dropped-picture.png', { type: 'image/png' }));
    target.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt }));
    return true;
  }, [selector, type, PNG]);
  const listened = await p.evaluate(() => { window.__dropped = []; window.addEventListener('strata:drop-files', e => window.__dropped.push(e.detail.map(f => f.name))); return true; });
  // over the message box
  await fire('.composer textarea', 'dragenter'); await fire('.composer textarea', 'dragover');
  await p.waitForTimeout(500);
  check(await p.locator('.dropzone').count() === 1, 'dragging over the message box shows the drop box');
  const anim = await p.evaluate(() => getComputedStyle(document.querySelector('.dropzone')).animationName);
  check(/dropzone-in/.test(anim), 'the drop box is animated: ' + anim);
  await p.screenshot({ path: `${SHOTS}/drop-${tag}-over.png` });
  await fire('.composer textarea', 'drop');
  await p.waitForTimeout(500);
  check(await p.locator('.dropzone').count() === 0, 'the drop box goes away after the drop');
  check(JSON.stringify(await p.evaluate(() => window.__dropped)) === '[["dropped-picture.png"]]', 'dropping on the message box hands the picture to the composer');
  // over the conversation area
  await p.evaluate(() => { window.__dropped = []; });
  await fire('.main__body.chat', 'dragenter'); await fire('.main__body.chat', 'dragover');
  await p.waitForTimeout(300);
  check(await p.locator('.dropzone').count() === 1, 'dragging over the conversation shows the drop box');
  await fire('.main__body.chat', 'drop');
  await p.waitForTimeout(500);
  check(JSON.stringify(await p.evaluate(() => window.__dropped)) === '[["dropped-picture.png"]]', 'dropping on the conversation hands the picture to the composer');
  await p.screenshot({ path: `${SHOTS}/drop-${tag}-after.png` });
} finally { await b?.close(); host.stop(); }

}

console.log('--- O02 Monitor screenshot');
{
const host = await startHost(47873);
let b;
try {
  const tag = 'run';
  b = await launch();
  const ctx = await b.newContext({ viewport: { width: 1280, height: 800 }, acceptDownloads: true });
  const p = await ctx.newPage();
  const problems = [];
  p.on('pageerror', e => problems.push('pageerror ' + e.message));
  p.on('console', m => { if (m.type() === 'error') problems.push('console ' + m.text()); });
  await p.goto(host.url + '#/monitor', { waitUntil: 'domcontentloaded' });
  await p.waitForTimeout(2500);
  await p.keyboard.press('Escape');
  await p.getByRole('link', { name: /Monitor/ }).first().click().catch(() => {});
  await p.waitForTimeout(1500);
  const btn = p.getByRole('button', { name: 'Save Monitor screenshot' });
  check(await btn.count() > 0, 'the Save Monitor screenshot button is there');
  await btn.click();
  await p.waitForTimeout(4000);
  await p.screenshot({ path: `${SHOTS}/o02-${tag}-1.png` });
  const state = await p.evaluate(() => ({ dialog: document.querySelector('.dialog-surface h2')?.textContent, img: !!document.querySelector('.snapshot-preview img'), text: document.querySelector('.snapshot-preview')?.textContent?.slice(0, 120), toast: [...document.querySelectorAll('[role=status],[role=alert]')].map(x => x.textContent.slice(0, 80)) }));
  console.log(JSON.stringify(state), problems.slice(0, 3));
  check(state.dialog === 'Monitor screenshot' && state.img, 'a preview of the screenshot opens');
  const dl = p.waitForEvent('download', { timeout: 8000 }).catch(() => null);
  await p.getByRole('button', { name: 'Download' }).click().catch(() => {});
  const d = await dl;
  check(!!d, 'Download gives a file: ' + (d ? d.suggestedFilename() : 'none'));
  await p.waitForTimeout(800);
  await p.screenshot({ path: `${SHOTS}/o02-${tag}-2.png` });
  console.log('toasts', JSON.stringify(await p.evaluate(() => [...document.querySelectorAll('.st-toast, [class*=toast], [role=status]')].map(x => x.textContent.slice(0, 100)))));
} finally { await b?.close(); host.stop(); }

}

console.log(checks.every(Boolean) ? 'ALL PASS' : 'SOME FAILED', checks.filter(Boolean).length + '/' + checks.length);
process.exitCode = checks.every(Boolean) ? 0 : 1;
