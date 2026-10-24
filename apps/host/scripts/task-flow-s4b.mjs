/**
 * Real-clock check of a scheduled task through the real app: creates a task due two minutes from now (the host's real scheduler runs it),
 * keeps the Tasks page open to capture the toast, then shows Last result. node scripts/task-flow-s4b.mjs  (STRATA_URL, SHOTS, MODE, PROMPT)
 */
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
const BASE = process.env.STRATA_URL; if (!BASE) { console.error('STRATA_URL required'); process.exit(2); }
const SHOTS = resolve(process.env.SHOTS ?? 'task-shots'); mkdirSync(SHOTS, { recursive: true });
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);
let failed = false; const check = (ok, label) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}`); if (!ok) failed = true; };
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const shot = async name => { await page.screenshot({ path: join(SHOTS, `${name}.png`) }); console.log(`shot ${name}.png`); };
try {
  await page.goto(`${BASE}/#/tasks`); await page.waitForTimeout(2500);
  const skip = page.getByText('Go straight to the app'); if (await skip.count()) { await skip.click(); await page.waitForTimeout(500); }
  const due = new Date(Date.now() + 2 * 60_000);
  const at = `${String(due.getHours()).padStart(2, '0')}:${String(due.getMinutes()).padStart(2, '0')}`;
  const created = await (await fetch(`${BASE}/api/tasks`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ title: process.env.TITLE ?? 'Open my downloads folder', prompt: process.env.PROMPT ?? 'Open my Downloads folder and tell me how many items are in it.', frequency: 'daily', at, weekday: 1, timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone, mode: process.env.MODE ?? 'bypass', enabled: true }) })).json();
  console.log(`task ${created.id} due ${created.nextRunAt} (local ${at})`);
  await page.reload(); await page.waitForTimeout(2000); await shot('01-task-scheduled');
  let final = null; const t0 = Date.now();
  for (let i = 0; i < 80 && !final; i++) {
    await page.waitForTimeout(5000);
    const t = (await (await fetch(`${BASE}/api/tasks`)).json()).tasks.find(x => x.id === created.id);
    if (t.status === 'running' && !globalThis.sawRunning) { globalThis.sawRunning = true; console.log(`running after ${Math.round((Date.now() - t0) / 1000)} s`); await shot('02-task-running'); }
    if (['done', 'error', 'cancelled'].includes(t.status)) final = t;
    if (await page.locator('.st-toast').count() && !globalThis.sawToast) { globalThis.sawToast = true; await shot('03-toast'); }
  }
  check(!!final && final.status === 'done', `the task ran by itself and finished (${final?.status}, ${final?.error ?? ''})`);
  console.log('result:', String(final?.output ?? '').replace(/\s+/g, ' ').slice(0, 400));
  check(/What it did/.test(final?.output ?? ''), 'the result lists what it did');
  await page.waitForTimeout(1500); await shot('04-toast-after');
  await page.goto(`${BASE}/#/tasks/${created.id}`); await page.waitForTimeout(2500); await shot('05-last-result');
  check(await page.getByText('Last result').count() >= 1, 'Last result is shown on the task page');
} catch (e) { failed = true; console.log('FAIL', e.message); await page.screenshot({ path: join(SHOTS, 'zz-failure.png') }).catch(() => {}); }
finally { await browser.close(); }
process.exit(failed ? 1 : 0);
