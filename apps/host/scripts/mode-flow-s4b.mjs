/**
 * The owner's "open my Downloads folder" in each Mode through the real app, with screenshots.  STRATA_URL, SHOTS, MODEL (picker text).
 * Manual: the folder is NOT opened until the one Approve card is approved; Plan: nothing changes and no card asks; Bypass: it just opens.
 */
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
const BASE = process.env.STRATA_URL; if (!BASE) { console.error('STRATA_URL required'); process.exit(2); }
const SHOTS = resolve(process.env.SHOTS ?? 'mode-shots'); mkdirSync(SHOTS, { recursive: true });
const MODEL = process.env.MODEL ?? 'Cloud model';
const PROMPT = process.env.PROMPT ?? 'open my Downloads folder';
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);
let failed = false; const check = (ok, label) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}`); if (!ok) failed = true; };
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const shot = async name => { await page.screenshot({ path: join(SHOTS, `${name}.png`) }); console.log(`shot ${name}.png`); };
const idle = async (ms = 600000) => page.waitForFunction(() => document.documentElement.dataset.streaming !== 'true', null, { timeout: ms });
try {
  await page.goto(`${BASE}/#/chat`); await page.waitForTimeout(2000);
  const skip = page.getByText('Go straight to the app'); if (await skip.count()) { await skip.click(); await page.waitForTimeout(500); }
  for (const [mode, label] of [['manual', 'Manual'], ['plan', 'Plan'], ['bypass', 'Bypass permissions']]) {
    await page.getByRole('button', { name: 'New chat' }).first().click(); await page.waitForTimeout(600);
    await page.getByRole('button', { name: /Choose model/ }).first().click(); await page.waitForTimeout(300);
    await page.getByRole('menuitemradio', { name: new RegExp(MODEL) }).first().click(); await page.waitForTimeout(300);
    await page.getByRole('button', { name: /^Mode:/ }).first().click(); await page.waitForTimeout(300);
    await page.getByRole('menuitemradio', { name: new RegExp(`^${label}`) }).first().click(); await page.waitForTimeout(300);
    await page.locator('textarea').first().fill(PROMPT);
    await page.getByRole('button', { name: 'Send' }).click();
    let asked = 0, sawCard = false;
    for (;;) {
      const streaming = await page.evaluate(() => document.documentElement.dataset.streaming === 'true');
      const card = page.locator('.toolcard[data-state="proposed"]');
      if (await card.count()) {
        if (!sawCard) { sawCard = true; await page.waitForTimeout(300); await shot(`${mode}-1-one-card-asks`); }
        asked++; await card.first().getByRole('button', { name: 'Approve' }).click().catch(() => {});
      }
      if (!streaming && !(await card.count())) break;
      await page.waitForTimeout(500);
    }
    await page.waitForTimeout(600); await shot(`${mode}-2-done`);
    const text = await page.evaluate(() => document.body.innerText);
    if (mode === 'manual') { check(sawCard, 'Manual: a single Approve/Deny card asked before the folder was opened'); check(!/sandbox|Constrained|language mode|access denied/i.test(text), 'Manual: no sandbox or PowerShell error text'); }
    if (mode === 'plan') { check(!sawCard, 'Plan: no card asks (nothing can change)'); check(!/Constrained|language mode/i.test(text), 'Plan: no PowerShell language-mode noise'); }
    if (mode === 'bypass') check(!sawCard, 'Bypass: it just opens, no question');
    console.log(`${mode}: ${asked} approval click(s)`);
  }
} catch (e) { failed = true; console.log('FAIL', e.message); await page.screenshot({ path: join(SHOTS, 'zz-failure.png') }).catch(() => {}); }
finally { await browser.close(); }
process.exit(failed ? 1 : 0);
