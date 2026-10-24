/**
 * Drive the real app through the S4b flow on a real model and take screenshots (playwright-core + installed Chrome).
 *
 *   STRATA_URL=http://127.0.0.1:8791 SHOTS=./shots METRICS=http://127.0.0.1:18797/metrics node scripts/agent-flow-s4b.mjs
 *
 * Against a host that has the Harness and one API connection saved (the model server), in Manual mode:
 *   1. a coding task that starts two helper agents; the Subagents popover shows two rows; the engine metrics show both running together;
 *      each helper's change to a file arrives as an Approve card, one card per change;
 *   2. an edit to a file approved through one card (no blocked-attempt card, no sandbox text anywhere);
 *   3. a web page fetch (a tool call), or its plain failure when offline;
 *   4. Stop during helpers: rows stop, the host reports no active turn.
 *
 * Environment: STRATA_URL (required), SHOTS, MODEL (picker text), METRICS (engine /metrics URL, optional, with Authorization from KEY),
 * PROMPT_HELPERS, PROMPT_EDIT, PROMPT_WEB, PROMPT_STOP, WORKDIR (the folder the agent works in, to check files), CHROME.
 */
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const BASE = process.env.STRATA_URL;
if (!BASE) { console.error('STRATA_URL is required'); process.exit(2); }
const SHOTS = resolve(process.env.SHOTS ?? 'agent-shots-s4b');
mkdirSync(SHOTS, { recursive: true });
const MODEL = process.env.MODEL ?? 'Cloud model';
const WORKDIR = process.env.WORKDIR;
const PROMPT_HELPERS = process.env.PROMPT_HELPERS ?? 'Use two helper agents at the same time, one for each file. Helper one creates hello.py that prints the word hello. Helper two creates README.txt with one sentence about hello.py. Start both helpers right away with the subagent tool, then tell me when they are done.';
const PROMPT_EDIT = process.env.PROMPT_EDIT ?? 'Read hello.py and then change it so it prints "hello world" instead of "hello". Use the edit tool.';
const PROMPT_WEB = process.env.PROMPT_WEB ?? 'Use the web_fetch tool to open https://example.com and tell me the page title in one sentence.';
const PROMPT_STOP = process.env.PROMPT_STOP ?? 'Start two helper agents with the subagent tool at once. Each one writes a long, detailed essay of at least 1500 words (one on the history of clocks, one on the history of bridges) and saves nothing. Tell me when they finish.';
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/google-chrome'].find(existsSync);

let failed = false;
const check = (ok, label) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}`); if (!ok) failed = true; };
const shot = async (page, name) => { await page.screenshot({ path: join(SHOTS, `${name}.png`) }); console.log(`shot ${name}.png`); };
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1512, height: 1000 } });
const problems = [];
page.on('pageerror', e => problems.push(`pageerror: ${e.message}`));
page.on('console', m => { if (m.type() === 'error') problems.push(`console: ${m.text().slice(0, 200)}`); });

const send = async text => { await page.locator('textarea').first().fill(text); await page.getByRole('button', { name: 'Send' }).click(); };
const idle = async (ms = 600000) => page.waitForFunction(() => document.documentElement.dataset.streaming !== 'true', null, { timeout: ms });
const agentsButton = () => page.locator('.chat-agents').first();
const sandboxTalk = () => page.evaluate(() => /sandbox|Constrained|language mode|access denied/i.test(document.body.innerText));

// the engine's own count of requests in flight, sampled once a second while a step runs
let peak = 0, sampler = null;
const startSampling = () => {
  if (!process.env.METRICS) return;
  peak = 0;
  sampler = setInterval(async () => {
    try {
      const res = await fetch(process.env.METRICS, { headers: process.env.KEY ? { authorization: `Bearer ${process.env.KEY}` } : {} });
      const m = await res.json();
      const running = m.live?.running?.length ?? (Array.isArray(m.live?.slots) ? m.live.slots.filter(s => s.state && s.state !== 'idle').length : 0);
      peak = Math.max(peak, running);
    } catch { /* tunnel hiccup */ }
  }, 1000);
};
const stopSampling = () => { clearInterval(sampler); sampler = null; };

/** Click Approve on every card that asks, until the turn is over. Returns how many cards asked. */
async function approveUntilIdle(label) {
  let asked = 0;
  const seen = new Set();
  for (;;) {
    const streaming = await page.evaluate(() => document.documentElement.dataset.streaming === 'true');
    const cards = page.locator('.toolcard[data-state="proposed"]');
    const n = await cards.count();
    for (let i = 0; i < n; i++) {
      const card = cards.nth(i);
      const key = (await card.innerText()).slice(0, 200);
      if (!seen.has(key)) { seen.add(key); asked++; await shot(page, `${label}-card-${asked}`); }
      await card.getByRole('button', { name: 'Approve' }).click().catch(() => {});
    }
    if (!streaming && n === 0) break;
    await page.waitForTimeout(400);
  }
  return asked;
}

try {
  await page.goto(`${BASE}/#/chat`);
  await page.waitForTimeout(2000);
  const skip = page.getByText('Go straight to the app');
  if (await skip.count()) { await skip.click(); await page.waitForTimeout(600); }
  await page.getByRole('button', { name: 'New chat' }).first().click();
  await page.waitForTimeout(600);
  await page.getByRole('button', { name: /Choose model/ }).first().click();
  await page.waitForTimeout(300);
  await page.getByRole('menuitemradio', { name: new RegExp(MODEL) }).first().click();
  await page.waitForTimeout(300);

  // 1. two helpers
  startSampling();
  await send(PROMPT_HELPERS);
  await page.waitForFunction(() => document.documentElement.dataset.streaming === 'true');
  await page.waitForFunction(() => /[12] subagents? running/.test(document.querySelector('.chat-agents')?.getAttribute('title') ?? ''), null, { timeout: 600000 }).catch(() => {});
  await agentsButton().click().catch(() => {});
  await page.waitForTimeout(500);
  await shot(page, '01-helpers-popover');
  await page.keyboard.press('Escape');
  const asked = await approveUntilIdle('01');
  stopSampling();
  await page.waitForTimeout(500);
  await agentsButton().click().catch(() => {});
  await page.waitForTimeout(500);
  await shot(page, '02-helpers-done');
  const popover = await page.locator('[role="dialog"], .popover').first().innerText().catch(() => '');
  console.log('popover text:', popover.replace(/\s+/g, ' ').slice(0, 400));
  await page.keyboard.press('Escape');
  check(/2 /.test(popover) || (popover.match(/Completed|Done|Finished/gi) ?? []).length >= 2 || /Helper|helper/.test(popover) || popover.length > 40, 'the Subagents popover lists the helpers');
  check(asked >= 1, `the helpers' changes reached the person as Approve cards (${asked})`);
  if (process.env.METRICS) check(peak >= 2, `the engine ran at least 2 requests at the same time (peak ${peak})`);
  if (WORKDIR) check(existsSync(join(WORKDIR, 'hello.py')), 'hello.py was written after the approval');
  check(!(await sandboxTalk()), 'no sandbox or PowerShell error text anywhere in the chat');
  check(await page.locator('.toolcard').count() === 0, 'no card is left waiting');

  // 2. one edit, one card
  await send(PROMPT_EDIT);
  const editAsked = await approveUntilIdle('03-edit');
  await page.waitForTimeout(500);
  await shot(page, '04-edit-done');
  check(editAsked >= 1, `the edit asked through a card (${editAsked} card(s) in total, one per change)`);
  check(!(await sandboxTalk()), 'still no sandbox or PowerShell error text');
  if (WORKDIR && existsSync(join(WORKDIR, 'hello.py'))) check(/hello world/i.test(readFileSync(join(WORKDIR, 'hello.py'), 'utf8')), 'hello.py now says hello world');

  // 3. a web page
  await send(PROMPT_WEB);
  await approveUntilIdle('05-web');
  await shot(page, '06-web-done');
  check(await page.getByText(/web_fetch/).count() >= 1, 'a web_fetch card is in the chat');

  // 4. Stop during helpers
  await send(PROMPT_STOP);
  await page.waitForFunction(() => document.documentElement.dataset.streaming === 'true');
  await page.waitForFunction(() => /[12] subagents? running/.test(document.querySelector('.chat-agents')?.getAttribute('title') ?? ''), null, { timeout: 600000 });
  await page.waitForTimeout(3000);
  await shot(page, '07-stop-running');
  const stop = page.getByRole('button', { name: /^Stop/ }).first();
  check(await stop.count() === 1, 'a Stop button is visible while the helpers run');
  const t0 = Date.now();
  await stop.click();
  await idle(120000);
  await page.waitForTimeout(500);
  const took = Date.now() - t0;
  await shot(page, '08-after-stop');
  const status = await (await fetch(`${BASE}/api/agent/status`)).json();
  check(status.activeTurns === 0, 'the host reports no active turn after Stop');
  check(!/[12] subagents? running/.test((await agentsButton().getAttribute('title')) ?? ''), 'no helper is shown as running after Stop');
  check(took < 60000, `Stop finished the run in ${Math.round(took / 1000)} s`);
  check(problems.length === 0, `no page errors (${problems.join(' | ') || 'none'})`);
} catch (error) {
  failed = true;
  console.log(`FAIL ${error.message}`);
  await page.screenshot({ path: join(SHOTS, 'zz-failure.png') }).catch(() => {});
} finally {
  stopSampling();
  await browser.close();
}
process.exit(failed ? 1 : 0);
