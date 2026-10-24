/**
 * The three slash commands through the real app on a real model, with screenshots (playwright-core + installed Chrome).
 *
 *   STRATA_URL=http://127.0.0.1:8899 SHOTS=./shots WORKDIR=<the host's workspace> MODEL='Qwen test' node scripts/agent2-flow.mjs
 *
 * Steps (each prints ok/FAIL):
 *   1. typing "/" lists /time-skip, /goal and /loop with plain descriptions;
 *   2. /loop 1m <message>: the first answer arrives, the loop repeats it a minute later, /loop shows it, /loop stop ends it;
 *   3. /goal <objective>: the assistant works until the goal is complete (Manual mode: Approve cards are clicked), /goal shows it;
 *   4. /time-skip <request>: a job runs on a copy, the sidebar lists the changed file, Keep the update changes the file in WORKDIR,
 *      Put the original back restores it.
 * Environment: STRATA_URL (required), SHOTS, WORKDIR, MODEL (picker text), CHROME, SKIP (comma list of steps to skip: menu,loop,goal,timeskip),
 * LOOP_WAIT_MS (how long to wait for the second loop run, default 150000).
 */
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const BASE = process.env.STRATA_URL;
if (!BASE) { console.error('STRATA_URL is required'); process.exit(2); }
const SHOTS = resolve(process.env.SHOTS ?? 'agent2-shots');
mkdirSync(SHOTS, { recursive: true });
const WORKDIR = process.env.WORKDIR;
const MODEL = process.env.MODEL ?? 'Qwen test';
const SKIP = new Set((process.env.SKIP ?? '').split(',').filter(Boolean));
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);

let failed = false;
const check = (ok, label) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}`); if (!ok) failed = true; };
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1512, height: 1000 } });
const problems = [];
page.on('pageerror', e => problems.push(`pageerror: ${e.message}`));
page.on('console', m => { if (m.type() === 'error') problems.push(`console: ${m.text().slice(0, 200)}`); });
const shot = async name => { await page.waitForTimeout(400); await page.screenshot({ path: join(SHOTS, `${name}.png`) }); console.log(`shot ${name}.png`); };
const send = async text => { await page.locator('textarea').first().fill(text); await page.getByRole('button', { name: 'Send' }).click(); };
const idle = async (ms = 900000) => page.waitForFunction(() => document.documentElement.dataset.streaming !== 'true', null, { timeout: ms });
const started = async () => page.waitForFunction(() => document.documentElement.dataset.streaming === 'true', null, { timeout: 30000 });
const text = () => page.evaluate(() => document.querySelector('.chat')?.innerText ?? document.body.innerText);

/** Approve every card that asks until the turn is over. */
async function approveUntilIdle(label) {
  let asked = 0; const seen = new Set();
  for (;;) {
    const streaming = await page.evaluate(() => document.documentElement.dataset.streaming === 'true');
    const cards = page.locator('.toolcard[data-state="proposed"]');
    const n = await cards.count();
    for (let i = 0; i < n; i++) {
      const card = cards.nth(i);
      const key = (await card.innerText()).slice(0, 200);
      if (!seen.has(key)) { seen.add(key); asked++; await shot(`${label}-approve-${asked}`); }
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

  // 1. the menu
  if (!SKIP.has('menu')) {
    await page.locator('textarea').first().fill('/');
    await page.waitForSelector('.slashmenu', { timeout: 5000 });
    const menu = await page.locator('.slashmenu').innerText();
    await shot('01-slash-menu');
    check(/\/time-skip/.test(menu) && /Try a change on a copy/.test(menu), '/time-skip is listed with its plain description');
    check(/\/goal/.test(menu) && /keep working until it is done/.test(menu), '/goal is listed with its plain description');
    check(/\/loop/.test(menu) && /every few minutes/.test(menu), '/loop is listed with its plain description');
    await page.locator('textarea').first().fill('/lo');
    await shot('01b-slash-menu-filtered');
    await page.locator('textarea').first().fill('');
  }

  // 2. /loop
  if (!SKIP.has('loop')) {
    await send('/loop 1m Reply with exactly three words about the sea.');
    await page.waitForTimeout(1500);
    await shot('02a-loop-started');
    let body = await text();
    check(/Loop started: .Reply with exactly three words about the sea\.. every minute/.test(body), 'a note says the loop started and how often');
    await started().catch(() => {});
    await idle();
    await shot('02b-loop-first-answer');
    const firstCount = (await text()).split('Reply with exactly three words about the sea.').length - 1;
    const t0 = Date.now();
    const wait = Number(process.env.LOOP_WAIT_MS ?? 150000);
    await page.waitForFunction(n => (document.querySelector('.chat')?.innerText ?? '').split('Reply with exactly three words about the sea.').length - 1 > n, firstCount, { timeout: wait }).catch(() => {});
    await idle();
    await shot('02c-loop-second-run');
    const secondCount = (await text()).split('Reply with exactly three words about the sea.').length - 1;
    check(secondCount > firstCount, `the loop sent the message again on its own (${Math.round((Date.now() - t0) / 1000)} s after the first answer)`);
    await send('/loop');
    await page.waitForTimeout(800);
    check(/A loop is running in this chat/.test(await text()), '/loop says a loop is running');
    await send('/loop stop');
    await page.waitForTimeout(800);
    await shot('02d-loop-stopped');
    check(/The loop is stopped/.test(await text()), '/loop stop says it is stopped');
    const afterStop = (await text()).split('Reply with exactly three words about the sea.').length - 1;
    await page.waitForTimeout(75000);
    await idle();
    check((await text()).split('Reply with exactly three words about the sea.').length - 1 === afterStop, 'nothing more was sent after stopping');
  }

  // 3. /goal
  if (!SKIP.has('goal')) {
    await page.getByRole('button', { name: 'New chat' }).first().click();
    await page.waitForTimeout(600);
    await page.getByRole('button', { name: /Choose model/ }).first().click();
    await page.waitForTimeout(300);
    await page.getByRole('menuitemradio', { name: new RegExp(MODEL) }).first().click();
    await page.waitForTimeout(300);
    await send('/goal Create a file called goal-done.txt in the current folder that contains the single word done, then confirm it exists.');
    await started().catch(() => {});
    const asked = await approveUntilIdle('03');
    await idle();
    await shot('03a-goal-finished');
    const body = await text();
    check(!!WORKDIR && existsSync(join(WORKDIR, 'goal-done.txt')), 'the goal produced goal-done.txt');
    check(/goal is complete|goal-done\.txt/i.test(body), 'the chat says the goal is complete or names the file');
    console.log(`approval cards: ${asked}`);
    await send('/goal');
    await page.waitForTimeout(1500);
    await shot('03b-goal-status');
  }

  // 4. /time-skip
  if (!SKIP.has('timeskip')) {
    await page.getByRole('button', { name: 'New chat' }).first().click();
    await page.waitForTimeout(600);
    await page.getByRole('button', { name: /Choose model/ }).first().click();
    await page.waitForTimeout(300);
    await page.getByRole('menuitemradio', { name: new RegExp(MODEL) }).first().click();
    await page.waitForTimeout(300);
    const notes = WORKDIR ? join(WORKDIR, 'notes.md') : null;
    const before = notes && existsSync(notes) ? readFileSync(notes, 'utf8') : null;
    await send('/time-skip Add a second line to notes.md that says: second line added.');
    await page.waitForTimeout(2500);
    await shot('04a-timeskip-working');
    await page.waitForFunction(() => /Ready to compare|Update kept|Saved version kept|could not be completed|stopped|interrupted/i.test(document.querySelector('#time-skip-content .ts-heading')?.innerText ?? ''), null, { timeout: 1200000 }).catch(() => {});
    await shot('04b-timeskip-ready');
    const panel = await page.locator('#time-skip-content').innerText().catch(() => '');
    check(/Ready to compare/.test(panel), 'the sidebar says the result is ready to compare');
    check(/notes\.md/.test(panel), 'the sidebar names the changed file');
    check(!notes || readFileSync(notes, 'utf8') === before, 'your file is untouched until you choose');
    await page.getByRole('button', { name: 'Compare' }).first().click().catch(() => {});
    await page.waitForTimeout(800);
    await shot('04c-timeskip-compare');
    await page.getByRole('button', { name: /Back to chat|Done comparing/ }).first().click().catch(() => {});
    await page.getByRole('button', { name: 'Keep the update' }).first().click();
    await page.waitForTimeout(4000);
    await shot('04d-timeskip-kept');
    check(!notes || /second line added/i.test(readFileSync(notes, 'utf8')), 'Keep the update put the revision in your file');
    await page.getByRole('button', { name: 'Put the original back' }).first().click();
    await page.waitForTimeout(4000);
    await shot('04e-timeskip-restored');
    check(!notes || readFileSync(notes, 'utf8') === before, 'Put the original back restored your file');
  }
} catch (e) {
  console.log('ERROR', e.message);
  await shot('zz-error').catch(() => {});
  failed = true;
} finally {
  const text = problems.filter(p => !/Failed to load resource|ERR_|favicon/.test(p));
  console.log(`page problems: ${text.length}`); text.slice(0, 8).forEach(p => console.log('  ' + p));
  await browser.close();
}
process.exit(failed ? 1 : 0);
