/**
 * Settings -> Tools and what it gives the assistant, through the real app on a real model, with screenshots (playwright-core + Chrome).
 *
 *   STRATA_URL=http://127.0.0.1:8899 SHOTS=./shots MODEL='Qwen test' node scripts/agent2-tools-flow.mjs
 *
 * The host must run with STRATA_BROWSER_HEADLESS=1 (the browser step must not open a window on the person's desktop) and a scratch
 * STRATA_HOME / STRATA_TIMESKIP_DIR. Steps (each prints ok/FAIL):
 *   1. Settings -> Tools shows the groups; two tools of my own are added through the dialog (instructions, command);
 *   2. web search with no key: "Searched the web" card, real DuckDuckGo results in the answer;
 *   3. the instruction tool is used without asking and shapes the answer; the command tool asks first in Manual, runs after Approve;
 *   4. Search the web switched off: the assistant has no search tool any more;
 *   5. the browser switched on: Manual asks before opening https://example.com, the answer names the page's heading.
 */
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

const BASE = process.env.STRATA_URL;
if (!BASE) { console.error('STRATA_URL is required'); process.exit(2); }
const SHOTS = resolve(process.env.SHOTS ?? 'agent2-tools-shots');
mkdirSync(SHOTS, { recursive: true });
const MODEL = process.env.MODEL ?? 'Qwen test';
const SKIP = new Set((process.env.SKIP ?? '').split(',').filter(Boolean));
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);

let failed = false;
const check = (ok, label) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}`); if (!ok) failed = true; };
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1360, height: 980 } });
const problems = [];
page.on('pageerror', e => problems.push(`pageerror: ${e.message}`));
page.on('console', m => { if (m.type() === 'error') problems.push(`console: ${m.text().slice(0, 200)}`); });
const shot = async name => { await page.waitForTimeout(500); await page.screenshot({ path: join(SHOTS, `${name}.png`) }); console.log(`shot ${name}.png`); };
const send = async text => { await page.locator('textarea').first().fill(text); await page.getByRole('button', { name: 'Send' }).click(); };
const idle = async (ms = 900000) => page.waitForFunction(() => document.documentElement.dataset.streaming !== 'true', null, { timeout: ms });
const started = async () => page.waitForFunction(() => document.documentElement.dataset.streaming === 'true', null, { timeout: 30000 });
const chatText = () => page.evaluate(() => document.querySelector('.chat')?.innerText ?? document.body.innerText);

async function approveUntilIdle(label) {
  let asked = 0; const seen = new Set(); const reasons = [];
  for (;;) {
    const streaming = await page.evaluate(() => document.documentElement.dataset.streaming === 'true');
    const cards = page.locator('.toolcard[data-state="proposed"]');
    const n = await cards.count();
    for (let i = 0; i < n; i++) {
      const card = cards.nth(i);
      const key = (await card.innerText()).slice(0, 200);
      if (!seen.has(key)) { seen.add(key); asked++; reasons.push(key.replace(/\s+/g, ' ')); await shot(`${label}-approve-${asked}`); }
      await card.getByRole('button', { name: 'Approve' }).click().catch(() => {});
    }
    if (!streaming && n === 0) break;
    await page.waitForTimeout(400);
  }
  return { asked, reasons };
}
async function openTools() {
  await page.goto(`${BASE}/#/settings`);
  await page.waitForSelector('#settings-tab-Tools', { timeout: 20000 });
  await page.locator('#settings-tab-Tools').evaluate(el => el.click());
  await page.waitForSelector('.settings-tools', { timeout: 10000 });
  await page.waitForSelector('.settings-tools [role="switch"]', { timeout: 10000 });
}
async function newChat() {
  await page.goto(`${BASE}/#/chat`);
  await page.waitForTimeout(1200);
  await page.getByRole('button', { name: 'New chat' }).first().click();
  await page.waitForTimeout(600);
  await page.getByRole('button', { name: /Choose model/ }).first().click();
  await page.waitForTimeout(300);
  await page.getByRole('menuitemradio', { name: new RegExp(MODEL) }).first().click();
  await page.waitForTimeout(300);
}

try {
  await page.goto(`${BASE}/`);
  const skip = page.getByText('Go straight to the app');
  await skip.waitFor({ timeout: 8000 }).then(() => skip.click()).catch(() => {});

  // 1. the screen, and two tools of my own
  if (!SKIP.has('own')) {
    await openTools();
    await shot('01-tools');
    const names = await page.locator('.settings-tools .field__label, .settings-tools [class*="field__label"]').allInnerTexts();
    check(['Run commands', 'Search the web', 'Open web pages', 'Goals'].every(n => names.some(x => x.includes(n))), 'the groups are listed in plain words');
    await page.getByRole('button', { name: 'Add a tool' }).click();
    await page.getByRole('dialog').getByLabel('Name').fill('House style');
    await page.getByRole('dialog').getByLabel('What it does').fill('Our house style for every answer.');
    await page.getByRole('dialog').getByLabel('Instructions').fill('End your answer with the single word FINISHED on a line of its own.');
    await shot('02-add-instructions');
    await page.getByRole('button', { name: 'Save tool' }).click();
    await page.waitForTimeout(800);
    await page.getByRole('button', { name: 'Add a tool' }).click();
    await page.getByRole('dialog').getByLabel('Name').fill('Say hello');
    await page.getByRole('dialog').getByLabel('What it does').fill('Prints a greeting from this computer.');
    await page.getByRole('dialog').getByLabel('How it works').click();
    await page.getByRole('option', { name: 'Run a command' }).click();
    await page.getByRole('dialog').getByLabel('Command to run').fill('echo hello from my own tool');
    await shot('03-add-command');
    await page.getByRole('button', { name: 'Save tool' }).click();
    await page.waitForTimeout(800);
    await page.getByRole('heading', { name: 'Your own tools' }).evaluate(el => el.scrollIntoView({ block: 'start' }));
    await shot('04-own-tools');
    const own = await page.locator('.settings-tools').innerText();
    check(/House style/.test(own) && /Say hello/.test(own), 'both tools of my own are listed');
  }

  // 2. web search with no key
  if (!SKIP.has('search')) {
    await newChat();
    await send('Use the web_search tool to find the GitHub page of the Strata local LLM engine by Niko1221. Tell me its address.');
    await started().catch(() => {});
    const r = await approveUntilIdle('05');
    await idle();
    await shot('05-search');
    const body = await chatText();
    check(/Searched the web/.test(body), 'a "Searched the web" card appeared');
    check(r.asked === 0, 'searching asked nothing (it only reads)');
    check(/github\.com\/Niko1221\/Strata/i.test(body), 'the answer names the real GitHub address from the results');
  }

  // 3. my own tools
  if (!SKIP.has('use')) {
    await newChat();
    await send('Use my House style tool, then tell me what 2 plus 2 is.');
    await started().catch(() => {});
    const a = await approveUntilIdle('06');
    await idle();
    await shot('06-house-style');
    const body = await chatText();
    check(/Read your instructions/.test(body), 'a "Read your instructions" card appeared');
    check(a.asked === 0, 'an instruction tool asked nothing');
    check(/FINISHED\s*$/m.test(body.trim().split('\n').filter(l => !/tok\/s|total tokens|of context/.test(l)).join('\n')), 'the answer follows the instructions (ends with FINISHED)');
    await send('Now use my Say hello tool and tell me what it printed.');
    await started().catch(() => {});
    const b = await approveUntilIdle('07');
    await idle();
    await shot('07-say-hello');
    const after = await chatText();
    check(b.asked >= 1 && b.reasons.some(x => /Running your tool.*say hello/i.test(x)), `Manual asked first, in plain words (${b.reasons[0] ?? 'no card'})`);
    check(/hello from my own tool/i.test(after), 'after Approve the command ran and the answer has its output');
  }

  // 4. Search the web switched off
  if (!SKIP.has('off')) {
    await openTools();
    await page.getByRole('switch', { name: 'Search the web' }).click();
    await page.waitForTimeout(800);
    await shot('08-search-off');
    check((await page.getByRole('switch', { name: 'Search the web' }).getAttribute('aria-checked')) === 'false', 'the switch is off');
    await newChat();
    await send('Use the web_search tool to search for the word Strata. If you have no such tool, say exactly: NO SEARCH TOOL.');
    await started().catch(() => {});
    await approveUntilIdle('08');
    await idle();
    await shot('09-no-search');
    const body = await chatText();
    check(!/Searched the web|Searching the web/.test(body), 'no search card: the assistant has no search tool now');
    await openTools();
    await page.getByRole('switch', { name: 'Search the web' }).click();
    await page.waitForTimeout(500);
  }

  // 5. the browser
  if (!SKIP.has('browser')) {
    await openTools();
    await page.getByRole('heading', { name: 'Web browser' }).evaluate(el => el.scrollIntoView({ block: 'start' }));
    const sw = page.getByRole('switch', { name: 'Use a web browser' });
    check(await sw.isEnabled(), 'the browser switch is available (Chrome or Edge and the tool server were found)');
    if ((await sw.getAttribute('aria-checked')) !== 'true') await sw.click();
    await page.waitForTimeout(800);
    await shot('10-browser-on');
    await newChat();
    await send('Open https://example.com in the browser and tell me the big heading on that page.');
    await started().catch(() => {});
    const r = await approveUntilIdle('11');
    await idle();
    await shot('11-browser');
    const body = await chatText();
    check(r.asked >= 1 && r.reasons.some(x => /Opening a page in the browser.*example\.com/i.test(x)), `Manual asked before opening the page (${r.reasons[0] ?? 'no card'})`);
    check(/Opened a page in the browser/.test(body), 'a plain "Opened a page in the browser" card appeared');
    check(/Example Domain/.test(body), 'the answer names the page heading');
    // leave the settings as they were found
    await openTools();
    await page.getByRole('heading', { name: 'Web browser' }).evaluate(el => el.scrollIntoView({ block: 'start' }));
    if ((await page.getByRole('switch', { name: 'Use a web browser' }).getAttribute('aria-checked')) === 'true') await page.getByRole('switch', { name: 'Use a web browser' }).click();
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
