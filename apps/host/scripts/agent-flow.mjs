/**
 * Drive the real app through an agent turn and take screenshots (playwright-core + installed Chrome).
 *
 *   STRATA_URL=http://127.0.0.1:8791 SHOTS=./shots node scripts/agent-flow.mjs
 *
 * What it does, against a host that has the Harness installed and one API connection saved:
 *   1. opens the app, picks the API model in the chat picker (the choice is for this chat only);
 *   2. sends a prompt whose reply is a tool call, waits for the approval card, clicks Approve;
 *   3. sends a long reply and clicks Stop mid-stream;
 *   4. reloads the page, reopens the chat and sends one more message (resume, no replayed tool).
 *
 * Environment:
 *   STRATA_URL     host base URL (required)
 *   SHOTS          output directory for screenshots (default ./agent-shots)
 *   MODEL          text of the API model in the picker (default "Cloud model")
 *   PROMPT_TOOL    prompt that makes the model call a tool
 *   PROMPT_LONG    prompt that makes the model write a long reply
 *   PROMPT_AFTER   prompt sent after reopening
 *   CHROME         path to Chrome (default: the usual Windows/macOS/Linux install)
 *
 * It prints one line per checkpoint and exits non-zero if a checkpoint fails.
 */
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

const BASE = process.env.STRATA_URL;
if (!BASE) { console.error('STRATA_URL is required'); process.exit(2); }
const SHOTS = resolve(process.env.SHOTS ?? 'agent-shots');
mkdirSync(SHOTS, { recursive: true });
const MODEL = process.env.MODEL ?? 'Cloud model';
const PROMPT_TOOL = process.env.PROMPT_TOOL ?? 'SCRIPT:tool notes.txt hello from the agent';
const PROMPT_LONG = process.env.PROMPT_LONG ?? 'SCRIPT:slow 400';
const PROMPT_AFTER = process.env.PROMPT_AFTER ?? 'SCRIPT:echo still here after reopening';
const CHROME = process.env.CHROME ?? [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
].find(existsSync);

let failed = false;
const check = (ok, label) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}`); if (!ok) failed = true; };
const shot = async (page, name) => { await page.screenshot({ path: join(SHOTS, `${name}.png`) }); console.log(`shot ${name}.png`); };

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1512, height: 1000 } });
const problems = [];
page.on('pageerror', e => problems.push(`pageerror: ${e.message}`));
page.on('console', m => { if (m.type() === 'error') problems.push(`console: ${m.text().slice(0, 200)}`); });

const send = async text => {
  const box = page.locator('textarea').first();
  await box.fill(text);
  await page.getByRole('button', { name: 'Send' }).click();
};
const idle = async (ms = 120000) => page.waitForFunction(() => document.documentElement.dataset.streaming !== 'true', null, { timeout: ms });

try {
  await page.goto(`${BASE}/#/chat`);
  await page.waitForTimeout(2000);
  const skip = page.getByText('Go straight to the app');
  if (await skip.count()) { await skip.click(); await page.waitForTimeout(600); }
  await page.getByRole('button', { name: 'New chat' }).first().click();
  await page.waitForTimeout(600);
  await page.getByRole('button', { name: /Choose model/ }).first().click();
  await page.waitForTimeout(300);
  await shot(page, '01-picker');
  await page.getByRole('menuitemradio', { name: new RegExp(MODEL) }).first().click();
  await page.waitForTimeout(300);
  check((await page.getByRole('button', { name: /Choose model/ }).first().getAttribute('aria-label'))?.includes(MODEL), `the picker shows ${MODEL} for this chat`);

  // 2. a tool call that needs approval
  await send(PROMPT_TOOL);
  await page.locator('.toolcard[data-state="proposed"]').first().waitFor({ timeout: 120000 });
  await page.waitForTimeout(300);
  await shot(page, '02-approval-card');
  check(await page.getByRole('button', { name: 'Approve' }).count() === 1, 'an Approve button is on the card');
  check(await page.getByText('Needs your approval').count() === 1, 'the card says it needs approval, not "Demo approval"');
  await page.getByRole('button', { name: 'Approve' }).click();
  await idle();
  await page.waitForTimeout(500);
  await shot(page, '03-after-approve');
  check(await page.locator('.tool-summary').count() >= 1, 'the tool finished and collapsed into its summary');
  check(await page.locator('.toolcard').count() === 0, 'no card is left waiting');

  // 3. Stop mid-stream
  const baseline = await page.evaluate(() => document.body.innerText.length);
  await send(PROMPT_LONG);
  await page.waitForFunction(() => document.documentElement.dataset.streaming === 'true');
  // Stop while text is arriving, not during the prompt read: wait until the page has grown by a few lines.
  await page.waitForFunction(b => document.body.innerText.length > b + 400, baseline, { timeout: 300000 });
  const stop = page.getByRole('button', { name: /^Stop/ }).first();
  check(await stop.count() === 1, 'a Stop button is visible while the turn runs');
  await shot(page, '04-streaming');
  await stop.click();
  await idle(60000);
  await page.waitForTimeout(500);
  await shot(page, '05-after-stop');
  const afterStop = await (await fetch(`${BASE}/api/agent/status`)).json();
  check(afterStop.activeTurns === 0, 'the host reports no active turn after Stop (the turn really ended)');
  check(await page.getByRole('button', { name: 'Send' }).count() === 1, 'the composer is free again');

  // 4. reload, reopen, continue
  const toolsBefore = await page.locator('.tool-summary').count();
  await page.reload();
  await page.waitForTimeout(2000);
  await page.locator('a.navsession').first().click();
  await page.waitForTimeout(800);
  await shot(page, '06-reopened');
  check(await page.locator('.tool-summary').count() === toolsBefore && toolsBefore >= 1, `reopening shows the same ${toolsBefore} finished tool card(s) and no waiting card`);
  await send(PROMPT_AFTER);
  await idle();
  await page.waitForTimeout(500);
  await shot(page, '07-after-reopen');
  check(await page.locator('.toolcard').count() === 0, 'no tool is waiting after the reopened turn');
  if (process.env.EXPECT_AFTER) check(await page.getByText(new RegExp(process.env.EXPECT_AFTER)).count() >= 1, `the reopened chat answered (${process.env.EXPECT_AFTER})`);
  // 5. the model choice belongs to the chat: a new chat starts on the local model again
  await page.getByRole('button', { name: 'New chat' }).first().click();
  await page.waitForTimeout(600);
  const fresh = await page.getByRole('button', { name: /Choose model/ }).first().getAttribute('aria-label');
  check(!fresh?.includes(MODEL), `a new chat does not inherit the API model (picker says: ${fresh})`);
  await shot(page, '08-new-chat-model');

  // 6. two chats at once (optional: a real model has one inference slot, so the second just queues)
  if (process.env.TWO_CHATS === '1') {
    await page.locator('a.navsession').nth(1).click().catch(() => {});
    await page.getByRole('button', { name: 'New chat' }).first().click();
    await page.getByRole('button', { name: /Choose model/ }).first().click();
    await page.getByRole('menuitemradio', { name: new RegExp(MODEL) }).first().click();
    await send(process.env.PROMPT_A ?? 'SCRIPT:slow 150');
    await page.waitForFunction(() => document.documentElement.dataset.streaming === 'true');
    await page.waitForTimeout(800);
    await page.getByRole('button', { name: 'New chat' }).first().click();
    await page.getByRole('button', { name: /Choose model/ }).first().click();
    await page.getByRole('menuitemradio', { name: new RegExp(MODEL) }).first().click();
    check(await page.getByRole('button', { name: 'Send' }).count() === 1, 'the second chat can send while the first is mid-turn');
    await send(process.env.PROMPT_B ?? 'SCRIPT:echo the other chat');
    await page.getByText(process.env.EXPECT_B ?? 'the other chat').first().waitFor({ timeout: 60000 });
    const stillRunning = await page.evaluate(() => document.documentElement.dataset.streaming);
    await shot(page, '09-two-chats-second-done');
    check(stillRunning === 'true', 'the first chat is still running when the second one has answered');
    await idle();
    await page.locator('a.navsession').first().click();
    await page.waitForTimeout(500);
    await shot(page, '10-two-chats-first-done');
    check(await page.getByText(/w149/).count() >= 1, 'the first chat finished its own stream');
  }
  check(problems.length === 0, `no page errors (${problems.join(' | ') || 'none'})`);
} catch (error) {
  failed = true;
  console.log(`FAIL ${error.message}`);
  await page.screenshot({ path: join(SHOTS, 'zz-failure.png') }).catch(() => {});
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
