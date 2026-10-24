#!/usr/bin/env node
/**
 * Reading-measure check.
 *
 *   node scripts/measure.mjs
 *
 * Long-form reading wants 45–75 characters per line, and ~66 is the classic
 * target. This asserts the transcript stays inside that band at every content
 * font size the settings offer, in both normal and focus mode — the property
 * most likely to silently regress when a width or a padding changes.
 *
 * Note: `--st-measure` is expressed in `ch`, but `ch` is the width of the "0"
 * glyph, which is narrower than average running text in GT Walsheim. Measured
 * characters therefore run ~1.2× the ch value, which is why the assertion is on
 * the rendered result rather than on the token.
 *
 * Requires the dev server. Exits non-zero if any size falls outside 45–75.
 */
import { chromium } from 'playwright-core';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const BASE = process.env.STRATA_URL ?? 'http://localhost:5199';
const MIN = 45;
const MAX = 75;

const executablePath = [
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  join(homedir(), 'Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell'),
].find((p) => existsSync(p));

if (!executablePath) {
  console.error('No cached Chromium found.');
  process.exit(2);
}

const browser = await chromium.launch({ executablePath, args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1512, height: 1110 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));

/** Characters per line of assistant prose, measured from the rendered element. */
const measure = async () => {
  await page.locator('.st-msg--assistant .st-bubble p').first().scrollIntoViewIfNeeded();
  return page.evaluate(() => {
    const para = document.querySelector('.st-msg--assistant .st-bubble p');
    if (!para) return null;
    const fs = getComputedStyle(para).fontSize;
    const span = document.createElement('span');
    span.style.cssText = `position:absolute;visibility:hidden;white-space:nowrap;font:400 ${fs} var(--st-font)`;
    span.textContent = 'abcdefghijklmnopqrstuvwxyz'.repeat(8);
    document.body.appendChild(span);
    const avg = span.getBoundingClientRect().width / 208;
    span.remove();
    return { chars: Math.round(para.getBoundingClientRect().width / avg), fontSize: fs };
  });
};

async function setSize(px) {
  await page.goto(`${BASE}/#/settings`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.st-slider', { timeout: 8000 });
  // React tracks the previous value of a controlled input, so assigning .value
  // directly is ignored. Go through the native setter, then dispatch.
  await page.evaluate((value) => {
    const el = document.querySelector('.st-slider');
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    setter.call(el, String(value));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, px);
  await page.waitForTimeout(250);
  // Confirm the setting actually took, so this check can never pass vacuously.
  const applied = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--st-content-fs').trim(),
  );
  if (applied !== `${px}px`) {
    throw new Error(`text size did not apply: asked for ${px}px, --st-content-fs is "${applied}"`);
  }
}

async function prepareConversation() {
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1300);
  await page.click('button.nav-new');
  await page.waitForTimeout(300);
  await page.fill('textarea', 'read the readme');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(6500);
}

const rows = [];
await prepareConversation();

for (const size of [10, 16, 22]) {
  await setSize(size);
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1000);
  rows.push([`${size}px · normal`, await measure()]);

  await page.keyboard.press('Meta+Shift+F');
  await page.waitForTimeout(700);
  rows.push([`${size}px · focus`, await measure()]);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(400);
}

await browser.close();

let failures = 0;
console.log('reading measure (target 45–75 characters per line)\n');
for (const [label, m] of rows) {
  if (!m) {
    console.log(`  ${label.padEnd(16)} no prose found`);
    failures += 1;
    continue;
  }
  const ok = m.chars >= MIN && m.chars <= MAX;
  if (!ok) failures += 1;
  console.log(`  ${label.padEnd(16)} ${String(m.chars).padStart(3)} chars/line at ${m.fontSize}  ${ok ? 'PASS' : 'FAIL'}`);
}
if (errors.length) {
  failures += errors.length;
  console.log(`\n  page errors: ${errors.join('; ')}`);
}
console.log(failures ? `\nreading measure: ${failures} failure(s)` : '\nreading measure: every size inside the comfortable band');
process.exit(failures ? 1 : 0);
