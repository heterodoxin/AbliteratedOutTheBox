/**
 * Screenshot the running app with the Playwright browser already on this machine.
 *
 *   node scripts/shoot.mjs <out.png> <url> [width] [height] [settleMs] [waitSelector]
 *
 * Uses playwright-core (no browser download) against the cached Chromium build.
 */
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { homedir } from 'node:os';

const CANDIDATES = [
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  join(homedir(), 'Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell'),
];

const executablePath = CANDIDATES.find((p) => existsSync(p));
if (!executablePath) {
  console.error('No cached Chromium found. Looked in:\n' + CANDIDATES.join('\n'));
  process.exit(1);
}

const [, , out, url, w = '1512', h = '1110', settle = '1400', selector] = process.argv;
if (!out || !url) {
  console.error('usage: node scripts/shoot.mjs <out.png> <url> [w] [h] [settleMs] [waitSelector]');
  process.exit(1);
}

mkdirSync(dirname(resolve(out)), { recursive: true });

const browser = await chromium.launch({
  executablePath,
  args: ['--no-sandbox', '--disable-gpu', '--force-device-scale-factor=1'],
});
const page = await browser.newPage({ viewport: { width: Number(w), height: Number(h) }, deviceScaleFactor: 1 });

const problems = [];
page.on('console', (msg) => {
  if (msg.type() === 'error') problems.push(`console: ${msg.text()}`);
});
page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
page.on('requestfailed', (req) => {
  if (!req.url().includes('/api/events')) problems.push(`requestfailed: ${req.url()} ${req.failure()?.errorText ?? ''}`);
});

await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20000 });
if (selector) await page.waitForSelector(selector, { timeout: 12000 }).catch(() => problems.push(`missing selector: ${selector}`));
await page.waitForTimeout(Number(settle));

await page.screenshot({ path: resolve(out) });
await browser.close();

console.log(`${resolve(out)}${problems.length ? `\n  ${problems.slice(0, 12).join('\n  ')}` : ''}`);
if (problems.length) process.exitCode = 2;
