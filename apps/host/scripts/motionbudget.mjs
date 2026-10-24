#!/usr/bin/env node
/**
 * Strata motion budget.
 *
 *   node scripts/motionbudget.mjs
 *
 * The plans ask for one thing about the scroll-driven effects that a static
 * audit cannot answer: what do they cost while you actually scroll? Six rules
 * are bound to the transcript's scroll timeline, and "it looks smooth" is not a
 * measurement.
 *
 * So this scrolls a real transcript twice — once as shipped, once with the
 * scroll-driven animations switched off — and compares p95 frame time. The
 * budget is 1.2x, taken from plan 15. A ratio rather than an absolute figure
 * because absolute frame times depend on the machine running the check.
 *
 * Requires the dev server.
 */
import { chromium } from 'playwright-core';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const BASE = process.env.STRATA_URL ?? 'http://localhost:5199';
const BUDGET = 1.2;

const executablePath = [
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  join(homedir(), 'Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell'),
].find((p) => existsSync(p));

if (!executablePath) {
  console.error('No cached Chromium found. Run `npm run shot -- x.png <url>` once to confirm the harness.');
  process.exit(2);
}

const browser = await chromium.launch({ executablePath, args: ['--no-sandbox'] });

/**
 * Confirms the harness can see a cost at all, before trusting it to report one.
 *
 * This was written after the check passed with a deliberately expensive effect
 * injected — blur, saturation and scale bound to every message's scroll
 * timeline — and the ratio did not move. In this environment, programmatic
 * scrolling does not appear to drive scroll-timeline paint cost, so the
 * measurement cannot distinguish cheap from expensive here.
 *
 * A budget check that cannot detect a six-fold increase in per-frame work is
 * worse than no check: it reports "within budget" with confidence it has not
 * earned. So it measures a known-expensive case first and says so when it
 * cannot see the difference.
 */
async function canDiscriminate() {
  const page = await browser.newPage({ viewport: { width: 1512, height: 700 } });
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  await page.click('button.nav-new');
  await page.waitForTimeout(400);
  for (const m of ['write a paragraph about tides', 'and another about the moon']) {
    await page.fill('textarea', m);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(8000);
    await page.keyboard.press('Enter').catch(() => {});
    await page.waitForTimeout(3500);
  }
  await scrollProfile(page);
  const before = await scrollProfile(page);
  await page.addStyleTag({
    content: `@keyframes probe-expensive { from { filter: blur(0px); } to { filter: blur(8px) saturate(2); } }
              .st-msg { scroll-timeline: --probe block; animation: probe-expensive linear both; animation-timeline: --probe; }`,
  });
  await scrollProfile(page);
  const after = await scrollProfile(page);
  await page.close();
  return { before: stats(before.frames), after: stats(after.frames) };
}

/**
 * Scrolls the transcript and returns the frame deltas.
 *
 * Deliberately drives the scroll from inside the page on rAF: driving it over
 * the protocol would measure the round-trip rather than the rendering.
 */
async function scrollProfile(page) {
  return page.evaluate(
    () =>
      new Promise((resolve) => {
        const el = document.querySelector('.main__body.chat');
        if (!el) return resolve(null);
        const frames = [];
        let last = performance.now();
        const started = last;
        let dir = 1;
        const step = () => {
          const now = performance.now();
          frames.push(now - last);
          last = now;
          // sweep down then up, so both directions are represented
          el.scrollTop += dir * 24;
          if (el.scrollTop <= 0) dir = 1;
          if (el.scrollTop >= el.scrollHeight - el.clientHeight) dir = -1;
          if (frames.length < 240) requestAnimationFrame(step);
          else resolve({ frames: frames.slice(5), wall: performance.now() - started });
        };
        requestAnimationFrame(step);
      }),
  );
}

/**
 * What is measured, and why not p95 frame time.
 *
 * The first version of this compared p95 frame deltas and reported a ratio of
 * exactly 1.000 — then still reported 1.006 against a deliberately expensive
 * scroll-driven effect (blur, saturation and scale on every message). p95 sits
 * at the 60fps frame cap in both runs, so it is saturated: any cost that fits
 * inside a frame reads as free. A check that cannot detect a six-fold increase
 * in per-frame work is not measuring the thing it claims to.
 *
 * What does discriminate is **total scripting time for a fixed amount of
 * scrolling**, which accumulates work that individual frames hide.
 */
const stats = (xs) => {
  const sorted = [...xs].sort((a, b) => a - b);
  const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
  return { p50: at(0.5), p95: at(0.95), total: xs.reduce((a, b) => a + b, 0), n: xs.length };
};

async function makeTranscript() {
  const page = await browser.newPage({ viewport: { width: 1512, height: 700 } });
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  await page.click('button.nav-new');
  await page.waitForTimeout(400);
  for (const m of ['write a paragraph about tides', 'and another about the moon', 'and one about the wind']) {
    await page.fill('textarea', m);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(8000);
    await page.keyboard.press('Enter').catch(() => {});
    await page.waitForTimeout(3500);
  }
  return page;
}

/* ---- run ---------------------------------------------------------------- */

// First: can this harness see a cost at all? Without this the check would
// report "within budget" whether or not it was measuring anything.
const probe = await canDiscriminate();
const probeRatio = probe.before.total > 0 ? probe.after.total / probe.before.total : 1;
const blind = probeRatio < 1.05;

console.log('self-check: known-expensive effect');
console.log(`  total frame time ${probe.before.total.toFixed(0)}ms -> ${probe.after.total.toFixed(0)}ms (${probeRatio.toFixed(3)}x)`);

if (blind) {
  console.log('  the harness cannot see a deliberately expensive effect, so it cannot');
  console.log('  measure the real one either. Reporting nothing rather than passing.');
  console.log('\nmotion budget: NOT MEASURED — see plan 15');
  await browser.close();
  process.exit(3);
}
console.log('  detected, so the measurement below is meaningful\n');

const page = await makeTranscript();
const room = await page.evaluate(() => {
  const el = document.querySelector('.main__body.chat');
  return el ? el.scrollHeight - el.clientHeight : 0;
});

if (room < 200) {
  console.log(`motion budget: not enough transcript to scroll (${room}px) — skipped`);
  await browser.close();
  process.exit(0);
}

// warm up so the first pass is not paying for layout the second one gets free
await scrollProfile(page);
const shipped = await scrollProfile(page);

// the same page, with the scroll-driven animations switched off
await page.addStyleTag({
  content: `*, *::before, *::after { animation-timeline: none !important; animation-name: none !important; }`,
});
await scrollProfile(page);
const withoutEffects = await scrollProfile(page);

await browser.close();

const a = stats(shipped.frames);
const b = stats(withoutEffects.frames);
// Compared on total frame time for the same number of scroll steps.
const ratio = b.total > 0 ? a.total / b.total : 1;
// Reported alongside, to show whether the p95 was saturated.
const p95Ratio = b.p95 > 0 ? a.p95 / b.p95 : 1;

console.log(`scroll room           : ${Math.round(room)}px`);
console.log(`frames sampled        : ${a.n} (${Math.round(shipped.wall)}ms as shipped, ${Math.round(withoutEffects.wall)}ms without)`);
console.log(`total frame time      : ${a.total.toFixed(0)}ms shipped vs ${b.total.toFixed(0)}ms without`);
console.log(`p95 frame             : ${a.p95.toFixed(2)}ms vs ${b.p95.toFixed(2)}ms (ratio ${p95Ratio.toFixed(3)}x — usually saturated at the frame cap)`);
console.log(`ratio                 : ${ratio.toFixed(3)}x (budget ${BUDGET}x)`);

if (ratio > BUDGET) {
  console.log(`\nmotion budget: FAIL — scroll-driven effects cost ${ratio.toFixed(2)}x, over the ${BUDGET}x budget`);
  process.exit(1);
}
console.log('\nmotion budget: within budget');
