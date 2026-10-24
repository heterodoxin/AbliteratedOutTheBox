#!/usr/bin/env node
/**
 * Motion capture and interruptibility check.
 *
 *   node scripts/motion.mjs
 *
 * Two jobs:
 *
 *  1. Record the canonical interactions and sample frames from each recording
 *     into a contact sheet (`.shots/motion-<name>.png`), so choreography can be
 *     reviewed as pictures rather than argued about.
 *  2. Assert the thing a screenshot cannot show: that a surface toggled
 *     mid-flight *reverses* rather than queueing or jumping to the end.
 *
 * Requires the dev server and a cached Chromium. The bundled ffmpeg that ships
 * with Playwright is used to extract frames.
 */
import { chromium } from 'playwright-core';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { existsSync, mkdirSync, rmSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { homedir } from 'node:os';

const run = promisify(execFile);
const BASE = process.env.STRATA_URL ?? 'http://localhost:5199';
const ROOT = resolve(process.env.STRATA_SHOTS ?? '.shots');
const TMP = join(ROOT, 'motion-frames');
mkdirSync(TMP, { recursive: true });

const executablePath = [
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  join(homedir(), 'Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell'),
].find((p) => existsSync(p));

const ffmpeg = [
  join(homedir(), 'Library/Caches/ms-playwright/ffmpeg-1011/ffmpeg-mac'),
  '/opt/homebrew/bin/ffmpeg',
].find((p) => existsSync(p));

if (!executablePath) {
  console.error('No cached Chromium found.');
  process.exit(2);
}

const W = 1280;
const H = 720;
const failures = [];

/** Record one interaction, then sample frames from the resulting webm. */
async function clip(browser, name, script) {
  const dir = join(TMP, name);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });

  const t0 = Date.now();
  const context = await browser.newContext({
    viewport: { width: W, height: H },
    recordVideo: { dir, size: { width: W, height: H } },
  });
  const page = await context.newPage();
  page.on('pageerror', (e) => failures.push(`${name}: page error — ${e.message}`));
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  // Recording starts at context creation, so the first second is page load.
  // Each clip reports when its interaction actually happened, and frames are
  // sampled from that window — otherwise every sheet shows an idle app.
  const started = Date.now();
  await script(page);
  const window = { from: (started - t0) / 1000, to: (Date.now() - t0) / 1000 };
  const video = page.video();
  await context.close();

  const videoPath = video ? await video.path() : null;
  if (!videoPath || !existsSync(videoPath)) {
    failures.push(`${name}: no recording produced`);
    return null;
  }

  if (!ffmpeg) {
    console.log(`  ${name}: recorded, but no ffmpeg found to sample frames`);
    return videoPath;
  }

  // The ffmpeg Playwright bundles is a stripped build (no `fps` or `hstack`
  // filters), so frames are taken by seeking to a timestamp and grabbing one.
  const span = Math.max(0.6, window.to - window.from);
  const stamps = Array.from({ length: 6 }, (_, i) => Number((window.from + (span * i) / 5).toFixed(2)));
  const frames = [];
  for (const t of stamps) {
    const out = join(dir, `t${String(t).replace('.', '_')}.png`);
    await run(ffmpeg, ['-y', '-ss', String(t), '-i', videoPath, '-frames:v', '1', out]).catch(() => {});
    if (existsSync(out)) frames.push(out);
  }
  if (!frames.length) failures.push(`${name}: ffmpeg produced no frames`);
  return { videoPath, frames, window };
}

/**
 * Stitch sampled frames into one image. Done in Python rather than ffmpeg
 * because the bundled build has no `hstack` filter.
 */
async function contactSheet(name, frames) {
  if (!frames?.length) return;
  const out = join(ROOT, `motion-${name}.png`);
  const script = `
import sys
from PIL import Image
paths, out = sys.argv[1:-1], sys.argv[-1]
ims = [Image.open(p) for p in paths]
w = sum(i.width for i in ims)
h = max(i.height for i in ims)
sheet = Image.new('RGB', (w, h), '#ffffff')
x = 0
for im in ims:
    sheet.paste(im, (x, 0))
    x += im.width
sheet.save(out)
`;
  await run('python3', ['-c', script, ...frames, out]).catch(() =>
    failures.push(`${name}: could not build the contact sheet`),
  );
  if (existsSync(out)) console.log(`  ${name}: ${frames.length} frames -> ${out}`);
}

const browser = await chromium.launch({ executablePath, args: ['--no-sandbox'] });
console.log('recording interactions\n');

const clips = {
  'route-change': async (page) => {
    await page.keyboard.press('Meta+3');
    await page.waitForTimeout(700);
    await page.keyboard.press('Meta+5');
    await page.waitForTimeout(700);
  },
  'drawer-toggle': async (page) => {
    await page.click('button.nav-new');
    await page.waitForTimeout(400);
    await page.getByRole('button', {name:'Add a task',exact:true}).click();
    await page.waitForTimeout(600);
    await page.getByRole('button', {name:/Close .*task/i}).click();
    await page.waitForTimeout(600);
  },
  'palette-toggle': async (page) => {
    await page.keyboard.press('Meta+k');
    await page.waitForTimeout(600);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(600);
  },
  disclosure: async (page) => {
    await page.click('button.nav-new');
    await page.waitForTimeout(300);
    await page.fill('textarea', 'read the readme');
    await page.keyboard.press('Enter');
    await page.waitForTimeout(7000);
    const details = page.locator('.st-collapse').first();
    if (await details.count()) {
      await details.locator('summary').click();
      await page.waitForTimeout(400);
      await details.locator('summary').click();
      await page.waitForTimeout(500);
    }
  },
};

for (const [name, script] of Object.entries(clips)) {
  const result = await clip(browser, name, script);
  if (result && typeof result === 'object') {
    console.log(`  ${name}: sampling ${result.window.from.toFixed(2)}s–${result.window.to.toFixed(2)}s`);
    await contactSheet(name, result.frames);
  }
}

/* ---------------------------------------------------- interruptibility ---- */

/**
 * Open a surface, reverse it mid-flight, and confirm it comes to rest in the
 * reversed state. A queued or keyframe-driven animation fails this: it finishes
 * the entrance first and only then plays the exit.
 */
async function reversesMidFlight(label, open, close, readState) {
  const page = await browser.newPage({ viewport: { width: W, height: H } });
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  await open(page);
  await page.waitForTimeout(60); // deliberately inside the entrance
  await close(page);
  await page.waitForTimeout(500); // longer than any transition on the scale
  const settled = await readState(page);
  await page.close();
  if (!settled) failures.push(`${label}: did not settle in the reversed state`);
  console.log(`  ${label}: reversed mid-flight — ${settled ? 'settled closed' : 'STUCK OPEN'}`);
}

/* A step that throws must still leave the other checks reported — a crash
   before the summary tells the reader nothing about what did pass. */
try {
  await reversesMidFlight(
    'task dialog',
    (p) => p.click('button.nav-new').then(() => p.waitForTimeout(300)).then(() => p.getByRole('button', {name:'Add a task',exact:true}).click()),
    (p) => p.getByRole('button', {name:/Close .*task/i}).click(),
    (p) =>
      p.evaluate(() => {
        const el = document.querySelector('.dialog-layer');
        if (!el) return true;
        return getComputedStyle(el).visibility === 'hidden';
      }),
  );

  await reversesMidFlight(
    'command palette',
    (p) => p.keyboard.press('Meta+k'),
    (p) => p.keyboard.press('Escape'),
    (p) =>
      p.evaluate(() => {
        const el = document.querySelector('.modal');
        return !!el && !el.matches('[data-open="true"]') && getComputedStyle(el).visibility === 'hidden';
      }),
  );

  /**
   * One loop per region.
   *
   * Several indefinite animations running in the same part of the screen read as
   * a fault rather than as activity — two pulsing things side by side have no
   * hierarchy, and the eye cannot settle on either. Each screen is checked for
   * regions containing more than one looping animation.
   *
   * "Region" is the nearest ancestor that is a card, a panel or the page body, so
   * a spinner inside a list row is judged against that row rather than against
   * everything on the screen.
   */
  async function loopBudget(route, theme) {
    const page = await browser.newPage({ viewport: { width: 1512, height: 1110 } });
    await page.goto(`${BASE}/#/${route}`, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(1500);
    if (theme === 'dark') {
      /* Set the theme the way the app does rather than by clicking a control a
         modal may be covering — the setup screen renders one over the whole
         window, and clicking through it timed out. */
      await page.evaluate(() => {
        const raw = JSON.parse(localStorage.getItem('strata.prefs.v2') ?? '{}');
        localStorage.setItem('strata.prefs.v2', JSON.stringify({ ...raw, theme: 'dark' }));
      });
      await page.reload({ waitUntil: 'domcontentloaded' });
      await page.waitForTimeout(1200);
    }
    const out = await page.evaluate(() => {
      const loops = [...document.querySelectorAll('*')].filter((el) => {
        const cs = getComputedStyle(el);
        if (!cs.animationIterationCount.split(',').some((v) => v.trim() === 'infinite')) return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden';
      });
      const regions = new Map();
      for (const el of loops) {
        const region =
          el.closest('.st-card, .st-drawer, section, aside, table') ?? document.querySelector('.main__body') ?? document.body;
        const label = region.className?.toString().split(' ').slice(0, 2).join('.') || region.tagName.toLowerCase();
        regions.set(label, (regions.get(label) ?? 0) + 1);
      }
      return [...regions.entries()].map(([label, n]) => ({ label, n }));
    });
    await page.close();
    return out;
  }

  for (const theme of ['light', 'dark']) {
    // Every route a reader can reach, not just the ones the reverse checks visit.
  for (const route of ['chat', 'models', 'tuning', 'monitor', 'serving', 'workspace', 'skills', 'settings', 'setup']) {
      const regions = await loopBudget(route, theme);
      const over = regions.filter((r) => r.n > 1);
      for (const r of over) failures.push(`${theme} · ${route}: ${r.n} loops at once in "${r.label}"`);
    }
  }
} catch (err) {
  failures.push(`a step threw before finishing: ${err instanceof Error ? err.message.split('\n')[0] : err}`);
}

await browser.close();

/* The summary has to name what was checked. Saying "every surface reverses
   mid-flight" when the failure was two spinners in one card would send the
   reader looking in the wrong place. */
console.log(
  failures.length
    ? `\nmotion: ${failures.length} failure(s)\n  ${failures.join('\n  ')}`
    : [
        '',
        'motion: every surface reverses mid-flight',
        'motion: one loop per region on every screen, both themes',
      ].join('\n'),
);
process.exit(failures.length ? 1 : 0);
