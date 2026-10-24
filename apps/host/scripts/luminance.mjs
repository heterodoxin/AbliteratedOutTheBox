#!/usr/bin/env node
/**
 * Dark-mode brightness budget, and material layering.
 *
 *   node scripts/luminance.mjs
 *
 * The complaint this exists to prevent: "the green buttons are too rough on the
 * eye if you keep working longer". It was correct — the primary button used to
 * be a solid #34d399 at relative luminance 0.496 against a page at 0.005, making
 * the brightest and most saturated object in the interface the one thing you are
 * invited to click, usually several times a screen.
 *
 * So: in dark mode, no *large* surface may exceed a luminance ceiling. Small
 * elements are exempt — a status dot or a 2px progress bar is meant to be vivid,
 * and it is the area of a bright fill that fatigues, not its existence.
 *
 * It also asserts the Liquid Glass layering rule: **no material may be a
 * descendant of another material.** Glass cannot sample glass — nesting blurs an
 * already-blurred backdrop, doubling the compositor cost for a muddier result.
 * The slash menu was doing exactly this inside the composer.
 *
 * Requires the dev server.
 */
import { chromium } from 'playwright-core';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const BASE = process.env.STRATA_URL ?? 'http://localhost:5199';
/** Anything at or above this projected area is "large". Roughly a 64×64 block. */
const LARGE_AREA = 4000;

/**
 * And it has to be large in *both* directions.
 *
 * Area alone called a 501×8px progress fill a "large bright surface" — the same
 * verdict as a 64×64 block, for a hairline. Worse, a progress fill's width is
 * its value, so the check passed or failed on the same build depending on how
 * full the bar happened to be. A minimum thickness keeps blocks and drops
 * hairlines, meters and slivers.
 */
const MIN_THICKNESS = 32;
/** Ceiling for the relative luminance of a large surface, in dark mode. */
const CEILING = 0.15;

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

/**
 * Walk every element, composite its background over its ancestors, and report
 * the large bright ones. Runs in the page.
 */
const SURVEY = ({ largeArea, minThickness, ceiling }) => {
  // Chrome serialises color-mix() results as `color(srgb r g b)` with 0–1
  // components. Reading those as 0–255 made a tinted fill look almost black,
  // which silently defeated both the contrast and luminance measurements.
  const parse = (c) => {
    const s = String(c);
    const srgb = s.match(/^color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+))?\)/);
    if (srgb) {
      return { r: +srgb[1] * 255, g: +srgb[2] * 255, b: +srgb[3] * 255, a: srgb[4] === undefined ? 1 : +srgb[4] };
    }
    const n = s.match(/[\d.]+/g);
    if (!n) return null;
    return { r: +n[0], g: +n[1], b: +n[2], a: n[3] === undefined ? 1 : +n[3] };
  };
  const lum = ({ r, g, b }) => {
    const f = (v) => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
    };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };
  const bgOf = (el) => {
    const stack = [];
    let n = el;
    while (n) {
      const c = parse(getComputedStyle(n).backgroundColor);
      if (c && c.a > 0) stack.push(c);
      if (c && c.a === 1) break;
      n = n.parentElement;
    }
    if (!stack.length || stack.at(-1).a < 1) stack.push({ r: 14, g: 17, b: 19, a: 1 });
    let acc = stack.at(-1);
    for (let i = stack.length - 2; i >= 0; i -= 1) {
      const f = stack[i];
      acc = {
        r: f.r * f.a + acc.r * (1 - f.a),
        g: f.g * f.a + acc.g * (1 - f.a),
        b: f.b * f.a + acc.b * (1 - f.a),
        a: 1,
      };
    }
    return acc;
  };

  const offenders = [];
  for (const el of document.querySelectorAll('*')) {
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || +cs.opacity === 0) continue;
    const rect = el.getBoundingClientRect();
    const area = rect.width * rect.height;
    if (area < largeArea) continue;
    if (Math.min(rect.width, rect.height) < minThickness) continue;
    // Only judge elements that are actually painted: a transparent wrapper
    // inherits its parent's colour and would double-count.
    const own = parse(cs.backgroundColor);
    if (!own || own.a === 0) continue;
    const L = lum(bgOf(el));
    if (L > ceiling) {
      offenders.push({
        selector: el.className && typeof el.className === 'string'
          ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.')
          : el.tagName.toLowerCase(),
        luminance: Number(L.toFixed(3)),
        area: Math.round(area),
        colour: cs.backgroundColor,
      });
    }
  }
  // Brightest first, and collapse duplicates of the same selector.
  const seen = new Map();
  for (const o of offenders.sort((a, b) => b.luminance - a.luminance)) {
    if (!seen.has(o.selector)) seen.set(o.selector, o);
  }

  // Material layering: a glass surface must not sit inside another one.
  const nested = [];
  for (const el of document.querySelectorAll('*')) {
    const bdf = getComputedStyle(el).backdropFilter;
    if (!bdf || bdf === 'none') continue;
    let n = el.parentElement;
    while (n && n !== document.body) {
      const outer = getComputedStyle(n).backdropFilter;
      if (outer && outer !== 'none') {
        nested.push({
          inner: '.' + (String(el.className).trim().split(/\s+/)[0] || el.tagName),
          outer: '.' + (String(n.className).trim().split(/\s+/)[0] || n.tagName),
        });
        break;
      }
      n = n.parentElement;
    }
  }

  return { bright: [...seen.values()], nested: [...new Map(nested.map((n) => [n.inner + n.outer, n])).values()] };
};

const routes = ['launch', 'chat', 'models', 'serving', 'monitor', 'tuning', 'workspace', 'skills', 'settings'];
let failures = 0;

console.log(`dark-mode surface budget — large fills (≥ ${LARGE_AREA}px²) under ${CEILING}, and no glass inside glass\n`);

for (const route of routes) {
  await page.goto(`${BASE}/#/${route}`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1100);
  const theme = await page.evaluate(() => document.documentElement.dataset.theme);
  if (theme !== 'dark') {
    await page.click('button[title="Switch to dark mode"]');
    await page.waitForTimeout(450);
  }
  const { bright, nested } = await page.evaluate(SURVEY, {
    largeArea: LARGE_AREA,
    minThickness: MIN_THICKNESS,
    ceiling: CEILING,
  });
  const problems = [];
  for (const o of bright) problems.push(`bright   ${String(o.luminance).padStart(6)}  ${String(o.area).padStart(7)}px²  ${o.selector}`);
  for (const n of nested) problems.push(`layering ${n.inner} is inside glass ${n.outer}`);
  if (problems.length) {
    failures += problems.length;
    console.log(`  ${route}`);
    for (const p of problems) console.log(`    ${p}`);
  } else {
    console.log(`  ${route.padEnd(10)} ok`);
  }
}

await browser.close();

if (errors.length) {
  failures += errors.length;
  console.log(`\n  page errors: ${errors.join('; ')}`);
}
console.log(
  failures
    ? `\nluminance: ${failures} problem(s) — a bright fill this large fatigues over hours, and glass cannot sample glass`
    : '\nluminance: every large dark surface is within budget, and no material is nested in another',
);
process.exit(failures ? 1 : 0);
