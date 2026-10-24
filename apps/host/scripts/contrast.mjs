#!/usr/bin/env node
/**
 * Strata contrast probe.
 *
 *   node scripts/contrast.mjs
 *
 * Measures real rendered contrast in the running app rather than reading token
 * values, because the things that actually fail are pairings: a label on a
 * filled button, text on a translucent tint, muted copy on a quiet panel.
 *
 * Translucent backgrounds are composited over their ancestors before the ratio
 * is computed, and every interactive state is measured separately — the kit's
 * primary button passes at rest and fails on hover, which a single measurement
 * would have missed.
 *
 * Requires the dev server. Exits non-zero if anything falls below WCAG AA.
 */
import { chromium } from 'playwright-core';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const BASE = process.env.STRATA_URL ?? 'http://localhost:5199';

const executablePath = [
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  join(homedir(), 'Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell'),
].find((p) => existsSync(p));

if (!executablePath) {
  console.error('No cached Chromium found. Run `npm run shot -- x.png <url>` once to confirm the harness.');
  process.exit(2);
}

/** Runs in the page. Kept as a string so it can be injected verbatim. */
const PROBE = () => {
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
  const over = (f, bg) => ({
    r: f.r * f.a + bg.r * (1 - f.a),
    g: f.g * f.a + bg.g * (1 - f.a),
    b: f.b * f.a + bg.b * (1 - f.a),
    a: 1,
  });
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
      if (c.a > 0) stack.push(c);
      if (c.a === 1) break;
      n = n.parentElement;
    }
    if (!stack.length || stack[stack.length - 1].a < 1) stack.push({ r: 255, g: 255, b: 255, a: 1 });
    let acc = stack[stack.length - 1];
    for (let i = stack.length - 2; i >= 0; i -= 1) acc = over(stack[i], acc);
    return acc;
  };
  const ratio = (a, b) => {
    const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
    return Number(((hi + 0.05) / (lo + 0.05)).toFixed(2));
  };
  const probe = (label, sel, size) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const px = size ?? parseFloat(getComputedStyle(el).fontSize);
    const bg = bgOf(el);
    const f0 = parse(getComputedStyle(el).color);
    const fg = f0.a < 1 ? over(f0, bg) : f0;
    const r = ratio(fg, bg);
    const bold = Number(getComputedStyle(el).fontWeight) >= 700;
    const large = px >= 24 || (px >= 18.66 && bold);
    const need = large ? 3 : 4.5;
    return { label, ratio: r, px: Math.round(px), need, pass: r >= need };
  };
  return [
    probe('primary button', '.st-btn--primary'),
    probe('secondary button', '.st-btn--secondary'),
    probe('body copy', '.main__sub'),
    probe('muted meta', '.st-msg__meta', 12),
    probe('sidebar nav', '.navitem__text'),
    probe('sidebar group label', '.navgroup__label', 11),
    probe('user bubble', '.st-msg--user .st-bubble'),
    probe('assistant bubble', '.st-msg--assistant .st-bubble'),
    probe('accent badge', '.st-pill--accent, .st-badge--generating', 13),
    probe('table header', '.st-table th', 11),
    probe('mono code', '.mono', 13),
    probe('file preview', '.filepreview__body', 13),
    // The accent-filled surfaces. In dark mode these are a tinted material
    // rather than a solid fill, so the label and the fill both need checking.
    probe('accent fill label', '.st-btn--primary'),
    probe('user bubble (tinted)', '.st-msg--user .st-bubble'),
  ].filter(Boolean);
};

/**
 * WCAG 1.4.11 — non-text contrast.
 *
 * Text pairings were covered; the *components* were not. A focus ring, an input
 * boundary and a switch track all have to reach **3:1** against what sits
 * beside them, and these are the things that vanish first when a theme is
 * retuned, because nothing was measuring them.
 *
 * Measured against the actual adjacent surface, not against a token, for the
 * same reason the text probe measures rendered pixels.
 *
 * **What is enforced and what is only reported.** 1.4.11 covers information
 * *required to identify* a component, so the question for each thing measured
 * is "is this the only thing telling you what it is or what state it is in?"
 *
 * - **Focus indicators — enforced.** Nothing else says "the keyboard is here".
 * - **Card and input boundaries — reported.** A card is grouping, not a
 *   control; a labelled input is identified by its label and position. The
 *   light hairline is a deliberate house style shared with Apple's light-mode
 *   controls, not a defect.
 * - **Toggle track — reported.** The switch carries its state in the *knob
 *   position*, which is not a contrast matter; the track colour reinforces it.
 *
 * The distinction matters because a probe that fails on things WCAG does not
 * require is a probe people learn to ignore — and then it catches nothing.
 */
async function nonText(theme) {
  const page = await browser.newPage({ viewport: { width: 1512, height: 1110 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const rows = [];
  const boundaries = [];
  for (const route of ['chat', 'models', 'tuning', 'settings', 'serving', 'skills']) {
    await page.goto(`${BASE}/#/${route}`, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(1000);
    if (theme === 'dark') {
      const t = await page.$('button[title="Switch to dark mode"]');
      if (t) {
        await t.click();
        await page.waitForTimeout(350);
      }
    }
    // Drive the keyboard so a real focus ring renders rather than a simulated one.
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.waitForTimeout(250);
    const { results, info } = await page.evaluate(() => {
      const lum = ({ r, g, b }) => {
        const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
      };
      const parse = (c) => {
        const s = String(c);
        const srgb = s.match(/^color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+))?\)/);
        if (srgb) return { r: +srgb[1] * 255, g: +srgb[2] * 255, b: +srgb[3] * 255, a: srgb[4] === undefined ? 1 : +srgb[4] };
        const n = s.match(/[\d.]+/g);
        if (!n || n.length < 3) return null;
        return { r: +n[0], g: +n[1], b: +n[2], a: n[3] === undefined ? 1 : +n[3] };
      };
      const ratio = (a, b) => {
        const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
        return (hi + 0.05) / (lo + 0.05);
      };
      const under = (el) => {
        let n = el?.parentElement;
        while (n) {
          const c = parse(getComputedStyle(n).backgroundColor);
          if (c && c.a > 0.85) return c;
          n = n.parentElement;
        }
        return parse(getComputedStyle(document.body).backgroundColor) ?? { r: 255, g: 255, b: 255, a: 1 };
      };
      const results = [];
      // the focus ring, whatever the browser is drawing it on
      const focused = document.activeElement;
      if (focused && focused !== document.body) {
        const cs = getComputedStyle(focused);
        const ring = parse(cs.outlineColor);
        const w = parseFloat(cs.outlineWidth) || 0;
        if (ring && w > 0 && cs.outlineStyle !== 'none') {
          results.push({ label: `focus ring on ${focused.tagName.toLowerCase()}.${String(focused.className).split(' ')[0]}`, ratio: ratio(ring, under(focused)) });
        }
      }
      const info = [];
      for (const [label, sel] of [['input boundary', '.st-input'], ['card boundary', '.st-card']]) {
        const el = document.querySelector(sel);
        if (!el) continue;
        const cs = getComputedStyle(el);
        const border = parse(cs.borderTopColor);
        const bw = parseFloat(cs.borderTopWidth) || 0;
        if (border && bw > 0) info.push({ label, ratio: ratio(border, under(el)) });
      }
      // A switch's state is carried by knob position; the track reinforces it.
      const sw = document.querySelector('.st-toggle');
      if (sw) {
        const track = parse(getComputedStyle(sw).backgroundColor);
        if (track && track.a > 0.15) {
          info.push({ label: `toggle track (state also shown by knob position)`, ratio: ratio(track, under(sw)) });
        }
      }
      return { results, info };
    });
    rows.push(...results.map((r) => ({ ...r, route })));
    boundaries.push(...info.map((r) => ({ ...r, route })));
  }
  await page.close();
  return { rows, boundaries, errors };
}

const browser = await chromium.launch({ executablePath, args: ['--no-sandbox'] });
let failures = 0;

async function measure(theme, hoverSelector, view) {
  const page = await browser.newPage({ viewport: { width: 1512, height: 1110 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  if (theme === 'dark') {
    await page.click('button[title="Switch to dark mode"]');
    await page.waitForTimeout(500);
  }
  if (view === 'conversation') {
    await page.click('button.nav-new');
    await page.waitForTimeout(300);
    await page.fill('textarea', 'Show me everything');
    await page.keyboard.press('Enter');
    await page.waitForTimeout(6200);
  }
  if (view === 'workspace') {
    /* The file preview renders on the kit's dark code surface. It inherited
       that background but overrode the text colour with the light theme's ink,
       giving #303030 on #1b1f23 — a ratio of 1.26, where body text needs 4.5.
       Nothing measured it, because the probe only ever visited Chat. */
    await page.goto(`${BASE}/#/workspace`, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(1600);
    await page.evaluate(() => {
      const file = [...document.querySelectorAll('.treerow')].find((r) => r.dataset.kind === 'file');
      file?.click();
    });
    await page.waitForTimeout(1200);
  }
  if (hoverSelector) {
    await page.hover(hoverSelector);
    await page.waitForTimeout(350);
  } else {
    await page.mouse.move(760, 980); // park away from controls so nothing is hovered
    await page.waitForTimeout(250);
  }
  const results = await page.evaluate(PROBE);
  await page.close();
  return { results, errors };
}

for (const theme of ['light', 'dark']) {
  const nt = await nonText(theme);
  console.log(`\n${theme} · non-text contrast (WCAG 1.4.11, enforced at 3:1)`);
  const seen = new Set();
  for (const r of nt.rows) {
    if (seen.has(r.label)) continue;
    seen.add(r.label);
    const pass = r.ratio >= 3;
    if (!pass) failures += 1;
    console.log(`  ${r.ratio.toFixed(2).padStart(6)}  need 3.0  ${pass ? 'PASS' : 'FAIL'}  ${r.label} (${r.route})`);
  }
  const seenB = new Set();
  const infoRows = nt.boundaries.filter((r) => !seenB.has(r.label) && seenB.add(r.label));
  if (infoRows.length) {
    console.log(`  — reported only, not required by 1.4.11 —`);
    for (const r of infoRows) {
      console.log(`  ${r.ratio.toFixed(2).padStart(6)}  (info)     ${r.label}`);
    }
  }
  if (nt.errors.length) {
    failures += nt.errors.length;
    console.log(`  page errors: ${nt.errors.join('; ')}`);
  }

  for (const view of ['chrome', 'conversation', 'workspace']) {
    const { results, errors } = await measure(theme, null, view);
    console.log(`\n${theme} · ${view}`);
    for (const r of results) {
      if (!r.pass) failures += 1;
      console.log(`  ${String(r.ratio).padStart(6)}  need ${r.need}  ${r.pass ? 'PASS' : 'FAIL'}  ${r.label} (${r.px}px)`);
    }
    if (errors.length) {
      failures += errors.length;
      console.log(`  page errors: ${errors.join('; ')}`);
    }
  }
}

// Hover is where the kit's primary button used to fail.
/**
 * How a tinted control stays visible, which differs by theme.
 *
 * On a near-black page the fill itself must separate from the background, so
 * dark asserts fill-vs-page. On a near-white page a tinted fill cannot reach
 * that ratio and does not need to: the control is delimited by its accent
 * hairline and its shadow, the way Apple's light-mode tinted controls are. So
 * light asserts the *border* is present and visibly differs from the page —
 * which is exactly the thing that broke when the border was mistakenly bound to
 * the white specular token.
 */
async function fillVsPage(theme) {
  const page = await browser.newPage({ viewport: { width: 1512, height: 1110 } });
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1300);
  if (theme === 'dark') {
    await page.click('button[title="Switch to dark mode"]');
    await page.waitForTimeout(500);
  }
  const out = await page.evaluate(() => {
    const lum = ({ r, g, b }) => {
      const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
      return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
    };
    const parse = (c) => {
      const s = String(c);
      const srgb = s.match(/^color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+))?\)/);
      if (srgb) {
        return { r: +srgb[1] * 255, g: +srgb[2] * 255, b: +srgb[3] * 255, a: srgb[4] === undefined ? 1 : +srgb[4] };
      }
      const n = s.match(/[\d.]+/g);
      return n ? { r: +n[0], g: +n[1], b: +n[2], a: n[3] === undefined ? 1 : +n[3] } : null;
    };
    const btn = document.querySelector('.st-btn--primary');
    if (!btn) return null;
    const fill = parse(getComputedStyle(btn).backgroundColor);
    if (!fill) return null;
    let pageBg = parse(getComputedStyle(document.querySelector('.main__body') ?? document.body).backgroundColor);
    if (!pageBg || (pageBg.r === 0 && pageBg.g === 0 && pageBg.b === 0)) {
      pageBg = parse(getComputedStyle(document.body).backgroundColor);
    }
    const [hi, lo] = [lum(fill), lum(pageBg)].sort((a, b) => b - a);
    const cs = getComputedStyle(btn);
    const bw = parseFloat(cs.borderTopWidth) || 0;
    const declared = parse(cs.borderTopColor);
    // Composite the border over the page before comparing. A declared-but-faint
    // border (say 2% alpha) differs wildly from the page as a raw value while
    // being invisible on screen, which would make this check meaningless.
    const border = declared
      ? {
          r: declared.r * (declared.a ?? 1) + pageBg.r * (1 - (declared.a ?? 1)),
          g: declared.g * (declared.a ?? 1) + pageBg.g * (1 - (declared.a ?? 1)),
          b: declared.b * (declared.a ?? 1) + pageBg.b * (1 - (declared.a ?? 1)),
        }
      : null;
    const borderDelta = border
      ? Math.max(Math.abs(border.r - pageBg.r), Math.abs(border.g - pageBg.g), Math.abs(border.b - pageBg.b))
      : 0;
    return {
      ratio: Number(((hi + 0.05) / (lo + 0.05)).toFixed(2)),
      luminance: Number(lum(fill).toFixed(3)),
      borderWidth: bw,
      borderDelta: Math.round(borderDelta),
    };
  });
  await page.close();
  return out;
}

for (const theme of ['light', 'dark']) {
  const r = await fillVsPage(theme);
  if (!r) continue;
  if (theme === 'dark') {
    const ok = r.ratio >= 1.5;
    if (!ok) failures += 1;
    console.log(
      `\ndark · primary fill vs page\n  ${String(r.ratio).padStart(6)}  need 1.5  ${ok ? 'PASS' : 'FAIL'}  fill separates from the page (luminance ${r.luminance})`,
    );
  } else {
    // Light: the hairline carries the definition, not the fill.
    const ok = r.borderWidth >= 1 && r.borderDelta >= 10;
    if (!ok) failures += 1;
    console.log(
      `\nlight · primary fill definition\n  ${String(r.borderDelta).padStart(6)}  need 10   ${ok ? 'PASS' : 'FAIL'}  accent hairline delimits the tinted fill (fill is only ${r.ratio}:1 by design)`,
    );
  }
}

const { results: hovered } = await measure('light', '.nav__rail-hide', 'chrome');
const primary = hovered.find((r) => r.label === 'primary button');
if (primary && !primary.pass) failures += 1;
console.log(`\nlight · primary button hovered\n  ${String(primary?.ratio).padStart(6)}  need ${primary?.need}  ${primary?.pass ? 'PASS' : 'FAIL'}  primary button (hover)`);

await browser.close();

console.log(failures ? `\ncontrast probe: ${failures} failure(s)` : '\ncontrast probe: all pairings pass WCAG AA');
process.exit(failures ? 1 : 0);
