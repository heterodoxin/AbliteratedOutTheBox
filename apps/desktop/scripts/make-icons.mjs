#!/usr/bin/env node
/**
 * The app and installer icon, refined from the existing Strata mark (apps/host/public/brand/strata-mark-*.svg and the current
 * icons/icon.png): the same three rounded bars on the same dark tile, in the same colours. Nothing is redrawn. What changes is
 * how it is cut for small sizes: at 16-48 px the bars are snapped to whole pixels and made a little bolder and wider so they
 * stay crisp instead of blurring; from 64 px up the geometry is the mark's own proportions.
 *
 *   node apps/desktop/scripts/make-icons.mjs [--out <folder>]    writes icon-<size>.png for every size and icon.ico
 */
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const here = dirname(fileURLToPath(import.meta.url));
const host = resolve(here, '../../host');
const out = resolve(args.out ?? join(here, '../src-tauri/icons'));
mkdirSync(out, { recursive: true });
const { chromium } = await import(pathToFileURL(join(host, 'node_modules/playwright-core/index.mjs')).href).catch(() => import('playwright-core'));
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);

const TILE = '#161a1d', LIGHT = '#eef1f3', GREEN = '#34d399';   // the colours of the current icon.png

/** Integer geometry for the small sizes: bars on whole pixels, centred. Returns [{x,y,w,h,fill}]. */
function hinted(size) {
  const bar = Math.max(2, Math.round(size * 0.14)), gap = Math.max(2, Math.round(size * 0.1)), mid = bar + 1;
  const total = 2 * bar + mid + 2 * gap, top = Math.floor((size - total) / 2);
  const even = w => (w % 2 === size % 2 ? w : w + 1);
  const wide = even(Math.round(size * 0.5)), long = even(Math.round(size * 0.76));
  return [
    { x: (size - wide) / 2, y: top, w: wide, h: bar, fill: LIGHT },
    { x: (size - long) / 2, y: top + bar + gap, w: long, h: mid, fill: GREEN },
    { x: (size - wide) / 2, y: top + bar + gap + mid + gap, w: wide, h: bar, fill: LIGHT },
  ];
}
/** The mark's own proportions (from icon.png at 512): bars 34% / 50% / 34% wide, 9% high, on a tile with 21.5% corners. */
function proportional(size) {
  const k = size / 512;
  return [
    { x: 170 * k, y: 156 * k, w: 173 * k, h: 46 * k, fill: LIGHT },
    { x: 128 * k, y: 232 * k, w: 256 * k, h: 48 * k, fill: GREEN },
    { x: 170 * k, y: 309 * k, w: 173 * k, h: 46 * k, fill: LIGHT },
  ];
}
function svg(size) {
  const small = size <= 48;
  const bars = (small ? hinted(size) : proportional(size));
  const radius = small ? Math.round(size * 0.22) : size * 0.215;
  const rects = bars.map(b => `<rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="${small ? Math.min(b.h / 2, 1.5 + size / 32) : b.h / 2}" fill="${b.fill}"${small ? ' shape-rendering="crispEdges"' : ''}/>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"><rect width="${size}" height="${size}" rx="${radius}" fill="${TILE}"/>${rects}</svg>`;
}

const SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256, 512, 1024];
const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const png = {};
for (const size of SIZES) {
  const page = await browser.newPage({ viewport: { width: size, height: size }, deviceScaleFactor: 1 });
  await page.setContent(`<!doctype html><body style="margin:0;background:transparent">${svg(size)}</body>`);
  png[size] = await page.screenshot({ omitBackground: true, clip: { x: 0, y: 0, width: size, height: size } });
  await page.close();
  writeFileSync(join(out, `icon-${size}.png`), png[size]);
}
await browser.close();

// The .ico: one PNG image per size (16-256), the sizes Windows asks for at 100-250% scaling.
const ICO = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256];
const head = Buffer.alloc(6); head.writeUInt16LE(0, 0); head.writeUInt16LE(1, 2); head.writeUInt16LE(ICO.length, 4);
let offset = 6 + 16 * ICO.length;
const entries = [], blobs = [];
for (const size of ICO) {
  const data = png[size], e = Buffer.alloc(16);
  e.writeUInt8(size === 256 ? 0 : size, 0); e.writeUInt8(size === 256 ? 0 : size, 1); e.writeUInt8(0, 2); e.writeUInt8(0, 3);
  e.writeUInt16LE(1, 4); e.writeUInt16LE(32, 6); e.writeUInt32LE(data.length, 8); e.writeUInt32LE(offset, 12);
  offset += data.length; entries.push(e); blobs.push(data);
}
writeFileSync(join(out, 'icon.ico'), Buffer.concat([head, ...entries, ...blobs]));
writeFileSync(join(out, 'icon.png'), png[512]);
console.log('icons in', out, ICO.join(','));
