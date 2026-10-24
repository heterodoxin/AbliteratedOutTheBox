/**
 * Sync the Strata UI kit (../ui) into the app's public/ folder.
 *
 *   node scripts/sync-design-kit.mjs
 *
 * Copies fonts, and rebuilds a single clean icon sprite at public/icons/sprite.svg.
 * The kit's own SVG files carry C2PA manifest metadata (~12 KB each, base64);
 * we strip <metadata> so the shipped sprite stays small.
 *
 * Also mirrors tokens.css / components.css into src/styles/kit/ so the app has a
 * versioned copy that can be diffed against the kit.
 */
import { readFileSync, writeFileSync, mkdirSync, readdirSync, copyFileSync, existsSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const kit = join(root, '..', 'ui');
const pub = join(root, 'public');
const styles = join(root, 'src', 'styles', 'kit');

if (!existsSync(kit)) {
  console.error(`Design kit not found at ${kit}`);
  process.exit(1);
}

mkdirSync(join(pub, 'fonts'), { recursive: true });
mkdirSync(join(pub, 'icons'), { recursive: true });
mkdirSync(styles, { recursive: true });

// ---- fonts -----------------------------------------------------------------
let fonts = 0;
for (const f of readdirSync(join(kit, 'fonts'))) {
  if (!f.endsWith('.woff2')) continue;
  copyFileSync(join(kit, 'fonts', f), join(pub, 'fonts', f));
  fonts++;
}

// ---- icon sprite -----------------------------------------------------------
const strip = (svg) =>
  svg
    .replace(/<metadata>[\s\S]*?<\/metadata>/g, '')
    .replace(/<svg[^>]*>/, '')
    .replace(/<\/svg>\s*$/, '')
    .replace(/\s+xmlns(:\w+)?="[^"]*"/g, '')
    .trim();

const iconDir = join(kit, 'icons');
const sources = new Map(readdirSync(iconDir)
  .filter((f) => f.endsWith('.svg') && f !== 'sprite.svg')
  .map((f) => [basename(f, '.svg'), join(iconDir, f)]));
const appIconDir = join(root, 'src', 'assets', 'icons');
if (existsSync(appIconDir)) {
  for (const file of readdirSync(appIconDir).filter(f => f.endsWith('.svg'))) {
    sources.set(basename(file, '.svg'), join(appIconDir, file));
  }
}
const names = [...sources.keys()].sort();

const symbols = names.map((name) => {
  const raw = readFileSync(sources.get(name), 'utf8');
  const body = strip(raw);
  const viewBox = /viewBox="([^"]+)"/.exec(raw)?.[1] ?? '0 0 24 24';
  // The root <svg> carries fill="none"; without it every stroked path inherits
  // the default black fill and the icons render as solid blobs.
  const rootFill = /<svg[^>]*\sfill="([^"]+)"/.exec(raw)?.[1] ?? 'none';
  return `  <symbol id="i-${name}" viewBox="${viewBox}" fill="${rootFill}">\n    ${body}\n  </symbol>`;
});

const sprite = `<svg xmlns="http://www.w3.org/2000/svg" style="display:none" aria-hidden="true">\n${symbols.join('\n')}\n</svg>\n`;
writeFileSync(join(pub, 'icons', 'sprite.svg'), sprite);

// ---- styles ----------------------------------------------------------------
// The kit references fonts relatively (fonts/x.woff2). In the app the fonts are
// served from public/fonts, so rewrite the URLs to be absolute — otherwise the
// browser resolves them against the CSS file and 404s in both dev and build.
for (const f of ['tokens.css', 'components.css']) {
  const css = readFileSync(join(kit, f), 'utf8').replace(/url\((["']?)fonts\//g, 'url($1/fonts/');
  writeFileSync(join(styles, f), css);
}

const kb = (n) => `${(n / 1024).toFixed(1)} KB`;
console.log(`synced ${fonts} fonts -> public/fonts/`);
console.log(`sprite: ${names.length} icons -> public/icons/sprite.svg (${kb(sprite.length)})`);
console.log(`styles: tokens.css, components.css -> src/styles/kit/`);
console.log(`icons: ${names.join(', ')}`);
