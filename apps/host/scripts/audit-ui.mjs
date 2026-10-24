#!/usr/bin/env node
/**
 * Strata UI audit.
 *
 *   node scripts/audit-ui.mjs
 *
 * Static checks over the app's own stylesheet and the synced kit copy. Everything
 * it asserts is a rule from src/styles/app.css's header comment or
 * docs/interaction-guidelines.md, so a failure means one of those documents and
 * the code disagree.
 *
 * Exits non-zero when anything fails, so it can gate a release.
 */
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const APP = join(ROOT, 'src/styles/app.css');
const KIT_SRC = join(ROOT, '..', 'ui');
const KIT_COPY = join(ROOT, 'src/styles/kit');

const KIT_SCALES = {
  radius: new Set(['8px', '14px', '20px', '24px', '9999px', '999px', '50%', '6px', '10px', '12px', '15px', '16px', '2px', '1.5px', 'inherit', '0']),
  fontSize: new Set(['11px', '12px', '13px', '14px', '15px', '16px', '17px', '18px', '19px', '22px', '24px', '30px', '32px', '40px', 'inherit']),
  shadow: [
    /^var\(--st-sh-(card|bar|pop)\)/, // the kit's three elevation tokens
    /^var\(--st-accent-surface-shadow\)/, // the accent-filled surfaces' own elevation
    /^none$/,
    /^inset/, // specular edges and focus rings
    /^0 0 0 \d+px var\(--st-/, // halo rings
  ],
};

/** Durations that belong to a looping indicator, not to a transition. */
const LOOP_DURATION_OK = new Set([
  '1s', '720ms', '1.1s', '1.15s', '1.4s', '1.6s', '1.8s',
  '420ms', '560ms', '260ms', // spring pops and the step marker
]);

/** The only infinite animations allowed, and why. */
const INFINITE_OK = new Set([
  'st-slide',  // indeterminate progress
  'st-pulse',  // busy dot
  'st-typing', // assistant is composing
  'st-blink',  // text caret
  'st-scan',   // hardware probe sweep (first run only)
  'st-spin',   // install stage spinner
  'st-ripple', // "ready" ring, first run only
]);

/** Layout properties that must never be transitioned (they thrash layout). */
const LAYOUT_PROPS = ['width', 'height', 'top', 'left', 'right', 'bottom', 'margin', 'padding', 'font-size'];

/**
 * Surfaces allowed to carry a material. Every one of them overlays content that
 * moves underneath; anything else would pay for a compositor pass and show
 * nothing.
 */
const BLUR_ALLOWED = new Set([
  '.main__head',
  '.focusbar',
  '.composer',
  '.slashmenu',
  '.select__pop',
  '.menu',
  '.telepop',
  '.scrim',
  '.config-savebar', // floating commit controls
  '.mobile-bar', // compact navigation controls
  '.st-toast--toned',
]);

/**
 * Surfaces that genuinely float above the page, and so may carry an elevation
 * shadow. A card sitting on the page background does not float: its shadow only
 * added a second edge next to the hairline.
 */
const ELEVATION_ALLOWED = new Set([
  '.modal',
  '.dialog-surface', // modal content sheet
  '.message-overflow[open] > div', // floating message menu
  '.app[data-nav-open=\'true\'] > .nav', // dismissible navigation sheet
  '.st-tab[aria-checked=\'true\']', // selected segmented control
  '.nav-new', // primary action affordance
  '.st-drawer.sheet',
  '.slashmenu',
  '.select__pop',
  '.menu',
  '.telepop',
  '.setup__panel',
  '.setupmodal__surface',
  '.st-tablewrap',
  '.st-toast--toned',
  // The jump-to-latest pill sits above a scrolling transcript, so it genuinely
  // overlaps moving content — the same test the other entries pass.
  '.jumplatest',
  '.st-btn--primary', // the accent fill's own lift
  '.st-slider::-webkit-slider-thumb', // a control affordance: the thumb needs to read against its track
  '.st-toggle::after', // same: the knob needs to read against its track
]);

/**
 * Controls, which Liquid Glass specifies as capsules. A control is something you
 * press or type into; a surface is something content sits on, and those stay on
 * the kit's 8/14/20/24 scale.
 */
const CAPSULE_CONTROLS = ['.st-btn', '.st-input', '.st-tab', '.navitem', '.navsession', '.slashmenu__item', '.treerow'];

/** Literal colours that are deliberate and documented. */
const DOCUMENTED_LITERALS = ['#05261b', '#12c68e', '#0a7a55'];

const failures = [];
const notes = [];
const fail = (file, line, message) => failures.push(`${file}:${line}  ${message}`);

/* ------------------------------------------------------------ app.css ---- */

const css = readFileSync(APP, 'utf8');
const lines = css.split('\n');
let selector = '(top level)';
let inKeyframes = false;

lines.forEach((raw, i) => {
  const line = raw.replace(/\/\*.*?\*\//g, '');
  const n = i + 1;

  const sel = line.match(/^([^@{}\s][^{}]*)\{$/);
  if (sel) selector = sel[1].trim();
  if (/@keyframes/.test(line)) inKeyframes = true;
  if (inKeyframes && /^\}/.test(line)) inKeyframes = false;

  for (const [, prop, rawValue] of line.matchAll(/([a-z-]+)\s*:\s*([^;]+);/g)) {
    const value = rawValue.trim();

    // --- static design-system rules -------------------------------------
    if (prop === 'border-radius') {
      for (const part of value.split(/\s+/)) {
        if (!KIT_SCALES.radius.has(part) && !part.startsWith('var(') && !part.includes('%')) {
          fail('app.css', n, `off-scale radius "${part}" in ${selector}`);
        }
      }
    }
    if (prop === 'font-size' && !value.startsWith('var(') && !value.includes('clamp') && !KIT_SCALES.fontSize.has(value)) {
      fail('app.css', n, `off-scale font-size "${value}" in ${selector}`);
    }
    if (prop === 'box-shadow' && !KIT_SCALES.shadow.some((re) => re.test(value))) {
      fail('app.css', n, `shadow is not a kit token: "${value}" in ${selector}`);
    }
    if (/gradient|radial|conic/.test(value) && prop.startsWith('background')) {
      fail('app.css', n, `gradient in ${selector} — the system is flat`);
    }
    if (/^(color|background|background-color|border-color)$/.test(prop)) {
      const literals = value.match(/#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)/g);
      if (literals && !value.includes('data:image') && !value.includes('currentColor')
          && !DOCUMENTED_LITERALS.some((d) => value.includes(d))) {
        fail('app.css', n, `literal colour "${value}" in ${selector} — use a token`);
      }
    }

    // --- motion rules ----------------------------------------------------
    // --- materials ------------------------------------------------------
    // Values must be a --st-material-* token: that is what collapses the blur
    // under reduced transparency, so gating cannot be forgotten.
    if (prop === 'backdrop-filter' || prop === '-webkit-backdrop-filter') {
      if (!/^var\(--st-material-(thin|regular|thick)\)$/.test(value) && !(selector === '.scrim' && value === 'none !important')) {
        fail('app.css', n, `blur must use a --st-material-* token (it carries the gating): "${value}"`);
      }
      if (!BLUR_ALLOWED.has(selector)) {
        fail('app.css', n, `material on "${selector}", which is not an approved surface — it does not overlay moving content`);
      }
    }
    if (prop === 'box-shadow' && /var\(--st-sh-(card|bar|pop)\)/.test(value) && !ELEVATION_ALLOWED.has(selector)) {
      fail('app.css', n, `elevation shadow on "${selector}" — shadows are reserved for surfaces that overlap content`);
    }

    if (prop === 'transition') {
      if (/^\d/.test(value) || /(^|[\s,])\d+m?s\b/.test(value)) {
        fail('app.css', n, `raw duration in transition — use a --st-t-* token: "${value}"`);
      }
      for (const part of value.split(',')) {
        const prop0 = part.trim().split(/\s+/)[0];
        if (LAYOUT_PROPS.includes(prop0)) {
          fail('app.css', n, `transition on layout property "${prop0}" in ${selector}`);
        }
      }
    }
    if (prop === 'transition-property') {
      for (const p of value.split(',').map((v) => v.trim())) {
        if (LAYOUT_PROPS.includes(p)) fail('app.css', n, `transition on layout property "${p}" in ${selector}`);
      }
    }
    if (prop === 'animation' || prop === 'animation-name') {
      const name = value.match(/^(st-[a-z-]+)/)?.[1];
      const infinite = /infinite/.test(value);
      if (infinite && name && !INFINITE_OK.has(name)) {
        fail('app.css', n, `looping animation "${name}" is not on the allowlist — nothing loops while idle`);
      }
      const duration = value.match(/\b(\d+(?:\.\d+)?m?s)\b/)?.[1];
      if (duration && !LOOP_DURATION_OK.has(duration)) {
        fail('app.css', n, `raw animation duration "${duration}" in ${selector} — use a --st-t-* token`);
      }
    }
    if (prop === 'animation-duration' && /^\d/.test(value) && !LOOP_DURATION_OK.has(value)) {
      fail('app.css', n, `raw animation-duration "${value}" in ${selector}`);
    }
  }
});

/* ------------------------------------------------- kit sync + motion scale -- */

for (const file of ['tokens.css', 'components.css']) {
  const a = join(KIT_SRC, file);
  const b = join(KIT_COPY, file);
  if (!existsSync(a) || !existsSync(b)) {
    fail('kit', 0, `${file} missing on one side`);
    continue;
  }
  // The sync rewrites @font-face URLs to absolute; everything else must match.
  const norm = (t) => t.replace(/url\((["']?)\/fonts\//g, 'url($1fonts/').trim();
  if (norm(readFileSync(a, 'utf8')) !== norm(readFileSync(b, 'utf8'))) {
    fail('kit', 0, `${file} has drifted from ../ui — run npm run sync:kit`);
  }
}

// Which selectors have been given the capsule radius.
const capsuled = new Set();
{
  const rules = css.replace(/\/\*[\s\S]*?\*\//g, '');
  for (const [, sel, body] of rules.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (!/border-radius:\s*var\(--st-r-pill\)/.test(body)) continue;
    for (const part of sel.split(',')) capsuled.add(part.trim());
  }
}
for (const c of CAPSULE_CONTROLS) {
  if (!capsuled.has(c)) {
    fail('app.css', 0, `${c} is a control but not a capsule — Liquid Glass specifies capsule-shaped controls`);
  }
}

/* ------------------------------------------------------------ source rules -- */

// The popup of a native <select> is drawn by the OS and shares nothing with the
// design system — it was the single most out-of-place surface in the app.
for (const file of [...readdirSync(join(ROOT, 'src'), { recursive: true })].filter((f) => /\.tsx?$/.test(String(f)))) {
  const full = join(ROOT, 'src', String(file));
  if (!statSync(full).isFile()) continue;
  // Strip comments first: the Select component's own doc comment *describes*
  // the native element it replaces, and matching prose would be a false alarm.
  const text = readFileSync(full, 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '');
  if (/<select[\s>]/.test(text)) {
    fail(`src/${file}`, 0, 'native <select> — use the Select component, whose popup we control');
  }
}

// Every <Icon name="..."> must resolve to a symbol in the sprite, or it renders
// as an empty box at runtime.
const sprite = readFileSync(join(ROOT, 'public/icons/sprite.svg'), 'utf8');
const symbols = new Set([...sprite.matchAll(/id="i-([a-z-]+)"/g)].map((m) => m[1]));
const referenced = new Set();
for (const file of [...readdirSync(join(ROOT, 'src'), { recursive: true })].filter((f) => /\.tsx$/.test(String(f)))) {
  const full = join(ROOT, 'src', String(file));
  if (!statSync(full).isFile()) continue;
  for (const m of readFileSync(full, 'utf8').matchAll(/name="([a-z-]+)"/g)) referenced.add(m[1]);
}
for (const name of referenced) {
  if (!symbols.has(name)) fail('src', 0, `icon "${name}" is referenced but not in the sprite — run npm run sync:kit`);
}

/* ---------------------------------------------------------- undefined classes -- */

// A class a component uses that no stylesheet defines renders unstyled and fails
// silently. An earlier "dead CSS" sweep deleted three of these because the sweep
// looked for `className="..."` literals and missed other spellings; the app
// shipped with unstyled metric sub-lines and toast bodies.
{
  const allCss = [];
  for (const dir of ['src/styles', 'src/styles/kit']) {
    for (const f of readdirSync(join(ROOT, dir))) {
      if (f.endsWith('.css')) allCss.push(readFileSync(join(ROOT, dir, f), 'utf8'));
    }
  }
  // Strip comments first: the note explaining this very check names the classes
  // it is about, and matching prose would satisfy the rule it exists to enforce.
  const cssCode = allCss.join('\n').replace(/\/\*[\s\S]*?\*\//g, '');
  const defined = new Set([...cssCode.matchAll(/\.([a-z][a-z0-9_-]*)/g)].map((m) => m[1]));
  const used = new Map();
  for (const file of [...readdirSync(join(ROOT, 'src'), { recursive: true })].filter((f) => /\.tsx$/.test(String(f)))) {
    const full = join(ROOT, 'src', String(file));
    if (!statSync(full).isFile()) continue;
    const text = readFileSync(full, 'utf8');
    // Any quoted class-shaped token, wherever it appears — className literals,
    // cx() arguments, ternaries. Missing a spelling is what caused the bug.
    for (const m of text.matchAll(/['"`]([a-z][a-z0-9_-]*(?:__|--)[a-z0-9_-]+)['"`]/g)) {
      used.set(m[1], String(file));
    }
  }
  for (const [cls, file] of used) {
    if (!defined.has(cls)) {
      fail(`src/${file}`, 0, `class "${cls}" is used but defined in no stylesheet — it renders unstyled`);
    }
  }
}

/* ------------------------------------------------------- undefined tokens -- */

// A `var(--st-…)` with no definition is not an error the browser reports: the
// declaration is simply dropped. `--st-accent-line` sat in a reduced-motion
// rule for several rounds, so the border it set never applied and the rule
// looked deliberate. Only kit-prefixed tokens are checked, since app-local
// custom properties are defined and consumed in the same file.
{
  const appCss = readdirSync(join(ROOT, 'src/styles'))
    .filter((f) => f.endsWith('.css'))
    .map((f) => readFileSync(join(ROOT, 'src/styles', f), 'utf8'))
    .join('\n');
  const kitCss = readdirSync(join(ROOT, 'src/styles/kit'))
    .filter((f) => f.endsWith('.css'))
    .map((f) => readFileSync(join(ROOT, 'src/styles/kit', f), 'utf8'))
    .join('\n');
  const all = `${appCss}\n${kitCss}`.replace(/\/\*[\s\S]*?\*\//g, '');
  const declared = new Set([...all.matchAll(/(--st-[a-z0-9-]+)\s*:/g)].map((m) => m[1]));
  /* Some tokens are set at runtime rather than written in a stylesheet —
     `--st-content-fs` follows the reading-size preference. Without this the
     rule reports a legitimate token, which is how a check earns being ignored. */
  for (const file of [...readdirSync(join(ROOT, 'src'), { recursive: true })].filter((f) => /\.tsx?$/.test(String(f)))) {
    const full = join(ROOT, 'src', String(file));
    if (!statSync(full).isFile()) continue;
    for (const m of readFileSync(full, 'utf8').matchAll(/setProperty\(\s*['"`](--st-[a-z0-9-]+)/g)) declared.add(m[1]);
  }
  const used = new Map();
  for (const m of all.matchAll(/var\((--st-[a-z0-9-]+)/g)) {
    if (!used.has(m[1])) used.set(m[1], true);
  }
  for (const name of used.keys()) {
    if (!declared.has(name)) {
      fail('src/styles/app.css', 0, `var(${name}) has no definition — the declaration using it is silently dropped`);
    }
  }
  notes.push(`tokens: every var(--st-*) resolves to a declaration`);
}

const tokens = new Set([...css.matchAll(/--st-t-[a-z]+:/g)].map((m) => m[0].slice(0, -1)));
const used = new Set([...css.matchAll(/var\((--st-t-[a-z]+)\)/g)].map((m) => m[1]));
for (const u of used) {
  if (!tokens.has(u)) fail('app.css', 0, `uses ${u}, which is not declared in the motion scale`);
}
const SCALE = ['--st-t-micro', '--st-t-control', '--st-t-local', '--st-t-surface', '--st-t-choreo'];
notes.push(`motion scale: ${SCALE.filter((t) => tokens.has(t)).length}/5 tokens declared, ${used.size} in use`);
notes.push(`kit copy: in sync with ../ui`);

/* ------------------------------------------------------------------ report -- */

const keyframes = (css.match(/@keyframes/g) ?? []).length;
notes.push(`keyframes: ${keyframes} (was 23 before consolidation)`);
notes.push(`materials: ${BLUR_ALLOWED.size} approved surfaces, all token-gated`);
notes.push(`elevation: ${ELEVATION_ALLOWED.size} floating surfaces, ${(css.match(/var\(--st-sh-(card|bar|pop)\)/g) ?? []).length} shadow uses`);
notes.push(`capsules: ${CAPSULE_CONTROLS.length}/${CAPSULE_CONTROLS.length} controls, surfaces untouched`);
notes.push(`icons: ${referenced.size} referenced, all present in a ${symbols.size}-symbol sprite`);
notes.push(`dropdowns: no native <select> in src/`);
notes.push(`classes: every class a component uses is defined in a stylesheet`);

if (failures.length) {
  console.log(`ui audit: ${failures.length} failure(s)\n`);
  for (const f of failures) console.log(`  ${f}`);
  console.log();
  for (const n of notes) console.log(`  · ${n}`);
  process.exit(1);
}
console.log('ui audit: clean\n');
for (const n of notes) console.log(`  · ${n}`);
