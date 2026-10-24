#!/usr/bin/env node
/**
 * Mock-ups of the Windows installer in Strata's own look: four screens (welcome, where, installing, done), light and dark, drawn with
 * the app's built CSS (apps/host/dist) and brand files, then photographed with Chrome. Nothing here is built or shipped: it is what
 * the owner is shown before an installer is chosen.
 *
 *   node apps/desktop/scripts/installer-mockups.mjs [--out <folder>]
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const here = dirname(fileURLToPath(import.meta.url));
const host = resolve(here, '../../host');
const out = resolve(args.out ?? join(here, '../installer-mockups'));
mkdirSync(out, { recursive: true });
const dist = join(host, 'dist');
const css = readdirSync(join(dist, 'assets')).filter(f => f.endsWith('.css')).map(f => readFileSync(join(dist, 'assets', f), 'utf8')).join('\n');
const logo = theme => readFileSync(join(dist, 'brand', `strata-lockup-${theme}.svg`), 'utf8');
const { chromium } = await import(pathToFileURL(join(host, 'node_modules/playwright-core/index.mjs')).href).catch(() => import('playwright-core'));
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);

const check = '<svg class="st-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
const folder = '<svg class="st-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>';

const screens = {
  welcome: theme => `<header class="onboard-header"><span class="mock-logo">${logo(theme)}</span></header>
    <div class="onboard-welcome-copy"><h1>State of the art. At home.</h1><p>Strata runs AI on your own computer.<br>Setting it up takes a few minutes.</p>
    <div class="onboard-benefits"><span>${check}Private</span><span>${check}Fast</span><span>${check}Easy to use</span></div></div>
    <footer class="onboard-footer onboard-footer--welcome"><button class="st-btn st-btn--primary">Install Strata</button><small>Installs for you only. No administrator needed.</small></footer>`,
  where: theme => `<header class="onboard-header"><span class="mock-logo">${logo(theme)}</span></header>
    <div class="mock-body"><h2 class="mock-title">Where should Strata go?</h2><p class="mock-sub">This is the right place for most people.</p>
    <div class="onboard-folder">${folder}<div><strong>Your Strata folder</strong><small>C:\\Users\\Alex\\AppData\\Local\\Strata</small></div><button class="st-btn st-btn--sm">Change</button></div>
    <p class="mock-note">About 310 MB. Your models and chats are kept somewhere else and are never removed by an update.</p></div>
    <footer class="onboard-footer"><button class="st-btn st-btn--ghost">Back</button><span></span><button class="st-btn st-btn--primary">Install</button></footer>`,
  installing: theme => `<header class="onboard-header"><span class="mock-logo">${logo(theme)}</span></header>
    <div class="mock-body"><h2 class="mock-title">Installing Strata…</h2><p class="mock-sub">Copying the app and its engine room. You can keep working meanwhile.</p>
    <div class="onboard-progress" style="--p:62%"><progress max="100" value="62"></progress></div>
    <div class="onboard-progress-meta"><span>62% · 190 / 310 MB</span><div class="onboard-progress-actions"><span>About 1 min left</span><button>Cancel</button></div></div></div>
    <footer class="onboard-footer"><span></span><span class="mock-hint">Next: choose your first model inside Strata.</span><button class="st-btn st-btn--primary" disabled>Installing…</button></footer>`,
  done: theme => `<header class="onboard-header"><span class="mock-logo">${logo(theme)}</span></header>
    <div class="onboard-welcome-copy"><h1>Strata is ready.</h1><p>Open it to pick your first model. Strata downloads it for you and gets it ready.</p></div>
    <footer class="onboard-footer onboard-footer--welcome"><button class="st-btn st-btn--primary">Open Strata</button><small>A shortcut was added to your Start menu and desktop.</small></footer>`,
};

const page = (name, theme) => `<!doctype html><html data-theme="${theme}"><head><meta charset="utf-8"><style>${css}
body{background:${theme === 'dark' ? '#0b0d0e' : '#e9edef'};margin:0;display:grid;place-items:center;min-height:100vh}
.mock-window{width:720px;border-radius:14px;overflow:hidden;background:var(--st-surface,#fff);border:1px solid var(--st-line,#dde);box-shadow:0 30px 80px #0003}
.mock-titlebar{height:34px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;font-size:12px;color:var(--st-ink-muted);background:var(--st-surface-2,#f3f5f6);border-bottom:1px solid var(--st-line-soft,#e6e9eb)}
.mock-logo svg{height:22px;width:auto;display:block}
.mock-body{padding:8px 36px 28px}.mock-title{font-size:28px;margin:16px 0 8px}.mock-sub{color:var(--st-ink-muted);margin:0 0 22px;line-height:1.6}
.mock-note{color:var(--st-ink-muted);font-size:12px;margin-top:16px;line-height:1.6}.mock-hint{font-size:12px;color:var(--st-ink-muted)}
.onboard-progress progress{width:100%}
.guided-onboarding{min-height:430px;display:flex;flex-direction:column}.onboard-footer{margin-top:auto}
</style></head><body><div class="mock-window"><div class="mock-titlebar"><span>Strata setup</span><span>–  ☐  ✕</span></div>
<div class="setup guided-onboarding" data-step="${name === 'welcome' || name === 'done' ? 1 : 2}">${screens[name](theme)}</div></div></body></html>`;

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const written = [];
for (const theme of ['light', 'dark']) {
  for (const name of Object.keys(screens)) {
    const tab = await browser.newPage({ viewport: { width: 860, height: 620 }, deviceScaleFactor: 1 });
    await tab.setContent(page(name, theme));
    await tab.waitForTimeout(250);
    const file = join(out, `${name}-${theme}.png`);
    await tab.screenshot({ path: file });
    written.push(file);
    await tab.close();
  }
}
await browser.close();
console.log(written.join('\n'));
