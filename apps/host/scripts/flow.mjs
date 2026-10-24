/**
 * Scripted UI flows for verification screenshots.
 *
 *   node scripts/flow.mjs chat|expert|setup|serving|light
 */
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { homedir } from 'node:os';

const executablePath = [
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  join(homedir(), 'Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell'),
].find((p) => existsSync(p));

const BASE = process.env.STRATA_URL ?? 'http://localhost:5199';
const flow = process.argv[2] ?? 'chat';
const shots = resolve('.shots');
mkdirSync(shots, { recursive: true });

const browser = await chromium.launch({ executablePath, args: ['--no-sandbox', '--disable-gpu'] });
const page = await browser.newPage({ viewport: { width: 1512, height: 1110 }, deviceScaleFactor: 1 });

const problems = [];

/**
 * A flow that deliberately breaks something — aborting a request to exercise a
 * failure path — would otherwise report its own instrumentation as a problem.
 * Flows declare the failures they intend; anything else is still a problem.
 */
const expectedFailures = [];
const expectFailure = (pattern) => expectedFailures.push(pattern);

page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
page.on('console', (m) => {
  if (m.type() !== 'error') return;
  const text = m.text();
  if (expectedFailures.some((re) => re.test(text))) return;
  problems.push(`console: ${text}`);
});

const shot = async (name) => {
  await page.screenshot({ path: join(shots, `${name}.png`) });
  console.log(join(shots, `${name}.png`));
};

await page.goto(`${BASE}/#/launch`, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(1200);

if (flow === 'chat') {
  await page.click('text=New conversation');
  await page.waitForTimeout(500);
  await page.fill('textarea', 'Explain what a KV cache does, in plain words');
  await page.waitForTimeout(200);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(2600);
  await shot('20-chat-streaming');
  await page.waitForTimeout(6000);
  await shot('21-chat-done');
  await page.goto(`${BASE}/#/monitor`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1600);
  await shot('22-monitor-after-chat');
  await page.goto(`${BASE}/#/serving`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1400);
  await shot('23-serving-with-requests');
}

if (flow === 'expert') {
  await page.goto(`${BASE}/#/tuning`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(900);
  /* The mode control is `.st-tab`, not `.seg__btn` — the flow kept the older
     class and had been failing on a selector that no longer existed. */
  await page.click('.st-tab:has-text("Expert")');
  await page.waitForTimeout(900);
  await shot('30-tuning-expert');
  await page.fill('input[placeholder="Search parameters"]', 'cache');
  await page.waitForTimeout(600);
  await shot('31-tuning-expert-search');
  await page.click('.st-tab:has-text("Raw")');
  await page.waitForTimeout(600);
  await shot('32-tuning-raw');
}

if (flow === 'expert-light') {
  await page.goto(`${BASE}/#/tuning`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  await page.click('.st-tab:has-text("Expert")');
  await page.waitForTimeout(800);
  await shot('98-expert-light');
}

if (flow === 'setup') {
  const next = () => page.getByRole('button', { name: 'Continue', exact: true }).first();
  await page.goto(`${BASE}/#/setup`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1800);
  await shot('40-setup-1-check');
  /* The wizard only offers "Continue" while it is still being worked through.
     Once setup is complete the host says so and the screen shows the finished
     path instead — so this flow reports that rather than timing out on a button
     that is correctly absent. */
  const state = await page.evaluate(async () => {
    const r = await fetch('/api/setup').then((x) => x.json());
    return { complete: r.complete };
  });
  if (state.complete) {
    console.log('setup is already complete on this host — the wizard shows its finished path.');
    console.log('The mid-wizard steps are not reachable without resetting it; see plan 09.');
    await browser.close();
    process.exit(0);
  }
  await next().click();
  await page.waitForTimeout(900);
  await shot('41-setup-2-model');
  await page.getByText('Qwen · Q2_K', { exact: true }).first().click();
  await page.waitForTimeout(400);
  await next().click();
  await page.waitForTimeout(800);
  await shot('42-setup-3-basics');
  await page.getByRole('button', { name: /Download & install/ }).click();
  await page.waitForTimeout(3400);
  await shot('43-setup-4-install');
  await page.waitForTimeout(17000);
  await shot('44-setup-4-done');
}

if (flow === 'tools') {
  await page.click('text=New conversation');
  await page.waitForTimeout(500);
  await page.fill('textarea', 'Read the README in this workspace and summarise it for me');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(3200);
  await shot('70-tool-approval');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(700);
  await shot('71-tool-approved');
  await page.waitForTimeout(6000);
  await shot('72-tool-answered');
}

if (flow === 'readysummary') {
  await page.goto(`${BASE}/#/settings`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2300);
  await page.click('text=Run setup again');
  await page.waitForTimeout(2500);
  const here = await page.evaluate(() => document.querySelector('.step[aria-current]')?.textContent.replace(/\s+/g, ' ').trim());
  console.log('wizard opens at:', here);

  const out = await page.evaluate(() => {
    const grids = [...document.querySelectorAll('.setupmodal__surface .metricgrid')];
    const grid = grids[grids.length - 1];
    if (!grid) return null;
    const cards = [...grid.querySelectorAll('.st-metric')];
    const tops = cards.map((m) => Math.round(m.closest('.st-card').getBoundingClientRect().top));
    const perRow = {};
    tops.forEach((t) => { perRow[t] = (perRow[t] ?? 0) + 1; });
    return {
      items: cards.map((m) => ({
        label: m.querySelector('.st-metric__label')?.textContent.trim(),
        value: m.querySelector('.st-metric__value')?.textContent.trim(),
        sub: m.querySelector('.spec__sub')?.textContent.trim(),
      })),
      perRow: Object.values(perRow),
    };
  });
  if (!out) { console.log('summary not found'); }
  else {
    out.items.forEach((c) => console.log(`  ${String(c.label).padEnd(14)} ${String(c.value).padEnd(22)} ${c.sub ?? ''}`));
    const ctx = out.items.find((c) => c.label === 'Context');
    const api = out.items.find((c) => c.label === 'Local API');
    console.log('context shows a size, not a choice:', /^\d+K$/i.test(ctx?.value ?? ''), '<- it said "Recommended"');
    console.log('local API leads with plain meaning:', api?.value === 'On this computer', '<- it said "127.0.0.1"');
    console.log('cards per row                    :', JSON.stringify(out.perRow),
      out.perRow.every((n) => n === out.perRow[0]) ? '(balanced)' : '(one orphaned)');
  }
  await shot('580-ready-summary');
}

if (flow === 'addskill') {
  const calls = [];
  await page.route('**/api/skills/add', (r) => {
    calls.push(r.request().method());
    return r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ cancelled: true }) });
  });
  await page.goto(`${BASE}/#/skills`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1800);

  await page.fill('input[placeholder="Search skills"]', 'commit');
  await page.waitForTimeout(400);
  const typed = await page.inputValue('input[placeholder="Search skills"]');
  await page.click('text=Add from folder');
  await page.waitForTimeout(1100);
  const after = await page.inputValue('input[placeholder="Search skills"]');
  console.log('search box untouched:', after === typed, '<- the button used to clear it, and do nothing else');
  console.log('asks the host to add:', calls.length === 1, `(${calls[0] ?? 'no'} request)`);

  const titles = await page.evaluate(() =>
    [...document.querySelectorAll('.st-tab')].slice(0, 3).map((t) => `${t.textContent} — ${t.title}`));
  console.log('scope options now say:');
  titles.forEach((t) => console.log('   ', t));
  console.log('"Here" names the workspace:', /design-iteration|workspace/.test(titles[1] ?? ''), '<- it said only "Here"');
  await shot('570-add-from-folder');
}

if (flow === 'filepreview') {
  await page.goto(`${BASE}/#/workspace`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1800);
  await page.evaluate(() => {
    const file = [...document.querySelectorAll('.treerow')].find((r) => r.dataset.kind === 'file');
    file?.click();
  });
  await page.waitForTimeout(1300);

  const read = () => page.evaluate(() => {
    const el = document.querySelector('.filepreview__body');
    if (!el) return null;
    const parse = (c) => { const n = String(c).match(/[\d.]+/g); return n ? { r: +n[0], g: +n[1], b: +n[2], a: n[3] === undefined ? 1 : +n[3] } : null; };
    const lum = ({ r, g, b }) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
    const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x); return Number(((hi + 0.05) / (lo + 0.05)).toFixed(2)); };
    const cs = getComputedStyle(el);
    let bg = null;
    let n = el;
    while (n) { const c = parse(getComputedStyle(n).backgroundColor); if (c && c.a > 0.9) { bg = c; break; } n = n.parentElement; }
    return {
      ratio: ratio(parse(cs.color), bg),
      overflowX: cs.overflowX,
      clipped: el.scrollWidth - el.clientWidth,
      scrollable: el.scrollWidth > el.clientWidth && cs.overflowX !== 'hidden',
      lines: el.textContent.split('\n').length,
      longest: Math.max(...el.textContent.split('\n').map((l) => l.length)),
    };
  });

  const before = await read();
  console.log('contrast          :', before.ratio, '(needs 4.5) <- it was #303030 on #1b1f23, a ratio of 1.26');
  console.log('lines / longest   :', before.lines, '/', before.longest, 'chars');
  console.log('overflows by      :', before.clipped + 'px');
  console.log('can be reached    :', before.scrollable, '<- overflow was hidden, so the rest of every long line was unreachable');

  await page.evaluate(() => { document.querySelector('.filepreview__body').scrollLeft = 300; });
  await page.waitForTimeout(300);
  const scrolled = await page.evaluate(() => document.querySelector('.filepreview__body').scrollLeft);
  console.log('scrolls           :', scrolled > 0, `(scrollLeft ${scrolled})`);
  await shot('560-file-preview');
}

if (flow === 'endpoints') {
  await page.goto(`${BASE}/#/serving`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2000);
  const read = () => page.evaluate(() => [...document.querySelectorAll('.compatrow')].map((r) => {
    const url = r.querySelector('.compatrow__url');
    const lines = (el) => { const rg = document.createRange(); rg.selectNodeContents(el); return rg.getClientRects().length; };
    return {
      label: r.querySelector('.compatrow__label').textContent,
      url: url.textContent,
      clipped: url.scrollWidth > url.clientWidth + 1,
      urlLines: lines(url),
      rowH: Math.round(r.getBoundingClientRect().height),
    };
  }));

  const wide = await read();
  wide.forEach((r) => console.log(`  ${r.label.padEnd(22)} ${r.rowH}px  url ${r.urlLines} line(s)  ${r.clipped ? 'CLIPPED' : 'full'}`));
  console.log('nothing truncated      :', wide.every((r) => !r.clipped), '<- the longest address used to end in "…/completi…"');
  console.log('rows equal at this width:', new Set(wide.map((r) => r.rowH)).size === 1);

  const access = await page.evaluate(() => {
    const card = [...document.querySelectorAll('.grid--2 > .st-card')].find((c) => /^Access/.test(c.textContent.trim()));
    const r = card.getBoundingClientRect();
    const rows = [...card.querySelectorAll('.field, .row')];
    const last = Math.max(...rows.map((x) => x.getBoundingClientRect().bottom));
    return { cardH: Math.round(r.height), contentH: Math.round(last - r.top) };
  });
  console.log('access card           :', access.cardH + 'px card,', access.contentH + 'px content',
    '<- it was 758px tall holding 260px');

  await page.click('.compatrow .st-btn--icon >> nth=0');
  await page.waitForTimeout(400);
  const copied = await page.evaluate(() => navigator.clipboard.readText().catch(() => 'unavailable'));
  console.log('copy still returns the full url:', copied);
  await shot('550-endpoints');
}

if (flow === 'contexthistory') {
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  await page.click('text=New session');
  await page.waitForTimeout(400);
  const ctx = () => page.evaluate(async () => {
    const m = await fetch('/api/monitor').then((r) => r.json());
    return { used: m.context.usedTokens, msgs: m.context.parts.messages, pct: Number(m.context.pct.toFixed(1)) };
  });
  const start = await ctx();
  console.log('start        :', JSON.stringify(start));

  let prev = start.msgs;
  let grew = 0;
  for (const msg of ['write a short paragraph about tides', 'and one about the moon', 'and one about wind']) {
    await page.fill('textarea', msg);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(8000);
    await page.keyboard.press('Enter').catch(() => {});
    await page.waitForTimeout(4000);
    const c = await ctx();
    const up = c.msgs > prev;
    if (up) grew += 1;
    console.log(`turn ${grew + 1}       : used ${c.used}, messages-part ${c.msgs}`, up ? '' : '<- NO GROWTH');
    prev = c.msgs;
  }
  console.log('grows with the conversation:', grew >= 2, '<- it sat at ~350 however long the conversation got');
  console.log('the model receives prior turns:', grew >= 2, '<- history was never sent, so every message was a first message');
  const end = await ctx();
  console.log('used == current occupancy, not a running total:', end.used < 32768, `(${end.pct}%)`);
  await shot('540-context-history');
}

if (flow === 'effectpanel') {
  await page.goto(`${BASE}/#/tuning`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2000);
  const out = await page.evaluate(() => {
    const rows = [...document.querySelectorAll('.effectrow')].map((r) => {
      const label = r.querySelector(':first-child');
      const value = r.querySelector(':last-child');
      // where does the text actually break?
      const range = document.createRange();
      range.selectNodeContents(value);
      const boxes = [...range.getClientRects()].map((b) => Math.round(b.width));
      return { label: label.textContent.trim(), value: value.textContent.trim(), lines: boxes.length, widths: boxes };
    });
    const notes = [...document.querySelectorAll('.st-card ul li')].map((li) => li.textContent.trim());
    return { rows, notes, breakdown: notes.filter((n) => /weights/.test(n)).length };
  });
  out.rows.forEach((r) => console.log(`  ${r.label.padEnd(16)} ${r.lines} line(s)  ${JSON.stringify(r.widths)}  ${r.value}`));
  console.log('model breakdown mentioned:', out.breakdown, 'time(s) <- it appeared twice');
  console.log('notes carry it once      :', out.breakdown === 1);
  await shot('530-effect-panel');
}

if (flow === 'loadguard') {
  await page.goto(`${BASE}/#/models`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1900);

  const rows = await page.evaluate(() =>
    [...document.querySelectorAll('.listrow')]
      .filter((r) => r.querySelector('.listrow__title'))
      .map((r) => {
        const btn = r.querySelector('.listrow__side .st-btn');
        return {
          model: r.querySelector('.listrow__title').textContent.trim().slice(0, 20),
          verdict: [...r.querySelectorAll('.st-pill')].map((x) => x.textContent.trim()).join('/'),
          button: btn?.textContent.trim(),
          primary: btn ? btn.className.includes('primary') : null,
        };
      }));
  const oversized = rows.filter((r) => /Too large/.test(r.verdict));
  console.log('unusable variants :', oversized.length);
  console.log('their button      :', JSON.stringify(oversized.map((r) => r.button)));
  console.log('none is a primary action:', oversized.every((r) => r.primary === false),
    '<- a green one-click Load invited an action the app itself says cannot work');

  // clicking one must ask first, and say what will happen
  let asked = null;
  page.on('dialog', async (d) => {
    asked = d.message();
    await d.dismiss();
  });
  await page.evaluate(() => {
    const r = [...document.querySelectorAll('.listrow')].find((x) => /Too large/.test(x.textContent));
    r?.querySelector('.listrow__side .st-btn')?.click();
  });
  await page.waitForTimeout(1200);
  console.log('asks first        :', asked !== null);
  console.log('names the cost    :', JSON.stringify(asked?.split('\n')[0]));
  const loaded = await page.evaluate(async () => (await fetch('/api/models').then((x) => x.json())).loaded);
  console.log('dismissing loads nothing:', loaded === 'qwen2.5-0.5b-q4km', `(loaded: ${loaded})`);
  await shot('520-load-guard');
}

if (flow === 'focusmode') {
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'say hello in one line');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(8000);
  await page.keyboard.press('Enter').catch(() => {});
  await page.waitForTimeout(4000);
  // A second turn, so there is an earlier exchange to recede.
  await page.fill('textarea', 'and one more line');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(8000);
  await page.keyboard.press('Enter').catch(() => {});
  await page.waitForTimeout(4000);

  const read = () => page.evaluate(() => {
    const g = (sel) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { l: Math.round(r.left), r: Math.round(r.right), h: Math.round(r.height) };
    };
    const bubble = g('.st-msg--assistant .st-bubble');
    const composer = g('.st-composer');
    return {
      composer: composer?.h,
      reading: Math.round(document.querySelector('.main__body.chat').getBoundingClientRect().height),
      alignL: bubble && composer ? bubble.l - composer.l : null,
      alignR: bubble && composer ? composer.r - bubble.r : null,
      hints: (() => { const e = document.querySelector('.composer__hints'); return e ? getComputedStyle(e).display : 'absent'; })(),
      icons: [...document.querySelectorAll('.st-composer__bar .st-btn--icon')].filter((x) => getComputedStyle(x).display !== 'none').length,
      telemetry: (() => { const e = document.querySelector('.telemetry'); return e ? getComputedStyle(e).display : 'absent'; })(),
      canType: !!document.querySelector('.composer textarea'),
      barKeys: [...document.querySelectorAll('.focusbar__keys > span')].map((s) => s.textContent.replace(/\s+/g, ' ').trim()),
      exitLabel: document.querySelector('.focusbar button:last-of-type')?.textContent.trim() ?? '',
      exitTitle: document.querySelector('.focusbar button:last-of-type')?.getAttribute('title'),
    };
  });

  const normal = await read();
  console.log('normal      : composer', normal.composer + 'px, reading area', normal.reading + 'px, alignment', normal.alignL);

  await page.keyboard.press('Meta+Shift+F');
  await page.mouse.move(1200, 820);
  await page.waitForTimeout(1100);
  const focus = await read();
  console.log('focus       : composer', focus.composer + 'px, reading area', focus.reading + 'px');
  console.log('reclaimed   :', focus.reading - normal.reading, 'px for the conversation');
  console.log('composer aligns with the cards:', focus.alignL === 0 && focus.alignR === 0);
  console.log('hint line   :', focus.hints, '| icon strip:', focus.icons, 'icons | telemetry:', focus.telemetry);
  console.log('still usable:', focus.canType, '- the point is removing chrome, not the ability to reply');
  console.log('keys grouped:', JSON.stringify(focus.barKeys));
  console.log('exit is quiet:', focus.exitLabel === '', '| labelled', JSON.stringify(focus.exitTitle));

  /* Earlier turns recede — or did not, for as long as this mode has existed.
     `.st-msg` carried `animation: st-enter ... both`, and an animated value
     outranks a normal declaration, so the keyframe's closing `opacity: 1` held
     every turn at full strength. The dimming rule was in the stylesheet,
     matched the element, and lost the cascade. */
  const opacities = await page.evaluate(() =>
    [...document.querySelectorAll('.chat__inner > .st-msg')].map((m) => getComputedStyle(m).opacity));
  const earlier = opacities.slice(0, -2);
  const current = opacities.slice(-2);
  console.log('earlier turns recede :', earlier.length > 0 && earlier.every((o) => Number(o) < 0.9),
    JSON.stringify(opacities), `(${opacities.length} turns)`);
  console.log('current stays full   :', current.every((o) => Number(o) > 0.9));
  await shot('510-focus-mode');
}

if (flow === 'railgeometry') {
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'read the readme');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await page.keyboard.press('Enter').catch(() => {});
  await page.waitForTimeout(4000);
  await page.evaluate(() => document.querySelector('.nav__collapse')?.click());
  await page.mouse.move(900, 700);
  await page.waitForTimeout(800);

  const out = await page.evaluate(() => {
    const nav = document.querySelector('.nav');
    const right = nav.getBoundingClientRect().right;
    const items = [...nav.querySelectorAll('button, a, .navsession, .ws__name, .nav__brand')]
      .filter((el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; });
    // text actually painted, not merely present in the DOM
    const painted = (el) => [...el.querySelectorAll('*')].some((c) =>
      !c.children.length && getComputedStyle(c).display !== 'none' &&
      c.getBoundingClientRect().width > 6 && c.textContent.trim());
    return {
      items: items.length,
      overflowing: items.filter((el) => el.getBoundingClientRect().right > right + 0.5).length,
      clippedText: items.filter(painted).length,
      sizes: [...new Set(items.map((el) => {
        const r = el.getBoundingClientRect();
        return `${Math.round(r.width)}x${Math.round(r.height)}`;
      }))],
      unlabelled: items.filter((el) => el.querySelector('svg') &&
        !(el.getAttribute('title') || el.getAttribute('aria-label') || el.textContent.trim())).length,
    };
  });
  console.log('items              :', out.items);
  console.log('clipped text       :', out.clippedText, '<- a conversation used to render as the first 5px of its title');
  console.log('overflowing the rail:', out.overflowing, '<- the 22px tree indent pushed a session past the edge');
  console.log('target sizes       :', JSON.stringify(out.sizes));
  console.log('icons with no label:', out.unlabelled);
  await shot('500-rail-geometry');
}

if (flow === 'archive') {
  for (const m of ['keep me', 'archive me']) {
    await page.click('text=New session');
    await page.waitForTimeout(400);
    await page.fill('textarea', m);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(7000);
    await page.keyboard.press('Enter').catch(() => {});
    await page.waitForTimeout(3200);
  }
  const rows = () => page.evaluate(() =>
    [...document.querySelectorAll('.navsession__title')].map((t) => t.textContent));
  console.log('before        :', JSON.stringify(await rows()));

  await page.click('.navsession >> nth=0', { button: 'right' });
  await page.waitForTimeout(500);
  console.log('menu          :', JSON.stringify(await page.evaluate(() =>
    [...document.querySelectorAll('.menu__item')].map((i) => i.textContent.trim()))));
  await page.click('.menu__item:has-text("Archive")');
  await page.waitForTimeout(700);
  console.log('after archive :', JSON.stringify(await rows()), '<- leaves the list, nothing deleted');

  // the search field is behind its own toggle, so open it the way the sidebar does
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('button')]
      .find((x) => /search/i.test(x.getAttribute('title') ?? x.getAttribute('aria-label') ?? ''));
    b?.click();
  });
  await page.waitForTimeout(500);
  await page.fill('input[placeholder="Search conversations"]', 'archive me');
  await page.waitForTimeout(700);
  console.log('found by search:', JSON.stringify(await rows()), '<- archived stays findable');
  await shot('490-archive');
}

if (flow === 'aboutengine') {
  await page.goto(`${BASE}/#/settings`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2100);
  const metrics = await page.evaluate(() =>
    [...document.querySelectorAll('.settings-body .st-metric')]
      .map((m) => ({
        label: m.querySelector('.st-metric__label')?.textContent.trim(),
        value: m.querySelector('.st-metric__value')?.textContent.trim(),
        sub: m.querySelector('.spec__sub')?.textContent.trim(),
      }))
      .filter((m) => m.label));
  metrics.forEach((m) => console.log(`  ${String(m.label).padEnd(14)} ${String(m.value).padEnd(10)} ${m.sub ?? ''}`));
  const engine = metrics.find((m) => m.label === 'Engine');
  console.log('engine named    :', !!engine, '<- the one thing About was missing');
  console.log('with its backend:', /metal|vulkan|cuda|cpu/.test(engine?.sub ?? ''));
  // folder paths belong to Storage, not duplicated here
  console.log('no duplicated paths:', !metrics.some((m) => /folder/i.test(m.label)));
  await shot('480-about-engine');
}

if (flow === 'rowweight') {
  await page.goto(`${BASE}/#/models`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1900);
  const rows = await page.evaluate(() =>
    [...document.querySelectorAll('.listrow')]
      .filter((r) => r.querySelector('.listrow__title'))
      .map((r) => ({
        title: r.querySelector('.listrow__title').textContent.trim().slice(0, 22),
        rec: r.hasAttribute('data-recommended'),
        size: getComputedStyle(r.querySelector('.listrow__title')).fontSize,
        edge: getComputedStyle(r, '::after').width,
        separator: getComputedStyle(r, '::before').height,
      })));
  const rec = rows.find((r) => r.rec);
  const rest = rows.filter((r) => !r.rec);
  console.log('recommended row  :', JSON.stringify(rec));
  console.log('one step larger  :', rec && rest.every((r) => parseFloat(rec.size) > parseFloat(r.size)));
  console.log('accent edge only on it:', rec?.edge === '2px' && rest.every((r) => r.edge === 'auto'));
  console.log('separator intact  :', rows.every((r) => r.separator === '1px'), '<- the edge uses ::after, so ::before keeps the separator');
  await shot('470-model-weight');
}

if (flow === 'dismissal') {
  // Needs a transcript with room to scroll, or "scroll" tests nothing.
  const tall = await browser.newPage({ viewport: { width: 1512, height: 560 }, deviceScaleFactor: 1 });
  await tall.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await tall.waitForTimeout(1500);
  await tall.click('text=New session');
  await tall.waitForTimeout(400);
  for (const m of ['write a paragraph about tides', 'and another about the moon']) {
    await tall.fill('textarea', m);
    await tall.keyboard.press('Enter');
    await tall.waitForTimeout(7500);
    await tall.keyboard.press('Enter').catch(() => {});
    await tall.waitForTimeout(3200);
  }
  const room = await tall.evaluate(() => {
    const el = document.querySelector('.main__body.chat');
    return el.scrollHeight - el.clientHeight;
  });
  console.log('scroll room      :', room, 'px (needed, or the scroll test proves nothing)');

  const isOpen = (kind) => tall.evaluate((k) =>
    k === 'telepop' ? !!document.querySelector('.telepop') : !!document.querySelector('.modal[data-open="true"]'), kind);
  const blur = () => tall.evaluate(() => document.activeElement?.blur());

  for (const [name, open] of [
    ['palette', async () => { await blur(); await tall.keyboard.press('Meta+k'); }],
    ['shortcuts', async () => { await blur(); await tall.keyboard.press('?'); }],
    ['telepop', async () => { await tall.click('.telemetry__chip >> nth=0'); }],
  ]) {
    const out = [];
    for (const [how, act] of [
      ['esc', async () => { await blur(); await tall.keyboard.press('Escape'); }],
      ['outside', async () => { await tall.mouse.click(200, 300); }],
      ['scroll', async () => {
        // upward: the transcript starts pinned to the bottom, so adding to
        // scrollTop clamps to nothing and no scroll event fires
        await tall.evaluate(() => {
          const el = document.querySelector('.main__body.chat');
          el.scrollTop = Math.max(0, el.scrollTop - 150);
        });
      }],
    ]) {
      await open();
      await tall.waitForTimeout(550);
      if (!(await isOpen(name))) { out.push(`${how}: N/A`); continue; }
      await act();
      await tall.waitForTimeout(500);
      out.push(`${how}: ${(await isOpen(name)) ? 'stays' : 'closes'}`);
    }
    console.log(`${name.padEnd(10)}: ${out.join(' | ')}`);
    await tall.keyboard.press('Escape');
    await tall.waitForTimeout(300);
  }
  console.log('rule: anchored overlays leave with their content; unanchored ones ignore scroll');
  await tall.screenshot({ path: '.shots/460-dismissal.png' });
  await tall.close();
}

if (flow === 'railweight') {
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'say hello');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await page.keyboard.press('Enter').catch(() => {});
  await page.waitForTimeout(3500);
  await page.click('button[title="Conversation context"]');
  await page.waitForTimeout(800);

  const out = await page.evaluate(() => {
    const sec = document.querySelector('.railsection');
    const cs = getComputedStyle(sec);
    const body = document.querySelector('.contextpanel__body')?.innerText ?? '';
    return {
      sectionIsQuiet: cs.backgroundColor === 'rgba(0, 0, 0, 0)' && cs.boxShadow === 'none',
      panelIsOneSurface: getComputedStyle(document.querySelector('.contextpanel')).backgroundColor !== 'rgba(0, 0, 0, 0)',
      heads: [...document.querySelectorAll('.railsection .h4')].map((h) => h.textContent.trim()),
      contextWindowMentions: (body.match(/Context window/g) ?? []).length,
    };
  });
  console.log('sections are quiet   :', out.sectionIsQuiet, '(no fill, no shadow)');
  console.log('the rail is one object:', out.panelIsOneSurface);
  console.log('section heads        :', JSON.stringify(out.heads));
  console.log('no repeated heading  :', out.contextWindowMentions === 1, '<- read "Context window / Context window / 6%" before');
  await shot('450-rail-weight');
}

if (flow === 'skillscope') {
  await page.goto(`${BASE}/#/skills`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1900);
  const read = () => page.evaluate(() =>
    [...document.querySelectorAll('.listrow')].map((r) => ({
      name: r.querySelector('.listrow__title')?.textContent.slice(0, 18),
      where: r.querySelector('.skillScope__where')?.textContent.trim() ?? null,
    })));
  console.log('at rest    :', JSON.stringify(await read()));

  // narrow the first skill to this workspace, through the control
  await page.evaluate(() => {
    const row = document.querySelector('.listrow');
    const here = [...row.querySelectorAll('.skillScope button')].find((b) => /Here/.test(b.textContent));
    here?.click();
  });
  await page.waitForTimeout(900);
  const scoped = await read();
  console.log('after Here :', JSON.stringify(scoped[0]), '<- narrowed');

  // the host should honour it: fires here, not elsewhere
  await page.evaluate(async () => {
    await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ message: 'write a commit message for these changes', workspaceId: 'root' }),
    });
  });
  await page.waitForTimeout(2500);
  const here = await page.evaluate(async () => {
    const r = await fetch('/api/skills').then((x) => x.json());
    return r[0].useCount;
  });
  await page.evaluate(async () => {
    await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ message: 'write a commit message for these changes', workspaceId: 'elsewhere' }),
    });
  });
  await page.waitForTimeout(2500);
  const elsewhere = await page.evaluate(async () => {
    const r = await fetch('/api/skills').then((x) => x.json());
    return r[0].useCount;
  });
  console.log('fires in its workspace   :', here);
  console.log('silent in another         :', elsewhere === here);

  // widen again so the flow leaves no trace
  await page.evaluate(() => {
    const row = document.querySelector('.listrow');
    const always = [...row.querySelectorAll('.skillScope button')].find((b) => /Always/.test(b.textContent));
    always?.click();
  });
  await page.waitForTimeout(900);
  console.log('restored   :', JSON.stringify((await read())[0]));
  await shot('440-skill-scope');
}

if (flow === 'turnnav') {
  const tall = await browser.newPage({ viewport: { width: 1512, height: 640 }, deviceScaleFactor: 1 });
  await tall.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await tall.waitForTimeout(1500);
  await tall.click('text=New session');
  await tall.waitForTimeout(400);
  for (const m of ['first question about tides', 'second question about the moon', 'third question about wind']) {
    await tall.fill('textarea', m);
    await tall.keyboard.press('Enter');
    await tall.waitForTimeout(7500);
    await tall.keyboard.press('Enter').catch(() => {});
    await tall.waitForTimeout(3200);
  }
  await tall.keyboard.press('Meta+Shift+F');
  await tall.waitForTimeout(800);
  console.log('focus bar  :', await tall.evaluate(() => !!document.querySelector('.focusbar')));
  console.log('keys taught:', await tall.evaluate(() =>
    document.querySelector('.focusbar__keys')?.textContent.replace(/\s+/g, ' ').trim()));

  const pos = () => tall.evaluate(() => Math.round(document.querySelector('.main__body.chat').scrollTop));
  await tall.evaluate(() => { document.querySelector('.main__body.chat').scrollTop = 0; });
  await tall.waitForTimeout(600);
  const down = [await pos()];
  for (let i = 0; i < 4; i += 1) {
    await tall.keyboard.press('Alt+ArrowDown');
    await tall.waitForTimeout(750);
    down.push(await pos());
  }
  console.log('Alt+Down   :', JSON.stringify(down));
  console.log('distinct   :', new Set(down).size, 'of', down.length, '<- every press advances, none stick');
  const up = [];
  for (let i = 0; i < 2; i += 1) {
    await tall.keyboard.press('Alt+ArrowUp');
    await tall.waitForTimeout(750);
    up.push(await pos());
  }
  console.log('Alt+Up     :', JSON.stringify(up), '<- retreats');
  await tall.keyboard.press('Meta+ArrowDown');
  await tall.waitForTimeout(1000);
  console.log('Cmd+Down   : gap', await tall.evaluate(() => {
    const el = document.querySelector('.main__body.chat');
    return Math.round(el.scrollHeight - el.scrollTop - el.clientHeight);
  }), '<- ~0');
  await tall.screenshot({ path: '.shots/430-focus-turns.png' });
  await tall.close();
}

if (flow === 'herorest') {
  await page.goto(`${BASE}/#/monitor`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1900);
  const idle = await page.evaluate(() => {
    const rest = document.querySelector('.herorest');
    return {
      resting: !!rest,
      figures: rest
        ? [...rest.querySelectorAll('.eyebrow')].map((e, i) =>
            `${e.textContent}: ${rest.querySelectorAll('.h3')[i]?.textContent.replace(/\s+/g, ' ').trim()}`)
        : [],
      /* Scoped to the hero. The page has five other progress bars — VRAM, RAM,
         temperature and two unlabelled — and an unscoped query finds those and
         reports the hero as unchanged. */
      bar: (document.querySelector('.stack .st-card')?.querySelectorAll('.st-progress').length ?? 0) > 0,
    };
  });
  console.log('idle — resting block :', idle.resting);
  console.log('idle — figures       :', JSON.stringify(idle.figures, null, 1));
  console.log('idle — 0% bar gone   :', !idle.bar);

  await page.evaluate(() => {
    void fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ message: 'write a long paragraph about the sea and the moon' }),
    });
  });
  await page.waitForTimeout(2200);
  const busy = await page.evaluate(() => ({
    phase: document.querySelector('.stack .st-card .h2')?.textContent,
    resting: !!document.querySelector('.herorest'),
    bar: (document.querySelector('.stack .st-card')?.querySelectorAll('.st-progress').length ?? 0) > 0,
    label: [...document.querySelectorAll('.stack .st-card .muted')].map((e) => e.textContent)[0],
  }));
  console.log('busy — phase         :', JSON.stringify(busy.phase));
  console.log('busy — progress bar  :', busy.bar, '| labelled', JSON.stringify(busy.label), '| resting hidden:', !busy.resting);
  await shot('420-hero-rest');
}

if (flow === 'sparkhover') {
  await page.goto(`${BASE}/#/monitor`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  // give the series something to show
  await page.evaluate(() => {
    void fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ message: 'write a long paragraph about the sea and the moon and the wind' }),
    });
  });
  await page.waitForTimeout(9000);
  const box = await page.evaluate(() => {
    const el = document.querySelector('.spark');
    const r = el.getBoundingClientRect();
    return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
  });
  for (const frac of [0.75, 0.9, 0.98]) {
    await page.mouse.move(box.x + box.w * frac, box.y + box.h / 2);
    await page.waitForTimeout(300);
    console.log(`at ${String(Math.round(frac * 100)).padStart(2)}% :`, JSON.stringify(await page.evaluate(() =>
      document.querySelector('.spark__readout')?.textContent ?? 'no readout')));
  }
  console.log('marker drawn :', await page.evaluate(() => !!document.querySelector('.spark line')));
  await page.mouse.move(10, 400);
  await page.waitForTimeout(300);
  console.log('clears on leave:', await page.evaluate(() => !document.querySelector('.spark__readout')));
  await shot('410-spark-hover');
}

if (flow === 'whythisone') {
  await page.goto(`${BASE}/#/models`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1900);
  const out = await page.evaluate(() => {
    const why = document.querySelector('.whyline');
    return {
      recommendedPills: [...document.querySelectorAll('.st-pill')].filter((x) => /Recommended/.test(x.textContent)).length,
      why: why?.textContent.replace(/\s+/g, ' ').trim() ?? null,
      onRow: why?.closest('.listrow')?.querySelector('.listrow__title')?.textContent.trim().slice(0, 28) ?? null,
      // exactly one variant is recommended, so exactly one explains itself
      whyCount: document.querySelectorAll('.whyline').length,
    };
  });
  console.log('recommended pills:', out.recommendedPills, '(want exactly 1)');
  console.log('explains itself  :', JSON.stringify(out.why));
  console.log('on which row     :', JSON.stringify(out.onRow));
  console.log('one reason only  :', out.whyCount === 1);
  await shot('400-why-this-one');
}

if (flow === 'paramgroups') {
  await page.goto(`${BASE}/#/tuning`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1700);
  await page.click('.st-tab:has-text("Expert")');
  await page.waitForTimeout(900);
  const out = await page.evaluate(() => ({
    headings: [...document.querySelectorAll('.paramgroup')].map((r) => r.textContent.replace(/\s+/g, ' ').trim()),
    params: document.querySelectorAll('.st-table tbody tr:not(.paramgroup)').length,
    // the group name should appear as a heading, not repeated on every row
    repeats: [...document.querySelectorAll('td .xs.muted')]
      .filter((e) => /^(Context & memory|Performance|Sampling|Server|Advanced)$/.test(e.textContent)).length,
  }));
  console.log('sections     :', JSON.stringify(out.headings, null, 1));
  console.log('param rows   :', out.params);
  console.log('name repeats :', out.repeats, '<- 0 means structure, not a label read 30 times');
  await page.fill('input[placeholder="Search parameters"]', 'cache');
  await page.waitForTimeout(600);
  console.log('after search :', JSON.stringify(await page.evaluate(() =>
    [...document.querySelectorAll('.paramgroup')].map((r) => r.textContent.replace(/\s+/g, ' ').trim()))));
  await shot('390-param-groups');
}

if (flow === 'dlcost') {
  await page.goto(`${BASE}/#/models`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1800);
  console.log('visible rows:', await page.evaluate(() => document.querySelectorAll('.listrow').length));
  /* Scoped to the body: the sidebar also has a "More" button, and an unscoped
     text match clicks that one instead. */
  await page.click('.main__body button:has-text("Show")');
  await page.waitForTimeout(900);
  const cost = await page.evaluate(() =>
    [...document.querySelectorAll('button')].map((b) => b.textContent.trim()).filter((t) => /Download/.test(t)));
  console.log('download cost as size and time:', JSON.stringify(cost, null, 1));
  console.log('every download names a time:', cost.length > 0 && cost.every((c) => /about \d+ min/.test(c)));
  await shot('380-download-cost');
}

if (flow === 'servingsort') {
  await page.goto(`${BASE}/#/serving`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1800);
  const first = () => page.evaluate(() => {
    const row = document.querySelector('tbody tr');
    return row ? [...row.querySelectorAll('td')].map((td) => td.textContent.trim())[0] : 'no rows';
  });
  const active = () => page.evaluate(() => {
    const th = document.querySelector('th [aria-sort]:not([aria-sort="none"])');
    return th ? `${th.textContent.trim()} ${th.getAttribute('aria-sort')}` : 'none';
  });
  console.log('rows        :', await page.evaluate(() => document.querySelectorAll('tbody tr').length));
  console.log('default     :', await active(), '| first row', await first());
  await page.click('th:has-text("Duration") button');
  await page.waitForTimeout(450);
  console.log('by duration :', await active(), '| first row', await first());
  await page.click('th:has-text("Duration") button');
  await page.waitForTimeout(450);
  console.log('reversed    :', await active(), '| first row', await first(), '<- the order really changed');
  await page.click('th:has-text("Client") button');
  await page.waitForTimeout(450);
  console.log('by client   :', await active());
  console.log('one column sorted at a time:', await page.evaluate(() =>
    document.querySelectorAll('th [aria-sort]:not([aria-sort="none"])').length) === 1);
  await shot('370-serving-sort');
}

if (flow === 'skillgroups') {
  await page.goto(`${BASE}/#/skills`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1900);
  const read = () => page.evaluate(() => ({
    groups: [...document.querySelectorAll('.skillgroup__head')].map((h) => h.textContent.replace(/\s+/g, ' ').trim()),
    rows: document.querySelectorAll('.listrow').length,
  }));
  const all = await read();
  console.log('groups        :', JSON.stringify(all.groups));
  console.log('rows          :', all.rows);
  await page.fill('input[type=search], .st-input', 'commit');
  await page.waitForTimeout(600);
  const filtered = await read();
  console.log('filtered      :', JSON.stringify(filtered.groups), '<- empty groups drop out');
  console.log('matching rows :', JSON.stringify(await page.evaluate(() =>
    [...document.querySelectorAll('.listrow__title')].map((t) => t.textContent))));
  await shot('360-skill-groups');
}

if (flow === 'overlayresize') {
  const wide = await browser.newPage({ viewport: { width: 1512, height: 1000 }, deviceScaleFactor: 1 });
  await wide.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await wide.waitForTimeout(1500);
  await wide.click('text=New session');
  await wide.waitForTimeout(400);
  await wide.fill('textarea', 'hello');
  await wide.keyboard.press('Enter');
  await wide.waitForTimeout(7000);
  await wide.keyboard.press('Enter').catch(() => {});
  await wide.waitForTimeout(3500);
  await wide.click('.telemetry__chip >> nth=0');
  await wide.waitForTimeout(500);

  const read = () => wide.evaluate(() => {
    const el = document.querySelector('.telepop');
    if (!el) return { present: false };
    const r = el.getBoundingClientRect();
    return { present: true, right: Math.round(r.right), vw: window.innerWidth,
             offscreen: r.right > window.innerWidth + 2 || r.left < -2 };
  });
  console.log('at 1512 :', JSON.stringify(await read()));
  for (const w of [1100, 900]) {
    await wide.setViewportSize({ width: w, height: 1000 });
    await wide.waitForTimeout(500);
    console.log(`at ${w}  :`, JSON.stringify(await read()), '<- follows its chip, stays on screen');
  }
  await wide.setViewportSize({ width: 1512, height: 1000 });
  await wide.close();
}

if (flow === 'completioncue') {
  const short = await browser.newPage({ viewport: { width: 1512, height: 520 }, deviceScaleFactor: 1 });
  await short.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await short.waitForTimeout(1500);
  await short.click('text=New session');
  await short.waitForTimeout(400);
  for (const m of ['write a paragraph about tides', 'and another about the moon']) {
    await short.fill('textarea', m);
    await short.keyboard.press('Enter');
    await short.waitForTimeout(8000);
    await short.keyboard.press('Enter').catch(() => {});
    await short.waitForTimeout(3500);
  }
  await short.evaluate(() => { document.querySelector('.main__body.chat').scrollTop = 0; });
  await short.waitForTimeout(700);
  console.log('pill while scrolled up:', await short.locator('.jumplatest').count());

  await short.fill('textarea', 'one more paragraph please');
  await short.keyboard.press('Enter');
  let saw = false;
  for (let i = 0; i < 90; i += 1) {
    if (await short.evaluate(() => document.querySelector('.jumplatest')?.hasAttribute('data-finished') ?? false)) {
      saw = true;
      break;
    }
    await short.waitForTimeout(300);
  }
  console.log('completion cue fired  :', saw, '<- a reply finished off screen and said so');
  console.log('cue animation         :', JSON.stringify(await short.evaluate(() => {
    const cs = getComputedStyle(document.querySelector('.jumplatest'));
    return { name: cs.animationName, duration: cs.animationDuration };
  })));
  await short.screenshot({ path: '.shots/350-completion-cue.png' });
  await short.close();
}

if (flow === 'privacy') {
  await page.goto(`${BASE}/#/settings`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2200);
  const state = () => page.evaluate(() => {
    const t = [...document.querySelectorAll('.st-toggle')]
      .find((x) => /usage data/i.test(x.getAttribute('aria-label') ?? ''));
    return {
      checked: t?.getAttribute('aria-checked'),
      stored: JSON.parse(localStorage.getItem('strata.prefs.v2') ?? '{}').shareTelemetry,
    };
  });
  console.log('at rest      :', JSON.stringify(await state()), '<- a real false, not an omitted attribute');
  await page.click('.st-toggle[aria-label="Share anonymous usage data"]');
  await page.waitForTimeout(700);
  console.log('after toggling:', JSON.stringify(await state()));
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2200);
  console.log('after reload :', JSON.stringify(await state()), '<- the choice is recorded');

  // a privacy choice is not an appearance setting
  await page.fill('.settingsindex__search', 'reset appearance', { timeout: 8000 });
  await page.waitForTimeout(600);
  const clickIn = (label) => page.evaluate((t) => {
    const f = [...document.querySelectorAll('.settings-body .field')].find((x) => !x.hidden);
    [...f.querySelectorAll('button')].find((b) => b.textContent.trim() === t).click();
  }, label);
  await clickIn('Reset');
  await page.waitForTimeout(400);
  await clickIn('Reset');
  await page.waitForTimeout(900);
  console.log('after a reset:', JSON.stringify(await state()), '<- untouched by an appearance reset');
  await shot('340-privacy');
}

if (flow === 'railclose') {
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'say hello');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await page.keyboard.press('Enter').catch(() => {});
  await page.waitForTimeout(3500);
  await page.click('button[title="Conversation context"]');
  await page.waitForTimeout(700);
  console.log('header     :', JSON.stringify(await page.evaluate(() => {
    const head = document.querySelector('.contextpanel__head');
    const btn = head?.querySelector('button');
    return { closeTitle: btn?.getAttribute('title'), buttonText: btn?.textContent.trim() };
  })));
  console.log('sections   :', await page.evaluate(() =>
    document.querySelectorAll('.contextpanel__sections .railsection').length));
  await page.click('.contextpanel__head button');
  await page.waitForTimeout(800);
  console.log('rail dismissed:', await page.evaluate(() =>
    getComputedStyle(document.querySelector('.contextpanel')).visibility === 'hidden'));
  await shot('330-rail-close');
}

if (flow === 'treekeys') {
  await page.goto(`${BASE}/#/workspace`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1800);
  const at = () => page.evaluate(() => {
    const el = document.activeElement;
    return el?.classList.contains('treerow') ? `${el.dataset.path} (d${el.dataset.depth})` : el?.tagName;
  });
  console.log('tabbable rows:', await page.evaluate(() =>
    document.querySelectorAll('.treerow[tabindex="0"]').length), '(want exactly 1 — roving tabindex)');
  console.log('of          :', await page.evaluate(() => document.querySelectorAll('.treerow').length), 'rows');

  await page.evaluate(() => document.querySelector('.treerow[tabindex="0"]').focus());
  console.log('start       :', await at());
  await page.keyboard.press('ArrowDown');
  console.log('Down        :', await at());
  await page.keyboard.press('ArrowDown');
  console.log('Down        :', await at());
  await page.keyboard.press('ArrowUp');
  console.log('Up          :', await at());
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(350);
  console.log('Right       :', await at(), '<- opens or steps in');
  await page.keyboard.press('ArrowLeft');
  await page.waitForTimeout(350);
  console.log('Left        :', await at(), '<- closes or steps out');
  await page.keyboard.press('End');
  console.log('End         :', await at());
  await page.keyboard.press('Home');
  console.log('Home        :', await at());
  console.log('one tabbable after moving:', await page.evaluate(() =>
    document.querySelectorAll('.treerow[tabindex="0"]').length) === 1);
  console.log('scope figures:', await page.evaluate(() =>
    document.querySelector('.scopes')?.textContent.replace(/\s+/g, ' ').trim()));
  await shot('320-tree-keys');
}

if (flow === 'readmark') {
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1600);
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'read the readme file in the repo');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(9000);
  console.log('proposal names a real file:',
    await page.evaluate(() => /strata\/README\.md/.test(document.body.innerText)));
  await page.click('button:has-text("Approve")');
  await page.waitForTimeout(2500);
  console.log('persisted  :', await page.evaluate(() => localStorage.getItem('strata.touched.v1')));

  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1800);
  await page.evaluate(() => { window.location.hash = '#/workspace'; });
  await page.waitForTimeout(2000);
  console.log('marked rows:', JSON.stringify(await page.evaluate(() =>
    [...document.querySelectorAll('.treerow[data-touched]')].map((r) => r.textContent.trim().slice(0, 22)))));
  console.log('marker title:', await page.evaluate(() =>
    document.querySelector('.treerow[data-touched]')?.getAttribute('title')));
  console.log('unread files unmarked:', await page.evaluate(() =>
    document.querySelectorAll('.treerow:not([data-touched])').length > 50));
  await shot('310-read-marker');
}

if (flow === 'settingsreset') {
  await page.goto(`${BASE}/#/settings`, { waitUntil: 'domcontentloaded' });
  // seed a deliberately non-default set, then reload so the app applies it
  await page.evaluate(() => {
    localStorage.setItem('strata.prefs.v2', JSON.stringify({
      theme: 'dark', rail: true, focus: false, contentFontSize: 22,
      density: 'compact', motion: 'reduced', transparency: 'opaque', uiScale: 1.25,
    }));
  });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2200);
  const read = () => page.evaluate(() => ({
    theme: document.documentElement.dataset.theme ?? '(none)',
    scale: getComputedStyle(document.documentElement).getPropertyValue('--st-ui-scale').trim(),
    rail: JSON.parse(localStorage.getItem('strata.prefs.v2') ?? '{}').rail,
  }));
  console.log('seeded       :', JSON.stringify(await read()));

  await page.fill('.settingsindex__search', 'reset appearance', { timeout: 8000 });
  await page.waitForTimeout(600);
  console.log('visible fields:', JSON.stringify(await page.evaluate(() =>
    [...document.querySelectorAll('.settings-body .field')].filter((f) => !f.hidden)
      .map((f) => f.querySelector('.field__label')?.textContent))));

  const clickIn = (labelText) => page.evaluate((t) => {
    const f = [...document.querySelectorAll('.settings-body .field')].find((x) => !x.hidden);
    [...f.querySelectorAll('button')].find((b) => b.textContent.trim() === t).click();
  }, labelText);

  await clickIn('Reset');
  await page.waitForTimeout(400);
  console.log('confirmation :', JSON.stringify(await page.evaluate(() => {
    const f = [...document.querySelectorAll('.settings-body .field')].find((x) => !x.hidden);
    return [...f.querySelectorAll('button')].map((b) => b.textContent.trim());
  })));
  await clickIn('Keep');
  await page.waitForTimeout(400);
  console.log('after Keep   :', JSON.stringify(await read()), '<- unchanged');

  await clickIn('Reset');
  await page.waitForTimeout(400);
  await clickIn('Reset');
  await page.waitForTimeout(900);
  console.log('after Reset  :', JSON.stringify(await read()), '<- theme and scale restored, rail untouched');
  await shot('300-settings-reset');
}

if (flow === 'jumplatest') {
  const short = await browser.newPage({ viewport: { width: 1512, height: 520 }, deviceScaleFactor: 1 });
  await short.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await short.waitForTimeout(1500);
  await short.click('text=New session');
  await short.waitForTimeout(400);
  for (const m of ['write a paragraph about tides', 'and another about the moon', 'and one about the wind']) {
    await short.fill('textarea', m);
    await short.keyboard.press('Enter');
    await short.waitForTimeout(8000);
    await short.keyboard.press('Enter').catch(() => {});
    await short.waitForTimeout(4000);
  }
  const gap = () => short.evaluate(() => {
    const el = document.querySelector('.main__body.chat');
    return Math.round(el.scrollHeight - el.scrollTop - el.clientHeight);
  });
  console.log('turns            :', await short.locator('.st-msg').count());
  console.log('content-visibility:', await short.evaluate(() => getComputedStyle(document.querySelector('.st-msg')).contentVisibility));
  console.log('pinned at rest   : gap', await gap(), '| pill:', await short.locator('.jumplatest').count(), '<- 0 and hidden');
  await short.evaluate(() => { document.querySelector('.main__body.chat').scrollTop = 0; });
  await short.waitForTimeout(900);
  console.log('scrolled up      : gap', await gap(), '| pill:', await short.locator('.jumplatest').count());
  await short.click('.jumplatest');
  await short.waitForTimeout(2200);
  console.log('after jump       : gap', await gap(), '| pill:', await short.locator('.jumplatest').count(), '<- 0 and hidden');
  await short.screenshot({ path: '.shots/290-jumplatest.png' });
  await short.close();
}

if (flow === 'settingssearch') {
  await page.goto(`${BASE}/#/settings`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1700);
  const vis = () => page.evaluate(() => ({
    fields: [...document.querySelectorAll('.settings-body .field')].filter((f) => !f.hidden)
      .map((f) => f.querySelector('.field__label')?.textContent),
    sections: [...document.querySelectorAll('.settings-body section[id^=settings-]')].filter((s) => !s.hidden).length,
  }));
  const rest = await vis();
  console.log('at rest       :', rest.fields.length, 'fields in', rest.sections, 'sections');

  await page.fill('.settingsindex__search', 'dark');
  await page.waitForTimeout(450);
  console.log('search "dark" :', JSON.stringify((await vis()).fields), '<- matched on the hint, not just the label');

  await page.fill('.settingsindex__search', 'cache');
  await page.waitForTimeout(450);
  console.log('search "cache":', JSON.stringify((await vis()).fields));

  await page.fill('.settingsindex__search', 'zzzz');
  await page.waitForTimeout(450);
  console.log('no match      :', await page.evaluate(() => document.querySelector('.empty__title')?.textContent),
    '| action:', await page.evaluate(() => document.querySelector('.empty button')?.textContent.trim()));
  await page.click('.empty button');
  await page.waitForTimeout(500);
  const back = await vis();
  console.log('after clear   :', back.fields.length, 'fields in', back.sections, 'sections');
  await shot('280-settings-search');
}

if (flow === 'unread') {
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'a thread');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await page.keyboard.press('Enter').catch(() => {});
  await page.waitForTimeout(3500);
  await page.click('text=New session');
  await page.waitForTimeout(600);

  // make the older conversation look updated since it was last read
  await page.evaluate(() => {
    const raw = JSON.parse(localStorage.getItem('strata.sessions.v1') ?? '[]');
    const older = raw.filter((s) => s.messages.length > 0).at(-1);
    if (older) {
      older.updatedAt = new Date(Date.now() + 120000).toISOString();
      older.lastReadAt = new Date(Date.now() - 120000).toISOString();
    }
    localStorage.setItem('strata.sessions.v1', JSON.stringify(raw));
  });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1800);
  const info = await page.evaluate(() => {
    const rows = [...document.querySelectorAll('.ws__session')];
    const unread = rows.find((r) => r.querySelector('.navsession__unread'));
    return {
      marked: rows.map((r) => r.querySelector('.navsession__unread') ? r.querySelector('.navsession__title')?.textContent : null).filter(Boolean),
      dot: unread ? getComputedStyle(unread.querySelector('.navsession__unread')).width : null,
      metaFlex: rows[0] ? getComputedStyle(rows[0].querySelector('.navsession__meta')).flex : null,
    };
  });
  console.log('unread rows   :', JSON.stringify(info.marked));
  console.log('dot is a dot  :', info.dot, '(a broad `.navsession span` rule made this 42px)');
  console.log('timestamp not stretched:', info.metaFlex);
  await shot('270-unread');
}

if (flow === 'servingfilter') {
  await page.goto(`${BASE}/#/serving`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1600);
  const rows = () => page.locator('tbody tr').count();
  const all = await rows();
  console.log('rows total    :', all);
  console.log('filter chips  :', JSON.stringify(await page.evaluate(() =>
    [...document.querySelectorAll('.servefilter button')].map((x) => x.textContent.trim()))));

  await page.click('.servefilter >> text=Errors');
  await page.waitForTimeout(500);
  console.log('errors filter :', await rows(), 'rows |',
    await page.evaluate(() => document.querySelector('.empty__title')?.textContent ?? 'table shown'));

  await page.click('.servefilter >> text=All');
  await page.waitForTimeout(400);
  await page.fill('input[aria-label="Filter requests by client"]', 'zzzz');
  await page.waitForTimeout(500);
  console.log('no match      :', await page.evaluate(() => document.querySelector('.empty__title')?.textContent ?? 'NO MESSAGE'),
    '| action:', await page.evaluate(() => document.querySelector('.empty button')?.textContent.trim()));
  await page.click('.empty button');
  await page.waitForTimeout(500);
  console.log('after clear   :', await rows(), 'rows (was', all + ')');
  console.log('state pills carry a glyph:', await page.evaluate(() =>
    !!document.querySelector('tbody .st-badge svg use')));
  await shot('260-serving-filter');
}

if (flow === 'empties') {
  // Every empty state should state the next move, not only the absence.
  const read = () => page.evaluate(() => {
    const el = document.querySelector('.empty');
    return el
      ? { title: el.querySelector('.empty__title')?.textContent,
          action: el.querySelector('button')?.textContent.trim() ?? null }
      : null;
  });

  await page.goto(`${BASE}/#/skills`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  await page.fill('input[type=search], .st-input', 'zzzzz');
  await page.waitForTimeout(600);
  console.log('skills filtered  :', JSON.stringify(await read()));
  await page.click('.empty button');
  await page.waitForTimeout(600);
  console.log('cleared          :', await page.locator('.listrow').count(), 'skills back');

  await page.goto(`${BASE}/#/workspace`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  await page.fill('input[placeholder="Filter files"]', 'zzzzz');
  await page.waitForTimeout(600);
  console.log('workspace filtered:', JSON.stringify(await read()));
  await page.click('.empty button');
  await page.waitForTimeout(700);
  console.log('cleared          :', await page.locator('.treerow').count(), 'rows back');
  // the assertion that matters: the pane is not blank, and it offers a way back
  const after = await read();
  console.log('filtered pane is not blank:', after !== null || (await page.locator('.treerow').count()) > 0);
  await shot('250-empty-states');
}

if (flow === 'branch') {
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'say hello');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(6500);

  const view = () => page.evaluate(() => ({
    roles: [...document.querySelectorAll('.st-msg')].map((m) =>
      m.classList.contains('st-msg--user') ? 'user' : 'assistant'),
    url: location.hash.slice(0, 24),
  }));
  console.log('before    :', JSON.stringify(await view()));

  // Continue is only offered on the newest reply, so check it before branching
  // away from the state that has one.
  const offeredBefore = await page.evaluate(() =>
    [...document.querySelectorAll('.st-msg--assistant')].map((m) =>
      [...m.querySelectorAll('button')].some((b) => b.title === 'Continue from here')));
  console.log('continue on the newest reply only:', JSON.stringify(offeredBefore));
  await page.click('.st-msg--assistant .st-msg__actions button[title="Continue from here"]');
  await page.waitForTimeout(1200);
  console.log('after continue:', JSON.stringify((await view()).roles), '<- a new turn was added');
  await page.waitForTimeout(8000);

  // remember the original so "untouched" means unchanged by the branch, not a
  // hardcoded length
  const before = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('strata.sessions.v1') ?? '[]')
      .map((s) => ({ title: s.title, roles: s.messages.map((m) => m.role) })));
  const originalBefore = before.find((s) => !s.title.includes('branch'));

  // now branch, from the very first user turn
  await page.click('.st-msg--user >> nth=0');
  await page.hover('.st-msg--user >> nth=0');
  await page.waitForTimeout(300);
  await page.click('.st-msg--user >> nth=0 >> xpath=.. >> css=.st-msg__actions button[title="Branch from here"]').catch(async () => {
    await page.evaluate(() => {
      const u = document.querySelector('.st-msg--user');
      [...u.querySelectorAll('button')].find((b) => b.title === 'Branch from here').click();
    });
  });
  await page.waitForTimeout(1100);
  console.log('from user :', JSON.stringify(await view()), '<- branch must show only the user turn, and the url must follow');

  const sessions = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('strata.sessions.v1') ?? '[]')
      .map((s) => ({ title: s.title, roles: s.messages.map((m) => m.role) })));
  console.log('stored    :', JSON.stringify(sessions));
  const originalAfter = sessions.find((s) => !s.title.includes('branch'));
  console.log('original untouched by the branch:',
    JSON.stringify(originalAfter) === JSON.stringify(originalBefore));

  await shot('240-branch-continue');
}

if (flow === 'loaderror') {
  await page.goto(`${BASE}/#/models`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1600);
  console.log('at rest:', (await page.locator('.verror').count()) === 0);

  expectFailure(/Failed to load resource|ERR_FAILED/);
  await page.route('**/api/models/*/load', (r) => r.abort());
  /* Every installed variant on this machine is either already loaded or rated
     beyond it, so the load action is "Load anyway" and asks first. The flow
     accepts the confirmation and exercises the failure path behind it. */
  page.on('dialog', (d) => void d.accept());
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll('button')]
      .find((b) => /^load( anyway)?$/i.test(b.textContent.trim()));
    btn?.click();
  });
  await page.waitForTimeout(1400);
  console.log('failed load:', await page.evaluate(() => {
    const el = document.querySelector('.verror');
    return el ? el.textContent.replace(/\s+/g, ' ').trim().slice(0, 60) : 'NOT SHOWN';
  }));
  await page.waitForTimeout(7000);
  console.log('t=8.4s inline:', await page.locator('.verror').count(), '| toasts:', await page.locator('.st-toast').count());
  await page.unroute('**/api/models/*/load');
  await page.click('.verror button');
  await page.waitForTimeout(3000);
  console.log('retry when recovered:', (await page.locator('.verror').count()) === 0 ? 'cleared' : 'STILL SHOWING');
  await shot('230-load-error');
}

if (flow === 'actionerror') {
  await page.goto(`${BASE}/#/tuning`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1600);
  console.log('no error at rest:', (await page.locator('.verror').count()) === 0);

  // make the engine action fail — the abort is the point of this flow, so the
  // network error it produces is expected rather than a regression
  expectFailure(/Failed to load resource|ERR_FAILED/);
  await page.route('**/api/engine/restart', (r) => r.abort());
  await page.evaluate(() => {
    [...document.querySelectorAll('button')].find((b) => /restart engine/i.test(b.textContent))?.click();
  });
  await page.waitForTimeout(1200);
  console.log('t=1.2s  inline:', await page.locator('.verror').count(), '| toasts:', await page.locator('.st-toast').count());
  await page.waitForTimeout(7000);
  console.log('t=8.2s  inline:', await page.locator('.verror').count(), '| toasts:', await page.locator('.st-toast').count(),
    '<- toast expired, failure remains');

  // retry while still failing, then with the route restored
  await page.click('.verror button');
  await page.waitForTimeout(1400);
  console.log('retry while failing :', (await page.locator('.verror').count()) > 0 ? 'error returns' : 'SILENTLY CLEARED');
  await page.unroute('**/api/engine/restart');
  await page.click('.verror button');
  await page.waitForTimeout(2000);
  console.log('retry when recovered:', (await page.locator('.verror').count()) === 0 ? 'cleared' : 'STILL SHOWING');
  await shot('220-action-error');
}

if (flow === 'skilluse') {
  // a prompt that should trigger one skill and nothing else
  await page.goto(`${BASE}/#/chat`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'review this code and check my code for bugs');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await page.keyboard.press('Enter').catch(() => {});
  await page.waitForTimeout(4000);

  await page.goto(`${BASE}/#/skills`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  const rows = await page.evaluate(() =>
    [...document.querySelectorAll('.listrow')].map((r) => ({
      name: r.querySelector('.listrow__title')?.textContent,
      meta: [...r.querySelectorAll('.xs.muted')].map((e) => e.textContent).find((t) => /Used|Not used/.test(t)),
    })));
  for (const r of rows) console.log(`  ${String(r.name).padEnd(20)} ${r.meta}`);
  console.log('every skill states its usage:', rows.every((r) => r.meta));
  await shot('210-skills-usage');
}

if (flow === 'sessionmenu') {
  for (const msg of ['keep this one', 'delete this one']) {
    await page.click('text=New session');
    await page.waitForTimeout(350);
    await page.fill('textarea', msg);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(3200);
  }
  const count = () => page.locator('.navsession').count();
  console.log('sessions:', await count());

  await page.click('.navsession >> nth=0', { button: 'right' });
  await page.waitForTimeout(400);
  console.log('menu:', JSON.stringify(await page.evaluate(() =>
    [...document.querySelectorAll('.menu__item')].map((i) => i.textContent.trim()))));
  await page.click('.menu__item--danger');
  await page.waitForTimeout(400);
  await page.click('.st-btn--danger');
  await page.waitForTimeout(600);
  console.log('after delete:', await count());

  await page.evaluate(() => document.querySelector('.navsession').focus());
  await page.keyboard.press('Delete');
  await page.waitForTimeout(350);
  console.log('Delete key arms confirm:', (await page.locator('.st-btn--danger').count()) > 0);
  await page.keyboard.press('Escape');
  await shot('200-session-menu');
}

if (flow === 'rail') {
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'read the readme');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(6000);
  await page.click('button[title="Conversation context"]');
  await page.waitForTimeout(700);

  const state = () => page.evaluate(() => {
    const body = document.querySelector('.contextpanel__body');
    const secs = [...document.querySelectorAll('.railsection')];
    return {
      sections: secs.length,
      scrolls: body.scrollHeight > body.clientHeight + 2,
      firstRows: getComputedStyle(secs[0].querySelector('.railsection__body')).gridTemplateRows,
    };
  });
  console.log('all open    :', JSON.stringify(await state()));
  await page.click('.railsection__head >> nth=0');
  await page.waitForTimeout(120);
  console.log('mid-collapse:', JSON.stringify(await state()), '<- animating, not snapping');
  await page.waitForTimeout(450);
  console.log('collapsed   :', JSON.stringify(await state()));

  const tools = await page.evaluate(() => {
    const sec = [...document.querySelectorAll('.railsection')].find((x) => x.textContent.includes('Tools in reach'));
    return [...sec.querySelectorAll('.railsection__inner .row')].map((r) => ({
      name: r.querySelector('.small.strong')?.textContent,
      used: r.hasAttribute('data-used'),
    }));
  });
  console.log('tools used  :', JSON.stringify(tools.filter((t) => t.used).map((t) => t.name)));
  await shot('190-context-rail');
}

if (flow === 'skills') {
  await page.goto(`${BASE}/#/skills`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
  const rows = await page.locator('.listrow').count();
  console.log('skills listed:', rows);
  await page.click('text=Read instructions >> nth=0');
  await page.waitForTimeout(400);
  const detail = await page.evaluate(() => {
    const el = document.querySelector('.skilldetail__body');
    const tr = document.querySelector('.skilldetail__triggers');
    return { chars: el?.textContent.length ?? 0, triggers: tr ? tr.querySelectorAll('.st-pill').length : 0 };
  });
  console.log('instructions shown:', detail.chars, 'chars ·', detail.triggers, 'triggers');
  const all = await page.evaluate(async () => {
    const r = await fetch('/api/skills').then((x) => x.json());
    return r.filter((s) => !s.body).map((s) => s.name);
  });
  console.log('skills with no instruction text:', all.length ? all : 'none');
  await shot('180-skills-instructions');
}

if (flow === 'sidebar') {
  for (const msg of ['alpha conversation', 'beta conversation']) {
    await page.click('text=New session');
    await page.waitForTimeout(350);
    await page.fill('textarea', msg);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(3200);
  }
  const titles = () => page.evaluate(() =>
    [...document.querySelectorAll('.navsession__title')].map((e) => e.textContent));
  console.log('newest first :', JSON.stringify(await titles()));

  await page.evaluate(() => {
    const rows = [...document.querySelectorAll('.ws__session')];
    rows[rows.length - 1].querySelector('.ws__pin').click();
  });
  await page.waitForTimeout(450);
  console.log('after pinning:', JSON.stringify(await titles()));

  await page.evaluate(() => document.querySelector('.nav__collapse').focus());
  const walk = [];
  for (let i = 0; i < 4; i += 1) {
    await page.keyboard.press('ArrowDown');
    walk.push(await page.evaluate(() => (document.activeElement?.textContent ?? '').trim().slice(0, 20)));
  }
  console.log('arrow walk   :', JSON.stringify(walk));
  await page.keyboard.press('End');
  console.log('End -> ', await page.evaluate(() => (document.activeElement?.textContent ?? '').trim().slice(0, 20)));
  await shot('170-sidebar-keys');
}

if (flow === 'retryedit') {
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'read the readme');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);

  const counts = () => page.evaluate(() => ({
    user: document.querySelectorAll('.st-msg--user').length,
    assistant: document.querySelectorAll('.st-msg--assistant').length,
  }));
  console.log('after a turn  :', JSON.stringify(await counts()));

  await page.hover('.st-msg--assistant');
  await page.waitForTimeout(250);
  await page.click('button[title="Ask again"]');
  await page.waitForTimeout(600);
  console.log('mid-retry     :', JSON.stringify(await counts()), '<- reply dropped, not duplicated');
  await page.waitForTimeout(8000);
  await page.keyboard.press('Enter').catch(() => {});
  await page.waitForTimeout(6000);
  console.log('after retry   :', JSON.stringify(await counts()));

  await page.hover('.st-msg--user');
  await page.waitForTimeout(250);
  await page.click('button[title="Edit and ask again"]');
  await page.waitForTimeout(600);
  console.log('after edit    :', JSON.stringify(await counts()),
    '| composer:', JSON.stringify(await page.inputValue('textarea')),
    '| focus:', await page.evaluate(() => document.activeElement?.tagName));
  await shot('160-retry-edit');
}

if (flow === 'palette') {
  await page.keyboard.press('Meta+k');
  await page.waitForTimeout(500);

  // focus must land inside the surface, and stay there
  const inside = () => page.evaluate(() => {
    const m = [...document.querySelectorAll('.modal')].find((x) => x.dataset.open === 'true');
    return !!(m && document.activeElement && m.contains(document.activeElement));
  });
  console.log('focus inside on open:', await inside());
  for (let i = 0; i < 14; i += 1) await page.keyboard.press('Tab');
  console.log('focus inside after 14 Tabs:', await inside());

  const top = () => page.evaluate(() =>
    [...document.querySelectorAll('.palette__row')].slice(0, 2).map((r) =>
      (r.querySelector('.navitem__text')?.textContent ?? '') +
      '[' + [...r.querySelectorAll('mark')].map((m) => m.textContent).join('') + ']'));
  for (const q of ['mtr', 'tune', 'api', '>unload', '@']) {
    await page.fill('.modal input', q);
    await page.waitForTimeout(260);
    console.log(`  ${q.padEnd(9)} -> ${JSON.stringify(await top())}`);
  }
  await page.keyboard.press('Escape');
  await page.waitForTimeout(400);
  console.log('closed:', await page.locator('.modal[data-open="true"]').count() === 0);
  await shot('150-palette-fuzzy');
}

if (flow === 'telemetry') {
  await page.click('text=New session');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'read the readme and summarise it');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);

  const chips = await page.locator('.telemetry__chip').count();
  console.log('telemetry chips:', chips);
  const titles = ['Session statistics', 'Token usage', 'Context'];
  for (let i = 0; i < titles.length; i += 1) {
    await page.locator('.telemetry__chip').nth(i).click();
    await page.waitForTimeout(450);
    const head = await page.textContent('.telepop__head').catch(() => null);
    const rows = await page.locator('.telepop__row').count();
    console.log(`  ${titles[i]}: head="${head?.trim()}" rows=${rows}`);
    await shot(`14${i}-telepop-${titles[i].toLowerCase().replace(/ /g, '-')}`);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
    const gone = await page.locator('.telepop').count();
    if (gone !== 0) console.log(`  WARNING: ${titles[i]} did not close on Escape`);
  }
  await page.click('button[title="Switch to dark"]');
  await page.waitForTimeout(500);
  await page.locator('.telemetry__chip').nth(2).click();
  await page.waitForTimeout(450);
  await shot('143-telepop-dark');
  await page.keyboard.press('Escape');
}

if (flow === 'focus') {
  await page.click('text=New conversation');
  await page.waitForTimeout(400);
  await page.fill('textarea', 'Explain what a KV cache does, in plain words');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(7000);
  await shot('130-chat-normal');
  await page.keyboard.press('Meta+Shift+F');
  await page.waitForTimeout(900);
  await shot('131-focus-mode');
  // reading size should move only the conversation
  await page.keyboard.press('Escape');
  await page.waitForTimeout(700);
  const exited = await page.evaluate(() => !document.documentElement.dataset.focus);
  console.log('Esc left focus mode:', exited);
}

if (flow === 'overlays') {
  await page.goto(`${BASE}/#/launch`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  await page.keyboard.press('Meta+k');
  await page.waitForTimeout(700);
  await shot('120-palette');
  await page.type('.modal input', 'serv');
  await page.waitForTimeout(500);
  await shot('121-palette-filtered');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(400);
  await page.keyboard.press('Meta+Slash');
  await page.waitForTimeout(700);
  await shot('122-shortcuts');
}

if (flow === 'panel') {
  await page.click('text=New conversation');
  await page.waitForTimeout(500);
  await page.fill('textarea', 'Read the README and summarise it');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(5200);
  await page.click('button[title="Conversation context"]');
  await page.waitForTimeout(1000);
  await shot('100-context-panel');
}

if (flow === 'sampling') {
  await page.click('text=New conversation');
  await page.waitForTimeout(500);
  await page.click('button[title="Sampling — temperature and friends"]');
  await page.waitForTimeout(900);
  await shot('95-sampling-drawer');
  const slider = page.locator('.st-slider').first();
  await slider.focus();
  for (let i = 0; i < 6; i += 1) await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(900);
  await shot('96-sampling-changed');
}

if (flow === 'commands') {
  await page.click('text=New conversation');
  await page.waitForTimeout(400);
  await page.click('textarea');
  await page.type('textarea', '/');
  await page.waitForTimeout(600);
  await shot('80-slash-menu');
  await page.type('textarea', 'mon');
  await page.waitForTimeout(500);
  await shot('81-slash-filtered');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(300);
  await page.keyboard.press('Meta+Slash');
  await page.waitForTimeout(700);
  await shot('82-shortcuts');
}

if (flow.startsWith('tour')) {
  const prefix = process.argv[3] ?? 'tour';
  // tour-dark flips the theme first so both palettes are captured from one run.
  if (flow === 'tour-dark') {
    await page.click('button[title="Switch to dark"]');
    await page.waitForTimeout(600);
  }
  const routes = ['launch', 'chat', 'models', 'tuning', 'serving', 'monitor', 'workspace', 'skills', 'settings'];
  for (const r of routes) {
    await page.goto(`${BASE}/#/${r}`, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(1600);
    await shot(`${prefix}-${r}`);
  }
  await page.goto(`${BASE}/#/setup`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2000);
  await shot(`${prefix}-setup`);
}

if (flow === 'light') {
  /* Sets the theme rather than toggling to it: this flow used to click
     "Switch to light", which only exists while the app is dark. Light is the
     default now, so the control was correctly absent and the flow failed on a
     button that should not have been there. */
  await page.evaluate(() => {
    const raw = JSON.parse(localStorage.getItem('strata.prefs.v2') ?? '{}');
    localStorage.setItem('strata.prefs.v2', JSON.stringify({ ...raw, theme: 'light' }));
  });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  await page.goto(`${BASE}/#/models`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);
  await shot('50-models-light');
  await page.goto(`${BASE}/#/monitor`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1400);
  await shot('51-monitor-light');
}

await browser.close();
console.log(problems.length ? `problems:\n  ${problems.slice(0, 10).join('\n  ')}` : 'no console errors');
