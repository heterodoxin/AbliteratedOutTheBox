/**
 * Settings -> Tools screenshots from the real UI. The host's /api/agent/tools answer is replaced by a fixture (page.route) so every state can be
 * shown without a model or a Harness.
 *
 *   STRATA_URL=http://127.0.0.1:8899 SHOTS=./tools-shots node scripts/tools-shots.mjs
 */
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

const BASE = process.env.STRATA_URL;
if (!BASE) { console.error('STRATA_URL is required'); process.exit(2); }
const SHOTS = resolve(process.env.SHOTS ?? 'tools-shots');
mkdirSync(SHOTS, { recursive: true });
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);

const group = (id, name, about, area, asks = false, enabled = true) => ({ id, name, about, area, asks, enabled });
const GROUPS = [
  group('shell', 'Run commands', 'Lets the assistant run commands, for example to open a folder or start a program.', 'computer', true),
  group('files', 'Read and change files', 'Lets the assistant read, create and edit files in your chat folder.', 'computer', true),
  group('find', 'Find files and text', 'Lets the assistant look for files by name and search inside them. It only reads.', 'computer'),
  group('jobs', 'Long-running jobs', 'Lets the assistant start a slow command in the background and check on it later.', 'computer', true),
  group('fetch', 'Open web pages', 'Lets the assistant read a web page you point it to.', 'web'),
  group('search', 'Search the web', 'Lets the assistant search the web when it needs fresh information.', 'web'),
  group('helpers', 'Helper assistants', 'Lets the assistant start helpers that work on parts of a task at the same time.', 'together'),
  group('goals', 'Goals', 'Lets you give a goal with /goal. The assistant keeps working until it is done.', 'together'),
  group('todo', 'To-do list', 'Lets the assistant keep a checklist while it works on a big task.', 'together'),
  group('skills', 'Skills', 'Lets the assistant use your saved skills when they fit the task.', 'together'),
  group('present', 'Show results', 'Lets the assistant show a page or file in the preview next to the chat.', 'together'),
];
const base = () => ({ groups: GROUPS.map(g => ({ ...g })), search: { provider: 'builtin', keySet: false }, browser: { enabled: false, found: true, name: 'Google Chrome' }, custom: [] });
const STATES = {
  fresh: base(),
  busy: (() => { const s = base(); s.groups[3].enabled = false; s.groups[5].enabled = false; s.browser.enabled = true;
    s.custom = [
      { id: 'c1', name: 'Release notes', about: 'Write release notes in our house style.', kind: 'instructions', enabled: true, summary: 'Start with a one-line summary…' },
      { id: 'c2', name: 'Check website', about: 'Check that our website answers.', kind: 'command', enabled: true, summary: 'curl -I https://example.com', command: 'curl -I https://example.com' },
      { id: 'c3', name: 'Issue tracker', about: 'Look up and update issues.', kind: 'server', enabled: false, summary: 'https://issues.example/mcp', address: 'https://issues.example/mcp' },
    ]; return s; })(),
  key: (() => { const s = base(); s.search = { provider: 'exa', keySet: false }; return s; })(),
  nobrowser: (() => { const s = base(); s.browser = { enabled: false, found: false, name: null }; return s; })(),
};

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
let page = null, context = null;
let current = 'fresh';
const shot = async name => { await page.waitForTimeout(500); await page.screenshot({ path: join(SHOTS, `${name}.png`) }); console.log(`shot ${name}.png`); };
const open = async state => {
  current = state;
  if (!page) {
    context = await browser.newContext({ viewport: { width: 1360, height: 980 } });
    page = await context.newPage();
    page.on('pageerror', e => console.log('pageerror', e.message.slice(0, 300)));
    page.on('console', m => { if (m.type() === 'error') console.log('console', m.text().slice(0, 300)); });
    await page.route('**/api/agent/tools**', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(STATES[current]) }));
    await page.goto(`${BASE}/`);
    const skip = page.getByText('Go straight to the app');
    await skip.waitFor({ timeout: 8000 }).then(() => skip.click()).catch(() => {});
    await page.goto(`${BASE}/#/settings`);
    await page.waitForSelector('#settings-tab-Tools', { timeout: 20000 });
    await page.locator('#settings-tab-Tools').evaluate(el => el.click());
  } else {
    // The Tools panel loads its list when it is shown: leave it and come back to load the next fixture.
    await page.locator('#settings-tab-Appearance').evaluate(el => el.click());
    await page.waitForTimeout(300);
  }
  await page.locator('#settings-tab-Tools').evaluate(el => el.click());
  await page.waitForSelector('.settings-tools', { timeout: 10000 });
};
await open('fresh'); await shot('01-tools-top');
await page.getByRole('heading', { name: 'Web search' }).evaluate(el => el.scrollIntoView({ block: 'start' })); await shot('02-tools-web-and-browser');
await open('busy'); await page.getByRole('heading', { name: 'Your own tools' }).evaluate(el => el.scrollIntoView({ block: 'start' })); await shot('03-your-own-tools');
await open('key'); await page.getByRole('heading', { name: 'Web search' }).evaluate(el => el.scrollIntoView({ block: 'start' })); await shot('04-search-with-key');
await open('nobrowser'); await page.getByRole('heading', { name: 'Web browser' }).evaluate(el => el.scrollIntoView({ block: 'start' })); await shot('05-no-browser-found');
await open('fresh');
await page.getByRole('button', { name: 'Add a tool' }).click(); await shot('06-add-tool-instructions');
await page.getByRole('dialog').getByLabel('How it works').click();
await page.getByRole('option', { name: 'Run a command' }).click(); await shot('07-add-tool-command');
await page.getByRole('dialog').getByLabel('How it works').click();
await page.getByRole('option', { name: 'Connect a tool server (MCP)' }).click(); await shot('08-add-tool-server');
await browser.close();
