// Shared helpers for driving a throwaway host with Playwright (the app page in a real browser). STRATA_UI_BASE = the host's address, CHROME_PATH = the browser.
export { chromium } from 'playwright-core';
import { chromium as browserType } from 'playwright-core';
export const BASE = process.env.STRATA_UI_BASE ?? 'http://127.0.0.1:18791';
export const CHROME = process.env.CHROME_PATH ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';
export async function api(path, method = 'GET', body) {
  const r = await fetch(BASE + path, { method, headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: r.status, body: await r.json().catch(() => null) };
}
export async function ensureMock(name = 'Mock model', models = [{ id: 'mock-model', name: 'Mock model' }], baseUrl = 'http://127.0.0.1:18899/v1') {
  const list = (await api('/api/model-providers')).body;
  const have = list.connections.find(c => c.name === name);
  const res = await api('/api/model-providers', 'POST', { ...(have ? { id: have.id } : {}), kind: 'custom', provider: null, name, baseUrl, protocol: 'openai-chat', apiKey: '', models });
  return res.body.connections.find(c => c.name === name);
}
export async function open({ theme = 'light', width = 1400, height = 900 } = {}) {
  const browser = await browserType.launch({ executablePath: CHROME, headless: true });
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1, colorScheme: theme, permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await context.newPage();
  await page.goto(BASE + '/');
  await page.waitForTimeout(1500);
  const skip = page.getByText('Go straight to the app');
  await skip.waitFor({ timeout: 10000 }).catch(() => {});
  if (await skip.count()) { await skip.click(); await page.waitForTimeout(800); }
  return { browser, context, page };
}
