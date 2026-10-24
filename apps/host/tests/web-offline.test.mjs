/** With no internet the assistant gets plain words from the web tools, not "fetch failed", and can go on without them. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createDuckDuckGoSearch, UNREACHABLE_MESSAGE } from '../server/web-search-ddg.mjs';
import { plainToolError, withPlain } from '../server/agent/events.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('search with no internet: one readable sentence that tells the model to go on without it', async () => {
  let t = 5_000_000;   // a pretend clock that moves when the search waits, as in web-search.test.mjs
  const offline = createDuckDuckGoSearch({ fetch: async () => { throw new TypeError('fetch failed'); }, now: () => t, sleep: async ms => { t += ms; }, random: () => 0.5 });
  const error = await offline.search({ query: 'heat pumps' }).catch(e => e);
  assert.equal(error.message, UNREACHABLE_MESSAGE);
  assert.match(error.message, /could not reach the internet/);
  assert.match(error.message, /go on with what you know/);
  assert.doesNotMatch(error.message, /fetch failed|ENOTFOUND|TypeError/);
});

test('opening a page with no internet: the card says so in plain words and keeps the technical text behind Details', () => {
  for (const raw of ['Error: web fetch failed: fetch failed', 'Error: getaddrinfo ENOTFOUND example.com', 'Error: connect ECONNREFUSED 93.184.216.34:443']) {
    const text = withPlain(raw);
    assert.match(text, /^That web page could not be opened\. Check the address and your internet connection\./, raw);
    assert.match(text, /Details: /);
  }
  assert.equal(plainToolError(UNREACHABLE_MESSAGE), null, 'the search sentence is already plain and is left as it is');
});

test('real Harness: the model gets the plain sentence when the search cannot reach the internet', { skip }, async t => {
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-weboff-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768 }], connections: () => [], env: process.env, startTimeoutMs: 90_000, searchMock: 'offline' });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const frames = [];
  await bridge.runTurn({ chatId: 'off1', message: 'SCRIPT:call web_search {"queries":["heat pumps"]}', workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
  const card = frames.filter(f => f.type === 'agent-tool' && f.tool.name === 'web_search').at(-1).tool;
  assert.equal(card.state, 'error');
  assert.match(card.result, /could not reach the internet/);
  assert.doesNotMatch(card.result, /fetch failed/);
  assert.equal(frames.at(-1).type, 'done', 'the turn ends normally');
});
