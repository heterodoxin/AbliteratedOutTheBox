// The turn guard: a search-heavy /plan that keeps failing searches and rewriting its plan is stopped by the app and ends with a plain message;
// a context at 100% never ends silently. The pure counting always runs; the end-to-end ones need a built Harness (STRATA_HARNESS_DIR) and
// run the real search code on a pretend web that refuses (no network).
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TurnGuard, GUARD_LIMITS, GUARD_NOTES, guardNote } from '../server/agent/turn-guard.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

const search = (id, state, result = '') => ({ id, name: 'web_search', state, result });
const LIMITED = 'Web search is limited right now. Try again in a minute, or add a search key (Exa, Perplexity or DeepSeek) in Settings → Tools → Web search. Do not retry this search';

test('three failed searches in a row stop the turn; a search that worked starts the count again', () => {
  const g = new TurnGuard({ plan: true, contextMax: 32768 });
  assert.equal(g.card(search('a', 'running')), null, 'a running search is not counted yet');
  assert.equal(g.card(search('a', 'error', LIMITED)), null);
  assert.equal(g.card(search('a', 'error', LIMITED)), null, 'the same card twice is one search');
  assert.equal(g.card(search('b', 'done', 'three results')), null);
  assert.equal(g.card(search('c', 'error', LIMITED)), null);
  assert.equal(g.card(search('d', 'done', LIMITED)), null, 'a refusal that arrives as a normal result still counts');
  assert.equal(g.card(search('e', 'error', LIMITED)), 'searches-failed');
  assert.equal(g.card(search('f', 'error', LIMITED)), 'searches-failed', 'once stopped it stays stopped');
  assert.equal(g.searches, 5);
});

test('other tools are not searches; too many searches in one turn stop it even when they work', () => {
  const g = new TurnGuard({ plan: true });
  for (let i = 0; i < 50; i++) assert.equal(g.card({ id: `r${i}`, name: 'read', state: 'done' }), null);
  for (let i = 0; i < GUARD_LIMITS.searchesPlan - 1; i++) assert.equal(g.card(search(`s${i}`, 'done', 'ok')), null);
  assert.equal(g.card(search('last', 'done', 'ok')), 'searches');
  const t = new TurnGuard({ plan: false });
  for (let i = 0; i < GUARD_LIMITS.searchesTurn - 1; i++) assert.equal(t.card(search(`s${i}`, 'done', 'ok')), null);
  assert.equal(t.card(search('last', 'done', 'ok')), 'searches');
});

test('a plan turn that writes its numbered plan from 1. again and again is stopped; a single plan is not', () => {
  const g = new TurnGuard({ plan: true });
  const plan = '## Plan\n\n1. First\n2. Second\n3. Third\n\n';
  let reason = null;
  for (let i = 0; i < GUARD_LIMITS.planRewrites - 1; i++) reason ??= g.text(plan);
  assert.equal(reason, null, `${GUARD_LIMITS.planRewrites - 1} versions are fine`);
  assert.equal(g.text(plan), 'rewrites');
  const one = new TurnGuard({ plan: true });
  for (const piece of ['Some looking around.\n\n## Plan\n\n', '1. First\n', '2. Second\n', '   1. a sub point\n', '3. Third\n']) assert.equal(one.text(piece), null, piece);
  const ordinary = new TurnGuard({ plan: false });
  for (let i = 0; i < 10; i++) assert.equal(ordinary.text(plan), null, 'only a plan turn counts rewrites');
});

test('a nearly full context stops the turn; a normal one does not', () => {
  const g = new TurnGuard({ contextMax: 10000 });
  assert.equal(g.context(5000), null);
  assert.equal(g.context(9200), null);
  assert.equal(g.context(9300), 'context');
  assert.equal(new TurnGuard({ contextMax: 0 }).context(1e9), null, 'an unknown window is never guessed');
});

test('a turn that ends with no words always gets a plain message: out of room, or nothing written', () => {
  const g = new TurnGuard({ contextMax: 10000 });
  const done = { kind: 'completed' };
  assert.equal(g.silent({ ended: done, wrote: false, contextTokens: 9700 }), 'context');
  assert.equal(g.silent({ ended: { kind: 'max-tokens' }, wrote: false, contextTokens: 9000 }), 'context');
  assert.equal(g.silent({ ended: done, wrote: false, contextTokens: 3000 }), 'empty');
  assert.equal(g.silent({ ended: done, wrote: false, acted: true, contextTokens: 3000 }), null, 'cards of finished work are an answer of a kind');
  assert.equal(g.silent({ ended: done, wrote: false, acted: true, contextTokens: 9700 }), 'context', 'but a full context is never silent');
  assert.equal(g.silent({ ended: done, wrote: true, contextTokens: 9700 }), null, 'there are words: nothing to add');
  assert.equal(g.silent({ ended: { kind: 'error', error: { message: 'x' } }, wrote: false, contextTokens: 9700 }), null, 'an error says why by itself');
  assert.equal(g.silent({ ended: done, wrote: false, contextTokens: 9700, stoppedByPerson: true }), null, 'Stop is the person\'s own');
});

test('every message is a plain sentence with a next step, carries its interface key, and is in the catalog in all eight languages', () => {
  const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko'];
  const catalogs = Object.fromEntries(langs.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/${l}.json`, import.meta.url), 'utf8'))]));
  const idOf = text => Object.entries(catalogs.en).find(([, v]) => v === text)?.[0];
  for (const [reason, text] of Object.entries(GUARD_NOTES)) {
    const frame = guardNote(reason);
    assert.equal(frame.type, 'delta'); assert.equal(frame.key, text); assert.equal(frame.own, true);
    assert.ok(text.length < 220, reason);
    assert.match(text, /(Try again|Tell me|Say|Send|approve)/i, `${reason} gives a next step`);
    const id = idOf(text);
    assert.ok(id, `${reason} is in the English catalog`);
    for (const l of langs) assert.ok(catalogs[l][id]?.trim(), `${reason} in ${l}`);
  }
  assert.match(GUARD_NOTES.context, /ran out of room/);
});

// ------------------------------------------------------------------------------------------------ with the real Harness

async function rig(t, options = {}) {
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-guard-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const logs = [];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: options.context ?? 32768, slots: 2 }], connections: () => [],
    // these chats are in Manual and search without a person at the card: the Settings switch that skips the web question is on
    tools: () => ({ off: new Set(), search: { provider: 'builtin', key: '' }, servers: {}, webWithoutAsking: true }),
    env: process.env, startTimeoutMs: 90_000, searchMock: 'limited', log: l => logs.push(l) });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  return { worker, bridge, ws, logs, base: (chatId, onFrame) => ({ chatId, workspacePath: ws, mode: 'manual', onFrame }) };
}
const words = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');

test('a search-heavy /plan on a web that keeps refusing stops after 3 failed searches, proposes the plan it has, and says why', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.planCommand({ ...r.base('p1', f => out.push(f)), line: 'SCRIPT:searchloop' });
  const cards = out.filter(f => f.type === 'agent-tool' && f.tool.name === 'web_search' && ['done', 'error'].includes(f.tool.state));
  assert.ok(cards.length >= 3 && cards.length <= 4, `searches stopped at the limit (${cards.length})`);
  assert.ok(cards.some(f => /Web search is limited right now/.test(f.tool.result ?? '')), 'the tool said web search is limited');
  assert.equal(out.filter(f => f.type === 'error').length, 0, 'not an error: a normal end');
  const done = out.find(f => f.type === 'done');
  assert.ok(done && !done.cancelled, 'the turn ended normally');
  const plan = out.find(f => f.type === 'plan');
  assert.equal(plan?.event, 'proposed', 'the plan it had is offered');
  assert.equal(plan.plan.steps.length, 3);
  assert.equal(plan.plan.caution, GUARD_NOTES['searches-failed'], 'and the card says why it stopped early');
  const requests = r.worker.requests.filter(q => q.directive === 'searchloop');
  assert.ok(requests.length <= 5, `the model was not asked again and again (${requests.length} requests)`);
  assert.ok(r.logs.some(l => /turn guard: searches-failed/.test(l)));
});

test('an ordinary turn whose searches keep failing ends with the plain sentence as its last words', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.runTurn({ ...r.base('c1', f => out.push(f)), message: 'SCRIPT:searchloop' });
  assert.equal(out.filter(f => f.type === 'error').length, 0);
  assert.ok(out.find(f => f.type === 'done'));
  const note = out.filter(f => f.type === 'delta' && f.key).at(-1);
  assert.equal(note?.key, GUARD_NOTES['searches-failed']);
  assert.match(words(out), /Settings → Tools → Web search/);
  assert.ok(out.findIndex(f => f.key) < out.findIndex(f => f.type === 'done'), 'the sentence comes before the end of the turn');
});

test('a context at 100% with no answer is never silent: the plain message comes, and the older messages are summarised', { skip }, async (t) => {
  const r = await rig(t, { context: 32768 });
  const out = [];
  await r.bridge.runTurn({ ...r.base('f1', f => out.push(f)), message: 'SCRIPT:bloat 32000' });
  assert.equal(out.filter(f => f.type === 'error').length, 0);
  assert.ok(out.find(f => f.type === 'done'));
  assert.equal(out.filter(f => f.type === 'delta' && f.key).at(-1)?.key, GUARD_NOTES.context);
  assert.match(words(out), /ran out of room/);
  await new Promise(res => setTimeout(res, 1500));
  const compacted = r.logs.find(l => /compact after a stopped turn/.test(l));
  assert.match(compacted ?? '', /compact after a stopped turn: \w+/, `compaction was asked for and the Harness answered (${compacted})`);
  const out2 = [];
  await r.bridge.runTurn({ ...r.base('f2', f => out2.push(f)), message: 'SCRIPT:bloat 3000' });
  assert.equal(out2.filter(f => f.type === 'delta' && f.key).at(-1)?.key, GUARD_NOTES.empty, 'no words at all, plenty of room: still a message');
});
