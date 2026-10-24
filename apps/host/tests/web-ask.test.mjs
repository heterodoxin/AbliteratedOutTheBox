/**
 * Manual mode asks once per chat before the first web search or web page. Approve holds for the rest of that chat, Deny gives the model a
 * readable refusal, Plan and Bypass are unchanged, and a switch in Settings -> Tools (off by default) skips the question.
 *
 * The gate is plain ESM, so most of this drives its plugin with a pretend Harness context. The bridge-level tests at the end use the real
 * Harness with a scripted model and skip without STRATA_HARNESS_DIR.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { build } from 'esbuild';
import { apply, manualDecision, planDecision, READ_ONLY_TOOLS, MANUAL_PRESET, PLAN_PRESET, WEB_REFUSAL } from '../server/agent/plugins/strata-gate.mjs';
import { buildProfile } from '../server/agent/profile.mjs';
import { createAgentTools } from '../server/agent/tools-settings.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { createAgentTaskRunner } from '../server/agent/task-run.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

const SEARCH = { name: 'web_search', arguments: { queries: ['heat pump prices'] } };
const FETCH = { name: 'web_fetch', arguments: { url: 'https://example.com/a' } };

// ---------------------------------------------------------------------------------------------- the question

test('the question says what leaves this computer and where it goes', () => {
  const ask = manualDecision(SEARCH, { provider: 'DuckDuckGo' });
  assert.equal(ask.kind, 'ask');
  assert.equal(ask.reason, 'The assistant wants to search the web for: heat pump prices. Search words go to DuckDuckGo.');
  assert.equal(manualDecision({ name: 'web_search', arguments: { queries: ['a', 'b'] } }, { provider: 'Exa' }).reason, 'The assistant wants to search the web for: a; b. Search words go to Exa.');
  assert.equal(manualDecision(FETCH).reason, "The assistant wants to open the web page https://example.com/a. The page's site sees the request.");
  assert.ok(!READ_ONLY_TOOLS.has('web_search') && !READ_ONLY_TOOLS.has('web_fetch'), 'the web is no longer "look only"');
  // a very long query is cut, not pasted whole into the card
  assert.ok(manualDecision({ name: 'web_search', arguments: { queries: ['x'.repeat(900)] } }).reason.length < 300);
});

test('Plan keeps its rules: the web needs no question there', () => {
  assert.equal(planDecision(SEARCH), null);
  assert.equal(planDecision(FETCH), null);
});

// ---------------------------------------------------------------------------------------------- the gate

/** A pretend Harness: enough of the context for the gate. `answers` is what the person says to each question, in order. */
function rig({ preset = MANUAL_PRESET, answers = ['allowed-once'], config = {} } = {}) {
  const handlers = [];
  const sessions = new Map();
  const asked = [];
  let policy = new Map();
  const ctx = {
    sessions: { get: id => sessions.get(id) },
    permissionPresets: { current: session => session.preset ?? preset },
    approval: {
      overrideOf: session => policy.get(session.id) ?? null,
      request: async req => { asked.push(req.reason); await new Promise(r => setTimeout(r, 5)); return answers[Math.min(asked.length - 1, answers.length - 1)]; },
    },
    on: (event, handler) => { if (event === 'tools/pre-execute') handlers.push(handler); },
    inject: () => {},
  };
  apply(ctx, config);
  const session = (id, extra = {}) => { const s = { id, header: {}, append: (type, data) => { if (type === 'approval/policy') policy.set(id, data.policy); }, ...extra }; sessions.set(id, s); return s; };
  const call = async (sessionObj, exec) => {
    let n = 0;
    const result = await handlers[0]({ ...exec, callId: `c${++n}${Math.random()}`, agent: { session: sessionObj } }, async () => ({ kind: 'allow' }));
    return result;
  };
  return { ctx, session, call, asked };
}

test('Manual: the first search asks, Approve lets the rest of that chat search and open pages without asking again', async () => {
  const r = rig({ config: { web: { provider: 'DuckDuckGo' } } });
  const chat = r.session('chat-1');
  const first = await r.call(chat, SEARCH);
  assert.deepEqual(r.asked, ['The assistant wants to search the web for: heat pump prices. Search words go to DuckDuckGo.']);
  assert.equal(first.kind, 'allow', 'approved, so the search runs');
  assert.equal((await r.call(chat, { name: 'web_search', arguments: { queries: ['another'] } })).kind, 'allow');
  assert.equal((await r.call(chat, FETCH)).kind, 'allow');
  assert.equal(r.asked.length, 1, 'one question for the chat');
  // another chat asks for itself
  await r.call(r.session('chat-2'), SEARCH);
  assert.equal(r.asked.length, 2);
});

test('Manual: Deny gives the model a readable refusal, and the next try asks again', async () => {
  const r = rig({ answers: ['rejected', 'allowed-once'] });
  const chat = r.session('chat-1');
  const refused = await r.call(chat, SEARCH);
  assert.equal(refused.kind, 'deny');
  assert.equal(refused.reason, WEB_REFUSAL);
  assert.match(refused.reason, /did not allow web access/);
  assert.match(refused.reason, /Go on without it and say so/);
  assert.doesNotMatch(refused.reason, /tool "|web_search/, 'no tool names in what the model reads');
  assert.equal((await r.call(chat, SEARCH)).kind, 'allow', 'asked again, and this time yes');
  assert.equal(r.asked.length, 2);
  // an unavailable approval channel is a refusal too, not a silent allow
  const none = rig({ answers: ['unavailable'] });
  assert.equal((await none.call(none.session('c'), SEARCH)).reason, WEB_REFUSAL);
});

test('two searches started together share one question', async () => {
  const r = rig();
  const chat = r.session('chat-1');
  const [a, b] = await Promise.all([r.call(chat, SEARCH), r.call(chat, { name: 'web_search', arguments: { queries: ['second'] } })]);
  assert.equal(r.asked.length, 1);
  assert.equal(a.kind, 'allow'); assert.equal(b.kind, 'allow');
});

test('a helper agent works under the chat that started it: one yes covers both, and its question can reach the person', async () => {
  const r = rig();
  const chat = r.session('chat-1');
  const helper = r.session('helper-1', { header: { origin: 'subagent', parentSession: 'chat-1' } });
  assert.equal((await r.call(helper, SEARCH)).kind, 'allow');
  assert.equal(r.ctx.approval.overrideOf(helper), 'ask', 'the helper is set to ask so the card reaches the person');
  assert.equal((await r.call(chat, SEARCH)).kind, 'allow');
  assert.equal(r.asked.length, 1);
});

test('Bypass never asks, Plan is unchanged, and the Settings switch skips the question in Manual', async () => {
  const bypass = rig({ preset: 'strata-bypass' });
  assert.equal((await bypass.call(bypass.session('c'), SEARCH)).kind, 'allow');
  assert.equal(bypass.asked.length, 0);
  const plan = rig({ preset: PLAN_PRESET });
  assert.equal((await plan.call(plan.session('c'), SEARCH)).kind, 'allow');
  assert.equal(plan.asked.length, 0);
  const switchedOn = rig({ config: { web: { provider: 'DuckDuckGo', skipAsk: true } } });
  assert.equal((await switchedOn.call(switchedOn.session('c'), SEARCH)).kind, 'allow');
  assert.equal(switchedOn.asked.length, 0);
  // other things still ask with the switch on
  const write = await switchedOn.call(switchedOn.session('c2'), { name: 'write', arguments: { file_path: 'a.txt' } });
  assert.equal(write.kind, 'ask');
});

// ---------------------------------------------------------------------------------------------- the setting and the profile

test('Settings -> Tools: "search the web without asking" is off at first, remembered, and reaches the gate', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-web-ask-'));
  try {
    let saved = {};
    const tools = createAgentTools({ read: () => saved, write: v => { saved = v; }, dataDir: dir });
    assert.equal(tools.view().webWithoutAsking, false);
    assert.equal(tools.profileView().webWithoutAsking, false);
    assert.equal(tools.update({ webWithoutAsking: true }).webWithoutAsking, true);
    assert.equal(saved.webWithoutAsking, true, 'saved in the settings file');
    assert.equal(tools.profileView().webWithoutAsking, true);
    assert.equal(tools.update({ search: { provider: 'exa' } }).webWithoutAsking, true, 'other changes keep it');
    assert.equal(tools.update({ webWithoutAsking: false }).webWithoutAsking, false);
    assert.equal(tools.update({ webWithoutAsking: 'yes' }).webWithoutAsking, false, 'only a real on/off counts');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('the profile names the search service in the gate\'s question, and carries the switch', () => {
  const gateOf = tools => JSON.parse(buildProfile({ local: [], tools }).patch.split('\n').slice(1).join('\n')).flatMap(e => e.insert ?? []).find(e => e.id === 'strata-gate').config.web;
  const base = { off: new Set(), search: { provider: 'builtin', key: '' }, servers: {}, webWithoutAsking: false };
  assert.deepEqual(gateOf(base), { provider: 'DuckDuckGo, or Bing when DuckDuckGo is busy', skipAsk: false });
  assert.deepEqual(gateOf({ ...base, search: { provider: 'exa', key: 'k' }, webWithoutAsking: true }), { provider: 'Exa', skipAsk: true });
  assert.equal(gateOf({ ...base, search: { provider: 'perplexity', key: '' } }).provider, 'DuckDuckGo, or Bing when DuckDuckGo is busy', 'a service without its key is not used, so it is not named');
});

// ---------------------------------------------------------------------------------------------- the words, in every language

test('the card text and the Settings switch exist in all eight languages', async () => {
  const compiled = await build({ stdin: { contents: "export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, platform: 'node', format: 'esm' });
  const { tr, setTranslationLanguage } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
  const search = manualDecision(SEARCH, { provider: 'DuckDuckGo, or Bing when DuckDuckGo is busy' }).reason;
  const open = manualDecision(FETCH).reason;
  const english = [search, open, 'Let the assistant search the web without asking'];
  for (const language of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko']) {
    setTranslationLanguage(language);
    for (const text of english) assert.notEqual(tr(text), text, `${language}: ${text}`);
    const translated = tr(search);
    assert.ok(translated.includes('heat pump prices'), `${language}: the person's own words stay as typed`);
    assert.ok(!translated.includes('Bing when DuckDuckGo is busy'), `${language}: the service's description is translated too`);
    assert.ok(!/\{\d\}/.test(translated), language);
  }
  setTranslationLanguage('en');
  assert.equal(tr(search), search);
});

// ---------------------------------------------------------------------------------------------- with the real Harness

async function bridgeRig(t) {
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-webask-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  let n = 0;   // one data folder per bridge: two bridges on one folder share a pid file and end each other's Harness
  const make = (tools = () => undefined) => createAgentBridge({ dataDir: join(dir, 'bridge-' + (++n)), localInstances: () => local, connections: () => [], tools, env: process.env, startTimeoutMs: 90_000, searchMock: 'limited' });
  const bridges = [];
  t.after(async () => { for (const b of bridges) await b.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const open = tools => { const b = make(tools); bridges.push(b); return b; };
  const run = (bridge, chatId, message, { mode = 'manual', answer } = {}) => {
    const frames = [];
    return bridge.runTurn({
      chatId, message, workspacePath: ws, mode, onFrame: f => {
        frames.push(f);
        if (answer && f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId, callId: f.tool.id, decision: answer }).catch(() => {});
      },
    }).then(() => frames);
  };
  return { open, run, ws, local };
}
const searchCards = frames => frames.filter(f => f.type === 'agent-tool' && f.tool.name === 'web_search');
const stateList = frames => searchCards(frames).map(f => f.tool.state);
const CALL = 'SCRIPT:call web_search {"queries":["heat pump prices"]}';

test('real Harness, Manual: the first search shows the Approve/Deny card with the plain question, and Approve lets it run', { skip }, async t => {
  const r = await bridgeRig(t);
  const bridge = r.open();
  const asked = [];
  const frames = [];
  await bridge.runTurn({
    chatId: 'w1', message: CALL, workspacePath: r.ws, mode: 'manual',
    onFrame: f => {
      frames.push(f);
      if (f.type === 'agent-tool' && f.tool.state === 'proposed') { asked.push(f.tool.effect); void bridge.approve({ chatId: 'w1', callId: f.tool.id, decision: 'approve' }).catch(() => {}); }
    },
  });
  assert.equal(asked.length, 1);
  assert.match(asked[0], /^The assistant wants to search the web for: heat pump prices\. Search words go to DuckDuckGo, or Bing when DuckDuckGo is busy\.$/);
  const last = searchCards(frames).at(-1).tool;
  assert.equal(last.approved, true);
  assert.ok(['done', 'error'].includes(last.state), `the search ran (${last.state}); the pretend web is rate limited`);
  // second search in the same chat: no card
  const again = await r.run(bridge, 'w1', CALL, { answer: 'deny' });
  assert.ok(!stateList(again).includes('proposed'), 'no second question in the same chat');
  assert.ok(!stateList(again).includes('denied'));
});

test('real Harness, Manual: Deny gives a readable refusal and the search does not run', { skip }, async t => {
  const r = await bridgeRig(t);
  const bridge = r.open();
  const frames = await r.run(bridge, 'w2', CALL, { answer: 'deny' });
  const states = stateList(frames);
  assert.ok(states.includes('proposed'), 'it asked');
  const card = searchCards(frames).at(-1).tool;
  assert.equal(card.state, 'denied');
  assert.doesNotMatch(String(card.result ?? ''), /rate limit|limited right now/, 'nothing was searched');
  assert.equal(frames.at(-1).type, 'done');
});

test('real Harness: Bypass and Plan do not ask, and the Settings switch skips the question in Manual', { skip }, async t => {
  const r = await bridgeRig(t);
  const bridge = r.open();
  for (const mode of ['bypass', 'plan']) {
    const frames = await r.run(bridge, `w3-${mode}`, CALL, { mode, answer: 'deny' });
    assert.ok(!stateList(frames).includes('proposed'), `${mode} did not ask`);
  }
  const tools = { off: new Set(), search: { provider: 'builtin', key: '' }, servers: {}, webWithoutAsking: true };
  const switched = r.open(() => tools);
  const frames = await r.run(switched, 'w4', CALL, { answer: 'deny' });
  assert.ok(!stateList(frames).includes('proposed'), 'the switch is on, so no question');
  assert.ok(searchCards(frames).length > 0);
});

test('real Harness: a scheduled task in Manual is answered "no" at once, so it cannot search unless the switch is on', { skip }, async t => {
  const r = await bridgeRig(t);
  const task = { id: 't1', prompt: CALL, mode: 'manual' };
  const bridge = r.open();
  const off = await createAgentTaskRunner({ agent: bridge, choice: () => ({ instanceId: 'i' }), workspace: r.ws })(task);
  assert.ok(off, 'the task ran');
  const offCards = JSON.stringify(off);
  assert.doesNotMatch(offCards, /rate limit|limited right now/, 'no search happened');
  assert.match(offCards, /did not allow web access/, 'the model was told');
  assert.match(offCards, /needed a yes, which nobody could give/, 'the report says why');
  const tools = { off: new Set(), search: { provider: 'builtin', key: '' }, servers: {}, webWithoutAsking: true };
  const on = await createAgentTaskRunner({ agent: r.open(() => tools), choice: () => ({ instanceId: 'i' }), workspace: r.ws })({ ...task, id: 't2' });
  assert.ok(on);
  assert.doesNotMatch(JSON.stringify(on), /did not allow web access|nobody could give/, 'with the switch on the search was not refused');
});
