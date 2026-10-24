/** Pure parts of the agent bridge: profile generation, environment scrubbing, event projection. No Harness needed. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildProfile, outputBudget, routeFor, ASSUMED_API_CONTEXT } from '../server/agent/profile.mjs';
import { agentEnvironment, redact } from '../server/agent/harness-runtime.mjs';
import { TurnProjector, presentTool, toolScope, toolVerb, friendlyFailure } from '../server/agent/events.mjs';
import { withEarlierConversation } from '../server/agent/agent-bridge.mjs';

const local = [{ id: 'inst-1', label: 'Qwen', baseUrl: 'http://127.0.0.1:8095/v1', modelId: 'qwen', context: 32768, key: 'local-secret-key-1' }];
const conn = { id: 'c0nn-1234-abcd', name: 'Cloud', baseUrl: 'https://api.example.com/v1', protocol: 'openai-chat', apiKey: 'sk-live-ABCDEFGH', models: [{ id: 'm1', name: 'M1', contextMax: 64000 }, { id: 'm2', name: 'M2', contextMax: null }] };

test('the generated profile names credentials, never carries them, and declares only what exists', () => {
  const p = buildProfile({ local, connections: [conn] });
  assert.ok(!p.patch.includes('sk-live-ABCDEFGH') && !p.patch.includes('local-secret-key-1'));
  assert.match(p.patch, /apiKeyEnv/);
  assert.deepEqual(Object.values(p.env).sort(), ['local-secret-key-1', 'sk-live-ABCDEFGH']);
  const entries = JSON.parse(p.patch.replace(/^#.*\n/, ''));
  const providers = entries.find(e => e.id === 'llm-pi-ai').config.providers;
  assert.deepEqual(Object.keys(providers).sort(), ['api-c0nn-1234-ab', 'strata-local']);
  assert.equal(providers['strata-local'].models[0].contextWindow, 32768);
  assert.equal(providers['strata-local'].models[0].maxTokens, 9830, 'a local model gets 30 percent of its window for a reply (it was 8192, which cut big answers off) and the rest stays for the conversation, so the Harness can compact it');
  assert.equal(providers['strata-local'].models[0].reasoning, true);
  assert.equal(providers['strata-local'].compat.thinkingFormat, 'chat-template');
  assert.equal(providers['api-c0nn-1234-ab'].models[1].contextWindow, ASSUMED_API_CONTEXT, 'an unknown API window is a stated assumption, not a catalog default');
  assert.equal(entries.find(e => e.id === 'agent-default-model').config.provider, 'strata-local', 'local stays the default');
  assert.equal(entries.find(e => e.id === 'session-title-llm').disabled, true);
});

test('no local model means no default model, so nothing can quietly fall to a cloud route', () => {
  const p = buildProfile({ local: [], connections: [conn] });
  assert.equal(p.defaultRoute, null);
  assert.ok(!JSON.parse(p.patch.replace(/^#.*\n/, '')).some(e => e.id === 'agent-default-model'));
  assert.equal(routeFor(p, null), null);
});

test('a route fingerprint changes when its key or address changes, and only then', () => {
  const a = buildProfile({ local, connections: [conn] });
  const b = buildProfile({ local, connections: [{ ...conn, apiKey: 'sk-live-OTHER-KEY' }] });
  const c = buildProfile({ local: [{ ...local[0], context: 16384 }], connections: [conn] });
  const fp = (p, kind) => p.routes.find(r => r.kind === kind).fp;
  assert.notEqual(fp(a, 'api'), fp(b, 'api'));
  assert.equal(fp(a, 'local'), fp(b, 'local'));
  assert.notEqual(fp(a, 'local'), fp(c, 'local'));
  assert.equal(fp(a, 'api'), fp(c, 'api'));
});

test('routeFor picks the chat\'s own choice', () => {
  const p = buildProfile({ local, connections: [conn] });
  assert.equal(routeFor(p, { connectionId: conn.id, modelId: 'm2' }).modelId, 'm2');
  assert.equal(routeFor(p, { instanceId: 'inst-1' }).route, 'strata-local');
  assert.equal(routeFor(p, { connectionId: 'gone', modelId: 'm1' }), null);
  assert.equal(outputBudget(2048), 1536, 'a small window keeps a quarter for the prompt');
  assert.equal(outputBudget(1_000_000), 32768, 'a big window with no known maximum gets a generous 32K, not 8K');
  assert.equal(outputBudget(128_000), 32768); assert.equal(outputBudget(null), 32768);
  assert.equal(outputBudget(16_384), 16_384 - 4096, 'never more than the window minus room for the prompt');
  assert.equal(outputBudget(200_000, 64_000), 64_000, 'the connection own maximum is used when known');
  assert.equal(outputBudget(8192, 64_000), 8192 - 2048, 'a known maximum is still bounded by the window');
  assert.equal(outputBudget(200_000, 4096), 4096, 'a small known maximum is respected');
});

test('the agent environment keeps what a shell needs and drops ambient credentials', () => {
  const env = agentEnvironment({ PATH: '/bin', HOME: '/h', OPENAI_API_KEY: 'x', GITHUB_TOKEN: 'y', AWS_SECRET_ACCESS_KEY: 'z', SSH_AUTH_SOCK: 's', SystemRoot: 'C:\\Windows' }, { STRATA_API_KEY_X: 'k' });
  assert.deepEqual(Object.keys(env).sort(), ['DSH_PERMISSION_MODE', 'DSH_TELEMETRY_DISABLED', 'HOME', 'NODE_USE_SYSTEM_CA', 'PATH', 'STRATA_API_KEY_X', 'SystemRoot']);
});

test('redact removes secret values and launch tokens from log text', () => {
  const out = redact('GET /?token=abcDEF_123-x failed with key sk-live-ABCDEFGH', ['sk-live-ABCDEFGH']);
  assert.ok(!out.includes('sk-live-ABCDEFGH') && !out.includes('abcDEF_123-x'));
});

const ev = (type, data, seq = 1, time = 1000) => ({ type: 'event', event: { type, seq, time, data } });
const chunk = c => ({ type: 'assistant-stream', frame: { type: 'chunk', attemptId: 'a:1', chunk: c } });

test('a tool call walks running -> proposed -> done with its approval, and carries its result', () => {
  const p = new TurnProjector({ startedAt: 0 });
  const call = p.consume(ev('tool/call', { callId: 'c1', name: 'pwsh', arguments: JSON.stringify({ command: 'ls', description: 'List files' }) }, 1, 1000));
  assert.equal(call[0].tool.state, 'running'); assert.equal(call[0].tool.scope, 'shell'); assert.equal(call[0].tool.preview, 'ls');
  const asked = p.consume(ev('approval/asked', { id: 'ap1', callId: 'c1', reason: 'escalate sandbox' }));
  assert.equal(asked[0].tool.state, 'proposed'); assert.match(asked[0].tool.effect, /escalate/);
  const decided = p.consume(ev('approval/decided', { id: 'ap1', outcome: 'allowed-once' }));
  assert.equal(decided[0].tool.state, 'running');
  const done = p.consume(ev('tool/result', { message: { toolCallId: 'c1', content: [{ type: 'text', text: 'ok' }] } }, 3, 1500));
  assert.equal(done[0].tool.state, 'done'); assert.equal(done[0].tool.result, 'ok'); assert.equal(done[0].tool.durationMs, 500);
});

test('a rejected approval ends as denied even though the Harness returns an error result', () => {
  const p = new TurnProjector({ startedAt: 0 });
  p.consume(ev('tool/call', { callId: 'c2', name: 'pwsh', arguments: '{}' }));
  p.consume(ev('approval/asked', { id: 'ap2', callId: 'c2' }));
  assert.equal(p.consume(ev('approval/decided', { id: 'ap2', outcome: 'rejected' }))[0].tool.state, 'denied');
  const res = p.consume(ev('tool/result', { message: { toolCallId: 'c2', isError: true, content: [{ type: 'text', text: 'the user rejected tool' }] } }));
  assert.equal(res[0].tool.state, 'denied');
});

test('live frames become deltas, usage is summed across steps, the turn ends with one terminal frame', () => {
  const p = new TurnProjector({ startedAt: 0 });
  assert.deepEqual(p.consume(chunk({ type: 'text-delta', index: 0, text: 'Hi' })), [{ type: 'delta', text: 'Hi' }]);
  assert.deepEqual(p.consume(chunk({ type: 'reasoning-delta', index: 0, text: 'hmm' })), [{ type: 'reasoning', text: 'hmm' }]);
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 100, outputTokens: 5 } }));
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 120, outputTokens: 7 } }));
  p.consume(ev('turn/end', { turn: 1, reason: { kind: 'completed' } }));
  const [done] = p.finish({ model: 'M', contextMax: 1000 });
  assert.equal(done.type, 'done'); assert.equal(done.tokensIn, 220); assert.equal(done.tokensOut, 12); assert.equal(done.usageSource, 'reported');
});

test('every turn-end reason maps to a frame the chat knows', () => {
  const finish = reason => { const p = new TurnProjector(); p.ended = reason; return p.finish()[0]; };
  assert.equal(finish({ kind: 'aborted', reason: { kind: 'user' } }).cancelled, true);
  assert.equal(finish({ kind: 'max-tokens' }).finishReason, 'length');
  assert.equal(finish({ kind: 'error', error: { code: 'X', message: 'context window exceeded' } }).type, 'error');
  assert.match(finish({ kind: 'error', error: { message: 'request exceeds the context window' } }).message, /no longer fits/);
  assert.equal(finish({ kind: 'blocked' }).code, 'BLOCKED');
  assert.equal(finish(null).code, 'INTERRUPTED');
  assert.equal(finish({ kind: 'completed' }).usageSource, 'unavailable', 'no usage reported means unavailable, not zero');
});

test('a text the live frames missed is completed from the committed message, once', () => {
  const p = new TurnProjector();
  p.consume(chunk({ type: 'text-delta', index: 0, text: 'Hel' }));
  const out = p.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: 'Hello there' }] } }));
  assert.deepEqual(out, [{ type: 'delta', text: 'lo there' }]);
  assert.deepEqual(p.consume(ev('assistant/message', { message: { content: [{ type: 'text', text: 'Hello there' }] } })), []);
});

test('tool scopes and previews', () => {
  assert.equal(toolScope('write'), 'write'); assert.equal(toolScope('grep'), 'read'); assert.equal(toolScope('web_fetch'), 'connected');
  assert.equal(presentTool({ callId: 'x', name: 'write', args: { file_path: 'a.txt' } }).preview, 'a.txt');
  assert.equal(friendlyFailure({ message: 'No API key' }), 'The model rejected the credentials. Check the connection in Settings.');
});

test('a chat that existed before the agent carries a bounded summary into its first agent turn', () => {
  const history = Array.from({ length: 30 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `m${i}`.padEnd(2000, 'x') }));
  const prompt = withEarlierConversation(history, 'NOW');
  assert.match(prompt, /Current message:\nNOW$/);
  assert.ok(prompt.length < 12 * 1600 + 200, 'bounded');
  assert.equal(withEarlierConversation([], 'NOW'), 'NOW');
});

test('the browser and the person\'s own tools have plain sentences, translated in every language', async () => {
  const { readFileSync } = await import('node:fs');
  const catalog = Object.fromEntries(['en', 'de', 'es', 'ja', 'ko', 'pt', 'ru', 'zh'].map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/${l}.json`, import.meta.url), 'utf8'))]));
  const idOf = text => Object.entries(catalog.en).find(([, v]) => v === text)?.[0];
  const cases = [
    ['mcp__browser__browser_navigate', 'Opening a page in the browser', 'Opened a page in the browser', "Couldn't open the page"],
    ['mcp__browser__browser_click', 'Clicking on the page', 'Clicked on the page', "Couldn't click on the page"],
    ['mcp__browser__browser_type', 'Typing on the page', 'Typed on the page', "Couldn't type on the page"],
    ['mcp__browser__browser_snapshot', 'Looking at the page', 'Looked at the page', "Couldn't look at the page"],
    ['mcp__browser__browser_evaluate', 'Running a small program on the page', 'Ran a small program on the page', "Couldn't run the program on the page"],
    ['mcp__browser__browser_wait_for', 'Waiting for the page', 'Waited for the page', "Couldn't wait for the page"],
    ['mcp__browser__browser_navigate_back', 'Going back a page', 'Went back a page', "Couldn't go back"],
    ['mcp__browser__browser_tabs', 'Managing browser tabs', 'Managed browser tabs', "Couldn't manage the browser tabs"],
    ['mcp__browser__browser_close', 'Closing the browser page', 'Closed the browser page', "Couldn't close the browser page"],
    ['mcp__browser__browser_something_new', 'Using the browser', 'Used the browser', 'The browser step failed'],
    ['mcp__strata-instructions__house_style', 'Reading your instructions', 'Read your instructions', "Couldn't read your instructions"],
    ['mcp__strata-commands__check_site', 'Running your tool', 'Ran your tool', 'Your tool failed'],
    ['mcp__github__create_issue', 'Using a tool', 'Used a tool', 'The tool failed'],
  ];
  for (const [name, running, done, failed] of cases) {
    assert.equal(toolVerb(name, 'running'), running, name); assert.equal(toolVerb(name, 'done'), done, name); assert.equal(toolVerb(name, 'error'), failed, name);
    for (const text of [running, done, failed]) { const id = idOf(text); assert.ok(id, `${text} is in the catalog`); for (const l of ['de', 'es', 'ja', 'ko', 'pt', 'ru', 'zh']) assert.ok(catalog[l][id]?.trim(), `${l}: ${text}`); }
  }
  const nav = presentTool({ callId: 'c1', name: 'mcp__browser__browser_navigate', args: { url: 'https://example.com' } });
  assert.equal(nav.target, 'https://example.com'); assert.equal(nav.verb, 'Opening a page in the browser');
  const own = presentTool({ callId: 'c2', name: 'mcp__strata-commands__check_site', args: {} });
  assert.equal(own.target, 'check site');
  assert.equal(presentTool({ callId: 'c3', name: 'mcp__github__create_issue', args: {} }).target, 'mcp__github__create_issue', 'an unknown connected tool keeps its name');
  assert.equal(presentTool({ callId: 'c4', name: 'mcp__browser__browser_snapshot', args: {} }).preview, '', 'no empty braces on a card');
  assert.equal(presentTool({ callId: 'c5', name: 'mcp__browser__browser_evaluate', args: { function: '() => document.title' } }).preview, '', 'a script in the page is not shown as code on the card');
});

test('an API model\'s route gets a generous answer room, or the connection\'s own maximum when the provider listed one', () => {
  const profile = buildProfile({ local: [], connections: [{ id: 'c1', name: 'C', baseUrl: 'https://x/v1', protocol: 'openai-chat', apiKey: 'k', models: [
    { id: 'unknown', contextMax: null }, { id: 'big', contextMax: 200_000, maxOutput: 64_000 }, { id: 'small', contextMax: 16_384 }] }] });
  const providers = JSON.parse(profile.patch.split('\n').slice(1).join('\n')).find(e => e.id === 'llm-pi-ai').config.providers;
  const models = Object.values(providers)[0].models;
  const by = Object.fromEntries(models.map(m => [m.id, m.maxTokens]));
  assert.equal(by.unknown, 32768); assert.equal(by.big, 64_000); assert.equal(by.small, 16_384 - 4096);
});
