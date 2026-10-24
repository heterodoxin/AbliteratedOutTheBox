/**
 * Settings -> Tools, behind the screen: what is saved, what reaches the agent, web search with no key, and the person's own tools.
 *
 * Pure tests always run. The two that start the real Harness need STRATA_HARNESS_DIR (the model is scripted; what is real is the Harness,
 * its web seam and tool executor, the Strata gate plugin, the bridge and the app's own tool server).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { buildProfile, CUSTOM_TOOLS_SCRIPT } from '../server/agent/profile.mjs';
import { standardPreset } from '../server/agent/preset.mjs';
import { TOOL_GROUPS } from '../server/agent/tools-catalog.mjs';
import { createAgentTools } from '../server/agent/tools-settings.mjs';
import { findBrowser, browserCandidates, findPlaywrightMcp } from '../server/agent/browser-probe.mjs';
import { manualDecision, planDecision } from '../server/agent/plugins/strata-gate.mjs';
import { createDuckDuckGoSearch, looksBlocked, parseHtmlResults, parseLiteResults, realUrl, BLOCKED_MESSAGE } from '../server/web-search-ddg.mjs';
import { createExaSearch, createPerplexitySearch } from '../server/web-search-keyed.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const tmp = () => mkdtempSync(join(tmpdir(), 'strata-tools-set-'));
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };
const make = (dir, state = {}) => { let saved = state; return { tools: createAgentTools({ read: () => saved, write: v => { saved = v; }, dataDir: dir }), saved: () => saved }; };

// ------------------------------------------------------------------------------------------------ the saved choices

test('Tools: every tool is on at first, in plain words, and turning one off or on is remembered', () => {
  const dir = tmp();
  try {
    const { tools } = make(dir);
    const v = tools.view();
    assert.equal(v.groups.length, TOOL_GROUPS.length);
    assert.ok(v.groups.every(g => g.enabled && g.name && g.about && ['computer', 'web', 'together'].includes(g.area)));
    assert.equal(v.search.provider, 'builtin'); assert.equal(v.search.keySet, false);
    const off = tools.update({ groups: { jobs: false, search: false } });
    assert.deepEqual(off.groups.filter(g => !g.enabled).map(g => g.id), ['jobs', 'search']);
    assert.deepEqual([...tools.profileView().off].sort(), ['jobs', 'search']);
    assert.ok(tools.update({ groups: { jobs: true } }).groups.find(g => g.id === 'jobs').enabled);
    assert.throws(() => tools.update({ groups: { nonsense: false } }), /not known/);
  } finally { cleanup(dir); }
});

test('Tools: a search key is saved for its service only, is never shown again, and is refused when it looks wrong', () => {
  const dir = tmp();
  try {
    const { tools, saved } = make(dir);
    tools.update({ search: { provider: 'exa' } });
    assert.throws(() => tools.update({ search: { key: 'has a space' } }), /does not look right/);
    const v = tools.update({ search: { key: 'exa-key-123456' } });
    assert.deepEqual(v.search, { provider: 'exa', keySet: true });
    assert.ok(!JSON.stringify(v).includes('exa-key-123456'), 'the screen never gets the key back');
    assert.equal(tools.profileView().search.key, 'exa-key-123456');
    assert.equal(tools.update({ search: { provider: 'perplexity' } }).search.keySet, false, 'a key belongs to its service');
    assert.throws(() => tools.update({ search: { provider: 'bing' } }), /listed search services/);
    assert.equal(saved().search.provider, 'perplexity');
  } finally { cleanup(dir); }
});

test('Tools: the person\'s own tools are checked in plain words and saved with the three ways of working', () => {
  const dir = tmp();
  try {
    const { tools } = make(dir);
    assert.throws(() => tools.addCustom({ name: '', about: 'x', kind: 'instructions', instructions: 'y' }), /short name/);
    assert.throws(() => tools.addCustom({ name: 'Notes', about: '', kind: 'instructions', instructions: 'y' }), /what the tool does/);
    assert.throws(() => tools.addCustom({ name: 'Notes', about: 'a', kind: 'instructions', instructions: '  ' }), /instructions/);
    assert.throws(() => tools.addCustom({ name: 'Run', about: 'a', kind: 'command', command: 'a\nb' }), /one line/);
    assert.throws(() => tools.addCustom({ name: 'Srv', about: 'a', kind: 'server', address: 'https://u:p@x.example/mcp' }), /user name or password/);
    let v = tools.addCustom({ name: 'Release notes', about: 'Write release notes in our house style.', kind: 'instructions', instructions: 'Start with one line.' });
    v = tools.addCustom({ name: 'Check site', about: 'Check that the site answers.', kind: 'command', command: 'curl -I https://example.com' });
    v = tools.addCustom({ name: 'Issue tracker', about: 'Look up issues.', kind: 'server', address: 'https://issues.example/mcp' });
    assert.deepEqual(v.custom.map(t => [t.id, t.kind]), [['release_notes', 'instructions'], ['check_site', 'command'], ['issue_tracker', 'server']]);
    assert.throws(() => tools.addCustom({ name: 'release notes', about: 'a', kind: 'instructions', instructions: 'y' }), /already have a tool/);
    const p = tools.profileView();
    assert.equal(p.hasInstructionTools && p.hasCommandTools, true);
    assert.deepEqual(p.servers, { 'custom-issue-tracker': { url: 'https://issues.example/mcp' } });
    const file = JSON.parse(readFileSync(tools.filePath, 'utf8'));
    assert.deepEqual(Object.keys(file.instructions), ['release_notes']); assert.deepEqual(Object.keys(file.commands), ['check_site']);
    v = tools.update({ custom: { check_site: false } });
    assert.equal(tools.profileView().hasCommandTools, false); assert.deepEqual(JSON.parse(readFileSync(tools.filePath, 'utf8')).commands, {});
    v = tools.editCustom('release_notes', { name: 'Release notes', about: 'Changed.', kind: 'instructions', instructions: 'New text.' });
    assert.equal(v.custom[0].about, 'Changed.');
    v = tools.removeCustom('issue_tracker');
    assert.equal(v.custom.length, 2);
    assert.throws(() => tools.removeCustom('nope'), /not in your list/);
  } finally { cleanup(dir); }
});

// ---------------------------------------------------------------------------------------------- what the agent gets

const entriesOf = profile => JSON.parse(profile.patch.split('\n').slice(1).join('\n'));
const toolWeb = entries => entries.find(e => e.id === 'preset-standard').config.plugins.find(p => p.id === 'tool-web');

test('profile: web search is on with no key (DuckDuckGo), pinned so a second backend cannot make the choice ambiguous', () => {
  const e = entriesOf(buildProfile({ local: [], connections: [] }));
  assert.equal(toolWeb(e).config.search, true); assert.equal(toolWeb(e).config.fetch, true);
  assert.deepEqual(e.find(x => x.id === 'web').config, { searchProvider: 'strata-duckduckgo', fetchProvider: 'http' });
  assert.equal(e.find(x => x.id === 'web-search-deepseek').disabled, true);
  const plugin = e.flatMap(x => x.insert ?? []).find(x => x.id === 'strata-web-search');
  assert.ok(existsSync(plugin.name)); assert.equal(plugin.config.provider, 'duckduckgo');
});

test('profile: a keyed service is used only with its key saved; the key travels in the environment and is in no file', () => {
  const view = (provider, key) => ({ off: new Set(), search: { provider, key }, servers: {}, hasInstructionTools: false, hasCommandTools: false, toolFile: null, browserEnabled: false });
  const noKey = buildProfile({ local: [], connections: [], tools: view('exa', '') });
  assert.equal(entriesOf(noKey).find(x => x.id === 'web').config.searchProvider, 'strata-duckduckgo', 'no key: DuckDuckGo, not a broken search');
  const exa = buildProfile({ local: [], connections: [], tools: view('exa', 'exa-secret-key') });
  assert.equal(entriesOf(exa).find(x => x.id === 'web').config.searchProvider, 'exa');
  assert.equal(exa.env.STRATA_SEARCH_KEY, 'exa-secret-key'); assert.ok(!exa.patch.includes('exa-secret-key'));
  const ds = buildProfile({ local: [], connections: [], tools: view('deepseek', 'ds-secret') });
  assert.deepEqual(entriesOf(ds).find(x => x.id === 'web-search-deepseek').config, { apiKeyEnv: 'STRATA_SEARCH_KEY' });
  assert.equal(entriesOf(ds).find(x => x.id === 'web').config.searchProvider, 'deepseek-official');
  assert.notEqual(noKey.signature, exa.signature, 'a changed choice restarts the agent');
});

test('profile: the groups switched off are not loaded; both web halves off drops the web plugin', () => {
  const base = { servers: {}, hasInstructionTools: false, hasCommandTools: false, toolFile: null, browserEnabled: false, search: { provider: 'builtin', key: '' } };
  const ids = off => standardPreset({ off: new Set(off), search: true }).plugins.map(p => p.id);
  assert.ok(!ids(['shell']).includes('tool-pwsh') && !ids(['shell']).includes('tool-bash'));
  assert.ok(!ids(['helpers']).includes('delegation') && ids(['helpers']).includes('tool-fs'));
  assert.ok(!ids(['goals']).includes('tool-goal') && !ids(['goals']).includes('command-goal'));
  assert.ok(ids(['fetch']).includes('tool-web') && !ids(['fetch', 'search']).includes('tool-web'));
  const e = entriesOf(buildProfile({ local: [], connections: [], tools: { ...base, off: new Set(['search']) } }));
  assert.equal(toolWeb(e).config.search, false);
  assert.equal(e.find(x => x.id === 'web'), undefined, 'no search service is set up when search is off');
});

test('profile: the person\'s instruction and command tools are the app\'s own tool server; a person\'s server is a connected tool', () => {
  const dir = tmp();
  try {
    const { tools } = make(dir);
    tools.addCustom({ name: 'Notes', about: 'a', kind: 'instructions', instructions: 'text' });
    tools.addCustom({ name: 'Run it', about: 'a', kind: 'command', command: 'echo hi' });
    tools.addCustom({ name: 'Remote', about: 'a', kind: 'server', address: 'https://r.example/mcp' });
    const e = entriesOf(buildProfile({ local: [], connections: [], tools: tools.profileView() }));
    const rows = e.flatMap(x => x.insert ?? []).filter(r => r.name === '@deepseek-ai/dsh-mcp-client');
    assert.deepEqual(rows.map(r => r.config.serverName).sort(), ['custom-remote', 'strata-commands', 'strata-instructions']);
    const own = rows.find(r => r.config.serverName === 'strata-commands').config;
    assert.deepEqual([own.command, own.args[0], own.args[1]], [process.execPath, CUSTOM_TOOLS_SCRIPT, 'commands']);
  } finally { cleanup(dir); }
});

test('the gate: instruction tools only hand back text, so they do not ask; commands, servers and the browser ask in Manual and are refused in Plan', () => {
  assert.equal(manualDecision({ name: 'mcp__strata-instructions__notes', arguments: {} }), null);
  assert.equal(manualDecision({ name: 'mcp__strata-commands__run_it', arguments: {} })?.kind, 'ask');
  assert.equal(manualDecision({ name: 'mcp__strata-commands__check_site', arguments: { input: 'example.com' } }).reason, 'Run your tool “check site” with: example.com');
  assert.equal(manualDecision({ name: 'mcp__custom-remote__lookup', arguments: {} })?.kind, 'ask');
  assert.equal(planDecision({ name: 'mcp__strata-instructions__notes' }), null);
  assert.equal(planDecision({ name: 'read' }), null);
  for (const name of ['mcp__strata-commands__run_it', 'mcp__custom-remote__lookup', 'mcp__browser__browser_navigate']) assert.equal(planDecision({ name })?.kind, 'deny', name);
  assert.match(planDecision({ name: 'mcp__x__y' }).reason, /Plan mode only looks/);
});

// ------------------------------------------------------------------------------------------------- web search alone

const HTML = `<div class="result results_links results_links_deep web-result">
<div class="links_main links_deep result__body"><h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc">Example <b>Domain</b> &amp; more</a></h2>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">A <b>short</b> snippet &#x27;quoted&#x27;.</a></div></div>
<div class="result result--ad"><h2 class="result__title"><a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_domain=ad.example">An ad</a></h2></div>
<div class="result"><h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fsecond.example%2F">Second</a></h2>
<a class="result__snippet" href="x">Second snippet</a></div>`;
const LITE = `<table><tr><td>1.&nbsp;</td><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flite.example%2F" class='result-link'>Lite <b>title</b></a></td></tr>
<tr><td class='result-snippet'>Lite snippet text</td></tr>
<tr><td>2.&nbsp;</td><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flite2.example%2F" class='result-link'>Lite two</a></td></tr></table>`;

test('search: the DuckDuckGo result list is read into titles, addresses and snippets; ads and internal links are dropped', () => {
  const rows = parseHtmlResults(HTML);
  assert.deepEqual(rows, [
    { url: 'https://example.com/a?x=1', title: 'Example Domain & more', snippet: "A short snippet 'quoted'." },
    { url: 'https://second.example/', title: 'Second', snippet: 'Second snippet' },
  ]);
  assert.deepEqual(parseLiteResults(LITE).map(r => [r.url, r.title, r.snippet]), [['https://lite.example/', 'Lite title', 'Lite snippet text'], ['https://lite2.example/', 'Lite two', undefined]]);
  assert.equal(realUrl('//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2F'), 'https://a.example/');
  assert.equal(realUrl('https://duckduckgo.com/y.js?x=1'), null);
  assert.equal(realUrl('javascript:alert(1)'), null);
});

test('search: a human check or a refusal is told in plain words, and a real search with no hits is an answer, not an error', async () => {
  assert.equal(looksBlocked(202, ''), true); assert.equal(looksBlocked(429, ''), true);
  assert.equal(looksBlocked(200, '<div class="anomaly-modal">Unfortunately, bots use DuckDuckGo too</div>'), true);
  assert.equal(looksBlocked(200, HTML), false);
  const sleeps = [];
  const tick = () => { let t = 1_000_000; return { now: () => t, sleep: async ms => { sleeps.push(ms); t += ms; } }; };
  const blocked = createDuckDuckGoSearch({ ...tick(), fetch: async () => new Response('<html>captcha anomaly-modal</html>', { status: 200 }) });
  await assert.rejects(blocked.search({ query: 'x' }), e => e.message === BLOCKED_MESSAGE && e.code === 'WEB_RATE_LIMITED');
  const empty = createDuckDuckGoSearch({ ...tick(), fetch: async () => new Response('<html><body>No results.</body></html>', { status: 200 }) });
  assert.deepEqual(await empty.search({ query: 'zzzz' }), { sources: [], truncated: false });
  const offline = createDuckDuckGoSearch({ ...tick(), fetch: async () => { throw new TypeError('fetch failed'); } });
  await assert.rejects(offline.search({ query: 'x' }), /could not reach the internet/);
});

test('search: polite by design: one request at a time, spaced out, a minute limit, and the same query is answered from a short cache', async () => {
  let clock = 1_000_000, calls = 0;
  const waits = [];
  const provider = createDuckDuckGoSearch({
    now: () => clock, sleep: async ms => { waits.push(ms); clock += ms; },
    fetch: async (url, init) => { calls++; assert.equal(init.method, 'POST'); assert.match(init.headers['user-agent'], /Mozilla/); assert.match(String(url), /duckduckgo\.com/); return new Response(HTML, { status: 200 }); },
  });
  const a = await provider.search({ query: 'Strata engine', maxResults: 1 });
  assert.equal(a.sources.length, 1);
  await provider.search({ query: 'strata ENGINE' });          // same words: from the cache
  assert.equal(calls, 1);
  await provider.search({ query: 'another one' });
  assert.equal(calls, 2); assert.ok(waits.some(ms => ms >= 1400), 'the second request waited for the gap');
  for (let i = 0; i < 25; i++) { try { await provider.search({ query: `q${i}` }); } catch (e) { assert.equal(e.code, 'WEB_RATE_LIMITED'); break; } }
  assert.ok(calls <= 23, 'no more than about 20 a minute');
});

test('search: Exa and Perplexity use the person\'s key, map their answers, and say plainly when the key is refused', async () => {
  const real = globalThis.fetch;
  try {
    globalThis.fetch = async (url, init) => {
      assert.equal(init.headers.authorization, 'Bearer the-key');
      if (String(url).includes('exa.ai')) return new Response(JSON.stringify({ results: [{ url: 'https://e.example', title: 'E', highlights: ['hl'] }, { url: 'https://no-snippet.example', highlights: [] }] }));
      return new Response(JSON.stringify({ choices: [{ message: { content: 'Answer.' } }], search_results: [{ url: 'https://p.example', title: 'P', snippet: 's' }] }));
    };
    assert.deepEqual((await createExaSearch({ key: 'the-key' }).search({ query: 'q' })).sources, [{ url: 'https://e.example', title: 'E', snippet: 'hl' }]);
    const p = await createPerplexitySearch({ key: () => 'the-key' }).search({ query: 'q' });
    assert.equal(p.content, 'Answer.'); assert.equal(p.sources[0].url, 'https://p.example');
    assert.equal(createExaSearch({ key: () => '' }).available(), false);
    globalThis.fetch = async () => new Response('{}', { status: 401 });
    await assert.rejects(createExaSearch({ key: 'the-key' }).search({ query: 'q' }), /did not accept the key/);
  } finally { globalThis.fetch = real; }
});

// ------------------------------------------------------------------------------------------------------- browser probe

test('browser: Chrome, then Edge, then Chromium; nothing found says so', () => {
  const win = { PROGRAMFILES: 'C:\\Program Files', LOCALAPPDATA: 'C:\\Users\\x\\AppData\\Local' };
  assert.equal(browserCandidates('win32', win)[0].name, 'Google Chrome');
  assert.deepEqual(findBrowser({ platform: 'win32', env: win, exists: p => /msedge\.exe$/.test(p) }).name, 'Microsoft Edge');
  assert.equal(findBrowser({ platform: 'win32', env: win, exists: () => false }).found, false);
  assert.equal(findBrowser({ platform: 'darwin', exists: p => p.includes('Google Chrome') }).found, true);
});

// ------------------------------------------------------------------------------------ the app's own tool server (MCP)

function mcpSession(dir, mode) {
  const child = spawn(process.execPath, [CUSTOM_TOOLS_SCRIPT, mode, join(dir, 'agent', 'custom-tools.json')], { stdio: ['pipe', 'pipe', 'inherit'], windowsHide: true });
  const waiting = new Map();
  createInterface({ input: child.stdout }).on('line', line => { const m = JSON.parse(line); waiting.get(m.id)?.(m); });
  let id = 0;
  const ask = (method, params) => new Promise(resolve => { const n = ++id; waiting.set(n, resolve); child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: n, method, params }) + '\n'); });
  return { ask, close: () => child.kill() };
}

test('tool server: instruction tools return the person\'s text; command tools run their command with the assistant\'s text only in STRATA_INPUT', async () => {
  const dir = tmp();
  try {
    const script = join(dir, 'show.mjs');
    writeFileSync(script, "console.log('got:' + process.env.STRATA_INPUT); if (process.env.STRATA_INPUT === 'fail') process.exit(3);");
    const { tools } = make(dir);
    tools.addCustom({ name: 'Notes', about: 'House style.', kind: 'instructions', instructions: 'Start with one line.' });
    tools.addCustom({ name: 'Show', about: 'Shows its input.', kind: 'command', command: `"${process.execPath}" "${script}"` });
    const instr = mcpSession(dir, 'instructions');
    try {
      assert.equal((await instr.ask('initialize', { protocolVersion: '2024-11-05' })).result.serverInfo.name, 'strata-instructions');
      const list = (await instr.ask('tools/list')).result.tools;
      assert.deepEqual(list.map(t => [t.name, t.description]), [['notes', 'House style.']]);
      assert.equal((await instr.ask('tools/call', { name: 'notes', arguments: {} })).result.content[0].text, 'Start with one line.');
    } finally { instr.close(); }
    const cmd = mcpSession(dir, 'commands');
    try {
      await cmd.ask('initialize', {});
      assert.deepEqual((await cmd.ask('tools/list')).result.tools.map(t => t.name), ['show']);
      const ok = (await cmd.ask('tools/call', { name: 'show', arguments: { input: 'hello; echo HACKED' } })).result;
      assert.equal(ok.isError, false); assert.equal(ok.content[0].text, 'got:hello; echo HACKED', 'the text is data, never a command');
      const bad = (await cmd.ask('tools/call', { name: 'show', arguments: { input: 'fail' } })).result;
      assert.equal(bad.isError, true);
      assert.match((await cmd.ask('tools/call', { name: 'gone', arguments: {} })).result.content[0].text, /no longer available/);
      tools.update({ custom: { show: false } });
      assert.deepEqual((await cmd.ask('tools/list')).result.tools, [], 'a change in Settings is seen without a restart');
    } finally { cmd.close(); }
  } finally { cleanup(dir); }
});

// ------------------------------------------------------------------------------------------------- real Harness

async function rig(t, setup) {
  const worker = await startScriptedWorker({});
  const dir = tmp();
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const { tools } = make(dir);
  setup?.(tools, dir);
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], tools: () => tools.profileView(), env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const run = async (chatId, tool, args, { mode = 'bypass', answer } = {}) => {
    const frames = [];
    await bridge.runTurn({ chatId, message: `SCRIPT:call ${tool} ${JSON.stringify(args)}`, workspacePath: ws, mode, onFrame: f => {
      frames.push(f);
      if (answer && f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId, callId: f.tool.id, decision: answer }).catch(() => {});
    } });
    return frames.filter(f => f.type === 'agent-tool').map(f => f.tool);
  };
  return { worker, bridge, tools, dir, ws, run, offered: () => [...worker.requests].reverse().find(q => q.toolNames.length)?.toolNames.sort() };
}

test('real Harness: the person\'s own tools are offered; an instruction tool never asks, a command asks in Manual and is refused in Plan', { skip }, async (t) => {
  const r = await rig(t, (tools, dir) => {
    const script = join(dir, 'touch.mjs');
    writeFileSync(script, "import { writeFileSync } from 'node:fs'; writeFileSync(process.argv[2], 'ran'); console.log('ok:' + process.env.STRATA_INPUT);");
    tools.addCustom({ name: 'House style', about: 'Write in our house style.', kind: 'instructions', instructions: 'Always start with one line.' });
    tools.addCustom({ name: 'Make marker', about: 'Writes a marker file.', kind: 'command', command: `"${process.execPath}" "${script}" "${join(dir, 'marker.txt')}"` });
  });
  const marker = join(r.dir, 'marker.txt');
  const first = await r.run('c1', 'mcp__strata-instructions__house_style', {}, { mode: 'manual' });
  assert.ok(r.offered().includes('mcp__strata-instructions__house_style') && r.offered().includes('mcp__strata-commands__make_marker'), 'both of the person\'s tools are offered');
  assert.ok(!first.some(c => c.state === 'proposed'), 'an instruction tool never asks');
  assert.match(first.at(-1).result, /Always start with one line/);
  const plan = await r.run('c2', 'mcp__strata-commands__make_marker', { input: 'p' }, { mode: 'plan' });
  assert.ok(!existsSync(marker), 'Plan mode refused the command');
  assert.ok(!plan.some(c => c.state === 'done' && /ok:/.test(c.result ?? '')));
  const denied = await r.run('c3', 'mcp__strata-commands__make_marker', { input: 'd' }, { mode: 'manual', answer: 'deny' });
  assert.ok(denied.some(c => c.state === 'proposed') && !existsSync(marker), 'Manual asked first, and a No ran nothing');
  const approved = await r.run('c4', 'mcp__strata-commands__make_marker', { input: 'yes' }, { mode: 'manual', answer: 'approve' });
  assert.ok(approved.some(c => c.state === 'proposed'));
  assert.equal(approved.at(-1).state, 'done'); assert.match(approved.at(-1).result, /ok:yes/); assert.ok(existsSync(marker));
});

test('real Harness: switching a tool off in Settings takes it away from the next message; web search is there with no key', { skip }, async (t) => {
  const r = await rig(t);
  await r.run('o1', 'read', { file_path: join(r.ws, 'nothing.txt') });
  const before = r.offered();
  assert.ok(before.includes('web_search') && before.includes('web_fetch') && before.includes('subagent') && before.includes('job_list'));
  r.tools.update({ groups: { jobs: false, helpers: false, search: false } });
  await r.run('o2', 'read', { file_path: join(r.ws, 'nothing.txt') });
  const after = r.offered();
  assert.ok(!after.includes('web_search') && !after.includes('subagent') && !after.includes('job_list'), 'the three groups are gone');
  assert.ok(after.includes('web_fetch') && after.includes('read') && after.includes('pwsh') !== after.includes('bash'), 'the rest stays');
});

// ------------------------------------------------------------------------------------------------------------ browser

test('gate: in Manual the browser asks before opening a page, clicking or typing, and not before looking; Plan refuses it', () => {
  const nav = manualDecision({ name: 'mcp__browser__browser_navigate', arguments: { url: 'https://example.com/a' } });
  assert.equal(nav.kind, 'ask'); assert.equal(nav.reason, 'Open the web page https://example.com/a');
  assert.match(manualDecision({ name: 'mcp__browser__browser_click', arguments: { element: 'Sign in button' } }).reason, /^Click on the page: Sign in button/);
  assert.match(manualDecision({ name: 'mcp__browser__browser_type', arguments: { element: 'Search box' } }).reason, /^Type into the page/);
  assert.equal(manualDecision({ name: 'mcp__browser__browser_file_upload', arguments: {} }).kind, 'ask');
  assert.match(manualDecision({ name: 'mcp__browser__browser_evaluate', arguments: {} }).reason, /small program/);
  for (const name of ['browser_snapshot', 'browser_take_screenshot', 'browser_console_messages', 'browser_network_requests', 'browser_wait_for']) assert.equal(manualDecision({ name: `mcp__browser__${name}`, arguments: {} }), null, name);
  assert.equal(manualDecision({ name: 'mcp__browser__browser_tabs', arguments: { action: 'list' } }), null);
  assert.equal(manualDecision({ name: 'mcp__browser__browser_tabs', arguments: { action: 'close' } }).kind, 'ask');
  assert.equal(planDecision({ name: 'mcp__browser__browser_snapshot' }).kind, 'deny');
});

test('browser: the tool server is found in the agent component; the screen can say what is missing', () => {
  const exists = p => p.replaceAll(String.fromCharCode(92), '/').endsWith('/node_modules/@playwright/mcp/cli.js') && p.replaceAll(String.fromCharCode(92), '/').includes('/pkg/node_modules/@playwright/');
  assert.equal(findPlaywrightMcp('', {}), null);
  assert.ok(findPlaywrightMcp('/pkg', { exists }).replaceAll(String.fromCharCode(92), '/').endsWith('/pkg/node_modules/@playwright/mcp/cli.js'));
  assert.equal(findPlaywrightMcp('/nowhere', { exists: () => false, list: () => { throw new Error('none'); } }), null);
  const dir = tmp();
  try {
    const only = createAgentTools({ read: () => ({ browser: { enabled: true } }), write: () => {}, dataDir: dir, browser: () => ({ found: true, name: 'Google Chrome', path: 'C:/chrome.exe', cli: null }) });
    assert.deepEqual(only.view().browser, { enabled: true, found: false, name: 'Google Chrome', missing: 'component' });
    assert.equal(only.profileView().browserEnabled, false, 'no tool server, no browser rows');
    const ready = createAgentTools({ read: () => ({ browser: { enabled: true } }), write: () => {}, dataDir: dir, browser: () => ({ found: true, name: 'Google Chrome', path: 'C:/chrome.exe', cli: 'C:/pw/cli.js' }) });
    assert.equal(ready.view().browser.found, true);
    const rows = entriesOf(buildProfile({ local: [], connections: [], tools: ready.profileView() })).flatMap(x => x.insert ?? []).filter(r => r.config?.serverName === 'browser');
    assert.equal(rows.length, 1);
    assert.deepEqual(rows[0].config.args, ['C:/pw/cli.js', '--browser', 'chromium', '--isolated', '--executable-path', 'C:/chrome.exe']);
    const off = createAgentTools({ read: () => ({ browser: { enabled: false } }), write: () => {}, dataDir: dir, browser: () => ({ found: true, name: 'x', path: 'p', cli: 'c' }) });
    assert.equal(entriesOf(buildProfile({ local: [], connections: [], tools: off.profileView() })).flatMap(x => x.insert ?? []).some(r => r.config?.serverName === 'browser'), false);
  } finally { cleanup(dir); }
});

const browserNow = findBrowser();
const playwrightCli = HARNESS ? findPlaywrightMcp(HARNESS) : null;
// Opt-in: it starts a real (headless) Chrome, which a test run must not do on someone's PC unasked.
const browserSkip = process.env.STRATA_TEST_BROWSER !== '1' ? 'starts a real browser: set STRATA_TEST_BROWSER=1' : skip || !browserNow.found || !playwrightCli ? 'needs STRATA_HARNESS_DIR with the browser tool server and Chrome or Edge on this computer' : false;

test('real browser: the assistant opens a local page in the person\'s Chrome or Edge and reads it; Manual asks before opening it, Plan refuses', { skip: browserSkip, timeout: 240000 }, async (t) => {
  const http = await import('node:http');
  const page = http.createServer((req, res) => { res.setHeader('content-type', 'text/html'); res.end('<html><head><title>Strata browser test</title></head><body><h1>Hello from the browser test</h1><button>Press me</button></body></html>'); });
  await new Promise(r => page.listen(0, '127.0.0.1', r));
  t.after(() => page.close());
  const url = `http://127.0.0.1:${page.address().port}/`;
  const previous = process.env.STRATA_BROWSER_HEADLESS;
  process.env.STRATA_BROWSER_HEADLESS = '1';
  t.after(() => { if (previous === undefined) delete process.env.STRATA_BROWSER_HEADLESS; else process.env.STRATA_BROWSER_HEADLESS = previous; });
  const worker = await startScriptedWorker({});
  const dir = tmp();
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const real = createAgentTools({ read: () => ({ browser: { enabled: true } }), write: () => {}, dataDir: dir, browser: () => ({ ...browserNow, cli: playwrightCli }) });
  const local = [{ id: 'inst-b', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], tools: () => real.profileView(), env: process.env, startTimeoutMs: 120_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const go = async (chatId, tool, args, mode, answer) => {
    const frames = [];
    await bridge.runTurn({ chatId, message: `SCRIPT:call ${tool} ${JSON.stringify(args)}`, workspacePath: ws, mode, onFrame: f => {
      frames.push(f);
      if (answer && f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId, callId: f.tool.id, decision: answer }).catch(() => {});
    } });
    return frames.filter(f => f.type === 'agent-tool').map(f => f.tool);
  };
  const plan = await go('b0', 'mcp__browser__browser_navigate', { url }, 'plan');
  assert.ok(!plan.some(c => c.state === 'done' && /Hello from the browser test/.test(c.result ?? '')), 'Plan did not open the page');
  const denied = await go('b1', 'mcp__browser__browser_navigate', { url }, 'manual', 'deny');
  assert.ok(denied.some(c => c.state === 'proposed' && /Open the web page/.test(c.effect ?? '')), 'Manual asked first');
  const opened = await go('b2', 'mcp__browser__browser_navigate', { url }, 'manual', 'approve');
  assert.ok(opened.some(c => c.state === 'proposed'));
  assert.equal(opened.at(-1).state, 'done');
  const looked = await go('b2', 'mcp__browser__browser_snapshot', {}, 'manual', 'approve');
  assert.ok(!looked.some(c => c.state === 'proposed'), 'looking at the page does not ask');
  assert.match(looked.at(-1).result, /Hello from the browser test/);
});
