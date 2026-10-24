/**
 * The agent's tools in the app: the inventory, a tool from each family, and Manual mode's one approval card.
 *
 * Pure tests (the gate's rules, the preset list, the profile) always run. The rest need a built Harness
 * (STRATA_HARNESS_DIR) and skip loudly without it. The model is scripted (tests/helpers/scripted-worker.mjs): what is real is
 * the Harness, its tool executor, its sandbox, its approval service, the Strata gate plugin and the bridge.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { buildProfile, GATE_PLUGIN } from '../server/agent/profile.mjs';
import { STANDARD_PLUGIN_IDS, REMOVED_PLUGIN_IDS, standardPreset } from '../server/agent/preset.mjs';
import { isReadOnlyShell, manualDecision, maskQuotes, READ_ONLY_TOOLS } from '../server/agent/plugins/strata-gate.mjs';
import { plainToolError, withPlain } from '../server/agent/events.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* the Harness sandbox leaves access entries; best effort */ } };
const fwd = p => p.replaceAll('\\', '/');

// ---------------------------------------------------------------------------------------------------------- pure

test('the gate: only commands that provably just look are allowed without a question', () => {
  for (const ok of ['Get-ChildItem', 'Get-ChildItem -Recurse src | Select-Object -First 5', 'git status', 'git log --oneline -5', 'git diff', 'cat notes.txt', 'ls -la', 'node --version', 'Get-Content a.txt | Select-String needle', 'git branch', 'git remote -v']) {
    assert.equal(isReadOnlyShell(ok), true, ok);
  }
  for (const bad of ['', 'Set-Content a.txt x', 'echo hi > a.txt', 'Get-ChildItem; Remove-Item a.txt', 'rm -rf x', 'git commit -m x', 'git push', 'git branch -D x', 'git config user.name x', 'git remote add o u', 'curl http://x', 'Get-ChildItem | ForEach-Object { $_ }', 'Start-Process explorer.exe', 'explorer.exe C:/Users/x/Downloads', 'ii ~/Downloads', 'Invoke-Item $env:USERPROFILE/Downloads', 'start .', 'open .', 'xdg-open .', 'notepad a.txt', 'powershell -Command x', 'node script.js', 'python x.py', 'ls && rm a', 'find . -delete', 'echo $(whoami)', 'Invoke-WebRequest u', 'Start-Process x', 'New-Item a', 'cat a | tee b']) {
    assert.equal(isReadOnlyShell(bad), false, bad);
  }
  assert.equal(isReadOnlyShell('x'.repeat(700)), false, 'a long command is never trusted');
});

test('the gate: quoted text is data, so an & ( ) ; { } in a quoted path does not make a look-only command ask (2026-10-22)', () => {
  for (const ok of [
    'Get-ChildItem "C:\\Users\\AI-Server\\Downloads\\GUI Q&A" | Select-Object -First 5 Name',
    'Get-Content "C:\\Users\\AI-Server\\Downloads\\GUI Q&A\\a.txt"',
    "Get-ChildItem 'C:\\Users\\x\\Notes (old) {draft}; v2'",
    'Select-String -Path a.txt "a|b"',
    'Select-String needle "C:\\x\\a & b.txt" | Measure-Object',
    'git log --oneline "docs/Q&A.md"',
    'cat "my file (1).txt"',
    "Get-ChildItem 'it''s here'",
  ]) assert.equal(isReadOnlyShell(ok), true, ok);
  // Quoting does not hide anything that acts: outside the quotes the same characters still ask, and inside double quotes an expansion does.
  for (const bad of [
    'Get-ChildItem C:\\Q&A',
    'Get-ChildItem "C:\\a" ; Remove-Item x',
    'Get-ChildItem "C:\\a" & calc',
    'Get-ChildItem "C:\\a" > out.txt',
    'Get-ChildItem "$env:USERPROFILE"',
    'Get-ChildItem "$(whoami)"',
    'Get-ChildItem "a`b"',
    "Get-ChildItem 'unterminated",
    'Get-ChildItem "unterminated',
    'git diff "--output=out.txt"',
    'git diff --out"put"=out.txt',
    'git diff --"output"=out.txt',
    'Get-Content "a"b',
    '"Remove-Item" x',
    'Set-Content "a.txt" "x"',
    'Get-ChildItem "a" | Remove-Item',
    'curl "http://x"',
  ]) assert.equal(isReadOnlyShell(bad), false, bad);
  assert.equal(maskQuotes('Get-ChildItem "C:\\Q&A" | Select-Object Name'), 'Get-ChildItem "_" | Select-Object Name');
  assert.equal(maskQuotes("echo 'a' \"b\""), 'echo "_" "_"');
  assert.equal(maskQuotes('echo "$HOME"'), null);
});

test('the gate: every tool outside the read-only list asks, including unknown and connected ones', () => {
  for (const name of READ_ONLY_TOOLS) assert.equal(manualDecision({ name, arguments: {} }), null, name);
  for (const name of ['write', 'edit', 'job_kill', 'mcp__github__create_issue', 'something_new']) assert.equal(manualDecision({ name, arguments: {} })?.kind, 'ask', name);
  assert.equal(manualDecision({ name: 'pwsh', arguments: { command: 'Get-ChildItem' } }), null);
  assert.match(manualDecision({ name: 'pwsh', arguments: { command: 'Remove-Item x', description: 'Delete x' } }).reason, /Delete x/);
  assert.match(manualDecision({ name: 'write', arguments: { file_path: 'C:/a/b.txt' } }).reason, /b\.txt/);
  assert.equal(manualDecision({ name: 'pwsh', arguments: {} }).kind, 'ask', 'no command is not a read-only command');
});

test('the preset: our list is the Harness standard list minus exactly what the app replaces', () => {
  const ids = standardPreset({ platform: 'win32' }).plugins.map(p => p.id);
  assert.deepEqual(ids, STANDARD_PLUGIN_IDS.filter(id => !REMOVED_PLUGIN_IDS.includes(id) && id !== 'tool-bash'));
  assert.ok(standardPreset({ platform: 'linux' }).plugins.some(p => p.id === 'tool-bash') && !standardPreset({ platform: 'linux' }).plugins.some(p => p.id === 'tool-pwsh'));
  const web = standardPreset().plugins.find(p => p.id === 'tool-web').config;
  assert.equal(web.fetch, true); assert.equal(web.search, false, 'the preset alone leaves search to the profile, which switches it on (DuckDuckGo needs no key)');
  assert.equal(standardPreset({ search: true }).plugins.find(p => p.id === 'tool-web').config.search, true);
  assert.ok(!ids.includes('tool-schedule'), 'Tasks is the only scheduler');
});

test('the profile loads the gate by path, sizes helper agents to the engine slots and keeps no secret', () => {
  const p = buildProfile({ local: [{ id: 'a', baseUrl: 'http://127.0.0.1:1/v1', modelId: 'm', context: 32768, slots: 2, key: 'sekret-key-123' }] });
  const entries = JSON.parse(p.patch.split('\n').slice(1).join('\n'));
  assert.deepEqual(entries.find(e => e.insert)?.insert, [{ id: 'strata-gate', name: GATE_PLUGIN, config: { web: { provider: 'DuckDuckGo, or Bing when DuckDuckGo is busy', skipAsk: false } } }]);
  assert.ok(existsSync(GATE_PLUGIN));
  assert.equal(entries.find(e => e.id === 'subagent').config.maxActiveSubagents, 4, 'at least the 4 helpers; more only with more slots');
  assert.equal(entries.find(e => e.id === 'preset-standard').config.plugins.some(x => x.id === 'tool-schedule'), false);
  assert.ok(!p.patch.includes('sekret-key-123'));
  const api = buildProfile({ local: [], connections: [{ id: 'c1', name: 'C', baseUrl: 'https://x/v1', protocol: 'openai-chat', models: [{ id: 'm' }] }] });
  assert.equal(JSON.parse(api.patch.split('\n').slice(1).join('\n')).find(e => e.id === 'subagent').config.maxActiveSubagents, 8, 'API models take the Harness limit');
});

test('tool failures are said in plain words with the original text kept after them', () => {
  assert.match(plainToolError('Error: URL hostname "127.0.0.1" resolves to a non-public IP address'), /web page could not be opened/);
  assert.match(plainToolError('Error: invalid arguments: missing required property "files"'), /wrong way/);
  assert.equal(plainToolError('something unusual'), null);
  const text = withPlain('Error: web fetch failed: Error: getaddrinfo ENOTFOUND example.invalid');
  assert.match(text, /^That web page could not be opened/);
  assert.match(text, /Details: web fetch failed/);
});

// ------------------------------------------------------------------------------------------------- real Harness

async function rig(t) {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-tools-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 4 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const run = async (chatId, message, { mode = 'bypass', answer } = {}) => {
    const frames = [];
    await bridge.runTurn({
      chatId, message, workspacePath: ws, mode,
      onFrame: f => {
        frames.push(f);
        if (answer && f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId, callId: f.tool.id, decision: answer }).catch(e => frames.push({ type: 'test-error', message: e.message }));
      },
    });
    return frames;
  };
  const call = (chatId, tool, args, options) => run(chatId, `SCRIPT:call ${tool} ${JSON.stringify(args)}`, options);
  return { worker, bridge, dir, ws, run, call };
}
const cards = frames => frames.filter(f => f.type === 'agent-tool').map(f => f.tool);
const last = frames => cards(frames).at(-1);

test('the agent is offered exactly these tools: everything useful including web search (no key needed), no scheduler, no unanswerable question', { skip }, async (t) => {
  const r = await rig(t);
  await r.run('inv', 'SCRIPT:echo hi');
  const offered = [...r.worker.requests].reverse().find(q => q.toolNames.length)?.toolNames.sort();
  assert.deepEqual(offered, ['create_goal', 'edit', 'get_goal', 'glob', 'grep', 'interrupt_agent', 'job_kill', 'job_list', 'job_output', 'list_agents', 'list_subagent_models', 'present', 'pwsh', 'read', 'read_image', 'send_message', 'skill', 'subagent', 'subagent_fork', 'todo_write', 'update_goal', 'web_fetch', 'web_search', 'workflow', 'working_directory', 'write'].sort());
  assert.ok(!offered.some(n => n.startsWith('schedule_')) && offered.includes('web_search') && !offered.includes('ask_user_question') && !offered.includes('exit_plan_mode'));
});

test('a tool from each family runs and shows a card the page can render', { skip }, async (t) => {
  const r = await rig(t);
  const file = join(r.ws, 'notes.txt');
  writeFileSync(file, 'alpha\nneedle here\nomega\n');
  const f = fwd(file);
  // read-only family
  assert.match(last(await r.call('fam-r', 'read', { file_path: f })).result, /needle here/);
  assert.match(last(await r.call('fam-g', 'glob', { pattern: '*.txt' })).result, /notes\.txt/);
  assert.match(last(await r.call('fam-s', 'grep', { pattern: 'needle' })).result, /Line 2/);
  assert.equal(last(await r.call('fam-w', 'working_directory', {})).state, 'done');
  assert.equal(last(await r.call('fam-j', 'job_list', {})).state, 'done');
  // plan / list family
  assert.equal(last(await r.call('fam-t', 'todo_write', { todos: [{ content: 'check the build', status: 'pending' }] })).state, 'done');
  const goal = last(await r.call('fam-goal', 'create_goal', { objective: 'finish the notes' }));
  assert.equal(goal.state, 'done'); assert.equal(goal.effect, 'Sets a goal to work towards.');
  // write family (Bypass: no card asks)
  const wrote = await r.call('fam-fs', 'write', { file_path: fwd(join(r.ws, 'new.txt')), content: 'made\n' });
  assert.equal(last(wrote).state, 'done'); assert.equal(readFileSync(join(r.ws, 'new.txt'), 'utf8').trim(), 'made');
  await r.call('fam-fs', 'read', { file_path: f });
  const edited = await r.call('fam-fs', 'edit', { file_path: f, old_string: 'alpha', new_string: 'ALPHA' });
  assert.equal(last(edited).state, 'done'); assert.match(readFileSync(file, 'utf8'), /^ALPHA/);
  // shell family
  assert.match(last(await r.call('fam-sh', 'pwsh', { command: 'Get-ChildItem', description: 'list files' })).result, /notes\.txt/);
  // web family: offline, loopback is refused by the Harness itself; the failure is plain with Details after it
  const web = last(await r.call('fam-web', 'web_fetch', { url: 'http://127.0.0.1:9/' }));
  assert.equal(web.state, 'error'); assert.match(web.result, /^That web page could not be opened/); assert.match(web.result, /Details: /);
  assert.equal(web.effect, 'Opens a web page.'); assert.equal(web.preview, 'http://127.0.0.1:9/');
  // no card shows raw JSON as its explanation
  for (const c of [last(await r.call('fam-t2', 'todo_write', { todos: [{ content: 'x', status: 'pending' }] }))]) assert.ok(!c.preview.startsWith('{'));
});

test('Manual: one change is one card. Nothing ran when it appeared, and no sandbox or PowerShell error was ever shown', { skip }, async (t) => {
  const r = await rig(t);
  const a = join(r.ws, 'a.txt'), b = join(r.ws, 'b.txt'), c = join(r.ws, 'c.txt');
  // approve a file write
  let frames = await r.call('man1', 'write', { file_path: fwd(a), content: 'hello\n' }, { mode: 'manual', answer: 'approve' });
  assert.deepEqual(cards(frames).map(x => x.state), ['running', 'proposed', 'running', 'done']);
  assert.equal(new Set(cards(frames).map(x => x.id)).size, 1, 'one card, updated in place');
  assert.equal(readFileSync(a, 'utf8').trim(), 'hello');
  // deny a file write
  frames = await r.call('man2', 'write', { file_path: fwd(b), content: 'no\n' }, { mode: 'manual', answer: 'deny' });
  assert.deepEqual(cards(frames).map(x => x.state), ['running', 'proposed', 'denied', 'denied']);
  assert.equal(last(frames).result, 'Denied. Nothing ran.', 'the card says it in plain words, not the model-facing error');
  assert.equal(existsSync(b), false);
  // a shell command that changes things
  frames = await r.call('man3', 'pwsh', { command: `Set-Content -Path '${c}' -Value x`, description: 'Make c.txt' }, { mode: 'manual', answer: 'approve' });
  assert.deepEqual(cards(frames).map(x => x.state), ['running', 'proposed', 'running', 'done']);
  assert.ok(existsSync(c));
  // a shell command that only looks needs no question
  frames = await r.call('man4', 'pwsh', { command: 'Get-ChildItem', description: 'List files' }, { mode: 'manual' });
  assert.deepEqual(cards(frames).map(x => x.state), ['running', 'done']);
  // nowhere in any of it does the person see the sandbox talking
  const all = JSON.stringify([...cards(frames), ...cards(await r.call('man5', 'write', { file_path: fwd(join(r.ws, 'd.txt')), content: 'd' }, { mode: 'manual', answer: 'approve' }))]);
  assert.ok(!/sandbox|Constrained|language mode|access denied/i.test(all), all.slice(0, 300));
});

test('Plan stays read-only and Bypass asks nothing', { skip }, async (t) => {
  const r = await rig(t);
  const p = join(r.ws, 'p.txt'), q = join(r.ws, 'q.txt');
  const planned = await r.call('pl', 'write', { file_path: fwd(p), content: 'x' }, { mode: 'plan' });
  assert.equal(existsSync(p), false);
  assert.ok(!cards(planned).some(x => x.state === 'proposed'));
  const free = await r.call('bp', 'write', { file_path: fwd(q), content: 'x' }, { mode: 'bypass' });
  assert.equal(existsSync(q), true);
  assert.ok(!cards(free).some(x => x.state === 'proposed'));
});

test('Manual refuses to start when the gate is not loaded (it would be an open sandbox that nobody asks about)', { skip }, async (t) => {
  // A Harness started without the gate plugin (the bridge's `gate: false`).
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-nogate-'));
  mkdirSync(join(dir, 'ws'), { recursive: true });
  const local = [{ id: 'i', baseUrl: worker.url, modelId: 'scripted-model', context: 32768 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, gate: false, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  await assert.rejects(bridge.runTurn({ chatId: 'x', message: 'SCRIPT:echo hi', workspacePath: join(dir, 'ws'), mode: 'manual', onFrame: () => {} }), error => error.code === 'mode-unavailable' && /asks before changing/.test(error.message) && !/sandbox/i.test(error.message));
});

test('a tool server set up for the model is connected to the agent: its tools are offered, Manual asks before using them, and its secret never reaches a file', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-mcp-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const mcpServers = { demo: { command: process.execPath, args: [fwd(join(import.meta.dirname, 'helpers', 'mini-mcp.mjs'))], env: { DEMO_TOKEN: 'tok-123456-secret' } }, off: { command: 'x', disabled: true } };
  const local = [{ id: 'i', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2, mcpServers }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const frames = [];
  await bridge.runTurn({ chatId: 'mcp-bypass', message: `SCRIPT:call mcp__demo__shout ${JSON.stringify({ text: 'hello' })}`, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
  assert.ok([...worker.requests].reverse().find(q => q.toolNames.length).toolNames.includes('mcp__demo__shout'));
  assert.ok(![...worker.requests].some(q => q.toolNames.some(n => n.startsWith('mcp__off'))), 'a disabled server is not connected');
  const card = cards(frames).at(-1);
  assert.equal(card.state, 'done'); assert.match(card.result, /HELLO\|token:set/, 'the secret reached the server through the environment');
  assert.equal(card.scope, 'connected');
  // Manual: a connected tool has unknown effects, so it asks first
  const manual = [];
  await bridge.runTurn({
    chatId: 'mcp-manual', message: `SCRIPT:call mcp__demo__shout ${JSON.stringify({ text: 'ask me' })}`, workspacePath: ws, mode: 'manual',
    onFrame: f => { manual.push(f); if (f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId: 'mcp-manual', callId: f.tool.id, decision: 'approve' }); },
  });
  assert.deepEqual(cards(manual).map(c => c.state), ['running', 'proposed', 'running', 'done']);
  assert.match(cards(manual)[1].effect, /Use the connected tool demo \/ shout/);
  // the secret is in the environment of the agent only: never in the profile file
  const files = JSON.stringify(readdirDeep(join(dir, 'agent')).filter(f => f.endsWith('.yml')).map(f => readFileSync(f, 'utf8')));
  assert.ok(files.includes('process.env.STRATA_MCP_0_E_DEMO_TOKEN') && !files.includes('tok-123456-secret'));
});

function readdirDeep(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) { const p = join(dir, e.name); if (e.isDirectory()) out.push(...readdirDeep(p)); else out.push(p); }
  return out;
}

test('the output budget is not the old 8192, and the Reasoning menu reaches the request (off sends enable_thinking false)', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-think-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'i', baseUrl: worker.url, modelId: 'scripted-model', context: 65536 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const ask = async (chatId, thinking) => {
    worker.requests.length = 0;
    await bridge.runTurn({ chatId, message: 'SCRIPT:echo hi', workspacePath: ws, mode: 'bypass', thinking, onFrame: () => {} });
    return worker.requests.find(r => r.toolNames.length).params;
  };
  const ctk = p => p.chat_template_kwargs ?? {};
  const off = await ask('think-off', 'Off');
  assert.equal(ctk(off).enable_thinking, false); assert.equal(ctk(off).reasoning_effort, undefined);
  assert.equal(off.enable_thinking, undefined, 'a top-level enable_thinking is the field the Strata server ignores');
  assert.ok(off.max_completion_tokens === 19660, `budget ${off.max_completion_tokens} (half the window, at most 32K: never the old 8192, and room is left for the conversation to be compacted)`);
  const high = await ask('think-high', 'High');
  assert.equal(ctk(high).enable_thinking, true); assert.equal(ctk(high).reasoning_effort, 'high');
  const low = await ask('think-low', 'Low');
  assert.equal(ctk(low).enable_thinking, true); assert.equal(ctk(low).reasoning_effort, 'low');
  const max = await ask('think-max', 'Max');
  assert.equal(ctk(max).enable_thinking, true); assert.equal(ctk(max).reasoning_effort, 'high');
  // the same chat switching its level between turns
  assert.equal(ctk(await ask('think-off', 'High')).enable_thinking, true);
  assert.equal(ctk(await ask('think-off', 'Off')).enable_thinking, false);
});
