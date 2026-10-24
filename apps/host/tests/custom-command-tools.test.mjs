// T7 (tools and images review, 2026-10-22): a custom command tool runs in the shell that suits it, in the folder it needs, and with a time
// limit the person can raise. These tests run real commands; they use only what every machine has (node).
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, realpathSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createAgentTools } from '../server/agent/tools-settings.mjs';
import { runCommand, planShell, looksLikePowerShell, resolveCwd, clampTimeoutSec, DEFAULT_TIMEOUT_SEC } from '../server/agent/custom-command-run.mjs';

const SERVER = fileURLToPath(new URL('../server/agent/custom-tools-mcp.mjs', import.meta.url));
const node = JSON.stringify(process.execPath);
const WIN = process.platform === 'win32';

test('a Windows command that reads as PowerShell goes to PowerShell, the rest to cmd.exe, and a named shell is honoured', () => {
  for (const c of ['Get-Content notes.txt', 'Start-Process notepad', 'Get-ChildItem | Where-Object { $_.Length -gt 1 }', '$env:HOME', 'echo $(Get-Date)'])
    assert.ok(looksLikePowerShell(c), c);
  for (const c of ['dir', 'type notes.txt', 'python tools\\foo.py --x', 'npm run build && echo done', 'echo %PATH%', 'git status'])
    assert.ok(!looksLikePowerShell(c), c);
  const ps = planShell('Get-Content x', { platform: 'win32' });
  assert.match(ps.file, /^(pwsh|powershell\.exe)$/);
  assert.deepEqual(ps.args.slice(0, 5), ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command']);
  assert.equal(ps.args.at(-1), 'Get-Content x');
  assert.deepEqual(planShell('dir', { platform: 'win32' }), { file: 'dir', args: [], shell: true, name: 'cmd.exe' });
  assert.equal(planShell('dir', { platform: 'win32', shell: 'powershell' }).args.at(-1), 'dir');
  assert.equal(planShell('Get-Content x', { platform: 'win32', shell: 'cmd' }).shell, true);
  assert.deepEqual(planShell('ls -la', { platform: 'linux' }), { file: 'ls -la', args: [], shell: true, name: 'sh' });
  assert.deepEqual(planShell('ls | wc -l', { platform: 'linux', shell: 'bash' }), { file: 'bash', args: ['-c', 'ls | wc -l'], shell: false, name: 'bash' });
  assert.equal(planShell('Get-Content x', { platform: 'linux' }).name, 'sh', 'on Linux and macOS a PowerShell-looking command is not rerouted');
});

test('the time allowed is 2 minutes unless the tool says more, at most an hour', () => {
  assert.equal(DEFAULT_TIMEOUT_SEC, 120);
  assert.equal(clampTimeoutSec(undefined), 120);
  assert.equal(clampTimeoutSec(600), 600);
  assert.equal(clampTimeoutSec(99999), 3600);
  assert.equal(clampTimeoutSec(0), 120);
  assert.equal(clampTimeoutSec('abc'), 120);
});

test('a command runs in the folder the call names, then the tool\'s, then the home folder; a missing folder is a plain refusal', async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), 'strata-cmd-')));
  const other = join(dir, 'other'); mkdirSync(other);
  try {
    const pwd = `${node} -e "console.log(process.cwd())"`;
    const here = await runCommand({ command: pwd, cwd: dir });
    assert.equal(realpathSync(here.text), dir);
    const named = await runCommand({ command: pwd, cwd: dir, callCwd: other });
    assert.equal(realpathSync(named.text), other, 'the folder named in the call wins');
    const home = await runCommand({ command: pwd });
    assert.equal(realpathSync(home.text), realpathSync(homedir()));
    const gone = await runCommand({ command: pwd, cwd: join(dir, 'nope') });
    assert.equal(gone.ok, false);
    assert.match(gone.text, /does not exist, so the command was not run/);
    assert.deepEqual(resolveCwd(''), { cwd: homedir() });
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a relative path in the command finds its file in the tool\'s folder', async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), 'strata-cmd-')));
  try {
    mkdirSync(join(dir, 'tools'));
    writeFileSync(join(dir, 'tools', 'foo.js'), 'console.log("from foo " + (process.env.STRATA_INPUT || ""));');
    const r = await runCommand({ command: `${node} tools/foo.js`, cwd: dir, input: 'hello' });
    assert.deepEqual([r.ok, r.text], [true, 'from foo hello']);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a command over the limit is ended and says so; output over the cap says it was cut', async () => {
  const slow = await runCommand({ command: `${node} -e "setTimeout(()=>{}, 30000)"`, timeoutSec: 1 });
  assert.equal(slow.ok, false);
  assert.match(slow.text, /stopped after 1 seconds/);
  const big = await runCommand({ command: `${node} -e "process.stdout.write('x'.repeat(50000))"` });
  assert.equal(big.ok, true);
  assert.match(big.text, /\(The output was cut after 20,000 characters\.\)$/);
  assert.ok(big.text.length < 20100);
});

test('PowerShell commands run on Windows (Get-ChildItem, $env:)', { skip: !WIN && 'PowerShell is checked on Windows' }, async () => {
  const r = await runCommand({ command: 'Write-Output ("a" + $env:STRATA_INPUT)', input: 'b' });
  assert.deepEqual([r.ok, r.text], [true, 'ab']);
  const dir = realpathSync(mkdtempSync(join(tmpdir(), 'strata-cmd-')));
  try {
    writeFileSync(join(dir, 'note.txt'), 'hi');
    const read = await runCommand({ command: 'Get-Content note.txt', cwd: dir });
    assert.deepEqual([read.ok, read.text], [true, 'hi']);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('Settings keeps a command tool\'s folder, shell and time, validates them, and an edit that does not send them keeps them', () => {
  let stored;
  const dir = mkdtempSync(join(tmpdir(), 'strata-cmdset-'));
  try {
    const tools = createAgentTools({ read: () => stored, write: v => { stored = v; }, dataDir: dir });
    tools.addCustom({ name: 'Build', about: 'Builds the project.', kind: 'command', command: 'npm run build', cwd: 'C:\\work\\app', shell: 'cmd', timeoutSec: 900 });
    let tool = tools.view().custom[0];
    assert.deepEqual([tool.cwd, tool.shell, tool.timeoutSec], ['C:\\work\\app', 'cmd', 900]);
    tools.editCustom(tool.id, { name: 'Build', about: 'Builds the project, quickly.', kind: 'command', command: 'npm run build' });   // the form sends none of the three
    tool = tools.view().custom[0];
    assert.deepEqual([tool.about, tool.cwd, tool.shell, tool.timeoutSec], ['Builds the project, quickly.', 'C:\\work\\app', 'cmd', 900]);
    assert.throws(() => tools.editCustom(tool.id, { name: 'Build', about: 'x', kind: 'command', command: 'npm run build', timeoutSec: 99999 }), /1 to 3600/);
    assert.throws(() => tools.editCustom(tool.id, { name: 'Build', about: 'x', kind: 'command', command: 'npm run build', shell: 'fish' }), /Choose how the command runs/);
    tools.addCustom({ name: 'Plain', about: 'Plain.', kind: 'command', command: 'echo hi' });
    const plain = tools.view().custom[1];
    assert.deepEqual([plain.cwd, plain.shell, plain.timeoutSec], ['', 'auto', 120]);
    // the tool server's file carries them
    const file = JSON.parse(require_read(tools.filePath));
    assert.deepEqual(file.commands.build, { description: 'Builds the project, quickly.', command: 'npm run build', cwd: 'C:\\work\\app', shell: 'cmd', timeoutSec: 900 });
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

import { readFileSync } from 'node:fs';
const require_read = file => readFileSync(file, 'utf8');

test('the tool server over stdio: the call may name a folder, and the tool\'s own settings apply', async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), 'strata-cmdsrv-')));
  const work = join(dir, 'work'); mkdirSync(work);
  const file = join(dir, 'tools.json');
  writeFileSync(file, JSON.stringify({ commands: {
    where: { description: 'Prints its folder.', command: `${node} -e "console.log(process.cwd())"` },
    pinned: { description: 'Runs in work.', command: `${node} -e "console.log(process.cwd())"`, cwd: work },
  } }));
  const child = spawn(process.execPath, [SERVER, 'commands', file], { stdio: ['pipe', 'pipe', 'inherit'] });
  const lines = []; let buffer = '';
  child.stdout.on('data', d => { buffer += d; let i; while ((i = buffer.indexOf('\n')) >= 0) { lines.push(JSON.parse(buffer.slice(0, i))); buffer = buffer.slice(i + 1); } });
  const ask = async (id, method, params) => {
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
    for (let n = 0; n < 200; n++) { const hit = lines.find(l => l.id === id); if (hit) return hit.result; await new Promise(r => setTimeout(r, 25)); }
    throw new Error('no answer');
  };
  try {
    const listed = await ask(1, 'tools/list', {});
    assert.ok(listed.tools.every(t => t.inputSchema.properties.cwd?.type === 'string'), 'the optional folder is offered');
    const pinned = await ask(2, 'tools/call', { name: 'pinned', arguments: {} });
    assert.equal(realpathSync(pinned.content[0].text), work);
    const named = await ask(3, 'tools/call', { name: 'where', arguments: { cwd: dir } });
    assert.equal(realpathSync(named.content[0].text), dir);
    const missing = await ask(4, 'tools/call', { name: 'where', arguments: { cwd: join(dir, 'nope') } });
    assert.equal(missing.isError, true);
    assert.match(missing.content[0].text, /does not exist/);
  } finally { child.kill(); rmSync(dir, { recursive: true, force: true }); }
});
