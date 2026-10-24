import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chooseFolder, cleanChosenPath, typedFolder, windowsFolderScript } from '../server/folder-dialog.mjs';

const HOST_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const failure = (code) => Object.assign(new Error(`exit ${code}`), { code });
const recorder = () => {
  const seen = [];
  return { seen, run: (outcome) => async (command, args) => { seen.push({ command, args }); if (outcome instanceof Error) throw outcome; return { stdout: outcome }; } };
};

test('macOS keeps the osascript chooser and treats any dismissal as a cancel', async () => {
  const c = recorder();
  assert.deepEqual(await chooseFolder('Pick', { platform: 'darwin', run: c.run('/Users/me/skills/\n') }), { path: '/Users/me/skills' });
  assert.equal(c.seen[0].command, 'osascript');
  assert.equal(c.seen[0].args[1], 'POSIX path of (choose folder with prompt "Pick")');
  assert.deepEqual(await chooseFolder('Pick', { platform: 'darwin', run: c.run(failure(1)) }), { cancelled: true });
});

test('Windows uses a PowerShell FolderBrowserDialog passed as an encoded command', async () => {
  const c = recorder();
  const result = await chooseFolder("Don't panic", { platform: 'win32', run: c.run('C:\\Users\\me\\Skills\r\n') });
  assert.deepEqual(result, { path: 'C:\\Users\\me\\Skills' });
  assert.match(c.seen[0].command, /powershell\.exe$/i);
  assert.ok(c.seen[0].args.includes('-STA'));
  const script = Buffer.from(c.seen[0].args.at(-1), 'base64').toString('utf16le');
  assert.equal(script, windowsFolderScript("Don't panic"));
  assert.match(script, /FolderBrowserDialog/);
  assert.match(script, /Description = 'Don''t panic'/);
  assert.deepEqual(await chooseFolder('Pick', { platform: 'win32', run: c.run('') }), { cancelled: true });
  const missing = await chooseFolder('Pick', { platform: 'win32', run: c.run(failure('ENOENT')) });
  assert.equal(missing.unavailable, true);
  await assert.rejects(chooseFolder('Pick', { platform: 'win32', run: c.run(failure(2)) }), /could not open/);
});

test('the generated PowerShell parses without errors', { skip: process.platform !== 'win32' }, () => {
  const shell = join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
  const out = execFileSync(shell,
    ['-NoProfile', '-NonInteractive', '-Command', '$e=$null; [void][System.Management.Automation.Language.Parser]::ParseInput($env:SCRIPT,[ref]$null,[ref]$e); $e.Count'],
    { encoding: 'utf8', env: { ...process.env, SCRIPT: windowsFolderScript('Choose a folder for Strata') }, windowsHide: true }).trim();
  assert.equal(out, '0');
});

test('Linux tries zenity, falls back to kdialog, then reports that none exists', async () => {
  const order = [];
  const run = async (command) => { order.push(command); if (command === 'zenity') throw failure('ENOENT'); return { stdout: '/home/me/skills/\n' }; };
  assert.deepEqual(await chooseFolder('Pick', { platform: 'linux', run }), { path: '/home/me/skills' });
  assert.deepEqual(order, ['zenity', 'kdialog']);
  const none = await chooseFolder('Pick', { platform: 'linux', run: async () => { throw failure('ENOENT'); } });
  assert.equal(none.unavailable, true);
  assert.match(none.reason, /zenity or kdialog/);
  assert.deepEqual(await chooseFolder('Pick', { platform: 'linux', run: async () => { throw failure(1); } }), { cancelled: true });
  assert.deepEqual(await chooseFolder('Pick', { platform: 'linux', run: async () => ({ stdout: '' }) }), { cancelled: true });
});

test('chosen paths lose trailing separators but roots survive', () => {
  assert.equal(cleanChosenPath('/a/b/\n'), '/a/b');
  assert.equal(cleanChosenPath('/'), '/');
  assert.equal(cleanChosenPath('C:\\Users\\me\\'), 'C:\\Users\\me');
  assert.equal(cleanChosenPath('C:\\'), 'C:\\');
});

test('a typed path must be an absolute existing directory', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'strata-typed-'));
  try {
    await writeFile(join(dir, 'file.txt'), 'x');
    assert.equal(await typedFolder(dir, { isAbsolute }), dir);
    assert.equal(await typedFolder(undefined, { isAbsolute }), null);
    await assert.rejects(typedFolder('relative/dir', { isAbsolute }), /full path/);
    await assert.rejects(typedFolder(join(dir, 'missing'), { isAbsolute }), /does not exist/);
    await assert.rejects(typedFolder(join(dir, 'file.txt'), { isAbsolute }), /not a file/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('the model page takes a folder or one .gguf file, with or without quotes around the path', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'strata-gguf-typed-'));
  try {
    await writeFile(join(dir, 'model-00001-of-00002.gguf'), 'x');
    await writeFile(join(dir, 'notes.txt'), 'x');
    const only = { isAbsolute, fileType: /\.gguf$/i };
    assert.equal(await typedFolder(dir, only), dir);
    assert.equal(await typedFolder(join(dir, 'model-00001-of-00002.gguf'), only), join(dir, 'model-00001-of-00002.gguf'));
    assert.equal(await typedFolder(`"${join(dir, 'model-00001-of-00002.gguf')}"`, only), join(dir, 'model-00001-of-00002.gguf'));
    assert.equal(await typedFolder(`  "${dir}"  `, only), dir);
    await assert.rejects(typedFolder(join(dir, 'notes.txt'), only), /folder, or a \.gguf file/);
    await assert.rejects(typedFolder(join(dir, 'missing.gguf'), only), /does not exist/);
    // the other pickers still refuse a file
    await assert.rejects(typedFolder(join(dir, 'model-00001-of-00002.gguf'), { isAbsolute }), /not a file/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

async function startHost() {
  const home = await mkdtemp(join(tmpdir(), 'strata-folder-home-'));
  const net = await import('node:net');
  const probe = net.createServer();
  await new Promise(r => probe.listen(0, '127.0.0.1', r));
  const { port } = probe.address();
  await new Promise(r => probe.close(r));
  const child = spawn(process.execPath, [join(HOST_ROOT, 'server', 'index.mjs'), '--dist', join(HOST_ROOT, 'dist')], {
    cwd: HOST_ROOT, env: { ...process.env, STRATA_HOME: home, STRATA_WORKSPACE: home, STRATA_PORT: String(port), STRATA_REAL_ENGINE: '0' }, stdio: 'ignore',
  });
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 150; i++) {
    try { if ((await fetch(`${base}/api/health`)).ok) return { child, base, home }; } catch { /* not up yet */ }
    await new Promise(r => setTimeout(r, 100));
  }
  child.kill();
  throw new Error('the host did not become healthy');
}

test('the host accepts a typed folder path for the picker and for adding a skill, on every OS', { timeout: 60000 }, async () => {
  const { child, base, home } = await startHost();
  const skill = await mkdtemp(join(tmpdir(), 'strata-skill-'));
  try {
    const post = (path, body) => fetch(`${base}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    let r = await post('/api/workspaces/pick', { path: skill });
    assert.deepEqual(await r.json(), { path: skill });
    r = await post('/api/workspaces/pick', { path: join(skill, 'nope') });
    assert.equal(r.status, 400);
    assert.match((await r.json()).error, /does not exist/);
    r = await post('/api/skills/add', { path: skill });
    assert.equal(r.status, 400);
    assert.match((await r.json()).error, /no SKILL\.md/);
    await writeFile(join(skill, 'SKILL.md'), '---\nname: Typed\n---\nbody\n');
    r = await post('/api/skills/add', { path: skill });
    assert.equal(r.status, 200);
    assert.ok((await r.json()).added);
  } finally {
    child.kill();
    await rm(skill, { recursive: true, force: true });
    await rm(home, { recursive: true, force: true }).catch(() => {});
  }
});

// QA report F06: a Windows folder window that never opens must not leave the button disabled forever.
import { EventEmitter } from 'node:events';
import { watchedWindowsDialog } from '../server/folder-dialog.mjs';
function fakeShell() {
  const child = new EventEmitter(); child.stdout = new EventEmitter(); child.killed = false; child.kill = () => { child.killed = true; };
  return { child, spawnProcess: () => child };
}
test('Windows: no READY from the dialog script means the host gives up and the app asks for the path', async () => {
  const f = fakeShell();
  const result = await watchedWindowsDialog('powershell.exe', [], { readyMs: 30, spawnProcess: f.spawnProcess });
  assert.deepEqual(result, { noDialog: true }); assert.equal(f.child.killed, true, 'the stuck PowerShell is ended');
  const answer = await chooseFolder('Pick', { platform: 'win32', windowsDialog: async () => ({ noDialog: true }) });
  assert.equal(answer.unavailable, true); assert.match(answer.reason, /Type the folder path/);
});
test('Windows: once READY has come the person may take their time, and the answer is read without the READY line', async () => {
  const f = fakeShell();
  const pending = watchedWindowsDialog('powershell.exe', [], { readyMs: 30, spawnProcess: f.spawnProcess });
  f.child.stdout.emit('data', 'READY\r\n');
  await new Promise(r => setTimeout(r, 80));   // longer than readyMs: still waiting, not given up
  assert.equal(f.child.killed, false);
  f.child.stdout.emit('data', 'C:\Users\me\Skills'); f.child.emit('close', 0);
  const done = await pending;
  const answer = await chooseFolder('Pick', { platform: 'win32', windowsDialog: async () => done });
  assert.deepEqual(answer, { path: 'C:\Users\me\Skills' });
  const cancelled = await chooseFolder('Pick', { platform: 'win32', windowsDialog: async () => ({ stdout: 'READY\r\n' }) });
  assert.deepEqual(cancelled, { cancelled: true });
});
test('Windows: a PowerShell that is missing or fails is reported as before', async () => {
  const f = fakeShell();
  const missing = watchedWindowsDialog('powershell.exe', [], { spawnProcess: f.spawnProcess });
  f.child.emit('error', Object.assign(new Error('x'), { code: 'ENOENT' }));
  await assert.rejects(missing, e => e.code === 'ENOENT');
  const g = fakeShell(); const failed = watchedWindowsDialog('powershell.exe', [], { spawnProcess: g.spawnProcess });
  g.child.emit('close', 1); await assert.rejects(failed, e => e.code === 1);
});
