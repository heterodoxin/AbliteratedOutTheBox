// The chat's own folder and opening its files by path.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, symlinkSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createChatFolders, openFile, openCommand, resolveChatFile, refusedFolder } from '../server/chat-files.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
const BS = String.fromCharCode(92);
const HARNESS =process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('a chat folder is remembered per chat, must exist, and is never a system folder or a whole drive', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-cf-'));
  try {
    const mine = join(dir, 'pagoda'); mkdirSync(mine);
    const file = join(dir, 'folders.json');
    const a = createChatFolders({ file });
    assert.equal(a.get('c1'), null);
    assert.equal(a.set('c1', mine).name, 'pagoda');
    assert.equal(createChatFolders({ file }).get('c1'), mine, 'it survives a restart');
    assert.equal(createChatFolders({ file }).get('c2'), null, 'per chat');
    assert.throws(() => a.set('c1', join(dir, 'nope')), /doesn't exist/);
    writeFileSync(join(dir, 'f.txt'), 'x'); assert.throws(() => a.set('c1', join(dir, 'f.txt')), /not a folder/);
    assert.throws(() => a.set('c1', ''), /Choose a folder/);
    assert.throws(() => a.set('c1', process.platform === 'win32' ? 'C:' + BS : '/'), /whole drive/);
    assert.ok(refusedFolder(process.platform === 'win32' ? 'C:' + BS + 'Windows' + BS + 'System32' : '/etc'));
    rmSync(mine, { recursive: true }); assert.equal(a.get('c1'), null, 'a deleted folder falls back to the default'); assert.equal(a.info('c1').exists, false);
    a.clear('c1'); assert.equal(a.info('c1'), null);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('files open by path from inside the folder only; the browser gets a file:// address', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-cf2-'));
  try {
    const root = join(dir, 'proj'); mkdirSync(root); writeFileSync(join(root, 'pagoda.html'), '<p>hi</p>'); writeFileSync(join(dir, 'secret.txt'), 's');
    const rec = [];
    const r = openFile(root, 'pagoda.html', 'browser', { recorder: rec, platform: 'win32' });
    assert.match(r.url, /^file:\/\/\/.*pagoda\.html$/); assert.deepEqual(rec[0].slice(0, 2), ['rundll32.exe', 'url.dll,FileProtocolHandler']); assert.match(rec[0][2], /^file:\/\//);
    assert.ok(!/127\.0\.0\.1|localhost/.test(rec[0].join(' ')));
    const WIN = 'C:' + BS + 'a' + BS + 'b.html';
    assert.deepEqual(openCommand('folder', WIN, 'win32'), ['explorer.exe', ['/select,' + WIN]]);
    assert.equal(openCommand('folder', '/a/b.html', 'darwin')[1][0], '-R');
    assert.equal(rec.length, 1);
    assert.throws(() => openFile(root, '../secret.txt', 'app', { recorder: rec }), /outside this chat's folder/);
    assert.throws(() => openFile(root, join(dir, 'secret.txt'), 'app', { recorder: rec }), /outside/);
    assert.throws(() => openFile(root, 'missing.html', 'app', { recorder: rec }), /no longer there/);
    assert.throws(() => openFile(null, 'pagoda.html', 'app', { recorder: rec }), /no folder/);
    assert.throws(() => openFile(root, 'pagoda.html', 'weird', { recorder: rec }), /not known/);
    try { symlinkSync(join(dir, 'secret.txt'), join(root, 'link.txt')); assert.throws(() => resolveChatFile(root, 'link.txt'), /outside/, 'a link out of the folder is followed and refused'); } catch { /* no symlink right on this system */ }
    assert.equal(rec.length, 1, 'nothing was started for a refused request');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a chat that moves to another folder gets a session there: the agent writes into the new folder', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-cf3-')); const a = join(dir, 'a'), b = join(dir, 'b'); mkdirSync(a); mkdirSync(b);
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }], connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const fwd = p => p.replaceAll(BS, '/');
  const run = (ws, name) => bridge.runTurn({ chatId: 'one', message: `SCRIPT:call write ${JSON.stringify({ file_path: fwd(join(ws, name)), content: name })}`, workspacePath: ws, mode: 'manual', onFrame: f => { if (f.type === 'agent-tool' && f.tool.state === 'proposed') void bridge.approve({ chatId: 'one', callId: f.tool.id, decision: 'approve' }); } });
  await run(a, 'in-a.txt');
  const first = bridge.sessionOf('one');
  await run(b, 'in-b.txt');
  assert.notEqual(bridge.sessionOf('one'), first, 'a new session for the new folder');
  assert.ok(existsSync(join(a, 'in-a.txt')) && existsSync(join(b, 'in-b.txt')));
  // the sandbox follows the folder: Plan mode shows the old folder is not the working folder any more
  await run(b, 'again.txt');
  assert.equal(readFileSync(join(b, 'again.txt'), 'utf8').trim(), 'again.txt');
});

import { chatRootOf, previewFile } from '../server/chat-files.mjs';
test('a chat with no chosen folder still has a folder: the one its agent worked in, else the default workspace; files there open, preview and reveal', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-root-'));
  try {
    const def = join(dir, 'Documents', 'Strata', 'Workspace'); mkdirSync(def, { recursive: true });
    const chosen = join(dir, 'mine'); mkdirSync(chosen);
    const used = join(dir, 'other'); mkdirSync(used);
    assert.equal(chatRootOf({ chosen, used, fallback: def }), chosen, 'the chosen folder wins');
    assert.equal(chatRootOf({ chosen: null, used, fallback: def }), used, 'the folder the agent really wrote to');
    assert.equal(chatRootOf({ chosen: null, used: join(dir, 'gone'), fallback: def }), def, 'a folder that is gone is skipped');
    const root = chatRootOf({ chosen: null, used: null, fallback: def });
    assert.equal(root, def, 'no chosen folder: the default workspace');
    writeFileSync(join(def, 'pagoda.html'), '<h1>hi</h1>');
    assert.equal(previewFile(root, 'pagoda.html', 'c1').kind, 'html');
    const rec = [];
    const r = openFile(root, join(def, 'pagoda.html'), 'folder', { platform: 'darwin', recorder: rec });
    assert.equal(r.ok, true); assert.deepEqual(rec[0].slice(0, 2), ['open', '-R']);
    assert.equal(openFile(root, 'pagoda.html', 'app', { platform: 'linux', recorder: rec }).ok, true);
    assert.equal(openFile(root, 'pagoda.html', 'app', { platform: 'win32', recorder: rec }).ok, true);
    assert.throws(() => resolveChatFile(root, join(dir, 'secret.txt')), /no longer there|outside/, 'the default folder is still a boundary');
    assert.equal(chatRootOf({}), null);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
