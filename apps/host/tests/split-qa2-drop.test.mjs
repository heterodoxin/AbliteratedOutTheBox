// Files dropped on the app window on Linux and macOS: the shell hands the paths to the host, the page fetches each file once.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile, readFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { transform } from 'esbuild';
import { createNativeDrops, typeOfName, MAX_FILES } from '../server/native-drop.mjs';

const root = resolve(import.meta.dirname, '..');
async function loadTs(path) {
  const { code } = await transform(await readFile(new URL(path, import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}
const { readDroppedRefs, fetchDropped, dragHasFiles } = await loadTs('../src/lib/native-drop.ts');

test('the host keeps only regular files it can read, each for one fetch, for a short time', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'strata-drop-'));
  try {
    await writeFile(join(dir, 'a.PNG'), Buffer.from([1, 2, 3])); await writeFile(join(dir, 'notes.md'), '# hi'); await writeFile(join(dir, 'thing.bin'), 'x');
    await mkdir(join(dir, 'folder'));
    let clock = 1000;
    const drops = createNativeDrops({ now: () => clock });
    const got = await drops.add([join(dir, 'a.PNG'), join(dir, 'folder'), join(dir, 'missing.txt'), 'relative/path.txt', 42, join(dir, 'notes.md'), join(dir, 'thing.bin')]);
    assert.deepEqual(got.map(f => [f.name, f.type, f.size]), [['a.PNG', 'image/png', 3], ['notes.md', 'text/markdown', 4], ['thing.bin', 'application/octet-stream', 1]]);
    assert.match(got[0].id, /^[0-9a-f-]{36}$/);
    assert.deepEqual([...drops.take(got[0].id).bytes], [1, 2, 3]);
    assert.equal(drops.take(got[0].id), null, 'one fetch per file');
    clock += 3 * 60_000;
    assert.equal(drops.take(got[1].id), null, 'gone after a couple of minutes');
    assert.deepEqual(await drops.add('not a list'), []);
    const many = await drops.add(Array.from({ length: 30 }, () => join(dir, 'notes.md')));
    assert.equal(many.length, MAX_FILES);
    const big = createNativeDrops({ info: async () => ({ isFile: () => true, size: 26 * 1024 * 1024 }) });
    assert.deepEqual(await big.add([join(dir, 'notes.md')]), [], 'a file over 25 MB is left out');
    assert.equal(typeOfName('x.JPG'), 'image/jpeg');
  } finally { await rm(dir, { recursive: true, force: true, maxRetries: 3 }); }
});

test('only the app shell can hand files over; the page fetches each file once', { timeout: 60000 }, async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-drop-host-'));
  const socket = createServer(); await new Promise(r => socket.listen(0, '127.0.0.1', r));
  const port = socket.address().port; await new Promise(r => socket.close(r));
  const secret = 'a'.repeat(40);
  const child = spawn(process.execPath, ['server/index.mjs'], { cwd: root, env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_REAL_ENGINE: '0', STRATA_WORKSPACE: join(home, 'work'), STRATA_SHUTDOWN_TOKEN: secret }, stdio: 'ignore', windowsHide: true });
  const base = `http://127.0.0.1:${port}`;
  try {
    for (let i = 0; i < 100; i++) { try { if ((await fetch(base + '/api/skills')).ok) break; } catch { /* not up */ } await new Promise(r => setTimeout(r, 100)); }
    const picture = join(home, 'qa.png'); await writeFile(picture, Buffer.from([137, 80, 78, 71]));
    const post = (headers) => fetch(base + '/api/app/drop', { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify({ paths: [picture] }) });
    assert.equal((await post({})).status, 403, 'a page cannot hand over paths');
    assert.equal((await post({ 'x-strata-shutdown': 'b'.repeat(40) })).status, 403);
    const ok = await post({ 'x-strata-shutdown': secret });
    assert.equal(ok.status, 200);
    const [file] = (await ok.json()).files;
    assert.deepEqual([file.name, file.type, file.size], ['qa.png', 'image/png', 4]);
    const first = await fetch(`${base}/api/app/drop/${file.id}`);
    assert.equal(first.status, 200); assert.equal(first.headers.get('content-type'), 'image/png');
    assert.deepEqual([...new Uint8Array(await first.arrayBuffer())], [137, 80, 78, 71]);
    assert.equal((await fetch(`${base}/api/app/drop/${file.id}`)).status, 404, 'once');
    assert.equal((await fetch(`${base}/api/app/drop/not-an-id`)).status, 404);
  } finally { child.kill(); await new Promise(r => setTimeout(r, 300)); await rm(home, { recursive: true, force: true, maxRetries: 3 }).catch(() => {}); }
});

test('the page reads what the shell announced and makes Files of it', async () => {
  const id = '123e4567-e89b-12d3-a456-426614174000';
  assert.deepEqual(readDroppedRefs([{ id, name: 'a.png', type: 'image/png', size: 3 }, { id: 'bad', name: 'x' }, { id, name: '' }, null, 7]), [{ id, name: 'a.png', type: 'image/png' }]);
  assert.deepEqual(readDroppedRefs('nope'), []);
  const files = await fetchDropped([{ id, name: 'a.png', type: 'image/png' }, { id: id.replace('0', '1'), name: 'gone.png', type: 'image/png' }],
    async url => url.endsWith(id) ? { ok: true, blob: async () => new Blob([new Uint8Array([1, 2])]) } : { ok: false, blob: async () => new Blob([]) });
  assert.deepEqual(files.map(f => [f.name, f.type, f.size]), [['a.png', 'image/png', 2]]);
  assert.equal(dragHasFiles(['text/uri-list', 'text/html']), true); assert.equal(dragHasFiles(['Files']), true); assert.equal(dragHasFiles(['text/plain']), false); assert.equal(dragHasFiles(null), false);
});

test('the window passes the drop on (Linux and macOS) and ignores a file dropped outside the chat', async () => {
  const rust = await readFile(new URL('../../desktop/src-tauri/src/main.rs', import.meta.url), 'utf8');
  assert.match(rust, /#\[cfg\(windows\)\]\s+let builder = builder\.disable_drag_drop_handler\(\);/);
  assert.match(rust, /window\.on_webview_event\(/); assert.match(rust, /\/api\/app\/drop/);
  const zone = await readFile(new URL('../src/components/DropZone.tsx', import.meta.url), 'utf8');
  assert.match(zone, /NATIVE_DROP_EVENT/); assert.match(zone, /export function installDropGuard/);
  assert.match(await readFile(new URL('../src/main.tsx', import.meta.url), 'utf8'), /installDropGuard\(\);/);
});
