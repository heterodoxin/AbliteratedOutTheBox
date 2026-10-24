import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { transform } from 'esbuild';
import { createCloudflare, tunnelHostname } from '../server/cloudflare.mjs';
import { saveSnapshot, snapshotBytes } from '../server/snapshot.mjs';

function fixture(extra = {}) {
  let stopped = 0, launched;
  const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter();
  child.kill = () => { queueMicrotask(() => child.emit('exit', 0)); };
  const connector = createCloudflare({ origin: { start: async () => {}, stop: async () => { stopped++; } }, originUrl: 'http://127.0.0.1:8793', hasKey: () => true, locate: async () => '/test/cloudflared', launch: (...args) => { launched = args; return child; }, ...extra });
  return { connector, child, launched: () => launched, stopped: () => stopped };
}
test('quick tunnel publishes its URL only after registration, and stops its private origin', async () => {
  const f = fixture();
  await f.connector.start({ mode: 'quick' });
  f.child.stderr.emit('data', 'URL https://quiet-field.trycloud');
  f.child.stderr.emit('data', 'flare.com\n');
  assert.equal(f.connector.state().status, 'starting');
  assert.equal(f.connector.state().url, null);
  f.child.stderr.emit('data', 'Registered tunnel connection connIndex=0\n');
  assert.equal(f.connector.state().url, 'https://quiet-field.trycloudflare.com/v1');
  assert.equal(f.connector.state().status, 'connected');
  await assert.rejects(f.connector.start({ mode: 'quick' }), /Stop/);
  await f.connector.stop();
  assert.equal(f.connector.state().status, 'idle'); assert.equal(f.stopped(), 1);
});
test('domain tunnel validates hostnames and keeps the token out of arguments and state', async () => {
  for (const value of ['https://ai.example.com', 'example.com/path', 'localhost', 'ai.example.com;echo x']) assert.throws(() => tunnelHostname(value));
  const f = fixture(), token = 'test-only-tunnel-token-12345';
  await f.connector.start({ mode: 'domain', hostname: 'ai.example.com', token });
  assert.equal(f.launched()[2].env.TUNNEL_TOKEN, token);
  assert.ok(!JSON.stringify(f.launched()[1]).includes(token));
  assert.ok(!JSON.stringify(f.connector.state()).includes(token));
  f.child.stderr.emit('data', 'Registered tunnel connection\n');
  assert.equal(f.connector.state().url, 'https://ai.example.com/v1');
  await f.connector.stop();
});
test('missing prerequisites and connector failure leave no inference listener running', async () => {
  const noKey = fixture({ hasKey: () => false });          // without a key the tunnel starts (owner decision, 0.2.2) and the state carries a warning
  await noKey.connector.start({ mode: 'quick' });
  assert.notEqual(noKey.launched(), undefined);
  const missing = fixture({ locate: async () => null });
  await assert.rejects(missing.connector.start({ mode: 'quick' }), /connector could not be prepared automatically/i);
  assert.equal(missing.connector.state().status, 'error');
  const failed = fixture(); await failed.connector.start({ mode: 'quick' });
  failed.child.emit('exit', 1); await new Promise(resolve => setImmediate(resolve));
  assert.equal(failed.connector.state().status, 'error'); assert.equal(failed.stopped(), 1);
});
test('snapshot validates PNGs, writes only a dialog-selected path, and respects Cancel', async () => {
  const png = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aQ1sAAAAASUVORK5CYII=';
  assert.throws(() => snapshotBytes('data:image/png;base64,ZXZpbA=='));
  let writes = [];
  const saved = await saveSnapshot(png, { platform: 'darwin', run: async () => ({ stdout: '/tmp/user-selected.png\n' }), write: async (...args) => { writes.push(args); } });
  assert.equal(saved.saved, true); assert.equal(writes[0][0], '/tmp/user-selected.png'); assert.deepEqual(writes[0][1], snapshotBytes(png));
  const cancelled = await saveSnapshot(png, { platform: 'darwin', run: async () => { throw { stderr: 'User canceled. (-128)' }; }, write: async () => { throw new Error('Must not write'); } });
  assert.equal(cancelled.cancelled, true);
});
test('shortcut overrides persist, clear, restore, and change the composer send combination', async () => {
  const memory = new Map(); globalThis.localStorage = { getItem: k => memory.get(k), setItem: (k,v) => memory.set(k,v), removeItem: k => memory.delete(k) };
  const require = createRequire(import.meta.url);
  const source = (await readFile(new URL('../src/lib/shortcut-bindings.ts', import.meta.url), 'utf8')).replace("from 'react'", `from '${pathToFileURL(require.resolve('react')).href}'`);
  const { code } = await transform(source, { loader: 'ts', format: 'esm' });
  const m = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  const enter = { key: 'Enter', ctrlKey: false, metaKey: false, altKey: false, shiftKey: false };
  assert.equal(m.composerSends(enter), true);
  m.setBinding('chat:send', { key: 'enter', mod: true, shift: false, alt: false });
  assert.equal(m.composerSends(enter), false); assert.equal(m.composerSends({ ...enter, metaKey: true }), true);
  assert.ok(memory.get('strata.shortcuts.v1').includes('chat:send'));
  m.setBinding('new', null); assert.equal(m.bindingFor('new', { key: 'n' }), null);
  m.resetBindings(); assert.equal(m.composerSends(enter), true); assert.equal(memory.size, 0);
  delete globalThis.localStorage;
});
