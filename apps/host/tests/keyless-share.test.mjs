import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { createLocalShare, NO_KEY_WARNING } from '../server/local-share.mjs';
import { createCloudflare, NO_KEY_TUNNEL_WARNING } from '../server/cloudflare.mjs';

test('sharing on the network works without an API key and says so; with a key there is no warning', async () => {
  let key = false;
  const share = createLocalShare({ handle: async (req, res) => res.end('ok'), hasKey: () => key, host: '127.0.0.1', port: 0 });
  try {
    const started = await share.start();
    assert.equal(started.active, true);
    assert.equal(started.warning, NO_KEY_WARNING);
    key = true;
    assert.equal(share.state().warning, null);
  } finally { await share.stop(); }
});

test('a Cloudflare tunnel connects without an API key and carries the stronger warning', async () => {
  let key = false;
  const origin = { start: async () => {}, stop: async () => {} };
  const launch = () => {
    const child = new EventEmitter();
    child.stdout = new EventEmitter(); child.stderr = new EventEmitter();
    child.kill = () => { setImmediate(() => child.emit('exit')); };
    setImmediate(() => child.stderr.emit('data', 'https://demo-abc.trycloudflare.com\nRegistered tunnel connection\n'));
    return child;
  };
  const tunnel = createCloudflare({ origin, originUrl: 'http://127.0.0.1:1', hasKey: () => key, launch, locate: async () => '/bin/cloudflared', timeoutMs: 5000 });
  try {
    const first = await tunnel.start({ mode: 'quick' });
    assert.ok(['starting', 'connected'].includes(first.status));
    await new Promise(resolve => setTimeout(resolve, 50));
    const now = tunnel.state();
    assert.equal(now.status, 'connected');
    assert.equal(now.warning, NO_KEY_TUNNEL_WARNING);
    assert.match(NO_KEY_TUNNEL_WARNING, /internet/);
    key = true;
    assert.equal(tunnel.state().warning, null);
  } finally { await tunnel.stop(); }
});
