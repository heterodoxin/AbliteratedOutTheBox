/** The API key is optional (owner decision, 0.2.2): removing it while sharing or the Cloudflare tunnel runs is allowed, and the key stays removed. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const home = mkdtempSync(join(tmpdir(), 'strata-serving-key-'));
process.env.STRATA_HOME = home;
const serving = await import('../server/serving.mjs');

test('removing the API key while local sharing and the tunnel run is allowed', () => {
  serving.init({ send() {} }, { port: 8080, host: '127.0.0.1' });
  serving.patch({ generateKey: true });
  assert.equal(serving.state().access.apiKeySet, true);
  serving.setSharing({ active: true, url: 'http://192.168.1.5:8792/v1' });
  serving.setTunnel({ status: 'connected', mode: 'quick', url: 'https://example.trycloudflare.com', error: null, installed: true, originUrl: null });
  assert.equal(serving.state().access.otherDevices, true);
  const after = serving.patch({ apiKey: null });
  assert.equal(after.access.apiKeySet, false);
  assert.equal(serving.authorized({ headers: {} }), true);
});
test.after(() => { try { rmSync(home, { recursive: true, force: true }); } catch { /* temp folder */ } });
