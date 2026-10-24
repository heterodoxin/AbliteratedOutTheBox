// Connect an app: Anthropic-style apps get the server root (they add /v1/messages themselves), OpenAI-style apps keep /v1,
// and the host accepts /v1/v1/messages as the same request for people who pasted the old address.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import http from 'node:http';
import { build } from 'esbuild';
import { apiPath } from '../server/api-paths.mjs';
import { createLocalShare } from '../server/local-share.mjs';

const bundled = await build({ entryPoints: ['src/lib/connect-urls.ts'], bundle: true, write: false, format: 'esm', platform: 'node' });
const { serverRoot, appBaseUrl } = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);

test('the Anthropic tab shows the address without /v1, the other tabs keep it', () => {
  const base = 'http://127.0.0.1:8792/v1';
  assert.equal(appBaseUrl(base, 'anthropic'), 'http://127.0.0.1:8792');
  assert.equal(appBaseUrl(base, 'openai'), base);
  assert.equal(appBaseUrl(base, 'responses'), base);
  assert.equal(serverRoot('https://abc.trycloudflare.com/v1/'), 'https://abc.trycloudflare.com');
  assert.equal(serverRoot('http://192.168.1.5:8792'), 'http://192.168.1.5:8792');
  assert.equal(serverRoot('http://host/api/v1'), 'http://host/api');
});

test('the Connect dialog gives the Anthropic tab the root address and says why in plain words', async () => {
  const screen = await readFile('src/screens/Serving.tsx', 'utf8');
  assert.match(screen, /appBaseUrl\(serving\.baseUrl, protocol\)/);
  assert.match(screen, /ANTHROPIC_BASE_URL/);
});

test('/v1/v1/messages is read as /v1/messages, and nothing else is aliased', () => {
  assert.equal(apiPath('/v1/v1/messages'), '/v1/messages');
  assert.equal(apiPath('/v1/v1/messages/'), '/v1/messages');
  assert.equal(apiPath('/v1/v1/messages/count_tokens'), '/v1/messages/count_tokens');
  assert.equal(apiPath('/v1/messages'), '/v1/messages');
  for (const other of ['/v1/v1/chat/completions', '/v1/v1/models', '/v1/v1/responses', '/v1/v1/v1/messages']) assert.equal(apiPath(other), other);
});

test('the shared (key-protected) listener serves /v1/v1/messages as /v1/messages', async () => {
  const seen = [];
  const share = createLocalShare({ handle: async (req, res, path) => { seen.push(path); res.writeHead(200); res.end('ok'); }, hasKey: () => true, host: '127.0.0.1', port: 0 });
  const { url } = await share.start();
  try {
    const root = url.replace(/\/v1$/, '');
    for (const path of ['/v1/messages', '/v1/v1/messages']) {
      const res = await fetch(root + path, { method: 'POST', body: '{}' });
      assert.equal(res.status, 200, path);
    }
    assert.deepEqual(seen, ['/v1/messages', '/v1/messages']);
    assert.equal((await fetch(root + '/v1/v1/chat/completions', { method: 'POST', body: '{}' })).status, 404);
  } finally { await share.stop(); }
});
