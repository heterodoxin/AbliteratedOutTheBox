// A message too large for the window, made small enough to send: the text as a file in the chat's folder, or a shortened preview.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { prepareOversized } from '../server/oversize.mjs';

const big = 'q=0x1f ok;'.repeat(11000);   // 110,000 characters
const engine = (tokens) => async () => ({ ok: true, json: async () => ({ input_tokens: tokens }) });
const instance = { port: 18500, key: 'k', apiModelId: 'm', context: 32768 };

test('file: the whole text is saved in the chat folder, the message names it and shows how it starts', async (t) => {
  const root = mkdtempSync(join(tmpdir(), 'strata-oversize-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const r = await prepareOversized({ mode: 'file', text: big, root, instance, fetchImpl: engine(36000), now: new Date(2026, 9, 22, 14, 5, 9) });
  assert.equal(r.mode, 'file');
  assert.equal(r.tokens, 36000);
  assert.equal(r.path, join(root, 'pasted-message-20261022-140509.txt'));
  assert.equal(readFileSync(r.path, 'utf8'), big, 'nothing is lost: the file holds the whole text');
  assert.ok(r.message.includes(r.path));
  assert.ok(r.message.length < 2000, 'the message that is sent is short');
  assert.ok(r.message.includes('q=0x1f ok;q=0x1f ok;'), 'and it shows how the text starts');
  // The same second twice never overwrites a file.
  await assert.rejects(prepareOversized({ mode: 'file', text: 'other', root, instance, fetchImpl: engine(5), now: new Date(2026, 9, 22, 14, 5, 9) }));
});

test('file: a chat with no folder says so instead of guessing one', async () => {
  await assert.rejects(prepareOversized({ mode: 'file', text: big, root: null, instance: null }), /no folder/);
});

test('preview: the start and the end, cut to what the window can hold, using the engine\'s own count when it has one', async () => {
  const r = await prepareOversized({ mode: 'preview', text: big, root: null, instance, fetchImpl: engine(36000) });
  assert.equal(r.mode, 'preview');
  assert.ok(r.message.length < big.length / 2 && r.message.length > 20000, `kept a good part (${r.message.length})`);
  assert.match(r.message, /characters from the middle are left out because the message was too long to send/);
  assert.ok(r.message.startsWith('q=0x1f ok;') && r.message.endsWith('q=0x1f ok;'));
  // Counted by the engine: 36,000 tokens for 110,000 characters is 3.05 characters a token; the kept text is about 70% of the room that is left.
  assert.ok(r.kept < (32768 - 12000 - 4096) * 3.05, 'the preview fits the window by the engine\'s own ratio');
});

test('without an engine to count, three characters a token is assumed (code and logs are denser than prose)', async () => {
  const r = await prepareOversized({ mode: 'preview', text: big, root: null, instance: null });
  assert.equal(r.tokens, null);
  assert.ok(r.message.length < big.length);
  const down = await prepareOversized({ mode: 'preview', text: big, root: null, instance, fetchImpl: async () => { throw new Error('refused'); } });
  assert.equal(down.tokens, null, 'a server that cannot be asked is not an error');
  assert.ok(down.message.length < big.length);
});

test('a short text is returned as it is', async () => {
  const r = await prepareOversized({ mode: 'preview', text: 'short', root: null, instance: null });
  assert.equal(r.message, 'short');
});

test('the host serves it at /api/chat/oversize, from the chat\'s own folder and the loaded model', () => {
  const index = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const at = index.indexOf("path === '/api/chat/oversize'");
  assert.ok(at > 0);
  const route = index.slice(at, at + 900);
  assert.match(route, /prepareOversized\(/);
  assert.match(route, /chatRoot\(String\(body\.sessionId/);
  assert.match(route, /native\.resolveInstance\(body\.nativeModelId/);
  assert.ok(!existsSync('/nonexistent'), 'sanity');
});
