/** #3604 / #3603: the chat list is saved at most once a second, changed chats only, and capped; a streamed answer is a handful of writes. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

async function bundle(entry, plugins = []) {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins,
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

const persist = await bundle('../src/lib/session-persist.ts');
const { createSessionWriter, serializeSession, MAX_SESSION_CHARS } = persist;

test('2,000 schedules inside a second produce one write, and the newest state wins', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const writes = [];
  const writer = createSessionWriter({ write: (json) => writes.push(JSON.parse(json)), delayMs: 1_000 });
  for (let i = 0; i < 2_000; i++) writer.schedule([{ id: 's1', seq: i, messages: [] }]);
  assert.equal(writes.length, 0, 'nothing is written synchronously');
  t.mock.timers.tick(1_000);
  assert.equal(writes.length, 1);
  assert.equal(writes[0][0].seq, 1_999);
  writer.schedule([{ id: 's1', seq: 2_000, messages: [] }]);
  writer.flush();
  assert.equal(writes.length, 2);
  assert.equal(writes[1][0].seq, 2_000);
  writer.flush();
  assert.equal(writes.length, 2, 'a flush with nothing pending writes nothing');
});

test('only the chat that changed is serialised again', () => {
  const seen = [];
  const writes = [];
  const writer = createSessionWriter({ write: (j) => writes.push(j), onSerialize: (s) => seen.push(s.id) });
  const a = { id: 'a', updatedAt: '1', messages: [{ id: 'm', content: 'x'.repeat(1000) }] };
  const b = { id: 'b', updatedAt: '2', messages: [] };
  writer.schedule([a, b]); writer.flush();
  assert.deepEqual(seen.sort(), ['a', 'b']);
  seen.length = 0;
  for (let i = 0; i < 50; i++) { writer.schedule([a, { ...b, title: `t${i}` }]); writer.flush(); }
  assert.equal(seen.filter((id) => id === 'a').length, 0, 'the untouched chat is never stringified again');
  assert.equal(seen.filter((id) => id === 'b').length, 50);
  assert.equal(JSON.parse(writes.at(-1)).length, 2);
});

test('the saved copy is capped: a huge tool result is shortened, an over-long chat loses its oldest messages, old chats make room', () => {
  const big = 'y'.repeat(1_600_000);
  const one = serializeSession({ id: 'a', messages: [{ id: '1', tools: [{ result: big }] }] });
  assert.ok(one.length < 250_000, 'one 1.6M field is clipped');
  const many = { id: 'b', messages: Array.from({ length: 40 }, (_, i) => ({ id: String(i), content: 'z'.repeat(150_000) })) };
  const text = serializeSession(many);
  assert.ok(text.length <= MAX_SESSION_CHARS);
  const kept = JSON.parse(text).messages;
  assert.equal(kept.at(-1).id, '39', 'the latest message is kept');

  const writes = [];
  const writer = createSessionWriter({ write: (j) => writes.push(JSON.parse(j)), maxTotalChars: 320_000 });
  const chats = ['old', 'mid', 'new'].map((id, i) => ({ id, updatedAt: String(i), messages: [{ id: 'm', content: 'q'.repeat(150_000) }] }));
  writer.schedule(chats); writer.flush();
  assert.deepEqual(writes[0].map((s) => s.id), ['mid', 'new'], 'the oldest chat is left out of the saved copy');
});

test('the saved list keeps the format older builds read (a JSON array of chats)', () => {
  let out = '';
  const w = createSessionWriter({ write: (j) => { out = j; } });
  w.schedule([{ id: 'a', title: 'A', messages: [] }]); w.flush();
  assert.deepEqual(JSON.parse(out), [{ id: 'a', title: 'A', messages: [] }]);
});

/* End to end through the real store: a 1,500-token streamed answer. */
test('a streamed answer of 1,500 tokens is at most a few storage writes, and the final text is saved', async () => {
  const writes = [];
  globalThis.window ??= globalThis;
  globalThis.document ??= { addEventListener() {}, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
  const memory = new Map();
  globalThis.localStorage = {
    getItem: (k) => memory.get(k) ?? null,
    setItem: (k, v) => { if (k === 'strata.sessions.v1') writes.push(v); memory.set(k, v); },
    removeItem: (k) => memory.delete(k),
  };
  globalThis.navigator ??= { language: 'en' };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
  const TOKENS = 1_500;
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({
        loader: 'js',
        contents: `
          export class ApiError extends Error {}
          const reject = async () => { throw new Error('offline'); };
          export const api = { get: reject, post: async () => ({}), put: reject, patch: reject, del: reject, delete: reject };
          export function onEvent() { return () => {}; }
          export async function streamPost(path, body, onFrame) {
            onFrame({ type: 'start', model: 'm', contextMax: 8192, requestId: 'r1' });
            for (let i = 0; i < ${TOKENS}; i++) onFrame({ type: 'delta', text: 'word ' });
            onFrame({ type: 'done', tokensIn: 10, tokensOut: ${TOKENS}, durationMs: 100, usageSource: 'reported' });
          }`,
      }));
    },
  };
  const store = await bundle('../src/lib/store.ts', [mockClient]);
  const before = writes.length;
  await store.sendMessage('Write a long answer');
  const during = writes.length - before;
  assert.ok(during <= 6, `a 1,500-token answer wrote the chat list ${during} times`);
  const saved = JSON.parse(writes.at(-1));
  const answer = saved[0].messages.find((m) => m.role === 'assistant');
  assert.ok(answer.content.length >= TOKENS * 5 - 10, 'the finished answer is what was saved');
  assert.equal(answer.streaming, false);
});
