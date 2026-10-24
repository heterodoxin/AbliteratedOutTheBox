// The page side of the context fixes (2026-10-22 hands-on test): what the error card offers, what the context meter reads while a turn runs,
// and that every sentence the host or the page says about the context is in the catalog in all eight languages.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { FIT_TEXT } from '../server/agent/context-fit.mjs';

const here = rel => fileURLToPath(new URL(rel, import.meta.url));
const read = rel => readFileSync(here(rel), 'utf8').replace(/\r\n/g, '\n');
globalThis.window ??= globalThis; globalThis.localStorage ??= { getItem: () => null, setItem: () => {} };
globalThis.navigator ??= { language: 'en' };

async function load(file, stubs = {}) {
  const result = await build({
    entryPoints: [here(`../src/lib/${file}`)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent', loader: { '.json': 'json' },
    plugins: [{ name: 'stubs', setup(b) { b.onResolve({ filter: /^\.\/(store)$/ }, () => ({ path: 'store', namespace: 'stub' })); b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({ contents: stubs.store, loader: 'js' })); } }],
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

const calls = [];
globalThis.__calls = calls;
const recovery = await load('context-recovery.ts', {
  store: `export const sendOversized = (mode) => globalThis.__calls.push(['oversized', mode]);
          export const summarizeAndContinue = () => globalThis.__calls.push(['summarize']);
          export const startNewChat = () => globalThis.__calls.push(['new-chat']);`,
});
const { chatMetrics } = await load('chat-metrics.ts');

test('a conversation that no longer fits offers "Summarize and continue" and "Start a new chat", never "try again"', () => {
  const r = recovery.contextRecovery('CONTEXT_WINDOW_EXCEEDED');
  assert.equal(r.action, 'Summarize and continue');
  assert.equal(r.secondary.label, 'Start a new chat');
  assert.match(r.next, /Summarize it now and continue, or start a new chat\./);
  assert.doesNotMatch(r.next, /again/i);
  calls.length = 0;
  r.onRetry(); r.secondary.onClick();
  assert.deepEqual(calls, [['summarize'], ['new-chat']]);
});

test('a message that is too large offers the two ways that fit: a file the assistant reads in parts, or a shortened preview', () => {
  const r = recovery.contextRecovery('CONTEXT_MESSAGE_TOO_LARGE');
  assert.equal(r.action, 'Attach as a file');
  assert.equal(r.secondary.label, 'Send a shortened preview');
  calls.length = 0;
  r.onRetry(); r.secondary.onClick();
  assert.deepEqual(calls, [['oversized', 'file'], ['oversized', 'preview']]);
});

test('any other error keeps the card as it was: one sentence, no buttons of ours', () => {
  for (const code of [undefined, 'UNKNOWN', 'AGENT_STOPPED']) {
    const r = recovery.contextRecovery(code);
    assert.equal(r.next, 'Try sending the message again.');
    assert.equal(r.onRetry, undefined);
    assert.equal(r.secondary, undefined);
  }
});

test('the chat shows the error card with Details and the recovery buttons, and the card can hold a second button', () => {
  const chat = read('../src/screens/Chat.tsx');
  assert.match(chat, /<ErrorNotice message=\{m\.error\} detail=\{m\.errorDetail\} \{\.\.\.contextRecovery\(m\.errorCode\)\}\/>/);
  const ui = read('../src/components/ui.tsx');
  assert.match(ui, /secondary\?: \{ label: string; onClick: \(\) => void \}/);
  assert.match(ui, /<Btn size="sm" variant="ghost" onClick=\{secondary\.onClick\}>/);
});

test('the store keeps the code and the engine\'s text of an error, shows a sentence the host wrote in the person\'s language, and follows the context live', () => {
  const store = read('../src/lib/store.ts');
  assert.match(store, /typeof frame\.key === 'string' \? tr\(frame\.key,/);
  assert.match(store, /errorCode: frame\.code/);
  assert.match(store, /errorDetail: frame\.details/);
  assert.match(store, /type === 'context'/);
  const summarize = store.slice(store.indexOf('export async function summarizeAndContinue'), store.indexOf('export async function sendOversized'));
  assert.ok(summarize.indexOf("sendMessage('/compact')") > 0 && summarize.indexOf("sendMessage('Please answer my last message now.')") > summarize.indexOf("sendMessage('/compact')"), 'summarise first, then ask for the answer');
  assert.doesNotMatch(summarize, /failed.prompt|dropFailedExchange/, 'the refused question is not sent a second time (it is already in the conversation, and a large one twice would not fit)');
  const oversized = store.slice(store.indexOf('export async function sendOversized'), store.indexOf('export function startNewChat'));
  assert.match(oversized, /'\/api\/chat\/oversize'/);
  assert.ok(oversized.indexOf('dropFailedExchange') > oversized.indexOf("'/api/chat/oversize'"), 'the failed message stays until the shorter one is ready');
});

const reply = (extra) => ({ id: 'r', role: 'assistant', content: '', at: '2026-10-22T12:00:00Z', ...extra });
const session = messages => ({ id: 's', title: 't', createdAt: '', updatedAt: '', messages, tokensIn: 0, tokensOut: 0, statsAvailable: true });

test('the context meter follows the turn that is running: a tool step does not read as 1%', () => {
  // The 2026-10-22 chat: an earlier reply left the meter at 1% (about 600 tokens), then a turn with tools ran on a 65,536 window.
  const earlier = reply({ id: 'a', inputTokens: 590, tokens: 20, contextUsed: 610, contextMax: 65536 });
  const running = reply({ id: 'b', streaming: true, contextMax: 65536 });
  const before = chatMetrics(session([earlier, running]), null, 65536);
  assert.equal(Math.round(before.pct), 1, 'with nothing reported yet for the turn, the earlier reply is all there is');
  const stepped = reply({ id: 'b', streaming: true, contextMax: 65536, contextUsed: 43000 });
  const during = chatMetrics(session([earlier, stepped]), null, 65536);
  assert.equal(during.used, 43000);
  assert.equal(Math.round(during.pct), 66);
  const monitor = { phase: 'idle', context: { sessionId: 's', usedTokens: 600, maxTokens: 65536 } };
  assert.equal(chatMetrics(session([earlier, stepped]), monitor, 65536).used, 43000, 'the engine\'s figure for the request in flight does not override the turn\'s own');
  const done = reply({ id: 'b', inputTokens: 43000, tokens: 90, contextUsed: 43090, contextMax: 65536 });
  assert.equal(chatMetrics(session([earlier, done]), null, 65536).used, 43090, 'when the turn has ended its final figure stands');
});

test('every sentence about the context is in the catalog in all eight languages, with the same slots', () => {
  const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko'];
  const catalogs = Object.fromEntries(langs.map(l => [l, JSON.parse(read(`../src/locales/${l}.json`))]));
  const idOf = text => Object.entries(catalogs.en).find(([, v]) => v === text)?.[0];
  const sentences = [...Object.values(FIT_TEXT), 'Attach as a file', 'Send a shortened preview', 'Summarize and continue', 'Start a new chat',
    'The message could not be prepared. Try a shorter message.', 'The last Strata session is still closing down.', 'Wait a moment and try again. If it keeps happening, restart the computer.'];
  const slots = t => (t.match(/\{\d+\}/g) ?? []).sort().join();
  for (const text of sentences) {
    const id = idOf(text);
    assert.ok(id, `"${text.slice(0, 60)}" is in the English catalog`);
    for (const l of langs) {
      const t = catalogs[l][id];
      assert.ok(typeof t === 'string' && t.trim(), `${id} in ${l}`);
      assert.equal(slots(t), slots(text), `${id} in ${l} has the same slots`);
      if (l !== 'en' && text.length > 25) assert.notEqual(t, text, `${id} is translated in ${l}`);
    }
  }
});
