/** Owner round 7: live numbers in chat, a steady status line, the subagents counts on the title row, plain Privacy, a round arrow instead of "Latest". */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const src = f => fileURLToPath(new URL(`../src/${f}`, import.meta.url));
const read = f => readFile(src(f), 'utf8');
async function load(file) {
  const result = await build({ entryPoints: [src(`lib/${file}`)], bundle: true, format: 'esm', platform: 'node', write: false, alias: { react: fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url)) }, logLevel: 'silent', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}
globalThis.window ??= globalThis; globalThis.localStorage ??= { getItem: () => null, setItem: () => {} };
globalThis.navigator ??= { language: 'en' };
const { tweenValue, easeOutCubic } = await load('tween.ts');
const { chatMetrics } = await load('chat-metrics.ts');

test('item 5: a number travels between two readings in a short ease-out, never past them', () => {
  assert.equal(tweenValue(40, 50, 0, 600), 40);
  assert.equal(tweenValue(40, 50, 600, 600), 50);
  assert.ok(tweenValue(40, 50, 150, 600) > 40 && tweenValue(40, 50, 150, 600) < 50);
  assert.ok(tweenValue(50, 40, 150, 600) < 50 && tweenValue(50, 40, 150, 600) > 40, 'down as well as up');
  assert.equal(tweenValue(NaN, 7, 100, 600), 7); assert.equal(tweenValue(1, 2, 0, 0), 2);
  assert.ok(easeOutCubic(0.5) > 0.5, 'fast at first, settling at the end');
});

const reply = (extra = {}) => ({ role: 'assistant', at: new Date(1_000_000).toISOString(), content: 'x', ...extra });
test('item 5: during a goal the chat follows the Monitor even though the engine names another session', () => {
  const session = { id: 'chat-a', statsAvailable: true, tokensIn: 5000, tokensOut: 400, messages: [{ role: 'user', content: 'goal' }, reply({ inputTokens: 5000, tokens: 400, tps: 41, contextMax: 65536, durationMs: 9000, reasoningMs: 3000 }), reply({ streaming: true, tokens: 120, inputTokens: 5200, reasoningStartedAt: 1_070_000 })] };
  const monitor = { phase: 'generating', outputTps: 47.4, context: { sessionId: 'harness-xyz', usedTokens: 9100, maxTokens: 65536 } };
  const m = chatMetrics(session, monitor, 65536, 1_100_000);
  assert.equal(m.speed, 47.4, 'the live speed comes from the Monitor');
  assert.equal(m.used, 9100, 'and so does the context in use');
  assert.equal(m.llmMs, 9000 + 100_000, 'a reply being written counts its running time');
  assert.equal(m.thinkMs, 3000 + 30_000);
  assert.equal(m.writeMs, 9000 + 100_000 - 33_000);
  assert.equal(m.total, 5000 + 400 + 120 + 5200);
  // an idle engine does not move a finished chat's numbers
  const idle = chatMetrics({ ...session, messages: session.messages.slice(0, 2) }, { phase: 'idle', outputTps: 0, context: { sessionId: 'harness-xyz', usedTokens: 1, maxTokens: 2 } }, 65536, 1_100_000);
  assert.equal(idle.speed, 41); assert.notEqual(idle.used, 1);
});

test('item 10: the status line has fixed columns, so the word changing length moves nothing', async () => {
  const css = await read('styles/app.css'), line = await read('components/StatusLine.tsx');
  assert.match(css, /\.live-status \{ display: grid; grid-template-columns: 14px minmax\(0, 24ch\) 6ch/);
  assert.match(css, /\.live-status__time, \.live-status__tokens \{ font-variant-numeric: tabular-nums/);
  assert.match(line, /live-status__verb/); assert.match(line, /live-status__time/); assert.match(line, /live-status__tokens/);
});

test('items 15, 16, 17: counts on the Subagents title row, Privacy as plain text, a round arrow', async () => {
  const bar = await read('components/ChatToolbar.tsx'), pop = await read('components/Popover.tsx'), privacy = await read('components/PrivacyNotes.tsx'), chat = await read('screens/Chat.tsx');
  assert.match(bar, /title=\{tr\("Subagents"\)\} detail=\{/); assert.ok(!/subagents-overview-summary/.test(bar), 'no second line of counts');
  assert.match(pop, /telepop__detail/);
  assert.ok(!/Icon/.test(privacy), 'no icons in Privacy');
  assert.ok(!/>\{tr\("Latest"\)\}</.test(chat) && (chat.match(/aria-label=\{tr\("Scroll to the latest message"\)\}/g) ?? []).length === 2, 'the Latest button is a round arrow');
});
