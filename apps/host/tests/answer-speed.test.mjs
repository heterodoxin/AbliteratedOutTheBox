/** "Answer speed" is one number everywhere: every token the model produced over its decode time, never over a time that holds the prompt being read. */
import { fileURLToPath } from 'node:url';
import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { sumEngineRequests, turnEngineSpeed } from '../server/agent/engine-speed.mjs';
import { TurnProjector } from '../server/agent/events.mjs';
import { answerSpeed } from '../server/model-providers.mjs';
import * as monitor from '../server/monitor.mjs';
import * as serving from '../server/serving.mjs';

async function moduleAt(path) { const result = await build({ entryPoints: [fileURLToPath(new URL(path, import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm' }); return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`); }
const { chatMetrics } = await moduleAt('../src/lib/chat-metrics.ts');

// Two real requests from the owner PC (Gyro-S with MTP drafting, RTX 5070), as the engine's history lists them.
const REAL = [
  { time: 1000.0, prompt_tokens: 62, reused: 0, prompt_ms: 941, decode_ms: 711, engine_generated: 34, output_tokens: 34, drafts_offered: 34, drafts_accepted: 18 },
  { time: 1010.0, prompt_tokens: 308, reused: 0, prompt_ms: 1212, decode_ms: 1108, engine_generated: 73, output_tokens: 73, drafts_offered: 68, drafts_accepted: 45 },
];

test('the engine speed of a turn: produced tokens over decode time, prompt reading kept apart', () => {
  const one = sumEngineRequests(REAL, { since: 1009_000 });
  assert.equal(one.tokens, 73);
  assert.equal(one.tps, 65.9);
  assert.equal(one.promptTps, 254.1);
  const both = sumEngineRequests(REAL, { since: 999_000 });
  assert.equal(both.tokens, 107);
  assert.equal(both.tps, Number((107 / 1.819).toFixed(1)));       // 58.8: the sum over the sum, not the mean of the two
  // dividing by the whole time (reading + decoding) is the number users saw as "lower than before"
  assert.ok(73 / ((1212 + 1108) / 1000) < 32);
});

test('drafted tokens are not output: the speed uses the engine count, whatever was offered', () => {
  const r = sumEngineRequests([{ ...REAL[1], drafts_offered: 9999 }], { since: 1009_000 });
  assert.equal(r.tps, 65.9);
});

test("another chat's requests on the same engine are not this turn's", () => {
  const stranger = { time: 1011.0, prompt_tokens: 10, reused: 0, prompt_ms: 100, decode_ms: 1000, engine_generated: 10 };
  const r = sumEngineRequests([REAL[1], stranger], { since: 1009_000, attempts: 1 });
  assert.equal(r.tokens, 10);           // the newest one only: the turn made one model request
  assert.equal(sumEngineRequests([], { since: 0 }), null);
  assert.equal(sumEngineRequests([{ time: 5, engine_generated: 5, decode_ms: 0 }], { since: 0 }), null);
});

test('reading the engine: its own /metrics with the key, null when it fails', async () => {
  let seen;
  const fetcher = async (url, init) => { seen = { url, auth: init.headers.authorization }; return { ok: true, json: async () => ({ requests: REAL }) }; };
  const r = await turnEngineSpeed({ baseUrl: 'http://127.0.0.1:8080/v1', key: 'k', since: 1009_000, fetcher });
  assert.equal(r.tps, 65.9);
  assert.equal(seen.url, 'http://127.0.0.1:8080/metrics?requests=all');
  assert.equal(seen.auth, 'Bearer k');
  assert.equal(await turnEngineSpeed({ baseUrl: 'http://127.0.0.1:8080/v1', since: 0, fetcher: async () => { throw new Error('down'); } }), null);
  assert.equal(await turnEngineSpeed({ baseUrl: 'http://127.0.0.1:8080/v1', since: 0, fetcher: async () => ({ ok: false }) }), null);
});

const chunk = (attemptId, c) => ({ type: 'assistant-stream', frame: { type: 'chunk', attemptId, chunk: c } });

test('the finished agent turn shows the engine speed and the prompt reading speed', () => {
  const p = new TurnProjector({ startedAt: Date.now() });
  p.consume(chunk('a1', { type: 'text-delta', text: 'Done.' }));
  p.consume(chunk('a1', { type: 'usage', usage: { inputTokens: 308, outputTokens: 73 } }));
  p.ended = { kind: 'completed' };
  const done = p.finish({ model: 'm', contextMax: 32768, engine: { tps: 65.9, promptTps: 254.1 } }).find(f => f.type === 'done');
  assert.equal(done.tps, 65.9);
  assert.equal(done.promptTps, 254.1);
  assert.equal(done.tokensOut, 73);
});

test('a short reply still gets a speed (it used to come out empty under 0.3 s of streaming)', () => {
  mock.timers.enable({ apis: ['Date'], now: 3_000_000 });
  try {
    const p = new TurnProjector({ startedAt: Date.now() });
    mock.timers.tick(2000);                       // reading the prompt
    p.consume(chunk('a1', { type: 'text-delta', text: 'Hi!' }));
    p.consume(chunk('a1', { type: 'usage', usage: { inputTokens: 40, outputTokens: 3 } }));
    mock.timers.tick(100);
    p.ended = { kind: 'completed' };
    const done = p.finish({}).find(f => f.type === 'done');
    assert.ok(done.tps > 0 && done.tps <= 60, `${done.tps}`);
  } finally { mock.timers.reset(); }
});

test('a model API reply is timed from its first token, not from the request', () => {
  assert.equal(answerSpeed(100, 12_000, 2_000), 10);        // 100 tokens in the 10 s after the first token
  assert.ok(100 / 12 < answerSpeed(100, 12_000, 2_000));    // the whole-wait number is the lower one
  assert.equal(answerSpeed(50, 1000, null), 50);            // no first-token time: the whole time is all there is
  assert.equal(answerSpeed(50, 1000, 900), 50);             // under a quarter second left: not used
  assert.equal(answerSpeed(null, 1000, 0), null);
  assert.equal(answerSpeed(0, 1000, 0), null);
});

test('the monitor and the request list average answer speeds, not tokens over the whole request', () => {
  monitor.record({ tokensIn: 308, tokensOut: 73, durationMs: 2320, tps: 65.9 });
  monitor.record({ tokensIn: 62, tokensOut: 34, durationMs: 1652, tps: 47.8 });
  assert.equal(monitor.snapshot().totals.avgTps, 56.9);
  monitor.record({ tokensIn: 1, tokensOut: 5, durationMs: 100 });      // no decode figure: left out, not divided by the wall time
  assert.equal(monitor.snapshot().totals.avgTps, 56.9);
  const a = serving.open({ client: 't', endpoint: '/x', model: 'm' });
  serving.mark(a.id, { tps: 65.9 });
  serving.finish(a.id, { tokensOut: 73, durationMs: 2320 });
  assert.equal(serving.rateSample(), 65.9);
});

const reply = (o) => ({ role: 'assistant', at: new Date().toISOString(), content: 'x', ...o });
const chat = (messages) => ({ id: 's', messages, tokensIn: 0, tokensOut: 0, statsAvailable: true });

test('the footer shows the last reply own speed, and the prompt reading speed next to it', () => {
  const s = chat([reply({ tps: 61, inputTokens: 10, tokens: 5 }), reply({ tps: 66, promptTps: 254, inputTokens: 20, tokens: 7 })]);
  const m = chatMetrics(s, null, 32768);
  assert.equal(m.speed, 66);
  assert.equal(m.promptSpeed, 254);
  // an older reply's number is never shown under "last reply"
  const m2 = chatMetrics(chat([reply({ tps: 61, inputTokens: 10, tokens: 5 }), reply({ inputTokens: 20, tokens: 7 })]), null, 32768);
  assert.equal(m2.speed, null);
});

test('while it thinks the footer reads the engine like the monitor does', () => {
  const s = chat([reply({ streaming: true, tps: 12 })]);
  const monitorState = { phase: 'thinking', outputTps: 64.2, context: { sessionId: 's', usedTokens: 100, maxTokens: 32768 } };
  assert.equal(chatMetrics(s, monitorState, 32768).speed, 64.2);
  assert.equal(chatMetrics(s, { ...monitorState, outputTps: 0 }, 32768).speed, 12);   // no reading yet: the stream's own estimate
});
