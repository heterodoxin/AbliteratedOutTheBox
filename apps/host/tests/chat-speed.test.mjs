/** The chat footer's "tok/s · last reply" is the answer-writing speed: the engine's output tokens over the time spent decoding them. */
import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { TurnProjector } from '../server/agent/events.mjs';

const chunk = (attemptId, c) => ({ type: 'assistant-stream', frame: { type: 'chunk', attemptId, chunk: c } });

/** One model request: `pieces` chunks `gap` ms apart of the given kind, then the engine's usage; tool runs and the wait for the prompt take their own time. */
test('writing a big tool call counts as decoding time: 600 tokens over 10 s is 60 tok/s, not hundreds', () => {
  mock.timers.enable({ apis: ['Date'], now: 1_000_000 });
  try {
    const p = new TurnProjector({ startedAt: Date.now() });
    // the model reads a 30 s prompt first: no chunk yet
    mock.timers.tick(30_000);
    // it writes a file as a tool call: 100 chunks, 100 ms apart, 600 output tokens in the engine's count
    for (let i = 0; i < 100; i++) {
      p.consume(chunk('a1', { type: 'tool-call-delta', id: 'call-1', name: 'write', argumentsDelta: 'x'.repeat(24) }));
      mock.timers.tick(100);
    }
    p.consume(chunk('a1', { type: 'usage', usage: { inputTokens: 5000, outputTokens: 600 } }));
    // a short closing sentence
    p.consume(chunk('a2', { type: 'text-delta', text: 'Done.' }));
    p.consume({ type: 'event', event: { seq: 9, time: 9, type: 'turn/end', data: { reason: { kind: 'completed' } } } });
    p.ended = { kind: 'completed' };
    const done = p.finish({ model: 'm', contextMax: 32768 }).find(f => f.type === 'done');
    assert.ok(done.tps >= 55 && done.tps <= 65, `${done.tps} tok/s`);
    assert.equal(done.tokensOut, 600, "the engine's own count");
  } finally { mock.timers.reset(); }
});

test('text and reasoning keep their speed, and tool-run time is not decoding time', () => {
  mock.timers.enable({ apis: ['Date'], now: 2_000_000 });
  try {
    const p = new TurnProjector({ startedAt: Date.now() });
    for (let i = 0; i < 50; i++) { p.consume(chunk('a1', { type: 'reasoning-delta', text: 'think ' })); mock.timers.tick(20); }
    for (let i = 0; i < 50; i++) { p.consume(chunk('a1', { type: 'text-delta', text: 'word ' })); mock.timers.tick(20); }
    p.consume(chunk('a1', { type: 'usage', usage: { inputTokens: 100, outputTokens: 100 } }));
    mock.timers.tick(60_000);   // a slow command runs: not decoding
    p.ended = { kind: 'completed' };
    const done = p.finish({}).find(f => f.type === 'done');
    assert.ok(done.tps >= 45 && done.tps <= 55, `${done.tps} tok/s`);
  } finally { mock.timers.reset(); }
});
