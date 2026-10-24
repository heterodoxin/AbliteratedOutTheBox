// /loop stops by itself: a count or a time said in words is read when the loop is created and enforced by the scheduler, and an
// answer that says "done" / "stopped" ends it. The model is never left to keep replying "stopped" every minute.
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const bundled = await build({ stdin: { contents: "export * from './src/lib/loop'; export * from './src/lib/loop-runner';", resolveDir: process.cwd(), loader: 'ts' }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { parseLoop, readLimits, replySaysDone, judgeAnswer, withStopRule, LoopScheduler, LOOP_MAX_FAILURES, LOOP_MAX_RUNS } = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);

const start = text => { const c = parseLoop(text); assert.equal(c?.kind, 'start', text); return c; };

test('a count in words or digits is read when the loop is created, and left out of the message', () => {
  for (const [line, runs, prompt] of [
    ['/loop 1m say hi, stop after 3 answers', 3, 'say hi'],
    ['/loop 1m say hi and stop after three replies', 3, 'say hi'],
    ['/loop 5m say hi. Stop the loop after 2 runs.', 2, 'say hi'],
    ['/loop 5m check the build, only 4 times', 4, 'check the build'],
    ['/loop 5m ping 3 times', 3, 'ping'],
    ['/loop every 2 minutes say hi for 5 answers', 5, 'say hi'],
    ['/loop 1m report the time, up to 6 times', 6, 'report the time'],
    ['/loop 1m say hi then stop after 1 answer', 1, 'say hi'],
    ['/loop 1m say hi, end after the 3rd run', 3, 'say hi'],
  ]) {
    const c = start(line);
    assert.equal(c.maxRuns, runs, line);
    assert.equal(c.prompt, prompt, line);
  }
});

test('a time limit and a stop condition in words are read too', () => {
  let c = start('/loop 10m check the build for the next 2 hours');
  assert.equal(c.maxMs, 2 * 3_600_000); assert.equal(c.prompt, 'check the build');
  c = start('/loop 5m check the build, stop after 30 minutes');
  assert.equal(c.maxMs, 30 * 60_000); assert.equal(c.prompt, 'check the build');
  c = start('/loop 2m check the build, stop when it passes');
  assert.equal(c.stopWhen, 'it passes'); assert.equal(c.prompt, 'check the build'); assert.equal(c.maxRuns, undefined);
  c = start('/loop 2m look for the file until it exists');
  assert.equal(c.stopWhen, 'it exists'); assert.equal(c.prompt, 'look for the file');
});

test('an ordinary message is not mistaken for a limit, and a limit with nothing to do is explained', () => {
  for (const line of ['/loop 10m summarise my inbox notes', '/loop 5m say hi', '/loop 10m check whether the build passes', '/loop 5m run the tests faster than last time']) {
    const c = start(line);
    assert.equal(c.maxRuns, undefined, line); assert.equal(c.maxMs, undefined, line); assert.equal(c.stopWhen, undefined, line);
    assert.equal(c.prompt, line.replace(/^\/loop \S+ /, ''));
  }
  assert.equal(parseLoop('/loop 5m stop after 3 answers').kind, 'help');
  assert.match(parseLoop('/loop 5m stop after 3 answers').problem, /what to do/);
  assert.equal(readLimits('say hi').prompt, 'say hi');
  assert.equal(start('/loop 5m say hi, stop after 500 answers').maxRuns, LOOP_MAX_RUNS, 'the count is capped like the loop');
});

test('a reply that says done or stopped ends the loop, a normal reply does not', () => {
  for (const reply of ['stopped', 'Stopped.', 'Done', 'DONE the build passes', 'LOOP DONE', 'The loop is stopped.']) assert.equal(replySaysDone(reply, true), true, reply);
  for (const reply of ['stopped', 'Done.', 'DONE: it passes', 'LOOP STOPPED']) assert.equal(replySaysDone(reply, false), true, reply + ' (nothing stated)');
  for (const reply of ['The build is done and the tests pass: 41 of 41.', 'Hi! Nice to see you.', 'I checked the folder and found nothing new.', '']) assert.equal(replySaysDone(reply, true), false, reply);
  // Without a stated limit only the bare words count: "Done. The deploy finished" could be a normal answer.
  assert.equal(replySaysDone('Done. The build passes.', false), false);
  assert.equal(replySaysDone('Done. The build passes.', true), true);
  assert.match(withStopRule('check the build', 'it passes'), /begin your reply with the word DONE/);
  assert.equal(withStopRule('check the build'), 'check the build');
});

test('judgeAnswer: count, time, failures and the said-done rule, in that situation order', () => {
  const p = (answers, failures = 0, startedAt = 0) => ({ answers, failures, startedAt });
  assert.equal(judgeAnswer({ maxRuns: 3 }, p(2), { text: 'hi', failed: false }, 0), null);
  assert.equal(judgeAnswer({ maxRuns: 3 }, p(3), { text: 'hi', failed: false }, 0), 'count');
  assert.equal(judgeAnswer({ maxMs: 1000 }, p(1), { text: 'hi', failed: false }, 999), null);
  assert.equal(judgeAnswer({ maxMs: 1000 }, p(1), { text: 'hi', failed: false }, 1000), 'time');
  assert.equal(judgeAnswer({}, p(1, LOOP_MAX_FAILURES - 1), { text: '', failed: true }, 0), null);
  assert.equal(judgeAnswer({}, p(1, LOOP_MAX_FAILURES), { text: '', failed: true }, 0), 'failures');
  assert.equal(judgeAnswer({}, p(1), { text: 'stopped', failed: false }, 0), 'said-done');
  assert.equal(judgeAnswer({}, p(LOOP_MAX_RUNS), { text: 'hi', failed: false }, 0), 'cap');
});

/** A scheduler on a fake clock with a scripted model. `reply(n, prompt)` is the model's n-th answer. */
function harness(reply, { ready = () => true } = {}) {
  const sent = []; const notes = []; const timers = new Map(); let t = 0, nextTimer = 1;
  const host = {
    async send(_id, message) {
      sent.push(message);
      const r = await reply(sent.length, message.text);
      return r === null ? { answered: false, text: '', failed: false } : typeof r === 'string' ? { answered: true, text: r, failed: false } : { answered: true, ...r };
    },
    note: (_id, text) => notes.push(text),
    exists: () => true,
    ready,
    now: () => t,
    setTimer: (fn, ms) => { const id = nextTimer++; timers.set(id, { fn, at: t + ms }); return id; },
    clearTimer: id => timers.delete(id),
  };
  const scheduler = new LoopScheduler(host);
  /** Move the clock to the next timer and run it; false when nothing is waiting. */
  const advance = async () => {
    const next = [...timers.entries()].sort((a, b) => a[1].at - b[1].at)[0];
    if (!next) return false;
    timers.delete(next[0]); t = Math.max(t, next[1].at); next[1].fn();
    for (let i = 0; i < 20; i++) await Promise.resolve();
    await new Promise(r => setImmediate(r));
    return true;
  };
  const drain = async (max = 400) => { let n = 0; while (n < max && await advance()) n++; return n; };
  return { scheduler, sent, notes, timers, advance, drain, clock: () => t };
}

test('"stop after 3 answers" stops the scheduler after the third answer and nothing is sent again', async () => {
  const h = harness(() => 'Hello!');
  const c = start('/loop 1m say hi, stop after 3 answers');
  await h.scheduler.start('chat', c);
  await h.drain();
  assert.equal(h.sent.length, 3);
  assert.equal(h.sent[0].text, 'say hi', 'the model is not told about the stop rule');
  assert.equal(h.timers.size, 0, 'no timer is left');
  assert.equal(h.scheduler.isRunning('chat'), false);
  assert.match(h.notes.at(-1), /The loop is done: it ran 3 times/);
  assert.equal(h.notes.filter(n => /stops by itself after 3 answers/.test(n)).length, 1, 'the start note says how it ends');
});

test('the owner case: a model that answers "stopped" every minute is cut off at once', async () => {
  // With a count in the line the scheduler stops at 3 by itself, and a "stopped" answer before that also ends it.
  const h = harness(n => (n < 2 ? 'Hi!' : 'stopped'));
  await h.scheduler.start('chat', start('/loop 1m say hi, stop after 3 answers'));
  await h.drain();
  assert.equal(h.sent.length, 2, 'the "stopped" answer ended the loop');
  assert.match(h.notes.at(-1), /last answer said the work is done/);
  // No limit in the line at all: a bare "Stopped." still ends it, instead of 100 minutes of "stopped".
  const g = harness(n => (n < 3 ? 'Still going.' : 'Stopped.'));
  await g.scheduler.start('chat', start('/loop 1m keep an eye on the queue'));
  await g.drain();
  assert.equal(g.sent.length, 3); assert.equal(g.timers.size, 0);
});

test('a stop condition in words: the model is told how to say it is met, and its DONE ends the loop', async () => {
  const h = harness(n => (n < 4 ? 'Not yet: 2 tests fail.' : 'DONE all tests pass.'));
  await h.scheduler.start('chat', start('/loop 2m run the tests, stop when they all pass'));
  await h.drain();
  assert.equal(h.sent.length, 4);
  assert.match(h.sent[0].text, /begin your reply with the word DONE/);
  assert.equal(h.sent[0].shown, 'run the tests', 'the chat shows the message without the rule');
});

test('a time limit ends the loop by the clock, between answers', async () => {
  const h = harness(() => 'ok');
  await h.scheduler.start('chat', start('/loop 10m check the queue for the next 35 minutes'));
  await h.drain();
  // answers at 0, 10, 20, 30 minutes; the timer at 40 finds the 35 minutes over
  assert.equal(h.sent.length, 4);
  assert.match(h.notes.at(-1), /time you set is up/);
  assert.equal(h.timers.size, 0);
});

test('a busy chat waits, and does not use up the count', async () => {
  let busy = true;
  const h = harness(() => 'ok', { ready: () => !busy });
  await h.scheduler.start('chat', start('/loop 1m ping, stop after 2 answers'));
  assert.equal(h.sent.length, 1);
  await h.advance();                                   // 1 minute: busy, retried in 15 s
  await h.advance(); await h.advance();
  assert.equal(h.sent.length, 1, 'nothing is sent while the chat is busy');
  busy = false;
  await h.drain();
  assert.equal(h.sent.length, 2);
  assert.equal(h.scheduler.isRunning('chat'), false);
});

test('three failed answers in a row end the loop, one failure does not', async () => {
  const h = harness(n => (n === 2 ? { text: '', failed: true } : 'ok'));
  await h.scheduler.start('chat', start('/loop 1m check something, stop after 4 answers'));
  await h.drain();
  assert.equal(h.sent.length, 5, 'the failed run is not an answer, so 4 good answers took 5 sends');
  const bad = harness(() => ({ text: '', failed: true }));
  await bad.scheduler.start('chat', start('/loop 1m check something'));
  await bad.drain();
  assert.equal(bad.sent.length, LOOP_MAX_FAILURES);
  assert.match(bad.notes.at(-1), /did not work/);
});

test('/loop stop while an answer is coming means nothing more is sent', async () => {
  let release;
  const h = harness(() => new Promise(resolve => { release = () => resolve('ok'); }));
  const started = h.scheduler.start('chat', start('/loop 1m ping'));
  await Promise.resolve();
  assert.equal(h.scheduler.stop('chat'), true);
  release(); await started;
  assert.equal(h.timers.size, 0);
  assert.equal(h.scheduler.isRunning('chat'), false);
  assert.equal(await h.advance(), false);
  assert.equal(h.sent.length, 1);
});

test('a second /loop in the same chat is refused with one plain sentence; a loop with no limit still runs on', async () => {
  const h = harness(() => 'ok');
  await h.scheduler.start('chat', start('/loop 1m ping'));
  assert.equal(await h.scheduler.start('chat', start('/loop 1m other')), false);
  assert.match(h.notes.at(-1), /already has a loop/);
  for (let i = 0; i < 5; i++) await h.advance();
  assert.equal(h.sent.length, 6, 'no limit: it keeps going until the person stops it');
  assert.equal(h.scheduler.stop('chat'), true);
});
