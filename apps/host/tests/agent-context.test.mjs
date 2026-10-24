// Context pressure and compaction (2026-10-22 hands-on test): compaction reads the size the model server measured, a summary request
// always fits the window, and a refused prompt is recovered from. The Harness is real; the model server is the scripted worker, set to
// count a prompt the way the engine does (characters over a density) and to refuse one that leaves no room.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { buildProfile } from '../server/agent/profile.mjs';
import { clipMessage, densityOf, estimateMessage, isContextOverflow, pressureNow, promptTokensOf, splitMessages, summaryBudget, usageOfEvent } from '../server/agent/context-budget.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('the profile loads the context plugin, so compaction and /compact read the measured prompt', () => {
  const profile = buildProfile({ local: [{ id: 'a', modelId: 'm', baseUrl: 'http://x', context: 65536 }], connections: [] });
  assert.match(profile.patch, /"id": "strata-context"/);
  assert.match(profile.patch, /strata-context\.mjs/);
});

test('density: the engine said 59,048 tokens for a prompt the Harness priced at 31,794', () => {
  const d = densityOf(59048, 31794);
  assert.ok(d > 1.85 && d < 1.87, `density ${d}`);
  assert.equal(densityOf(0, 100), 1);
  assert.equal(densityOf(100, 0), 1);
  assert.equal(densityOf(10, 100), 1, 'plain prose is never priced lower than the Harness already does');
  assert.equal(densityOf(10000, 100), 6, 'a wild sample is capped');
});

test('pressure: the measured part is exact, what came after is priced at the measured density, and it never reads lower than the Harness', () => {
  const sample = { real: 59048, estimated: 31794 };
  assert.equal(pressureNow({ sample, now: 31794 }), 59048, 'nothing added since the sample: the measured number');
  const grown = pressureNow({ sample, now: 31794 + 1000 });
  assert.ok(grown > 59048 + 1800 && grown < 59048 + 1900, `1000 added tokens are worth about 1,860 real ones (${grown})`);
  assert.ok(pressureNow({ sample, now: 31794 - 5000 }) < 59048, 'a compaction lowers it');
  assert.equal(pressureNow({ sample: null, now: 1234 }), 1234, 'no measurement yet: the Harness price');
  assert.equal(pressureNow({ sample: { real: 100, estimated: 5000 }, now: 5000 }), 5000, 'never below the Harness price');
});

test('the compaction threshold on a 64K window leaves room for the reply even when the prompt is measured in real tokens', () => {
  // Gyro-S at 64K: the reply keeps 19,660 tokens and compaction keeps 5,734 of headroom (profile.mjs), so it starts at 40,142 REAL tokens.
  const window = 65536, reply = 19660, headroom = 5734;
  const threshold = Math.floor(Math.min(window * 0.8, window - reply - headroom));
  assert.equal(threshold, 40142);
  assert.ok(threshold + reply < window, 'the prompt at the threshold plus a full reply still fits');
  // The 2026-10-22 chat: 31,794 priced = 59,048 real. It is over the threshold now (before: under it, and the engine refused the next request).
  assert.ok(pressureNow({ sample: { real: 59048, estimated: 31794 }, now: 31794 }) > threshold);
});

test('usage helpers: prompt side only, from a message or from its stream', () => {
  assert.equal(promptTokensOf({ inputTokens: 100, outputTokens: 50, cacheReadTokens: 30, cacheWriteTokens: 5 }), 135);
  assert.equal(promptTokensOf(null), 0);
  assert.deepEqual(usageOfEvent({ data: { usage: { inputTokens: 7 } } }), { inputTokens: 7 });
  assert.deepEqual(usageOfEvent({ data: { stream: [{ type: 'chunk', chunk: { type: 'text-delta' } }, { type: 'chunk', chunk: { type: 'usage', usage: { inputTokens: 9 } } }] } }), { inputTokens: 9 });
  assert.equal(usageOfEvent({ data: {} }), null);
});

test('a refusal for the context window is recognised by its code and by the engine\'s own words', () => {
  assert.ok(isContextOverflow({ code: 'CONTEXT_WINDOW_EXCEEDED', message: 'x' }));
  assert.ok(isContextOverflow(new Error('400: {"type":"invalid_request_error","message":"request exceeds the context window: prompt (70141 tokens) leaves no room to answer in the context (65536); requests are never truncated"}')));
  assert.ok(!isContextOverflow(new Error('fetch failed')));
  assert.ok(!isContextOverflow(null));
});

test('a summary request is split into pieces that fit, each message whole when it can be, a huge one cut in the middle', () => {
  const text = n => ({ role: 'user', content: [{ type: 'text', text: 'x'.repeat(n) }] });
  const many = Array.from({ length: 12 }, () => text(4000));   // 1,000 tokens each by the Harness price
  const runs = splitMessages(many, 3500, 1);
  assert.deepEqual(runs.map(r => r.length), [3, 3, 3, 3]);
  assert.deepEqual(runs.flat().length, 12, 'nothing is dropped');
  const dense = splitMessages(many, 3500, 2);
  assert.ok(dense.length > runs.length, 'a denser text means more pieces');
  const huge = clipMessage(text(400000), 8000, 1);
  assert.ok(estimateMessage(huge) <= 8000, `cut to its share (${estimateMessage(huge)})`);
  assert.match(huge.content[0].text, /left out of this summary request/);
  assert.equal(clipMessage(text(100), 8000, 1).content[0].text.length, 100, 'a small message is untouched');
  const oneBig = splitMessages([text(10), text(600000), text(10)], 6000, 1);
  assert.ok(oneBig.every(r => r.reduce((n, m) => n + estimateMessage(m), 0) <= 6000), 'every piece fits after the cut');
  // A tool call and its result stay together when a clean cut exists.
  const call = { role: 'assistant', content: [{ type: 'tool-call', id: 'c', name: 'read', arguments: '{}' }] };
  const result = { role: 'tool', toolCallId: 'c', content: [{ type: 'text', text: 'y'.repeat(2000) }] };
  const paired = splitMessages([text(2000), call, result, text(2000)], 1200, 1);
  assert.ok(paired.every(r => !r.includes(call) || r.includes(result)), 'the call and its result are in one piece');
  assert.ok(summaryBudget({ window: 65536, summaryTokens: 5734, toolsTokens: 10133, systemTokens: 400 }) > 40000);
});

/** The scripted model server counting like the engine (3 characters a token here: logs and code) in a 32K window. */
async function rig(t, { window = 32768, charsPerToken = 3, noCount = false } = {}) {
  const worker = await startScriptedWorker({ window, charsPerToken, noCount });
  const dir = mkdtempSync(join(tmpdir(), 'strata-ctx-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: window, slots: 2 }], connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const send = async (message) => { const frames = []; await bridge.runTurn({ chatId: 'c', message, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) }); return frames; };
  const harnessLog = () => { try { return readFileSync(join(dir, 'agent', 'harness.log'), 'utf8'); } catch { return ''; } };
  return { worker, bridge, send, ws, harnessLog };
}
const answerOf = frames => frames.filter(f => f.type === 'delta' && !f.noteReason).map(f => f.text).join('');
const dense = 'q=0x1f ok;';   // 10 characters: 2.5 tokens by the Harness price, 3.3 by the model server's count here

test('the plugin loads in the real Harness (a plugin that fails to import would leave compaction as it was, silently)', { skip }, async (t) => {
  const { send, harnessLog } = await rig(t);
  const frames = await send('SCRIPT:echo hello');
  assert.equal(frames.at(-1).type, 'done');
  assert.doesNotMatch(harnessLog(), /did not activate|failed to import|strata-context/, harnessLog());
});

test('dense text compacts in time: no request is refused, and every message is answered', { skip }, async (t) => {
  const { worker, send } = await rig(t);
  const replies = [];
  for (let i = 0; i < 8; i++) {
    const frames = await send(`SCRIPT:echo round${i} ${dense.repeat(900)}`);
    assert.equal(frames.at(-1).type, 'done', `round ${i} ended normally: ${JSON.stringify(frames.at(-1)).slice(0, 300)}`);
    replies.push(answerOf(frames).slice(0, 12));
  }
  assert.ok(replies.every(r => /^round[0-9]/.test(r)), replies.join(' | '));
  const refused = worker.requests.filter(r => r.refused);
  assert.equal(refused.length, 0, `the model server refused ${refused.length} request(s): ${refused.map(r => r.refused).join(', ')} tokens`);
  assert.ok(worker.requests.some(r => r.summary), 'the older history was summarised');
  // Compaction starts before the next message would push the prompt past the threshold, so no answer was asked for from a nearly full window.
  const peak = Math.max(...worker.requests.filter(r => !r.summary).map(r => Math.round(r.promptChars / 3)));
  assert.ok(peak < 32768 - 4000, `no prompt came within 4,000 tokens of the window (peak ${peak})`);
});

test('a chat the model server refused once is not stuck: the next message compacts (the summary itself fits the window) and goes through', { skip }, async (t) => {
  // (An engine that cannot count a message before it is sent: the check lets it through, and this is the recovery that remains.)
  // 2026-10-22: a 103 KB message made a 70K-token prompt, the 400 left it in the history, and then neither the next message nor /compact
  // could go through, because the summary request carried the same history and was refused for its size too.
  const { worker, send } = await rig(t, { noCount: true });
  assert.equal((await send(`SCRIPT:echo one ${dense.repeat(300)}`)).at(-1).type, 'done');
  const huge = await send(`SCRIPT:echo two ${dense.repeat(7000)}`);
  assert.equal(huge.at(-1).type, 'done', 'the Harness compacted after the refusal and asked again: ' + JSON.stringify(huge.at(-1)).slice(0, 200));
  const small = await send('SCRIPT:echo three please');
  assert.equal(small.at(-1).type, 'done', JSON.stringify(small.at(-1)).slice(0, 300));
  assert.match(answerOf(small), /three/);
  const again = await send('SCRIPT:echo four please');
  assert.equal(again.at(-1).type, 'done');
  const refused = worker.requests.filter(r => r.refused);
  assert.equal(refused.length, 1, `only the first, too large, prompt was refused: ${refused.map(r => r.refused).join(', ')}`);
  assert.ok(worker.requests.filter(r => r.summary).every(r => Math.round(r.promptChars / 3) < 32768 - 64), 'every summary request fitted the window');
});

// ------------------------------------------------------------------------------------------------ the message is checked before it is sent

import { COUNT_ABOVE_CHARS, FIT_TEXT, countTokens, decideFit, fill, oversizedReplacement } from '../server/agent/context-fit.mjs';
import { TurnProjector, friendlyFailure, isContextFailure } from '../server/agent/events.mjs';

test('fit: a message that fits is sent, one that fits after a summary is summarised first, one that never fits is not sent', () => {
  // Gyro-S at 64K: 65,536 window, a new chat already holds about 12,000 (system prompt and tools), the reply needs 4,096.
  assert.equal(decideFit({ window: 65536, pressure: 12000, messageTokens: 5000 }).action, 'send');
  assert.equal(decideFit({ window: 65536, pressure: 30000, messageTokens: 25000 }).action, 'send', 'it fits with the reply room');
  assert.equal(decideFit({ window: 65536, pressure: 59048, messageTokens: 5000 }).action, 'summarize', 'the chat is full but the message alone is small');
  const huge = decideFit({ window: 65536, pressure: 0, messageTokens: 70141 });
  assert.equal(huge.action, 'too-large', 'the 103 KB message of 2026-10-22 (70,141 tokens) can never fit a 65,536 window');
  assert.equal(decideFit({ window: 65536, pressure: 0, messageTokens: 49000 }).action, 'send');
  assert.equal(decideFit({ window: 65536, pressure: 20000, messageTokens: 49000 }).action, 'summarize');
  assert.equal(decideFit({ window: 65536, pressure: 20000, messageTokens: 50000 }).action, 'too-large', 'with 12,000 for the empty chat and 4,096 for the reply, 49,440 is the most');
  assert.equal(decideFit({ window: 0, messageTokens: 99999 }).action, 'send', 'no window known: nothing is blocked');
  assert.equal(decideFit({ window: 8192, pressure: 0, messageTokens: 5000 }).action, 'too-large', 'a small window has less room for a message');
});

test('fit: a big message on a chat already near the compaction threshold summarises first, so the window is never come close to', () => {
  // 64K window: the Harness starts summarising at 40,142 (profile.mjs), but looks only before its NEXT step; a message that arrives now is not in that count.
  const at = { window: 65536, threshold: 40142, messageTokens: 14400 };
  const near = decideFit({ ...at, pressure: 39840 });
  assert.equal(near.action, 'summarize'); assert.equal(near.why, 'threshold');
  assert.equal(decideFit({ ...at, pressure: 15000 }).action, 'send', 'still under the threshold with the message on top');
  assert.equal(decideFit({ ...at, pressure: 13000, messageTokens: 30000 }).action, 'send', 'a chat that holds little beyond its first prompt has nothing worth summarising');
  assert.equal(decideFit({ window: 65536, messageTokens: 14400, pressure: 39840 }).action, 'send', 'no threshold given: only the window decides');
  assert.equal(decideFit({ ...at, pressure: 59048, messageTokens: 5000 }).why, undefined, 'a refusal-sized case is the old rule, not the threshold');
});

test('count: the engine\'s own number, or nothing when it cannot be had (then nothing is blocked)', async () => {
  const sent = [];
  const ok = async (url, init) => { sent.push([url, JSON.parse(init.body), init.headers]); return { ok: true, json: async () => ({ input_tokens: 70141 }) }; };
  assert.equal(await countTokens({ baseUrl: 'http://127.0.0.1:18500/v1', key: 'k', model: 'm', text: 'hello', fetchImpl: ok }), 70141);
  assert.equal(sent[0][0], 'http://127.0.0.1:18500/v1/messages/count_tokens');
  assert.deepEqual(sent[0][1].messages, [{ role: 'user', content: 'hello' }]);
  assert.equal(sent[0][2].authorization, 'Bearer k');
  assert.equal(await countTokens({ baseUrl: 'http://x/v1', text: 'a', fetchImpl: async () => ({ ok: false }) }), null);
  assert.equal(await countTokens({ baseUrl: 'http://x/v1', text: 'a', fetchImpl: async () => { throw new Error('refused'); } }), null);
  assert.equal(await countTokens({ baseUrl: 'http://x/v1', text: 'a', fetchImpl: async () => ({ ok: true, json: async () => ({}) }) }), null);
  assert.ok(COUNT_ABOVE_CHARS >= 1000);
});

test('the two ways out for a message that is too large: a file the assistant reads in parts, or a shortened preview', () => {
  const text = 'q=0x1f ok;'.repeat(11000);   // 110,000 characters
  const file = oversizedReplacement({ text, mode: 'file', tokens: 36000, window: 32768, path: 'C:\\work\\pasted-message.txt' });
  assert.match(file.message, /saved it as a file: C:\\work\\pasted-message\.txt/);
  assert.ok(file.message.length < 2000, 'the file message is short');
  const preview = oversizedReplacement({ text, mode: 'preview', tokens: 36000, window: 32768, pressure: 12000 });
  assert.ok(preview.message.length < text.length / 2, 'cut to what the window can hold');
  assert.match(preview.message, /characters from the middle are left out/);
  assert.ok(preview.message.startsWith('q=0x1f ok;') && preview.message.endsWith('q=0x1f ok;'), 'the start and the end are kept');
  const small = oversizedReplacement({ text: 'short', mode: 'preview', tokens: 5, window: 32768 });
  assert.equal(small.message, 'short');
  assert.equal(fill(FIT_TEXT.tooLarge, ['70,141', '65,536']), 'This message is about 70,141 tokens long, and this model can read about 65,536 at a time. It is too large to send as it is.');
});

test('a context failure is told in plain words, with the engine\'s text kept for Details and a code the page acts on', () => {
  const raw = '400: {"type":"invalid_request_error","message":"request exceeds the context window: prompt (85340 tokens) leaves no room to answer in the context (65536); requests are never truncated"}';
  assert.ok(isContextFailure({ message: raw }));
  assert.ok(isContextFailure({ code: 'CONTEXT_WINDOW_EXCEEDED', message: 'x' }));
  assert.ok(!isContextFailure({ message: 'fetch failed' }));
  const said = friendlyFailure({ message: raw });
  assert.equal(said, FIT_TEXT.full);
  assert.doesNotMatch(said, /\{|invalid_request|400/, 'no raw JSON in what the person reads');
  const projector = new TurnProjector();
  projector.consume({ type: 'event', event: { type: 'turn/end', data: { reason: { kind: 'error', error: { message: raw } } } } });
  const [frame] = projector.finish({ model: 'm', contextMax: 65536 });
  assert.equal(frame.type, 'error');
  assert.equal(frame.code, 'CONTEXT_WINDOW_EXCEEDED');
  assert.equal(frame.message, FIT_TEXT.full);
  assert.equal(frame.details, raw);
});

test('compaction shows in the chat: a status line while it runs and one plain line when it is done, with how full the chat was', () => {
  const projector = new TurnProjector({ window: 65536 });
  const statuses = [];
  projector.onStatus = u => statuses.push(u);
  const usage = { type: 'assistant-stream', frame: { type: 'chunk', attemptId: 'a', chunk: { type: 'usage', usage: { inputTokens: 58000, outputTokens: 1048 } } } };
  const live = projector.consume(usage);
  assert.deepEqual(live.find(f => f.type === 'context'), { type: 'context', contextTokens: 59048, contextMax: 65536 }, 'the page is told what the context holds after every request');
  assert.deepEqual(projector.consume({ type: 'event', event: { type: 'compaction/start', data: { compactionId: 'c' } } }), []);
  assert.equal(statuses.at(-1).code, 'compacting');
  assert.equal(statuses.at(-1).text, FIT_TEXT.summarizing);
  const done = projector.consume({ type: 'event', event: { type: 'compaction/end', data: { compactionId: 'c' } } });
  assert.equal(done.length, 1);
  assert.equal(done[0].key, FIT_TEXT.summarizedWas);
  assert.deepEqual(done[0].values, [90]);
  assert.match(done[0].text, /Older messages were summarized to make room\. The chat was 90% full\./);
  assert.equal(done[0].own, true);
  // A summary that failed says nothing here (the turn's own error does), and an end with no start (an older log) says nothing.
  const failed = new TurnProjector({ window: 65536 });
  failed.consume({ type: 'event', event: { type: 'compaction/start', data: {} } });
  assert.deepEqual(failed.consume({ type: 'event', event: { type: 'compaction/end', data: { error: 'x' } } }), []);
  assert.deepEqual(new TurnProjector().consume({ type: 'event', event: { type: 'compaction/end', data: {} } }), []);
});

test('a message too large for the window is not sent: the person is told plainly and nothing reaches the model', { skip }, async (t) => {
  const { worker, send } = await rig(t);
  const frames = await send(`SCRIPT:echo big ${dense.repeat(11000)}`);   // 110,000 characters: about 36,000 tokens in a 32,768 window
  const last = frames.at(-1);
  assert.equal(last.type, 'error');
  assert.equal(last.code, 'CONTEXT_MESSAGE_TOO_LARGE');
  assert.match(last.message, /^This message is about 3\d,\d{3} tokens long, and this model can read about 32,768 at a time/);
  assert.equal(last.key, FIT_TEXT.tooLarge);
  assert.equal(worker.requests.length, 0, 'no prompt was sent to the model');
  assert.equal(worker.counts.length, 1, 'the model server counted it once');
});

test('a message that fits only after a summary is sent after one, and the chat says so', { skip }, async (t) => {
  const { worker, send } = await rig(t);
  for (let i = 0; i < 2; i++) assert.equal((await send(`SCRIPT:echo part${i} ${dense.repeat(900)}`)).at(-1).type, 'done');
  const before = worker.requests.length;
  const frames = await send(`SCRIPT:echo then ${dense.repeat(4000)}`);   // 40,000 characters: about 13,000 tokens on top of about 18,000
  assert.equal(frames.at(-1).type, 'done', JSON.stringify(frames.at(-1)).slice(0, 300));
  const note = frames.find(f => f.type === 'delta' && f.noteReason === 'compacted');
  assert.ok(note, 'a line says the older messages were summarized');
  assert.match(note.text, /summarized to make room/);
  const later = worker.requests.slice(before);
  const firstAnswer = later.findIndex(r => !r.summary);
  assert.ok(later.slice(0, firstAnswer).some(r => r.summary), 'the summary was asked for before the message was sent');
  assert.equal(worker.requests.filter(r => r.refused).length, 0, 'nothing was refused');
});

test('/compact summarises the older messages with the real figures, and says so plainly when there is nothing to summarise', { skip }, async (t) => {
  const { worker, bridge, ws } = await rig(t);
  const run = async (line) => { const frames = []; await bridge.compactCommand({ chatId: 'c', message: line, workspacePath: ws, mode: 'bypass', choice: null, onFrame: f => frames.push(f) }); return frames; };
  const fresh = await run('/compact');
  assert.equal(fresh.find(f => f.type === 'delta').key, FIT_TEXT.noChatYet, 'a chat that has not started has nothing to summarise');
  const frames = [];
  for (let i = 0; i < 3; i++) { await bridge.runTurn({ chatId: 'c', message: `SCRIPT:echo r${i} ${dense.repeat(700)}`, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) }); }
  const done = await run('/compact');
  const said = done.find(f => f.type === 'delta');
  assert.equal(said.key, FIT_TEXT.summarized, 'the figures are the measured ones: ' + said.text);
  const [was, now] = said.values;
  assert.ok(was > now && now > 0 && was <= 100, `the chat was ${was}% full and is now ${now}%`);
  assert.equal(done.at(-1).type, 'done');
  assert.ok(worker.requests.some(r => r.summary));
  const again = await run('/compact');
  assert.ok([FIT_TEXT.nothingToSummarize, FIT_TEXT.nothingFull, FIT_TEXT.summarized].includes(again.find(f => f.type === 'delta').key), 'a second /compact is answered in plain words');
});
