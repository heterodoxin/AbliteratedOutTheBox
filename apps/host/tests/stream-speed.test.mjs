import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { proxyCompletion } from '../server/inference.mjs';

/* The speed shown while an answer streams must be the decode speed of everything generated, reasoning included, from the first
   token (the owner saw 4-5 tok/s in chat while the engine log said 48-57): the stand-in engine streams 90 reasoning tokens at about
   40 a second, a 4 s "prompt" before the first one, then a short answer. */
function engine({ promptMs, reasoning, answer, perSecond }) {
  return createServer((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    const send = obj => res.write(`data: ${JSON.stringify(obj)}\n\n`);
    let sent = 0;
    setTimeout(function tick() {
      if (sent < reasoning + answer) {
        send({ choices: [{ delta: sent < reasoning ? { reasoning_content: 'tok ' } : { content: 'ans ' } }] });
        sent++;
        return setTimeout(tick, 1000 / perSecond);
      }
      send({ choices: [], usage: { prompt_tokens: 20, completion_tokens: reasoning + answer } });
      res.write('data: [DONE]\n\n'); res.end();
    }, promptMs);
  });
}

test('live speed counts reasoning tokens over decode time, not the answer over the whole wait', async () => {
  const server = engine({ promptMs: 4000, reasoning: 90, answer: 10, perSecond: 40 });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const frames = [];
  try {
    const result = await proxyCompletion({ baseUrl: `http://127.0.0.1:${server.address().port}/v1`, message: 'hi', onFrame: f => frames.push(f), engine: { beginReading() {}, beginThinking() {}, beginGenerating() {} } });
    const stats = frames.filter(f => f.type === 'stats');
    assert.ok(stats.length >= 2, 'live frames while it streams');
    const last = stats.at(-1);
    assert.ok(last.tps > 22 && last.tps < 80, `live tok/s ${last.tps} is the decode rate, not tokens over the wait including the prompt (~14)`);
    assert.ok(last.tokensOut >= 50, 'reasoning tokens are counted');
    assert.equal(result.tokensOut, 100);
    assert.ok(result.tps > 22 && result.tps < 80, `final tok/s ${result.tps}`);
  } finally { server.close(); }
});
