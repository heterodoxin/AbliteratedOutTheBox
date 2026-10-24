// The agent's keyless web search against a MOCKED, rate-limited DuckDuckGo: spaced requests, a session cache, a cooldown that doubles,
// Bing as the second source, and one plain sentence (with the place to add a key) when every source refuses.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createDuckDuckGoSearch, BLOCKED_MESSAGE, LIMITED_SENTENCE, UNREACHABLE_MESSAGE } from '../server/web-search-ddg.mjs';
import { bingTarget, parseBingResults, bingLooksBlocked } from '../server/web-search-bing.mjs';

const BING = readFileSync(new URL('./fixtures/bing-results.html', import.meta.url), 'utf8');
const DDG = `<div class="result"><h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fddg.example%2Fa">DDG one</a></h2>
<a class="result__snippet" href="x">First snippet</a></div>
<div class="result"><h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fddg.example%2Fb">DDG two</a></h2></div>`;
const ANOMALY = '<html><body><div class="anomaly-modal">Unfortunately, bots use DuckDuckGo too.</div></body></html>';

/** A fake clock whose sleep moves time, and a fake web: `script(url, n)` decides each answer. */
function world(script, { random = () => 0.5 } = {}) {
  let t = 5_000_000;
  const calls = [];
  const fetch = async (url, init) => {
    const at = t;
    const call = { url: String(url), method: init?.method ?? 'GET', at, headers: init?.headers ?? {} };
    calls.push(call);
    const out = await script(call, calls.length);
    if (out instanceof Error) throw out;
    return out;
  };
  const search = createDuckDuckGoSearch({ fetch, now: () => t, sleep: async ms => { t += ms; }, random });
  return { search, calls, advance: ms => { t += ms; }, clock: () => t, hostCalls: host => calls.filter(c => c.url.includes(host)) };
}
const page = (body, status = 200, headers = {}) => new Response(body, { status, headers });
const duck = url => /duckduckgo\.com/.test(url);

test('requests to one source are spaced 2.5 s or more apart, never in a burst', async () => {
  const w = world(() => page(DDG));
  for (const q of ['one', 'two', 'three', 'four']) await w.search.search({ query: q });
  const at = w.calls.map(c => c.at);
  assert.equal(at.length, 4);
  for (let i = 1; i < at.length; i++) assert.ok(at[i] - at[i - 1] >= 2500, `gap ${i}: ${at[i] - at[i - 1]} ms`);
});

test('the request looks like the page\'s own form post: browser headers, Referer, Origin, no hidden tricks', async () => {
  const w = world(() => page(DDG));
  await w.search.search({ query: 'strata' });
  const [c] = w.calls;
  assert.equal(c.method, 'POST');
  assert.match(c.headers['user-agent'], /Mozilla\/5\.0 .*Chrome\/\d+/);
  assert.equal(c.headers.referer, 'https://html.duckduckgo.com/');
  assert.equal(c.headers.origin, 'https://html.duckduckgo.com');
  assert.equal(c.headers['sec-fetch-mode'], 'navigate');
  assert.match(c.headers['accept-language'], /en/);
});

test('the same question is answered from the session cache, even while DuckDuckGo is refusing', async () => {
  let refuse = false;
  const w = world(call => (duck(call.url) ? (refuse ? page(ANOMALY, 202) : page(DDG)) : page(BING)));
  const a = await w.search.search({ query: 'Strata  Engine' });
  assert.equal(a.sources.length, 2);
  refuse = true;
  const b = await w.search.search({ query: 'strata engine' });
  assert.deepEqual(b.sources, a.sources);
  assert.equal(w.calls.length, 1, 'no second request');
  w.advance(20 * 60_000);
  assert.equal((await w.search.search({ query: 'STRATA ENGINE' })).sources.length, 2, 'still cached after 20 minutes');
  assert.equal(w.calls.length, 1);
  w.advance(11 * 60_000);
  await w.search.search({ query: 'strata engine' });
  assert.ok(w.calls.length > 1, 'asked again after the cache is 30 minutes old');
});

test('DuckDuckGo refuses (HTTP 202): Bing answers, DuckDuckGo is left alone for a minute, and its lite page is not tried', async () => {
  const w = world(call => (duck(call.url) ? page(ANOMALY, 202) : page(BING)));
  const first = await w.search.search({ query: 'node release notes', maxResults: 3 });
  assert.equal(first.sources.length, 3);
  assert.equal(first.sources[0].url, 'https://github.com/Niko1221/Strata');
  assert.equal(w.hostCalls('html.duckduckgo.com').length, 1);
  assert.equal(w.hostCalls('lite.duckduckgo.com').length, 0, 'a refusal is not followed by the lite page');
  assert.equal(w.hostCalls('bing.com').length, 1);
  // The next question goes straight to Bing: DuckDuckGo is cooling down.
  await w.search.search({ query: 'another question' });
  assert.equal(w.hostCalls('duckduckgo.com').length, 1, 'no request to DuckDuckGo during the cooldown');
  assert.equal(w.hostCalls('bing.com').length, 2);
  assert.ok(w.search.status().duckduckgo.blockedForMs > 0);
  // After the cooldown one probe goes out again.
  w.advance(61_000);
  await w.search.search({ query: 'a third question' });
  assert.equal(w.hostCalls('html.duckduckgo.com').length, 2);
});

test('every source refuses: one plain sentence with the place to add a key; later searches make no request at all', async () => {
  const w = world(call => page(duck(call.url) ? ANOMALY : '<html>captcha</html>', duck(call.url) ? 202 : 429));
  const error = await w.search.search({ query: 'hello world' }).catch(e => e);
  assert.equal(error.code, 'WEB_RATE_LIMITED');
  assert.equal(error.message, BLOCKED_MESSAGE);
  assert.ok(error.message.startsWith(LIMITED_SENTENCE));
  assert.match(LIMITED_SENTENCE, /^Web search is limited right now\. Try again in a minute, or add a search key \(Exa, Perplexity or DeepSeek\) in Settings → Tools → Web search\.$/);
  assert.match(error.message, /Do not retry this search/, 'the model is told to stop');
  const used = w.calls.length;
  assert.equal(used, 2, 'one request per source, nothing more');
  for (const q of ['again', 'and again', 'once more']) {
    const e = await w.search.search({ query: q }).catch(x => x);
    assert.equal(e.code, 'WEB_RATE_LIMITED', q);
  }
  assert.equal(w.calls.length, used, 'a model that keeps asking does not extend the block');
});

test('the cooldown doubles each time a source refuses again, and a Retry-After header is honoured', async () => {
  const w = world(call => (duck(call.url) ? page(ANOMALY, 202) : page('<html>captcha</html>', 429)));
  await w.search.search({ query: 'q1' }).catch(() => {});
  const first = w.search.status().duckduckgo.blockedForMs;
  assert.ok(first > 55_000 && first <= 60_000, `first cooldown ${first}`);
  w.advance(61_000);
  await w.search.search({ query: 'q2' }).catch(() => {});
  const second = w.search.status().duckduckgo.blockedForMs;
  assert.ok(second > 115_000 && second <= 120_000, `second cooldown ${second}`);
  w.advance(121_000);
  await w.search.search({ query: 'q3' }).catch(() => {});
  assert.ok(w.search.status().duckduckgo.blockedForMs > 235_000);
  const r = world(call => (duck(call.url) ? page(ANOMALY, 429, { 'retry-after': '300' }) : page('<html>captcha</html>', 429)));
  await r.search.search({ query: 'x' }).catch(() => {});
  assert.ok(r.search.status().duckduckgo.blockedForMs > 295_000, 'Retry-After: 300');
});

test('a recovered source resets the cooldown', async () => {
  let refuse = true;
  const w = world(call => (duck(call.url) ? (refuse ? page(ANOMALY, 202) : page(DDG)) : page('<html>captcha</html>', 429)));
  await w.search.search({ query: 'a' }).catch(() => {});
  w.advance(61_000);
  refuse = false;
  assert.equal((await w.search.search({ query: 'b' })).sources[0].url, 'https://ddg.example/a');
  assert.equal(w.search.status().duckduckgo.strikes, 0);
  assert.equal(w.search.status().duckduckgo.blockedForMs, 0);
});

test('a timeout or a server error gets one retry; two in a row moves on to the next source; nothing reachable says so', async () => {
  let n = 0;
  const flaky = world(call => (duck(call.url) ? (++n === 1 ? page('oops', 503) : page(DDG)) : page(BING)));
  assert.equal((await flaky.search.search({ query: 'q' })).sources[0].url, 'https://ddg.example/a');
  assert.equal(flaky.hostCalls('duckduckgo.com').length, 2);
  const down = world(call => (duck(call.url) ? new TypeError('fetch failed') : page(BING)));
  const viaBing = await down.search.search({ query: 'q' });
  assert.equal(viaBing.sources.length, 5, 'DuckDuckGo unreachable twice: Bing answers');
  const none = world(() => new TypeError('fetch failed'));
  const e = await none.search.search({ query: 'q' }).catch(x => x);
  assert.equal(e.message, UNREACHABLE_MESSAGE);
  assert.notEqual(e.code, 'WEB_RATE_LIMITED');
});

test('a search with no hits is an answer; a cancelled search is cancelled', async () => {
  const w = world(() => page('<html><body>No results found.</body></html>'));
  assert.deepEqual(await w.search.search({ query: 'zzzz qqqq' }), { sources: [], truncated: false });
  const ctrl = new AbortController(); ctrl.abort();
  const e = await w.search.search({ query: 'later' }, ctrl.signal).catch(x => x);
  assert.equal(e.code, 'WEB_ABORTED');
});

test('too many searches in a minute are held back by the app, not sent to the sources', async () => {
  const w = world(() => page(DDG));
  let held = null;
  for (let i = 0; i < 12 && !held; i++) held = await w.search.search({ query: `question ${i}` }).then(() => null, e => e);
  assert.ok(held, 'the ninth search in a minute is held back');
  assert.equal(held.code, 'WEB_RATE_LIMITED');
  assert.ok(w.hostCalls('duckduckgo.com').length <= 8);
});

test('Bing: the real result page is read (redirect links decoded, snippets kept); a challenge page is a refusal; ordinary pages are not', () => {
  const rows = parseBingResults(BING);
  assert.equal(rows.length, 5);
  assert.ok(rows.every(r => /^https:\/\//.test(r.url) && !r.url.includes('bing.com')), 'real addresses, not Bing redirects');
  assert.equal(rows[1].url, 'https://github.com/Niko1221/Strata/releases');
  assert.ok(rows[0].snippet.length > 20);
  assert.equal(bingTarget('https://www.bing.com/ck/a?!&&p=1&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9hP3g9MQ&ntb=1'), 'https://example.com/a?x=1');
  assert.equal(bingTarget('https://www.bing.com/images/search?q=x'), null);
  assert.equal(bingTarget('https://example.org/direct'), 'https://example.org/direct');
  assert.equal(bingLooksBlocked(429, ''), true);
  assert.equal(bingLooksBlocked(200, '<html><form>captcha</form></html>'), true);
  assert.equal(bingLooksBlocked(200, BING), false);
  assert.deepEqual(parseBingResults('<html>nothing here</html>'), []);
});
