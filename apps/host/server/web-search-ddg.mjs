/**
 * Web search with no account and no key: DuckDuckGo's own web-page search results, with Bing's plain result page as the second source.
 *
 * The Harness has three search backends and all of them need a paid service key. This is the fourth: it asks DuckDuckGo's plain
 * HTML search page the way a browser does and reads the result list out of the page. It is the default for the agent's
 * `web_search` tool; the person can switch to a keyed service in Settings -> Tools.
 *
 * Being polite is part of the design, because DuckDuckGo refuses a program that asks too fast (2026-10-18, a Linux box: two searches in
 * 1.2 s were answered, the third got HTTP 202 "anomaly" and every DuckDuckGo page then refused for minutes):
 *   - an ordinary browser User-Agent and the page's own request shape (Referer, Origin, Sec-Fetch-*; nothing hidden);
 *   - one request at a time per source, 2.5 to 4 s apart (random, so it does not look like a clock), at most 8 a minute;
 *   - the same question is answered from a session cache (30 minutes), so a model that repeats itself costs nothing;
 *   - when a source refuses (HTTP 202/403/429 or a challenge page) it is left alone for a cooldown that doubles every time it refuses
 *     again (1, 2, 4 ... up to 10 minutes; a Retry-After header is honoured) instead of being asked again, which only extends a block;
 *   - a refused DuckDuckGo page is not followed by the lite page (same refusal, one more strike): the next source is tried instead;
 *   - a timeout or a server error gets one retry after a short pause, then the next source;
 *   - when every source refuses the answer is one plain sentence that also tells the model not to retry (see LIMITED_MESSAGE).
 * Results come from DuckDuckGo or Bing; they are not ranked or filtered here.
 *
 * Nothing in this file touches the Harness, so it can be tested alone. `strata-search.mjs` is the plugin that registers it.
 */
import { BING_URL, bingLooksBlocked, parseBingResults } from './web-search-bing.mjs';

export const DDG_PROVIDER_ID = 'strata-duckduckgo';

const HTML_URL = 'https://html.duckduckgo.com/html/';
const LITE_URL = 'https://lite.duckduckgo.com/lite/';
const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';
const CACHE_MS = 30 * 60 * 1000;
const CACHE_SIZE = 120;
const GAP_MS = 2500;
const JITTER_MS = 1500;
const PER_MINUTE = 8;
const COOLDOWN_MS = 60_000;
const COOLDOWN_MAX_MS = 10 * 60_000;
const RETRY_PAUSE_MS = 1500;

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', '#39': "'", '#x27': "'", '#x2F': '/' };
const decode = text => String(text).replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (whole, key) => {
  if (key[0] === '#') {
    const code = key[1].toLowerCase() === 'x' ? parseInt(key.slice(2), 16) : parseInt(key.slice(1), 10);
    return Number.isFinite(code) && code > 0 && code < 0x110000 ? String.fromCodePoint(code) : whole;
  }
  return ENTITIES[key.toLowerCase()] ?? whole;
});
const plain = html => decode(String(html).replace(/<[^>]*>/g, '')).replace(/\s+/g, ' ').trim();

/** DuckDuckGo wraps every result link in its own redirect; the real address is in `uddg`. */
export function realUrl(href) {
  let value = decode(href).trim();
  if (value.startsWith('//')) value = `https:${value}`;
  try {
    const url = new URL(value, 'https://duckduckgo.com/');
    const target = url.searchParams.get('uddg');
    if (target) return new URL(target).href;
    if (url.hostname.endsWith('duckduckgo.com')) return null;      // an internal link, an ad or a "more results" link
    return /^https?:$/.test(url.protocol) ? url.href : null;
  } catch { return null; }
}

/** The result list of the HTML page. Tolerant: a result needs a link and a title; the snippet is optional. */
export function parseHtmlResults(html) {
  const out = [];
  const links = [...String(html).matchAll(/<a\b[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*>/gi)];
  for (let i = 0; i < links.length; i++) {
    const open = links[i];
    const start = open.index + open[0].length;
    const next = links[i + 1]?.index ?? html.length;
    const chunk = html.slice(start, next);
    const close = chunk.indexOf('</a>');
    if (close < 0) continue;
    const href = /href="([^"]*)"/i.exec(open[0])?.[1];
    const url = href ? realUrl(href) : null;
    const title = plain(chunk.slice(0, close));
    if (!url || !title) continue;
    // An ad is a result block whose own opening tag says so (the block that holds this link, not the one before it).
    const block = html.lastIndexOf('<div class="result', open.index);
    if (block >= 0 && /result--ad|result__ad/.test(html.slice(block, html.indexOf('>', block) + 1))) continue;
    const snippetMatch = /<(?:a|td|div)\b[^>]*class="[^"]*\bresult__snippet\b[^"]*"[^>]*>([\s\S]*?)<\/(?:a|td|div)>/i.exec(chunk);
    const snippet = snippetMatch ? plain(snippetMatch[1]) : '';
    out.push({ url, title, ...(snippet ? { snippet } : {}) });
  }
  return dedupe(out);
}

/** The result list of the lite page (a table), used when the HTML page gives nothing. */
export function parseLiteResults(html) {
  const out = [];
  const links = [...String(html).matchAll(/<a\b[^>]*class=['"][^'"]*\bresult-link\b[^'"]*['"][^>]*>/gi)];
  for (let i = 0; i < links.length; i++) {
    const open = links[i];
    const start = open.index + open[0].length;
    const next = links[i + 1]?.index ?? html.length;
    const chunk = html.slice(start, next);
    const close = chunk.indexOf('</a>');
    if (close < 0) continue;
    const href = /href=['"]([^'"]*)['"]/i.exec(open[0])?.[1];
    const url = href ? realUrl(href) : null;
    const title = plain(chunk.slice(0, close));
    if (!url || !title) continue;
    const snippetMatch = /<td\b[^>]*class=['"][^'"]*\bresult-snippet\b[^'"]*['"][^>]*>([\s\S]*?)<\/td>/i.exec(chunk);
    const snippet = snippetMatch ? plain(snippetMatch[1]) : '';
    out.push({ url, title, ...(snippet ? { snippet } : {}) });
  }
  return dedupe(out);
}

function dedupe(rows) {
  const seen = new Set();
  return rows.filter(r => !seen.has(r.url) && seen.add(r.url));
}

/** DuckDuckGo asks a program to prove it is a person by serving a challenge page instead of results. */
export function looksBlocked(status, html) {
  if (status === 202 || status === 403 || status === 418 || status === 429) return true;
  return /anomaly-modal|anomaly\.js|unusual traffic|are you a (?:robot|human)|captcha|challenge-form/i.test(String(html).slice(0, 60000))
    && !/class="[^"]*\bresult__a\b/.test(html);
}


export class SearchUnavailable extends Error {
  constructor(message, code = 'WEB_PROVIDER_ERROR', options) { super(message, options); this.name = 'WebError'; this.code = code; }
}

/** The one plain sentence for the person. The words "Settings → Tools → Web search" are where the key goes in the app. */
export const LIMITED_SENTENCE = 'Web search is limited right now. Try again in a minute, or add a search key (Exa, Perplexity or DeepSeek) in Settings → Tools → Web search.';
/** What the tool answers: the sentence, then what the model should do (it must not keep searching). */
export const BLOCKED_MESSAGE = `${LIMITED_SENTENCE} Do not retry this search: go on with what you know and tell the person, in one plain sentence, that web search was limited.`;
export const UNREACHABLE_MESSAGE = 'The search could not reach the internet. Check the connection and try again. Do not retry this search now: go on with what you know and say that search was unavailable.';

/** One request at a time, spaced out with a little randomness, and at most PER_MINUTE a minute. */
class Pacer {
  constructor({ now, sleep, random, gap = GAP_MS, jitter = JITTER_MS, perMinute = PER_MINUTE }) {
    Object.assign(this, { now, sleep, random, gap, jitter, perMinute });
    this.starts = []; this.last = 0; this.chain = Promise.resolve();
  }
  turn(signal) {
    const run = this.chain.then(async () => {
      const spacing = this.gap + Math.floor(this.random() * this.jitter);
      for (;;) {
        if (signal?.aborted) throw new SearchUnavailable('The search was cancelled.', 'WEB_ABORTED');
        const t = this.now();
        while (this.starts.length && t - this.starts[0] > 60_000) this.starts.shift();
        const wait = this.last ? Math.max(spacing - (t - this.last), this.starts.length >= this.perMinute ? 60_000 - (t - this.starts[0]) : 0) : 0;
        if (wait <= 0) break;
        if (wait > 30_000) throw new SearchUnavailable(LIMITED_SENTENCE, 'WEB_RATE_LIMITED');
        await this.sleep(wait);
      }
      this.last = this.now();
      this.starts.push(this.last);
    });
    this.chain = run.catch(() => {});
    return run;
  }
}

const REFUSED = Symbol('refused');
const retryAfterMs = res => {
  const v = Number(res?.headers?.get?.('retry-after'));
  return Number.isFinite(v) && v > 0 ? Math.min(v * 1000, COOLDOWN_MAX_MS) : 0;
};
const serverError = status => status >= 500 && status < 600;
const queryKey = query => query.toLowerCase().replace(/\s+/g, ' ').trim();

/**
 * @param {{fetch?: typeof fetch, now?: () => number, sleep?: (ms: number) => Promise<void>, random?: () => number}} [deps]
 */
export function createDuckDuckGoSearch({ fetch: fetchImpl = globalThis.fetch, now = Date.now, sleep = ms => new Promise(r => setTimeout(r, ms)), random = Math.random } = {}) {
  const cache = new Map();

  /** A source: its pacing, and what to do after it refuses. */
  const source = (id, name) => ({ id, name, pacer: new Pacer({ now, sleep, random }), blockedUntil: 0, strikes: 0, refusals: 0 });
  const ddg = source('duckduckgo', 'DuckDuckGo');
  const bing = source('bing', 'Bing');
  const refuse = (s, res) => {
    s.strikes += 1; s.refusals += 1;
    s.blockedUntil = now() + Math.max(retryAfterMs(res), Math.min(COOLDOWN_MS * 2 ** (s.strikes - 1), COOLDOWN_MAX_MS));
  };
  const accepted = s => { s.strikes = 0; s.blockedUntil = 0; };

  const headers = (origin, extra = {}) => ({
    'user-agent': USER_AGENT, accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 'accept-language': 'en-US,en;q=0.9',
    'upgrade-insecure-requests': '1', 'sec-fetch-dest': 'document', 'sec-fetch-mode': 'navigate', 'sec-fetch-site': 'same-origin', 'sec-fetch-user': '?1',
    ...(origin ? { referer: `${origin}/`, origin } : {}), ...extra,
  });

  async function get(url, init, signal) {
    const timeout = AbortSignal.timeout(15_000);
    const res = await fetchImpl(url, { redirect: 'follow', ...init, signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
    return { res, status: res.status, html: await res.text() };
  }

  /** One page request with one retry after a pause for a timeout, a network error or a server error. Throws SearchUnavailable only for cancel. */
  async function ask(s, makeRequest, signal) {
    let failure;
    for (let attempt = 0; attempt < 2; attempt++) {
      await s.pacer.turn(signal);
      try {
        const page = await get(...makeRequest(), signal);
        if (!serverError(page.status)) return page;
        failure = new Error(`HTTP ${page.status}`);
      } catch (error) {
        if (signal?.aborted) throw new SearchUnavailable('The search was cancelled.', 'WEB_ABORTED', { cause: error });
        failure = error;
      }
      if (attempt === 0) await sleep(RETRY_PAUSE_MS + Math.floor(random() * 1000));
    }
    return { failure };
  }

  async function searchDuckDuckGo(query, signal) {
    const form = new URLSearchParams({ q: query, b: '', kl: 'wt-wt', df: '' }).toString();
    const post = (url, origin) => () => [url, { method: 'POST', headers: headers(origin, { 'content-type': 'application/x-www-form-urlencoded' }), body: form }];
    let reachedOnce = false;
    for (const [url, origin, parse] of [[HTML_URL, 'https://html.duckduckgo.com', parseHtmlResults], [LITE_URL, 'https://lite.duckduckgo.com', parseLiteResults]]) {
      const page = await ask(ddg, post(url, origin), signal);
      if (page.failure) return { failure: page.failure };
      reachedOnce = true;
      // A refusal on the first page is a refusal for the whole address: the lite page would only add one more strike.
      if (looksBlocked(page.status, page.html)) { refuse(ddg, page.res); return { refused: true }; }
      const sources = parse(page.html);
      if (sources.length) { accepted(ddg); return { sources }; }
    }
    if (reachedOnce) accepted(ddg);
    return { sources: [] };
  }

  async function searchBing(query, signal) {
    const url = `${BING_URL}?${new URLSearchParams({ q: query, setlang: 'en' })}`;
    const page = await ask(bing, () => [url, { method: 'GET', headers: headers(null) }], signal);
    if (page.failure) return { failure: page.failure };
    if (bingLooksBlocked(page.status, page.html)) { refuse(bing, page.res); return { refused: true }; }
    accepted(bing);
    return { sources: parseBingResults(page.html) };
  }

  return {
    id: DDG_PROVIDER_ID,
    available: () => typeof fetchImpl === 'function',
    /** What happened so far, for tests and for the status line in the app's log. */
    status: () => ({ duckduckgo: { strikes: ddg.strikes, refusals: ddg.refusals, blockedForMs: Math.max(0, ddg.blockedUntil - now()) }, bing: { strikes: bing.strikes, refusals: bing.refusals, blockedForMs: Math.max(0, bing.blockedUntil - now()) }, cached: cache.size }),
    async search(request, signal) {
      const query = String(request.query ?? '').trim().slice(0, 400);
      if (!query) throw new SearchUnavailable('Type what to search for.', 'WEB_PROVIDER_ERROR');
      const key = queryKey(query);
      const want = request.maxResults ?? 10;
      const hit = cache.get(key);
      if (hit && now() - hit.at < CACHE_MS) return { sources: hit.sources.slice(0, want), truncated: false };
      let limited = false, unreachable = false;
      for (const [s, run] of [[ddg, searchDuckDuckGo], [bing, searchBing]]) {
        if (s.blockedUntil > now()) { limited = true; continue; }    // asked again too soon: that only extends a block
        const result = await run(query, signal);
        if (result.refused) { limited = true; continue; }
        if (result.failure) { unreachable = true; continue; }
        if (!result.sources.length) return { sources: [], truncated: false };      // a real search with no results is an answer, not an error
        cache.set(key, { at: now(), sources: result.sources });
        if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value);
        return { sources: result.sources.slice(0, want), truncated: false };
      }
      if (limited) throw new SearchUnavailable(BLOCKED_MESSAGE, 'WEB_RATE_LIMITED');
      throw new SearchUnavailable(unreachable ? UNREACHABLE_MESSAGE : BLOCKED_MESSAGE, unreachable ? 'WEB_PROVIDER_ERROR' : 'WEB_RATE_LIMITED');
    },
  };
}
