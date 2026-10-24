/**
 * The second keyless source for the agent's web search: Bing's plain result page, used only when DuckDuckGo refuses.
 *
 * It is read the way a browser reads it (an ordinary User-Agent, the page's own request shape, nothing hidden) and is held to the same
 * manners as DuckDuckGo (see web-search-ddg.mjs): one request at a time, spaced out, a cooldown after a refusal. Results are not ranked or
 * filtered here. Pure parsing lives in this file so it can be tested on saved pages.
 */

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', '#39': "'", '#x27': "'", '#x2F': '/', '#183': '·' };
const decode = text => String(text).replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (whole, key) => {
  if (key[0] === '#') {
    const code = key[1].toLowerCase() === 'x' ? parseInt(key.slice(2), 16) : parseInt(key.slice(1), 10);
    return Number.isFinite(code) && code > 0 && code < 0x110000 ? String.fromCodePoint(code) : whole;
  }
  return ENTITIES[key.toLowerCase()] ?? whole;
});
const plain = html => decode(String(html).replace(/<[^>]*>/g, '')).replace(/\s+/g, ' ').trim();

export const BING_URL = 'https://www.bing.com/search';

/** Bing wraps every result link in its own redirect (`/ck/a?...&u=a1<base64url of the real address>`). */
export function bingTarget(href) {
  const value = decode(href).trim();
  try {
    const url = new URL(value, 'https://www.bing.com/');
    if (/(^|\.)bing\.com$/i.test(url.hostname)) {
      const u = url.searchParams.get('u');
      if (u && /^a1/.test(u)) {
        const target = Buffer.from(u.slice(2).replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('utf8');
        const real = new URL(target);
        return /^https?:$/.test(real.protocol) ? real.href : null;
      }
      return null;                                       // Bing's own page (images, maps, "more results")
    }
    return /^https?:$/.test(url.protocol) ? url.href : null;
  } catch { return null; }
}

/** The result list of the page. Tolerant: a result needs a link and a title; the snippet is optional. */
export function parseBingResults(html) {
  const out = [];
  const seen = new Set();
  const blocks = String(html).split(/<li\b[^>]*\bclass="[^"]*\bb_algo\b[^"]*"/i).slice(1);
  for (const block of blocks) {
    const end = block.search(/<li\b[^>]*\bclass="[^"]*\bb_(?:algo|ans|ad|pag)\b/i);
    const body = end > 0 ? block.slice(0, end) : block;
    const h2 = /<h2\b[^>]*>\s*<a\b([^>]*)>([\s\S]*?)<\/a>/i.exec(body);
    if (!h2) continue;
    const href = /\bhref="([^"]*)"/i.exec(h2[1])?.[1];
    const url = href ? bingTarget(href) : null;
    const title = plain(h2[2]);
    if (!url || !title || seen.has(url)) continue;
    const snippetMatch = /<p\b[^>]*class="[^"]*\bb_lineclamp\d*\b[^"]*"[^>]*>([\s\S]*?)<\/p>/i.exec(body)
      ?? /<div\b[^>]*class="[^"]*\bb_caption\b[^"]*"[^>]*>[\s\S]*?<p\b[^>]*>([\s\S]*?)<\/p>/i.exec(body);
    const snippet = snippetMatch ? plain(snippetMatch[1]) : '';
    seen.add(url);
    out.push({ url, title, ...(snippet ? { snippet } : {}) });
  }
  return out;
}

/** Bing serves a challenge page, or a refusal, instead of results. A page with results is never "blocked". */
export function bingLooksBlocked(status, html) {
  if (status === 403 || status === 429 || status === 503) return true;
  const text = String(html).slice(0, 80000);
  if (/\bclass="[^"]*\bb_algo\b/.test(text)) return false;
  return /captcha|unusual traffic|are you a robot|verify you are (?:a )?human|\bcf-chl|challenge-platform/i.test(text);
}
