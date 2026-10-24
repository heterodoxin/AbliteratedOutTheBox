/**
 * Does a message fit the model's window? Asked before it is sent, with the model server's own count, so a message the server would refuse
 * is never read for half a minute first (2026-10-22: a 103 KB paste was 70,141 tokens in a 65,536-token window, and the 400 came back after
 * the whole prompt had been read). No I/O except the one count request; everything else is arithmetic the tests call directly.
 *
 * Three answers: send it; summarise the chat first and then send it (the message fits once the older messages are a summary); or it can
 * never fit, and the person is offered the two ways out that do (the text as a file the assistant reads in parts, or a shortened preview).
 */

/** What a new chat already holds before the person's first word: the system prompt and the tool list. Measured 10,510 by the Harness's price. */
export const BASE_TOKENS = 12000;
/** The least room a reply needs after the prompt: an eighth of the window, at most this. */
export const REPLY_FLOOR = 4096;
/** A message shorter than this many characters cannot matter to a window of 8K or more, so it is not even counted. */
export const COUNT_ABOVE_CHARS = 3000;

export const replyRoom = window => Math.max(512, Math.min(REPLY_FLOOR, Math.floor(window / 8)));

/**
 * @param {{window:number, pressure?:number, base?:number, messageTokens:number, threshold?:number}} input
 *   `pressure` is what the chat holds now (measured), `base` what an empty chat holds, `messageTokens` the new message, `threshold` where the
 *   Harness starts summarising by itself. The Harness looks at that only before its NEXT step, so a big message that arrives now would otherwise
 *   be sent on top of a chat already near the threshold (2026-10-22: 54K of 64K held by the time it was looked at): summarise first.
 * @returns {{action:'send'|'summarize'|'too-large', room:number, needs:number}}
 */
export function decideFit({ window, pressure = 0, base: given = BASE_TOKENS, messageTokens, threshold = 0 }) {
  const reply = replyRoom(window);
  const base = Math.min(given, Math.floor(window * 0.6));   // on a small window the first prompt cannot be what the window is made of
  // A chat holds at least what an empty one does (the system prompt and the tools), so a new chat is not counted as empty.
  const needs = Math.max(pressure, base) + messageTokens + reply;
  if (!(window > 0) || !(messageTokens >= 0)) return { action: 'send', room: Infinity, needs: 0 };
  if (needs <= window) {
    const worthIt = pressure > base + window * 0.1;   // older messages worth summarising: more than a tenth of the window beyond the first prompt
    if (threshold > 0 && worthIt && pressure + messageTokens > threshold) return { action: 'summarize', room: window - needs, needs, why: 'threshold' };
    return { action: 'send', room: window - needs, needs };
  }
  // Not as it stands. Does the message fit in an otherwise empty chat (only the system prompt and the tools before it)?
  if (base + messageTokens + reply <= window) return { action: 'summarize', room: window - (base + messageTokens + reply), needs };
  return { action: 'too-large', room: window - (base + messageTokens + reply), needs };
}

/**
 * The model server's count of a text (its tokenizer and chat template), through its Anthropic-style count endpoint.
 * Returns the number, or null when it cannot be had (an old engine, a server that is busy, a timeout): then nothing is blocked.
 */
export async function countTokens({ baseUrl, key = '', model = '', text, fetchImpl = fetch, timeoutMs = 8000 }) {
  try {
    const root = String(baseUrl ?? '').replace(/\/v1\/?$/, '').replace(/\/$/, '');
    const res = await fetchImpl(`${root}/v1/messages/count_tokens`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(key ? { authorization: `Bearer ${key}`, 'x-api-key': key } : {}) },
      body: JSON.stringify({ model, messages: [{ role: 'user', content: String(text) }] }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    const n = Number((await res.json())?.input_tokens);
    return Number.isFinite(n) && n >= 0 ? Math.round(n) : null;
  } catch { return null; }
}

/** The plain sentences (also in the interface catalog in all eight languages). `{0}` slots are filled by the page. */
export const FIT_TEXT = Object.freeze({
  tooLarge: 'This message is about {0} tokens long, and this model can read about {1} at a time. It is too large to send as it is.',
  tooLargeNext: 'Attach it as a file for the assistant to read in parts, or send a shortened preview.',
  full: 'This conversation no longer fits the model’s window, and summarizing the older messages did not make enough room.',
  fullNext: 'Summarize it now and continue, or start a new chat.',
  summarizing: 'Summarizing earlier messages…',
  summarized: 'Older messages were summarized to make room. The chat was {0}% full and is now {1}% full.',
  summarizedWas: 'Older messages were summarized to make room. The chat was {0}% full.',
  summarizedShort: 'Older messages were summarized to make room.',
  nothingFull: 'This chat is {0}% full, but there is nothing older to summarize: the newest message is what fills it. Start a new chat, or send a shorter message.',
  nothingToSummarize: 'There is nothing older to summarize yet. Everything in this chat is still needed by the assistant.',
  noChatYet: 'There is nothing to summarize yet. This chat has not started a conversation with the assistant.',
  compactFailed: 'The older messages could not be summarized right now. Try again in a moment, or start a new chat.',
  compactBusy: 'This chat is still working. Stop it or wait for it to finish, then try again.',
});
export const fill = (template, values = []) => String(template).replace(/\{(\d+)\}/g, (_, i) => String(values[Number(i)] ?? ''));

/** A number with thousands separators, the way the page writes it. */
export const grouped = n => Math.round(n).toLocaleString('en-US');

/**
 * The two ways to send a message that is too large.
 *   file     the whole text goes in a file in the chat's folder; the message names it and shows how it starts
 *   preview  the start and the end of the text, cut to what the window can hold, with the number of characters left out
 * @param {{text:string, mode:'file'|'preview', tokens:number, window:number, pressure?:number, base?:number, path?:string}} input
 * @returns {{message:string, kept:number}}
 */
export function oversizedReplacement({ text, mode, tokens, window, pressure = 0, base = BASE_TOKENS, path = '' }) {
  const t = String(text);
  const charsPerToken = tokens > 0 ? Math.max(0.8, t.length / tokens) : 3;
  if (mode === 'file') {
    const start = t.slice(0, 1200).trimEnd();
    return { kept: start.length, message: `My message was too long to send in the chat, so I saved it as a file: ${path}\n\nIt starts like this:\n\n${start}\n\n[the rest is in the file]\n\nPlease read the file in parts (it is long) and then do what my message asks.` };
  }
  const room = Math.max(1000, Math.floor((window - Math.max(pressure, base) - replyRoom(window)) * 0.7));
  const keep = Math.max(500, Math.floor(room * charsPerToken * 0.9));
  const head = Math.ceil(keep * 0.7), tail = keep - head;
  const left = Math.max(0, t.length - head - tail);
  if (!left) return { kept: t.length, message: t };
  return { kept: head + tail, message: `${t.slice(0, head)}\n\n[… ${grouped(left)} characters from the middle are left out because the message was too long to send …]\n\n${t.slice(t.length - tail)}` };
}
