import type { Session } from './types';

/**
 * Saving the chat list without hurting the page (#3604, #3603).
 *
 * Every streamed token used to stringify every chat and write the lot to localStorage. On a long history that is
 * megabytes of string work per token, and the WebView2 renderer grew to gigabytes before it died (the window goes
 * white). Three rules fix it, and the stored format is unchanged, so older builds and the flow scripts still read it:
 *
 *  1. Coalesce: the newest state wins and at most one write happens per `delayMs`. The caller flushes when a turn ends
 *     or the page goes away.
 *  2. Only the chats that changed are serialised again. A chat is replaced by a new object when it changes (the store
 *     never mutates), so the previous JSON of every other chat is reused by object identity.
 *  3. The stored copy is capped: a huge tool result or an old chat that no longer fits is left out of the saved copy,
 *     never out of memory. The saved list is a cache, not the source of truth.
 */

/** Longest text kept for one field in the saved copy. */
export const MAX_FIELD_CHARS = 200_000;
/** Longest saved copy of one chat (JSON characters). A longer chat is shortened from its oldest messages. */
export const MAX_SESSION_CHARS = 1_500_000;
/** Longest saved chat list (JSON characters). localStorage allows about 5M; keep a margin for everything else. */
export const MAX_TOTAL_CHARS = 3_500_000;

type Json = unknown;

function clipFields(value: Json, depth = 0): Json {
  if (typeof value === 'string') return value.length > MAX_FIELD_CHARS ? `${value.slice(0, MAX_FIELD_CHARS)}\n[shortened in the saved copy]` : value;
  if (depth > 6 || value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(v => clipFields(v, depth + 1));
  const out: Record<string, Json> = {};
  for (const [k, v] of Object.entries(value as Record<string, Json>)) out[k] = clipFields(v, depth + 1);
  return out;
}

/** The JSON text kept for one chat: whole when it fits, otherwise clipped and then shortened from the oldest message. */
export function serializeSession(session: Session): string {
  const whole = JSON.stringify(session);
  if (whole.length <= MAX_SESSION_CHARS) return whole;
  let copy = clipFields(session) as Session;
  let text = JSON.stringify(copy);
  if (text.length <= MAX_SESSION_CHARS) return text;
  let messages = copy.messages ?? [];
  // Drop the oldest messages in halves until it fits; always keep the latest two.
  while (text.length > MAX_SESSION_CHARS && messages.length > 2) {
    messages = messages.slice(Math.max(1, Math.floor(messages.length / 2)));
    copy = { ...copy, messages };
    text = JSON.stringify(copy);
  }
  return text;
}

export interface SessionWriterOptions {
  /** Where the list is saved (the storage write). Receives the finished JSON text. */
  write: (json: string) => void;
  delayMs?: number;
  maxTotalChars?: number;
  /** Counts a serialisation of one chat; for tests. */
  onSerialize?: (session: Session) => void;
}

export function createSessionWriter(options: SessionWriterOptions) {
  const delayMs = options.delayMs ?? 1_000;
  const maxTotal = options.maxTotalChars ?? MAX_TOTAL_CHARS;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: Session[] | null = null;
  // Chat object -> its JSON. WeakMap: a replaced chat is forgotten with its object.
  const cache = new WeakMap<Session, string>();

  const textOf = (s: Session): string => {
    let text = cache.get(s);
    if (text === undefined) {
      options.onSerialize?.(s);
      text = serializeSession(s);
      cache.set(s, text);
    }
    return text;
  };

  const flush = () => {
    if (timer !== null) { clearTimeout(timer); timer = null; }
    const sessions = pending;
    pending = null;
    if (!sessions) return;
    const parts = sessions.map(s => ({ s, text: textOf(s) }));
    let total = parts.reduce((n, p) => n + p.text.length + 1, 2);
    if (total > maxTotal) {
      // Leave the oldest unpinned chats out of the saved copy until the rest fits. The newest chat is always kept.
      const order = parts
        .map((p, i) => ({ i, p }))
        .sort((a, b) => Number(!!a.p.s.pinned) - Number(!!b.p.s.pinned) || (a.p.s.updatedAt < b.p.s.updatedAt ? -1 : 1));
      const drop = new Set<number>();
      for (const { i, p } of order) {
        if (total <= maxTotal || drop.size >= parts.length - 1) break;
        drop.add(i);
        total -= p.text.length + 1;
      }
      options.write(`[${parts.filter((_, i) => !drop.has(i)).map(p => p.text).join(',')}]`);
      return;
    }
    options.write(`[${parts.map(p => p.text).join(',')}]`);
  };

  const schedule = (sessions: Session[]) => {
    pending = sessions;
    if (timer === null) timer = setTimeout(flush, delayMs);
  };

  return { schedule, flush, hasPending: () => pending !== null };
}
