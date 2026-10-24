/**
 * The turn guard: a turn that keeps failing the same way, or keeps rewriting the same thing, is stopped by the app and ends with a plain message.
 * Pure (no I/O); the bridge feeds it what the turn shows and acts on what it answers.
 *
 * Why: a search-heavy /plan on a small model searched again after every refusal and rewrote its plan after every search until the context
 * reached 100%, then the turn ended with no answer and no message. Nothing here relies on the model behaving: the app counts.
 *
 *   searches-failed   this many web searches in a row failed (web search was limited, or could not be reached)
 *   searches          this many searches in one turn (a plan or an ordinary answer needs a handful, not dozens)
 *   rewrites          a plan turn wrote its numbered plan from "1." again this many times
 *   context           the context is nearly full: the Harness compaction did not keep up
 *
 * What the person is told is `guardNote`: fixed English sentences that are also in the interface catalog in all eight languages (the page
 * translates a `delta` frame that carries `key`). Each says what happened and gives one next step.
 */

export const GUARD_LIMITS = Object.freeze({
  failedSearches: 3,
  searchesPlan: 10,
  searchesTurn: 25,
  planRewrites: 4,
  contextFraction: 0.93,
  /** A turn that ends with no words at all and a context this full is "out of room", not "nothing to say". */
  contextEmptyFraction: 0.85,
});

const SEARCH_TOOLS = new Set(['web_search']);
const SEARCH_FAILED = /web search is limited|search could not reach|could not reach the internet|WEB_RATE_LIMITED|WEB_PROVIDER_ERROR|did not accept the key|too many searches/i;

/** The plain sentences, keyed by reason. `values` are the slots the page fills in. */
const NOTES = {
  'searches-failed': 'Web search is limited right now, so I stopped searching and went on with what I have. Try again in a minute, or add a search key in Settings → Tools → Web search.',
  searches: 'I searched a lot and stopped so this chat does not fill up. This is what I have so far. Tell me what to look into next.',
  rewrites: 'I kept rewriting the plan, so I stopped here with the latest version. Tell me what to change, or approve it.',
  context: 'I ran out of room in this chat, so I stopped. Strata is summarising the older messages to make space. Say “continue” to go on, or start a new chat.',
  empty: 'I stopped without writing an answer. Send your message again, or say “continue”.',
};
export const GUARD_NOTES = Object.freeze({ ...NOTES });

/** A `delta` frame the page shows as ordinary assistant text and translates (`key` is the English sentence). */
export function guardNote(reason) {
  const text = NOTES[reason];
  return text ? { type: 'delta', own: true, text: `\n\n${text}`, key: text, noteReason: reason } : null;
}

export class TurnGuard {
  /**
   * @param {{plan?: boolean, contextMax?: number, limits?: Partial<typeof GUARD_LIMITS>}} [options]
   */
  constructor({ plan = false, contextMax = 0, limits = {} } = {}) {
    this.plan = plan; this.contextMax = Number(contextMax) || 0;
    this.limits = { ...GUARD_LIMITS, ...limits };
    this.seen = new Set();
    this.searches = 0; this.failedInARow = 0; this.failedTotal = 0;
    this.rewrites = 0; this.written = '';
    this.reason = null;
  }

  #fire(reason) { this.reason ??= reason; return this.reason; }

  /** A tool card frame (`agent-tool`). Counts a search once, when it has finished. Returns the reason to stop, or null. */
  card(tool) {
    if (this.reason || !tool || !SEARCH_TOOLS.has(tool.name)) return this.reason;
    const done = tool.state === 'done' || tool.state === 'error' || tool.state === 'denied';
    if (!done || this.seen.has(tool.id)) return this.reason;
    this.seen.add(tool.id);
    this.searches += 1;
    const failed = tool.state === 'error' || SEARCH_FAILED.test(String(tool.result ?? ''));
    if (failed) { this.failedInARow += 1; this.failedTotal += 1; } else this.failedInARow = 0;
    if (this.failedInARow >= this.limits.failedSearches) return this.#fire('searches-failed');
    if (this.searches >= (this.plan ? this.limits.searchesPlan : this.limits.searchesTurn)) return this.#fire(this.failedTotal >= this.searches / 2 ? 'searches-failed' : 'searches');
    return this.reason;
  }

  /** Text the assistant wrote (not our own lines). A plan turn that starts its numbered list again and again is rewriting. */
  text(delta) {
    if (this.reason || !this.plan || !delta) return this.reason;
    this.written += delta;
    if (this.written.length > 200_000) this.written = this.written.slice(-100_000);
    if (!delta.includes('\n') && !/\d[.)]\s/.test(delta)) return this.reason;
    this.rewrites = (this.written.match(/(?:^|\n)[ \t]{0,3}1[.)][ \t]+\S/g) ?? []).length;
    if (this.rewrites >= this.limits.planRewrites) return this.#fire('rewrites');
    return this.reason;
  }

  /** What the context holds now (tokens of the last request). */
  context(tokens) {
    if (this.reason || !this.contextMax || !Number.isFinite(tokens)) return this.reason;
    if (tokens / this.contextMax >= this.limits.contextFraction) return this.#fire('context');
    return this.reason;
  }

  /**
   * The turn is over. If the person has seen no words from it, which sentence do they get? `ended` is the Harness's turn end reason,
   * `wrote` whether any answer text was shown, `acted` whether tool cards were shown, `contextTokens` the last request's size. Returns a reason or null.
   */
  silent({ ended, wrote, contextTokens, acted = false, stoppedByPerson = false }) {
    if (this.reason) return this.reason;
    if (stoppedByPerson || wrote) return null;
    const kind = ended?.kind;
    if (kind !== 'completed' && kind !== 'max-tokens') return null;       // an error says why by itself; Stop is the person's own
    if (this.contextMax && Number(contextTokens) / this.contextMax >= this.limits.contextEmptyFraction) return 'context';
    return acted ? null : 'empty';                                         // cards of work done are an answer of a kind; no cards and no words is not
  }
}
