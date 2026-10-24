/**
 * Keeps markup that belongs to the model's reasoning, not to the answer, out of the answer text (QA report F01: a literal "<analysis>" with
 * reasoning Off, and a "p> " in front of a short answer, also in the Markdown export).
 *
 * This only removes the wrapper itself, never the words around it, and only for names the reasoning formats use. The words stay, so a model that
 * answers inside such a tag still shows its answer. If the raw text of a model really holds these, that is the engine's output and is reported
 * there: this is not a fix for it, only a guard so the page never shows a wrapper to a person.
 *
 * Streaming: the page calls this with everything received so far, plus the next piece, every time, so it is cheap and leaves a tag that has not
 * arrived completely yet alone (it is removed when the rest arrives).
 */
const WRAPPER = '(?:think|thinking|thought|analysis|reasoning)';
/** <think>, </think>, <analysis ...> : one reasoning wrapper tag, open or close. */
const TAG = new RegExp(`</?${WRAPPER}(?:\\s[^<>]*)?/?>`, 'gi');
/**
 * A tag that lost its "<" at the very start of the answer ("p> CEDAR-742", "/p> done"): only well-known markup names, only at the start.
 * The text after it (and the single space that follows) is kept.
 */
const LOST_LEAD = /^\s*\/?(?:p|br|div|span|li|ul|ol|h[1-6]|b|i|em|strong|code|pre|answer|reply|final|analysis|think|thinking|reasoning)>[ \t]+/i;

/** The answer with reasoning wrapper tags and a stray leading tag fragment removed. Anything else is returned unchanged. */
export function tidyAnswer(text: string): string {
  if (!text) return text;
  let out = text;
  if (out.indexOf('>') !== -1) {
    out = out.replace(TAG, '');
    out = out.replace(LOST_LEAD, '');
  }
  return out;
}

/** True when `tidyAnswer` would change nothing, so a caller can skip a copy. */
export const isTidy = (text: string) => tidyAnswer(text) === text;
