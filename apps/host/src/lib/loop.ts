import { tr } from './i18n';

/**
 * /loop: send the same message again and again, every few minutes, while this chat is open.
 *
 *   /loop 10m check whether the build passes                       start (the message is sent now, then every 10 minutes)
 *   /loop 5m say hi, stop after 3 answers                           stops by itself after the 3rd answer (a count in words works too)
 *   /loop 10m check the build for the next 2 hours                  stops by itself when the time is up
 *   /loop 5m check the build, stop when it passes                   stops when the answer says it is done
 *   /loop stop                                                      stop
 *   /loop                                                           say whether a loop is running
 *
 * The loop lives in the app window: it waits for the chat to be idle, and it stops when the window closes.
 * For work that must go on with the app closed, a task on the Tasks screen is the right tool.
 *
 * A stop condition is the scheduler's job, not the model's: the count and the time are read when the loop is created and are
 * enforced by the scheduler (loop-runner.ts), and the scheduler also ends the loop when an answer says the work is done or stopped.
 * The model is never left to keep answering "stopped" every minute.
 */

export const LOOP_MIN_MS = 60_000;
export const LOOP_MAX_MS = 24 * 3_600_000;
export const LOOP_MAX_RUNS = 100;
/** This many answers in a row that failed (no model, an error) end the loop, so it does not retry a broken thing all day. */
export const LOOP_MAX_FAILURES = 3;

export type LoopCommand =
  | { kind: 'start'; everyMs: number; prompt: string; maxRuns?: number; maxMs?: number; stopWhen?: string }
  | { kind: 'stop' }
  | { kind: 'status' }
  | { kind: 'help'; problem: string };

const UNITS: Record<string, number> = { s: 1000, sec: 1000, second: 1000, seconds: 1000, m: 60_000, min: 60_000, mins: 60_000, minute: 60_000, minutes: 60_000, h: 3_600_000, hr: 3_600_000, hour: 3_600_000, hours: 3_600_000 };

/** The words shown when a /loop line is not understood. */
export const LOOP_USAGE = 'Type /loop, how often, then what to do. For example: /loop 10m check whether the build passes. Add "stop after 3 answers" to end it by itself. Use /loop stop to stop.';

const NUMBER_WORDS: Record<string, number> = { a: 1, an: 1, one: 1, once: 1, two: 2, twice: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, fifteen: 15, twenty: 20, thirty: 30, fifty: 50 };
const NUM = '(\\d{1,4}|[a-z]+)';
const THINGS = '(?:answers?|repl(?:y|ies)|responses?|runs?|times?|messages?|rounds?|checks?|iterations?|turns?|updates?|reports?|passes|tries|attempts?|loops?)';
const STOP_VERB = '(?:stop|end|quit|finish|halt|cancel|exit)';
const number = (word: string): number | null => {
  const n = /^\d+$/.test(word) ? Number(word) : NUMBER_WORDS[word.toLowerCase()];
  return Number.isFinite(n) && n >= 1 ? n : null;
};
/** A clause with its comma or "and" in front and its full stop behind, so removing it leaves a clean sentence. */
const clause = (body: string) => new RegExp(`(?:[,;.]\\s*|\\s+(?:and|then)\\s+|\\s+)?(?:please\\s+)?${body}(?=\\s*[,;.!]|\\s+(?:and|then)\\b|\\s*$)[,;.!]?`, 'i');

interface Limits { prompt: string; maxRuns?: number; maxMs?: number; stopWhen?: string }

/** The count, the time and the condition the person put in words, with those words taken out of the message the model gets. */
export function readLimits(text: string): Limits {
  let prompt = text.trim();
  let maxRuns: number | undefined;
  let maxMs: number | undefined;
  let stopWhen: string | undefined;
  const take = (re: RegExp, use: (m: RegExpMatchArray) => boolean) => {
    const m = prompt.match(re);
    if (!m || !use(m)) return;
    prompt = (prompt.slice(0, m.index) + ' ' + prompt.slice((m.index ?? 0) + m[0].length)).replace(/\s+/g, ' ').replace(/\s+([,.;!])/g, '$1').replace(/^[\s,;.]+|[\s,;]+$/g, '').trim();
  };
  const setRuns = (n: number | null) => { if (n === null) return false; maxRuns = Math.min(n, LOOP_MAX_RUNS); return true; };
  // "stop after 3 answers", "stop the loop after three replies", "stop after the 3rd run", "end after 5 more checks"
  take(clause(`${STOP_VERB}(?:\\s+(?:the\\s+)?(?:loop|looping|it|this|all))?\\s+after\\s+(?:the\\s+)?${NUM}(?:st|nd|rd|th)?\\s+(?:more\\s+)?${THINGS}`), m => setRuns(number(m[1])));
  // "only 3 times", "just 3 answers", "3 times only", "3 times", "for 3 answers", "up to 3 runs", "at most 3 times", "max 3 runs", "run it 3 times"
  if (maxRuns === undefined) take(clause(`(?:only|just|exactly|up\\s+to|at\\s+most|no\\s+more\\s+than|max(?:imum)?(?:\\s+of)?|for|do\\s+this|do\\s+it|repeat(?:\\s+this|\\s+it)?|run(?:\\s+this|\\s+it)?|send(?:\\s+this|\\s+it)?|ask(?:\\s+me)?)\\s+${NUM}\\s+${THINGS}(?:\\s+only)?`), m => setRuns(number(m[1])));
  if (maxRuns === undefined) take(clause(`${NUM}\\s+${THINGS}\\s+(?:only|in\\s+total|total|at\\s+most)`), m => setRuns(number(m[1])));
  if (maxRuns === undefined) take(clause(`${NUM}\\s+times`), m => setRuns(number(m[1])));
  if (maxRuns === undefined) take(clause(`(?:${STOP_VERB}\\s+)?(?:for\\s+)?(?:the\\s+)?(?:first|next)\\s+${NUM}\\s+${THINGS}`), m => setRuns(number(m[1])));
  // "for the next 2 hours", "for 30 minutes", "stop after 2 hours", "for an hour"
  const span = (m: RegExpMatchArray) => {
    const n = number(m[1]); const unit = UNITS[m[2].toLowerCase()];
    if (n === null || !unit || unit < 60_000) return false;
    maxMs = Math.min(n * unit, LOOP_MAX_MS);
    return true;
  };
  take(clause(`${STOP_VERB}(?:\\s+(?:the\\s+)?(?:loop|it|this))?\\s+after\\s+(?:about\\s+)?${NUM}\\s*(minutes?|mins?|hours?|hrs?|[mh])\\b`), span);
  if (maxMs === undefined) take(clause(`(?:for|over)\\s+(?:the\\s+)?(?:next\\s+)?(?:about\\s+)?${NUM}\\s*(minutes?|mins?|hours?|hrs?)\\b`), span);
  // "stop when the build passes", "stop once it is green", "until the file exists": the model judges it; the scheduler stops on its word
  take(clause(`(?:${STOP_VERB}\\s+(?:the\\s+loop\\s+)?(?:when|once|as\\s+soon\\s+as|if|the\\s+moment)|until)\\s+([^.;!]{3,160})`), m => { stopWhen = m[1].trim(); return true; });
  return { prompt, ...(maxRuns !== undefined ? { maxRuns } : {}), ...(maxMs !== undefined ? { maxMs } : {}), ...(stopWhen ? { stopWhen } : {}) };
}

/** null means the text is an ordinary message. */
export function parseLoop(text: string): LoopCommand | null {
  const match = text.trim().match(/^\/loop(?:\s+([\s\S]*))?$/i);
  if (!match) return null;
  const rest = (match[1] ?? '').trim();
  if (!rest) return { kind: 'status' };
  if (/^(stop|off|cancel|end)$/i.test(rest)) return { kind: 'stop' };
  const timed = rest.match(/^(?:every\s+)?(\d+(?:\.\d+)?)\s*([a-z]+)\s+([\s\S]+)$/i);
  const unit = timed ? UNITS[timed[2].toLowerCase()] : undefined;
  if (!timed || !unit) return { kind: 'help', problem: 'Say how often first, like 10m (minutes) or 2h (hours), then what to do.' };
  const everyMs = Math.round(Number(timed[1]) * unit);
  if (everyMs < LOOP_MIN_MS) return { kind: 'help', problem: 'The shortest time between runs is 1 minute, so the model has time to finish each one.' };
  if (everyMs > LOOP_MAX_MS) return { kind: 'help', problem: 'The longest time between runs is 24 hours. For longer waits, use a task on the Tasks screen.' };
  const limits = readLimits(timed[3]);
  if (!limits.prompt) return { kind: 'help', problem: 'Say what to do each time, not only when to stop.' };
  return { kind: 'start', everyMs, ...limits };
}

/** "every 10 minutes", "every hour". */
export function loopEvery(ms: number): string {
  if (ms % 3_600_000 === 0) { const h = ms / 3_600_000; return h === 1 ? tr('every hour') : tr('every {0} hours', [h]); }
  const m = Math.round(ms / 60_000);
  return m === 1 ? tr('every minute') : tr('every {0} minutes', [m]);
}

/** The line added to a message that carries a stop condition in words, so the model can say it is met (it is not shown in the chat). */
export const LOOP_DONE_WORD = 'DONE';
export function withStopRule(prompt: string, stopWhen?: string): string {
  return stopWhen ? `${prompt}\n\n(This repeats on a timer. If this is finished or "${stopWhen}" is true, begin your reply with the word ${LOOP_DONE_WORD} and say so in one sentence. Otherwise just answer.)` : prompt;
}

const plain = (text: string) => text.replace(/[*_`#>~]/g, '').replace(/\s+/g, ' ').trim();

/**
 * Does this reply say the work is finished or stopped? The scheduler ends the loop on it.
 *
 * Always: the model's own marker ("DONE ...", "LOOP DONE", "LOOP STOPPED") or a reply that is only "stopped" / "done" / "finished".
 * When the person stated a stop condition (a count, a time or a condition in words), also the short ways a model closes the job:
 * "Stopped.", "The loop is stopped.", "I have stopped as requested.", "Done, the build passes."
 */
export function replySaysDone(reply: string, stated: boolean): boolean {
  const text = plain(reply);
  if (!text) return false;
  if (/\bLOOP[ _-]?(?:DONE|STOPPED|STOP|COMPLETE|FINISHED|ENDED)\b/i.test(text)) return true;
  if (new RegExp(`^${LOOP_DONE_WORD}\\b`).test(text)) return true;
  const bare = text.replace(/[.!\s]+$/g, '').toLowerCase();
  if (/^(?:ok(?:ay)?[,. ]+)?(?:stopped|done|finished|complete|completed|all done|ended|stopping)$/.test(bare)) return true;
  if (!stated || text.length > 280) return false;
  return /^(?:ok(?:ay)?[,. ]+|sure[,. ]+|yes[,. ]+)?(?:the\s+)?(?:loop\s+(?:is\s+|has\s+)?(?:now\s+)?)?(?:stopped|done|finished|complete(?:d)?|ended|stopping|all done)\b/i.test(text)
    || /\b(?:loop|repeat\w*|schedule\w*)\s+(?:is\s+|has\s+been\s+|was\s+)?(?:now\s+)?(?:stopped|finished|complete(?:d)?|ended|over|done)\b/i.test(text)
    || /\bi(?:'ve|\s+have|'m|\s+am)?\s+(?:now\s+)?(?:stopped|stopping|done|finished)\b/i.test(text)
    || /\b(?:stop|end)\s+condition\s+(?:is\s+|has\s+been\s+)?met\b/i.test(text)
    || /\bnothing\s+(?:more|else|left)\s+to\s+(?:do|check|report)\b/i.test(text);
}

export interface LoopLimits { maxRuns?: number; maxMs?: number; stopWhen?: string }
export interface LoopProgress { answers: number; failures: number; startedAt: number }
export type LoopStop = 'count' | 'said-done' | 'time' | 'failures' | 'cap';

/** After an answer: should the loop end, and why? `reply` is the model's text of that run; `failed` is an answer that did not work. */
export function judgeAnswer(limits: LoopLimits, progress: LoopProgress, reply: { text: string; failed: boolean }, now: number): LoopStop | null {
  if (reply.failed) return progress.failures >= LOOP_MAX_FAILURES ? 'failures' : null;
  const stated = limits.maxRuns !== undefined || limits.maxMs !== undefined || !!limits.stopWhen;
  if (replySaysDone(reply.text, stated)) return 'said-done';
  if (limits.maxRuns !== undefined && progress.answers >= limits.maxRuns) return 'count';
  if (limits.maxMs !== undefined && now - progress.startedAt >= limits.maxMs) return 'time';
  if (progress.answers >= LOOP_MAX_RUNS) return 'cap';
  return null;
}

/** Plain words for the person when a loop ends by itself. */
export function loopEndNote(why: LoopStop, answers: number): string {
  switch (why) {
    case 'count': return tr('The loop is done: it ran {0} times, as you asked. Type /loop to start another.', [answers]);
    case 'said-done': return tr('The loop stopped because the last answer said the work is done. Type /loop to start another.');
    case 'time': return tr('The loop stopped because the time you set is up. Type /loop to start another.');
    case 'failures': return tr('The loop stopped because {0} runs in a row did not work. Check the model, then type /loop to start it again.', [LOOP_MAX_FAILURES]);
    default: return tr('The loop stopped after {0} runs. Type /loop again to start it again.', [LOOP_MAX_RUNS]);
  }
}

/** The part of the "Loop started" note that says how it ends by itself ('' when it only ends when the person stops it). */
export function loopLimitNote(limits: LoopLimits): string {
  const parts: string[] = [];
  if (limits.maxRuns !== undefined) parts.push(limits.maxRuns === 1 ? tr('It stops by itself after the first answer.') : tr('It stops by itself after {0} answers.', [limits.maxRuns]));
  if (limits.maxMs !== undefined) parts.push(tr('It stops by itself after {0} minutes.', [Math.max(1, Math.round(limits.maxMs / 60_000))]));
  if (limits.stopWhen) parts.push(tr('It stops by itself when an answer says the work is done.'));
  return parts.join(' ');
}
