import { tr } from './i18n';
import type { LoopRecord } from './types';
import { LOOP_MAX_FAILURES, LOOP_MAX_RUNS, judgeAnswer, loopEndNote, loopEvery, loopLimitNote, withStopRule, type LoopLimits, type LoopStop } from './loop';

/**
 * The /loop scheduler. It owns the timer, the count, the time limit and the "the answer says it is done" rule, so a stop condition
 * is enforced here, in the app, and never left to the model: once a loop has ended nothing is sent again.
 *
 * Everything outside (sending the message, reading the answer, adding a line to the chat, the clock) is passed in, so the same code
 * runs in the app (store.ts) and in the tests with a scripted model and a fake clock.
 */
export interface LoopHost {
  /** Send the message as an ordinary chat turn and resolve when the answer is complete. `answered` is false when nothing was sent (the chat was busy). */
  send(sessionId: string, message: { text: string; shown: string }): Promise<{ answered: boolean; text: string; failed: boolean }>;
  /** A line from the app itself in the chat (never sent to the model). */
  note(sessionId: string, text: string): void;
  /** False when the chat was deleted: the loop ends quietly. */
  exists(sessionId: string): boolean;
  /** True when this chat is open and nothing is being answered. */
  ready(sessionId: string): boolean;
  now(): number;
  setTimer(fn: () => void, ms: number): unknown;
  clearTimer(handle: unknown): void;
  /** Keep a small record of the loop on the chat (running, or how it ended) so the task board can show it. Optional: the tests do not need it. */
  track?(sessionId: string, record: LoopRecord): void;
}

export interface LoopStart extends LoopLimits { everyMs: number; prompt: string }

interface LoopState extends LoopStart {
  answers: number;
  failures: number;
  startedAt: number;
  timer: unknown;
  running: boolean;
  lastText?: string;
  nextAt?: number;
}

export class LoopScheduler {
  private readonly loops = new Map<string, LoopState>();
  /** How long to wait before looking again when the chat is busy or not open. */
  busyRetryMs = 15_000;

  constructor(private readonly host: LoopHost) {}

  isRunning(sessionId: string | null): boolean { return !!sessionId && this.loops.has(sessionId); }

  /** The chats that have a loop going. */
  ids(): string[] { return [...this.loops.keys()]; }

  private record(sessionId: string, loop: LoopState, endedWhy?: NonNullable<LoopRecord['endedWhy']>) {
    const record: LoopRecord = { prompt: loop.prompt, everyMs: loop.everyMs, startedAt: loop.startedAt, state: endedWhy ? 'ended' : 'running', answers: loop.answers, ...(loop.maxRuns !== undefined ? { maxRuns: loop.maxRuns } : {}), ...(loop.lastText ? { lastText: loop.lastText } : {}) };
    if (endedWhy) { record.endedAt = this.host.now(); record.endedWhy = endedWhy; } else if (loop.nextAt !== undefined) record.nextAt = loop.nextAt;
    try { this.host.track?.(sessionId, record); } catch { /* the record is a courtesy to the board */ }
  }

  /** What the person is told for /loop with nothing after it. */
  describe(sessionId: string): string | null {
    const loop = this.loops.get(sessionId);
    if (!loop) return null;
    const limit = loopLimitNote(loop);
    return tr('A loop is running in this chat: it sends "{0}" {1}, while this chat is open. Type /loop stop to stop it.', [loop.prompt, loopEvery(loop.everyMs)]) + (limit ? ' ' + limit : '');
  }

  /** The person stopped it (/loop stop or the board's Stop loop). */
  stop(sessionId: string): boolean { return this.end(sessionId, 'stopped'); }

  private end(sessionId: string, why: NonNullable<LoopRecord['endedWhy']>): boolean {
    const loop = this.loops.get(sessionId);
    if (!loop) return false;
    if (loop.timer) this.host.clearTimer(loop.timer);
    this.loops.delete(sessionId);
    this.record(sessionId, loop, why);
    return true;
  }

  /** Start a loop and send its first message now. Returns false (and says why in the chat) when the chat already has one. */
  async start(sessionId: string, command: LoopStart): Promise<boolean> {
    if (this.loops.has(sessionId)) { this.host.note(sessionId, tr('This chat already has a loop. Type /loop stop first, then start a new one.')); return false; }
    const loop: LoopState = { ...command, answers: 0, failures: 0, startedAt: this.host.now(), timer: null, running: false };
    this.loops.set(sessionId, loop);
    this.record(sessionId, loop);
    const limit = loopLimitNote(loop);
    this.host.note(sessionId, tr('Loop started: "{0}" {1}, while this chat is open. Type /loop stop to stop it. For work that must go on with Strata closed, use the Tasks screen.', [loop.prompt, loopEvery(loop.everyMs)]) + (limit ? ' ' + limit : ''));
    await this.run(sessionId);
    return true;
  }

  /** Send once, judge the answer, then either end the loop with a plain note or wait for the next time. */
  private async run(sessionId: string): Promise<void> {
    const loop = this.loops.get(sessionId);
    if (!loop || loop.running) return;
    loop.running = true;
    let why: LoopStop | null = null;
    try {
      const result = await this.host.send(sessionId, { text: withStopRule(loop.prompt, loop.stopWhen), shown: loop.prompt });
      // The person may have typed /loop stop while the answer was coming: nothing more happens then.
      if (this.loops.get(sessionId) !== loop) return;
      if (result.answered) {
        loop.lastText = String(result.text ?? '').replace(/\s+/g, ' ').trim().slice(0, 200);
        if (result.failed) loop.failures += 1; else { loop.failures = 0; loop.answers += 1; }
        why = judgeAnswer(loop, loop, result, this.host.now());
      }
    } catch {
      if (this.loops.get(sessionId) !== loop) return;
      loop.failures += 1;
      if (loop.failures >= LOOP_MAX_FAILURES) why = 'failures';
    } finally {
      loop.running = false;
    }
    if (why) { this.finish(sessionId, loop, why); return; }
    this.schedule(sessionId, loop.everyMs);
  }

  private finish(sessionId: string, loop: LoopState, why: LoopStop) {
    this.end(sessionId, why);
    this.host.note(sessionId, loopEndNote(why, loop.answers));
  }

  private schedule(sessionId: string, ms: number) {
    const loop = this.loops.get(sessionId);
    if (!loop) return;
    loop.nextAt = this.host.now() + ms;
    this.record(sessionId, loop);
    loop.timer = this.host.setTimer(() => void this.tick(sessionId), ms);
  }

  private async tick(sessionId: string): Promise<void> {
    const loop = this.loops.get(sessionId);
    if (!loop) return;
    loop.timer = null;
    if (!this.host.exists(sessionId)) { this.end(sessionId, 'stopped'); return; }
    // The time is up, or the count is reached: never fire again, even if the last answer was not judged (a busy chat).
    if (loop.maxMs !== undefined && this.host.now() - loop.startedAt >= loop.maxMs) { this.finish(sessionId, loop, 'time'); return; }
    if (loop.maxRuns !== undefined && loop.answers >= loop.maxRuns) { this.finish(sessionId, loop, 'count'); return; }
    if (loop.answers >= LOOP_MAX_RUNS) { this.finish(sessionId, loop, 'cap'); return; }
    // Wait for a quiet moment: this chat is open and nothing is being answered.
    if (!this.host.ready(sessionId)) { this.schedule(sessionId, this.busyRetryMs); return; }
    await this.run(sessionId);
  }
}
