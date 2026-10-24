/**
 * Helper agents (the Harness's subagents) as the rows the chat already has.
 *
 * A helper is a child session of the chat's Harness session. This module is the pure part: it folds one child's
 * `session/follow` items into
 *   - an `agent` frame (the shape `apps/host/src/lib/agents.ts updateAgent` merges into the chat's existing rows:
 *     id, name, description, status, durationMs, tokens, updatedAt), and
 *   - tool cards, but only for a helper's request that needs the person (an approval). Everything else a helper does
 *     stays in its row; the transcript is not filled with the work of four helpers at once.
 *
 * A helper's approval card is the same card a direct request gets (state `proposed`, answered with the same
 * Approve/Deny route). Its id is `<childSessionId>:<callId>`, which is also the key the bridge keeps the pending
 * question under, so the answer finds the helper that asked and no other chat can answer it.
 *
 * Wording is for the person reading the row, not for a log: no session ids, no tool internals.
 */
import { friendlyFailure, parseArguments, presentTool, resultText, toolScope } from './events.mjs';

const clip = (value, max) => {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

/** What a helper is doing, in words (shown as the row's description while it works). */
export function activityOf(name, args = {}) {
  const scope = toolScope(name);
  if (scope === 'shell') return 'Running a command';
  if (scope === 'write') return 'Changing files';
  if (name === 'web_fetch' || name === 'web_search') return 'Looking something up online';
  if (name === 'read' || name === 'glob' || name === 'grep' || name === 'read_image') return 'Reading files';
  if (name === 'subagent' || name === 'subagent_fork') return 'Handing work to another helper';
  if (scope === 'read') return 'Looking through the project';
  return 'Using a tool';
}

export class ChildProjector {
  /**
   * @param {{id:string, name?:string, task?:string, now?:()=>number}} options
   */
  constructor({ id, name = 'Helper', task = '', now = () => Date.now() }) {
    this.id = id;
    this.name = clip(name, 60) || 'Helper';
    this.task = clip(task, 140);
    this.now = now;
    this.startedAt = now();
    this.status = 'queued';
    this.tokens = 0;
    this.sawUsage = false;
    this.activity = null;
    this.closing = '';              // the helper's last words, shown once it finished
    this.failure = null;
    this.endedAt = null;
    this.tools = new Map();         // callId -> { card, calledAt, asked }
    this.asked = new Map();         // approval id -> callId
    this.lastEmit = 0;
  }

  /** The row as the UI merges it. */
  agent() {
    const at = this.now();
    const done = this.status === 'completed' || this.status === 'failed' || this.status === 'cancelled';
    let description = this.task;
    if (this.status === 'queued') description = this.task ? `Waiting for its turn. ${this.task}` : 'Waiting for its turn.';
    if (this.status === 'working' && this.activity) description = this.activity;
    if (this.status === 'completed') description = this.closing ? clip(this.closing, 140) : 'Finished.';
    if (this.status === 'failed') description = this.failure ?? 'Stopped because something went wrong.';
    if (this.status === 'cancelled') description = 'Stopped.';
    return {
      id: this.id, name: this.name, description, status: this.status,
      durationMs: Math.max(0, (done ? this.endedAt : at) - this.startedAt), tokens: this.tokens, updatedAt: at,
    };
  }

  #cardId(callId) { return `${this.id}:${callId}`; }

  #card(callId, patch) {
    const entry = this.tools.get(callId);
    if (!entry) return null;
    entry.card = { ...entry.card, ...patch };
    return { type: 'agent-tool', tool: entry.card };
  }

  /**
   * Fold one follow item. Returns `{ frames, row }`: card frames to send now, and whether the row changed
   * (the caller decides when to send it, so a stream of usage chunks does not flood the page).
   */
  consume(item) {
    const frames = [];
    let row = false;
    const set = (status) => { if (this.status !== status) { this.status = status; row = true; } };

    if (item.type === 'assistant-stream') {
      const f = item.frame;
      if (f.type === 'chunk' && f.chunk?.type === 'usage' && f.chunk.usage) {
        this.sawUsage = true;
        this.tokens += (Number(f.chunk.usage.inputTokens) || 0) + (Number(f.chunk.usage.outputTokens) || 0);
        row = true;
      } else if (f.type === 'chunk' && (f.chunk?.type === 'text-delta' || f.chunk?.type === 'reasoning-delta')) {
        if (this.status === 'queued') set('working');
      }
      return { frames, row };
    }
    if (item.type !== 'event') return { frames, row };
    const e = item.event;
    const d = e.data ?? {};
    switch (e.type) {
      case 'step/start':
      case 'turn/start':
        // A request was sent. Whether the engine has room for it yet is told by its first word: until then the helper is waiting for its turn.
        // A helper that had finished and got a new message starts over as waiting.
        if (this.finished) { this.endedAt = null; this.failure = null; set('queued'); }
        break;
      case 'assistant/message': {
        const text = (d.message?.content ?? []).filter(p => p.type === 'text').map(p => p.text).join('').trim();
        if (text) this.closing = text;
        break;
      }
      case 'tool/call': {
        const args = parseArguments(d.arguments);
        const card = { ...presentTool({ callId: this.#cardId(d.callId), name: d.name, args }), state: 'running', agentName: this.name };
        this.tools.set(d.callId, { card, calledAt: e.time, asked: false });
        this.activity = activityOf(d.name, args);
        set('working');
        row = true;
        break;
      }
      case 'approval/asked': {
        if (!d.callId || !this.tools.has(d.callId)) break;
        this.asked.set(d.id, d.callId);
        const entry = this.tools.get(d.callId);
        entry.asked = true;
        const frame = this.#card(d.callId, { state: 'proposed', effect: clip(`${this.name} asks: ${d.reason ?? 'to use a tool'}`, 400) });
        if (frame) frames.push(frame);
        break;
      }
      case 'approval/decided': {
        const callId = this.asked.get(d.id);
        const entry = callId ? this.tools.get(callId) : null;
        if (!entry) break;
        const frame = d.outcome === 'allowed-once'
          ? this.#card(callId, { state: 'running', approved: true })
          : this.#card(callId, { state: 'denied', result: d.outcome === 'rejected' ? 'Denied. Nothing ran.' : 'Cancelled. Nothing ran.' });
        if (frame) frames.push(frame);
        break;
      }
      case 'tool/result': {
        const callId = d.message?.toolCallId;
        const entry = this.tools.get(callId);
        if (!entry) break;
        if (entry.asked) {
          const wasDenied = entry.card.state === 'denied';
          const frame = this.#card(callId, {
            state: wasDenied ? 'denied' : d.message?.isError === true ? 'error' : 'done',
            result: wasDenied ? entry.card.result : resultText(d.message),
            durationMs: Math.max(0, (e.time ?? 0) - entry.calledAt),
          });
          if (frame) frames.push(frame);
        }
        this.activity = null;
        row = true;
        break;
      }
      case 'turn/end': {
        const reason = d.reason ?? { kind: 'completed' };
        this.endedAt = this.now();
        if (reason.kind === 'aborted') set('cancelled');
        else if (reason.kind === 'error') { this.failure = clip(friendlyFailure(reason.error), 140); set('failed'); }
        else if (reason.kind === 'blocked' || reason.kind === 'interrupted') { this.failure = 'Stopped before it finished.'; set('failed'); }
        else set('completed');
        // A request still waiting when its helper ended can no longer be answered.
        for (const [callId, entry] of this.tools) {
          if (entry.asked && entry.card.state === 'proposed') {
            const frame = this.#card(callId, { state: 'denied', result: 'Cancelled. Nothing ran.' });
            if (frame) frames.push(frame);
          }
        }
        row = true;
        break;
      }
      default:
        break;
    }
    return { frames, row };
  }

  get finished() { return this.status === 'completed' || this.status === 'failed' || this.status === 'cancelled'; }
}
