import type { Ask } from './ask';
import type { Session, ToolCall } from './types';
import type { Task } from './tasks';

/**
 * The task board (Tasks screen, "Board"). One pure function turns what the app already knows (scheduled tasks from the host, chats and
 * their loops from the browser, the approvals the host is holding) into cards in four columns. Nothing here is stored as "board state":
 * a card is where the facts put it, and it moves when the facts change. It never touches the DOM, the store or the clock, so it is tested
 * like loop.ts.
 *
 * One card per thing: a chat shows once, in the first place that fits (waiting for you, running, needs attention, done), and a task
 * shows once. A task or chat the person dismissed stays off the board until a newer run or turn brings a problem back.
 */

export type BoardColumn = 'running' | 'waiting' | 'attention' | 'done';

/** What a button on a card does. The component maps each one to a call; the order of `menu` is the order shown. */
export type BoardAction = 'stop-task' | 'run-again' | 'open-task' | 'open-chat' | 'stop-chat' | 'stop-loop' | 'approve' | 'deny' | 'continue' | 'dismiss' | 'view-result';

/** Why a card needs the person. `kind` names the sentence; the component writes it in the person's language. */
export interface BoardReason {
  tone: 'bad' | 'warn' | 'ask';
  kind: 'task-failed' | 'task-stopped' | 'task-closed' | 'task-needs-yes' | 'chat-failed' | 'loop-failed' | 'ask';
  /** The original error text, for the failures a person can open under Details. */
  detail?: string;
  /** The approval question, for `ask`. */
  ask?: Ask;
  /** What the tool wants to do, when the host sent no question. */
  fallback?: string;
  params?: (string | number)[];
}

export interface BoardCard {
  /** Stable across refreshes: the source and its id. */
  key: string;
  column: BoardColumn;
  source: 'task' | 'chat' | 'loop';
  /** Task id for a task; chat id for a chat or a loop. */
  id: string;
  title: string;
  /** Milliseconds. What the time on the card is about, and the sort key (newest first). */
  at: number;
  timeKind: 'started' | 'waiting' | 'next' | 'stopped' | 'plain';
  /** For a loop that is running: when its next answer is due. */
  nextAt?: number;
  /** A loop's rhythm in milliseconds, for "Loop · every 10 min". */
  everyMs?: number;
  folder?: string;
  /** The muted part after the folder: "2 helpers working", "answer 3 of 6". */
  meta?: { text: string; params?: (string | number)[] };
  line?: string;
  reason?: BoardReason;
  /** Done cards: the chip beside the button. */
  chip?: { text: string; tone: 'good' | 'plain' };
  /** A card whose run is going on shows a moving bar. */
  busy?: boolean;
  primary: BoardAction;
  menu: BoardAction[];
  /** Set on a Waiting card that may be answered from the board (a file read or write inside the chat's folder). */
  approve?: { messageId: string; toolId: string };
}

export interface BoardPending { chatId: string; callIds: string[]; at?: number; folder?: string | null }

export interface BoardInput {
  tasks: readonly Task[];
  sessions: readonly Session[];
  /** Chats with a turn going. */
  liveChats: readonly string[];
  /** Chats with a /loop going. */
  loopChats: readonly string[];
  /** What the host holds, per chat. `null` while it is not known (not asked yet or unreachable): the cards in the chats are trusted then. */
  pending: readonly BoardPending[] | null;
  activeSession: string | null;
  now: number;
  days?: number;
}

export interface BoardResult {
  columns: Record<BoardColumn, BoardCard[]>;
  /** Waiting for you plus Needs attention: the number the sidebar shows. */
  need: number;
  /** The next scheduled runs. */
  upcoming: { id: string; title: string; at: number }[];
  /** Task ids that have a problem the person has not dismissed (the sidebar's dot). */
  problemTasks: string[];
}

export const BOARD_DAYS = 7;
const DAY = 86_400_000;

const time = (iso?: string | null) => { const n = iso ? Date.parse(iso) : NaN; return Number.isFinite(n) ? n : 0; };

/** The last folder name of a path: how a chat's working folder is shown on a card. */
export function folderName(path?: string | null): string | undefined {
  const parts = String(path ?? '').split(/[\\/]+/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : undefined;
}

const normal = (p: string) => p.replace(/\\/g, '/').replace(/\/+$/, '');

/**
 * Is `target` inside `folder`? Relative paths count (the agent works in the folder) unless they climb out with "..". Anything else
 * (another drive, another tree, a path that cannot be told) is not: those need the whole conversation in front of the person.
 */
export function isInside(target: string | undefined | null, folder: string | undefined | null): boolean {
  if (!target || !folder) return false;
  const t = normal(String(target).trim()), f = normal(String(folder).trim());
  if (!t || !f || t.split('/').includes('..')) return false;
  const windows = /^[a-z]:/i.test(f) || String(folder).includes('\\');
  const absolute = /^([a-z]:)?\//i.test(t);
  if (!absolute) return !/^[a-z]:/i.test(t);
  const a = windows ? t.toLowerCase() : t, b = windows ? f.toLowerCase() : f;
  return a === b || a.startsWith(b + '/');
}

/** Can this waiting card be answered from the board? A file read or write inside the chat's folder; shell commands and connected tools open the chat. */
export function canApproveOnBoard(tool: Pick<ToolCall, 'scope' | 'target'>, folder: string | undefined | null): boolean {
  return (tool.scope === 'read' || tool.scope === 'write') && isInside(tool.target, folder);
}

/** One plain line from an answer: markdown off, first sentence (or two when the first is tiny), cut at 160 characters. */
export function plainLine(text: string | undefined | null, max = 160): string {
  let body = String(text ?? '').replace(/```[\s\S]*?```/g, ' ').replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/^\s*(#{1,6}|[-*+]|\d+[.)]|>)\s+/gm, '').replace(/[*_`~]/g, '');
  body = body.replace(/\s+/g, ' ').trim();
  if (!body) return '';
  const sentence = /^.*?[.!?](?=\s|$)/.exec(body)?.[0] ?? body;
  const two = /^.*?[.!?](?=\s|$)\s*.*?[.!?](?=\s|$)/.exec(body)?.[0] ?? body;
  const pick = sentence.length < 24 && body.length > sentence.length ? two : sentence;
  return pick.length > max ? `${pick.slice(0, max - 1).trimEnd()}…` : pick;
}

function lastAssistant(s: Session) {
  for (let i = s.messages.length - 1; i >= 0; i--) { const m = s.messages[i]; if (m.role === 'assistant' && !(m as { local?: boolean }).local) return m; }
  return undefined;
}

/** Dismissed: the person dismissed it at or after the moment of the thing shown. */
const dismissed = (ackedAt: string | null | undefined, since: number) => !!ackedAt && time(ackedAt) >= since;

export function boardCards(input: BoardInput): BoardResult {
  const { now } = input, window = (input.days ?? BOARD_DAYS) * DAY;
  const live = new Set(input.liveChats), loops = new Set(input.loopChats);
  const columns: Record<BoardColumn, BoardCard[]> = { running: [], waiting: [], attention: [], done: [] };
  const problemTasks: string[] = [];
  const upcoming: BoardResult['upcoming'] = [];

  /* ---- scheduled tasks ---- */
  for (const t of input.tasks) {
    if (t.archived || !t.id) continue;
    const base = { key: `task:${t.id}`, source: 'task' as const, id: t.id, title: t.title };
    const ran = time(t.lastRunAt), fin = time(t.finishedAt) || ran;
    if (t.enabled && t.nextRunAt) upcoming.push({ id: t.id, title: t.title, at: time(t.nextRunAt) });
    if (t.status === 'running') { columns.running.push({ ...base, column: 'running', at: ran, timeKind: 'started', line: plainLine(t.lastLine ?? '') || undefined, busy: true, primary: 'stop-task', menu: ['open-task'] }); continue; }
    if (dismissed(t.ackedAt, fin)) continue;
    const failed = t.status === 'error' || t.status === 'cancelled' || t.status === 'interrupted';
    if (failed) {
      problemTasks.push(t.id);
      const kind = t.status === 'error' ? 'task-failed' : t.status === 'cancelled' ? 'task-stopped' : 'task-closed';
      columns.attention.push({ ...base, column: 'attention', at: fin || ran, timeKind: 'plain', reason: { tone: 'bad', kind, ...(t.status === 'error' && t.error ? { detail: t.error } : {}) }, primary: 'run-again', menu: ['open-task', 'continue', 'dismiss'] });
    } else if (t.attention) {
      problemTasks.push(t.id);
      // A run that was auto-denied: the useful next step is a chat that can carry on with the person's yes.
      columns.attention.push({ ...base, column: 'attention', at: fin || ran, timeKind: 'plain', line: plainLine(t.lastLine ?? t.output) || undefined, reason: { tone: 'warn', kind: 'task-needs-yes', detail: t.attention }, primary: 'continue', menu: ['open-task', 'dismiss'] });
    } else if (t.status === 'done' && fin && now - fin <= window) {
      columns.done.push({ ...base, column: 'done', at: fin, timeKind: 'plain', line: plainLine(t.lastLine ?? t.output) || undefined, chip: { text: 'Finished', tone: 'good' }, primary: 'view-result', menu: ['dismiss'] });
    }
  }

  /* ---- chats and loops ---- */
  const pendingBy = new Map((input.pending ?? []).map(p => [p.chatId, p]));
  for (const s of input.sessions) {
    if (s.archived || s.showcase) continue;
    const base = { id: s.id, title: s.title, folder: folderName(s.workspace) };
    const answer = lastAssistant(s);

    // Waiting for you: a card the host still holds.
    const held = pendingBy.get(s.id);
    let proposed: { messageId: string; tool: ToolCall } | undefined;
    for (const m of s.messages) for (const tool of m.tools ?? []) if (tool.live && tool.state === 'proposed' && (input.pending === null || held?.callIds.includes(tool.id))) { proposed = { messageId: m.id, tool }; break; }
    if (proposed) {
      const { tool, messageId } = proposed;
      const folder = held?.folder ?? s.workspace;
      const ok = canApproveOnBoard(tool, folder);
      columns.waiting.push({ ...base, key: `chat:${s.id}`, source: 'chat', column: 'waiting', at: held?.at || time(s.updatedAt), timeKind: 'waiting',
        reason: { tone: 'ask', kind: 'ask', ...(tool.ask ? { ask: tool.ask } : {}), fallback: tool.effect || tool.preview || tool.name },
        primary: ok ? 'approve' : 'open-chat', menu: ok ? ['deny', 'open-chat'] : [], ...(ok ? { approve: { messageId, toolId: tool.id } } : {}) });
      continue;
    }

    // Running: a loop that is going, or a turn that is.
    if (loops.has(s.id) && s.loop) {
      const l = s.loop;
      columns.running.push({ ...base, key: `loop:${s.id}`, source: 'loop', column: 'running', at: l.startedAt, timeKind: 'next', nextAt: live.has(s.id) ? undefined : l.nextAt, everyMs: l.everyMs, title: l.prompt || s.title,
        meta: { text: l.maxRuns ? 'answer {0} of {1}' : 'answer {0}', params: l.maxRuns ? [l.answers + 1, l.maxRuns] : [l.answers + 1] }, line: plainLine(l.lastText) || undefined, busy: live.has(s.id), primary: 'stop-loop', menu: ['open-chat'] });
      continue;
    }
    if (live.has(s.id)) {
      const helpers = (s.agents ?? []).filter(a => a.status === 'working').length;
      const lastUser = [...s.messages].reverse().find(m => m.role === 'user');
      columns.running.push({ ...base, key: `chat:${s.id}`, source: 'chat', column: 'running', at: time(lastUser?.at) || time(s.updatedAt), timeKind: 'started', busy: true,
        ...(helpers ? { meta: { text: helpers === 1 ? '1 helper working' : '{0} helpers working', params: [helpers] } } : {}), line: plainLine(answer?.content) || undefined, primary: 'open-chat', menu: ['stop-chat'] });
      continue;
    }

    // A loop that ended (or that the window closed on) speaks for the whole chat.
    if (s.loop) {
      const l = s.loop;
      const why = l.state === 'running' ? 'closed' : l.endedWhy ?? 'closed';
      const when = l.endedAt ?? time(s.updatedAt);
      if (dismissed(s.ackedAt, when) || now - when > window) continue;
      const common = { ...base, key: `loop:${s.id}`, source: 'loop' as const, at: when, timeKind: 'stopped' as const, everyMs: l.everyMs, title: l.prompt || s.title, line: plainLine(l.lastText) || undefined };
      if (why === 'failures') columns.attention.push({ ...common, column: 'attention', reason: { tone: 'bad', kind: 'loop-failed' }, primary: 'open-chat', menu: ['continue', 'dismiss'] });
      else columns.done.push({ ...common, column: 'done', chip: why === 'stopped' ? { text: 'Stopped by you', tone: 'plain' } : why === 'closed' ? { text: 'Stopped when Strata closed', tone: 'plain' } : { text: 'Ended on its own', tone: 'plain' }, primary: 'open-chat', menu: ['dismiss'] });
      continue;
    }

    // Chats: only what has something to do, in the window.
    const when = time(s.updatedAt);
    if (!when || now - when > window || dismissed(s.ackedAt, when) || !answer || answer.streaming) continue;
    if (answer.error) {
      columns.attention.push({ ...base, key: `chat:${s.id}`, source: 'chat', column: 'attention', at: when, timeKind: 'plain', reason: { tone: 'bad', kind: 'chat-failed', detail: answer.error }, primary: 'open-chat', menu: ['continue', 'dismiss'] });
      continue;
    }
    const unread = s.id !== input.activeSession && (s.lastReadAt ? s.updatedAt > s.lastReadAt : s.messages.length > 0);
    if (unread && answer.content?.trim() && s.messages[s.messages.length - 1] === answer) {
      columns.done.push({ ...base, key: `chat:${s.id}`, source: 'chat', column: 'done', at: when, timeKind: 'plain', line: plainLine(answer.content) || undefined, chip: { text: 'Finished', tone: 'good' }, primary: 'open-chat', menu: ['dismiss'] });
    }
  }

  for (const list of Object.values(columns)) list.sort((a, b) => b.at - a.at);
  upcoming.sort((a, b) => a.at - b.at);
  return { columns, need: columns.waiting.length + columns.attention.length, upcoming: upcoming.slice(0, 5), problemTasks };
}
