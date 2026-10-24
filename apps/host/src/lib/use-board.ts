import { useEffect, useMemo, useState } from 'react';
import { api } from './client';
import { boardCards, type BoardPending, type BoardResult } from './board';
import { liveChatIds, loopChatIds, useStore } from './store';
import type { Task } from './tasks';

/**
 * What the board and the sidebar's count read: the host's tasks and its waiting approvals (asked every few seconds, by one timer however
 * many parts of the page use them), turned into cards together with the chats in this browser.
 */
interface Shared { tasks: Task[]; pending: BoardPending[] | null; ready: boolean; error: string }
let shared: Shared = { tasks: [], pending: null, ready: false, error: '' };
const listeners = new Set<() => void>();
let timer: ReturnType<typeof setInterval> | null = null;
let inFlight = false;

async function refresh() {
  if (inFlight) return;
  inFlight = true;
  try {
    const [tasks, waiting] = await Promise.all([
      api.get<{ tasks: Task[] }>('/api/tasks'),
      api.get<{ chats: { chatId: string; folder: string | null; pending: { callId: string; at: number }[] }[] }>('/api/agent/pending').catch(() => null),
    ]);
    shared = {
      tasks: tasks.tasks,
      // Unknown (the host did not answer) is not the same as "nothing waits": the chats' own cards are trusted then.
      pending: waiting ? waiting.chats.map(c => ({ chatId: c.chatId, folder: c.folder, callIds: c.pending.map(p => p.callId), at: Math.min(...c.pending.map(p => p.at)) })) : null,
      ready: true, error: '',
    };
  } catch (e) {
    shared = { ...shared, ready: true, error: e instanceof Error ? e.message : String(e) };
  } finally {
    inFlight = false;
    listeners.forEach(l => l());
  }
}

/** Ask now (after an action), instead of waiting for the next tick. */
export function refreshBoard() { void refresh(); }

function subscribe(listener: () => void) {
  listeners.add(listener);
  if (listeners.size === 1) { void refresh(); timer = setInterval(() => void refresh(), 4000); window.addEventListener('strata:tasks', refreshBoard); }
  return () => {
    listeners.delete(listener);
    if (!listeners.size && timer) { clearInterval(timer); timer = null; window.removeEventListener('strata:tasks', refreshBoard); }
  };
}

export function useBoard(): { board: BoardResult; ready: boolean; error: string; tasks: Task[] } {
  const [, tick] = useState(0);
  useEffect(() => subscribe(() => tick(n => n + 1)), []);
  const sessions = useStore(s => s.sessions), activeSession = useStore(s => s.activeSession), live = useStore(s => s.liveChats);
  // The clock moves the "waiting 4 min" labels and the seven-day window on, without needing a new fetch.
  const [minute, setMinute] = useState(() => Math.floor(Date.now() / 60_000));
  useEffect(() => { const t = setInterval(() => setMinute(Math.floor(Date.now() / 60_000)), 30_000); return () => clearInterval(t); }, []);
  const board = useMemo(
    () => boardCards({ tasks: shared.tasks, sessions, liveChats: liveChatIds(), loopChats: loopChatIds(), pending: shared.pending, activeSession, now: Date.now() }),
    // `live` (how many chats are mid-turn) changes whenever a turn starts or ends; the shared snapshot changes by object identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [shared, sessions, activeSession, live, minute],
  );
  return { board, ready: shared.ready, error: shared.error, tasks: shared.tasks };
}
