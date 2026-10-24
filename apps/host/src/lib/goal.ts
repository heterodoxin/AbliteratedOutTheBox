import { useEffect, useSyncExternalStore } from 'react';
import { api } from './client';

/** A chat's goal as the host reports it (notes-split-agent3.md): the bar above the input draws it; the host owns it. */
export interface Goal {
  objective: string;
  phase: 'active' | 'paused' | 'blocked' | 'complete';
  round: number;
  maxRounds: number;
  blocked?: string;
  /** While a plan runs: which step the goal is on (notes-split-agent6.md). */
  step?: { index: number; total: number; text: string };
}

const goals = new Map<string, Goal>();
const listeners = new Set<() => void>();
const emit = () => listeners.forEach(fn => fn());

/** A goal object from a frame, an event or a route answer; anything that is not one means "no goal". */
export function readGoal(raw: unknown): Goal | null {
  if (!raw || typeof raw !== 'object') return null;
  const g = raw as Record<string, unknown>;
  const phase = g.phase;
  if (typeof g.objective !== 'string' || !g.objective || !['active', 'paused', 'blocked', 'complete'].includes(String(phase))) return null;
  return {
    objective: g.objective, phase: phase as Goal['phase'],
    round: Number(g.round) || 0, maxRounds: Number(g.maxRounds) || 0,
    ...(typeof g.blocked === 'string' && g.blocked ? { blocked: g.blocked } : {}),
    ...readStep(g.step),
  };
}

function readStep(raw: unknown): { step?: Goal['step'] } {
  if (!raw || typeof raw !== 'object') return {};
  const s = raw as Record<string, unknown>;
  const index = Number(s.index), total = Number(s.total);
  return index > 0 && total >= index ? { step: { index, total, text: typeof s.text === 'string' ? s.text : '' } } : {};
}

export function applyGoalFrame(chatId: string, raw: unknown) {
  const goal = readGoal(raw);
  const before = goals.get(chatId);
  if (!goal) { if (goals.delete(chatId)) emit(); return; }
  if (before && JSON.stringify(before) === JSON.stringify(goal)) return;
  goals.set(chatId, goal);
  emit();
}

export const goalOf = (chatId: string | null | undefined): Goal | null => (chatId ? goals.get(chatId) ?? null : null);

export function useGoal(chatId: string | null | undefined): Goal | null {
  const goal = useSyncExternalStore(fn => { listeners.add(fn); return () => { listeners.delete(fn); }; }, () => goalOf(chatId), () => null);
  // Page load or a switch to another chat: ask the host once.
  useEffect(() => { if (chatId) void loadGoal(chatId); }, [chatId]);
  return goal;
}

export async function loadGoal(chatId: string) {
  try { applyGoalFrame(chatId, (await api.get<{ goal: unknown }>(`/api/goal?chatId=${encodeURIComponent(chatId)}`)).goal); }
  catch { /* an older host has no goal route: no bar */ }
}

/** Pause, change or remove the goal. The host answers at once with the goal as it now is. */
export async function goalAction(chatId: string, action: 'pause' | 'edit' | 'clear', text?: string) {
  const result = await api.post<{ goal: unknown }>(`/api/goal/${action}`, { chatId, ...(text !== undefined ? { text } : {}) });
  applyGoalFrame(chatId, result.goal);
}
