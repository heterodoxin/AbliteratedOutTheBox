import { useEffect, useSyncExternalStore } from 'react';
import { api, ApiError } from './client';

/**
 * A chat's /plan and its queued messages as the host reports them (notes-split-agent6.md, "Backend contract for the ui5 worker"). The plan
 * arrives whole every time it changes, in the chat stream as {type:'plan', event, plan} and as the SSE event 'plan' {chatId, event, plan}; the
 * queue as {type:'queued-message', queue} and the SSE event 'queue' {chatId, queue}. A snapshot replaces what was there; null or an empty
 * list clears it. The host owns both; every button asks it and draws the answer.
 */
export type PlanState = 'proposed' | 'approved' | 'rejected' | 'running' | 'paused' | 'done' | 'stopped';
export type StepState = 'pending' | 'running' | 'done' | 'failed' | 'skipped';
export interface PlanStep { id: string; text: string; detail?: string; summary?: string; state: StepState }
export interface Plan { id: string; title: string; source: 'plan' | 'goal'; state: PlanState; steps: PlanStep[]; /** A plain sentence (an interface string) when the app stopped the planning turn early. */ caution?: string }
export interface QueuedMessage { id: string; text: string; deliverAt: 'step' | 'end' }

const PLAN_STATES = ['proposed', 'approved', 'rejected', 'running', 'paused', 'done', 'stopped'];
const STEP_STATES = ['pending', 'running', 'done', 'failed', 'skipped'];

export function readPlan(raw: unknown): Plan | null {
  if (!raw || typeof raw !== 'object') return null;
  const p = raw as Record<string, unknown>;
  if (typeof p.id !== 'string' || !p.id || !PLAN_STATES.includes(String(p.state)) || !Array.isArray(p.steps)) return null;
  const steps: PlanStep[] = [];
  p.steps.forEach((s, i) => {
    if (!s || typeof s !== 'object') return;
    const r = s as Record<string, unknown>;
    if (typeof r.text !== 'string' || !r.text.trim()) return;
    steps.push({
      id: typeof r.id === 'string' && r.id ? r.id : `s${i + 1}`, text: r.text,
      state: STEP_STATES.includes(String(r.state)) ? r.state as StepState : 'pending',
      ...(typeof r.detail === 'string' && r.detail ? { detail: r.detail } : {}),
      ...(typeof r.summary === 'string' && r.summary ? { summary: r.summary } : {}),
    });
  });
  if (!steps.length) return null;
  return { id: p.id, title: typeof p.title === 'string' ? p.title : '', source: p.source === 'goal' ? 'goal' : 'plan', state: p.state as PlanState, steps };
}

export function readQueue(raw: unknown): QueuedMessage[] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((q): QueuedMessage[] => {
    if (!q || typeof q !== 'object') return [];
    const r = q as Record<string, unknown>;
    return typeof r.id === 'string' && r.id && typeof r.text === 'string' && r.text
      ? [{ id: r.id, text: r.text, deliverAt: r.deliverAt === 'end' ? 'end' : 'step' }] : [];
  });
}

const plans = new Map<string, Plan>();
const queues = new Map<string, QueuedMessage[]>();
const listeners = new Set<() => void>();
const emit = () => listeners.forEach(fn => fn());
const subscribe = (fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn); }; };

export function applyPlanFrame(chatId: string, raw: unknown) {
  const plan = readPlan(raw), before = plans.get(chatId);
  if (!plan) { if (plans.delete(chatId)) emit(); return; }
  if (before && JSON.stringify(before) === JSON.stringify(plan)) return;
  plans.set(chatId, plan);
  emit();
}

export function applyQueueFrame(chatId: string, raw: unknown) {
  const items = readQueue(raw), before = queues.get(chatId) ?? [];
  if (JSON.stringify(before) === JSON.stringify(items)) return;
  if (items.length) queues.set(chatId, items); else queues.delete(chatId);
  emit();
}

const NONE: QueuedMessage[] = [];
export const planOf = (chatId: string | null | undefined): Plan | null => (chatId ? plans.get(chatId) ?? null : null);
export const queueOf = (chatId: string | null | undefined): QueuedMessage[] => (chatId ? queues.get(chatId) ?? NONE : NONE);

export function usePlan(chatId: string | null | undefined): Plan | null {
  const plan = useSyncExternalStore(subscribe, () => planOf(chatId), () => null);
  useEffect(() => { if (chatId) void loadPlan(chatId); }, [chatId]);
  return plan;
}

export function useQueue(chatId: string | null | undefined): QueuedMessage[] {
  const items = useSyncExternalStore(subscribe, () => queueOf(chatId), () => NONE);
  useEffect(() => { if (chatId) void loadQueue(chatId); }, [chatId]);
  return items;
}

export async function loadPlan(chatId: string) {
  try { applyPlanFrame(chatId, (await api.get<{ plan: unknown }>(`/api/plan?sessionId=${encodeURIComponent(chatId)}`)).plan); }
  catch { /* an older host has no plan route: no plan */ }
}

/** The waiting messages. Without `release` this only reads; the host never sends them from here. */
export async function loadQueue(chatId: string) {
  try { applyQueueFrame(chatId, (await api.get<{ queue: unknown }>(`/api/chat/queue?sessionId=${encodeURIComponent(chatId)}`)).queue); }
  catch { /* an older host has no queue route: nothing queued */ }
}

/** Save the edited steps. Each step keeps its grey detail line when its words did not change. */
export async function planEdit(chatId: string, plan: Plan, texts: string[]) {
  const steps = texts.map((text, i) => { const was = plan.steps[i]; return was && was.text === text && was.detail ? { text, detail: was.detail } : text; });
  const result = await api.post<{ plan: unknown }>('/api/plan/edit', { sessionId: chatId, planId: plan.id, steps });
  applyPlanFrame(chatId, result.plan);
}

export async function planReject(chatId: string, planId: string) {
  const result = await api.post<{ plan: unknown }>('/api/plan/reject', { sessionId: chatId, planId });
  applyPlanFrame(chatId, result.plan);
}

/** Approve: the host answers with the plan and the message that carries it out; the caller sends that message as an ordinary turn. */
export async function planApprove(chatId: string, planId: string): Promise<{ prompt: string }> {
  const result = await api.post<{ plan: unknown; prompt: string }>('/api/plan/approve', { sessionId: chatId, planId });
  applyPlanFrame(chatId, result.plan);
  return { prompt: String(result.prompt ?? '') };
}

/** Put a message behind the running goal. Returns false when no goal is running (409), so the caller can send it the normal way. */
export async function queueMessage(chatId: string, message: string): Promise<boolean> {
  try {
    const result = await api.post<{ queue: unknown }>('/api/chat/queue', { sessionId: chatId, message });
    applyQueueFrame(chatId, result.queue);
    return true;
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) return false;
    throw e;
  }
}

/** Take a queued message back before it is sent. */
export async function cancelQueued(chatId: string, id: string) {
  const result = await api.post<{ queue: unknown }>('/api/chat/queue/cancel', { sessionId: chatId, id });
  applyQueueFrame(chatId, result.queue);
}

/** The step the queued message waits for: the one running now, or the next that has not started. 1-based, 0 if there is none. */
export function currentStep(plan: Plan | null): number {
  if (!plan) return 0;
  const running = plan.steps.findIndex(s => s.state === 'running');
  if (running >= 0) return running + 1;
  const next = plan.steps.findIndex(s => s.state === 'pending');
  return next >= 0 ? next + 1 : 0;
}

/** The words under a queued message: when it will be sent. The first one goes out after the current step, the next ones after that. */
export function queuedWhen(item: QueuedMessage, index: number, plan: Plan | null): { id: string; args?: unknown[] } {
  if (item.deliverAt === 'end') return { id: 'Queued — sends when the goal finishes' };
  if (index > 0) return { id: 'Queued — sends after that' };
  const step = currentStep(plan);
  return step > 0 ? { id: 'Queued — sends when step {0} finishes', args: [step] } : { id: 'Queued — sends when the goal finishes' };
}

export const stepsDone = (plan: Plan) => plan.steps.filter(s => s.state === 'done').length;

/** The stored plan of a chat without asking the host (the one that asks is usePlan, used once per chat). */
export function usePlanQuiet(chatId: string | null | undefined): Plan | null {
  return useSyncExternalStore(subscribe, () => planOf(chatId), () => null);
}
