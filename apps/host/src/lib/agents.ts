import type { SessionAgent } from './types';

const statuses = new Set(['working', 'completed', 'failed', 'queued', 'cancelled']);
const metric = (value: unknown, fallback = 0) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : fallback;

/** Merge partial agent frames without replacing other agents or their readings. */
export function updateAgent(agents: SessionAgent[], frame: unknown, now = Date.now()): SessionAgent[] {
  if (!frame || typeof frame !== 'object') return agents;
  const next = frame as Record<string, unknown>;
  if (typeof next.id !== 'string' || !next.id.trim()) return agents;
  const previous = agents.find(a => a.id === next.id);
  const at = metric(next.updatedAt, now);
  if (previous?.updatedAt && at < previous.updatedAt) return agents;
  const previousDuration = (previous?.durationMs ?? 0) + (previous?.status === 'working' && previous.updatedAt ? Math.max(0, at - previous.updatedAt) : 0);
  const agent: SessionAgent = {
    id: next.id,
    name: typeof next.name === 'string' ? next.name : previous?.name ?? 'Subagent',
    description: typeof next.description === 'string' ? next.description : previous?.description ?? '',
    status: statuses.has(String(next.status)) ? next.status as SessionAgent['status'] : previous?.status ?? 'queued',
    durationMs: metric(next.durationMs, previousDuration),
    tokens: metric(next.tokens, previous?.tokens),
    updatedAt: at,
  };
  return previous ? agents.map(a => a.id === agent.id ? agent : a) : [...agents, agent];
}

export const runningAgents = (agents: SessionAgent[]) => agents.filter(a => a.status === 'working').length;
