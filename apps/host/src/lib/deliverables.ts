import { api } from './client';
import type { PreviewArtifact } from './artifacts';
import { tidyAnswer } from './answer-text';

/**
 * Deliverables and grouped activity (notes-split-agent8.md, item 12). The host sends `deliverable` frames for every file the assistant wrote or
 * changed and `activity-group` frames for runs of tool use between two pieces of text. A reply keeps them in the order they came, so the text
 * that arrives between activities reads as one continuous piece.
 */
export type DeliverableKind = 'html' | 'markdown' | 'pdf' | 'csv' | 'image' | 'code' | 'text' | 'other';
export interface Deliverable { path: string; abs: string; name: string; kind: DeliverableKind; status: 'created' | 'changed'; size: number; lines: number | null; added: number; removed: number;
  /** Which write came last in this reply (set when the host's frames are merged): the newest is the run's main result. */ order?: number }
export interface GroupFile { path: string; name: string; kind: DeliverableKind; status: string; added: number; removed: number }
export interface ActivityGroup { id: string; closed: boolean; line: string; added: number; removed: number; files: GroupFile[] }
export type MessageBlock = { kind: 'text'; text: string } | ({ kind: 'group' } & ActivityGroup);

const KINDS = ['html', 'markdown', 'pdf', 'csv', 'image', 'code', 'text', 'other'];
const num = (v: unknown) => (Number.isFinite(Number(v)) ? Number(v) : 0);

export function readDeliverable(raw: unknown): Deliverable | null {
  if (!raw || typeof raw !== 'object') return null;
  const f = raw as Record<string, unknown>;
  if (typeof f.path !== 'string' || !f.path) return null;
  const name = typeof f.name === 'string' && f.name ? f.name : f.path.split(/[\\/]/).at(-1) ?? f.path;
  return {
    path: f.path, abs: typeof f.abs === 'string' ? f.abs : f.path, name, kind: KINDS.includes(String(f.kind)) ? f.kind as DeliverableKind : 'other',
    status: f.status === 'changed' ? 'changed' : 'created', size: num(f.size), lines: f.lines === null || f.lines === undefined ? null : num(f.lines), added: num(f.added), removed: num(f.removed),
  };
}

export function readGroup(raw: unknown): ActivityGroup | null {
  if (!raw || typeof raw !== 'object') return null;
  const g = raw as Record<string, unknown>;
  if (typeof g.id !== 'string' || !g.id) return null;
  const files = Array.isArray(g.files) ? g.files.flatMap((x): GroupFile[] => {
    const d = readDeliverable(x);
    return d ? [{ path: d.path, name: d.name, kind: d.kind, status: d.status, added: d.added, removed: d.removed }] : [];
  }) : [];
  return { id: g.id, closed: g.closed === true, line: typeof g.line === 'string' ? g.line : '', added: num(g.added), removed: num(g.removed), files };
}

/** The `skills-used` frame: the skills this reply used, each with the name the person gave it and a one-line description. */
export function readSkills(raw: unknown): { name: string; description: string }[] {
  const list = (raw as { skills?: unknown } | null)?.skills;
  if (!Array.isArray(list)) return [];
  return list.flatMap(x => {
    const s = x as Record<string, unknown>;
    return typeof s?.name === 'string' && s.name ? [{ name: s.name, description: typeof s.description === 'string' ? s.description : '' }] : [];
  }).slice(0, 20);
}

/** Add streamed text to the reply's ordered pieces: it continues the last text piece, or starts a new one after an activity line. */
export function appendText(blocks: MessageBlock[] | undefined, text: string): MessageBlock[] {
  const list = blocks ?? [], last = list.at(-1);
  if (last?.kind === 'text') return [...list.slice(0, -1), { kind: 'text', text: tidyAnswer(last.text + text) }];
  return [...list, { kind: 'text', text: tidyAnswer(text) }];
}

/** A group frame is re-sent while it grows: the same id replaces its line; a new id goes after what came before. */
export function upsertGroup(blocks: MessageBlock[] | undefined, group: ActivityGroup): MessageBlock[] {
  const list = blocks ?? [], at = list.findIndex(b => b.kind === 'group' && b.id === group.id);
  const next: MessageBlock = { kind: 'group', ...group };
  return at >= 0 ? list.map((b, i) => (i === at ? next : b)) : [...list, next];
}

/** The same file may be written more than once in a run: one card per path, with its latest numbers. */
export function mergeDeliverable(list: Deliverable[] | undefined, file: Deliverable): Deliverable[] {
  const have = list ?? [], at = have.findIndex(f => f.path === file.path);
  const order = have.reduce((n, f) => Math.max(n, f.order ?? 0), 0) + 1;
  if (at < 0) return [...have, { ...file, order }];
  const old = have[at];
  return have.map((f, i) => (i === at ? { ...file, order, status: old.status === 'created' ? 'created' : file.status, added: old.added + file.added, removed: old.removed + file.removed } : f));
}

/** The kinds the sidebar preview shows and a finished run may open by itself: pages, markdown, PDFs, pictures, tables and plain text (not code or unknown files). */
export const SIDEBAR_KINDS: DeliverableKind[] = ['html', 'markdown', 'pdf', 'image', 'csv', 'text'];

/**
 * The one file a finished run opens in the sidebar, or null. The file the answer names wins (the last one it mentions); otherwise the one
 * written last. Only kinds the sidebar can show count, so a run that made a script and a page opens the page, and several files open one.
 */
export function mainResult(files: Deliverable[] | undefined, answer: string): Deliverable | null {
  const shown = (files ?? []).filter(f => SIDEBAR_KINDS.includes(f.kind));
  if (!shown.length) return null;
  const text = answer.toLowerCase();
  let named: Deliverable | null = null, at = -1;
  for (const f of shown) {
    const mention = Math.max(text.lastIndexOf(f.name.toLowerCase()), text.lastIndexOf(f.path.toLowerCase()));
    if (mention > at) { at = mention; named = f; }
  }
  if (named) return named;
  return shown.reduce((best, f) => ((f.order ?? 0) >= (best.order ?? 0) ? f : best));
}

export const fileTypeLabel = (d: { kind: DeliverableKind; name: string }) => {
  const ext = d.name.includes('.') ? d.name.split('.').at(-1)!.toUpperCase() : '';
  return d.kind === 'markdown' ? 'MD' : d.kind === 'html' ? 'HTML' : d.kind === 'csv' ? 'CSV' : d.kind === 'pdf' ? 'PDF' : ext && ext.length <= 10 ? ext : d.kind.toUpperCase();
};

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const csvText = (rows: string[][]) => rows.map(r => r.map(c => (/[",\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(',')).join('\n');

interface PreviewAnswer { kind: DeliverableKind; name: string; path: string; size: number; url?: string; text?: string; rows?: string[][]; rawUrl?: string; note?: string }

/** The sidebar preview of a file: the host reads it (only inside the chat's folder) and the page draws it as a website, markdown, table, picture, PDF or code. */
export async function previewOf(sessionId: string, file: { path: string; name: string }): Promise<PreviewArtifact> {
  const a = await api.get<PreviewAnswer>(`/api/files/preview?sessionId=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(file.path)}`);
  const id = `file:${a.path}`, title = a.name || file.name;
  if (a.kind === 'image' && a.rawUrl) return { id, kind: 'image', title, url: a.rawUrl };
  if (a.kind === 'pdf' && a.rawUrl) return { id, kind: 'pdf', title, url: a.rawUrl };
  if (a.kind === 'html' && a.text !== undefined) return { id, kind: 'website', title, content: a.text };
  if (a.kind === 'markdown' && a.text !== undefined) return { id, kind: 'markdown', title, content: a.text };
  if (a.kind === 'csv' && a.rows) return { id, kind: 'table', title, content: csvText(a.rows) };
  if (a.text !== undefined) return { id, kind: 'code', title, content: a.text };
  throw new Error(a.note || 'This file has no preview. Open it in its own app.');
}

export const fileAction = {
  open: (sessionId: string, path: string, how: 'browser' | 'app' | 'folder') => api.post('/api/files/open', { sessionId, path, how }),
  fullPath: async (sessionId: string, path: string) => (await api.get<{ path: string }>(`/api/files/path?sessionId=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(path)}`)).path,
  attach: (sessionId: string, path: string) => api.post('/api/chat/attachments', { sessionId, path }),
};

export const rawUrlOf = (sessionId: string, path: string) => `/api/files/raw?sessionId=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(path)}`;
