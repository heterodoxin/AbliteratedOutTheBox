import { useEffect, useSyncExternalStore } from 'react';
import { api } from './client';
import { postWithFolderFallback } from './folder-pick';

/** A chat's project folder (notes-split-agent8.md, item 4): the assistant works inside it and can only touch files in it. */
export interface ChatFolder { path: string; name: string; exists: boolean }
const folders = new Map<string, ChatFolder | null>();
const listeners = new Set<() => void>();
const subscribe = (fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn); }; };
const emit = () => listeners.forEach(fn => fn());

export function readFolder(raw: unknown): ChatFolder | null {
  if (!raw || typeof raw !== 'object') return null;
  const f = raw as Record<string, unknown>;
  if (typeof f.path !== 'string' || !f.path) return null;
  return { path: f.path, name: typeof f.name === 'string' && f.name ? f.name : f.path.split(/[\/]/).filter(Boolean).at(-1) ?? f.path, exists: f.exists !== false };
}

function put(chatId: string, folder: ChatFolder | null) {
  if (JSON.stringify(folders.get(chatId) ?? null) === JSON.stringify(folder) && folders.has(chatId)) return;
  folders.set(chatId, folder); emit();
}

export const folderOf = (chatId: string | null | undefined) => (chatId ? folders.get(chatId) ?? null : null);

export function useChatFolder(chatId: string | null | undefined): ChatFolder | null {
  const value = useSyncExternalStore(subscribe, () => folderOf(chatId), () => null);
  useEffect(() => { if (chatId) void loadFolder(chatId); }, [chatId]);
  return value;
}

export async function loadFolder(chatId: string) {
  try { put(chatId, readFolder((await api.get<{ folder: unknown }>(`/api/chat/folder?sessionId=${encodeURIComponent(chatId)}`)).folder)); }
  catch { /* an older host has no folder route: no chip */ }
}

/** Ask the system for a folder (null when the person closes the dialog), then make it this chat's. */
export async function chooseChatFolder(chatId: string): Promise<'chosen' | 'cancelled'> {
  const picked = await postWithFolderFallback<{ path?: string; cancelled?: boolean }>('/api/workspaces/pick') as { path?: string };
  if (!picked.path) return 'cancelled';
  await setChatFolder(chatId, picked.path);
  return 'chosen';
}

/** `null` goes back to the default workspace. The host answers a plain sentence when the folder is not allowed (409 while the chat runs). */
export async function setChatFolder(chatId: string, path: string | null) {
  const result = await api.post<{ folder: unknown }>('/api/chat/folder', { sessionId: chatId, path });
  put(chatId, readFolder(result.folder));
}
