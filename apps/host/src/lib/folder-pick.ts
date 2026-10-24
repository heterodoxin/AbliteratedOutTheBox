import { useSyncExternalStore } from 'react';
import { ApiError, api } from './client';

/**
 * Picking a folder when the system has no working chooser (Linux without zenity, kdialog or a desktop portal, or one that fails).
 * The host answers 501 with code `folder-chooser-unavailable` and a plain sentence; the app then asks for the path in a small dialog
 * and sends it back as `{ path }`, which the host checks the same way as a picked folder. Closing the dialog is a cancel.
 */
export interface FolderPrompt { message: string; resolve: (path: string | null) => void }

let current: FolderPrompt | null = null;
const listeners = new Set<() => void>();
const subscribe = (fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn); }; };
const emit = () => listeners.forEach(fn => fn());

export function useFolderPrompt() { return useSyncExternalStore(subscribe, () => current, () => null); }

/** Ask for a folder path in words; resolves with the typed text, or null when the person cancels. */
export function askForFolderPath(message: string): Promise<string | null> {
  return new Promise(resolve => {
    current?.resolve(null);
    current = { message, resolve: path => { current = null; emit(); resolve(path); } };
    emit();
  });
}

export const isChooserUnavailable = (e: unknown) => e instanceof ApiError && (e.body as { code?: string } | undefined)?.code === 'folder-chooser-unavailable';

/** POST to a host route that opens the system folder chooser; if there is none, ask for the path and POST it instead. */
export async function postWithFolderFallback<T>(url: string): Promise<T | { cancelled: true }> {
  try { return await api.post<T>(url, {}); }
  catch (e) {
    if (!isChooserUnavailable(e)) throw e;
    for (;;) {
      const typed = await askForFolderPath((e as ApiError).message);
      if (!typed) return { cancelled: true };
      try { return await api.post<T>(url, { path: typed }); }
      catch (again) {
        // A path the host refuses ("That folder does not exist.") is shown and asked for again; anything else is a real failure.
        if (!(again instanceof ApiError) || again.status !== 400) throw again;
        e = Object.assign(new ApiError(again.message, 501, { code: 'folder-chooser-unavailable' }));
      }
    }
  }
}

/** Is this page open on the computer that runs Strata? A folder window opens on that computer's screen, so a page opened from another one cannot use it. */
export function pageIsOnThisComputer(hostname: string): boolean {
  return /^(localhost|127(\.\d{1,3}){3}|\[?::1\]?)$/i.test(hostname);
}
