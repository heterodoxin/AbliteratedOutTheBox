/**
 * Files dropped on the app window on Linux and macOS. The page gets the drag events there but not the files (WebKit hides them), so the app
 * shell reads the drop, hands the paths to the host, and tells the page with these two events (apps/desktop/src-tauri/src/main.rs):
 *   strata:native-drag  detail 'enter' | 'leave'      the drop box shows or goes away
 *   strata:native-drop  detail [{ id, name, type }]    the host holds each file once under its id
 */
export const NATIVE_DRAG_EVENT = 'strata:native-drag';
export const NATIVE_DROP_EVENT = 'strata:native-drop';
export interface DroppedRef { id: string; name: string; type: string }

/** The files the shell announced; anything that is not a well-formed entry is left out. */
export function readDroppedRefs(detail: unknown): DroppedRef[] {
  if (!Array.isArray(detail)) return [];
  return detail.flatMap(x => {
    const r = x as Record<string, unknown>;
    return typeof r?.id === 'string' && /^[0-9a-f-]{36}$/.test(r.id) && typeof r.name === 'string' && r.name
      ? [{ id: r.id, name: r.name, type: typeof r.type === 'string' ? r.type : 'application/octet-stream' }] : [];
  }).slice(0, 10);
}

/** Fetch each announced file from the host and make it a File, the way a browser drop would have delivered it. A file that cannot be fetched is skipped. */
export async function fetchDropped(refs: DroppedRef[], fetcher: (url: string) => Promise<{ ok: boolean; blob(): Promise<Blob> }> = url => fetch(url)): Promise<File[]> {
  const files: File[] = [];
  for (const ref of refs) {
    try {
      const response = await fetcher(`/api/app/drop/${ref.id}`);
      if (response.ok) files.push(new File([await response.blob()], ref.name, { type: ref.type }));
    } catch { /* skipped */ }
  }
  return files;
}

/** True for a drag that carries files, as far as the page can tell (WebKit lists them as text/uri-list, others as Files). */
export const dragHasFiles = (types: ArrayLike<string> | null | undefined) => Array.from(types ?? []).some(t => t === 'Files' || t === 'text/uri-list');
