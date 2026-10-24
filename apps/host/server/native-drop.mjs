/**
 * Files dropped on the app window (Linux and macOS).
 *
 * The page cannot read a dragged file there: WebKit gives it the drag events but no files. The app shell sees the drop with the real paths and
 * hands them to the host with its shell secret (POST /api/app/drop); the host reads the files and keeps them for a short time under
 * unguessable ids; the page asks for each id once (GET /api/app/drop/<id>) and attaches the files as if they had been dropped in a browser.
 * Only regular files up to MAX_BYTES, at most MAX_FILES per drop; folders and anything else are skipped.
 */
import { randomUUID } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import { basename, extname, isAbsolute } from 'node:path';

export const MAX_FILES = 10, MAX_BYTES = 25 * 1024 * 1024, KEEP_MS = 2 * 60_000;

const TYPES = {
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', gif: 'image/gif', webp: 'image/webp', bmp: 'image/bmp',
  txt: 'text/plain', md: 'text/markdown', markdown: 'text/markdown', csv: 'text/csv', json: 'application/json', log: 'text/plain',
  html: 'text/html', htm: 'text/html', xml: 'text/xml', yaml: 'text/yaml', yml: 'text/yaml', pdf: 'application/pdf',
};
export const typeOfName = name => TYPES[extname(String(name)).slice(1).toLowerCase()] ?? 'application/octet-stream';

export function createNativeDrops({ now = Date.now, read = readFile, info = stat } = {}) {
  const held = new Map();   // id -> { name, type, bytes, at }
  const sweep = () => { for (const [id, f] of held) if (now() - f.at > KEEP_MS) held.delete(id); };
  return {
    /** Read the dropped paths; answer what the page may ask for: [{ id, name, type, size }]. */
    async add(paths) {
      sweep();
      const out = [];
      for (const raw of (Array.isArray(paths) ? paths : []).slice(0, MAX_FILES)) {
        const path = typeof raw === 'string' ? raw : '';
        if (!path || path.includes('\0') || !isAbsolute(path)) continue;
        try {
          const st = await info(path);
          if (!st.isFile() || st.size > MAX_BYTES) continue;
          const bytes = await read(path);
          const id = randomUUID();
          const name = basename(path);
          held.set(id, { name, type: typeOfName(name), bytes, at: now() });
          out.push({ id, name, type: typeOfName(name), size: bytes.length });
        } catch { /* an unreadable file is skipped, the rest still arrive */ }
      }
      return out;
    },
    /** One use per id. */
    take(id) { sweep(); const f = held.get(id); if (f) held.delete(id); return f ?? null; },
  };
}
