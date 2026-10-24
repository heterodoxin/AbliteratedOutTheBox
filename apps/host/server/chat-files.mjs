/**
 * The folder a chat works in, and what can be done with a file in it.
 *
 *   - Each chat can have its own project folder ("Select a folder" above the input). It becomes the agent's working folder and
 *     sandbox for that chat and is remembered per chat.
 *   - A file in that folder is opened by its PATH (file:///...), never through a local web address: in the browser, in its own app,
 *     shown in the folder (Explorer /select), or copied as a path. Only files inside the chat's folder (or one the agent made
 *     there) can be named, so a request cannot reach anywhere else.
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync, realpathSync } from 'node:fs';
import { basename, dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { externalEnvironment } from './external-env.mjs';

/** A folder no chat may use as its project: the top of a drive, or the system's own folders. */
export function refusedFolder(dir) {
  const p = resolve(dir);
  if (dirname(p) === p) return 'Choose a folder inside a drive, not the whole drive.';
  if (process.platform === 'win32') {
    const win = (process.env.SystemRoot ?? 'C:\\Windows').toLowerCase();
    const q = p.toLowerCase();
    if (q === win || q.startsWith(win + sep) || /^[a-z]:\\(program files( \(x86\))?|programdata)(\\|$)/.test(q)) return 'That is a system folder. Choose one of your own folders.';
  } else if (/^\/(bin|boot|dev|etc|lib|lib64|proc|root|run|sbin|sys|usr|var)(\/|$)/.test(p)) return 'That is a system folder. Choose one of your own folders.';
  return null;
}

export const isInside = (root, target) => {
  const rel = relative(resolve(root), resolve(target));
  return rel === '' || (!!rel && !rel.startsWith('..') && !isAbsolute(rel));
};

/** The chat -> folder table, kept in one small file. */
export function createChatFolders({ file }) {
  let table = {};
  try { const parsed = JSON.parse(readFileSync(file, 'utf8')); if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) table = parsed; } catch { /* first run */ }
  const save = () => {
    try { mkdirSync(dirname(file), { recursive: true }); const tmp = file + '.tmp'; writeFileSync(tmp, JSON.stringify(table, null, 2)); renameSync(tmp, file); } catch { /* kept in memory */ }
  };
  const alive = p => { try { return statSync(p).isDirectory(); } catch { return false; } };
  return {
    /** The folder of a chat, or null (none chosen, or it was deleted: then the chat falls back to the default). */
    get(chatId) { const p = table[chatId]; return typeof p === 'string' && alive(p) ? p : null; },
    info(chatId) { const p = table[chatId]; return p ? { path: p, name: basename(p) || p, exists: alive(p) } : null; },
    set(chatId, dir) {
      const target = resolve(String(dir ?? '').trim());
      if (!String(dir ?? '').trim() || !isAbsolute(target)) throw Object.assign(new Error('Choose a folder first.'), { code: 'folder-missing' });
      let st; try { st = statSync(target); } catch { throw Object.assign(new Error("That folder doesn't exist."), { code: 'folder-missing' }); }
      if (!st.isDirectory()) throw Object.assign(new Error("That's a file, not a folder."), { code: 'folder-missing' });
      const why = refusedFolder(target);
      if (why) throw Object.assign(new Error(why), { code: 'folder-refused' });
      table[chatId] = target; save();
      return { path: target, name: basename(target) || target, exists: true };
    },
    clear(chatId) { delete table[chatId]; save(); },
  };
}

/**
 * The folder a chat's files live in, one rule for the agent's writes, Open, Show in folder, preview and attach:
 *   1. the folder the person chose for this chat; else
 *   2. the folder the agent actually worked in for it (its session's working folder, so it is the one it wrote to); else
 *   3. the default workspace (Documents/Strata/Workspace, or STRATA_WORKSPACE), where a chat with no folder is run.
 * A folder that no longer exists is skipped. Never null while the default exists, so a chat with no chosen folder can still open its files.
 */
export function chatRootOf({ chosen = null, used = null, fallback = null } = {}) {
  const alive = p => { try { return !!p && statSync(p).isDirectory(); } catch { return false; } };
  for (const p of [chosen, used, fallback]) if (alive(p)) return resolve(p);
  return fallback ? resolve(fallback) : null;
}

/** The real path of a file the chat may touch, or throws a plain sentence. Symbolic links are followed before the check. */
export function resolveChatFile(root, rawPath) {
  if (!root) throw Object.assign(new Error('This chat has no folder yet. Choose one first.'), { code: 'no-folder' });
  const given = String(rawPath ?? '').trim();
  if (!given) throw Object.assign(new Error('Which file?'), { code: 'file-missing' });
  const full = isAbsolute(given) ? resolve(given) : resolve(root, given);
  let real; try { real = realpathSync(full); } catch { throw Object.assign(new Error('That file is no longer there.'), { code: 'file-missing' }); }
  let realRoot; try { realRoot = realpathSync(root); } catch { realRoot = resolve(root); }
  if (!isInside(realRoot, real)) throw Object.assign(new Error("That file is outside this chat's folder, so it was not opened."), { code: 'outside' });
  return real;
}

/**
 * Start a program and return at once: detached, no shared output, never waited for.
 * `STRATA_OPENER=record` (tests) records the command instead of starting anything.
 */
export function launch(command, args, { recorder = null } = {}) {
  if (process.env.STRATA_OPENER === 'record' || recorder) { (recorder ?? launch.recorded).push([command, ...args]); return; }
  // xdg-open and the browser it starts are the system's programs: they get the environment the AppImage was started from, not Strata's libraries.
  const child = spawn(command, args, { detached: true, stdio: 'ignore', windowsHide: false, env: externalEnvironment() });
  child.on('error', () => {});
  child.unref();
}
launch.recorded = [];

/** What "open" means on each system, for the browser, the file's own app, or the folder. */
export function openCommand(how, file, platform = process.platform) {
  const url = pathToFileURL(file).href;
  if (how === 'folder') {
    if (platform === 'win32') return ['explorer.exe', [`/select,${file}`]];
    if (platform === 'darwin') return ['open', ['-R', file]];
    return ['xdg-open', [dirname(file)]];
  }
  if (how === 'browser') {
    if (platform === 'win32') return ['rundll32.exe', ['url.dll,FileProtocolHandler', url]];   // the default browser, handed a file:// address
    if (platform === 'darwin') return ['open', [url]];
    return ['xdg-open', [url]];
  }
  if (platform === 'win32') return ['rundll32.exe', ['url.dll,FileProtocolHandler', file]];
  if (platform === 'darwin') return ['open', [file]];
  return ['xdg-open', [file]];
}

export function openFile(root, rawPath, how = 'app', { platform = process.platform, recorder = null } = {}) {
  if (!['browser', 'app', 'folder'].includes(how)) throw Object.assign(new Error('That way of opening is not known.'), { code: 'bad-how' });
  const file = resolveChatFile(root, rawPath);
  const [cmd, args] = openCommand(how, file, platform);
  launch(cmd, args, { recorder });
  return { ok: true, started: true, path: file, url: pathToFileURL(file).href, how };
}

// ---- previews, raw bytes and "attach as context" -----------------------------------------------------------------------------------

import { readFileSync as readBytes } from 'node:fs';
import { fileType } from './agent/deliverables.mjs';

const MIME = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', gif: 'image/gif', webp: 'image/webp', svg: 'image/svg+xml', bmp: 'image/bmp', ico: 'image/x-icon', pdf: 'application/pdf' };
export const mimeOf = file => MIME[String(file).split('.').pop().toLowerCase()] ?? 'application/octet-stream';
const MAX_PREVIEW_TEXT = 1_000_000;

/** A small CSV/TSV reader for the table preview: quotes, doubled quotes, commas/tabs inside quotes. */
export function parseCsv(text, delimiter = ',', maxRows = 500) {
  const rows = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length && rows.length < maxRows; i++) {
    const ch = text[i];
    if (quoted) { if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else quoted = false; } else cell += ch; }
    else if (ch === '"') quoted = true;
    else if (ch === delimiter) { row.push(cell); cell = ''; }
    else if (ch === '\n' || ch === '\r') { if (ch === '\r' && text[i + 1] === '\n') i++; row.push(cell); rows.push(row); row = []; cell = ''; }
    else cell += ch;
  }
  if (rows.length < maxRows && (cell !== '' || row.length)) { row.push(cell); rows.push(row); }
  return rows;
}

/**
 * What the sidebar needs to show a file: { kind, name, size, text?, rows?, rawUrl? }.
 * html: the text (the page renders it sandboxed); markdown: the text; csv: parsed rows; pdf and images: the address of the raw bytes
 * (/api/files/raw), which only serves a file inside the chat's folder; code and text: the text. Too big to show: kind only and a plain note.
 */
export function previewFile(root, rawPath, sessionId) {
  const file = resolveChatFile(root, rawPath);
  const st = statSync(file); const kind = fileType(file);
  const base = { kind, name: basename(file), path: file, size: st.size, url: pathToFileURL(file).href };
  const raw = `/api/files/raw?sessionId=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(file)}`;
  if (kind === 'image' || kind === 'pdf') return { ...base, rawUrl: raw, mime: mimeOf(file) };
  if (kind === 'other') return { ...base, note: 'This kind of file has no preview. Open it in its own app.' };
  if (st.size > MAX_PREVIEW_TEXT) return { ...base, note: 'This file is too big to show here. Open it in its own app.' };
  const buf = readBytes(file);
  if (buf.subarray(0, 2048).includes(0)) return { ...base, kind: 'other', note: 'This file is not text, so it has no preview.' };
  const text = buf.toString('utf8');
  if (kind === 'csv') return { ...base, rows: parseCsv(text, file.toLowerCase().endsWith('.tsv') ? '\t' : ','), truncated: text.split('\n').length > 500 };
  return { ...base, text };
}

/** The bytes of a picture or PDF inside the chat's folder. */
export function rawFile(root, rawPath) {
  const file = resolveChatFile(root, rawPath);
  const mime = mimeOf(file);
  if (mime === 'application/octet-stream') throw Object.assign(new Error('Only pictures and PDFs are served this way.'), { code: 'bad-type' });
  return { bytes: readBytes(file), mime };
}

/** Files the person attached from a deliverable; the next message to the chat names them (@path), so the assistant reads them again. */
export function createAttachments() {
  const table = new Map();
  return {
    list: chatId => [...(table.get(chatId) ?? [])],
    add(chatId, file) { const l = table.get(chatId) ?? []; if (!l.includes(file)) l.push(file); table.set(chatId, l.slice(-10)); return [...table.get(chatId)]; },
    remove(chatId, file) { const l = (table.get(chatId) ?? []).filter(f => f !== file); table.set(chatId, l); return l; },
    /** Take them for the next message (and forget them). */
    take(chatId) { const l = table.get(chatId) ?? []; table.delete(chatId); return l; },
  };
}
/** The message with its attachments named in front, as the assistant's instructions explain (@path is a file the person pointed at). */
export const withAttachments = (message, files) => (files.length ? files.map(f => '@' + (/\s/.test(f) ? '"' + f + '"' : f)).join(' ') + ' ' + message : message);
