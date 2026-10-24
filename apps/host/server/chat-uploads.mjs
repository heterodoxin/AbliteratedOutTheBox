/**
 * Files and pictures dropped or pasted into a chat.
 *
 * `POST /api/chat/upload?sessionId=&name=` takes the bytes in any of three forms: multipart/form-data (one or more files), a JSON body
 * { name, dataUrl } (a pasted or read picture), or the raw bytes with the file's type. A picture is kept as an upload the next message
 * can name (imageIds) and is shown again by its id; any other file is saved inside the chat's folder under .strata/attachments and named
 * in the next message (@path), so the assistant reads it. Nothing is ever written outside those two places.
 */

import { randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import { basename, extname, join } from 'node:path';

export const IMAGE_TYPES = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp' };
export const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;
export const MAX_IMAGE_BYTES = 6 * 1024 * 1024;
export const MAX_IMAGES = 4;

const safeName = name => String(name ?? 'file').replace(/[\\/:*?"<>|\x00-\x1f]/g, '_').replace(/^\.+/, '').slice(0, 120) || 'file';

/** The files of a multipart/form-data body: [{ name, type, bytes }]. */
export function parseMultipart(body, contentType) {
  const m = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(String(contentType ?? ''));
  if (!m) throw new Error('This upload is not readable. Try again.');
  const boundary = Buffer.from('--' + (m[1] ?? m[2]).trim());
  const files = [];
  let at = body.indexOf(boundary);
  while (at >= 0) {
    const next = body.indexOf(boundary, at + boundary.length);
    if (next < 0) break;
    let part = body.subarray(at + boundary.length, next);
    if (part.subarray(0, 2).toString() === '\r\n') part = part.subarray(2);
    const head = part.indexOf('\r\n\r\n');
    if (head > 0) {
      const headers = part.subarray(0, head).toString('utf8');
      let data = part.subarray(head + 4);
      if (data.subarray(-2).toString() === '\r\n') data = data.subarray(0, -2);
      const file = /filename="([^"]*)"/i.exec(headers);
      if (file) files.push({ name: file[1] || 'file', type: (/content-type:\s*([^\r\n;]+)/i.exec(headers)?.[1] ?? 'application/octet-stream').trim().toLowerCase(), bytes: Buffer.from(data) });
    }
    at = next;
  }
  if (!files.length) throw new Error('No file was in this upload.');
  return files;
}

const typeFromBytes = b => (b.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ? 'image/png' : b.subarray(0, 3).equals(Buffer.from([255, 216, 255])) ? 'image/jpeg' : b.subarray(0, 4).toString() === 'RIFF' && b.subarray(8, 12).toString() === 'WEBP' ? 'image/webp' : null);

export function createUploads({ dir, folderOf, attachments }) {
  mkdirSync(dir, { recursive: true });
  const index = new Map();   // id -> { id, chatId, name, type, size, file }
  const metaFile = id => join(dir, id + '.json');
  const load = id => {
    if (index.has(id)) return index.get(id);
    try { const m = JSON.parse(readFileSync(metaFile(id), 'utf8')); index.set(id, m); return m; } catch { return null; }
  };

  /** Store one file; returns what the page needs to show it. */
  function saveOne(chatId, f) {
    if (f.bytes.length > MAX_UPLOAD_BYTES) throw Object.assign(new Error(`"${safeName(f.name)}" is bigger than ${MAX_UPLOAD_BYTES / 1024 / 1024} MB.`), { code: 'too-big' });
    // A picture is recognised by its first bytes, not by the name or the type the sender claimed.
    const real = typeFromBytes(f.bytes);
    if (real) {
      if (f.bytes.length > MAX_IMAGE_BYTES) throw Object.assign(new Error(`The picture "${safeName(f.name)}" is bigger than ${MAX_IMAGE_BYTES / 1024 / 1024} MB. Choose a smaller one.`), { code: 'too-big' });
      const id = randomUUID();
      const file = join(dir, id + '.' + IMAGE_TYPES[real]);
      writeFileSync(file, f.bytes);
      const meta = { id, chatId, name: safeName(f.name), type: real, size: f.bytes.length, file, kind: 'image' };
      writeFileSync(metaFile(id), JSON.stringify(meta)); index.set(id, meta);
      return { id, kind: 'image', name: meta.name, type: real, size: meta.size, url: `/api/chat/upload/${id}` };
    }
    if (/^image\//i.test(f.type)) throw Object.assign(new Error(`"${safeName(f.name)}" is a picture type the model cannot read. Use PNG, JPEG or WebP.`), { code: 'bad-image' });
    const root = folderOf(chatId);
    const target = join(root, '.strata', 'attachments');
    mkdirSync(target, { recursive: true });
    let name = safeName(f.name), n = 1;
    while (existsSync(join(target, name))) name = `${basename(f.name, extname(f.name))}-${++n}${extname(f.name)}`.replace(/[\\/:*?"<>|]/g, '_');
    const path = join(target, name);
    writeFileSync(path, f.bytes);
    attachments.add(chatId, path);
    return { id: null, kind: 'file', name, type: f.type, size: f.bytes.length, path, attached: true };
  }

  return {
    /** Accept a request body in any of the three forms. */
    accept(chatId, { body, contentType, name }) {
      const ct = String(contentType ?? '');
      let files;
      if (/^multipart\/form-data/i.test(ct)) files = parseMultipart(body, ct);
      else if (/^application\/json/i.test(ct)) {
        const j = JSON.parse(body.toString('utf8') || '{}');
        const list = Array.isArray(j.files) ? j.files : [j];
        files = list.map(x => {
          const m = /^data:([^;,]+);base64,([A-Za-z0-9+/=]+)$/.exec(String(x.dataUrl ?? x.url ?? ''));
          if (!m) throw new Error('A picture must be sent as a data address (data:image/png;base64,...).');
          return { name: x.name ?? 'Picture', type: m[1], bytes: Buffer.from(m[2], 'base64') };
        });
      } else files = [{ name: name ?? 'file', type: ct || 'application/octet-stream', bytes: body }];
      if (files.length > 10) throw new Error('Drop up to ten files at a time.');
      return files.map(f => saveOne(chatId, f));
    },
    /** The picture of an upload as { bytes, type, name }, for the page and for the model. */
    image(id) {
      const m = load(String(id));
      if (!m || m.kind !== 'image') return null;
      try { return { bytes: readFileSync(m.file), type: m.type, name: m.name, chatId: m.chatId }; } catch { return null; }
    },
    /** The pictures a message names (imageIds), as data addresses the engine and the agent take. */
    dataUrls(chatId, ids) {
      const list = Array.isArray(ids) ? ids : [];
      if (list.length > MAX_IMAGES) throw new Error(`Attach up to ${MAX_IMAGES} pictures per message.`);
      return list.map(id => {
        const img = this.image(id);
        if (!img || (img.chatId && chatId && img.chatId !== chatId)) throw new Error('A picture is no longer available. Attach it again.');
        return { url: `data:${img.type};base64,${img.bytes.toString('base64')}`, name: img.name };
      });
    },
    /** Remove uploads older than a day (the folder never grows without end). */
    sweep(maxAgeMs = 24 * 3600_000) {
      let removed = 0;
      for (const f of readdirSync(dir)) { const p = join(dir, f); try { if (Date.now() - statSync(p).mtimeMs > maxAgeMs) { unlinkSync(p); removed += 1; } } catch { /* gone already */ } }
      return removed;
    },
  };
}
