/** Shared helpers for the Strata host. */
import { randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { whichCommand } from './platform-paths.mjs';

export const exec = promisify(execFile);
export const uid = (prefix = 'id') => `${prefix}_${randomUUID().slice(0, 8)}`;
export const nowIso = () => new Date().toISOString();
export const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
export const round = (n, d = 1) => Number(n.toFixed(d));

export function json(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(payload),
    'cache-control': 'no-store',
  });
  res.end(payload);
}

export function error(res, status, message, extra = {}) {
  json(res, status, { error: message, ...extra });
}

export async function readBody(req, limit = 4 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new Error('Request body too large');
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  const raw = Buffer.concat(chunks).toString('utf8');
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error('Body must be JSON');
  }
}

/** Run a command, returning stdout or '' when it is missing or fails. */
export async function probe(cmd, args = [], timeout = 4000) {
  try {
    const { stdout } = await exec(cmd, args, { timeout, maxBuffer: 4 * 1024 * 1024 });
    return stdout.trim();
  } catch {
    return '';
  }
}

/** Where `binary` is on PATH, or null. `/usr/bin/env which` on POSIX, System32\where.exe on Windows (first match). */
export async function which(binary, { env = process.env, platform = process.platform, run = probe } = {}) {
  const { command, args } = whichCommand(binary, env, platform);
  const out = await run(command, args);
  return out.split(/\r?\n/)[0].trim() || null;
}

/** Minimal SSE hub. */
export class EventHub {
  #clients = new Set();

  add(res) {
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    });
    res.write(': connected\n\n');
    this.#clients.add(res);
    const ping = setInterval(() => res.write(': ping\n\n'), 15000);
    res.on('close', () => {
      clearInterval(ping);
      this.#clients.delete(res);
    });
  }

  send(event, payload) {
    const frame = `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
    for (const res of this.#clients) {
      try {
        res.write(frame);
      } catch {
        this.#clients.delete(res);
      }
    }
  }

  get size() {
    return this.#clients.size;
  }
}

/** The raw bytes of a request body (an upload), up to `limit`. */
export async function readRaw(req, limit = 16 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new Error('That upload is too big.');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}
