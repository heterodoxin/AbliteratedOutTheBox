/**
 * A preload (`node --import acquire-spy.mjs server/index.mjs`) that watches a host process for anything that acquires
 * software, records each attempt as a JSON line in $ACQUIRE_SPY_LOG and blocks it:
 *   - fetch / http(s) requests / sockets to anything but loopback;
 *   - a child process that is a package manager, a compiler, git, a downloader or `python -m pip`.
 * The test that uses it fails if the log has a line. A host that was never allowed to acquire leaves it empty.
 */
import { appendFileSync } from 'node:fs';
import { basename } from 'node:path';
import childProcess from 'node:child_process';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';
import { syncBuiltinESMExports } from 'node:module';

const LOG = process.env.ACQUIRE_SPY_LOG;
const note = (kind, detail) => { try { appendFileSync(LOG, JSON.stringify({ kind, detail }) + '\n'); } catch { /* the test reads the log */ } };
const LOOPBACK = new Set(['127.0.0.1', 'localhost', '::1', '[::1]', '0.0.0.0', '']);
const FORBIDDEN = new Set(['pip', 'pip3', 'npm', 'npx', 'pnpm', 'yarn', 'git', 'curl', 'wget', 'cmake', 'ninja', 'cl', 'link', 'msbuild',
  'winget', 'choco', 'scoop', 'apt', 'apt-get', 'dnf', 'pacman', 'brew', 'make', 'gcc', 'g++', 'cc', 'clang', 'rustup', 'cargo']);

const hostOf = value => {
  try { return new URL(typeof value === 'string' ? value : value?.url ?? String(value)).hostname.replace(/^\[|\]$/g, ''); } catch { return ''; }
};

const realFetch = globalThis.fetch;
globalThis.fetch = (input, init) => {
  const host = hostOf(input);
  if (!LOOPBACK.has(host)) { note('fetch', String(input?.url ?? input)); return Promise.reject(new TypeError('blocked by the acquisition spy')); }
  return realFetch(input, init);
};

for (const [module, label] of [[http, 'http'], [https, 'https']]) {
  for (const name of ['request', 'get']) {
    const original = module[name];
    module[name] = function patched(...args) {
      const first = args[0];
      const host = typeof first === 'string' || first instanceof URL ? hostOf(first) : (first?.hostname ?? first?.host ?? '').replace(/:\d+$/, '');
      if (!LOOPBACK.has(host)) { note(`${label}.${name}`, String(host)); throw new Error('blocked by the acquisition spy'); }
      return original.apply(this, args);
    };
  }
}

const originalConnect = net.Socket.prototype.connect;
net.Socket.prototype.connect = function patched(...args) {
  const first = args[0];
  const host = typeof first === 'object' && first ? (first.host ?? 'localhost') : (typeof args[1] === 'string' ? args[1] : 'localhost');
  if (typeof first === 'object' && first?.path) return originalConnect.apply(this, args);
  if (!LOOPBACK.has(String(host))) { note('socket', String(host)); throw new Error('blocked by the acquisition spy'); }
  return originalConnect.apply(this, args);
};

const looksLikeAcquisition = (command, args) => {
  const name = basename(String(command)).toLowerCase().replace(/\.(exe|cmd|bat)$/, '');
  if (FORBIDDEN.has(name)) return true;
  const list = (Array.isArray(args) ? args : []).map(String);
  if (/^python[\d.]*$|^py$/.test(name) && (list.includes('pip') || list.some(a => /(^|[\/])setup\.py$/.test(a)))) return true;
  return false;
};
for (const name of ['spawn', 'spawnSync', 'execFile', 'execFileSync']) {
  const original = childProcess[name];
  childProcess[name] = function patched(command, ...rest) {
    const args = Array.isArray(rest[0]) ? rest[0] : [];
    if (looksLikeAcquisition(command, args)) { note(name, [command, ...args].join(' ')); throw new Error('blocked by the acquisition spy'); }
    return original.call(this, command, ...rest);
  };
}
for (const name of ['exec', 'execSync']) {
  const original = childProcess[name];
  childProcess[name] = function patched(line, ...rest) {
    const first = String(line).trim().split(/\s+/)[0];
    if (looksLikeAcquisition(first, String(line).split(/\s+/).slice(1))) { note(name, String(line)); throw new Error('blocked by the acquisition spy'); }
    return original.call(this, line, ...rest);
  };
}
syncBuiltinESMExports();
