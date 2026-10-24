#!/usr/bin/env node
/**
 * One command for the whole dev loop.
 *
 *   npm run dev
 *
 * Starts the engine host (:8791) and the Vite dev server (:5199), prefixes their
 * output, and shuts both down together. Vite proxies /api to the host, so the
 * browser only ever talks to one origin — the same shape as the packaged app.
 */
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const HOST_PORT = process.env.STRATA_PORT ?? '8791';

const palette = { host: '\u001b[38;5;114m', web: '\u001b[38;5;111m', dim: '\u001b[2m', off: '\u001b[0m' };

/** @type {import('node:child_process').ChildProcess[]} */
const children = [];
let shuttingDown = false;

function run(name, command, args, env) {
  const child = spawn(command, args, {
    cwd: ROOT,
    env: { ...process.env, ...env },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const prefix = `${palette[name]}${name.padEnd(4)}${palette.off} ${palette.dim}│${palette.off} `;
  const pipe = (stream, target) => {
    let buffer = '';
    stream.on('data', (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) target.write(`${prefix}${line}\n`);
    });
  };
  pipe(child.stdout, process.stdout);
  pipe(child.stderr, process.stderr);
  child.on('exit', (code, signal) => {
    if (shuttingDown) return;
    console.log(`${prefix}exited (${signal ?? code})`);
    shutdown(code ?? 0);
  });
  children.push(child);
  return child;
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) child.kill('SIGTERM');
  setTimeout(() => process.exit(code), 250);
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));

console.log(`${palette.dim}strata dev · host :${HOST_PORT} · web :5199${palette.off}`);
// --watch restarts the host on server edits without taking the web server down.
run('host', process.execPath, ['--watch', join(ROOT, 'server', 'index.mjs')], { STRATA_PORT: HOST_PORT });
run('web', process.execPath, [join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js')], {
  STRATA_HOST: `http://127.0.0.1:${HOST_PORT}`,
});
