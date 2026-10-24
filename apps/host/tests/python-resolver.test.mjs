import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolvePython, shellPrefix, venvPython } from '../server/python-resolver.mjs';

const none = () => false;
const pick = (options) => resolvePython({ useCache: false, env: {}, exists: none, starts: none, ...options });

test('an explicit STRATA_PYTHON wins over everything', () => {
  assert.deepEqual(pick({ platform: 'win32', repo: 'C:\\r', env: { STRATA_PYTHON: 'D:\\py\\python.exe' }, exists: () => true }),
    { command: 'D:\\py\\python.exe', args: [], source: 'environment' });
  assert.equal(pick({ platform: 'linux', env: { STRATA_NATIVE_PYTHON: '/opt/py' } }).command, '/opt/py');
});

test('the repository venv is used with the platform layout', () => {
  assert.match(venvPython('C:\\repo', 'win32'), /Scripts[\\/]python\.exe$/);
  assert.match(venvPython('/repo', 'linux'), /bin[\\/]python$/);
  const seen = [];
  const found = pick({ platform: 'win32', repo: 'C:\\repo', exists: (p) => { seen.push(p); return /Scripts[\\/]python\.exe$/.test(p); } });
  assert.equal(found.source, 'venv');
  assert.match(found.command, /Scripts[\\/]python\.exe$/);
  assert.ok(seen.length >= 1);
});

test('Windows falls back to py -3, then python.exe, and skips an interpreter that does not start', () => {
  const tried = [];
  const starts = (command, args = []) => { tried.push([command, ...args].join(' ')); return command === 'py'; };
  assert.deepEqual(pick({ platform: 'win32', starts }), { command: 'py', args: ['-3'], source: 'py launcher' });
  const second = pick({ platform: 'win32', starts: (command) => command === 'python.exe' });
  assert.deepEqual(second, { command: 'python.exe', args: [], source: 'PATH' });
  // the Microsoft Store alias exists on PATH but fails to start: nothing works, so the plain name is returned for a clear error
  assert.equal(pick({ platform: 'win32' }).command, 'python.exe');
  assert.equal(pick({ platform: 'win32' }).source, 'fallback');
});

test('POSIX keeps Homebrew first, then python3', () => {
  assert.equal(pick({ platform: 'darwin', exists: (p) => p.endsWith('python3.11') }).command, '/opt/homebrew/bin/python3.11');
  assert.deepEqual(pick({ platform: 'linux' }), { command: 'python3', args: [], source: 'PATH' });
});

test('the shell prefix keeps the launcher arguments', () => {
  assert.equal(shellPrefix({ command: 'py', args: ['-3'] }), '"py" -3');
  assert.equal(shellPrefix({ command: 'python3', args: [] }), '"python3"');
});

test('the real resolver finds a Python that starts on this machine', () => {
  const found = resolvePython({ useCache: false, env: {} });
  const { spawnSync } = process.getBuiltinModule('node:child_process');
  const run = spawnSync(found.command, [...found.args, '-c', 'import sys; print(sys.version_info[0])'], { encoding: 'utf8', windowsHide: true });
  assert.equal(run.stdout.trim(), '3');
});

test('no server module spawns a bare python3 any more', () => {
  const dir = resolve(fileURLToPath(new URL('../server', import.meta.url)));
  for (const name of readdirSync(dir).filter((n) => /\.(mjs|cjs)$/.test(n))) {
    const text = readFileSync(join(dir, name), 'utf8');
    assert.doesNotMatch(text, /spawn\w*\(\s*['"]python3?['"]/, `${name} spawns python directly`);
    if (name !== 'python-resolver.mjs') assert.doesNotMatch(text, /\|\|\s*'python3'/, `${name} defaults to python3`);
  }
});
