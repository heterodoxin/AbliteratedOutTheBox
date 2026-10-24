/**
 * Every build and staging script must at least parse. A string literal broken over two lines in stage-harness.mjs (2026-10-17) made
 * every Harness staging fail with a SyntaxError, on every operating system, and nothing noticed until a build was started.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';

const scripts = resolve(import.meta.dirname, '..', '..', 'desktop', 'scripts');
const packaging = resolve(import.meta.dirname, '..', '..', '..', 'packaging');
// On Windows a bare `bash` can be the WSL launcher in System32, which cannot read a C:\ path (exit 127): use Git bash when it is there.
const bash = process.platform === 'win32'
  ? ['C:/Program Files/Git/usr/bin/bash.exe', 'C:/Program Files/Git/bin/bash.exe'].find(p => existsSync(p)) ?? 'bash'
  : 'bash';

for (const file of readdirSync(scripts).filter(f => f.endsWith('.mjs'))) {
  test(`node --check ${file}`, () => {
    const r = spawnSync(process.execPath, ['--check', join(scripts, file)], { encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
  });
}

for (const file of readdirSync(scripts).filter(f => f.endsWith('.sh'))) {
  test(`bash -n ${file}`, t => {
    const probe = spawnSync(bash, ['--version'], { encoding: 'utf8' });
    if (probe.error || probe.status !== 0) { t.skip('no bash here'); return; }
    const r = spawnSync(bash, ['-n', join(scripts, file).replaceAll('\\', '/')], { encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
  });
}

for (const dir of [scripts, packaging]) {
  for (const file of readdirSync(dir).filter(f => f.endsWith('.py'))) {
    test(`python compile ${file}`, t => {
      const python = process.platform === 'win32' ? 'python' : 'python3';
      const probe = spawnSync(python, ['--version'], { encoding: 'utf8' });
      if (probe.error || probe.status !== 0) { t.skip('no python here'); return; }
      const r = spawnSync(python, ['-c', 'import ast,sys; ast.parse(open(sys.argv[1], encoding="utf-8").read())', join(dir, file)], { encoding: 'utf8' });
      assert.equal(r.status, 0, r.stderr);
    });
  }
}
