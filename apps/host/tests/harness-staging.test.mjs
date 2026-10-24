/**
 * The staged Harness keeps every package its plugins import (tester report 2026-10-17: Linux and Mac agent chats failed with
 * "session/create ... waiting for workflowEngine" because the other-OS prune removed @deepseek-ai/dsh-win32-process).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, sep } from 'node:path';
import { isHarnessCodePackage, isOtherOsBinaryFolder, missingHarnessImports } from '../../desktop/scripts/harness-prune.mjs';

const nm = (...p) => ['harness', 'node_modules', ...p].join(sep);

test('the Harness own code packages are never pruned, on any platform', () => {
  for (const platform of ['linux', 'darwin', 'win32']) {
    for (const name of ['dsh-win32-process', 'dsh-darwin-thing', 'dsh-linux-thing', 'dsh-sandbox-windows-acl']) {
      assert.equal(isOtherOsBinaryFolder(platform, name, nm('@deepseek-ai', name)), false, `${platform}: ${name}`);
    }
  }
  assert.equal(isHarnessCodePackage(nm('@deepseek-ai', 'dsh-win32-process')), true);
  assert.equal(isHarnessCodePackage(nm('@deepseek-ai', 'dsh-win32-process', 'lib')), false);
});

test('other systems prebuilt binaries are still dropped', () => {
  assert.equal(isOtherOsBinaryFolder('linux', 'sharp-win32-x64', nm('@img', 'sharp-win32-x64')), true);
  assert.equal(isOtherOsBinaryFolder('linux', 'musl', nm('@deepseek-ai', 'node-addon-system-linux-x64', 'bin', 'musl')), true);
  assert.equal(isOtherOsBinaryFolder('darwin', 'win32-x64', nm('node-pty', 'prebuilds', 'win32-x64')), true);
  assert.equal(isOtherOsBinaryFolder('win32', 'linux-x64', nm('node-pty', 'prebuilds', 'linux-x64')), true);
  // this platform's own binaries stay
  assert.equal(isOtherOsBinaryFolder('linux', 'linux-x64', nm('node-pty', 'prebuilds', 'linux-x64')), false);
});

test('missingHarnessImports names a package a plugin imports that is not staged', () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-stage-'));
  try {
    const scope = join(root, 'node_modules', '@deepseek-ai');
    mkdirSync(join(scope, 'dsh-subprocess-local', 'lib'), { recursive: true });
    writeFileSync(join(scope, 'dsh-subprocess-local', 'lib', 'index.js'), 'import { x } from "@deepseek-ai/dsh-win32-process";\nawait import("@deepseek-ai/dsh-here");\n');
    mkdirSync(join(scope, 'dsh-here'), { recursive: true });
    assert.deepEqual(missingHarnessImports(root), [{ from: 'dsh-subprocess-local', missing: '@deepseek-ai/dsh-win32-process' }]);
    mkdirSync(join(scope, 'dsh-win32-process'), { recursive: true });
    assert.deepEqual(missingHarnessImports(root), []);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('a staged Harness (STRATA_HARNESS_DIR) has every package its plugins import', { skip: !process.env.STRATA_HARNESS_DIR }, () => {
  assert.deepEqual(missingHarnessImports(process.env.STRATA_HARNESS_DIR), []);
});
