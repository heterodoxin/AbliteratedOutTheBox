import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { POWERSHELL_UTF8, utf8Script } from '../server/powershell-utf8.mjs';

test('the UTF-8 line comes first and leaves the script alone', () => {
  assert.equal(utf8Script('Get-Date'), POWERSHELL_UTF8 + 'Get-Date');
  assert.match(POWERSHELL_UTF8, /^\[Console\]::OutputEncoding = New-Object System\.Text\.UTF8Encoding \$false;/);
});

// #3202: PowerShell writes to a pipe in the OEM code page (cp866 on a Russian Windows, cp437 elsewhere); a Cyrillic folder came back
// as mojibake or "?". The same happens on any non-UTF-8 OEM page, so this fails on this machine without the line, too.
test('a Cyrillic temp folder name survives PowerShell through a pipe', { skip: process.platform !== 'win32' }, t => {
  const word = String.fromCharCode(0x414, 0x43e, 0x43a, 0x443, 0x43c, 0x435, 0x43d, 0x442, 0x44b);   // "Документы"
  const dir = mkdtempSync(join(tmpdir(), word + '-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const read = script => execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script],
    { encoding: 'utf8', windowsHide: true, env: { ...process.env, STRATA_UTF8_TEST: dir } }).trim();
  const ask = '(Get-Item -LiteralPath $env:STRATA_UTF8_TEST).FullName';
  const name = path => basename(path.replaceAll('\\', '/')).toLowerCase();   // the temp folder may come back as a short 8.3 path
  assert.equal(name(read(utf8Script(ask))), name(dir));
  assert.equal(name(read(utf8Script(`[Console]::Out.Write((${ask}))`))), name(dir));
});
