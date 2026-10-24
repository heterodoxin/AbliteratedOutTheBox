import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { currentUserSid } from '../../server/private-file.mjs';

/** The effective protection of a file: mode 0600 on POSIX, the exact DACL (one protected entry for this user) on NTFS. */
export function assertPrivate(path) {
  if (process.platform !== 'win32') { assert.equal(statSync(path).mode & 0o777, 0o600); return; }
  const sddl = execFileSync(join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'), ['-NoProfile', '-NonInteractive', '-Command', `(Get-Acl -LiteralPath '${path.replace(/'/g, "''")}').Sddl`],
    { encoding: 'utf8', timeout: 30000, windowsHide: true }).trim();
  const dacl = sddl.slice(sddl.indexOf('D:'));
  assert.equal(dacl, `D:PAI(A;;FA;;;${currentUserSid()})`, `ACL is not owner-only: ${sddl}`);
}
