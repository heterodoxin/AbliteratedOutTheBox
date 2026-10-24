/** Owner-only files. POSIX: mode 0600. Windows: a protected ACL naming only the current user (NTFS has no mode bits). */
import { chmodSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

// Name the Windows tools by path: Git for Windows puts a different whoami first on PATH.
const system32 = name => join(process.env.SystemRoot || 'C:\\Windows', 'System32', name);

let userSid = null;
/** The current user's SID, read once with whoami (language independent). */
export function currentUserSid() {
  if (userSid) return userSid;
  const out = execFileSync(system32('whoami.exe'), ['/user', '/fo', 'csv', '/nh'], { encoding: 'utf8', timeout: 10000, windowsHide: true });
  const match = /S-1-\d+(?:-\d+)+/.exec(out);
  if (!match) throw new Error('Could not read the current Windows user.');
  return (userSid = match[0]);
}

export function makePrivate(path, { platform = process.platform } = {}) {
  if (platform !== 'win32') { chmodSync(path, 0o600); return; }
  // Grant first, then drop every inherited entry, so the file is never left without an owner entry.
  execFileSync(system32('icacls.exe'), [path, '/grant:r', `*${currentUserSid()}:F`, '/inheritance:r'],
    { stdio: 'ignore', timeout: 15000, windowsHide: true });
  // The owner too, so a file an elevated run made is not left owned by Administrators (needs the right to; best effort).
  try { execFileSync(system32('icacls.exe'), [path, '/setowner', `*${currentUserSid()}`], { stdio: 'ignore', timeout: 15000, windowsHide: true }); } catch { /* the explicit entry above already opens it to the user */ }
}
