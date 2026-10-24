/**
 * The host's native "choose a folder" dialog, one per platform:
 *   macOS    osascript (choose folder)            -- unchanged
 *   Windows  PowerShell FolderBrowserDialog (STA, topmost owner so it never opens behind the app)
 *   Linux    zenity, then kdialog, then the desktop portal (gdbus), all with the environment the AppImage was started from
 * Where no dialog works the caller gets { unavailable: true } and the app shows a plain message and a field for the path.
 */
import { spawn } from 'node:child_process';
import { stat } from 'node:fs/promises';
import { exec } from './util.mjs';
import { externalEnvironment } from './external-env.mjs';

const MAC_SCRIPT = prompt => `POSIX path of (choose folder with prompt "${prompt.replace(/["\\]/g, '')}")`;

export function windowsFolderScript(prompt) {
  const text = prompt.replace(/'/g, "''");
  return [
    'Add-Type -AssemblyName System.Windows.Forms',
    '[Console]::OutputEncoding = [System.Text.Encoding]::UTF8',
    '$owner = New-Object System.Windows.Forms.Form',
    '$owner.TopMost = $true; $owner.ShowInTaskbar = $false; $owner.StartPosition = "CenterScreen"; $owner.Size = New-Object System.Drawing.Size(0, 0)',
    '$owner.Show(); $owner.Activate()',
    '$dialog = New-Object System.Windows.Forms.FolderBrowserDialog',
    `$dialog.Description = '${text}'`,
    '$dialog.ShowNewFolderButton = $true',
    // READY tells the host the window is about to open; a host that never sees it gives up and asks for the path in words (QA F06).
    '[Console]::Out.WriteLine("READY"); [Console]::Out.Flush()',
    'if ($dialog.ShowDialog($owner) -eq "OK") { [Console]::Out.Write($dialog.SelectedPath) }',
    '$owner.Close()',
  ].join('; ');
}

/** powershell.exe takes the script as UTF-16LE base64, so no quoting survives to be mangled. */
export function windowsFolderArgs(prompt) {
  return ['-NoProfile', '-NonInteractive', '-STA', '-EncodedCommand', Buffer.from(windowsFolderScript(prompt), 'utf16le').toString('base64')];
}

/** Strip trailing separators, but leave a drive or filesystem root intact. */
export function cleanChosenPath(value) {
  const text = String(value ?? '').trim();
  if (/^[A-Za-z]:[\\/]*$/.test(text)) return text.slice(0, 2) + '\\';
  return text.length > 1 ? text.replace(/[\\/]+$/, '') || text.slice(0, 1) : text;
}

const missing = error => error?.code === 'ENOENT';

/** What the script printed, without its READY line. */
const answerOf = stdout => String(stdout ?? '').split(/\r?\n/).filter(line => line.trim() && line.trim() !== 'READY').join('\n');

/**
 * Runs the Windows dialog and watches it. Resolves { stdout } when it ends, { noDialog: true } when its window never got as far as
 * opening (no READY within `readyMs`: PowerShell blocked, stuck or without a desktop), and rejects with { code } as exec does.
 * The person's own time with the open dialog is not limited (a day at most).
 */
export function watchedWindowsDialog(shell, args, { readyMs = 30_000, openMs = 24 * 3600_000, spawnProcess = spawn } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawnProcess(shell, args, { windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] });
    let stdout = '', ready = false, settled = false;
    child.stdout?.setEncoding?.('utf8');                 // a multi-byte letter of a Cyrillic folder must not be cut between two chunks
    const end = (fn, value) => { if (settled) return; settled = true; clearTimeout(timer); try { child.kill(); } catch { /* gone */ } fn(value); };
    const timer = setTimeout(() => end(resolve, ready ? { stdout: '' } : { noDialog: true }), readyMs);
    const later = () => { clearTimeout(timer); ready = true; setTimeout(() => end(resolve, { stdout: '' }), openMs).unref?.(); };
    child.stdout?.on('data', chunk => { stdout += chunk; if (!ready && /^READY\s*$/m.test(stdout)) { clearTimeout(timer); later(); } });
    child.on('error', error => end(reject, error));
    child.on('close', code => { if (settled) return; settled = true; clearTimeout(timer); code === 0 ? resolve({ stdout }) : reject(Object.assign(new Error('exit ' + code), { code })); });
  });
}

/**
 * Ask for a folder. Resolves { path } | { cancelled: true } | { unavailable: true, reason }.
 * `run` is injectable for tests: (command, args) => Promise<{ stdout }>.
 */
export async function chooseFolder(prompt, options = {}) {
  const { platform = process.platform, run = exec, portal = portalFolder, windowsDialog = options.run ? null : watchedWindowsDialog } = options;
  if (platform === 'darwin') {
    try {
      const { stdout } = await run('osascript', ['-e', MAC_SCRIPT(prompt)]);
      return stdout.trim() ? { path: cleanChosenPath(stdout) } : { cancelled: true };
    } catch {
      return { cancelled: true }; // dismissing the dialog is an exit code, not a failure
    }
  }
  if (platform === 'win32') {
    const shell = `${process.env.SystemRoot || 'C:\\Windows'}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe`;
    try {
      const done = windowsDialog ? await windowsDialog(shell, windowsFolderArgs(prompt)) : await run(shell, windowsFolderArgs(prompt));
      if (done.noDialog) return { unavailable: true, failed: true, reason: 'The folder window did not open. Type the folder path instead.' };
      const answer = answerOf(done.stdout).trim();
      return answer ? { path: cleanChosenPath(answer) } : { cancelled: true };
    } catch (error) {
      if (missing(error)) return { unavailable: true, reason: 'Windows PowerShell was not found.' };
      throw new Error('The folder chooser could not open.');
    }
  }
  return chooseLinuxFolder(prompt, { run, portal });
}

/** A chooser that is there but broke (the display is missing, a library does not load) must say so; a bare exit 1 is a cancel. */
const BROKEN = /unable to init server|cannot open display|can't open display|no protocol specified|error while loading shared libraries|symbol lookup error|version `[^']*' not found|cannot connect to|failed to connect|could not connect/i;
const firstLine = text => String(text ?? '').split(/\r?\n/).map(l => l.trim()).find(Boolean)?.slice(0, 160) ?? '';

/**
 * Linux: zenity, then kdialog, then the desktop portal's file chooser through gdbus. Every one of them runs with the environment the
 * AppImage was started from (external-env.mjs), not Strata's own libraries. A chooser that is missing is skipped; one that fails is
 * remembered and the next is tried; when none works the answer says why and the person types the path.
 */
async function chooseLinuxFolder(prompt, { run, portal }) {
  const env = externalEnvironment();
  const candidates = [
    ['zenity', ['--file-selection', '--directory', `--title=${prompt}`]],
    ['kdialog', ['--getexistingdirectory', '.', '--title', prompt]],
  ];
  const failures = [];
  for (const [command, args] of candidates) {
    try {
      const { stdout } = await run(command, args, { env });
      return stdout.trim() ? { path: cleanChosenPath(stdout) } : { cancelled: true };
    } catch (error) {
      if (missing(error)) continue;
      const detail = `${error?.stderr ?? ''}`;
      if (error?.code === 1 && !BROKEN.test(detail)) return { cancelled: true }; // both tools exit 1 when the dialog is dismissed
      failures.push(`${command}: ${firstLine(detail) || firstLine(error?.message) || `exit ${error?.code}`}`);
    }
  }
  try {
    const picked = await portal(prompt, { run, env });
    if (picked) return picked;
  } catch (error) {
    failures.push(`portal: ${firstLine(error?.message)}`);
  }
  return {
    unavailable: true, failed: failures.length > 0, ...(failures.length ? { detail: failures.join('; ') } : {}),
    reason: failures.length
      ? 'The folder chooser could not open. Install zenity or kdialog to pick a folder, or type the path.'
      : 'No folder chooser was found. Install zenity or kdialog to pick a folder, or type the path.',
  };
}

/**
 * xdg-desktop-portal FileChooser through gdbus. Resolves { path } | { cancelled: true } | null when there is no portal or no gdbus.
 * The answer arrives as a Response signal on the request the call returns, so a monitor is started first.
 */
export async function portalFolder(prompt, { run = exec, env = externalEnvironment(), spawnProcess = spawn, timeoutMs = 10 * 60_000 } = {}) {
  const monitor = spawnProcess('gdbus', ['monitor', '--session', '--dest', 'org.freedesktop.portal.Desktop'], { env, stdio: ['ignore', 'pipe', 'ignore'] });
  let seen = '';
  let ended = false;
  monitor.on('error', () => { ended = true; });
  monitor.on('close', () => { ended = true; });
  monitor.stdout?.on('data', chunk => { seen += chunk; });
  try {
    let call;
    try {
      call = await run('gdbus', ['call', '--session', '--dest', 'org.freedesktop.portal.Desktop', '--object-path', '/org/freedesktop/portal/desktop',
        '--method', 'org.freedesktop.portal.FileChooser.OpenFile', '', prompt, "{'directory': <true>, 'modal': <true>}"], { env });
    } catch {
      return null;                                           // no gdbus, or no portal on the session bus
    }
    const handle = /'(\/org\/freedesktop\/portal\/desktop\/request\/[^']+)'/.exec(call.stdout)?.[1];
    if (!handle) return null;
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const line = seen.split('\n').find(l => l.startsWith(`${handle}: org.freedesktop.portal.Request.Response`));
      if (line) {
        const m = /\(uint32 (\d+), (.*)\)\s*$/.exec(line);
        if (!m || m[1] !== '0') return { cancelled: true };
        const uri = /'(file:\/\/[^']*)'/.exec(m[2])?.[1];
        return uri ? { path: cleanChosenPath(decodeURIComponent(uri.replace(/^file:\/\//, ''))) } : { cancelled: true };
      }
      if (ended) return null;
      await new Promise(r => setTimeout(r, 100));
    }
    return { cancelled: true };
  } finally {
    try { monitor.kill(); } catch { /* already gone */ }
  }
}

export const TYPE_A_PATH = 'No folder chooser is available on this system, so type the folder path instead.';

/** The host's answer when no chooser works: a plain sentence and a code the app turns into a field for typing the path. */
export function chooserUnavailable(picked) {
  return { message: picked.reason, code: 'folder-chooser-unavailable', details: picked.detail };
}

/** A typed path stands in for the dialog: it must be an absolute directory that exists. `fileType` (a RegExp on the name) also lets a
 *  file of that kind through (the model page takes a folder or one .gguf file). Quotes around the path (Windows "Copy as path") go. */
export async function typedFolder(value, { isAbsolute, fileType = null }) {
  let text = typeof value === 'string' ? value.trim() : '';
  if (text.length > 1 && /^(["'])[\s\S]*\1$/.test(text)) text = text.slice(1, -1).trim();
  if (!text || text.includes('\0')) return null;
  if (!isAbsolute(text)) throw new Error(fileType ? 'Enter the full path of the folder or the file.' : 'Enter the full path of the folder.');
  let info;
  try { info = await stat(text); } catch { throw new Error(fileType ? 'That folder or file does not exist.' : 'That folder does not exist.'); }
  if (info.isFile() && fileType?.test(text)) return text;
  if (!info.isDirectory()) throw new Error(fileType ? 'Choose a folder, or a .gguf file.' : 'Choose a folder, not a file.');
  return cleanChosenPath(text);
}
