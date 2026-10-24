/**
 * Finding the Cloudflare connector (cloudflared) that is already on this computer.
 *
 * Strata used to look only in its own folders, so a cloudflared installed by the system (apt, brew, winget, a .deb) was
 * reported as "not installed" (Linux tester, cloudflared 2026.5.0 in /usr/bin). Now the places are, in this order:
 *   1. Strata's own connector folder
 *   2. the optional connector component folder
 *   3. every folder on PATH
 *   4. the usual install places (/usr/bin, /usr/local/bin, ~/.local/bin, Homebrew, Program Files, winget/choco/scoop)
 * The first one that runs `cloudflared --version` and is new enough is used; the answer says which one and from where.
 * Nothing here downloads or installs anything.
 */
import { utf8Script } from './powershell-utf8.mjs';
import { execFile } from 'node:child_process';
import { access, stat } from 'node:fs/promises';
import { constants } from 'node:fs';
import { homedir } from 'node:os';
import { win32, posix } from 'node:path';

/** The quick tunnel flags Strata passes (tunnel --url, --no-autoupdate, --grace-period) and the "Registered tunnel connection" log line it waits for exist from 2022.1.0. */
export const MIN_VERSION = '2022.1.0';

export function parseVersion(text) {
  const m = /(\d{4})\.(\d{1,2})\.(\d{1,3})/.exec(String(text ?? ''));
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
}
export function versionAtLeast(found, minimum = MIN_VERSION) {
  const a = Array.isArray(found) ? found : parseVersion(found), b = Array.isArray(minimum) ? minimum : parseVersion(minimum);
  if (!a || !b) return true; // an unreadable version (a custom build) is not a reason to refuse it
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i];
  return true;
}

const SOURCE_LABEL = {
  strata: 'Strata’s own connector folder',
  component: 'the optional connector component',
  path: 'found on your PATH',
  system: 'a usual install location',
  setting: 'chosen with the STRATA_CLOUDFLARED setting',
};

/** Candidate paths in search order, each as { path, source }. Pure: no disk access. */
export function connectorCandidates({ directory, componentDirectory = null, platform = process.platform, env = process.env, home = homedir(), search = true } = {}) {
  // STRATA_CLOUDFLARED=<file> uses exactly that connector; STRATA_CLOUDFLARED_SEARCH=own keeps the search to Strata's own folders.
  const windows = platform === 'win32', p = windows ? win32 : posix;
  const name = windows ? 'cloudflared.exe' : 'cloudflared';
  if (env.STRATA_CLOUDFLARED) return [{ path: env.STRATA_CLOUDFLARED, source: 'setting' }];
  if (env.STRATA_CLOUDFLARED_SEARCH === 'own') search = false;
  const out = [], seen = new Set();
  const add = (dir, source) => {
    if (!dir) return;
    const path = p.join(dir, name), key = windows ? path.toLowerCase() : path;
    if (!seen.has(key)) { seen.add(key); out.push({ path, source }); }
  };
  add(directory, 'strata');
  add(componentDirectory, 'component');
  if (!search) return out;
  const pathVariable = windows ? (env.Path ?? env.PATH ?? '') : (env.PATH ?? '');
  for (const dir of pathVariable.split(windows ? ';' : ':')) add(dir.replace(/^"|"$/g, '').trim(), 'path');
  if (windows) {
    const roots = [env.ProgramFiles ?? 'C:\\Program Files', env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)'];
    for (const root of roots) add(win32.join(root, 'cloudflared'), 'system');
    if (env.LOCALAPPDATA) add(win32.join(env.LOCALAPPDATA, 'Microsoft', 'WinGet', 'Links'), 'system');
    add(win32.join(env.ProgramData ?? 'C:\\ProgramData', 'chocolatey', 'bin'), 'system');
    if (home) add(win32.join(home, 'scoop', 'shims'), 'system');
  } else {
    for (const dir of ['/usr/bin', '/usr/local/bin', '/opt/homebrew/bin', '/home/linuxbrew/.linuxbrew/bin', '/snap/bin', '/opt/cloudflared']) add(dir, 'system');
    if (home) { add(posix.join(home, '.local', 'bin'), 'system'); add(posix.join(home, 'bin'), 'system'); add(posix.join(home, '.cloudflared'), 'system'); }
  }
  return out;
}

/** Runs `<path> --version`; resolves to the combined output, rejects when it does not run. */
export function runVersion(path) {
  return new Promise((resolve, reject) => {
    execFile(path, ['--version'], { timeout: 10000, windowsHide: true, maxBuffer: 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) reject(error); else resolve(`${stdout}\n${stderr}`.trim());
    });
  });
}

async function isRunnableFile(path, platform) {
  try {
    if (!(await stat(path)).isFile()) return false;
    await access(path, platform === 'win32' ? constants.F_OK : constants.X_OK);
    return true;
  } catch { return false; }
}

/**
 * The connector to use, and what was seen on the way.
 * Resolves to { found: { path, version, source, label } | null, skipped: [{ path, source, reason, version? }] }.
 * `tooOld` is set when something was found but every copy is older than the minimum version.
 */
export async function findConnector(options = {}) {
  const platform = options.platform ?? process.platform;
  const run = options.run ?? runVersion;
  const usable = options.usable ?? (path => isRunnableFile(path, platform));
  const skipped = [];
  for (const candidate of connectorCandidates(options)) {
    if (!(await usable(candidate.path))) continue;
    let output;
    try { output = await run(candidate.path); }
    catch { skipped.push({ ...candidate, reason: 'does-not-run' }); continue; }
    if (!/cloudflared/i.test(output) && !parseVersion(output)) { skipped.push({ ...candidate, reason: 'not-cloudflared' }); continue; }
    const parsed = parseVersion(output), version = parsed ? parsed.join('.') : null;
    if (!versionAtLeast(parsed)) { skipped.push({ ...candidate, reason: 'too-old', version }); continue; }
    return { found: { path: candidate.path, version, source: candidate.source, label: SOURCE_LABEL[candidate.source] }, skipped };
  }
  return { found: null, skipped };
}

/** A person-readable sentence about what was found (shown in the Connector step and the review). */
export function describeConnector(found) {
  if (!found) return null;
  return `${found.version ? `cloudflared ${found.version}` : 'cloudflared'} · ${found.path} (${found.label})`;
}

/** A person-readable reason when nothing usable was found but something unusable was. Null when nothing was seen. */
export function skippedMessage(skipped) {
  const old = skipped.find(s => s.reason === 'too-old');
  if (old) return `The cloudflared at ${old.path} is version ${old.version}, which is too old. Strata needs ${MIN_VERSION} or newer. Update cloudflared, then retry.`;
  const broken = skipped.find(s => s.reason === 'does-not-run' || s.reason === 'not-cloudflared');
  if (broken) return `The file at ${broken.path} did not run as cloudflared. Reinstall cloudflared, then retry.`;
  return null;
}

/**
 * Which cloudflared processes run on this computer (a system service, one the person started, another program's tunnel)?
 * `exclude` is the list of pids that belong to Strata itself. Resolves to { running, kind: 'service' | 'process' | null, pids, procs }
 * where procs is [{ pid, command }] (the command line is for matching only: it can hold a tunnel token and is never shown).
 * Failing to look is "not running": a missing ps must never block the tunnel.
 */
export async function findRunningConnector({ platform = process.platform, exclude = [], run = defaultProbe } = {}) {
  const skip = new Set(exclude.filter(Number.isInteger));
  let procs = [];
  if (platform === 'win32') {
    const out = await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', utf8Script("Get-CimInstance Win32_Process -Filter \"Name='cloudflared.exe'\" | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")]);
    procs = out.split(/\r?\n/).map(l => /^(\d+)\t(.*)$/.exec(l)).filter(Boolean).map(m => ({ pid: Number(m[1]), command: m[2] }));
    if (!procs.length) {   // no PowerShell answer: at least the pids
      const list = await run('tasklist', ['/FI', 'IMAGENAME eq cloudflared.exe', '/NH', '/FO', 'CSV']);
      procs = [...list.matchAll(/"cloudflared\.exe","(\d+)"/gi)].map(m => ({ pid: Number(m[1]), command: '' }));
    }
  } else {
    const out = await run('ps', ['-axo', 'pid=,comm=,args=']);
    procs = out.split('\n').map(l => /^\s*(\d+)\s+(\S+)\s*(.*)$/.exec(l)).filter(m => m && /(^|\/)cloudflared$/.test(m[2])).map(m => ({ pid: Number(m[1]), command: m[3] }));
  }
  procs = procs.filter(p => !skip.has(p.pid));
  if (!procs.length) return { running: false, kind: null, pids: [], procs: [] };
  let kind = 'process';
  if (platform === 'linux') {
    const active = (await run('systemctl', ['is-active', 'cloudflared'])).trim();
    if (active === 'active') kind = 'service';
  } else if (platform === 'win32') {
    const service = await run('sc.exe', ['query', 'Cloudflared']);
    if (/RUNNING/.test(service)) kind = 'service';
  }
  return { running: true, kind, pids: procs.map(p => p.pid), procs };
}

/**
 * Split the running connectors into Strata's own tunnel (same port: its command line has --url pointing at `originUrl`'s port; same tunnel:
 * its command line holds the same token) and tunnels of something else, which are none of Strata's business.
 */
export function classifyRunning(running, { originUrl = '', token = '' } = {}) {
  const port = /:(\d+)\/?$/.exec(String(originUrl))?.[1] ?? null;
  const own = [], others = [];
  for (const p of running?.procs ?? []) {
    const cmd = p.command ?? '';
    const samePort = !!port && new RegExp(`--url[= ]\\S*:${port}(?:/|\\s|$)`).test(cmd);
    const sameToken = token.length >= 20 && cmd.includes(token);
    (samePort || sameToken ? own : others).push(p.pid);
  }
  return { own, others };
}

const pidList = pids => pids.join(', ');
/** Plain sentence about tunnels that are none of Strata's business (information only, never a refusal). */
export function otherTunnelMessage(pids) {
  if (!pids?.length) return null;
  return pids.length === 1 ? `cloudflared is already running for another tunnel (PID ${pids[0]}).` : `cloudflared is already running for other tunnels (PIDs ${pidList(pids)}).`;
}
/** The refusal for Strata's own tunnel already running. */
export function ownTunnelMessage(pids) {
  if (!pids?.length) return null;
  return `Strata’s tunnel for this address is already running (PID ${pidList(pids)}), so Strata did not start a second one. Stop it first (or use it as it is), then connect again here.`;
}

function defaultProbe(cmd, args) {
  return new Promise(resolve => {
    execFile(cmd, args, { timeout: 4000, windowsHide: true, maxBuffer: 1024 * 1024 }, (error, stdout) => resolve(error ? '' : String(stdout)));
  });
}
