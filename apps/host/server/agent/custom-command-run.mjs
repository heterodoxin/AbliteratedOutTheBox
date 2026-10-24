/**
 * Runs the command of a command tool the person wrote (Settings -> Tools). Split from custom-tools-mcp.mjs so it can be tested.
 *
 * T7 (tools and images review, 2026-10-22). It used to run every command with `shell: true` (cmd.exe on Windows, so `Get-Content x` failed with
 * "not recognized"), in whatever folder the host happened to run in (so `python tools/foo.py` could not find its file), and ended it after two
 * minutes with 20,000 characters of output, silently cut.
 *
 *   shell      'auto' (default): PowerShell for a Windows command that reads as PowerShell (Get-Content, Start-Process, $env:X, $(...)), cmd.exe for
 *              any other Windows command (so every command that worked keeps working), /bin/sh elsewhere. 'cmd', 'powershell', 'bash' and 'sh' name one.
 *   cwd        the folder to run in: the tool's own, or the one the assistant names in the call (it knows the chat's folder); else the person's home.
 *   timeoutSec the time allowed, 1 to 3600 (default 120). A command that runs over is ended together with the programs it started.
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { homedir } from 'node:os';

export const DEFAULT_TIMEOUT_SEC = 120;
export const MAX_TIMEOUT_SEC = 3600;
export const OUT_MAX = 20000;
export const SHELLS = ['auto', 'cmd', 'powershell', 'bash', 'sh'];

/** A Windows command that is PowerShell and would fail in cmd.exe. */
const POWERSHELL_LOOKS = /(^|[\s;|(&])[A-Z][A-Za-z]+-[A-Z][A-Za-z]+\b|\$\w|\$\(|\$env:|@\(|\|\s*(?:%|\?)\s*\{/;
export const looksLikePowerShell = command => POWERSHELL_LOOKS.test(String(command ?? ''));

let powershellFound;
/** pwsh when it is installed (PowerShell 7), else the Windows PowerShell every Windows has. */
function powershellExe(platform) {
  if (powershellFound) return powershellFound;
  if (platform !== 'win32') return (powershellFound = 'pwsh');
  try { if (spawnSync('where', ['pwsh'], { windowsHide: true, timeout: 5000 }).status === 0) return (powershellFound = 'pwsh'); } catch { /* none */ }
  return (powershellFound = 'powershell.exe');
}

/**
 * -> { file, args, shell } for spawn: which program runs `command`.
 * @param {string} command
 * @param {{shell?: string, platform?: string}} [options]
 */
export function planShell(command, { shell = 'auto', platform = process.platform } = {}) {
  const windows = platform === 'win32';
  let choice = SHELLS.includes(shell) ? shell : 'auto';
  if (choice === 'auto') choice = windows ? (looksLikePowerShell(command) ? 'powershell' : 'cmd') : 'sh';
  if (choice === 'powershell') return { file: powershellExe(platform), args: ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', command], shell: false, name: 'PowerShell' };
  if (choice === 'bash') return { file: 'bash', args: ['-c', command], shell: false, name: 'bash' };
  if (choice === 'sh') return windows ? { file: 'sh', args: ['-c', command], shell: false, name: 'sh' } : { file: command, args: [], shell: true, name: 'sh' };
  return { file: command, args: [], shell: true, name: windows ? 'cmd.exe' : 'sh' };       // cmd: spawn's shell:true is cmd.exe on Windows
}

/** The folder a command runs in: -> { cwd } or { error } with a plain sentence. */
export function resolveCwd(...candidates) {
  for (const candidate of candidates) {
    const text = typeof candidate === 'string' ? candidate.trim() : '';
    if (!text) continue;
    const expanded = text.replace(/^~(?=$|[\\/])/, homedir());
    try {
      if (!existsSync(expanded) || !statSync(expanded).isDirectory()) return { error: `The folder “${text}” does not exist, so the command was not run.` };
    } catch { return { error: `The folder “${text}” could not be opened, so the command was not run.` }; }
    return { cwd: expanded };
  }
  return { cwd: homedir() };
}

export const clampTimeoutSec = value => {
  const n = Number(value);
  return Number.isFinite(n) && n >= 1 ? Math.min(Math.floor(n), MAX_TIMEOUT_SEC) : DEFAULT_TIMEOUT_SEC;
};

function endTree(child) {
  try {
    if (process.platform === 'win32' && child.pid) spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, timeout: 10000 });
    else child.kill('SIGKILL');
  } catch { /* gone */ }
}

/**
 * @param {{command: string, input?: string, cwd?: string, callCwd?: string, shell?: string, timeoutSec?: number, platform?: string}} job
 * @returns {Promise<{ok: boolean, text: string}>}
 */
export function runCommand({ command, input = '', cwd, callCwd, shell = 'auto', timeoutSec, platform = process.platform }) {
  return new Promise(resolve => {
    const where = resolveCwd(callCwd, cwd);
    if (where.error) return resolve({ ok: false, text: where.error });
    const plan = planShell(command, { shell, platform });
    const limit = clampTimeoutSec(timeoutSec);
    let child;
    try {
      child = spawn(plan.file, plan.args, { shell: plan.shell, windowsHide: true, cwd: where.cwd, env: { ...process.env, STRATA_INPUT: String(input ?? '') }, stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (e) { return resolve({ ok: false, text: `The command could not start in ${plan.name}: ${e.message}` }); }
    let text = '', cut = false, timedOut = false;
    const take = chunk => { if (text.length < OUT_MAX + 1) text += chunk.toString(); else cut = true; };
    child.stdout.on('data', take); child.stderr.on('data', take);
    const timer = setTimeout(() => { timedOut = true; endTree(child); }, limit * 1000);
    child.on('error', e => { clearTimeout(timer); resolve({ ok: false, text: `The command could not start in ${plan.name}: ${e.message}` }); });
    child.on('close', code => {
      clearTimeout(timer);
      if (text.length > OUT_MAX) { cut = true; text = text.slice(0, OUT_MAX); }
      const out = text.trim() + (cut ? `\n\n(The output was cut after ${OUT_MAX.toLocaleString('en-US')} characters.)` : '');
      if (timedOut) return resolve({ ok: false, text: `The command was stopped after ${limit} seconds.${out ? '\n' + out : ''}` });
      resolve({ ok: code === 0, text: out || (code === 0 ? 'The command finished without any error. It printed nothing.' : `The command failed (exit code ${code}) and printed nothing.`) });
    });
  });
}
