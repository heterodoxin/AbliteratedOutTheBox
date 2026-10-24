/**
 * The one place the App Host decides which Python to run. Every spawn of Python goes through it
 * (tuning, Time-skip, the revise command, the native MCP client, setup/calibration jobs, the native server).
 *
 * Order:
 *   1. STRATA_PYTHON, then STRATA_NATIVE_PYTHON      (an explicit choice is never second-guessed)
 *   1b. the Control component's own interpreter, when the app ships one (control-runtime.mjs findControl):
 *      the installed app never looks at the user's Python
 *   2. the repository's own venv: .venv\Scripts\python.exe on Windows, .venv/bin/python elsewhere
 *   3. Windows: `py -3` (the launcher), then python.exe; each one must actually start
 *      (the Microsoft Store alias in WindowsApps exists on PATH but only opens the Store)
 *      POSIX: Homebrew 3.11 / 3.10, then python3, then python
 * It returns { command, args } so `py -3` can be spawned as spawn(command, [...args, script]).
 */
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

export function venvPython(repo, platform = process.platform) {
  return join(repo, '.venv', platform === 'win32' ? 'Scripts' : 'bin', platform === 'win32' ? 'python.exe' : 'python');
}

/** Does this interpreter start? A bare name is looked up on PATH by the spawn itself. */
export function pythonStarts(command, args = []) {
  try {
    const result = spawnSync(command, [...args, '-c', 'import sys'], { timeout: 15000, windowsHide: true, stdio: 'ignore' });
    return !result.error && result.status === 0;
  } catch { return false; }
}

const cache = new Map();

export function resolvePython({ repo = null, control = null, env = process.env, platform = process.platform, exists = existsSync, starts = pythonStarts, useCache = true } = {}) {
  const key = JSON.stringify([repo, control?.python ?? null, env.STRATA_PYTHON, env.STRATA_NATIVE_PYTHON, platform]);
  if (useCache && cache.has(key)) return cache.get(key);
  const found = (() => {
    const explicit = env.STRATA_PYTHON || env.STRATA_NATIVE_PYTHON;
    if (explicit) return { command: explicit, args: [], source: 'environment' };
    if (control?.python) return { command: control.python, args: [], source: 'bundled' };
    if (repo) {
      const venv = venvPython(repo, platform);
      if (exists(venv)) return { command: venv, args: [], source: 'venv' };
    }
    if (platform === 'win32') {
      if (starts('py', ['-3'])) return { command: 'py', args: ['-3'], source: 'py launcher' };
      if (starts('python.exe')) return { command: 'python.exe', args: [], source: 'PATH' };
      return { command: 'python.exe', args: [], source: 'fallback' };
    }
    for (const brew of ['/opt/homebrew/bin/python3.11', '/opt/homebrew/bin/python3.10']) {
      if (exists(brew)) return { command: brew, args: [], source: 'homebrew' };
    }
    return { command: 'python3', args: [], source: 'PATH' };
  })();
  // A failed lookup (the fallback) is not cached: the person may install Python and retry.
  if (useCache && found.source !== 'fallback') cache.set(key, found);
  return found;
}

/** A shell-ready command prefix such as `"py" -3`, for the few places that run a command string. */
export function shellPrefix(python) {
  return [`"${python.command}"`, ...python.args].join(' ');
}
