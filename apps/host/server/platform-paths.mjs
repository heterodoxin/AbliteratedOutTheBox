/**
 * Where things live, and which system tool to call, per platform.
 *
 * The host used to assume POSIX in three places: it called /usr/bin/tar and /usr/bin/env, and it kept the tuning
 * and Time-skip state under $HOME/.local/state. On Windows those paths do not exist. State goes to
 * %LOCALAPPDATA%\Strata (the same place the Python side uses: strata_control/paths.py), and the system tools are
 * the ones Windows ships (System32\tar.exe, System32\where.exe). POSIX behavior is unchanged.
 *
 * Every function takes the environment and platform as arguments so tests can exercise both without a second OS.
 */
import { homedir } from 'node:os';
import { posix } from 'node:path';

const winPath = (...parts) => parts.join('\\');

/** The folder under which Strata keeps per-user state (tuning jobs, Time-skip runs). */
export function stateRoot(env = process.env, platform = process.platform, home = homedir()) {
  if (platform === 'win32') {
    const local = env.LOCALAPPDATA || winPath(env.USERPROFILE || home, 'AppData', 'Local');
    return winPath(local, 'Strata');
  }
  return posix.join(env.XDG_STATE_HOME || posix.join(env.HOME || '.', '.local', 'state'), 'strata');
}

const under = (root, platform, name) => (platform === 'win32' ? winPath(root, name) : posix.join(root, name));

export function tuningDir(env = process.env, platform = process.platform, home = homedir()) {
  return env.STRATA_TUNE_DIR || under(stateRoot(env, platform, home), platform, 'tuning');
}

export function timeSkipDir(env = process.env, platform = process.platform, home = homedir()) {
  return env.STRATA_TIMESKIP_DIR || under(stateRoot(env, platform, home), platform, 'time-skip');
}

function systemRoot(env) {
  return env.SystemRoot || env.SYSTEMROOT || env.windir || 'C:\\Windows';
}

/** The archive tool: Windows 10+ ships bsdtar as System32\tar.exe (it reads .zip and .tar.*); POSIX has /usr/bin/tar. */
export function tarBinary(env = process.env, platform = process.platform) {
  return platform === 'win32' ? winPath(systemRoot(env), 'System32', 'tar.exe') : '/usr/bin/tar';
}

/** The command that prints where a program is on PATH. */
export function whichCommand(binary, env = process.env, platform = process.platform) {
  if (platform === 'win32') return { command: winPath(systemRoot(env), 'System32', 'where.exe'), args: [binary] };
  return { command: '/usr/bin/env', args: ['which', binary] };
}
