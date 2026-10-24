/**
 * Locates and runs the pinned DeepSeek Harness (`dsh`) for the web profile.
 *
 * The Harness is a separate component, not part of the host: it is found through `STRATA_HARNESS_DIR`
 * (or `<components>/harness`), checked for the files a run needs, and started with the Strata profile
 * on a loopback port the OS picks. Nothing here downloads or installs anything.
 *
 * One process, one gateway, many sessions: every chat is a Harness session inside it, and cancelling
 * one chat never touches the process. Stopping the process is a lifecycle act (shutdown, or a model
 * route change while idle), never a way to implement Stop.
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { externalEnvironment } from '../external-env.mjs';
import { utf8Script } from '../powershell-utf8.mjs';

const URL_PATTERN = /http:\/\/127\.0\.0\.1:(\d+)\/\?token=([A-Za-z0-9_-]+)/;

/** Environment names that carry ambient credentials and must not reach the agent unless named. */
const SECRET_NAME = /(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)/i;
/** Names the agent and its shell tools need to behave like a normal user process. */
const KEEP = /^(PATH|PATHEXT|SYSTEMROOT|WINDIR|COMSPEC|TEMP|TMP|TMPDIR|HOME|USERPROFILE|HOMEDRIVE|HOMEPATH|APPDATA|LOCALAPPDATA|PROGRAMDATA|PROGRAMFILES(\(X86\))?|PROGRAMW6432|COMMONPROGRAMFILES(\(X86\))?|USERNAME|USERDOMAIN|COMPUTERNAME|OS|PROCESSOR_ARCHITECTURE|NUMBER_OF_PROCESSORS|LANG|LC_ALL|TERM|SHELL|USER|LOGNAME|XDG_.*|SYSTEMDRIVE|PSMODULEPATH)$/i;

/** The environment the agent gets: a filtered copy of the host's, plus exactly the names passed in `extra`. */
export function agentEnvironment(base = process.env, extra = {}) {
  const out = {};
  // The shell and tools the assistant runs for the person are the person's own programs: they get the environment the AppImage was
  // started from, never Strata's bundled libraries and GTK/GIO settings (external-env.mjs). Outside an AppImage this is `base`.
  for (const [k, v] of Object.entries(externalEnvironment(base))) {
    if (v === undefined || !KEEP.test(k) || SECRET_NAME.test(k)) continue;
    out[k] = v;
  }
  // DSH_PERMISSION_MODE is the composition default for a session no chat mode has touched (a child agent, a
  // seeded session): the strictest sandbox, which matches the Manual preset, never the Harness's wider default.
  return { ...out, DSH_TELEMETRY_DISABLED: '1', DSH_PERMISSION_MODE: 'read-only', NODE_USE_SYSTEM_CA: '1', ...extra };
}

/**
 * Find a usable Harness. Returns `{ ok:true, dir, bin, version }` or `{ ok:false, reason }`; the reason
 * is user-readable because it becomes the capability state shown in the app.
 */
export function resolveHarness({ env = process.env, componentsDir = null, bundledDir = null } = {}) {
  // Order: an explicit folder, the folder the installer ships next to the host, then the user's component folder.
  const candidates = [env.STRATA_HARNESS_DIR, bundledDir, componentsDir ? join(componentsDir, 'harness') : null].filter(Boolean);
  if (!candidates.length) return { ok: false, reason: 'The agent component (DeepSeek Harness) is not installed.' };
  for (const dir of candidates.map(c => resolve(c))) {
    // A source checkout keeps the CLI in apps/cli; the installer's pruned production copy IS the CLI package.
    const nested = join(dir, 'apps', 'cli', 'lib', 'bin.js');
    const bin = existsSync(nested) ? nested : join(dir, 'lib', 'bin.js');
    if (!existsSync(bin)) continue;
    if (!existsSync(join(dir, 'node_modules'))) return { ok: false, reason: `The agent component at ${dir} has no installed dependencies.` };
    let version = null;
    try { version = JSON.parse(readFileSync(existsSync(nested) ? join(dir, 'apps', 'cli', 'package.json') : join(dir, 'package.json'), 'utf8')).version ?? null; } catch { /* version is informational */ }
    return { ok: true, dir, bin, version };
  }
  return { ok: false, reason: `The agent component was not found at ${candidates.join(' or ')}. Install or repair it.` };
}

export class HarnessProcess {
  #child = null;
  /** @type {{baseUrl:string, token:string, port:number}|null} */
  info = null;
  exited = null;
  #onExit;

  constructor({ harness, home, patchFile, cwd, env = {}, nodePath = process.execPath, logFile = null, pidFile = null, onExit = () => {} }) {
    Object.assign(this, { harness, home, patchFile, cwd, extraEnv: env, nodePath, logFile, pidFile });
    this.#onExit = onExit;
  }

  get pid() { return this.#child?.pid ?? null; }
  get running() { return this.#child !== null && this.exited === null; }

  /** Start the agent and resolve with its loopback URL and launch token. */
  async start({ timeoutMs = 120_000 } = {}) {
    mkdirSync(this.home, { recursive: true });
    mkdirSync(this.cwd, { recursive: true });
    const args = [this.harness.bin, '--profile', 'web', '--patch', this.patchFile, '--port', '0', '--no-open'];
    const child = spawn(this.nodePath, args, {
      cwd: this.cwd, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
      env: agentEnvironment(process.env, { ...this.extraEnv, DSH_HOME: this.home }),
    });
    this.#child = child;
    let text = '';
    const log = [];
    const keep = chunk => { log.push(chunk.toString()); if (log.length > 200) log.shift(); };
    child.on('exit', (code, signal) => {
      this.exited = { code, signal };
      if (this.logFile) { try { writeFileSync(this.logFile, redact(log.join(''), Object.values(this.extraEnv))); } catch { /* best effort */ } }
      this.#onExit(this.exited);
    });
    child.stderr.on('data', keep);
    const info = await new Promise((resolveInfo, reject) => {
      const timer = setTimeout(() => reject(new Error(`the agent did not print its address within ${Math.round(timeoutMs / 1000)}s`)), timeoutMs);
      child.stdout.on('data', chunk => {
        keep(chunk);
        text += chunk.toString();
        const m = URL_PATTERN.exec(text);
        if (m) { clearTimeout(timer); resolveInfo({ port: Number(m[1]), token: m[2], baseUrl: `http://127.0.0.1:${m[1]}` }); }
      });
      child.on('exit', code => { clearTimeout(timer); reject(new Error(`the agent exited during start (code ${code}): ${redact(log.join('').slice(-600), Object.values(this.extraEnv))}`)); });
      child.on('error', e => { clearTimeout(timer); reject(e); });
    }).catch(async error => { await this.stop(); throw error; });
    this.info = info;
    if (this.pidFile) { try { writeFileSync(this.pidFile, JSON.stringify({ pid: child.pid, at: new Date().toISOString() })); } catch { /* the record is a courtesy */ } }
    return info;
  }

  /** Stop this process and its children. Only ever touches the PID this object started. */
  async stop({ graceMs = 3000 } = {}) {
    const child = this.#child;
    if (!child || this.exited) return;
    const done = new Promise(r => child.once('exit', r));
    if (process.platform === 'win32') {
      // /T takes the shells and PTYs the agent started; /PID is this child only.
      spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true });
    } else {
      child.kill('SIGTERM');
      setTimeout(() => { if (!this.exited) child.kill('SIGKILL'); }, graceMs).unref();
    }
    await Promise.race([done, new Promise(r => setTimeout(r, graceMs + 2000))]);
    if (this.pidFile) rmSync(this.pidFile, { force: true });
  }
}

/** The command line of a running process, or null. Used only to confirm a recorded PID is still our agent. */
function commandLineOf(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return null;
  if (process.platform === 'win32') {
    const r = spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', utf8Script(`(Get-CimInstance Win32_Process -Filter "ProcessId=${pid}").CommandLine`)], { encoding: 'utf8', windowsHide: true, timeout: 15000 });
    return r.status === 0 && r.stdout.trim() ? r.stdout.trim() : null;
  }
  const r = spawnSync('ps', ['-p', String(pid), '-o', 'args='], { encoding: 'utf8' });
  return r.status === 0 && r.stdout.trim() ? r.stdout.trim() : null;
}

/**
 * A host that died without stopping its agent leaves the agent running, still holding its sessions'
 * writer locks, so the next start could not resume a chat. The agent records its PID; at the next start
 * a recorded PID is stopped only when it is still a `dsh` web launch (never a reused PID).
 */
export function stopStaleAgent(pidFile) {
  if (!existsSync(pidFile)) return null;
  let pid = null;
  try { pid = Number(JSON.parse(readFileSync(pidFile, 'utf8')).pid); } catch { /* unreadable record */ }
  rmSync(pidFile, { force: true });
  const cmd = commandLineOf(pid);
  if (!cmd || !/lib[\\/]bin\.js/.test(cmd) || !/--profile web/.test(cmd)) return null;
  if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true });
  else { try { process.kill(pid, 'SIGTERM'); } catch { /* already gone */ } }
  return pid;
}

/** Remove any occurrence of a secret value from text bound for a log. */
export function redact(text, secrets) {
  let out = String(text);
  for (const s of secrets) if (typeof s === 'string' && s.length >= 6) out = out.split(s).join('[redacted]');
  return out.replace(/token=[A-Za-z0-9_-]+/g, 'token=[redacted]');
}
