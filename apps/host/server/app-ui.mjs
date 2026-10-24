/**
 * Which window the person is using, and what only the app window can do.
 *
 * The Linux app starts the host itself (the "shell" process, which has no window) and shows the page in a separate window
 * process. WebKitGTK can crash that process on some computers; the shell then keeps the host running and opens the same page
 * in the default browser ("browser mode"). A person can also choose browser mode (`--browser`, STRATA_UI=browser).
 *
 *   mode 'window'   the app window shows the page; the window process registers the app updater here (Tauri's signed updater).
 *   mode 'browser'  an ordinary browser tab shows the page. There is no window process, so the app cannot update itself, and
 *                   closing the tab does not stop the engine: the menu has "Quit Strata" (POST /api/app/quit) and the command
 *                   line has `--quit`.
 *   mode null       not started by the app (the host run by hand, a test, a source checkout): nothing here applies.
 *
 * What needs the Tauri bridge and what happens without it (browser mode):
 *   folder picker        works: the host runs zenity / kdialog / the desktop portal itself (folder-dialog.mjs), else the page asks for the path
 *   open a file / folder works: the host runs xdg-open itself (chat-files.mjs, artifacts.mjs)
 *   window controls      none to offer: the browser has its own tab and window; the page never called the shell for them
 *   notifications        the page does not use the shell for them; the host sends the one desktop notification it needs (notify-send)
 *   app updater          NOT available: Tauri's updater lives in the window process. Settings says so in plain words and the
 *                        engine and component updates (Strata Control) still work.
 */
import { execFile, spawn } from 'node:child_process';
import { externalEnvironment } from './external-env.mjs';
import { createAppUpdater } from './app-updater.mjs';

export const APP_UPDATE_UNAVAILABLE = 'app-update-unavailable';
export const BROWSER_UPDATE_MESSAGE = "The Strata app itself can't be updated in browser mode. Start Strata from its icon, without --browser, to update it.";
const BRIDGE_URL = /^http:\/\/127\.0\.0\.1:\d+$/;
const MODES = new Set(['window', 'browser']);

/** Browsers and openers to try, in order, when the address of the page is opened for the person. */
export const OPENERS = [['xdg-open'], ['gio', 'open'], ['sensible-browser'], ['x-www-browser'], ['firefox'], ['chromium'], ['google-chrome']];

/**
 * Start one opener and say whether it took the address: false when the program is missing or ends with an error within a couple of
 * seconds. xdg-open often stays running until the browser closes, so a program that is still running counts as success.
 */
export function startOpener(command, args, { spawnProcess = spawn, env = externalEnvironment(), settleMs = 2500 } = {}) {
  return new Promise(resolve => {
    let child;
    try { child = spawnProcess(command, args, { detached: true, stdio: 'ignore', env }); } catch { return resolve(false); }
    let done = false;
    const finish = value => { if (!done) { done = true; clearTimeout(timer); resolve(value); } };
    const timer = setTimeout(() => { try { child.unref?.(); } catch { /* ignore */ } finish(true); }, settleMs);
    child.on('error', () => finish(false));
    child.on('exit', code => finish(code === 0));
  });
}

export async function openInBrowser(url, { openers = OPENERS, start = startOpener } = {}) {
  for (const [command, ...rest] of openers) if (await start(command, [...rest, url])) return true;
  return false;
}

/** One desktop notification (notify-send), best effort: no notification daemon or no notify-send is not an error. */
export function sendNotification(message, { run = execFile, env = externalEnvironment() } = {}) {
  return new Promise(resolve => {
    try { run('notify-send', ['--app-name=Strata', '--icon=dialog-information', 'Strata', message], { env, timeout: 5000 }, error => resolve(!error)); }
    catch { resolve(false); }
  });
}

export function createAppUi({ env = process.env, shutdown = () => {}, open = openInBrowser, notify = sendNotification, createUpdater = createAppUpdater } = {}) {
  let mode = MODES.has(env.STRATA_UI_MODE) ? env.STRATA_UI_MODE : null;
  let reason = mode === 'browser' ? (env.STRATA_UI_REASON || 'chosen') : null;
  let bridge = null;
  const originalBridge = createUpdater({ env });   // the older way: the shell passed the bridge in the environment

  /** The shell's updater as the rest of the host sees it, or null when no app started this host. */
  const updater = mode === null && !originalBridge ? null : {
    get unavailable() { return mode === 'browser' ? BROWSER_UPDATE_MESSAGE : null; },
    async check() { return (await current()).check(); },
    async install(options) { return (await current()).install(options); },
  };
  async function current() {
    if (bridge) return bridge;
    if (originalBridge) return originalBridge;
    throw Object.assign(new Error(mode === 'browser' ? BROWSER_UPDATE_MESSAGE : 'The app updater is not ready yet. Try again in a moment.'), { code: APP_UPDATE_UNAVAILABLE });
  }

  return {
    updater,
    get mode() { return mode; },
    /** What the page reads to decide what to show. */
    state: () => ({ mode, reason, canQuit: mode === 'browser', appUpdate: mode === 'browser' || mode === null ? 'unavailable' : 'available' }),
    /** The window process hands over its updater bridge (a loopback address and a token). */
    setBridge({ url, token } = {}) {
      if (typeof url !== 'string' || !BRIDGE_URL.test(url) || typeof token !== 'string' || token.length < 16) throw new Error('That is not a bridge address.');
      bridge = createUpdater({ env: { STRATA_SHELL_BRIDGE: url, STRATA_SHELL_TOKEN: token } });
      return true;
    },
    /** Open the page in the default browser without changing the mode (a second start while Strata runs). */
    openPage: url => open(url),
    /** The shell moves everyone to a browser tab: the window could not start or has stopped. */
    async switchToBrowser({ url, reason: why = 'window-failed', notice = '', openPage = true } = {}) {
      mode = 'browser'; reason = String(why).slice(0, 40); bridge = null;
      const opened = openPage && url ? await open(url) : false;
      if (notice) void notify(notice);
      return { mode, reason, opened };
    },
    /** The "Quit Strata" menu entry: only in browser mode, where closing the tab leaves the engine running. */
    quit() {
      if (mode !== 'browser') return { ok: false, status: 409, error: 'Close the Strata window to quit.' };
      setImmediate(() => void shutdown());
      return { ok: true, status: 202 };
    },
  };
}
