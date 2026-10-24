/**
 * Finds a browser already installed on this computer (Chrome, then Edge, then Chromium) for the assistant's browser tool, so nothing
 * big has to be downloaded. Returns the program's path and the name the screen shows.
 */
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

/** @returns {Array<{name: string, path: string}>} candidate programs for a platform, best first */
export function browserCandidates(platform = process.platform, env = process.env) {
  if (platform === 'win32') {
    const roots = [env.PROGRAMFILES, env['PROGRAMFILES(X86)'], env.LOCALAPPDATA].filter(Boolean);
    const at = rel => roots.map(r => join(r, rel));
    return [
      ...at('Google\\Chrome\\Application\\chrome.exe').map(path => ({ name: 'Google Chrome', path })),
      ...at('Microsoft\\Edge\\Application\\msedge.exe').map(path => ({ name: 'Microsoft Edge', path })),
      ...at('Chromium\\Application\\chrome.exe').map(path => ({ name: 'Chromium', path })),
    ];
  }
  if (platform === 'darwin') {
    return [
      { name: 'Google Chrome', path: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' },
      { name: 'Microsoft Edge', path: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge' },
      { name: 'Chromium', path: '/Applications/Chromium.app/Contents/MacOS/Chromium' },
    ];
  }
  return [
    ...['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/opt/google/chrome/chrome'].map(path => ({ name: 'Google Chrome', path })),
    ...['/usr/bin/microsoft-edge', '/usr/bin/microsoft-edge-stable', '/opt/microsoft/msedge/msedge'].map(path => ({ name: 'Microsoft Edge', path })),
    ...['/usr/bin/chromium', '/usr/bin/chromium-browser', '/snap/bin/chromium'].map(path => ({ name: 'Chromium', path })),
  ];
}

/** @returns {{found: boolean, name: string|null, path: string|null}} */
export function findBrowser({ platform = process.platform, env = process.env, exists = existsSync } = {}) {
  const hit = browserCandidates(platform, env).find(c => exists(c.path));
  if (hit) return { found: true, name: hit.name, path: hit.path };
  if (platform === 'linux') {
    for (const [name, bin] of [['Google Chrome', 'google-chrome'], ['Microsoft Edge', 'microsoft-edge'], ['Chromium', 'chromium']]) {
      const r = spawnSync('which', [bin], { encoding: 'utf8' });
      if (r.status === 0 && r.stdout.trim()) return { found: true, name, path: r.stdout.trim() };
    }
  }
  return { found: false, name: null, path: null };
}

/**
 * The browser tool server (Playwright's MCP, MIT/Apache) inside the agent component. The installer ships it next to the Harness; a source
 * checkout of the Harness keeps it under node_modules/.pnpm. Returns the file to run with Node, or null.
 */
export function findPlaywrightMcp(harnessDir, { exists = existsSync, list = readdirSync } = {}) {
  if (!harnessDir) return null;
  const direct = join(harnessDir, 'node_modules', '@playwright', 'mcp', 'cli.js');
  if (exists(direct)) return direct;
  try {
    const store = join(harnessDir, 'node_modules', '.pnpm');
    const hit = list(store).filter(n => /^@playwright\+mcp@/.test(n)).sort().at(-1);
    const nested = hit ? join(store, hit, 'node_modules', '@playwright', 'mcp', 'cli.js') : null;
    return nested && exists(nested) ? nested : null;
  } catch { return null; }
}
