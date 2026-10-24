/**
 * Which files and folders of the staged Harness are dropped, and the check that nothing the Harness imports is gone.
 *
 * The 2026-10-17 tester report: on Linux and Mac every agent chat failed ("session/create ... waiting for workflowEngine").
 * Cause: the old rule "drop any folder named after another operating system" also removed the Harness's own JavaScript package
 * `@deepseek-ai/dsh-win32-process`. It is plain JS (no binary), and `dsh-subprocess-local` and `dsh-sandbox-local` import it by
 * name on every platform, so those two plugins failed to load, and everything waiting on them (shell, subprocess, ptcRuntime,
 * workflowEngine) never started. Windows kept it (the rule differs there), so the Windows tests passed.
 */
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, sep } from 'node:path';

export const OTHER_OS = {
  win32: /(^|[\\/._-])(darwin|linux|android|freebsd|musl)([\\/._-]|$)/i,
  darwin: /(^|[\\/._-])(win32|linux|android|freebsd|musl)([\\/._-]|$)/i,
  // Linux: the AppImage bundler (linuxdeploy) resolves the libraries of every ELF file under resources and stops on one it
  // cannot satisfy (koffi's musl build wants libc.musl). The app is glibc x86_64, so musl and other-CPU builds are dropped.
  linux: /(^|[\\/._-])(win32|darwin|android|freebsd|musl|arm64|aarch64|armv7|arm|ia32|riscv64|ppc64|s390x|loong64)([\\/._-]|$)/i,
};
const NATIVE_FOLDER = /prebuild|bin|native|platform|-(win32|darwin|linux)/i;

/** The Harness's own JavaScript packages (`@deepseek-ai/dsh-*`) are code every platform imports by name; never drop one. */
export function isHarnessCodePackage(path) {
  const parts = path.split(sep);
  const i = parts.lastIndexOf('@deepseek-ai');
  return i >= 0 && i === parts.length - 2 && parts[i + 1].startsWith('dsh-');
}

/** True when the folder `path` (a directory named `name`) is another operating system's prebuilt binary and can go. */
export function isOtherOsBinaryFolder(platform, name, path) {
  const rule = OTHER_OS[platform] ?? OTHER_OS.linux;
  if (isHarnessCodePackage(path)) return false;
  return rule.test(name) && NATIVE_FOLDER.test(path);
}

/** Imports the Windows copy (which works) is also without: browser-side or dev-only bundles, and the document converter the app drops on purpose. */
const KNOWN_ABSENT = new Set([
  '@deepseek-ai/libreoffice-kit', '@deepseek-ai/dsh-client-ui-dockkit', '@deepseek-ai/dsh-client-web',
  '@deepseek-ai/dsh-experimental-webworker-runtime', '@deepseek-ai/dsh-remote-mock', '@deepseek-ai/dsh-client-test-runtime',
]);

function jsFiles(dir, out = []) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isSymbolicLink()) continue;
    if (e.isDirectory()) jsFiles(p, out);
    else if (/\.(m?js|cjs)$/.test(e.name)) out.push(p);
  }
  return out;
}

/**
 * Every bare `@deepseek-ai/<package>` the staged Harness's own packages import (static `from "x"` and `import("x")`) must exist
 * in the staged node_modules. Returns the list of `{from, missing}` problems (empty = good). Plugins that are only loaded on
 * another platform still count: Node resolves a static import whether or not the code runs.
 */
export function missingHarnessImports(harnessDir) {
  const scope = join(harnessDir, 'node_modules', '@deepseek-ai');
  if (!existsSync(scope)) return [{ from: scope, missing: '@deepseek-ai (no installed packages)' }];
  const problems = [];
  const seen = new Set();
  const re = /(?:from|import)\s*\(?\s*["'](@deepseek-ai\/[^"'/]+)(?:\/[^"']*)?["']/g;
  for (const pkg of readdirSync(scope)) {
    const lib = join(scope, pkg, 'lib');
    if (!existsSync(lib)) continue;
    for (const file of jsFiles(lib)) {
      const text = readFileSync(file, 'utf8');
      for (const m of text.matchAll(re)) {
        const name = m[1];
        const key = `${pkg}>${name}`;
        if (seen.has(key) || KNOWN_ABSENT.has(name)) continue;
        seen.add(key);
        if (!existsSync(join(harnessDir, 'node_modules', ...name.split('/')))
          && !existsSync(join(scope, pkg, 'node_modules', ...name.split('/')))) problems.push({ from: `${pkg}`, missing: name });
      }
    }
  }
  return problems;
}
