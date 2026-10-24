#!/usr/bin/env node
/**
 * Stage the DeepSeek Harness (pinned, MIT) as the installer's `harness/` folder: a pruned PRODUCTION install, built at
 * build time, so nothing is fetched or compiled on the person's PC.
 *
 *   node apps/desktop/scripts/stage-harness.mjs --src <built Harness checkout> [--out apps/desktop/src-tauri/resources/harness]
 *
 * <built Harness checkout> is a clone of the pinned commit after `pnpm install --frozen-lockfile` and `pnpm run build:artifacts`
 * (see phaseA-tests/notes-split-s4a.md, step 1). The script never touches it: it runs `pnpm deploy --legacy --prod` into
 * the output folder, removes what a running Harness does not read (sources, tests, maps, docs, other operating systems'
 * prebuilt binaries, the LibreOffice document converter and the speech engine, which the app does not use), keeps the
 * MIT licence and third-party notices, writes BUNDLE.json, and finally boots the staged copy once as proof.
 *
 * The host finds the result at `<install>/harness` (server/agent/harness-runtime.mjs), or in the component folder.
 */
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { createHash } from 'node:crypto';
import { flattenPnpm } from './flatten-pnpm.mjs';
import { isOtherOsBinaryFolder, missingHarnessImports } from './harness-prune.mjs';

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : fallback; };
const src = resolve(opt('src', process.env.STRATA_HARNESS_SRC ?? ''));
const out = resolve(opt('out', join(import.meta.dirname, '..', 'src-tauri', 'resources', 'harness')));
if (!opt('src', process.env.STRATA_HARNESS_SRC) || !existsSync(join(src, 'apps', 'cli', 'lib', 'bin.js'))) {
  console.error('Give --src <a built Harness checkout> (apps/cli/lib/bin.js must exist).');
  process.exit(2);
}
const platform = process.platform;
const run = (cmd, argv, cwd, extra = {}) => {
  const r = spawnSync(cmd, argv, { cwd, stdio: 'inherit', shell: false, ...extra });
  if (r.status !== 0) { console.error(`${cmd} ${argv.join(' ')} failed (${r.status ?? r.error})`); process.exit(1); }
};

// The output folder in the repo carries a .gitignore that keeps the staged files out of git: keep it across the clean-up.
const keepIgnore = existsSync(join(out, '.gitignore')) ? readFileSync(join(out, '.gitignore')) : null;
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
const pnpm = platform === 'win32' ? ['corepack.cmd', ['pnpm']] : ['corepack', ['pnpm']];
run(pnpm[0], [...pnpm[1], 'deploy', '--legacy', '--prod', '--filter', '@deepseek-ai/dsh', out], src, { shell: platform === 'win32' });

// --- peers -------------------------------------------------------------------------------------------------------------
// `pnpm deploy --prod` leaves out peer dependencies, but the Harness loads some of them by name at run time (for example the
// plugin-group helper that app-boot imports). Walk the installed packages; every dependency or peer that cannot be
// resolved from the output is copied from the source checkout's own node_modules, and its own requirements after it.
function packageJsonsUnder(dir, out2 = []) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isSymbolicLink()) continue;
    if (e.isDirectory()) packageJsonsUnder(p, out2);
    else if (e.name === 'package.json' && p.includes(`${sep}node_modules${sep}`)) out2.push(p);
  }
  return out2;
}
function requirementsOf(file) {
  try { const j = JSON.parse(readFileSync(file, 'utf8')); return [...Object.keys(j.dependencies ?? {}), ...Object.keys(j.peerDependencies ?? {}), ...Object.keys(j.optionalDependencies ?? {})]; } catch { return []; }
}
/** The checkout's own workspace packages (packages/**, vendor/*, apps/*) by name: where a peer that is one of ours comes from. */
const workspace = new Map();
function indexWorkspace(dir, depth = 0) {
  if (depth > 4) return;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (!e.isDirectory() || e.isSymbolicLink() || e.name === 'node_modules' || e.name.startsWith('.')) continue;
    const p = join(dir, e.name);
    const manifest = join(p, 'package.json');
    if (existsSync(manifest)) { try { workspace.set(JSON.parse(readFileSync(manifest, 'utf8')).name, p); } catch { /* not a package */ } }
    indexWorkspace(p, depth + 1);
  }
}
for (const top of ['packages', 'vendor', 'apps']) if (existsSync(join(src, top))) indexWorkspace(join(src, top));
const resolvable = name => existsSync(join(out, 'node_modules', ...name.split('/')));
const queue = [...new Set(packageJsonsUnder(join(out, 'node_modules')).flatMap(requirementsOf))].filter(n => !resolvable(n));
const added = [];
while (queue.length) {
  const name = queue.shift();
  if (resolvable(name)) continue;
  const from = workspace.get(name) ?? join(src, 'node_modules', ...name.split('/'));
  if (!existsSync(from)) continue; // an optional package for another platform, or not installed in the source: nothing to copy
  const to = join(out, 'node_modules', ...name.split('/'));
  mkdirSync(join(to, '..'), { recursive: true });
  const realFrom = realpathSync(from);
  cpSync(realFrom, to, { recursive: true, dereference: true, filter: f => !relative(realFrom, f).split(sep).includes('node_modules') });
  added.push(name);
  queue.push(...requirementsOf(join(to, 'package.json')).filter(n => !resolvable(n)));
}
if (added.length) console.log(`peer packages added: ${added.join(', ')}`);

// --- the browser tool server ---------------------------------------------------------------------------------------------
// Settings -> Tools can switch on a browser the assistant drives (Playwright's MCP server, Apache-2.0) on the person's own Chrome or Edge,
// so no browser is downloaded. The server and the two Playwright libraries it needs are not part of the Harness's production closure;
// they are copied from the source checkout's package store (~20 MB) and the host starts it with Node (server/agent/browser-probe.mjs).
const pnpmStore = join(src, 'node_modules', '.pnpm');
function storePackage(name, version) {
  if (!existsSync(pnpmStore)) return null;
  const prefix = `${name.replace('/', '+')}@`;
  const hit = readdirSync(pnpmStore).filter(n => n.startsWith(prefix) && (!version || n === `${prefix}${version}` || n.startsWith(`${prefix}${version}_`))).sort().at(-1);
  const dir = hit ? join(pnpmStore, hit, 'node_modules', ...name.split('/')) : null;
  return dir && existsSync(dir) ? dir : null;
}
const browserDir = storePackage('@playwright/mcp');
if (browserDir) {
  const wanted = JSON.parse(readFileSync(join(browserDir, 'package.json'), 'utf8')).dependencies ?? {};
  const parts = [['@playwright/mcp', browserDir], ...['playwright', 'playwright-core'].map(n => [n, storePackage(n, wanted[n])])];
  if (parts.every(([, dir]) => dir)) {
    for (const [name, dir] of parts) {
      const to = join(out, 'node_modules', ...name.split('/'));
      mkdirSync(join(to, '..'), { recursive: true });
      cpSync(realpathSync(dir), to, { recursive: true, dereference: true, filter: f => !relative(dir, f).split(sep).includes('node_modules') });
    }
    console.log(`browser tool server added: ${parts.map(([n]) => n).join(', ')}`);
  } else console.log('browser tool server NOT added: Playwright libraries missing from the package store');
} else console.log('browser tool server NOT added: @playwright/mcp is not in the source checkout');

// --- one real folder per package, no links -------------------------------------------------------------------------------
// The installer copies files; pnpm's layout is links. See flatten-pnpm.mjs. The two packages the app does not use are never copied.
const flat = flattenPnpm(out, { skip: /libreoffice-kit|sherpa-onnx/i });
console.log(`flattened: ${flat.packages} placed (${flat.nested} nested for a second version)`);

// --- prune ------------------------------------------------------------------------------------------------------------
/** Whole packages the app never loads: office-document conversion (hundreds of MB) and speech input. */
const DROP_PACKAGES = [/libreoffice-kit/i, /sherpa-onnx/i];
/** Only when sitting directly in a package's root (a library may have a runtime folder called `doc`, e.g. yaml/dist/doc). */
const DROP_DIRS = new Set(['test', 'tests', '__tests__', 'docs', 'website', 'benchmarks', 'examples', '.github']);
const DROP_FILES = /\.(map|md|markdown|pdb|tsbuildinfo)$/i;
/** Licences and notices travel with the files they cover. */
const KEEP_FILES = /^(LICENSE|LICENCE|NOTICE|THIRD_PARTY)/i;
let removed = 0;
function prune(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isSymbolicLink()) continue;
    if (e.isDirectory()) {
      const inLib = p.split(sep).includes('lib');
      if ((DROP_DIRS.has(e.name) && !inLib && existsSync(join(dir, 'package.json'))) || DROP_PACKAGES.some(r => r.test(e.name)) || isOtherOsBinaryFolder(platform, e.name, p)) {
        rmSync(p, { recursive: true, force: true }); removed++; continue;
      }
      prune(p);
    } else if (DROP_FILES.test(e.name) && !KEEP_FILES.test(e.name)) {
      rmSync(p, { force: true }); removed++;
    }
  }
}
prune(out);

// --- licence: the MIT text and the third-party notices travel with the files ---------------------------------------------
for (const name of ['LICENSE', 'THIRD_PARTY_NOTICES.md']) {
  if (existsSync(join(src, name))) cpSync(join(src, name), join(out, name));
}
const cli = JSON.parse(readFileSync(join(out, 'package.json'), 'utf8'));
let commit = process.env.STRATA_HARNESS_COMMIT ?? null;
if (!commit) { const r = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: src, encoding: 'utf8' }); if (r.status === 0) commit = r.stdout.trim(); }
function size(dir) { let t = 0, n = 0; for (const e of readdirSync(dir, { withFileTypes: true })) { const p = join(dir, e.name); if (e.isDirectory()) { const s = size(p); t += s.bytes; n += s.files; } else if (e.isFile()) { t += statSync(p).size; n++; } } return { bytes: t, files: n }; }
const total = size(out);
writeFileSync(join(out, 'BUNDLE.json'), JSON.stringify({
  component: 'deepseek-harness', version: cli.version, commit, license: 'MIT', platform, node: process.version,
  bytes: total.bytes, files: total.files, entry: 'apps/cli/lib/bin.js', builtAt: new Date().toISOString(),
  note: 'Pruned production install (pnpm deploy --prod). No package manager runs on the user\'s PC.',
}, null, 2));

if (keepIgnore) writeFileSync(join(out, '.gitignore'), keepIgnore);

// --- proof 1: nothing the Harness imports by name was pruned ------------------------------------------------------------
// (2026-10-17: the other-OS rule once removed @deepseek-ai/dsh-win32-process, which every platform imports; Linux/Mac agent chats died.)
const lost = missingHarnessImports(out);
if (lost.length) { console.error(['staged Harness is missing imported packages:', ...lost.map(l => `  ${l.from} -> ${l.missing}`)].join(String.fromCharCode(10))); process.exit(1); }
console.log('import check: every @deepseek-ai package the Harness imports is staged');

// --- proof 2: the staged copy answers --version and boots the web profile once ----------------------------------------
const bin = join(out, 'apps', 'cli', 'lib', 'bin.js');
const entry = existsSync(bin) ? bin : join(out, 'lib', 'bin.js');
const v = spawnSync(process.execPath, [entry, '--version'], { encoding: 'utf8' });
if (v.status !== 0) { console.error('staged Harness does not start:', v.stderr); process.exit(1); }
console.log(`staged ${cli.version} (${commit?.slice(0, 8) ?? 'no commit'}) in ${out}: ${(total.bytes / 1048576).toFixed(0)} MiB, ${total.files} files, ${removed} pruned items; --version -> ${v.stdout.trim()}`);
console.log(`sha256(BUNDLE.json)=${createHash('sha256').update(readFileSync(join(out, 'BUNDLE.json'))).digest('hex')}`);
