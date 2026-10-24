/**
 * Turn a pnpm "isolated" node_modules (a virtual store in `.pnpm` full of links) into an ordinary hoisted tree of real folders.
 *
 * Why: the installer copies the Harness folder file by file. pnpm's layout is thousands of links (junctions on Windows); a copy
 * that follows them duplicates every package once per dependent and one that does not breaks on the first dangling link. A hoisted
 * tree has each package once (nested only where two versions of the same name are needed), no links, and Node resolves it
 * exactly as it resolves any npm install.
 *
 * Algorithm (the one npm uses, in short): the packages the root names come first, at the top. Then, walking each placed package's
 * own dependencies: look the name up from the package's own folder upwards; if the same instance is already visible, nothing to
 * do; if a DIFFERENT instance of that name is visible, nest this one inside the dependent; if none is visible, hoist it to the top.
 *
 * @param {string} out        folder holding `node_modules`
 * @param {{skip?: RegExp}} [options]  package names never copied (the caller removes them anyway)
 * @returns {{packages:number, copied:number, nested:number}}
 */
import { cpSync, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, renameSync, rmSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';

export function flattenPnpm(out, { skip = null } = {}) {
  const nm = join(out, 'node_modules');
  const store = join(nm, '.pnpm');
  const flatRoot = join(out, 'node_modules.flat');
  rmSync(flatRoot, { recursive: true, force: true });
  mkdirSync(join(flatRoot, 'node_modules'), { recursive: true });

  const manifest = dir => { try { return JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8')); } catch { return null; } };
  const nameOf = dir => manifest(dir)?.name ?? null;
  /**
   * What makes two instances "the same package": name and version. pnpm keeps one copy per set of peer packages (the same workspace
   * package under many hashed folders); in one hoisted tree they are one package.
   */
  const identity = dir => { const m = manifest(dir); return m ? `${m.name}@${m.version}` : dir; };
  /** The sibling links of a store instance: its dependencies, by name -> real instance folder. */
  function storeDeps(instance, name) {
    const container = name.startsWith('@') ? dirname(dirname(instance)) : dirname(instance);
    const found = new Map();
    for (const e of readdirSync(container, { withFileTypes: true })) {
      if (e.name === '.bin') continue;
      const entries = e.name.startsWith('@') ? readdirSync(join(container, e.name), { withFileTypes: true }).map(x => [`${e.name}/${x.name}`, join(container, e.name, x.name)]) : [[e.name, join(container, e.name)]];
      for (const [dep, p] of entries) {
        if (dep === name) continue;
        try { if (lstatSync(p).isSymbolicLink() || existsSync(p)) found.set(dep, realpathSync(p)); } catch { /* a dangling link: the package was removed on purpose */ }
      }
    }
    return found;
  }
  /** Every instance in the store by package name, for packages that arrived without a store entry (copied peers). */
  const byName = new Map();
  if (existsSync(store)) {
    for (const id of readdirSync(store, { withFileTypes: true })) {
      if (!id.isDirectory()) continue;
      const container = join(store, id.name, 'node_modules');
      if (!existsSync(container)) continue;
      for (const e of readdirSync(container, { withFileTypes: true })) {
        if (e.name === '.bin') continue;
        const dirs = e.name.startsWith('@') ? readdirSync(join(container, e.name)).map(x => join(container, e.name, x)) : [join(container, e.name)];
        for (const d of dirs) {
          let real; try { real = realpathSync(d); } catch { continue; }
          if (!real.startsWith(store)) continue;       // a link to a workspace package outside the store is handled through the instance table
          const n = nameOf(real);
          if (n && !byName.has(n)) byName.set(n, real);
        }
      }
    }
  }

  const placed = new Map();   // folder (in the flat tree) -> real instance folder it holds
  let copied = 0, nested = 0;

  const visibleFrom = (chain, name) => {
    for (const dir of chain) {
      const slot = join(dir, 'node_modules', ...name.split('/'));
      if (placed.has(slot)) return { slot, instance: placed.get(slot) };
    }
    return null;
  };
  const copyInto = (slot, instance) => {
    mkdirSync(dirname(slot), { recursive: true });
    cpSync(instance, slot, { recursive: true, dereference: true, filter: f => !relative(instance, f).split(sep).includes('node_modules') });
    placed.set(slot, identity(instance));
    copied++;
  };

  const queue = [];   // { slotDir (folder of the placed package), chain (folders searched upwards), instance, name }
  const rootChain = [flatRoot];
  const place = (name, instance, chain, forceHere) => {
    if (skip?.test(name)) return;
    const seen = visibleFrom(chain, name);
    if (seen && seen.instance === identity(instance)) return;
    const hoist = !seen && !forceHere;
    const base = hoist ? flatRoot : chain[0];
    const slot = join(base, 'node_modules', ...name.split('/'));
    if (placed.has(slot)) return;      // another instance already sits where this one would go (it nests below, handled by the caller's chain)
    copyInto(slot, instance);
    if (!hoist && base !== flatRoot) nested++;
    queue.push({ slotDir: slot, chain: [slot, ...(hoist ? rootChain : chain)], instance, name });
  };

  // 1. the packages the root names
  const rootEntries = [];
  for (const e of readdirSync(nm, { withFileTypes: true })) {
    if (e.name === '.pnpm' || e.name === '.bin' || e.name.startsWith('.')) continue;
    const entries = e.name.startsWith('@') ? readdirSync(join(nm, e.name)).map(x => [`${e.name}/${x}`, join(nm, e.name, x)]) : [[e.name, join(nm, e.name)]];
    for (const [name, p] of entries) rootEntries.push([name, p]);
  }
  for (const [name, p] of rootEntries) {
    let real; try { real = realpathSync(p); } catch { continue; }
    place(name, real, rootChain, true);
  }
  // 2. their dependencies, breadth first
  while (queue.length) {
    const { slotDir, chain, instance, name } = queue.shift();
    let wants;
    if (instance.startsWith(store)) wants = storeDeps(instance, name);
    else {
      // a package copied in without a store entry: its declared requirements, taken from the store by name when not visible yet
      const m = manifest(instance) ?? {};
      wants = new Map();
      for (const dep of [...Object.keys(m.dependencies ?? {}), ...Object.keys(m.peerDependencies ?? {}), ...Object.keys(m.optionalDependencies ?? {})]) {
        if (visibleFrom(chain, dep)) continue;
        const real = byName.get(dep);
        if (real) wants.set(dep, real);
      }
    }
    for (const [dep, real] of wants) place(dep, real, chain, false);
  }

  rmSync(nm, { recursive: true, force: true });
  renameSync(join(flatRoot, 'node_modules'), nm);
  rmSync(flatRoot, { recursive: true, force: true });
  return { packages: placed.size, copied, nested };
}
