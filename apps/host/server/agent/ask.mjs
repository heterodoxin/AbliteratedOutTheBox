/**
 * An approval question in plain words: what the agent is about to do, as a sentence a person who never opens a terminal can answer.
 *
 * `askFor(name, args)` returns `{ id, params, why, detail }`:
 *   - `id` + `params` name one fixed sentence the page translates (src/lib/ask.ts holds the English templates and `{0}` slots),
 *   - `why` is an optional second fixed sentence id, `detail` a short plain line (a path) shown beneath it,
 *   - the exact command or arguments stay on the card under "Details" for experts.
 * Anything not recognised falls back to a generic question, never to the raw command as the headline.
 */

const KNOWN_FOLDERS = { documents: 'Documents', downloads: 'Downloads', desktop: 'Desktop', pictures: 'Pictures', music: 'Music', videos: 'Videos' };

const clip = (value, max = 120) => { const t = String(value ?? '').replace(/\s+/g, ' ').trim(); return t.length > max ? `${t.slice(0, max - 1)}…` : t; };
const unquote = text => String(text ?? '').trim().replace(/^(['"])(.*)\1$/, '$2');
const baseName = path => { const t = unquote(path).replace(/[\\/]+$/, ''); return t.split(/[\\/]/).pop() || t; };
const hasExtension = path => /\.[A-Za-z0-9]{1,8}$/.test(baseName(path));
const isExecutable = path => /\.(exe|msi|bat|cmd|ps1|com|scr|msix|appx|vbs|jar|sh)$/i.test(unquote(path));

/** Words of a command line, quotes kept together. */
function words(text) {
  const out = []; const re = /"([^"]*)"|'([^']*)'|(\S+)/g; let m;
  while ((m = re.exec(text))) out.push(m[1] ?? m[2] ?? m[3]);
  return out;
}

/** The first argument that is not a switch (-Name value pairs are skipped by the callers that need them). */
const firstPath = list => list.find(w => !/^-/.test(w)) ?? '';

function folderAsk(target) {
  const clean = unquote(target);
  const known = KNOWN_FOLDERS[baseName(clean).toLowerCase()];
  if (known) return { id: 'open-known-folder', params: [known], why: 'opens-window', detail: clip(clean, 160) };
  if (hasExtension(clean)) return { id: 'open-file', params: [baseName(clean)], why: 'opens-window', detail: clip(clean, 160) };
  return { id: 'open-folder', params: [baseName(clean) || clean], why: 'opens-window', detail: clip(clean, 160) };
}

function shellAsk(command) {
  const text = String(command ?? '').trim();
  // Several steps in one command: say so rather than guessing at one of them.
  const steps = text.split(/\s*(?:;|&&|\|\|)\s*|\r?\n/).filter(Boolean);
  const first = steps[0] ?? '';
  const w = words(first);
  const verb = (w[0] ?? '').toLowerCase().replace(/\.exe$/, '');
  const rest = w.slice(1);
  if (steps.length > 1) {
    const kinds = new Set(steps.map(s => (words(s)[0] ?? '').toLowerCase()));
    if (kinds.size > 1) return { id: 'run-steps', params: [steps.length], why: 'run-steps-why', detail: clip(first, 100) };
  }
  if (verb === 'start-process' || verb === 'start' || verb === 'saps') {
    const pathAt = rest.findIndex((x, i) => !/^-/.test(x) && !(i > 0 && /^-(argumentlist|args|workingdirectory|verb|wait)$/i.test(rest[i - 1])));
    const nameFlag = rest.findIndex(x => /^-(filepath|path)$/i.test(x));
    const target = nameFlag >= 0 ? rest[nameFlag + 1] : rest[pathAt] ?? '';
    const lower = baseName(target).toLowerCase();
    if (/^explorer(\.exe)?$/.test(lower)) {
      const argAt = rest.findIndex(x => /^-(argumentlist|args)$/i.test(x));
      const tail = argAt >= 0 ? rest.slice(argAt + 1) : rest.filter((x, i) => i !== (nameFlag >= 0 ? nameFlag + 1 : pathAt));
      const folder = tail.find(x => !/^-/.test(x) && !/^\/(select|e|root)/i.test(x));
      return folder ? folderAsk(folder) : { id: 'open-explorer', params: [], why: 'opens-window' };
    }
    if (isExecutable(target)) return { id: 'run-program', params: [baseName(target)], why: 'run-program-why', detail: clip(unquote(target), 160) };
    if (target) return folderAsk(target);
  }
  if (verb === 'explorer') return rest.length ? folderAsk(firstPath(rest.filter(x => !/^\/(select|e|root)/i.test(x))) || rest[0]) : { id: 'open-explorer', params: [], why: 'opens-window' };
  if (verb === 'invoke-item' || verb === 'ii' || verb === 'xdg-open' || verb === 'open') return folderAsk(firstPath(rest));
  if (verb === 'remove-item' || verb === 'rm' || verb === 'del' || verb === 'erase' || verb === 'rmdir' || verb === 'rd') {
    const target = firstPath(rest.filter(x => !/^-(recurse|force|r|rf|fr|f|path|literalpath)$/i.test(x)));
    return { id: 'delete', params: [baseName(target) || 'these files'], why: 'delete-why', detail: clip(unquote(target), 160) };
  }
  if (verb === 'new-item' || verb === 'mkdir' || verb === 'md' || verb === 'ni') {
    const pathAt = rest.findIndex(x => /^-path$/i.test(x));
    const target = pathAt >= 0 ? rest[pathAt + 1] : firstPath(rest);
    const asFile = rest.some((x, i) => /^-(itemtype|type)$/i.test(x) && /file/i.test(rest[i + 1] ?? '')) || (verb === 'new-item' && hasExtension(target));
    return asFile ? { id: 'create-file', params: [baseName(target)], why: 'replaces-if-exists', detail: clip(unquote(target), 160) } : { id: 'create-folder', params: [baseName(target) || target], detail: clip(unquote(target), 160) };
  }
  if (['copy-item', 'cp', 'copy'].includes(verb)) return { id: 'copy', params: [baseName(firstPath(rest))], detail: clip(rest.filter(x => !/^-/.test(x)).join('  →  '), 160) };
  if (['move-item', 'mv', 'move'].includes(verb)) return { id: 'move', params: [baseName(firstPath(rest))], detail: clip(rest.filter(x => !/^-/.test(x)).join('  →  '), 160) };
  if (['rename-item', 'ren', 'rni'].includes(verb)) return { id: 'rename', params: [baseName(firstPath(rest))], detail: clip(rest.filter(x => !/^-/.test(x)).join('  →  '), 160) };
  if (/(set-content|add-content|out-file|tee-object)/i.test(first) || />{1,2}\s*\S+/.test(first)) {
    const redirect = />{1,2}\s*("[^"]+"|'[^']+'|\S+)/.exec(first)?.[1];
    const all = words(first);
    const cmdAt = all.findIndex(x => /^(set-content|add-content|out-file|tee-object)$/i.test(x));
    const after = cmdAt >= 0 ? all.slice(cmdAt + 1) : rest;
    const pathAt = after.findIndex(x => /^-(path|filepath|literalpath)$/i.test(x));
    const target = unquote(redirect ?? (pathAt >= 0 ? after[pathAt + 1] : firstPath(after)));
    return { id: 'write-file', params: [baseName(target) || 'a file'], why: 'replaces-if-exists', detail: clip(target, 160) };
  }
  if (verb === 'git') {
    const sub = (rest[0] ?? '').toLowerCase();
    if (['commit', 'push', 'pull', 'merge', 'rebase', 'reset', 'checkout', 'clone', 'fetch', 'stash', 'switch', 'tag', 'branch', 'add', 'rm', 'mv', 'cherry-pick', 'revert'].includes(sub)) return { id: 'git', params: [sub], why: 'changes-repo' };
  }
  if (['npm', 'pnpm', 'yarn', 'pip', 'pip3', 'winget', 'choco', 'scoop', 'apt', 'apt-get', 'brew', 'cargo', 'dotnet', 'npx'].includes(verb)) {
    const sub = (rest[0] ?? '').toLowerCase();
    if (/^(install|i|add|update|upgrade|remove|uninstall|restore|publish)$/.test(sub)) return { id: 'install', params: [verb], why: 'install-why', detail: clip(rest.slice(1).filter(x => !/^-/.test(x)).join(' '), 120) };
  }
  if (isExecutable(w[0] ?? '') || /^[.&]/.test(first) && isExecutable(rest[0] ?? '')) return { id: 'run-program', params: [baseName(isExecutable(w[0]) ? w[0] : rest[0])], why: 'run-program-why' };
  if (['python', 'python3', 'py', 'node', 'pwsh', 'powershell', 'bash', 'sh', 'cmd'].includes(verb)) {
    const script = rest.find(x => /\.(py|js|mjs|ps1|sh|bat|cmd)$/i.test(x));
    return script ? { id: 'run-script', params: [baseName(script)], why: 'run-program-why' } : { id: 'run-command', params: [], why: 'run-command-why' };
  }
  return { id: 'run-command', params: [], why: 'run-command-why' };
}

/** @param {string} name tool name @param {object} args @param {string|null} [described] the model's own one-line description */
export function askFor(name, args = {}, described = null) {
  const path = args.file_path ?? args.path ?? args.filePath ?? '';
  switch (name) {
    case 'write': return { id: 'create-file', params: [baseName(path) || 'a file'], why: 'replaces-if-exists', detail: clip(path, 160) };
    case 'edit': return { id: 'change-file', params: [baseName(path) || 'a file'], detail: clip(path, 160) };
    case 'pwsh': case 'bash': {
      const ask = shellAsk(args.command);
      // A command nothing above could read: the model's own one-line description is better than "Run a command" alone.
      if (ask.id === 'run-command' && described) return { ...ask, detail: clip(described, 140) };
      return ask;
    }
    case 'job_kill': return { id: 'stop-task', params: [] };
    case 'web_fetch': {
      let host = String(args.url ?? '');
      try { host = new URL(host).hostname; } catch { /* keep the text */ }
      return { id: 'open-page', params: [clip(host, 80)], detail: clip(args.url, 160) };
    }
    default:
      if (String(name).startsWith('mcp__')) return { id: 'use-tool', params: [String(name).split('__').slice(1).join(' / ')], why: 'use-tool-why' };
      return { id: 'use-other', params: [String(name)] };
  }
}
