/**
 * What the person chose in Settings -> Tools, and the single place that validates it.
 *
 * Saved (inside the host's settings file): which tool groups are off, which web search service to use (and its key, which never leaves
 * this file: the screen only learns whether one is saved), whether the browser tool is on, and the person's own tools.
 *
 * The profile builder reads `profileView()`; the screen reads `view()`. Custom command and instruction tools are served by the app's
 * own small tool server (custom-tools-mcp.mjs) from a JSON file this module writes, so they appear to the assistant as ordinary tools:
 * `mcp__strata-instructions__<name>` (look-only: it just returns the person's instructions) and `mcp__strata-commands__<name>`
 * (runs the person's command; asks first in Manual and is refused in Plan). A person's own tool server (a web address or a program)
 * is added as `mcp__custom-<name>__...` and asks first in Manual like any connected tool.
 */
import { mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { GROUP_IDS, TOOL_GROUPS } from './tools-catalog.mjs';
import { SHELLS, clampTimeoutSec } from './custom-command-run.mjs';

export const SEARCH_PROVIDERS = ['builtin', 'exa', 'perplexity', 'deepseek'];
const KINDS = ['instructions', 'command', 'server'];
const NAME = /^[A-Za-z][A-Za-z0-9 _-]{0,39}$/;
const MAX_CUSTOM = 40;

export const INSTRUCTIONS_SERVER = 'strata-instructions';
export const COMMANDS_SERVER = 'strata-commands';

const slug = value => String(value).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40) || 'tool';
const clean = (value, max) => String(value ?? '').replace(/\0/g, '').slice(0, max);

function normalize(raw) {
  const r = raw && typeof raw === 'object' ? raw : {};
  const search = r.search && typeof r.search === 'object' ? r.search : {};
  return {
    off: [...new Set((Array.isArray(r.off) ? r.off : []).filter(id => GROUP_IDS.includes(id)))],
    search: { provider: SEARCH_PROVIDERS.includes(search.provider) ? search.provider : 'builtin', key: typeof search.key === 'string' ? search.key : '' },
    browser: { enabled: r.browser?.enabled === true },
    // Manual mode asks once per chat before the first web search or page. This switch (off unless the person turned it on) skips that question.
    webWithoutAsking: r.webWithoutAsking === true,
    custom: (Array.isArray(r.custom) ? r.custom : []).filter(t => t && NAME.test(t.name ?? '') && KINDS.includes(t.kind)).slice(0, MAX_CUSTOM).map(t => ({
      id: String(t.id), name: String(t.name), about: clean(t.about, 300), kind: t.kind, enabled: t.enabled !== false,
      instructions: clean(t.instructions, 20000), command: clean(t.command, 2000), address: clean(t.address, 2000),
      args: (Array.isArray(t.args) ? t.args : []).slice(0, 50).map(a => clean(a, 1000)),
      // a command tool's folder, shell and time allowed (custom-command-run.mjs); '' / 'auto' / 120 when not set
      cwd: clean(t.cwd, 1000).trim(), shell: SHELLS.includes(t.shell) ? t.shell : 'auto', timeoutSec: clampTimeoutSec(t.timeoutSec),
    })),
  };
}

function validateCustom(body, existing) {
  const name = String(body?.name ?? '').trim();
  if (!NAME.test(name)) throw new Error('Give the tool a short name: letters, numbers, spaces or hyphens.');
  const about = clean(body?.about, 300).trim();
  if (!about) throw new Error('Say what the tool does, in a sentence. The assistant reads this to decide when to use it.');
  const kind = body?.kind;
  if (!KINDS.includes(kind)) throw new Error('Choose how the tool works.');
  const out = { name, about, kind, instructions: '', command: '', address: '', args: [], cwd: '', shell: 'auto', timeoutSec: clampTimeoutSec(undefined) };
  if (kind === 'instructions') {
    out.instructions = clean(body.instructions, 20000).trim();
    if (!out.instructions) throw new Error('Write the instructions the assistant should follow when it uses this tool.');
  } else if (kind === 'command') {
    out.command = clean(body.command, 2000).trim();
    if (!out.command) throw new Error('Enter the command to run.');
    if (/[\r\n]/.test(out.command)) throw new Error('Put the command on one line.');
    // The edit form of Settings does not send these three: an edit keeps what the tool had.
    const before = existing.find(t => t.id === body.id && t.kind === 'command');
    out.cwd = body.cwd !== undefined ? clean(body.cwd, 1000).trim() : before?.cwd ?? '';
    if (/[\r\n]/.test(out.cwd)) throw new Error('Put the folder on one line.');
    if (body.shell !== undefined && !SHELLS.includes(body.shell)) throw new Error('Choose how the command runs: automatic, cmd, PowerShell, bash or sh.');
    out.shell = body.shell !== undefined ? body.shell : before?.shell ?? 'auto';
    if (body.timeoutSec !== undefined && !(Number(body.timeoutSec) >= 1 && Number(body.timeoutSec) <= 3600)) throw new Error('Give the time allowed in seconds, from 1 to 3600.');
    out.timeoutSec = body.timeoutSec !== undefined ? clampTimeoutSec(body.timeoutSec) : before?.timeoutSec ?? clampTimeoutSec(undefined);
  } else {
    out.address = clean(body.address, 2000).trim();
    if (!out.address) throw new Error('Enter the tool server’s web address, or the program that starts it.');
    if (/[\r\n]/.test(out.address)) throw new Error('Put the address or program on one line.');
    if (/^https?:\/\//i.test(out.address)) {
      let url; try { url = new URL(out.address); } catch { throw new Error('That web address is not valid.'); }
      if (url.username || url.password) throw new Error('Use a web address without a user name or password in it.');
    }
    out.args = (Array.isArray(body.args) ? body.args : []).slice(0, 50).map(a => clean(a, 1000)).filter(Boolean);
  }
  if (existing.some(t => t.name.toLowerCase() === name.toLowerCase() && t.id !== body.id)) throw new Error('You already have a tool with that name.');
  return out;
}

const idFor = (name, existing) => {
  const base = slug(name);
  let id = base, n = 2;
  while (existing.some(t => t.id === id)) id = `${base}_${n++}`;
  return id;
};

/**
 * @param {{read: () => object|undefined, write: (value: object) => void, dataDir: string, browser?: () => {found: boolean, name: string|null, path?: string|null, cli?: string|null}}} deps
 */
export function createAgentTools({ read, write, dataDir, browser = () => ({ found: false, name: null }) }) {
  const state = () => normalize(read());
  const filePath = join(dataDir, 'agent', 'custom-tools.json');

  /** The file the app's own tool server reads: instruction tools and command tools that are on. */
  function writeToolFile(s) {
    const instructions = {}, commands = {};
    for (const t of s.custom.filter(t => t.enabled)) {
      const key = t.id;
      if (t.kind === 'instructions') instructions[key] = { description: t.about, text: t.instructions };
      else if (t.kind === 'command') commands[key] = { description: t.about, command: t.command, cwd: t.cwd, shell: t.shell, timeoutSec: t.timeoutSec };
    }
    mkdirSync(join(dataDir, 'agent'), { recursive: true });
    const tmp = `${filePath}.tmp`;
    writeFileSync(tmp, JSON.stringify({ instructions, commands }, null, 1));
    renameSync(tmp, filePath);
    return { instructions: Object.keys(instructions).length, commands: Object.keys(commands).length };
  }

  const save = s => { write(s); writeToolFile(s); };

  function view() {
    const s = state();
    const probe = browser();
    return {
      groups: TOOL_GROUPS.map(g => ({ id: g.id, name: g.name, about: g.about, area: g.area, asks: g.asks, enabled: !s.off.includes(g.id) })),
      search: { provider: s.search.provider, keySet: !!s.search.key },
      webWithoutAsking: s.webWithoutAsking,
      // `found` means the browser tool can run: a browser on this computer AND the tool server in the agent component.
      browser: { enabled: s.browser.enabled, found: !!(probe.found && probe.cli), name: probe.name ?? null, ...(probe.found && !probe.cli ? { missing: 'component' } : {}) },
      custom: s.custom.map(t => ({
        id: t.id, name: t.name, about: t.about, kind: t.kind, enabled: t.enabled,
        summary: t.kind === 'instructions' ? t.instructions.slice(0, 80) : t.kind === 'command' ? t.command : t.address,
        instructions: t.instructions, command: t.command, address: t.address, args: t.args,
        ...(t.kind === 'command' ? { cwd: t.cwd, shell: t.shell, timeoutSec: t.timeoutSec } : {}),
      })),
    };
  }

  return {
    view,
    filePath,
    /** For the profile builder: plain values, secrets included. Never sent to the screen. */
    profileView() {
      const s = state();
      const custom = s.custom.filter(t => t.enabled);
      const servers = {};
      for (const t of custom.filter(t => t.kind === 'server')) {
        const name = `custom-${t.id.replace(/_/g, "-")}`.slice(0, 32);
        servers[name] = /^https?:\/\//i.test(t.address) ? { url: t.address } : { command: t.address, args: t.args };
      }
      return {
        off: new Set(s.off),
        search: s.search,
        webWithoutAsking: s.webWithoutAsking,
        browserEnabled: s.browser.enabled && !!(browser().found && browser().cli),
        browserPath: browser().path ?? null,
        browserCli: browser().cli ?? null,
        servers,
        hasInstructionTools: custom.some(t => t.kind === 'instructions'),
        hasCommandTools: custom.some(t => t.kind === 'command'),
        toolFile: filePath,
      };
    },
    update(change) {
      const s = state();
      if (change?.groups) {
        for (const [id, on] of Object.entries(change.groups)) {
          if (!GROUP_IDS.includes(id)) throw new Error('That tool is not known.');
          s.off = on === false ? [...new Set([...s.off, id])] : s.off.filter(x => x !== id);
        }
      }
      if (change?.search) {
        const provider = change.search.provider;
        if (provider !== undefined) {
          if (!SEARCH_PROVIDERS.includes(provider)) throw new Error('Choose one of the listed search services.');
          if (provider !== s.search.provider) s.search = { provider, key: '' };   // a key belongs to its service
        }
        if (change.search.key !== undefined) {
          const key = String(change.search.key).trim();
          if (!key || key.length > 500 || /[\s\0]/.test(key)) throw new Error('That key does not look right. Paste it again without spaces.');
          s.search = { ...s.search, key };
        }
      }
      if (typeof change?.webWithoutAsking === 'boolean') s.webWithoutAsking = change.webWithoutAsking;
      if (change?.browser && typeof change.browser.enabled === 'boolean') s.browser = { enabled: change.browser.enabled };
      if (change?.custom) {
        for (const [id, on] of Object.entries(change.custom)) {
          const t = s.custom.find(x => x.id === id);
          if (!t) throw new Error('That tool is not in your list.');
          t.enabled = on !== false;
        }
      }
      save(s);
      return view();
    },
    addCustom(body) {
      const s = state();
      if (s.custom.length >= MAX_CUSTOM) throw new Error(`You can have up to ${MAX_CUSTOM} tools of your own.`);
      const t = validateCustom(body, s.custom);
      s.custom.push({ id: idFor(t.name, s.custom), enabled: true, ...t });
      save(s);
      return view();
    },
    editCustom(id, body) {
      const s = state();
      const at = s.custom.findIndex(t => t.id === id);
      if (at < 0) throw new Error('That tool is not in your list.');
      const t = validateCustom({ ...body, id }, s.custom);
      s.custom[at] = { ...s.custom[at], ...t };
      save(s);
      return view();
    },
    removeCustom(id) {
      const s = state();
      if (!s.custom.some(t => t.id === id)) throw new Error('That tool is not in your list.');
      s.custom = s.custom.filter(t => t.id !== id);
      save(s);
      return view();
    },
    /** Make sure the tool server's file exists (host start). */
    ensureFile() { writeToolFile(state()); },
  };
}
