/**
 * The Strata approval gate: a small Harness plugin (plain ESM, no imports, loaded by absolute path from the profile).
 *
 * Why it exists. The Harness only asks a person when a tool escalates its sandbox. With the sandbox read-only that
 * produced a blocked attempt first (a card full of sandbox and PowerShell errors) and the real Approve/Deny card
 * only on the model's second try. In Manual mode this plugin asks BEFORE a change runs, through the Harness's own
 * approval service, so the person sees exactly one card per change and nothing has run when it appears.
 *
 * How it works. It listens to `tools/pre-execute` (the same seam the Harness's own Auto-review layer uses). For a
 * session whose effective preset is Manual it returns an `ask` decision for every call that is not known to be
 * read-only. The Harness then asks, runs the tool only after an approval, and records the answer. Manual runs with
 * the sandbox open, so an approved change can really happen anywhere the person agreed to; the approval is the
 * control, which is why anything this file cannot prove is read-only asks (fail closed).
 *
 * Children. A subagent's approval policy is pinned to `never` by the Harness (its requests are rejected on the
 * spot). For a child of a Manual chat that would hide the card and the child could not do anything. The gate
 * writes `ask` into that child's log, so the child's request reaches the same single Approve/Deny card.
 * Plan and Bypass children are untouched.
 *
 * The status command `/strata-gate` lets the host confirm the gate is really loaded before it lets a Manual turn
 * start (a Manual chat with no gate would be a chat with the sandbox open and nobody asking).
 */

export const name = 'strata-gate';
export const inject = ['approval', 'permissionPresets', 'sessions', 'tools'];

/** Preset key the host writes for Manual (apps/host/server/agent/profile.mjs MODE_PRESETS). */
export const MANUAL_PRESET = 'strata-manual';
/** Preset key the host writes for Plan (read-only, nothing may change). */
export const PLAN_PRESET = 'strata-plan';
/** The app's own look-only tool server (the person's instruction tools): it only hands back text. */
const LOOK_ONLY_SERVER = 'mcp__strata-instructions__';
/** The tools a person wrote as commands (Settings -> Tools): the card says which one runs and with what text. */
const COMMANDS_SERVER = 'mcp__strata-commands__';
/** The browser tool server's name in the agent (Settings -> Tools). Looking at the page does not ask; acting on it does. */
const BROWSER_SERVER = 'mcp__browser__';
const BROWSER_LOOKING = new Set(['browser_snapshot', 'browser_take_screenshot', 'browser_console_messages', 'browser_network_requests', 'browser_wait_for']);

/** Tools that cannot change anything on this computer. Everything else asks. */
export const READ_ONLY_TOOLS = new Set([
  'read', 'glob', 'grep', 'read_image', 'working_directory', 'todo_write', 'skill', 'present', 'lsp',
  'get_goal', 'create_goal', 'update_goal', 'list_agents', 'list_subagent_models', 'job_list', 'job_output',
  'ask_user_question', 'exit_plan_mode',
  // (web_search and web_fetch are not here: the first use in a chat asks, see webDecision.)
  // Delegation: the child it starts is governed by this same gate.
  'subagent', 'subagent_fork', 'send_message', 'interrupt_agent', 'workflow',
]);

const SHELL_TOOLS = new Set(['pwsh', 'bash']);

/** Commands that only look. Matched against the first word of every pipeline segment. */
const SAFE_COMMANDS = new Set([
  // PowerShell
  'get-childitem', 'gci', 'ls', 'dir', 'get-content', 'gc', 'cat', 'type', 'get-item', 'gi', 'get-location', 'pwd', 'gl',
  'select-string', 'sls', 'measure-object', 'measure', 'select-object', 'select', 'where-object', 'where', 'sort-object', 'sort',
  'format-table', 'ft', 'format-list', 'fl', 'out-string', 'test-path', 'resolve-path', 'get-filehash', 'get-date', 'get-command',
  'get-process', 'get-itemproperty', 'write-output', 'echo', 'hostname', 'whoami', 'get-host', 'get-psdrive', 'get-help',
  'group-object', 'group', 'convertto-json', 'convertfrom-json', 'select-xml', 'compare-object', 'get-unique', 'tee-object-none',
  // POSIX
  'head', 'tail', 'wc', 'grep', 'rg', 'egrep', 'fgrep', 'stat', 'file', 'du', 'df', 'uname', 'which', 'basename', 'dirname',
  'realpath', 'readlink', 'sort', 'uniq', 'cut', 'tr', 'less', 'more', 'date', 'id', 'env-none', 'tree',
]);

/** git subcommands that only read. */
const SAFE_GIT = new Set(['status', 'log', 'diff', 'show', 'branch', 'rev-parse', 'ls-files', 'blame', 'describe', 'remote', 'tag', 'shortlog', 'rev-list', 'cat-file', 'grep', 'ls-tree', 'config']);
const SAFE_VERSION = /^(node|python|python3|py|npm|pnpm|git|dotnet|cargo|rustc|go|java|pip|pip3)\s+(--version|-v|-version|version)$/i;

/** Anything that can chain, redirect, expand or start something else is not read-only. */
const UNSAFE_SYNTAX = /[;&<>`${}()]|\r|\n|--%|\bout-file\b|\bset-content\b|\badd-content\b|\binvoke-|\biex\b|\bstart-|\bnew-|\bremove-|\bcopy-|\bmove-|\brename-|\bclear-|\bset-|\bpush-|\bexpand-|\bsave-|\binstall-|\bcurl\b|\bwget\b|\bsudo\b|\bsh\b|\bbash\b|\bpwsh\b|\bpowershell\b|\bcmd\b|-exec\b|-delete\b|-execdir\b/i;

/**
 * The command with every quoted text replaced by a plain stand-in, or null when it cannot be read safely. What is inside quotes is data
 * (a path like "C:\Users\x\Q&A", a pattern like 'a|b'), so an & ; | ( ) < > or { in it must not decide anything. Two things still can:
 * inside double quotes a $ or a backtick expands (a variable, a sub-command), and an unterminated quote means the text was not understood.
 * Either one gives null, and the command asks as before.
 * @param {string} text
 * @returns {string|null}
 */
export function maskQuotes(text) {
  let out = '';
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c !== '"' && c !== "'") { out += c; continue; }
    let end = text.indexOf(c, i + 1);
    // PowerShell writes a quote inside a quoted text by doubling it ('it''s'): the text goes on past the pair.
    while (end >= 0 && text[end + 1] === c) end = text.indexOf(c, end + 2);
    if (end < 0) return null;
    if (c === '"' && /[$`]/.test(text.slice(i + 1, end))) return null;
    // A quoted text is a whole word. One glued to other characters (--out"put"=f, "a"b) or one that starts with a dash could hide an option from the
    // word checks below (git diff "--output=f" writes a file), so those are not understood either.
    const before = text[i - 1], after = text[end + 1];
    if ((before !== undefined && !/[\s|]/.test(before)) || (after !== undefined && !/[\s|]/.test(after)) || text[i + 1] === '-') return null;
    out += '"_"';
    i = end;
  }
  return out;
}

/**
 * Whether a shell command only reads. Conservative by design: one unrecognised word and the answer is no.
 * Quoted text is data and is looked at after it has been set aside (maskQuotes): Get-ChildItem "C:\Users\x\Q&A" only lists.
 * @param {string} command
 * @returns {boolean}
 */
export function isReadOnlyShell(command) {
  const text = String(command ?? '').trim();
  if (!text || text.length > 600) return false;
  if (SAFE_VERSION.test(text)) return true;
  const masked = maskQuotes(text);
  if (masked === null || UNSAFE_SYNTAX.test(masked)) return false;
  const segments = masked.split('|').map(s => s.trim());
  return segments.every((segment) => {
    if (!segment) return false;
    const words = segment.split(/\s+/);
    const first = words[0].toLowerCase().replace(/^\.\\/, '');
    if (first === 'git') {
      // `git config` and `git tag`/`branch`/`remote` can write when given a value; only the bare listing forms are safe.
      const sub = (words[1] ?? '').toLowerCase();
      if (!SAFE_GIT.has(sub)) return false;
      if (['config', 'tag', 'branch', 'remote'].includes(sub)) {
        const rest = words.slice(2).filter(w => !w.startsWith('-') || /^(-l|-a|-v|-vv|--list|--all|--show-current|--get|--get-all)$/i.test(w));
        return rest.length === 0 || (sub === 'config' && words[2] === '--get') || (sub === 'remote' && words[2] === '-v');
      }
      return !words.some(w => /^--(output|ext-diff|textconv)\b/i.test(w));
    }
    return SAFE_COMMANDS.has(first) && !words.some(w => /^-(delete|exec|execdir|fprint|fls|ok)$/i.test(w));
  });
}

const clip = (value, max) => {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

/** The tools that reach the internet. In Manual the first one in a chat asks once; the answer holds for the rest of that chat. */
export const WEB_TOOLS = new Set(['web_search', 'web_fetch']);
/** What the model reads when the person says no. Plain words, and it tells the model what to do next. */
export const WEB_REFUSAL = 'The person did not allow web access, so nothing was searched or opened. Go on without it and say so.';

/**
 * The question for a web call, in plain words: what is about to leave this computer and where it goes.
 * @param {{name:string, arguments:any}} exec
 * @param {{provider?: string}} [options] provider is the search service's name as the person knows it
 */
export function webDecision(exec, options = {}) {
  const tool = String(exec.name ?? '');
  const args = exec.arguments && typeof exec.arguments === 'object' ? exec.arguments : {};
  if (tool === 'web_search') {
    const queries = Array.isArray(args.queries) ? args.queries : [args.query ?? args.q];
    const words = clip(queries.filter(q => typeof q === 'string' && q.trim()).join('; '), 160);
    return { kind: 'ask', web: true, reason: `The assistant wants to search the web for: ${words}. Search words go to ${options.provider || 'the search service'}.` };
  }
  return { kind: 'ask', web: true, reason: `The assistant wants to open the web page ${clip(args.url, 200)}. The page's site sees the request.` };
}

/**
 * The decision for one call in Manual mode, or null when it needs no question.
 * @param {{name:string, arguments:any}} exec
 * @param {{provider?: string}} [options]
 * @returns {{kind:'ask', reason:string, web?:true}|null}
 */
export function manualDecision(exec, options = {}) {
  const tool = String(exec.name ?? '');
  const args = exec.arguments && typeof exec.arguments === 'object' ? exec.arguments : {};
  if (WEB_TOOLS.has(tool)) return webDecision(exec, options);
  if (READ_ONLY_TOOLS.has(tool)) return null;
  if (SHELL_TOOLS.has(tool)) {
    if (isReadOnlyShell(args.command)) return null;
    const what = args.description ? clip(args.description, 160) : clip(args.command, 160);
    return { kind: 'ask', reason: `Run a command that may change things: ${what}` };
  }
  if (tool === 'write') return { kind: 'ask', reason: `Create or replace the file ${clip(args.file_path ?? args.path, 200)}` };
  if (tool === 'edit') return { kind: 'ask', reason: `Change the file ${clip(args.file_path ?? args.path, 200)}` };
  if (tool === 'job_kill') return { kind: 'ask', reason: 'Stop a background task' };
  if (tool.startsWith(LOOK_ONLY_SERVER)) return null;
  if (tool.startsWith(BROWSER_SERVER)) return browserDecision(tool.slice(BROWSER_SERVER.length), args);
  if (tool.startsWith(COMMANDS_SERVER)) return { kind: 'ask', reason: `Run your tool “${tool.slice(COMMANDS_SERVER.length).replace(/_/g, ' ')}”${args.input ? ` with: ${clip(args.input, 120)}` : ''}` };
  if (tool.startsWith('mcp__')) return { kind: 'ask', reason: `Use the connected tool ${tool.split('__').slice(1).join(' / ')}` };
  return { kind: 'ask', reason: `Use the tool ${tool}` };
}

/** The browser in plain words: opening a page, clicking or typing asks; looking at what is already open does not. */
function browserDecision(name, args) {
  if (BROWSER_LOOKING.has(name) || (name === 'browser_tabs' && args.action === 'list')) return null;
  const url = clip(args.url, 200);
  if (name === 'browser_navigate') return { kind: 'ask', reason: `Open the web page ${url}` };
  if (name === 'browser_click' || name === 'browser_hover' || name === 'browser_drag') return { kind: 'ask', reason: `Click on the page${args.element ? `: ${clip(args.element, 120)}` : ''}` };
  if (name === 'browser_type' || name === 'browser_fill_form' || name === 'browser_press_key' || name === 'browser_select_option') return { kind: 'ask', reason: `Type into the page${args.element ? `: ${clip(args.element, 120)}` : ''}` };
  if (name === 'browser_file_upload') return { kind: 'ask', reason: 'Send a file from this computer to the page' };
  if (name === 'browser_evaluate' || name === 'browser_run_code') return { kind: 'ask', reason: 'Run a small program inside the web page' };
  return { kind: 'ask', reason: `Use the browser (${name.replace(/^browser_/, '').replace(/_/g, ' ')})` };
}

/**
 * Plan mode changes nothing. A connected tool (a person's tool server, a command they wrote, the browser) is not under the sandbox, so
 * Plan refuses it instead of letting it run. The look-only instruction tools are fine.
 * @returns {{kind:'deny', reason:string}|null}
 */
export function planDecision(exec) {
  const tool = String(exec.name ?? '');
  if (!tool.startsWith('mcp__') || tool.startsWith(LOOK_ONLY_SERVER)) return null;
  return { kind: 'deny', reason: 'Plan mode only looks and changes nothing, so it cannot use this tool. Switch to Manual or Bypass to use it.' };
}

/**
 * @param {object} ctx
 * @param {{web?: {provider?: string, skipAsk?: boolean}}} [config] `web.provider`: the search service's name for the question;
 *        `web.skipAsk`: the person's switch in Settings -> Tools, "search the web without asking" (off unless they turned it on)
 */
export function apply(ctx, config = {}) {
  const web = config.web ?? {};
  /** Chats (their first session) where the person said yes to the web. One yes holds for the rest of the chat. */
  const webAllowed = new Set();
  /** A question already on the screen for a chat, so two searches started together share one card. */
  const webAsking = new Map();
  const chatOf = (session, depth = 0) => {
    const parentId = session.header?.parentSession;
    if (session.header?.origin === 'subagent' && parentId && depth < 6) {
      const parent = ctx.sessions.get(parentId);
      if (parent) return chatOf(parent, depth + 1);
    }
    return session.id;
  };
  async function askWeb(exec, gate, agent) {
    const chat = chatOf(agent.session);
    let question = webAsking.get(chat);
    if (!question) {
      question = ctx.approval.request({ agent, toolName: exec.name, callId: exec.callId, reason: gate.reason, displayReason: { en: gate.reason }, signal: exec.signal });
      webAsking.set(chat, question);
      const forget = () => { if (webAsking.get(chat) === question) webAsking.delete(chat); };
      question.then(forget, forget);
    }
    const outcome = await question;
    if (outcome === 'allowed-once') { webAllowed.add(chat); return null; }
    if (outcome === 'cancelled') return { kind: 'cancel' };
    return { kind: 'deny', reason: WEB_REFUSAL };
  }

  /** The preset a session is really working under: a child follows the chat that started it. */
  const effectivePreset = (session, depth = 0) => {
    const parentId = session.header?.parentSession;
    if (session.header?.origin === 'subagent' && parentId && depth < 6) {
      const parent = ctx.sessions.get(parentId);
      if (parent) return effectivePreset(parent, depth + 1);
    }
    return ctx.permissionPresets.current(session);
  };

  ctx.on('tools/pre-execute', async (exec, next) => {
    const agent = exec.agent;
    if (!agent || exec.name === 'run_code') return next();
    // Fail closed (T4): a session whose preset cannot be read (a reset in progress, a helper whose chat is gone) is treated as Manual, so
    // anything this file cannot prove is read-only still asks. It used to run unasked.
    let preset;
    try { preset = effectivePreset(agent.session); } catch (e) {
      preset = MANUAL_PRESET;
      try { console.warn(`[strata-gate] could not read the permission mode for this call (${e?.message ?? e}); treating it as Manual`); } catch { /* logging must never decide a permission */ }
    }
    // A helper that was set to ask while the chat was in Manual must stop asking the moment the chat leaves Manual,
    // so a mode change applies to the very next tool call of a running goal and its helpers.
    if (preset !== MANUAL_PRESET && agent.session.header?.origin === 'subagent' && ctx.approval.overrideOf(agent.session) === 'ask') {
      agent.session.append('approval/policy', { policy: 'never' });
    }
    if (preset === PLAN_PRESET) return planDecision(exec) ?? next();
    if (preset !== MANUAL_PRESET) return next();
    const gate = manualDecision(exec, { provider: web.provider });
    if (!gate) return next();
    if (gate.web) {
      // The person's switch, or a yes already given in this chat: no question.
      if (web.skipAsk === true || webAllowed.has(chatOf(agent.session))) return next();
      // A helper's own approval policy is `never`; it must be `ask` for the question to reach the person.
      if (agent.session.header?.origin === 'subagent' && ctx.approval.overrideOf(agent.session) !== 'ask') {
        agent.session.append('approval/policy', { policy: 'ask' });
      }
      return (await askWeb(exec, gate, agent)) ?? next();
    }
    // A child's approval policy is pinned to `never` by the Harness; asking is how its request reaches the person.
    if (agent.session.header?.origin === 'subagent' && ctx.approval.overrideOf(agent.session) !== 'ask') {
      agent.session.append('approval/policy', { policy: 'ask' });
    }
    const downstream = await next();
    if (downstream.kind !== 'allow') return downstream;
    return { kind: 'ask', reason: gate.reason, displayReason: { en: gate.reason } };
  }, { prepend: true });

  // Lets the host prove the gate is live before it starts a Manual turn.
  ctx.inject(['commands'], (commandCtx) => {
    commandCtx.commands.register({
      definitionId: 'strata-gate',
      name: 'strata-gate',
      description: 'Reports that the Strata approval gate is active',
      handler: () => ({ kind: 'success', text: 'strata-gate active' }),
    });
  });
}
