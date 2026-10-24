/**
 * Pure projections from Harness facts to the frames the chat UI already renders.
 *
 * No I/O here, so the mapping can be tested without an agent. The inputs are the two things a
 * `session/follow` stream carries: durable session events (`tool/call`, `approval/asked`,
 * `tool/result`, `turn/end`, ...) and the process-local `assistant-stream` frames (text and
 * reasoning deltas, usage).
 */

import { toolUpdate, TokenCount, fmtTokens } from './status.mjs';
import { askFor } from './ask.mjs';
import { FIT_TEXT, fill } from './context-fit.mjs';
import { isContextOverflow } from './context-budget.mjs';

const SHELL = new Set(['pwsh', 'bash', 'job_kill', 'job_output', 'job_list']);
const WRITE = new Set(['write', 'edit']);
const READ = new Set(['read', 'glob', 'grep', 'read_image', 'working_directory', 'todo_write', 'get_goal', 'list_agents', 'skill', 'present']);

/** Delegation tools have no card: the helper agents show as rows, and the tool's raw reply ("started subagent <id>") is not for the person. */
export const NO_CARD = new Set(['subagent', 'subagent_fork', 'send_message', 'interrupt_agent', 'list_agents']);

export function toolScope(name) {
  if (SHELL.has(name)) return 'shell';
  if (WRITE.has(name)) return 'write';
  if (READ.has(name)) return 'read';
  return 'connected';
}

/** Shown on a card the mode refused (Plan, Bypass): the model's own error text would say "the user rejected". */
export const NOT_IN_MODE = 'Not allowed in this mode. Nothing ran.';

const clip = (value, max) => {
  const text = String(value ?? '');
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

export function parseArguments(raw) {
  if (raw && typeof raw === 'object') return raw;
  try { const v = JSON.parse(String(raw ?? '{}')); return v && typeof v === 'object' ? v : {}; } catch { return {}; }
}

/**
 * One plain sentence and one short detail per tool, so a card never shows raw JSON or an internal tool name as its
 * explanation. Unknown tools (a connected MCP server's) fall back to a generic sentence with their arguments.
 */
const WORDS = {
  read: a => ['Reads a file.', a.file_path ?? a.path],
  glob: a => ['Looks for files by name.', a.pattern],
  grep: a => ['Searches inside files.', a.pattern ?? a.query],
  read_image: a => ['Looks at a picture.', a.file_path ?? a.path],
  working_directory: () => ['Checks which folder it is working in.', ''],
  todo_write: a => ['Updates its to-do list.', Array.isArray(a.todos) ? a.todos.map(t => t?.content ?? t?.title ?? '').filter(Boolean).join('; ') : ''],
  skill: a => ['Uses a saved skill.', a.name ?? a.skill],
  present: a => ['Shows you a result.', a.file_path ?? a.path ?? a.title],
  create_goal: a => ['Sets a goal to work towards.', a.objective],
  update_goal: a => ['Updates the goal.', a.objective ?? a.status ?? ''],
  get_goal: () => ['Checks the current goal.', ''],
  web_fetch: a => ['Opens a web page.', a.url],
  web_search: a => ['Searches the web.', Array.isArray(a.queries) ? a.queries.join('; ') : a.query],
  workflow: a => ['Runs a multi-step plan.', a.description ?? a.name ?? ''],
  job_list: () => ['Checks the background tasks.', ''],
  job_output: a => ['Reads what a background task printed.', a.job_id ?? a.id],
  job_kill: a => ['Stops a background task.', a.job_id ?? a.id],
  lsp: a => ['Looks up how the code fits together.', a.file_path],
};

/**
 * What a card says it is doing, in the person's words: [while it runs, when it worked, when it failed]. The UI translates
 * these fixed sentences and writes the target (a file path, the command) beside them, so a row never shows `pwsh · Ran`.
 */
const VERBS = {
  write: ['Writing a file', 'Wrote a file', "Couldn't write the file"],
  edit: ['Changing a file', 'Changed a file', "Couldn't change the file"],
  read: ['Reading a file', 'Read a file', "Couldn't read the file"],
  read_image: ['Looking at a picture', 'Looked at a picture', "Couldn't look at the picture"],
  glob: ['Searching for files', 'Searched for files', "Couldn't search for files"],
  grep: ['Searching inside files', 'Searched inside files', "Couldn't search inside files"],
  pwsh: ['Running a command', 'Ran a command', 'The command failed'],
  bash: ['Running a command', 'Ran a command', 'The command failed'],
  web_fetch: ['Opening a web page', 'Opened a web page', "Couldn't open the web page"],
  web_search: ['Searching the web', 'Searched the web', "Couldn't search the web"],
  create_goal: ['Setting the goal', 'Set the goal', "Couldn't set the goal"],
  update_goal: ['Updating the goal', 'Updated the goal', "Couldn't update the goal"],
  get_goal: ['Checking the goal', 'Checked the goal', "Couldn't check the goal"],
  skill: ['Loading your skill', 'Loaded your skill', "Couldn't load the skill"],
  todo_write: ['Updating its to-do list', 'Updated its to-do list', "Couldn't update its to-do list"],
  present: ['Showing you a result', 'Showed you a result', "Couldn't show the result"],
  working_directory: ['Checking the folder', 'Checked the folder', "Couldn't check the folder"],
  job_list: ['Checking background tasks', 'Checked background tasks', "Couldn't check background tasks"],
  job_output: ['Reading a background task', 'Read a background task', "Couldn't read the background task"],
  job_kill: ['Stopping a background task', 'Stopped a background task', "Couldn't stop the background task"],
};
const GENERIC_VERBS = ['Using a tool', 'Used a tool', 'The tool failed'];
/** The browser tool server's tools (Settings -> Tools): the card says what happened to the page, not the tool's internal name. */
const BROWSER_VERBS = {
  browser_navigate: ['Opening a page in the browser', 'Opened a page in the browser', "Couldn't open the page"],
  browser_click: ['Clicking on the page', 'Clicked on the page', "Couldn't click on the page"],
  browser_hover: ['Pointing at the page', 'Pointed at the page', "Couldn't point at the page"],
  browser_type: ['Typing on the page', 'Typed on the page', "Couldn't type on the page"],
  browser_fill_form: ['Typing on the page', 'Typed on the page', "Couldn't type on the page"],
  browser_press_key: ['Typing on the page', 'Typed on the page', "Couldn't type on the page"],
  browser_snapshot: ['Looking at the page', 'Looked at the page', "Couldn't look at the page"],
  browser_take_screenshot: ['Looking at the page', 'Looked at the page', "Couldn't look at the page"],
  browser_evaluate: ['Running a small program on the page', 'Ran a small program on the page', "Couldn't run the program on the page"],
  browser_run_code: ['Running a small program on the page', 'Ran a small program on the page', "Couldn't run the program on the page"],
  browser_wait_for: ['Waiting for the page', 'Waited for the page', "Couldn't wait for the page"],
  browser_navigate_back: ['Going back a page', 'Went back a page', "Couldn't go back"],
  browser_tabs: ['Managing browser tabs', 'Managed browser tabs', "Couldn't manage the browser tabs"],
  browser_close: ['Closing the browser page', 'Closed the browser page', "Couldn't close the browser page"],
};
const BROWSER_GENERIC = ['Using the browser', 'Used the browser', 'The browser step failed'];
const BROWSER_PREFIX = 'mcp__browser__', INSTRUCTIONS_PREFIX = 'mcp__strata-instructions__', COMMANDS_PREFIX = 'mcp__strata-commands__';
/** The person's own tool's name as they wrote it (spaces for underscores). */
const ownToolName = name => name.replace(/^mcp__strata-(?:instructions|commands)__/, '').replace(/_/g, ' ');
/** The three sentences (running, worked, failed) for a tool, and whether this is one the app knows by name. */
function verbsFor(name) {
  if (VERBS[name]) return { verbs: VERBS[name], known: true };
  if (name.startsWith(BROWSER_PREFIX)) return { verbs: BROWSER_VERBS[name.slice(BROWSER_PREFIX.length)] ?? BROWSER_GENERIC, known: true };
  if (name.startsWith(INSTRUCTIONS_PREFIX)) return { verbs: ['Reading your instructions', 'Read your instructions', "Couldn't read your instructions"], known: true };
  if (name.startsWith(COMMANDS_PREFIX)) return { verbs: ['Running your tool', 'Ran your tool', 'Your tool failed'], known: true };
  return { verbs: GENERIC_VERBS, known: false };
}

/** The sentence for a card in this state (the first for a running or waiting card). */
export function toolVerb(name, state) {
  const v = verbsFor(name).verbs;
  return state === 'error' ? v[2] : state === 'done' ? v[1] : v[0];
}
/** The ToolCall fields the card needs, from one Harness tool call. */
export function presentTool({ callId, name, args, reason = null }) {
  const scope = toolScope(name);
  const words = WORDS[name]?.(args);
  const preview = scope === 'shell' && !words ? args.command ?? ''
    : scope === 'write' ? args.file_path ?? args.path ?? args.filePath ?? ''
      : words ? words[1] ?? '' : name.startsWith(INSTRUCTIONS_PREFIX) || name.startsWith(COMMANDS_PREFIX) ? ownToolName(name) : /^mcp__browser__browser_(evaluate|run_code)$/.test(name) ? '' : args.file_path ?? args.path ?? args.pattern ?? args.url ?? args.element ?? args.query ?? args.text ?? args.action ?? (Object.keys(args).length ? JSON.stringify(args) : '');
  const effect = reason ?? (words ? words[0] : scope === 'shell' ? (args.description ?? 'Runs a command in the workspace.')
    : scope === 'write' ? 'Changes a file in the workspace.'
      : scope === 'read' ? 'Reads from the workspace.' : 'Uses a connected tool.');
  return { id: callId, name, args, preview: clip(preview, 600), effect: clip(effect, 400), scope, verb: toolVerb(name, 'running'), target: clip(verbsFor(name).known ? preview : name, 300), ask: askFor(name, args, typeof args.description === 'string' ? args.description : null) };
}

/** A tool's failure in plain words, or null when there is nothing better than the original text. */
export function plainToolError(text) {
  const t = String(text ?? '');
  if (/non-public ip|enotfound|getaddrinfo|web fetch failed|econnrefused|econnreset|etimedout|fetch failed/i.test(t)) return 'That web page could not be opened. Check the address and your internet connection.';
  if (/invalid arguments|missing required property|not a declared property/i.test(t)) return 'The assistant asked this tool in the wrong way. It will usually try again.';
  if (/unknown or no longer available/i.test(t)) return 'That skill is not available.';
  if (/cannot modify|must (?:first )?be read|has not been read|changed since/i.test(t)) return 'The file could not be changed. It may need to be read again first, or it changed in the meantime.';
  if (/sandbox|access denied|permission denied|eacces|eperm/i.test(t)) return 'The computer did not allow this.';
  if (/timed out|timeout/i.test(t)) return 'This took too long and was stopped.';
  if (/does not declare image input|image input/i.test(t)) return 'Picture reading is off for this model. Open Models and turn on Read pictures for it (load it again if it is a local model), or choose a model that reads pictures.';
  if (/extension|not a supported image|cannot read/i.test(t)) return 'This file could not be read.';
  return null;
}

/** A failed tool's result: the plain sentence first, the original text after it (the card's disclosure is the Details). */
export function withPlain(raw) {
  const plain = plainToolError(raw);
  return plain ? `${plain}\n\nDetails: ${raw.replace(/^Error:\s*/, '')}` : raw;
}

/** What a launch that printed nothing proves, and no more. */
export const LAUNCHED_TEXT = 'The command finished without an error, so the program was started. Strata cannot confirm that a window or page actually opened.';

/** A command whose whole point is to open something: it has no output to show when it worked. */
const LAUNCH = /(^|[\s;|&(])(start-process|start|invoke-item|ii|explorer(?:\.exe)?|xdg-open|open|notepad(?:\.exe)?|code)(?=[\s;|&)]|$)/i;

/**
 * A command that succeeded and printed nothing says so in words (the card used to show "(no output)" under a "Ran" that could not
 * tell the person whether anything happened). Opening a folder or a window is not something a command can prove it did, so the
 * sentence says only what is known.
 */
export function outcomeText(tool, text, args = {}) {
  if (tool === 'pwsh' || tool === 'bash') {
    // In the read-only modes PowerShell runs in Constrained Language Mode and complains about the Harness's own output-encoding line:
    // noise appended after a command that worked. It says nothing about the command, so it is not shown.
    text = text.replace(/\r?\n?\[stderr\]\r?\n(?:Cannot create type\. Only core types are supported in this language mode\.[\s\S]*?CannotCreateTypeConstrainedLanguage\s*)+$/, '');
    if (!text.trim()) text = '(no output)';
  }
  // Starting a program or opening a folder prints nothing when it worked, and also when it did not (a browser given a file:// address, a
  // program that closed at once). T9 / #3576: the card used to say "was opened", and the model repeated it ("I previewed it in a browser to
  // confirm it renders"). It says what is known: the command returned without an error.
  if ((tool === 'pwsh' || tool === 'bash') && /^\(no output\)$/.test(text.trim()) && LAUNCH.test(String(args?.command ?? ''))) return LAUNCHED_TEXT;
  if ((tool === 'pwsh' || tool === 'bash') && /^\(no output\)$/.test(text.trim())) return 'The command finished without any error. It printed nothing, so there is nothing more to show.';
  return text;
}

/** Text of a tool-result message, bounded. */
export function resultText(message) {
  const parts = Array.isArray(message?.content) ? message.content : [];
  const text = parts.map(p => (p?.type === 'text' ? p.text : p?.type ? `[${p.type}]` : '')).join('\n').trim();
  return clip(text || '(no output)', 4000);
}

/**
 * Accumulates one turn's frames. Feed it follow items; it returns the frames to send.
 * It owns the per-turn state (tool cards, usage, text emitted) so the I/O layer stays thin.
 */
export class TurnProjector {
  /**
   * `policy` is the session's approval policy for this turn. Under `never` the Harness rejects an ask by itself, so
   * no question ever reaches the person: the card goes straight from running to denied, without a moment where it
   * offers Approve and Deny buttons that would do nothing.
   */
  constructor({ startedAt = Date.now(), policy = 'ask', window = 0 } = {}) {
    this.startedAt = startedAt;
    this.window = Number(window) || 0;   // the model's window, for the figures in the line that says the chat was summarised
    this.summarizing = false;
    this.policy = policy;
    this.byPolicy = new Set();       // callIds the approval policy rejected without asking
    this.tools = new Map();          // callId -> { card, calledAt }
    this.tokensIn = 0;
    this.tokensOut = 0;
    this.contextTokens = null;       // the LAST model request's prompt + reply: what the context holds now (the sums above count every request again)
    this.usageSeen = false;
    this.firstTokenAt = null;
    this.streams = new Map();        // attemptId -> { first, last, chunks, chars }: the time a model request actually spent producing
    this.lastStats = 0;
    this.emittedText = new Map();    // attemptId -> chars sent live
    this.ended = null;               // the turn/end reason, once seen
    this.denied = new Set();         // callIds a human or the policy rejected
    this.asked = new Map();          // approval id -> callId
    this.onStatus = null;            // (update) => void: what the assistant is doing now (status.mjs); set by the bridge
    this.reasoning = new TokenCount();
    this.composing = new Map();      // callId -> { name, head, bytes }: a tool call the model is still writing
    this.running = [];               // callIds of tools that are running, oldest first
  }

  #status(update) { if (this.onStatus) this.onStatus(update); }

  /** After a tool ends: another tool still running is what is happening, otherwise the model is working out what to do next. */
  #afterTool() {
    const id = this.running.at(-1);
    const card = id ? this.tools.get(id)?.card : null;
    if (card) this.#status(toolUpdate('tool', 'tool', card.name, card.args, {}, { args: { tool: card.name, verb: card.verb, target: card.target } }));
    else this.#status({ phase: 'waiting', code: 'model-working', text: 'Working out the next step…' });
  }

  #card(callId, patch) {
    const entry = this.tools.get(callId);
    if (!entry) return null;
    entry.card = { ...entry.card, ...patch };
    entry.card.verb = toolVerb(entry.card.name, entry.card.state);
    return { type: 'agent-tool', tool: entry.card };
  }

  /* The speed shown while a reply streams, and after it: everything the model generated (reasoning and tool calls included) over the time its
     requests spent producing it, not over the whole turn (tool runs and the wait for the prompt are not decoding). One stream per model
     request; tokens are one per streamed chunk, or characters / 4 when a server batches. */
  #produced(attemptId, text, out) {
    const now = Date.now();
    const s = this.streams.get(attemptId) ?? { first: now, last: now, chunks: 0, chars: 0 };
    s.last = now; s.chunks += 1; s.chars += text.length;
    this.streams.set(attemptId, s);
    if (now - this.lastStats < 400) return;
    const { tokens, seconds } = this.decode();
    if (seconds < 0.5) return;
    this.lastStats = now;
    out.push({ type: 'stats', tokensOut: tokens, tps: Number((tokens / seconds).toFixed(1)), source: 'stream' });
  }

  decode() {
    let tokens = 0, ms = 0;
    for (const s of this.streams.values()) { tokens += s.chars / Math.max(1, s.chunks) < 8 ? s.chunks : Math.ceil(s.chars / 4); ms += s.last - s.first; }
    return { tokens, seconds: ms / 1000 };
  }

  /** Process one `session/follow` item (event or assistant-stream). Returns frames. */
  consume(item) {
    const out = [];
    if (item.type === 'assistant-stream') {
      const f = item.frame;
      if (f.type === 'chunk') {
        const c = f.chunk;
        if (c.type === 'text-delta' && c.text) {
          this.firstTokenAt ??= Date.now();
          this.#produced(f.attemptId, c.text, out);
          this.emittedAny = true;
          this.#status({ phase: 'answering', code: 'answering', text: 'Writing the answer…' });
          this.emittedText.set(f.attemptId, (this.emittedText.get(f.attemptId) ?? 0) + c.text.length);
          out.push({ type: 'delta', text: c.text });
        } else if (c.type === 'reasoning-delta' && c.text) {
          this.firstTokenAt ??= Date.now();
          this.#produced(f.attemptId, c.text, out);
          // The count is for the model's current thinking (each model request starts again from zero), not the whole turn.
          if (this.reasoningAttempt !== f.attemptId) { this.reasoning = new TokenCount(); this.reasoningAttempt = f.attemptId; }
          this.reasoning.add(c.text);
          this.#status({ phase: 'thinking', code: 'thinking', text: `Thinking… ${fmtTokens(this.reasoning.tokens)} tokens`, key: 'Thinking… {0} tokens', values: [fmtTokens(this.reasoning.tokens)], tokens: this.reasoning.tokens, args: { tokens: this.reasoning.tokens } });
          out.push({ type: 'reasoning', text: c.text });
        } else if (c.type === 'tool-call-delta' && c.id) {
          // The model is writing a tool call. For a big file this is the long silent part: nothing else arrives until it is finished.
          const call = this.composing.get(c.id) ?? { name: c.name ?? '', head: '', bytes: 0 };
          if (c.name) call.name = c.name;
          const piece = String(c.argumentsDelta ?? '');
          // Writing a tool call (a whole file, say) is decoding too: its tokens are in the engine's output count, so its time belongs in the speed.
          this.#produced(f.attemptId, piece, out);
          call.bytes += piece.length;
          if (call.head.length < 4000) call.head += piece;
          this.composing.set(c.id, call);
          if (call.name && !NO_CARD.has(call.name)) {
            const raw = (/"(?:file_path|path|filePath)"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(call.head) ?? /"url"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(call.head))?.[1];
            let target = raw; try { if (raw !== undefined) target = JSON.parse('"' + raw + '"'); } catch { /* a half-written escape: show it as it is */ }
            const args = target ? { file_path: target, url: target } : {};
            this.#status(toolUpdate('preparing', 'preparing-tool', call.name, args, { bytes: call.bytes }, { args: { tool: call.name, verb: toolVerb(call.name, 'running'), target: target ?? '', bytes: call.bytes } }));
          }
        } else if (c.type === 'usage' && c.usage) {
          this.usageSeen = true;
          // The Harness counts input, cache reads and cache writes apart (a prompt the server's conversation cache already held is `cacheReadTokens`,
          // not `inputTokens`): the prompt of a request is all three. Counting only `inputTokens` made a long agent chat look nearly empty (0%).
          const prompt = (Number(c.usage.inputTokens) || 0) + (Number(c.usage.cacheReadTokens) || 0) + (Number(c.usage.cacheWriteTokens) || 0);
          this.tokensIn += prompt;
          this.tokensOut += Number(c.usage.outputTokens) || 0;
          this.contextTokens = prompt + (Number(c.usage.outputTokens) || 0);
          out.push({ type: 'context', contextTokens: this.contextTokens, contextMax: this.window || undefined });
        }
      }
      return out;
    }
    if (item.type !== 'event') return out;
    const e = item.event;
    const d = e.data ?? {};
    switch (e.type) {
      case 'assistant/message': {
        // Reconcile a text the live frames did not fully carry (a socket gap), never duplicating it.
        const live = [...this.emittedText.values()].pop() ?? 0;
        const text = (d.message?.content ?? []).filter(p => p.type === 'text').map(p => p.text).join('');
        if (text.length > live && live > 0) out.push({ type: 'delta', text: text.slice(live) });
        this.emittedText.clear();
        break;
      }
      case 'tool/call': {
        if (NO_CARD.has(d.name)) break;
        const args = parseArguments(d.arguments);
        const card = { ...presentTool({ callId: d.callId, name: d.name, args }), state: 'running' };
        this.tools.set(d.callId, { card, calledAt: e.time });
        this.composing.delete(d.callId);
        this.running.push(d.callId);
        this.#status(toolUpdate('tool', 'tool', d.name, args, {}, { args: { tool: d.name, verb: card.verb, target: card.target } }));
        out.push({ type: 'agent-tool', tool: card });
        break;
      }
      case 'approval/asked': {
        if (!d.callId) break;
        this.asked.set(d.id, d.callId);
        if (this.policy === 'never') break;
        const entry = this.tools.get(d.callId);
        this.#status({ phase: 'approval', code: 'waiting-approval', text: 'Waiting for your answer…', args: { tool: entry?.card.name ?? '' } });
        const patch = { state: 'proposed', effect: clip(d.reason ?? entry?.card.effect ?? 'Needs your approval.', 400) };
        const frame = this.#card(d.callId, patch);
        if (frame) out.push(frame);
        break;
      }
      case 'approval/decided': {
        const callId = this.asked.get(d.id);
        if (callId && this.policy === 'never' && d.outcome !== 'allowed-once') {
          this.byPolicy.add(callId); this.denied.add(callId);
          const frame = this.#card(callId, { state: 'denied', result: NOT_IN_MODE });
          if (frame) out.push(frame);
          break;
        }
        this.#afterTool();
        const frame = callId ? this.decide(callId, d.outcome) : null;
        if (frame) out.push(frame);
        break;
      }
      case 'tool/result': {
        const callId = d.message?.toolCallId;
        const entry = this.tools.get(callId);
        if (!entry) break;
        this.running = this.running.filter(id => id !== callId);
        this.#afterTool();
        const isError = d.message?.isError === true;
        const denied = this.denied.has(callId);
        const frame = this.#card(callId, {
          state: denied ? 'denied' : isError ? 'error' : 'done',
          // A card the person (or the mode) refused keeps its plain sentence; the model's own error text ("the user rejected ...") is not for them.
          result: this.byPolicy.has(callId) ? NOT_IN_MODE : denied ? (entry.card.result ?? 'Denied. Nothing ran.') : isError ? withPlain(resultText(d.message)) : outcomeText(entry.card.name, resultText(d.message), entry.card.args),
          durationMs: Math.max(0, (e.time ?? 0) - entry.calledAt),
        });
        if (frame) out.push(frame);
        break;
      }
      case 'compaction/start':
        // The Harness is summarising the older messages to make room: say so while it does, and again (once) when it is done.
        this.summarizing = true;
        this.#status({ phase: 'compacting', code: 'compacting', text: FIT_TEXT.summarizing, key: FIT_TEXT.summarizing, values: [] });
        break;
      case 'compaction/end':
        if (!this.summarizing) break;
        this.summarizing = false;
        this.#afterTool();
        if (!d.error) {
          const before = this.window > 0 && this.contextTokens ? Math.min(100, Math.round(100 * this.contextTokens / this.window)) : null;
          const text = before === null ? FIT_TEXT.summarizedShort : FIT_TEXT.summarizedWas;
          const values = before === null ? [] : [before];
          out.push({ type: 'delta', own: true, text: '\n\n' + fill(text, values), key: text, values, noteReason: 'compacted' });
        }
        break;
      case 'turn/start':
        // The parent woke up again (a helper finished and reported back): the message goes on, as a new paragraph.
        this.ended = null;
        if (this.emittedAny) out.push({ type: 'delta', text: '\n\n' });
        break;
      case 'turn/end':
        this.ended = d.reason ?? { kind: 'completed' };
        break;
      default:
        break;
    }
    return out;
  }

  /** An approval was decided. `outcome` is the Harness vocabulary. */
  decide(callId, outcome) {
    if (!this.tools.has(callId)) return null;
    // `approved` is what lets the card say "Approved" (the user answered) rather than "Ran" (it never asked).
    if (outcome === 'allowed-once') return this.#card(callId, { state: 'running', approved: true });
    this.denied.add(callId);
    return this.#card(callId, { state: 'denied', result: outcome === 'rejected' ? 'Denied. Nothing ran.' : outcome === 'cancelled' ? 'Cancelled. Nothing ran.' : 'No one could answer this request, so nothing ran.' });
  }

  /** The terminal frames for a finished turn. */
  finish({ model, contextMax, engine = null } = {}) {
    const reason = this.ended ?? { kind: 'interrupted' };
    const durationMs = Date.now() - this.startedAt;
    const usage = this.usageSeen
      ? { tokensIn: this.tokensIn, tokensOut: this.tokensOut, usageSource: 'reported' }
      : { tokensIn: null, tokensOut: null, usageSource: 'unavailable' };
    const decoded = this.decode();
    const generated = this.usageSeen && this.tokensOut > 0 ? this.tokensOut : decoded.tokens;
    // Answer speed, one definition everywhere: all the tokens the model produced over its decode time. The engine's own figures (the ones the
    // Monitor shows) when the bridge could read them; otherwise the stream's own span. A very short reply has almost no span, so its decode time
    // is then the time from its first chunk to the end of the turn (only when no tool ran, so no tool time is in it): the number is still shown.
    let tps = null;
    if (engine?.tps > 0) tps = engine.tps;
    else if (generated > 0 && decoded.seconds >= 0.3) tps = Number((generated / decoded.seconds).toFixed(1));
    else if (generated > 0 && this.tools.size === 0 && this.firstTokenAt && Date.now() - this.firstTokenAt >= 50) tps = Number((generated / ((Date.now() - this.firstTokenAt) / 1000)).toFixed(1));
    const promptTps = engine?.promptTps > 0 ? engine.promptTps : null;
    const base = { durationMs, ttftMs: this.firstTokenAt ? this.firstTokenAt - this.startedAt : null, model, contextMax, tps, ...(promptTps ? { promptTps } : {}), contextTokens: this.contextTokens };
    switch (reason.kind) {
      case 'completed':
        return [{ type: 'done', ...base, ...usage, finishReason: 'stop' }];
      case 'max-tokens':
        return [{ type: 'done', ...base, ...usage, finishReason: 'length' }];
      case 'aborted':
        return [{ type: 'done', ...base, ...usage, finishReason: 'aborted', cancelled: true }];
      case 'error':
        return [{ type: 'error', message: friendlyFailure(reason.error), code: isContextFailure(reason.error) ? 'CONTEXT_WINDOW_EXCEEDED' : reason.error?.code ?? 'UNKNOWN', ...(isContextFailure(reason.error) ? { key: FIT_TEXT.full, details: String(reason.error?.message ?? '') } : {}) }];
      case 'blocked':
        return [{ type: 'error', message: 'The agent stopped because this turn was blocked by policy.', code: 'BLOCKED' }];
      default:
        return [{ type: 'error', message: 'The agent turn was interrupted before it finished.', code: 'INTERRUPTED' }];
    }
  }
}

/** Whether a failure is the model server saying the conversation does not fit its window (its code, or the engine's own words). */
export function isContextFailure(failure) {
  return isContextOverflow(failure) || /context window|too long for|exceeds the context/i.test(String(failure?.message ?? ''));
}

/** A model-route failure in words a user can act on. */
export function friendlyFailure(failure) {
  const message = String(failure?.message ?? 'The model request failed.');
  if (isContextFailure(failure)) return FIT_TEXT.full;
  if (/ECONNREFUSED|fetch failed|connect/i.test(message)) return `The model is not reachable. ${clip(message, 200)}`;
  if (/401|403|credential|api key/i.test(message)) return 'The model rejected the credentials. Check the connection in Settings.';
  return clip(message, 400);
}
