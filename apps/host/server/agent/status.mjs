/**
 * Live status for a running turn: one small `status` frame that says what the assistant is doing right now.
 *
 * The page draws only the LATEST status, so a status must be replaced the moment it stops being true. This reporter therefore sends a
 * new frame on every change of phase or target (at most one per `gapMs`, the newest change wins), and between changes only repeats the
 * frame to refresh a counter (reasoning tokens, bytes written) every `beatMs`. It never sends the same words twice in a row.
 *
 *   { type:'status', seq, phase, code, text, key?, values?, args?, tokens?, elapsedMs }
 *   `key`/`values` (only when the sentence has a name or a number in it): the catalog template ("Writing {0}…") and its values, for translation.
 *
 * phases: starting waiting thinking answering preparing tool approval helpers goal idle   (see notes-split-agent3.md)
 */

/** Fill the `{0}` slots of a template (the catalog key of a sentence) with values. */
export const fillTemplate = (key, values = []) => String(key).replace(/\{(\d+)\}/g, (_, i) => String(values[Number(i)] ?? ''));

/**
 * The sentence for a tool, in the person's words: what it is doing to what ("Writing pagoda.html…"), as a catalog template and its values
 * so the page can translate it: `{ key: 'Writing {0}…', values: ['pagoda.html'] }`.
 */
export function toolStatus(name, args = {}, { bytes = 0 } = {}) {
  const base = value => String(value ?? '').split(/[\\/]/).filter(Boolean).pop() ?? '';
  const file = base(args.file_path ?? args.path ?? args.filePath);
  const kb = bytes >= 1500 ? (bytes / 1024).toFixed(bytes >= 10240 ? 0 : 1) : null;
  const act = (verb, noun) => file
    ? (kb ? { key: `${verb} {0}… ({1} KB)`, values: [file, kb] } : { key: `${verb} {0}…`, values: [file] })
    : (kb ? { key: `${verb} ${noun}… ({0} KB)`, values: [kb] } : { key: `${verb} ${noun}…`, values: [] });
  switch (name) {
    case 'write': return act('Writing', 'a file');
    case 'edit': return act('Changing', 'a file');
    case 'read': return file ? { key: 'Reading {0}…', values: [file] } : { key: 'Reading a file…', values: [] };
    case 'read_image': return { key: 'Looking at a picture…', values: [] };
    case 'pwsh': case 'bash': return { key: 'Running a command…', values: [] };
    case 'glob': case 'grep': return { key: 'Searching your files…', values: [] };
    case 'web_search': return { key: 'Searching the web…', values: [] };
    case 'web_fetch': { let host = ''; try { host = new URL(args.url).hostname; } catch { /* partial address */ } return host ? { key: 'Opening {0}…', values: [host] } : { key: 'Opening a web page…', values: [] }; }
    case 'create_goal': return { key: 'Setting the goal…', values: [] };
    case 'update_goal': return { key: 'Updating the goal…', values: [] };
    case 'get_goal': return { key: 'Checking the goal…', values: [] };
    case 'todo_write': return { key: 'Updating its to-do list…', values: [] };
    case 'skill': return { key: 'Using a saved skill…', values: [] };
    case 'present': return { key: 'Showing you a result…', values: [] };
    case 'workflow': return { key: 'Running a multi-step plan…', values: [] };
    case 'job_list': case 'job_output': case 'job_kill': return { key: 'Checking background tasks…', values: [] };
    default:
      if (name.startsWith('mcp__browser__')) return name.endsWith('navigate') ? (args.url ? { key: 'Opening {0}…', values: [String(args.url)] } : { key: 'Opening a page…', values: [] }) : { key: 'Using the browser…', values: [] };
      if (name.startsWith('mcp__strata-instructions__')) return { key: 'Reading your instructions…', values: [] };
      if (name.startsWith('mcp__strata-commands__')) return { key: 'Running your tool…', values: [] };
      return { key: 'Using a tool…', values: [] };
  }
}

/** The same, as one English sentence. */
export function toolStatusText(name, args = {}, options = {}) {
  const s = toolStatus(name, args, options);
  return fillTemplate(s.key, s.values);
}

/** A status update (what the reporter takes) for a tool: text, catalog template and values together. */
export function toolUpdate(phase, code, name, args, options, extra = {}) {
  const s = toolStatus(name, args, options);
  return { phase, code, text: fillTemplate(s.key, s.values), key: s.key, values: s.values, ...extra };
}

const COUNTERS = new Set(['tokens', 'bytes']);
const keyOf = u => `${u.phase}|${u.code}|${JSON.stringify(Object.fromEntries(Object.entries(u.args ?? {}).filter(([k]) => !COUNTERS.has(k))))}`;

export class StatusReporter {
  /**
   * @param {{emit: (frame: object) => void, now?: () => number, gapMs?: number, beatMs?: number, timers?: {set: Function, clear: Function}}} options
   */
  constructor({ emit, now = Date.now, gapMs = 300, beatMs = 2000, timers = { set: setTimeout, clear: clearTimeout } }) {
    Object.assign(this, { emitFrame: emit, now, gapMs, beatMs, timers });
    this.seq = 0;
    this.current = null;      // what the page was last told
    this.wanted = null;       // the newest thing that is true now
    this.phaseStartedAt = 0;
    this.lastEmitAt = -Infinity;
    this.timer = null;
    this.closed = false;
  }

  /** Say what is true now. `{phase, code, text, args?, tokens?}`. */
  set(update) {
    if (this.closed) return;
    const next = { ...update, args: update.args ? { ...update.args } : undefined };
    const changed = !this.wanted || keyOf(next) !== keyOf(this.wanted);
    if (changed) this.phaseStartedAt = this.now();
    this.wanted = next;
    const sinceEmit = this.now() - this.lastEmitAt;
    const due = changed ? this.gapMs : this.beatMs;
    if (!this.current || sinceEmit >= due || update.phase === 'goal') this.#flush();   // a goal step is rare and the page should always hear it
    else if (this.#differs()) this.#arm(due - sinceEmit);
  }

  /** Nothing is happening any more: tell the page to drop the line. */
  clear() {
    if (this.closed) return;
    if (this.current && this.current.phase !== 'idle') { this.wanted = { phase: 'idle', code: 'idle', text: '' }; this.#flush(); }
    this.dispose();
  }

  dispose() { this.closed = true; if (this.timer) this.timers.clear(this.timer); this.timer = null; }

  #differs() {
    const a = this.wanted, b = this.current;
    return !b || a.text !== b.text || a.tokens !== b.tokens;
  }

  #arm(ms) {
    if (this.timer || this.closed) return;
    this.timer = this.timers.set(() => { this.timer = null; if (!this.closed && this.#differs()) this.#flush(); }, Math.max(0, ms));
    this.timer.unref?.();
  }

  #flush() {
    if (this.timer) { this.timers.clear(this.timer); this.timer = null; }
    const w = this.wanted;
    if (!w || (this.current && this.current.text === w.text && this.current.phase === w.phase && this.current.tokens === w.tokens)) return;
    this.current = w;
    this.lastEmitAt = this.now();
    this.emitFrame({
      type: 'status', seq: ++this.seq, phase: w.phase, code: w.code, text: w.text,
      ...(w.key && w.key !== w.text ? { key: w.key, values: w.values ?? [] } : {}),
      ...(w.args ? { args: w.args } : {}), ...(w.tokens !== undefined ? { tokens: w.tokens } : {}),
      elapsedMs: Math.max(0, this.now() - this.phaseStartedAt),
    });
  }
}

/** Counts streamed tokens: one per chunk, or characters / 4 when a server batches many characters into a chunk. */
export class TokenCount {
  constructor() { this.chunks = 0; this.chars = 0; }
  add(text) { this.chunks += 1; this.chars += String(text).length; }
  get tokens() { return this.chars / Math.max(1, this.chunks) < 8 ? this.chunks : Math.ceil(this.chars / 4); }
}

export const fmtTokens = n => Number(n).toLocaleString('en-US');

/**
 * The same status line for a plain chat turn (no tools): the request is being read, the model is thinking (tokens), the model is writing.
 * Wraps the function that writes frames to the client; the returned function has `dispose()` for a client that left early.
 */
export function withPlainStatus(write, options = {}) {
  const reporter = new StatusReporter({ emit: write, ...options });
  const reasoning = new TokenCount();
  reporter.set({ phase: 'starting', code: 'starting', text: 'Starting…' });
  const frame = f => {
    switch (f.type) {
      case 'start': reporter.set({ phase: 'waiting', code: 'reading-prompt', text: 'Reading your message…' }); break;
      case 'reasoning': reasoning.add(f.text ?? ''); reporter.set({ phase: 'thinking', code: 'thinking', text: `Thinking… ${fmtTokens(reasoning.tokens)} tokens`, key: 'Thinking… {0} tokens', values: [fmtTokens(reasoning.tokens)], tokens: reasoning.tokens, args: { tokens: reasoning.tokens } }); break;
      case 'delta': reporter.set({ phase: 'answering', code: 'answering', text: 'Writing the answer…' }); break;
      case 'done': case 'error': reporter.clear(); break;
      default: break;
    }
    write(f);
  };
  frame.dispose = () => reporter.dispose();
  return frame;
}
