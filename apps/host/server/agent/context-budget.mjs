/**
 * Token arithmetic for the agent's context, with no I/O (the Harness plugin `plugins/strata-context.mjs` and the host both use it).
 *
 * Why it exists. The Harness prices a conversation with a fixed 4 characters per token. Code, logs and JSON are about twice as dense
 * (2.09 characters per token in the 2026-10-22 hands-on test), so its pressure read 31,794 tokens while the engine had been sent 59,048,
 * and compaction waited until the window was already past full. The engine reports the real prompt size with every answer; this module
 * turns that report into the number compaction decides on, and splits a summarising request that would not fit the window.
 */

/** The Harness's own price of a text: characters over this, plus a fixed cost per block and per message. */
export const CHARS_PER_TOKEN = 4;
const BLOCK_OVERHEAD = 4;
const MESSAGE_OVERHEAD = 4;
/** The most a real token can be worth in the Harness's units (one character per token is the worst a tokenizer does). */
export const MAX_DENSITY = 6;
/** Room kept free in a window for the engine's own framing and the chat template (the engine keeps 64 tokens of slack itself). */
export const WINDOW_SLACK = 1024;

/** The Harness's price (4 characters a token) of one message, the same arithmetic as its token meter. */
export function estimateMessage(message) {
  const content = Array.isArray(message?.content) ? message.content : [];
  if (message?.role === 'system') {
    if (!content.length) return 0;
    let chars = 0;
    for (const block of content) chars += block?.type === 'text' ? String(block.text ?? '').length : JSON.stringify(block ?? {}).length;
    return Math.ceil(chars / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD;
  }
  let tokens = MESSAGE_OVERHEAD;
  for (const block of content) {
    if (block?.type === 'text' || block?.type === 'reasoning') tokens += Math.ceil(String(block.text ?? '').length / CHARS_PER_TOKEN) + BLOCK_OVERHEAD;
    else if (block?.type === 'tool-call') tokens += Math.ceil(String(block.name ?? '').length / CHARS_PER_TOKEN) + Math.ceil(String(block.arguments ?? '').length / CHARS_PER_TOKEN) + BLOCK_OVERHEAD;
    else tokens += BLOCK_OVERHEAD + Math.ceil(JSON.stringify(block ?? {}).length / CHARS_PER_TOKEN);
  }
  return tokens;
}

/** The Harness's price of the tool list of a request header. */
export function estimateTools(header) {
  const tools = header?.tools;
  if (!Array.isArray(tools) || !tools.length) return 0;
  return Math.ceil(JSON.stringify(tools).length / CHARS_PER_TOKEN) + BLOCK_OVERHEAD;
}

/**
 * How many real tokens one of the Harness's tokens is worth, from one measured request: the engine said the prompt was `real` tokens,
 * the Harness priced the same prompt at `estimated`. Never below 1 (a conversation of plain prose is priced about right, and a lower
 * number would only delay compaction) and never above MAX_DENSITY (a wild sample must not make every step compact).
 */
export function densityOf(real, estimated) {
  if (!(real > 0) || !(estimated > 0)) return 1;
  return Math.min(MAX_DENSITY, Math.max(1, real / estimated));
}

/**
 * The prompt size compaction decides on. `sample` is the last measured request: `real` tokens for a prompt the Harness priced at
 * `estimated`. `now` is the Harness's price of the conversation now. What was added or removed since the sample is scaled by the sample's
 * density; the measured part is exact. Without a sample, or when the measurement is lower than the Harness's own price, the Harness's
 * price stands (it is never an underestimate that makes this smaller).
 */
export function pressureNow({ sample = null, now }) {
  if (!sample || !(sample.real > 0) || !(sample.estimated > 0)) return Math.max(0, now);
  const density = densityOf(sample.real, sample.estimated);
  return Math.max(Math.round(now), Math.round(sample.real + density * (now - sample.estimated)));
}

/** The usage a finished model request reported, as the prompt side only: input plus the cache's reads and writes (output is not in the prompt). */
export function promptTokensOf(usage) {
  if (!usage || typeof usage !== 'object') return 0;
  const n = value => (Number.isFinite(Number(value)) ? Math.max(0, Number(value)) : 0);
  return n(usage.inputTokens) + n(usage.cacheReadTokens) + n(usage.cacheWriteTokens);
}

/** The usage of a settled assistant event (a `usage` on the message, or the last usage chunk of its stream). */
export function usageOfEvent(event) {
  const data = event?.data;
  if (!data) return null;
  if (data.usage && typeof data.usage === 'object') return data.usage;
  const stream = Array.isArray(data.stream) ? data.stream : [];
  for (let i = stream.length - 1; i >= 0; i--) if (stream[i]?.type === 'chunk' && stream[i].chunk?.type === 'usage') return stream[i].chunk.usage ?? null;
  return null;
}

/**
 * A summarising request is: the conversation's system prompt and tools, the messages to condense, then the instruction. This is how many
 * real tokens the messages may take in a window so that all of it, the summary and a safety margin fit.
 */
export function summaryBudget({ window, summaryTokens, toolsTokens = 0, systemTokens = 0, instructionTokens = 900, slack = WINDOW_SLACK }) {
  return Math.floor(window - summaryTokens - toolsTokens - systemTokens - instructionTokens - slack);
}

/** Text blocks of a message, longest first: the ones worth shortening when a single message is bigger than the budget. */
function textBlocks(message) {
  return (Array.isArray(message?.content) ? message.content : []).filter(b => (b?.type === 'text' || b?.type === 'reasoning') && typeof b.text === 'string');
}

/**
 * One message cut down to `maxTokens` (in the Harness's price times `density`): the start and the end of its longest text are kept and
 * the middle is replaced by a line saying how much was left out. Only the copy sent to the summariser is cut; the conversation keeps
 * the whole message until the summary replaces it.
 */
export function clipMessage(message, maxTokens, density = 1) {
  if (estimateMessage(message) * density <= maxTokens) return message;
  const blocks = textBlocks(message);
  if (!blocks.length) return message;
  const others = estimateMessage({ ...message, content: (message.content ?? []).filter(b => !blocks.includes(b)) }) * density;
  const room = Math.max(256, maxTokens - others);
  const share = room / blocks.reduce((n, b) => n + b.text.length, 0) / density * CHARS_PER_TOKEN;   // characters kept per character
  const content = (message.content ?? []).map((block) => {
    if (!blocks.includes(block)) return block;
    const keep = Math.max(200, Math.floor(block.text.length * Math.min(1, share) - 120));
    if (block.text.length <= keep) return block;
    const head = Math.ceil(keep * 0.6), tail = keep - head;
    const left = block.text.length - head - tail;
    return { ...block, text: `${block.text.slice(0, head)}\n\n[… ${left.toLocaleString('en-US')} characters from the middle are left out of this summary request …]\n\n${block.text.slice(block.text.length - tail)}` };
  });
  return { ...message, content };
}

/** Whether cutting the list in front of index `i` would separate a tool call from its result. */
function balancedBefore(messages, i) {
  let open = 0;
  for (let k = 0; k < i; k++) {
    const m = messages[k];
    if (m?.role === 'assistant') open += (Array.isArray(m.content) ? m.content : []).filter(b => b?.type === 'tool-call').length;
    else if (m?.role === 'tool') open -= 1;
  }
  return open <= 0;
}

/**
 * Split messages into runs that each fit `budget` real tokens (price `density`), cutting between a tool call's result and the next
 * message when it can. A message bigger than the budget by itself is clipped first. Order is kept.
 */
export function splitMessages(messages, budget, density = 1) {
  const room = Math.max(512, budget);
  const sized = messages.map(m => clipMessage(m, Math.floor(room * 0.8), density));
  const cost = sized.map(m => estimateMessage(m) * density);
  const runs = [];
  let start = 0, used = 0;
  for (let i = 0; i < sized.length; i++) {
    if (i > start && used + cost[i] > room) {
      // Cut as late as possible, but never between a tool call and its result when a clean place exists inside the run.
      let cut = i;
      while (cut > start + 1 && !balancedBefore(sized, cut)) cut--;
      if (!balancedBefore(sized, cut)) cut = i;
      runs.push(sized.slice(start, cut));
      start = cut;
      used = cost.slice(start, i).reduce((n, x) => n + x, 0);
    }
    used += cost[i];
  }
  if (start < sized.length) runs.push(sized.slice(start));
  return runs;
}

/** Whether a failure is the model server saying the prompt does not fit its window. */
export function isContextOverflow(error) {
  if (!error) return false;
  if (error.code === 'CONTEXT_WINDOW_EXCEEDED') return true;
  const text = [error.message, error.cause?.message, error.failure?.message, error.failure?.code].filter(Boolean).join(' ');
  return /request exceeds the context window|leaves no room to answer|context[_ ]length|context window|too many tokens|maximum context/i.test(text);
}
