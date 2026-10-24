/**
 * The Strata profile for the Harness, generated from what the host knows right now.
 *
 * The static profile in `integrations/harness/profile/cordis.patch.yml` names one local route on a
 * fixed port. The app needs more than that: whichever local Strata instances are ready, and one route
 * per saved API connection, so that each chat can pick its own model. This module writes the same two
 * Cordis entries (`agent-default-model`, `llm-pi-ai`) for the current set.
 *
 * Rules carried over from the integration notes (docs/work/IMPLEMENTATION_STATUS.md section 1.8):
 *   - `llm-pi-ai` config is `{ providers: { "<route>": {...} } }`; a route key at the top level is
 *     silently ignored.
 *   - the output budget must fit the whole request, so `maxTokens` is derived from the real context.
 *   - no cloud route is declared unless the user saved that connection, so nothing can fall back to a
 *     hosted model by itself.
 *
 * Secrets: a route names a credential by an environment-variable NAME (`apiKeyEnv`). The values are
 * returned separately in `env` for the host to hand to the agent process; they are never written to
 * the profile file, a prompt, or a log.
 */
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { standardPreset } from './preset.mjs';
import { validMcpName } from '../mcp-names.mjs';
import { BUDGETS, isPrivateHost } from './llm-shape.mjs';
import { DDG_PROVIDER_ID } from '../web-search-ddg.mjs';

const PROTOCOL = { 'openai-chat': 'openai-completions', 'openai-responses': 'openai-responses', anthropic: 'anthropic-messages' };

/** A conservative window for an API model whose connection does not report one. */
export const ASSUMED_API_CONTEXT = 128_000;

/**
 * The three chat modes as Harness permission presets (see README, "Modes"). A preset is a
 * sandbox mode plus an approval policy; the Harness switches them with its own `/permission` command.
 *   manual  open sandbox + ask:        the Strata gate (plugins/strata-gate.mjs) asks BEFORE any change runs; nothing is denied first
 *   plan    read-only sandbox + never: nothing can change, and the agent is told not to ask
 *   bypass  danger-full-access + never: the Harness's own full-access, never-ask bundle (its `danger-full-access` preset)
 */
export const MODE_PRESETS = Object.freeze({
  manual: { key: 'strata-manual', sandbox: 'danger-full-access', approval: 'ask', name: 'Manual', description: 'Asks before changing a file or running a command that changes anything.' },
  plan: { key: 'strata-plan', sandbox: 'read-only', approval: 'never', name: 'Plan', description: 'Read-only. Nothing is changed and no approval prompts appear.' },
  bypass: { key: 'strata-bypass', sandbox: 'danger-full-access', approval: 'never', name: 'Bypass', description: 'Full access without approval prompts. Files and commands anywhere on this computer run without asking.' },
});
export const DEFAULT_MODE = 'manual';
/** Helper agents running at once when no local engine sets the limit (the Harness's own default). */
export const DEFAULT_API_HELPERS = 8;
/**
 * Helpers a local chat may have at once: at least the engine's request slots, and never fewer than this. With one slot the engine
 * serves its requests one after the other (its queue is unbounded), so the extra helpers show as waiting for their turn instead
 * of being refused.
 */
export const MIN_LOCAL_HELPERS = 4;
/** The preset for a chat's stored mode. Anything unknown (older values such as `auto`) gets the strictest ask-first mode. */
export const modeOf = value => (Object.hasOwn(MODE_PRESETS, value) ? value : DEFAULT_MODE);

/** The approval gate plugin (plain ESM, loaded by absolute path). */
export const GATE_PLUGIN = fileURLToPath(new URL('./plugins/strata-gate.mjs', import.meta.url));
/** The app's web search (DuckDuckGo, or the person's own Exa / Perplexity key), registered with the Harness's web seam. */
/** Compaction that measures the real prompt and summarises in pieces that fit the window (plugins/strata-context.mjs). */
export const CONTEXT_PLUGIN = fileURLToPath(new URL('./plugins/strata-context.mjs', import.meta.url));
export const MODELS_PLUGIN = fileURLToPath(new URL('./plugins/strata-models.mjs', import.meta.url));
export const SEARCH_PLUGIN = fileURLToPath(new URL('./plugins/strata-search.mjs', import.meta.url));
/** The app's own tool server for the tools a person writes (instructions, commands). */
export const CUSTOM_TOOLS_SCRIPT = fileURLToPath(new URL('./custom-tools-mcp.mjs', import.meta.url));

/** The search service as the person knows it, for the question that comes before the first web search in a chat. */
const SEARCH_NAMES = Object.freeze({ builtin: 'DuckDuckGo, or Bing when DuckDuckGo is busy', exa: 'Exa', perplexity: 'Perplexity', deepseek: 'DeepSeek' });

/** What Settings -> Tools says when nothing was chosen. */
const NO_TOOLS_SETTINGS = Object.freeze({ off: new Set(), search: { provider: 'builtin', key: '' }, webWithoutAsking: false, browserEnabled: false, browserPath: null, browserCli: null, servers: {}, hasInstructionTools: false, hasCommandTools: false, toolFile: null });

/**
 * The tool servers (MCP) set up for the model in Settings, as Harness client rows. Their tools appear to the agent as
 * `mcp__<server>__<tool>`. A command or URL goes into the patch as text; a secret (an env value or a header) never does: the row
 * carries `!!js process.env.<NAME>` and the value travels in the agent process's environment (returned in `env`).
 */
export function mcpRows(servers, env, skipped = []) {
  const rows = [];
  const ids = new Set();
  let index = 0;
  for (const [name, server] of Object.entries(servers ?? {})) {
    if (!server || server.disabled === true) continue;       // switched off on purpose: not an error
    // A server that cannot be wired is REPORTED (T6): the model never sees its tools, and a silent drop looked like "the tool does not work".
    if (!validMcpName(name)) { skipped.push({ name, why: 'name' }); continue; }
    if (!server.command && !server.url) { skipped.push({ name, why: 'address' }); continue; }
    if (ids.has(`mcp-${slug(name)}`)) { skipped.push({ name, why: 'duplicate' }); continue; }
    ids.add(`mcp-${slug(name)}`);
    const token = (kind, key, value) => {
      const variable = `STRATA_MCP_${index}_${kind}_${slug(key).replace(/-/g, '_').toUpperCase()}`;
      env[variable] = String(value);
      return `@@JS:process.env.${variable}@@`;
    };
    const config = { serverName: name, ...(server.command
      ? { transport: 'stdio', command: String(server.command), args: Array.isArray(server.args) ? server.args.map(String) : [], ...(server.cwd ? { cwd: String(server.cwd) } : {}),
        ...(server.env && Object.keys(server.env).length ? { env: Object.fromEntries(Object.entries(server.env).map(([k, v]) => [k, token('E', k, v)])) } : {}) }
      : { transport: 'streamable-http', url: String(server.url), ...(server.headers && Object.keys(server.headers).length ? { headers: Object.fromEntries(Object.entries(server.headers).map(([k, v]) => [k, token('H', k, v)])) } : {}) }) };
    rows.push({ id: `mcp-${slug(name)}`, name: '@deepseek-ai/dsh-mcp-client', config });
    index += 1;
  }
  return rows;
}

/**
 * What the chat says when a tool server was left out (one notice per reason). The sentences are interface copy: they are in every language's
 * catalog (src/locales/tools-<lang>.json), and the {0} is the list of names exactly as the person wrote them.
 */
const SKIPPED_TEXT = {
  name: 'These tool servers were left out because their names are not allowed: {0}. A name can have up to 32 letters, numbers, hyphens or underscores.',
  address: 'These tool servers were left out because they have no program or web address: {0}.',
  duplicate: 'These tool servers were left out because another one has almost the same name: {0}.',
};
export function mcpSkippedNotices(skipped) {
  const out = [];
  for (const why of Object.keys(SKIPPED_TEXT)) {
    const names = (skipped ?? []).filter(s => s.why === why).map(s => String(s.name).slice(0, 40));
    if (names.length) out.push({ code: `mcp-skipped-${why}-${names.join(',')}`.slice(0, 160), message: SKIPPED_TEXT[why].replace('{0}', names.join(', ')) });
  }
  return out;
}

const slug = value => String(value).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 24) || 'x';

/** Identity of one route's wire configuration, secret included by hash only. */
function fingerprint(provider, secret) {
  return createHash('sha256').update(JSON.stringify(provider)).update('|').update(secret ?? '').digest('hex').slice(0, 16);
}

/**
 * Output budget for an API model, whose real limit the app does not know: a quarter of the window, between 1K and 8K.
 * (A local model has its own, much larger budget: `localOutputBudget`.)
 */
/**
 * Output budget for a LOCAL model: nearly the whole window, so a thinking model is never cut off mid-thought by the app's own
 * limit (it was 8192, which stopped big coding answers at exactly 8192 tokens). The engine clamps the request to what is left
 * of the context (`fit_max_tokens`, which the app turns on for the models it loads), so a long prompt never turns this into an error.
 */
export function localOutputBudget(context) {
  const c = Number.isFinite(context) && context > 0 ? context : 32768;
  // 30 percent of the window, at most 32K (9,830 tokens at 32K, 19,660 at 64K): far more than the old 8192 that cut big answers off, and it leaves
  // the rest for the conversation. The Harness's automatic compaction reads this number as the room the reply reserves: with nearly the whole
  // window reserved it had no room left to measure pressure in and never compacted; with half of it reserved a 32K window compacted every few
  // steps (the system prompt and tool list alone are about 7K tokens), which slowed a long task to a crawl.
  return Math.max(4096, Math.min(32768, Math.floor(c * 0.3)));
}

/**
 * When the Harness's automatic compaction starts condensing old history, for one model: its own default keeps 64K tokens of headroom,
 * which on a local window of 32K to 128K leaves no room at all (it then reports a configuration error once and never compacts). An eighth
 * of what the window has left after the reply's room works from 16K up to the largest windows. Compaction then starts at about 60 percent.
 */
export function compactionHeadroom(context, outputBudget) {
  const room = Math.max(0, (Number.isFinite(context) && context > 0 ? context : 32768) - outputBudget);
  return Math.max(512, Math.min(65536, Math.floor(room * 0.125)));
}

/**
 * The thinking levels the chat's Reasoning menu offers, as wire values for a local Strata model. `off` sends
 * chat_template_kwargs.enable_thinking false; the rest send it true plus reasoning_effort (the server maps high and max to its top level).
 */
export const LOCAL_EFFORTS = Object.freeze({ off: null, low: 'low', medium: 'medium', high: 'high', max: 'high' });
/** The menu's words to the Harness's level names. */
export const EFFORT_LEVEL = Object.freeze({ off: 'off', low: 'low', medium: 'medium', high: 'high', max: 'max' });
/**
 * The thinking levels for a model on OpenRouter: its `reasoning.effort` words (none, low, medium, high, xhigh). A model takes them when its entry in
 * OpenRouter's model list names `reasoning` among its supported parameters; the Harness (pi-ai) writes `reasoning: { effort }` into the request.
 */
export const OPENROUTER_EFFORTS = Object.freeze({ off: 'none', low: 'low', medium: 'medium', high: 'high', max: 'xhigh' });
/** Whether a connection is OpenRouter (the catalog entry, or its address), whatever address it is reached at. */
export const isOpenRouter = conn => conn?.provider === 'openrouter' || (() => { try { return /(^|.)openrouter.ai$/i.test(new URL(conn.baseUrl).hostname); } catch { return false; } })();
export const effortOf = value => EFFORT_LEVEL[String(value ?? '').toLowerCase()] ?? null;

/**
 * Output budget for an API model. The connection's own maximum output is used when it is known (`maxOutput`, read from the provider's model
 * list). Otherwise a generous 32K, never more than the window minus room for the prompt (a quarter of a small window at most is the prompt's).
 * It used to be a quarter of the window capped at 8,192, which cut a thinking model off while it was still thinking: `length` and no answer.
 * The budget is the answer's room: a model that thinks keeps what is left of it for the answer, and the Harness sends the window minus the prompt.
 */
export const DEFAULT_API_OUTPUT = 32768;
export function outputBudget(context, maxOutput = null) {
  const c = Number.isFinite(context) && context > 0 ? context : null;
  const window = c === null ? Infinity : Math.max(1024, c - Math.min(4096, Math.floor(c / 4)));
  if (Number.isFinite(maxOutput) && maxOutput > 0) return Math.max(1024, Math.min(Math.floor(maxOutput), window));
  return Math.max(1024, Math.min(DEFAULT_API_OUTPUT, window));
}

/**
 * @param {object} input
 * @param {Array<{id:string,label?:string,baseUrl:string,modelId:string,context?:number,key?:string,vision?:boolean,slots?:number}>} input.local
 *        ready local instances; the first is the default route (`strata-local`)
 * @param {Array<{id:string,name:string,baseUrl:string,protocol:string,apiKey?:string,models:Array<{id:string,name?:string,contextMax?:number|null}>}>} input.connections
 * @returns {{patch:string, env:Record<string,string>, routes:Array, defaultRoute:object|null, signature:string}}
 */
/** What a local model, or an OpenAI-style server on this network, takes for the thinking level (see the comment where a local route is built). */
const THINKING_COMPAT = { supportsStore: false, thinkingFormat: 'chat-template', chatTemplateKwargs: { enable_thinking: { $var: 'thinking.enabled' }, reasoning_effort: { $var: 'thinking.effort', omitWhenOff: true } } };

export function buildProfile({ local = [], connections = [], gate = true, tools = NO_TOOLS_SETTINGS, skillsDir = null, proxyBase = null, searchMock = null, defaultLocalId = null }) {
  const providers = {};
  const env = {};
  const routes = [];
  const mcpEnv = {};
  // Helper agents: at least as many as the engine has request slots (a local model); API models take the Harness's own limit.
  let slots = 0;
  /** Per-model compaction timing (see compactionHeadroom). */
  const compaction = [];
  /** Routes the person's Response settings can reach: a copy of the provider whose address is this host's forwarding endpoint (see llm-shape.mjs). */
  const shapeTargets = {};
  function addShaped(route, def, realBase) {
    if (!proxyBase) return null;
    const keys = {};
    for (const budget of [0, ...BUDGETS]) {
      const key = `${route}~s${budget}`;
      providers[key] = { ...def, displayName: `${def.displayName} (with your response settings)`, baseURL: `${proxyBase}/${key}/v1`, ...(budget ? { headers: { ...(def.headers ?? {}), 'x-strata-budget': String(budget) } } : {}) };
      shapeTargets[key] = realBase;
      for (const m of def.models) { const w = m.contextWindow, out = m.maxTokens; compaction.push({ provider: key, model: m.id, headroomTokens: compactionHeadroom(w, out) }); }
      keys[budget] = key;
    }
    return keys;
  }

  local.forEach((inst, index) => {
    const route = index === 0 ? 'strata-local' : `strata-local-${slug(inst.id)}`;
    const keyEnv = `STRATA_LOCAL_KEY_${index}`;
    // pi-ai's OpenAI-compatible client insists on some key before it sends anything. A worker that
    // does not authenticate gets a fixed placeholder; one that does gets its real key, by reference.
    env[keyEnv] = inst.key || 'strata-local';
    const context = inst.context ?? 32768;
    providers[route] = {
      displayName: inst.label ? `Strata (local) ${inst.label}` : 'Strata (local)',
      api: 'openai-completions',
      baseURL: inst.baseUrl,
      apiKeyEnv: keyEnv,
      // A long prompt read sends only SSE comments, which pi-ai's OpenAI client drops, so its 300 s stream-idle timer would cut the
      // request five minutes in (#3623). 60 minutes is a backstop: serve ends a hung engine and the bridge's watchdog ends a lost turn.
      streamIdleTimeoutMs: 60 * 60_000,
      // The Strata server reads the thinking level from `chat_template_kwargs` (enable_thinking, reasoning_effort) or from a top-level
      // reasoning_effort; a top-level `enable_thinking` (pi-ai's `qwen` format) is ignored, so the model went on thinking with the menu on Off.
      compat: THINKING_COMPAT,
      models: [{
        id: inst.modelId, name: inst.label || inst.modelId, contextWindow: context, maxTokens: localOutputBudget(context),
        input: inst.vision ? ['text', 'image'] : ['text'],
        reasoning: true, reasoningEfforts: LOCAL_EFFORTS,
      }],
    };
    compaction.push({ provider: route, model: inst.modelId, headroomTokens: compactionHeadroom(context, localOutputBudget(context)) });
    slots = Math.max(slots, Number.isFinite(inst.slots) && inst.slots >= 1 ? Math.floor(inst.slots) : 1);
    const shaped = addShaped(route, providers[route], inst.baseUrl);
    routes.push({ route, fp: fingerprint(providers[route], env[keyEnv]), kind: 'local', connectionName: 'this computer', instanceId: inst.id, modelId: inst.modelId, name: inst.label || inst.modelId, contextWindow: context, ...(shaped ? { shaped } : {}) });
  });

  connections.forEach(conn => {
    const protocol = PROTOCOL[conn.protocol];
    if (!protocol) return;
    const route = `api-${slug(conn.id).slice(0, 12)}`;
    const keyEnv = `STRATA_API_KEY_${slug(conn.id).replace(/-/g, '_').toUpperCase().slice(0, 12)}`;
    // A model server on this network that wants no key (llama.cpp, LM Studio, another Strata) still needs some value: the OpenAI-style client
    // refuses to send anything without one, and the person saw "the model rejected the credentials".
    const keyless = !conn.apiKey && protocol === 'openai-completions';
    if (conn.apiKey) env[keyEnv] = conn.apiKey;
    else if (keyless) env[keyEnv] = 'not-needed';
    let baseURL = conn.baseUrl;
    if (protocol === 'anthropic-messages') baseURL = baseURL.replace(/\/v1\/?$/, '');
    // A model server on this computer or network (another Strata PC, llama.cpp, LM Studio) takes the same thinking vocabulary as a local model:
    // without it the Reasoning menu did nothing for a connection to another Strata machine.
    const openRouter = protocol === 'openai-completions' && isOpenRouter(conn);
    const ownServer = protocol === 'openai-completions' && !openRouter && isPrivateHost(baseURL);
    // A model server on this computer or network has a window of its own size like a local model: the reply keeps half of it, so the Harness can compact the conversation
    // before the window is full (with nearly all of it reserved for the reply a long task ended with "the prompt leaves no room to answer").
    const reply = m => (ownServer && Number.isFinite(m.contextMax) && m.contextMax > 0 && !m.maxOutput ? localOutputBudget(m.contextMax) : outputBudget(m.contextMax ?? ASSUMED_API_CONTEXT, m.maxOutput));
    providers[route] = {
      displayName: conn.name,
      api: protocol,
      baseURL,
      ...(conn.apiKey || keyless ? { apiKeyEnv: keyEnv } : {}),
      ...(ownServer ? { compat: THINKING_COMPAT } : openRouter ? { compat: { thinkingFormat: 'openrouter' } } : {}),
      models: conn.models.map(m => ({
        id: m.id, name: m.name || m.id,
        contextWindow: m.contextMax ?? ASSUMED_API_CONTEXT, maxTokens: reply(m),
        // Pictures are on by default (owner rule): only an explicit no (the person's switch, or the server's own model list or /props) leaves them out.
        // (any OpenAI-style API: a cloud model the person marked, or whose model list says it reads pictures, takes image_url the same way)
        input: protocol === 'openai-completions' && m.vision !== false ? ['text', 'image'] : ['text'],
        ...(ownServer ? { reasoning: true, reasoningEfforts: LOCAL_EFFORTS } : openRouter && m.reasoning === true ? { reasoning: true, reasoningEfforts: OPENROUTER_EFFORTS } : {}),
      })),
    };
    const fp = fingerprint(providers[route], conn.apiKey ?? '');
    const shaped = ownServer ? addShaped(route, providers[route], baseURL) : null;
    for (const m of conn.models) { const ctx = m.contextMax ?? ASSUMED_API_CONTEXT; compaction.push({ provider: route, model: m.id, headroomTokens: compactionHeadroom(ctx, reply(m)) }); }
    for (const m of conn.models) routes.push({ route, fp, kind: 'api', connectionId: conn.id, connectionName: conn.name, modelId: m.id, name: m.name || m.id, contextWindow: m.contextMax ?? ASSUMED_API_CONTEXT, ...(shaped ? { shaped } : {}) });
  });

  // The model the chat really runs on is the default instance, not whichever one was loaded first (T5: with two models loaded, the tool servers of the
  // first one were wired to a chat on the second).
  const defaultInstance = local.find(i => i.id === defaultLocalId) ?? local[0] ?? null;
  const defaultRoute = routes.find(r => r.kind === 'local' && r.instanceId === defaultInstance?.id) ?? routes.find(r => r.kind === 'local') ?? null;
  // One Harness, one set of tool servers: the default local model's.
  const mcpSkipped = [];
  const mcp = mcpRows({ ...(defaultInstance?.mcpServers ?? {}), ...(tools.servers ?? {}) }, mcpEnv, mcpSkipped);
  Object.assign(env, mcpEnv);
  // The person's own instruction and command tools come from the app's tool server; a command tool asks in Manual like any connected tool.
  const own = (kind, on) => on && tools.toolFile ? [{ id: `mcp-strata-${kind}`, name: '@deepseek-ai/dsh-mcp-client', config: { serverName: `strata-${kind}`, transport: 'stdio', command: process.execPath, args: [CUSTOM_TOOLS_SCRIPT, kind, tools.toolFile] } }] : [];
  // The browser: Playwright's tool server on the person's own Chrome or Edge, in a separate empty profile (what the Harness's own browser provider does). Every action asks in Manual.
  const browserRows = tools.browserEnabled && tools.browserPath && tools.browserCli
    ? [{ id: 'mcp-browser', name: '@deepseek-ai/dsh-mcp-client', config: { serverName: 'browser', transport: 'stdio', command: process.execPath, args: [tools.browserCli, '--browser', 'chromium', '--isolated', ...(process.env.STRATA_BROWSER_HEADLESS === '1' ? ['--headless'] : []), '--executable-path', tools.browserPath] } }] : [];
  const ownRows = [...own('instructions', tools.hasInstructionTools), ...own('commands', tools.hasCommandTools), ...browserRows];
  // Web search: DuckDuckGo with no key unless the person chose a keyed service AND saved its key (then that one; never a half-set-up one).
  const searchOn = !tools.off.has('search');
  const keyed = ['exa', 'perplexity', 'deepseek'].includes(tools.search.provider) && tools.search.key ? tools.search.provider : null;
  if (searchOn && keyed) env.STRATA_SEARCH_KEY = tools.search.key;
  const webRows = !searchOn ? [] : keyed === 'deepseek'
    ? [{ id: 'web', config: { searchProvider: 'deepseek-official', fetchProvider: 'http' } }, { id: 'web-search-deepseek', config: { apiKeyEnv: 'STRATA_SEARCH_KEY' } }]
    : [{ id: 'web', config: { searchProvider: keyed ?? DDG_PROVIDER_ID, fetchProvider: 'http' } }, { id: 'web-search-deepseek', disabled: true },
      { insert: [{ id: 'strata-web-search', name: SEARCH_PLUGIN, config: { provider: keyed ?? 'duckduckgo', keyEnv: 'STRATA_SEARCH_KEY', ...(searchMock ? { mock: searchMock } : {}) } }] }];
  /** The models a helper agent may be started on: the table the model-choice plugin and the Harness's allow-list are written from. */
  const subagentRoutes = routes.map(r => ({ provider: r.route, model: r.modelId, name: r.name, connection: r.connectionName, local: r.kind === 'local' }));
  const entries = [
    ...(defaultRoute ? [{ id: 'agent-default-model', name: '@deepseek-ai/dsh-agent-default-model', config: { provider: defaultRoute.route, model: defaultRoute.modelId } }] : []),
    { id: 'llm-pi-ai', name: '@deepseek-ai/dsh-llm-pi-ai', config: { providers } },
    // The chat modes. Replacing the table also drops the Harness's own three presets: a chat is always in one of ours.
    { id: 'permission', config: { presets: Object.fromEntries(Object.values(MODE_PRESETS).map(({ key, sandbox, approval, name, description }) => [key, { sandbox, approval, name, description }])), defaultPreset: MODE_PRESETS[DEFAULT_MODE].key } },
    { id: 'subagent', config: { maxActiveSubagents: slots > 0 ? Math.max(slots, MIN_LOCAL_HELPERS) : DEFAULT_API_HELPERS } },
    // Manual asks before a change runs, so the person sees one Approve/Deny card and nothing has happened yet.
    // Its web question names the search service the person chose (a keyed one only when its key is saved), and the person's switch skips the question.
    ...(gate ? [{ insert: [{ id: 'strata-gate', name: GATE_PLUGIN, config: { web: { provider: SEARCH_NAMES[keyed ?? 'builtin'], skipAsk: tools.webWithoutAsking === true } } }] }] : []),
    // Compaction decides from what the model server measured, and its summary requests always fit the window (plugins/strata-context.mjs).
    { insert: [{ id: 'strata-context', name: CONTEXT_PLUGIN, config: {} }] },
    // Helper agents on a model the person names: every model the app can reach is allowed, and the plugin makes the choice forgiving (plugins/strata-models.mjs).
    ...(subagentRoutes.length ? [{ id: 'subagent-model-selection-settings', config: { enabled: true, allowedModels: subagentRoutes.map(r => ({ provider: r.provider, model: r.model })) } }] : []),
    ...(subagentRoutes.length ? [{ insert: [{ id: 'strata-models', name: MODELS_PLUGIN, config: { routes: subagentRoutes } }] }] : []),
    ...(mcp.length || ownRows.length ? [{ insert: [...mcp, ...ownRows] }] : []),
    ...webRows,
    // The reminder tools would be a second scheduler beside the app's Tasks. The tool groups the person switched off in Settings -> Tools are left out.
    { id: 'preset-standard', name: '@deepseek-ai/dsh-agent-preset', config: standardPreset({ off: tools.off, search: searchOn, skillsDir, compaction }) },
    // Strata names its chats itself; the Harness's own title request would be an extra model call on
    // every first prompt (and, for an API route, an unrequested one).
    { id: 'session-title-llm', disabled: true },
    // Office documents are not previewed in the app (the converter is not shipped): do not try to start it.
    { id: 'office-to-pdf', disabled: true },
  ];
  // JSON is valid YAML, and a JSON document cannot be mis-quoted by a model id or a display name.
  // JSON is valid YAML. A tool server's secret is a `!!js process.env.NAME` value, swapped in after stringifying.
  const json = JSON.stringify(entries, null, 2).replace(/"@@JS:([A-Za-z0-9_.]+)@@"/g, '!!js $1');
  const patch = `# Generated by the Strata host. Contains credential NAMES only, never values.\n${json}\n`;
  // The signature decides whether the running agent must be restarted. It covers the secret values
  // through a hash so a changed key is noticed, and the hash is never reversible into the key.
  const signature = createHash('sha256').update(patch).update(JSON.stringify(Object.entries(env).sort())).digest('hex').slice(0, 16);
  // What the agent is offered (tool groups, web search, tool servers, the person's own tools) decides whether a running agent must be
  // restarted for the next message; the routes are compared separately (a changed context size alone does not restart it).
  const toolsSignature = createHash('sha256').update(JSON.stringify([mcp, ownRows, webRows, entries.find(e => e.id === 'preset-standard'), env.STRATA_SEARCH_KEY ?? '', mcpEnv])).digest('hex').slice(0, 16);
  return { patch, env, routes, defaultRoute, signature, toolsSignature, shapeTargets, mcpSkipped };
}

/** Find the route for a chat's model choice. */
export function routeFor(profile, choice) {
  if (!choice) return profile.defaultRoute;
  if (choice.connectionId) return profile.routes.find(r => r.kind === 'api' && r.connectionId === choice.connectionId && r.modelId === choice.modelId) ?? null;
  if (choice.instanceId) return profile.routes.find(r => r.kind === 'local' && r.instanceId === choice.instanceId) ?? null;
  return profile.defaultRoute;
}
