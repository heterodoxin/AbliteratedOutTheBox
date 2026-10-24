/**
 * The agent bridge: Strata chats as Harness sessions.
 *
 * What it owns
 *   - the pinned Harness process (one, lazily started, restarted only when a route it needs changed);
 *   - the stable Strata-chat -> Harness-session map, persisted next to the host's other state;
 *   - one `session/follow` stream per live turn, projected into the NDJSON frames the chat UI renders;
 *   - approvals: the Harness `approval/request` waterfall in, the user's Approve/Deny back out;
 *   - Stop: `session/cancel` for exactly one session, resolved only once the turn has really ended.
 *
 * What it deliberately does not do
 *   - it never falls back to another model: a chat picks a route (local instance or saved API
 *     connection) and a missing route is a typed error;
 *   - it never implements Stop by killing the Harness (that would cancel every other chat);
 *   - it never holds a secret outside the environment of the agent process it starts.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { homedir, tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { GatewayClient } from './gateway-client.mjs';
import { HarnessProcess, resolveHarness, redact, stopStaleAgent } from './harness-runtime.mjs';
import { StatusReporter } from './status.mjs';
import { buildProfile, routeFor, MODE_PRESETS, modeOf, effortOf, localOutputBudget, compactionHeadroom } from './profile.mjs';
import { TurnProjector, parseArguments as parseArgs } from './events.mjs';
import { ChildProjector } from './children.mjs';
import { nearestBudget } from './llm-shape.mjs';
import { DeliverableTracker, ActivityGroups } from './deliverables.mjs';
import { SkillsUsed, skillInfoFrom } from './skills-used.mjs';
import { turnEngineSpeed } from './engine-speed.mjs';
import { TurnGuard, guardNote, GUARD_NOTES } from './turn-guard.mjs';
import { BASE_TOKENS, COUNT_ABOVE_CHARS, FIT_TEXT, countTokens, decideFit, fill, grouped } from './context-fit.mjs';
import { PlanStore, planPrompt, parsePlan, parsePoints, stepObjective, stepLine, firstSentence, stepSummary, planAsPrompt, editedSteps, planView, researchPrompt, researchWritePrompt, RESEARCH_NUDGE, parseSources } from './plan.mjs';

/**
 * A failure the person can read. `message` is a plain sentence with the next step; `details` keeps the technical
 * text for the Details disclosure (experts), already free of secrets.
 */
export class AgentUnavailable extends Error {
  constructor(message, code = 'agent-unavailable', details = null) { super(message); this.name = 'AgentUnavailable'; this.code = code; this.details = details; }
}

/** The plain sentence for a raw agent failure text, whatever produced it. */
export function plainAgentFailure(raw) {
  const text = String(raw ?? '');
  if (/did not print its address|exited during start|could not start|ECONNREFUSED.*127\.0\.0\.1|refused the launch token|socket/i.test(text)) return 'The coding assistant could not start. Try again. If it keeps happening, close and reopen Strata.';
  if (/process stopped|lost the session stream|stream did not open|session stream/i.test(text)) return 'The coding assistant stopped unexpectedly, so this answer is incomplete. Send your message again.';
  if (/did not accept the message/i.test(text)) return 'The coding assistant did not take this message. Send it again in a moment.';
  return text;
}

/**
 * A failure thrown while a turn starts, as `{ message, details, code }` for the chat's error card. The agent's own errors read like
 * `session/create: tool-bash (...): waiting for shell ...` (a plugin chain that never started): never shown raw. The plain sentence
 * goes in `message`; the technical text goes in `details`, which the app shows only behind Details.
 */
export function plainTurnFailure(err) {
  const raw = String(err?.message ?? err ?? '');
  const details = typeof err?.details === 'string' && err.details ? err.details : raw;
  if (err instanceof AgentUnavailable) return { message: err.message, details, code: err.code ?? 'AGENT_ERROR' };
  if (err?.code === 'agent-preset/invalid' || /waiting for (shell|subprocess|ptcRuntime|workflowEngine|sandbox)|failed to import|did not activate/i.test(raw)) {
    return { message: 'The coding assistant is missing a part it needs, so it cannot work on files yet. Close and reopen Strata. If it keeps happening, reinstall Strata. Chatting without tools still works.', details, code: 'AGENT_INCOMPLETE' };
  }
  if (/^[a-z][a-z-]*\/[a-z][a-z-]*: /i.test(raw) || /^(gateway|rpc)/i.test(raw)) {
    return { message: 'The coding assistant could not start this message. Try again. If it keeps happening, close and reopen Strata.', details, code: err?.code ?? 'AGENT_ERROR' };
  }
  return { message: plainAgentFailure(raw), details, code: err?.code ?? 'AGENT_ERROR' };
}

/** The Harness's plugins start one after another: until they are all up, session/create answers "<plugin>: waiting for <service>". */
const STARTING_UP = /waiting for (shell|subprocess|ptcRuntime|workflowEngine|sandbox)/i;

/**
 * Create a Harness session, waiting for its plugins to finish starting. Right after the app opens (or the Harness restarts) the first
 * message can arrive before they are up: that is retried every `retryMs` for up to `readyMs` (`onWait` tells the page "Starting the assistant…")
 * instead of failing the first chat. Any other error is thrown at once; a start that never finishes is a plain AGENT_NOT_READY error
 * with the Harness's text kept for Details. Returns null when `cancelled()` turns true while waiting.
 */
export async function createSessionWhenReady(create, { readyMs = 60_000, retryMs = 1_500, onWait = () => {}, cancelled = () => false, now = Date.now } = {}) {
  const until = now() + readyMs;
  for (;;) {
    try { return await create(); } catch (e) {
      const text = String(e?.message ?? e ?? '');
      if (!STARTING_UP.test(text)) throw e;
      if (now() >= until) throw new AgentUnavailable('The assistant could not start. Try again in a moment.', 'AGENT_NOT_READY', text);
      onWait();
      await new Promise(r => setTimeout(r, retryMs));
      if (cancelled()) return null;
    }
  }
}

/** Attached pictures (data addresses) as the Harness's image prompt parts. */
export function imageParts(images) {
  return (Array.isArray(images) ? images : []).map(i => {
    const m = /^data:(image\/(?:png|jpeg|webp));base64,(.+)$/.exec(String(i?.url ?? ''));
    return m ? { type: 'image', mediaType: m[1], data: m[2], name: String(i.name ?? 'Picture').slice(0, 120) } : null;
  }).filter(Boolean);
}

/** A timer that never keeps the process alive by itself. */
const after = (ms, fn) => { const t = setTimeout(fn, ms); t.unref?.(); return t; };
/** The app's own host folder: never a place an answer's files can be (this file lives in <host>/server/agent). */
const HOST_DIR = fileURLToPath(new URL('../..', import.meta.url));

const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };

/**
 * @param {object} options
 * @param {string} options.dataDir                      the host's data directory (agent state lives in `<dataDir>/agent`)
 * @param {string|null} [options.componentsDir]         where installed components live
 * @param {() => Array} options.localInstances          ready local instances (see profile.mjs)
 * @param {() => Array} options.connections             saved API connections WITH their keys (host-internal)
 * @param {() => (string|null)} [options.defaultLocalId]  the local instance the chat runs on: its tool servers are the agent's (T5)
 * @param {NodeJS.ProcessEnv} [options.env]
 * @param {string} [options.nodePath]                   the Node runtime that runs the Harness
 * @param {(line:string)=>void} [options.log]
 */
export function createAgentBridge({ dataDir, componentsDir = null, bundledDir = null, gate = true, localInstances, connections, tools = () => undefined, defaultLocalId = () => null, skillsDir = () => null, onGoal = () => {}, onPlan = () => {}, onQueue = () => {}, proxyBase = null, searchMock = null, expertSampling = () => null, engineProgress = () => null, engineGraceMs = 30 * 60_000, stallMs = 4 * 60_000, stallToolMs = 20 * 60_000, env = process.env, nodePath = process.execPath, log = () => {}, startTimeoutMs = 120_000, sessionReadyMs = 60_000, sessionRetryMs = 1_500 }) {
  const root = join(dataDir, 'agent');
  const home = join(root, 'home');
  const sessionsFile = join(root, 'sessions.json');
  mkdirSync(root, { recursive: true });

  /**
   * Where the turn's file tracker must never look (deliverables.mjs): the app, the engine and its components, Strata's own data, AppData
   * and temp. `forbid` ones also switch tracking off when the chat's folder is one of them or inside one.
   */
  function trackerFolders() {
    const user = env.USERPROFILE ?? env.HOME ?? homedir();
    const forbid = [HOST_DIR, componentsDir, bundledDir].filter(Boolean);
    const exclude = [dataDir, tmpdir(), env.LOCALAPPDATA, env.APPDATA, env.ProgramData, env.ProgramFiles, env['ProgramFiles(x86)'], env.TEMP, env.TMP,
      join(user, 'AppData'), join(user, '.cache'), join(user, '.config'), join(user, '.local'), join(user, 'Library')].filter(Boolean);
    return { forbid, exclude };
  }

  /** chatId -> { sessionId, cwd, createdAt } */
  let sessionMap = {};
  if (existsSync(sessionsFile)) {
    try { sessionMap = JSON.parse(readFileSync(sessionsFile, 'utf8')) ?? {}; } catch { sessionMap = {}; }
  }
  const persist = () => {
    const temp = `${sessionsFile}.${randomUUID()}.tmp`;
    writeFileSync(temp, JSON.stringify(sessionMap, null, 2));
    renameSync(temp, sessionsFile);
  };

  /** @type {{hp:HarnessProcess, gw:GatewayClient, profile:object, clientId:string|null, generation:number, events:object}|null} */
  let rt = null;
  let generation = 0;
  let lastError = null;
  let starting = null;
  const chats = new Map();         // chatId -> { turn, route, routeGeneration }
  const plans = new PlanStore(join(root, 'plans.json'));
  /** chatId -> messages the person typed while a goal ran: [{ id, text, at, deliverAt }]; delivered between steps or when the goal ends */
  const queues = new Map();
  const queueView = chatId => (queues.get(chatId) ?? []).map(({ id, text, at, deliverAt }) => ({ id, text, at, deliverAt }));
  const pending = new Map();       // `${sessionId}:${callId}` -> { eventId, sessionId, callId, at }
  const idleWaiters = new Set();

  const chat = id => { let c = chats.get(id); if (!c) { c = { id, turn: null, route: null, routeGeneration: -1 }; chats.set(id, c); } return c; };
  const activeTurns = () => [...chats.values()].filter(c => c.turn).length;
  const notifyIdle = () => { if (activeTurns() === 0) { for (const w of idleWaiters) w(); idleWaiters.clear(); } };
  const waitIdle = (signal, ms) => new Promise((resolve, reject) => {
    if (activeTurns() === 0) return resolve();
    const timer = setTimeout(() => { idleWaiters.delete(done); reject(new AgentUnavailable('Another chat is still working and the list of models changed. Try again when it finishes.', 'agent-busy')); }, ms);
    const done = () => { clearTimeout(timer); resolve(); };
    idleWaiters.add(done);
    signal?.addEventListener('abort', () => { clearTimeout(timer); idleWaiters.delete(done); reject(signal.reason ?? new Error('aborted')); }, { once: true });
  });

  /** The agent's profile now, plus where its shaped routes really lead (read by the forwarding endpoint). */
  let shapeTargets = {};
  function buildProfileNow() {
    const profile = buildProfile({ local: localInstances(), connections: connections(), gate, tools: tools(), skillsDir: skillsDir(), proxyBase, searchMock, defaultLocalId: defaultLocalId() });
    shapeTargets = profile.shapeTargets ?? {};
    return profile;
  }
  const llmTarget = key => shapeTargets[key] ?? null;

  function harnessInfo() { return resolveHarness({ env, componentsDir, bundledDir }); }
  /** Where the agent component is (the browser tool server lives inside it), or null. */
  const harnessDir = () => { const h = harnessInfo(); return h.ok ? h.dir : null; };

  /** The capability state the app shows. Never carries a token or a key. */
  function status() {
    const h = harnessInfo();
    if (!h.ok) return { state: 'unavailable', reason: 'The coding assistant isn’t installed on this computer yet. Chat works without it.', detail: h.reason, code: 'component-missing' };
    const profile = buildProfileNow();
    return {
      state: rt?.hp.running ? 'ready' : starting ? 'starting' : lastError ? 'error' : 'idle',
      reason: lastError ? plainAgentFailure(lastError) === lastError ? 'The coding assistant could not start. Try again.' : plainAgentFailure(lastError) : null,
      detail: lastError ?? null,
      harness: { version: h.version },
      runtime: rt?.hp.running ? { pid: rt.hp.pid, generation: rt.generation } : null,
      routes: profile.routes.map(({ route, kind, modelId, name, contextWindow }) => ({ route, kind, modelId, name, contextWindow })),
      // tool servers that could not be wired (a name the rule does not allow, no program or address): the chat says so
      skippedServers: profile.mcpSkipped ?? [],
      activeTurns: activeTurns(),
      pendingApprovals: pending.size,
    };
  }

  async function startRuntime(profile) {
    const h = harnessInfo();
    if (!h.ok) throw new AgentUnavailable('The coding assistant isn’t installed on this computer yet. Chat works without it.', 'component-missing', h.reason);
    mkdirSync(home, { recursive: true });
    const stale = stopStaleAgent(join(root, 'harness.pid'));
    if (stale) log(`agent · stopped a leftover Harness process (pid ${stale}) from an earlier run`);
    const patchFile = join(root, `profile-${profile.signature}.patch.yml`);
    writeFileSync(patchFile, profile.patch);
    const workdir = join(root, 'run');
    const hp = new HarnessProcess({
      harness: h, home, patchFile, cwd: workdir, env: profile.env, nodePath, logFile: join(root, 'harness.log'), pidFile: join(root, 'harness.pid'),
      onExit: exit => {
        log(`agent · the Harness process exited (code ${exit.code})`);
        if (rt?.hp === hp) { lastError = 'The agent process stopped.'; rt = null; }
        for (const c of chats.values()) c.turn?.abortFromRuntime('The agent process stopped before this turn finished.');
      },
    });
    const info = await hp.start({ timeoutMs: startTimeoutMs });
    const gw = new GatewayClient({ baseUrl: info.baseUrl, token: info.token });
    await gw.connect();
    const next = { hp, gw, profile, clientId: null, generation: ++generation, events: null };
    next.events = gw.openStream('$events', {}, {
      onItem: value => onHostEvent(next, value),
      onError: e => log(`agent · event stream: ${redact(e.message, Object.values(profile.env))}`),
    });
    rt = next;
    lastError = null;
    log(`agent · Harness ${h.version ?? ''} ready (pid ${hp.pid}, ${profile.routes.length} route(s))`);
    return next;
  }

  /** The local engine's own speed for this turn's requests (null when it cannot be read: the stream's estimate stays). */
  async function engineSpeedOf(route, projector) {
    const inst = localInstances().find(i => i.id === route.instanceId);
    if (!inst) return null;
    return turnEngineSpeed({ baseUrl: inst.baseUrl, key: inst.key, since: projector.startedAt, attempts: projector.streams.size, timeoutMs: 800 });
  }

  /** Make sure a runtime exists that knows the route this turn needs. */
  async function ensureRuntime(choice, signal) {
    for (;;) {
      if (starting) { await starting.catch(() => {}); continue; }
      const profile = buildProfileNow();
      const route = routeFor(profile, choice);
      if (!route) {
        throw new AgentUnavailable(choice?.connectionId
          ? 'This API model is no longer available. Choose another model in the chat picker.'
          : 'No local model is ready for the agent. Load a model in Models, or choose an API model.', 'model-unavailable');
      }
      const known = rt?.hp.running && rt.profile.toolsSignature === profile.toolsSignature && rt.profile.routes.some(r => r.route === route.route && r.modelId === route.modelId && r.fp === route.fp);
      if (known) return { rt, route };
      starting = (async () => {
        if (rt?.hp.running) {
          await waitIdle(signal, 120_000);
          const old = rt; rt = null;
          old.events?.close(); old.gw.close(); await old.hp.stop();
        }
        try { return await startRuntime(profile); } catch (e) {
          lastError = redact(String(e.message ?? e), Object.values(profile.env));
          throw e instanceof AgentUnavailable ? e : new AgentUnavailable(plainAgentFailure(lastError) === lastError ? 'The coding assistant could not start. Try again. If it keeps happening, close and reopen Strata.' : plainAgentFailure(lastError), 'agent-start-failed', lastError);
        }
      })();
      try { await starting; } finally { starting = null; }
    }
  }

  function onHostEvent(runtime, value) {
    if (value.type === 'ready') { runtime.clientId = value.clientId; return; }
    if (value.type === 'waterfall' && value.event === 'approval/request') {
      const r = value.request ?? {};
      if (!r.callId) return;
      pending.set(`${value.agentId}:${r.callId}`, { eventId: value.eventId, sessionId: value.agentId, callId: r.callId, tool: r.toolName, reason: r.reason ?? null, at: Date.now(), clientId: runtime.clientId });
    } else if (value.type === 'waterfall' && value.event === 'user-questions/request') {
      // The app has no surface for the agent's own questions yet. Declaring that nobody can answer makes
      // the tool return an error the model reads and carries on from, instead of a turn that waits forever.
      void runtime.gw.answerEvent({ clientId: runtime.clientId, eventId: value.eventId, outcome: 'next' }).catch(() => {});
    } else if (value.type === 'cancel') {
      for (const [k, v] of pending) if (v.eventId === value.eventId) pending.delete(k);
    }
  }

  async function sessionFor(c, cwd, runtime, { onWait = () => {} } = {}) {
    let known = sessionMap[c.id];
    // The chat's folder changed: the agent's working folder and sandbox are fixed when its session starts, so the chat gets a new session
    // in the new folder (the earlier conversation is carried into its first message, as for any fresh session).
    if (known && cwd && known.cwd && resolve(known.cwd) !== resolve(cwd)) { delete sessionMap[c.id]; persist(); known = null; }
    if (known) return { sessionId: known.sessionId, fresh: false };
    const created = await createSessionWhenReady(() => runtime.gw.rpc('session/create', { request: { cwd } }), { readyMs: sessionReadyMs, retryMs: sessionRetryMs, onWait, cancelled: () => c.cancelStart });
    if (!created) return null;
    sessionMap[c.id] = { sessionId: created.sessionId, cwd, createdAt: new Date().toISOString() };
    persist();
    return { sessionId: created.sessionId, fresh: true };
  }

  /**
   * Put the session into the preset of the chat's mode, through the Harness's own `/permission` command, and
   * check that it took. A turn never starts in a mode other than the one asked for: if the switch cannot be
   * confirmed the turn fails closed with a typed error instead of running with unknown permissions.
   */
  async function applyMode(runtime, sessionId, mode) {
    const want = MODE_PRESETS[mode].key;
    const read = async () => {
      const r = await runtime.gw.rpc('session/projections', { request: { sessionId, keys: ['permissions'] } });
      return r?.values?.permissions?.currentValue ?? null;
    };
    try {
      if (await read() !== want) {
        const done = await runtime.gw.rpc('commands/execute', { agentId: sessionId, line: `/permission ${want}`, submittedAttachments: [] });
        if (done?.result?.kind !== 'success') throw new Error(done?.result?.text ?? 'the agent did not recognise the command');
        const now = await read();
        if (now !== want) throw new Error(`the session reports "${now}" instead of "${want}"`);
      }
    } catch (e) {
      throw new AgentUnavailable(`Strata could not set this chat to ${MODE_PRESETS[mode].name} mode, so nothing ran. Try again.`, 'mode-unavailable', e.message);
    }
    // Manual leaves the sandbox open and relies on the Strata gate to ask first. Without the gate it must not start.
    if (mode === 'manual' && !runtime.gateChecked) {
      try {
        const probe = await runtime.gw.rpc('commands/execute', { agentId: sessionId, line: '/strata-gate', submittedAttachments: [] });
        if (probe?.result?.kind !== 'success' || !String(probe.result.text ?? '').includes('strata-gate active')) throw new Error('the approval gate is not loaded');
        runtime.gateChecked = true;
      } catch (e) {
        throw new AgentUnavailable('Strata could not make sure the coding assistant asks before changing anything, so nothing ran. Close and reopen Strata, then try again.', 'mode-unavailable', e.message);
      }
    }
  }

  /**
   * Run one agent turn for a Strata chat and stream it as frames.
   * `signal` means the client went away: the turn keeps running (that is not Stop).
   */
  async function runTurn(options) {
    const c = chat(options.chatId);
    if (c.turn || c.starting) throw new AgentUnavailable('This chat is still working. Stop it or wait for it to finish.', 'turn-active');
    // From here until the Harness has the prompt, Stop has no session to cancel; it cancels the start.
    c.starting = true; c.cancelStart = false;
    try { return await runTurnInner(c, options); } finally { c.starting = false; }
  }

  async function runTurnInner(c, { chatId, message, workspacePath, choice = null, mode = null, history = [], onFrame, requestId = `req-${randomUUID()}`, signal, goal = null, thinking = null, thinkingBudget = 0, images = [], planTask = null, planRun = null, researchTopic = null }) {
    let send = onFrame; let attached = true;
    // A research run gathers first; what the assistant writes while gathering is notes, not the answer, so it is not shown.
    let hideDeltas = !!researchTopic;
    signal?.addEventListener('abort', () => { send = () => {}; attached = false; }, { once: true });
    // What the assistant itself wrote in this run (not our own step lines): a plan turn reads its plan from it, a step reads its summary.
    const written = { all: '', step: '' };
    // The turn guard (turn-guard.mjs): counts failing searches, searches, plan rewrites and a nearly full context. Set up once the route is known.
    let guard = null;
    const raw = frame => { try { send(frame); } catch { send = () => {}; attached = false; } };
    // What the turn made (files) and compact activity lines (notes-split-agent8.md): derived from the cards the turn already sends.
    const tracker = new DeliverableTracker(workspacePath, trackerFolders());
    let toolsRan = false;   // a turn that ran no tool wrote nothing, so there is nothing to look for at its end
    try { tracker.snapshot(); } catch { /* a folder that cannot be read has no deliverables */ }
    const groups = new ActivityGroups({ emit: raw });
    const skills = new SkillsUsed({ emit: raw, info: skillInfoFrom(skillsDir()) });
    const emit = frame => {
      if (frame.type === 'delta' && !frame.own) { written.all += frame.text; written.step += frame.text; if (guard) trip(guard.text(frame.text)); if (hideDeltas) return; }
      if (frame.own) { const { own, ...rest } = frame; frame = rest; }
      if (frame.type === 'delta' && frame.text) groups.text();
      if (frame.type === 'agent-tool' && frame.tool) {
        if (['done', 'error'].includes(frame.tool.state) && !['read', 'read_image', 'glob', 'grep', 'web_fetch', 'web_search'].includes(frame.tool.name)) toolsRan = true;
        let made = null;
        if (frame.tool.state === 'done' && (frame.tool.name === 'write' || frame.tool.name === 'edit')) { try { made = tracker.check(frame.tool.args?.file_path ?? frame.tool.args?.path); } catch { made = null; } }
        raw(frame);
        if (made) raw(made);
        if (guard) trip(guard.card(frame.tool));
        groups.card(frame.tool, made);
        skills.card(frame.tool);
        return;
      }
      raw(frame);
    };
    // What the assistant is doing right now, one small frame (status.mjs). Cleared before the final frame of the turn.
    const reporter = new StatusReporter({ emit });

    emit({ type: 'queued', requestId });
    reporter.set({ phase: 'starting', code: 'starting', text: 'Starting…' });
    const { rt: runtime, route } = await ensureRuntime(choice, signal);
    const made = await sessionFor(c, workspacePath, runtime, { onWait: () => reporter.set({ phase: 'starting', code: 'assistant-starting', text: 'Starting the assistant…' }) });
    const { sessionId, fresh } = made ?? {};
    const cancelled = () => { reporter.clear(); emit({ type: 'done', finishReason: 'aborted', cancelled: true, tokensIn: null, tokensOut: null, usageSource: 'unavailable', durationMs: 0, model: route.name, contextMax: route.contextWindow }); return { cancelled: true }; };
    if (c.cancelStart || !made) return cancelled();

    // The model and the chat's thinking level (the Reasoning menu) are sent to the Harness when either changed. A level the route does
    // not have (an API model with its own vocabulary) is dropped and the model's own default stands: never an error for the person.
    const level = effortOf(thinking ?? 'Medium');   // the menu's default is Medium
    // Response settings the Harness cannot send itself (the thinking budget, the Expert sampling) go through this host's forwarding
    // endpoint, on a copy of the route; with every setting at its default the route is used directly, as before.
    const budget = level === 'off' ? 0 : nearestBudget(thinkingBudget);
    const sampling = proxyBase && route.shaped ? expertSampling() : null;
    const provider = route.shaped && (budget || sampling) ? route.shaped[budget] : route.route;
    const wanted = `${provider}/${route.modelId}/${level ?? ''}`;
    if (c.route !== wanted || c.routeGeneration !== runtime.generation) {
      const select = effort => runtime.gw.rpc('session/selectModel', { request: { sessionId, provider, model: route.modelId, ...(effort ? { reasoningEffort: effort } : {}) } });
      try {
        try { await select(level); } catch (e) { if (!level || e.code === 'session/not-found') throw e; await select(null); }
      } catch (e) {
        if (e.code === 'session/not-found') { delete sessionMap[c.id]; persist(); }
        throw new AgentUnavailable(`The coding assistant could not use ${route.name}. Pick the model again and retry.`, 'model-unavailable', e.message);
      }
      c.route = wanted; c.routeGeneration = runtime.generation;
    }

    // The chat's permission mode is read once, here, for this whole turn. A change the user makes while the
    // turn runs applies to the next one, so a running turn never sees its permissions move under it.
    const appliedMode = planTask ? 'plan' : modeOf(mode);
    await applyMode(runtime, sessionId, appliedMode);

    // Does the message fit the model's window? Asked before anything is sent, with the model server's own count (context-fit.mjs). A message
    // that fits once the older messages are a summary is sent after a summary; one that can never fit is not sent, and the person is told
    // plainly and offered the two ways out. Whatever cannot be counted goes through: the Harness recovers from a refusal by itself.
    if (!goal && route.kind === 'local') {
      const fit = await fitCheck({ runtime, route, sessionId, fresh, history, message }).catch(() => null);
      if (fit?.action === 'too-large') {
        reporter.clear();
        const values = [grouped(fit.tokens), grouped(fit.window)];
        const text = fill(FIT_TEXT.tooLarge, values);
        emit({ type: 'error', message: text, key: FIT_TEXT.tooLarge, values, code: 'CONTEXT_MESSAGE_TOO_LARGE', tokens: fit.tokens, window: fit.window });
        return { error: text };
      }
      if (fit?.action === 'summarize') {
        reporter.set({ phase: 'compacting', code: 'compacting', text: FIT_TEXT.summarizing, key: FIT_TEXT.summarizing, values: [] });
        const done = await compactChat(runtime, sessionId);
        reporter.clear();
        if (done.ok) emit(summarizedNote(done.before, done.after, fit.window));
      }
    }

    const projector = new TurnProjector({ policy: MODE_PRESETS[appliedMode].approval, window: route.contextWindow });
    if (!goal) guard = new TurnGuard({ plan: !!planTask, contextMax: route.kind === 'local' ? route.contextWindow : 0 });   // a saved connection's window is often only assumed: never judged by a guess
    projector.onStatus = update => { if (!turn?.parentOpen && turn?.children.size && [...turn.children.values()].some(h => !h.done) && update.phase === 'waiting') return; reporter.set(update); };
    const turn = {
      sessionId, projector, route, requestId, live: false, lastSeq: -1,
      ended: deferred(), follow: null, snapshot: deferred(), finished: false, prompted: false, stopBeforePrompt: false,
      abortFromRuntime: reason => finishTurn(reason), finish: failure => finishTurn(failure),
      // Helper agents (subagents) the Harness started for this chat, and the state of the run around them.
      children: new Map(), subagentCalls: new Map(), parentOpen: false, stopping: false, settleTimer: null, mode: appliedMode,
      plan: null, advancing: false, goalRun: false, lastActivity: Date.now(), heard: Date.now(), engineMark: null, watchdog: null,
      research: researchTopic ? { phase: 'gather', nudged: false, waitingStart: false } : null,
    };
    c.turn = turn;
    /**
     * A turn must always end. If the Harness sends nothing at all for a while (a tool that never returns, a model server that went
     * quiet, a lost event) the turn is stopped and ended with a plain message instead of showing "thinking" forever. A running tool gets
     * a longer allowance than a quiet model (a build can be silent for minutes). Whatever the assistant finished is kept.
     */
    turn.watchdog = setInterval(() => {
      if (turn.finished || turn.stopping) return;
      const toolRunning = [...projector.tools.values()].some(x => x.card?.state === 'running' || x.card?.state === 'proposed') || pending.size > 0 && pendingFor(chatId).length > 0;
      const limit = toolRunning ? stallToolMs : stallMs;
      // A local engine reading a long prompt sends the Harness nothing for minutes (#3623). Its /metrics progress counts as activity,
      // for at most engineGraceMs after the Harness last spoke: progress for another chat cannot keep a lost turn alive forever.
      if (turn.route?.kind === 'local' && Date.now() - turn.heard < engineGraceMs) {
        let mark = null;
        try { mark = engineProgress(turn.route.instanceId); } catch { /* no progress known: the Harness alone decides */ }
        if (mark != null && mark !== turn.engineMark) { turn.engineMark = mark; turn.lastActivity = Date.now(); return; }
      }
      if (Date.now() - turn.lastActivity < limit) return;
      turn.stalledFor = Math.round((Date.now() - turn.lastActivity) / 1000);
      turn.stopping = true;
      void runtime.gw.rpc('session/cancel', { request: { sessionId } }).catch(() => {});
      for (const h of turn.children.values()) if (!h.done) void runtime.gw.rpc('subagents/interruptByParent', { childSessionId: h.id, parentSessionId: sessionId, mode: 'continuable' }).catch(() => {});
      const mins = Math.max(1, Math.round(turn.stalledFor / 60));
      turn.finish(`The assistant stopped answering for ${mins > 1 ? mins + ' minutes' : 'a minute'}, so I ended this turn. Anything it already finished is kept. Send your message again to go on.`);
    }, Math.min(15_000, Math.max(200, Math.floor(stallMs / 4))));
    turn.watchdog.unref?.();
    async function finishTurn(failure) {
      if (turn.finished) return;
      turn.finished = true;
      clearTimeout(turn.settleTimer);
      clearInterval(turn.watchdog);
      turn.follow?.close();
      for (const child of turn.children.values()) child.follow?.close();
      if (turn.stopping && !failure) projector.ended = turn.guard ? { kind: 'completed' } : { kind: 'aborted', reason: { kind: 'user' } };
      reporter.clear();
      const extra = [];
      try { if (toolsRan) for (const d of tracker.sweep()) { emit(d); groups.addFile(d); } } catch { /* never blocks the end of a turn */ }
      groups.end();
      if (researchTopic && !failure && !turn.stopping) { const sources = parseSources(written.all); if (sources.length) extra.push({ type: 'research', topic: researchTopic, sources }); }
      try { extra.push(...settlePlan(failure)); } catch { /* a plan frame must never break the end of a turn */ }
      // The turn guard: a turn the app stopped, or one that ended with no words, always ends with one plain message (a plan card carries it itself).
      let noteReason = turn.guard ?? null;
      if (!noteReason && guard && !failure && !turn.stopping && !turn.goalRun) {
        noteReason = guard.silent({ ended: projector.ended, wrote: projector.emittedAny === true, acted: projector.tools.size > 0, contextTokens: projector.contextTokens });
      }
      if (noteReason && !extra.some(f => f.type === 'plan' && f.event === 'proposed')) { const note = guardNote(noteReason); if (note) extra.push(note); }
      for (const f of extra) emit(f);
      // The answer speed of a turn on this computer is the engine's own (tokens it produced over its decode time, as the Monitor shows it).
      const engine = !failure && route.kind === 'local' ? await engineSpeedOf(route, projector) : null;
      const frames = failure ? [{ type: 'error', message: plainAgentFailure(failure), details: failure, code: 'AGENT_STOPPED' }] : projector.finish({ model: route.name, contextMax: route.contextWindow, engine });
      for (const f of frames.slice(0, -1)) emit(f);
      // Messages typed while the goal ran and not delivered between steps go back to the page now, to be sent as ordinary messages.
      if (turn.goalRun && attached) {
        const left = queueView(chatId);
        if (left.length) { queues.delete(chatId); onQueue(chatId, []); emit({ type: 'queued-release', messages: left.map(({ id, text }) => ({ id, text })) }); }
      }
      for (const f of frames.slice(-1)) emit(f);
      c.turn = null;
      turn.ended.resolve({ reason: projector.ended, failure: failure ?? null });
      notifyIdle();
      // A turn that filled the context, or that the guard stopped, leaves a long history behind: summarise the older part now so the next message has room.
      if (noteReason === 'context' || noteReason === 'searches' || noteReason === 'searches-failed' || noteReason === 'rewrites' || /context window|too long|exceeds/i.test(String(projector.ended?.error?.message ?? ''))) void compactSession(runtime, sessionId);
    }

    emit({ type: 'start', model: route.name, contextMax: route.contextWindow, demo: false, remoteModel: route.kind === 'api', agent: true, mode: appliedMode, sessionId, requestId, skills: [], ...(planTask ? { planning: true } : {}) });

    turn.follow = runtime.gw.openStream('session/follow', { request: { address: { kind: 'session', sessionId }, assistantStream: true } }, {
      onItem: item => {
        if (item.type === 'snapshot') {
          if (turn.lastSeq < 0) turn.lastSeq = item.cursor;
          else for (const r of item.records ?? []) if (r.type === 'event' && r.event.seq > turn.lastSeq) handle({ type: 'event', event: r.event });
          turn.snapshot.resolve();
          return;
        }
        handle(item);
      },
      onError: e => finishTurn(`The agent lost the session stream: ${e.message}`),
    });

    /** The guard says stop: end the turn the way Stop does, but as a normal end with a plain message (finishTurn). */
    function trip(reason) {
      if (!reason || turn.guard || turn.finished || turn.stopping) return;
      turn.guard = reason;
      log(`agent · turn guard: ${reason} (searches ${guard?.searches ?? 0}, failed in a row ${guard?.failedInARow ?? 0}, plan rewrites ${guard?.rewrites ?? 0})`);
      void stop(chatId).catch(() => {});
    }

    function handle(item) {
      if (turn.finished) return;
      turn.lastActivity = Date.now();
      turn.heard = turn.lastActivity;
      if (item.type === 'event') {
        const e = item.event;
        if (e.seq <= turn.lastSeq) return;
        turn.lastSeq = e.seq;
        // The turn is ours from the user message that carries our request id. Until then an older,
        // detached turn on the same session may still be finishing; its events are not ours.
        if (!turn.live) {
          if (e.type === 'user/message' && e.data?.source?.rpcId === requestId) turn.live = true;
          return;
        }
      } else if (!turn.live) return;
      for (const f of projector.consume(item)) emit(f);
      if (guard && projector.contextTokens != null) trip(guard.context(projector.contextTokens));
      if (item.type === 'event') watchParent(item.event);
    }

    // -- helper agents ---------------------------------------------------------------------------------------

    const rowFrame = child => ({ type: 'agent', agent: child.cp.agent() });
    const emitRow = (child, force = false) => {
      const now = Date.now();
      if (!force && now - child.lastRow < 400) { child.rowPending = true; return; }
      child.lastRow = now; child.rowPending = false;
      emit(rowFrame(child));
      const working = [...turn.children.values()].filter(h => !h.done).length;
      if (working && !turn.parentOpen) reporter.set({ phase: 'helpers', code: 'helpers', text: `Helpers are working (${working})…`, key: 'Helpers are working ({0})…', values: [working], args: { n: working } });
    };

    function openChild(d) {
      const id = d.childId;
      if (!id || turn.children.has(id)) return;
      const cp = new ChildProjector({ id, name: d.label || 'Helper' });
      const child = { id, cp, follow: null, noticed: false, done: false, lastRow: 0, rowPending: false, endedAt: 0 };
      turn.children.set(id, child);
      emitRow(child, true);
      const fold = item => {
        if (turn.finished) return;
        turn.lastActivity = turn.heard = Date.now();
        // A helper that finishes before its throttled "working" row went out would show queued -> completed and never "working": send the held row first.
        if (item.type === 'event' && item.event.type === 'turn/end' && child.rowPending) emitRow(child, true);
        const { frames, row } = cp.consume(item);
        for (const f of frames) emit(f);
        if (item.type === 'event' && item.event.type === 'turn/end') { child.done = true; child.endedAt = Date.now(); emitRow(child, true); armSettle(); return; }
        if (item.type === 'event' && child.done && cp.status === 'working') { child.done = false; }
        if (row) emitRow(child);
      };
      child.follow = runtime.gw.openStream('session/follow', {
        request: { address: { kind: 'subagent', parentSessionId: sessionId, childSessionId: id, mode: d.mode ?? 'unknown' }, assistantStream: true },
      }, {
        onItem: item => {
          if (item.type === 'snapshot') { for (const r of item.records ?? []) if (r.type === 'event') fold({ type: 'event', event: r.event }); return; }
          fold(item);
        },
        onError: e => {
          log(`agent · helper stream ended: ${e.message}`);
          if (!child.done) { child.done = true; child.endedAt = Date.now(); cp.status = 'failed'; cp.failure = 'Lost contact with this helper.'; cp.endedAt = Date.now(); emitRow(child, true); armSettle(); }
        },
      });
      // A late reader still gets the final row: flush any row held back by the throttle.
      child.flush = after(500, function tick() { if (!turn.finished && child.rowPending) emitRow(child, true); if (!turn.finished && !child.done) child.flush = after(500, tick); });
    }

    function watchParent(e) {
      const d = e.data ?? {};
      switch (e.type) {
        case 'turn/start':
          turn.parentOpen = true;
          if (turn.research) turn.research.waitingStart = false;
          clearTimeout(turn.settleTimer);
          // After Stop, a helper's report would start a new turn that nobody asked for.
          if (turn.stopping) void runtime.gw.rpc('session/cancel', { request: { sessionId } }).catch(() => {});
          break;
        case 'turn/end':
          turn.parentOpen = false;
          armSettle();
          break;
        // The first turn of a prompt may have started before this follow stream attached (no turn/start seen): any step of the parent is proof that it is
        // still working. Without this a helper that finished while the parent was still answering ended the run early ("The agent turn was interrupted").
        case 'step/start':
          if (!turn.parentOpen) { turn.parentOpen = true; clearTimeout(turn.settleTimer); }
          break;
        case 'tool/call':
          if (d.name === 'subagent' || d.name === 'subagent_fork') turn.subagentCalls.set(d.callId, parseArgs(d.arguments));
          break;
        case 'subagent/catalog':
          openChild(d);
          break;
        case 'tool/result': {
          const call = turn.subagentCalls.get(d.message?.toolCallId);
          const text = (d.message?.content ?? []).map(p => p.text ?? '').join(' ');
          const started = /started subagent ([0-9a-f-]{36})/i.exec(text);
          const child = started ? turn.children.get(started[1]) : null;
          if (call && child) { child.cp.task = String(call.prompt ?? call.description ?? '').replace(/\s+/g, ' ').trim().slice(0, 140); emitRow(child, true); }
          break;
        }
        case 'user/message':
          if (d.source?.kind === 'goal' && Number.isInteger(d.source.round)) { turn.goalRound = d.source.round; void announceGoal(); }
        // falls through
        case 'agent/inbox/spliced': {
          const hit = JSON.stringify(d);
          if (!/Background subagent/.test(hit)) break;
          for (const child of turn.children.values()) if (hit.includes(child.id)) child.noticed = true;
          break;
        }
        default:
          break;
      }
    }

    // -- plans and goal steps ----------------------------------------------------------------------------------

    const planFrame = (event, plan, extra = {}) => ({ type: 'plan', event, plan: planView(plan), ...extra });
    const publishPlan = (event, plan) => { onPlan(chatId, event, planView(plan)); return planFrame(event, plan); };
    const stepFrame = (plan, index) => {
      const st = plan.steps[index];
      return { type: 'plan-step', planId: plan.id, stepId: st.id, index: index + 1, total: plan.steps.length, state: st.state, text: st.text, summary: st.summary ?? null, line: stepLine(index, plan.steps.length, st.text, st.state === 'running' ? 'started' : st.state) };
    };
    const currentStep = plan => plan.steps.findIndex(x => x.state !== 'done' && x.state !== 'skipped');

    /** What the end of the turn means for the plan: a plan turn makes one; a step run records where it stopped. */
    function settlePlan(failure) {
      if (planTask) {
        if (failure || (turn.stopping && !turn.guard)) return [];
        const parsed = parsePlan(written.all, planTask);
        if (!parsed) return [];
        const plan = plans.create(chatId, { title: parsed.title, steps: parsed.steps, request: planTask, ...(turn.guard ? { caution: GUARD_NOTES[turn.guard] } : {}) });
        return [publishPlan('proposed', plan)];
      }
      const run = turn.plan;
      if (!run) return [];
      const plan = plans.get(chatId);
      if (!plan || plan.id !== run.id) return [];
      if (plan.state !== 'running') return [];
      const out = [];
      const at = currentStep(plan);
      plans.update(chatId, p => {
        if (at < 0) { p.state = 'done'; return; }
        // The step that was running goes back to waiting, so Resume runs it again.
        p.steps[at].state = 'pending';
        p.state = 'paused';
      });
      const updated = plans.get(chatId);
      if (at >= 0) out.push(stepFrame(updated, at));
      out.push(publishPlan(updated.state === 'done' ? 'done' : 'paused', updated));
      return out;
    }

    /** The goal the Harness runs for the plan's next waiting step, and the frames that say so. */
    async function startStep(plan, index) {
      plans.update(chatId, p => { p.state = 'running'; p.goalStarted = true; p.steps[index].state = 'running'; });
      const now = plans.get(chatId);
      await runtime.gw.rpc('goals/create', { agentId: sessionId, request: { objective: stepObjective(now, index) } });
      written.step = ''; written.stepAt = Date.now();
      emit(stepFrame(now, index));
      emit(publishPlan('step', now));
      const st = now.steps[index];
      reporter.set({ phase: 'goal', code: 'goal-step', text: stepLine(index, now.steps.length, st.text, 'started'), key: '{0}', values: [st.text], args: { step: index + 1, steps: now.steps.length } });
    }

    /** One goal step finished: say so, hand over queued messages, and begin the next step (or end the plan). */
    async function advancePlan() {
      const run = turn.plan;
      const plan = plans.get(chatId);
      if (!run || !plan || plan.id !== run.id) return false;
      const at = currentStep(plan);
      if (at < 0) return false;
      const cards = [...projector.tools.values()].filter(x => x.calledAt >= (written.stepAt ?? 0) - 50 && x.card?.state === 'done').map(x => x.card);
      const summary = stepSummary(cards, written.step.split(/\n\s*\n/).filter(Boolean).at(-1) ?? written.step);
      plans.update(chatId, p => { p.steps[at].state = 'done'; p.steps[at].summary = summary; });
      let now = plans.get(chatId);
      emit(stepFrame(now, at));
      emit({ type: 'delta', own: true, text: `\n\n${stepLine(at, now.steps.length, now.steps[at].text, 'done')}\n` });
      const next = currentStep(now);
      emit(publishPlan(next < 0 ? 'done' : 'step', next < 0 ? (plans.update(chatId, p => { p.state = 'done'; }), plans.get(chatId)) : now));
      await deliverQueued(at + 1);
      if (turn.stopping || turn.finished) return true;
      if (next < 0) return false;
      // One step per turn (Time-skip): the plan stays open for the next step, which starts in a session of its own so the conversation never grows across steps.
      if (run.once) { plans.update(chatId, p => { p.state = 'approved'; }); emit(publishPlan('step', plans.get(chatId))); return false; }
      await startStep(plans.get(chatId), next);
      void announceGoal();
      return true;
    }

    /** Hand the person's queued messages to the assistant, one at a time, each answered before the next. */
    async function deliverQueued(afterStep) {
      const list = queues.get(chatId) ?? [];
      while (list.length && !turn.stopping && !turn.finished) {
        const item = list.shift();
        if (!list.length) queues.delete(chatId);
        onQueue(chatId, queueView(chatId));
        emit({ type: 'queued-message', state: 'delivered', message: { id: item.id, text: item.text, at: item.at, deliverAt: item.deliverAt }, afterStep, queue: queueView(chatId) });
        try { await runtime.gw.rpc('session/prompt', { request: { requestId: `queued-${randomUUID()}`, sessionId, mode: 'queue', content: [{ type: 'text', text: item.text }] } }); }
        catch { emit({ type: 'queued-release', messages: [{ id: item.id, text: item.text }] }); continue; }
        // Wait for the answer: the parent opens a turn for it and closes it again.
        const t0 = Date.now();
        while (!turn.parentOpen && Date.now() - t0 < 4000 && !turn.finished) await new Promise(r => setTimeout(r, 50));
        while (turn.parentOpen && !turn.finished && !turn.stopping) await new Promise(r => setTimeout(r, 100));
      }
    }

    /**
     * The run is over when the parent is idle and every helper has finished and reported back. A helper's report
     * wakes the parent a moment after the helper ends, so a helper that ended but has not reported yet keeps the run
     * open (for a few seconds at most); after Stop nobody waits for reports.
     */
    function armSettle() {
      clearTimeout(turn.settleTimer);
      if (turn.finished) return;
      turn.settleTimer = after(turn.stopping ? 1200 : turn.children.size ? 600 : 0, async () => {
        if (turn.finished || turn.parentOpen || turn.advancing) return;
        const open = [...turn.children.values()];
        if (open.some(c => !c.done)) return;
        if (!turn.stopping) {
          if (open.some(c => !c.noticed && Date.now() - c.endedAt < 8000)) { armSettle(); return; }
          // An armed goal starts its next round as soon as the agent is idle: the run is not over until the goal stops.
          const current = await goalGet();
          if (turn.finished || turn.parentOpen) return;
          if (current && current.phase === 'active' && current.activation === 'armed') { turn.settleTimer = after(700, armSettle); return; }
          // Research: gather (search and read pages), then write the answer in the fixed shape. A step that did not open pages is sent back once.
          if (turn.research && !turn.stopping) {
            const r = turn.research;
            if (r.waitingStart) { turn.settleTimer = after(300, armSettle); return; }
            const opened = [...projector.tools.values()].filter(x => x.card?.name === 'web_fetch' && x.card.state === 'done').length;
            const prompt = r.phase === 'write' ? null : (opened < 2 && !r.nudged) ? RESEARCH_NUDGE : researchWritePrompt(researchTopic);
            if (prompt) {
              if (prompt !== RESEARCH_NUDGE) { r.phase = 'write'; hideDeltas = false; written.all = ''; reporter.set({ phase: 'waiting', code: 'research-writing', text: 'Writing the answer…', key: 'Writing the answer…', values: [], args: {} }); } else r.nudged = true;
              r.waitingStart = true;
              try { await runtime.gw.rpc('session/prompt', { request: { requestId: `research-${randomUUID()}`, sessionId, mode: 'queue', content: [{ type: 'text', text: prompt }] } }); }
              catch { r.waitingStart = false; hideDeltas = false; finishTurn(null); return; }
              turn.settleTimer = after(500, armSettle);
              return;
            }
          }
          // A plan step's goal is complete: record it, deliver queued messages, start the next step.
          if (turn.plan && current && current.phase === 'complete' && !turn.advancing) {
            turn.advancing = true;
            let more = false;
            try { more = await advancePlan(); } catch (e) { emit({ type: 'delta', own: true, text: `\n\nThe plan stopped: ${plainGoalError(e)}\n` }); }
            turn.advancing = false;
            if (more && !turn.finished && !turn.stopping) { turn.settleTimer = after(700, armSettle); return; }
          }

          if (turn.goalRun && current) { const line = goalEnding(current); if (line) emit({ type: 'delta', text: line }); }
          if (turn.goalRun) await announceGoal(true);
        }
        finishTurn(null);
      });
    }
    const goalGet = () => runtime.gw.rpc('goals/get', { agentId: sessionId }).catch(() => undefined);
    /** The goal's state to the page: a `goal` frame, the SSE event for the bar, and (while a round runs) the status line. */
    async function announceGoal(final = false) {
      const view = await goalGet();
      if (turn.finished && !final) return;
      const goal = goalObject(view, chatId);
      emit({ type: 'goal', goal });
      onGoal(chatId, goal);
      if (!final && goal && goal.phase === 'active' && turn.goalRound) reporter.set({ phase: 'goal', code: 'goal-step', text: 'Working on the goal…', args: { round: turn.goalRound, max: goal.maxRounds } });
    }
    turn.announceGoal = announceGoal;
    turn.announceQueue = frame => emit(frame);

    const snapshotTimer = {};
    await Promise.race([turn.snapshot.promise, new Promise((_, rej) => { snapshotTimer.t = after(20_000, () => rej(new Error('the session stream did not open'))); })])
      .catch(e => { finishTurn(e.message); })
      .finally(() => clearTimeout(snapshotTimer.t));
    if (turn.finished) return { error: true };
    if (turn.stopBeforePrompt || c.cancelStart) { projector.ended = { kind: 'aborted', reason: { kind: 'user' } }; finishTurn(null); return { cancelled: true }; }

    if (goal) {
      // A goal run has no prompt of its own: the Harness's goal driver starts the rounds. Everything from now on is ours.
      turn.live = true; turn.goalRun = true;
      try {
        if (planRun) {
          const plan = plans.get(chatId);
          if (!plan || plan.id !== planRun.planId) throw new AgentUnavailable('That plan is no longer there. Ask for a new one with /plan.', 'plan-gone');
          turn.plan = { id: plan.id, once: !!planRun.once };
          const at = currentStep(plan);
          if (at < 0) throw new AgentUnavailable('Every step of this plan is already done.', 'plan-done');
          if (goal.action === 'resume' && plan.goalStarted) {
            plans.update(chatId, p => { p.state = 'running'; p.steps[at].state = 'running'; });
            await startGoal(runtime, sessionId, goal);
            emit(publishPlan('step', plans.get(chatId)));
          } else await startStep(plan, at);
        } else {
          await startGoal(runtime, sessionId, goal);
        }
      } catch (e) { finishTurn(e instanceof AgentUnavailable ? e.message : plainGoalError(e)); return { error: true }; }
      reporter.set({ phase: 'waiting', code: 'reading-prompt', text: 'Reading your message…' });
      void announceGoal();
    } else {
      let prompt = message;
      if (fresh && history.length) prompt = withEarlierConversation(history, message);
      try {
        await runtime.gw.rpc('session/prompt', { request: { requestId, sessionId, mode: 'queue', content: [{ type: 'text', text: prompt }, ...imageParts(images)] } });
      } catch (e) {
        finishTurn(`The agent did not accept the message: ${e.message}`);
        return { error: true };
      }
    }
    if (!goal) reporter.set({ phase: 'waiting', code: 'reading-prompt', text: 'Reading your message…' });
    turn.prompted = true;
    if (turn.stopBeforePrompt) {
      // Stop landed in the instant between 'prompt sent' and 'prompt accepted'. A cancel before the turn
      // starts arms nothing, so wait for the turn to be ours, then cancel it.
      for (let i = 0; i < 100 && !turn.live && !turn.finished; i++) await new Promise(r => setTimeout(r, 50));
      if (!turn.finished) await runtime.gw.rpc('session/cancel', { request: { sessionId } }).catch(() => {});
    }

    // Wait for the turn, or for the client to leave. Leaving does not stop the turn.
    const gone = new Promise(r => signal?.addEventListener('abort', () => r('detached'), { once: true }));
    const outcome = await Promise.race([turn.ended.promise, gone]);
    if (outcome === 'detached') return { detached: true, sessionId, ended: turn.ended.promise.then(r => ({ ...r, tokensOut: projector.tokensOut })) };
    return { sessionId, reason: outcome.reason, tokensIn: projector.tokensIn, tokensOut: projector.tokensOut };
  }

  // -- /goal -----------------------------------------------------------------------------------------------

  /** Create or resume the chat's goal; the Harness's goal driver then runs the rounds by itself. */
  async function startGoal(runtime, sessionId, goal) {
    if (goal.action === 'create') {
      const request = { objective: goal.objective, ...(goal.maxRounds ? { maxGoalRounds: goal.maxRounds } : {}) };
      await runtime.gw.rpc('goals/create', { agentId: sessionId, request });
      return;
    }
    const current = await runtime.gw.rpc('goals/get', { agentId: sessionId });
    if (!current) throw new AgentUnavailable('There is no goal to continue. Start one with /goal followed by what you want done.', 'goal-none');
    await runtime.gw.rpc('goals/resume', { agentId: sessionId, ref: { id: current.id, revision: current.revision } });
  }

  async function pauseGoal(runtime, sessionId, chatId = null) {
    try {
      const current = await runtime.gw.rpc('goals/get', { agentId: sessionId });
      if (current && current.phase === 'active') {
        await runtime.gw.rpc('goals/pause', { agentId: sessionId, ref: { id: current.id, revision: current.revision } });
        if (chatId) onGoal(chatId, goalObject(await runtime.gw.rpc('goals/get', { agentId: sessionId }).catch(() => undefined)));
      }
    } catch { /* no goal service, or nothing to pause */ }
  }

  /** The goal as the page and the goal bar see it (notes-split-agent3.md), or null. */
  function goalObject(view, chatId = null) {
    if (!view) return null;
    const phase = ['active', 'paused', 'complete', 'blocked'].includes(view.phase) ? view.phase : 'blocked';
    // A goal that is running a plan (or the points of a goal) shows the plan's title and which step it is on.
    const plan = chatId ? plans.get(chatId) : null;
    const running = plan && plan.goalStarted && ['running', 'paused'].includes(plan.state) ? plan : null;
    const at = running ? running.steps.findIndex(x => x.state !== 'done' && x.state !== 'skipped') : -1;
    const step = running && at >= 0 ? { index: at + 1, total: running.steps.length, text: running.steps[at].text } : null;
    return { objective: step ? running.title : String(view.objective ?? ''), ...(step ? { step } : {}), phase, round: view.roundsStarted ?? 0, maxRounds: view.maxGoalRounds ?? null, ...(phase === 'blocked' ? { blocked: plainBlocker(view.blockedReason) } : {}) };
  }

  /** The goal of a chat now, from the Harness (null: none, or the agent is not running). */
  async function goalState(chatId) {
    const known = sessionMap[chatId]?.sessionId;
    if (!known || !rt?.hp.running) return null;
    return goalObject(await rt.gw.rpc('goals/get', { agentId: known }).catch(() => undefined), chatId);
  }

  /** Pause, edit or remove the goal from the goal bar. Instant; resume runs rounds and goes through the chat stream. */
  async function goalControl({ chatId, action, text }) {
    const known = sessionMap[chatId]?.sessionId;
    if (!known || !rt?.hp.running) throw new AgentUnavailable('There is no goal in this chat yet.', 'goal-none');
    const current = await rt.gw.rpc('goals/get', { agentId: known }).catch(() => undefined);
    if (!current) throw new AgentUnavailable('There is no goal in this chat yet.', 'goal-none');
    const ref = { id: current.id, revision: current.revision };
    try {
      if (action === 'pause') { if (current.phase === 'active') await rt.gw.rpc('goals/pause', { agentId: known, ref }); }
      else if (action === 'clear') await rt.gw.rpc('goals/clear', { agentId: known, ref });
      else if (action === 'edit') {
        const objective = String(text ?? '').trim().slice(0, 4000);
        if (!objective) throw new AgentUnavailable('A goal needs a few words about what you want done.', 'goal-text');
        await rt.gw.rpc('goals/edit', { agentId: known, ref, request: { objective } });
      } else throw new AgentUnavailable('That goal action is not known.', 'goal-action');
    } catch (e) { throw e instanceof AgentUnavailable ? e : new AgentUnavailable(plainGoalError(e), 'goal-failed', e.message); }
    if (action === 'clear') { const plan = plans.get(chatId); if (plan && ['running', 'paused'].includes(plan.state)) { plans.update(chatId, p => { p.state = 'stopped'; for (const st of p.steps) if (st.state === 'running') st.state = 'pending'; }); onPlan(chatId, 'stopped', planView(plans.get(chatId))); } }
    const goal = action === 'clear' ? null : goalObject(await rt.gw.rpc('goals/get', { agentId: known }).catch(() => undefined), chatId);
    onGoal(chatId, goal);
    return { goal };
  }

  /** The goal's state in plain words. */
  function describeGoal(view) {
    if (!view) return 'No goal is set. Start one with /goal followed by what you want done.';
    const what = `"${view.objective}"`;
    if (view.phase === 'active') return `Working on the goal ${what}.`;
    if (view.phase === 'paused') return `The goal ${what} is paused. Type /goal resume to go on.`;
    if (view.phase === 'complete') return `The goal ${what} is complete.`;
    return `The goal ${what} stopped: ${plainBlocker(view.blockedReason)}`;
  }

  /**
   * Handle a "/goal ..." line for a chat, as the Harness's own /goal command reads it: nothing = show the goal,
   * clear, pause, resume, edit <text>, or any other text = a new goal. Starting or resuming runs the rounds and
   * streams them like a chat turn (Stop pauses the goal); every other verb answers at once.
   */
  async function goalCommand(options) {
    const { chatId, line, onFrame } = options;
    const text = String(line ?? '').trim();
    const verb = text.split(/\s+/)[0]?.toLowerCase() ?? '';
    const rest = text.slice(verb.length).trim();
    const say = reply => {
      onFrame({ type: 'queued', requestId: `goal-${randomUUID()}` });
      onFrame({ type: 'start', model: '', contextMax: 0, demo: false, remoteModel: false, agent: true, mode: modeOf(options.mode), skills: [] });
      onFrame({ type: 'delta', text: reply });
      onFrame({ type: 'done', finishReason: 'stop', tokensIn: null, tokensOut: null, usageSource: 'unavailable', durationMs: 0 });
      return { goal: true };
    };
    if (tools()?.off?.has('goals')) return say('Goals are turned off. You can turn them back on in Settings, Tools.');
    const known = sessionMap[chatId]?.sessionId;
    const quick = ['', 'clear', 'pause', 'edit', 'complete'].includes(verb) || (verb === 'status');
    let runtime = rt;
    if (quick) {
      if (!known || !runtime?.hp.running) return say(verb === '' || verb === 'status' ? describeGoal(null) : 'There is no goal in this chat yet.');
      const current = await runtime.gw.rpc('goals/get', { agentId: known }).catch(() => undefined);
      const ref = current ? { id: current.id, revision: current.revision } : null;
      try {
        if (verb === '' || verb === 'status') return say(describeGoal(current));
        if (!ref) return say('There is no goal in this chat yet.');
        if (verb === 'clear') { await runtime.gw.rpc('goals/clear', { agentId: known, ref }); return say('The goal is cleared.'); }
        if (verb === 'pause') {
          if (current.phase === 'active') await runtime.gw.rpc('goals/pause', { agentId: known, ref });
          return say(describeGoal(await runtime.gw.rpc('goals/get', { agentId: known })));
        }
        if (verb === 'complete') { await runtime.gw.rpc('goals/complete', { agentId: known, ref }); return say('The goal is marked complete.'); }
        if (verb === 'edit') {
          if (!rest) return say('Type /goal edit followed by the new wording of the goal.');
          await runtime.gw.rpc('goals/edit', { agentId: known, ref, request: { objective: rest } });
          return say(`The goal now reads: "${rest}".`);
        }
      } catch (e) { return say(plainGoalError(e)); }
    }
    if (verb === 'resume') {
      const plan = plans.get(chatId);
      const goalPlan = plan && plan.goalStarted && plan.state === 'paused' ? { planId: plan.id } : null;
      return runTurn({ ...options, goal: { action: 'resume' }, ...(goalPlan ? { planRun: goalPlan } : {}) });
    }
    // A goal whose own words hold numbered points ("1. ... 2. ... 3. ...") runs point by point, each one reported.
    const points = parsePoints(text);
    if (points) {
      const plan = plans.create(chatId, { title: points.title, steps: points.steps, request: text, source: 'goal', state: 'approved' });
      onPlan(chatId, 'approved', planView(plan));
      return runTurn({ ...options, goal: { action: 'create', objective: text }, planRun: { planId: plan.id } });
    }
    // Any other text is a new goal. One unfinished goal at a time (the Harness's rule): say so in plain words.
    return runTurn({ ...options, goal: { action: 'create', objective: text } });
  }

  // -- /plan -----------------------------------------------------------------------------------------------

  /** A short answer in the chat that the page translates (the sentence is in the interface catalog, "values" fill its slots). */
  function sayKey(options, template, values = []) {
    const { onFrame } = options;
    onFrame({ type: 'queued', requestId: 'compact-' + randomUUID() });
    onFrame({ type: 'start', model: '', contextMax: 0, demo: false, remoteModel: false, agent: true, mode: modeOf(options.mode), skills: [] });
    onFrame({ type: 'delta', text: fill(template, values), key: template, values });
    onFrame({ type: 'done', finishReason: 'stop', tokensIn: null, tokensOut: null, usageSource: 'unavailable', durationMs: 0 });
    return { compact: true };
  }

  /** A short answer in the chat (no model): the same frames a turn sends. */
  function sayOnce(options, reply) {
    const { onFrame } = options;
    onFrame({ type: 'queued', requestId: `plan-${randomUUID()}` });
    onFrame({ type: 'start', model: '', contextMax: 0, demo: false, remoteModel: false, agent: true, mode: modeOf(options.mode), skills: [] });
    onFrame({ type: 'delta', text: reply });
    onFrame({ type: 'done', finishReason: 'stop', tokensIn: null, tokensOut: null, usageSource: 'unavailable', durationMs: 0 });
    return { plan: true };
  }

  /**
   * "/plan <what you want>": the assistant looks around (the chat is put in Plan mode for this turn, so nothing can change), writes the
   * plan, and the host sends it as a `plan` frame. Nothing runs until the person approves it.
   */
  async function planCommand(options) {
    const task = String(options.line ?? '').trim();
    if (!task) {
      const open = plans.get(options.chatId);
      return sayOnce(options, open && ['proposed', 'approved'].includes(open.state)
        ? 'A plan is waiting for you above. Approve it, edit it or reject it, or type /plan followed by something else to make a new one.'
        : 'Type /plan followed by what you want done, for example: /plan build a small pagoda page. I will look around and write a plan; nothing changes until you approve it.');
    }
    return runTurn({ ...options, message: planPrompt(task), planTask: task, mode: 'plan' });
  }

  /**
   * "/research <topic>": the assistant searches the web, opens and reads pages, and writes a short structured answer (Summary, Key findings,
   * Sources). The host also sends the sources as a `research` frame before the end of the turn. Needs web search turned on.
   */
  async function researchCommand(options) {
    const topic = String(options.line ?? '').trim();
    if (!topic) return sayOnce(options, 'Type /research followed by what you want to know, for example: /research how do heat pumps compare with gas boilers. I will search the web, read the pages and answer with sources.');
    if (tools()?.off?.has('search')) return sayOnce(options, 'Research needs web search, which is turned off. Turn it on in Settings, Tools (DuckDuckGo needs no key), then try again.');
    return runTurn({ ...options, message: researchPrompt(topic), researchTopic: topic });
  }

  const planGone = () => new AgentUnavailable('There is no plan waiting in this chat. Ask for one with /plan followed by what you want done.', 'plan-gone');
  function planOf(chatId, planId) {
    const plan = plans.get(chatId);
    if (!plan || (planId && plan.id !== planId)) throw planGone();
    return plan;
  }
  const mustBeOpen = plan => { if (plan.state !== 'proposed' && plan.state !== 'approved') throw new AgentUnavailable('This plan has already been used. Ask for a new one with /plan.', 'plan-used'); };

  /** A short note added to every step of the chat's plan (the Time-skip B path: where its work comes from). */
  function setPlanBriefing(chatId, text) { plans.update(chatId, p => { p.briefing = String(text ?? '').slice(0, 600); }); }

  /** The plan of a chat, or null. */
  const planState = chatId => planView(plans.get(chatId));

  /** Approve, edit or reject the plan; instant. Approve also returns the message that carries the plan out as an ordinary turn. */
  function planControl({ chatId, planId, action, steps, title }) {
    const plan = planOf(chatId, planId);
    if (action === 'edit') {
      mustBeOpen(plan);
      const next = editedSteps(steps, plan.steps);
      const t = String(title ?? '').replace(/\s+/g, ' ').trim().slice(0, 80);
      plans.update(chatId, p => { p.steps = next; if (t) p.title = t; p.state = 'proposed'; });
      onPlan(chatId, 'edited', planView(plans.get(chatId)));
      return { plan: planView(plans.get(chatId)) };
    }
    if (action === 'approve') {
      mustBeOpen(plan);
      plans.update(chatId, p => { p.state = 'approved'; });
      const approved = plans.get(chatId);
      onPlan(chatId, 'approved', planView(approved));
      return { plan: planView(approved), prompt: planAsPrompt(approved) };
    }
    if (action === 'reject') {
      mustBeOpen(plan);
      plans.update(chatId, p => { p.state = 'rejected'; });
      onPlan(chatId, 'rejected', planView(plans.get(chatId)));
      return { plan: planView(plans.get(chatId)) };
    }
    throw new AgentUnavailable('That plan action is not known.', 'plan-action');
  }

  /** "Start as goal": each step of the plan becomes a goal step, run one after another and reported. Streams like a chat turn. */
  async function planGoal(options) {
    const { chatId, planId } = options;
    let plan;
    try { plan = planOf(chatId, planId); } catch (e) { return sayOnce(options, e.message); }
    if (modeOf(options.mode) === 'plan') return sayOnce(options, 'Plan mode only looks, so it cannot carry out a plan. Switch this chat to Manual or Bypass, then start it again.');
    if (plan.state === 'paused' && plan.goalStarted) return goalCommand({ ...options, line: 'resume' });
    try { mustBeOpen(plan); } catch (e) { return sayOnce(options, e.message); }
    if (tools()?.off?.has('goals')) return sayOnce(options, 'Goals are turned off. You can turn them back on in Settings, Tools.');
    plans.update(chatId, p => { p.state = 'approved'; });
    onPlan(chatId, 'approved', planView(plans.get(chatId)));
    return runTurn({ ...options, goal: { action: 'create', objective: plan.title }, planRun: { planId: plan.id, once: options.once === true } });
  }

  /** A paused plan becomes open again (a step that did not finish is tried again in a fresh session). */
  function reopenPlan(chatId) { const p = plans.get(chatId); if (p && p.state === 'paused') { plans.update(chatId, x => { x.state = 'approved'; }); onPlan(chatId, 'approved', planView(plans.get(chatId))); } }

  /** Give up on one step with a plain note (a check that cannot be done here) so the rest of the plan can go on. */
  function skipStep(chatId, index, note) { plans.update(chatId, p => { const st = p.steps[index]; if (st) { st.state = 'skipped'; st.summary = String(note ?? 'Skipped.').slice(0, 140); } if (p.state === 'paused') p.state = 'approved'; }); onPlan(chatId, 'step', planView(plans.get(chatId))); }

  /** Forget the chat's agent session (not its plan): the next turn starts a fresh one in the same folder. */
  function resetSession(chatId) {
    if (chats.get(chatId)?.turn || chats.get(chatId)?.starting) throw new AgentUnavailable('This chat is still working.', 'turn-active');
    delete sessionMap[chatId]; persist();
  }

  // -- messages typed while a goal runs ----------------------------------------------------------------------

  /** Queue a message for the running goal. It is delivered between steps, or handed back to the page when the goal ends or pauses. */
  function queueMessage({ chatId, text }) {
    const turn = chats.get(chatId)?.turn;
    if (!turn || !turn.goalRun || turn.stopping || turn.finished) throw new AgentUnavailable('No goal is running in this chat, so there is nothing to queue behind. Send the message as usual.', 'queue-idle');
    const clean = String(text ?? '').trim().slice(0, 8000);
    if (!clean) throw new AgentUnavailable('Type a few words to queue.', 'queue-empty');
    const list = queues.get(chatId) ?? [];
    if (list.length >= 20) throw new AgentUnavailable('Twenty messages are already waiting. Wait for some to be delivered.', 'queue-full');
    const item = { id: randomUUID(), text: clean, at: new Date().toISOString(), deliverAt: turn.plan ? 'step' : 'end' };
    list.push(item); queues.set(chatId, list);
    const message = { id: item.id, text: item.text, at: item.at, deliverAt: item.deliverAt };
    onQueue(chatId, queueView(chatId));
    turn.announceQueue?.({ type: 'queued-message', state: 'queued', message, queue: queueView(chatId) });
    return { message, queue: queueView(chatId) };
  }

  /** Take a queued message back (the x on it). A message already being delivered cannot be taken back. */
  function cancelQueued({ chatId, id }) {
    const list = queues.get(chatId) ?? [];
    const at = list.findIndex(m => m.id === id);
    if (at < 0) throw new AgentUnavailable('That message was already sent or removed.', 'queue-gone');
    const [item] = list.splice(at, 1);
    if (!list.length) queues.delete(chatId);
    onQueue(chatId, queueView(chatId));
    chats.get(chatId)?.turn?.announceQueue?.({ type: 'queued-message', state: 'cancelled', message: { id: item.id, text: item.text, at: item.at, deliverAt: item.deliverAt }, queue: queueView(chatId) });
    return { queue: queueView(chatId) };
  }

  /** Messages waiting for a chat, and (after a goal ended while the page was away) the ones it should now send. */
  function takeQueue(chatId, { release = false } = {}) {
    const turn = chats.get(chatId)?.turn;
    const list = queueView(chatId);
    if (release && !turn && list.length) { queues.delete(chatId); onQueue(chatId, []); return { queue: [], released: list.map(({ id, text }) => ({ id, text })) }; }
    return { queue: list, released: [] };
  }

  /**
   * The person changed the chat's mode while a turn (a goal and its helpers included) is running. The new preset
   * is put on the session at once, the running turn's own bookkeeping follows, and questions already waiting are
   * settled the way the new mode would have settled them: Bypass allows them, Plan rejects them.
   */
  async function setMode(chatId, mode) {
    const want = modeOf(mode);
    const c = chats.get(chatId);
    const sessionId = sessionMap[chatId]?.sessionId;
    if (!rt || !sessionId) return { applied: false, mode: want, reason: 'no session yet' };
    if (c?.turn && c.turn.mode === want) return { applied: true, mode: want, settled: 0 };
    await applyMode(rt, sessionId, want);
    let settled = 0;
    if (c?.turn) {
      c.turn.mode = want;
      c.turn.projector.policy = MODE_PRESETS[want].approval;
      if (want !== 'manual') {
        const helpers = c.turn.children;
        for (const [key, entry] of [...pending.entries()]) {
          if (entry.sessionId !== sessionId && !helpers?.has(entry.sessionId)) continue;
          pending.delete(key);
          try { await rt.gw.answerEvent({ clientId: entry.clientId ?? rt.clientId, eventId: entry.eventId, outcome: want === 'bypass' ? 'allowed-once' : 'rejected' }); settled++; } catch { /* already answered */ }
        }
      }
    }
    return { applied: true, mode: want, settled };
  }

  /** Answer one pending approval. `decision` is `approve` or `deny`. */
  async function approve({ chatId, callId, decision }) {
    const sessionId = sessionMap[chatId]?.sessionId;
    // A helper agent's question carries the helper's session id in front of its call id. Only a helper that belongs to
    // THIS chat's run can be answered from this chat.
    const [maybeChild] = String(callId).split(':');
    const ownHelper = callId.includes(':') && chats.get(chatId)?.turn?.children.has(maybeChild);
    const key = ownHelper ? callId : `${sessionId}:${callId}`;
    // The durable `approval/asked` (which makes the card say "needs approval") and the waterfall frame
    // (which makes it answerable) travel on different streams; a fast click can beat the second one.
    let entry = sessionId ? pending.get(key) : null;
    for (let i = 0; !entry && sessionId && i < 40; i++) {
      await new Promise(r => setTimeout(r, 50));
      entry = pending.get(key);
    }
    if (!entry) throw new AgentUnavailable('That question is no longer waiting for an answer.', 'approval-gone');
    if (!rt?.hp.running) throw new AgentUnavailable('That question expired because the coding assistant restarted. Ask again.', 'approval-gone');
    pending.delete(key);
    try {
      await rt.gw.answerEvent({ clientId: entry.clientId ?? rt.clientId, eventId: entry.eventId, outcome: decision === 'approve' ? 'allowed-once' : 'rejected' });
    } catch (e) {
      throw new AgentUnavailable('Your answer could not be delivered. Ask again.', 'approval-gone', e.message);
    }
    return { ok: true };
  }

  /**
   * Stop one chat's turn. Resolves only once the turn has ended (or the wait times out), because the
   * Harness acknowledges a cancel as admitted, not as done.
   */
  /** What the chat holds now and the model's window, as the context plugin measures them (null when it cannot be read). */
  async function contextOf(runtime, sessionId) {
    try {
      const r = await runtime.gw.rpc('commands/execute', { agentId: sessionId, line: '/strata-context', submittedAttachments: [] });
      if (r?.result?.kind !== 'success') return null;
      const v = JSON.parse(String(r.result.text));
      return Number.isFinite(v.tokens) ? { tokens: Math.round(v.tokens), window: Number(v.window) > 0 ? Math.round(v.window) : 0 } : null;
    } catch { return null; }
  }

  /** The decision of context-fit.mjs for the message about to be sent, or null when the model server cannot count it. */
  async function fitCheck({ runtime, route, sessionId, fresh, history, message }) {
    const inst = localInstances().find(i => i.id === route.instanceId);
    if (!inst || !(route.contextWindow > 0)) return null;
    const carried = fresh && history.length ? withEarlierConversation(history, message) : message;
    if (carried.length < COUNT_ABOVE_CHARS) return null;
    const count = text => countTokens({ baseUrl: inst.baseUrl, key: inst.key, model: inst.modelId, text });
    let tokens = await count(carried);
    if (tokens == null) return null;
    const now = fresh ? null : await contextOf(runtime, sessionId);
    const window = now?.window || route.contextWindow;
    const pressure = now?.tokens ?? 0;
    // Where the Harness starts summarising by itself on this window (profile.mjs: the reply keeps its room, then the headroom).
    const reply = localOutputBudget(window);
    const threshold = Math.floor(Math.min(window * 0.8, window - reply - compactionHeadroom(window, reply)));
    let verdict = decideFit({ window, pressure, messageTokens: tokens, threshold });
    // The earlier conversation carried into a fresh session is what pushed it over: send the message alone instead.
    if (verdict.action === 'too-large' && carried !== message) {
      const alone = await count(message);
      if (alone != null) { tokens = alone; verdict = decideFit({ window, pressure, messageTokens: alone, threshold }); }
    }
    return { ...verdict, tokens, window, pressure };
  }

  /**
   * Summarise the older part of a chat with the Harness's /compact and say what happened. Never throws.
   * ok:true carries the figures before and after; ok:false says why (nothing to summarise, busy, failed).
   */
  async function compactChat(runtime, sessionId) {
    const before = await contextOf(runtime, sessionId);
    let result;
    try { result = (await runtime.gw.rpc('commands/execute', { agentId: sessionId, line: '/compact', submittedAttachments: [] }))?.result; }
    catch (e) { return { ok: false, reason: /not.found|no such/i.test(String(e.message)) ? 'nothing' : 'failed', before, text: String(e.message ?? e) }; }
    const text = String(result?.text ?? '');
    log('agent · compact: ' + (result?.kind ?? 'none') + ' ' + text.slice(0, 140));
    if (result?.kind === 'success' && /^Compacted/i.test(text)) return { ok: true, before, after: await contextOf(runtime, sessionId) };
    if (result?.kind === 'success') return { ok: false, reason: 'nothing', before, text };
    if (/active compaction|not idle/i.test(text)) return { ok: false, reason: 'busy', before, text };
    // The Harness cannot make a summary smaller than what is left when the chat holds little beyond its first prompt: that is nothing to summarise, not a failure.
    const little = before && before.window > 0 && before.tokens - BASE_TOKENS < before.window * 0.25;
    return { ok: false, reason: little && /useful summary|not smaller/i.test(text) ? 'nothing' : 'failed', before, text };
  }

  /** The line in the chat that explains why the context meter fell: how full it was and how full it is. */
  function summarizedNote(before, after, window) {
    const w = after?.window || before?.window || window || 0;
    const pct = v => Math.max(0, Math.min(100, Math.round(100 * v.tokens / w)));
    const known = w > 0 && before && after;
    const template = known ? FIT_TEXT.summarized : FIT_TEXT.summarizedShort;
    const values = known ? [pct(before), pct(after)] : [];
    return { type: 'delta', own: true, text: '\n\n' + fill(template, values), key: template, values, noteReason: 'compacted' };
  }

  /** "/compact": summarise the older messages of this chat now, and say in plain words what happened. */
  async function compactCommand(options) {
    const { chatId, choice, signal } = options;
    const c = chat(chatId);
    const say = (template, values = []) => sayKey(options, template, values);
    if (c.turn || c.starting) return say(FIT_TEXT.compactBusy);
    const known = sessionMap[chatId]?.sessionId;
    if (!known) return say(FIT_TEXT.noChatYet);
    let runtime;
    try { ({ rt: runtime } = await ensureRuntime(choice, signal)); } catch { return say(FIT_TEXT.compactFailed); }
    const done = await compactChat(runtime, known);
    if (done.ok) {
      const note = summarizedNote(done.before, done.after, 0);
      return say(note.key, note.values);
    }
    if (done.reason === 'busy') return say(FIT_TEXT.compactBusy);
    if (done.reason === 'nothing') {
      const w = done.before?.window || 0;
      const pct = w > 0 && done.before ? Math.round(100 * done.before.tokens / w) : 0;
      return say(pct >= 60 ? FIT_TEXT.nothingFull : FIT_TEXT.nothingToSummarize, pct >= 60 ? [pct] : []);
    }
    return say(FIT_TEXT.compactFailed);
  }

  /** Summarise the older part of a session with the Harness's own /compact. Best effort: the next message compacts again by itself when it must. */
  async function compactSession(runtime, sessionId) {
    try {
      const r = await runtime.gw.rpc('commands/execute', { agentId: sessionId, line: '/compact', submittedAttachments: [] });
      log(`agent · compact after a stopped turn: ${r?.result?.kind ?? 'done'} ${String(r?.result?.text ?? '').slice(0, 120)}`);
    } catch (e) { log(`agent · compact after a stopped turn failed: ${e.message}`); }
  }

  async function stop(chatId, { waitMs = 20_000 } = {}) {
    const c = chats.get(chatId);
    const turn = c?.turn;
    if (!turn && c?.starting) { c.cancelStart = true; return { stopped: true, ended: 'cancelled-before-start', ms: 0 }; }
    if (!turn || !rt) return { stopped: false, reason: 'no active turn' };
    const t0 = Date.now();
    if (!turn.prompted) {
      // The Harness has not been given the prompt yet, so there is nothing to cancel there.
      turn.stopBeforePrompt = true;
      const early = await Promise.race([turn.ended.promise, new Promise(r => setTimeout(() => r(null), 100))]);
      if (early) return { stopped: true, ended: early.reason?.kind ?? 'aborted', ms: Date.now() - t0 };
    }
    turn.stopping = true;
    await pauseGoal(rt, turn.sessionId, chatId);
    // Helpers first: a helper still working would otherwise report back and wake the parent again.
    await Promise.allSettled([...turn.children.values()].filter(h => !h.done).map(h => rt.gw.rpc('subagents/interruptByParent', { childSessionId: h.id, parentSessionId: turn.sessionId, mode: 'continuable' })));
    await rt.gw.rpc('session/cancel', { request: { sessionId: turn.sessionId } }).catch(e => { if (turn.parentOpen) throw e; });
    // Nothing is running in the parent (it was waiting for its helpers): the run ends once they are all down.
    if (!turn.parentOpen) { clearTimeout(turn.settleTimer); turn.settleTimer = after(300, () => { if (!turn.finished && !turn.parentOpen && [...turn.children.values()].every(h => h.done)) turn.finish(null); }); }
    let timer; const result = await Promise.race([turn.ended.promise, new Promise(r => { timer = after(waitMs, () => r(null)); })]);
    clearTimeout(timer);
    if (!result) return { stopped: false, pending: true, reason: 'the turn is still stopping' };
    return { stopped: true, ended: result.reason?.kind ?? 'unknown', ms: Date.now() - t0 };
  }

  function pendingFor(chatId) {
    const sessionId = sessionMap[chatId]?.sessionId;
    const helpers = chats.get(chatId)?.turn?.children;
    return [...pending.values()].filter(p => p.sessionId === sessionId || helpers?.has(p.sessionId))
      .map(p => (p.sessionId === sessionId ? p : { ...p, callId: `${p.sessionId}:${p.callId}` })).map(({ callId, tool, reason, at }) => ({ callId, tool, reason, at }));
  }

  /** Every chat that has an approval waiting, for the task board: the chat, its working folder and what is waiting (same items as pendingFor). */
  function pendingAll() {
    if (!pending.size) return [];
    return Object.keys(sessionMap).map(chatId => ({ chatId, folder: sessionMap[chatId]?.cwd ?? null, pending: pendingFor(chatId) })).filter(c => c.pending.length);
  }

  async function close() {
    const r = rt; rt = null;
    if (r) { r.events?.close(); r.gw.close(); await r.hp.stop(); }
  }

  /** Let go of a chat's Harness session mapping (a task starts every run in a session of its own). */
  function forget(chatId) { if (chats.get(chatId)?.turn) return; delete sessionMap[chatId]; chats.delete(chatId); persist(); }

  return { compactCommand, goalState, goalControl, harnessDir, status, runTurn, goalCommand, planCommand, researchCommand, llmTarget, setPlanBriefing, resetSession, reopenPlan, skipStep, planGoal, planControl, planState, queueMessage, cancelQueued, takeQueue, forget, approve, setMode, stop, pendingFor, pendingAll, close, sessionOf: chatId => sessionMap[chatId]?.sessionId ?? null, workingFolder: chatId => sessionMap[chatId]?.cwd ?? null, isRunning: chatId => !!chats.get(chatId)?.turn, _debug: () => ({ rt, sessionMap }) };
}

/** A goal that stopped by itself, in words. */
export function plainBlocker(reason) {
  if (!reason) return 'it could not go on.';
  if (reason.code === 'round-limit') return 'it worked for the most steps it was allowed. Type /goal clear, then start a new goal to go on.';
  return String(reason.message ?? 'it could not go on.');
}

/** What a goal run says when it ends. */
export function goalEnding(view) {
  if (view.phase === 'complete') return '\n\n*The goal is complete.*';
  if (view.phase === 'blocked') return `\n\n*The goal stopped: ${plainBlocker(view.blockedReason)}*`;
  if (view.phase === 'paused') return '\n\n*The goal is paused. Type /goal resume to go on.*';
  return '';
}

/** A goal request the Harness refused, as a sentence. */
export function plainGoalError(e) {
  const message = String(e?.message ?? e);
  if (/already|unfinished|current goal/i.test(message)) return 'This chat already has a goal that is not finished. Type /goal edit to change it, /goal clear to remove it, or /goal to see it.';
  if (/GOAL_NONE|no goal/i.test(message) || e?.code === 'goal-none') return 'There is no goal to continue. Start one with /goal followed by what you want done.';
  if (/objective/i.test(message)) return 'A goal needs a few words about what you want done, for example: /goal make every test pass.';
  return 'The goal could not be changed right now. Try again in a moment.';
}

/** The Harness session of a chat that existed before the agent did knows nothing; carry a bounded summary. */
export function withEarlierConversation(history, message) {
  const rows = history.filter(m => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim()).slice(-12);
  if (!rows.length) return message;
  const text = rows.map(m => `${m.role}: ${m.content.length > 1500 ? `${m.content.slice(0, 1499)}…` : m.content}`).join('\n');
  return `Earlier in this conversation, before you were given tools:\n${text}\n\nCurrent message:\n${message}`;
}
