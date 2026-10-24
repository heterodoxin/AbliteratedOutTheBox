#!/usr/bin/env node
import { engineProgressMark } from './agent/engine-progress.mjs';
import './system-ca.mjs';        // #3722: the operating system's certificates, before any HTTPS request
import { createCloudflare } from './cloudflare.mjs';
import { createConnectorInstaller } from './cloudflared-install.mjs';
import { findRunningConnector } from './cloudflared-find.mjs';
import { openArtifact } from './artifacts.mjs';
import { saveSnapshot } from './snapshot.mjs';
import { resolvePython, shellPrefix } from './python-resolver.mjs';
import { chooseFolder, chooserUnavailable, typedFolder } from './folder-dialog.mjs';
import { createLocalShare } from './local-share.mjs';
import { createTaskService, validateTask } from './tasks.mjs';
import { createAgentTaskRunner } from './agent/task-run.mjs';
import { migrateSetup } from './setup.mjs';
import { skillEdit } from './skill-edits.mjs';
import { materializeSkills, parseSkillFile, skillSlug, writeCreatedSkill } from './skill-files.mjs';
import { systemFonts } from './fonts.mjs';
import { tuningDir } from './platform-paths.mjs';
import { createAgentBridge, plainTurnFailure } from './agent/agent-bridge.mjs';
import { mcpSkippedNotices } from './agent/profile.mjs';
import { createAgentTools } from './agent/tools-settings.mjs';
import { withPlainStatus } from './agent/status.mjs';
import { findBrowser, findPlaywrightMcp } from './agent/browser-probe.mjs';
/**
 * Strata engine host.
 *
 *   node server/index.mjs                 # API only (use with `npm run dev`)
 *   node server/index.mjs --dist dist     # API + the built web app on one port
 *
 * Owns hardware detection, the model library, engine tuning, the serving log,
 * live metrics and the local inference API. No runtime dependencies.
 */
import http from 'node:http';
import { homedir } from 'node:os';
import { cpSync, createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { basename, dirname, extname, isAbsolute, join, normalize, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

import * as hardware from './hardware.mjs';
import * as models from './models.mjs';
import * as engine from './engine.mjs';
import * as monitor from './monitor.mjs';
import * as serving from './serving.mjs';
import * as store from './store.mjs';
import * as supervisor from './supervisor.mjs';
import * as engineInstall from './engine-install.mjs';
import { detectBackend, currentBackend, streamChat, inferenceLine } from './inference.mjs';
import { CONTEXT_TOKENS, VARIANTS, localModels, variant } from './catalog.mjs';
import { browseLocalModels, inspectLocalModel } from './local-models.mjs';
import { EventHub, error, json, readBody, readRaw, uid } from './util.mjs';
import { createUploads } from './chat-uploads.mjs';
import { createNativeDrops } from './native-drop.mjs';
import { createVisionProbe, effectiveVision, readsPicturesByDefault } from './vision-probe.mjs';
import { RequestQueue } from './request-queue.mjs';
import { selectSkills, skillInstructions } from './skill-selection.mjs';
import { createModelProviders } from './model-providers.mjs';
import { createForkTimeskip } from './timeskip-fork.mjs';
import { prepareOversized } from './oversize.mjs';
import { createChatFolders, chatRootOf, openFile, resolveChatFile, previewFile, rawFile, createAttachments, withAttachments } from './chat-files.mjs';
import { createLlmProxy, samplingFromExpert } from './agent/llm-shape.mjs';
import { checkUpdates, checkComponentUpdates } from './updates.mjs';
import { plainNoModel } from './no-model-text.mjs';
import { createAppUi } from './app-ui.mjs';
import { appVersionFrom, appVersionToShow } from './app-version.mjs';
import { createImageModels } from './image-models.mjs';
import { createRuntimeModels } from './runtime-models.mjs';
import { createNativeControl } from './native-control.mjs';
import { nativeUnloadPlan } from './storage-check.mjs';
import { installBundledEngine } from './bundled-engine.mjs';
import { findControl, createControl, componentsDirectory } from './control-runtime.mjs';
import { forbidden as acquisitionForbidden } from './acquisition.mjs';
import { randomBytes, timingSafeEqual } from 'node:crypto';
import { nativeGateway } from './native-gateway.mjs';
import { proxyCompletion } from './inference.mjs';
import { apiPath } from './api-paths.mjs';
import { findFreePort, listenProblem } from './ports.mjs';
import { validateChatImages } from './chat-images.mjs';
import { createTimeSkip, measureFolder, timeskipRevisionMessage, imageTypeOf } from './timeskip.mjs';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const ROOT = resolve(HERE, '..');
const argv = process.argv.slice(2);
const distFlag = argv.indexOf('--dist');
// `STRATA_DIST` exists for carriers that load this module directly rather than starting it as its
// own process (the Electron app does that). Such a carrier cannot rely on `--dist`: when the host
// runs as the Electron main process, `process.argv` is Chromium's switch list and the host's own
// flags are not reliably present. An explicit environment variable is unambiguous, and the CLI
// `--dist` form keeps working unchanged.
const DIST = process.env.STRATA_DIST
  ? resolve(process.env.STRATA_DIST)
  : distFlag >= 0
    ? resolve(argv[distFlag + 1] ?? join(ROOT, 'dist'))
    : join(ROOT, 'dist');
const PORT = Number(process.env.STRATA_PORT ?? 8791);
const HOST = process.env.STRATA_BIND ?? '127.0.0.1';
/* The folder chats work in when nothing else is chosen. It is never the app's own folder (the host is started inside its install
   directory, so `process.cwd()` was "host" in the sidebar and the agent could read and write the app): the person's own
   Documents\Strata\Workspace (the models folder is next to it, not in it), created on first use, unless STRATA_WORKSPACE names one. */
function defaultWorkspace() {
  if (process.env.STRATA_WORKSPACE) return resolve(process.env.STRATA_WORKSPACE);
  const user = process.env.USERPROFILE ?? process.env.HOME ?? homedir();
  for (const base of [join(user, 'Documents'), user]) {
    const dir = join(base, 'Strata', 'Workspace');
    try { mkdirSync(dir, { recursive: true }); return dir; } catch { /* try the next place */ }
  }
  return user;
}
const WORKSPACE = defaultWorkspace();
const WORKSPACE_NAME = process.env.STRATA_WORKSPACE ? basename(WORKSPACE) : 'Strata';
/** True for the app's own install/source folder or anything inside it. */
const insideApp = path => { const rel = relative(ROOT, resolve(path)); return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel)); };

const hub = new EventHub();
const modelProviders = createModelProviders({ directory: store.DATA_DIR, changed: value => hub.send('providers', withPictureAnswers(value)) });
const imageModels = createImageModels({ dataDir: store.DATA_DIR, modelsDir: () => store.MODELS_DIR, real: models.REAL, changed: value => hub.send('image-models', value) });
/* The installed app ships Strata Control (python + strata_control) next to the host; a source checkout has none and keeps its older paths. */
const CONTROL = findControl();
const control = createControl({ control: CONTROL });
// Files a run as administrator made in Strata's own folders (owner Administrators) are given back to the user at every start; best effort, never blocks.
if (control && process.platform === 'win32' && !process.env.STRATA_NO_REPAIR_ACCESS) void control.repairAccess().catch(() => {});
/* Which window shows the page (the app window or, when that cannot run, a browser tab) and what only the app window can do: app-ui.mjs. */
const appUi = createAppUi({ shutdown: () => shutdown() });
/* The app's version has one source: the desktop shell passes it (STRATA_APP_VERSION, from apps/desktop/src-tauri/Cargo.toml). Outside the app it is read from the same file in a checkout, else 'dev'. */
function appVersion() { return appVersionFrom({ env: process.env, here: HERE }); }
const appUpdater = appUi.updater;
/* The engine that ships inside the installer is put in the components folder when this PC has none (or an older one); no network, hashes checked. The model list waits for it. */
const bundledEngine = control && !process.env.STRATA_NO_BUNDLED_ENGINE
  ? installBundledEngine({ bundleDir: process.env.STRATA_ENGINE_BUNDLE_DIR || resolve(HERE, '..', '..', 'engine-bundle'), components: componentsDirectory(), log: line => console.log(line) })
  : Promise.resolve({ action: 'none' });
let bundledEngineSettled = !(control && !process.env.STRATA_NO_BUNDLED_ENGINE);
bundledEngine.then(() => { bundledEngineSettled = true; }, () => { bundledEngineSettled = true; });
/* A build made for another make of graphics card is not unpacked; Control's answer then carries the plain sentence (with the right download) instead of its own. */
let bundledWrongMake = null;
bundledEngine.then(result => { if (result?.action === 'wrong-make') { bundledWrongMake = result.notice; console.log(result.notice); } }, () => {});
if (control) {
  const engineOf = control.engine.bind(control);
  control.engine = async (...args) => {
    const answer = await engineOf(...args);
    return answer && !answer.usable && bundledWrongMake ? { ...answer, notice: bundledWrongMake, wrongMake: true } : answer;
  };
}
const native=createNativeControl({control,ready:bundledEngine,appUpdater,hardwareProbe:()=>(hardware.cachedHardware()?.gpus??[]).map((g,n)=>({index:g.index??n,vram_gb:g.vramGb})),directory:store.DATA_DIR,modelsDirectory:()=>store.MODELS_DIR,nativeDataDirectory:store.nativeDataDirectory,beforeLoad:async()=>{if(models.REAL&&models.library().loaded){if(serving.state().requests.some(r=>runtimeModels.state().instances.some(i=>i.id===r.instanceId)&&["queued","generating"].includes(r.state)))throw new Error("Finish requests before switching engines.");await models.unload();runtimeState();}},requests:()=>serving.state().requests,changed:value=>hub.send('native',value)});
/* The agent: chats become DeepSeek Harness sessions when that component is installed. It is a separate
   pinned component (STRATA_HARNESS_DIR), not part of this host, so its absence is a visible capability
   state and the chat falls back to the direct path with a notice, never silently to another model. */
// What the person chose in Settings -> Tools (tool groups, web search service and key, the browser, their own tools).
const llmProxy = createLlmProxy({ targetOf: key => agent.llmTarget(key), samplingNow: () => samplingFromExpert(store.get().expert) });
const agentTools = createAgentTools({ read: () => store.get().agentTools, write: value => store.update({ agentTools: value }), dataDir: store.DATA_DIR, browser: () => ({ ...findBrowser(), cli: findPlaywrightMcp(agent.harnessDir()) }) });
agentTools.ensureFile();
const visionProbe = createVisionProbe();
/** The model APIs as the chat picker sees them: each model says whether it reads pictures (what the person ticked, else what the server answered). Unanswered models are asked in the background. */
function withPictureAnswers(list) {
  const secrets = new Map(modelProviders.withSecrets().map(c => [c.id, c]));
  const connections = (list.connections ?? []).map(c => ({ ...c, models: c.models.map(m => {
    const asked = visionProbe.known(c, m.id);
    if ((m.vision == null || m.reasoning == null) && asked === undefined && c.protocol === 'openai-chat' && secrets.has(c.id)) void visionProbe.check(secrets.get(c.id), m.id).catch(() => {});
    // Pictures are on unless the person or the server says no; a model that thinks says so too (OpenRouter's supported_parameters).
    return { ...m, reads: readsPicturesByDefault(m, asked), thinks: m.reasoning ?? visionProbe.knownThinks(c, m.id) ?? null };
  }) }));
  return { ...list, connections };
}
/** Why a picture cannot go to this API model, in plain words and with the way to fix it; null when it can. */
async function pictureRefusal(connection, model) {
  if (!connection || !model) return 'This API model is no longer available. Choose another model in the chat picker.';
  if (connection.protocol !== 'openai-chat') return `Pictures can only be sent to model APIs that use the OpenAI chat style. "${connection.name}" uses a different style, so the picture stays in the preview. Add the model through an OpenAI-compatible address (a /v1 address) to send pictures.`;
  // Pictures are on by default: only an explicit no (the person's switch, or the server's own list) stops one.
  if (readsPicturesByDefault(model, await visionProbe.check(connection, model.id))) return null;
  const fix = `open Settings → Models, edit "${connection.name}" and turn on "Read pictures with this model"`;
  return `"${model.name}" is set not to read pictures (or its server says it cannot: a llama-server started without --mmproj, or a text-only model). If that is wrong, ${fix}. Otherwise choose a model that reads pictures.`;
}
const agent = createAgentBridge({
  tools: () => agentTools.profileView(),
  skillsDir: () => AGENT_SKILLS,
  onGoal: (chatId, goal) => hub.send('goal', { chatId, goal }),
  // The person's Response settings (thinking budget, Expert sampling) reach the agent's model through this host's own forwarding endpoint.
  proxyBase: `http://127.0.0.1:${PORT}/api/internal/agent-llm`,
  expertSampling: () => samplingFromExpert(store.get().expert),
  // How far a local engine has got (the 2 s /metrics poll), for the agent's stall watchdog: it moves while a long prompt is read.
  engineProgress: instanceId => engineProgressMark(native.readyInstances().find(i => i.id === instanceId)),
  onPlan: (chatId, event, plan) => hub.send('plan', { chatId, event, plan }),
  onQueue: (chatId, queue) => hub.send('queue', { chatId, queue }),
  dataDir: store.DATA_DIR,
  componentsDir: componentsDirectory(),
  // The installer ships the Harness next to the host (<install>/host/server -> <install>/harness).
  bundledDir: resolve(HERE, '..', '..', 'harness'),
  defaultLocalId: () => native.defaultInstanceId(),
  localInstances: () => native.readyInstances().map(i => ({
    id: i.id, label: i.label, baseUrl: `http://127.0.0.1:${i.port}/v1`, modelId: i.apiModelId,
    context: i.metrics?.engine?.max_context ?? i.context ?? undefined, key: i.key, vision: i.vision,
    // The engine's request slots: helper agents never outnumber them.
    slots: i.metrics?.live?.slots?.length || i.parallel || 1,
    // The tool servers set up for this model (Settings, Tools): the agent connects to the same ones.
    mcpServers: i.mcpServers,
  })),
  connections: () => modelProviders.withSecrets().map(c => ({ ...c, models: c.models.map(m => ({ ...m, vision: readsPicturesByDefault(m, visionProbe.known(c, m.id)), reasoning: m.reasoning ?? visionProbe.knownThinks(c, m.id) ?? null })) })),
  log: line => console.log(line),
});
let storageChanging = false;
let modelOperations = 0;
const requestQueue = new RequestQueue(n => monitor.queueWaiting(n));
// The supervisor raises its own event names; surface them on the client stream.
let runtimeModels;
const rawSend = hub.send.bind(hub);
hub.send = (event, payload) => {
  if (event === 'engine-status') rawSend('engine-status', payload);
  else if (event === 'engine-log') rawSend('log', payload);
  else if(event==='models' && runtimeModels){
    runtimeModels.adopt(payload.loaded,{beginner:store.get().beginner,expert:engine.params(),extra:store.get().configExtra??{}});
    rawSend('runtime',runtimeModels.state());rawSend(event,runtimeModels.libraryState(payload));
  } else rawSend(event, payload);
};

/* ------------------------------------------------------------------- boot --- */

const boot = await hardware.detect({ force: true });
models.init(hub);
engine.init(hub);
engineInstall.init(hub);
monitor.init(hub, boot);
serving.init(hub, { port: PORT, host: HOST });
runtimeModels=createRuntimeModels({hardware:()=>hardware.cachedHardware()??boot,library:()=>models.library(),requests:()=>serving.state().requests,real:models.REAL,changed:value=>{
  if(!models.REAL){const primary=value.instances.find(i=>i.id===value.defaultId);if(store.get().loaded!==(primary?.variantId??null))store.update({loaded:primary?.variantId??null});}
  rawSend('runtime',value);rawSend('models',runtimeModels.libraryState(models.library()));
}});
runtimeModels.adopt(models.library().loaded,{beginner:store.get().beginner,expert:engine.params(),extra:store.get().configExtra??{}});
function runtimeState(){
  if(models.REAL)runtimeModels.adopt(models.library().loaded,{beginner:store.get().beginner,expert:engine.params(),extra:store.get().configExtra??{}});
  return runtimeModels.state();
}

const startupModelId = process.env.STRATA_REAL_ENGINE === '1' ? (store.get().loaded ?? store.get().current) : null;
if (startupModelId) {
  monitor.setModel(startupModelId);
  const lastId = startupModelId;
  setTimeout(() => {
    const v = variant(lastId);
    if (!v || !models.isOnDisk(lastId)) return;
    console.log(`autostart    ·  loading ${v.label}`);
    void models
      .load(lastId, {
        hardware: boot,
        config: { beginner: store.get().beginner, expert: engine.params(), serving: store.get().serving },
        onPhase: async (phase, id) => {
          if (phase === 'loading') await monitor.beginLoading(id);
          if (phase === 'ready') {
            await monitor.setModel(id);
            engine.markRunning(boot.accelerator.backend);
            hub.send('engine', await engine.config());
          }
        },
      })
      .catch((err) => console.error('autostart failed:', err.message));
  }, 700);
}
const backend = await detectBackend({ force: true });

console.log(`strata host  ·  http://${HOST}:${PORT}`);
console.log(`hardware     ·  ${boot.accelerator.label} / ${boot.accelerator.backend} / ${boot.cpu.cores} cores / ${Math.round(boot.memory.totalGb)} GB`);
const printInference = engine => console.log(`inference    ·  ${models.REAL ? inferenceLine({ backend, engine, settled: bundledEngineSettled }) : 'simulated (design iteration) — set STRATA_REAL_ENGINE=1 for a real engine'}`);
if (models.REAL && control && backend.kind === 'unavailable' && bundledEngineSettled) void control.engine().catch(() => null).then(printInference);
else printInference(null);
console.log(`models dir   ·  ${store.MODELS_DIR}`);
console.log(`workspace    ·  ${WORKSPACE}`);

/* ------------------------------------------------------------------- data --- */

function setupState() {
  const s = store.get();
  return {
    complete: !!s.setup.complete,
    dismissed: !!s.setup.dismissed,
    choices: s.setup.choices ?? {},
    step: s.setup.step ?? 1,
    flowVersion: 2,
    completedAt: s.setup.completedAt,
    history: s.setup.history ?? [],
    app: { version: appVersion(), modelsDir: store.MODELS_DIR, dataDir: store.DATA_DIR, modelsDirNotice: store.MODELS_DIR_NOTICE },
  };
}

/** Skills written in the Skills page live here (one folder each, SKILL.md inside). */
const CREATED_SKILLS = join(store.DATA_DIR, 'skills');
/** What the agent scans: the whole current list, written out by syncAgentSkills (the Harness watches it, so no restart). */
const AGENT_SKILLS = join(store.DATA_DIR, 'agent', 'skills-live');

function skills() {
  const bundled = [
    {
      id: 'commit-messages',
      body: `Write the commit message for the staged change.

Subject line: conventional commits, imperative mood, at most 72 characters.
Body: one paragraph explaining why the change was made, not what changed —
the diff already says what changed. Wrap at 72 columns.

If the diff contains more than one unrelated change, say so instead of
inventing a single narrative for it.`,
      name: 'Commit messages',
      description: 'Writes conventional-commit messages from a diff, with a scope line and a why-paragraph.',
      source: 'bundled',
      enabled: true,
      triggers: ['commit', 'git message', 'stage these changes'],
      sizeKb: 3.2,
    },
    {
      id: 'code-review',
      body: `Review the code that follows and report findings.

Correctness first: wrong results, unhandled cases, races, resource leaks.
Then security. Then clarity. Style last, and only where it hides a bug.

For each finding give the line, what is wrong, and what would fix it. If the
code is fine, say that plainly rather than inventing minor notes to seem
useful. Never restate what the code does.`,
      name: 'Code review',
      description: 'Reviews a file or diff for correctness first, style second, and reports findings with line numbers.',
      source: 'bundled',
      enabled: true,
      triggers: ['review this', 'find bugs', 'check my code'],
      sizeKb: 5.6,
    },
    {
      id: 'release-notes',
      body: `Turn the merged commits into release notes a user would understand.

Group by Feature, Fix, and Breaking change. Write for someone who uses the
product, not someone who wrote it: no commit hashes, no internal file names,
no "refactored X".

Omit anything with no user-visible effect. If nothing user-visible shipped,
say so in one line rather than padding.`,
      name: 'Release notes',
      description: 'Turns merged commits into user-facing release notes grouped by feature, fix and breaking change.',
      source: 'bundled',
      enabled: false,
      triggers: ['release notes', 'changelog', 'what shipped'],
      sizeKb: 2.8,
    },
    {
      id: 'hardware-report',
      body: `Explain what this machine can run, in plain language.

Lead with the verdict: which model classes will run comfortably, which will be
tight, which will not fit. Then the evidence — CPU, accelerator, memory — in
that order of importance to the reader.

Avoid jargon a non-technical person would not know: "unified memory", not
"UMA"; "graphics memory", not "VRAM". Give sizes in GB and say what they mean
for the choice being made.`,
      name: 'Hardware report',
      description: 'Explains what this machine can run, in plain language, from the detected CPU, GPU and memory.',
      source: 'bundled',
      enabled: true,
      triggers: ['can this run', 'my hardware', 'how fast'],
      sizeKb: 2.1,
    },
  ];

  // Two places hold skills a person brought or wrote: the app's own folder (written in the Skills page) and the workspace's skills folder (added
  // with "Add skill"). Both are plain `<name>/SKILL.md` folders; the frontmatter (name, description) is optional.
  const extra = [];
  for (const [dir, source, prefix] of [[CREATED_SKILLS, 'created', 'my-'], [join(WORKSPACE, 'skills'), 'workspace', 'ws-']]) {
    if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
    for (const name of readdirSync(dir)) {
      const full = join(dir, name);
      try { if (!statSync(full).isDirectory()) continue; } catch { continue; }
      let text = '', sizeKb = 1;
      const md = join(full, 'SKILL.md');
      try { text = readFileSync(md, 'utf8'); sizeKb = Math.max(0.1, statSync(md).size / 1024); } catch { /* a folder with no SKILL.md is listed but has nothing to show */ }
      const parsed = parseSkillFile(text);
      extra.push({
        id: `${prefix}${name}`,
        name: parsed.title ?? name.replace(/-/g, ' ').replace(/^./, c => c.toUpperCase()),
        description: parsed.description ?? (source === 'created' ? 'A skill you wrote.' : `Workspace skill loaded from ${name}/SKILL.md`),
        source,
        enabled: true,
        triggers: [name.replace(/-/g, ' ')],
        body: parsed.body,
        sizeKb: Number(sizeKb.toFixed(1)),
      });
    }
  }
  return [...bundled, ...extra].map(s => ({ ...s, ...store.get().skillEdits?.[s.id] }));
}

let skillState = skills().map((s) => ({
  ...s,
  // `enabled` is the standing answer; `onlyIn` narrows it to one workspace.
  // A skill written for one project is noise in another, and a plain on/off
  // switch cannot express "here but not there".
  enabled: store.get().skillPreferences?.[s.id]?.enabled ?? s.enabled,
  onlyIn: store.get().skillPreferences?.[s.id]?.onlyIn ?? null,
  useCount: 0,
  lastUsedAt: null,
}));

/**
 * Write the skills the agent may use into its folder. A skill narrowed to one workspace is left out (the agent has no such scope), a skill
 * switched off is written as hidden. Called after every change to the list.
 */
function syncAgentSkills() {
  try { materializeSkills(AGENT_SKILLS, skillState.filter(s => !s.onlyIn)); } catch (e) { console.warn('skills · could not write the agent skills folder:', e.message); }
}
syncAgentSkills();

/** Whether a skill applies, given where the reader is. */
/**
 * Records which enabled skills a prompt actually triggered.
 *
 * The triggers were decorative: nothing consulted them, so a skill's list was a
 * claim the product could not back up. Matching them here is what makes "used in
 * N conversations" possible rather than invented.
 */
function noteSkillUse(selected) {
  const hit = new Set(selected.map(s => s.id));
  skillState = skillState.map((s) => {
    if (!hit.has(s.id)) return s;
    return { ...s, useCount: (s.useCount ?? 0) + 1, lastUsedAt: new Date().toISOString() };
  });
  if (hit.size) hub.send('skills', skillState);
}

const execFileAsync = promisify(execFile);

/* ---------------------------------------------------------------- prompt ---
   What Strata puts in front of every conversation. Counted so the context
   breakdown can show where the window actually goes rather than reporting a
   single opaque total. */
const SYSTEM_PROMPT = `You are Strata, a local inference assistant running entirely on this computer.
Answer plainly and briefly. Use only tools actually offered to you. Ask for file content when no file tool is available.
Say when you do not know. Never claim to have done something you have not done.`;

/**
 * Files a tool proposal could plausibly name.
 *
 * Scanned from the workspace so the demo proposes something that exists — it
 * used to name `README.md` unconditionally, including on machines with none.
 */
function readableCandidates(root = WORKSPACE, depth = 3) {
  const out = [];
  const walk = (dir, at) => {
    if (at > depth || out.length > 60) return;
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      if (out.length > 60) return;
      if (e.name.startsWith('.') || e.name === 'node_modules') continue;
      const full = join(dir, e.name);
      const rel = full.slice(root.length + 1);
      if (e.isDirectory()) walk(full, at + 1);
      else if (/\.(md|txt|json|ts|tsx|mjs|js|css)$/.test(e.name)) out.push(rel);
    }
  };
  walk(root, 0);
  return out;
}

/* ------------------------------------------------------------- workspaces ---
   The host used to expose exactly one root, taken from the environment. A
   sidebar that offers "add a workspace" over a single-root host would be a lie,
   so the roots are now a small persisted registry. The original root is always
   first and cannot be removed, which keeps existing sessions and file paths
   valid. */
const WORKSPACES_FILE = join(store.DATA_DIR, 'workspaces.json');

function loadWorkspaces() {
  try {
    const parsed = JSON.parse(readFileSync(WORKSPACES_FILE, 'utf8'));
    if (Array.isArray(parsed) && parsed.length) {
      // An earlier version saved the app's own folder as the first workspace. The first entry keeps its id (chats point at it) and moves to the default folder.
      if (parsed[0]?.id === 'root' && typeof parsed[0].path === 'string' && insideApp(parsed[0].path)) parsed[0] = { ...parsed[0], aliases: [...new Set([...(parsed[0].aliases ?? []), parsed[0].path])], path: WORKSPACE, name: WORKSPACE_NAME };
      return parsed;
    }
  } catch {
    /* first run */
  }
  return [{ id: 'root', name: WORKSPACE_NAME, path: WORKSPACE }];
}

let workspaces = loadWorkspaces();
/* The folder each chat works in ("Select a folder"): the agent's working folder and sandbox for that chat. */
const chatAttachments = createAttachments();
const nativeDrops = createNativeDrops();
const uploads = createUploads({ dir: join(store.DATA_DIR, 'uploads'), folderOf: chatId => chatRoot(chatId), attachments: chatAttachments });
try { uploads.sweep(); } catch { /* best effort */ }
const chatFolders = createChatFolders({ file: join(store.DATA_DIR, 'chat-folders.json') });
/** Where a chat's files are: its chosen folder, else the folder its agent worked in, else the default workspace (chat-files.mjs). One source for every file route. */
const chatRoot = chatId => chatRootOf({ chosen: chatFolders.get(chatId), used: agent.workingFolder?.(chatId), fallback: workspaces[0]?.path ?? WORKSPACE });

function saveWorkspaces() {
  try {
    mkdirSync(store.DATA_DIR, { recursive: true });
    writeFileSync(WORKSPACES_FILE, JSON.stringify(workspaces, null, 2));
  } catch (err) {
    console.warn('workspaces · could not persist', err.message);
  }
}

const SKIP_DIRS = new Set(['node_modules', '.git', '.npm-cache', '.strata', '.shots', 'dist', '.vite', '__pycache__', '.cache']);

function workspaceListing(root = WORKSPACE) {
  const entries = [];
  const walk = (dir, prefix = '', depth = 0) => {
    if (depth > 2) return;
    let names = [];
    try {
      names = readdirSync(dir);
    } catch {
      return;
    }
    for (const name of names.slice(0, 60)) {
      if (SKIP_DIRS.has(name) || name.startsWith('.DS')) continue;
      const full = join(dir, name);
      let st;
      try {
        st = statSync(full);
      } catch {
        continue;
      }
      const rel = prefix ? `${prefix}/${name}` : name;
      if (st.isDirectory()) {
        entries.push({ path: rel, kind: 'dir' });
        walk(full, rel, depth + 1);
      } else {
        entries.push({ path: rel, kind: 'file', sizeKb: st.size / 1024, updatedAt: st.mtime.toISOString() });
      }
    }
  };
  walk(root);
  entries.sort((a, b) =>
    a.kind === b.kind ? a.path.localeCompare(b.path) : a.kind === 'dir' ? -1 : 1,
  );
  return { path: root, entries };
}

/* ----------------------------------------------------------------- chat io --- */

function clientName(req) {
  const ua = req.headers['user-agent'] ?? '';
  const header = req.headers['x-strata-client'];
  if (typeof header === 'string' && header) return header;
  if (/curl/i.test(ua)) return 'curl';
  if (/python|httpx|requests/i.test(ua)) return 'Python client';
  if (/node|undici|axios/i.test(ua)) return 'Node client';
  if (/strata/i.test(ua)) return 'Strata chat';
  return 'Local client';
}

async function runRemoteTurn({ selection, message, history, images = [], onFrame, signal, workspaceId }) {
  const controller = new AbortController();
  const requestSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  let entry;
  const started = Date.now();
  try {
    const { connection, model } = modelProviders.resolve(selection);
    const appliedSkills = selectSkills(skillState, message, workspaceId);
    entry = serving.open({ client: 'Strata chat', endpoint: connection.protocol === 'anthropic' ? '/messages' : connection.protocol === 'openai-responses' ? '/responses' : '/chat/completions', model: `${connection.name} · ${model.name}`, cancel: () => controller.abort(new DOMException('Cancelled by the user', 'AbortError')) });
    serving.mark(entry.id, { state: 'generating' });
    const result = await modelProviders.stream({ selection, message, history, images, signal: requestSignal, systemPrompt: ['You are Strata, a helpful assistant. Answer clearly. Do not claim to have read files or performed actions unless their results are included in the conversation.', skillInstructions(appliedSkills)].filter(Boolean).join('\n\n'), onFrame: frame => {
      if (frame.type === 'start') noteSkillUse(appliedSkills);
      if (frame.type === 'done') serving.mark(entry.id, { tokensIn: frame.tokensIn, tokensOut: frame.tokensOut, usageSource: frame.usageSource });
      onFrame({ ...frame, requestId: entry.id, model: model.name, contextMax: model.contextMax, remoteModel: true, ...(frame.type === 'start' ? { skills: appliedSkills.map(s => s.name) } : {}) });
    } });
    serving.mark(entry.id,{tps:result.tps??null});
    serving.finish(entry.id, { tokensOut: result.tokensOut, durationMs: Date.now() - started });
    return result;
  } catch (err) {
    const refusedPicture = images.length && /HTTP [45]\d\d/.test(String(err.message)) ? ' If this model does not read pictures, turn off "Read pictures with this model" in Settings → Models (edit this connection), or choose a model that does.' : '';
    const message = requestSignal.aborted ? 'Request cancelled. Partial output is preserved.' : err.name === 'TimeoutError' ? 'The model API timed out. Try again.' : err.message + refusedPicture;
    if (entry) serving.finish(entry.id, { durationMs: Date.now() - started, error: message });
    onFrame({ type: 'error', message, requestId: entry?.id });
    return { error: message };
  }
}

/** One agent turn for a chat, streamed to the caller as the NDJSON frames the chat already renders. */
async function runAgentTurn({ res, controller, chatId, body, message, workspaceId, images = [] }) {
  const write = frame => { if (!res.writableEnded && !res.destroyed) res.write(`${JSON.stringify(frame)}\n`); };
  const cwd = chatFolders.get(chatId) ?? (workspaceId ? workspaces.find(w => w.id === workspaceId)?.path : null) ?? WORKSPACE;
  let choice, instance = null;
  if (body.remoteModel) {
    choice = { connectionId: String(body.remoteModel.connectionId), modelId: String(body.remoteModel.modelId) };
    // Does this server read pictures? Asked before the agent is built, so a picture is never sent to one that cannot.
    const own = modelProviders.withSecrets().find(c => c.id === choice.connectionId);
    if (own && own.protocol === 'openai-chat') await visionProbe.check(own, choice.modelId);
  }
  else {
    try { instance = native.resolveInstance(body.nativeModelId ?? native.state().defaultId); }
    catch (e) { write({ type: 'error', message: plainNoModel(e.message) }); return; }
    choice = { instanceId: instance.id };
  }
  const started = Date.now();
  // A turn the engine is serving must be visible to the rest of the host (Serving, unload guards).
  const entry = instance ? serving.open({ client: 'Strata chat', endpoint: '/agent', model: instance.apiModelId, instanceId: instance.id, gpuIds: instance.gpus.map(String), cancel: () => { void agent.stop(chatId); } }) : null;
  if (entry) serving.mark(entry.id, { state: 'generating' });
  const settle = (tokensOut, error) => { if (entry) serving.finish(entry.id, { tokensOut, durationMs: Date.now() - started, ...(error ? { error } : {}) }); };
  try {
    // "/goal ..." is the Harness's goal command (set, show, pause, resume, clear); everything else is a message.
    const attached = chatAttachments.take(chatId);
    if (attached.length && !/^\//.test(message.trim())) message = withAttachments(message, attached);
    const goalLine = /^\/goal(?:\s+([\s\S]*))?$/i.exec(message.trim());
    const planLine = /^\/plan(?:\s+([\s\S]*))?$/i.exec(message.trim());
    const researchLine = /^\/research(?:\s+([\s\S]*))?$/i.exec(message.trim());
    // "/compact" summarises the older messages of this chat now (it used to go to the model as text, which answered it as a question).
    const compactLine = /^\/compact\s*$/i.test(message.trim());
    const turnOptions = { chatId, message, workspacePath: cwd, choice, images, mode: typeof body.mode === 'string' ? body.mode : null, thinking: typeof body.thinking === 'string' ? body.thinking : null, thinkingBudget: Number.isFinite(body.thinkingBudget) ? body.thinkingBudget : 0, history: Array.isArray(body.history) ? body.history : [], onFrame: write, signal: controller.signal };
    const result = typeof body.planId === 'string' && body.message === '/plan start' ? await agent.planGoal({ ...turnOptions, planId: body.planId })
      : compactLine ? await agent.compactCommand(turnOptions)
      : researchLine ? await agent.researchCommand({ ...turnOptions, line: researchLine[1] ?? '' })
      : planLine ? await agent.planCommand({ ...turnOptions, line: planLine[1] ?? '' })
      : goalLine ? await agent.goalCommand({ ...turnOptions, line: goalLine[1] ?? '' }) : await agent.runTurn(turnOptions);
    if (result?.detached) void result.ended.then(r => settle(r?.tokensOut ?? null), () => settle(null));
    else settle(result?.tokensOut ?? null, result?.error ? 'The agent turn failed.' : undefined);
  } catch (err) {
    const plain = plainTurnFailure(err);
    write({ type: 'error', message: plain.message, ...(plain.details ? { details: plain.details } : {}), code: plain.code });
    settle(null, plain.message);
  }
}

async function runTurn({ message, history = [], onFrame, signal, client, workspaceId = null, sessionId = null, remoteModel = null, runtimeModelId = undefined, nativeModelId, images=[], useTools=false, thinking='medium', thinkingBudget=0 }) {
  if (remoteModel) return runRemoteTurn({ selection: remoteModel, message, history, images, onFrame, signal, workspaceId });
  if(native.hasLive()||nativeModelId){
    let instance;try{instance=native.resolveInstance(nativeModelId??native.state().defaultId);}catch(e){const message=plainNoModel(e.message);onFrame({type:'error',message});return {error:message};}const started=Date.now(),cancel=new AbortController();
    const entry=serving.open({client:client??'Strata chat',endpoint:'/chat/completions',model:instance.apiModelId,instanceId:instance.id,gpuIds:instance.gpus.map(String),cancel:()=>cancel.abort()});
    try{serving.mark(entry.id,{state:'generating'});const applied=selectSkills(skillState,message,workspaceId);const result=await proxyCompletion({baseUrl:`http://127.0.0.1:${instance.port}/v1`,modelId:instance.apiModelId,authorization:instance.key,message,history,images,extra:{strata_mcp:useTools===true,...(thinkingBudget>0&&String(thinking).toLowerCase()!=='off'?{reasoning_budget_tokens:thinkingBudget}:{}),reasoning_effort:({off:'none',max:'high'}[String(thinking).toLowerCase()]??String(thinking).toLowerCase())},systemPrompt:[SYSTEM_PROMPT,skillInstructions(applied)].filter(Boolean).join('\n\n'),signal:signal?AbortSignal.any([signal,cancel.signal]):cancel.signal,onFrame:frame=>onFrame({...frame,requestId:entry.id,model:instance.label,contextMax:instance.metrics?.engine?.max_context??instance.context,skills:applied.map(s=>s.name)})});serving.mark(entry.id,{...result,usageSource:result.usageSource});serving.finish(entry.id,{tokensOut:result.tokensOut,durationMs:Date.now()-started});return result;}catch(e){serving.finish(entry.id,{durationMs:Date.now()-started,error:e.message});onFrame({type:'error',message:e.message,requestId:entry.id});return {error:e.message};}
  }
  runtimeState();
  let instance;
  try{instance=runtimeModels.resolve(runtimeModelId,{chatDefault:runtimeModelId===undefined});}
  catch(err){onFrame({type:'error',message:err.message});return {error:err.message};}
  const v = variant(instance.variantId);

  if (!v) {
    onFrame({ type: 'error', message: 'No model is loaded. Pick one in Models, then try again.' });
    return;
  }

  const controller = new AbortController();
  const requestSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  const entry = serving.open({ client: client ?? 'Strata chat', endpoint: '/chat/completions', model: instance.apiModelId, instanceId:instance.id, gpuIds:instance.gpuIds, cancel: () => controller.abort(new DOMException('Cancelled by the user', 'AbortError')) });
  const started = Date.now();
  let result = { tokensIn: null, tokensOut: null };
  // Time to first token: the gap between asking and the first token arriving.
  let firstTokenMs = null;

  try {
    onFrame({ type: 'queued', requestId: entry.id });
    await (models.REAL?requestQueue:runtimeModels.queue(instance.id)).run(requestSignal, async () => {
      requestSignal.throwIfAborted();
      if(!runtimeModels.state().instances.some(i=>i.id===instance.id))throw new Error('This model was unloaded before dispatch.');
      runtimeModels.update(instance.id,{phase:'reading'});
      // Scope is resolved again at dispatch: a disabled skill never stays active
      // merely because the request waited in the queue.
      const appliedSkills = selectSkills(skillState, message, workspaceId);
      const systemPrompt = [SYSTEM_PROMPT, skillInstructions(appliedSkills)].filter(Boolean).join('\n\n');
      serving.mark(entry.id, { state: 'generating' });
      const contextMax = instance.memory.context;
      const modelMonitor=models.REAL?monitor:{
        progressOutput:()=>{},
        beginReading:(total)=>runtimeModels.update(instance.id,{phase:'reading',prompt:{done:0,total,pct:0}}),
        progressReading:(done)=>{const total=runtimeModels.state().instances.find(i=>i.id===instance.id)?.prompt.total??0;runtimeModels.update(instance.id,{prompt:{done,total,pct:total?done/total*100:0}});},
        beginGenerating:()=>runtimeModels.update(instance.id,{phase:'generating'}),
        beginThinking:()=>runtimeModels.update(instance.id,{phase:'thinking'}),
      };
      try {
        result = await streamChat({
          message,
          history,
          systemPrompt,
          modelLabel: v.label,
          candidates: readableCandidates(),
          onFrame: (frame) => {
            if(frame.type==='stats'||frame.type==='done')modelMonitor.progressOutput(frame.tokensOut);
            // Time to first *token*. A `tool` frame is a proposal, not generated
            // output, and it arrives before the prompt is even read — counting it
            // reported a 0ms first token. The stream emits `delta` for both
            // reasoning and content; the real-engine path also reports ttftMs
            // itself on the done frame.
            if (firstTokenMs === null && (frame.type === 'delta' || frame.type === 'reasoning')) {
              firstTokenMs = Date.now() - started;
            }
            if (frame.type === 'done' && typeof frame.ttftMs === 'number') firstTokenMs = frame.ttftMs;
            if (frame.type === 'start') {
              // Demo responses cannot demonstrate that instructions affected a model.
              if (!frame.demo) noteSkillUse(appliedSkills);
              serving.mark(entry.id, { tokensIn: frame.promptTokens ?? null, usageSource: frame.demo ? 'demo' : 'unavailable' });
            }
            if (frame.type === 'done') {
              serving.mark(entry.id, { tokensIn: frame.tokensIn, tokensOut: frame.tokensOut, usageSource: frame.usageSource });
            }
            onFrame({ ...frame, ...(frame.type==='done'?{ttftMs:firstTokenMs}:{}), requestId: entry.id, model: v.label, contextMax, ...(frame.type === 'start' ? { skills: frame.demo ? [] : appliedSkills.map(s => s.name) } : {}) });
          },
          signal: requestSignal,
          engine: {
            variantId: v.id,
            beginReading: (tokens, id) => modelMonitor.beginReading(tokens, id),
            progressReading: (done) => modelMonitor.progressReading(done),
            beginGenerating: () => modelMonitor.beginGenerating(),
            beginThinking: () => modelMonitor.beginThinking(),
          },
        });
        requestSignal.throwIfAborted();
        serving.mark(entry.id,{cachedTokens:result.cachedTokens??null,tps:result.tps??null,promptTps:result.promptTps??null,hitRate:result.hitRate??null,pcieShare:result.pcieShare??null,finishReason:result.finishReason??null});
        serving.finish(entry.id, {
          tokensOut: result.tokensOut,
          durationMs: Date.now() - started,
        });
        runtimeModels.update(instance.id,{phase:'idle',outputTps:result.tps??0,promptTps:result.promptTps??0,context:{usedTokens:(result.tokensIn??0)+(result.tokensOut??0),maxTokens:contextMax,pct:((result.tokensIn??0)+(result.tokensOut??0))/contextMax*100}});
        if(models.REAL)monitor.record({
          tokensIn: result.tokensIn,
          tokensOut: result.tokensOut,
          durationMs: Date.now() - started,
          ttftMs: firstTokenMs ?? undefined,
          cachedTokens: result.cachedTokens,
          tps: result.tps,
          usageSource: result.usageSource,
        });
        if(models.REAL){monitor.setRecordedContext({ ...result, requestId: entry.id, sessionId });monitor.idle();}
      } catch (error) {
        runtimeModels.update(instance.id,{phase:'idle'});
        if(models.REAL)monitor.idle(requestSignal.aborted ? 'Request cancelled' : 'Last request failed');
        throw error;
      }
    });
    return result;
  } catch (err) {
    const message = requestSignal.aborted ? 'Request cancelled. Partial output is preserved.' : err.message;
    serving.finish(entry.id, { durationMs: Date.now() - started, error: message });
    onFrame({ type: 'error', message, requestId: entry.id });
    return { error: message };
  }
}

// `HERE` is apps/host/server, so the repository root is three levels up (server -> host -> apps).
const REPO_ROOT = CONTROL ? CONTROL.dir : resolve(HERE, '..', '..', '..');
// The interpreter that drives the job is the same one the launcher would pick, named explicitly so
// a packaged run uses the runtime the bundle carries rather than whatever `python3` resolves to.
const PYTHON = resolvePython({ repo: REPO_ROOT, control: CONTROL });
const TS_PYTHON = PYTHON.command;
const REVISE_SCRIPT = resolve(REPO_ROOT, 'integrations', 'harness', 'revise_once.py');
// A revision attempt is an agent turn, performed by the Harness through the pinned integration
// script — which is a *Python* program, so it is invoked with the interpreter, not with Node.
// This is a configured path, never a value an HTTP caller supplies: accepting one would hand a
// remote caller command execution on the user's machine.
// The default revision step is the app's own assistant (server/timeskip-revise.mjs, run by Node), not the Harness's Python bridge, which
// needs a Harness build next to the checkout that the installed app does not have.
const REVISE_NODE_SCRIPT = resolve(HERE, 'timeskip-revise.mjs');
const TIMESKIP_REVISE = process.env.STRATA_TIMESKIP_REVISE
  ?? `"${process.execPath}" "${REVISE_NODE_SCRIPT}"`;
/** Proves a revision request comes from a job this host started, not from some other program on this PC. */
const INTERNAL_SECRET = randomBytes(24).toString('hex');
/** The model a trial runs on, by run id: the chat's saved API model when it uses one, else the loaded local model. */
const timeskipChoices = new Map();
/** Trials this host is driving: run id -> { chatId (the chat that asked), turnChat (the assistant's own chat id), controller, step (the assistant's current status), lastSent }. */
const trials = new Map();
const trialByChat = new Map();
/** What the job is doing, in the person's words, from its phase and the assistant's own current step (notes-split-agent3.md). */
function trialActivity(run) {
  const trial = trials.get(run.id);
  switch (run.phase) {
    case 'queued': case 'checkpointing': return { code: 'reading-files', text: 'Reading your files…' };
    case 'generating': return { code: 'trying-version', text: 'Trying an improved version…', ...(trial?.step ? { args: { step: trial.step.text, stepCode: trial.step.code, ...(trial.step.args ? { stepArgs: trial.step.args } : {}) } } : {}) };
    case 'checking': return { code: 'checking-result', text: 'Checking the result…' };
    case 'decision': return { code: 'comparing', text: 'Comparing the two versions…' };
    default: return null;
  }
}
/** Tell the page when the job's phase or the assistant's step changed (SSE `timeskip`), at most every 2 s. */
function watchTrial(id) {
  const trial = trials.get(id);
  if (!trial) return;
  const timer = setInterval(() => {
    const run = timeskip.status(id);
    if (!run) { clearInterval(timer); return; }
    const activity = trialActivity(run);
    const key = JSON.stringify([run.phase, activity]);
    if (key !== trial.lastSent) { trial.lastSent = key; hub.send('timeskip', { id, phase: run.phase, activity }); }
    if (['completed', 'cancelled', 'failed', 'interrupted'].includes(run.phase)) { clearInterval(timer); trials.delete(id); if (trial.chatId) trialByChat.delete(trial.chatId); }
  }, 2000);
  timer.unref?.();
  trial.timer = timer;
}
/** Stop a trial the person stopped: the assistant's turn, the job and its processes, and the copies. */
async function stopTrial(id) {
  const trial = trials.get(id);
  if (trial) { trial.controller?.abort(); if (trial.turnChat) void agent.stop(trial.turnChat).catch(() => {}); }
  const stopped = await timeskip.stop(id);
  if (trial) { clearInterval(trial.timer); trials.delete(id); if (trial.chatId) trialByChat.delete(trial.chatId); }
  if (stopped) hub.send('timeskip', { id, phase: 'cancelled', activity: null });
  return stopped;
}
/**
 * Tuning, from the App Host.
 *
 * The job contract is Strata Control's (`strata_control/tuning.py`) and is driven through
 * `strata tune`. Two things the UI needs and neither is computable here:
 *
 *   * **what a sweep would search, and why the rest was excluded** — that depends on the selected
 *     engine binary's own flag list, so it is asked of the CLI rather than guessed;
 *   * **recorded jobs and their reports** — read from the job's durable `job.json`.
 *
 * There is deliberately no "run a sweep" route yet. A sweep needs a measurement harness that boots
 * the engine with each candidate, and inventing measurements is exactly what the handoff forbids.
 * The plan and the report are real; the sweep is gated on that harness.
 */
function createTuning({ repoRoot, python, pythonArgs = [], env = process.env } = {}) {
  const cli = join(repoRoot, 'strata.py');
  const stateDir = () => tuningDir(env);

  function runCli(args) {
    return new Promise((resolve) => {
      const child = spawn(python, [...pythonArgs, cli, ...args, '--output', 'json'], {
        cwd: repoRoot, windowsHide: true, env: { ...env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', STRATA_TUNE_DIR: stateDir() },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let out = '', err = '';
      child.stdout.on('data', (b) => { out += b.toString(); });
      child.stderr.on('data', (b) => { err += b.toString(); });
      child.on('error', (e) => resolve({ ok: false, error: e.message }));
      child.on('close', (code) => {
        let parsed = null;
        try { parsed = JSON.parse(out.trim().split('\n').filter(Boolean).pop() ?? ''); } catch { parsed = null; }
        if (!parsed) return resolve({ ok: false, code, error: err.trim() || out.trim() || `exit ${code}` });
        resolve({ ok: Boolean(parsed.ok), code, envelope: parsed });
      });
    });
  }

  return {
    stateDir,
    /** The reviewable plan for one backend/engine pair. */
    async plan({ engine, backend, model, objective = 'generation', trials = 16, repeats = 3 }) {
      if (!engine || !backend) return { ok: false, code: 2, error: 'engine and backend are required' };
      const r = await runCli(['tune', 'plan', '--engine', engine, '--backend', backend,
        '--model', model ?? 'unknown', '--objective', objective,
        '--trials', String(Math.max(0, Math.min(512, Number(trials) || 16))),
        '--repeats', String(Math.max(1, Math.min(10, Number(repeats) || 3)))]);
      return r.ok ? { ok: true, plan: r.envelope.plan } : { ok: false, code: r.code, error: r.error };
    },
    /** Recorded jobs, newest state only: the durable record is the authority. */
    jobs() {
      const root = stateDir();
      if (!existsSync(root)) return [];
      const out = [];
      for (const entry of readdirSync(root, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const file = join(root, entry.name, 'job.json');
        if (!existsSync(file)) continue;
        try {
          const data = JSON.parse(readFileSync(file, 'utf8'));
          out.push({
            id: data.id, state: data.state,
            backend: data.plan?.backend ?? null, objective: data.plan?.objective ?? null,
            trials: data.report?.trials_run ?? null,
            recommendation: data.report?.recommendation_available === true,
            reason: data.report?.reason ?? null,
            startedAt: data.started_at ?? null, finishedAt: data.finished_at ?? null,
          });
        } catch {
          // A partially written record is skipped rather than reported as an empty job.
        }
      }
      return out;
    },
    job(id) {
      if (!/^[0-9a-f]{8,64}$/.test(String(id ?? ''))) return null;
      const file = join(stateDir(), id, 'job.json');
      if (!existsSync(file)) return null;
      try { return JSON.parse(readFileSync(file, 'utf8')); } catch { return null; }
    },
  };
}

const tuning = createTuning({ repoRoot: REPO_ROOT, python: TS_PYTHON, pythonArgs: PYTHON.args });

/* Time-skip: plan, lever, A all the way, B forked at the lever (notes-split-agent8.md). The older revise-the-draft job above stays reachable with legacy:true. */
const forkTs = createForkTimeskip({ agent, dir: join(store.DATA_DIR, 'timeskip-fork'), notify: (type, data) => hub.send(type, data) });
const timeskip = createTimeSkip({
  repoRoot: REPO_ROOT,
  python: TS_PYTHON,
  pythonArgs: PYTHON.args,
  env: { ...process.env, STRATA_TIMESKIP_DIR: process.env.STRATA_TIMESKIP_DIR ?? '' },
});

const runAgentTask = createAgentTaskRunner({
  agent, workspace: WORKSPACE,
  // The loaded local model. (STRATA_TASK_REMOTE_MODEL="<connection id>:<model id>" lets a developer or an API-only setup name a saved API model instead.)
  choice: () => {
    try { return { instanceId: native.resolveInstance(native.state().defaultId).id }; } catch { /* no local model is ready */ }
    const [connectionId, ...rest] = String(process.env.STRATA_TASK_REMOTE_MODEL ?? '').split(':');
    return connectionId && rest.length ? { connectionId, modelId: rest.join(':') } : null;
  },
});
const tasks = createTaskService({
  read: () => store.get().tasks ?? [],
  write: records => store.update({ tasks: records }),
  notify: task => hub.send('task', { id: task.id, title: task.title, status: task.status, attention: task.attention ?? null, error: task.error ?? null }),
  run: async (task, signal) => {
    if (agent.status().state !== 'unavailable') { const agentResult = await runAgentTask(task, signal); if (agentResult) return agentResult; }
    let text = '', failure = null, demo = !models.REAL;
    await runTurn({ message: task.prompt, signal, client: `Task: ${task.title}`, onFrame: frame => {
      if (frame.type === 'delta') text += frame.text ?? '';
      if (frame.type === 'start') demo = !!frame.demo;
      if (frame.type === 'error') failure = frame.message;
    }});
    if (failure) throw new Error(failure);
    signal.throwIfAborted();
    if (!text) throw new Error('The model returned no answer.');
    return { text, demo };
  },
});
const taskTimer = setInterval(() => tasks.tick(), 15000);
taskTimer.unref();

// The share and tunnel listeners take a free port (the next ones above the host's own) instead of a fixed PORT+1 / PORT+2 that Docker, WSL or Hyper-V may hold or reserve.
const sharePort = await findFreePort(Number(process.env.STRATA_SHARE_PORT ?? PORT + 1), '0.0.0.0', { skip: [PORT] });
const localShare = createLocalShare({ handle: openai, hasKey: () => !!store.get().serving?.apiKey, port: sharePort, changed: () => serving.setSharing(localShare.state()) });
const tunnelPort = await findFreePort(Number(process.env.STRATA_TUNNEL_PORT ?? PORT + 2), '127.0.0.1', { skip: [PORT, sharePort] });
const tunnelOrigin = createLocalShare({ handle: openai, hasKey: () => !!store.get().serving?.apiKey, host: '127.0.0.1', port: tunnelPort });
const connector = createConnectorInstaller({ directory: join(store.HOME, 'engine', 'cloudflare'), componentDirectory: join(componentsDirectory(), 'cloudflared') });
const cloudflare = createCloudflare({ origin: tunnelOrigin, originUrl: `http://127.0.0.1:${tunnelPort}`, hasKey: () => !!store.get().serving?.apiKey, changed: serving.setTunnel, locate: connector.locate, prepareConnector: connector.ensure, connectorInfo: connector.connector, findRunning: findRunningConnector });
await cloudflare.inspect();

/* --------------------------------------------------------------- routing --- */

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  const path = url.pathname;
  const method = req.method ?? 'GET';

  res.setHeader('access-control-allow-origin', '*');
  res.setHeader('access-control-allow-headers', 'content-type, authorization, x-api-key, x-strata-client');
  res.setHeader('access-control-allow-methods', 'GET, POST, PATCH, DELETE, OPTIONS');
  if (method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  try {
    if (path.startsWith('/api/') && req.headers.origin) {
      const origin = new URL(req.headers.origin);
      if (!['localhost', '127.0.0.1', '[::1]'].includes(origin.hostname)) return error(res, 403, 'Open the local Strata app to use host controls.');
    }
    if (path.startsWith('/api/')) {
      const modelOperation = !['GET', 'OPTIONS'].includes(method) && /^\/api\/(models|local-models|runtime|image-models|engine|native)(\/|$)/.test(path) && !['/api/runtime/plan', '/api/engine/validate', '/api/local-models/inspect'].includes(path);
      if (modelOperation) modelOperations++;
      try { return await api(req, res, path, method, url); }
      finally { if (modelOperation) modelOperations--; }
    }
    if (path.startsWith('/v1/')) {
      if (storageChanging && method !== 'GET') return error(res, 409, 'Wait for the model folder check to finish.');
      return await openai(req, res, apiPath(path), method);
    }
    return await staticFile(req, res, path);
  } catch (err) {
    console.error('[strata]', err);
    error(res, 500, err.message ?? 'Internal error');
  }
});

async function api(req, res, path, method, url) {
  if(path==='/api/fonts'&&method==='GET')return json(res,200,await systemFonts());
  if (storageChanging && !['GET', 'OPTIONS'].includes(method)) return error(res, 409, 'Wait for the model folder check to finish.');
  // --- events -------------------------------------------------------------
  if (path === '/api/events') return hub.add(res);
  if(path.startsWith('/api/native')){try{
  if(path==='/api/native'&&method==='GET')return json(res,200,await native.refresh(url.searchParams.get('refresh')==='1'));
  if(path==='/api/native'&&method==='PATCH')return json(res,200,await native.configure(await readBody(req)));
  if(path==='/api/native/directory'&&method==='POST')return json(res,200,native.setDownloadDirectory((await readBody(req)).directory));
  /* Settings > Storage > Models folder: pick (or type) a folder, say what it means, then start the choice. Copying is Control's (strata models move). */
  if(path==='/api/native/models-folder/plan'&&method==='POST'){
   const typed=await typedFolder((await readBody(req)).path,{isAbsolute});
   const picked=typed?{path:typed}:await chooseFolder('Choose a folder for your models');
   if(picked.unavailable){const u=chooserUnavailable(picked);return error(res,501,u.message,{code:u.code});}
   if(picked.cancelled)return json(res,200,{cancelled:true});
   return json(res,200,{plan:await native.planModelsFolder(picked.path)});}
  if(path==='/api/native/models-folder/start'&&method==='POST'){const body=await readBody(req);const result=await native.startModelsFolder({folder:body.path,choice:body.choice,stopRunning:body.stopRunning===true});return json(res,result.id?202:200,result);}
  if(path==='/api/native/models-folder/interrupted'&&method==='GET')return json(res,200,{interrupted:await native.interruptedMove()});
  if(path==='/api/native/models-folder/resume'&&method==='POST'){const body=await readBody(req);return json(res,202,await native.finishInterruptedMove({stopRunning:body.stopRunning===true}));}
  if(path==='/api/native/models-folder/leave'&&method==='POST')return json(res,200,await native.leaveInterruptedMove());
  if(path==='/api/native/plan'&&method==='POST')return json(res,200,await native.plan(await readBody(req)));
  if(path==='/api/native/operations'&&method==='POST'){if(storageChanging)return error(res,409,'Wait for the model folder check to finish.');const body=await readBody(req);return json(res,202,native.operation(body.kind,body.args));}
  if(path==='/api/native/logs'&&method==='GET')return json(res,200,{logs:native.logs()});
  if(path==='/api/native/logs/save'&&method==='POST'){
   /* Diagnostics to a file: the engine messages the app kept, failed operations, and the tail of each model's server log under Control.
      Written to the user's Downloads folder (the window cannot save a download); the path is returned to show. */
   const NL=String.fromCharCode(10);
   const lines=[`Strata diagnostics ${new Date().toISOString()}`,`Version ${native.state().version??'unknown'}`,'','== Engine messages =='];
   for(const l of native.logs())lines.push(`${l.at} ${l.instanceId}: ${l.line}`);
   lines.push('','== Operations ==');for(const j of native.state().jobs)lines.push(`${j.startedAt} ${j.kind} ${j.state}${j.error?`: ${j.error}`:''}${j.details?`${NL}  ${j.details}`:''}`);
   try{const logs=join(dirname(componentsDirectory()),'logs');const names=readdirSync(logs).filter(n=>/^desktop-.*\.log$/.test(n)).sort();const crash=names.filter(n=>n.startsWith('desktop-crash-')).slice(-3).reverse();const runs=names.filter(n=>!n.startsWith('desktop-crash-')).slice(-2).reverse();lines.push('',`== Desktop shell: window crash logs (${crash.length} newest) ==`);for(const n of crash)lines.push('',`-- ${n}`,readFileSync(join(logs,n),'utf8'));lines.push('',`== Desktop shell: last lines of the newest ${runs.length} run logs ==`);for(const n of runs)lines.push('',`-- ${n}`,...readFileSync(join(logs,n),'utf8').split(NL).slice(-300));}catch{/* no desktop logs (not the Linux app, or no crash) */}
   try{const runtime=process.env.STRATA_RUNTIME_DIR??join(dirname(componentsDirectory()),'runtime');for(const name of readdirSync(runtime).filter(n=>n.endsWith('.server.log'))){lines.push('',`== ${name} (last lines) ==`,...readFileSync(join(runtime,name),'utf8').split(NL).slice(-200));}}catch{/* no server logs yet */}
   const folder=join(process.env.USERPROFILE??process.env.HOME??homedir(),'Downloads');mkdirSync(folder,{recursive:true});
   const file=join(folder,`strata-diagnostics-${new Date().toISOString().replace(/[:.]/g,'-').slice(0,19)}.txt`);writeFileSync(file,lines.join(NL),'utf8');
   return json(res,200,{path:file});}
  if(path==='/api/native/local/preview'&&method==='POST'){
   /* "Use a local GGUF": pick (or type) the folder, and Control says what is in it. Nothing is copied or changed. */
   const typed=await typedFolder((await readBody(req)).path,{isAbsolute,fileType:/\.gguf$/i});
   const picked=typed?{path:typed}:await chooseFolder('Choose the folder that holds your GGUF files');
   if(picked.unavailable){const u=chooserUnavailable(picked);return error(res,501,u.message,{code:u.code});};
   if(picked.cancelled)return json(res,200,{cancelled:true});
   return json(res,200,{path:picked.path,preview:await native.localPreview(picked.path)});}
  if(path==='/api/native/benchmark'&&method==='GET')return json(res,200,native.benchmarkReport());
  /* The auto-tuner's saved tunings (Strata Control's presets, shared with the Terminal). Starting a tune or a re-check is POST /api/native/operations with kind autotune / recheck. */
  if(path==='/api/native/tune/status'&&method==='GET')return json(res,200,await native.tuneStatus({model:url.searchParams.get('model'),resourceProfile:url.searchParams.get('resourceProfile')||undefined,vision:url.searchParams.get('vision')||undefined}));
  if(path==='/api/native/tune/effective'&&method==='GET')return json(res,200,await native.tuneEffective({model:url.searchParams.get('model'),resourceProfile:url.searchParams.get('resourceProfile')||undefined,vision:url.searchParams.get('vision')||undefined}));
  if(path==='/api/native/tune/off'&&method==='POST'){const body=await readBody(req);return json(res,200,await native.tuneOff({model:body.model,resourceProfile:body.resourceProfile,vision:body.vision}));}
  if(path==='/api/native/tune/presets'&&method==='GET')return json(res,200,await native.tunePresets.list());
  if(path==='/api/native/tune/undo'&&method==='POST'){const body=await readBody(req);return json(res,200,await native.tunePresets.undoFor({model:body.model,resourceProfile:body.resourceProfile,vision:body.vision}));}
  const tunePreset=/^\/api\/native\/tune\/presets\/([0-9a-f]{8,32})\/(apply|rename|delete|undo)$/.exec(path);
  if(tunePreset&&method==='POST')return json(res,200,await native.tunePresets.action(tunePreset[2],tunePreset[1],await readBody(req)));
  const nativeOperation=/^\/api\/native\/operations\/([a-z0-9-]+)\/cancel$/.exec(path);
  if(nativeOperation&&method==='POST')return json(res,200,await native.cancel(nativeOperation[1]));
  const nativeInstance=/^\/api\/native\/instances\/([a-z0-9-]+)\/(unload|default)$/.exec(path);
  if(nativeInstance&&method==='POST')return json(res,200,nativeInstance[2]==='unload'?await native.unload(nativeInstance[1]):native.setDefault(nativeInstance[1]));
  const nativeSettings=/^\/api\/native\/models\/([a-z0-9_-]+)\/settings$/.exec(path);
  /* Expert mode shows what will actually run: the saved tuning's values come with the settings (null when none applies or unreadable). */
  if(nativeSettings&&method==='GET'){const settings=native.settings(nativeSettings[1]);let tuned=null;try{const t=await native.tuneEffective({model:nativeSettings[1],resourceProfile:url.searchParams.get('resourceProfile')||undefined,vision:url.searchParams.get('vision')||undefined});tuned=t.applied?t:null;}catch{tuned=null;}return json(res,200,{...settings,tuned});}
  if(nativeSettings&&method==='PATCH')return json(res,200,native.save(nativeSettings[1],await readBody(req)));
  const nativeSettingsFile=/^\/api\/native\/models\/([a-z0-9_-]+)\/settings-file$/.exec(path);
  if(nativeSettingsFile&&method==='GET')return json(res,200,native.settingsFile(nativeSettingsFile[1]));
  if(nativeSettingsFile&&method==='POST')return json(res,200,await native.openSettingsFile(nativeSettingsFile[1]));

  // Credentials are handled only on the host. Responses and events expose a key-set flag.
  return error(res,404,'Unknown native control action.');}catch(e){return error(res,/Finish|Unload|running|current operation/.test(e.message)?409:400,e.message,{...(e.code&&typeof e.code==='string'?{code:e.code}:{}),...(e.details?{details:String(e.details)}:{})});}}

  if (path === '/api/model-providers' && method === 'GET') return json(res, 200, withPictureAnswers(modelProviders.list()));
  if (path === '/api/model-providers/discover' && method === 'POST') {
    const body = await readBody(req);
    const controller = new AbortController();
    res.on('close', () => { if (!res.writableEnded) controller.abort(); });
    try { return json(res, 200, await modelProviders.discover(body, controller.signal)); }
    catch (err) { return error(res, 400, err.name === 'TimeoutError' ? 'The model API timed out. Check the address and try again.' : err.message); }
  }
  if (path === '/api/model-providers' && method === 'POST') {
    try { return json(res, 200, withPictureAnswers(modelProviders.save(await readBody(req)))); }
    catch (err) { return error(res, 400, err.message); }
  }
  const providerMatch = /^\/api\/model-providers\/([a-zA-Z0-9-]+)$/.exec(path);
  if (providerMatch && method === 'DELETE') return json(res, 200, withPictureAnswers(modelProviders.remove(providerMatch[1])));

  if(path === '/api/artifacts/open' && method === 'POST') {
    const body=await readBody(req, 30*1024*1024);
    return json(res,200,await openArtifact(body,{examplesDir:join(ROOT,'public','examples'),previewsDir:join(store.HOME,'previews')}));
  }

  // Host-owned recurring prompts. Creating a draft never enables a schedule.
  // ------------------------------------------------------------------- Tuning
  //
  // The review surface and the recorded reports. Both are real: the plan is computed from the
  // selected engine binary's own flag list, and a report is read from the job's durable record.
  // There is no run route, because a sweep needs a measurement harness that does not exist yet and
  // a fabricated measurement is worse than no measurement.
  if (path === '/api/tune/plan' && method === 'POST') {
    const body = await readBody(req);
    const result = await tuning.plan({
      engine: String(body.engine ?? '').trim(),
      backend: String(body.backend ?? '').trim(),
      model: body.model ? String(body.model) : undefined,
      objective: body.objective, trials: body.trials, repeats: body.repeats,
    });
    if (!result.ok) return error(res, result.code === 2 ? 400 : 502, result.error);
    return json(res, 200, { plan: result.plan });
  }
  if (path === '/api/tune/engine' && method === 'GET') {
    // The plan is only meaningful for a real engine binary, so the panel asks rather than guessing.
    // `available` is false when nothing is installed, which is a state the panel must show honestly
    // instead of offering a plan that cannot be computed.
    const binary = supervisor.findBinary();
    const status = supervisor.status();
    return json(res, 200, {
      available: Boolean(binary?.path),
      path: binary?.path ?? null,
      backend: status?.backend ?? null,
      running: Boolean(status?.running),
    });
  }
  if (path === '/api/tune/jobs' && method === 'GET') {
    return json(res, 200, { jobs: tuning.jobs(), stateDir: tuning.stateDir() });
  }
  if (path === '/api/tune/job' && method === 'GET') {
    const job = tuning.job(url.searchParams.get('id'));
    if (!job) return error(res, 404, 'No such tuning job.');
    return json(res, 200, { job });
  }

  // ---------------------------------------------------------------- Time-skip
  //
  // The job lives in Strata Control (`strata_control/time_skip.py`) and is driven through its CLI.
  // These routes only start it, follow it and apply its result; they never compute a status or a
  // score of their own, so the UI cannot show progress the job did not report.
  if (path === '/api/timeskip' && method === 'GET') {
    return json(res, 200, { runs: [...forkTs.list(), ...timeskip.list()], stateDir: timeskip.stateDir() });
  }
  if (path === '/api/timeskip/status' && method === 'GET') {
    const id = url.searchParams.get('id');
    if (id && forkTs.has(id)) return json(res, 200, { run: forkTs.status(id) });
    const run = id ? timeskip.status(id) : null;
    if (!run) return error(res, 404, 'No such Time-skip run.');
    return json(res, 200, { run: { ...run, activity: trialActivity(run) } });
  }
  if (path === '/api/timeskip/start' && method === 'POST') {
    const body = await readBody(req);
    const workspaceId = typeof body.workspaceId === 'string' ? body.workspaceId : null;
    const workspace = String(body.workspace ?? '').trim() || ((workspaceId ? workspaces.find(w => w.id === workspaceId)?.path : null) ?? WORKSPACE);
    const prompt = String(body.prompt ?? '').trim().slice(0, 4000);
    if (body.legacy !== true) {
      if (!prompt) return error(res, 400, 'Say what you want made or improved after /time-skip.');
      if (!existsSync(workspace)) return error(res, 400, 'Time-skip works on a folder of your own. Choose a folder for this chat, then try again.');
      const cwd = chatFolders.get(typeof body.sessionId === 'string' ? body.sessionId : '') ?? workspace;
      if (resolve(cwd) === resolve(ROOT) || resolve(cwd).startsWith(resolve(ROOT) + sep)) return error(res, 400, 'Time-skip works on a folder of your own. Choose a workspace folder for this chat, then try again.');
      if (measureFolder(cwd).tooBig) return error(res, 400, 'This folder is too big to try a change on a copy. Open a smaller folder for this chat, then try again.');
      let choice, slots = 1;
      if (body.remoteModel?.connectionId && body.remoteModel?.modelId) { choice = { connectionId: String(body.remoteModel.connectionId), modelId: String(body.remoteModel.modelId) };
        { const own = modelProviders.withSecrets().find(c => c.id === choice.connectionId); if (own && own.protocol === 'openai-chat') await visionProbe.check(own, choice.modelId); } slots = Number(body.slots) >= 1 ? Math.floor(Number(body.slots)) : 1; }
      else { try { const inst = native.resolveInstance(native.state().defaultId); choice = { instanceId: inst.id }; slots = inst.metrics?.live?.slots?.length || inst.parallel || 1; } catch { return error(res, 503, 'Load a model first, then try Time-skip again.'); } }
      const chatId = typeof body.sessionId === 'string' && body.sessionId ? body.sessionId : null;
      if (chatId && forkTs.byChat(chatId)) return error(res, 409, 'A Time-skip is already running in this chat.');
      const run = forkTs.start({ prompt, workspace: cwd, chatId, choice, thinking: typeof body.thinking === 'string' ? body.thinking : 'Low', slots });
      return json(res, 200, { run, simulated: false });
    }
    const reviseCommand = body.useAgent === false ? null : TIMESKIP_REVISE;
    // The folder of the app itself is never a person's work: "Keep the update" would write into the installation.
    if (resolve(workspace) === resolve(ROOT) || resolve(workspace).startsWith(resolve(ROOT) + sep)) return error(res, 400, 'Time-skip works on a folder of your own. Choose a workspace folder for this chat, then try again.');
    // A trial copies the folder (twice), so a huge folder is refused in plain words instead of filling the disk.
    if (existsSync(workspace)) {
      const size = measureFolder(workspace);
      if (size.tooBig) return error(res, 400, 'This folder is too big to try a change on a copy. Open a smaller folder for this chat, then try again.');
    }
    const result = await timeskip.startDetached({
      workspace, prompt,
      attempts: body.attempts ?? 1,
      reviseCommand,
      extraEnv: { STRATA_HOST_URL: `http://127.0.0.1:${PORT}`, STRATA_INTERNAL_SECRET: INTERNAL_SECRET },
    });
    if (!result.ok) return error(res, result.code === 2 ? 400 : 502, result.error);
    const chatId = typeof body.sessionId === 'string' && body.sessionId ? body.sessionId : null;
    trials.set(result.run.id, { chatId, turnChat: null, controller: null, step: null, lastSent: '' });
    if (chatId) trialByChat.set(chatId, result.run.id);
    watchTrial(result.run.id);
    if (body.remoteModel?.connectionId && body.remoteModel?.modelId) timeskipChoices.set(result.run.id, { connectionId: String(body.remoteModel.connectionId), modelId: String(body.remoteModel.modelId) });
    return json(res, 200, { run: { ...result.run, activity: trialActivity(result.run) }, simulated: false });
  }
  if (path === '/api/timeskip/files' && method === 'GET') {
    const fid = url.searchParams.get('id');
    const found = forkTs.has(fid) ? forkTs.files(fid) : timeskip.files(fid);
    if (!found) return error(res, 404, 'No such Time-skip run.');
    return json(res, 200, found);
  }
  /* One picture of a run, original or updated, for the Compare view. Only pictures inside the run's own folders; served as an image and nothing else. */
  if (path === '/api/timeskip/file' && method === 'GET') {
    const iid = url.searchParams.get('id');
    const found = forkTs.has(iid) ? (() => { const p = forkTs.image(iid, url.searchParams.get('path'), url.searchParams.get('version')); return p ? { path: p, type: imageTypeOf(p) } : null; })() : timeskip.image(iid, url.searchParams.get('path'), url.searchParams.get('version'));
    if (!found) return error(res, 404, 'No such picture in this Time-skip run.');
    res.writeHead(200, { 'content-type': found.type, 'cache-control': 'private, max-age=300', 'x-content-type-options': 'nosniff', 'content-security-policy': "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:" });
    createReadStream(found.path).pipe(res);
    return;
  }
  if (path === '/api/timeskip/stop' && method === 'POST') {
    const body = await readBody(req);
    if (forkTs.has(String(body.id ?? ''))) return json(res, 200, { stopped: await forkTs.stop(String(body.id)) });
    return json(res, 200, { stopped: await stopTrial(String(body.id ?? '')) });
  }
  // One revision attempt for a job this host started: the app's assistant works inside the candidate folder.
  if (path === '/api/internal/timeskip-revise' && method === 'POST') {
    const given = Buffer.from(String(req.headers['x-strata-internal'] ?? ''));
    const want = Buffer.from(INTERNAL_SECRET);
    if (given.length !== want.length || !timingSafeEqual(given, want)) return error(res, 403, 'Not allowed.');
    const body = await readBody(req);
    const cwd = String(body.cwd ?? '');
    // Only the job's own candidate folders may be worked in.
    const root = resolve(timeskip.stateDir());
    if (!cwd || !resolve(cwd).startsWith(root + sep)) return error(res, 400, 'That folder is not part of a Time-skip run.');
    const runId = resolve(cwd).slice(root.length + 1).split(sep)[0];
    const chosen = timeskipChoices.get(runId) ?? (() => { try { return { instanceId: native.resolveInstance(native.state().defaultId).id }; } catch { return null; } })();
    if (!chosen) return error(res, 503, 'Load a model first, then try Time-skip again.');
    const chatId = `timeskip:${randomBytes(6).toString('hex')}`;
    // The answer is a stream that starts at once and carries a beat every 15 s: a turn on a slow computer can run for many minutes with
    // nothing to say, and a caller that hears no byte for five minutes gives up (that was "The assistant could not be reached").
    res.writeHead(200, { 'content-type': 'application/x-ndjson; charset=utf-8', 'cache-control': 'no-cache, no-transform' });
    const write = frame => { if (!res.writableEnded && !res.destroyed) res.write(`${JSON.stringify(frame)}\n`); };
    write({ type: 'beat' });
    const beat = setInterval(() => write({ type: 'beat' }), 15_000);
    const controller = new AbortController();
    const trial = trials.get(runId);
    if (trial) { trial.turnChat = chatId; trial.controller = controller; }
    res.on('close', () => { clearInterval(beat); if (!res.writableEnded) { controller.abort(); void agent.stop(chatId).catch(() => {}); } });
    let text = '', failure = null;
    try {
      await agent.runTurn({ chatId, message: timeskipRevisionMessage(String(body.prompt ?? '')), workspacePath: cwd, choice: chosen, mode: 'bypass', thinking: 'low', history: [], signal: controller.signal, onFrame: frame => {
        if (frame.type === 'delta') text += frame.text ?? '';
        else if (frame.type === 'error') failure = frame.message;
        else if (frame.type === 'status' && trial) {
          // the assistant's own current step (what it is writing or running) rides along with the job's phase
          trial.step = frame.phase === 'idle' ? null : { code: frame.code, text: frame.text, args: frame.args };
        }
      } });
    } catch (e) { failure = failure ?? e.message; } finally { agent.forget(chatId); clearInterval(beat); if (trial) trial.step = null; }
    if (controller.signal.aborted) failure = failure ?? 'The trial was stopped.';
    write(failure ? { type: 'error', message: failure } : { type: 'result', text: text.slice(-600) });
    res.end();
    return undefined;
  }
  if (path === '/api/timeskip/apply' && method === 'POST') {
    const body = await readBody(req);
    if (forkTs.has(String(body.id ?? ''))) {
      try { return json(res, 200, { apply: forkTs.apply({ id: String(body.id), which: body.which === 'a' ? 'a' : 'b', restore: Boolean(body.restore) }) }); }
      catch (e) { return error(res, 400, e.message); }
    }
    const result = await timeskip.apply({
      id: body.id, workspace: body.workspace ?? null,
      dryRun: Boolean(body.dryRun), restore: Boolean(body.restore), attempt: body.attempt ?? null,
    });
    if (!result.ok) {
      // A concurrent user edit is a conflict the caller must see, not a server error to retry.
      return json(res, result.conflict ? 409 : 400, { error: result.error, conflict: Boolean(result.conflict) });
    }
    return json(res, 200, { apply: result.apply });
  }
  if (path === '/api/tasks' && method === 'GET') return json(res, 200, { tasks: tasks.list() });
  if (path === '/api/tasks/draft' && method === 'POST') {
    const body = await readBody(req);
    const description = String(body.description ?? '').trim().slice(0, 12000);
    if (!description) return error(res, 400, 'Describe the task first.');
    let draft = { title: description.slice(0, 70), prompt: description, frequency: /week/i.test(description) ? 'weekly' : /hour/i.test(description) ? 'hourly' : 'daily', at: '09:00', weekday: 1, timeZone: body.timeZone ?? 'UTC', enabled: true };
    if (models.REAL) {
      let text = '', failure = null;
      await runTurn({ message: 'Draft a recurring prompt task from the description below. Return only JSON with title (short), prompt (self-contained; text reasoning only, no file, web, email or external tools), frequency (hourly/daily/weekdays/weekly), at (HH:MM), weekday (0 Sunday to 6 Saturday). Do not claim to execute it. Description: ' + description, signal: AbortSignal.timeout(120000), client: 'Task drafting', onFrame: frame => { if (frame.type === 'delta') text += frame.text ?? ''; if (frame.type === 'error') failure = frame.message; } });
      if (failure) return error(res, 400, failure);
      try { draft = { ...draft, ...JSON.parse(text.replace(/^```(?:json)?\s*|\s*```$/g, '').trim()), timeZone: body.timeZone ?? 'UTC', enabled: true }; }
      catch { return error(res, 400, 'The model did not return a usable draft. Try again or set up manually.'); }
    }
    try { return json(res, 200, { draft: validateTask(draft), demo: !models.REAL }); }
    catch (e) { return error(res, 400, e.message); }
  }
  if (path === '/api/tasks' && method === 'POST') {
    try { return json(res, 201, tasks.put(await readBody(req))); } catch(e) { return error(res, 400, e.message); }
  }
  const taskRoute = path.match(/^\/api\/tasks\/([^/]+)(?:\/(run|stop))?$/);
  if (taskRoute) {
    const [, id, action] = taskRoute;
    if (!tasks.list().some(t => t.id === id)) return error(res, 404, 'Task not found.');
    try {
      if (method === 'PATCH' && !action) return json(res, 200, tasks.put(await readBody(req), id));
      if (method === 'POST' && action === 'run') { if (tasks.list().find(t => t.id === id)?.archived) return error(res, 400, 'Restore this task first.'); void tasks.execute(id); return json(res, 202, { accepted:true }); }
      if (method === 'POST' && action === 'stop') { tasks.stop(id); return json(res, 200, { stopped:true }); }
    } catch (e) { return error(res, 400, e.message); }
  }

  // --- health / setup -----------------------------------------------------
  if (path === '/api/health') {
    return json(res, 200, {
      ok: true,
      uptimeSec: Math.round(process.uptime()),
      backend: currentBackend(),
      engine: supervisor.status(),
      clients: hub.size,
    });
  }

  // --- Strata Control: what the installed app runs on, and what it will not do ---------------------
  if (path === '/api/control/status' && method === 'GET') {
    const engine = control ? await control.engine().catch(e => ({ usable: false, notice: e.message })) : null;
    // the first start copies the engine that came with the app (about a minute): say so instead of "no engine"
    if (engine && !engine.usable && !bundledEngineSettled) engine.notice = 'The engine that came with the app is being set up. This takes a minute the first time.';
    return json(res, 200, {
      packaged: process.env.STRATA_PACKAGED === '1',
      control: CONTROL ? { dir: CONTROL.dir, python: CONTROL.python, version: control.version, bundle: CONTROL.bundle } : null,
      engine,
      engineSetup: !bundledEngineSettled,
      componentsDir: componentsDirectory(),
      acquisition: { forbidden: acquisitionForbidden(), env: process.env.STRATA_NO_ACQUIRE ?? null },
    });
  }
  // The shell (and nothing else) holds the secret that lets it end the host or tell it which window is in use.
  const fromShell = () => {
    const token = process.env.STRATA_SHUTDOWN_TOKEN ?? '';
    const given = String(req.headers['x-strata-shutdown'] ?? '');
    return token.length >= 32 && given.length === token.length && timingSafeEqual(Buffer.from(given), Buffer.from(token));
  };
  // Which window the person is using: the page shows "Quit Strata" in browser mode only.
  if (path === '/api/app/mode' && method === 'GET') return json(res, 200, appUi.state());
  if (path === '/api/app/quit' && method === 'POST') {
    // A web page cannot send this header without the browser asking the host first, and the host refuses pages that are not local (above).
    if (!req.headers['x-strata-client']) return error(res, 403, 'Not allowed.');
    const verdict = appUi.quit();
    return verdict.ok ? json(res, verdict.status, { stopping: true }) : error(res, verdict.status, verdict.error);
  }
  // The shell: the window process registers its updater; or the window could not run and everything moves to a browser tab.
  if (path === '/api/control/ui' && method === 'POST') {
    if (!fromShell()) return error(res, 403, 'Not allowed.');
    const body = await readBody(req);
    try {
      if (body.bridge) appUi.setBridge(body.bridge);
      if (body.mode === 'browser') return json(res, 200, await appUi.switchToBrowser({ url: `http://127.0.0.1:${PORT}/`, reason: body.reason, notice: typeof body.notice === 'string' ? body.notice.slice(0, 300) : '', openPage: body.open !== false }));
      if (body.open === true) return json(res, 200, { ...appUi.state(), opened: await appUi.openPage(`http://127.0.0.1:${PORT}/`) });
      return json(res, 200, appUi.state());
    } catch (err) { return error(res, 400, err.message); }
  }
  // The shell asks for a clean stop (engines this app started are stopped through Control) before it ends the process.
  if (path === '/api/control/shutdown' && method === 'POST') {
    if (!fromShell()) return error(res, 403, 'Not allowed.');
    json(res, 202, { stopping: true });
    setImmediate(() => void shutdown());
    return;
  }

  // Files dropped on the app window (Linux, macOS): the shell hands over the paths, the page fetches each file once (native-drop.mjs).
  if (path === '/api/app/drop' && method === 'POST') {
    if (!fromShell()) return error(res, 403, 'Not allowed.');
    try { return json(res, 200, { files: await nativeDrops.add((await readBody(req)).paths) }); } catch (err) { return error(res, 400, err.message); }
  }
  const dropGet = /^\/api\/app\/drop\/([0-9a-f-]{36})$/.exec(path);
  if (dropGet && method === 'GET') {
    const file = nativeDrops.take(dropGet[1]);
    if (!file) return error(res, 404, 'That file is no longer available. Drop it again.');
    res.writeHead(200, { 'content-type': file.type, 'content-length': file.bytes.length, 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' }); res.end(file.bytes); return undefined;
  }

  if (path === '/api/setup' && method === 'GET') return json(res, 200, setupState());
  if (path === '/api/updates/check' && method === 'GET' && control?.updateCheck) return json(res, 200, await checkComponentUpdates({ control, appUpdater, appVersion: process.env.STRATA_APP_VERSION || null }));
  if (path === '/api/updates/check' && method === 'GET') return json(res, 200, await checkUpdates({ appVersion:appVersionToShow(setupState().app.version,native.state().version), engineVersion:native.state().available?(native.state().installed?.engine?.version??null):(await engine.config()).engine.version }));

  if (path === '/api/setup/step' && method === 'POST') {
    const body = await readBody(req);
    const s = store.get();
    const step = Math.max(1, Math.min(4, migrateSetup({ step: body.step, flowVersion: body.flowVersion }).step));
    store.update({
      setup: { ...s.setup, flowVersion: 2, complete: false, step, dismissed: !!body.dismissed, choices: { ...s.setup.choices, ...body.choices }, history: [...(s.setup.history ?? []), { step, at: new Date().toISOString() }] },
    });
    const next = setupState();
    hub.send('setup', next);
    return json(res, 200, next);
  }

  if (path === '/api/setup/complete' && method === 'POST') {
    const s = store.get();
    if (!models.library().loaded&&!native.hasLive()) return error(res, 409, 'Load a model before completing setup.');
    store.update({ setup: { ...s.setup, dismissed: false, complete: true, flowVersion: 2, step: 4, completedAt: new Date().toISOString() } });
    const next = setupState();
    hub.send('setup', next);
    return json(res, 200, next);
  }

  // --- hardware -----------------------------------------------------------
  if (path === '/api/hardware') {
    const force = url.searchParams.get('reprobe') === '1';
    const hw = await hardware.detect({ force });
    if (force) monitor.init(hub, hw);
    return json(res, 200, hw);
  }

  // --- models -------------------------------------------------------------
  if (path === '/api/local-models/browse' && method === 'GET') {
    const shortcuts = [{name:'Model directory',path:store.MODELS_DIR},{name:'Home',path:process.env.USERPROFILE??process.env.HOME}].filter(s=>s.path);
    try { return json(res, 200, { ...browseLocalModels(url.searchParams.get('directory') ?? store.MODELS_DIR), shortcuts }); }
    catch (err) { return error(res, 400, err.message); }
  }
  if (path === '/api/local-models/inspect' && method === 'POST') {
    try { const body=await readBody(req); return json(res,200,inspectLocalModel(body.path)); }
    catch (err) { return error(res,400,err.message); }
  }
  if (path === '/api/local-models' && method === 'POST') {
    const body=await readBody(req);
    if(storageChanging)return error(res,409,'Wait for the model folder check to finish.');
    try {
      const preview=inspectLocalModel(body.path),previous=variant(preview.variant.id);
      if(body.name!==undefined){if(typeof body.name!=='string'||!body.name.trim()||body.name.length>160)return error(res,400,'Enter a display name up to 160 characters.');preview.variant.name=body.name.replace(/[\x00-\x1f\x7f]/g,' ').trim();if(!preview.variant.name)return error(res,400,'Enter a display name.');preview.variant.label=`${preview.variant.name} · ${preview.variant.quant}`;}
      if(previous && runtimeModels.hasVariant(previous.id) && JSON.stringify(previous.local)!==JSON.stringify(preview.variant.local))return error(res,409,'Unload this local model before replacing its file reference.');
      localModels.add(preview);
      return json(res,200,{library:runtimeModels.libraryState(models.localModelAdded()),variantId:preview.variant.id});
    } catch (err) { return error(res,400,err.message); }
  }
  if (path === '/api/storage/models-directory' && method === 'POST') {
    const body = await readBody(req);
    let directory;
    try { directory = store.validateModelsDirectory(body.directory); }
    catch (err) { return error(res, 400, err.message); }
    const changed = directory !== store.MODELS_DIR;
    const instances = runtimeState().instances;
    if (storageChanging || modelOperations || models.directoryBusy() || imageModels.busy()||native.state().jobs.some(j=>j.state==='running')) return error(res, 409, 'Finish or cancel model downloads and loading before changing the folder.');
    if (serving.state().requests.some(r => ['queued', 'generating'].includes(r.state))) return error(res, 409, 'Finish or cancel model requests before changing the folder.');
    if (changed && instances.length && body.unload !== true) return error(res, 409, 'Unload the current models before changing the folder.');
    const nativePlan=nativeUnloadPlan({changed,unload:body.unload,instances:native.state().instances});
    if(nativePlan.refuse)return error(res,409,nativePlan.refuse);
    for(const id of nativePlan.ids)await native.unload(id);
    storageChanging = true;
    try {
      if (changed) {
        if (models.REAL) await models.unload();
        else for (const instance of instances) runtimeModels.unload(instance.id);
        store.changeModelsDirectory(directory);
        engine.markStopped(); monitor.idle('Model directory changed');
      }
      models.init(hub);
      const images = await imageModels.rescan(changed);
      const hw = await hardware.detect({ force: true });
      const library = runtimeModels.libraryState(models.library());
      hub.send('models', library); hub.send('setup', setupState()); hub.send('engine', await engine.config());
      return json(res, 200, { library, imageModels: images, runtime: runtimeState(), hardware: hw, setup: setupState() });
    } catch (err) { return error(res, 400, err.message); }
    finally { storageChanging = false; hub.send('models', models.library()); hub.send('setup', setupState()); }
  }
  if (path === '/api/image-models' && method === 'GET') return json(res, 200, imageModels.library());
  if (path === '/api/image-models' && method === 'PATCH') {
    const body = await readBody(req);
    return json(res, 200, imageModels.select(body.selected));
  }
  if (path === '/api/image-models/apply' && method === 'POST') return json(res, 200, imageModels.apply());
  const imageMatch = /^\/api\/image-models\/(q4|q5|q6|q8)\/(install|cancel)$/.exec(path);
  if (imageMatch && method === 'POST') {
    const [, id, action] = imageMatch;
    return json(res, 200, action === 'install' ? imageModels.install(id) : await imageModels.cancel(id));
  }
  if(path==='/api/runtime' && method==='GET')return json(res,200,runtimeState());
  if(path==='/api/runtime/plan' && method==='POST')return json(res,200,runtimeModels.plan(await readBody(req)));
  if(path==='/api/runtime/load' && method==='POST'){
    const body=await readBody(req);const checked=await engine.validate({...body.config,variantId:body.variantId});
    if(storageChanging)return error(res,409,'Wait for the model folder check to finish.');
    if(!checked.ok)return error(res,400,checked.issues.filter(i=>i.level==='error').map(i=>i.message).join(' '));
    if(models.REAL)return error(res,409,'This native host supports one model on automatic GPU placement.');
    try{const plan=runtimeModels.plan(body);if(!plan.allowed)return error(res,409,'This model does not fit the selected setup.');for(const instance of native.state().instances)await native.unload(instance.id);return json(res,200,runtimeModels.load(body));}catch(err){return error(res,409,err.message);}
  }
  const runtimeMatch=/^\/api\/runtime\/([^/]+)\/(unload|default)$/.exec(path);
  if(runtimeMatch && method==='POST'){
    const [,id,action]=runtimeMatch;runtimeState();
    if(models.REAL && action==='unload'){
      const instance=runtimeModels.state().instances.find(i=>i.id===id);
      if(!instance)return error(res,404,'This model is no longer loaded.');
      if(serving.state().requests.some(r=>r.instanceId===id && ['queued','generating'].includes(r.state)))return error(res,409,'Finish or cancel this model’s requests first.');
      await models.unload();engine.markStopped();hub.send('engine',await engine.config());runtimeState();return json(res,200,runtimeState());
    }
    try{return json(res,200,action==='unload'?runtimeModels.unload(id):runtimeModels.setDefault(id));}
    catch(err){return error(res,409,err.message);}
  }
  if (path === '/api/models' && method === 'GET') return json(res, 200, runtimeModels.libraryState(models.library()));

  const modelMatch = /^\/api\/models\/([^/]+)(?:\/(install|cancel|load))?$/.exec(path);
  if (modelMatch) {
    const [, id, action] = modelMatch;
    if (method === 'DELETE') {if(runtimeModels.hasVariant(id))return error(res,409,'Unload every instance of this model before deleting its files.');return json(res, 200, runtimeModels.libraryState(models.remove(id)));}
    if (method === 'POST' && action === 'install') { try { return json(res, 200, runtimeModels.libraryState(await models.install(id))); } catch (err) { return error(res, err.status ?? 500, err.message); } }
    if (method === 'POST' && action === 'cancel') return json(res, 200, runtimeModels.libraryState(models.cancel(id)));
    if (method === 'POST' && action === 'load') {
      const hw = await hardware.detect();
      if(storageChanging)return error(res,409,'Wait for the model folder check to finish.');
      if(!models.REAL){
        const config={beginner:store.get().beginner,expert:engine.params(),extra:store.get().configExtra??{}};
        const gpuIds=config.expert.find(p=>p.id==='gpu.layers')?.value==='CPU only'?[]:hw.gpus.map(g=>g.id??`${g.vendor}:${g.index}`);
        try{const choice={variantId:id,gpuIds,config,executionMode:'single'};const plan=runtimeModels.plan(choice);if(!plan.allowed)throw new Error('This model does not fit the selected setup.');for(const instance of native.state().instances)await native.unload(instance.id);runtimeModels.load(choice);}
        catch(err){return error(res,409,err.message);}
        await monitor.setModel(id);engine.markRunning(hw.accelerator.backend);hub.send('engine',await engine.config());
        return json(res,200,runtimeModels.libraryState(models.library()));
      }
      if(!supervisor.findBinary())return error(res,409,'Install a compatible llama-server engine before switching from native Strata.');const candidate=models.library().variants.find(v=>v.id===id);if(!candidate||!['installed','loaded'].includes(candidate.state))return error(res,409,'Download this model before loading it.');for(const instance of native.state().instances)await native.unload(instance.id);
      const lib = await models.load(id, {
        hardware: hw,
        config: { beginner: store.get().beginner, expert: engine.params(), serving: store.get().serving },
        onPhase: async (phase, variantId) => {
          if (phase === 'loading') await monitor.beginLoading(variantId);
          if (phase === 'ready') {
            await monitor.setModel(variantId);
            monitor.idle('Model loaded — waiting for a request');
            monitor.setContextMax(CONTEXT_TOKENS[store.get().beginner.contextLength] ?? 32768);
            engine.markRunning(hw.accelerator.backend);
            hub.send('engine', await engine.config());
          }
        },
      });
      return json(res, 200, runtimeModels.libraryState(lib));
    }
  }

  if (path === '/api/models/unload' && method === 'POST') {
    if(!models.REAL && runtimeModels.state().defaultId){runtimeModels.unload(runtimeModels.state().defaultId);return json(res,200,runtimeModels.libraryState(models.library()));}
    const lib = await models.unload({ onPhase: () => monitor.idle('Model unloaded') });
    engine.markStopped();
    hub.send('engine', await engine.config());
    return json(res, 200, runtimeModels.libraryState(lib));
  }

  // --- engine -------------------------------------------------------------
  if (path === '/api/engine' && method === 'GET') return json(res, 200, await engine.config());
  if (path === '/api/engine' && method === 'PATCH') {
    const body = await readBody(req);
    const checked = await engine.validate(body);
    if (!checked.ok) return error(res, 400, checked.issues.filter(i => i.level === 'error').map(i => i.message).join(' '));
    const cfg = await engine.patch(body);
    if (body.beginner?.contextLength) {
      monitor.setContextMax(CONTEXT_TOKENS[body.beginner.contextLength] ?? 32768);
    }
    return json(res, 200, cfg);
  }
  if (path === '/api/engine/restart' && method === 'POST') {
    monitor.idle('Engine restarting');
    const cfg = await engine.restart();
    monitor.setModel(store.get().loaded);
    return json(res, 200, cfg);
  }
  if (path === '/api/engine/reset' && method === 'POST') return json(res, 200, await engine.reset());
  if (path === '/api/engine/validate' && method === 'POST') {
    const body = await readBody(req);
    return json(res, 200, await engine.validate(body ?? {}));
  }
  if (path === '/api/engine/status' && method === 'GET') return json(res, 200, supervisor.status());
  if (path === '/api/engine/logs' && method === 'GET') {
    return json(res, 200, {
      logs: supervisor.logs(Number(url.searchParams.get('limit') ?? 200)),
      status: supervisor.status(),
    });
  }
  if (path === '/api/engine/stop' && method === 'POST') {
    const status = await supervisor.stop();
    store.update({ loaded: null });
    monitor.idle('Model unloaded');
    engine.markStopped();
    hub.send('engine', await engine.config());
    hub.send('models', models.library());
    return json(res, 200, status);
  }
  if (path === '/api/engine/binary' && method === 'GET') return json(res, 200, engineInstall.status());
  if (path === '/api/engine/binary' && method === 'POST') {
    try {
      const result = await engineInstall.install({ force: url.searchParams.get('force') === '1' });
      return json(res, 200, result);
    } catch (err) {
      return error(res, err.status ?? 500, err.message, { ...engineInstall.status(), error: err.message });
    }
  }
  if (path === '/api/engine/export' && method === 'GET') return json(res, 200, await engine.exportConfig());

  // --- serving ------------------------------------------------------------
  if (path === '/api/serving/tunnel' && method === 'GET') return json(res, 200, await cloudflare.inspect());
  if (path === '/api/serving/tunnel' && method === 'POST') {
    try { const body = await readBody(req); if (body.action === 'stop') await cloudflare.stop(); else if (body.action === 'start') await cloudflare.start(body); else if (body.action === 'prepare') await cloudflare.prepare(); else throw new Error('Unknown tunnel action.'); return json(res, 200, serving.state()); }
    catch(e) { return error(res, 400, e.message); }
  }
  if (path === '/api/monitor/snapshot' && method === 'POST') {
    try { const body = await readBody(req, 25 * 1024 * 1024); return json(res, 200, await saveSnapshot(body.data)); }
    catch(e) { return error(res, 400, e.message); }
  }
  if (path === '/api/serving/share' && method === 'POST') {
    try { const body = await readBody(req); if (body.enabled === true) await localShare.start(); else await localShare.stop(); return json(res, 200, serving.state()); }
    catch(e) { return error(res, 400, e.message); }
  }
  if (path === '/api/serving' && method === 'GET') return json(res, 200, serving.state());
  if (path === '/api/serving' && method === 'PATCH') {
    const body = await readBody(req);
    return json(res, 200, serving.patch(body));
  }
  const cancelMatch = /^\/api\/serving\/requests\/([^/]+)\/cancel$/.exec(path);
  if (cancelMatch && method === 'POST') return json(res, 200, serving.cancel(cancelMatch[1]));

  // --- monitor ------------------------------------------------------------
  if (path === '/api/monitor') return json(res, 200, monitor.snapshot());

  // --- skills / workspace -------------------------------------------------
  if (path === '/api/skills' && method === 'GET') {
    /* Re-scan rather than returning the array built at startup.
       The screen offers a "Check again" action for a skill dropped into the
       workspace while the app is open; returning the cached list made that
       button a promise the host did not keep. Enabled state and usage counts
       are carried across, since they belong to the skill's identity rather than
       to the scan. */
    const known = new Map(skillState.map((s) => [s.id, s]));
    skillState = skills().map((s) => {
      const before = known.get(s.id);
      return before
        ? { ...s, enabled: before.enabled, onlyIn: before.onlyIn ?? null, useCount: before.useCount, lastUsedAt: before.lastUsedAt }
        : { ...s, onlyIn: null, useCount: 0, lastUsedAt: null };
    });
    syncAgentSkills();
    return json(res, 200, skillState);
  }
  // A skill written in the app: a name, what it is for, and the instructions. Saved as a folder the agent can read; no restart.
  if (path === '/api/skills' && method === 'POST') {
    const body = await readBody(req);
    try {
      const name = String(body.name ?? '').trim().slice(0, 80);
      const instructions = String(body.body ?? '').trim();
      if (!name) throw new Error('Give the skill a name.');
      if (!instructions) throw new Error('Write the instructions the assistant should follow.');
      // The form's own choices are kept: a skill created with "Skill enabled" off stays off (QA F04), and its trigger phrases are saved with it.
      const triggers = body.triggers === undefined ? [] : skillEdit({ triggers: body.triggers }).triggers;
      const id = writeCreatedSkill(CREATED_SKILLS, { name, description: String(body.description ?? '').trim().slice(0, 400), body: instructions.slice(0, 40000) });
      const newId = `my-${id}`, off = body.enabled === false;
      if (off || triggers.length) {
        const saved = store.get();
        store.update({
          ...(off ? { skillPreferences: { ...saved.skillPreferences, [newId]: { enabled: false, onlyIn: null } } } : {}),
          ...(triggers.length ? { skillEdits: { ...saved.skillEdits, [newId]: { ...saved.skillEdits?.[newId], triggers } } } : {}),
        });
      }
      const before = new Map(skillState.map(s => [s.id, s]));
      skillState = skills().map(s => before.has(s.id) ? { ...s, enabled: before.get(s.id).enabled, onlyIn: before.get(s.id).onlyIn ?? null, useCount: before.get(s.id).useCount, lastUsedAt: before.get(s.id).lastUsedAt } : { ...s, enabled: s.id === newId && off ? false : s.enabled, onlyIn: null, useCount: 0, lastUsedAt: null });
      syncAgentSkills();
      hub.send('skills', skillState);
      return json(res, 201, { id: newId, skills: skillState });
    } catch (e) { return error(res, 400, e.message); }
  }
  const skillDelete = /^\/api\/skills\/([^/]+)$/.exec(path);
  if (skillDelete && method === 'DELETE') {
    const skill = skillState.find(s => s.id === decodeURIComponent(skillDelete[1]));
    if (!skill) return error(res, 404, 'Skill not found.');
    if (skill.source !== 'created') return error(res, 400, 'Only a skill you wrote here can be removed from this page.');
    rmSync(join(CREATED_SKILLS, skill.id.replace(/^my-/, '')), { recursive: true, force: true });
    skillState = skillState.filter(s => s.id !== skill.id);
    syncAgentSkills();
    hub.send('skills', skillState);
    return json(res, 200, skillState);
  }
  const skillMatch = /^\/api\/skills\/([^/]+)$/.exec(path);
  if (skillMatch && method === 'PATCH') {
    const body = await readBody(req);
    const skillId = decodeURIComponent(skillMatch[1]);
    if (!skillState.some(s => s.id === skillId)) return json(res, 404, { error: 'Skill not found.' });
    let edit;
    try { edit = skillEdit(body); } catch (error) { return json(res, 400, { error: error.message }); }
    if (body.onlyIn != null && !workspaces.some(w => w.id === body.onlyIn)) return json(res, 400, { error: 'Choose an available workspace.' });
    skillState = skillState.map((s) =>
      s.id === skillId
        ? {
            ...s,
            ...edit,
            enabled: body.enabled === undefined ? s.enabled : !!body.enabled,
            // null clears the narrowing; a workspace id restricts to it.
            onlyIn: body.onlyIn === undefined ? (s.onlyIn ?? null) : (body.onlyIn || null),
          }
        : s,
    );
    store.update({ skillPreferences: Object.fromEntries(skillState.map(s => [s.id, { enabled: s.enabled, onlyIn: s.onlyIn }])), skillEdits: { ...store.get().skillEdits, [skillId]: { ...store.get().skillEdits?.[skillId], ...edit } } });
    syncAgentSkills();
    hub.send('skills', skillState);
    return json(res, 200, skillState);
  }

  if (path === '/api/workspace') {
    const root = url.searchParams.get('root');
    const found = root ? workspaces.find((w) => w.id === root) : null;
    // A stale id from a saved session falls back to the primary root rather than
    // erroring, so a removed folder cannot break the app.
    return json(res, 200, workspaceListing(found?.path ?? WORKSPACE));
  }

  // --- workspaces ---------------------------------------------------------
  // Read one file for the preview pane. Bounded deliberately: only inside a
  // registered workspace, only text, only up to a size that can be shown.
  if (path === '/api/workspace/file') {
    const rootId = url.searchParams.get('root');
    const rel = url.searchParams.get('path') ?? '';
    const root = (rootId ? workspaces.find((w) => w.id === rootId)?.path : null) ?? WORKSPACE;
    const target = resolve(root, rel);
    // A relative path may not escape its workspace.
    if (target !== resolve(root) && !target.startsWith(resolve(root) + '/')) {
      return error(res, 400, 'That file is outside the workspace.');
    }
    let st;
    try {
      st = statSync(target);
    } catch {
      return error(res, 404, "That file doesn't exist any more.");
    }
    if (!st.isFile()) return error(res, 400, "That's a folder, not a file.");
    const LIMIT = 200 * 1024;
    if (st.size > LIMIT) {
      return json(res, 200, { path: rel, tooLarge: true, sizeKb: st.size / 1024, text: '' });
    }
    let text = '';
    try {
      text = readFileSync(target, 'utf8');
    } catch {
      return json(res, 200, { path: rel, binary: true, sizeKb: st.size / 1024, text: '' });
    }
    // A NUL byte in the first block is the usual signal for a binary file.
    if (text.slice(0, 1024).includes('\u0000')) {
      return json(res, 200, { path: rel, binary: true, sizeKb: st.size / 1024, text: '' });
    }
    return json(res, 200, { path: rel, text, sizeKb: st.size / 1024, updatedAt: st.mtime.toISOString() });
  }

  // The folder a chat works in, and opening its files by path (notes-split-agent8.md).
  if (path === '/api/chat/folder' && method === 'GET') {
    const id = url.searchParams.get('sessionId') ?? '';
    return json(res, 200, { folder: chatFolders.info(id), default: workspaces[0]?.path ?? WORKSPACE });
  }
  if (path === '/api/chat/folder' && method === 'POST') {
    const body = await readBody(req);
    const id = String(body.sessionId ?? '');
    if (!id) return error(res, 400, 'sessionId is required');
    if (agent.isRunning(id)) return error(res, 409, 'This chat is still working. Stop it or wait for it to finish, then choose the folder.');
    if (body.path === null || body.path === '') { chatFolders.clear(id); return json(res, 200, { folder: null }); }
    try { return json(res, 200, { folder: chatFolders.set(id, body.path) }); } catch (e) { return error(res, 400, e.message); }
  }
  if (path === '/api/files/open' && method === 'POST') {
    const body = await readBody(req);
    try { return json(res, 200, openFile(chatRoot(String(body.sessionId ?? '')), body.path, String(body.how ?? 'app'))); }
    catch (e) { return error(res, e.code === 'outside' ? 403 : 400, e.message); }
  }
  /* Files and pictures dropped or pasted into a chat: multipart, a JSON data address, or raw bytes. A picture comes back with an id the next message names (imageIds). */
  if (path === '/api/chat/upload' && method === 'POST') {
    const sid = url.searchParams.get('sessionId') ?? '';
    if (!sid) return error(res, 400, 'sessionId is required');
    try { const body = await readRaw(req); return json(res, 200, { files: uploads.accept(sid, { body, contentType: req.headers['content-type'], name: url.searchParams.get('name') }) }); }
    catch (e) { return error(res, e.code === 'too-big' ? 413 : 400, e.message); }
  }
  const uploadGet = /^\/api\/chat\/upload\/([0-9a-f-]{36})$/.exec(path);
  if (uploadGet && method === 'GET') {
    const img = uploads.image(uploadGet[1]);
    if (!img) return error(res, 404, 'That picture is no longer available.');
    res.writeHead(200, { 'content-type': img.type, 'content-length': img.bytes.length, 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' }); res.end(img.bytes); return undefined;
  }
  if (path === '/api/files/preview' && method === 'GET') {
    const sid = url.searchParams.get('sessionId') ?? '';
    try { return json(res, 200, previewFile(chatRoot(sid), url.searchParams.get('path'), sid)); }
    catch (e) { return error(res, e.code === 'outside' ? 403 : 400, e.message); }
  }
  if (path === '/api/files/raw' && method === 'GET') {
    try { const f = rawFile(chatRoot(url.searchParams.get('sessionId') ?? ''), url.searchParams.get('path')); res.writeHead(200, { 'content-type': f.mime, 'content-length': f.bytes.length, 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' }); res.end(f.bytes); return undefined; }
    catch (e) { return error(res, e.code === 'outside' ? 403 : 400, e.message); }
  }
  /* A message too large for the model's window, made small enough to send: its text as a file in the chat's folder, or a shortened preview (oversize.mjs). */
  if (path === '/api/chat/oversize' && method === 'POST') {
    const body = await readBody(req, 12 * 1024 * 1024);
    const text = String(body.text ?? '');
    if (!text.trim()) return error(res, 400, 'text is required');
    let instance = null;
    try { instance = native.resolveInstance(body.nativeModelId ?? native.state().defaultId); } catch { /* no local model: the length alone decides */ }
    try { return json(res, 200, await prepareOversized({ mode: body.mode === 'file' ? 'file' : 'preview', text, root: chatRoot(String(body.sessionId ?? '')), instance })); }
    catch (e) { return error(res, 400, 'The message could not be prepared. Try a shorter message.'); }
  }
  /* "Attach as context": the file is named in the next message of the chat, so the assistant reads it again. */
  if (path === '/api/chat/attachments' && method === 'GET') return json(res, 200, { files: chatAttachments.list(url.searchParams.get('sessionId') ?? '') });
  if (path === '/api/chat/attachments' && method === 'POST') {
    const body = await readBody(req); const id = String(body.sessionId ?? '');
    try { const file = resolveChatFile(chatRoot(id), body.path); return json(res, 200, { files: body.remove === true ? chatAttachments.remove(id, file) : chatAttachments.add(id, file) }); }
    catch (e) { return error(res, e.code === 'outside' ? 403 : 400, e.message); }
  }
  if (path === '/api/files/path' && method === 'GET') {
    try { const p = resolveChatFile(chatRoot(url.searchParams.get('sessionId') ?? ''), url.searchParams.get('path')); return json(res, 200, { path: p }); }
    catch (e) { return error(res, e.code === 'outside' ? 403 : 400, e.message); }
  }
  if (path === '/api/workspaces' && method === 'GET') {
    return json(res, 200, { workspaces });
  }

  if (path === '/api/workspaces' && method === 'POST') {
    const body = await readBody(req);
    const target = resolve(String(body.path ?? '').trim().replace(/^~/, store.HOME));
    if (!target) return error(res, 400, 'Choose a folder first.');
    let st;
    try {
      st = statSync(target);
    } catch {
      return error(res, 400, "That folder doesn't exist.");
    }
    if (!st.isDirectory()) return error(res, 400, "That's a file, not a folder.");
    if (workspaces.some((w) => w.path === target)) {
      return error(res, 400, 'That folder is already in your workspaces.');
    }
    const ws = {
      id: `ws_${Math.random().toString(36).slice(2, 8)}`,
      name: basename(target) || target,
      path: target,
    };
    workspaces = [...workspaces, ws];
    saveWorkspaces();
    return json(res, 200, ws);
  }

  /* A workspace's name is only what the sidebar shows; its folder stays. */
  if (path.startsWith('/api/workspaces/') && method === 'PATCH') {
    const id = path.split('/')[3];
    const body = await readBody(req);
    const name = String(body.name ?? '').trim().slice(0, 80);
    if (!name) return error(res, 400, 'Type a name for this workspace.');
    const at = workspaces.findIndex(w => w.id === id);
    if (at < 0) return error(res, 404, 'That workspace is not in your list.');
    workspaces = workspaces.map((w, i) => (i === at ? { ...w, name } : w));
    saveWorkspaces();
    return json(res, 200, workspaces[at]);
  }

  if (path.startsWith('/api/workspaces/') && method === 'DELETE') {
    const id = path.split('/')[3];
    if (id === workspaces[0]?.id) {
      return error(res, 400, 'The first workspace cannot be removed.');
    }
    workspaces = workspaces.filter((w) => w.id !== id);
    saveWorkspaces();
    return json(res, 200, { ok: true });
  }

  // A native folder chooser. Asking someone who finds computers daunting to type
  // an absolute path is not a real option, so this borrows the system dialog.
  /* "Add from folder" promised to install a skill and, until now, cleared the
     search box instead — a button whose label and action had nothing to do with
     each other. This does the thing the label says: choose a folder, check it
     looks like a skill, copy it into the workspace's skills directory. */
  if (path === '/api/skills/add' && method === 'POST') {
    try {
      // The dialog is the normal path; a typed `path` in the body stands in where none exists.
      const typed = await typedFolder((await readBody(req)).path, { isAbsolute });
      const picked = typed ? { path: typed } : await chooseFolder('Choose a skill folder (it should contain SKILL.md)');
      if (picked.unavailable) {const u=chooserUnavailable(picked);return error(res,501,u.message,{code:u.code});};
      if (picked.cancelled) return json(res, 200, { cancelled: true });
      const chosen = picked.path;
      const base = basename(chosen);
      const mdPath = join(chosen, 'SKILL.md');
      if (!existsSync(mdPath)) {
        return error(res, 400, `That folder has no SKILL.md, so there is nothing for the model to read. A skill folder holds a SKILL.md with its instructions.`);
      }
      const dest = join(WORKSPACE, 'skills', base);
      if (existsSync(dest)) {
        return error(res, 400, `A skill called "${base}" is already in this workspace. Rename the folder, or remove the existing one first.`);
      }
      mkdirSync(join(WORKSPACE, 'skills'), { recursive: true });
      cpSync(chosen, dest, { recursive: true });
      return json(res, 200, { added: base });
    } catch (e) {
      // Dismissing the dialog is { cancelled: true } above; anything thrown here is a real problem.
      return error(res, 400, e.message);
    }
  }

  if (path === '/api/workspaces/pick' && method === 'POST') {
    try {
      const typed = await typedFolder((await readBody(req)).path, { isAbsolute });
      const picked = typed ? { path: typed } : await chooseFolder('Choose a folder for Strata');
      if (picked.unavailable) {const u=chooserUnavailable(picked);return error(res,501,u.message,{code:u.code});};
      if (picked.cancelled) return json(res, 200, { cancelled: true });
      return json(res, 200, { path: picked.path });
    } catch (e) {
      return error(res, 400, e.message);
    }
  }

  // --- agent --------------------------------------------------------------
  if (path === '/api/agent/status' && method === 'GET') return json(res, 200, agent.status());
  // The assistant's model requests, with the person's Response settings added (llm-shape.mjs). This computer only.
  const llmRoute = /^\/api\/internal\/agent-llm\/([A-Za-z0-9_.~-]+)(\/.*)$/.exec(path);
  if (llmRoute) { llmProxy(req, res, llmRoute[1], llmRoute[2] + (url.search ?? '')); return undefined; }
  // The goal bar (notes-split-agent3.md): the state, and the instant actions. Resume runs rounds, so it is a chat stream with {goalAction:'resume'}.
  if (path === '/api/goal' && method === 'GET') {
    try { return json(res, 200, { goal: await agent.goalState(url.searchParams.get('chatId') ?? '') }); } catch (e) { return error(res, 400, e.message); }
  }
  const goalAct = /^\/api\/goal\/(pause|edit|clear)$/.exec(path);
  if (goalAct && method === 'POST') {
    const body = await readBody(req);
    try { return json(res, 200, await agent.goalControl({ chatId: String(body.chatId ?? body.sessionId ?? ''), action: goalAct[1], text: body.text })); } catch (e) { return error(res, 400, e.message); }
  }
  // Plans (notes-split-agent6.md): the plan of a chat, and the instant actions. "Start as goal" streams, so it is /api/plan/start-goal below.
  if (path === '/api/plan' && method === 'GET') return json(res, 200, { plan: agent.planState(url.searchParams.get('sessionId') ?? url.searchParams.get('chatId') ?? '') });
  const planAct = /^\/api\/plan\/(approve|edit|reject)$/.exec(path);
  if (planAct && method === 'POST') {
    const body = await readBody(req);
    try { return json(res, 200, agent.planControl({ chatId: String(body.sessionId ?? body.chatId ?? ''), planId: typeof body.planId === 'string' ? body.planId : null, action: planAct[1], steps: body.steps, title: body.title })); }
    catch (e) { return error(res, e.code === 'plan-gone' || e.code === 'plan-used' ? 409 : 400, e.message); }
  }
  // Messages typed while a goal runs: queued (shown as queued), taken back, or listed.
  if (path === '/api/chat/queue' && method === 'GET') return json(res, 200, agent.takeQueue(url.searchParams.get('sessionId') ?? '', { release: url.searchParams.get('release') === '1' }));
  if (path === '/api/chat/queue' && method === 'POST') {
    const body = await readBody(req);
    try { return json(res, 200, agent.queueMessage({ chatId: String(body.sessionId ?? ''), text: body.message })); }
    catch (e) { return error(res, e.code === 'queue-idle' ? 409 : 400, e.message); }
  }
  if (path === '/api/chat/queue/cancel' && method === 'POST') {
    const body = await readBody(req);
    try { return json(res, 200, agent.cancelQueued({ chatId: String(body.sessionId ?? ''), id: String(body.id ?? '') })); }
    catch (e) { return error(res, e.code === 'queue-gone' ? 409 : 400, e.message); }
  }
  if (path === '/api/agent/tools' && method === 'GET') return json(res, 200, agentTools.view());
  if (path === '/api/agent/tools' && method === 'PUT') {
    try { return json(res, 200, agentTools.update(await readBody(req))); } catch (e) { return error(res, 400, e.message); }
  }
  if (path === '/api/agent/tools/custom' && method === 'POST') {
    try { return json(res, 200, agentTools.addCustom(await readBody(req))); } catch (e) { return error(res, 400, e.message); }
  }
  const customToolRoute = /^\/api\/agent\/tools\/custom\/([A-Za-z0-9_-]+)$/.exec(path);
  if (customToolRoute && method === 'PATCH') {
    try { return json(res, 200, agentTools.editCustom(customToolRoute[1], await readBody(req))); } catch (e) { return error(res, 400, e.message); }
  }
  if (customToolRoute && method === 'DELETE') {
    try { return json(res, 200, agentTools.removeCustom(customToolRoute[1])); } catch (e) { return error(res, 400, e.message); }
  }
  if (path === '/api/agent/pending' && method === 'GET') {
    const params = new URL(req.url, 'http://x').searchParams;
    // No chat named: every chat with an approval waiting (the task board lists them without opening each chat).
    if (!params.has('chatId')) return json(res, 200, { chats: agent.pendingAll() });
    return json(res, 200, { pending: agent.pendingFor(params.get('chatId') ?? '') });
  }
  /* Approve / Deny a tool the agent asked about. The answer reaches the Harness approval for exactly
     that chat and call; a stale or foreign id is refused. */
  if (path === '/api/chat/approval' && method === 'POST') {
    const b = await readBody(req);
    if (typeof b.sessionId !== 'string' || typeof b.toolId !== 'string' || !['approve', 'deny'].includes(b.decision)) return error(res, 400, 'sessionId, toolId and decision (approve or deny) are required');
    try { return json(res, 200, await agent.approve({ chatId: b.sessionId, callId: b.toolId, decision: b.decision })); }
    catch (e) { return error(res, e.code === 'approval-gone' ? 409 : 400, e.message); }
  }
  /* The person changed the chat's mode while a turn runs: it applies to the next tool call, goal and helpers included. */
  if (path === '/api/chat/mode' && method === 'POST') {
    const b = await readBody(req);
    if (typeof b.sessionId !== 'string' || !['manual', 'plan', 'bypass'].includes(b.mode)) return error(res, 400, 'sessionId and mode (manual, plan or bypass) are required');
    try { return json(res, 200, await agent.setMode(b.sessionId, b.mode)); } catch (e) { return error(res, e.code === 'mode-unavailable' ? 409 : 502, e.message); }
  }
  /* Stop one chat's turn. Resolves once the turn has really ended, not when the cancel was admitted. */
  if (path === '/api/chat/stop' && method === 'POST') {
    const b = await readBody(req);
    if (typeof b.sessionId !== 'string') return error(res, 400, 'sessionId is required');
    // A Time-skip trial of this chat is stopped by the same Stop button (and cleans up its copies).
    const trialId = trialByChat.get(b.sessionId);
    if (trialId) return json(res, 200, { stopped: await stopTrial(trialId), ended: 'time-skip' });
    try { return json(res, 200, await agent.stop(b.sessionId)); } catch (e) { return error(res, 502, e.message); }
  }

  // --- chat ---------------------------------------------------------------
  if ((path === '/api/chat/stream' || path === '/api/plan/start-goal') && method === 'POST') {
    const body = await readBody(req,12*1024*1024);
    if (path === '/api/plan/start-goal') { body.message = '/plan start'; if (typeof body.planId !== 'string') return error(res, 400, 'planId is required'); }
    let images;try{const named=Array.isArray(body.imageIds)&&body.imageIds.length?uploads.dataUrls(typeof body.sessionId==='string'?body.sessionId:'',body.imageIds):[];images=validateChatImages([...(Array.isArray(body.images)?body.images:[]),...named]);}catch(e){return error(res,400,e.message);}
    if(images.length&&body.remoteModel){const own=modelProviders.withSecrets().find(c=>c.id===body.remoteModel.connectionId);const row=own?.models.find(m=>m.id===body.remoteModel.modelId);const refused=await pictureRefusal(own,row);if(refused)return error(res,409,refused);}
    else if(images.length){let selected;try{selected=native.resolveInstance(body.nativeModelId??native.state().defaultId);}catch(e){return error(res,409,e.message);}if(!selected.vision)return error(res,409,'This model was started without picture reading. Open Models, turn on Read pictures and load it again, or choose a model that reads pictures.');}
    // The goal bar's Resume has no words of its own: it is the same turn as "/goal resume".
    const message = body.goalAction === 'resume' ? '/goal resume' : String(body.message ?? '').trim();
    if (!message) return error(res, 400, 'message is required');
    /* Which workspace the caller is in, so a skill narrowed to one project
       does not fire in another. */
    const workspaceId = typeof body.workspaceId === 'string' ? body.workspaceId : null;

    res.writeHead(200, {
      'content-type': 'application/x-ndjson; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
    });
    const controller = new AbortController();
    res.on('close', () => { if (!res.writableEnded) controller.abort(); });

    /* Agent turn, when the component is there and this request can use it. A client that disappears
       detaches from the turn (it keeps running in the Harness); Stop is its own request. */
    const chatId = typeof body.sessionId === 'string' && body.sessionId ? body.sessionId : null;
    const agentSkip = process.env.STRATA_AGENT === 'off' ? 'The agent is turned off (STRATA_AGENT=off).'
      : body.agent === false || body.tools === false ? 'This chat runs without the agent.'
      : !chatId ? 'This request did not name a chat.'
      : !(body.remoteModel || native.hasLive()) ? 'The agent needs a loaded Strata model or an API model.'
      : null;
    if (chatId && !agentSkip) {
      const capability = agent.status();
      if (capability.state !== 'unavailable') {
        const skipped = mcpSkippedNotices(capability.skippedServers);
        for (const notice of skipped) res.write(`${JSON.stringify({ type: 'notice', ...notice })}\n`);
        await runAgentTurn({ res, controller, chatId, body, message, workspaceId, images });
        res.end();
        return undefined;
      }
      res.write(`${JSON.stringify({ type: 'notice', code: 'agent-unavailable', message: `${capability.reason}`, ...(capability.detail ? { details: capability.detail } : {}) })}\n`);
    } else if (agentSkip && chatId && body.agent !== false && body.tools !== false && process.env.STRATA_AGENT !== 'off') {
      res.write(`${JSON.stringify({ type: 'notice', code: 'agent-skipped', message: agentSkip })}\n`);
    }

    // A plain chat turn says what it is doing too (reading the message, thinking with a token count, writing).
    const plain = withPlainStatus(frame => { if (!res.writableEnded && !res.destroyed) res.write(`${JSON.stringify(frame)}\n`); });
    res.on('close', () => plain.dispose());
    await runTurn({
      message,
      history: Array.isArray(body.history) ? body.history : [],
      onFrame: plain,
      signal: controller.signal,
      req,
      client: 'Strata chat',
      workspaceId,
      sessionId: typeof body.sessionId === 'string' ? body.sessionId : null,
      remoteModel: body.remoteModel ?? null,
      nativeModelId:body.nativeModelId,
      images,useTools:body.useTools===true,thinking:body.thinking??'high',thinkingBudget:Number.isInteger(body.thinkingBudget)&&body.thinkingBudget>0?Math.min(body.thinkingBudget,262144):0,
    });
    res.end();
    return undefined;
  }

  return error(res, 404, `No route for ${method} ${path}`);
}

/* -------------------------------------------------- OpenAI-compatible API --- */

async function openai(req, res, path, method) {
  if (!serving.authorized(req)) return error(res, 401, 'Invalid API key');
  if(native.state().instances.length){
    if(path==='/v1/models'&&method==='GET')return json(res,200,{object:'list',data:native.state().instances.filter(i=>i.status==='ready').map(i=>({id:i.apiModelId,object:'model',owned_by:'strata',meta:{instanceId:i.id,gpus:i.gpus,label:i.label,aliases:i.aliases}}))});
    if(method==='POST'&&['/v1/chat/completions','/v1/messages','/v1/responses'].includes(path)){if(store.get().serving?.pauseServing)return error(res,503,'New API requests are paused.');if(path==='/v1/responses'&&!native.state().capabilities.responses)return error(res,501,'Update the native Strata installation to use the Responses API.');return nativeGateway({req,res,path,body:await readBody(req,12*1024*1024),control:native,serving});}
  }
  if(path==='/v1/responses'&&method==='POST')return error(res,503,'Load a native Strata model to use the Responses API.');

  if (path === '/v1/models' && method === 'GET') {
    const runtime=runtimeState();
    return json(res,200,{object:'list',data:runtime.instances.map(i=>({id:i.apiModelId,object:'model',created:Math.floor(new Date(i.loadedAt).getTime()/1000),owned_by:'strata',meta:{label:i.family,context:i.memory.context,instanceId:i.id,gpuIds:i.gpuIds,mode:runtime.mode}}))});
  }

  if ((path === '/v1/chat/completions' || path === '/v1/messages') && method === 'POST') {
    if (store.get().serving?.pauseServing) return error(res, 503, 'New API requests are paused. Resume them in Serving.');
    const body = await readBody(req);
    const client = clientName(req);
    const anthropic = path === '/v1/messages';
    // An API client can say which workspace it is acting for; without it, a
    // skill narrowed to one project does not fire.
    const workspaceId = typeof body.workspaceId === 'string' ? body.workspaceId : null;

    const messages = anthropic
      ? [
          ...(body.system ? [{ role: 'system', content: body.system }] : []),
          ...(body.messages ?? []),
        ]
      : (body.messages ?? []);
    const last = [...messages].reverse().find((m) => m.role === 'user');
    const text =
      typeof last?.content === 'string'
        ? last.content
        : Array.isArray(last?.content)
          ? last.content.map((c) => c.text ?? '').join(' ')
          : '';

    if (!text) return error(res, 400, 'No user message found');

    runtimeState();
    let loaded;
    try{loaded=runtimeModels.resolve(body.model);}
    catch(err){return error(res,409,err.message);}
    loaded={...loaded,id:loaded.apiModelId};

    const id = uid('chatcmpl');
    const created = Math.floor(Date.now() / 1000);
    const wantsStream = body.stream === true;
    const started = Date.now();
    const controller = new AbortController();
    res.on('close', () => { if (!res.writableEnded) controller.abort(); });

    if (!wantsStream) {
      let out = '';
      let usage = { tokensIn: null, tokensOut: null };
      const result = await runTurn({
        message: text,
        runtimeModelId:loaded.apiModelId,
        history: messages.slice(0, -1),
        onFrame: (frame) => {
          if (frame.type === 'delta') out += frame.text;
          if (frame.type === 'done') usage = { tokensIn: frame.tokensIn, tokensOut: frame.tokensOut };
        },
        client,
        req,
        workspaceId,
        signal: controller.signal,
      });
      if (result?.error) return error(res, 500, result.error);
      return anthropic
        ? json(res, 200, {
            id,
            type: 'message',
            role: 'assistant',
            model: loaded.id,
            content: [{ type: 'text', text: out }],
            usage: { input_tokens: usage.tokensIn, output_tokens: usage.tokensOut },
          })
        : json(res, 200, {
            id,
            object: 'chat.completion',
            created,
            model: loaded.id,
            choices: [{ index: 0, message: { role: 'assistant', content: out }, finish_reason: 'stop' }],
            usage: { prompt_tokens: usage.tokensIn, completion_tokens: usage.tokensOut, total_tokens: Number.isFinite(usage.tokensIn) && Number.isFinite(usage.tokensOut) ? usage.tokensIn + usage.tokensOut : null },
          });
    }

    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
    });
    const write = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`);

    if (!anthropic) {
      write({ id, object: 'chat.completion.chunk', created, model: loaded.id, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] });
    } else {
      res.write(`event: message_start\ndata: ${JSON.stringify({ type: 'message_start', message: { id, role: 'assistant', model: loaded.id } })}\n\n`);
    }

    await runTurn({
      message: text,
      runtimeModelId:loaded.apiModelId,
      history: messages.slice(0, -1),
      workspaceId,
      onFrame: (frame) => {
        if (frame.type === 'delta') {
          if (anthropic) {
            res.write(`event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: frame.text } })}\n\n`);
          } else {
            write({ id, object: 'chat.completion.chunk', created, model: loaded.id, choices: [{ index: 0, delta: { content: frame.text }, finish_reason: null }] });
          }
        }
        if (frame.type === 'done' && !anthropic) {
          write({ id, object: 'chat.completion.chunk', created, model: loaded.id, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: frame.tokensIn, completion_tokens: frame.tokensOut } });
        }
        if (frame.type === 'error') {
          if (anthropic) res.write(`event: error\ndata: ${JSON.stringify({ type: 'error', error: { type: 'api_error', message: frame.message } })}\n\n`);
          else write({ error: { message: frame.message, type: 'server_error' } });
        }
      },
      signal: controller.signal,
      client,
      req,
    });

    if (anthropic) {
      res.write(`event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`);
    } else {
      res.write('data: [DONE]\n\n');
    }
    res.end();
    void started;
    return undefined;
  }

  return error(res, 404, `No route for ${method} ${path}`);
}

/* ---------------------------------------------------------------- static --- */

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.ico': 'image/x-icon',
  '.map': 'application/json; charset=utf-8',
  '.pdf': 'application/pdf',
  '.md': 'text/markdown; charset=utf-8',
};

async function staticFile(req, res, path) {
  if (!existsSync(DIST)) {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    res.end(
      `<!doctype html><meta charset="utf-8"><title>Strata host</title>
       <body style="font:16px/1.6 -apple-system,system-ui;background:#0e1113;color:#eef1f3;padding:48px">
       <h1 style="margin:0 0 8px">Strata host is running</h1>
       <p style="color:#8e979f">No built web app found at <code>${DIST}</code>.</p>
       <p>For development run <code>npm run dev</code> (Vite on :5199 proxies /api here).<br>
       For a single port run <code>npm run build:web &amp;&amp; npm run serve</code>.</p>
       <p style="color:#8e979f">API: <a style="color:#4ade80" href="/api/health">/api/health</a> ·
       <a style="color:#4ade80" href="/api/hardware">/api/hardware</a> ·
       <a style="color:#4ade80" href="/v1/models">/v1/models</a></p></body>`,
    );
    return;
  }

  const safe = normalize(path).replace(/^(\.\.[/\\])+/, '');
  let file = join(DIST, safe);
  if (!file.startsWith(DIST)) return error(res, 403, 'Forbidden');

  if (!existsSync(file) || statSync(file).isDirectory()) {
    if (extname(path)) return error(res, 404, 'Asset not found');
    const index = join(DIST, 'index.html');
    if (!existsSync(index)) return error(res, 404, 'Not found');
    file = index;
  }

  const type = MIME[extname(file)] ?? 'application/octet-stream';
  const size=statSync(file).size;
  const headers={'content-type':type,'cache-control':file.endsWith('index.html')?'no-store':'public, max-age=3600','accept-ranges':'bytes'};
  if(req.headers.range) {
    const match=/^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    const start=match?.[1]?Number(match[1]):match?.[2]?Math.max(0,size-Number(match[2])):NaN;
    const end=match?.[1]&&match[2]?Math.min(size-1,Number(match[2])):size-1;
    if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start>end||start<0||start>=size){res.writeHead(416,{'content-range':`bytes */${size}`});res.end();return;}
    res.writeHead(206,{...headers,'content-range':`bytes ${start}-${end}/${size}`,'content-length':end-start+1});
    if(req.method==='HEAD')res.end();else createReadStream(file,{start,end}).pipe(res);
    return;
  }
  res.writeHead(200,{...headers,'content-length':size});
  if(req.method==='HEAD')res.end();else createReadStream(file).pipe(res);
}

/* ------------------------------------------------------------------ start --- */

server.on('error', error => {
  if (server.listening) { console.error(`[strata] ${error.message}`); return; }
  // a taken or Windows-reserved port: one plain sentence instead of a stack trace
  console.error(`[strata] ${listenProblem(error, PORT, HOST)}`);
  process.exit(1);
});
server.listen(PORT, HOST, () => {
  if (process.send) process.send({ type: 'strata-ready', port: PORT });
  // Engines a lost owner left behind (an earlier crash, an older version): ended after a PID + start time + command-line check.
  if (control && process.env.STRATA_PACKAGED === '1') control.cleanup().catch(() => {});
});

let stopping=false;
const shutdown = async () => {
  if(stopping)return;stopping=true;
  clearInterval(taskTimer);tasks.close();
  const timeout=setTimeout(()=>process.exit(0),control?45000:10000);timeout.unref();   // stopping an engine through Control: drained (15 s), then ended; the shell's deadline is longer
  server.close();server.closeAllConnections?.();
  await Promise.allSettled([localShare.stop(),cloudflare.stop(),native.close(),agent.close()]);
  process.exit(0);
};
// Every ordinary way of being told to stop takes the clean path: Ctrl+C, kill, the terminal closing (SIGHUP), Ctrl+Break.
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK']) { try { process.on(signal, shutdown); } catch { /* not on this platform */ } }
// Started by the desktop shell: when the shell is gone (killed, crashed, logged off) stop what this app started and leave.
// POSIX re-parents us (ppid changes); Windows keeps the old ppid, so the pid is probed (signal 0 never signals).
if (process.env.STRATA_EXIT_WITH_PARENT === '1') {
  const parent = process.ppid;
  const gone = () => {
    if (process.ppid !== parent) return true;
    try { process.kill(parent, 0); return false; } catch (e) { return e.code === 'ESRCH'; }
  };
  setInterval(() => { if (gone()) void shutdown(); }, 1000).unref();
}

export { server, VARIANTS };
