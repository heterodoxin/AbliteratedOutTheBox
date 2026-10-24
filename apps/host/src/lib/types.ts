/** Shared shapes for the Strata host API. Mirrors server/types.d.ts. */

export type CheckState = 'pass' | 'warn' | 'fail';
export interface SoftwareUpdate {
  installedVersion?: string;
  status: 'available' | 'current' | 'unavailable' | 'error' | 'no-releases';
  latestVersion?: string;
  releaseUrl?: string;
  canInstall: boolean;
  message: string;
}
export interface UpdateCheckResult { checkedAt: string; strata: SoftwareUpdate; engine: SoftwareUpdate }

export interface HardwareCheck {
  id: 'gpu' | 'memory' | 'cpu' | 'storage';
  label: string;
  state: CheckState;
  detail: string;
  value?: string;
}

export interface ProcessorInfo { id:string; socket:string; model:string; cores:number; threads:number; arch:string; ghz:number; logicalIds?:number[] }
export interface CpuReading { id:string; utilizationPct:number|null }
export interface GpuReading { id:string; temperatureC:number|null; utilizationPct:number|null; memoryUsedGb:number|null; source:string; sampledAt:string; powerW?:number|null; powerLimitW?:number|null; pcieGen?:number|null; pcieGenMax?:number|null; pcieWidth?:number|null; pcieRxMbs?:number|null }

export interface GpuInfo {
  id?: string;
  index: number;
  vendor: 'apple' | 'nvidia' | 'amd' | 'intel' | 'unknown';
  name: string;
  vramGb: number;
  freeVramGb?: number;
  unified: boolean;
  backend: 'metal' | 'cuda' | 'rocm' | 'vulkan' | 'cpu';
  driver?: string;
}

export interface Hardware {
  probedAt: string;
  host: { name: string; platform: string; release: string; arch: string; app: string };
  processors?: ProcessorInfo[];
  cpu: { topologyKnown?:boolean; model: string; cores: number; threads: number; arch: string; ghz: number };
  memory: { totalGb: number; freeGb: number; usedGb: number; pressure: 'low' | 'medium' | 'high' };
  disk: { path: string; totalGb: number; freeGb: number; modelsDir: string };
  gpus: GpuInfo[];
  accelerator: { backend: string; label: string; usableVramGb: number; unified: boolean };
  checks: HardwareCheck[];
  verdict: { ok: boolean; headline: string; detail: string };
  recommendedVariantId: string;
}

export type VariantState =
  | 'available'
  | 'downloading'
  | 'installed'
  | 'loading'
  | 'loaded'
  | 'unloading'
  | 'error';

export interface ModelVariant {
  local?: { path: string; files: { path: string; bytes: number; mtimeMs: number; ctimeMs: number }[]; architecture: string; contextLength: number; hasChatTemplate: boolean; shards: number; metadataSignature: string };
  source?: { repo: string; file: string; bytes?: number };
  downloadUrl?: string;
  kvKbPerToken?: number;
  catalog?: { family: string; version: string; purpose: string; capabilities: string[]; agentApproved: boolean; balancedRank: number; review: string };
  performance?: import('./experience').SpeedEvidence[];
  id: string;
  name: string;
  family: string;
  quant: string;
  label: string;
  params: string;
  blurb: string;
  role: 'best-fit' | 'option';
  ramClass: { minGb: number; maxGb: number; label: string };
  diskGb: number;
  downloadGb: number;
  contextMax: number;
  layerCount: number;
  experts: number;
  hasDraft: boolean;
  kvModes: string[];
  tags: string[];
  state: VariantState;
  progress?: { pct: number; mbps: number; etaSec: number; stage: string };
  error?: string;
  installedAt?: string;
}

export interface LocalModelPreview { variant: ModelVariant; files: string[]; warnings: string[] }
export interface LocalModelFolder { directory: string; parent: string | null; entries: { name: string; path: string; kind: 'folder' | 'model'; bytes?: number }[]; truncated: boolean; shortcuts: { name: string; path: string }[] }

export interface ModelLibrary {
  mode?: 'real' | 'simulated';
  current: string | null;
  loaded: string | null;
  variants: ModelVariant[];
  downloads: { active: number; queued: number; totalGbToday: number };
  disk: { modelsDir: string; usedGb: number; freeGb: number };
}

export type ImageModelChoice = 'off' | 'q4' | 'q5' | 'q6' | 'q8';
export interface ImageModelVariant {
  id: Exclude<ImageModelChoice, 'off'>;
  label: string;
  quant: string;
  file: string;
  bytes: number;
  downloadGb: number;
  downloadUrl: string;
  state: 'available' | 'downloading' | 'installed' | 'error';
  progress?: { pct: number; stage: string };
  error?: string;
}
export interface ImageModelLibrary {
  mode: 'real' | 'simulated';
  family: string;
  sourceUrl: string;
  revision: string;
  selected: ImageModelChoice;
  applied: ImageModelChoice;
  generationAvailable: boolean;
  variants: ImageModelVariant[];
}

export type ModelProtocol = 'openai-chat' | 'openai-responses' | 'anthropic';
export interface ApiModel { id: string; name: string; contextMax?: number | null; /** What the person ticked ("This model reads pictures"): true, false, or null for not set. */ vision?: boolean | null; /** The answer in use: the tick, else what the server said; null when nobody knows. */ reads?: boolean | null; /** The provider says the model can think (OpenRouter's supported_parameters): the Reasoning menu then works. */ reasoning?: boolean | null; thinks?: boolean | null }
export interface ModelProviderPreset { id: string; name: string; baseUrl: string; protocol: ModelProtocol; baseUrlHint?: string }
export interface ModelConnection {
  id: string; kind: 'provider' | 'custom'; provider: string | null; name: string;
  baseUrl: string; protocol: ModelProtocol; apiKeySet: boolean; models: ApiModel[];
}
export interface ModelProviders { catalog: ModelProviderPreset[]; connections: ModelConnection[] }
export interface RemoteModelSelection { connectionId: string; modelId: string }
export interface ModelConnectionDraft {
  id?: string; kind: 'provider' | 'custom'; provider: string | null; name: string;
  baseUrl: string; protocol: ModelProtocol; apiKey: string; clearKey?: boolean; models: ApiModel[];
}

export type Effort = 'fastest' | 'balanced' | 'quality' | 'max';

export interface BeginnerConfig {
  priority: 'speed' | 'balanced' | 'quality' | 'coding';
  contextLength: 'recommended' | `${number}k`;
  readImages: boolean;
  gpuIdle: 'never' | '5m' | '15m' | '1h';
}

export interface ExpertParam {
  applies?: string;
  id: string;
  group: string;
  label: string;
  effect: string;
  type: 'select' | 'number' | 'text' | 'toggle';
  options?: string[];
  value: string | number | boolean;
  recommended: string | number | boolean;
  advanced?: boolean;
}

export interface EngineConfig {
  extra?: Record<string, unknown>;
  variantId: string;
  beginner: BeginnerConfig;
  expert: ExpertParam[];
  predicted: {
    responseSpeed: string;
    memory: string;
    restartNeeded: boolean;
    notes: string[];
  };
  dirty: boolean;
  appliedAt?: string;
  engine: {
    status?: string;
    error?: string | null;
    port?: number;
    running: boolean;
    pid?: number;
    uptimeSec: number;
    version: string;
    backend: string;
    startedAt?: string;
  };
}

export interface RuntimeModel {
  id:string; apiModelId:string; variantId:string; family:string; name:string; quant:string;
  gpuIds:string[]; loadedAt:string; status:'ready'; phase:MonitorState['phase'];
  active:number; queued:number; prompt:MonitorState['prompt']; outputTps:number; promptTps:number;
  context:MonitorState['context'];
  memory:{context:number;ramGb:number;gpuGb:number;spillGb:number;reserveGb:number;totalGb:number;devices:{id:string;gb:number}[]};
}
export interface RuntimeModels {
  revision:number; mode:'real'|'simulated'; capabilities:{multipleModels:boolean;deviceAssignment:boolean}; defaultId:string|null;
  instances:RuntimeModel[];
  memory:{ramGb:number;budgetGb:number;reserveGb:number;totalGb:number;fits:boolean;devices:{id:string;name:string;usedGb:number;budgetGb:number}[]};
}

export interface ServeRequest {
  instanceId?:string;
  gpuIds?:string[];
  id: string;
  at: string;
  client: string;
  endpoint: string;
  state: 'generating' | 'queued' | 'done' | 'error';
  tokensIn: number | null;
  tokensOut: number | null;
  cancellable?: boolean;
  cancelRequested?: boolean;
  usageSource?: 'reported' | 'demo' | 'unavailable' | 'estimated';
  durationMs: number;
  error?: string;
  model: string;
  cachedTokens?: number | null;
  tps?: number | null;
  promptTps?: number | null;
  hitRate?: number | null;
  pcieShare?: number | null;
  finishReason?: string | null;
}

export interface TunnelState { status: 'idle' | 'starting' | 'connected' | 'stopping' | 'error'; mode: 'quick' | 'domain' | null; url: string | null; error: string | null; warning?: string | null; installed: boolean; originUrl: string; connector?: { path: string; version: string | null; source: string; label: string; summary: string } | null; external?: { kind: 'service' | 'process'; others: number[]; own: number[]; message: string } | null; }

export interface ServingState {
  tunnel?: TunnelState;
  sharing?: { active: boolean; url: string | null; warning?: string | null };
  revision?: number;
  running: boolean;
  baseUrl: string;
  port: number;
  endpoints: { openai: string; anthropic: string; models: string };
  access: { otherDevices: boolean; apiKeySet: boolean; apiKeyHint: string | null; allowOrigin: string };
  stats: { running: number; waiting: number; completed: number; failed: number; tokensOut: number };
  requests: ServeRequest[];
  pauseServing: boolean;
}

export interface MonitorSample {
  /** Device IDs covered by aggregate GPU history; omitted for legacy host-wide samples. */
  gpuIds?: string[];
  t: number;
  outputTps: number | null;
  promptTps: number | null;
  gpuLoad: number | null;
  vramUsedGb: number | null;
  ramUsedGb: number | null;
  gpuTempC: number | null;
  diskReadGbs: number | null;
  cpuLoad?: number | null;
  powerW?: number | null;
  pcieRxMbs?: number | null;
  gpus?: GpuReading[];
  cpus?: CpuReading[];
}

export interface MonitorState {
  telemetry?: { throughput: string; thermals: string; memory: string; system?:string; context?: string; ttftMs?: number | null };
  phase: 'idle' | 'reading' | 'thinking' | 'generating' | 'loading' | 'error';
  phaseLabel: string;
  variantId: string | null;
  prompt: { done: number; total: number; pct: number };
  generation?: { done:number|null; total:number|null };
  outputTps: number;
  promptTps: number;
  queue: { running: number; waiting: number };
  context: {
    requestId?: string;
    sessionId?: string;
    usedTokens: number;
    maxTokens: number;
    pct: number;
    parts?: { system: number; tools: number; messages: number };
  };
  expertCache: { used: number; total: number; memoryGb?:number|null; source?:string };
  hardware: {
    gpus?: GpuReading[];
    cpus?: CpuReading[];
    cpuLoad?: number | null;
    powerW?: number | null;
    powerLimitW?: number | null;
    pcieRxMbs?: number | null;
    diskReadGbs?: number | null;
    diskWriteMbs?: number | null;
    gpuLoad: number | null;
    vramUsedGb: number | null;
    vramTotalGb: number;
    ramUsedGb: number | null;
    ramTotalGb: number;
    gpuTempC: number | null;
    tempLimitC: number;
  };
  history: MonitorSample[];
  requests?: ServeRequest[];
  conversationCache?: { enabled?:boolean; parked?:number; slots?:number; memoryGb?:number; budgetGb?:number; parks?:number; restores?:number; evictions?:number; lastEvent?:string; lastAt?:number; lastTokens?:number; requests?:number;requestsReused?:number;lastPrompt?:number;lastReused?:number; source:string };
  mcp?: { servers:{ name:string; state:string; transport?:string; info?:{name?:string;version?:string}; error?:string; tools:{name:string;description?:string}[] }[]; tools:number };
  totals: {
    promptMs?: number;
    decodeMs?: number;
    usageRequests?: number;
    latencyRequests?: number;
    tokensIn: number;
    tokensOut: number;
    requests: number;
    avgTps: number;
    since: string;
    ttftMs: number;
    cachedTokens: number;
    uncachedTokens: number;
  };
}

export interface ChatMessage {
  /** The person's message is written with Markdown (a task handover: bold headings and lists), so the bubble draws it as such. */
  markdown?: boolean;
  photos?:import('./chat-photos').ChatPhoto[];
  /** Reply came from a saved API connection, rather than the local engine. */
  remoteModel?: boolean;
  timeSkipId?: string;
  /** The plan this reply proposed: shown as the plan card in place of the reply text. */
  planId?: string;
  /** The reply in the order it came: text and grouped activity lines (round 8, item 12). Absent on older replies. */
  blocks?: import('./deliverables').MessageBlock[];
  /** The saved skills this reply used (the line under the answer), in the order they were loaded. */
  skills?: { name: string; description: string }[];
  /** Files the assistant made or changed in this reply, one card each. */
  deliverables?: import('./deliverables').Deliverable[];
  /** A line from the app itself (for example from /loop), never sent to the model. */
  local?: boolean;
  ttftMs?: number;
  example?: boolean;
  media?: {kind: 'image' | 'video' | 'gif'; src:string; title:string; caption?:string; poster?:string}[];
  artifacts?: import('./artifacts').PreviewArtifact[];
  requestId?: string;
  model?: string;
  inputTokens?: number;
  /** What the context held after this reply (the last model request's prompt + reply); an agent turn's inputTokens add every request up. */
  contextUsed?: number;
  /** When the first reasoning arrived, and how long the model reasoned before it began the answer (for "Thought for 38s"). */
  reasoningStartedAt?: number;
  reasoningMs?: number;
  contextMax?: number;
  usageSource?: 'reported' | 'demo' | 'unavailable' | 'estimated';
  skillsUsed?: string[];
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  at: string;
  streaming?: boolean;
  demo?: boolean;
  reasoning?: string;
  reasoningOpen?: boolean;
  tokens?: number;
  tps?: number;
  promptTps?: number;
  durationMs?: number;
  error?: string;
  /** What went wrong, as a code the page acts on (a context error offers its own recovery buttons). */
  errorCode?: string;
  /** The engine's own text of the error, kept behind Details. */
  errorDetail?: string;
  tools?: ToolCall[];
}

export type ToolState = 'proposed' | 'approved' | 'running' | 'done' | 'denied' | 'error';

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  preview: string;
  effect: string;
  scope: 'read' | 'write' | 'shell' | 'connected';
  /** The card's headline in plain words for its current state ("Wrote a file"); the host sends it, the UI translates it. */
  verb?: string;
  /** What the action is on: a file path, the command, a web address. */
  target?: string;
  /** The question an approval card asks in plain words (server/agent/ask.mjs); the page translates it. */
  ask?: { id: string; params?: (string | number)[]; why?: string; detail?: string };
  state: ToolState;
  result?: string;
  durationMs?: number;
  /** Reported by the agent runtime: an approval answers a real request. Absent for design-build examples. */
  live?: boolean;
  /** The person approved this call. A finished card without it was never asked about. */
  approved?: boolean;
}

/** A folder the model may read, and that sessions are grouped under. */
export interface Workspace {
  id: string;
  name: string;
  path: string;
  /** Earlier folders whose chats belong here (the first workspace moved off the app's own folder). */
  aliases?: string[];
}

export interface SessionAgent {
  id: string;
  name: string;
  description: string;
  status: 'working' | 'completed' | 'failed' | 'queued' | 'cancelled';
  durationMs: number;
  tokens: number;
  /** Time the latest duration measurement was received. */
  updatedAt?: number;
}

export type ChatMode = 'auto' | 'manual' | 'accept-edits' | 'plan' | 'bypass';

/**
 * What the board needs to know about a /loop after the window closes: loops live in memory, so the chat keeps a small record of the
 * last one (running, or how it ended). A record that says `running` with no loop behind it is a loop the window closed on.
 */
export interface LoopRecord {
  prompt: string;
  everyMs: number;
  startedAt: number;
  state: 'running' | 'ended';
  answers: number;
  maxRuns?: number;
  /** When the next answer is due (ms), while running. */
  nextAt?: number;
  endedAt?: number;
  /** count, said-done, time, failures, cap; `stopped` when the person typed /loop stop; `closed` when the window closed on it. */
  endedWhy?: 'count' | 'said-done' | 'time' | 'failures' | 'cap' | 'stopped' | 'closed';
  /** The last answer's first words, for the card. */
  lastText?: string;
}

export interface Session {
  /** The person dismissed this chat's (or its loop's) card on the board at this time. A later problem brings it back. */
  ackedAt?: string;
  loop?: LoopRecord;
  timeSkips?: import('./time-skip').TimeSkipRun[];
  showcase?: boolean;
  agents?: SessionAgent[];
  previewOpen?: string;
  closedPreviews?: string[];
  showcaseAgentStates?: boolean;
  preferences?: { access?: 'read' | 'write' | 'full'; thinking?: string; thinkingBudget?: number; mode?: ChatMode; useTools?:boolean; /** false: plain conversation, no tools of any kind. */ agent?: boolean; /** This chat's API model; absent or null means the local model. */ remoteModel?: RemoteModelSelection | null };
  /** A branch does not have retained per-turn input usage. */
  statsAvailable?: boolean;
  id: string;
  title: string;
  defaultTitle?: boolean;
  updatedAt: string;
  createdAt: string;
  messages: ChatMessage[];
  workspace: string;
  tokensIn: number;
  tokensOut: number;
  pinned?: boolean;
  /** Put away without deleting. Archiving is for "done with this for now" —
   *  deleting is for "never want to see it again", and conflating the two makes
   *  people keep things they are finished with. */
  archived?: boolean;
  /** When this conversation was last on screen. A session is unread when it has
   *  been updated since — which is the only way to tell, with more than a
   *  handful of conversations, which ones have something new in them. */
  lastReadAt?: string;
}

export interface Skill {
  id: string;
  name: string;
  description: string;
  source: 'bundled' | 'workspace' | 'user';
  enabled: boolean;
  triggers: string[];
  /** The instruction text itself — what the model is actually told. Deciding
   *  whether to trust a skill means reading this, not its name. */
  body?: string;
  /** When set, the skill applies only in this workspace. A skill written for
   *  one project is noise in another, and on/off cannot express "here, not
   *  there". */
  onlyIn?: string | null;
  /** How many times a prompt has actually triggered this skill, and when last. */
  useCount?: number;
  lastUsedAt?: string | null;
  sizeKb: number;
}

export interface WorkspaceFile {
  path: string;
  kind: 'file' | 'dir';
  sizeKb?: number;
  updatedAt?: string;
}

export interface SetupState {
  flowVersion?: number;
  dismissed?: boolean;
  choices?: { onboarding?: { step:number; family?:string; size?:string; context?:number; vision?:boolean; directory?:string; profile?:'work'|'performance'; installJobId?:string; tuneJobId?:string; calibrate?:boolean; manual?:boolean }; variantId?: string; contextLength?: BeginnerConfig["contextLength"]; profile?: import("./experience").Profile; batch?: number; batchAuto?: boolean };
  complete: boolean;
  step: number;
  completedAt?: string;
  history: { step: number; at: string }[];
  app: { version: string; modelsDir: string; dataDir: string; modelsDirNotice?: string };
}

export interface EngineEvent {
  type: 'engine' | 'monitor' | 'models' | 'serving' | 'download' | 'log';
  at: string;
  payload: unknown;
}
