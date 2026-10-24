#!/usr/bin/env node
/**
 * The context fixes against a REAL engine (not the scripted worker): Gyro-S at 64K on this PC, the agent bridge and the real Harness.
 *
 *   gpu + heavy locks held first (python phaseA-tests\lock.py acquire gpu "023-agent: context e2e", then heavy); one engine; nothing of the
 *   installed app is used or touched: its model files are read, its Harness is copied (STRATA_HARNESS_DIR), data goes to a scratch folder.
 *
 *   node scripts/context-e2e.mjs                       (needs STRATA_HARNESS_DIR; see the E2E_* names below)
 *
 * It starts serve/server.py as ITS OWN child (foreground: this script waits for it and stops it by PID at the end or on a memory alarm),
 * runs four scenarios and prints one table. Exit code 0 when every expectation held.
 *
 *   A  a message that can never fit (about 60K tokens in a 64K window) is refused in seconds, before the engine reads it
 *   B  a chat of dense log messages grows past the window several times over: automatic compaction runs, nothing is refused
 *   C  /compact at a high fill: summarises, says the figures; and a second /compact is answered in plain words
 *   D  the needle: a fact from the first message is asked about at the end (reported, not asserted: it is the summary's quality)
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { countTokens } from '../server/agent/context-fit.mjs';

const REPO = join(import.meta.dirname, '..', '..', '..');
const env = (name, fallback) => process.env[name] ?? fallback;
const DOCS = env('E2E_DOCS', 'C:\\Users\\AI-Server\\Documents\\Strata');
const ENGINE = env('E2E_ENGINE', 'C:\\Users\\AI-Server\\Desktop\\Strata\\GitHub\\wt-023-int\\build-cuda\\strata.exe');
const PROFILE = env('E2E_EXPERT_PROFILE', 'C:\\Users\\AI-Server\\AppData\\Local\\Programs\\Strata\\control\\data\\expert-profile.bin');
const PYTHON = env('E2E_PYTHON', 'python');
const PORT = Number(env('E2E_PORT', '18590'));
const MIN_FREE_GB = Number(env('E2E_MIN_FREE_GB', '7'));
const GGUF = join(DOCS, 'models', 'gyro-Gyro-S', 'Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf');
const CUDA_BIN = env('E2E_CUDA_BIN', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin');
const WINDOW = 65536;
/** A model server that is already running (the scripted worker, to try this script without a GPU): nothing is started or stopped. */
const EXISTING = process.env.E2E_BASE_URL ?? '';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const freeGb = () => {
  const out = spawnSync('powershell', ['-NoProfile', '-Command', '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)'], { encoding: 'utf8' }).stdout.trim();
  return Number(out);
};
const rows = [];
const note = (id, ok, detail) => { rows.push({ id, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${id}  ${detail}`); };

if (!process.env.STRATA_HARNESS_DIR) { console.error('STRATA_HARNESS_DIR is not set'); process.exit(2); }
if (!EXISTING) for (const f of [ENGINE, GGUF]) if (!existsSync(f)) { console.error(`missing: ${f}`); process.exit(2); }
if (!EXISTING && freeGb() < 20) { console.error(`only ${freeGb()} GB of RAM are free; wait for the other streams (lock.py status) and run again`); process.exit(2); }

const scratch = mkdtempSync(join(tmpdir(), 'strata-ctx-e2e-'));
const configFile = join(scratch, 'server.json');
const logFile = join(scratch, 'server.log');
const readyFile = join(scratch, 'ready.json');
writeFileSync(configFile, JSON.stringify({
  exe: ENGINE,
  args: ['--pack', join(DOCS, 'packs', 'gyro-gyro-s'), '--native', GGUF, '--ple-gguf', GGUF, ...(existsSync(PROFILE) ? ['--expert-profile', PROFILE] : []),
    '--prefill', 'auto', '--spec-min-p', '0.5', '--mtp', join(DOCS, 'mtp', 'rt'), '--max-context', String(WINDOW), '--kv', 'int8', '--pool-workers', '5',
    '--expert-cache', 'auto', '--kv-resident', '32768', '--spec', '4', '--vram-reserve-mib', '1536'],
  cwd: REPO, tokenizer: join(DOCS, 'packs', 'gyro-gyro-s', 'tokenizer'), model_name: 'qwen3.8-flash-next-gyro-gyro-s', log: join(scratch, 'engine.log'), lib_dirs: [], port: PORT, gpu: 0,
}, null, 2));

let server = null, bridge = null, alarm = null;
const stopServer = () => {
  if (server?.pid) spawnSync('taskkill', ['/T', '/F', '/PID', String(server.pid)], { stdio: 'ignore' });
  server = null;
};
const finish = async (code) => {
  clearInterval(alarm);
  try { await bridge?.close(); } catch { /* going away */ }
  stopServer();
  await sleep(1500);
  console.log(`\nscratch folder: ${scratch}`);
  process.exit(code);
};
process.on('SIGINT', () => finish(130));

if (!EXISTING) server = spawn(PYTHON, ['serve/server.py', '--engine', 'strata', '--config', configFile, '--port', String(PORT), '--no-api-key', '--ready-file', readyFile], {
  cwd: REPO, env: { ...process.env, PATH: `${CUDA_BIN};${process.env.PATH}`, STRATA_DATA_DIR: join(scratch, 'data') }, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
});
const sink = [];
if (server) { server.stdout.on('data', d => sink.push(String(d))); server.stderr.on('data', d => sink.push(String(d))); server.on('exit', code => { sink.push(`[server exited ${code}]`); server = null; }); }
// Memory alarm: this run must never be what freezes the PC. Under the floor the engine is stopped (by PID) and the run ends.
if (!EXISTING) alarm = setInterval(() => { const free = freeGb(); if (free < MIN_FREE_GB) { console.error(`ALARM: ${free} GB of RAM free, stopping the engine`); finish(3); } }, 3000);
alarm?.unref?.();

const base = EXISTING || `http://127.0.0.1:${PORT}`;
let ready = false;
for (let i = 0; i < 120 && !ready; i++) {
  if (!EXISTING && !server) break;
  try { ready = (await fetch(`${base}/health`, { signal: AbortSignal.timeout(2000) })).ok; } catch { /* still loading */ }
  if (!ready) await sleep(2000);
}
writeFileSync(logFile, sink.join(''));
if (!ready) { console.error('the engine did not become ready:\n' + sink.join('').slice(-1500)); await finish(4); }
const model = (await (await fetch(`${base}/v1/models`)).json()).data[0].id;
console.log(`engine ready: ${model}, free RAM ${freeGb()} GB`);

// ---------------------------------------------------------------------------------------------------------------- the pieces

const line = i => `2026-10-22 13:${String(Math.floor(i / 60) % 60).padStart(2, '0')}:${String(i % 60).padStart(2, '0')} INFO svc-${i % 17} req=${(i * 2654435761 >>> 0).toString(16)} status=${200 + (i % 7) * 3} ms=${(i * 37) % 900} path=/api/v1/items/${(i * 7919) % 100000}`;
const logText = (chars, from = 0, needle = '') => {
  const out = [];
  let n = 0, i = from;
  while (n < chars) { const l = line(i++); out.push(l); n += l.length + 1; }
  if (needle) out.splice(Math.floor(out.length / 2), 0, needle);
  return out.join('\n');
};
const count = text => countTokens({ baseUrl: `${base}/v1`, model, text });

const turnsOf = async (message) => {
  const frames = [];
  const t0 = Date.now();
  await bridge.runTurn({ chatId: 'e2e', message, workspacePath: scratch, mode: 'bypass', thinking: 'Off', onFrame: f => frames.push(f) });
  return { frames, seconds: (Date.now() - t0) / 1000, last: frames.at(-1) };
};
const textOf = frames => frames.filter(f => f.type === 'delta' && !f.noteReason).map(f => f.text).join('').trim();
const noteOf = frames => frames.find(f => f.type === 'delta' && f.noteReason === 'compacted')?.text?.trim() ?? null;

bridge = createAgentBridge({
  dataDir: join(scratch, 'agent'), env: process.env, startTimeoutMs: 120_000, log: l => console.log('   bridge:', l),
  localInstances: () => [{ id: 'e2e', label: 'Gyro-S', baseUrl: `${base}/v1`, modelId: model, context: WINDOW, slots: 1, key: '' }], connections: () => [],
});
mkdirSync(join(scratch, 'agent'), { recursive: true });

// ---------------------------------------------------------------------------------------------------------------- A
{
  // Grow a log until the engine itself counts about 60K tokens.
  let chars = 150_000, tokens = await count(logText(chars));
  while (tokens && tokens < 58_000 && chars < 400_000) { chars = Math.round(chars * (60_000 / tokens)); tokens = await count(logText(chars)); }
  const t = await turnsOf(logText(chars) + '\n\nWhat is the most common status?');
  const refused = t.last.type === 'error' && t.last.code === 'CONTEXT_MESSAGE_TOO_LARGE';
  note('A  too large is refused before it is read', refused && t.seconds < 15, `${chars} characters = ${tokens} tokens by the engine; answered in ${t.seconds.toFixed(1)} s; frame ${t.last.type}/${t.last.code ?? '-'}: ${String(t.last.message ?? '').slice(0, 110)}`);
}

// ---------------------------------------------------------------------------------------------------------------- B
const needle = 'NOTE FROM OPS: the vault code is COBALT-27. Do not lose it.';
const timeline = [];
let notes = 0, refusedAny = false, peak = 0;
{
  for (let k = 0; k < 5; k++) {
    const body = logText(34_000, k * 1000, k === 0 ? needle : '');
    const t = await turnsOf(`${body}\n\nReply with the single word OK.`);
    const note = noteOf(t.frames);
    if (note) notes++;
    if (t.last.type === 'error') refusedAny = true;
    const used = t.last.contextTokens ?? 0;
    peak = Math.max(peak, ...t.frames.filter(f => f.type === 'context').map(f => f.contextTokens), used);
    timeline.push(`msg${k + 1}: ${t.last.type}${t.last.code ? '/' + t.last.code : ''} ${t.seconds.toFixed(0)} s context ${used} (${Math.round(100 * used / WINDOW)}%)${note ? ' [' + note.slice(0, 80) + ']' : ''}`);
    console.log('   ' + timeline.at(-1));
    if (t.last.type === 'error') break;
  }
  note('B  five dense messages (about 14K tokens each) in a 64K window: no error', !refusedAny, `peak context ${peak} of ${WINDOW}; ${notes} compaction note(s) in the chat`);
  note('B  compaction ran and was announced', notes >= 1, timeline.join(' | '));
}

// ---------------------------------------------------------------------------------------------------------------- C
{
  const run = async (line) => { const frames = []; await bridge.compactCommand({ chatId: 'e2e', message: line, workspacePath: scratch, mode: 'bypass', choice: null, onFrame: f => frames.push(f) }); return frames; };
  await turnsOf(`${logText(30_000, 90_000)}\n\nReply with the single word OK.`);
  const t0 = Date.now();
  const first = await run('/compact');
  const said = first.find(f => f.type === 'delta');
  const ok = said?.key === 'Older messages were summarized to make room. The chat was {0}% full and is now {1}% full.' && said.values[0] > said.values[1];
  note('C  /compact summarises and states the figures', ok, `${((Date.now() - t0) / 1000).toFixed(0)} s: ${said?.text}`);
  const second = await run('/compact');
  const again = second.find(f => f.type === 'delta')?.text ?? '';
  note('C  a second /compact is answered in plain words', /summarized|nothing|could not/i.test(again) && !/\{|400|invalid_request/.test(again), again);
}

// ---------------------------------------------------------------------------------------------------------------- D
{
  const t = await turnsOf('What was the vault code in the very first message? Answer with the code only.');
  const answer = textOf(t.frames);
  console.log(`D  needle after compaction: "${answer.slice(0, 80)}"  (${t.last.type})`);
  rows.push({ id: 'D  the needle', ok: /COBALT-27/.test(answer), detail: `reported only: answer "${answer.slice(0, 80)}"` });
}

console.log('\n' + rows.map(r => `${r.ok ? 'PASS' : r.id.startsWith('D') ? 'INFO' : 'FAIL'}  ${r.id}${r.id.startsWith('D') ? `  ${r.detail}` : ''}`).join('\n'));
const failed = rows.filter(r => !r.ok && !r.id.startsWith('D')).length;
writeFileSync(join(scratch, 'result.json'), JSON.stringify(rows, null, 2));
await finish(failed ? 1 : 0);
