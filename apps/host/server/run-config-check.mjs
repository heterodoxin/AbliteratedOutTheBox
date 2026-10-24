/**
 * The run config of a model (strata-<model>.json in the installation folder) is the text file a power user can edit by hand:
 * every engine option goes in "args", every STRATA_* environment flag in "env", the rest are named keys. Changes apply the next
 * time the model starts. This checks the file BEFORE the model starts and says what is wrong in plain words, with the line, so a
 * typo is never a start that just fails.
 *
 * Errors (the start is held back until they are fixed): the file is not valid JSON, or a key the app knows has a value it cannot use.
 * Warnings (the start goes on): a key nothing reads (probably a typo; the nearest real name is offered), an option the app manages itself.
 *
 * Each issue is { level, code, key, line, params, message }: `code` and `params` are what the screen translates (src/lib/run-config-issues.ts),
 * `message` is the same sentence in English for logs and the Details of a refused start.
 */
import { NATIVE_FIELDS } from './native-settings.mjs';

/** Every top-level key a run config may carry: what setup writes, what serve/server.py reads, the app's own fields. */
export const KNOWN_KEYS = new Set([
  ...NATIVE_FIELDS.map(f => f.id),
  'exe', 'args', 'cwd', 'tokenizer', 'model_name', 'model', 'arch', 'log', 'lib_dirs', 'port', 'host', 'api_key', 'cuda', 'backend', 'gpu', 'gpus_asked',
  'gpu_policy', 'hip_ordinal', 'env', 'env_to_user', 'sampling', 'mcp_servers', 'mcpServers', 'before_load', 'before_load_ignore_failure',
  'before_load_strict', 'before_load_timeout_s', 'cors_origins', 'trusted_origins', 'allowed_hosts', 'open_browser', 'prevent_sleep',
  'max_queue', 'queue_timeout', 'max_tokens', 'min_tokens', 'answer_room', 'threads', 'replicas', 'replica_cpus', 'replica_load_tokens',
  'elastic', 'elastic_idle_s', 'elastic_mib', 'min_free_vram_mib', 'vram_segment_mib', 'slot_save_path', 'telemetry_read_key', 'batch_telemetry',
  'reasoning_effort', 'reasoning_close_retry', 'reasoning_loop_recovery', 'repeat_stop_tokens', 'tool_call_recovery', 'codex_compaction_cache',
  'codex_thread_titles', 'expert_profile_save', 'expert_profile_save_every', 'images', 'vision', 'mmproj', 'draft_vocab', 'ui_resource_profile',
  'ui_vision_enabled', 'component', 'native', 'uma', 'remote_expert_opt', 'diffusion_model', 'llm', 'vae', 'quant', 'type', 'url', 'timeout_s',
]);

const BOOLS = new Set(['open_browser', 'prevent_sleep', 'elastic', 'env_to_user', 'gpus_asked', 'batch_telemetry', 'tool_call_recovery', 'reasoning_loop_recovery']);
const NUMBERS = { port: [1, 65535], max_queue: [0, 1e6], queue_timeout: [0, 1e6], threads: [1, 4096], elastic_idle_s: [0, 1e7], min_free_vram_mib: [0, 1e7] };
const REASONING = ['none', 'low', 'medium', 'high'];
const SAMPLING = { temperature: [0, 10], top_p: [0, 1], top_k: [1, 64], min_p: [0, 1], presence_penalty: [-10, 10], repetition_penalty: [0, 10], repeat_penalty: [0, 10], frequency_penalty: [-10, 10], penalty_last_n: [0, 1048576], repeat_last_n: [0, 1048576], seed: [0, 4294967295] };
const WHOLE = new Set(['top_k', 'penalty_last_n', 'repeat_last_n', 'seed']);

function distance(a, b) {
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = row[0]; row[0] = i;
    for (let j = 1; j <= b.length; j++) { const keep = row[j]; row[j] = Math.min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1)); prev = keep; }
  }
  return row[b.length];
}
/** The real key a mistyped one most likely meant, or null. */
export function nearestKey(key, known = KNOWN_KEYS) {
  let best = null, score = Infinity;
  const lower = key.toLowerCase().replace(/[-\s]/g, '_');
  for (const candidate of known) {
    const d = lower === candidate ? 0 : distance(lower, candidate);
    if (d < score) { score = d; best = candidate; }
  }
  return score <= Math.max(2, Math.floor(key.length / 4)) ? best : null;
}

const escape = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
/** 1-based line of a top-level key in the text, or null. */
function lineOf(text, key) {
  const m = new RegExp(`^[ \\t]{0,8}"${escape(key)}"\\s*:`, 'm').exec(text);
  return m ? text.slice(0, m.index).split('\n').length : null;
}

const show = v => { const t = JSON.stringify(v); return t === undefined ? 'nothing' : t.length > 40 ? `${t.slice(0, 37)}…` : t; };

/**
 * `text` is the file as it is on disk. Returns { ok, issues } where ok means no errors (warnings may remain). Never throws.
 */
export function checkRunConfig(text, { known = KNOWN_KEYS } = {}) {
  text = String(text ?? '');
  const issues = [];
  const add = (level, code, key, params, message) => issues.push({ level, code, key, line: key ? lineOf(text, key) : null, params, message });
  let config;
  try { config = JSON.parse(text.replace(/^﻿/, '')); }
  catch (e) {
    const position = /position (\d+)/.exec(e.message ?? '');
    const line = position ? text.slice(0, Number(position[1])).split('\n').length : null;
    issues.push({ level: 'error', code: 'json', key: null, line, params: { line: line ?? 0 },
      message: `The settings file is not valid JSON${line ? `; look near line ${line}` : ''}. Check for a missing comma between two entries, a comma after the last entry, or a missing quote.` });
    return { ok: false, issues };
  }
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    issues.push({ level: 'error', code: 'not-object', key: null, line: null, params: {}, message: 'The settings file must hold one object, { … }, with the settings inside.' });
    return { ok: false, issues };
  }
  const fields = new Map(NATIVE_FIELDS.map(f => [f.id, f]));
  for (const [key, value] of Object.entries(config)) {
    if (!known.has(key) && !key.startsWith('ui_')) {   // ui_* are the app's own notes in the file (resource profile, picture device)
      const near = nearestKey(key, known);
      add('warning', near ? 'unknown-key-near' : 'unknown-key', key, { key, near: near ?? '' }, `"${key}" is not a setting Strata reads, so it is ignored.${near ? ` Did you mean "${near}"?` : ''}`);
      continue;
    }
    if (value === null) continue;
    if (key === 'args') {
      if (!Array.isArray(value) || value.some(v => typeof v !== 'string')) add('error', 'args-type', key, {}, '"args" must be a list of quoted text entries, for example ["--ple-io", "mmap"]. Numbers need quotes too.');
      else if (value.some(v => ['--host', '--port', '--api-key'].includes(v))) add('warning', 'args-managed', key, {}, 'Strata manages the network address, port and API key itself, so --host, --port and --api-key in "args" are not honoured.');
    } else if (key === 'env') {
      if (!value || typeof value !== 'object' || Array.isArray(value)) add('error', 'env-type', key, {}, '"env" must be an object of names with quoted text values, for example { "STRATA_X": "1" }.');
      else for (const [name, v] of Object.entries(value)) {
        if (typeof v !== 'string') add('error', 'env-value', key, { name }, `"env" entry ${name} must be quoted text, for example "1", not ${show(v)}.`);
      }
    } else if (key === 'sampling') {
      if (!value || typeof value !== 'object' || Array.isArray(value)) add('error', 'sampling-type', key, {}, '"sampling" must be an object such as { "temperature": 0.7 }.');
      else for (const [name, n] of Object.entries(value)) {
        if (n === null) continue;
        if (name === 'experimental_speed_projection') { if (typeof n !== 'boolean') add('error', 'bool', key, { key: `sampling.${name}` }, `"sampling.${name}" must be true or false.`); continue; }
        if (!SAMPLING[name]) { add('warning', 'unknown-key', key, { key: `sampling.${name}`, near: '' }, `"sampling.${name}" is not a setting Strata reads, so it is ignored.`); continue; }
        const [lo, hi] = SAMPLING[name];
        if (typeof n !== 'number' || !Number.isFinite(n) || n < lo || n > hi || (WHOLE.has(name) && !Number.isInteger(n))) add('error', WHOLE.has(name) ? 'whole-range' : 'number-range', key, { key: `sampling.${name}`, min: lo, max: hi, value: show(n) }, `"sampling.${name}" must be ${WHOLE.has(name) ? 'a whole ' : 'a '}number from ${lo} to ${hi}, not ${show(n)}.`);
      }
    } else if (fields.has(key)) {
      const f = fields.get(key);
      if (value === '') continue;
      if (f.type === 'number' && (typeof value !== 'number' || !Number.isInteger(value) || value < f.min || value > f.max)) add('error', 'whole-range', key, { key, min: f.min, max: f.max, value: show(value) }, `"${key}" must be a whole number from ${f.min} to ${f.max}, not ${show(value)}.`);
      else if (f.type === 'boolean' && typeof value !== 'boolean') add('error', 'bool', key, { key, value: show(value) }, `"${key}" must be true or false, not ${show(value)}.`);
      else if (f.type === 'enum' && !f.choices.includes(value)) add('error', 'choice', key, { key, choices: f.choices.join(', '), value: show(value) }, `"${key}" must be one of ${f.choices.join(', ')}, not ${show(value)}.`);
      else if (f.type === 'string' && (typeof value !== 'string' || value.length > 256)) add('error', 'text', key, { key }, `"${key}" must be text of at most 256 characters.`);
      else if (f.type === 'array') {
        const list = typeof value === 'string' ? value.split(',').map(v => v.trim()).filter(Boolean) : value;
        if (!Array.isArray(list) || list.length > 24 || list.some(v => typeof v !== 'string' || !v.trim() || v.length > 80)) add('error', 'text', key, { key }, `"${key}" must be a list of short names, for example ["chat", "coding"].`);
      }
      if (key === 'layer_split' && typeof value === 'string' && value !== 'auto') {
        const points = value.split(',').map(Number);
        if (!points.length || points.some((n, i) => !Number.isInteger(n) || n < 1 || n >= 48 || (i > 0 && n <= points[i - 1]))) add('error', 'layer-split', key, {}, '"layer_split" points must increase from 1 to 47, for example "16,32", or be "auto".');
      }
    } else if (BOOLS.has(key)) {
      if (typeof value !== 'boolean') add('error', 'bool', key, { key, value: show(value) }, `"${key}" must be true or false, not ${show(value)}.`);
    } else if (key in NUMBERS) {
      const [lo, hi] = NUMBERS[key];
      if (typeof value !== 'number' || !Number.isFinite(value) || value < lo || value > hi) add('error', 'number-range', key, { key, min: lo, max: hi, value: show(value) }, `"${key}" must be a number from ${lo} to ${hi}, not ${show(value)}.`);
    } else if (key === 'reasoning_effort') {
      if (!REASONING.includes(value)) add('error', 'choice', key, { key, choices: REASONING.join(', '), value: show(value) }, `"reasoning_effort" must be one of ${REASONING.join(', ')}, not ${show(value)}.`);
    } else if (key === 'gpu') {
      const ok = Number.isInteger(value) && value >= 0 || Array.isArray(value) && value.length > 0 && value.every(n => Number.isInteger(n) && n >= 0);
      if (!ok) add('error', 'gpu', key, {}, '"gpu" must be a graphics card number such as 0, or a list such as [0, 1].');
    }
  }
  return { ok: !issues.some(i => i.level === 'error'), issues };
}

/** The English lines of the issues, `Line 12: …`, for the Details of a refused start. */
export function describeIssues(issues, level) {
  return issues.filter(i => !level || i.level === level).map(i => `${i.line ? `Line ${i.line}: ` : ''}${i.message}`);
}
