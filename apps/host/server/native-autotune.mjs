/**
 * The app's side of the auto-tuner. The tuner itself is Strata Control's (`strata tune auto`, `strata presets`): the same code and the
 * same preset store the Terminal uses, so tuning in one place covers the other. This module only runs those commands, follows the
 * status file the tuner rewrites at every step, and turns the answer into the sentence the app shows. It measures nothing itself.
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const MODES = new Set(['quick', 'thorough']);
const USES = new Set(['chat', 'coding']);

/** The one sentence a finished tune is reported with (plain words; the numbers are the tuner's own). */
export function resultSentence(result) {
  if (!result) return 'The tune did not finish.';
  const name = result.preset?.name;
  if (result.code === 'busy_pc') return 'Your PC was busy, so the measurements were not reliable. Nothing was changed. Try again when it is quiet.';
  if (result.outcome === 'improved') {
    const gain = Math.max(0, Math.round(result.gain_pct ?? 0));
    return `+${gain}% faster answers${name ? `, saved as '${name}'` : ''}`;
  }
  if (result.outcome === 'default_confirmed') return `Your current settings are already the fastest we found${name ? `. Saved as '${name}' so it is not tuned again` : ''}`;
  return result.message || 'Nothing was changed.';
}

export function createAutoTune({ launcher, workDir, spawnProcess = spawn, pollMs = 1500 }) {
  mkdirSync(workDir, { recursive: true });

  /** Run one CLI command and read its JSON. Rejects only when the command cannot start or times out. */
  function run(args, { timeoutMs = 60_000 } = {}) {
    const l = launcher();
    return new Promise((resolve, reject) => {
      const child = spawnProcess(l.command, [...(l.prefix ?? []), ...args], { cwd: l.cwd, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'], env: l.env });
      let out = '', err = '', done = false;
      const finish = (fn, v) => { if (done) return; done = true; clearTimeout(alarm); fn(v); };
      const alarm = setTimeout(() => { try { child.kill(); } catch { /* gone */ } finish(reject, new Error('Strata did not answer in time.')); }, timeoutMs);
      child.stdout?.on('data', c => { out += c; });
      child.stderr?.on('data', c => { err += c; if (err.length > 100_000) err = err.slice(-50_000); });
      child.on('error', e => finish(reject, new Error(`Strata could not start: ${e.message}`)));
      child.on('close', code => {
        let json = null;
        const text = out.trim();
        if (text) { try { json = JSON.parse(text.slice(text.indexOf('{'))); } catch { /* not JSON */ } }
        finish(resolve, { code, json, text, stderr: err.trim() });
      });
    });
  }
  const need = (r, what) => {
    if (r.json?.ok === false) throw new Error(r.json.error?.message ?? what);
    if (r.code !== 0 && !r.json) throw new Error(r.stderr.split(/\r?\n/).filter(Boolean).pop() || what);
    return r.json ?? {};
  };

  return {
    run,
    /** Which saved tuning these choices start with, and whether it is still valid (reads only). */
    async status(configPath) { return need(await run(['presets', 'status', configPath, '--output', 'json'], { timeoutMs: 45_000 }), 'The saved tuning could not be read.'); },
    /** What a start will really run: the saved tuning's values on top of the configuration (Expert mode shows these). */
    async effective(configPath) { return need(await run(['presets', 'effective', configPath, '--output', 'json'], { timeoutMs: 45_000 }), 'The tuned values could not be read.'); },
    /** Switch the saved tuning off for these choices (kept, can be switched on again with apply). */
    async off(configPath) { return need(await run(['presets', 'off', configPath, '--output', 'json'], { timeoutMs: 45_000 }), 'The saved tuning could not be switched off.'); },
    async list() { return need(await run(['presets', 'list', '--output', 'json'], { timeoutMs: 30_000 }), 'The saved tunings could not be read.'); },
    async action(kind, id, extra = {}) {
      if (!/^[0-9a-f]{8,32}$/.test(String(id ?? '')) && kind !== 'undo') throw new Error('Choose a saved tuning.');
      const args = {
        apply: ['presets', 'apply', id], delete: ['presets', 'delete', id], undo: ['presets', 'undo', id],
        rename: ['presets', 'rename', id, String(extra.name ?? '').slice(0, 80)],
      }[kind];
      if (!args) throw new Error('Choose a supported action.');
      return need(await run([...args, '--output', 'json'], { timeoutMs: 30_000 }), 'That did not work.');
    },
    /** Undo by configuration: back to the tuning that was active before (or the standard settings). */
    async undoFor(configPath) { return need(await run(['presets', 'undo', 'x', '--config', configPath, '--output', 'json'], { timeoutMs: 45_000 }), 'There is nothing to undo.'); },

    /**
     * The long jobs: `tune auto` and `presets recheck`. `job` is a native-control job: `job.process` lets Stop end it,
     * `job.progress` carries what the status file says (stage, minutes left, runs done) and `job.summary` the sentence.
     */
    async start(job, kind, opts, { onProgress = () => {} } = {}) {
      const l = launcher();
      const tag = job.id.slice(0, 8);
      const statusFile = join(workDir, `tune-${tag}.status.json`);
      const work = join(workDir, `tune-${tag}`);
      rmSync(statusFile, { force: true });
      let args;
      if (kind === 'recheck') {
        if (!opts.presetId) throw new Error('Choose a saved tuning to check.');
        args = ['presets', 'recheck', opts.presetId, opts.configPath, '--output', 'json', '--status-file', statusFile, '--work-dir', work];
      } else {
        const mode = MODES.has(opts.mode) ? opts.mode : 'quick';
        const use = USES.has(opts.use) ? opts.use : 'chat';
        args = ['tune', 'auto', opts.configPath, '--mode', mode, '--use', use, '--output', 'json', '--status-file', statusFile, '--work-dir', work];
        args.push('--goal', opts.goal === 'room' ? 'room' : 'speed');
        if (opts.leaveRam) args.push('--leave-ram', String(Math.max(0, Math.min(50, Math.round(opts.leaveRam)))));
        if (opts.allowSmallDifferences) args.push('--allow-small-differences');
        // For headless checks on a PC where another engine runs: do not wait for it (the tuner would sit in its quiet check).
        if (process.env.STRATA_TUNE_NO_QUIET_CHECK === '1') args.push('--no-quiet-check');
        if (opts.name) args.push('--name', String(opts.name).slice(0, 80));
      }
      const child = spawnProcess(l.command, [...(l.prefix ?? []), ...args], { cwd: l.cwd, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'], env: l.env });
      job.process = child;
      let out = '', err = '';
      child.stdout?.on('data', c => { out += c; });
      child.stderr?.on('data', c => { err += c; if (err.length > 100_000) err = err.slice(-50_000); });
      const poll = () => {
        try {
          const p = JSON.parse(readFileSync(statusFile, 'utf8'));
          if (p && p.state !== 'done') { job.progress = { stage: p.stage, text: p.text ?? null, minutesLeft: p.minutes_left ?? null, runsDone: p.runs_done ?? 0, runsPlanned: p.runs_planned ?? 0, elapsedS: p.elapsed_s ?? 0, step: p.message ?? '', speeds: Array.isArray(p.speeds) ? p.speeds : [] }; onProgress(job.progress); }
        } catch { /* not written yet */ }
      };
      const timer = setInterval(poll, pollMs);
      try {
        const code = await new Promise((resolve, reject) => { child.on('error', reject); child.on('close', resolve); });
        clearInterval(timer);
        let json = null;
        const text = out.trim();
        if (text) { try { json = JSON.parse(text.slice(text.indexOf('{'))); } catch { /* not JSON */ } }
        if (job.state && job.state !== 'running') return null;      // stopped by the user
        if (!json) throw new Error(err.split(/\r?\n/).filter(Boolean).pop() || `The tune stopped (exit ${code}).`);
        if (json.ok === false) throw new Error(json.error?.message ?? 'The tune failed.');
        const result = json.result ?? json;
        if (kind === 'recheck') return { summary: json.verdict === 'keep' ? 'Still as fast as when it was saved.' : (json.reason ?? 'Checked.'), ...json };
        return { summary: resultSentence(result), ...result };
      } finally {
        clearInterval(timer);
        job.process = null;
        rmSync(statusFile, { force: true });
        try { rmSync(work, { recursive: true, force: true }); } catch { /* keep logs if busy */ }
      }
    },
  };
}

export function writeTuneConfig(directory, model, cfg) {
  const dir = join(directory, 'native-tune');
  mkdirSync(dir, { recursive: true });
  const path = join(dir, `${model}.json`);
  const { api_key, port, ...rest } = cfg;            // the key and port belong to a running instance, not to the choices
  writeFileSync(path, JSON.stringify(rest, null, 2), { mode: 0o600 });
  return path;
}
