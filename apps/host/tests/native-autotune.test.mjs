import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';
import { resultSentence } from '../server/native-autotune.mjs';

const arg = (args, name) => { const i = args.indexOf(name); return i < 0 ? undefined : args[i + 1]; };

/** A pretend `strata` command: records what it was asked, writes status-file steps, then prints its JSON. */
function handler(log, { result, hold = false, delayMs = 60 } = {}) {
  return (child, args) => {
    log.push(args.slice(args.findIndex(a => String(a).endsWith('strata.py')) + 1));
    const cmd = log.at(-1);
    const out = json => { child.stdout.emit('data', Buffer.from(JSON.stringify(json))); child.exitCode = 0; child.emit('exit', 0); child.emit('close', 0); };
    if (cmd[0] === 'presets' && cmd[1] === 'status') return out({ ok: true, active: null, suggested_name: 'IQ2_XS 32K – Full speed', presets: [], state: null, reasons: [] });
    if (cmd[0] === 'presets' && cmd[1] === 'list') return out({ ok: true, presets: [{ id: 'a'.repeat(32), name: 'x' }] });
    if (cmd[0] === 'presets' && cmd[1] === 'effective') return out({ ok: true, applied: true, preset: { id: 'a'.repeat(32), name: 'IQ2_XS 32K – Work' }, args: ['--kv-resident', '65536'], changed: [{ flag: '--kv-resident', was: '32768', now: '65536' }] });
    if (cmd[0] === 'presets') return out({ ok: true, done: cmd[1] });
    if (cmd[0] === 'tune') {
      const status = arg(cmd, '--status-file');
      writeFileSync(status, JSON.stringify({ state: 'running', stage: 'rounds', text: 'Finding the fastest settings for your PC… about 9 min left', minutes_left: 9, runs_done: 3, runs_planned: 9, elapsed_s: 400, message: 'Trying: host core' }));
      if (hold) return;                       // stays running until killed
      setTimeout(() => out({ ok: true, result }), delayMs);
    }
  };
}

const improved = { outcome: 'improved', gain_pct: 11.6, seconds: 900, preset: { id: 'b'.repeat(32), name: 'IQ2_XS 32K – Full speed' }, answers: { class: 'exact', fields: [] }, settings: { args: { '--host-core': 'last' }, env: {}, per_request: {} } };

test('the result sentence is plain words with the gain and the saved name', () => {
  assert.equal(resultSentence(improved), "+12% faster answers, saved as 'IQ2_XS 32K – Full speed'");
  assert.match(resultSentence({ outcome: 'default_confirmed', preset: { name: 'X' } }), /already the fastest/);
  assert.equal(resultSentence({ outcome: 'inconclusive', message: 'The PC was too noisy.' }), 'The PC was too noisy.');
});

test('a tune is started on the config a load would use, unloads nothing it did not have, follows progress and reports the saved name', async () => {
  const log = [];
  const f = nativeFixture({ tuneHandler: handler(log, { result: improved, delayMs: 2500 }) });
  try {
    await f.control.refresh(true);
    const job = f.control.operation('autotune', { model: 'iq2_xs', mode: 'quick', use: 'chat', resourceProfile: 'work' });
    let seen = null;
    for (let n = 0; n < 200 && !seen; n++) { await new Promise(r => setTimeout(r, 30)); const j = f.control.state().jobs.find(x => x.id === job.id); if (j.progress) seen = j.progress; }
    assert.ok(seen, 'progress from the status file reaches the job');
    assert.equal(seen.minutesLeft, 9);
    const done = await finishJob(f.control, job.id);
    assert.equal(done.state, 'done', done.error);
    assert.equal(done.result.summary, "+12% faster answers, saved as 'IQ2_XS 32K – Full speed'");
    const cmd = log.find(c => c[0] === 'tune');
    assert.equal(cmd[1], 'auto');
    assert.equal(arg(cmd, '--mode'), 'quick');
    assert.equal(arg(cmd, '--goal'), 'room');                 // "Leave 20% for my work"
    assert.equal(arg(cmd, '--leave-ram'), '20');
    assert.ok(!cmd.includes('--allow-small-differences'));
    const cfg = JSON.parse(readFileSync(cmd[2], 'utf8'));      // the config the tuner measures is the one a load would start
    assert.equal(cfg.ui_resource_profile, 'work');
    assert.ok(Array.isArray(cfg.args) && cfg.args.includes('--max-context'));
    assert.equal(cfg.api_key, undefined);
    assert.ok(!existsSync(arg(cmd, '--status-file')), 'the status file is removed afterwards');
  } finally { await f.cleanup(); }
});

test('full performance asks for speed with no memory left over, and the tick for small differences is only passed when ticked', async () => {
  const log = [];
  const f = nativeFixture({ tuneHandler: handler(log, { result: improved }) });
  try {
    await f.control.refresh(true);
    const done = await finishJob(f.control, f.control.operation('autotune', { model: 'iq3_s', mode: 'thorough', resourceProfile: 'performance', allowSmallDifferences: true }).id);
    assert.equal(done.state, 'done', done.error);
    const cmd = log.find(c => c[0] === 'tune');
    assert.equal(arg(cmd, '--goal'), 'speed');
    assert.equal(cmd.includes('--leave-ram'), false);
    assert.equal(arg(cmd, '--mode'), 'thorough');
    assert.ok(cmd.includes('--allow-small-differences'));
  } finally { await f.cleanup(); }
});

test('Stop ends the tune and keeps your settings; only one operation at a time', async () => {
  const log = [];
  const f = nativeFixture({ tuneHandler: handler(log, { hold: true }) });
  try {
    await f.control.refresh(true);
    const job = f.control.operation('autotune', { model: 'iq2_xs' });
    await new Promise(r => setTimeout(r, 80));
    assert.throws(() => f.control.operation('autotune', { model: 'iq3_s' }), /current engine operation/);
    await f.control.cancel(job.id);
    const j = f.control.state().jobs.find(x => x.id === job.id);
    assert.equal(j.state, 'cancelled');
    assert.equal(j.result, null);
  } finally { await f.cleanup(); }
});

test('a tune is refused while a model is generating, and loaded models are unloaded first', async () => {
  const log = [];
  const f = nativeFixture({ tuneHandler: handler(log, { result: improved }) });
  try {
    await f.control.refresh(true);
    const loaded = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(loaded.state, 'done', loaded.error);
    f.requests.push({ instanceId: loaded.result.instanceId, state: 'generating' });
    const refused = await finishJob(f.control, f.control.operation('autotune', { model: 'iq2_xs' }).id);
    assert.equal(refused.state, 'error');
    assert.match(refused.error, /Finish requests/);
    f.requests.length = 0;
    const done = await finishJob(f.control, f.control.operation('autotune', { model: 'iq2_xs' }).id);
    assert.equal(done.state, 'done', done.error);
    assert.equal(f.control.state().instances.length, 0, 'the model was unloaded so the tuner has the graphics card');
  } finally { await f.cleanup(); }
});

test('saved tunings: status for the choices, list, and actions go to Strata Control by id', async () => {
  const log = [];
  const f = nativeFixture({ tuneHandler: handler(log, {}) });
  try {
    await f.control.refresh(true);
    const st = await f.control.tuneStatus({ model: 'iq2_xs', resourceProfile: 'performance' });
    assert.equal(st.suggested_name, 'IQ2_XS 32K – Full speed');
    const listed = await f.control.tunePresets.list();
    assert.equal(listed.presets.length, 1);
    await f.control.tunePresets.action('rename', 'a'.repeat(32), { name: 'My fast one' });
    assert.deepEqual(log.at(-1).slice(0, 4), ['presets', 'rename', 'a'.repeat(32), 'My fast one']);
    await assert.rejects(f.control.tunePresets.action('apply', '../x'), /Choose a saved tuning/);
    await assert.rejects(f.control.tunePresets.action('format', 'a'.repeat(32)), /supported action/);
  } finally { await f.cleanup(); }
});

test('a model that is not prepared cannot be tuned', async () => {
  const f = nativeFixture({ tuneHandler: handler([], {}) });
  try {
    await f.control.refresh(true);
    f.installed.models[0].ready = false;
    await f.control.refresh(true);
    const job = await finishJob(f.control, f.control.operation('autotune', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /Prepare the model/);
  } finally { await f.cleanup(); }
});

test('Expert mode can read the tuned values and the tuning can be switched off for these choices', async () => {
  const log = [];
  const f = nativeFixture({ tuneHandler: handler(log, {}) });
  try {
    await f.control.refresh(true);
    const eff = await f.control.tuneEffective({ model: 'iq2_xs', resourceProfile: 'work' });
    assert.equal(eff.applied, true);
    assert.deepEqual(eff.changed, [{ flag: '--kv-resident', was: '32768', now: '65536' }]);
    const cmd = log.find(c => c[1] === 'effective');
    const cfg = JSON.parse(readFileSync(cmd[2], 'utf8'));
    assert.equal(cfg.ui_resource_profile, 'work');                 // the choices a load would make, so the right tuning is found
    await f.control.tuneOff({ model: 'iq2_xs', resourceProfile: 'work' });
    assert.deepEqual(log.at(-1).slice(0, 2), ['presets', 'off']);
  } finally { await f.cleanup(); }
});

test('a busy PC is a plain result, not an error', () => {
  assert.match(resultSentence({ outcome: 'inconclusive', code: 'busy_pc', message: 'x' }), /PC was busy/);
});
