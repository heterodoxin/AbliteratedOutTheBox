// Time-skip v2: plan, lever, A all the way, B forked at the lever with an improvement plan, compare, keep.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { isViewStep, stepObjective } from '../server/agent/plan.mjs';
import { UNATTENDED, createForkTimeskip, leverPrompt, parseLever, defaultLever, improvementPrompt, copyProject } from '../server/timeskip-fork.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('the lever answer is read strictly; a bad answer gets a fixed rule with its reason said plainly', () => {
  assert.deepEqual(parseLever('Sure! {"step": 3, "why": "The shape is set here."} done', 5), { step: 3, why: 'The shape is set here.' });
  assert.equal(parseLever('{"step": 1, "why": "x"}', 5), null, 'never step 1');
  assert.equal(parseLever('{"step": 9, "why": "x"}', 5), null);
  assert.equal(parseLever('no json', 5), null);
  assert.equal(parseLever('{"step": 2}', 5).why.length > 5, true);
  const steps = ['Look around', 'Write the page', 'Add controls', 'Check it'].map(text => ({ text }));
  const d = defaultLever(steps);
  assert.deepEqual([d.step, d.defaulted], [3, true]); assert.match(d.why, /default rule/);
  assert.ok(defaultLever([{ text: 'a' }, { text: 'b' }, { text: 'c' }]).step >= 2);
  assert.match(leverPrompt('make x', steps), /ONE biggest lever/); assert.match(leverPrompt('make x', steps), /"step"/);
  const ip = improvementPrompt({ request: 'make x', steps, lever: { step: 3, why: 'shape' }, doneSummaries: ['1. Look around: done.'] });
  assert.match(ip, /IMPROVEMENT PLAN/); assert.match(ip, /Steps 1 to 2 are done/); assert.match(ip, /shape/); assert.match(ip, /\.timeskip\/context\.md/);
});

test('copyProject leaves out dependency and VCS folders', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-cp-'));
  try {
    const from = join(dir, 'p'); mkdirSync(join(from, 'node_modules'), { recursive: true }); mkdirSync(join(from, 'src'), { recursive: true }); mkdirSync(join(from, '.git'));
    writeFileSync(join(from, 'node_modules', 'x.js'), 'x'); writeFileSync(join(from, 'src', 'a.txt'), 'a'); writeFileSync(join(from, '.git', 'h'), 'h');
    copyProject(from, join(dir, 'c'));
    assert.ok(existsSync(join(dir, 'c', 'src', 'a.txt'))); assert.ok(!existsSync(join(dir, 'c', 'node_modules'))); assert.ok(!existsSync(join(dir, 'c', '.git')));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

async function rig(t, slots, opts = {}) {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-tsf-'));
  const project = join(dir, 'project'); mkdirSync(project); writeFileSync(join(project, 'keep.txt'), 'original');
  const bridge = createAgentBridge({ dataDir: join(dir, 'data'), localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots }], connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  const events = [];
  const ts = createForkTimeskip({ agent: bridge, dir: join(dir, 'ts'), notify: (type, data) => events.push(data), ...opts });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  return { ts, events, project, dir, worker };
}
const until = async (fn, ms = 90_000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { const v = fn(); if (v) return v; await new Promise(r => setTimeout(r, 100)); } throw new Error('timed out'); };

test('plan -> lever (with its why) -> A runs every step -> B forks after step 1 with an improvement plan -> A and B are compared, and Keep B changes the folder', { skip }, async (t) => {
  const r = await rig(t, 2);
  const run = r.ts.start({ prompt: 'SCRIPT:planreply', workspace: r.project, choice: { instanceId: 'i' }, slots: 2 });
  assert.ok(['queued', 'checkpointing'].includes(run.phase), 'a run starts at once and does not wait for the work');
  const done = await until(() => { const s = r.ts.status(run.id); return ['completed', 'failed'].includes(s.phase) ? s : null; });
  assert.equal(done.phase, 'completed', done.error);
  assert.equal(done.plan.steps.length, 3);
  assert.deepEqual([done.lever.step, done.lever.total, done.lever.defaulted], [2, 3, false]); assert.match(done.lever.why, /layout/);
  assert.deepEqual([done.a.state, done.a.step, done.a.total], ['done', 3, 3]);
  assert.equal(done.b.state, 'done'); assert.equal(done.b.plan.length, 2);
  assert.equal(done.needsUserReview, true);
  assert.match(done.activity.text, /^Both versions are ready/); assert.equal(done.b.note ?? null, null, 'no stale warning when B finished');
  assert.deepEqual(done.labels, { original: 'A', updated: 'B' });
  // events told the story in order
  const phases = r.events.map(e => e.phase);
  assert.ok(phases.indexOf('checkpointing') < phases.indexOf('generating') && phases.indexOf('generating') < phases.indexOf('completed'));
  assert.ok(r.events.some(e => e.lever && e.lever.step === 2), 'the lever and its reason were sent');
  assert.ok(r.events.some(e => e.a?.line && / — /.test(e.a.line)) && r.events.some(e => e.b?.line && / — /.test(e.b.line)), "A's and B's progress were sent");
  assert.ok(r.events.some(e => /^B, /.test(e.activity?.text ?? '')), 'the activity line names B');
  // B was copied from A at the fork and got the context attachment; the project is untouched
  assert.ok(existsSync(join(r.dir, 'ts', run.id, 'B', '.timeskip', 'context.md')));
  assert.match(readFileSync(join(r.dir, 'ts', run.id, 'B', '.timeskip', 'context.md'), 'utf8'), /Why this step: The layout/);
  assert.equal(readFileSync(join(r.project, 'keep.txt'), 'utf8'), 'original');
  // the compare is A against B
  const found = r.ts.files(run.id);
  assert.ok(Array.isArray(found.files));
  // keep B, then put the original back
  const kept = r.ts.apply({ id: run.id, which: 'b' });
  assert.equal(kept.applied, 'b');
  assert.throws(() => r.ts.apply({ id: run.id, which: 'a' }), /already in your files/);
  assert.deepEqual(r.ts.apply({ id: run.id, restore: true }), { restored: true });
  assert.equal(readFileSync(join(r.project, 'keep.txt'), 'utf8'), 'original');
});

test('with one model slot B waits for A, says so, and still finishes', { skip }, async (t) => {
  const r = await rig(t, 1);
  const run = r.ts.start({ prompt: 'SCRIPT:planreply', workspace: r.project, choice: { instanceId: 'i' }, slots: 1 });
  const done = await until(() => { const s = r.ts.status(run.id); return ['completed', 'failed'].includes(s.phase) ? s : null; });
  assert.equal(done.phase, 'completed', done.error);
  assert.ok(r.events.some(e => e.b?.waiting === true), 'B was shown as waiting');
  assert.ok(r.events.some(e => /B waited for A/.test(e.activity?.text ?? '')));
  assert.equal(done.b.state, 'done');
});

test('Stop ends both agents, removes the copies and leaves the folder alone', { skip }, async (t) => {
  const r = await rig(t, 2);
  const run = r.ts.start({ prompt: 'SCRIPT:planreply', workspace: r.project, choice: { instanceId: 'i' }, slots: 2 });
  await until(() => r.ts.status(run.id).a.state === 'running');
  assert.equal(await r.ts.stop(run.id), true);
  assert.equal(r.ts.status(run.id).phase, 'cancelled');
  assert.ok(!existsSync(join(r.dir, 'ts', run.id)));
  assert.equal(readFileSync(join(r.project, 'keep.txt'), 'utf8'), 'original');
  assert.equal(r.ts.files(run.id), null);
});

test('a step about looking at the page in a browser is recognised, carries the no-hang rule, and the unattended note forbids windows', () => {
  assert.ok(isViewStep('Open goku.html in a browser to confirm rendering')); assert.ok(isViewStep('Preview the page in a window')); assert.ok(!isViewStep('Write the HTML file')); assert.ok(!isViewStep('Count the voxels'));
  const plan = { title: 'T', steps: [{ text: 'Write it' }, { text: 'Open it in a browser to check', detail: '' }] };
  assert.match(stepObjective(plan, 1), /Never wait for a program to close/); assert.match(stepObjective(plan, 1), /Start-Process/); assert.ok(!/Start-Process/.test(stepObjective(plan, 0)));
  assert.match(UNATTENDED, /Never open a visible window/);
});

test('a step that never finishes is stopped after the time limit and left as it is with a plain note; the version still completes', { skip }, async (t) => {
  const r = await rig(t, 2, { stepTimeoutMs: 4000 });
  const run = r.ts.start({ prompt: 'SCRIPT:planslow', workspace: r.project, choice: { instanceId: 'i' }, slots: 2 });
  const done = await until(() => { const s = r.ts.status(run.id); return ['completed', 'failed'].includes(s.phase) ? s : null; }, 150_000);
  assert.equal(done.phase, 'completed', done.error);
  assert.ok(r.events.some(e => /step 2 took too long and was left as it is/.test(e.activity?.text ?? '')), 'the person was told which step was left');
  assert.equal(done.a.state, 'done');
});
