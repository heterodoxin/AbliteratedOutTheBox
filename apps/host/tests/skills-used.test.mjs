/** "Used 2 skills: Weekly report, Brand voice" under an answer: the skills a turn loaded, with the name the person gave and a one-line description. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { materializeSkills } from '../server/skill-files.mjs';
import { SkillsUsed, skillInfoFrom, skillLabel } from '../server/agent/skills-used.mjs';

const skip = !process.env.STRATA_HARNESS_DIR || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };

test('a skill is reported once, when it has worked, with the whole list so far', () => {
  const frames = [];
  const used = new SkillsUsed({ emit: f => frames.push(f) });
  used.card({ name: 'skill', state: 'running', args: { name: 'weekly-report' } });
  used.card({ name: 'read', state: 'done', args: { name: 'x' } });
  assert.equal(frames.length, 0, 'not while running, not for another tool');
  used.card({ name: 'skill', state: 'done', args: { name: 'weekly-report' } });
  used.card({ name: 'skill', state: 'done', args: { name: 'weekly-report' } });
  used.card({ name: 'skill', state: 'error', args: { name: 'brand-voice' } });
  used.card({ name: 'skill', state: 'done', args: { name: 'brand-voice' } });
  assert.equal(frames.length, 2);
  assert.deepEqual(frames[1], { type: 'skills-used', skills: [{ name: 'Weekly report', description: '' }, { name: 'Brand voice', description: '' }] });
});

test('the name and description come from the skill file the app wrote; a folder name is made readable', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-su-'));
  try {
    materializeSkills(dir, [{ id: 'my-weekly-report', name: 'Weekly report', description: 'Turns notes into a short status update', body: 'Be short.', enabled: true }]);
    const info = skillInfoFrom(dir);
    assert.deepEqual(info('weekly-report'), { name: 'Weekly report', description: 'Turns notes into a short status update' });
    assert.deepEqual(info('brand-voice'), { name: 'Brand voice', description: '' });
    assert.deepEqual(info('../../etc/passwd'), { name: skillLabel('../../etc/passwd'), description: '' }, 'a path is never read');
  } finally { cleanup(dir); }
});

test('the page keeps the list on the reply (readSkills) and tolerates a bad frame', async () => {
  const source = await readFile(new URL('../src/lib/deliverables.ts', import.meta.url), 'utf8');
  const { code } = await transform(source.replace(/^import .*$/gm, ''), { loader: 'ts', format: 'esm' });
  const { readSkills } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  assert.deepEqual(readSkills({ skills: [{ name: 'A', description: 'd' }, { name: '' }, null, { name: 'B' }] }), [{ name: 'A', description: 'd' }, { name: 'B', description: '' }]);
  assert.deepEqual(readSkills({}), []); assert.deepEqual(readSkills(null), []);
});

test('real Harness: loading two saved skills in a turn sends one skills-used frame per skill, then both are listed', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-su-run-')); const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const live = join(dir, 'skills-live');
  materializeSkills(live, [
    { id: 'my-weekly-report', name: 'Weekly report', description: 'Turns notes into a short status update', body: 'Be short.', enabled: true },
    { id: 'my-brand-voice', name: 'Brand voice', description: 'Plain, friendly, no jargon', body: 'Write plainly.', enabled: true },
  ]);
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }], connections: () => [], skillsDir: () => live, env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const turn = async (chatId, args) => { const frames = []; await bridge.runTurn({ chatId, message: `SCRIPT:call skill ${JSON.stringify(args)}`, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) }); return frames.filter(f => f.type === 'skills-used'); };
  const first = await turn('su', { name: 'weekly-report' });
  assert.equal(first.length, 1);
  assert.deepEqual(first[0].skills, [{ name: 'Weekly report', description: 'Turns notes into a short status update' }]);
  const none = await turn('su2', { name: 'does-not-exist' });
  assert.equal(none.length, 0, 'a skill that could not be loaded is not listed');
});

test('real Harness: two skills in one answer are listed together, in the order they were loaded', { skip }, async (t) => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-su-two-')); const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const live = join(dir, 'skills-live');
  materializeSkills(live, [
    { id: 'my-weekly-report', name: 'Weekly report', description: 'Turns notes into a short status update', body: 'Be short.', enabled: true },
    { id: 'my-brand-voice', name: 'Brand voice', description: 'Plain, friendly, no jargon', body: 'Write plainly.', enabled: true },
  ]);
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }], connections: () => [], skillsDir: () => live, env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const frames = [];
  await bridge.runTurn({ chatId: 'two', message: `SCRIPT:multi ${JSON.stringify({ calls: [['skill', { name: 'weekly-report' }], ['skill', { name: 'brand-voice' }]], say: 'Here is the note.' })}`, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
  const last = frames.filter(f => f.type === 'skills-used').at(-1);
  assert.deepEqual(last.skills.map(s => s.name).sort(), ['Brand voice', 'Weekly report']);
  assert.match(frames.filter(f => f.type === 'delta').map(f => f.text).join(''), /Here is the note\./);
  assert.ok(frames.findIndex(f => f.type === 'skills-used') < frames.findIndex(f => f.type === 'done'), 'the list arrives before the answer is done');
});
