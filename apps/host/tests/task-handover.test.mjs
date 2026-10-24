/** "Continue in a new chat" on a task: the first message of the new chat carries what the task found, did, and the files it made. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createAgentTaskRunner } from '../server/agent/task-run.mjs';
import { createTaskService } from '../server/tasks.mjs';

const compiled = await build({ stdin: { contents: "export * from './src/lib/task-handover'; export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { splitTaskOutput, taskHandoverMessage, taskHandoverTitle, setTranslationLanguage } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const OUTPUT = 'Two things need your attention.\n\n- Disk space is low.\n- The card ran hot twice.\n\n---\n\n**What it did**\n\n- Read the temperature log\n- Checked free space on C: and D:\n\n> This task wanted to change something and needed a yes. Nothing was changed.';
const task = { id: 't', title: 'Hardware check', prompt: 'Check it', frequency: 'daily', at: '08:00', weekday: 1, timeZone: 'UTC', enabled: true, lastRunAt: '2026-10-18T06:00:00.000Z', output: OUTPUT,
  files: [{ name: 'report.md', path: 'C:\\Users\\you\\Documents\\Strata\\Workspace\\report.md' }] };

test('the saved answer is split into what it found, what it did, and the note that needed a yes', () => {
  const parts = splitTaskOutput(OUTPUT);
  assert.equal(parts.found, 'Two things need your attention.\n\n- Disk space is low.\n- The card ran hot twice.');
  assert.deepEqual(parts.did, ['Read the temperature log', 'Checked free space on C: and D:']);
  assert.match(parts.attention, /needed a yes/);
  assert.deepEqual(splitTaskOutput('Just an answer.'), { found: 'Just an answer.', did: [], attention: '' });
  assert.deepEqual(splitTaskOutput(''), { found: '', did: [], attention: '' });
});

test('the first message names the task and carries the findings, the steps, the files and a next step', () => {
  setTranslationLanguage('en');
  const text = taskHandoverMessage(task);
  assert.match(text, /^\*\*Continuing from the task “Hardware check”\*\* \(ran .*, finished\)\./);
  for (const part of ['**What it found**', 'Disk space is low.', '**Needs your attention**', 'needed a yes', '**What it did**', '- Read the temperature log', '**Files**', '- C:\\Users\\you\\Documents\\Strata\\Workspace\\report.md', 'Please help me take it from here.']) assert.ok(text.includes(part), part);
  assert.ok(text.indexOf('What it found') < text.indexOf('What it did') && text.indexOf('What it did') < text.indexOf('Files'), 'in reading order');
  assert.equal(taskHandoverTitle(task), 'Continuing from the task “Hardware check”');
});

test('a task that did not finish says what went wrong; a task with no files has no Files part', () => {
  setTranslationLanguage('en');
  const failed = taskHandoverMessage({ ...task, error: 'The model is not reachable.', output: '', files: [] });
  assert.match(failed, /did not finish/); assert.match(failed, /\*\*What went wrong\*\*\n\nThe model is not reachable\./);
  assert.ok(!failed.includes('**Files**') && !failed.includes('**What it found**'));
});

test('the labels follow the language of the app', () => {
  setTranslationLanguage('de');
  const text = taskHandoverMessage(task);
  assert.match(text, /Weiter mit der Aufgabe „Hardware check“/); assert.match(text, /\*\*Was sie gefunden hat\*\*/); assert.match(text, /Dateien/);
  setTranslationLanguage('en');
});

test('a scheduled run keeps the files the agent made, and a later failed run does not keep the old ones', async () => {
  const frames = [{ type: 'delta', text: 'Done.' }, { type: 'deliverable', abs: '/w/report.md', name: 'report.md', path: 'report.md' }, { type: 'deliverable', abs: '/w/report.md', name: 'report.md', path: 'report.md' }];
  const agent = { runTurn: async ({ onFrame }) => { for (const f of frames) onFrame(f); }, stop: async () => {}, forget() {}, approve: async () => {} };
  const run = createAgentTaskRunner({ agent, choice: () => ({ instanceId: 'i' }), workspace: '/w' });
  let store = [];
  const service = createTaskService({ read: () => store, write: v => { store = v; }, run });
  const saved = service.put({ title: 'T', prompt: 'p', frequency: 'daily', at: '08:00', timeZone: 'UTC' });
  await service.execute(saved.id);
  assert.deepEqual(service.list()[0].files, [{ name: 'report.md', path: '/w/report.md' }]);
  frames.length = 0; frames.push({ type: 'error', message: 'boom' });
  await service.execute(saved.id);
  assert.equal(service.list()[0].status, 'error'); assert.deepEqual(service.list()[0].files, []);
});

test('a long run keeps the first steps, shortened, and says how many more', () => {
  setTranslationLanguage('en');
  const steps = Array.from({ length: 14 }, (_, i) => `Step ${i + 1} ${'x'.repeat(300)}`);
  const text = taskHandoverMessage({ ...task, output: `Done.\n\n---\n\n**What it did**\n\n${steps.map(s => `- ${s}`).join('\n')}`, files: [] });
  const lines = text.split('\n').filter(l => l.startsWith('- '));
  assert.equal(lines.length, 11); assert.ok(lines.slice(0, 10).every(l => l.length <= 162 && l.endsWith('…')));
  assert.equal(lines[10], '- …and 4 more steps');
});
