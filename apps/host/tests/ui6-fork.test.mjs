/** Round 8, item 11: the fork design's events (notes-split-agent8.md) are read into the job; the folder chip reads the host's answer. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const src = f => fileURLToPath(new URL(`../src/${f}`, import.meta.url));
const stub = fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url));
globalThis.window ??= globalThis; globalThis.localStorage ??= { getItem: () => null, setItem: () => {} }; globalThis.navigator ??= { language: 'en' };
async function load(file) {
  const r = await build({ entryPoints: [src(file)], bundle: true, format: 'esm', platform: 'node', write: false, alias: { react: stub }, logLevel: 'silent' });
  return import(`data:text/javascript;base64,${Buffer.from(r.outputFiles[0].text).toString('base64')}`);
}
const T = await load('lib/time-skip.ts');
const F = await load('lib/chat-folder.ts');

test('an event carries the plan, the lever with its why, both lanes and one friendly line', () => {
  const f = T.forkFields({ phase: 'running', activity: 'Step 2 of 5 in A: Draw the layout', plan: { steps: [{ id: 's1', text: 'Look' }, { text: 'Draw' }, 5] }, lever: { step: 3, total: 5, text: 'Styles', why: 'The layout decides everything after it.', defaulted: false }, a: { state: 'running', step: 2, total: 5, text: 'Draw', line: 'Step 2 of 5', waiting: false }, b: { state: 'waiting', waiting: true } });
  assert.deepEqual(f.plan.steps.map(s => s.id), ['s1', '2']);
  assert.equal(f.lever.why, 'The layout decides everything after it.');
  assert.equal(f.a.step, 2); assert.equal(f.b.waiting, true);
  assert.deepEqual(f.activity, { code: 'line', text: 'Step 2 of 5 in A: Draw the layout' });
  assert.equal(T.isForkJob(f), true);
  assert.equal(T.isForkJob({}), false);
});

test('an older job (an activity object, no plan) is not a fork job and keeps its activity', () => {
  const f = T.forkFields({ phase: 'generating', activity: { code: 'reading-files', text: 'Reading' } });
  assert.deepEqual(f.activity, { code: 'reading-files', text: 'Reading' });
  assert.equal(T.isForkJob(f), false);
  assert.equal(T.forkFields({ lever: null }).lever, null);
});

test('the folder answer: a path gives a name; nothing gives no folder', () => {
  assert.deepEqual(F.readFolder({ path: 'C:\Users\me\pagoda-site', name: 'pagoda-site', exists: true }), { path: 'C:\Users\me\pagoda-site', name: 'pagoda-site', exists: true });
  assert.equal(F.readFolder({ path: '/home/me/site' }).name, 'site');
  assert.equal(F.readFolder(null), null);
  assert.equal(F.readFolder({ path: '' }), null);
});

test('the sample time-skip copy appears only when the host cannot be reached at all', async () => {
  const { readFileSync } = await import('node:fs');
  const store = readFileSync(src('lib/store.ts'), 'utf8');
  const chat = readFileSync(src('screens/Chat.tsx'), 'utf8');
  const panel = readFileSync(src('components/TimeSkipPanel.tsx'), 'utf8');
  const route = store.slice(store.indexOf('const demoPrompt = '), store.indexOf('const imageTarget'));
  assert.match(route, /connection === 'live'/);
  assert.match(route, /connection === 'connecting'/);
  assert.ok(route.indexOf('startTimeSkipDemo') > route.indexOf("'connecting'"), 'the sample run is the last resort, after live and connecting');
  assert.match(chat, /hostLive \? tr\('This was a sample run/);
  assert.match(panel, /run\.job \? 'Working on a copy of your files\.'/);
});
