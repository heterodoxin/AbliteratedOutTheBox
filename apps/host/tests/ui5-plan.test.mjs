/** Owner round 7: /plan card, plan steps in the sidebar, queued messages, against agent6's contract (notes-split-agent6.md). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const src = f => fileURLToPath(new URL(`../src/${f}`, import.meta.url));
const stub = fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url));
async function load(file) {
  const result = await build({ entryPoints: [src(`lib/${file}`)], bundle: true, format: 'esm', platform: 'node', write: false, alias: { react: stub }, logLevel: 'silent', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}
globalThis.window ??= globalThis; globalThis.localStorage ??= { getItem: () => null, setItem: () => {} };
globalThis.navigator ??= { language: 'en' };
const P = await load('plan.ts');
const G = await load('goal.ts');

const plan = (state, steps, extra = {}) => ({ id: 'p1', chatId: 'c', title: 'Pagoda', request: 'x', source: 'plan', state, goalStarted: false, steps, ...extra });
const step = (id, text, state = 'pending', extra = {}) => ({ id, text, detail: null, state, summary: null, ...extra });

test('a plan from the host is read with step states, detail and summary; bad parts are dropped', () => {
  const p = P.readPlan(plan('running', [step('s1', 'Look', 'done', { summary: 'Found two files.' }), step('s2', 'Write', 'running', { detail: 'one file' }), step('s3', 'Check'), step('s4', '  '), 7]));
  assert.equal(p.steps.length, 3);
  assert.deepEqual(p.steps.map(s => s.state), ['done', 'running', 'pending']);
  assert.equal(p.steps[0].summary, 'Found two files.'); assert.equal(p.steps[1].detail, 'one file');
  assert.equal(P.stepsDone(p), 1); assert.equal(P.currentStep(p), 2);
  assert.equal(P.readPlan(plan('weird', [step('s1', 'x')])), null);
  assert.equal(P.readPlan(plan('proposed', [])), null);
  assert.equal(P.readPlan(null), null);
  assert.equal(P.readPlan(plan('paused', [step('s1', 'x', 'skipped')])).steps[0].state, 'skipped');
});

test('plan frames replace the plan; null clears it; the same plan changes nothing', () => {
  P.applyPlanFrame('c', plan('proposed', [step('s1', 'A'), step('s2', 'B')]));
  assert.equal(P.planOf('c').state, 'proposed');
  P.applyPlanFrame('c', plan('running', [step('s1', 'A', 'running'), step('s2', 'B')]));
  assert.equal(P.planOf('c').steps[0].state, 'running');
  assert.equal(P.planOf('other'), null);
  P.applyPlanFrame('c', null);
  assert.equal(P.planOf('c'), null);
});

test('queued messages: the first waits for the current step, the next ones "after that", a goal without points waits for the end', () => {
  P.applyQueueFrame('c', [{ id: 'q1', text: 'Add a moon', deliverAt: 'step' }, { id: 'q2', text: 'Warmer glow', deliverAt: 'step' }, { id: '', text: 'x' }, { id: 'q3' }]);
  const items = P.queueOf('c');
  assert.deepEqual(items.map(q => q.id), ['q1', 'q2']);
  const p = P.readPlan(plan('running', [step('s1', 'A', 'done'), step('s2', 'B', 'running')]));
  assert.deepEqual(P.queuedWhen(items[0], 0, p), { id: 'Queued — sends when step {0} finishes', args: [2] });
  assert.deepEqual(P.queuedWhen(items[1], 1, p), { id: 'Queued — sends after that' });
  assert.deepEqual(P.queuedWhen({ id: 'e', text: 'x', deliverAt: 'end' }, 0, null), { id: 'Queued — sends when the goal finishes' });
  P.applyQueueFrame('c', []);
  assert.deepEqual(P.queueOf('c'), []);
});

test('the goal bar reads the step the host reports', () => {
  const g = G.readGoal({ objective: 'Build a small pagoda page', phase: 'active', round: 9, maxRounds: 50, step: { index: 2, total: 3, text: 'Write it' } });
  assert.deepEqual(g.step, { index: 2, total: 3, text: 'Write it' });
  assert.equal(G.goalProgress, undefined, 'no step counter is shown for a goal');
  assert.equal(G.readGoal({ objective: 'x', phase: 'active', round: 1, maxRounds: 5 }).step, undefined);
});

test('the chat and the sidebar draw the plan from the same store, and the stream feeds it', async () => {
  const [card, chat, store, client] = await Promise.all(['components/PlanCard.tsx', 'screens/Chat.tsx', 'lib/store.ts', 'lib/client.ts'].map(f => readFile(src(f), 'utf8')));
  assert.match(card, /export function PlanCard/); assert.match(card, /export function PlanPanel/); assert.match(card, /export function QueuedMessages/);
  assert.match(card, /approvePlan\(chatId, plan\.id\)/); assert.match(card, /startPlanGoal\(plan\.id\)/); assert.match(card, /planReject\(chatId, plan\.id\)/);
  assert.match(chat, /<PlanPanel chatId/); assert.match(chat, /plan:\$\{session\.id\}/); assert.match(chat, /planCard \? <PlanCard/);
  assert.match(store, /type === 'plan'/); assert.match(store, /type === 'queued-message'/); assert.match(store, /type === 'queued-release'/); assert.match(store, /\/api\/plan\/start-goal/);
  assert.match(client, /'plan', 'queue'/);
});
