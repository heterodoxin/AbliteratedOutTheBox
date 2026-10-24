/** Owner test round 6: short model names, one logs button, the Monitor sentence and screenshot, Tasks without the deleted list and the suggestions. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const src = f => fileURLToPath(new URL(`../src/${f}`, import.meta.url));
const read = f => readFile(src(f), 'utf8');
async function load(file) {
  const result = await build({ entryPoints: [src(`lib/${file}`)], bundle: true, format: 'esm', platform: 'node', write: false, alias: { react: fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url)) }, logLevel: 'silent', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}
globalThis.window ??= globalThis; globalThis.localStorage ??= { getItem: () => null, setItem: () => {} };
globalThis.navigator ??= { language: 'en' };

test('item 1: downloaded models get a short name, family and quantization', async () => {
  const { nativeShortName } = await load('native-names.ts');
  assert.equal(nativeShortName('swift-iq2_xs'), 'Swift IQ2_XS');
  assert.equal(nativeShortName('iq3_s'), 'IQ3_S');
  assert.equal(nativeShortName('unsloth-ud-q4_k_xl'), 'Unsloth UD-Q4_K_XL');
  assert.equal(nativeShortName('gyro-gyro-s'), 'Gyro-S');
  assert.equal(nativeShortName('coder-iq1_m'), 'Coder IQ1_M');
  assert.equal(nativeShortName('nvfp4-nvfp4'), 'NVFP4 (Strata’s version)');
  assert.equal(nativeShortName('nvfp4-nvfp4-nvidia'), 'NVFP4 (NVIDIA’s version)');
  assert.equal(nativeShortName('nvfp4-nvfp4-qad'), 'NVFP4 (NVIDIA, trained for FP4)');
  const storage = await read('components/SettingsStorage.tsx');
  assert.match(storage, /title=\{m\.model_name \|\| m\.model\}/, 'the full name is the tooltip');
});

test('item 2: Diagnostics has one button, View logs; Save logs lives in the logs view', async () => {
  const ops = await read('components/NativeOperations.tsx');
  const card = ops.slice(ops.indexOf('tr("Diagnostics")'), ops.indexOf('<Dialog open={showLogs}'));
  assert.equal((card.match(/<Btn /g) ?? []).length, 1, 'one button in the Diagnostics card');
  assert.match(card, /View logs/); assert.ok(!/Save logs/.test(card));
  const dialog = ops.slice(ops.indexOf('<Dialog open={showLogs}'), ops.indexOf('</Dialog>', ops.indexOf('<Dialog open={showLogs}')));
  assert.match(dialog, /Save logs/);
});

test('item 4: the Monitor sentence is plain words; the screenshot leaves out the model state and the conversation cache', async () => {
  const { requestSummary } = await load('monitor-summary.ts');
  const f = { number: (n, d = 0) => n.toLocaleString('en', { maximumFractionDigits: d, minimumFractionDigits: d }), tokens: n => n >= 1000 ? `${Math.round(n / 1000)}K` : String(n) };
  assert.equal(requestSummary({ requests: 18, output: 47.4, reading: 1332, reused: 91000 }, f), 'In 18 requests your average output was 47.4 tok/s and prompt reading 1,332 tok/s, reusing 91K tokens.');
  assert.equal(requestSummary({ requests: 3, output: 40 }, f), 'In 3 requests your average output was 40.0 tok/s.');
  assert.equal(requestSummary({ requests: 2 }, f), '2 requests so far.');
  const monitor = await read('screens/Monitor.tsx');
  assert.match(monitor, /monitor-state" aria-label=\{tr\("Model state"\)\} data-snapshot-skip/);
  assert.match(monitor, /monitor-cache" data-snapshot-skip/);
  assert.match(await read('lib/snapshot.ts'), /querySelectorAll\('\[data-snapshot-skip\]'\)/);
});

test('items 6, 7, 8: Bypass permissions everywhere, no deleted-tasks list or suggestions, a + icon, and a plain "Task removed"', async () => {
  const tasks = await read('screens/Tasks.tsx'), detail = await read('components/TaskDetail.tsx'), editor = await read('components/TaskEditor.tsx'), tree = await read('app/TaskTree.tsx'), lib = await read('lib/tasks.ts');
  for (const [name, text] of [['Tasks', tasks], ['TaskDetail', detail], ['TaskEditor', editor], ['TaskTree', tree]]) assert.ok(!/Allowed to make changes/.test(text), `${name} still says it`);
  assert.match(detail, /Bypass permissions/); assert.match(editor, /Bypass permissions/);
  assert.match(lib, /mode:'bypass'/, 'a new task starts as Bypass permissions');
  assert.ok(!/Deleted tasks|task-templates|Restore/.test(tasks), 'no deleted list, suggestions or restore on the Tasks screen');
  assert.match(tasks, /IconBtn name="plus"/);
  assert.ok(!/back=\{|titleDetail=/.test(tasks), 'nothing on the left of the navbar of an opened task');
  assert.ok(!/modelName|"Model"/.test(detail), 'no Model section');
  assert.match(tree, /pushToast\('success', 'Task removed'\)/); assert.match(tasks, /pushToast\('success', 'Task removed'\)/);
});
