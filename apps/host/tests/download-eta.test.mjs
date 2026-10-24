import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const built = await build({ stdin: { contents: "export * from './src/lib/download-eta';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' }, external: ['react'] });
const src = built.outputFiles[0].text.replace(/from\s*["']react["']/, 'from "data:text/javascript,export const useRef=()=>({current:null}),useEffect=()=>{},useState=v=>[v,()=>{}]"');
const { etaMinutes, etaLabel, installProgress } = await import(`data:text/javascript;base64,${Buffer.from(src).toString('base64')}`);

test('the time left comes from the recent rate, and says nothing until there is one', () => {
  assert.equal(etaMinutes([{ at: 0, done: 1 }], 50), null);
  assert.equal(etaMinutes([{ at: 0, done: 1 }, { at: 5000, done: 3 }], 50), null, 'under ten seconds of samples');
  assert.equal(etaMinutes([{ at: 0, done: 1 }, { at: 20000, done: 1 }], 50), null, 'nothing moved');
  assert.equal(etaMinutes([{ at: 0, done: 0 }, { at: 20000, done: 10 }], 60), 2, '0.5 GB/s, 60 GB left: 120 s');
  assert.equal(etaMinutes([{ at: 0, done: 0 }, { at: 60000, done: 1 }], 0.1), 1, 'never below one minute');
});

test('the label is minutes, and hours past an hour and a half', () => {
  assert.equal(etaLabel(null), '');
  assert.equal(etaLabel(3), 'About 3 min left');
  assert.equal(etaLabel(125), 'About 2 h 5 min left');
  assert.equal(etaLabel(120), 'About 2 h left');
});

test('one count over all the parts of the job: the model files and, when there is one, the engine', () => {
  const model = { kind: 'model', done_gb: 30, total_gb: 60, state: 'downloading' };
  const engine = { kind: 'engine', done_gb: 0.3, total_gb: 0.7, state: 'downloading' };
  const both = installProgress({ parts: [model, engine], download: { done_gb: 0, total_gb: 1 } });
  assert.equal(both.total_gb, 60.7);
  assert.ok(Math.abs(both.done_gb - 30.3) < 1e-9);
  assert.equal(both.percent, 49);
  assert.ok(Math.abs(both.remaining_gb - 30.4) < 1e-9);
  const modelOnly = installProgress({ parts: [model] });
  assert.equal(modelOnly.total_gb, 60);
  assert.equal(installProgress({ parts: [model, { ...engine, state: 'skipped' }, { kind: 'engine', state: 'current', total_gb: 1, done_gb: 1 }] }).total_gb, 60, 'a part that is skipped or already current is not downloaded');
  // no parts: the job's own combined figures
  assert.deepEqual(installProgress({ download: { done_gb: 5, total_gb: 50, percent: 10 } }), { done_gb: 5, total_gb: 50, remaining_gb: 45, percent: 10, file: undefined });
  assert.equal(installProgress({}), undefined);
  assert.equal(installProgress(undefined), undefined);
});
