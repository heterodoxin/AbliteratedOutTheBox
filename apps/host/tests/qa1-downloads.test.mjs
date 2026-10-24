import test from 'node:test';
import { readFileSync as readSource } from 'node:fs';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createEngineStep } from '../server/engine-step.mjs';

const compiled = await build({
  stdin: {
    contents: "export * from './src/lib/native'; export * from './src/lib/download-eta'; export * from './src/lib/native-fit'; export * from './src/lib/native-names'; export * from './src/lib/native-catalog'; export * from './src/lib/i18n';",
    resolveDir: process.cwd(),
  },
  bundle: true, write: false, platform: 'node', format: 'esm',
});
const lib = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const job = (progress, state = 'running') => ({ id: 'j1', kind: 'install', state, startedAt: '', summary: '', error: null, result: { install: progress } });
const stateWith = (jobs, extra = {}) => ({ available: true, repo: '', summary: '', error: null, catalog: null, installed: null, hardware: null, instances: [], jobs, ...extra });
const answer = body => ({ ok: true, status: 200, statusText: 'OK', text: async () => JSON.stringify(body) });

test('a slow forced read never holds the light poll: a running download keeps moving', async () => {
  const seen = [];
  globalThis.fetch = async url => {
    seen.push(url);
    if (String(url).includes('refresh=1')) { await new Promise(r => setTimeout(r, 400)); return answer(stateWith([job({ download: { done_gb: 1, total_gb: 10 } })])); }
    return answer(stateWith([job({ download: { done_gb: 5, total_gb: 10 } })]));
  };
  const slow = lib.refreshNative(true);
  const t0 = Date.now();
  await lib.refreshNative();
  assert.ok(Date.now() - t0 < 200, 'the light read answered while the forced one was still running');
  assert.equal(lib.nativeState().jobs[0].result.install.download.done_gb, 5);
  await slow;
  assert.ok(seen.some(u => String(u).includes('refresh=1')) && seen.some(u => !String(u).includes('refresh=1')));
});

test('a second forced read waits for the first and then reads again (it never joins a read that began before the action)', async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; await new Promise(r => setTimeout(r, 50)); return answer(stateWith([], { summary: `read ${calls}` })); };
  const a = lib.refreshNative(true), b = lib.refreshNative(true), c = lib.refreshNative(true);
  await Promise.all([a, b, c]);
  assert.equal(calls, 2, 'one read for the first, one more for everything asked meanwhile');
  assert.equal(lib.nativeState().summary, 'read 2');
});

test('the host pushes its native state and the page takes it at once', () => {
  lib.applyNativeEvent(stateWith([job({ download: { done_gb: 7, total_gb: 10 } })], { summary: 'pushed' }));
  assert.equal(lib.nativeState().summary, 'pushed');
  lib.applyNativeEvent({ nonsense: true });
  lib.applyNativeEvent(null);
  assert.equal(lib.nativeState().summary, 'pushed', 'a frame that is not a native state changes nothing');
});

test('the poll is quick while a job runs, normal otherwise, and stops while the window is hidden', () => {
  assert.equal(lib.nativePollDelay(null), 2500);
  assert.equal(lib.nativePollDelay(stateWith([job({})], {})), 1000);
  assert.equal(lib.nativePollDelay(stateWith([job({}, 'done')])), 2500);
  assert.equal(lib.nativePollDelay(stateWith([job({})]), true), null);
});

const parts = (model, vision, engine = { kind: 'engine', name: 'Engine', state: 'done', done_gb: 0.4, total_gb: 0.4 }) => [model, vision, engine];
const modelPart = (state, done, total, files_done, files_total) => ({ kind: 'model', name: 'Model files', state, done_gb: done, total_gb: total, files_done, files_total });
const visionPart = (state, done = 0) => ({ kind: 'vision', name: 'Picture reader', state, done_gb: done, total_gb: 0.9 });

test('a model in parts says which part is under way, then the picture reader', () => {
  lib.setTranslationLanguage('en');
  const second = { parts: parts(modelPart('downloading', 20, 66, 1, 2), visionPart('waiting')) };
  assert.deepEqual(lib.installStage(second), { kind: 'model', index: 2, count: 2 });
  assert.equal(lib.stageText(lib.installStage(second)), 'Downloading model files · part 2 of 2');
  const reader = { parts: parts(modelPart('done', 66, 66, 2, 2), visionPart('downloading', 0.3)) };
  assert.deepEqual(lib.installStage(reader), { kind: 'vision' });
  assert.equal(lib.stageText(lib.installStage(reader)), 'Downloading the picture reader…');
  assert.equal(lib.stageText(lib.installStage(undefined)), 'Downloading model files…');
  const single = { parts: [modelPart('downloading', 1, 20, 0, 1), visionPart('waiting')] };
  assert.equal(lib.stageText(lib.installStage(single)), 'Downloading model files…', 'one file is not "part 1 of 1"');
});

test('the figure counts every part: the model whole but the picture reader not yet is not finished', () => {
  const p = lib.installProgress({ parts: parts(modelPart('done', 10, 10, 2, 2), visionPart('downloading', 0.3)) });
  assert.equal(p.percent, 94, 'the bar is not at its end while the picture reader is still coming');
  assert.ok(Math.abs(p.total_gb - 11.3) < 0.01, `total ${p.total_gb}`);
  assert.ok(p.remaining_gb > 0.5);
  const half = lib.installProgress({ parts: parts(modelPart('downloading', 33, 66, 1, 2), visionPart('waiting')) });
  assert.equal(half.percent, Math.floor(100 * (33 + 0.4) / (66 + 0.9 + 0.4)));
});

test('the engine step keeps the install job\'s own parts (model parts, picture reader) and adds the engine', () => {
  const step = createEngineStep({ control: null });
  const install = { running: true, step: '5/7: downloading', download: { file: 'picture reader', done_gb: 66.3, total_gb: 66.9, remaining_gb: 0.6, percent: 99 },
    parts: [modelPart('done', 66, 66, 2, 2), visionPart('downloading', 0.3)] };
  const merged = step.merge(install);
  assert.deepEqual(merged.parts.map(p => p.kind), ['model', 'vision', 'engine']);
  assert.equal(merged.parts[0].files_total, 2);
  const bare = step.merge({ running: true, download: { file: 'model files', done_gb: 1, remaining_gb: 9 } });
  assert.deepEqual(bare.parts.map(p => p.kind), ['model', 'engine'], 'an install without part information keeps the old two-part list');
});

test('the recommended four come first; the others are what the "More versions" list holds', () => {
  const v = (sourceFamily, model) => ({ sourceFamily, model, id: `${sourceFamily}-${model}` });
  const all = [v('qwen', 'Q2_0'), v('gyro', 'Gyro-S'), v('swift', 'IQ2_XS'), v('qwen', 'IQ3_S'), v('unsloth', 'UD-Q4_K_XL'), v('nvfp4', 'NVFP4')];
  const rec = lib.recommendedVersions(all).map(x => x.v.model);
  assert.deepEqual(rec, ['Gyro-S', 'IQ2_XS', 'IQ3_S', 'UD-Q4_K_XL']);
  assert.deepEqual(all.filter(x => !rec.includes(x.model)).map(x => x.model), ['Q2_0', 'NVFP4']);
  assert.deepEqual(lib.recommendedVersions([v('qwen36', 'UD-Q4_K_XL')]), [], 'Qwen3.6 has no recommended set: its versions are all shown');
});

test('the three NVFP4 versions have clear names and one line that tells them apart', () => {
  lib.setTranslationLanguage('en');
  const names = ['NVFP4', 'NVFP4-NVIDIA', 'NVFP4-QAD'].map(m => lib.nativeVersionLabel(m));
  assert.deepEqual(names, ['NVFP4 (Strata’s version)', 'NVFP4 (NVIDIA’s version)', 'NVFP4 (NVIDIA, trained for FP4)']);
  assert.equal(new Set(names).size, 3);
  const lines = ['NVFP4', 'NVFP4-NVIDIA', 'NVFP4-QAD'].map(m => lib.versionNote(m));
  assert.match(lines[0], /best quality/i);
  assert.match(lines[1], /NVIDIA’s own NVFP4 checkpoint/);
  assert.match(lines[2], /trained for FP4/);
  assert.match(lines[2], /faster on short prompts/);
  assert.equal(lib.versionNote('IQ3_S'), '');
  assert.equal(lib.nativeShortName('nvfp4-nvfp4-nvidia'), 'NVFP4 (NVIDIA’s version)');
  for (const language of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko']) {
    lib.setTranslationLanguage(language);
    const local = ['NVFP4', 'NVFP4-NVIDIA', 'NVFP4-QAD'].map(m => lib.nativeVersionLabel(m));
    assert.equal(new Set(local).size, 3, `${language}: three different names`);
    assert.ok(local.every((n, i) => n !== names[i]), `${language}: translated`);
    assert.ok(['NVFP4', 'NVFP4-NVIDIA', 'NVFP4-QAD'].every((m, i) => lib.versionNote(m) !== lines[i]), `${language}: the line is translated`);
  }
  lib.setTranslationLanguage('en');
});

test('4d: picture reading is on unless the person turned it off (no saved choice is not "off")', async () => {
  const { nativeFixture } = await import('./helpers/native-fixture.mjs');
  const { readFileSync, writeFileSync } = await import('node:fs');
  const { join } = await import('node:path');
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const first = () => f.control.state().installed.models.find(m => m.model === 'iq2_xs');
    assert.equal(first().visionEnabled, undefined, 'a model never loaded and never turned off');
    const path = join(f.repo, 'strata-iq2_xs.json');
    const cfg = JSON.parse(readFileSync(path, 'utf8'));
    writeFileSync(path, JSON.stringify({ ...cfg, ui_vision_enabled: false }));
    await f.control.refresh(true);
    assert.equal(first().visionEnabled, false, 'turned off by the person: stays off');
    writeFileSync(path, JSON.stringify({ ...cfg, ui_vision_enabled: true, ui_vision_device: 'cpu' }));
    await f.control.refresh(true);
    assert.equal(first().visionEnabled, true);
  } finally { await f.cleanup(); }
  const screen = readSource(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  assert.match(screen, /setVision\(installed\.visionEnabled === false \? 'no'/);
  assert.match(screen, /else\s+setVision\('yes'\)/, 'a version that is not downloaded starts with Read pictures on');
});

test('no recognised engine: the app still downloads, warns in the card, and loading says it with the next step', async () => {
  const { ENGINE_NOT_RECOGNISED } = await import('../server/native-control.mjs');
  assert.match(ENGINE_NOT_RECOGNISED, /Settings → Engine/);
  assert.match(ENGINE_NOT_RECOGNISED, /didn’t recognise an engine/);
  const source = readSource(new URL('../server/native-control.mjs', import.meta.url), 'utf8');
  assert.equal((source.match(/ENGINE_NOT_RECOGNISED\+/g) ?? []).length, 2, 'both load paths use it');
  const models = readSource(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  const guided = readSource(new URL('../src/components/GuidedSetup.tsx', import.meta.url), 'utf8');
  assert.match(models, /<EngineWarning native=\{native\}\/>/);
  assert.match(guided, /<EngineWarning native=\{native\}\/>/);
  const warn = readSource(new URL('../src/components/EngineWarning.tsx', import.meta.url), 'utf8');
  assert.match(warn, /The model will download, but it may not run until the right engine is installed \(Settings → Engine\)\./);
  // the install is not refused for a missing engine (only the engine-minimum refusal for a version that is too old stays)
  assert.doesNotMatch(source.slice(source.indexOf('async function install(job,args)'), source.indexOf('function terminate')), /engine\.usable/);
});

test('the draft layer download of setup step 6 is a stage of its own and counts in the one figure (#3257)', () => {
  lib.setTranslationLanguage('en');
  const draft = { parts: [{ kind: 'model', name: 'Model files', state: 'done', done_gb: 60, total_gb: 60 }, { kind: 'draft', name: 'Draft layer', state: 'downloading', done_gb: 2, total_gb: 5 }] };
  assert.deepEqual(lib.installStage(draft), { kind: 'draft' });
  assert.equal(lib.stageText(lib.installStage(draft)), 'Preparing your model…');
  const p = lib.installProgress(draft);
  assert.equal(p.total_gb, 65);
  assert.equal(p.done_gb, 62);
});
