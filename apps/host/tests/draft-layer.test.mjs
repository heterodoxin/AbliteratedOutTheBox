/**
 * The model's own choices outlive setup: "Draft layer (MTP)" off, picture reading off, a context size set by hand.
 * Setup writes --mtp, --spec-min-p and --vision into the arguments every time it runs; the host applies the saved choices on every
 * load. Nothing here starts an engine; Control is a fake that records what the host saved and asked it to start.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { withoutDraftLayer, NATIVE_FIELDS, APP_OWNED_FIELDS, validateNativeSettings } from '../server/native-settings.mjs';
import { checkRunConfig } from '../server/run-config-check.mjs';

const ARGS = ['--pack', '/p', '--expert-cache', 'auto', '--spec', '4', '--spec-min-p', '0.5', '--mtp', '/m/rt', '--max-context', '40000', '--kv', 'int8', '--vision'];

function fixture(config = {}, { gpuGb = 12 } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-draft-test-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const path = join(install, 'strata-q2_0.json');
  writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ARGS, model_name: 'Qwen q2_0', port: 8080, ...config }));
  const saved = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.2.0',
    async installedModels() { return { install_dir: install, models: [{ id: 'q2_0', name: 'Qwen q2_0', source_config: path, component_status: 'present', context_tokens: 40000 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.2.0', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const native = createNativeControl({ control, hardwareProbe: () => [{ index: 0, vram_gb: gpuGb }], directory: data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [] });
  return { native, saved, path, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
const visionSection = { exe: 'E:\\e\\strata-vision.exe', mmproj: 'E:\\mm.gguf', model: 'E:\\m.gguf', gpu: false, max_tokens: 300 };

test('withoutDraftLayer takes out --mtp, --spec-min-p and --batch-mtp and keeps --spec', () => {
  assert.deepEqual(withoutDraftLayer(ARGS), ['--pack', '/p', '--expert-cache', 'auto', '--spec', '4', '--max-context', '40000', '--kv', 'int8', '--vision']);
  assert.deepEqual(withoutDraftLayer(['--mtp', '2', '--spec', '2', '--batch-mtp']), ['--spec', '2'], 'Qwen3.6: a draft count');
  assert.deepEqual(withoutDraftLayer(['--spec', '4', '--mtp']), ['--spec', '4'], 'a flag with no value at the end');
  assert.deepEqual(withoutDraftLayer(['--mtp', '--spec', '4']), ['--spec', '4'], 'a flag whose next word is another flag');
});

test('"Draft layer (MTP)" is a setting of the app, not frozen, and "MTP per request" is not frozen either', async () => {
  assert.ok(NATIVE_FIELDS.some(f => f.id === 'mtp' && f.type === 'boolean'));
  assert.deepEqual(APP_OWNED_FIELDS.sort(), ['batchMtp', 'drafting', 'mtp']);
  const f = fixture();
  try {
    await f.native.refresh(true);
    for (const id of ['mtp', 'batchMtp', 'drafting']) assert.equal(f.native.state().fields.find(x => x.id === id).supported, true, `${id} must not be disabled`);
    assert.deepEqual(validateNativeSettings({ mtp: false }), { mtp: false });
    assert.equal(checkRunConfig(JSON.stringify({ exe: 'e', tokenizer: 't', args: [], mtp: false })).issues.length, 0, 'the file check knows the key');
  } finally { await f.native.close(); f.cleanup(); }
});

test('saving "Draft layer" off keeps it in the model file, and every load starts without --mtp', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    f.native.save('q2_0', { values: { mtp: false } });
    assert.equal(JSON.parse(readFileSync(f.path, 'utf8')).mtp, false);
    // a setup run rewrites the arguments (here: the same text with --mtp back) and keeps the key; the load applies it
    const job = await finish(f.native, f.native.operation('load', { model: 'q2_0', vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    const args = f.saved[0].args;
    assert.ok(!args.includes('--mtp') && !args.includes('--spec-min-p') && !args.includes('--batch-mtp'), args.join(' '));
    assert.equal(args[args.indexOf('--spec') + 1], '4', '--spec stays: prompt lookup still uses its windows');
    assert.equal(args[args.indexOf('--max-context') + 1], '40000', 'a context set by hand is not touched');
    // and back to the default: the key goes, the draft layer is in again
    f.native.save('q2_0', { values: { mtp: null } });
    assert.equal('mtp' in JSON.parse(readFileSync(f.path, 'utf8')), false);
  } finally { await f.native.close(); f.cleanup(); }
});

test('the load with the draft layer off in its own values, and a retry, both keep it off', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'q2_0', vision: 'no', values: { mtp: false } }).id);
    assert.equal(job.state, 'done', job.error);
    assert.ok(!f.saved[0].args.includes('--mtp'));
    assert.ok(!f.saved[0].args.includes('--batch-mtp'), 'no draft layer, so no draft layer per request either, even on a big card');
  } finally { await f.native.close(); f.cleanup(); }
});

test('picture reading off sticks: no vision section and no --vision on the engine, whatever setup wrote', async () => {
  const f = fixture({ vision: visionSection, ui_vision_enabled: false, ui_vision_device: 'yes' });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'q2_0' }).id);   // the screen sends nothing: the saved choice decides
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.saved[0].vision, undefined);
    assert.ok(!f.saved[0].args.includes('--vision'));
    assert.equal(JSON.parse(readFileSync(f.path, 'utf8')).ui_vision_enabled, false, 'the choice stays saved');
  } finally { await f.native.close(); f.cleanup(); }
});

test('picture reading on keeps --vision; the Models screen sees the prepared pictures instead of "off"', async () => {
  const f = fixture({ vision: visionSection });
  try {
    await f.native.refresh(true);
    assert.equal(f.native.state().installed.models[0].images, 'yes', 'prepared: the screen loads straight away, no setup run');
    const job = await finish(f.native, f.native.operation('load', { model: 'q2_0' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.ok(f.saved[0].args.includes('--vision'));
    assert.ok(f.saved[0].vision);
  } finally { await f.native.close(); f.cleanup(); }
  const g = fixture();
  try { await g.native.refresh(true); assert.equal(g.native.state().installed.models[0].images, 'off', 'no vision section: not prepared'); } finally { await g.native.close(); g.cleanup(); }
});
