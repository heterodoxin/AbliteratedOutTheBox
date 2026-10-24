import test from 'node:test';
import assert from 'node:assert/strict';
import { createEngineStep } from '../server/engine-step.mjs';

const GB = 1e9;
function control({ usable = false, check, update, checkError = null } = {}) {
  const calls = [];
  return {
    calls,
    async engine() { calls.push('engine'); return { usable }; },
    async updateCheck() { calls.push('updateCheck'); if (checkError) throw new Error(checkError); return check; },
    async update(options) { calls.push(['update', options.engine, options.control]); return update(options); },
  };
}
const offer = (status = 'install', extra = {}) => ({ engine: { status, installed: null, offered: { version: '0.1.46' }, download_bytes: 0.5 * GB, ...extra } });
const updated = { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.46' } }] };

test('no engine installed: the step downloads it with the model and reports the stages in order', async () => {
  const seen = [];
  const c = control({
    check: offer(),
    update: async ({ onEvent }) => {
      onEvent({ event: 'progress', percent: 50, done: 0.25 * GB, total: 0.5 * GB });
      onEvent({ event: 'phase', phase: 'verify', message: 'Checking' });
      onEvent({ event: 'phase', phase: 'test', message: 'Test-starting' });
      onEvent({ event: 'phase', phase: 'switch', message: 'Switching' });
      return updated;
    },
  });
  const step = createEngineStep({ control: c, onChange: p => seen.push(p.state) });
  const part = await step.run();
  assert.deepEqual(c.calls, ['engine', 'updateCheck', ['update', true, false]], 'the engine only: Control and the app are not part of a model install');
  assert.deepEqual([...new Set(seen)], ['checking', 'downloading', 'verifying', 'testing', 'switching', 'done']);
  assert.equal(part.state, 'done');
  assert.equal(part.version, '0.1.46');
  assert.equal(step.summaryNote(), null);
});

test('one combined total: model and engine bytes add up, the model-only numbers stay in the parts list', async () => {
  const c = control({ check: offer(), update: async ({ onEvent }) => { onEvent({ event: 'progress', percent: 40, done: 0.2 * GB, total: 0.5 * GB }); return new Promise(() => {}); } });
  const step = createEngineStep({ control: c });
  void step.run();
  await new Promise(r => setTimeout(r, 5));
  const install = { running: true, step: 'Downloading', download: { file: 'model-00001.gguf', percent: 10, done_gb: 6, remaining_gb: 60 } };
  const merged = step.merge(install);
  assert.equal(merged.download.done_gb, 6.2);
  assert.equal(merged.download.remaining_gb, 60.3);
  assert.equal(merged.download.percent, Math.round(100 * 6.2 / 66.5));
  assert.equal(merged.download.file, 'model-00001.gguf', 'the model is still the part that is moving');
  assert.deepEqual(merged.parts.map(p => p.kind), ['model', 'engine']);
  assert.deepEqual([merged.parts[0].done_gb, merged.parts[0].total_gb], [6, 66]);
  assert.equal(merged.engine.state, 'downloading');
  assert.equal(install.download.done_gb, 6, 'merge does not change its input');
});

test('a failed engine download keeps the model and says how to retry', async () => {
  const c = control({ check: offer(), update: async () => { throw new Error('the connection dropped'); } });
  const step = createEngineStep({ control: c });
  const part = await step.run();
  assert.equal(part.state, 'failed');
  assert.equal(part.retry, 'Update engine');
  assert.match(part.message, /^Your model is ready\. The engine for this PC couldn't be downloaded\. .*Update engine in Settings/);
  assert.doesNotMatch(part.message, /dropped|HTTP|exit|sha/i);
  assert.match(part.details, /connection dropped/, 'the reason is kept for the Details disclosure');
  assert.equal(step.summaryNote(), part.message);
  const merged = step.merge({ running: false, download: { file: 'm', percent: 100, done_gb: 10, remaining_gb: 0 } });
  assert.equal(merged.download.remaining_gb, 0, 'a failed engine does not leave the progress bar short');
});

test('a feed that cannot be reached is a failure only when there is no engine yet', async () => {
  const none = await createEngineStep({ control: control({ usable: false, checkError: 'could not be read' }) }).run();
  assert.equal(none.state, 'failed');
  const have = await createEngineStep({ control: control({ usable: true, checkError: 'could not be read' }) }).run();
  assert.equal(have.state, 'skipped');
  assert.match(have.message, /engine for this PC is installed/);
  assert.equal(have.details, undefined, 'nothing to say when an engine is installed');
});

test('an engine that is already current downloads nothing; one with no package for this PC is a failure', async () => {
  const c = control({ usable: true, check: { engine: { status: 'current', installed: { version: '0.1.46' } } } });
  const part = await createEngineStep({ control: c }).run();
  assert.equal(part.state, 'current');
  assert.deepEqual(c.calls, ['engine', 'updateCheck']);
  const none = await createEngineStep({ control: control({ usable: false, check: { engine: { status: 'none-for-this-pc', reasons: ['0.1.46 is for linux-x86_64, this PC is windows-x86_64'] } } }) }).run();
  assert.equal(none.state, 'failed');
  assert.doesNotMatch(none.message, /windows-x86_64/);
  assert.match(none.details, /windows-x86_64/);
});

test('without a component updater the step is skipped and never throws', async () => {
  const part = await createEngineStep({ control: {} }).run();
  assert.equal(part.state, 'skipped');
  assert.equal(createEngineStep({ control: {} }).merge(null), null);
});
