/** One answer for "can this model take a picture?" (src/lib/pictures.ts): the composer and Send agree, pictures are on by default. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const source = await readFile(new URL('../src/lib/pictures.ts', import.meta.url), 'utf8');
const { code } = await transform(source, { loader: 'ts', format: 'esm' });
const { pictureStatus } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

const native = (vision, { images = true, ready = true } = {}) => ({
  defaultId: 'i1', instances: [{ id: 'i1', model: 'Gyro-S', status: ready ? 'ready' : 'loading', vision }],
  catalog: { families: [{ family: 'gyro', images, sizes: [{ id: 'Gyro-S' }] }] },
});
const api = (model, protocol = 'openai-chat') => ({ connections: [{ id: 'c', protocol, models: [{ id: 'm', ...model }] }] });
const pick = { connectionId: 'c', modelId: 'm' };

test('a local model that was loaded with picture reading takes pictures', () => {
  assert.equal(pictureStatus(native(true), null, null).ok, true);
});

test('a local model that could read pictures but was loaded without says what to do (the stale-flag case of imageissue2)', () => {
  const r = pictureStatus(native(false), null, null);
  assert.equal(r.ok, false); assert.equal(r.why, 'local-off'); assert.match(r.message, /Open Models, turn on Read pictures and load it again/);
});

test('a local model that cannot read pictures, and no model at all, say so plainly', () => {
  assert.equal(pictureStatus(native(false, { images: false }), null, null).why, 'local-cannot');
  assert.equal(pictureStatus(native(false, { ready: false }), null, null).why, 'none');
  assert.equal(pictureStatus(null, null, null).why, 'none');
});

test('an API model takes pictures unless it was switched off or its server said no', () => {
  assert.equal(pictureStatus(null, pick, api({})).ok, true, 'nothing said: on');
  assert.equal(pictureStatus(null, pick, api({ reads: true })).ok, true);
  const off = pictureStatus(null, pick, api({ reads: false }));
  assert.equal(off.ok, false); assert.equal(off.why, 'api-off'); assert.match(off.message, /turn on Read pictures with this model/);
  assert.equal(pictureStatus(null, pick, api({}, 'anthropic')).why, 'api-style');
});

const noModelSource = await readFile(new URL('../src/lib/no-model.ts', import.meta.url), 'utf8');
const { noModelNotice } = await import(`data:text/javascript;base64,${Buffer.from((await transform(noModelSource, { loader: 'ts', format: 'esm' })).code).toString('base64')}`);
test('no model: Send says so plainly, shows a running download, and stays quiet when a model can answer', () => {
  assert.equal(noModelNotice({ instances: [], jobs: [] }).message, 'No model is loaded. Load one in Models first.');
  assert.equal(noModelNotice(null).message, 'No model is loaded. Load one in Models first.');
  assert.match(noModelNotice({ instances: [], jobs: [{ kind: 'install', state: 'running', result: { install: { download: { percent: 41.6 } } } }] }).message, /downloading \(42%\)/);
  assert.match(noModelNotice({ instances: [{ status: 'loading' }], jobs: [] }).message, /still loading/);
  assert.equal(noModelNotice({ instances: [{ status: 'ready' }], jobs: [] }), null);
  assert.equal(noModelNotice({ instances: [], jobs: [] }, { remote: true }), null);
  assert.equal(noModelNotice({ instances: [], jobs: [] }, { libraryLoaded: true }), null);
  assert.equal(noModelNotice({ instances: [], jobs: [{ kind: 'benchmark', state: 'running' }] }).title, 'No model is loaded');
});
