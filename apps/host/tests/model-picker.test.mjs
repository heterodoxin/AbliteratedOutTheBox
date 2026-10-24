import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/model-picker.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { pickerGroups } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const version = (id, family, state, diskGb, extra = {}) => ({ id, family, state, diskGb, ...extra });

test('chat picker groups every downloaded family and puts its running version first', () => {
  const library = { loaded: 'qwen-balanced', variants: [
    version('other-small', 'Other model', 'installed', 3),
    version('qwen-small', 'Qwen', 'installed', 11),
    version('qwen-balanced', 'Qwen', 'loaded', 19),
    version('qwen-original', 'Qwen', 'available', 61),
    version('other-large', 'Other model', 'installed', 8),
    version('not-downloaded', 'Missing family', 'available', 1),
  ] };
  const before = structuredClone(library);
  assert.deepEqual(pickerGroups(library).map(g => [g.family, g.versions.map(v => v.id)]), [
    ['Qwen', ['qwen-balanced', 'qwen-small']],
    ['Other model', ['other-small', 'other-large']],
  ]);
  assert.deepEqual(library, before);
});

test('downloaded files stay visible during load transitions and recoverable failures', () => {
  const library = { loaded: null, variants: [
    version('fetching', 'Qwen', 'downloading', 1),
    version('failed-download', 'Qwen', 'error', 2),
    version('loading', 'Qwen', 'loading', 3),
    version('unloading', 'Qwen', 'unloading', 4),
    version('failed-load', 'Qwen', 'error', 5, { installedAt: '2026-10-10' }),
  ] };
  assert.deepEqual(pickerGroups(library)[0].versions.map(v => v.id), ['loading', 'unloading', 'failed-load']);
  assert.deepEqual(pickerGroups({ loaded: null, variants: [library.variants[0]] }), []);
});
