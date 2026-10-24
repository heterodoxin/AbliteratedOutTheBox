/** Settings shows the last known update check from this computer and never asks the network when it opens. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const load = async name => {
  const { code } = await transform(await readFile(new URL(`../src/lib/${name}.ts`, import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const { readUpdateCache, saveUpdateCache, clearUpdateCache } = await load('update-cache');

const store = new Map();
globalThis.localStorage = { getItem: k => store.get(k) ?? null, setItem: (k, v) => void store.set(k, String(v)), removeItem: k => void store.delete(k) };
const result = (status = 'available') => ({ checkedAt: '2026-10-19T10:00:00Z', strata: { status: 'current', message: 'ok' }, engine: { status, message: 'x' } });

test('a saved check is read back for the same installed versions', () => {
  clearUpdateCache();
  assert.equal(readUpdateCache({ appVersion: '0.2.0', engineVersion: '0.1.45' }), null, 'never checked');
  saveUpdateCache(result(), { appVersion: '0.2.0', engineVersion: '0.1.45' });
  assert.equal(readUpdateCache({ appVersion: '0.2.0', engineVersion: '0.1.45' }).engine.status, 'available');
});

test('a check made for other versions is ignored, so an old "Update available" does not linger after an update', () => {
  saveUpdateCache(result(), { appVersion: '0.2.0', engineVersion: '0.1.45' });
  assert.equal(readUpdateCache({ appVersion: '0.2.0', engineVersion: '0.1.46' }), null);
  assert.equal(readUpdateCache({ appVersion: '0.2.1', engineVersion: '0.1.45' }), null);
});

test('a new Control version, or a version that is not known yet, hides the old result', () => {
  saveUpdateCache(result(), { appVersion: '0.2.0', engineVersion: '0.1.45', controlVersion: '0.1.45' });
  assert.equal(readUpdateCache({ appVersion: '0.2.0', engineVersion: '0.1.45', controlVersion: '0.1.45' }).engine.status, 'available');
  assert.equal(readUpdateCache({ appVersion: '0.2.0', engineVersion: '0.1.45', controlVersion: '0.1.46' }), null, 'Control changed');
  assert.equal(readUpdateCache({ appVersion: null, engineVersion: '0.1.45', controlVersion: '0.1.45' }), null, 'app version still loading');
  assert.equal(readUpdateCache({ appVersion: '0.2.0', engineVersion: '0.1.45' }), null, 'Control version not known');
});

test('unreadable or blocked storage is harmless', () => {
  store.set('strata.update-check.v1', '{not json');
  assert.equal(readUpdateCache({}), null);
  const saved = globalThis.localStorage;
  globalThis.localStorage = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); }, removeItem() { throw new Error('blocked'); } };
  assert.equal(readUpdateCache({}), null);
  assert.doesNotThrow(() => saveUpdateCache(result(), {}));
  assert.doesNotThrow(() => clearUpdateCache());
  globalThis.localStorage = saved;
});

test('opening the Engine and Update settings makes no update call: only a button press does', async () => {
  const src = async name => (await readFile(new URL(`../src/${name}`, import.meta.url), 'utf8')).replace(/\r\n/g, '\n');
  const ops = await src('components/NativeOperations.tsx'), upd = await src('components/SettingsUpdates.tsx'), lib = await src('lib/update-check.ts');
  assert.doesNotMatch(ops, /\/api\/updates\/check/, 'NativeOperations has no direct call');
  assert.doesNotMatch(upd, /\/api\/updates\/check/);
  assert.equal(lib.match(/\/api\/updates\/check/g).length, 1, 'one place asks the host');
  // every caller of the check sits inside a function that only a click starts
  for (const [name, text] of [['NativeOperations', ops], ['SettingsUpdates', upd]]) {
    const body = text.replace(/async function (checkForUpdates|check)\(\) \{[\s\S]*?\n    \}\n/, '');
    assert.doesNotMatch(body, /fetchUpdateCheck\(/, `${name}: the check runs only inside the button handler`);
    assert.match(text, /void (checkForUpdates|check)\(\)/, `${name}: a button starts it`);
  }
  // nothing in either component runs a check from an effect
  for (const text of [ops, upd]) for (const effect of text.match(/useEffect\(\(\) => \{[\s\S]*?\n    \}, \[[^\]]*\]\);/g) ?? []) assert.doesNotMatch(effect, /fetchUpdateCheck|updates\/check/);
});
