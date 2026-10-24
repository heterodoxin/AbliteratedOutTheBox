import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { nativeUnloadPlan } from '../server/storage-check.mjs';

const loaded = [{ id: 'a', model: 'gyro-gyro-s' }, { id: 'b', model: 'iq3_s' }];

test('O01: checking the same folder unloads nothing', () => {
  assert.deepEqual(nativeUnloadPlan({ changed: false, unload: true, instances: loaded }), { refuse: null, ids: [] });
  assert.deepEqual(nativeUnloadPlan({ changed: false, unload: undefined, instances: loaded }), { refuse: null, ids: [] });
});

test('O01: changing the folder unloads the loaded models only when the person agreed', () => {
  assert.match(nativeUnloadPlan({ changed: true, unload: false, instances: loaded }).refuse, /Unload native models/);
  assert.deepEqual(nativeUnloadPlan({ changed: true, unload: true, instances: loaded }), { refuse: null, ids: ['a', 'b'] });
  assert.deepEqual(nativeUnloadPlan({ changed: true, unload: false, instances: [] }), { refuse: null, ids: [] });
  assert.deepEqual(nativeUnloadPlan({ changed: true, unload: true, instances: undefined }), { refuse: null, ids: [] });
});

test('O01: the route uses that plan and the screen asks for an unload only when the folder differs', async () => {
  const route = await readFile(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(route, /nativeUnloadPlan\(\{changed,unload:body\.unload,instances:native\.state\(\)\.instances\}\)/);
  assert.doesNotMatch(route, /for\(const instance of native\.state\(\)\.instances\)await native\.unload\(instance\.id\);\}\s*\n\s*storageChanging = true/);
  const screen = await readFile(new URL('../src/components/SettingsStorage.tsx', import.meta.url), 'utf8');
  assert.match(screen, /changeModelDirectory\(path, path\.trim\(\) !== currentDirectory\)/);
});

test('F02: Settings -> Storage has ONE Downloaded models list and one total, with the shared files as a line of their own', async () => {
  const screen = await readFile(new URL('../src/components/SettingsStorage.tsx', import.meta.url), 'utf8');
  assert.equal((screen.match(/<Field wide label=\{tr\("Downloaded models"\)\}/g) ?? []).length, 1, 'no second section that contradicts the first');
  assert.match(screen, /totalGb/);
  assert.match(screen, /Shared files/);
  assert.match(screen, /No downloaded models/);
});

test('item 8: "Strata" in Settings is the app\'s version; Strata Control\'s 0.1.45 is only a fallback; the update check gets the app version too', async () => {
  const { appVersionToShow } = await import('../server/app-version.mjs');
  assert.equal(appVersionToShow('0.2.0-beta', '0.1.45'), '0.2.0-beta');
  assert.equal(appVersionToShow('dev', '0.1.45'), '0.1.45', 'a source checkout has no app version: the next best number');
  assert.equal(appVersionToShow(undefined, null), null);
  assert.equal(appVersionToShow('dev', null), 'dev');
  const { build } = await import('esbuild');
  const out = await build({ stdin: { contents: "export * from './src/lib/app-version';", resolveDir: process.cwd() }, bundle: true, write: false, platform: 'node', format: 'esm' });
  const { appVersionLabel } = await import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`);
  assert.equal(appVersionLabel('0.2.0-beta', '0.1.45'), '0.2.0-beta');
  assert.equal(appVersionLabel('dev', '0.1.45'), '0.1.45');
  assert.equal(appVersionLabel(undefined, undefined), undefined);
  const route = await readFile(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(route, /appVersion:appVersionToShow\(setupState\(\)\.app\.version,native\.state\(\)\.version\)/);
  const updates = await readFile(new URL('../src/components/SettingsUpdates.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(updates, /native\?\.version \?\? appVersion/, 'the Strata row no longer prefers Control\'s version');
  assert.match(updates, /AboutVersions/);
});

test('Cargo.toml, tauri.conf.json and the desktop package agree on the app version the host reports', async () => {
  const root = new URL('../../desktop/', import.meta.url);
  const cargo = /^version\s*=\s*"([^"]+)"/m.exec(await readFile(new URL('src-tauri/Cargo.toml', root), 'utf8'))[1];
  const pkg = JSON.parse(await readFile(new URL('package.json', root), 'utf8')).version;
  assert.equal(cargo, pkg);
  assert.equal(cargo, '0.2.3');
});
