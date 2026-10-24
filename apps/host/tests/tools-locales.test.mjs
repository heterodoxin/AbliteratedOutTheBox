// The strings of the tools and images fixes (src/locales/tools-<lang>.json) exist in every language with the same slots and no empty text,
// do not collide with the main catalog, and translate through tr().
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko'];
const catalogs = Object.fromEntries(await Promise.all(languages.map(async l => [l, JSON.parse(await readFile(`src/locales/tools-${l}.json`, 'utf8'))])));
const main = JSON.parse(await readFile('src/locales/en.json', 'utf8'));
const tune = JSON.parse(await readFile('src/locales/tune-en.json', 'utf8'));
const slots = text => [...text.matchAll(/\{\d+\}/g)].map(m => m[0]).sort();
const compiled = await build({ stdin: { contents: "export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { tr, setTranslationLanguage } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

test('every language has every tools string with the same slots and no empty text', () => {
  for (const l of languages) {
    assert.deepEqual(Object.keys(catalogs[l]), Object.keys(catalogs.en), l);
    for (const [id, source] of Object.entries(catalogs.en)) {
      assert.ok(catalogs[l][id].trim(), `${l}:${id} is empty`);
      assert.deepEqual(slots(catalogs[l][id]), slots(source), `${l}:${id}`);
    }
  }
});

test('the ids and texts do not collide with the main or Tune catalogs, and translate through tr()', () => {
  const seen = new Set([...Object.values(main), ...Object.values(tune)]);
  for (const [id, text] of Object.entries(catalogs.en)) {
    assert.ok(!(id in main) && !(id in tune), id);
    assert.ok(!seen.has(text), `already in another catalog: ${text}`);
  }
  const first = Object.keys(catalogs.en)[0];
  for (const l of languages) {
    setTranslationLanguage(l);
    assert.equal(tr(catalogs.en[first]), catalogs[l][first], l);
  }
  setTranslationLanguage('de');
  const withName = Object.entries(catalogs.en).find(([, t]) => t.includes('{0}'));
  const shown = tr(withName[1].replace('{0}', 'files, a.b'));
  assert.equal(shown, catalogs.de[withName[0]].replace('{0}', 'files, a.b'));
  setTranslationLanguage('en');
});
