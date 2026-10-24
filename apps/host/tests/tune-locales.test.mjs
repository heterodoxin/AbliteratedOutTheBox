import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko'];
const catalogs = Object.fromEntries(await Promise.all(languages.map(async l => [l, JSON.parse(await readFile(`src/locales/tune-${l}.json`, 'utf8'))])));
const main = JSON.parse(await readFile('src/locales/en.json', 'utf8'));
const slots = text => [...text.matchAll(/\{\d+\}/g)].map(m => m[0]).sort();
const compiled = await build({ stdin: { contents: "export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { tr, setTranslationLanguage } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

test('the Tune dialog strings exist in every language with the same slots and no empty text', () => {
  for (const l of languages) {
    assert.deepEqual(Object.keys(catalogs[l]), Object.keys(catalogs.en), l);
    for (const [id, source] of Object.entries(catalogs.en)) {
      assert.ok(catalogs[l][id].trim(), `${l}:${id} is empty`);
      assert.deepEqual(slots(catalogs[l][id]), slots(source), `${l}:${id}`);
    }
  }
});

test('the Tune strings do not collide with the main catalog ids or texts, and translate through tr()', () => {
  const mainTexts = new Set(Object.values(main));
  for (const [id, text] of Object.entries(catalogs.en)) {
    assert.ok(!(id in main), id);
    assert.ok(!mainTexts.has(text), `already in the main catalog: ${text}`);
  }
  setTranslationLanguage('de');
  assert.equal(tr('Tune this model'), catalogs.de.tune01);
  assert.equal(tr('about {0} min left', [7]), catalogs.de.tune03.replace('{0}', '7'));
  assert.equal(tr("+{0}% faster answers, saved as '{1}'", [12, 'IQ3_S 128K']), catalogs.de.tune10.replace('{0}', '12').replace('{1}', 'IQ3_S 128K'));
  setTranslationLanguage('en');
  assert.equal(tr('Tune this model'), 'Tune this model');
});

test('the Tune dialog shows the answer-changing switch in plain words, off by default, in the simple dialog', async () => {
  const source = await readFile('src/components/NativeTune.tsx', 'utf8');
  const label = 'Allow settings that change answers very slightly (same quality)';
  assert.ok(Object.values(catalogs.en).includes(label), 'the label is in the catalogs');
  assert.ok(source.includes(`tr("${label}")`));
  assert.match(source, /\[allowSmall, setAllowSmall\] = useState\(false\)/);
  assert.match(source, /type="checkbox" checked=\{allowSmall\}/);
  assert.match(source, /allowSmallDifferences: allowSmall/);
});
