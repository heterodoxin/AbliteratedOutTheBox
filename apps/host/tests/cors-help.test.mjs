// #3208: the "Allow browser origins" setting says in plain words that it does nothing in the app, where CORS really is, and how to limit it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { expertParams } from '../server/catalog.mjs';

const languages = ['en', 'de', 'es', 'ja', 'ko', 'pt', 'ru', 'zh'];

test('the CORS setting explains itself and the explanation exists in all 8 languages', async () => {
  const cors = expertParams().find(p => p.id === 'server.cors');
  assert.match(cors.effect, /does nothing in the app/);
  assert.match(cors.effect, /"cors_origins"/);
  const english = JSON.parse(await readFile(new URL('../src/locales/en.json', import.meta.url), 'utf8'));
  const id = Object.keys(english).find(key => english[key] === cors.effect);
  assert.ok(id, 'the English text is in the catalog');
  for (const language of languages) {
    const catalog = JSON.parse(await readFile(new URL(`../src/locales/${language}.json`, import.meta.url), 'utf8'));
    assert.ok(catalog[id]?.trim(), `${language} has ${id}`);
    assert.ok(catalog[id].includes('cors_origins'), `${language} keeps the config key`);
    assert.ok(catalog[id].includes('8080'), `${language} keeps the port`);
    if (language !== 'en') assert.notEqual(catalog[id], english[id], `${language} is translated`);
  }
});
