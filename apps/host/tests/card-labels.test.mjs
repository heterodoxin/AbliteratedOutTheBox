/** Chat card wording: no "Demo" labels, "Ran" for a tool nobody was asked about, "Approved" only when someone approved. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const languages = ['en', 'de', 'es', 'ja', 'ko', 'pt', 'ru', 'zh'];
const catalog = Object.fromEntries(languages.map(l => [l, JSON.parse(readFileSync(resolve(root, 'src/locales', `${l}.json`), 'utf8'))]));
const idOf = text => Object.entries(catalog.en).find(([, v]) => v === text)?.[0];

test('no catalog string, in any language, calls anything a demo', () => {
  const word = /demo|デモ|데모|Демо|Демонстр|演示|demonstr|demostr/i;
  for (const l of languages) {
    const hits = Object.entries(catalog[l]).filter(([k, v]) => word.test(v) && !/^s0171$/.test(k));
    assert.deepEqual(hits, [], `${l}: ${JSON.stringify(hits)}`);
  }
});

test('the approval and result texts of a chat card exist in every language, translated', () => {
  for (const text of ['Needs your approval', 'Denied. Nothing ran.', 'Approval recorded. Nothing was executed.', 'Ran', 'Not allowed in this mode. Nothing ran.']) {
    const id = idOf(text);
    assert.ok(id, `${text} has a catalog id`);
    for (const l of languages.filter(l => l !== 'en')) assert.notEqual(catalog[l][id], text, `${l} translates "${text}"`);
  }
});

test('the card shows "Ran" unless the person approved, and never "Demo approval"', () => {
  const chat = readFileSync(resolve(root, 'src/screens/Chat.tsx'), 'utf8');
  assert.ok(!/Demo approval/i.test(chat));
  assert.match(chat, /t\.live && t\.state === 'done' && t\.approved \? 'Approved' : 'Done'/);
  const store = readFileSync(resolve(root, 'src/lib/store.ts'), 'utf8');
  assert.ok(!/Demo approval/i.test(store));
  assert.match(store, /mode:state\.sessions\.find\(s=>s\.id===sessionId\)\?\.preferences\?\.mode\?\?'manual'/, 'the Mode menu choice travels with the turn');
});

test('every language has the same keys', () => {
  const keys = Object.keys(catalog.en).sort();
  for (const l of languages) assert.deepEqual(Object.keys(catalog[l]).sort(), keys, l);
});
