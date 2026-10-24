import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';

const bundled = await build({ entryPoints: ['src/lib/loop.ts'], bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { parseLoop, loopEvery, LOOP_MIN_MS } = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);

test('/loop reads how often, then what to do', () => {
  assert.deepEqual(parseLoop('/loop 10m check the build'), { kind: 'start', everyMs: 600000, prompt: 'check the build' });
  assert.deepEqual(parseLoop('/loop 2h summarise my inbox notes'), { kind: 'start', everyMs: 7200000, prompt: 'summarise my inbox notes' });
  assert.deepEqual(parseLoop('/loop every 5 minutes say hi'), { kind: 'start', everyMs: 300000, prompt: 'say hi' });
  assert.equal(parseLoop('hello /loop'), null, 'only a message that starts with /loop is a loop');
  assert.equal(parseLoop('/looping'), null);
});

test('/loop with no time, a tiny time or a huge time is explained in plain words, never run', () => {
  assert.equal(parseLoop('/loop check the build').kind, 'help');
  assert.match(parseLoop('/loop check the build').problem, /how often/);
  assert.match(parseLoop('/loop 10s ping').problem, /1 minute/);
  assert.match(parseLoop('/loop 48h ping').problem, /24 hours/);
  assert.equal(LOOP_MIN_MS, 60000);
});

test('/loop, /loop stop and the words for how often', () => {
  assert.deepEqual(parseLoop('/loop'), { kind: 'status' });
  assert.deepEqual(parseLoop('/loop stop'), { kind: 'stop' });
  assert.deepEqual(parseLoop('/loop OFF'), { kind: 'stop' });
  assert.equal(loopEvery(60000), 'every minute');
  assert.equal(loopEvery(600000), 'every 10 minutes');
  assert.equal(loopEvery(3600000), 'every hour');
  assert.equal(loopEvery(7200000), 'every 2 hours');
});

test('the slash menu lists /time-skip, /goal and /loop with plain descriptions, and /loop no longer just opens Tasks', async () => {
  const chat = await readFile('src/screens/Chat.tsx', 'utf8');
  for (const [label, words] of [['/time-skip', 'Try a change on a copy'], ['/goal', 'keep working until it is done'], ['/loop', 'every few minutes']]) {
    const line = chat.split('\n').find(l => l.includes(`label: '${label}'`));
    assert.ok(line, `${label} is in the menu`);
    assert.ok(line.includes(words), `${label} has a plain description`);
  }
  const commands = await readFile('src/lib/commands.ts', 'utf8');
  assert.ok(!/slash: \['loop'\]/.test(commands), 'the old /loop alias that only opened Tasks is gone');
});

test('the store sends /loop to the loop and /time-skip to a real job when a local model is ready', async () => {
  const store = await readFile('src/lib/store.ts', 'utf8');
  assert.match(store, /parseLoop\(text\.trim\(\)\)/);
  assert.match(store, /startTimeSkipReal\(demoPrompt\)/);
  assert.match(store, /nativeState\(\)\?\.instances\.some\(i => i\.status === 'ready'\)/);
  assert.match(store, /!m\.timeSkipId && !m\.local/, 'notes from the app and trial turns are not sent back to the model');
});

test('the new slash-command, Time-skip and Tools words are translated in every language, with their slots', async () => {
  const i18n = await build({ stdin: { contents: "export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
  const { tr, setTranslationLanguage } = await import(`data:text/javascript;base64,${Buffer.from(i18n.outputFiles[0].text).toString('base64')}`);
  const cases = [
    ['Keep the update', 'de', 'Aktualisierung behalten'],
    ['Put the original back', 'ja', '元に戻す'],
    ['Repeat a message every few minutes while this chat is open', 'es', 'Repite un mensaje cada pocos minutos mientras este chat esté abierto'],
    ['Search the web', 'zh', '搜索网络'],
  ];
  for (const [text, lang, want] of cases) { setTranslationLanguage(lang); assert.equal(tr(text), want, `${lang}: ${text}`); }
  setTranslationLanguage('de');
  assert.equal(tr('Revision {0}', [2]), 'Überarbeitung 2');
  assert.equal(tr('every {0} minutes', [10]), 'alle 10 Minuten');
  assert.equal(tr('Time-skip stopped. The model rejected the credentials. Your files were not changed.'), 'Time-skip wurde gestoppt. The model rejected the credentials. Deine Dateien wurden nicht geändert.');
  setTranslationLanguage('en');
  assert.equal(tr('Keep the update'), 'Keep the update');
});
