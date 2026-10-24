import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { KNOWN_KEYS, checkRunConfig, describeIssues, nearestKey } from '../server/run-config-check.mjs';
import { NATIVE_FIELDS } from '../server/native-settings.mjs';
import { openTextFile, textEditorCommand } from '../server/open-text-file.mjs';

const good = `{
  "exe": "/opt/strata/strata",
  "args": ["--native", "/m/a.gguf", "--max-context", "65536"],
  "env": {"STRATA_PCIE_FRAC_DEFAULT": "old"},
  "sampling": {"temperature": 0.7, "repeat_penalty": 1.1},
  "parallel": 2,
  "aliases": "chat, coding",
  "gpu": [0, 1],
  "layer_split": "16,32",
  "port": 8080
}`;

test('a normal run config has no issues', () => {
  assert.deepEqual(checkRunConfig(good), { ok: true, issues: [] });
  assert.deepEqual(checkRunConfig('\uFEFF{}'), { ok: true, issues: [] }, 'Notepad’s byte-order mark is fine');
});

test('every key the app edits is a known key', () => {
  for (const field of NATIVE_FIELDS) assert.ok(KNOWN_KEYS.has(field.id), field.id);
});

test('broken JSON is reported with its line, in plain words', () => {
  const text = '{\n  "args": ["--ple-io" "mmap"],\n  "port": 8080\n}';
  const result = checkRunConfig(text);
  assert.equal(result.ok, false);
  assert.equal(result.issues[0].code, 'json');
  assert.equal(result.issues[0].level, 'error');
  assert.equal(result.issues[0].line, 2);
  assert.match(describeIssues(result.issues)[0], /^Line 2: The settings file is not valid JSON; look near line 2\. Check for a missing comma/);
  assert.equal(checkRunConfig('[1,2]').issues[0].code, 'not-object');
  assert.equal(checkRunConfig('').ok, false);
});

test('an unknown key is a warning with the nearest real name, and does not hold the start back', () => {
  const result = checkRunConfig('{\n  "paralel": 2,\n  "idle_unload_secs": 5,\n  "totally_made_up_thing": 1\n}');
  assert.equal(result.ok, true);
  assert.deepEqual(result.issues.map(i => [i.level, i.code, i.key, i.params.near, i.line]), [
    ['warning', 'unknown-key-near', 'paralel', 'parallel', 2],
    ['warning', 'unknown-key-near', 'idle_unload_secs', 'idle_unload_s', 3],
    ['warning', 'unknown-key', 'totally_made_up_thing', '', 4]]);
  assert.match(result.issues[0].message, /"paralel" is not a setting Strata reads, so it is ignored\. Did you mean "parallel"\?/);
  assert.equal(nearestKey('Lazy-Load'), 'lazy_load');
  assert.equal(nearestKey('zzzzzzzz'), null);
});

test('a bad value is an error that names the key, the line, what is allowed and what was found', () => {
  const text = '{\n  "parallel": 12,\n  "lazy_load": "yes",\n  "anthropic_thinking": "sometimes",\n  "idle_unload_s": 1.5,\n  "port": 70000,\n  "args": ["--ple-io", 4],\n  "env": {"PATH": "/x", "STRATA_Y": 2},\n  "sampling": {"temperature": -1, "top_k": 0.5, "banana": 3},\n  "gpu": "first",\n  "layer_split": "32,16",\n  "reasoning_effort": "max"\n}';
  const result = checkRunConfig(text);
  assert.equal(result.ok, false);
  const by = key => result.issues.filter(i => i.key === key);
  assert.deepEqual(by('parallel').map(i => [i.code, i.line, i.params.min, i.params.max, i.params.value]), [['whole-range', 2, 1, 8, '12']]);
  assert.equal(by('lazy_load')[0].code, 'bool');
  assert.deepEqual([by('anthropic_thinking')[0].code, by('anthropic_thinking')[0].params.choices], ['choice', 'model, on_request']);
  assert.equal(by('idle_unload_s')[0].code, 'whole-range');
  assert.equal(by('port')[0].code, 'number-range');
  assert.equal(by('args')[0].code, 'args-type');
  assert.deepEqual(by('env').map(i => i.code), ['env-value'], 'names other than STRATA_ (PATH) are kept, only a non-text value is wrong');
  assert.deepEqual(by('sampling').map(i => [i.code, i.params.key]), [['number-range', 'sampling.temperature'], ['whole-range', 'sampling.top_k'], ['unknown-key', 'sampling.banana']]);
  assert.equal(by('gpu')[0].code, 'gpu');
  assert.equal(by('layer_split')[0].code, 'layer-split');
  assert.equal(by('reasoning_effort')[0].code, 'choice');
  assert.ok(result.issues.filter(i => i.level === 'error').every(i => i.message.length > 20));
});

test('null and empty values mean "use the default" and are never an error', () => {
  assert.deepEqual(checkRunConfig('{"parallel": null, "lazy_load": null, "reasoning_budget_tokens": ""}'), { ok: true, issues: [] });
});

test('--host, --port and --api-key in args are a warning, not a refusal', () => {
  const r = checkRunConfig('{"args": ["--port", "9"]}');
  assert.equal(r.ok, true);
  assert.equal(r.issues[0].code, 'args-managed');
});

test('the text editor: Notepad on Windows, text mode on a Mac, the system text-file program on Linux', async () => {
  assert.deepEqual(await textEditorCommand('C:\\s\\strata-q.json', { platform: 'win32' }), ['notepad.exe', ['C:\\s\\strata-q.json']]);
  assert.deepEqual(await textEditorCommand('/s/strata-q.json', { platform: 'darwin' }), ['open', ['-t', '/s/strata-q.json']]);
  const ask = async (command, args) => command === 'xdg-mime' ? 'org.gnome.gedit.desktop\n' : args[1] === 'command -v gtk-launch' ? '/usr/bin/gtk-launch\n' : '';
  assert.deepEqual(await textEditorCommand('/s/strata-q.json', { platform: 'linux', ask }), ['/usr/bin/gtk-launch', ['org.gnome.gedit.desktop', '/s/strata-q.json']]);
  const noEditor = async () => '';
  const seen = [];
  await openTextFile('/s/strata-q.json', { platform: 'linux', ask: noEditor, recorder: seen });
  assert.deepEqual(seen, [['xdg-open', '/s/strata-q.json']], 'with no text editor known the file’s own program opens it');
});

test('every issue sentence exists in all eight languages and keeps its slots', async () => {
  const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko'];
  const compiled = await build({ stdin: { contents: "export * from './src/lib/run-config-issues'; export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, platform: 'node', format: 'esm' });
  const { issueText, setTranslationLanguage } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
  const samples = [
    { code: 'json', line: 3, params: { line: 3 } }, { code: 'json', line: null, params: { line: 0 } }, { code: 'not-object', params: {} },
    { code: 'unknown-key-near', params: { key: 'paralel', near: 'parallel' } }, { code: 'unknown-key', params: { key: 'zzz' } },
    { code: 'args-type', params: {} }, { code: 'args-managed', params: {} }, { code: 'env-type', params: {} },
    { code: 'env-value', params: { name: 'STRATA_X' } }, { code: 'sampling-type', params: {} }, { code: 'bool', params: { key: 'lazy_load', value: '"yes"' } },
    { code: 'number-range', params: { key: 'port', min: 1, max: 65535, value: '0' } }, { code: 'whole-range', params: { key: 'parallel', min: 1, max: 8, value: '12' } },
    { code: 'choice', params: { key: 'gpu_order', choices: 'auto, as_given', value: '"x"' } }, { code: 'text', params: { key: 'aliases' } },
    { code: 'layer-split', params: {} }, { code: 'gpu', params: {} },
  ].map(s => ({ level: 'error', key: 'k', message: 'ENGLISH FALLBACK', line: s.line === undefined ? 5 : s.line, ...s }));
  const english = {};
  setTranslationLanguage('en');
  for (const s of samples) english[`${s.code}:${s.line}`] = issueText(s);
  for (const s of samples) assert.notEqual(english[`${s.code}:${s.line}`], 'ENGLISH FALLBACK', s.code);
  for (const language of languages.filter(l => l !== 'en')) {
    setTranslationLanguage(language);
    for (const s of samples) {
      const text = issueText(s);
      assert.notEqual(text, english[`${s.code}:${s.line}`], `${language}:${s.code} is not translated`);
      assert.ok(!/\{\d+\}/.test(text), `${language}:${s.code} left a slot`);
    }
  }
  setTranslationLanguage('en');
});

test('the notes the app itself writes into the file (ui_resource_baseline, ui_vision_device, ...) are not reported', () => {
  assert.deepEqual(checkRunConfig('{"ui_resource_baseline": {"parallel": 2}, "ui_vision_device": "cpu", "ui_resource_profile": "work"}'), { ok: true, issues: [] });
});
