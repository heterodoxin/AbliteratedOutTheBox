import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko'];
const catalogs = Object.fromEntries(await Promise.all(languages.map(async l => [l, JSON.parse(await readFile(`src/locales/${l}.json`, 'utf8'))])));
const built = await build({ stdin: { contents: "export * from './src/lib/plain-errors'; export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { plainFailure, plainFailureText, setTranslationLanguage } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

const cases = [
  ['Strata’s tunnel for this address is already running (PID 31), so Strata did not start a second one. Stop it first (or use it as it is), then connect again here.', 'tunnel-own-running', /tunnel for this address is already running/],
  ['GPU 2 is already used by qwen3.8-flash-next-q2_0, so this model did not fit there. Load it on another GPU, or unload qwen3.8-flash-next-q2_0 first.', 'gpu-shared', /no room for this one/],
  ['This model needs a newer engine: Gyro-S needs engine 0.1.45.1 or newer, and this one is 0.1.45. Update the engine in Settings, then try again.', 'engine-too-old', /needs a newer engine/],
  ['The settings file for this model has a mistake, so the model was not started. Line 4: The settings file is not valid JSON; look near line 4.', 'settings-file', /settings file has a mistake/],
  ['The Cloudflare connector (cloudflared) is not installed here, and the app never downloads it while it runs. Install it from the optional connector component.', 'connector-missing', /could not find Cloudflare’s connector/],
  ['The cloudflared at /usr/bin/cloudflared is version 2021.3.0, which is too old. Strata needs 2022.1.0 or newer. Update cloudflared, then retry.', 'connector-old', /connector on this computer is too old/],
  ['The file at /x/cloudflared did not run as cloudflared. Reinstall cloudflared, then retry.', 'connector-broken', /did not run/],
  ['lost host child is still present; retry after it exits', 'previous-session', /last Strata session is still closing down/],
  ['server_exited', 'start-failed', /model stopped while starting\./],
  ['server_exited (exit code 0xC0000135)', 'dll-missing', /program files is missing/],
  ['server_exited (exit code 0xC000007B)', 'dll-bad', /does not fit this computer or is damaged/],
  ['server_exited (exit code 0xC0000005)', 'crash', /crashed while starting/],
  ['server_exited (exit code 0xC000001D)', 'cpu-unsupported', /newer processor feature/],
  ['spawn C:/x/strata.exe ENOENT', 'file-missing', /could not find a file/],
  ['server_exited (exit code 0xC0000022)', 'blocked', /security software stopped/],
  ['server_exited: memory guard: 5.1 GiB of RAM free is under the 6 GiB limit', 'memory', /ran low on memory/],
  ['server_exited (exit code 137): Killed', 'memory', /ran low on memory/],
  ['Engine stopped (exit 3). Open diagnostics or reload.', 'start-failed', /model stopped while starting\./],
  ['Loading took more than five minutes. Check the engine log and free memory.', 'slow-load', /taking too long/],
  ['No engine for this PC was found in C:/x/components: the component folder holds no engine package. The app never downloads an engine while it starts or serves.', 'engine-missing', /engine for this computer isn’t installed yet/],
  ['cannot reach huggingface.co (<urlopen error [Errno 11001] getaddrinfo failed>)', 'network', /couldn’t reach the internet/],
  ['sha256 mismatch for Qwen3.8-Flash-Next-00001-of-00002.gguf', 'integrity', /safety check/],
  ['The agent component (DeepSeek Harness) is not installed.', 'assistant-missing', /coding assistant isn’t installed/],
  ['component-missing', 'assistant-missing', /coding assistant/],
  ['Strata Control did not answer within 60 s (models list).', 'control', /couldn’t read its own components/],
  ['Prepare this model’s vision files before enabling picture reading.', 'picture-reading', /Picture reading isn’t available/],
  ['server_exited (exit code 1): strata generate: GPU 0 (AMD Radeon 760M Graphics, gfx1103) is not an architecture this Strata engine was compiled for (gfx1200,gfx1201); compile it for this card', 'gpu-no-code', /no code for the graphics card/],
  ['server_exited (exit code 1): [strata] none of this computer\'s graphics cards has code in this Strata engine (it is built for gfx1200,gfx1201). Cards found: AMD Radeon RX 6800 XT (gfx1030).', 'gpu-no-code', /no code for the graphics card/],
  ['Traceback (most recent call last):\n  File "x.py", line 3', 'technical', /Something went wrong/],
];

for (const [raw, kind, message] of cases) {
  test(`plain words for: ${raw.slice(0, 60)}`, () => {
    const f = plainFailure(raw);
    assert.equal(f.kind, kind);
    assert.match(f.message, message);
    assert.ok(f.next, 'a next step');
    assert.equal(f.detail, raw.trim(), 'the original text stays for Details');
    assert.doesNotMatch(f.message, /server_exited|exit \d|component-missing|Traceback|0x[0-9A-F]{8}/);
  });
}

test('an ordinary sentence is left as it is and gets no Details', () => {
  const f = plainFailure('Some model files are missing. Use Download / repair to restore them.');
  assert.equal(f.kind, 'plain');
  assert.equal(f.detail, undefined);
});

test('every plain message and next step is translated in all eight languages', () => {
  const en = catalogs.en;
  const ids = Object.fromEntries(Object.entries(en).map(([id, text]) => [text, id]));
  for (const [raw] of cases) {
    const f = plainFailure(raw);
    for (const text of [f.message, f.next]) {
      const id = ids[text];
      assert.ok(id, `"${text}" is in the catalog`);
      for (const l of languages) if (l !== 'en') assert.notEqual(catalogs[l][id], text, `${l}: "${text}" is translated`);
    }
  }
  setTranslationLanguage('de');
  assert.notEqual(plainFailureText('server_exited').message, plainFailure('server_exited').message);
  // a missing DLL is never described as low memory
  assert.doesNotMatch(plainFailure('server_exited (exit code 0xC0000135)').message + plainFailure('server_exited (exit code 0xC0000135)').next, /memory|Leave 20/i);
  setTranslationLanguage('en');
});

test('missing model files are named, in every language, with a next step (#3478)', async () => {
  const { missingFilesError, missingNames } = await import('../server/missing-files.mjs');
  const names = missingNames([{ part: 'engine', file: 'strata.exe', status: 'missing' }, { part: 'native', file: 'Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf', status: 'missing' }, { part: 'vision/mmproj', file: 'mmproj-F16.gguf', status: 'missing' }],
    [['C:', 'models', 'Qwen3.6-35B-A3B-UD-Q4_K_XL-00002-of-00003.gguf'].join(String.fromCharCode(92)), '/data/models/mmproj-F16.gguf']);
  assert.deepEqual(names, ['Qwen3.6-35B-A3B-UD-Q4_K_XL-00002-of-00003.gguf', 'mmproj-F16.gguf', 'the engine program', 'Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf']);
  const error = missingFilesError('qwen36-q4', names);
  assert.match(error.message, /missing: Qwen3\.6-35B-A3B-UD-Q4_K_XL-00002-of-00003\.gguf, mmproj-F16\.gguf, the engine program, Qwen3\.6-35B-A3B-UD-Q4_K_XL\.gguf\. Use Download/);
  assert.match(error.details[0], /Missing for qwen36-q4: /);
  const more = missingFilesError('m', ['a.gguf', 'b.gguf', 'c.gguf', 'd.gguf', 'e.gguf', 'f.gguf']);
  assert.match(more.message, /a\.gguf, b\.gguf, c\.gguf, d\.gguf and 2 more\./);
  const f = plainFailure(error.message);
  assert.equal(f.kind, 'model-files-missing');
  assert.equal(f.message, 'These model files are missing: Qwen3.6-35B-A3B-UD-Q4_K_XL-00002-of-00003.gguf, mmproj-F16.gguf, the engine program, Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf.');
  assert.ok(f.next && f.detail === error.message);
  for (const language of languages) {
    setTranslationLanguage(language);
    const shown = plainFailureText(error.message);
    if (language !== 'en') {
      assert.notEqual(shown.message, f.message, `${language}: the sentence is translated`);
      assert.notEqual(shown.next, f.next, `${language}: the next step is translated`);
    }
    assert.ok(shown.message.includes('mmproj-F16.gguf'), `${language}: the file names survive`);
    assert.doesNotMatch(shown.message, /\{0\}/);
  }
  setTranslationLanguage('en');
  assert.equal(plainFailure(missingFilesError('m', []).message).kind, 'plain', 'with no names the old sentence is left as it was');
});
