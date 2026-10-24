import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

const configOf = f => join(f.repo, 'strata-iq2_xs.json');

test('the settings file is found, checked, and opened in the text editor', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh();
    const info = f.control.settingsFile('iq2_xs');
    assert.equal(info.path, configOf(f));
    assert.equal(info.name, 'strata-iq2_xs.json');
    assert.equal(info.ok, true);
    assert.match(info.note, /apply the next time the model starts/);
    assert.ok(info.issues.every(i => i.level === 'warning'), 'the fixture’s own extra key is a warning at most');
    const seen = [];
    const opened = await f.control.openSettingsFile('iq2_xs', { recorder: seen, platform: 'win32' });
    assert.deepEqual(seen, [['notepad.exe', info.path]]);
    assert.equal(opened.name, 'strata-iq2_xs.json');
    assert.throws(() => f.control.settingsFile('../etc/passwd'), /installed model/);
  } finally { await f.cleanup(); }
});

test('a mistake in the file holds the start back with its line; a typo’d key only warns', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const path = configOf(f), good = readFileSync(path, 'utf8');
    // 1. a syntax slip (missing comma): not valid JSON, said with the line, and nothing is started
    writeFileSync(path, JSON.stringify(JSON.parse(good), null, 2).replace('"gpu": 0,', '"gpu": 0\n  "x": 1,'));
    assert.equal(f.control.settingsFile('iq2_xs').ok, false);
    let done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(done.state, 'error');
    assert.match(done.error, /^The settings file for this model has a mistake, so the model was not started\. Line \d+: The settings file is not valid JSON/);
    assert.equal(f.control.state().instances.length, 0, 'no engine was started');
    assert.equal(f.calls.filter(c => /strata\.py|serve/.test(JSON.stringify(c.args))).length, 0);
    // 2. a bad value
    const config = JSON.parse(good); config.parallel = 99; config.lazy_load = 'yes';
    writeFileSync(path, JSON.stringify(config, null, 2));
    done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(done.state, 'error');
    assert.match(done.error, /"parallel" must be a whole number from 1 to 8, not 99/);
    assert.match(done.error, /"lazy_load" must be true or false, not "yes"/);
    // 3. an unknown key only warns and the model loads
    const typo = JSON.parse(good); typo.paralel = 2;
    writeFileSync(path, JSON.stringify(typo, null, 2));
    const info = f.control.settingsFile('iq2_xs');
    assert.equal(info.ok, true);
    assert.ok(info.issues.some(i => i.code === 'unknown-key-near' && i.params.near === 'parallel'));
    done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(done.state, 'done', done.error);
  } finally { await f.cleanup(); }
});
