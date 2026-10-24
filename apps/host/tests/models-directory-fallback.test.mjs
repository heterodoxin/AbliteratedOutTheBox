import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const cwd = fileURLToPath(new URL('..', import.meta.url));
function fixture(t) { const home = realpathSync(mkdtempSync(join(tmpdir(), 'strata-fallback-'))); t.after(() => rmSync(home, { recursive: true, force: true })); return home; }
function load(env) {
  const code = "const s = await import('./server/store.mjs'); console.log(JSON.stringify({ dir: s.MODELS_DIR, notice: s.MODELS_DIR_NOTICE }));";
  const run = spawnSync(process.execPath, ['--input-type=module', '-e', code], { cwd, encoding: 'utf8', env: { ...process.env, ...env } });
  assert.equal(run.status, 0, run.stderr);
  return JSON.parse(run.stdout.trim().split('\n').pop());
}

test('a Documents folder that cannot be made does not stop the host: models go to STRATA_HOME/models with a notice', t => {
  const home = fixture(t);
  const blocker = join(home, 'not-a-folder'); writeFileSync(blocker, 'x');            // mkdir below a file fails like a blocked Documents folder
  const { dir, notice } = load({ STRATA_HOME: join(home, 'h'), STRATA_MODELS_DIR: '', STRATA_DOCUMENTS_DIR: join(blocker, 'Documents') });
  assert.equal(dir, realpathSync(join(home, 'h', 'models')));
  assert.match(notice, /could not use .* for models/);
  assert.match(notice, /Settings/);
});

test('a usable Documents folder is used and says nothing', t => {
  const home = fixture(t);
  const docs = join(home, 'Documents'); mkdirSync(docs);
  const { dir, notice } = load({ STRATA_HOME: join(home, 'h'), STRATA_MODELS_DIR: '', STRATA_DOCUMENTS_DIR: docs });
  assert.equal(dir, realpathSync(join(docs, 'Strata', 'Models')));
  assert.equal(notice, '');
});
