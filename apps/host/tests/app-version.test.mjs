import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { appVersionFrom } from '../server/app-version.mjs';

const here = fileURLToPath(new URL('../server', import.meta.url));

test('the shell passes the version; a bad value is ignored', () => {
  assert.equal(appVersionFrom({ env: { STRATA_APP_VERSION: '0.2.0-beta' }, here: '/nowhere/a/b' }), '0.2.0-beta');
  assert.equal(appVersionFrom({ env: { STRATA_APP_VERSION: ' 1.2.3 ' }, here: '/nowhere/a/b' }), '1.2.3');
  assert.equal(appVersionFrom({ env: { STRATA_APP_VERSION: 'latest' }, here: '/nowhere/a/b' }), 'dev');
  assert.equal(appVersionFrom({ env: {}, here: '/nowhere/a/b' }), 'dev');
});

test('in a checkout the same file the shell is built from is read, so there is one source', async () => {
  const toml = await readFile(join(here, '..', '..', 'desktop', 'src-tauri', 'Cargo.toml'), 'utf8');
  const declared = /^version\s*=\s*"([^"]+)"/m.exec(toml)[1];
  assert.equal(appVersionFrom({ env: {}, here }), declared);
  assert.notEqual(declared, '0.1.0', 'the app version is the product version, not the 0.1.0 placeholder');
  const dir = await mkdtemp(join(tmpdir(), 'strata-appver-'));
  try {
    await mkdir(join(dir, 'host', 'server'), { recursive: true });
    await mkdir(join(dir, 'desktop', 'src-tauri'), { recursive: true });
    await writeFile(join(dir, 'desktop', 'src-tauri', 'Cargo.toml'), '[package]\nname = "x"\nversion = "9.8.7-rc.1"\n');
    assert.equal(appVersionFrom({ env: {}, here: join(dir, 'host', 'server') }), '9.8.7-rc.1');
  } finally { await rm(dir, { recursive: true, force: true }); }
});
