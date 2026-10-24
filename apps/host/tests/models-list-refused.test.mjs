import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

/**
 * The starter list (real-model-catalog.mjs) names five files by repository and file name with no hash. With the real engine on, nothing
 * may download from it. REAL is read when the module loads, and node --test runs each file in its own process, so this file sets it first.
 */
const home = mkdtempSync(join(tmpdir(), 'strata-list-'));
process.env.STRATA_HOME = home;
process.env.STRATA_MODELS_DIR = join(home, 'models');
process.env.STRATA_REAL_ENGINE = '1';
const calls = [];
const realFetch = globalThis.fetch;
globalThis.fetch = (...args) => { calls.push(String(args[0])); return Promise.reject(new Error('no network in this test')); };
const models = await import('../server/models.mjs');
const { REAL_VARIANTS } = await import('../server/real-model-catalog.mjs');
test.after(() => { globalThis.fetch = realFetch; rmSync(home, { recursive: true, force: true }); });

test('with the real engine on, installing from the starter list is refused before any request, for every entry', async () => {
  assert.equal(models.REAL, true);
  assert.ok(REAL_VARIANTS.length >= 1);
  for (const v of REAL_VARIANTS) {
    await assert.rejects(models.install(v.id), err => err.status === 409 && err.message === 'The app downloads models from the Models screen of the native engine, not from this list.', v.id);
  }
  assert.deepEqual(calls, [], 'no request was made');
  // nothing was left half-started
  for (const v of REAL_VARIANTS) assert.notEqual(models.library().variants.find(row => row.id === v.id)?.state, 'downloading');
});
