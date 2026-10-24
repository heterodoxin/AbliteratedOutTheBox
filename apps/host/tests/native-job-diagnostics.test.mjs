/** Ported from #3578 (medking82): a later finished update of the same component settles a restart interruption; the alert stays red. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const out = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/native-job-diagnostics.ts', import.meta.url))], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent' });
const { recoveredInterruptions, jobErrorText, INTERRUPTED } = await import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`);
const failed = { id: 'a', kind: 'update-strata', state: 'error', summary: 'Interrupted operation', error: INTERRUPTED, startedAt: '2026-10-20T11:41:02.916Z' };
const done = { id: 'b', kind: 'update-strata', state: 'done', summary: 'Complete', error: null, startedAt: '2026-10-20T23:11:18.883Z' };

test('a later finished update of the same component settles the interruption, in any array order, and the records are untouched', () => {
  const jobs = structuredClone([failed, done]), before = JSON.stringify(jobs);
  assert.deepEqual([...recoveredInterruptions(jobs)], ['a']);
  assert.deepEqual([...recoveredInterruptions([done, failed])], ['a']);
  assert.equal(JSON.stringify(jobs), before);
  assert.match(jobErrorText(jobs[0], jobs), /A later update finished, so nothing needs doing/);
});

for (const [name, other] of [['no retry', null], ['running retry', { ...done, state: 'running' }], ['cancelled retry', { ...done, state: 'cancelled' }],
  ['success before the interruption', { ...done, startedAt: '2026-10-19T00:00:00.000Z' }], ['same time', { ...done, startedAt: failed.startedAt }],
  ['invalid time', { ...done, startedAt: 'unknown' }], ['other component', { ...done, kind: 'update-engine' }]]) {
  test(`${name}: the interruption stays an ordinary failure`, () => {
    const jobs = other ? [failed, other] : [failed];
    assert.equal(recoveredInterruptions(jobs).size, 0);
    assert.equal(jobErrorText(failed, jobs), INTERRUPTED);
  });
}

test('an ordinary failure is never settled, and an invalid interruption time stays a failure', () => {
  const other = { ...failed, id: 'c', error: 'Installer Failed' };
  assert.equal(jobErrorText(other, [other, done]), 'Installer Failed');
  assert.equal(recoveredInterruptions([{ ...failed, startedAt: 'unknown' }, done]).size, 0);
});
