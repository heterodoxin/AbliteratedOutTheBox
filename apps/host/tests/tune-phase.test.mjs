/** Item 7: the tuner's stage names map onto the five steps shown, and the speeds it reports become the live readout. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/tune-phase.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { tunePhase, tuneSpeeds } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('every stage the tuner reports lands on one of the five steps, in order', () => {
  assert.deepEqual(['start', 'waiting', undefined, null].map(tunePhase), [0, 0, 0, 0], 'checking the PC, also while it waits for a quiet PC');
  assert.equal(tunePhase('base'), 1);
  assert.deepEqual(['plan', 'rounds', 'drift', 'per_request'].map(tunePhase), [2, 2, 2, 2]);
  assert.equal(tunePhase('confirm'), 3);
  assert.equal(tunePhase('done'), 4);
  assert.equal(tunePhase('something new'), 0, 'an unknown stage never breaks the page');
});

test('the speeds are what the tuner measured, base first, and nothing invented', () => {
  assert.deepEqual(tuneSpeeds(undefined), []);
  assert.deepEqual(tuneSpeeds({ stage: 'rounds' }), []);
  const speeds = tuneSpeeds({ stage: 'rounds', speeds: [{ label: 'base', tok_s: 41.2 }, { label: 'More experts in memory', tok_s: 46.8 }, { label: 'bad', tok_s: 0 }, { label: 'worse', tok_s: 'x' }] });
  assert.deepEqual(speeds, [{ label: 'base', tok_s: 41.2 }, { label: 'More experts in memory', tok_s: 46.8 }]);
});
