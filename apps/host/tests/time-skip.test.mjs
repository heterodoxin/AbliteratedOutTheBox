import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/time-skip.ts', import.meta.url), 'utf8'), { loader:'ts', format:'esm' });
const { timeSkipPrompt, timeSkipFrame, timeSkipTabId, timeSkipDocuments, timeSkipDocumentMarkdown, timeSkipResultState, withTimeSkipResult, TIME_SKIP_DURATION } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('only an exact leading time-skip command starts a demo', () => {
  assert.equal(timeSkipPrompt('/time-skip Make something nice'), 'Make something nice');
  assert.equal(timeSkipPrompt(' /TIME-SKIP  A plan\nwith two parts '), 'A plan\nwith two parts');
  assert.equal(timeSkipPrompt('/time-skip'), '');
  assert.equal(timeSkipPrompt('/time-skip   '), '');
  for (const message of ['/time-skipping hello', '/time-skipper', 'Explain /time-skip', '/time-skip: hello']) assert.equal(timeSkipPrompt(message), null);
});

test('the candidate is checked before promotion and a regressed revision cannot replace it', () => {
  const run = { id:'test', prompt:'Make a plan', startedAt:100000 };
  let compared = false, promoted = false, rejected = false;
  for (let elapsed = 0; elapsed <= TIME_SKIP_DURATION; elapsed += 250) {
    const frame = timeSkipFrame(run, run.startedAt + elapsed);
    if (frame.status === 'Comparing' && frame.pass === 1) compared = true;
    if (frame.chosen === 'b') { assert.ok(compared); promoted = true; }
    if (promoted) assert.equal(frame.chosen, 'b');
    if (frame.status === 'Keeping B') { assert.equal(frame.pass, 2); rejected = true; }
    if (frame.done) assert.ok(rejected);
  }
  assert.ok(promoted && rejected);
});

test('a restored or hidden run follows its original clock, and completion is stable', () => {
  const run = { id:'one', prompt:'Make a plan', startedAt:100000 };
  assert.equal(timeSkipFrame(run, 99999).index, 0);
  const saved = JSON.parse(JSON.stringify(run));
  assert.equal(timeSkipFrame(saved, 118000).pass, 2);
  assert.equal(timeSkipFrame(saved, 118000).chosen, 'b');
  assert.equal(timeSkipFrame(saved, 128000).done, true);
  assert.equal(timeSkipFrame({ ...saved, completedAt:128000 }, 100000).done, true);
  assert.notEqual(timeSkipTabId('one'), timeSkipTabId('two'));
});

test('original and kept outputs stay stable after the rejected later revision', () => {
  for (const prompt of ['Make a launch plan', 'Write a counter app']) {
    const run = { id:'stable', prompt, startedAt:100000 };
    const documents = timeSkipDocuments(prompt);
    const baseline = JSON.stringify(documents.original);
    const update = timeSkipDocumentMarkdown(documents.updated);
    for (const now of [109500, 116000, 124000, 128000]) {
      timeSkipResultState(run, now);
      assert.equal(JSON.stringify(timeSkipDocuments(prompt).original), baseline);
      assert.equal(timeSkipDocumentMarkdown(timeSkipDocuments(prompt).updated), update);
    }
    assert.notEqual(timeSkipDocumentMarkdown(documents.original), update);
    assert.notEqual(timeSkipDocumentMarkdown(documents.revision), update);
  }
});

test('restore and undo preserve both artifacts and survive session serialization', () => {
  const run = { id:'restore', prompt:'Make a launch plan', startedAt:100000 };
  assert.equal(withTimeSkipResult(run, 'original', 118000), run, 'cannot change the retained result during the demo');
  const restored = withTimeSkipResult(run, 'original', 128000);
  const saved = JSON.parse(JSON.stringify(restored));
  assert.equal(timeSkipResultState(saved, 130000).kept, 'original');
  assert.equal(timeSkipResultState(saved, 130000).status, 'Original restored');
  assert.deepEqual(timeSkipDocuments(saved.prompt), timeSkipDocuments(run.prompt));
  const undone = withTimeSkipResult(saved, 'updated', 130000);
  assert.equal(timeSkipResultState(undone, 130000).status, 'Updated result kept');
  assert.equal(run.retainedVersion, undefined, 'the baseline run was not mutated');
});

test('unfinished updates are distinguished from checked and retained results', () => {
  const run = { id:'progress', prompt:'Make a launch plan', startedAt:100000 };
  assert.equal(timeSkipResultState(run, 106000).ready, false);
  assert.equal(timeSkipResultState(run, 109500).status, 'Checking changes…');
  assert.equal(timeSkipResultState(run, 109500).kept, 'original');
  assert.equal(timeSkipResultState(run, 112500).kept, 'updated');
  assert.equal(timeSkipResultState(run, 120500).kept, 'updated');
  assert.equal(timeSkipResultState(run, 128000).status, 'Updated result kept');
});
