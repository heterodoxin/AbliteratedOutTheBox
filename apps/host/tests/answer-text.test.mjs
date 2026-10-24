// QA report F01: reasoning wrappers and stray tag fragments never show in an answer or its Markdown export.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';
async function loadTs(path) {
  const { code } = await transform(await readFile(new URL(path, import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}
const { tidyAnswer } = await loadTs('../src/lib/answer-text.ts');
const { appendText } = await loadTs('../src/lib/deliverables.ts').catch(() => ({}));

test('a literal <analysis> or <think> wrapper is removed, the words stay', () => {
  assert.equal(tidyAnswer('<analysis>The answer is 4.'), 'The answer is 4.');
  assert.equal(tidyAnswer('Sure.\n<think>\nplan\n</think>\nIt is 4.'), 'Sure.\n\nplan\n\nIt is 4.');
  assert.equal(tidyAnswer('x </ANALYSIS> y <Thinking style="a"> z'), 'x  y  z');
});

test('a tag that lost its "<" at the very start is not shown as a prefix', () => {
  assert.equal(tidyAnswer('p> CEDAR-742'), 'CEDAR-742');
  assert.equal(tidyAnswer('  /p> done'), 'done');
  assert.equal(tidyAnswer('analysis> 4'), '4');
});

test('ordinary text, code and comparisons are untouched', () => {
  for (const text of ['CEDAR-742', 'if a > b then c', 'x -> y', '```html\n<p>hi</p>\n```', 'The p> in the middle', '<div>kept</div>', 'a <b and c> d', 'Result: 5 > 3', '']) assert.equal(tidyAnswer(text), text);
});

test('streaming: a tag that arrives in pieces is removed once it is complete', () => {
  let content = '';
  for (const piece of ['<', 'ana', 'lysis', '>', 'The code is ', 'p', '> no']) content = tidyAnswer(content + piece);
  assert.equal(content, 'The code is p> no');
  let lead = '';
  for (const piece of ['p', '>', ' CEDAR', '-742']) lead = tidyAnswer(lead + piece);
  assert.equal(lead, 'CEDAR-742');
});

test('the store applies it to the answer, the ordered pieces and the Markdown export', async () => {
  const store = await readFile(new URL('../src/lib/store.ts', import.meta.url), 'utf8');
  assert.match(store, /tidyAnswer\(m\.content\+String\(frame\.text/);
  assert.match(store, /m\.role === 'assistant' \? tidyAnswer\(m\.content\) : m\.content\)/);
  assert.match(await readFile(new URL('../src/lib/deliverables.ts', import.meta.url), 'utf8'), /tidyAnswer\(last\.text \+ text\)/);
});
