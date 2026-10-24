// The saved Monitor screenshot starts with the Strata logo alone (no "Monitor" headline); the live Monitor page keeps its title.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const read = path => readFile(new URL(path, import.meta.url), 'utf8').then(s => s.replace(/\r\n/g, '\n'));

test('the exported Monitor picture has the logo and no "Monitor" headline; the page itself keeps its title', async () => {
  const source = await read('../src/screens/Monitor.tsx');
  const header = /\{tr\(frozen \? <header className="section-toolbar">(.*?)<\/header> : null\)\}/s.exec(source)?.[1];
  assert.ok(header, 'the picture has its own header (it only exists while the picture is taken)');
  assert.match(header, /<Brand variant="lockup"/);
  assert.doesNotMatch(header, /<h1/);
  assert.doesNotMatch(header, /tr\("Monitor"\)/);
  assert.match(header, /frozen\.modelLabel/, 'the model and time line stays');
  assert.match(source, /<Screen width="wide" plainHeader title=\{tr\("Monitor"\)\}/, 'the live page still says Monitor');
});
