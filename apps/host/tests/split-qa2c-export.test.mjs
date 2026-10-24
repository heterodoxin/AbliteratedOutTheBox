// The saved Monitor picture when a model is loaded (the normal case) starts with the Strata logo alone and the models/time line.
// (QA follow-up: it was checked with a loaded model through the real page; the picture is the one of the Monitor screen, which shows the
// loaded models. NativeMonitor.tsx is not used by any screen, so it needs no header.)
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { readdir } from 'node:fs/promises';

const read = path => readFile(new URL(path, import.meta.url), 'utf8').then(s => s.replace(/\r\n/g, '\n'));

test('with a model loaded the picture is headed by the logo and the line "<model> · <time>"; with none, "N loaded models"', async () => {
  const source = await read('../src/screens/Monitor.tsx');
  assert.match(source, /const scopeLabel = selectedInstance \? instanceLabel\(selectedInstance\) : `\$\{runtime\?\.instances\.length \?\? 0\} loaded models`;/);
  const header = /\{tr\(frozen \? <header className="section-toolbar">(.*?)<\/header> : null\)\}/s.exec(source)?.[1];
  assert.ok(header);
  assert.match(header, /<Brand variant="lockup"/); assert.doesNotMatch(header, /<h1/);
  assert.match(header, /frozen\.modelLabel.*frozen\.time/s);
  assert.match(source, /setFrozen\(\{ data: structuredClone\(scoped \?\? live\), hardware: structuredClone\(inventory\), time: new Date\(\)\.toLocaleString\(localeTag\(\)\), modelLabel: scopeLabel \}\)/);
});

test('no screen renders NativeMonitor, so the Monitor screen is the only picture source', async () => {
  const dirs = ['../src/screens/', '../src/app/', '../src/components/'];
  for (const dir of dirs) {
    for (const name of await readdir(new URL(dir, import.meta.url))) {
      if (!/\.tsx?$/.test(name) || name === 'NativeMonitor.tsx') continue;
      assert.doesNotMatch(await read(dir + name), /<NativeMonitor\b|from '\.\/NativeMonitor'|from '\.\.\/components\/NativeMonitor'/, `${dir}${name}`);
    }
  }
});
