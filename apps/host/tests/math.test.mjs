// #3268: LaTeX in answers is found (and drawn by KaTeX), money and code are not mistaken for it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';
// KaTeX is a dependency of the app (npm install / npm ci); a checkout that shares another folder's node_modules may not have it yet
const katex = await import('katex').then(m => m.default, () => null);

async function loadTs(path) {
  const { code } = await transform(await readFile(new URL(path, import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}
const { splitMath, displayBlock, hasMath } = await loadTs('../src/lib/math.ts');
const maths = text => splitMath(text).filter(p => p.kind === 'math').map(p => [p.tex, p.display]);

test('inline and display forms', () => {
  assert.deepEqual(maths('a sphere of radius $r \\le R$ here'), [['r \\le R', false]]);
  assert.deepEqual(maths('so \\(x^2\\) and \\[y^2\\]'), [['x^2', false], ['y^2', true]]);
  assert.deepEqual(maths('then $$g = -k r$$ done'), [['g = -k r', true]]);
  assert.deepEqual(splitMath('plain words').map(p => p.kind), ['text']);
});

test('money, escaped dollars and unclosed math stay text', () => {
  assert.deepEqual(maths('it costs $5 and $10 in total'), []);
  assert.deepEqual(maths('only $5 here'), []);
  assert.deepEqual(maths('a \\$5 and a \\$7 fee'), []);
  assert.deepEqual(maths('an open $$ x + y'), []);
  assert.equal(splitMath('a \\$5 fee').map(p => p.text).join(''), 'a $5 fee');
  assert.equal(hasMath('no math here'), false);
});

test('inline math does not cross a line, display math does', () => {
  assert.deepEqual(maths('start $a\nb$ end'), []);
  assert.deepEqual(maths('x $$a\nb$$ y'), [['a\nb', true]]);
});

test('display blocks: one line, several lines, bracket form, and unclosed ones (still streaming) stay text', () => {
  assert.deepEqual(displayBlock(['$$ a = b $$'], 0), { tex: ' a = b ', used: 1 });
  assert.deepEqual(displayBlock(['$$', '\\oint g = 1', '$$', 'after'], 0), { tex: '\n\\oint g = 1\n', used: 3 });
  assert.deepEqual(displayBlock(['\\[', 'x', '\\]'], 0), { tex: '\nx\n', used: 3 });
  assert.equal(displayBlock(['$$', 'x'], 0), null);
  assert.equal(displayBlock(['$$a$$ and more'], 0), null);
  assert.equal(displayBlock(['plain'], 0), null);
});

test('the formulas of the report are found in their paragraph and draw without an error', { skip: katex ? false : 'katex is not installed in node_modules' }, () => {
  const answer = [
    'Apply Gauss\'s Law for gravity to a sphere of radius $r \\le R$ concentric with the planet:',
    '',
    '$$\\oint \\vec{g}\\cdot d\\vec{A} = -4\\pi G\\, M_{\\text{enc}}(r)$$',
    '',
    'By spherical symmetry, $\\vec{g}$ is radial and constant on the Gaussian sphere, so:',
    '',
    '$$g\\,(4\\pi r^2) = -4\\pi G\\left(\\tfrac{4}{3}\\pi r^3\\rho\\right)$$',
    '',
    '$$\\boxed{g(r) = -\\frac{4\\pi G\\rho}{3}\\,r}$$',
  ].join('\n').split('\n');
  const found = [];
  for (let i = 0; i < answer.length;) {
    const block = displayBlock(answer, i);
    if (block) { found.push([block.tex, true]); i += block.used; continue; }
    for (const [tex, display] of maths(answer[i])) found.push([tex, display]);
    i++;
  }
  assert.equal(found.length, 5);
  for (const [tex, display] of found) {
    const html = katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore', trust: false });
    assert.ok(html.includes('class="katex'), tex);
    assert.ok(!html.includes('katex-error'), tex);
  }
  // trust is off: no link can come out of a model's formula
  const bad = katex.renderToString('\\href{https://example.com}{x}', { throwOnError: false, strict: 'ignore', trust: false });
  assert.ok(!bad.includes('<a '));
});
