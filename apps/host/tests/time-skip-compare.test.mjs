/** Item 4: the A versus B view adapts to the kind of file and keeps one window size. The pure parts. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

async function load(file) {
  const result = await build({ entryPoints: [fileURLToPath(new URL(`../src/lib/${file}`, import.meta.url))], bundle: true, format: 'esm', platform: 'node', write: false, alias: { react: fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url)) }, logLevel: 'silent', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}
globalThis.window ??= globalThis; globalThis.localStorage ??= { getItem: () => null, setItem: () => {} };
globalThis.navigator ??= { language: 'en' };
const cmp = await load('time-skip-compare.ts');
const act = await load('time-skip-activity.ts');

test('each kind of file gets its own view', () => {
  const k = (path, file) => cmp.fileKind(path, file);
  assert.equal(k('notes/plan.md'), 'markdown'); assert.equal(k('a.txt'), 'text'); assert.equal(k('site/index.html'), 'html'); assert.equal(k('b.HTM'), 'html');
  assert.equal(k('img/logo.PNG'), 'image'); assert.equal(k('x.svg'), 'image'); assert.equal(k('weird', { image: true }), 'image');
  assert.equal(k('src/app.py'), 'code'); assert.equal(k('data.bin', { unreadable: true }), 'other');
  assert.equal(k('photo.jpg', { unreadable: true }), 'image', 'a picture is not "unreadable": it is shown from the file');
});

test('changed words are marked, unchanged text is identical on both sides, and nothing is lost', () => {
  const a = 'Dear Hiring Manager, I am writing to express my sincere interest in the role.';
  const b = 'Hello team, I am writing because I love the role.';
  const d = cmp.wordDiff(a, b);
  assert.equal(d.original.map(s => s.text).join(''), a); assert.equal(d.updated.map(s => s.text).join(''), b);
  assert.ok(d.original.some(s => s.changed && /Dear/.test(s.text)) && d.updated.some(s => s.changed && /Hello/.test(s.text)));
  assert.ok(d.original.some(s => !s.changed && /I am writing/.test(s.text)), 'what stayed is not marked');
  assert.deepEqual(cmp.wordDiff('same text', 'same text').updated, [{ text: 'same text', changed: false }]);
  const lines = cmp.segmentLines(cmp.wordDiff('# Title\n- one\n', '# Title\n- two\n').updated);
  assert.equal(lines[0].map(s => s.text).join(''), '# Title'); assert.equal(lines[1].map(s => s.text).join(''), '- two');
  assert.ok(lines[1].some(s => s.changed));
});

test('a huge pair of texts does not freeze the page: the work is capped', () => {
  const big = Array.from({ length: 4000 }, (_, i) => `word${i}`).join(' ');
  const t = Date.now();
  const d = cmp.wordDiff(big, big + ' extra');
  assert.ok(Date.now() - t < 3000);
  assert.ok(d.updated.length >= 1);
});

test('code and other files get a short line diff with the unchanged stretches folded', () => {
  const a = Array.from({ length: 40 }, (_, i) => `line ${i}`).join('\n');
  const b = a.replace('line 20', 'line twenty') + '\nline 40';
  const rows = cmp.lineDiff(a, b);
  const stats = cmp.diffStats(rows);
  assert.deepEqual(stats, { added: 2, removed: 1 });
  assert.ok(rows.length < 14, `folded to ${rows.length} rows`);
  assert.ok(rows.some(r => r.t === 'gap' && r.n > 10));
  assert.deepEqual(cmp.lineDiff('', 'a\nb').map(r => r.t), ['add', 'add'], 'a new file is all additions');
  assert.deepEqual(cmp.lineDiff('a', '').map(r => r.t), ['del'], 'a removed file is all removals');
});

test('an HTML page is shown with no way to reach out: a policy is added to the page itself', () => {
  const withHead = cmp.htmlPreviewDocument('<html><head><title>x</title></head><body><script>alert(1)</script><img src="https://evil.example/a.png"></body></html>');
  assert.match(withHead, /Content-Security-Policy/); assert.match(withHead, /default-src 'none'/); assert.ok(withHead.indexOf('Content-Security-Policy') < withHead.indexOf('<script>'), 'the policy comes first');
  const bare = cmp.htmlPreviewDocument('<h1>Hi</h1>');
  assert.match(bare, /<body><h1>Hi<\/h1><\/body>/); assert.match(bare, /Content-Security-Policy/);
  assert.equal(cmp.timeSkipImageUrl('abc12345', 'img/a b.png', 'updated'), '/api/timeskip/file?id=abc12345&path=img%2Fa%20b.png&version=updated');
});

test('the working view says what is happening, from the host\'s words or the phase', async () => {
  assert.equal(act.activityWords({ phase: 'generating', activity: { code: 'trying-version', text: 'x', args: { step: 'Writing notes.md…' } } }), 'Writing notes.md…');
  assert.equal(act.activityWords({ phase: 'generating', activity: { code: 'trying-version', text: 'x' } }), 'Trying an improved version…');
  assert.equal(act.activityWords({ phase: 'checking', activity: { code: 'checking-result', text: 'x' } }), 'Checking the result…');
  assert.equal(act.activityWords({ phase: 'queued' }), 'Reading your files…');
  assert.equal(act.activityWords({ phase: 'checking' }), 'Checking the result…');
  assert.ok(act.phaseFraction('generating') > act.phaseFraction('checkpointing') && act.phaseFraction('checking') > act.phaseFraction('generating'));
  const panel = await readFile(new URL('../src/components/TimeSkipCompare.tsx', import.meta.url), 'utf8');
  // A page that draws itself with a script (a 3D scene) must run, or it shows as an empty dark page; the frame stays isolated and offline.
  assert.equal(cmp.HTML_PREVIEW_SANDBOX, 'allow-scripts');
  assert.match(panel, /sandbox={HTML_PREVIEW_SANDBOX}/);
  assert.ok(!/allow-same-origin|allow-forms|allow-popups|allow-top-navigation/.test(panel + cmp.HTML_PREVIEW_SANDBOX));
  const policy = cmp.htmlPreviewDocument('<p>x</p>');
  assert.match(policy, /script-src 'unsafe-inline'/); assert.ok(!/connect-src|https?:\/\//.test(policy.split('<body>')[0]), 'nothing may be fetched');
  assert.equal(cmp.needsOutsideFiles('<script src="https://cdn.example/three.js"></script>'), true);
  assert.equal(cmp.needsOutsideFiles('<script>draw()</script><a href="https://x.example">link</a>'), false);
});
