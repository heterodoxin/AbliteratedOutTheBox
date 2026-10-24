import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { previewFile, rawFile, parseCsv, createAttachments, withAttachments } from '../server/chat-files.mjs';

const NL = String.fromCharCode(10);

test('previews: html and markdown as text, csv as a table, pictures and pdf as raw addresses, nothing outside the folder', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-pv-'));
  try {
    const root = join(dir, 'p'); mkdirSync(root);
    writeFileSync(join(root, 'a.html'), '<h1>x</h1>'); writeFileSync(join(root, 'n.md'), '# T'); writeFileSync(join(root, 'd.csv'), ['name,qty', '"Tea, green",2', '"say ""hi""",3', ''].join(NL));
    writeFileSync(join(root, 'p.png'), Buffer.from([137, 80, 78, 71, 0])); writeFileSync(join(root, 'x.pdf'), Buffer.from('%PDF-1.4')); writeFileSync(join(root, 'z.zip'), Buffer.from([80, 75, 0, 0])); writeFileSync(join(dir, 'secret.txt'), 's');
    assert.deepEqual([previewFile(root, 'a.html', 's').kind, previewFile(root, 'a.html', 's').text], ['html', '<h1>x</h1>']);
    assert.equal(previewFile(root, 'n.md', 's').text, '# T');
    assert.deepEqual(previewFile(root, 'd.csv', 's').rows, [['name', 'qty'], ['Tea, green', '2'], ['say "hi"', '3']]);
    const png = previewFile(root, 'p.png', 'chat 1'); assert.equal(png.mime, 'image/png'); assert.match(png.rawUrl, /^\/api\/files\/raw\?sessionId=chat%201&path=/);
    assert.equal(previewFile(root, 'x.pdf', 's').mime, 'application/pdf');
    assert.match(previewFile(root, 'z.zip', 's').note, /no preview/);
    assert.equal(rawFile(root, 'p.png').mime, 'image/png'); assert.throws(() => rawFile(root, 'a.html'), /Only pictures and PDFs/);
    assert.throws(() => previewFile(root, join('..', 'secret.txt'), 's'), /outside/);
    assert.throws(() => rawFile(root, join('..', 'secret.txt')), /outside/);
    assert.deepEqual(parseCsv('a;b\n1;2', ';'), [['a', 'b'], ['1', '2']]);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('attach as context: the files are named in the next message once, then forgotten', () => {
  const a = createAttachments();
  a.add('c', '/p/a.html'); a.add('c', '/p/a.html'); a.add('c', '/p/with space.md');
  assert.deepEqual(a.list('c'), ['/p/a.html', '/p/with space.md']);
  assert.equal(withAttachments('improve it', a.take('c')), '@/p/a.html @"/p/with space.md" improve it');
  assert.deepEqual(a.take('c'), []); assert.equal(withAttachments('hi', []), 'hi');
  a.add('d', 'x'); assert.deepEqual(a.remove('d', 'x'), []);
});
