/** Round 8: deliverable cards, grouped activity and the pieces of a reply in order (contract: notes-split-agent8.md, item 12). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const src = f => fileURLToPath(new URL(`../src/${f}`, import.meta.url));
const stub = fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url));
const result = await build({ entryPoints: [src('lib/deliverables.ts')], bundle: true, format: 'esm', platform: 'node', write: false, alias: { react: stub }, logLevel: 'silent' });
const D = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);

test('a deliverable frame is read; unknown kinds become "other"; no path means no card', () => {
  const d = D.readDeliverable({ path: 'a/pagoda.html', abs: 'C:/a/pagoda.html', kind: 'html', status: 'created', size: 7270, lines: 225, added: 225, removed: 0 });
  assert.equal(d.name, 'pagoda.html'); assert.equal(D.fileTypeLabel(d), 'HTML');
  assert.equal(D.readDeliverable({ path: 'x.bin', kind: 'weird' }).kind, 'other');
  assert.equal(D.readDeliverable({ kind: 'html' }), null);
  assert.equal(D.fileTypeLabel({ kind: 'markdown', name: 'README.md' }), 'MD');
  assert.equal(D.fileSize(7270), '7.1 KB'); assert.equal(D.fileSize(512), '512 B');
});

test('text between activities stays in order: text, group, text; a group frame re-sent replaces its line', () => {
  let b = D.appendText(undefined, 'Building. ');
  b = D.appendText(b, 'Now.');
  b = D.upsertGroup(b, D.readGroup({ id: 'g1', closed: false, line: 'Ran a command', files: [] }));
  b = D.appendText(b, 'Next.');
  b = D.upsertGroup(b, D.readGroup({ id: 'g1', closed: true, line: 'Ran 2 commands', files: [] }));
  assert.deepEqual(b.map(x => x.kind), ['text', 'group', 'text']);
  assert.equal(b[0].text, 'Building. Now.'); assert.equal(b[1].line, 'Ran 2 commands'); assert.equal(b[1].closed, true);
});

test('one card per path; created stays created and the line counts add up', () => {
  const a = D.readDeliverable({ path: 'p.html', kind: 'html', status: 'created', size: 10, lines: 5, added: 5, removed: 0 });
  const b = D.readDeliverable({ path: 'p.html', kind: 'html', status: 'changed', size: 14, lines: 7, added: 3, removed: 1 });
  const list = D.mergeDeliverable(D.mergeDeliverable(undefined, a), b);
  assert.equal(list.length, 1); assert.equal(list[0].status, 'created'); assert.equal(list[0].added, 8); assert.equal(list[0].removed, 1); assert.equal(list[0].size, 14);
});

test('a finished run opens one main result in the sidebar: the file the answer names, else the one written last, only kinds the sidebar shows', () => {
  const f = (path, kind) => D.readDeliverable({ path, kind, status: 'created', size: 10, lines: 5, added: 5, removed: 0 });
  const add = (list, file) => D.mergeDeliverable(list, file);
  let list = add(add(add(undefined, f('notes.md', 'markdown')), f('pagoda.html', 'html')), f('build.py', 'code'));
  assert.equal(D.mainResult(list, 'All done.').name, 'pagoda.html', 'the last written one the sidebar can show (a script is not shown)');
  assert.equal(D.mainResult(list, 'The notes are in notes.md.').name, 'notes.md', 'the file the answer names wins');
  assert.equal(D.mainResult(list, 'Open Pagoda.HTML and read notes.md for details; the page is pagoda.html').name, 'pagoda.html', 'the last one it names');
  list = add(list, f('notes.md', 'markdown'));
  assert.equal(D.mainResult(list, '').name, 'notes.md', 'a file written again counts as written last');
  assert.equal(list.length, 3, 'still one card per path');
  assert.equal(D.mainResult(add(undefined, f('run.py', 'code')), 'Made run.py'), null, 'code is not opened by itself');
  assert.equal(D.mainResult(undefined, ''), null);
  for (const kind of ['html', 'markdown', 'pdf', 'image', 'csv', 'text']) assert.ok(D.SIDEBAR_KINDS.includes(kind), kind);
});

test('the file card: Open uses the system, the menu has Open in sidebar, and a run opens its result by itself', async () => {
  const { readFile } = await import('node:fs/promises');
  const card = await readFile(new URL('../src/components/Deliverables.tsx', import.meta.url), 'utf8');
  assert.match(card, /fileAction\.open\(sessionId, file\.path, 'app'\)[^]*?\{tr\("Open"\)\}/, 'Open hands the file to its own app or the browser');
  assert.match(card, /showOf\(file, sessionId, onPreview, file\.kind\); \}\}><Icon name="panel"\/>\{tr\("Open in sidebar"\)\}/, 'the menu opens the sidebar preview');
  const chat = await readFile(new URL('../src/screens/Chat.tsx', import.meta.url), 'utf8');
  assert.match(chat, /mainResult\(lastReply\.deliverables, lastReply\.content\)/);
  assert.match(chat, /before !== true \|\| lastReply\.streaming \|\| lastReply\.error/, 'only a run seen finishing, and not a failed one');
});
