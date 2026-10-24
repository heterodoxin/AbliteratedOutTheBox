import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createUploads, parseMultipart } from '../server/chat-uploads.mjs';
import { createAttachments } from '../server/chat-files.mjs';
import { imageParts } from '../server/agent/agent-bridge.mjs';

const PNG = Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), Buffer.alloc(40, 1)]);
const multipart = (parts, boundary = 'XBOUND') => ({
  contentType: 'multipart/form-data; boundary=' + boundary,
  body: Buffer.concat([...parts.flatMap(p => [Buffer.from('--' + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="' + p.name + '"\r\nContent-Type: ' + p.type + '\r\n\r\n'), p.bytes, Buffer.from('\r\n')]), Buffer.from('--' + boundary + '--\r\n')]),
});

test('pictures and files arrive as multipart, a data address or raw bytes; pictures get an id, files land in the chat folder', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-up-'));
  try {
    const proj = join(dir, 'proj'); mkdirSync(proj);
    const att = createAttachments();
    const up = createUploads({ dir: join(dir, 'uploads'), folderOf: () => proj, attachments: att });
    const m = multipart([{ name: 'a.png', type: 'image/png', bytes: PNG }, { name: 'notes.md', type: 'text/markdown', bytes: Buffer.from('# hi\r\nthere') }]);
    assert.equal(parseMultipart(m.body, m.contentType).length, 2);
    const out = up.accept('c1', m);
    assert.equal(out[0].kind, 'image'); assert.match(out[0].url, /^\/api\/chat\/upload\/[0-9a-f-]{36}$/);
    assert.equal(out[1].kind, 'file'); assert.ok(existsSync(join(proj, '.strata', 'attachments', 'notes.md'))); assert.deepEqual(att.list('c1'), [out[1].path]);
    assert.equal(readFileSync(out[1].path, 'utf8'), '# hi\r\nthere', 'the bytes are kept exactly');
    const j = up.accept('c1', { body: Buffer.from(JSON.stringify({ name: 'p.png', dataUrl: 'data:image/png;base64,' + PNG.toString('base64') })), contentType: 'application/json' });
    assert.equal(j[0].kind, 'image');
    const raw = up.accept('c1', { body: PNG, contentType: 'image/png', name: 'x.png' });
    assert.equal(up.image(raw[0].id).type, 'image/png');
    assert.equal(up.dataUrls('c1', [raw[0].id])[0].url.slice(0, 22), 'data:image/png;base64,');
    assert.throws(() => up.dataUrls('other-chat', [raw[0].id]), /no longer available/, 'a picture belongs to its chat');
    assert.throws(() => up.dataUrls('c1', ['1', '2', '3', '4', '5']), /up to 4/);
    // the first bytes decide: a renamed non-picture is a file, a picture type the model cannot read is refused
    assert.equal(up.accept('c1', { body: Buffer.from('plain'), contentType: 'text/plain', name: 'renamed.png' })[0].kind, 'file');
    assert.throws(() => up.accept('c1', { body: Buffer.from('plain'), contentType: 'image/png', name: 'fake.png' }), /cannot read/);
    assert.throws(() => up.accept('c1', { body: Buffer.from('GIF89a....'), contentType: 'image/gif', name: 'a.gif' }), /PNG, JPEG or WebP/);
    const bigPng = Buffer.concat([PNG, Buffer.alloc(7 * 1024 * 1024)]);
    assert.throws(() => up.accept('c1', { body: bigPng, contentType: 'image/png', name: 'big.png' }), /bigger than 6 MB/);
    assert.throws(() => up.accept('c1', { body: Buffer.alloc(13 * 1024 * 1024), contentType: 'application/pdf', name: 'huge.pdf' }), /bigger than 12 MB/);
    const evil = up.accept('c1', { body: Buffer.from('x'), contentType: 'text/plain', name: '..' + String.fromCharCode(92) + '..' + String.fromCharCode(92) + 'evil.txt' });
    assert.ok(evil[0].path.startsWith(join(proj, '.strata', 'attachments')), 'a file name cannot climb out of the attachments folder');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('pictures become the Harness image parts; anything else is dropped', () => {
  const parts = imageParts([{ url: 'data:image/png;base64,AAAA', name: 'a.png' }, { url: 'data:image/gif;base64,AAAA' }, { url: 'nope' }, null]);
  assert.deepEqual(parts, [{ type: 'image', mediaType: 'image/png', data: 'AAAA', name: 'a.png' }]);
  assert.deepEqual(imageParts(undefined), []);
});
