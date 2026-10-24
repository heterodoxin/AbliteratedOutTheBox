/** Item 4: the Compare view can show pictures side by side, so the host lists them as pictures and serves them safely. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { diffRun, imageTypeOf, runImagePath } from '../server/timeskip.mjs';

function fixture() {
  const dir = mkdtempSync(join(tmpdir(), 'strata-cmp-'));
  const base = join(dir, 'base'), upd = join(dir, 'upd'), outside = join(dir, 'secret.png');
  mkdirSync(join(base, 'img'), { recursive: true }); mkdirSync(join(upd, 'img'), { recursive: true });
  writeFileSync(outside, 'not for you');
  writeFileSync(join(base, 'index.html'), '<h1>Old</h1>'); writeFileSync(join(upd, 'index.html'), '<h1>New</h1>');
  writeFileSync(join(base, 'img', 'logo.png'), Buffer.from([137, 80, 78, 71, 0, 1, 2])); writeFileSync(join(upd, 'img', 'logo.png'), Buffer.from([137, 80, 78, 71, 0, 9, 9, 9]));
  writeFileSync(join(base, 'same.png'), Buffer.from([137, 80, 78, 71, 0])); writeFileSync(join(upd, 'same.png'), Buffer.from([137, 80, 78, 71, 0]));
  writeFileSync(join(upd, 'new.svg'), '<svg xmlns="http://www.w3.org/2000/svg"/>');
  const record = { checkpoints: [{ id: 'cp1', label: 'baseline', path: base }], selected_checkpoint: null, attempts: [{ index: 1, path: upd }] };
  return { dir, record };
}

test('pictures are listed as pictures (changed ones only), text files keep their text', () => {
  const { dir, record } = fixture();
  try {
    const found = diffRun(record);
    const by = Object.fromEntries(found.files.map(f => [f.path, f]));
    assert.equal(by['img/logo.png'].image, true); assert.equal(by['img/logo.png'].status, 'changed'); assert.equal(by['img/logo.png'].original, undefined);
    assert.equal(by['new.svg'].image, true); assert.equal(by['new.svg'].status, 'added');
    assert.equal(by['same.png'], undefined, 'an unchanged picture is not listed');
    assert.equal(by['index.html'].original, '<h1>Old</h1>'); assert.equal(by['index.html'].updated, '<h1>New</h1>');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a picture is served only from inside the run, only as a picture', () => {
  const { dir, record } = fixture();
  try {
    assert.ok(runImagePath(record, 'img/logo.png', 'original')?.endsWith('logo.png'));
    assert.ok(runImagePath(record, 'img/logo.png', 'updated'));
    assert.equal(runImagePath(record, 'img/logo.png', 'other'), null);
    assert.equal(runImagePath(record, '../secret.png', 'original'), null, 'no way out of the folder');
    assert.equal(runImagePath(record, 'img/../../secret.png', 'updated'), null);
    assert.equal(runImagePath(record, 'index.html', 'updated'), null, 'a page is not a picture');
    assert.equal(runImagePath(record, 'missing.png', 'updated'), null);
    assert.equal(imageTypeOf('a/b/C.JPG'), 'image/jpeg'); assert.equal(imageTypeOf('a.txt'), null);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('the page that was made comes first; a browser profile and other housekeeping come last, without their contents', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-ts-rank-'));
  try {
    const base = join(dir, 'base'), cand = join(dir, 'cand');
    for (const root of [base, cand]) { mkdirSync(join(root, '.chrome-profile', 'Default', 'Cache'), { recursive: true }); mkdirSync(join(root, 'notes'), { recursive: true }); }
    for (let i = 0; i < 12; i++) { writeFileSync(join(base, '.chrome-profile', 'Default', 'Cache', 'f' + i), 'a' + i); writeFileSync(join(cand, '.chrome-profile', 'Default', 'Cache', 'f' + i), 'b' + i); }
    writeFileSync(join(cand, 'voxel-pagoda.html'), '<canvas></canvas>');
    writeFileSync(join(base, 'notes', 'a.md'), 'old'); writeFileSync(join(cand, 'notes', 'a.md'), 'new');
    const found = diffRun({ checkpoints: [{ id: 'cp1-baseline', label: 'baseline', path: base }], selected_checkpoint: 'cp1-baseline', attempts: [{ index: 1, path: cand }] });
    assert.deepEqual(found.files.slice(0, 2).map(f => f.path), ['voxel-pagoda.html', 'notes/a.md'], 'the page first, then other readable work');
    assert.equal(found.files[0].updated, '<canvas></canvas>');
    const noise = found.files.filter(f => f.noise);
    assert.equal(noise.length, 5, 'only a few housekeeping files are listed'); assert.ok(noise.every(f => f.unreadable && f.original === undefined));
    assert.equal(found.more, 7, 'the rest is counted, not listed');
    assert.ok(found.files.findIndex(f => f.noise) > 1, 'housekeeping is last');
    rmSync(join(cand, 'voxel-pagoda.html')); writeFileSync(join(cand, 'notes', 'a.md'), 'old');
    const onlyNoise = diffRun({ checkpoints: [{ id: 'cp1-baseline', label: 'baseline', path: base }], selected_checkpoint: 'cp1-baseline', attempts: [{ index: 1, path: cand }] });
    assert.deepEqual(onlyNoise.files, [], 'when only housekeeping changed nothing is shown as changed'); assert.equal(onlyNoise.housekeeping, 12);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a big file that did not change is not listed as changed; one that did is', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-ts-big-'));
  try {
    const base = join(dir, 'base'), cand = join(dir, 'cand'); mkdirSync(base); mkdirSync(cand);
    const big = Buffer.alloc(400 * 1024, 7);
    writeFileSync(join(base, 'archive.bin'), big); writeFileSync(join(cand, 'archive.bin'), big);
    writeFileSync(join(base, 'data.json'), Buffer.alloc(300 * 1024, 65)); writeFileSync(join(cand, 'data.json'), Buffer.alloc(300 * 1024, 66));
    writeFileSync(join(cand, 'page.html'), '<p>x</p>');
    const found = diffRun({ checkpoints: [{ id: 'cp1-baseline', label: 'baseline', path: base }], selected_checkpoint: 'cp1-baseline', attempts: [{ index: 1, path: cand }] });
    assert.deepEqual(found.files.map(f => f.path), ['page.html', 'data.json'], 'the untouched big file is gone from the list; the changed one is last (too big to show)');
    assert.equal(found.files[1].unreadable, true);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
