import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createImageModels, IMAGE_CATALOG } from '../server/image-models.mjs';

function fixture(t, options = {}) {
  const home = mkdtempSync(join(tmpdir(), 'strata-image-test-'));
  t.after(() => rmSync(home, { recursive: true, force: true }));
  const paths = { dataDir: join(home, 'data'), modelsDir: join(home, 'models') };
  return { ...paths, service: createImageModels({ ...paths, demoDuration: 25, ...options }) };
}
const data = Buffer.from('GGUFtest-image-weights');
const smallCatalog = { ...IMAGE_CATALOG, variants: [{ ...IMAGE_CATALOG.variants[0], bytes: data.length, sha256: createHash('sha256').update(data).digest('hex') }] };

test('image selection starts Off and selecting a version never downloads it', t => {
  let requests = 0;
  const { service, dataDir } = fixture(t, { fetcher: () => { requests++; throw new Error('No automatic downloads'); } });
  assert.equal(service.library().selected, 'off');
  assert.deepEqual(service.library().variants.map(v => v.label), ['Q4','Q5','Q6','Q8']);
  service.select('q4');
  assert.equal(requests, 0);
  assert.equal(service.library().variants[0].state, 'available');
  assert.throws(() => service.apply(), /Download/);
  assert.throws(() => service.select('invalid'), /Choose/);
  service.select('off');
  assert.equal(service.library().applied, 'off');
  assert.equal(JSON.parse(readFileSync(join(dataDir, 'image-models.json'))).selected, 'off');
});
test('demo download has progress, survives restart and stays downloaded when switched Off', async t => {
  const events = [];
  const { service, ...paths } = fixture(t, { changed: value => events.push(value) });
  service.select('q5'); service.install('q5'); await service.whenIdle();
  assert.equal(service.library().variants[1].state, 'installed');
  assert.ok(events.some(e => e.variants[1].progress?.pct > 0));
  service.apply(); service.select('off'); service.apply();
  assert.equal(service.library().variants[1].state, 'installed');
  const restarted = createImageModels(paths);
  assert.equal(restarted.library().selected, 'off');
  assert.equal(restarted.library().variants[1].state, 'installed');
});
test('cancelled downloads do not become installed and can be retried', async t => {
  const { service } = fixture(t);
  service.install('q6');
  assert.throws(() => service.install('q4'), /Finish or cancel/);
  await service.cancel('q6');
  assert.equal(service.library().variants[2].state, 'available');
  service.install('q6'); await service.whenIdle();
  assert.equal(service.library().variants[2].state, 'installed');
});
test('real downloads verify exact bytes, GGUF header and hash at a pinned URL', async t => {
  let fetched;
  const { service, ...paths } = fixture(t, { real: true, catalog: smallCatalog, fetcher: async url => { fetched = url; return new Response(data); } });
  service.select('q4'); service.install('q4'); await service.whenIdle();
  assert.equal(service.library().variants[0].state, 'installed');
  assert.ok(fetched.includes(`/resolve/${IMAGE_CATALOG.revision}/`));
  assert.equal(service.library().generationAvailable, false);
  assert.throws(() => service.apply(), /compatible image engine/);
  const restarted = createImageModels({ ...paths, real: true, catalog: smallCatalog });
  assert.equal(restarted.library().variants[0].state, 'installed');
  assert.equal(restarted.library().applied, 'off');
});
test('failed verification and failed fetch do not leave installed or partial files', async t => {
  for (const body of [Buffer.from('bad!test-image-weights'), data.subarray(0, 7)]) {
    const { service, modelsDir } = fixture(t, { real: true, catalog: smallCatalog, fetcher: async () => new Response(body) });
    service.install('q4'); await service.whenIdle();
    assert.equal(service.library().variants[0].state, 'error');
    assert.equal(existsSync(join(modelsDir, 'image', 'q4', smallCatalog.variants[0].file)), false);
    assert.equal(existsSync(join(modelsDir, 'image', 'q4', smallCatalog.variants[0].file + '.part')), false);
  }
  const { service } = fixture(t, { real: true, catalog: smallCatalog, fetcher: async () => new Response('Unavailable', { status: 503 }) });
  service.install('q4'); await service.whenIdle();
  assert.equal(service.library().variants[0].state, 'error');
});
test('a demo install never becomes a real file or a real applied image model', async t => {
  const { service, ...paths } = fixture(t);
  service.select('q4'); service.install('q4'); await service.whenIdle(); service.apply();
  const real = createImageModels({ ...paths, real: true });
  assert.equal(real.library().variants[0].state, 'available');
  assert.equal(real.library().applied, 'off');
});
