import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

// WebKitGTK 2.52 crashes its UI process (null pointer in AcceleratedBackingStore::update) on document.startViewTransition,
// so the route morph must never be used there. This test pins the engine check and that the router uses it.
const source = await readFile(new URL('../src/lib/router.ts', import.meta.url), 'utf8');
const fn = source.match(/export function isWebKitGtk[\s\S]*?\r?\n}\r?\n/)[0];
const { code } = await transform(fn, { loader: 'ts', format: 'esm' });
const { isWebKitGtk } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('WebKitGTK is recognised, other engines are not', () => {
  assert.equal(isWebKitGtk('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15'), true);
  assert.equal(isWebKitGtk('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko)'), true);
  assert.equal(isWebKitGtk('Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0'), false);
  assert.equal(isWebKitGtk('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36'), false);
  assert.equal(isWebKitGtk('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15'), false);
  assert.equal(isWebKitGtk('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Edg/130.0'), false);
});

test('the router skips the view transition on WebKitGTK', () => {
  assert.match(source, /typeof document\.startViewTransition === 'function' &&\s*!isWebKitGtk\(\)/);
  assert.equal((source.match(/document\.startViewTransition\(/g) ?? []).length, 1, 'one call site only, guarded above');
});
