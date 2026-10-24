/** #3673: the route change froze every click for 4-5 s (Chromium and the WebView2 window about 4 s, Firefox about 5 s) because the view
    transition's update callback waited for requestAnimationFrame, which the browser holds while that callback is pending. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const result = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/router.ts', import.meta.url))], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent', alias: { react: fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url)) } });
globalThis.window ??= globalThis;
const { afterRouteUpdate } = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);

test('a held requestAnimationFrame cannot hold the route update: it resolves on the deadline', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  globalThis.requestAnimationFrame = () => 1;                 // the browser never calls back while the transition is pending
  globalThis.document = { body: {} };
  globalThis.MutationObserver = undefined;                    // nothing draws: only the deadline is left
  let done = 0;
  afterRouteUpdate(() => { done++; }, 150);
  t.mock.timers.tick(149);
  assert.equal(done, 0);
  t.mock.timers.tick(1);
  assert.equal(done, 1, 'resolved at the deadline, with no frame ever delivered');
  t.mock.timers.tick(1000);
  assert.equal(done, 1, 'resolves once');
});

test('the first change to the page after the hash flipped resolves it at once, and the observer is released', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  globalThis.requestAnimationFrame = () => 1;
  let notify; let disconnected = false;
  globalThis.document = { body: {} };
  globalThis.MutationObserver = class { constructor(fn) { notify = fn; } observe() {} disconnect() { disconnected = true; } };
  let done = 0;
  afterRouteUpdate(() => { done++; }, 150);
  notify();                                                   // React committed the new screen
  t.mock.timers.tick(0);
  assert.equal(done, 1, 'well before the 150 ms deadline');
  assert.equal(disconnected, true);
  t.mock.timers.tick(500);
  assert.equal(done, 1);
});

test('the transition callback no longer waits on a frame', async () => {
  const { readFileSync } = await import('node:fs');
  const text = readFileSync(new URL('../src/lib/router.ts', import.meta.url), 'utf8');
  const callback = text.slice(text.indexOf('const transition = document.startViewTransition'), text.indexOf('transition.ready.catch'));
  assert.ok(callback.length > 40);
  assert.match(callback, /afterRouteUpdate\(resolve\)/);
  assert.doesNotMatch(callback, /requestAnimationFrame/);
});
