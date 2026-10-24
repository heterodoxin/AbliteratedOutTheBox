// O3 (tools and images review, 2026-10-22): a Markdown image in an answer renders as a picture when it is already on this computer or inside
// the message, and never fetches anything from the internet by default.
import assert from 'node:assert/strict';
import test from 'node:test';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

async function load() {
  const bundle = await build({
    stdin: { contents: `import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server'; import { Markdown } from './src/components/Markdown.tsx'; import { imagePlan } from './src/lib/markdown-images.ts'; export { imagePlan }; export const render = (text, imageSession) => renderToStaticMarkup(React.createElement(Markdown, { text, imageSession }));`, resolveDir: root },
    bundle: true, write: false, platform: 'node', format: 'cjs', jsx: 'automatic', external: ['katex', 'katex/dist/katex.min.css'],
    plugins: [{ name: 'copy-control-stub', setup(b) {
      b.onResolve({ filter: /^\.\/ui$/ }, () => ({ path: 'ui', namespace: 'test-stub' }));
      b.onLoad({ filter: /.*/, namespace: 'test-stub' }, () => ({ contents: 'export const IconBtn = () => null; export const Btn = () => null;' }));
    } }],
  });
  const mod = { exports: {} };
  new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
  return mod.exports;
}

test('a data: PNG, JPEG, WebP or GIF renders as an <img>', async () => {
  const { render } = await load();
  const html = render(`Here it is: ![dot](${PNG}) done`);
  assert.match(html, /<img [^>]*src="data:image\/png;base64,/);
  assert.match(html, /alt="dot"/);
  assert.match(html, /referrerPolicy|referrerpolicy="no-referrer"/i);
  for (const type of ['jpeg', 'jpg', 'webp', 'gif']) assert.match(render(`![x](data:image/${type};base64,QUJD)`), /<img /, type);
});

test('SVG data, script addresses, file: and other schemes never become an <img>', async () => {
  const { render } = await load();
  for (const src of ['data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=', 'data:text/html;base64,QUJD', 'javascript:alert(1)', 'file:///C:/secret.png',
    'blob:https://x/1', 'ftp://x/y.png', '//host/share/x.png', '\\\\host\\share\\x.png', 'data:image/png;base64,QUJD" onerror="alert(1)']) {
    const html = render(`![x](${src})`, 'chat-1');
    assert.ok(!/<img /.test(html), src);
    assert.ok(!/<a /.test(html), src);
  }
});

test('a remote picture is a link, not a request: nothing is fetched from the internet by default', async () => {
  const { render, imagePlan } = await load();
  const html = render('![Monthly sales](https://example.com/sales.png)', 'chat-1');
  assert.ok(!/<img /.test(html));
  assert.match(html, /<a [^>]*href="https:\/\/example.com\/sales.png"[^>]*>Monthly sales<\/a>/);
  assert.deepEqual(imagePlan('http://example.com/a.png'), { kind: 'link', href: 'http://example.com/a.png' });
});

test('a path in the chat folder is read through the host, and only when the chat is known', async () => {
  const { render, imagePlan } = await load();
  const html = render('![chart](plots/sales.png)', 'chat 1');
  assert.match(html, /<img [^>]*src="\/api\/files\/raw\?sessionId=chat%201&amp;path=plots%2Fsales.png"/);
  assert.match(render('![chart](C:\\work\\sales.png)', 'c1'), /<img [^>]*path=C%3A%5Cwork%5Csales.png/);
  assert.ok(!/<img /.test(render('![chart](plots/sales.png)')));                 // no chat: shown as its alt text
  assert.match(render('![chart](plots/sales.png)'), /chart/);
  assert.deepEqual(imagePlan('/api/files/raw?sessionId=a&path=b'), { kind: 'img', src: '/api/files/raw?sessionId=a&path=b' });
});

test('links, code and emphasis around a picture are unchanged, and the old text-with-a-bang is gone', async () => {
  const { render } = await load();
  const html = render('See [the report](https://example.com/r) and `code` and **bold**. ![a](' + PNG + ')');
  assert.match(html, /<a [^>]*>the report<\/a>/);
  assert.match(html, /<code class="md-code">code<\/code>/);
  assert.match(html, /<strong>bold<\/strong>/);
  assert.ok(!html.includes('>!<') && !/!<a /.test(html));
  assert.ok(!/<img /.test(render('`![x](' + PNG + ')`')));        // inside a code span it stays code
});

test('every incomplete stream prefix of an answer with a picture renders', async () => {
  const { render } = await load();
  const text = `Chart:\n\n![sales](${PNG})\n\nand ![b](https://example.com/b.png) end`;
  for (let i = 0; i <= text.length; i += 7) assert.equal(typeof render(text.slice(0, i), 'c1'), 'string');
});
