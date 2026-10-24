import { ROOMY_HOST } from './helpers/hardware.mjs';
/** Host regressions use a disposable demo home. No browser automation or model downloads. */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createServer } from 'node:net';
import { createRequire } from 'node:module';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');

test('Markdown renders every incomplete stream prefix without looping', { timeout: 15000 }, async () => {
  const bundle = await build({
    stdin: { contents: `import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server'; import { Markdown } from './src/components/Markdown.tsx'; export {setTranslationLanguage as setLanguage} from './src/lib/i18n.ts'; export const render = text => renderToStaticMarkup(React.createElement(Markdown, { text }));`, resolveDir: root },
    bundle: true, write: false, platform: 'node', format: 'cjs', jsx: 'automatic', external: ['katex', 'katex/dist/katex.min.css'],   // math loads on demand in the browser
    plugins: [{ name: 'copy-control-stub', setup(b) {
      b.onResolve({ filter: /^\.\/ui$/ }, () => ({ path: 'ui', namespace: 'test-stub' }));
      b.onLoad({ filter: /.*/, namespace: 'test-stub' }, () => ({ contents: 'export const IconBtn = () => null; export const Btn = () => null;' }));
    } }],
  });
  // SSR is a parser regression, not a substitute for browser appearance checks.
  const mod = { exports: {} };
  new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
  const { render, setLanguage } = mod.exports;
  for(const language of ['zh','ja','de','es','ru','pt','ko']) { setLanguage(language); const markup=render('Ready\n\n**Settings**\n\n*Download*\n\n[Cancel](https://example.com)\n\n| Name | Value |\n| --- | --- |\n| Close | Language |'); for(const word of ['Ready','Settings','Download','Cancel','Name','Value','Close','Language'])assert(markup.includes(word),language+': user text changed: '+word); }
  setLanguage('en');
  for (const marker of ['- ', '* ', '+ ', '1. ', '2) ', '  - ', '  12. ']) assert.ok(render(marker).includes(marker));
  const text = '# Answer\n\n- First item\n- Second item\n\n1. First step\n2. Second step\n\n```js\nconsole.log("hello");\n```\n\n| Name | Value |\n| --- | --- |\n| One | Two |';
  for (let i = 0; i <= text.length; i++) assert.equal(typeof render(text.slice(0, i)), 'string');
  assert.match(render(text), /<ul>/); assert.match(render(text), /<ol>/); assert.match(render(text), /<table/);
  assert.ok(!render('<script>alert(1)</script>').includes('<script>'));
  // #3268: math is found in a paragraph and as a block; until KaTeX is loaded it shows as source, and money is plain text
  assert.match(render('Radius $r = 2$ here'), /md-math md-math--source/);
  assert.match(render('$$\nx = 1\n$$'), /md-math-block/);
  assert.ok(!render('costs $5 and $10').includes('md-math'));
  for (let i = 0; i <= 40; i++) assert.equal(typeof render('A $$x^2 + y$$ and $q$ b'.slice(0, i)), 'string');
});

test('setup, config and lifecycle preserve truthful outcomes in the demo host', { timeout: 30000 }, async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-reliability-'));
  const socket = createServer(); await new Promise(r => socket.listen(0, '127.0.0.1', r));
  const port = socket.address().port; await new Promise(r => socket.close(r));
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, env: { ...process.env, STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'), STRATA_PORT: String(port), STRATA_REAL_ENGINE: '0', STRATA_TEST_HARDWARE: ROOMY_HOST }, stdio: ['ignore', 'pipe', 'pipe'] });
  let logs = ''; host.stdout.on('data', d => logs += d); host.stderr.on('data', d => logs += d);
  const base = `http://127.0.0.1:${port}`;
  async function call(path, method = 'GET', body) {
    const response = await fetch(base + path, { method, headers: body ? { 'Content-Type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined });
    return { status: response.status, data: await response.json() };
  }
  try {
    for (let i = 0; i < 80; i++) { try { if ((await call('/api/models')).status === 200) break; } catch {} await new Promise(r => setTimeout(r, 100)); }
    assert.equal((await call('/api/setup/complete', 'POST')).status, 409);
    await call('/api/setup/step', 'POST', { step: 3, dismissed: true, choices: { variantId: 'qwen38-q4_k_m', contextLength: '8k', profile: 'performance', batch: 1024, batchAuto: false } });
    let setup = (await call('/api/setup')).data;
    assert.equal(setup.complete, false); assert.equal(setup.step, 2); assert.equal(setup.dismissed, true); assert.equal(setup.choices.contextLength, '8k'); assert.equal(setup.choices.profile, 'performance'); assert.equal(setup.choices.batch, 1024); assert.equal(setup.choices.batchAuto, false);
    const before = (await call('/api/engine')).data;
    const expert = before.expert.map(p => ({ id: p.id, value: p.id === 'perf.batch' ? 1024 : p.id === 'sample.temp' ? 0.9 : p.value }));
    expert.push({ id: 'custom.future-option', value: 77 });
    assert.equal((await call('/api/engine', 'PATCH', { beginner: before.beginner, expert, replaceExpert: true, extra: { futureAdapter: 'preserved' } })).status, 200);
    const invalid = await call('/api/engine', 'PATCH', { expert: [{ id: 'sample.temp', value: 99 }] });
    assert.equal(invalid.status, 400);
    assert.equal((await call('/api/engine')).data.expert.find(p => p.id === 'sample.temp').value, 0.9);
    await call('/api/engine', 'PATCH', { expert: before.expert.filter(p => p.id.startsWith('sample.')).map(p => ({ id: p.id, value: p.recommended })) });
    const after = (await call('/api/engine')).data;
    assert.equal(after.expert.find(p => p.id === 'sample.temp').value, 0.6);
    assert.equal(after.expert.find(p => p.id === 'perf.batch').value, 1024);
    assert.equal(after.expert.find(p => p.id === 'custom.future-option').value, 77);
    assert.equal(after.extra.futureAdapter, 'preserved');
    assert.equal((await call('/api/engine/restart', 'POST')).status, 500);
    const installing = await call('/api/models/qwen38-q4_k_m/install', 'POST');
    assert.equal(installing.data.mode, 'simulated'); assert.ok(Array.isArray(installing.data.variants));
    assert.equal(installing.data.loaded, null);
    assert.equal((await call('/api/setup/complete', 'POST')).status, 409);
    await call('/api/models/qwen38-q4_k_m/cancel', 'POST');
    assert.equal((await call('/api/models')).data.variants.find(v => v.id === 'qwen38-q4_k_m').state, 'available');
    await call('/api/models/qwen38-q4_k_m/install', 'POST');
    let installed = false;
    for (let i = 0; i < 120; i++) { const lib = (await call('/api/models')).data; if (lib.variants.find(v => v.id === 'qwen38-q4_k_m').state === 'installed') { installed = true; break; } await new Promise(r => setTimeout(r, 100)); }
    assert.ok(installed, logs);
    const loadingPromise = call('/api/models/qwen38-q4_k_m/load', 'POST');
    await new Promise(r => setTimeout(r, 100));
    // Simulated runtime placement can complete synchronously; native loads expose loading until healthy.
    assert.ok(['loading','loaded'].includes((await call('/api/models')).data.variants.find(v => v.id === 'qwen38-q4_k_m').state));
    assert.equal((await loadingPromise).data.loaded, 'qwen38-q4_k_m');
    assert.equal((await call('/api/setup/complete', 'POST')).data.complete, true);
    await call('/api/setup/step', 'POST', { step: 3, dismissed: true, choices: { contextLength: '16k' } });
    const resumed = (await call('/api/setup')).data;
    assert.equal(resumed.complete, false);
    assert.equal(resumed.step, 2);
    assert.equal(resumed.choices.contextLength, '16k'); assert.equal(resumed.choices.profile, 'performance'); assert.equal(resumed.choices.batch, 1024);
    const monitor = (await call('/api/monitor')).data;
    assert.equal(monitor.telemetry.memory, 'demo'); assert.equal(monitor.telemetry.thermals, 'demo');
    const skills = (await call('/api/skills')).data;
    await call('/api/skills/' + skills[0].id, 'PATCH', { enabled: false, onlyIn: 'root' });
    const fs = await import('node:fs/promises');
    const persisted = JSON.parse(await fs.readFile(join(home, 'data/state.json'), 'utf8'));
    assert.equal(persisted.skillPreferences[skills[0].id].enabled, false);
    assert.equal(persisted.skillPreferences[skills[0].id].onlyIn, 'root');
    const changedSkill = { name: 'Review helper', description: 'Checks the draft.', body: 'Find missing evidence. Explain each finding.', triggers: ['check draft', 'review text'] };
    assert.equal((await call('/api/skills/' + skills[0].id, 'PATCH', changedSkill)).status, 200);
    const rescannedSkill = (await call('/api/skills')).data.find(s => s.id === skills[0].id);
    assert.equal(rescannedSkill.body, changedSkill.body); assert.equal(rescannedSkill.name, changedSkill.name);
    assert.equal(rescannedSkill.enabled, false); assert.equal(rescannedSkill.onlyIn, 'root');
    assert.equal((await call('/api/skills/' + skills[0].id, 'PATCH', { body: '' })).status, 400);
    assert.equal((await call('/api/skills/not-found', 'PATCH', changedSkill)).status, 404);
    assert.equal((await call('/api/skills')).data.find(s => s.id === skills[0].id).body, changedSkill.body);
    const savedEdits = JSON.parse(await fs.readFile(join(home, 'data/state.json'), 'utf8'));
    assert.deepEqual(savedEdits.skillEdits[skills[0].id], changedSkill);
    await call('/api/models/unload', 'POST');
    assert.equal((await call('/api/models')).data.loaded, null);
    assert.equal((await call('/api/setup/complete', 'POST')).status, 409);
  } finally {
    host.kill('SIGTERM'); await new Promise(r => host.once('exit', r)); await rm(home, { recursive: true, force: true });
  }
});
