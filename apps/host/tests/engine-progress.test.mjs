import test from 'node:test';
import assert from 'node:assert/strict';
import { engineProgressMark } from '../server/agent/engine-progress.mjs';
import { readFileSync } from 'node:fs';

test('the mark moves with the prompt read and with generation, and is null when the engine is idle or silent', () => {
  const at = live => engineProgressMark({ metrics: { live } });
  assert.equal(engineProgressMark(null), null);
  assert.equal(engineProgressMark({ metrics: null }), null);
  assert.equal(at({ state: 'idle' }), null);
  assert.equal(at({ state: 'unloaded' }), null);
  const a = at({ state: 'reading', prompt_read: 4096, prompt_total: 90000 });
  const b = at({ state: 'reading', prompt_read: 8192, prompt_total: 90000 });
  assert.notEqual(a, b);
  assert.equal(a, at({ state: 'reading', prompt_read: 4096, prompt_total: 90000 }), 'no progress, same mark');
  assert.notEqual(at({ state: 'generating', generated: 1 }), at({ state: 'generating', generated: 2 }));
  assert.notEqual(at({ state: 'reading', slots: [{ state: 'reading', prompt_read: 1 }] }), at({ state: 'reading', slots: [{ state: 'reading', prompt_read: 2 }] }));
});

test('the host wires it in and the local profile lifts the 300 s stream-idle timer', () => {
  assert.match(readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8'), /engineProgress: instanceId => engineProgressMark\(/);
  assert.match(readFileSync(new URL('../server/agent/profile.mjs', import.meta.url), 'utf8'), /streamIdleTimeoutMs: 60 \* 60_000/);
});
