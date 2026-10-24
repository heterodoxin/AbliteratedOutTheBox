// T4 (tools and images review, 2026-10-22): the approval gate fails closed. When the permission mode of a session cannot be read, a call
// that is not provably read-only asks, as in Manual; it used to run without a card.
import test from 'node:test';
import assert from 'node:assert/strict';
import { apply, MANUAL_PRESET } from '../server/agent/plugins/strata-gate.mjs';

function gate(current) {
  let handler;
  apply({ on: (_e, h) => { handler = h; }, inject: () => {}, sessions: new Map(), permissionPresets: { current }, approval: { overrideOf: () => 'never' } });
  return handler;
}
const session = { header: { origin: 'user' } };
const allow = async () => ({ kind: 'allow' });
const quiet = async fn => { const warn = console.warn; const lines = []; console.warn = (...a) => lines.push(a.join(' ')); try { return [await fn(), lines]; } finally { console.warn = warn; } };

test('a call that changes things asks when the preset cannot be resolved', async () => {
  const handler = gate(() => { throw new Error('unknown session'); });
  const [result, lines] = await quiet(() => handler({ name: 'write', agent: { session }, input: { file_path: 'x' } }, allow));
  assert.equal(result.kind, 'ask');
  assert.match(lines.join('\n'), /could not read the permission mode.*unknown session.*Manual/);
});

test('a provably read-only call still goes through, and a shell write asks', async () => {
  const handler = gate(() => { throw new Error('gone'); });
  const [read] = await quiet(() => handler({ name: 'read', agent: { session }, input: { file_path: 'x' } }, allow));
  assert.equal(read.kind, 'allow');
  const [shell] = await quiet(() => handler({ name: 'bash', agent: { session }, input: { command: 'rm -rf x' } }, allow));
  assert.equal(shell.kind, 'ask');
});

test('with a readable preset nothing changes: Bypass runs, Manual asks', async () => {
  const bypass = gate(() => 'strata-bypass');
  assert.equal((await bypass({ name: 'write', agent: { session }, input: { file_path: 'x' } }, allow)).kind, 'allow');
  const manual = gate(() => MANUAL_PRESET);
  assert.equal((await manual({ name: 'write', agent: { session }, input: { file_path: 'x' } }, allow)).kind, 'ask');
});
