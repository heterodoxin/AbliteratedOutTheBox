// T5 / T6 (tools and images review, 2026-10-22): the agent's tool servers are the default model's, one naming rule is shared by the editor, the
// save check and the profile, and a server that is left out is reported instead of vanishing.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { mcpRows, mcpSkippedNotices, buildProfile } from '../server/agent/profile.mjs';
import { MCP_NAME_MAX, MCP_NAME_RE, validMcpName, MCP_NAME_HELP } from '../server/mcp-names.mjs';
import { validateMcpServers } from '../server/native-settings.mjs';

const cmd = { command: 'node', args: ['x.js'] };

test('one rule: 32 characters of letters, numbers, hyphens and underscores', () => {
  assert.equal(MCP_NAME_MAX, 32);
  for (const ok of ['files', 'my_server-1', 'A', 'x'.repeat(32)]) assert.ok(validMcpName(ok), ok);
  for (const bad of ['', 'my.server', 'x'.repeat(33), 'x'.repeat(40), 'a b', 'ä', 'a/b']) assert.ok(!validMcpName(bad), bad);
  assert.match(MCP_NAME_HELP, /up to 32/);
});

test('the editor keeps the same rule as the host (a copy in src/lib/mcp-names.ts)', async () => {
  const text = await readFile(new URL('../src/lib/mcp-names.ts', import.meta.url), 'utf8');
  assert.ok(text.includes(String(MCP_NAME_RE)), 'same regular expression');
  assert.ok(text.includes(`MCP_NAME_MAX = ${MCP_NAME_MAX}`));
  assert.ok(text.includes(MCP_NAME_HELP), 'same help sentence');
  const editor = await readFile(new URL('../src/components/NativeMcpEditor.tsx', import.meta.url), 'utf8');
  assert.ok(editor.includes('validMcpName(name)') && !editor.includes('{1,64}'));
});

test('the save check refuses a new name over 32 but does not block the other settings of a name saved before', () => {
  const long = 'x'.repeat(40);
  assert.throws(() => validateMcpServers({ [long]: cmd }, {}), /up to 32/);
  assert.throws(() => validateMcpServers({ 'my.server': cmd }, {}), /up to 32/);
  assert.deepEqual(Object.keys(validateMcpServers({ [long]: cmd }, { [long]: cmd })), [long]);
  assert.deepEqual(Object.keys(validateMcpServers({ files: cmd }, {})), ['files']);
});

test('a server that cannot be wired is reported with its reason, not dropped silently', () => {
  const skipped = [];
  const rows = mcpRows({
    files: cmd, [`${'y'.repeat(40)}`]: cmd, 'my.server': cmd, empty: {}, off: { ...cmd, disabled: true },
    Files2: cmd, 'files2': cmd,
  }, {}, skipped);
  assert.deepEqual(rows.map(r => r.config.serverName), ['files', 'Files2']);
  assert.deepEqual(skipped.map(s => [s.name.slice(0, 6), s.why]), [['yyyyyy', 'name'], ['my.ser', 'name'], ['empty', 'address'], ['files2', 'duplicate']]);
  // switched off on purpose is not an error
  assert.ok(!skipped.some(s => s.name === 'off'));
});

test('the notices name the servers, one sentence per reason, in the catalog wording', async () => {
  const notices = mcpSkippedNotices([{ name: 'a.b', why: 'name' }, { name: 'c'.repeat(33), why: 'name' }, { name: 'e', why: 'address' }]);
  assert.equal(notices.length, 2);
  assert.match(notices[0].message, /^These tool servers were left out because their names are not allowed: a\.b, c{33}\./);
  assert.match(notices[1].message, /no program or web address: e\.$/);
  assert.notEqual(notices[0].code, notices[1].code);
  assert.deepEqual(mcpSkippedNotices([]), []);
  // every sentence is in the catalog of every language (src/locales/tools-<lang>.json)
  for (const lang of ['en', 'de', 'es', 'pt', 'ru', 'ja', 'ko', 'zh']) {
    const catalog = JSON.parse(await readFile(new URL(`../src/locales/tools-${lang}.json`, import.meta.url), 'utf8'));
    const english = JSON.parse(await readFile(new URL('../src/locales/tools-en.json', import.meta.url), 'utf8'));
    for (const text of ['These tool servers were left out because their names are not allowed: {0}. A name can have up to 32 letters, numbers, hyphens or underscores.',
      'These tool servers were left out because they have no program or web address: {0}.',
      'These tool servers were left out because another one has almost the same name: {0}.']) {
      const id = Object.keys(english).find(k => english[k] === text);
      assert.ok(id, text);
      assert.ok(catalog[id]?.includes('{0}'), `${lang}:${id}`);
    }
  }
});

test('the agent gets the tool servers of the DEFAULT instance, not the first one loaded', () => {
  const local = [
    { id: 'first', modelId: 'm1', baseUrl: 'http://127.0.0.1:1/v1', mcpServers: { alpha: cmd } },
    { id: 'second', modelId: 'm2', baseUrl: 'http://127.0.0.1:2/v1', mcpServers: { beta: cmd } },
  ];
  const names = profile => profile.patch.match(/"serverName": "([^"]+)"/g)?.map(s => s.slice(15, -1)) ?? [];
  assert.deepEqual(names(buildProfile({ local, defaultLocalId: 'second' })), ['beta']);
  assert.deepEqual(names(buildProfile({ local, defaultLocalId: 'first' })), ['alpha']);
  assert.deepEqual(names(buildProfile({ local })), ['alpha'], 'no default named: the first, as before');
  assert.deepEqual(names(buildProfile({ local, defaultLocalId: 'gone' })), ['alpha'], 'a default that is not loaded: the first');
  // the agent's own default model follows the same instance, and a different default changes what a running agent must be restarted for
  assert.equal(buildProfile({ local, defaultLocalId: 'second' }).defaultRoute.instanceId, 'second');
  assert.notEqual(buildProfile({ local, defaultLocalId: 'second' }).toolsSignature, buildProfile({ local, defaultLocalId: 'first' }).toolsSignature);
  // the Settings servers are added for every default
  const withSettings = buildProfile({ local, defaultLocalId: 'second', tools: { off: new Set(), search: { provider: 'builtin', key: '' }, servers: { gamma: cmd } } });
  assert.deepEqual(names(withSettings).sort(), ['beta', 'gamma']);
});

test('buildProfile hands the skipped servers on for the chat to report', () => {
  const local = [{ id: 'a', modelId: 'm', baseUrl: 'http://127.0.0.1:1/v1', mcpServers: { ['z'.repeat(33)]: cmd, ok: cmd } }];
  const profile = buildProfile({ local, defaultLocalId: 'a' });
  assert.deepEqual(profile.mcpSkipped.map(s => s.why), ['name']);
  assert.match(profile.patch, /"serverName": "ok"/);
});
