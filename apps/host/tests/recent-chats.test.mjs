import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const source = await readFile(new URL('../src/lib/recent-chats.ts', import.meta.url), 'utf8');
const { code } = await transform(source, { loader: 'ts', format: 'esm' });
const { recentChats, recentChatDescription } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('recent chats use activity order, exclude archives, and leave the source list intact', () => {
  const sessions = [
    { id: 'old-pinned', updatedAt: '2026-10-01', pinned: true },
    { id: 'second', updatedAt: '2026-10-08' },
    { id: 'archived', updatedAt: '2026-10-11', archived: true },
    { id: 'latest', updatedAt: '2026-10-10' },
    { id: 'third', updatedAt: '2026-10-07' },
  ];
  const before = structuredClone(sessions);
  assert.deepEqual(recentChats(sessions).map(chat => chat.id), ['latest', 'second', 'third', 'old-pinned']);
  assert.deepEqual(sessions, before);
  assert.deepEqual(recentChats([]), []);
  assert.equal(recentChats([sessions[0]]).length, 1);
});

test('chat descriptions show the latest request and handle empty or long conversations', () => {
  assert.equal(recentChatDescription({ messages: [
    { role: 'user', content: 'First request' },
    { role: 'user', content: '  Plan\n our   weekend  ' },
    { role: 'assistant', content: 'A long reply' },
  ] }), 'Plan our weekend');
  assert.equal(recentChatDescription({ messages: [] }), 'No messages yet');
  assert.equal(recentChatDescription({ messages: [{ role: 'assistant', content: 'Welcome' }] }), 'Welcome');
  const text = recentChatDescription({ messages: [{ role: 'user', content: 'a'.repeat(200) }] });
  assert.ok(text.length <= 160);
  assert.ok(text.endsWith('…'));
});


test('sidebar includes an older current chat plus the latest five, with stable activity slots', () => {
  const sessions = Array.from({ length: 8 }, (_, i) => ({ id: String(i), updatedAt: `2026-10-${String(10-i).padStart(2, '0')}` }));
  assert.deepEqual(recentChats(sessions, '7').map(s => s.id), ['0','1','2','3','4','7']);
  assert.deepEqual(recentChats(sessions, '0').map(s => s.id), ['0','1','2','3','4','5']);
  assert.deepEqual(recentChats(sessions, '3').map(s => s.id), ['0','1','2','3','4','5']);
  assert.equal(recentChats(sessions, 'missing').length, 5);
  sessions[7].archived = true;
  assert.equal(recentChats(sessions, '7').some(s => s.id === '7'), false);
});
