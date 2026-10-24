/** Item 2: an approval card asks in plain words; the exact command stays in the card's Details. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { askFor } from '../server/agent/ask.mjs';
import { TurnProjector } from '../server/agent/events.mjs';

const sh = command => askFor('pwsh', { command });
const ev = (type, data, seq = 1, time = 1000) => ({ type: 'event', event: { type, seq, time, data } });

test('opening a folder is asked as opening that folder, with the common ones by name', () => {
  assert.deepEqual(sh('Start-Process explorer.exe -ArgumentList "C:\\Users\\Ana\\Documents"').id, 'open-known-folder');
  assert.deepEqual(sh('Start-Process explorer.exe -ArgumentList "C:\\Users\\Ana\\Documents"').params, ['Documents']);
  assert.deepEqual(sh('explorer.exe C:\\Users\\Ana\\Downloads').params, ['Downloads']);
  assert.deepEqual(sh('Start-Process explorer.exe C:\\work\\Reports'), { id: 'open-folder', params: ['Reports'], why: 'opens-window', detail: 'C:\\work\\Reports' });
  assert.equal(sh('Invoke-Item C:\\work\\plan.pdf').id, 'open-file');
  assert.equal(sh('explorer.exe').id, 'open-explorer');
});

test('running a program, deleting, creating and copying are named for what they are', () => {
  assert.deepEqual(sh('Start-Process "C:\\Users\\Ana\\Downloads\\setup.exe" -ArgumentList /S').params, ['setup.exe']);
  assert.equal(sh('Start-Process "C:\\Users\\Ana\\Downloads\\setup.exe" -ArgumentList /S').id, 'run-program');
  assert.deepEqual(sh('Remove-Item C:\\tmp\\old.txt -Force'), { id: 'delete', params: ['old.txt'], why: 'delete-why', detail: 'C:\\tmp\\old.txt' });
  assert.equal(sh('rm -rf build').id, 'delete');
  assert.equal(sh('New-Item -ItemType File -Path C:\\Users\\Ana\\Desktop\\notes.txt').id, 'create-file');
  assert.equal(sh('mkdir C:\\work\\out').id, 'create-folder');
  assert.equal(sh('Copy-Item a.txt b.txt').id, 'copy');
  assert.equal(sh('Get-Date | Out-File C:\\x\\a.txt').id, 'write-file');
  assert.equal(sh('echo hi > notes.txt').params[0], 'notes.txt');
  assert.deepEqual(sh('git push origin main'), { id: 'git', params: ['push'], why: 'changes-repo' });
  assert.equal(sh('npm install left-pad').id, 'install');
  assert.equal(sh('python build.py').id, 'run-script');
});

test('files the agent writes or edits, and the tools that are not commands', () => {
  assert.deepEqual(askFor('write', { file_path: 'C:/Users/Ana/Desktop/notes.txt' }), { id: 'create-file', params: ['notes.txt'], why: 'replaces-if-exists', detail: 'C:/Users/Ana/Desktop/notes.txt' });
  assert.equal(askFor('edit', { file_path: 'C:/a/b.md' }).id, 'change-file');
  assert.equal(askFor('job_kill', {}).id, 'stop-task');
  assert.deepEqual(askFor('web_fetch', { url: 'https://example.com/a?b=1' }).params, ['example.com']);
  assert.deepEqual(askFor('mcp__github__create_issue', {}).params, ['github / create_issue']);
});

test('what it cannot read it does not pretend to: a generic question, with the model\'s own description when there is one', () => {
  assert.deepEqual(sh('foo --bar'), { id: 'run-command', params: [], why: 'run-command-why' });
  assert.equal(askFor('pwsh', { command: 'foo --bar' }, 'Rebuild the index').detail, 'Rebuild the index');
  assert.equal(sh('Get-Date; Remove-Item x.txt').id, 'run-steps', 'two different steps are said to be steps');
});

test('the card carries the question, and the exact command stays in its arguments for Details', () => {
  const p = new TurnProjector({ startedAt: 0 });
  const card = p.consume(ev('tool/call', { callId: 'a1', name: 'pwsh', arguments: JSON.stringify({ command: 'Start-Process explorer.exe -ArgumentList "C:\\Users\\Ana\\Documents"', description: 'Open Documents' }) }))[0].tool;
  assert.equal(card.ask.id, 'open-known-folder'); assert.deepEqual(card.ask.params, ['Documents']);
  assert.match(card.preview, /Start-Process explorer\.exe/, 'the command is still sent for the Details disclosure');
});
