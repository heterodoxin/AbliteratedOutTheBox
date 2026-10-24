import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { stateRoot, tuningDir, timeSkipDir, tarBinary, whichCommand } from '../server/platform-paths.mjs';
import { which } from '../server/util.mjs';

const read = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');

test('Windows keeps tuning and Time-skip state under %LOCALAPPDATA%\\Strata, never HOME/.local/state', () => {
  const env = { LOCALAPPDATA: 'C:\\Users\\me\\AppData\\Local', HOME: '/home/ignored', XDG_STATE_HOME: '/ignored' };
  assert.equal(stateRoot(env, 'win32'), 'C:\\Users\\me\\AppData\\Local\\Strata');
  assert.equal(tuningDir(env, 'win32'), 'C:\\Users\\me\\AppData\\Local\\Strata\\tuning');
  assert.equal(timeSkipDir(env, 'win32'), 'C:\\Users\\me\\AppData\\Local\\Strata\\time-skip');
});

test('without LOCALAPPDATA, Windows falls back to the profile folder', () => {
  assert.equal(stateRoot({ USERPROFILE: 'C:\\Users\\me' }, 'win32', 'C:\\Users\\other'), 'C:\\Users\\me\\AppData\\Local\\Strata');
  assert.equal(stateRoot({}, 'win32', 'C:\\Users\\other'), 'C:\\Users\\other\\AppData\\Local\\Strata');
});

test('explicit directories win on every platform', () => {
  assert.equal(tuningDir({ STRATA_TUNE_DIR: 'D:\\tune' }, 'win32'), 'D:\\tune');
  assert.equal(timeSkipDir({ STRATA_TIMESKIP_DIR: '/t/ts' }, 'linux'), '/t/ts');
});

test('POSIX is unchanged: XDG_STATE_HOME, else ~/.local/state', () => {
  assert.equal(stateRoot({ XDG_STATE_HOME: '/x' }, 'linux'), '/x/strata');
  assert.equal(tuningDir({ HOME: '/home/me' }, 'linux'), '/home/me/.local/state/strata/tuning');
  assert.equal(timeSkipDir({ HOME: '/home/me' }, 'darwin'), '/home/me/.local/state/strata/time-skip');
});

test('the archive tool is the one the platform ships', () => {
  assert.equal(tarBinary({ SystemRoot: 'C:\\Windows' }, 'win32'), 'C:\\Windows\\System32\\tar.exe');
  assert.equal(tarBinary({}, 'win32'), 'C:\\Windows\\System32\\tar.exe');
  assert.equal(tarBinary({}, 'linux'), '/usr/bin/tar');
  assert.equal(tarBinary({}, 'darwin'), '/usr/bin/tar');
});

test('which asks where.exe on Windows (first match) and /usr/bin/env which elsewhere', async () => {
  assert.deepEqual(whichCommand('llama-server', { SystemRoot: 'C:\\Windows' }, 'win32'),
    { command: 'C:\\Windows\\System32\\where.exe', args: ['llama-server'] });
  assert.deepEqual(whichCommand('llama-server', {}, 'linux'), { command: '/usr/bin/env', args: ['which', 'llama-server'] });
  const seen = [];
  const win = await which('cloudflared', { platform: 'win32', env: { SystemRoot: 'C:\\Windows' },
    run: async (command, args) => { seen.push([command, args]); return 'C:\\tools\\cloudflared.exe\r\nC:\\other\\cloudflared.exe'; } });
  assert.equal(win, 'C:\\tools\\cloudflared.exe');
  assert.deepEqual(seen[0], ['C:\\Windows\\System32\\where.exe', ['cloudflared']]);
  assert.equal(await which('nothing', { platform: 'linux', run: async () => '' }), null);
  assert.equal(await which('sh', { platform: 'linux', run: async () => '/usr/bin/sh' }), '/usr/bin/sh');
});

test('the host no longer hard-codes /usr/bin/tar, /usr/bin/env or HOME/.local/state', () => {
  assert.ok(!read('../server/engine-install.mjs').includes('/usr/bin/tar'));
  assert.ok(!read('../server/util.mjs').includes("'/usr/bin/env'"));
  assert.ok(!read('../server/timeskip.mjs').includes('.local'));
  assert.ok(!read('../server/index.mjs').includes("'.local', 'state'"));
  assert.match(read('../server/engine-install.mjs'), /tarBinary\(\)/);
});
