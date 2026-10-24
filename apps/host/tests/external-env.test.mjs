/**
 * Programs that are not Strata's own (zenity, kdialog, xdg-open, the assistant's shell) get the environment the Linux AppImage was
 * started from, not the AppImage's libraries and settings (tester report 2026-10-17: zenity died with
 * "libssl.so.3: version OPENSSL_3.2.0 not found" and the folder picker did nothing).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { externalEnvironment, recordedOriginal, withoutAppImage } from '../server/external-env.mjs';
import { chooseFolder, chooserUnavailable, portalFolder } from '../server/folder-dialog.mjs';
import { agentEnvironment } from '../server/agent/harness-runtime.mjs';

const APPDIR = '/tmp/.mount_StrataAbc';
const leaky = {
  APPDIR, APPIMAGE: '/home/u/Strata.AppImage', ARGV0: 'Strata',
  LD_LIBRARY_PATH: `${APPDIR}/usr/lib:${APPDIR}/usr/lib/x86_64-linux-gnu:/opt/cuda/lib64`,
  PATH: `${APPDIR}/usr/bin:/usr/local/bin:/usr/bin`,
  XDG_DATA_DIRS: `${APPDIR}/usr/share:/usr/share:/usr/local/share`,
  GTK_PATH: `${APPDIR}//usr/lib/gtk-3.0`, GIO_MODULE_DIR: `${APPDIR}//usr/lib/gio/modules`, GDK_PIXBUF_MODULE_FILE: `${APPDIR}//usr/lib/gdk-pixbuf-2.0/2.10.0/loaders.cache`,
  GSETTINGS_SCHEMA_DIR: `${APPDIR}//usr/share/glib-2.0/schemas`, PYTHONHOME: `${APPDIR}/usr/python`, HOME: '/home/u', DISPLAY: ':0',
};

test('without a record, everything that points into the AppImage is removed and the user\'s own entries stay', () => {
  const env = externalEnvironment(leaky);
  assert.equal(env.APPDIR, undefined);
  assert.equal(env.APPIMAGE, undefined);
  assert.equal(env.LD_LIBRARY_PATH, '/opt/cuda/lib64', 'only the AppImage\'s folders go from the list');
  assert.equal(env.PATH, '/usr/local/bin:/usr/bin');
  assert.equal(env.XDG_DATA_DIRS, '/usr/share:/usr/local/share');
  for (const name of ['GTK_PATH', 'GIO_MODULE_DIR', 'GDK_PIXBUF_MODULE_FILE', 'GSETTINGS_SCHEMA_DIR', 'PYTHONHOME']) assert.equal(env[name], undefined, name);
  assert.equal(env.HOME, '/home/u');
  assert.equal(env.DISPLAY, ':0');
});

test('with the launcher\'s record, the original environment is restored exactly', () => {
  const original = { PATH: '/usr/bin', LD_LIBRARY_PATH: '/mine', HOME: '/home/u', DISPLAY: ':1' };
  const record = Buffer.from(Object.entries(original).map(([k, v]) => `${k}=${v}`).join('\0') + '\0').toString('base64');
  const env = externalEnvironment({ ...leaky, STRATA_ORIGINAL_ENV: record });
  assert.deepEqual(env, original);
  assert.deepEqual(recordedOriginal({ STRATA_ORIGINAL_ENV: record }), original);
  assert.equal(recordedOriginal({ STRATA_ORIGINAL_ENV: '!!!' }), null);
});

test('outside an AppImage nothing changes', () => {
  const plain = { PATH: '/usr/bin', LD_LIBRARY_PATH: '/opt/x/lib', HOME: '/h' };
  assert.deepEqual(externalEnvironment(plain), plain);
  assert.deepEqual(externalEnvironment(plain, { A: '1' }), { ...plain, A: '1' });
  assert.deepEqual(withoutAppImage(plain), plain);
});

test('the coding assistant\'s shell does not inherit the AppImage\'s settings either', () => {
  const env = agentEnvironment({ ...leaky, SOME_TOKEN: 'x' }, { DSH_HOME: '/h' });
  assert.equal(env.LD_LIBRARY_PATH, undefined);
  assert.ok(!/\.mount_/.test(JSON.stringify(env)), JSON.stringify(env));
  assert.equal(env.DSH_HOME, '/h');
  assert.equal(env.SOME_TOKEN, undefined);
});

const failure = (code, stderr = '') => Object.assign(new Error(`exit ${code}`), { code, stderr });

test('Linux folder chooser: a zenity that breaks is reported, not taken for a cancel, and the next chooser is tried', async () => {
  const seen = [];
  const libssl = failure(127, 'zenity: error while loading shared libraries: libssl.so.3: version `OPENSSL_3.2.0\' not found');
  const run = async (command, args, options) => { seen.push({ command, env: options?.env }); if (command === 'zenity') throw libssl; return { stdout: '/home/u/Models/\n' }; };
  const old = { ...process.env };
  Object.assign(process.env, leaky);
  try {
    const picked = await chooseFolder('Pick', { platform: 'linux', run, portal: async () => null });
    assert.deepEqual(picked, { path: '/home/u/Models' });
    assert.deepEqual(seen.map(s => s.command), ['zenity', 'kdialog']);
    for (const s of seen) { assert.equal(s.env.LD_LIBRARY_PATH, '/opt/cuda/lib64'); assert.equal(s.env.APPDIR, undefined); }
    // every chooser broken or missing: a plain sentence that names what to do, never a silent nothing
    const none = await chooseFolder('Pick', { platform: 'linux', run: async () => { throw libssl; }, portal: async () => null });
    assert.equal(none.unavailable, true);
    assert.equal(none.failed, true);
    assert.match(none.reason, /Install zenity or kdialog to pick a folder, or type the path/);
    const answer = chooserUnavailable(none);
    assert.equal(answer.code, 'folder-chooser-unavailable');
    const missingAll = await chooseFolder('Pick', { platform: 'linux', run: async () => { throw failure('ENOENT'); }, portal: async () => null });
    assert.equal(missingAll.unavailable, true);
    assert.match(missingAll.reason, /zenity or kdialog/);
    // a plain exit 1 is still a cancel; so is exit 1 with harmless warnings
    assert.deepEqual(await chooseFolder('Pick', { platform: 'linux', run: async () => { throw failure(1, 'Gtk-Message: Failed to load module "canberra-gtk-module"'); }, portal: async () => null }), { cancelled: true });
    // a zenity that cannot reach the display exits 1 too: that is a failure
    const noDisplay = await chooseFolder('Pick', { platform: 'linux', run: async () => { throw failure(1, 'Unable to init server: Could not connect: Connection refused'); }, portal: async () => null });
    assert.equal(noDisplay.unavailable, true);
  } finally { for (const k of Object.keys(leaky)) { if (old[k] === undefined) delete process.env[k]; else process.env[k] = old[k]; } }
});

test('Linux folder chooser: the desktop portal answers when zenity and kdialog are not installed', async () => {
  const run = async () => { throw failure('ENOENT'); };
  assert.deepEqual(await chooseFolder('Pick', { platform: 'linux', run, portal: async () => ({ path: '/home/u/Pics' }) }), { path: '/home/u/Pics' });
  assert.deepEqual(await chooseFolder('Pick', { platform: 'linux', run, portal: async () => ({ cancelled: true }) }), { cancelled: true });
});

test('portalFolder reads the Response signal of its request', async () => {
  const handle = '/org/freedesktop/portal/desktop/request/1_42/t1';
  const monitor = new EventEmitter();
  monitor.stdout = new EventEmitter();
  monitor.kill = () => { monitor.exitCode = 0; };
  monitor.exitCode = null;
  const spawnProcess = () => monitor;
  const run = async (command, args) => {
    assert.equal(command, 'gdbus');
    assert.ok(args.includes('org.freedesktop.portal.FileChooser.OpenFile'));
    queueMicrotask(() => monitor.stdout.emit('data', `${handle}: org.freedesktop.portal.Request.Response (uint32 0, {'uris': <['file:///home/u/My%20Models']>})\n`));
    return { stdout: `(objectpath '${handle}',)\n` };
  };
  assert.deepEqual(await portalFolder('Pick', { run, env: {}, spawnProcess }), { path: '/home/u/My Models' });
  const cancel = new EventEmitter();
  cancel.stdout = new EventEmitter(); cancel.kill = () => {}; cancel.exitCode = null;
  const run2 = async () => { queueMicrotask(() => cancel.stdout.emit('data', `${handle}: org.freedesktop.portal.Request.Response (uint32 1, {})\n`)); return { stdout: `(objectpath '${handle}',)\n` }; };
  assert.deepEqual(await portalFolder('Pick', { run: run2, env: {}, spawnProcess: () => cancel }), { cancelled: true });
  const none = new EventEmitter(); none.stdout = new EventEmitter(); none.kill = () => {}; none.exitCode = null;
  assert.equal(await portalFolder('Pick', { run: async () => { throw failure('ENOENT'); }, env: {}, spawnProcess: () => none }), null);
});
