/**
 * Where the person's models go: what a picked folder means (0.2.2).
 *
 * - A drive on its own (D:\ or /) is never filled with model files: it becomes D:\Strata (the data folder) and D:\Strata\Models (the
 *   model library), which the person is told.
 * - The data folder is the person's own choice, so it is written into the settings file Strata Control reads (`data_dir`). Control's
 *   installer accepts a data folder that is that setting, whatever the folder is called; for a folder it has not been told about it wants a
 *   name starting with "Strata" (a rule against an AI tool filling an arbitrary folder, tools/strata_mcp.py check_data_dir).
 * - The link that makes `<data>/models` and `<data>/Models` one folder is only needed where the file system tells them apart; where it cannot be
 *   made (exFAT, FAT, a network share) the models simply go in `<data>/models`.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, statSync, symlinkSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, parse, resolve } from 'node:path';

export const isDriveRoot = path => { const p = resolve(path); return p === parse(p).root; };

/** The data folder for a picked folder: a drive root gets a `Strata` folder of its own. `{ path, note }`. */
export function dataFolderFromPick(picked) {
  const path = resolve(picked);
  if (!isDriveRoot(path)) return { path, note: null };
  const target = join(path, 'Strata');
  return { path: target, note: `${path} is a whole drive, so the models go in ${target}.` };
}

/** The model library folder for a picked folder: a drive root gets `Strata/Models`. `{ path, note }`. */
export function libraryFolderFromPick(picked) {
  const path = resolve(picked);
  if (!isDriveRoot(path)) return { path, note: null };
  const target = join(path, 'Strata', 'Models');
  return { path: target, note: `${path} is a whole drive, so the models go in ${target}.` };
}

/** Make a folder the person asked for; a refusal of the disk is said in words. */
export function makeFolder(path) {
  try { mkdirSync(path, { recursive: true }); }
  catch (error) { throw new Error(`Strata could not create ${path} (${error.code === 'EACCES' || error.code === 'EPERM' ? 'access was denied' : error.code === 'ENOSPC' ? 'the drive is full' : error.message}). Choose another folder.`); }
  return path;
}

/** `models` beside `Models`: a link where the two are different names. A disk that cannot hold the link gets a plain `models` folder instead. */
export function alignModels(data, library, { link = symlinkSync } = {}) {
  const models = join(data, 'models');
  if (existsSync(models)) return { data, linked: false };
  try { link(process.platform === 'win32' ? library : 'Models', models, process.platform === 'win32' ? 'junction' : 'dir'); return { data, linked: true }; }
  catch { /* exFAT, FAT, NFS, a share: no links there */ }
  try { mkdirSync(models, { recursive: true }); } catch { /* the install says what it cannot write */ }
  return { data, linked: false };
}


function readSettings(file) {
  try { const parsed = JSON.parse(readFileSync(file, 'utf8').replace(/^﻿/, '')); return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}; } catch { return {}; }
}

/** The data folder a settings file names and when it was last changed (`data_dir_at`, seconds; the file's own time for a file that has none). */
export function readDataFolder(file) {
  if (!file) return null;
  const current = readSettings(file);
  if (typeof current.data_dir !== 'string' || !current.data_dir) return null;
  let at = Number(current.data_dir_at);
  if (!Number.isFinite(at) || at <= 0) { try { at = statSync(file).mtimeMs / 1000; } catch { at = 0; } }
  return { folder: current.data_dir, at };
}

/** The settings file the Terminal (command-line Strata) keeps: the same place setup.py and strata_control/discovery.py use. */
export function terminalSettingsFile(env = process.env, platform = process.platform) {
  if (env.STRATA_TERMINAL_SETTINGS_FILE) return env.STRATA_TERMINAL_SETTINGS_FILE;
  if (platform === 'win32') return join(env.APPDATA || join(homedir(), 'AppData', 'Roaming'), 'Strata', 'settings.json');
  return join(env.XDG_CONFIG_HOME || join(homedir(), '.config'), 'strata', 'settings.json');
}

/** Write the person's data folder into a settings file (merged, atomically) with the time of the change; returns whether it changed. Never
 *  throws: the folder still works for the app, only the Terminal and the installer's own check lose the shared setting. */
export function rememberDataFolder(file, folder, at = Date.now() / 1000, touch = false) {
  if (!file || !folder) return false;
  try {
    const current = readSettings(file);
    if (current.data_dir === folder && !touch) return false;
    mkdirSync(dirname(file), { recursive: true });
    const temp = `${file}.${process.pid}.tmp`;
    writeFileSync(temp, JSON.stringify({ ...current, data_dir: folder, data_dir_at: at }, null, 1) + '\n', { mode: 0o600 });
    renameSync(temp, file);
    return true;
  } catch { return false; }
}

const same = (a, b) => resolve(a).toLowerCase() === resolve(b).toLowerCase();

/**
 * One models folder for the app and the Terminal. The app keeps its choice in its own settings file (`appFile`), the Terminal in `terminalFile`;
 * whichever was changed last wins and the other file is brought up to date. Nothing is moved. A folder that is gone (a disk that is not
 * connected) is never adopted. `app` is the app's own answer when its file names none (`{ folder, at }`).
 * Returns `{ folder, from }`: the shared folder (null when neither names one) and `from` = 'terminal' when the app has to take over the
 * Terminal's folder, 'app' when the Terminal's file was updated, null when nothing changed.
 */
export function syncDataFolder({ appFile, terminalFile, app = null, exists = path => existsSync(path) }) {
  const mine = readDataFolder(appFile) ?? app;
  const theirs = readDataFolder(terminalFile);
  const theirsUsable = theirs && exists(theirs.folder);
  if (!theirsUsable) {
    if (mine && theirs && !same(mine.folder, theirs.folder)) { rememberDataFolder(terminalFile, mine.folder, mine.at); return { folder: mine.folder, from: 'app' }; }
    return { folder: mine?.folder ?? null, from: null };
  }
  if (!mine || (!same(mine.folder, theirs.folder) && theirs.at > mine.at)) {
    rememberDataFolder(appFile, theirs.folder, theirs.at);
    return { folder: theirs.folder, from: 'terminal' };
  }
  if (!same(mine.folder, theirs.folder)) { rememberDataFolder(terminalFile, mine.folder, mine.at); return { folder: mine.folder, from: 'app' }; }
  return { folder: mine.folder, from: null };
}
