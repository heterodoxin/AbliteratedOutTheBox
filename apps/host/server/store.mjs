import { migrateSetup } from './setup.mjs';
/** Tiny persisted state for the host: one JSON file under STRATA_HOME. */
import { accessSync, constants, mkdirSync, readFileSync, writeFileSync, existsSync, realpathSync, readdirSync, statSync, statfsSync, symlinkSync } from 'node:fs';
import { alignModels, isDriveRoot, libraryFolderFromPick, makeFolder } from './model-folder.mjs';
import { dirname, isAbsolute, join, parse, relative, resolve } from 'node:path';
import { homedir } from 'node:os';
import { defaultModelDirectory } from './model-directory-default.mjs';

/**
 * Prefer ~/.strata, the way the packaged app stores its data. Sandboxes and
 * locked-down accounts cannot write to the home directory, so fall back to a
 * folder next to the checkout rather than failing to boot.
 */
function resolveHome() {
  const candidates = [process.env.STRATA_HOME, join(homedir(), '.strata'), join(process.cwd(), '.strata')].filter(
    Boolean,
  );
  for (const dir of candidates) {
    try {
      mkdirSync(join(dir, 'models'), { recursive: true });
      mkdirSync(join(dir, 'data'), { recursive: true });
      return dir;
    } catch {
      /* try the next candidate */
    }
  }
  throw new Error('No writable data directory found — set STRATA_HOME to a writable path');
}

export const HOME = resolveHome();
export const DATA_DIR = join(HOME, 'data');
const STATE_FILE = join(DATA_DIR, 'state.json');

const DEFAULTS = {
  version: 1,
  setup: { complete: false, flowVersion: 2, step: 1, history: [] },
  installed: {}, // id -> { installedAt, bytes }
  current: null,
  loaded: null,
  beginner: {
    priority: 'balanced',
    contextLength: 'recommended',
    readImages: false,
    gpuIdle: 'never',
  },
  expert: {}, // id -> value
  serving: { otherDevices: false, apiKey: null, pauseServing: false },
};

function read() {
  try {
    if (!existsSync(STATE_FILE)) return structuredClone(DEFAULTS);
    const parsed = JSON.parse(readFileSync(STATE_FILE, 'utf8'));
    return { ...structuredClone(DEFAULTS), ...parsed, setup: migrateSetup(parsed.setup) };
  } catch {
    return structuredClone(DEFAULTS);
  }
}

let state = read();
const DEFAULT_MODELS_DIRECTORY=defaultModelDirectory();
/**
 * The model folder for a new install: the saved one, an existing library inside STRATA_HOME, else Documents/Strata/Models. A Documents folder
 * that cannot be made (Controlled folder access, a disconnected share, an offline OneDrive known folder) must not stop the host from starting:
 * the models go to STRATA_HOME/models instead, and `notice` says so in plain words.
 */
export function pickModelsDirectory({ saved, home, preferred, fixedByEnv = false }) {
 if (saved) return { directory: saved, notice: '' };
 const fallback = join(home, 'models');
 let legacy = fallback;
 try { legacy = realpathSync(fallback); } catch { /* the folder is made at start; keep the plain path */ }
 function hasWeights(folder,depth=0){try{return readdirSync(folder,{withFileTypes:true}).some(entry=>entry.isFile()&&/\.gguf(?:\.part|\.partial)?$/i.test(entry.name)||entry.isDirectory()&&depth<4&&hasWeights(join(folder,entry.name),depth+1));}catch{return false;}}
 // Keep an existing library in place. A new install uses Documents/Strata/Models.
 if (!fixedByEnv && hasWeights(legacy)) return { directory: legacy, notice: '' };
 try {
  mkdirSync(preferred, { recursive: true });
  accessSync(preferred, constants.R_OK | constants.W_OK | constants.X_OK);
  return { directory: realpathSync(preferred), notice: '' };
 } catch (error) {
  return { directory: legacy, notice: `Strata could not use ${preferred} for models (${error.code ?? error.message}), so models go to ${legacy}. You can pick another folder in Settings.` };
 }
}
const MODELS_PICK = pickModelsDirectory({ saved: state.modelsDir, home: HOME, preferred: DEFAULT_MODELS_DIRECTORY, fixedByEnv: !!process.env.STRATA_MODELS_DIR });
/** Plain words for the person when the usual model folder could not be used (empty otherwise). */
export const MODELS_DIR_NOTICE = MODELS_PICK.notice;
if (MODELS_DIR_NOTICE) console.warn(`[strata] ${MODELS_DIR_NOTICE}`);
export let MODELS_DIR = MODELS_PICK.directory;

/** Native setup writes weights to data_dir/models. Keep its data root one level up. */
export function nativeDataDirectory(input) {
 if(!input||!existsSync(input))return input;
 if(MODELS_DIR_NOTICE)return input;
 const root=realpathSync(input),legacy=realpathSync(join(HOME,'models'));
 // Retained legacy libraries are model folders too, not native data roots.
 // A named data root satisfies the installer's guard; its alias leaves weights in place.
 if(root===legacy){
  const data=join(HOME,'Strata-data'),nativeModels=join(data,'models');
  mkdirSync(data,{recursive:true});
  // Unlike a fresh default, a retained library must not fall back to an empty folder if linking fails.
  if(!existsSync(nativeModels))symlinkSync(process.platform==='win32'?root:relative(data,root),nativeModels,process.platform==='win32'?'junction':'dir');
  const a=statSync(nativeModels),b=statSync(root);
  if(realpathSync(nativeModels)!==root&&!(a.ino&&a.ino===b.ino&&a.dev===b.dev))throw new Error('The model folder conflicts with another library. Choose a different folder.');
  return data;
 }
 if(process.env.STRATA_MODELS_DIR||!existsSync(DEFAULT_MODELS_DIRECTORY)||root!==realpathSync(DEFAULT_MODELS_DIRECTORY))return input;
 const data=dirname(root);
 // `models` and `Models` are one folder on Windows and macOS; where they are not (Linux) a link makes them one, and where no link can be made (exFAT, a share) models go in `models`.
 alignModels(data,root);
 return data;
}

/** Existing folders only: a mistyped or disconnected drive must never become a new empty library. */
export function validateModelsDirectory(input) {
  if (typeof input !== 'string' || !input.trim() || input.includes('\0')) throw new Error('Enter a model directory.');
  const path = input.trim().replace(/^~(?=$|[\\/])/, homedir());
  if (!isAbsolute(path)) throw new Error('Enter the full path to an existing folder.');
  let directory;
  try {
    directory = realpathSync(path);
    if (!statSync(directory).isDirectory()) throw new Error();
    accessSync(directory, constants.R_OK | constants.W_OK | constants.X_OK);
  } catch { throw new Error('This folder is unavailable or not writable. Check the path and folder permissions.'); }
  // A whole drive (D:\) is not filled with models: D:\Strata\Models is made and used.
  if (isDriveRoot(directory)) directory = realpathSync(makeFolder(libraryFolderFromPick(directory).path));
  const data = realpathSync(DATA_DIR);
  const contains = (parent, child) => { const r = relative(parent, child); return !r || !r.startsWith('..') && !isAbsolute(r); };
  if (directory === parse(directory).root || contains(directory, data) || contains(data, directory)) throw new Error('Choose a separate folder for models, outside Strata’s app data.');
  return resolve(directory);
}

export function changeModelsDirectory(directory) {
  const next = validateModelsDirectory(directory);
  if (next === MODELS_DIR) return false;
  const installedByDirectory = { ...state.installedByDirectory, [MODELS_DIR]: state.installed ?? {} };
  const updated = { ...state, modelsDir: next, installedByDirectory, installed: installedByDirectory[next] ?? {}, current: null, loaded: null };
  // Commit the preference before exposing it; a persistence error leaves the old library active.
  writeFileSync(STATE_FILE, JSON.stringify(updated, null, 2));
  state = updated; MODELS_DIR = next;
  return true;
}

export function get() {
  return state;
}

export function save() {
  try {
    writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  } catch (err) {
    console.error('[strata] could not persist state:', err.message);
  }
}

export function update(patch) {
  state = { ...state, ...patch };
  save();
  return state;
}

export function diskFreeGb(path = HOME) {
  try {
    const s = statfsSync(path);
    return (s.bavail * s.bsize) / 1024 ** 3;
  } catch {
    return 0;
  }
}

export function diskTotalGb(path = HOME) {
  try {
    const s = statfsSync(path);
    return (s.blocks * s.bsize) / 1024 ** 3;
  } catch {
    return 0;
  }
}
