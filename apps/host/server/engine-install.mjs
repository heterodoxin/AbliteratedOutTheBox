/**
 * Engine installer.
 *
 * Strata ships no binaries: it fetches the llama.cpp release build that matches
 * this platform into $STRATA_HOME/engine, so "Install inference engine" in the
 * first-run wizard is a real step rather than a progress bar that lies.
 */
import { createWriteStream, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { exec } from './util.mjs';
import { HOME } from './store.mjs';
import { tarBinary } from './platform-paths.mjs';
import { nowIso } from './util.mjs';
import { require as requireAcquisition } from './acquisition.mjs';

const ENGINE_DIR = join(HOME, 'engine');
const RELEASES = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8';

const state = {
  status: 'idle', // idle | checking | downloading | extracting | done | error
  pct: 0,
  mbps: 0,
  stage: '',
  asset: null,
  version: null,
  error: null,
  installedPath: null,
};

let hubRef = null;
let running = null;

export function init(hub) {
  hubRef = hub;
}

export function status() {
  return { ...state, engineDir: ENGINE_DIR };
}

function emit() {
  hubRef?.send('engine-install', status());
}

/** Which release asset belongs on this machine. */
function assetMatcher() {
  const { platform, arch } = process;
  if (platform === 'darwin') {
    return {
      test: (name) => /macos/i.test(name) && (arch === 'arm64' ? /arm64/i.test(name) : /x64|x86_64/i.test(name)) && /\.tar\.gz$/.test(name),
      ext: 'tar.gz',
    };
  }
  if (platform === 'linux') {
    return {
      test: (name) => /ubuntu|linux/i.test(name) && (arch === 'arm64' ? /arm64|aarch64/i.test(name) : /x64|x86_64/i.test(name)) && /\.tar\.gz$/.test(name),
      ext: 'tar.gz',
    };
  }
  return {
    test: (name) => /win/i.test(name) && /x64|x86_64/i.test(name) && /\.zip$/.test(name),
    ext: 'zip',
  };
}

export async function findRelease() {
  requireAcquisition('The llama.cpp engine');
  state.status = 'checking';
  state.stage = 'Looking up the current llama.cpp build';
  emit();

  const res = await fetch(RELEASES, { headers: { accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(15000) });
  if (!res.ok) throw new Error(`GitHub responded ${res.status}`);
  const releases = await res.json();
  const matcher = assetMatcher();

  for (const release of releases) {
    const asset = (release.assets ?? []).find((a) => matcher.test(a.name));
    if (asset) return { asset, version: release.tag_name, ext: matcher.ext };
  }
  throw new Error('No matching llama.cpp build found for this platform');
}

export async function install({ force = false } = {}) {
  requireAcquisition('The llama.cpp engine');
  if (running) return running;

  running = (async () => {
    try {
      const found = await findRelease();
      state.asset = found.asset.name;
      state.version = found.version;

      if (!force) {
        const existing = findInstalledBinary();
        if (existing) {
          state.status = 'done';
          state.pct = 100;
          state.stage = `Already installed (${found.version})`;
          state.installedPath = existing;
          emit();
          return status();
        }
      }

      mkdirSync(ENGINE_DIR, { recursive: true });
      const archive = join(ENGINE_DIR, found.asset.name);

      state.status = 'downloading';
      state.stage = `Downloading ${found.asset.name}`;
      state.pct = 0;
      emit();

      const dl = await fetch(found.asset.browser_download_url, { redirect: 'follow', signal: AbortSignal.timeout(600_000) });
      if (!dl.ok || !dl.body) throw new Error(`Download failed (${dl.status})`);

      const total = Number(dl.headers.get('content-length')) || found.asset.size || 1;
      const startedAt = Date.now();
      let received = 0;
      let last = 0;

      const progress = new ReadableStream({
        start(controller) {
          const reader = dl.body.getReader();
          void (async () => {
            try {
              for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                received += value.byteLength;
                controller.enqueue(value);
                const now = Date.now();
                if (now - last > 250) {
                  last = now;
                  const seconds = (now - startedAt) / 1000;
                  const mbps = received / 1024 ** 2 / Math.max(seconds, 0.001);
                  state.pct = Math.min(99, (received / total) * 100);
                  state.mbps = mbps;
                  state.stage = `Downloading ${found.asset.name}`;
                  emit();
                }
              }
              controller.close();
            } catch (err) {
              controller.error(err);
            }
          })();
        },
      });

      await pipeline(Readable.fromWeb(progress), createWriteStream(archive));

      state.status = 'extracting';
      state.pct = 100;
      state.stage = 'Unpacking the engine';
      emit();

      await exec(tarBinary(), ['-xf', archive, '-C', ENGINE_DIR], { timeout: 180_000 });
      rmSync(archive, { force: true });

      const binary = findInstalledBinary();
      if (!binary) throw new Error('Engine unpacked but no llama-server binary was found');

      state.status = 'done';
      state.stage = `llama.cpp ${found.version} ready`;
      state.installedPath = binary;
      state.error = null;
      hubRef?.send('log', { level: 'info', message: `Engine installed: ${binary}`, at: nowIso() });
      emit();
      return status();
    } catch (err) {
      state.status = 'error';
      state.error = err.message;
      state.stage = 'Engine install failed';
      emit();
      throw err;
    } finally {
      running = null;
    }
  })();

  return running;
}

export function findInstalledBinary() {
  if (!existsSync(ENGINE_DIR)) return null;
  try {
    for (const entry of readdirSync(ENGINE_DIR)) {
      const dir = join(ENGINE_DIR, entry);
      if (!statSync(dir).isDirectory()) continue;
      const candidate = join(dir, 'llama-server');
      if (existsSync(candidate)) return candidate;
    }
  } catch {
    /* ignore */
  }
  return null;
}
