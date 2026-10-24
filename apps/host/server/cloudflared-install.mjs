/** Install the official connector into Strata's own data folder, without sudo. */
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { mkdir, mkdtemp, lstat, rename, rm, chmod, writeFile, access } from 'node:fs/promises';
import { constants } from 'node:fs';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { require as requireAcquisition } from './acquisition.mjs';
import { findConnector, skippedMessage } from './cloudflared-find.mjs';

const run = promisify(execFile);
const RELEASES = 'https://api.github.com/repos/cloudflare/cloudflared/releases/latest';
const LIMIT = 128 * 1024 * 1024;

export function connectorAsset(platform, arch) {
  const cpu = { x64: 'amd64', arm64: 'arm64', ia32: '386', arm: 'arm' }[arch];
  if (!cpu || !['darwin', 'linux', 'win32'].includes(platform) || platform === 'win32' && arch === 'arm64') {
    throw new Error('Automatic connector setup is not available for this operating system and processor.');
  }
  return `cloudflared-${platform === 'win32' ? 'windows' : platform}-${cpu}${platform === 'darwin' ? '.tgz' : platform === 'win32' ? '.exe' : ''}`;
}

export function verifyConnector(bytes, digest) {
  if (!/^sha256:[a-f0-9]{64}$/.test(digest ?? '') || `sha256:${createHash('sha256').update(bytes).digest('hex')}` !== digest) {
    throw new Error('The connector download could not be verified. Retry the automatic setup.');
  }
}

/**
 * `componentDirectory` is where an optional, explicitly installed connector component would live (the local
 * component folder). A connector found there is used as it is; one that is not there is never fetched while the app
 * runs (acquisition.mjs), it is a notice.
 */
export function createConnectorInstaller({ directory, componentDirectory = null, platform = process.platform, arch = process.arch, fetchImpl = fetch, env = process.env, home, run: runVersion, usable, search }) {
  const name = platform === 'win32' ? 'cloudflared.exe' : 'cloudflared';
  const binary = join(directory, name);
  let installing = null, last = { found: null, skipped: [] };
  /** Strata's own folder first, then the component folder, PATH and the usual install places (cloudflared-find.mjs). */
  async function look() {
    last = await findConnector({ directory, componentDirectory, platform, env, home, run: runVersion, usable, search });
    return last;
  }
  async function locate() { return (await look()).found?.path ?? null; }
  /** What is in use ({ path, version, source, label }), or null; set by the last look. */
  const connector = () => last.found;
  async function install() {
    const existing = await locate();
    if (existing) return existing;
    const unusable = skippedMessage(last.skipped);
    if (unusable) throw new Error(unusable);
    requireAcquisition('The Cloudflare connector (cloudflared)', 'the optional connector component');
    const name = connectorAsset(platform, arch);
    await mkdir(directory, { recursive: true });
    const stage = await mkdtemp(join(directory, '.install-'));
    const request = url => fetchImpl(url, { signal: AbortSignal.timeout(120000), headers: { 'user-agent': 'Strata-connector-setup', accept: 'application/vnd.github+json' } });
    try {
      const metadata = await request(RELEASES);
      if (!metadata.ok) throw new Error('Cloudflare’s download service is unavailable. Try again in a moment.');
      const release = await metadata.json();
      const asset = release.assets?.find(a => a.name === name);
      if (!asset || !asset.browser_download_url?.startsWith('https://github.com/cloudflare/cloudflared/releases/download/')) throw new Error('No official connector download was found for this computer.');
      if (asset.size > LIMIT) throw new Error('The connector download is unexpectedly large.');
      const response = await request(asset.browser_download_url);
      if (!response.ok || !response.body) throw new Error('The connector could not be downloaded. Check your connection and retry.');
      const chunks = []; let size = 0;
      for await (const chunk of response.body) {
        size += chunk.byteLength;
        if (size > LIMIT) throw new Error('The connector download is unexpectedly large.');
        chunks.push(chunk);
      }
      const bytes = Buffer.concat(chunks);
      verifyConnector(bytes, asset.digest);
      const download = join(stage, name);
      await writeFile(download, bytes, { mode: 0o600 });
      let candidate = download;
      if (platform === 'darwin') {
        const { stdout } = await run('tar', ['-tzf', download], { timeout: 10000 });
        const entry = stdout.split(/\r?\n/).find(p => p === 'cloudflared' || p === './cloudflared');
        if (!entry) throw new Error('The connector package did not contain the expected executable.');
        await run('tar', ['-xzf', download, '-C', stage, entry], { timeout: 10000 });
        candidate = join(stage, 'cloudflared');
      }
      if (!(await lstat(candidate)).isFile()) throw new Error('The connector package contained an invalid executable.');
      await chmod(candidate, 0o700);
      await run(candidate, ['--version'], { timeout: 10000, windowsHide: true });
      await rename(candidate, binary);
      return binary;
    } finally { await rm(stage, { recursive: true, force: true }); }
  }
  return { locate, connector, async inspect() { await look(); return last; }, ensure() {
    if (!installing) installing = install().finally(() => { installing = null; });
    return installing;
  } };
}
