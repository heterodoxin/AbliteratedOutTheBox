/** A supervised Cloudflare connector with a private, inference-only origin. */
import { spawn } from 'node:child_process';
import { probe, which } from './util.mjs';
import { classifyRunning, describeConnector, otherTunnelMessage, ownTunnelMessage } from './cloudflared-find.mjs';

/** A tunnel without a key is allowed (owner decision, 0.2.2); because the link is public the warning is stronger than the local one. */
export const NO_KEY_TUNNEL_WARNING = 'No API key is set, so anyone who finds this link can use your model from the internet. Create an API key to protect it.';
export function tunnelHostname(value) {
  const hostname = String(value ?? '').trim().toLowerCase();
  if (hostname.length > 253 || !/^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$/.test(hostname)) throw new Error('Enter a hostname such as ai.example.com, without https:// or a path.');
  return hostname;
}
export function createCloudflare({ origin, originUrl, hasKey, changed = () => {}, launch = spawn, locate = async () => process.platform === 'win32' ? (await probe('where.exe', ['cloudflared'])).split(/\r?\n/)[0] || null : which('cloudflared'), prepareConnector = null, connectorInfo = () => null, findRunning = async () => ({ running: false, kind: null, pids: [] }), timeoutMs = 45000 }) {
  let binary = null, other = null, child = null, timer = null, busy = false, generation = 0;
  let value = { status: 'idle', mode: null, url: null, error: null };
  const state = () => {
    const info = binary ? connectorInfo() : null;
    return { ...value, warning: value.status === 'connected' && !hasKey() ? NO_KEY_TUNNEL_WARNING : null, installed: !!binary, originUrl, connector: info ? { path: info.path, version: info.version, source: info.source, label: info.label, summary: describeConnector(info) } : null, external: other?.running ? externalState() : null };
  };
  const update = patch => { value = { ...value, ...patch }; changed(state()); };
  function externalState(token = '') {
    const { own, others } = classifyRunning(other, { originUrl, token });
    return { kind: other.kind, others, own, message: own.length ? ownTunnelMessage(own) : otherTunnelMessage(others) };
  }
  async function lookForOther() { try { other = await findRunning({ exclude: [child?.pid] }); } catch { other = null; } return other; }
  async function inspect() { binary = await locate(); await lookForOther(); changed(state()); return state(); }
  async function prepare() {
    if (busy) throw new Error('Connector setup is already in progress.');
    busy = true;
    try {
      binary = await locate();
      if (!binary && prepareConnector) binary = await prepareConnector();
      if (!binary) throw new Error('The connector could not be prepared automatically.');
      update({ error: null });
      return state();
    } finally { busy = false; }
  }
  async function terminate() {
    clearTimeout(timer); timer = null;
    const processRef = child; child = null;
    if (processRef) {
      await new Promise(resolve => {
        const fallback = setTimeout(() => { processRef.kill('SIGKILL'); resolve(); }, 1500);
        processRef.once('exit', () => { clearTimeout(fallback); resolve(); });
        processRef.kill('SIGTERM');
      });
    }
    await origin.stop();
  }
  async function fail(message, id) {
    if (id !== generation) return;
    ++generation; busy = true; update({ status: 'stopping', error: message, url: null });
    try { await terminate(); } finally { busy = false; update({ status: 'error' }); }
  }
  async function start(body) {
    if (busy || ['starting', 'connected', 'stopping'].includes(value.status)) throw new Error('Stop the current tunnel before starting another.');
    if (!['quick', 'domain'].includes(body.mode)) throw new Error('Choose a temporary URL or your own domain.');
    const hostname = body.mode === 'domain' ? tunnelHostname(body.hostname) : null;
    const token = body.mode === 'domain' ? String(body.token ?? '').trim() : '';
    if (body.mode === 'domain' && (!/^[A-Za-z0-9+/=_-]{20,8192}$/.test(token))) throw new Error('Paste the tunnel token from Cloudflare, not the full installation command.');
    busy = true; const id = ++generation;
    update({ status: 'starting', mode: body.mode, url: null, error: null });
    try {
      binary = await locate();
      if (!binary && prepareConnector) binary = await prepareConnector();
      if (!binary) throw new Error('The connector could not be prepared automatically. Retry connector setup.');
      // Strata's own tunnel (same port or same token) is never doubled up. Other cloudflared processes serve other tunnels and are left alone.
      if ((await lookForOther())?.running) { const { own } = classifyRunning(other, { originUrl, token }); if (own.length) throw new Error(ownTunnelMessage(own)); }
      await origin.start();
      // Tokens never appear in command arguments, logs, saved settings, or API responses.
      const env = { ...process.env };
      for (const key of Object.keys(env)) if (key.startsWith('TUNNEL_')) delete env[key];
      if (token) env.TUNNEL_TOKEN = token;
      const args = ['tunnel', '--no-autoupdate', '--loglevel', 'info', '--grace-period', '1s', ...(hostname ? ['run'] : ['--url', originUrl])];
      child = launch(binary, args, { env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
      let tail = '', publicUrl = hostname ? `https://${hostname}` : null, registered = false;
      const read = chunk => {
        if (generation !== id) return;
        tail = (tail + chunk.toString()).slice(-16384);
        const match = tail.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com\b/);
        if (!hostname && match) publicUrl = match[0];
        if (/Registered tunnel connection/i.test(tail)) registered = true;
        if (publicUrl && registered && value.status !== 'connected') { clearTimeout(timer); update({ status: 'connected', url: `${publicUrl}/v1` }); }
      };
      child.stdout?.on('data', read); child.stderr?.on('data', read);
      child.once('error', () => { void fail('Cloudflared could not start. Check the installation and try again.', id); });
      child.once('exit', () => { if (generation === id) { child = null; void fail('Cloudflared stopped. Check your network, token and Cloudflare configuration, then reconnect.', id); } });
      timer = setTimeout(() => { void fail('Cloudflare did not connect in time. Check your internet connection and tunnel setup, then try again.', id); }, timeoutMs);
      timer.unref?.();
      return state();
    } catch (e) {
      ++generation; await terminate(); update({ status: 'error', url: null, error: e.message }); throw e;
    } finally { busy = false; }
  }
  async function stop() {
    if (busy) throw new Error('The tunnel is updating. Try again in a moment.');
    busy = true; ++generation; update({ status: 'stopping', url: null, error: null });
    try { await terminate(); update({ status: 'idle', mode: null }); return state(); } finally { busy = false; }
  }
  return { state, inspect, prepare, start, stop };
}
