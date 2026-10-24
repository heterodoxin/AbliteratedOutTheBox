/** A separate, key-protected listener for inference only. Never serves the UI or host APIs. */
import http from 'node:http';
import os from 'node:os';
import { apiPath } from './api-paths.mjs';
import { listenProblem } from './ports.mjs';
/** Sharing without a key is allowed (owner decision, 0.2.2): the person is told once, in plain words, and the app keeps offering to make a key. */
export const NO_KEY_WARNING = 'No API key is set, so any device on this network can use your model. Create an API key to protect it.';
export function createLocalShare({ handle, hasKey, changed = () => {}, host = '0.0.0.0', port = 8792 }) {
  let listener = null, active = false;
  const address = () => Object.values(os.networkInterfaces()).flat().find(n => n?.family === 'IPv4' && !n.internal)?.address ?? '127.0.0.1';
  const state = () => ({ active, url: active ? `http://${host === '127.0.0.1' ? host : address()}:${listener.address().port}/v1` : null, warning: active && !hasKey() ? NO_KEY_WARNING : null });
  async function start() {
    if (active) return state();
    if (listener) throw new Error('Sharing is already starting.');
    listener = http.createServer(async (req, res) => {
      const path = apiPath(new URL(req.url ?? '/', 'http://localhost').pathname);
      if (!['/v1/models','/v1/chat/completions','/v1/messages'].includes(path)) { res.writeHead(404); res.end('Not found'); return; }
      try { await handle(req, res, path, req.method ?? 'GET'); }
      catch { if (!res.headersSent) res.writeHead(500); res.end('Request failed'); }
    });
    try { await new Promise((resolve,reject) => { listener.once('error',reject); listener.listen(port,host,resolve); }); }
    catch(e) { listener.close(); listener=null; throw new Error(listenProblem(e, port, host)); }
    active=true; changed(); return state();
  }
  async function stop() {
    if (!listener) return state();
    const current=listener; listener=null; active=false;
    current.closeAllConnections(); await new Promise(resolve=>current.close(resolve)); changed(); return state();
  }
  return { state, start, stop };
}
