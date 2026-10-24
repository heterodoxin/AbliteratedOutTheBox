/**
 * Preloaded into a host process (`node --import`) to make the network "unplugged": every fetch to an address that is not this
 * computer is written to the file in STRATA_NET_LOG (one URL per line) and fails the way a machine with no internet does
 * (TypeError: fetch failed). Loopback calls pass through, so the host's own engine and UI traffic still works.
 */
import { appendFileSync } from 'node:fs';

const real = globalThis.fetch;
const log = process.env.STRATA_NET_LOG;
const loopback = host => host === '127.0.0.1' || host === 'localhost' || host === '[::1]' || host === '::1';

globalThis.fetch = function offlineFetch(input, init) {
  let url;
  try { url = new URL(typeof input === 'string' ? input : input?.url ?? String(input)); } catch { return real(input, init); }
  if (loopback(url.hostname)) return real(input, init);
  if (log) appendFileSync(log, `${url.href}\n`);
  const error = new TypeError('fetch failed');
  error.cause = Object.assign(new Error(`getaddrinfo ENOTFOUND ${url.hostname}`), { code: 'ENOTFOUND' });
  return Promise.reject(error);
};
