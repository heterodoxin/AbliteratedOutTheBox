/** Ports the host listens on: a free one for the helpers, and a plain sentence when the main one cannot be taken. */
import net from 'node:net';

/** Why a listen failed, in words a person can act on (Windows keeps whole ranges for Hyper-V, WSL and Docker). */
export function listenProblem(error, port, host = '127.0.0.1') {
  const code = error?.code;
  if (code === 'EADDRINUSE') return `Port ${port} is already in use. Strata may already be running; close it or the other program, or start Strata with another port (STRATA_PORT).`;
  if (code === 'EACCES') return `Windows or a firewall does not let Strata use port ${port}. Hyper-V, WSL and Docker keep ranges of ports for themselves (see "netsh int ipv4 show excludedportrange protocol=tcp"). Start Strata with another port (STRATA_PORT), for example 8801.`;
  if (code === 'EADDRNOTAVAIL') return `${host} is not an address of this computer, so Strata cannot listen on it. Check STRATA_BIND.`;
  return `Strata cannot listen on ${host}:${port} (${error?.message ?? code}).`;
}

function tryListen(port, host) {
  return new Promise(resolve => {
    const probe = net.createServer();
    probe.once('error', () => resolve(null));
    probe.listen(port, host, () => { const got = probe.address().port; probe.close(() => resolve(got)); });
  });
}

/** `preferred` when it can be taken, else the next free port above it (up to `tries` of them), else any free port. Never throws. */
export async function findFreePort(preferred, host = '127.0.0.1', { skip = [], tries = 20 } = {}) {
  for (let offset = 0; offset < tries; offset++) {
    const port = preferred + offset;
    if (port > 65535 || skip.includes(port)) continue;
    const got = await tryListen(port, host);
    if (got) return got;
  }
  return (await tryListen(0, host)) ?? preferred;
}
