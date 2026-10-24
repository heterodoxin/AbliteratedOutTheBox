// The host's own HTTPS downloads (engine releases from GitHub, web search, a cloud model's API) check the server against the operating
// system's certificates as well as Node's bundled list (#3722). Node carries its own list of roots; Chrome and curl.exe use the Windows
// store, which also holds the roots an antivirus's HTTPS scanning, a company proxy or an OS preview build adds. Nothing is turned off:
// the system roots are ADDED to the bundled ones, and a connection that still fails keeps failing.
//
// Node 24.5 and newer: tls.getCACertificates('system') and tls.setDefaultCACertificates. An older Node (the app bundles 24.x) does nothing here;
// the agent process gets NODE_USE_SYSTEM_CA=1 (harness-runtime.mjs) for the same reason.
import tlsModule from 'node:tls';

/** @returns {'system'|'bundled'} what the process trusts after the call */
export function useSystemCertificates(tls = tlsModule, env = process.env) {
  if (env.STRATA_NO_SYSTEM_CA === '1') return 'bundled';          // a way back for a PC where reading the store misbehaves; not a way to skip checks
  try {
    if (typeof tls.getCACertificates !== 'function' || typeof tls.setDefaultCACertificates !== 'function') return 'bundled';
    const system = tls.getCACertificates('system');
    if (!Array.isArray(system) || !system.length) return 'bundled';
    tls.setDefaultCACertificates([...new Set([...tls.getCACertificates('bundled'), ...system])]);
    return 'system';
  } catch {
    return 'bundled';
  }
}

export const trust = useSystemCertificates();
