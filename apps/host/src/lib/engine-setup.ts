import { useEffect, useState } from 'react';
import { api } from './client';

/**
 * True while the host is still unpacking the engine that came inside the app (the first start; about a minute, longer on a busy disk).
 * The model list and the setup wait for it, so the page would otherwise sit on "Getting your setup ready…" with no reason given.
 */
export function useEngineUnpacking(active: boolean): boolean {
  const [unpacking, setUnpacking] = useState(false);
  useEffect(() => {
    if (!active) return undefined;
    let live = true;
    const ask = () => api.get<{ engineSetup?: boolean }>('/api/control/status').then(s => { if (live) setUnpacking(s.engineSetup === true); }, () => {});
    void ask();
    const timer = setInterval(() => void ask(), 2000);
    return () => { live = false; clearInterval(timer); };
  }, [active]);
  return unpacking;
}
