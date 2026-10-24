import { useEffect, useState } from 'react';
import { useStore } from './store';
import { useNative } from './native';
import { appVersionLabel } from './app-version';
import { readUpdateCache, saveUpdateCache, type InstalledVersions } from './update-cache';
import type { UpdateCheckResult } from './types';

/** The installed versions the cached check belongs to. Same values the Settings version rows show. */
export function useInstalledVersions(): InstalledVersions {
  const native = useNative();
  const engineVersion = useStore(s => s.engine?.engine.version);
  const appVersion = appVersionLabel(useStore(s => s.setup?.app.version), native?.version);
  return { appVersion: appVersion ?? null, engineVersion: (native?.available ? native.installed?.engine?.version : engineVersion) ?? null, controlVersion: native?.version ?? null };
}

/** The one place the app asks the host to check for updates. Called only from a button press. */
export async function fetchUpdateCheck(signal?: AbortSignal): Promise<UpdateCheckResult> {
  const response = await fetch('/api/updates/check', { signal });
  if (!response.ok) throw new Error('Could not check for updates. Try again.');
  const data = await response.json() as UpdateCheckResult;
  if (!data.strata?.status || !data.engine?.status || !data.checkedAt) throw new Error('The update service returned an incomplete result.');
  return data;
}

const EVENT = 'strata:update-check';

/** Remember a finished check and tell every Settings panel that shows it. */
export function publishUpdateCheck(result: UpdateCheckResult, now: InstalledVersions) {
  saveUpdateCache(result, now);
  try { window.dispatchEvent(new Event(EVENT)); } catch { /* No window, nothing to refresh. */ }
}

/** The last known check result, read from this computer. Never touches the network. */
export function useLastUpdateCheck(now: InstalledVersions): UpdateCheckResult | null {
  const [, tick] = useState(0);
  useEffect(() => {
    const refresh = () => tick(n => n + 1);
    window.addEventListener(EVENT, refresh);
    return () => window.removeEventListener(EVENT, refresh);
  }, []);
  return readUpdateCache(now);
}
