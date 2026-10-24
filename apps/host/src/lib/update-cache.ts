import type { UpdateCheckResult } from './types';

/**
 * The last update check, kept on this computer so Settings can show it without touching the network. A check runs only when the
 * person presses "Check for updates" (or "Update engine" / "Update Strata"). The result is kept together with the versions it was
 * for, and it is ignored once the installed version changes (an update, a roll back), so an old "Update available" never lingers.
 */
const KEY = 'strata.update-check.v1';

interface Stored { result: UpdateCheckResult; appVersion: string | null; engineVersion: string | null; controlVersion?: string | null }
export interface InstalledVersions { appVersion?: string | null; engineVersion?: string | null; controlVersion?: string | null }

const same = (a: string | null | undefined, b: string | null | undefined) => (a ?? null) === (b ?? null);

export function readUpdateCache(now: InstalledVersions): UpdateCheckResult | null {
  try {
    const stored = JSON.parse(localStorage.getItem(KEY) ?? 'null') as Stored | null;
    if (!stored?.result?.strata?.status || !stored.result.engine?.status || !stored.result.checkedAt) return null;
    // Still loading (no app version yet): show nothing rather than a result that may belong to the version before an update.
    if (!now.appVersion) return null;
    if (!same(stored.appVersion, now.appVersion) || !same(stored.engineVersion, now.engineVersion) || !same(stored.controlVersion, now.controlVersion)) return null;
    return stored.result;
  } catch { return null; }
}

export function saveUpdateCache(result: UpdateCheckResult, now: InstalledVersions) {
  try { localStorage.setItem(KEY, JSON.stringify({ result, appVersion: now.appVersion ?? null, engineVersion: now.engineVersion ?? null, controlVersion: now.controlVersion ?? null } satisfies Stored)); } catch { /* Not remembering a result is harmless. */ }
}

export function clearUpdateCache() {
  try { localStorage.removeItem(KEY); } catch { /* Nothing to clear. */ }
}
