/**
 * The app's version, from one place. The desktop shell reports its own (apps/desktop/src-tauri/Cargo.toml, which Tauri also uses for
 * the installer and file names) in STRATA_APP_VERSION. Outside the app, in a source checkout, the same file is read; otherwise 'dev'.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const VERSION = /^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?$/;

export function appVersionFrom({ env = process.env, here }) {
  const given = String(env.STRATA_APP_VERSION ?? '').trim();
  if (VERSION.test(given)) return given;
  try {
    const toml = readFileSync(join(here, '..', '..', 'desktop', 'src-tauri', 'Cargo.toml'), 'utf8');
    const found = /^version\s*=\s*"([^"]+)"/m.exec(toml);
    if (found && VERSION.test(found[1])) return found[1];
  } catch { /* not a checkout */ }
  return 'dev';
}

/** The version "Strata" means on screen and in the update check: the app's own, not Strata Control's (see src/lib/app-version.ts). */
export function appVersionToShow(app, control) {
  const real = value => (typeof value === 'string' && value.trim() && value.trim().toLowerCase() !== 'dev' ? value.trim() : null);
  return real(app) ?? real(control) ?? (typeof app === 'string' && app.trim() ? app.trim() : null);
}
