/**
 * The environment for programs that are NOT part of Strata: the folder chooser (zenity, kdialog, gdbus), xdg-open and the browser it
 * starts, and the shell and tools the coding assistant runs for the person.
 *
 * Why: the Linux AppImage starts Strata with its own libraries and settings in the environment (LD_LIBRARY_PATH, GTK_*, GIO_*,
 * GSETTINGS_SCHEMA_DIR, XDG_DATA_DIRS, PYTHONHOME, ...). A system program that inherits them loads Strata's copies of libssl, GTK and
 * GLib instead of its own and dies (tester report 2026-10-17: zenity "libssl.so.3: version OPENSSL_3.2.0 not found", so the folder
 * picker did nothing). Programs Strata ships itself (its Python, Node and engine) keep the environment they were built for; everything
 * else gets the environment the AppImage was started from.
 *
 * The AppImage's launcher records that original environment in STRATA_ORIGINAL_ENV (repack-appimage.sh adds the hook). An older
 * AppImage without the record is cleaned instead: every variable that points into the AppImage's own folder is dropped, and its
 * folders are removed from the path lists.
 */
const RECORD = 'STRATA_ORIGINAL_ENV';

/** Variables the AppImage tooling sets only for itself; never meant for another program. */
const APPIMAGE_ONLY = [
  'APPDIR', 'APPIMAGE', 'ARGV0', 'OWD', 'APPIMAGE_SILENT_INSTALL', 'APPIMAGE_EXTRACT_AND_RUN',
  'GTK_DATA_PREFIX', 'GTK_EXE_PREFIX', 'GTK_PATH', 'GTK_IM_MODULE_FILE', 'GTK_THEME', 'GDK_PIXBUF_MODULE_FILE', 'GDK_PIXBUF_MODULEDIR',
  'GSETTINGS_SCHEMA_DIR', 'GI_TYPELIB_PATH', 'GIO_MODULE_DIR', 'GIO_EXTRA_MODULES',
  'GST_REGISTRY_REUSE_PLUGIN_SCANNER', 'GST_PLUGIN_SYSTEM_PATH_1_0', 'GST_PLUGIN_PATH_1_0', 'GST_PLUGIN_SCANNER_1_0', 'GST_PTP_HELPER_1_0',
  'WEBKIT_EXEC_PATH', 'WEBKIT_INJECTED_BUNDLE_PATH', 'STRATA_ORIGINAL_ENV', 'STRATA_APPDIR',
];
/** Lists of folders: the AppImage prepends its own; only its entries are removed. */
const PATH_LISTS = ['PATH', 'XDG_DATA_DIRS', 'XDG_CONFIG_DIRS', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'PERLLIB', 'PERL5LIB', 'QT_PLUGIN_PATH', 'QML2_IMPORT_PATH', 'GST_PLUGIN_PATH', 'GTK_PATH', 'GIO_MODULE_DIR'];

/** The environment the program was started from, decoded from the launcher's record, or null when there is none. */
export function recordedOriginal(env = process.env) {
  const raw = env[RECORD];
  if (!raw) return null;
  try {
    const text = Buffer.from(raw, 'base64').toString('utf8');
    const out = {};
    for (const entry of text.split('\0')) {
      const at = entry.indexOf('=');
      if (at > 0) out[entry.slice(0, at)] = entry.slice(at + 1);
    }
    return Object.keys(out).length ? out : null;
  } catch { return null; }
}

/** `env` with everything that belongs to a running AppImage removed (no record needed). */
export function withoutAppImage(env = process.env) {
  const dir = env.APPDIR || '';
  const out = { ...env };
  const inside = value => !!dir && (value === dir || value.startsWith(`${dir}/`) || value.startsWith(`${dir}//`));
  for (const name of APPIMAGE_ONLY) delete out[name];
  if (dir) {
    for (const name of Object.keys(out)) if (/^(GTK|GDK|GIO|GST|GI_|GSETTINGS|WEBKIT)/.test(name) && typeof out[name] === 'string' && out[name].includes(dir)) delete out[name];
    for (const name of PATH_LISTS) {
      if (typeof out[name] !== 'string') continue;
      const kept = out[name].split(':').filter(part => part && !inside(part));
      if (kept.length) out[name] = kept.join(':'); else delete out[name];
    }
  }
  return out;
}

/** True when this process was started from an AppImage. */
export const insideAppImage = (env = process.env) => !!(env.APPIMAGE || env.APPDIR);

/**
 * The environment for an external program (see the top of this file). Outside an AppImage it is `env` unchanged.
 * `extra` is added last (a variable the caller really means to pass).
 */
export function externalEnvironment(env = process.env, extra = {}) {
  if (!insideAppImage(env) && !env[RECORD]) return { ...env, ...extra };
  const original = recordedOriginal(env);
  return { ...(original ?? withoutAppImage(env)), ...extra };
}
