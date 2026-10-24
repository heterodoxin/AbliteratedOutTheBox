import { OFFLINE_MESSAGE, looksOffline, plainUpdateError } from './update-messages.mjs';

/** Read-only release checks. Installing/rolling back needs a managed installer. */
const RELEASES = 'https://api.github.com/repos/Niko1221/Strata/releases?per_page=30';
const RELEASE_PREFIX = 'https://github.com/Niko1221/Strata/releases/';

function versionParts(value) {
  const match = /^v?(\d+)\.(\d+)\.(\d+)(?:\+[\w.-]+)?$/.exec(value ?? '');
  return match ? match.slice(1).map(Number) : null;
}
function compare(a, b) {
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i] ? 1 : -1;
  return 0;
}

export async function checkUpdates({ appVersion, engineVersion }, fetcher = fetch) {
  let strata;
  try {
    const response = await fetcher(RELEASES, { headers: { accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error(response.status === 403 || response.status === 429 ? 'The release service is busy. Try again later.' : 'Could not reach the Strata release feed. Try again.');
    const releases = await response.json();
    if (!Array.isArray(releases)) throw new Error('The release service returned an invalid response.');
    const candidates = releases.filter(r => !r.draft && !r.prerelease && versionParts(r.tag_name));
    candidates.sort((a, b) => compare(versionParts(b.tag_name), versionParts(a.tag_name)));
    const latest = candidates[0];
    const installed = versionParts(appVersion);
    if (!latest) strata = { status: 'no-releases', message: 'No stable Strata release is published in this feed yet.' };
    else if (!installed) strata = { status: 'unavailable', message: 'This installation has no comparable version number.' };
    else {
      const newer = compare(versionParts(latest.tag_name), installed) > 0;
      // Release content is untrusted. Only link back to the known repository.
      const releaseUrl = typeof latest.html_url === 'string' && latest.html_url.startsWith(RELEASE_PREFIX) ? latest.html_url : undefined;
      strata = { status: newer ? 'available' : 'current', latestVersion: latest.tag_name, releaseUrl, message: newer ? `Strata ${latest.tag_name} is available.` : 'Strata is up to date.' };
    }
  } catch (error) {
    strata = { status: 'error', message: looksOffline(error) ? OFFLINE_MESSAGE : error.name === 'TimeoutError' ? 'The release check timed out. Try again.' : error.message ?? 'Could not check for updates.' };
  }
  return {
    checkedAt: new Date().toISOString(),
    strata: { installedVersion: appVersion, ...strata, canInstall: false },
    // The existing binary downloader has no tested-release compatibility feed
    // or trustworthy installed build ID. Upstream's latest is not an update verdict.
    engine: { installedVersion: engineVersion, status: 'unavailable', canInstall: false, message: 'Engine update checks are not available in this build.' },
  };
}

/**
 * The packaged app's update check: the signed feed, read by Control (`strata update --check`), plus the shell's own verdict
 * on the app. Same result shape as `checkUpdates`, so the Settings screen needs no change. This is the only place the host
 * reaches the feed, and only when the user presses "Check for updates".
 */
export async function checkComponentUpdates({ control, appUpdater = null, appVersion = null }) {
  const checkedAt = new Date().toISOString();
  let report;
  try {
    report = await control.updateCheck({ appVersion });
  } catch (error) {
    const plain = plainUpdateError(error, 'Strata');
    const message = /update server|aren't available|No internet/.test(plain.headline) ? plain.message : `Couldn't check for updates. Try again in a moment.`;
    return { checkedAt, strata: { installedVersion: appVersion ?? control.version ?? null, status: 'error', message, details: plain.details, canInstall: false }, engine: { installedVersion: null, status: 'error', message, details: plain.details, canInstall: false } };
  }
  const mib = bytes => `${Math.round((bytes ?? 0) / (1 << 20))} MB`;
  const engine = report.engine ?? {};
  const installedEngine = engine.installed?.version ?? null;
  const offered = engine.offered?.version ?? null;
  const engineView = { installedVersion: installedEngine, latestVersion: offered ?? undefined, canInstall: engine.status === 'available' || engine.status === 'install' };
  if (engine.status === 'available') Object.assign(engineView, { status: 'available', message: `A new engine is available (${mib(engine.download_bytes)}).` });
  else if (engine.status === 'install') Object.assign(engineView, { status: 'available', message: `This PC needs an engine to run models. Press Update engine to download it (${mib(engine.download_bytes)}).` });
  else if (engine.status === 'current' || engine.status === 'failed-before' || engine.status === 'skipped') Object.assign(engineView, { status: 'current', message: 'The engine is up to date.', details: engine.note });
  else if (engine.status === 'wrong-make') Object.assign(engineView, { status: 'unavailable', message: engine.note ?? 'Nothing was changed: the update was for another make of graphics card.', details: (engine.reasons ?? []).join('\n') || undefined });
  else if (engine.status === 'not-in-feed') Object.assign(engineView, { status: 'current', message: 'No engine update is listed right now.' });
  else Object.assign(engineView, { status: 'unavailable', message: "There isn't an engine for this PC's graphics card yet.", details: (engine.reasons ?? []).join('\n') || undefined });
  const control_ = report.control ?? {};
  const app = report.app ?? {};
  let appNew = app.status === 'available' ? app.offered : null;
  let appNotes = app.notes ?? '';
  let appCannot = null;   // browser mode: the app window's updater is not running, so the app itself cannot be updated from here
  if (appUpdater) {
    try {
      const verdict = await appUpdater.check();
      appNew = verdict.available ? verdict.version : null;
      appNotes = verdict.notes ?? appNotes;
    } catch (error) {
      if (error?.code === 'app-update-unavailable' && appUpdater.unavailable) appCannot = appUpdater.unavailable;   // otherwise the feed's own view stands
    }
  }
  const controlNew = control_.status === 'available' ? control_.offered : null;
  const strata = { installedVersion: appVersion ?? control_.installed ?? control.version ?? null, canInstall: !!((appNew && !appCannot) || controlNew) };
  if (appNew || controlNew) Object.assign(strata, { status: 'available', latestVersion: appNew ?? controlNew, message: `A new version of Strata is available.${appNotes ? ` ${appNotes}` : ''}${appNew && appCannot ? ` ${appCannot}` : ''}` });
  else Object.assign(strata, { status: 'current', message: 'Strata is up to date.' });
  return { checkedAt, feed: report.feed, strata, engine: engineView, serving: report.serving ?? [] };
}
