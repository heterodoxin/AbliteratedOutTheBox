/**
 * Plain words for the app. Whatever the updater says (HTTP codes, hash names, exit codes, file names) stays in `details`, which the UI
 * shows behind the existing "Details" disclosure; the main text says what happened and the one next step (`next`). The Terminal keeps
 * the precise messages: this module is only used on the app side.
 */

export const PHASE_TEXT = {
  feed: 'Looking for updates…',
  download: 'Downloading…',
  verify: 'Checking the download…',
  test: 'Making sure it works…',
  switch: 'Finishing up…',
  done: 'Done.',
};

/** True when the failure looks like a computer with no internet (no DNS, no route, refused), not a slow or broken server. */
export function looksOffline(error) {
  const text = [error?.message, error?.cause?.message, error?.cause?.code, error?.code, error].filter(x => typeof x === 'string').join(' ');
  return /ENOTFOUND|EAI_AGAIN|ENETUNREACH|ENETDOWN|EHOSTUNREACH|ECONNREFUSED|getaddrinfo|fetch failed|network is unreachable|name or service not known|temporary failure in name resolution|nodename nor servname|No address associated|URLError/i.test(text);
}
export const OFFLINE_MESSAGE = "No internet connection. Try again when you're online.";

/** { message, details, next } for an error thrown by Control's update calls (or the shell bridge). `subject` is "engine" or "Strata". */
export function plainUpdateError(error, subject = 'engine') {
  const raw = String(error?.message ?? error ?? '');
  const code = error?.code ?? '';
  const exit = error?.exitCode ?? null;
  const button = `Update ${subject === 'engine' ? 'engine' : 'Strata'}`;
  const keep = 'Your current version still works.';
  let message = `The update didn't finish and nothing was changed. ${keep}`;
  let next = `Press ${button} in Settings to try again.`;
  if (code === 'engine_serving' || exit === 7 || /is serving/i.test(raw)) {
    message = 'A model is running, so nothing was changed.';
    next = `Unload the model in Models, then press ${button} in Settings.`;
  } else if (/another update is already running/i.test(raw)) {
    message = 'An update is already running.';
    next = `Wait a moment, then press ${button} in Settings.`;
  } else if (/test start|did not start correctly|did not answer a one-token|did not become ready/i.test(raw)) {
    message = `The new ${subject} didn't start correctly, so nothing was changed. ${keep}`;
  } else if (looksOffline(error) || looksOffline(raw)) {
    message = 'No internet connection.';
    next = "Try again when you're online.";
  } else if (/could not be read|interrupted|timed out|timeout|ECONN|ENOTFOUND|fetch failed|could not reach|server answered|not answer within/i.test(raw)) {
    message = "Couldn't reach the update server.";
    next = `Check your internet connection, then press ${button} in Settings.`;
  } else if (/free space|not enough/i.test(raw)) {
    message = 'There is not enough free disk space for the update.';
    next = `Free some disk space, then press ${button} in Settings.`;
  } else if (/no engine for this PC|has no engine|no package/i.test(raw)) {
    message = "There isn't an engine for this PC's graphics card yet.";
    next = 'Check for updates again later.';
  } else if (/trusts no update key/i.test(raw)) {
    message = "Updates aren't available in this version of Strata yet.";
    next = 'They will be after the next version.';
  } else if (/signature|SHA-256|does not match|not trusted|expired|older than|replayed|integrity|refused/i.test(raw) || exit === 5) {
    message = `The update didn't pass its safety check, so it was not installed. ${keep}`;
    next = 'Try again later.';
  }
  return { message: `${message} ${next}`.trim(), headline: message, next, details: raw || undefined };
}

/** An Error carrying the plain message and, behind it, the technical reason. */
export function plainError(error, subject) {
  const { message, details } = plainUpdateError(error, subject);
  return Object.assign(new Error(message), { details, code: error?.code });
}
