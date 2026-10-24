/**
 * The line between running the app and acquiring software, on the host side. It mirrors
 * strata_control/acquisition.py: starting a model, serving, and opening the app never download or install
 * an engine, a connector, a package or a compiler. Anything that would (the llama.cpp fetch, the cloudflared
 * fetch) calls `require()` first, and `require()` refuses where acquiring is forbidden.
 *
 * Forbidden means: STRATA_NO_ACQUIRE is 1/true/yes/on, or this is the packaged app (STRATA_PACKAGED=1).
 * The Tauri shell sets both for the host. Models the user chooses to download are a different, explicit action
 * with its own progress and consent and do not go through here.
 */

const TRUE = new Set(['1', 'true', 'yes', 'on']);

export class AcquisitionRefused extends Error {
  constructor(what, command = 'the Terminal command `strata install`') {
    super(`${what} is not installed here, and the app never downloads it while it runs. Install it from ${command}.`);
    this.name = 'AcquisitionRefused';
    this.code = 'acquisition_refused';
    this.status = 409;
    this.what = what;
  }
}

export function forbidden(env = process.env) {
  return TRUE.has(String(env.STRATA_NO_ACQUIRE ?? '').trim().toLowerCase()) || env.STRATA_PACKAGED === '1';
}

/** Call before any download or install. Throws AcquisitionRefused where acquiring is forbidden. */
export function require(what, command, env = process.env) {
  if (forbidden(env)) throw new AcquisitionRefused(what, command);
}
