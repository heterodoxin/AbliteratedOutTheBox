/**
 * Host bootstrap.
 *
 * `utilityProcess.fork` starts a plain Node service process from inside the app bundle. This tiny
 * CommonJS shim exists because the App Host itself is an ES module (`server/index.mjs`) and the
 * utility-process loader has been observed not to start an `.mjs` entry point from inside a
 * packaged bundle. Loading it dynamically from an ordinary CommonJS entry removes that variable
 * while keeping the host's own module graph unchanged.
 *
 * It intentionally does nothing else: no supervision, no lifecycle, no policy. All of that belongs
 * to the host and to Strata Control.
 */
'use strict';

const { pathToFileURL } = require('node:url');
const { join } = require('node:path');

const entry = join(__dirname, 'index.mjs');

import(pathToFileURL(entry).href).catch((err) => {
  process.stderr.write(`[strata] the App Host failed to load from ${entry}: ${err && err.stack ? err.stack : err}\n`);
  process.exitCode = 1;
});
