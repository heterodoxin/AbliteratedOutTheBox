#!/usr/bin/env node
/**
 * Runs every recorded flow and reports which ones fail.
 *
 *   node scripts/flows.mjs            all flows
 *   node scripts/flows.mjs chat rail  only these
 *
 * Why this exists: two flows silently stopped testing anything — one waited for
 * a button the completed setup screen correctly no longer shows, the other kept
 * a CSS class that had been renamed. Both looked like regressions when they were
 * finally run, and neither had been exercised in weeks. A suite nobody runs is
 * not a suite, so the whole set is one command now.
 */
import { execFile } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { promisify } from 'node:util';

const run = promisify(execFile);
const ROOT = new URL('..', import.meta.url).pathname;

const source = readFileSync(`${ROOT}scripts/flow.mjs`, 'utf8');
const all = [...new Set([...source.matchAll(/flow === '([a-z]+)'/g)].map((m) => m[1]))].sort();
const wanted = process.argv.slice(2).filter((a) => !a.startsWith('-'));
const flows = wanted.length ? wanted.filter((f) => all.includes(f)) : all;

if (wanted.length && flows.length !== wanted.length) {
  const unknown = wanted.filter((f) => !all.includes(f));
  console.error(`Unknown flow(s): ${unknown.join(', ')}`);
  console.error(`Available: ${all.join(', ')}`);
  process.exit(2);
}

const failed = [];
for (const flow of flows) {
  process.stdout.write(`${flow.padEnd(16)} `);
  try {
    const { stdout } = await run('node', ['scripts/flow.mjs', flow], { cwd: ROOT, timeout: 180_000 });
    const last = stdout.trim().split('\n').filter(Boolean).pop() ?? '';
    console.log(last.slice(0, 78));
  } catch (err) {
    const out = `${err.stdout ?? ''}${err.stderr ?? ''}`;
    const reason = /Timeout .*exceeded/.test(out)
      ? 'timed out waiting for something that is not there'
      : (out.trim().split('\n').filter(Boolean).pop() ?? 'failed').slice(0, 78);
    console.log(`FAIL — ${reason}`);
    failed.push(flow);
  }
}

console.log(
  failed.length
    ? `\nflows: ${failed.length} of ${flows.length} failed — ${failed.join(', ')}`
    : `\nflows: all ${flows.length} passed`,
);
process.exit(failed.length ? 1 : 0);
