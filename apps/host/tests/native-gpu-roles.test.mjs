/** GPU roles in the host: a load writes the roles and the legacy mirror (Main last), a file that names roles keeps them, every card Off is allowed. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

const load = async (f, args) => { const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs', ...args }).id); return done; };
const written = (f, done) => JSON.parse(readFileSync(f.control.resolveInstance(done.result.instanceId).configPath, 'utf8'));

test('roles chosen in the screen: Helpers first, Main LAST, Off left out, and "gpu_roles" beside the mirror', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const done = await load(f, { gpuRoles: { 0: 'main', 1: 'helper' } });
    assert.equal(done.state, 'done', done.error);
    const cfg = written(f, done);
    assert.deepEqual(cfg.gpu, [1, 0], 'card 0 is Main, so it is the last stage');
    assert.equal(cfg.gpu_order, 'as_given');
    assert.deepEqual(cfg.gpu_roles, { 0: 'main', 1: 'helper' });
    assert.equal(cfg.layer_split, 'auto');
    assert.deepEqual(f.control.resolveInstance(done.result.instanceId).gpus, [1, 0]);
    await f.control.unload(done.result.instanceId);
    const one = await load(f, { gpuRoles: { 0: 'off', 1: 'main' } });
    assert.equal(one.state, 'done', one.error);
    const c1 = written(f, one);
    assert.equal(c1.gpu, 1, 'one card is a number, never a list');
    assert.deepEqual(c1.gpu_roles, { 0: 'off', 1: 'main' });
  } finally { await f.cleanup(); }
});

test('every card Off is allowed: no card in the list, "gpu_none" says it was meant', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const done = await load(f, { gpuRoles: { 0: 'off', 1: 'off' } });
    assert.equal(done.state, 'done', done.error);
    const cfg = written(f, done);
    assert.deepEqual(cfg.gpu, []);
    assert.equal(cfg.gpu_none, true);
  } finally { await f.cleanup(); }
});

test('a settings file that names roles keeps them when the screen only suggests cards; a plain card list replaces them', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const path = join(f.repo, 'strata-iq2_xs.json');
    const config = JSON.parse(readFileSync(path, 'utf8'));
    Object.assign(config, { gpu: [0, 1], gpu_order: 'as_given', gpu_roles: { 0: 'helper', 1: 'main' } });
    writeFileSync(path, JSON.stringify(config));
    await f.control.refresh(true);
    const kept = await load(f, { gpus: [1, 0], gpusFromRecommendation: true });
    assert.equal(kept.state, 'done', kept.error);
    assert.deepEqual(written(f, kept).gpu, [0, 1]);
    await f.control.unload(kept.result.instanceId);
    const picked = await load(f, { gpus: [1] });
    assert.equal(picked.state, 'done', picked.error);
    const cfg = written(f, picked);
    assert.equal(cfg.gpu, 1);
    assert.equal('gpu_roles' in cfg, false, 'a card list picked by hand replaces the roles');
  } finally { await f.cleanup(); }
});

test('roles that make no sense are refused with a plain line', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const two = await load(f, { gpuRoles: { 0: 'main', 1: 'main' } });
    assert.equal(two.state, 'error');
    assert.match(two.error, /Main, Helper or Off/);
    const nope = await load(f, { gpuRoles: { 5: 'main' } });
    assert.equal(nope.state, 'error');
    assert.match(nope.error, /detected graphics cards/);
  } finally { await f.cleanup(); }
});

test('"Tune this model" keeps a two-card role map exactly as the person set it, in the file and in the engine it starts (#3655)', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const path = join(f.repo, 'strata-iq2_xs.json');
    const config = JSON.parse(readFileSync(path, 'utf8'));
    Object.assign(config, { gpu: [0, 1], gpu_order: 'as_given', gpu_roles: { 0: 'helper', 1: 'main' }, gpu_cards: { 0: 'GTX 1080', 1: 'GTX 1080' } });
    writeFileSync(path, JSON.stringify(config));
    await f.control.refresh(true);
    const tuned = await finishJob(f.control, f.control.operation('tune', { model: 'iq2_xs', profile: 'performance', calibrate: false }).id);
    assert.equal(tuned.state, 'done', tuned.error);
    const after = JSON.parse(readFileSync(path, 'utf8'));
    for (const key of ['gpu', 'gpu_order', 'gpu_roles', 'gpu_cards']) assert.deepEqual(after[key], config[key], `${key} in the settings file is untouched by Tune`);
    const started = written(f, tuned);
    assert.deepEqual(started.gpu, [0, 1], 'and the engine Tune loads runs on both cards');
    assert.deepEqual(started.gpu_roles, { 0: 'helper', 1: 'main' });
  } finally { await f.cleanup(); }
});
