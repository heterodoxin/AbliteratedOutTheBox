// #3722: the host trusts the operating system's certificates in addition to Node's bundled list, and only adds to it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { useSystemCertificates } from '../server/system-ca.mjs';
import { agentEnvironment } from '../server/agent/harness-runtime.mjs';

const fake = (system, bundled = ['B1', 'B2']) => {
  const calls = [];
  return { calls, getCACertificates: kind => (kind === 'system' ? system : bundled), setDefaultCACertificates: list => calls.push(list) };
};

test('system roots are added to the bundled ones', () => {
  const tls = fake(['S1', 'B1']);
  assert.equal(useSystemCertificates(tls, {}), 'system');
  assert.deepEqual(tls.calls, [['B1', 'B2', 'S1']]);
});

test('an empty or unreadable system store, or an older Node, changes nothing and never throws', () => {
  const empty = fake([]);
  assert.equal(useSystemCertificates(empty, {}), 'bundled');
  assert.equal(empty.calls.length, 0);
  assert.equal(useSystemCertificates({}, {}), 'bundled');
  assert.equal(useSystemCertificates({ getCACertificates() { throw new Error('store'); }, setDefaultCACertificates() {} }, {}), 'bundled');
});

test('STRATA_NO_SYSTEM_CA=1 leaves Node as it was', () => {
  const tls = fake(['S1']);
  assert.equal(useSystemCertificates(tls, { STRATA_NO_SYSTEM_CA: '1' }), 'bundled');
  assert.equal(tls.calls.length, 0);
});

test('the real Node of this machine can read its system store', () => {
  const tls = process.getBuiltinModule?.('node:tls');
  if (typeof tls?.getCACertificates !== 'function') return;
  assert.ok(Array.isArray(tls.getCACertificates('system')));
});

test('the agent process is started with NODE_USE_SYSTEM_CA=1, and no switch turns verification off', () => {
  const env = agentEnvironment({ PATH: 'x' }, {});
  assert.equal(env.NODE_USE_SYSTEM_CA, '1');
  assert.ok(!('NODE_TLS_REJECT_UNAUTHORIZED' in env));
});
