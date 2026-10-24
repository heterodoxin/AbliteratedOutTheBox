import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const built = await build({ stdin: { contents: "export * from './src/lib/native-monitor';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { withHostSensors } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

const none = { id: 'GPU-1', utilizationPct: null, temperatureC: null, memoryUsedGb: null, powerW: null, powerLimitW: null, pcieRxMbs: null };
const live = { hardware: { gpus: [none], gpuLoad: NaN, vramUsedGb: NaN, vramTotalGb: 11.9, gpuTempC: NaN, powerW: NaN, powerLimitW: NaN, pcieRxMbs: NaN, cpuLoad: 20 } };
const host = { hardware: { gpus: [{ id: 'GPU-1', utilizationPct: 7, temperatureC: 41, memoryUsedGb: 0.9, powerW: 22, powerLimitW: 250, pcieRxMbs: 3 }], gpuLoad: 7, vramUsedGb: 0.9, vramTotalGb: 11.9, gpuTempC: 41, powerW: 22, powerLimitW: 250, pcieRxMbs: 3 } };

test('with no model loaded the GPU temperature, load, VRAM, power and PCIe come from the host sensors', () => {
  const out = withHostSensors(live, host);
  assert.equal(out.hardware.gpus[0].temperatureC, 41);
  assert.equal(out.hardware.gpus[0].memoryUsedGb, 0.9);
  assert.equal(out.hardware.gpuLoad, 7);
  assert.equal(out.hardware.powerW, 22);
  assert.equal(out.hardware.pcieRxMbs, 3);
  assert.equal(out.hardware.cpuLoad, 20, 'what the engine reported is kept');
});

test('what a running engine reports wins over the host sensors', () => {
  const running = { hardware: { ...live.hardware, gpus: [{ ...none, temperatureC: 55, utilizationPct: 90 }], gpuTempC: 55, gpuLoad: 90 } };
  const out = withHostSensors(running, host);
  assert.equal(out.hardware.gpus[0].temperatureC, 55);
  assert.equal(out.hardware.gpuLoad, 90);
});

test('no host monitor, no change', () => {
  assert.equal(withHostSensors(live, null), live);
});
