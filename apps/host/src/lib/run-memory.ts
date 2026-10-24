import type { Experience } from './experience';
import type { Hardware } from './types';
import type { ComputerUse } from './run-profiles';

export interface RunMemory {
  total: number;
  weights: number;
  context: number;
  runtime: number;
  ram: number;
  gpu: number;
  ssd: number;
  budget: number;
  reserve: number;
  unified: boolean;
  cacheFits: boolean;
  devices: { name: string; budget: number }[];
}

/** A capacity planning preview. Actual placement must be verified by the engine. */
export function planMemory(experience: Experience, hardware: Hardware | null, use: ComputerUse, parallel = 1, allowSpill = true): RunMemory | null {
  if (!experience.memory || !hardware) return null;
  const m = experience.memory;
  const unified = hardware.accelerator.unified;
  const cpuOnly = hardware.accelerator.backend === 'cpu' || experience.config.placement === 'CPU only';
  const reserve = Math.min(hardware.memory.totalGb, use === 'shared' ? Math.max(4, hardware.memory.totalGb * .25) : Math.max(2, hardware.memory.totalGb * .125));
  const ramBudget = Math.max(0, hardware.memory.totalGb - reserve);
  const devices = unified || cpuOnly ? [] : hardware.gpus.filter(g => !g.unified && g.backend !== 'cpu').map(g => ({
    name: g.name,
    budget: Math.max(0, g.vramGb - (use === 'shared' ? Math.max(1, g.vramGb * .1) : Math.max(.5, g.vramGb * .05))),
  }));
  const gpuBudget = devices.reduce((sum, g) => sum + g.budget, 0);
  // Explicit partial offload needs layer sizes, which this catalog does not expose.
  if (!unified && !cpuOnly && !['Automatic', 'All layers'].includes(experience.config.placement)) return null;
  const budget = ramBudget + gpuBudget;
  const context = m.cache * Math.max(1, parallel);
  const total = m.weights + context + m.runtime;
  const fixed = context + m.runtime;
  const cacheFits = fixed <= (unified || cpuOnly ? ramBudget : gpuBudget || ramBudget);
  const ssd = cacheFits && allowSpill ? Math.min(m.weights, Math.max(0, total - budget)) : 0;
  const gpu = Math.min(gpuBudget, total - ssd);
  const ram = Math.min(ramBudget, Math.max(0, total - ssd - gpu));
  return { total, weights: m.weights, context, runtime: m.runtime, ram, gpu, ssd, budget, reserve, unified, cacheFits, devices };
}

/** Illustrative preview curves stay separate from measured/estimated evidence. */
export function demoRunExperience(experience: Experience, memory: RunMemory | null, use: ComputerUse): Experience {
  if (experience.source !== 'demo' || !memory || !experience.reading || !experience.answer) return experience;
  const spillFraction = memory.ssd / Math.max(1, memory.weights);
  const factor = (use === 'full' ? 1.08 : 1) / (1 + spillFraction * 3);
  const scale = (range: NonNullable<Experience['reading']>) => ({ low: Math.max(1, Math.round(range.low * factor)), high: Math.max(2, Math.round(range.high * factor)) });
  return { ...experience, reading: scale(experience.reading), answer: scale(experience.answer), provenance: 'Illustrative design preview including context, computer use and SSD spill. Not a benchmark.' };
}
