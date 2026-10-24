import type { Hardware } from './types';

/** Inventory total only. Placement and fit continue to use separate budgets. */
export function memoryInventory(hardware: Hardware) {
  const ram = hardware.memory.totalGb;
  const dedicated = hardware.gpus.filter(g => !g.unified && g.backend !== 'cpu');
  const vram = dedicated.reduce((sum, gpu) => sum + Math.max(0, gpu.vramGb || 0), 0);
  return { ram, vram, total: ram + vram, shared: hardware.accelerator.unified && !vram };
}

export function memoryReading(used:number,total:number){const safeTotal=Number.isFinite(total)?Math.max(0,total):0;const safeUsed=Number.isFinite(used)?Math.max(0,used):0;return {used:Math.min(safeUsed,safeTotal),total:safeTotal,free:Math.max(0,safeTotal-safeUsed),over:Math.max(0,safeUsed-safeTotal)};}
