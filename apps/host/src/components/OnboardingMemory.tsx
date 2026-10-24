import { tr } from '../lib/i18n';
import { memoryInventory } from '../lib/hardware-summary';
import { sizeGb } from '../lib/format';
import type { Hardware } from '../lib/types';
import type { NativeState } from '../lib/native';
import { Icon } from './ui';
export function OnboardingMemory({ hardware, native }: {
    hardware: Hardware | null;
    native: NativeState['hardware'] | undefined;
}) {
    const gpus = hardware ? hardware.gpus.filter(gpu => gpu.backend !== 'cpu').map(gpu => ({ id: gpu.id ?? String(gpu.index), name: gpu.name, memory: gpu.vramGb, shared: gpu.unified })) : native?.gpus.map(gpu => ({ id: String(gpu.index), name: gpu.name, memory: gpu.vram_gb, shared: false })) ?? [];
    const dedicatedCount = gpus.filter(gpu => !gpu.shared).length;
    const memory = hardware ? memoryInventory(hardware) : native ? { ram: native.ram_gb, vram: native.gpus.reduce((sum, gpu) => sum + Math.max(0, gpu.vram_gb || 0), 0), total: 0, shared: false } : null;
    if (!memory)
        return <div className="onboard-hardware" role="group" aria-label={tr("Computer memory")}><div className="onboard-memory-intro"><Icon name="memory" size={18}/><span>{tr("Reading memory\u2026")}</span></div></div>;
    const total = memory.ram + memory.vram;
    return <div className="onboard-hardware" role="group" aria-label={tr("Computer memory")} data-shared={memory.shared} title={tr(memory.shared ? 'CPU and GPU use the same memory pool. Shared VRAM is counted once in the total.' : 'Total physical capacity across RAM and all dedicated GPUs. Model placement determines how it is used.')}>
  <div className="onboard-memory-header"><div className="onboard-memory-intro"><Icon name="computer" size={18}/><span>{tr("Your computer")}</span></div>{tr(memory.shared ? <small>{tr("One shared memory pool")}</small> : null)}</div>
  <div className="onboard-memory-equation">
   <div className="onboard-memory-stat"><small>{tr("RAM")}</small><div className="onboard-memory-value"><Icon name="memory" size={20}/><strong>{tr(sizeGb(memory.ram))}</strong></div></div>
   <span className="onboard-memory-operator" aria-hidden="true">{tr(memory.shared ? '↔' : '+')}</span>
   <div className="onboard-memory-stat"><small>{tr("VRAM")}{tr(dedicatedCount > 1 ? <span> · {tr(dedicatedCount)}{tr(" GPUs")}</span> : null)}</small><div className="onboard-memory-value"><Icon name="gpu" size={20}/><strong>{tr(memory.shared ? 'Shared' : sizeGb(memory.vram))}</strong></div></div>
   <span className="onboard-memory-operator" aria-hidden="true">=</span>
   <div className="onboard-memory-stat onboard-memory-total"><small>{tr("Total memory")}</small><div className="onboard-memory-value"><Icon name="layers" size={20}/><strong>{tr(sizeGb(total))}</strong></div></div>
  </div>
  {tr(gpus.length > 1 ? <details className="onboard-memory-devices"><summary>{tr(gpus.length)}{tr(" GPUs connected")}<Icon name="chevron" size={14}/></summary><ul aria-label={tr("GPU memory breakdown")}>{tr(gpus.map(gpu => <li key={gpu.id}><span><Icon name="gpu" size={16}/>{tr(gpu.name)}</span><strong>{tr(gpu.shared ? 'Shared with RAM' : sizeGb(Math.max(0, gpu.memory || 0)))}</strong></li>))}</ul></details> : null)}
 </div>;
}
