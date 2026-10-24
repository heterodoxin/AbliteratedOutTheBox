import { tr } from '../lib/i18n';
import type { Experience } from '../lib/experience';
import { speedText } from '../lib/experience';
import type { RunMemory } from '../lib/run-memory';
import type { Hardware } from '../lib/types';
import { Icon } from './ui';
export function RunPreview({ experience, memory, hardware, imageEnabled = false }: {
    experience: Experience;
    memory: RunMemory | null;
    hardware: Hardware | null;
    imageEnabled?: boolean;
}) {
    const spill = !!memory && memory.ssd > .05;
    const cannotFit = !!memory && (!memory.cacheFits || memory.total > memory.budget && !spill);
    const max = memory ? Math.max(memory.total, memory.budget) : 1;
    return <aside className="run-preview">
    <section className="run-card run-forecast" aria-label={tr("Expected experience")}>
      <div className="run-card__head"><span><Icon name="gauge" size={20}/><h2>{tr("Expected experience")}</h2></span><span className="run-source">{tr(experience.source === 'measured' ? 'Measured' : 'Estimated')}</span></div>
      <div className="run-speeds">{tr([{ label: 'Reading', icon: 'file' as const, value: experience.reading }, { label: 'Generating', icon: 'chat' as const, value: experience.answer }].map(item => <div key={item.label}><span><Icon name={item.icon} size={15}/>{tr(item.label)}</span><strong>{tr(item.value ? speedText(item.value) : '—')}</strong><small>{tr(item.value ? 'tokens / second' : 'Needs a speed test')}</small></div>))}</div>
      {tr(memory ? <>
        <div className="run-memory__total"><span>{tr(imageEnabled ? 'Chat memory needed' : 'Memory needed')}</span><strong>{tr(memory.total.toFixed(1))} <small>{tr("GB")}</small></strong></div>
        <div className="run-memory-bar" role="img" aria-label={tr("Capacity estimate: {0} GB GPU, {1} GB {2}, {3} GB SSD spill{4}", [memory.gpu.toFixed(1), memory.ram.toFixed(1), memory.unified ? tr('unified memory') : tr('RAM'), memory.ssd.toFixed(1), cannotFit ? tr(', context cache does not fit') : tr('')])}>
          {tr(memory.gpu > 0 ? <span data-pool="gpu" style={{ flex: memory.gpu / max }}/> : null)}<span data-pool="ram" style={{ flex: memory.ram / max }}/>{tr(spill ? <span data-pool="ssd" style={{ flex: memory.ssd / max }}/> : null)}{tr(memory.budget > memory.total ? <span data-pool="free" style={{ flex: (memory.budget - memory.total) / max }}/> : null)}{tr(cannotFit ? <span data-pool="over" style={{ flex: Math.max(0, memory.total - memory.budget) / max }}/> : null)}
        </div>
        <div className="run-memory-legend">{tr(memory.gpu > 0 ? <span><i data-pool="gpu"/>{tr("GPU ")}<b>{tr(memory.gpu.toFixed(1))}{tr(" GB")}</b></span> : null)}<span><i data-pool="ram"/>{tr(memory.unified ? 'Unified' : 'RAM')} <b>{tr(memory.ram.toFixed(1))}{tr(" GB")}</b></span>{tr(spill ? <span><i data-pool="ssd"/>{tr("SSD ")}<b>{tr(memory.ssd.toFixed(1))}{tr(" GB")}</b></span> : null)}</div>
        <div className="run-fit-note" data-spill={spill || cannotFit || undefined}><Icon name={spill ? 'disk' : cannotFit ? 'warning' : 'check'} size={17}/><div><strong>{tr(cannotFit ? memory.cacheFits ? 'Model needs more memory' : 'Context cache needs more memory' : spill ? 'SSD spill · slower responses' : 'Fits in memory')}</strong><p>{tr(cannotFit ? memory.cacheFits ? 'This configuration needs more resident memory. Review memory mapping or choose a smaller version.' : 'The context cache and runtime need to fit in RAM or GPU memory.' : spill ? 'This is just a warning that model may run slower due to offloading model to SSD.' : `${Math.max(0, memory.budget - memory.total).toFixed(1)} GB of planned capacity remains.`)}</p></div></div>
        <details className="run-memory-details"><summary>{tr("Memory breakdown ")}<Icon name="chevron" size={14}/></summary><dl><div><dt>{tr("Model weights")}</dt><dd>{tr(memory.weights.toFixed(1))}{tr(" GB")}</dd></div><div><dt>{tr("Context cache")}</dt><dd>{tr(memory.context.toFixed(1))}{tr(" GB")}</dd></div><div><dt>{tr("Runtime")}</dt><dd>{tr(memory.runtime.toFixed(1))}{tr(" GB")}</dd></div><div><dt>{tr("RAM headroom")}</dt><dd>{tr(memory.reserve.toFixed(1))}{tr(" GB")}</dd></div>{tr(memory.devices.map((device, i) => <div key={i}><dt>{tr(device.name)}</dt><dd>{tr(device.budget.toFixed(1))}{tr(" GB budget")}</dd></div>))}</dl><p className="xs muted">{tr("Capacity plan from model metadata. Actual placement, free memory and SSD reads need engine verification. ")}{tr(memory.unified ? 'CPU and GPU share one memory pool.' : '')}</p></details>
      </> : <p className="small muted">{tr("Memory placement is not available for this configuration.")}</p>)}
      {tr(imageEnabled ? <p className="run-image-memory-note"><Icon name="image" size={14}/>{tr("Image model memory is not included in this estimate.")}</p> : null)}
    </section>
    {tr(hardware ? <section className="run-card run-computer" aria-label={tr("This computer")}>
      <div className="run-card__head"><span><Icon name="computer" size={18}/><h2>{tr("This computer")}</h2></span></div>
      {tr(hardware.processors?.length ? hardware.processors.map((cpu, i) => <div key={cpu.id}><Icon name="cpu" size={17}/><span>{tr(cpu.model)}<small>{tr(hardware.processors!.length > 1 ? `CPU ${i + 1} · ` : '')}{tr(cpu.cores)}{tr(" cores \u00B7 ")}{tr(cpu.threads)}{tr(" threads")}</small></span></div>) : <div><Icon name="cpu" size={17}/><span>{tr(hardware.cpu.model)}<small>{tr(hardware.cpu.cores)}{tr(" cores \u00B7 ")}{tr(hardware.cpu.threads)}{tr(" threads")}</small></span></div>)}
      <div><Icon name="memory" size={17}/><span>{tr(hardware.memory.totalGb.toFixed(0))}{tr(" GB ")}{tr(hardware.accelerator.unified ? 'unified memory' : 'RAM')}</span></div>
      {tr(hardware.gpus.map((gpu, i) => <div key={gpu.id ?? i}><Icon name="gpu" size={17}/><span>{tr(gpu.name)}<small>{tr(hardware.gpus.length > 1 ? `GPU ${i + 1} · ` : '')}{tr(gpu.unified ? 'Shared unified memory' : `${gpu.vramGb.toFixed(0)} GB VRAM`)} · {tr(gpu.backend.toUpperCase())}</small></span></div>))}
    </section> : null)}
  </aside>;
}
