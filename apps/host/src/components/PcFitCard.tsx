import { tr } from '../lib/i18n';
import type { Fit } from '../lib/native-fit';

const gb = (n: number) => `${Number.isInteger(n) ? n : n.toFixed(1)} GB`;

function Meter({ label, total, use }: { label: string; total: number; use: number }) {
  const pct = total > 0 ? Math.max(2, Math.min(100, (use / total) * 100)) : 0;
  return <div className="pc-meter">
    <div className="pc-meter__top"><span>{tr(label)}</span><b>{gb(use)}</b><span className="muted">{tr("of {0}", [gb(total)])}</span></div>
    <div className="pc-meter__track" role="img" aria-label={tr("{0}: about {1} of {2} used by this model", [tr(label), gb(use), gb(total)])}><i style={{ width: `${pct}%` }}/></div>
  </div>;
}

/**
 * "This computer": the graphics memory and RAM, and what the chosen version would take of each. One plain line says whether it fits
 * or how much would be read from the SSD (the n-gram table, which always sits on the SSD, is not counted).
 */
export function PcFitCard({ gpuName, ramGb, vramGb, fit, inUse = null }: { gpuName?: string | null; ramGb: number | null; vramGb: number; fit: Fit | null; inUse?: { ramGb: number; vramGb: number } | null }) {
  const freeVram = Math.max(0, vramGb - (inUse?.vramGb ?? 0)), freeRam = ramGb ? Math.max(0, ramGb - (inUse?.ramGb ?? 0)) : 0;
  return <section className="st-card pc-card">
    <div><h2 className="h4">{tr("This computer")}</h2>
      <p className="small muted">{tr(gpuName ? tr("{0} · {1} graphics memory", [gpuName, gb(vramGb)]) : tr("No supported graphics card reported by this installation."))}{ramGb ? tr(" · {0} RAM", [gb(ramGb)]) : ''}</p></div>
    {tr(fit?.known && vramGb > 0 ? <Meter label="Graphics memory" total={freeVram} use={fit.vramUseGb}/> : null)}
    {tr(fit?.known && ramGb ? <Meter label="RAM" total={freeRam} use={fit.ramUseGb}/> : null)}
    {tr(inUse ? <p className="xs muted">{tr("Models that stay loaded use about {0} GB of RAM and {1} GB of this card’s memory.", [inUse.ramGb, inUse.vramGb])}</p> : null)}
    {tr(fit?.known ? <p className="pc-note">{tr(fit.spillGb >= 1 ? 'About {0} GB will be read from the SSD, so answers can be a little slower.' : 'Fits fully in memory.', [Math.round(fit.spillGb)])}</p> : null)}
  </section>;
}
