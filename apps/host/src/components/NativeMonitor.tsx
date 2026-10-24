import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useRef, useState } from 'react';
import { useNative } from '../lib/native';
import { ReferenceSpark } from './charts';
import { Screen } from './layout';
import { IconBtn, Select } from './ui';
import { monitorPng } from '../lib/snapshot';
import { SnapshotDialog } from './SnapshotDialog';
const number = (n: any, d = 1) => typeof n === 'number' && Number.isFinite(n) ? n.toLocaleString(localeTag(), { maximumFractionDigits: d }) : '—';
export function NativeMonitor() {
    const native = useNative();
    const [selection, setSelection] = useState('all'), [snapshot, setSnapshot] = useState(''), [open, setOpen] = useState(false), [saving, setSaving] = useState(false), [error, setError] = useState('');
    const capture = useRef<HTMLDivElement>(null);
    const instances = native?.instances ?? [], selected = instances.find(i => i.id === selection) ?? (instances.length === 1 ? instances[0] : null), rows = selected ? [selected] : instances;
    const devices = new Map<number, Record<string, any>>();
    for (const i of instances)
        for (const gpu of i.metrics?.hardware?.gpus ?? [])
            devices.set(gpu.index, gpu);
    for (const i of instances) {
        const h = i.metrics?.hardware;
        if (!h || h.gpus?.length)
            continue;
        const index = i.gpus.length === 1 ? i.gpus[0] : native?.hardware?.gpus.length === 1 ? native.hardware.gpus[0].index : null;
        if (index != null && !devices.has(index))
            devices.set(index, { util: h.gpu_util, mem_used: h.gpu_mem_used, temp: h.gpu_temp, power: h.gpu_power, mem_total: h.gpu_mem_total });
    }
    const hardware = instances.find(i => i.metrics)?.metrics?.hardware;
    const toolServers = rows.flatMap(i => (i.metrics?.mcp?.servers ?? []).map((s: any) => ({ ...s, model: i.label, instanceId: i.id })));
    const requests = rows.flatMap(i => (i.metrics?.requests ?? []).map((r: any) => ({ ...r, model: i.apiModelId }))).sort((a, b) => b.time - a.time).slice(0, 20);
    async function save() {
        setOpen(true);
        setSaving(true);
        setError('');
        try {
            if (capture.current)
                setSnapshot(await monitorPng(capture.current));
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setSaving(false);
        }
    }
    const detail = selected?.metrics, live = detail?.live, last = detail?.requests?.[0];
    const context = live && live.state !== 'idle' ? (live.prompt_tokens ?? 0) + (live.generated ?? 0) : (last?.prompt_tokens ?? 0) + (last?.output_tokens ?? 0);
    return <Screen plainHeader title={tr("Monitor")} width="wide" titleDetail={instances.length > 1 ? <Select ariaLabel={tr("Monitor model")} value={selection} options={[{ value: 'all', label: `All models (${instances.length})` }, ...instances.map(i => ({ value: i.id, label: `${i.label} · ${i.gpus.map(g => `GPU ${g + 1}`).join(' + ') || 'Automatic'}` }))]} onChange={setSelection}/> : selected?.label} actions={<IconBtn name="image" title={tr("Save Monitor screenshot")} disabled={saving} onClick={() => void save()}/>}><div className="stack stack--lg" ref={capture}>
  <section className="st-card"><h2 className="h3">{tr(instances.length > 1 ? 'Model performance' : 'Model state')}</h2><div className="st-tablewrap"><table className="st-table native-performance-table"><thead><tr><th>{tr("Model / device")}</th><th>{tr("Generating tok/s")}</th><th>{tr("Reading tok/s")}</th><th>{tr("State")}</th><th>{tr("Active / queued")}</th></tr></thead><tbody>{tr(rows.map(i => { const raw = i.metrics, live = raw?.live, last = raw?.requests?.[0], active = live?.state === 'generating', reading = live?.state === 'reading'; return <tr key={i.id}><td><button className="text-btn strong" onClick={() => setSelection(i.id)}>{tr(i.label)}</button><small>{tr(i.gpus.map(g => `GPU ${g + 1}`).join(' + ') || 'Automatic GPU')}</small></td><td className="num">{tr(number(active ? live.tok_s : last?.decode_tok_s))}<small>{tr(active ? 'Now' : last ? 'Last request' : 'No request yet')}</small></td><td className="num">{tr(number(reading ? live.prefill_tok_s_mean : last?.prompt_ms > 0 ? ((last.prompt_tokens ?? 0) - (last.reused ?? 0)) / (last.prompt_ms / 1000) : null, 0))}<small>{tr(reading ? 'Now' : last ? 'Last request' : 'No request yet')}</small></td><td>{tr(i.status === 'ready' ? i.phase : i.status)}{tr(i.error ? <small className="inline-error">{tr(i.error)}</small> : null)}</td><td>{tr(live ? live.running ?? (['reading', 'generating'].includes(live.state) ? 1 : 0) : '—')} / {tr(live?.queued ?? '—')}</td></tr>; }))}</tbody></table></div><p className="xs muted">{tr("Speeds come from each engine instance. Missing readings show \u2014.")}</p></section>
  {tr(selected ? <section className="st-card"><div className="section-toolbar"><h2 className="h3">{tr(selected.label)}</h2><span className="small muted">{tr(selected.apiModelId)}</span></div><div className="native-monitor-detail"><div><h3 className="h4">{tr("Speed history")}</h3><ReferenceSpark label={tr("Generation speed")} data={detail?.history?.tok_s ?? []} timestamps={(detail?.history?.tok_s ?? []).map((_: unknown, i: number) => Date.now() - ((detail?.history?.tok_s?.length ?? 0) - i - 1) * 1000)} secondary={{ data: detail?.history?.prefill_tok_s_mean ?? [], label: 'Prompt reading', tone: 'var(--st-chart-prefill)' }} unit="tok/s" tone="var(--st-chart-decode)"/></div><div><h3 className="h4">{tr("Context")}</h3><strong>{tr(detail ? number(context, 0) : '—')} / {tr(number(detail?.engine?.max_context, 0))}</strong><p className="xs muted">{tr("Tokens in this model\u2019s current or last request.")}</p></div><div><h3 className="h4">{tr("Expert cache")}</h3><strong>{tr(number(detail?.engine?.expert_slots, 0))} / {tr(number(detail?.engine?.experts, 0))}</strong><p className="xs muted">{tr("Experts reported by this engine.")}{tr(detail?.engine?.expert_cache_mib != null ? ` ${number(detail.engine.expert_cache_mib / 1024)} GB cached.` : '')}</p></div></div></section> : null)}
  <section className="st-card"><h2 className="h3">{tr("This computer")}</h2><p className="xs muted">{tr("GPU readings include all work on that device. System RAM is shared by all models.")}</p><div className="native-host-metrics"><div><span>{tr("CPU load")}</span><strong>{tr(number(hardware?.cpu_pct ?? hardware?.cpu, 0))}%</strong></div><div><span>{tr("System RAM")}</span><strong>{tr(number(hardware?.ram_used == null ? null : hardware.ram_used / 1024 ** 3))} / {tr(number(hardware?.ram_total == null ? null : hardware.ram_total / 1024 ** 3))}{tr(" GB")}</strong></div><div><span>{tr("Disk read")}</span><strong>{tr(number(hardware?.disk_read_mb))}{tr(" MB/s")}</strong></div></div><div className="st-tablewrap"><table className="st-table"><thead><tr><th>{tr("GPU")}</th><th>{tr("Load")}</th><th>{tr("Memory")}</th><th>{tr("Temperature")}</th><th>{tr("Power")}</th></tr></thead><tbody>{tr((native?.hardware?.gpus ?? []).map(g => { const value = devices.get(g.index); return <tr key={g.index}><td>{tr("GPU ")}{tr(g.index + 1)} · {tr(g.name)}</td><td>{tr(number(value?.util, 0))}%</td><td>{tr(number(value?.mem_used == null ? null : value.mem_used / 1024 ** 3))} / {tr(number(value?.mem_total == null ? g.vram_gb : value.mem_total / 1024 ** 3))}{tr(" GB")}</td><td>{tr(number(value?.temp, 0))}{tr(" \u00B0C")}</td><td>{tr(number(value?.power, 0))}{tr(" W")}</td></tr>; }))}</tbody></table></div></section>
  {tr(toolServers.length ? <section className="st-card"><h2 className="h3">{tr("Connected tools")}</h2>{tr(toolServers.map((s: any) => <div key={`${s.instanceId}-${s.name}`} className="native-setting-row"><span><strong>{tr(s.name)}</strong><small>{tr(s.model)} · {tr(s.tools?.length ?? 0)}{tr(" tools")}</small>{tr(s.error ? <small className="inline-error">{tr(s.error)}</small> : null)}</span><span className="small muted">{tr(s.state ?? s.status ?? 'Configured')}</span></div>))}</section> : null)}
  <section className="st-card"><h2 className="h3">{tr("Recent requests")}</h2><div className="st-tablewrap"><table className="st-table"><thead><tr><th>{tr("Time")}</th><th>{tr("Model")}</th><th>{tr("Prompt")}</th><th>{tr("Reused")}</th><th>{tr("Output")}</th><th>{tr("tok/s")}</th><th>{tr("Duration")}</th></tr></thead><tbody>{tr(requests.length ? requests.map((r, i) => <tr key={`${r.model}-${r.id ?? i}`}><td>{tr(new Date(r.time * 1000).toLocaleTimeString(localeTag()))}</td><td>{tr(r.model)}</td><td>{tr(number(r.prompt_tokens, 0))}</td><td>{tr(number(r.reused, 0))}</td><td>{tr(number(r.output_tokens, 0))}</td><td>{tr(number(r.decode_tok_s))}</td><td>{tr(number(r.duration_s))}{tr(" s")}</td></tr>) : <tr><td colSpan={7}>{tr("No requests yet")}</td></tr>)}</tbody></table></div></section>
 </div><SnapshotDialog open={open} data={snapshot} pending={saving} error={error} onClose={() => setOpen(false)} onRetry={() => void save()}/></Screen>;
}
