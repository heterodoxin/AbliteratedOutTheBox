import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { scopedMonitor, instanceLabel, deviceLabel } from '../lib/runtime-models';
/** The original Strata dashboard, adapted to the current shell and device inventory. */
import { useRef, useState, type ReactNode } from 'react';
import { flushSync } from 'react-dom';
import { SnapshotDialog } from '../components/SnapshotDialog';
import { requestSummary } from '../lib/monitor-summary';
import { monitorPng } from '../lib/snapshot';
import { Gauge, ReferenceSpark } from '../components/charts';
import { Brand } from '../components/Brand';
import { Bar, Btn, Icon, IconBtn, Select, type IconName } from '../components/ui';
import { Screen } from '../components/layout';
import { contextLabel, duration, tokens, tps } from '../lib/format';
import { gpuCapacity, gpuId, gpuMetric, hasReading } from '../lib/monitor-hardware';
import { useStore } from '../lib/store';
import { useKept } from '../lib/kept-state';
import type { Hardware, MonitorState, ServeRequest } from '../lib/types';
import { useNative, nativeRouting, type NativeState } from '../lib/native';
import { nativeMonitorData, withHostSensors } from '../lib/native-monitor';
const number = (value: number | null | undefined, digits = 0) => hasReading(value) ? value.toLocaleString(localeTag(), { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
const percent = (used: number | null | undefined, total: number | null | undefined) => hasReading(used) && hasReading(total) && total > 0 ? Math.max(0, Math.min(100, used / total * 100)) : null;
const sourceLabel = (source: string | undefined) => source === 'demo' ? 'Illustrative' : source === 'sensor' || source === 'engine' || source === 'reported' ? 'Measured' : source === 'estimated' ? 'Estimated' : 'Unavailable';
function ReadingBar({ label, value, tone = 'accent' }: {
    label: string;
    value: number | null;
    tone?: 'accent' | 'info' | 'warn' | 'danger';
}) {
    return hasReading(value) ? <Bar label={tr(label)} value={value} tone={tone}/> : <div className="st-progress" role="img" aria-label={tr("{0}: unavailable", [label])}/>;
}
function MonitorMetric({ label, icon, value, unit, sub, chart, source, secondary }: {
    label: string;
    icon: IconName;
    value: ReactNode;
    unit?: string;
    sub: ReactNode;
    chart: ReactNode;
    source: string;
    secondary?: ReactNode;
}) {
    return <section className="st-card monitor-metric" aria-label={tr(label)}>
    <div className="st-metric__label"><Icon name={icon} size={18}/>{tr(label)}<span className="monitor-metric__source xs">{tr(source)}</span></div>
    <div className={secondary ? 'monitor-speed-values' : 'monitor-metric__reading'}><div><div className="st-metric__value">{tr(value)}{tr(unit ? <span className="st-metric__unit">{tr(unit)}</span> : null)}</div><div className="monitor-metric__sub xs muted">{tr(sub)}</div></div>{tr(secondary)}</div>
    {tr(chart)}
  </section>;
}
function requestStatus(request: ServeRequest) {
    if (request.state === 'error')
        return 'Error';
    if (request.state === 'queued')
        return 'Queued';
    if (request.state === 'generating')
        return 'Generating';
    if (request.cancelRequested || ['cancel', 'disconnect'].includes(request.finishReason ?? ''))
        return 'Stopped';
    return ['length', 'max_tokens'].includes(request.finishReason ?? '') ? 'Max tokens' : 'Done';
}
export function MonitorScreen() {
    const native = useNative();
    return <LegacyMonitorScreen native={native?.instances.length ? native : null}/>;
}
function LegacyMonitorScreen({ native }: {
    native: NativeState | null;
}) {
    const storedLive = useStore(s => s.monitor), inventory = useStore(s => s.hardware);
    const storedRuntime = useStore(s => s.runtime), serving = useStore(s => s.serving);
    const [instanceSelection, setInstanceSelection] = useState('all');
    const routing = native ? nativeRouting(native, inventory) : null;
    const runtime = routing && native ? { ...routing, instances: routing.instances.map(instance => { const raw = native.instances.find(item => item.id === instance.id)?.metrics?.live; return { ...instance, outputTps: raw?.tok_s ?? NaN, promptTps: raw?.prefill_tok_s_mean ?? NaN, active: raw ? (raw.running ?? (['reading', 'generating'].includes(raw.state) ? 1 : 0)) : NaN, queued: raw?.queued ?? NaN }; }) } : storedRuntime;
    const live = native ? withHostSensors(nativeMonitorData(native, inventory, instanceSelection), storedLive) : storedLive;
    const multipleModels = (runtime?.instances.length ?? 0) > 1;
    const selectedInstance = runtime?.instances.find(i => i.id === instanceSelection) ?? (!multipleModels ? runtime?.instances[0] : undefined);
    const selection = selectedInstance?.id ?? 'all';
    const scoped = native ? live : scopedMonitor(live, runtime ?? null, selection, serving?.requests ?? []);
    const scopeLabel = selectedInstance ? instanceLabel(selectedInstance) : `${runtime?.instances.length ?? 0} loaded models`;
    // null = the person has not chosen: a model with assigned cards shows those (#3211); a choice is kept while the app stays open.
    const [gpuSelection, setGpuSelection] = useKept<string | null>('monitor.gpu', null), [cpuSelection, setCpuSelection] = useState('all'), [showAll, setShowAll] = useState(false);
    const [frozen, setFrozen] = useState<{
        data: MonitorState;
        hardware: Hardware | null;
        time: string;
        modelLabel: string;
    } | null>(null);
    const snapshotRef = useRef<HTMLDivElement>(null);
    const [saving, setSaving] = useState(false), [saveError, setSaveError] = useState('');
    const [snapshotOpen, setSnapshotOpen] = useState(false), [screenshot, setScreenshot] = useState('');
    async function saveScreenshot() {
        if (!live || !snapshotRef.current || saving)
            return;
        setSnapshotOpen(true);
        setScreenshot('');
        setSaving(true);
        setSaveError('');
        flushSync(() => setFrozen({ data: structuredClone(scoped ?? live), hardware: structuredClone(inventory), time: new Date().toLocaleString(localeTag()), modelLabel: scopeLabel }));
        try {
            setScreenshot(await monitorPng(snapshotRef.current));
        }
        catch (error) {
            setSaveError(error instanceof Error ? error.message : 'Screenshot could not be saved.');
        }
        finally {
            setFrozen(null);
            setSaving(false);
        }
    }
    const data = frozen?.data ?? scoped, hardware = frozen?.hardware ?? inventory;
    if (!data)
        return <Screen title={tr("Monitor")}><p className="muted">{tr("Waiting for engine telemetry\u2026")}</p></Screen>;
    // A display adapter that has no sensors and no memory of its own (an integrated one) cannot be measured; counting it would blank every total.
    const devices = (hardware?.gpus ?? []).filter(g => g.backend !== 'vulkan' || g.vramGb > 0), processors = hardware?.processors?.length ? hardware.processors : hardware ? [{ ...hardware.cpu, id: 'cpu', socket: '' }] : [];
    const assignedDevices = selectedInstance ? devices.filter(g => selectedInstance.gpuIds.includes(gpuId(g))) : [];
    const gpuChoice = gpuSelection === 'assigned' && selectedInstance ? 'assigned' : devices.some(g => gpuId(g) === gpuSelection) ? gpuSelection! : gpuSelection === 'all' || !assignedDevices.length ? 'all' : 'assigned';
    const cpuChoice = processors.some(p => p.id === cpuSelection) ? cpuSelection : 'all';
    const selectedGpus = gpuChoice === 'assigned' ? devices.filter(g => selectedInstance?.gpuIds.includes(gpuId(g))) : gpuChoice === 'all' ? devices : devices.filter(g => gpuId(g) === gpuChoice);
    const selectedCpus = cpuChoice === 'all' ? processors : processors.filter(p => p.id === cpuChoice);
    const multipleGpus = selectedGpus.length > 1, shared = selectedGpus.length > 0 && selectedGpus.every(g => g.unified);
    const readings = data.hardware.gpus;
    const gpuValue = (key: Parameters<typeof gpuMetric>[2], mode: Parameters<typeof gpuMetric>[3] = 'sum') => gpuMetric(selectedGpus, readings, key, mode);
    const gpuSource = (key: Parameters<typeof gpuMetric>[2], mode: Parameters<typeof gpuMetric>[3] = 'sum') => hasReading(gpuValue(key, mode)) ? sourceLabel(readings?.find(r => r.id === gpuId(selectedGpus[0]))?.source) : 'Unavailable';
    const gpuLoad = gpuValue('utilizationPct', 'mean'), vram = gpuValue('memoryUsedGb'), temperature = gpuValue('temperatureC', 'max'), power = gpuValue('powerW'), powerLimit = gpuValue('powerLimitW'), pcieRx = gpuValue('pcieRxMbs');
    const vramTotal = gpuCapacity(selectedGpus);
    const singleReading = selectedGpus.length === 1 ? readings?.find(r => r.id === gpuId(selectedGpus[0])) : undefined;
    const gen = singleReading?.pcieGenMax ?? singleReading?.pcieGen, width = singleReading?.pcieWidth;
    const cpu = cpuChoice === 'all' ? data.hardware.cpuLoad : data.hardware.cpus?.find(p => p.id === cpuChoice)?.utilizationPct;
    const requests = data.requests ?? [], last = requests.find(r => r.state === 'done'), visibleRequests = showAll ? requests : requests.slice(0, 12);
    const throughput = sourceLabel(data.telemetry?.throughput), throughputKnown = throughput !== 'Unavailable';
    const writing = ['generating', 'thinking'].includes(data.phase);
    const decode = throughputKnown ? (writing ? data.outputTps : last?.tps) : null;
    const prefill = throughputKnown ? (data.phase === 'reading' || writing ? data.promptTps : last?.promptTps) : null;
    const contextKnown = ['reported', 'demo'].includes(data.telemetry?.context ?? '') || data.phase === 'reading' && data.prompt.total > 0;
    const contextUsed = data.phase === 'reading' ? data.prompt.total : data.context.usedTokens;
    const contextPct = contextKnown ? percent(contextUsed, data.context.maxTokens) : null;
    const cacheKnown = ['engine', 'demo'].includes(data.expertCache.source ?? '');
    const cacheMemory = cacheKnown ? data.expertCache.memoryGb : null;
    const cachePercent = hasReading(cacheMemory) ? percent(cacheMemory, gpuCapacity(devices)) : cacheKnown ? percent(data.expertCache.used, data.expertCache.total) : null;
    const ramPct = percent(data.hardware.ramUsedGb, data.hardware.ramTotalGb);
    const history = data.history, timestamps = history.map(s => s.t);
    const historyGpu = (key: Parameters<typeof gpuMetric>[2], fallback: 'gpuLoad' | 'vramUsedGb' | 'gpuTempC' | 'powerW' | 'pcieRxMbs', mode: Parameters<typeof gpuMetric>[3] = 'sum') => history.map(sample => sample.gpus ? gpuMetric(selectedGpus, sample.gpus, key, mode) : (sample.gpuIds ? selectedGpus.length === sample.gpuIds.length && selectedGpus.every(gpu => sample.gpuIds!.includes(gpuId(gpu))) : gpuChoice === 'all') ? sample[fallback] : null);
    const historyCpu = history.map(sample => cpuChoice === 'all' ? sample.cpuLoad : sample.cpus?.find(p => p.id === cpuChoice)?.utilizationPct);
    const chart = (label: string, values: (number | null | undefined)[], unit: string, maximum?: number, tone?: string) => <ReferenceSpark label={tr(label)} data={values} unit={unit} timestamps={timestamps} maximum={maximum} tone={tone}/>;
    const gpuName = multipleGpus ? `${selectedGpus.length} GPUs · mean utilization` : selectedGpus[0]?.name ?? (selectedInstance && !selectedInstance.gpuIds.length ? native ? 'Automatic GPU placement' : 'CPU only · no GPU assigned' : 'No GPU detected');
    const cpuName = !selectedCpus.length ? 'Processor information unavailable' : selectedCpus.length > 1 ? `${selectedCpus.length} processors · ${selectedCpus.reduce((n, p) => n + p.cores, 0)} cores · ${selectedCpus.reduce((n, p) => n + p.threads, 0)} threads` : `${selectedCpus[0]?.model ?? 'CPU'} · ${hardware?.cpu.topologyKnown === false ? '' : `${selectedCpus[0]?.cores ?? 0} cores · `}${selectedCpus[0]?.threads ?? 0} threads`;
    const phase = writing ? 'generating' : data.queue.waiting > 0 && data.phase === 'idle' ? 'queued' : data.phase;
    const progress = data.phase === 'reading' ? data.prompt.pct : writing ? percent(data.generation?.done, data.generation?.total) : 0;
    const statusDetail = data.phase === 'reading' && data.prompt.total ? `${tokens(data.prompt.done)} / ${tokens(data.prompt.total)} tokens · ${Math.round(data.prompt.pct)}%` : writing && hasReading(data.generation?.done) ? `${tokens(data.generation.done)} tokens · ${tps(data.outputTps)} tok/s` : last ? `Last: ${tokens(last.tokensOut)} tokens${hasReading(last.tps) ? ` at ${tps(last.tps)} tok/s` : ''}` : '';
    const cache = data.conversationCache;
    const reused = requests.filter(r => hasReading(r.cachedTokens));
    const lastRequest = requests.find(r => r.state === 'done');
    const lastReused = hasReading(lastRequest?.cachedTokens) ? lastRequest : null;
    const reusedTotal = reused.length ? reused.reduce((n, r) => n + (r.cachedTokens ?? 0), 0) : null;
    const lastCacheText = hasReading(cache?.lastReused) && hasReading(cache?.lastPrompt) ? `${tokens(cache.lastReused)} of ${tokens(cache.lastPrompt)} prompt tokens reused` : lastReused ? `${tokens(lastReused.cachedTokens)} of ${tokens(lastReused.tokensIn)} prompt tokens reused` : 'Reuse not reported';
    const promptSpeed = data.totals.promptMs ? data.totals.uncachedTokens / (data.totals.promptMs / 1000) : null;
    const outputSpeed = data.totals.decodeMs ? data.totals.tokensOut / (data.totals.decodeMs / 1000) : null;
    const multipleScope = selection === 'all' && (runtime?.instances.length ?? 0) > 1;
    const chooseInstance = (id: string) => { setInstanceSelection(id); setGpuSelection(null); };
    const overview = <div className="stack stack--lg monitor-capture monitor-dashboard" ref={snapshotRef}>
    {tr(frozen ? <header className="section-toolbar"><div><div className="snapshot__brand-title"><Brand variant="lockup" height={24} decorative/></div><p className="small muted">{tr(frozen.modelLabel)} · {tr(frozen.time)}</p></div></header> : null)}
    {tr(multipleModels ? <section className="st-card"><h2 className="h4">{tr("Model performance")}</h2><div className="st-tablewrap"><table className="st-table native-performance-table"><thead><tr><th>{tr("Model / device")}</th><th>{tr("Generating tok/s")}</th><th>{tr("Reading tok/s")}</th><th>{tr("State")}</th><th>{tr("Active / queued")}</th></tr></thead><tbody>{tr(runtime?.instances.map(i => { const recent = (data.requests ?? []).find(r => r.instanceId === i.id); const writing = i.phase === 'generating' || i.phase === 'thinking', reading = i.phase === 'reading'; return <tr key={i.id}><td><button className="text-btn strong" onClick={() => chooseInstance(i.id)}>{tr(instanceLabel(i))}</button><small>{tr(deviceLabel(i.gpuIds, inventory))}</small></td><td>{tr(number(writing ? i.outputTps : recent?.tps, 1))}<small>{tr(writing ? 'Now' : recent ? 'Last request' : 'No request yet')}</small></td><td>{tr(number(reading ? i.promptTps : recent?.promptTps, 0))}<small>{tr(reading ? 'Now' : recent ? 'Last request' : 'No request yet')}</small></td><td>{tr(i.phase)}{tr(runtime.mode === 'simulated' ? <small>{tr("Illustrative")}</small> : null)}</td><td>{tr(number(i.active))} / {tr(number(i.queued))}</td></tr>; }))}</tbody></table></div></section> : null)}
    {tr(multipleModels ? <p className="xs muted monitor-scope-note">{tr(selectedInstance ? selectedInstance.gpuIds.length ? 'Requests, context and speed are scoped to this model. GPU readings include all work on the selected devices.' : 'CPU-only model. Requests and context are scoped to this model; system metrics are shared host measurements.' : 'Select a model for its context and speed. GPU readings and system RAM are shared host measurements.')}</p> : null)}
    <section className="st-card monitor-state" aria-label={tr("Model state")} data-snapshot-skip>
      <div className="monitor-state__head"><h2 className="h4">{tr(multipleScope ? 'Model activity' : 'Model state')}</h2><div className="monitor-state__badges">{tr([{ id: 'idle', label: 'Idle' }, { id: 'reading', label: 'Reading' }, { id: 'generating', label: 'Generating' }, { id: 'queued', label: 'Queued' }, { id: 'error', label: 'Error' }].map(s => <span key={s.id} className="st-badge" data-phase={s.id} data-active={phase === s.id || s.id === 'queued' && data.queue.waiting > 0}>{tr(s.label)}{tr(s.id === 'queued' && data.queue.waiting > 0 ? ` (${data.queue.waiting})` : '')}</span>))}</div><div className="monitor-device-selectors">{tr(devices.length > 1 ? <Select ariaLabel={tr("Monitor GPU")} value={gpuChoice} options={[...(selectedInstance ? [{ value: 'assigned', label: 'Assigned GPUs' }] : []), { value: 'all', label: `All GPUs (${devices.length})` }, ...devices.map((gpu, i) => ({ value: gpuId(gpu), label: `GPU ${i + 1} · ${gpu.name}` }))]} menuMinWidth={340} onChange={setGpuSelection}/> : null)}{tr(processors.length > 1 ? <Select ariaLabel={tr("Monitor processor")} value={cpuChoice} options={[{ value: 'all', label: `All processors (${processors.length})` }, ...processors.map((p, i) => ({ value: p.id, label: `CPU ${i + 1} · ${p.model}` }))]} onChange={setCpuSelection}/> : null)}</div></div>
      <div className="monitor-state__reading small"><span>{tr(multipleScope ? `${number(data.queue.running)} active · ${number(data.queue.waiting)} queued` : data.phase === 'idle' ? 'Waiting for a request' : data.phaseLabel)}</span><span className="muted num">{tr(statusDetail)}</span></div>
      {tr(!multipleScope ? <Bar label={tr("Model progress")} value={progress ?? 0} indeterminate={writing && !hasReading(progress)} tone={data.phase === 'reading' ? 'info' : 'accent'}/> : null)}
    </section>
    <div className="monitor-metrics">
      {tr(!multipleScope ? <MonitorMetric label={tr("Speed")} icon="gauge" value={number(decode, 1)} unit={hasReading(decode) ? 't/s' : ''} source={throughput} sub={writing ? 'Decode now' : last ? 'Decode last request' : 'Decode'} secondary={<div><div className="st-metric__value">{tr(number(prefill))}{tr(hasReading(prefill) ? <span className="st-metric__unit">{tr("t/s")}</span> : null)}</div><div className="monitor-metric__sub xs muted">{tr(data.phase === 'reading' ? 'Prefill now' : writing ? 'Prefill this request' : last ? 'Prefill last request' : 'Prefill')}</div></div>} chart={<ReferenceSpark label={tr("Speed")} data={history.map(s => throughputKnown ? s.outputTps : null)} secondary={{ data: history.map(s => throughputKnown ? s.promptTps : null), label: 'Prefill', tone: 'var(--st-chart-prefill)' }} tone="var(--st-chart-decode)" unit="t/s" timestamps={timestamps}/>}/> : null)}
      <MonitorMetric label={tr("GPU load")} icon="gpu" value={number(gpuLoad)} unit={hasReading(gpuLoad) ? '%' : ''} source={gpuSource('utilizationPct', 'mean')} sub={gpuName} chart={chart('GPU load', historyGpu('utilizationPct', 'gpuLoad', 'mean'), '%', 100)}/>
      <MonitorMetric label={tr(shared ? 'Shared GPU memory' : 'VRAM')} icon="layers" value={number(vram, 1)} unit={hasReading(vram) ? vramTotal ? `/ ${number(vramTotal, 0)} GB` : 'GB' : ''} source={gpuSource('memoryUsedGb')} sub={shared ? 'Same memory pool as system RAM' : multipleGpus ? 'Total across selected GPUs' : cacheKnown && devices.length === 1 ? `${number(data.expertCache.used)} experts cached` : 'GPU memory'} chart={chart(shared ? 'Shared GPU memory' : 'VRAM', historyGpu('memoryUsedGb', 'vramUsedGb'), 'GB', vramTotal)}/>
      <MonitorMetric label={tr("GPU temp")} icon="thermometer" value={number(temperature)} unit={hasReading(temperature) ? '°C' : ''} source={gpuSource('temperatureC', 'max')} sub={multipleGpus ? 'Hottest selected GPU' : selectedGpus[0]?.name ?? (selectedInstance && !selectedInstance.gpuIds.length ? native ? 'Automatic GPU placement' : 'CPU only · no GPU assigned' : 'No GPU detected')} chart={chart('GPU temperature', historyGpu('temperatureC', 'gpuTempC', 'max'), '°C', 90, 'var(--st-warn)')}/>
      <MonitorMetric label={tr("Power")} icon="bolt" value={number(power)} unit={hasReading(power) ? 'W' : ''} source={gpuSource('powerW')} sub={hasReading(powerLimit) ? `of ${number(powerLimit)} W limit` : multipleGpus ? 'Total across selected GPUs' : 'GPU power draw'} chart={chart('GPU power', historyGpu('powerW', 'powerW'), 'W', powerLimit ?? undefined)}/>
      <MonitorMetric label={tr("PCIe")} icon="link" value={hasReading(gen) ? `Gen${gen}` : multipleGpus ? number(pcieRx, 1) : '—'} unit={hasReading(gen) && hasReading(width) ? `x${width}` : multipleGpus && hasReading(pcieRx) ? 'MB/s' : ''} source={hasReading(gen) ? sourceLabel(singleReading?.source) : gpuSource('pcieRxMbs')} sub={hasReading(pcieRx) ? `To GPU ${number(pcieRx, 1)} MB/s${hasReading(gen) && hasReading(singleReading?.pcieGen) && singleReading.pcieGen < gen ? ` · idle Gen${singleReading.pcieGen}` : ''}` : shared ? 'Shared memory · no PCIe link' : multipleGpus ? 'Total transfer to selected GPUs' : 'Transfer to GPU unavailable'} chart={chart('PCIe transfer', historyGpu('pcieRxMbs', 'pcieRxMbs'), 'MB/s', undefined, 'var(--st-info)')}/>
      <MonitorMetric label={tr("CPU")} icon="cpu" value={number(cpu)} unit={hasReading(cpu) ? '%' : ''} source={hasReading(cpu) ? 'Measured' : 'Unavailable'} sub={cpuName} chart={chart('CPU utilization', historyCpu, '%', 100)}/>
      <MonitorMetric label={tr("Disk read")} icon="disk" value={number(hasReading(data.hardware.diskReadGbs) ? data.hardware.diskReadGbs >= 1 ? data.hardware.diskReadGbs : data.hardware.diskReadGbs * 1024 : null, 1)} unit={hasReading(data.hardware.diskReadGbs) ? data.hardware.diskReadGbs >= 1 ? 'GB/s' : 'MB/s' : ''} source={hasReading(data.hardware.diskReadGbs) ? 'Measured' : 'Unavailable'} sub={hasReading(data.hardware.diskWriteMbs) ? `Write ${number(data.hardware.diskWriteMbs, 1)} MB/s` : 'Read throughput'} chart={chart('Disk read', history.map(s => hasReading(s.diskReadGbs) ? s.diskReadGbs * 1024 : null), 'MB/s', undefined, 'var(--st-info)')}/>
    </div>
    <div className="monitor-context-requests">
      <section className="st-card monitor-context"><h2 className="h4">{tr("Context fill")}</h2><div className="monitor-context__gauge"><Gauge value={contextPct ?? 0} label={tr(hasReading(contextPct) ? `${Math.round(contextPct)}%` : '—')} sub={contextKnown ? `${tokens(contextUsed)} / ${contextLabel(data.context.maxTokens)}` : selection === 'all' && (runtime?.instances.length ?? 0) > 1 ? 'Select a model to see context' : 'No recorded request usage'}/></div>
        <div className="monitor-bar-reading"><span>{tr("Experts in VRAM")}</span><span className="muted num">{tr(cacheKnown ? `${number(data.expertCache.used)}${hasReading(cacheMemory) ? ` · ${number(cacheMemory, 1)} GB` : ''}` : '—')}</span></div><ReadingBar label={tr("Experts in VRAM")} value={cachePercent}/>
        <div className="monitor-bar-reading"><span>{tr("System RAM")}</span><span className="muted num">{tr(hasReading(data.hardware.ramUsedGb) ? `${number(data.hardware.ramUsedGb, 1)} / ${number(data.hardware.ramTotalGb)} GB` : '—')}</span></div><ReadingBar label={tr("System RAM")} value={ramPct} tone={hasReading(ramPct) && ramPct > 92 ? 'danger' : 'accent'}/>
        <div className="monitor-bar-reading"><span>{tr("GPU temperature")}</span><span className="muted num">{tr(number(temperature))}{tr(hasReading(temperature) ? ' °C' : '')}</span></div><ReadingBar label={tr("GPU temperature")} value={temperature} tone="warn"/>
      </section>
      <section className="st-card monitor-requests"><div className="section-toolbar"><h2 className="h4">{tr("Recent requests")}</h2>{tr(requests.length > 12 ? <Btn size="sm" onClick={() => setShowAll(!showAll)}>{tr(showAll ? 'Show fewer' : `Show all (${requests.length})`)}</Btn> : null)}</div>
        <div className="st-tablewrap monitor-request-table" data-expanded={showAll}><table className="st-table"><thead><tr><th>{tr("Time")}</th><th>{tr("Status")}</th>{tr(multipleModels ? <th>{tr("Model")}</th> : null)}<th className="num">{tr("Prompt")}</th><th className="num">{tr("Reused")}</th><th className="num">{tr("Output")}</th><th className="num">{tr("Tok/s")}</th><th className="num" title={tr("Experts already in GPU memory; PCIe share is shown separately when reported.")}>{tr("VRAM hit rate")}</th><th className="num">{tr("Duration")}</th></tr></thead><tbody>{tr(visibleRequests.length ? visibleRequests.map(r => <tr key={r.id}><td>{tr(new Date(r.at).toLocaleTimeString(localeTag(), { hour: '2-digit', minute: '2-digit', second: '2-digit' }))}</td><td><span className="request-status" data-state={r.state}>{tr(requestStatus(r))}</span>{tr(r.usageSource === 'demo' ? <span className="monitor-request-source xs muted">{tr("Illustrative")}</span> : null)}</td>{tr(multipleModels ? <td><span className="xs">{tr(r.model)}</span></td> : null)}<td className="num">{tr(number(r.tokensIn))}</td><td className="num">{tr(number(r.cachedTokens))}</td><td className="num">{tr(number(r.tokensOut))}</td><td className="num">{tr(number(r.tps, 1))}</td><td className="num">{tr(hasReading(r.hitRate) ? `${number(r.hitRate * 100, 1)}%` : '—')}{tr(hasReading(r.pcieShare) && r.pcieShare > 0 ? <span className="xs muted"> +{tr(number(r.pcieShare * 100, 1))}{tr("% PCIe")}</span> : null)}</td><td className="num">{tr(r.durationMs ? duration(r.durationMs) : '—')}</td></tr>) : <tr><td colSpan={multipleModels ? 9 : 8} className="muted">{tr("No requests yet")}</td></tr>)}</tbody></table></div>
        {tr(data.totals.requests > 0 ? <p className="small muted monitor-request-totals">{requestSummary({ requests: data.totals.requests, output: outputSpeed, reading: promptSpeed, reused: data.totals.cachedTokens }, { number, tokens })}</p> : null)}
      </section>
    </div>
    <section className="st-card monitor-cache" data-snapshot-skip><div className="section-toolbar"><h2 className="h4">{tr("Conversation cache")}</h2>{tr(lastReused ? <span className="small muted">{tr("{0} of {1} requests reused prompt tokens", [number(cache?.requestsReused ?? reused.filter(r => (r.cachedTokens ?? 0) > 0).length),number(cache?.requests ?? requests.length)])}</span> : null)}</div>
      {tr(cache?.enabled ? <div className="monitor-cache-bars"><div><div className="monitor-bar-reading"><span>{tr("Parked conversations")}</span><span className="muted num">{tr(number(cache.parked))} / {tr(number(cache.slots))}</span></div><ReadingBar label={tr("Parked conversations")} value={percent(cache.parked, cache.slots)}/></div><div><div className="monitor-bar-reading"><span>{tr("Their memory (RAM)")}</span><span className="muted num">{tr(number(cache.memoryGb, 1))} / {tr(number(cache.budgetGb, 1))}{tr(" GB")}</span></div><ReadingBar label={tr("Conversation cache memory")} value={percent(cache.memoryGb, cache.budgetGb)}/></div></div> : null)}
      <dl className="monitor-cache-facts"><div><dt>{tr("Last request")}</dt><dd>{tr(lastCacheText)}</dd></div><div><dt>{tr("Reused since start")}</dt><dd>{tr(hasReading(reusedTotal) ? `${tokens(data.totals.cachedTokens)} tokens` : '—')}</dd></div>{tr(cache?.enabled ? <><div><dt>{tr("Parked / restored")}</dt><dd>{tr(number(cache.parks))} / {tr(number(cache.restores))}{tr(cache.evictions ? ` · ${number(cache.evictions)} evicted` : '')}</dd></div>{tr(cache.lastEvent ? <div><dt>{tr("Last switch")}</dt><dd>{tr(cache.lastEvent === 'parked' ? 'Parked' : 'Restored')} {tr(tokens(cache.lastTokens))}{tr(" tokens")}{tr(cache.lastAt ? ` · ${new Date(cache.lastAt * 1000).toLocaleTimeString(localeTag())}` : '')}</dd></div> : null)}</> : null)}</dl>
      <p className="small muted">{tr(cache?.enabled ? 'Parked conversations resume from their saved state. The oldest is removed when the cache fills.' : lastReused ? 'Follow-up requests can reuse the previous prompt’s cached tokens.' : 'Cache details appear when the engine reports prompt reuse.')}</p>
    </section>
    {tr(data.mcp?.servers.length ? <section className="st-card monitor-mcp"><div className="section-toolbar"><h2 className="h4">{tr("MCP servers")}</h2><span className="small muted">{tr("{0} tools · {1} of {2} servers connected", [number(data.mcp.tools),number(data.mcp.servers.filter(server => ['ready','stopped'].includes(server.state)).length),number(data.mcp.servers.length)])}</span></div>{tr(data.mcp.servers.map(server => <div key={server.name}><div className="row row--between"><strong>{server.name}</strong><span className="small muted">{tr({ ready: 'Connected', starting: 'Starting', failed: 'Failed', stopped: 'Stopped', idle: 'Waiting' }[server.state] ?? server.state)}</span></div><p className="xs muted">{server.transport}{tr(server.transport ? ' · ' : '')}{tr(server.tools.length)}{tr(" tools")}{tr(server.info?.name ? ` · ${server.info.name}${server.info.version ? ` ${server.info.version}` : ''}` : '')}</p>{tr(server.error ? <p className="small inline-error">{tr(server.error)}</p> : null)}<div className="monitor-mcp-tools">{tr(server.tools.map(tool => <span key={tool.name} title={tr(tool.description)} className="st-badge">{tool.name}</span>))}</div></div>))}</section> : null)}
  </div>;
    return <Screen width="wide" plainHeader title={tr("Monitor")} titleDetail={multipleModels ? <Select ariaLabel={tr("Monitor model")} value={selection} options={[{ value: 'all', label: `All models (${runtime?.instances.length ?? 0})` }, ...(runtime?.instances ?? []).map(i => ({ value: i.id, label: `${instanceLabel(i)} · ${deviceLabel(i.gpuIds, inventory)}` }))]} onChange={chooseInstance}/> : selectedInstance ? instanceLabel(selectedInstance) : 'No model loaded'} actions={<IconBtn name="image" title={tr(saving ? 'Preparing screenshot…' : 'Save Monitor screenshot')} disabled={saving} onClick={() => void saveScreenshot()}/>}>{tr(saveError ? <p className="inline-error" role="alert">{tr(saveError)}</p> : null)}{tr(overview)}<SnapshotDialog open={snapshotOpen} data={screenshot} pending={saving} error={saveError} onClose={() => setSnapshotOpen(false)} onRetry={() => void saveScreenshot()}/></Screen>;
}
