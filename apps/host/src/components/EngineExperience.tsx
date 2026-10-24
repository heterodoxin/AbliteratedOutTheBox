import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useEffect, useId, useRef, useState } from 'react';
import type { PointerEvent } from 'react';
import { closestStop, dialAngle, dialIndex, unwrapAngle } from '../lib/tuning-controls';
import { Btn, Dialog, Icon, Select } from './ui';
import { DEFAULT_EXECUTION, FIT_TEXT, OUTPUT_TOKENS, PROFILES, WORKLOADS, RECOMMENDED_CONTEXT_TOKENS, durationText, speedText } from '../lib/experience';
import type { Execution, Experience, Profile, Workload } from '../lib/experience';
export function ProfileChoice({ value, onChange }: {
    value: Profile;
    onChange: (profile: Profile) => void;
}) {
    return <div className="profile-choice" role="radiogroup" aria-label={tr("Model priority")}>{tr(PROFILES.map((p, i) => <button type="button" className="profile-option" role="radio" aria-checked={value === p.id} tabIndex={value === p.id ? 0 : -1} key={p.id} onClick={() => onChange(p.id)} onKeyDown={e => {
                if (!['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key))
                    return;
                e.preventDefault();
                const at = e.key === 'Home' ? 0 : e.key === 'End' ? 2 : (i + (['ArrowRight', 'ArrowDown'].includes(e.key) ? 1 : -1) + 3) % 3;
                onChange(PROFILES[at].id);
                (e.currentTarget.parentElement?.children[at] as HTMLElement)?.focus();
            }}><Icon name={p.icon} size={22}/><strong>{tr(p.label)}</strong><span>{tr(p.purpose)}</span>{tr(value === p.id ? <Icon name="check" size={16}/> : null)}</button>))}</div>;
}
function point(angle: number, r = 78) { const a = angle * Math.PI / 180; return { x: 110 + r * Math.sin(a), y: 110 - r * Math.cos(a) }; }
function angleAt(index: number, count: number) { return -135 + 270 * index / Math.max(1, count - 1); }
export function Dial({ label, hint, values, value, saved, recommended, onChange, onReset, tone = 'accent', gestureHint = true, exactEntry = true }: {
    label: string;
    hint: string;
    values: {
        value: number;
        label: string;
    }[];
    value: number;
    saved: number;
    recommended: number;
    onChange: (v: number) => void;
    onReset?: () => void;
    tone?: 'accent' | 'info';
    gestureHint?: boolean;
    exactEntry?: boolean;
}) {
    const id = useId();
    const drag = useRef<{
        angle: number;
    } | null>(null);
    const index = Math.max(0, values.findIndex(v => v.value === value));
    const [dragIndex, setDragIndex] = useState<number | null>(null);
    const [entry, setEntry] = useState(String(value));
    const [entryError, setEntryError] = useState('');
    useEffect(() => { setEntry(String(value)); setEntryError(''); }, [value]);
    const shown = dragIndex ?? index;
    function select(i: number) { onChange(values[Math.max(0, Math.min(values.length - 1, Math.round(i)))].value); }
    function track(e: PointerEvent<HTMLDivElement>, initial = false) {
        const r = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - r.left - r.width / 2, y = e.clientY - r.top - r.height / 2;
        if (Math.hypot(x, y) < r.width * .12)
            return;
        const raw = dialAngle(x, y);
        const angle = initial || !drag.current ? raw : unwrapAngle(raw, drag.current.angle);
        drag.current = { angle };
        const next = dialIndex(angle, values.length);
        setDragIndex(next);
        select(next);
    }
    function finish(e: PointerEvent<HTMLDivElement>) {
        if (!drag.current)
            return;
        track(e);
        drag.current = null;
        setDragIndex(null);
        if (e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.releasePointerCapture(e.pointerId);
    }
    function commitEntry() {
        const n = Number(entry);
        if (!Number.isFinite(n) || n <= 0) {
            setEntryError('Enter a positive number.');
            return;
        }
        const snapped = closestStop(n, values.map(v => v.value));
        onChange(snapped);
        setEntry(String(snapped));
        setEntryError('');
    }
    const pointer = angleAt(shown, values.length);
    const known = values.some(v => v.value === value);
    return <section className="tactile-control" data-tone={tone}>
    <div className="row row--between"><h3 className="h4" id={id}>{tr(label)}</h3><button type="button" className="dial-reset" onClick={() => onReset ? onReset() : onChange(recommended)} disabled={value === recommended}>{tr("Reset")}</button></div>
    <div className="dial" role="slider" aria-labelledby={id} aria-describedby={`${id}-hint`} aria-valuemin={values[0]?.value} aria-valuemax={values.at(-1)?.value} aria-valuenow={known ? value : undefined} aria-valuetext={values.find(v => v.value === value)?.label ?? `${value} custom`} tabIndex={0} data-custom={!known || undefined} data-dragging={dragIndex !== null} onPointerDown={e => {
            if (e.button !== 0)
                return;
            e.preventDefault();
            e.currentTarget.focus();
            e.currentTarget.setPointerCapture(e.pointerId);
            drag.current = { angle: angleAt(index, values.length) };
            track(e, true);
        }} onPointerMove={e => {
            if (drag.current)
                track(e);
        }} onPointerUp={finish} onPointerCancel={() => { drag.current = null; setDragIndex(null); }} onKeyDown={e => {
            if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown'].includes(e.key))
                return;
            e.preventDefault();
            select(e.key === 'Home' ? 0 : e.key === 'End' ? values.length - 1 : index + (['ArrowUp', 'ArrowRight', 'PageUp'].includes(e.key) ? 1 : -1));
        }}>
      <svg viewBox="0 0 220 220" aria-hidden="true">
        <circle className="dial-track" cx="110" cy="110" r="78" pathLength="360" strokeDasharray="270 90" transform="rotate(135 110 110)"/>
        <circle className="dial-fill" cx="110" cy="110" r="78" pathLength="360" strokeDasharray={`${270 * shown / Math.max(1, values.length - 1)} 360`} transform="rotate(135 110 110)"/>
        {tr(values.map((v, i) => { const p = point(angleAt(i, values.length), 96); const q = point(angleAt(i, values.length), v.value === recommended ? 87 : 91); return <line key={v.value} x1={p.x} y1={p.y} x2={q.x} y2={q.y} className={v.value === recommended ? 'dial-detent dial-detent--recommended' : 'dial-detent'}/>; }))}
        {tr(values.some(v => v.value === saved) ? <circle className="dial-saved" {...{ cx: point(angleAt(values.findIndex(v => v.value === saved), values.length)).x, cy: point(angleAt(values.findIndex(v => v.value === saved), values.length)).y }} r="5"/> : null)}
        <circle className="dial-face" cx="110" cy="110" r="62"/>
        <g className="dial-pointer" style={{ transform: `rotate(${pointer}deg)` }}><line x1="110" y1="53" x2="110" y2="66"/></g>
      </svg>
      <div className="dial-value"><strong>{tr(dragIndex !== null ? values[Math.round(dragIndex)].label : known ? values[index].label : value.toLocaleString(localeTag()))}</strong><span>{tr(value === recommended ? 'Recommended' : known ? 'Custom' : 'Expert value')}</span></div>
    </div>
    <div className="dial-stops" aria-hidden="true">{tr(values.map(v => <span key={v.value}>{tr(v.label)}</span>))}</div>{tr(gestureHint ? <p className="dial-caption">{tr("Turn the dial \u00B7 or use arrow keys")}</p> : null)}
    {tr(exactEntry ? <label className="dial-entry"><span>{tr("Exact value")}</span><input className="st-input" aria-label={tr("{0} exact value", [label])} aria-invalid={!!entryError} inputMode="numeric" value={entry} onChange={e => setEntry(e.target.value)} onBlur={commitEntry} onKeyDown={e => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    commitEntry();
                }
            }}/></label> : null)}
    {tr(entryError ? <p className="inline-error xs" role="alert">{tr(entryError)}</p> : null)}<p className="small muted" id={`${id}-hint`}>{tr(hint)}</p>
  </section>;
}
export function TuningDials({ config, saved = config, contextMax, batchSupported, batchAuto, onBatchAuto, onChange, showContext = true, showBatch = true, exactEntry = true }: {
    config: Execution;
    saved?: Execution;
    contextMax: number;
    batchSupported: boolean;
    batchAuto: boolean;
    onBatchAuto: (v: boolean) => void;
    onChange: (patch: Partial<Execution>) => void;
    showContext?: boolean;
    showBatch?: boolean;
    exactEntry?: boolean;
}) {
    const contexts = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576].filter(v => v <= contextMax).map(value => ({ value, label: `${value / 1024}K` }));
    const recommended = Math.min(RECOMMENDED_CONTEXT_TOKENS, contexts.at(-1)?.value ?? 4096);
    return <div className="tuning-dials">
    {tr(showContext ? contexts.length ? <Dial exactEntry={exactEntry} label={tr("Context")} hint={tr("More text uses more memory.")} values={contexts} value={config.context} saved={saved.context} recommended={recommended} onChange={context => onChange({ context })}/> : <p>{tr("No supported context length.")}</p> : null)}
    {tr(showBatch ? batchSupported ? <details className="disclosure batch-disclosure"><summary>{tr("Prompt batch \u00B7 ")}{tr(batchAuto ? 'Automatic' : `Custom: ${config.batch.toLocaleString(localeTag())}`)} <span className="accent">{tr("Adjust")}</span></summary><div><Dial gestureHint={false} label={tr("Prompt batch")} hint={tr("Larger batches can use more memory; they are not always faster.")} tone="info" values={[256, 512, 1024, 2048, 4096].map(value => ({ value, label: value.toLocaleString(localeTag()) }))} value={config.batch} saved={saved.batch} recommended={DEFAULT_EXECUTION.batch} onReset={() => onBatchAuto(true)} onChange={batch => { onBatchAuto(false); onChange({ batch }); }}/><label className="check-label batch-auto"><input type="checkbox" checked={batchAuto} onChange={e => { onBatchAuto(e.target.checked); }}/>{tr("Automatic \u00B7 recommended 2,048")}</label></div></details> : <div className="tactile-control"><h3 className="h4">{tr("Prompt batch")}</h3><p className="muted">{tr("This adapter cannot apply prompt batch settings.")}</p></div> : null)}
  </div>;
}
export function WorkloadChoice({ value, context, onChange }: {
    value: Workload;
    context: number;
    onChange: (w: Workload) => void;
}) {
    return <Select ariaLabel={tr("Sample workload")} value={value} options={WORKLOADS.map(w => ({ value: w.id, label: `${w.label} · ${w.tokens / 1024}K tokens${w.tokens + OUTPUT_TOKENS > context ? ' · needs more context' : ''}`, disabled: w.tokens + OUTPUT_TOKENS > context }))} onChange={onChange}/>;
}
export function SpeedPair({ experience: e }: {
    experience: Experience;
}) {
    return <div className="speed-summary" title={tr(e.provenance)} aria-label={tr("Reading and answer speeds. {0}.", [e.source === 'demo' ? tr('Illustrative estimates') : e.source])}><div className="speed-pair">{tr([{ label: 'Reading', value: e.reading, tone: 'info' }, { label: 'Answer', value: e.answer, tone: 'accent' }].map(m => <div key={m.label} data-tone={m.tone}><span className="small muted">{tr(m.label)}</span><strong className="speed-value">{tr(speedText(m.value))}{tr(m.value ? <small>{tr(" tok/s")}</small> : null)}</strong></div>))}</div></div>;
}
export function MemoryBudget({ experience: e, detailed = false }: {
    experience: Experience;
    detailed?: boolean;
}) {
    const m = e.memory;
    if (!m)
        return <p className="small muted">{tr("Memory \u00B7 Not estimated \u00B7 ")}{tr(FIT_TEXT[e.fit])}</p>;
    const max = Math.max(m.total, m.budget ?? m.total);
    return <div className="memory-budget"><div className="row row--between"><strong className="small">{tr(m.pool)}</strong><span className="xs muted">{tr("Estimated")}</span></div><div className="memory-summary"><strong>{tr(m.total.toFixed(1))} <small>{tr("GB required")}</small></strong><span>{tr(m.remaining === null ? 'Room remaining unknown' : m.remaining >= 0 ? `${m.remaining.toFixed(1)} GB remaining` : `${(-m.remaining).toFixed(1)} GB over budget`)}</span></div>
    <div className="memory-strip" role="img" aria-label={tr("Model {0} GB, context {1} GB, runtime {2} GB. {3}", [m.weights.toFixed(1), m.cache.toFixed(1), m.runtime.toFixed(1), FIT_TEXT[e.fit]])}>
      <span data-part="weights" style={{ flex: m.weights / max }}/><span data-part="cache" style={{ flex: m.cache / max }}/><span data-part="runtime" style={{ flex: m.runtime / max }}/><span data-part="free" style={{ flex: Math.max(0, m.remaining ?? 0) / max }}/>
    </div><p className="small muted">{tr(FIT_TEXT[e.fit])}</p>
    {tr(detailed ? <><div className="memory-legend">{tr(['Model weights', 'Context/cache', 'Runtime', 'Remaining'].map((l, i) => <span key={l}><i data-part={['weights', 'cache', 'runtime', 'free'][i]}/>{tr(l)}</span>))}</div><p className="small muted">{tr(m.unified ? 'One shared pool; CPU and GPU use the same memory.' : !m.placementKnown ? 'Partial offload: allocation between GPU and system memory is not estimated.' : m.systemBudget !== null ? `System budget ${m.systemBudget.toFixed(1)} GB separately; additional host use is not estimated.` : 'System memory budget reserves room for the OS.')}</p></> : null)}
  </div>;
}
export function ExperienceDetails({ experience: e }: {
    experience: Experience;
}) {
    const tokens = WORKLOADS.find(w => w.id === e.workload)!.tokens;
    return <div className="experience-details-content"><ol className="experience-path"><li><span>{tr("Read prompt")}<strong>{tr(e.fitsWorkload ? durationText(tokens, e.reading) : 'Context too short')}</strong></span></li><li><span>{tr("Start answering")}<strong>{tr("Not estimated")}</strong></span></li><li><span>{tr("Generate")}<strong>{tr(e.fitsWorkload ? durationText(OUTPUT_TOKENS, e.answer) : 'Unavailable')}</strong></span></li></ol>
    <MemoryBudget experience={e} detailed/><p>{tr(e.provenance)}</p><p>{tr(tokens.toLocaleString(localeTag()))}{tr(" input \u00B7 256 output \u00B7 ")}{tr(e.config.context / 1024)}{tr("K context \u00B7 batch ")}{tr(e.config.batch.toLocaleString(localeTag()))} · {tr(e.config.kv)}{tr(" cache \u00B7 ")}{tr(e.config.placement)} · {tr(e.config.threads)}{tr(" CPU threads.")}</p><p>{tr("Memory is a catalog approximation, not a free-memory measurement. Batch workspace and partial offload are not predicted. Reasoning and scheduling time are not estimated; these durations are not total task time.")}</p></div>;
}
export function PerformancePreview({ experience: e, saved, updating = false }: {
    experience: Experience;
    saved?: Experience;
    workload: Workload;
    onWorkload: (v: Workload) => void;
    updating?: boolean;
    compact?: boolean;
}) {
    const delta = e.memory && saved?.memory ? e.memory.total - saved.memory.total : 0;
    return <section className="performance-preview" aria-label={tr("Expected experience")}>
    <h2 className="h4">{tr("Expected experience")}</h2>
    <SpeedPair experience={e}/>
    {tr(!e.fitsWorkload ? <p className="inline-error small">{tr("Sample exceeds context, including 256 output tokens. Choose shorter text or increase context.")}</p> : null)}
    <MemoryBudget experience={e}/>
    {tr(updating || Math.abs(delta) > .005 ? <p className="small muted" aria-live="polite">{tr(updating ? 'Checking draft…' : `Draft: ${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(2)} GB estimated`)}</p> : null)}
  </section>;
}
export function DemoTest({ open, onClose, experience }: {
    open: boolean;
    onClose: () => void;
    experience: Experience;
}) {
    const status = useRef<HTMLParagraphElement>(null);
    const [stage, setStage] = useState(0);
    const [running, setRunning] = useState(false);
    useEffect(() => {
        if (open) {
            setStage(0);
            setRunning(false);
        }
    }, [open]);
    useEffect(() => {
        if (!open || !running || stage >= 4)
            return;
        const timer = setTimeout(() => setStage(s => s + 1), 900);
        return () => clearTimeout(timer);
    }, [open, running, stage]);
    useEffect(() => {
        if (open && running && (stage === 0 || stage === 4))
            status.current?.focus({ preventScroll: true });
    }, [open, running, stage]);
    const demoOnly = experience.source === 'demo';
    return <Dialog open={open} title={tr("Speed test preview")} onClose={onClose} actions={<><Btn onClick={() => { setRunning(false); onClose(); }}>{tr(running && stage < 4 ? 'Cancel preview' : 'Close')}</Btn><Btn variant="primary" disabled={!experience.fitsWorkload || running && stage < 4} onClick={() => { setStage(0); setRunning(true); }}>{tr(stage === 4 ? 'Replay preview' : 'Start preview')}</Btn></>}>
    <div className="stack"><p className="muted">{tr("Explore the test flow. Preview values are never stored as measured performance or used as real recommendation evidence.")}</p>
    <ol className="test-stages">{tr(['Warm up', 'Read sample prompt', 'Generate sample answer', 'Review results'].map((label, i) => <li key={label} data-state={running && stage > i ? 'done' : running && stage === i ? 'active' : 'pending'}><span>{tr(running && stage > i ? <Icon name="check" size={16}/> : i + 1)}</span>{tr(label)}</li>))}</ol>
    <p ref={status} tabIndex={-1} role="status" className="small">{tr(!running ? 'Ready to preview' : stage === 4 ? 'Preview complete. No measurement was taken.' : tr('Preview: {0}…', [tr(['warming up', 'reading', 'generating', 'preparing results'][stage])]))}</p>
    {tr(stage === 4 ? demoOnly ? <SpeedPair experience={experience}/> : <p className="small muted">{tr("The flow is ready. Speeds remain unmeasured until a local benchmark adapter is connected.")}</p> : null)}</div>
  </Dialog>;
}
