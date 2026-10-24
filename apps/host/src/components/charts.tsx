import { tr } from '../lib/i18n';
import { useState } from 'react';
import { referencePaths } from '../lib/monitor-chart';
/** Original Monitor chart geometry, including the separate prefill line and sensor gaps. */
export function ReferenceSpark({ data, maximum, tone = 'var(--st-accent)', secondary, label, unit, timestamps }: {
    data: (number | null | undefined)[];
    maximum?: number;
    tone?: string;
    secondary?: {
        data: (number | null | undefined)[];
        label: string;
        tone: string;
    };
    label: string;
    unit: string;
    timestamps: number[];
}) {
    const [hover, setHover] = useState<number | null>(null);
    const primary = referencePaths(data, maximum), other = secondary ? referencePaths(secondary.data) : null;
    const empty = !data.some(value => typeof value === 'number' && Number.isFinite(value)) && !secondary?.data.some(value => typeof value === 'number' && Number.isFinite(value));
    const value = (values: (number | null | undefined)[]) => hover !== null && typeof values[hover] === 'number' && Number.isFinite(values[hover]) ? `${Math.round(values[hover]! * 10) / 10} ${unit}` : 'Unavailable';
    const ago = hover !== null && timestamps[hover] != null ? Math.max(0, Math.round(((timestamps.at(-1) ?? 0) - timestamps[hover]) / 1000)) : 0;
    const line = (paths: ReturnType<typeof referencePaths>['paths'], color: string) => paths.map((path, i) => <g key={i} style={{ color }}><path d={path.area} fill="currentColor" opacity=".12"/><path d={path.line} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke"/></g>);
    return <div className="spark monitor-spark" tabIndex={data.length > 1 ? 0 : undefined} role="group" aria-label={tr("{0}, last 60 seconds. Use arrow keys to inspect samples.", [label])} onMouseMove={event => {
            const rect = event.currentTarget.getBoundingClientRect();
            if (rect.width && data.length)
                setHover(Math.round(Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * (data.length - 1)));
        }} onMouseLeave={() => setHover(null)} onFocus={() => setHover(data.length - 1)} onBlur={() => setHover(null)} onKeyDown={event => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key))
                return;
            event.preventDefault();
            setHover(current => event.key === 'Home' ? 0 : event.key === 'End' ? data.length - 1 : Math.max(0, Math.min(data.length - 1, (current ?? data.length - 1) + (event.key === 'ArrowLeft' ? -1 : 1))));
        }}>
    <svg viewBox="0 0 100 32" preserveAspectRatio="none" role="img" aria-label={tr("{0}, last 60 seconds", [label])}>
 {tr(empty ? <path d="M0,28 L100,28" fill="none" stroke="var(--st-line)" strokeDasharray="3 3" strokeWidth="1" vectorEffect="non-scaling-stroke"/> : null)}
      {tr(line(primary.paths, tone))}{tr(other && secondary ? line(other.paths, secondary.tone) : null)}
      {tr(hover !== null && data.length > 1 ? <line x1={hover / (data.length - 1) * 100} x2={hover / (data.length - 1) * 100} y1="2" y2="32" stroke={tone} strokeWidth="1" opacity=".45" vectorEffect="non-scaling-stroke"/> : null)}
    </svg>
    {tr(hover !== null ? <span className="spark__readout num" role="status">{tr(secondary ? 'Decode: ' : '')}{tr(value(data))}{tr(secondary ? ` · ${secondary.label}: ${value(secondary.data)}` : '')}<span className="spark__ago"> · {tr(ago ? `${ago}s ago` : 'now')}</span></span> : null)}
  </div>;
}
/**
 * Charts — drawn to the kit's own recipes.
 *
 * `Spark` is the reference sparkline (area at 12% + 1.5px line, 100×28 box).
 * `Gauge` is the reference 270° arc: r=50, stroke 10, dasharray 235.6 / 314.2,
 * rotated 135°, solid accent stroke — no gradients.
 */
export function Spark({ data, height = 28, tone = 'var(--st-accent)', domain, axisLabel, unit, timestamps, reference = false, }: {
    data: number[];
    height?: number;
    tone?: string;
    /**
     * A shared vertical range, in the series' own units.
     *
     * Scaling each sparkline to its own min and max makes a flat 2 tok/s series
     * and a spiky 200 tok/s one draw the identical picture, which is the opposite
     * of what putting them side by side is for. Passing one domain for a group
     * makes their shapes comparable.
     */
    domain?: [
        number,
        number
    ];
    /** Spoken and hover description of the span, e.g. "last 60 seconds". */
    axisLabel?: string;
    /**
     * Names the value on hover, e.g. "tok/s". Without it the readout still
     * appears — a bare figure is more useful than nothing — but a number with no
     * unit on a chart that could be measuring anything is only half an answer.
     */
    unit?: string;
    /** Actual sample times; index spacing is used only for untimed miniatures. */
    timestamps?: number[];
    /** Match the live GitHub interface's 100×32 geometry and line weight. */
    reference?: boolean;
}) {
    const [hover, setHover] = useState<number | null>(null);
    const series = data.length > 1 ? data : [0, 0];
    const max = domain ? domain[1] : Math.max(...series);
    const min = domain ? domain[0] : Math.min(...series);
    const span = max - min || 1;
    const points = series.map((v, i) => {
        const timed = timestamps?.length === series.length && timestamps.at(-1)! > timestamps[0];
        const x = timed ? ((timestamps[i] - timestamps[0]) / (timestamps.at(-1)! - timestamps[0])) * 100 : (i / (series.length - 1)) * 100;
        const y = reference ? 30 - ((v - min) / span) * 26 : 25 - ((v - min) / span) * 22;
        return [x, y] as const;
    });
    const flat = max - min < 0.05;
    const line = points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(reference ? 2 : 1)} ${(reference ? p[1] : flat ? 26 : p[1]).toFixed(reference ? 2 : 1)}`).join('');
    const bottom = reference ? 32 : 28;
    const area = `${line}L100 ${bottom}L0 ${bottom}Z`;
    /**
     * Which sample the pointer is nearest.
     *
     * The chart is stretched to its container (`preserveAspectRatio="none"`), so
     * the pointer's fraction of the width maps directly to an index — no need to
     * read the rendered geometry back.
     */
    const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
        const r = e.currentTarget.getBoundingClientRect();
        if (r.width === 0)
            return;
        const frac = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
        setHover(points.reduce((best, point, i) => Math.abs(point[0] / 100 - frac) < Math.abs(points[best][0] / 100 - frac) ? i : best, 0));
    };
    const shown = hover !== null ? series[hover] : null;
    const secondsAgo = hover !== null ? timestamps?.[hover] !== undefined ? Math.max(0, Math.round(((timestamps.at(-1) ?? 0) - timestamps[hover]) / 1000)) : series.length - 1 - hover : 0;
    return (<div className="spark" style={{ height }} onMouseMove={onMove} onMouseLeave={() => setHover(null)} tabIndex={0} role="group" aria-label={tr("{0}. Use arrow keys to inspect samples.", [axisLabel ?? 'Chart'])} onFocus={() => setHover(series.length - 1)} onBlur={() => setHover(null)} onKeyDown={e => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key))
                return;
            e.preventDefault();
            setHover(current => e.key === 'Home' ? 0 : e.key === 'End' ? series.length - 1 : Math.max(0, Math.min(series.length - 1, (current ?? series.length - 1) + (e.key === 'ArrowLeft' ? -1 : 1))));
        }}>
      <svg viewBox={`0 0 100 ${bottom}`} preserveAspectRatio="none" style={{ width: '100%', height, color: tone }} role="img" aria-label={tr(axisLabel ?? `${series.length} samples`)}>
        {tr(flat && !reference ? null : <path d={area} fill="currentColor" opacity="0.12"/>)}
        <path d={line} fill="none" stroke="currentColor" strokeWidth={reference ? 1.6 : 1.5} strokeLinecap="round" strokeLinejoin="round" opacity={flat && !reference ? 0.35 : 1} vectorEffect="non-scaling-stroke"/>
        {tr(hover !== null && !flat ? (
        /* The marker uses the same 100×28 space as the line, so it lands on
           the sample rather than near it. */
        <line x1={points[hover][0]} x2={points[hover][0]} y1={2} y2={bottom} stroke="currentColor" strokeWidth={1} opacity={0.45} vectorEffect="non-scaling-stroke"/>) : null)}
      </svg>
      {tr(shown !== null ? (<span className="spark__readout num" role="status">
          {tr(Math.round(shown * 10) / 10)}
          {tr(unit ? ` ${unit}` : '')}
          <span className="spark__ago"> · {tr(secondsAgo === 0 ? 'now' : `${secondsAgo}s ago`)}</span>
        </span>) : null)}
    </div>);
}
const ARC = 235.6; // 270° of a r=50 circle
const CIRC = 314.2;
export function Gauge({ value, size = 150, label, sub, tone = 'var(--st-accent)', }: {
    value: number; // 0..100
    size?: number;
    label?: string;
    sub?: string;
    tone?: string;
}) {
    const clamped = Math.max(0, Math.min(100, value));
    const dash = (ARC * clamped) / 100;
    return (<div className="st-gauge" style={{ width: size, height: size }}>
      <svg viewBox="0 0 120 120" width={size} height={size} role="img" aria-label={tr(label ?? `${Math.round(clamped)}%`)}>
        <circle className="st-gauge__track" cx="60" cy="60" r="50" fill="none" strokeWidth="10" strokeLinecap="round" strokeDasharray={`${ARC} ${CIRC}`} transform="rotate(135 60 60)"/>
        <circle className="st-gauge__fill" cx="60" cy="60" r="50" fill="none" stroke={tone} strokeWidth="10" strokeLinecap="round" strokeDasharray={`${dash} ${CIRC}`} transform="rotate(135 60 60)" style={{ transition: 'stroke-dasharray 400ms var(--st-ease)', opacity: dash >= 3 ? 1 : 0 }}/>
      </svg>
      <div className="gauge__readout">
        <span style={{ fontSize: size * 0.2, fontWeight: 900, lineHeight: 1, fontVariantNumeric: 'tabular-nums' }}>
          {tr(label ?? `${Math.round(clamped)}%`)}
        </span>
        {tr(sub ? <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--st-ink-muted)' }}>{tr(sub)}</span> : null)}
      </div>
    </div>);
}
