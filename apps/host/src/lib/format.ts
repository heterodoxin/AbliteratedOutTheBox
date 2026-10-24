import {localeTag} from './i18n';
/** Small formatting helpers used across screens. */
export const gb = (n: number, digits = 1) => `${n.toLocaleString(localeTag(), { minimumFractionDigits: digits, maximumFractionDigits: digits })} GB`;
/** Human file size from a KB value. */
export const fileSize = (kb: number) => {
    if (kb < 1)
        return `${Math.max(1, Math.round(kb * 1024))} B`;
    if (kb < 1024)
        return `${kb.toFixed(kb < 10 ? 1 : 0)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
};
/** Human size for model files: keeps one decimal below 1 GB. */
export const sizeGb = (n: number) => (n < 1 ? `${n.toFixed(1)} GB` : `${Math.round(n)} GB`);
export const gbShort = (n: number) => n >= 100 ? `${Math.round(n)}` : `${n.toFixed(n < 10 ? 1 : 0)}`;
export const pct = (n: number, digits = 0) => `${n.toFixed(digits)}%`;
export const num = (n: number) => n.toLocaleString(localeTag());
export const tps = (n: number) => n >= 1000 ? `${Math.round(n).toLocaleString(localeTag())}` : n.toFixed(1);
export function clockTime(iso: string | number | Date): string {
    const d = new Date(iso);
    return d.toLocaleTimeString(localeTag(), { hour: '2-digit', minute: '2-digit', hour12: false });
}
export function relTime(iso: string | number | Date): string {
    const then = new Date(iso).getTime();
    const diff = Date.now() - then;
    if (diff < 45000)
        return 'just now';
    const mins = Math.round(diff / 60000);
    if (mins < 60)
        return `${mins}m ago`;
    const hours = Math.round(mins / 60);
    if (hours < 24)
        return `${hours}h ago`;
    const days = Math.round(hours / 24);
    if (days < 7)
        return `${days}d ago`;
    return new Date(iso).toLocaleDateString(localeTag(), { month: 'short', day: 'numeric' });
}
export function duration(ms: number): string {
    if (ms < 1000)
        return `${Math.round(ms)} ms`;
    if (ms < 60000)
        return `${(ms / 1000).toFixed(1)} s`;
    const s = Math.round(ms / 1000);
    return `${Math.floor(s / 60)}m ${s % 60}s`;
}
export function uptime(sec: number): string {
    if (sec < 60)
        return `${Math.round(sec)}s`;
    const m = Math.floor(sec / 60);
    if (m < 60)
        return `${m}m ${Math.round(sec % 60)}s`;
    const h = Math.floor(m / 60);
    if (h < 24)
        return `${h}h ${m % 60}m`;
    return `${Math.floor(h / 24)}d ${h % 24}h`;
}
export function tokens(n: number | null | undefined): string {
    if (n == null || !Number.isFinite(n))
        return '—';
    if (n >= 1000000)
        return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 10000)
        return `${Math.round(n / 1000)}k`;
    if (n >= 1000)
        return `${(n / 1000).toFixed(1)}k`;
    return `${Math.round(n)}`;
}
export const contextLabel = (max: number) => (max >= 1024 ? `${Math.round(max / 1024)}K` : `${max}`);
/**
 * Whether a model can actually run on this computer.
 *
 * A variant carries its requirement as a RAM class range; the machine has one
 * figure. Turning that into three plain words is the whole job of the Models
 * screen — "48 GB RAM class" asks the reader to know their own memory and do the
 * comparison themselves, which is precisely what someone who finds computers
 * intimidating cannot do.
 */
export type Fit = 'good' | 'tight' | 'too-large';
export function fitFor(ramClass: {
    minGb: number;
    maxGb: number;
}, memoryGb: number): Fit {
    if (memoryGb >= ramClass.maxGb)
        return 'good';
    if (memoryGb >= ramClass.minGb)
        return 'tight';
    return 'too-large';
}
export const FIT_LABEL: Record<Fit, string> = {
    good: 'Likely to fit (estimate)',
    tight: 'Memory may be tight (estimate)',
    'too-large': 'Exceeds estimated memory budget',
};
/**
 * A rough wall-clock estimate for fetching a model.
 *
 * Downloads dominate the time the setup wizard and the Models screen are asking
 * about, and both need to give the same answer — a figure that differs between
 * two screens describing the same download is worse than no figure. Kept here so
 * there is one formula rather than two.
 *
 * Deliberately coarse: it is rounded to whole minutes and floored at 2, because
 * a precise-looking number from an unknown connection is a false promise.
 */
export function downloadMinutes(gb: number): number {
    return Math.max(2, Math.round(1 + gb * 1.2));
}
export function clamp(n: number, min = 0, max = 1): number {
    return Math.min(max, Math.max(min, n));
}
export function cx(...parts: (string | false | null | undefined)[]): string {
    return parts.filter(Boolean).join(' ');
}
/** Identity comes from the host; never substitute a different real model. */
export function formatModelName(model: {
    family: string;
    quant: string;
    name?: string;
    local?: unknown;
} | null | undefined): string {
    if (!model)
        return 'No model loaded';
    if (model.local)
        return `${model.name ?? model.family} · ${model.quant}`;
    const version = ({ Q2_K: 'Smallest', Q3_K_M: 'Compact', Q4_K_M: 'Balanced', Q5_K_M: 'High quality', Q6_K: 'Very high', Q8_0: 'Near-original', BF16: 'Original' } as Record<string, string>)[model.quant] ?? model.name;
    return `${model.family} · ${version ? `${version} (${model.quant})` : model.quant}`;
}
export function userError(message: string): string {
    return message.replace(/^\s*Fixture:\s*/i, '').replace(/^./, c => c.toUpperCase());
}
