import {localeTag} from './i18n';
/** Shared, deterministic UI estimates. Demo curves are never benchmark evidence. */
import type { Hardware, ModelVariant } from './types';
export type Profile = 'agentic' | 'balanced' | 'performance';
export type Workload = 'short' | 'document' | 'agent';
export const PROFILES = [
    { id: 'agentic' as const, label: 'Agentic', icon: 'tool' as const, purpose: 'Tools and long prompts' },
    { id: 'balanced' as const, label: 'Balanced', icon: 'layers' as const, purpose: 'Everyday work' },
    { id: 'performance' as const, label: 'Performance', icon: 'bolt' as const, purpose: 'Faster answers' },
];
export const WORKLOADS = [
    { id: 'short' as const, label: 'Short text', tokens: 1024 },
    { id: 'document' as const, label: 'Document text', tokens: 4096 },
    { id: 'agent' as const, label: 'Long context', tokens: 16384 },
];
export const OUTPUT_TOKENS = 256;
export type Range = {
    low: number;
    high: number;
};
export interface Execution {
    context: number;
    batch: number;
    kv: string;
    placement: string;
    threads: number;
}
export interface SpeedEvidence {
    source: 'estimated' | 'measured';
    hardware: string;
    backend: string;
    config: string;
    workload: Workload;
    cache: 'cold-prompt';
    reading: Range;
    answer: Range;
    date: string;
    attribution: string;
}
export interface Experience {
    config: Execution;
    workload: Workload;
    identity: string;
    reading: Range | null;
    answer: Range | null;
    source: 'demo' | 'estimated' | 'measured' | 'unavailable';
    provenance: string;
    memory: {
        weights: number;
        cache: number;
        runtime: number;
        total: number;
        budget: number | null;
        remaining: number | null;
        unified: boolean;
        systemBudget: number | null;
        pool: string;
        placementKnown: boolean;
    } | null;
    fit: 'comfortable' | 'tight' | 'too-large' | 'unknown' | 'unsupported';
    fitsWorkload: boolean;
}
export const DEFAULT_EXECUTION: Execution = { context: 32768, batch: 2048, kv: 'INT8', placement: 'Automatic', threads: 8 };
export const RECOMMENDED_CONTEXT_TOKENS = 128 * 1024;
export function contextTokens(value: string) { return value === 'recommended' ? RECOMMENDED_CONTEXT_TOKENS : Number.parseInt(value, 10) * 1024; }
export function execution(values: Record<string, unknown> = {}, context = 32768): Execution {
    return { context, batch: Number(values['perf.batch'] ?? 2048), kv: String(values['ctx.kvMode'] ?? 'INT8'), placement: String(values['gpu.layers'] ?? 'Automatic'), threads: Number(values['perf.threads'] ?? 8) };
}
export function hardwareKey(hw: Hardware | null) { return hw ? `${hw.cpu.model}|${hw.gpus.map(g => g.name).join('+')}|${hw.memory.totalGb}` : 'unknown'; }
export function configKey(v: ModelVariant, c: Execution) { return `${v.id}|${c.context}|${c.batch}|${c.kv}|${c.placement}|${c.threads}`; }
export function metadata(v: ModelVariant) {
    return v.catalog ?? { family: v.family.split('-')[0], version: v.family, purpose: v.blurb, capabilities: ['Text'], agentApproved: false, balancedRank: 999, review: 'Not reviewed for agent tasks' };
}
function validRange(value: Range) { return Number.isFinite(value.low) && Number.isFinite(value.high) && value.low > 0 && value.high >= value.low; }
export function estimate(v: ModelVariant | undefined, hw: Hardware | null, config: Execution, workload: Workload = 'document', demo = false): Experience {
    const needed = WORKLOADS.find(w => w.id === workload)!.tokens + OUTPUT_TOKENS;
    const result: Experience = { config, workload, identity: v ? configKey(v, config) : 'none', reading: null, answer: null, source: 'unavailable', provenance: 'No comparable speed evidence for this configuration.', memory: null, fit: 'unknown', fitsWorkload: needed <= config.context };
    if (!v || !Number.isFinite(config.context) || config.context <= 0 || !Number.isFinite(config.batch) || config.batch <= 0)
        return result;
    if (config.context > v.contextMax)
        result.fit = 'unsupported';
    if (v.kvKbPerToken !== undefined && Number.isFinite(config.context) && config.context > 0) {
        const cpu = hw?.accelerator.backend === 'cpu' || config.placement === 'CPU only';
        const unified = !!hw?.accelerator.unified;
        const placementKnown = cpu || unified || ['Automatic', 'All layers'].includes(config.placement);
        const budget = !hw ? null : cpu || unified ? hw.memory.totalGb * .72 : hw.accelerator.usableVramGb;
        const cache = v.kvKbPerToken * config.context / 1024 ** 2 * (config.kv === 'FP16' ? 2 : config.kv === 'Q4_0' ? .5 : 1);
        const runtime = v.experts > 0 ? 2.5 : 1.2;
        const total = v.diskGb + cache + runtime;
        result.memory = { weights: v.diskGb, cache, runtime, total, budget: placementKnown ? budget : null, remaining: budget !== null && placementKnown ? budget - total : null, unified, systemBudget: hw && !unified && !cpu ? hw.memory.totalGb * .72 : null, pool: cpu ? 'System memory' : unified ? 'Unified memory' : 'GPU memory', placementKnown };
        if (result.fit !== 'unsupported')
            result.fit = budget === null || !placementKnown ? 'unknown' : total > budget ? 'too-large' : total > budget * .85 ? 'tight' : 'comfortable';
    }
    if (!result.fitsWorkload || result.fit === 'unsupported') {
        result.provenance = 'The sample or context is not supported by this setup.';
        return result;
    }
    const evidence = v.performance?.find(e => e.hardware === hardwareKey(hw) && e.backend === hw?.accelerator.backend && e.config === result.identity && e.workload === workload && e.cache === 'cold-prompt' && validRange(e.reading) && validRange(e.answer));
    if (evidence) {
        result.reading = evidence.reading;
        result.answer = evidence.answer;
        result.source = evidence.source;
        result.provenance = `${evidence.attribution} · ${evidence.date} · ${evidence.backend} · uncached prompt`;
    }
    else if (demo && hw) {
        // Illustrative curves only: deliberately isolated from real evidence.
        const weight = Math.max(.5, v.diskGb);
        const batchEffect = Math.max(.65, 1 - Math.abs(Math.log2(Math.max(1, config.batch) / 2048)) * .08);
        const contextEffect = 1 / (1 + Math.max(0, config.context - 8192) / 262144);
        const reading = 1100 / Math.sqrt(weight + 1) * batchEffect * contextEffect;
        const answer = 80 / Math.sqrt(weight + 1) * contextEffect;
        result.reading = { low: Math.round(reading * .85), high: Math.round(reading * 1.15) };
        result.answer = { low: Math.max(1, Math.round(answer * .85)), high: Math.max(2, Math.round(answer * 1.15)) };
        result.source = 'demo';
        result.provenance = 'Illustrative UI fixture · not a measurement or a hardware prediction';
    }
    return result;
}
export const FIT_TEXT = { comfortable: 'Comfortable fit · estimated', tight: 'Memory may be tight', 'too-large': 'Exceeds memory budget', unknown: 'Fit unknown', unsupported: 'Context not supported' };
export function speedText(range: Range | null) { return range ? `${Math.round(range.low).toLocaleString(localeTag())}–${Math.round(range.high).toLocaleString(localeTag())}` : 'Not estimated'; }
export function durationText(tokens: number, range: Range | null) { return range ? `${(tokens / range.high).toFixed(1)}–${(tokens / range.low).toFixed(1)} s` : 'Not estimated'; }
export function sourceText(e: Experience) { return e.source === 'demo' ? 'Illustrative estimate' : e.source === 'estimated' ? 'Estimated' : e.source === 'measured' ? 'Measured' : 'Unavailable'; }
export interface Recommendation {
    unavailableReason?: 'hardware' | 'fit';
    profile: Profile;
    variant?: ModelVariant;
    experience?: Experience;
    provisional: boolean;
    explanation: string;
}
export function recommend(variants: ModelVariant[], hw: Hardware | null, config: Execution, workload: Workload, demo: boolean): Recommendation[] {
    const all = variants.map(variant => ({ variant, experience: estimate(variant, hw, config, workload, demo) }));
    const viable = all.filter(x => ['comfortable', 'tight'].includes(x.experience.fit) && x.experience.fitsWorkload);
    return PROFILES.map(({ id }) => {
        const candidates = viable.filter(x => id !== 'agentic' || metadata(x.variant).agentApproved || (demo && x.variant.experts > 0));
        const ranked = [...candidates].sort((a, b) => {
            if (id === 'balanced')
                return Number(b.experience.fit === 'comfortable') - Number(a.experience.fit === 'comfortable') || metadata(a.variant).balancedRank - metadata(b.variant).balancedRank || a.variant.id.localeCompare(b.variant.id);
            const metric = id === 'agentic' ? 'reading' : 'answer';
            return (b.experience[metric]?.low ?? -1) - (a.experience[metric]?.low ?? -1) || a.variant.id.localeCompare(b.variant.id);
        });
        const chosen = ranked[0];
        if (!chosen)
            return { profile: id, unavailableReason: !hw ? 'hardware' : 'fit', provisional: true, explanation: !hw ? 'Check your computer to assess model fit.' : id === 'agentic' ? 'No approved agent candidate fits this setup. Try Balanced or review a shorter context.' : 'No compatible model fits this context and memory budget. Review a shorter context.' };
        const provisional = chosen.experience.source !== 'measured' || id === 'balanced' && !chosen.variant.catalog?.review.startsWith('Reviewed');
        const explanation = demo ? 'Example selection using illustrative speeds and catalog preferences. Agent approval is simulated; this is not a production recommendation.' : id === 'balanced' ? `Catalog preference among fitting models. ${metadata(chosen.variant).review}.` : chosen.experience.source === 'unavailable' ? 'Provisional fit-based option; speed has not been established. No fastest-model claim.' : `Highest supported ${id === 'agentic' ? 'prompt-reading' : 'answer-speed'} lower bound among comparable, fitting candidates. ${metadata(chosen.variant).review}.`;
        return { profile: id, ...chosen, provisional, explanation };
    });
}
export function groupVariants(variants: ModelVariant[]) {
    const families = new Map<string, Map<string, ModelVariant[]>>();
    for (const v of variants) {
        const meta = metadata(v);
        const versions = families.get(meta.family) ?? new Map<string, ModelVariant[]>();
        versions.set(meta.version, [...(versions.get(meta.version) ?? []), v]);
        families.set(meta.family, versions);
    }
    return [...families].map(([family, versions]) => ({ family, versions: [...versions].map(([version, variants]) => ({ version, variants })) }));
}
/** Merge recommendations only when their exact execution identities match. */
export function groupRecommendations(items: Recommendation[]) {
    const groups = new Map<string, Recommendation[]>();
    for (const item of items) {
        const key = item.variant && item.experience ? configKey(item.variant, item.experience.config) : `missing:${item.profile}`;
        groups.set(key, [...(groups.get(key) ?? []), item]);
    }
    return [...groups.entries()].map(([identity, recommendations]) => ({ identity, recommendations }));
}
