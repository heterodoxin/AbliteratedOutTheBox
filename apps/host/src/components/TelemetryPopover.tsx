import { tr } from '../lib/i18n';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Icon, type IconName } from './ui';
import { contextTokens } from '../lib/experience';
import { ContextRing } from './ContextRing';
import { Popover } from './Popover';
import { chatMetrics, compactTokens, elapsed } from '../lib/chat-metrics';
import { useStore } from '../lib/store';
import { Num } from './Num';
import { useNative } from '../lib/native';
import type { Session } from '../lib/types';
function Chip({ icon, visual, label, live, title, children }: {
    icon?: IconName;
    /** The same words with the number as a moving element (the plain `label` stays for assistive technology). */
    live?: ReactNode;
    visual?: ReactNode;
    label: string;
    title: string;
    children: ReactNode;
}) {
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLButtonElement>(null);
    return <><button ref={ref} className="telemetry__chip" aria-label={tr("{0} \u2014 show details", [title])} aria-description={tr(label)} aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen(v => !v)}>{tr(visual ?? (icon ? <Icon name={icon} size={14}/> : null))}{live ?? tr(label)}</button>{tr(open ? <Popover anchor={ref.current} title={tr(title)} onClose={() => setOpen(false)} above>{tr(children)}</Popover> : null)}</>;
}
function Row({ label, value }: {
    label: string;
    value: ReactNode;
}) { return <div className="telepop__row"><span className="telepop__label">{tr(label)}</span><span className="telepop__value num">{tr(value)}</span></div>; }
export function ChatTelemetry({ session }: {
    session: Session | null;
}) {
    const monitor = useStore(s => s.monitor);
    const engine = useStore(s => s.engine);
    // The context shown is the one the loaded model runs with (the engine's own --max-context), not a catalog or settings default.
    const native = useNative();
    const running = native?.instances.find(i => i.id === native.defaultId && i.status === 'ready') ?? native?.instances.find(i => i.status === 'ready');
    const runningContext = running ? Number(running.metrics?.engine?.max_context) || running.context || undefined : undefined;
    // While a reply is being written the durations keep counting once a second; the readings come from the Monitor's own stream.
    const [now, setNow] = useState(() => Date.now());
    const working = !!session?.messages.some(x => x.streaming);
    useEffect(() => { if (!working) return; setNow(Date.now()); const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t); }, [working]);
    const m = chatMetrics(session, monitor, runningContext ?? (engine ? contextTokens(engine.beginner.contextLength) : undefined), now);
    const speed = m.speed;
    if (!m.hasReplies)
        return null;
    const int = (n: number) => String(Math.round(n));
    const PLACE = '';
    /** A translated sentence with its number as a moving element: the words around the number keep the language's own order. */
    const around = (template: string, before: string, node: ReactNode) => { const [head, tail = ''] = tr(template, [before, PLACE]).split(PLACE); return <>{head}{node}{tail}</>; };
    const approx = m.estimated ? tr('≈ ') : tr('');
    const dur = (n: number) => elapsed(n);
    return <div className="telemetry xs muted" aria-label={tr("Chat statistics")}>
    <Chip icon="gauge" label={tr("{0}{1} tok/s{2}", [approx, speed === null ? tr('—') : Math.round(speed), m.streaming ? tr('') : tr(' · last reply')])} live={around(m.streaming ? "{0}{1} tok/s" : "{0}{1} tok/s · last reply", approx, <Num value={speed} format={int}/>)} title={tr("Session statistics")}>{tr(m.estimated ? <p className="xs muted">{tr("Live estimate from response length and elapsed time. Final usage comes from the engine.")}</p> : null)}<Row label={tr("Thinking time")} value={<Num value={m.thinkMs} format={dur}/>}/><Row label={tr("Writing time")} value={<Num value={m.writeMs} format={dur}/>}/><Row label={tr("Tool time")} value={<Num value={m.toolMs} format={dur}/>}/><Row label={tr("Average time to first token")} value={m.ttft === null ? tr('Not reported') : <Num value={m.ttft} format={dur}/>}/><Row label={tr("Latest response speed")} value={speed === null ? tr('Not reported') : <>{<Num value={speed} format={int}/>}{tr(' tokens/s')}</>}/><Row label={tr("Prompt reading")} value={m.promptSpeed === null ? tr('Not reported') : <>{<Num value={m.promptSpeed} format={int}/>}{tr(' tokens/s')}</>}/><Row label={tr("Turns / tool calls")} value={<><Num value={m.turns} format={int}/> / <Num value={m.steps} format={int}/></>}/></Chip>
    <Chip icon="memory" label={tr("{0}{1} total tokens", [approx, compactTokens(m.total)])} live={around("{0}{1} total tokens", approx, <Num value={m.total} format={compactTokens}/>)} title={tr("Token usage")}><Row label={tr("Total in this chat")} value={<Num value={m.total} format={compactTokens}/>}/><Row label={tr("Input tokens")} value={<Num value={m.input} format={compactTokens}/>}/><Row label={tr("Output tokens")} value={<Num value={m.output} format={compactTokens}/>}/></Chip>
    <Chip visual={<ContextRing percent={m.pct}/>} label={tr(m.pct === null ? '—' : `${Math.round(m.pct)}% of context`)} live={m.pct === null ? undefined : <><Num value={m.pct} format={int}/>{tr('% of context')}</>} title={tr("Context length")}><Row label={tr("Context used")} value={m.pct === null ? tr('Not reported') : <><Num value={m.pct} format={int}/>%</>}/><div className="context-meter" role="meter" aria-label={tr("Context window used")} aria-valuemin={0} aria-valuemax={100} aria-valuenow={m.pct ?? undefined}><span style={{ width: `${m.pct ?? 0}%` }}/></div><Row label={tr("Current context")} value={<Num value={m.used} format={compactTokens}/>}/><Row label={tr("Context limit")} value={<Num value={m.max} format={compactTokens}/>}/><Row label={tr("Space remaining")} value={m.used !== null && m.max !== null ? <Num value={Math.max(0, m.max - m.used)} format={compactTokens}/> : '—'}/></Chip>
  </div>;
}
