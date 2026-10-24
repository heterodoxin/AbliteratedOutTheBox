import { tr } from '../lib/i18n';
import { Fragment, useEffect, useId, useRef, useState, type ReactNode } from 'react';
import type { Hardware, ModelVariant } from '../lib/types';
import { versionFit, VERSION_FIT_LABEL } from '../lib/model-versions';
import { sizeGb } from '../lib/format';
import { Icon } from './ui';
import { VersionMemoryTooltip } from './VersionMemoryTooltip';
export function VersionLadder({ versions, hardware, context, selected, recommended, onSelect, selectedDetail, detailId, fitLabel, compact = false, disabled = false, memoryNote }: {
    versions: ModelVariant[];
    hardware: Hardware | null;
    context: number;
    selected?: string;
    recommended?: string;
    onSelect: (id: string) => void;
    selectedDetail?: ReactNode;
    detailId?: string;
    fitLabel?: (version: ModelVariant) => string;
    compact?: boolean;
    disabled?: boolean;
    memoryNote?: (version: ModelVariant) => string | undefined;
}) {
    const tooltipId = useId();
    const [hint, setHint] = useState<{
        id: string;
        text: string;
        anchor: HTMLElement;
    } | null>(null);
    const closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
    function keepHint() { clearTimeout(closeTimer.current); }
    function closeHint() { keepHint(); setHint(null); }
    function showHint(id: string, text: string, anchor: HTMLElement) { keepHint(); setHint({ id, text, anchor }); }
    function delayClose() { keepHint(); closeTimer.current = setTimeout(() => setHint(null), 140); }
    useEffect(() => () => clearTimeout(closeTimer.current), []);
    return <div className={`version-ladder${compact ? ' version-ladder--compact' : ''}`} role="group" aria-label={tr("Model versions")}>{tr(versions.map(v => {
            const fit = versionFit(v, hardware, context), active = selected === v.id, note = compact ? memoryNote?.(v) : undefined;
            return <Fragment key={v.id}>
   <button className="version-rung" disabled={disabled} data-fit={fit.zone} data-state={v.state} aria-pressed={active} aria-expanded={selectedDetail ? active : undefined} aria-controls={selectedDetail ? detailId : undefined} aria-describedby={hint?.id === v.id ? tooltipId : undefined} onFocus={event => { const icon = event.currentTarget.querySelector<HTMLElement>('.version-rung__info'); if (note && icon)
                showHint(v.id, note, icon); }} onBlur={closeHint} onClick={() => onSelect(v.id)}><span className="version-rung__node">{tr(compact ? <i>{tr(active ? <Icon name="check" size={11}/> : null)}</i> : <Icon name="model-file" size={16}/>)}</span><span className="version-rung__label"><strong>{tr(v.name)}</strong><small>{tr(compact ? v.quant : `${v.quant} · ${sizeGb(v.diskGb)} · ${fitLabel ? fitLabel(v) : VERSION_FIT_LABEL[fit.zone]}`)}</small></span>{tr(compact ? <span className="version-rung__size">{tr(sizeGb(v.diskGb))}</span> : null)}<span className="version-rung__state">{tr(v.state === 'loaded' ? 'Running' : v.state === 'loading' ? 'Loading…' : v.state === 'unloading' ? 'Unloading…' : v.state === 'downloading' ? `${Math.round(v.progress?.pct ?? 0)}%` : !compact && recommended === v.id ? 'Recommended' : v.state === 'installed' ? 'Downloaded' : compact ? v.local ? v.state === 'error' ? 'Reconnect file' : 'Local file' : v.state === 'error' ? 'Retry download' : 'Not downloaded' : 'Available')}{tr(note ? <span className="version-rung__info" role="img" aria-label={tr("SSD spill information")} onPointerEnter={event => showHint(v.id, note, event.currentTarget)} onPointerLeave={delayClose}><Icon name="info" size={14}/></span> : null)}</span></button>
   {tr(active ? selectedDetail : null)}
  </Fragment>;
        }))}{tr(hint ? <VersionMemoryTooltip id={tooltipId} anchor={hint.anchor} text={hint.text} onClose={closeHint} onEnter={keepHint} onLeave={delayClose}/> : null)}</div>;
}
