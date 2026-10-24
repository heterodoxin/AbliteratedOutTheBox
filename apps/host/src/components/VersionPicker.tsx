import { useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import { Icon } from './ui';
import { FloatingMenu } from './FloatingMenu';
import { cx } from '../lib/format';
import { versionNote, type CatalogVersion } from '../lib/native-catalog';
import { nvfp4Version } from '../lib/native-names';
import { VERSION_GROUPS, fitOn, groupOpen, recommendedVersions, type FitHardware, type PcPick } from '../lib/native-fit';

/** The name a version goes by in the list: the quantization, with the maker where the quantization alone is ambiguous. */
export function versionName(v: CatalogVersion, withMaker = false): string {
  switch (v.sourceFamily) {
    case 'swift': return `Swift ${v.model}`;
    case 'unsloth': return `Unsloth ${v.model}`;
    case 'nvfp4': return tr(nvfp4Version(v.model)?.name ?? v.model);
    case 'coder': return `Coder ${v.model}`;
    case 'qwen': return withMaker ? `${v.model} (GSQ-RCO)` : v.model;
    default: return v.model;
  }
}

const gb = (n: number) => `${Number.isInteger(n) ? n : n.toFixed(1)} GB`;

/**
 * The version list of the Models screen: one line per model. The bold figure is the RAM + VRAM it needs together (the n-gram table,
 * which stays on the SSD, is not in it); the download size is secondary. "Uses SSD N GB" appears only where this computer would read
 * part of the model from its SSD. The first group is a fixed list of four; the rest fold away.
 */
export function VersionPicker({ versions, selectedId, hardware, onSelect, onOwnGguf, downloaded, showRecommended = true, moreLabel, pick }: {
  versions: CatalogVersion[];
  selectedId: string;
  hardware: FitHardware | null | undefined;
  onSelect: (id: string) => void;
  onOwnGguf: () => void;
  /** The versions already on this computer (ready to load): each gets a small "Downloaded" tag. */
  downloaded?: ReadonlySet<string>;
  /** False where the recommended four are shown elsewhere (the first-run page): the menu then holds only the groups. */
  showRecommended?: boolean;
  /** What the button says while the selected version is one of the recommended four shown elsewhere. */
  moreLabel?: string;
  /** The engine's own pick for this PC (catalog.recommendation); it comes first in the Recommended group. */
  pick?: PcPick | null;
}) {
  const trigger = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  // A group is open when the person opened it; until they touch it, only the group that holds the selected version is open.
  const [opened, setOpened] = useState<Record<string, boolean>>({});
  const selected = versions.find(v => v.id === selectedId);
  const recommended = recommendedVersions(versions, pick);
  // Where the recommended four are shown outside the menu, the menu holds only the others.
  const apart = new Set(showRecommended ? [] : recommended.map(x => x.v.id));
  const rest = versions.filter(v => !apart.has(v.id));
  const groups = [
    ...VERSION_GROUPS.map(g => ({ id: g.id, title: g.title, items: rest.filter(v => g.families.includes(v.sourceFamily)) })),
    { id: 'other', title: 'Qwen3.6 models', items: rest.filter(v => !VERSION_GROUPS.some(g => g.families.includes(v.sourceFamily))) },
  ].filter(g => g.items.length);
  const row = (v: CatalogVersion, label: string, note?: string) => {
    const fit = fitOn(v, hardware);
    const on = v.id === selectedId;
    const line = versionNote(v.model);
    return <button key={`${label}-${v.id}`} role="menuitemradio" aria-checked={on} className={cx('version-row', on && 'is-on')} onClick={() => { onSelect(v.id); setOpen(false); }}>
      <span className="version-row__name"><span className="version-row__title">{tr(on ? <Icon name="check" size={14}/> : null)}<strong>{label}</strong>{tr(note ? <span className="muted"> · {tr(note)}</span> : null)}{tr(downloaded?.has(v.id) ? <span className="version-row__have">{tr("Downloaded")}</span> : null)}{tr(fit.known && fit.spillGb >= 1 ? <span className="version-row__ssd">{tr("Uses SSD {0} GB", [Math.round(fit.spillGb)])}</span> : null)}</span>{tr(line ? <small className="version-row__note muted">{line}</small> : null)}</span>
      <span className="version-row__ram">{gb(fit.memoryGb)}</span>
      <span className="version-row__dl">{gb(v.download_gb)}</span>
    </button>;
  };
  const fit = selected && !apart.has(selected.id) ? fitOn(selected, hardware) : null;
  return <>
    <button ref={trigger} type="button" className="version-trigger" aria-haspopup="menu" aria-expanded={open} aria-label={tr("Model version")} onClick={() => setOpen(v => !v)}>
      <strong>{selected && !apart.has(selected.id) ? versionName(selected, true) : tr(moreLabel ?? "Choose a version")}</strong>{tr(selected && !apart.has(selected.id) && downloaded?.has(selected.id) ? <span className="version-row__have">{tr("Downloaded")}</span> : null)}
      <span className="version-trigger__need muted">{tr(fit ? `${gb(fit.memoryGb)} RAM + VRAM` : '')}</span>
      <Icon name="chevron" size={14} className="version-trigger__chev"/>
    </button>
    {tr(open ? <FloatingMenu anchor={trigger.current} title={tr("Model version")} width={620} onClose={() => setOpen(false)}>
      <div className="version-menu">
        {tr(showRecommended ? <><div className="version-head"><span>{tr("Recommended")}</span><span>{tr("RAM + VRAM")}</span><span>{tr("Download")}</span></div>{tr(recommended.map(({ r, v }) => row(v, versionName(v, true), r.note)))}</> : <div className="version-head"><span>{tr("All other versions")}</span><span>{tr("RAM + VRAM")}</span><span>{tr("Download")}</span></div>)}
        {tr(groups.map(g => {
          const closed = !groupOpen(opened, g.id, g.items.some(v => v.id === selectedId));
          return <div className="version-group" key={g.id} data-open={!closed || undefined}>
            <button role="menuitem" className="version-group__head" aria-expanded={!closed} onClick={() => setOpened(o => ({ ...o, [g.id]: closed }))}><span>{tr(g.title)} <span className="muted">{tr(g.items.length)}</span></span><Icon name="chevron" size={14} className={cx(!closed && 'is-open')}/></button>
            {tr(!closed ? g.items.map(v => row(v, versionName(v))) : null)}
          </div>;
        }))}
        <button role="menuitem" className="version-group__head version-own" onClick={() => { setOpen(false); onOwnGguf(); }}><span>{tr("Use your own GGUF…")}</span><Icon name="folder" size={14}/></button>
      </div>
    </FloatingMenu> : null)}
  </>;
}
