import { tr } from '../lib/i18n';
import { useRef, useState } from 'react';
import { Btn, Dialog, Icon } from './ui';
import { FloatingMenu } from './FloatingMenu';
type Preset = {
    id: string;
    name: string;
    values: Record<string, any>;
};
const key = 'strata.native-presets.v1';
export function NativePresets({ values, onApply }: {
    values: Record<string, any>;
    onApply: (value: Record<string, any>) => void;
}) {
    const [items, setItems] = useState<Preset[]>(() => { try {
        const parsed = JSON.parse(localStorage.getItem(key) ?? '[]');
        return Array.isArray(parsed) ? parsed.filter(p => p && typeof p.id === 'string' && typeof p.name === 'string' && p.values && typeof p.values === 'object').slice(-50) : [];
    }
    catch {
        return [];
    } }), [open, setOpen] = useState(false), [save, setSave] = useState(false), [name, setName] = useState(''), [error, setError] = useState('');
    const trigger = useRef<HTMLButtonElement>(null);
    function persist(next: Preset[]) { try {
        localStorage.setItem(key, JSON.stringify(next));
        setItems(next);
        return true;
    }
    catch {
        setError('Browser storage is full. Free space before saving settings.');
        return false;
    } }
    return <><button className="st-btn st-btn--secondary" ref={trigger} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(v => !v)}><Icon name="layers"/>{tr("Saved settings ")}<Icon name="chevron" size={12}/></button>{tr(open ? <FloatingMenu anchor={trigger.current} title={tr("Saved settings")} width={290} onClose={() => setOpen(false)}><p className="model-picker-heading">{tr("Saved settings")}</p>{tr(items.map(item => <button className="menu__item" role="menuitem" key={item.id} onClick={() => { onApply(item.values); setOpen(false); }}><Icon name="sliders"/><span>{item.name}<small>{tr(item.values.model)} · {tr((item.values.context ?? 32768) / 1024)}{tr("K")}</small></span></button>))}{tr(!items.length ? <p className="model-picker-empty">{tr("Save a setup to use it again.")}</p> : null)}<button className="menu__item" role="menuitem" onClick={() => { setOpen(false); setSave(true); setName(''); }}><Icon name="plus"/>{tr("Save current settings\u2026")}</button></FloatingMenu> : null)}<Dialog open={save} compact title={tr("Saved settings")} onClose={() => setSave(false)} actions={<><Btn onClick={() => setSave(false)}>{tr("Done")}</Btn><Btn variant="primary" disabled={!name.trim()} onClick={() => { if (persist([...items, { id: crypto.randomUUID(), name: name.trim(), values }]))
        setSave(false); }}>{tr("Save")}</Btn></>}><label className="input-field"><span>{tr("Name")}</span><input className="st-input" value={name} maxLength={80} onChange={e => setName(e.target.value)} placeholder={tr("Everyday")}/></label>{tr(items.map(item => <div className="row row--between" key={item.id}><span>{item.name}</span><Btn size="sm" variant="ghost" onClick={() => persist(items.filter(i => i.id !== item.id))}>{tr("Remove")}</Btn></div>))}{tr(error ? <p role="alert" className="small inline-error">{tr(error)}</p> : null)}</Dialog></>;
}
