import { tr } from '../lib/i18n';
import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { setFontFamily, useStore } from '../lib/store';
import { Btn, Select } from './ui';
export function FontChoice() {
    const value = useStore(state => state.fontFamily), [fonts, setFonts] = useState<string[]>([]), [loading, setLoading] = useState(true), [manual, setManual] = useState(false), [draft, setDraft] = useState(value);
    useEffect(() => { let active = true; void api.get<{
        fonts: string[];
    }>('/api/fonts').then(result => { if (active)
        setFonts(result.fonts); }).catch(() => { }).finally(() => { if (active)
        setLoading(false); }); return () => { active = false; }; }, []);
    return <div className="font-choice"><Select disabled={loading} translateLabels={false} ariaLabel={tr("Font family")} value={manual ? 'manual' : value} options={[{ value: '', label: tr('Strata font') }, ...([...new Set([...fonts, ...(value ? [value] : [])])].map(name => ({ value: name, label: name }))), { value: 'manual', label: tr('Use a font from your computer') }]} onChange={name => { if (name === 'manual') {
        setDraft(value);
        setManual(true);
    }
    else {
        setManual(false);
        setFontFamily(name);
    } }}/>{tr(loading ? <small className="muted">{tr("Loading installed fonts\u2026")}</small> : null)}{!loading && !fonts.length ? <small className="muted">{tr("Your browser cannot list installed fonts. Enter the name of a font installed on your computer.")}</small> : null}{tr(manual ? <div className="font-choice-manual"><input className="st-input" aria-label={tr("Font family name")} placeholder={tr("Enter a font family name")} value={draft} onChange={event => setDraft(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && draft.trim()) {
        setFontFamily(draft);
        setManual(false);
    } }}/><Btn size="sm" disabled={!draft.trim()} onClick={() => { setFontFamily(draft); setManual(false); }}>{tr("Apply font")}</Btn></div> : null)}</div>;
}
