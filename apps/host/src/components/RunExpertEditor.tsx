import { tr } from '../lib/i18n';
import { useRef, useState } from 'react';
import { Btn, Icon, Select } from './ui';
import { draftParams, readConfig, type ConfigDocument } from '../lib/config';
import type { EngineConfig, ExpertParam } from '../lib/types';
export function RunExpertEditor({ engine, config, text, onChange, onText, onRestore, onError }: {
    engine: EngineConfig;
    config: ConfigDocument | null;
    text: string;
    onChange: (config: ConfigDocument) => void;
    onText: (text: string) => void;
    onRestore: () => void;
    onError: (message: string) => void;
}) {
    const [tab, setTab] = useState('parameters');
    const [query, setQuery] = useState('');
    const [group, setGroup] = useState('all');
    const [changedOnly, setChangedOnly] = useState(false);
    const [unsupported, setUnsupported] = useState(false);
    const [descriptions, setDescriptions] = useState(false);
    const file = useRef<HTMLInputElement>(null);
    const params = config ? draftParams(engine, config) : [];
    const changed = params.filter(p => p.value !== p.recommended);
    const groups = [...new Set(params.map(p => p.group))];
    const rows = params.filter(p => p.id !== 'server.host' && (unsupported || !!p.applies && !p.applies.startsWith('Preserved')))
        .filter(p => (group === 'all' || p.group === group) && (!changedOnly || p.value !== p.recommended) && `${p.label} ${p.id} ${p.effect}`.toLowerCase().includes(query.toLowerCase()));
    function changeParam(id: string, value: ExpertParam['value']) {
        if (!config)
            return;
        const next: ConfigDocument = { ...config, beginner: { ...config.beginner }, expert: { ...config.expert, [id]: value } };
        if (id === 'ctx.length')
            next.beginner.contextLength = String(value).toLowerCase() as ConfigDocument['beginner']['contextLength'];
        if (id === 'perf.batch')
            next.batchAutomatic = false;
        onChange(next);
    }
    function download() {
        const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
        const link = document.createElement('a');
        link.href = url;
        link.download = 'strata-run-config.json';
        link.click();
        URL.revokeObjectURL(url);
    }
    return <section className="run-expert run-card" aria-label={tr("Expert settings")}>
    <div className="run-card__head"><span><Icon name="sliders" size={19}/><h2>{tr("Expert settings")}</h2></span><span className="xs muted">{tr(changed.length)}{tr(" custom values")}</span></div>
    <div className="run-expert__tabs" role="tablist" aria-label={tr("Expert editor")}>{tr(['parameters', 'json'].map(item => <button key={item} role="tab" aria-selected={tab === item} aria-controls={`run-editor-${item}`} id={`run-tab-${item}`} onClick={() => setTab(item)}>{tr(item === 'parameters' ? <><Icon name="sliders" size={15}/>{tr("Parameters")}</> : <><Icon name="file" size={15}/>{tr("JSON")}</>)}</button>))}</div>
    <div role="tabpanel" id={`run-editor-${tab}`} aria-labelledby={`run-tab-${tab}`}>
    {tr(tab === 'json' ? <div className="stack">
      <div className="row"><span className="small muted">{tr("One configuration for every control.")}</span><span className="spacer"/><Btn size="sm" onClick={download}>{tr("Export")}</Btn><Btn size="sm" onClick={() => file.current?.click()}>{tr("Import")}</Btn></div>
      <textarea className="st-input config-editor" spellCheck={false} aria-label={tr("Run configuration JSON")} value={text} onChange={e => onText(e.target.value)}/>
    </div> : <>
      <div className="run-expert__filter"><input className="st-input" type="search" aria-label={tr("Search engine parameters")} placeholder={tr("Search parameters…")} value={query} onChange={e => setQuery(e.target.value)}/><Select ariaLabel={tr("Parameter group")} value={group} options={[{ value: 'all', label: 'All groups' }, ...groups.map(g => ({ value: g, label: g }))]} onChange={setGroup}/></div>
      <div className="run-expert__options"><label><input type="checkbox" checked={changedOnly} onChange={e => setChangedOnly(e.target.checked)}/>{tr("Changed only")}</label><label><input type="checkbox" checked={descriptions} onChange={e => setDescriptions(e.target.checked)}/>{tr("Descriptions")}</label><label><input type="checkbox" checked={unsupported} onChange={e => setUnsupported(e.target.checked)}/>{tr("Unsupported")}</label></div>
      {tr(groups.filter(g => rows.some(p => p.group === g)).map(g => <section className="run-expert__group" key={g}><h3>{tr(g)}</h3>{tr(rows.filter(p => p.group === g).map(p => {
                    const disabled = !p.applies || p.applies.startsWith('Preserved');
                    return <div className="run-expert__row" key={p.id} data-changed={p.value !== p.recommended || undefined}><div><label htmlFor={`run-param-${p.id}`}>{tr(p.label)}</label><code>{p.id}</code>{tr(descriptions ? <p className="xs muted">{tr(p.effect)}</p> : null)}<p className="xs muted">{tr(disabled ? 'Not supported by this engine' : p.applies)}</p></div><div className="run-expert__control"><ExpertInput param={p} disabled={disabled} onChange={value => changeParam(p.id, value)}/><button className="run-field-reset" title={tr("Reset {0} to recommended", [tr(p.label)])} aria-label={tr("Reset {0} to recommended", [tr(p.label)])} disabled={disabled || p.value === p.recommended} onClick={() => changeParam(p.id, p.recommended)}><Icon name="refresh" size={15}/></button></div></div>;
                }))}</section>))}
      {tr(!rows.length ? <p className="small muted">{tr("No parameters match your filters.")}</p> : null)}
    </>)}
    </div>
    <div className="run-expert__foot"><Icon name="info" size={15}/><span className="xs muted">{tr("Load settings take effect when you run the model.")}</span><Btn size="sm" variant="ghost" onClick={onRestore}>{tr("Restore saved settings")}</Btn></div>
    <input ref={file} type="file" accept=".json,application/json" hidden onChange={e => {
            const selected = e.target.files?.[0];
            e.target.value = '';
            if (selected)
                void selected.text().then(value => { onChange(readConfig(value)); setTab('json'); }).catch(error => onError(error.message));
        }}/>
  </section>;
}
function ExpertInput({ param, disabled, onChange }: {
    param: ExpertParam;
    disabled: boolean;
    onChange: (value: ExpertParam['value']) => void;
}) {
    if (param.type === 'select' && param.options)
        return <Select ariaLabel={tr(param.label)} value={String(param.value)} disabled={disabled} options={param.options.map(value => ({ value, label: value }))} onChange={onChange}/>;
    if (param.type === 'toggle')
        return <button className="st-toggle" role="switch" aria-label={tr(param.label)} aria-checked={!!param.value} disabled={disabled} onClick={() => onChange(!param.value)}/>;
    return <input id={`run-param-${param.id}`} className="st-input" disabled={disabled} aria-label={tr(param.label)} inputMode={param.type === 'number' ? 'decimal' : undefined} value={String(param.value)} onChange={e => { const v = e.target.value; onChange(param.type === 'number' && v.trim() !== '' && Number.isFinite(Number(v)) ? Number(v) : v); }}/>;
}
