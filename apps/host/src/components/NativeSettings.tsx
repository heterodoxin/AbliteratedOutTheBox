import { tr } from '../lib/i18n';
import {nativeSettingsResponse,nativeSettingsValues} from '../lib/native-settings-response';
import { describeChange, type TunedChange } from '../lib/tune-values';
import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { refreshNative, useNative, type NativeField } from '../lib/native';
import { Btn, ErrorNotice, Select } from './ui';
import { NativeMcpEditor } from './NativeMcpEditor';
export function NativeSettings({ model }: {
    model: string;
}) {
    const native = useNative();
    const [revision,setRevision]=useState(0);
    const [tuned, setTuned] = useState<{ preset: { name: string } | null; changed: TunedChange[] } | null>(null);
    const [values, setValues] = useState<Record<string, any>>({}), [mcp, setMcp] = useState('{}'), [original, setOriginal] = useState(''), [text, setText] = useState(''), [tab, setTab] = useState('fields'), [pending, setPending] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState(''), [query, setQuery] = useState('');
    useEffect(() => {
        let active = true;
        setOriginal('');
        setValues({});
        setText('');
        setMcp('{}');
        setError('');
        setNotice('');
        void api.get<{
            values: Record<string, any>;
            mcp: Record<string, any>;
            tuned?: { preset: { name: string } | null; changed: TunedChange[] } | null;
        }>(`/api/native/models/${model}/settings`).then(input => {
            if (!active)
                return;
            setTuned(input.tuned ?? null);
            const value = nativeSettingsResponse(input);
            setValues(value.values);
            setText(JSON.stringify(value.values, null, 2));
            setMcp(JSON.stringify(value.mcp ?? {}, null, 2));
            setOriginal(JSON.stringify({ values: value.values, mcp: value.mcp ?? {} }));
        }).catch(e => {
            if (active)
                setError(e.message);
        });
        return () => { active = false; };
    }, [model,revision]);
    const change = (id: string, value: any) => { const next = { ...values, [id]: value }; setValues(next); setText(JSON.stringify(next, null, 2)); setNotice(''); };
    async function save() {
        setPending(true);
        setError('');
        try {
            const parsed = nativeSettingsValues(tab === 'json' ? JSON.parse(text) : values);
            const tools = JSON.parse(mcp);
            const input = await api.patch<{
                values: Record<string, any>;
                mcp: Record<string, any>;
            }>(`/api/native/models/${model}/settings`, { values: parsed, mcp: tools });
            const result = nativeSettingsResponse(input);
            setValues(result.values);
            setMcp(JSON.stringify(result.mcp, null, 2));
            setText(JSON.stringify(result.values, null, 2));
            setOriginal(JSON.stringify({ values: result.values, mcp: result.mcp }));
            setNotice('Saved. Load the model again to apply these settings. A backup of the previous configuration was kept.');
            await refreshNative(true);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    function chooseTab(next: string) {
        if (tab === 'json') {
            try {
                setValues(nativeSettingsValues(JSON.parse(text)));
            }
            catch {
                setError('Fix the JSON before changing tabs.');
                return;
            }
        }
        setTab(next);
    }
    const dirty = (() => {
        try {
            return original !== JSON.stringify({ values: tab === 'json' ? JSON.parse(text) : values, mcp: JSON.parse(mcp) });
        }
        catch {
            return true;
        }
    })();
    if (!original) return <details className="native-settings" open={!!error}>
      <summary>{tr('Native engine settings')}</summary>
      {error ? <><ErrorNotice message={tr('Couldn’t load model settings.')} next={tr('Check that the engine is connected and this model is installed, then try again.')} onRetry={() => setRevision(value => value + 1)}/><details className="native-settings-diagnostic"><summary className="small">{tr('Details')}</summary><p className="small muted">{error}</p></details></> : <p className="small muted" role="status">{tr('Reading saved settings…')}</p>}
    </details>;
    const rows = (native?.fields ?? []).filter(f => `${tr(f.label)} ${f.id} ${tr(f.group)}`.toLowerCase().includes(query.toLowerCase()));
    return <details className="native-settings"><summary>{tr("Native engine settings")}</summary>{tr(tuned?.changed?.length ? <div className="native-tuned"><p className="small"><strong>{tr("Your tuning '{0}' sets", [tuned.preset?.name ?? ''])}</strong></p><ul>{tuned.changed.map(c => { const d = describeChange(c); return <li key={c.flag}><span>{d.name}</span><b>{d.now || tr('off')}</b>{d.was !== null ? <small className="muted">{tr("was {0}", [d.was])}</small> : null}</li>; })}</ul></div> : null)}<p className="small muted">{tr(!original ? 'Reading saved settings… ' : '')}{tr("Settings for this model. Blank values use Strata\u2019s defaults.")}</p><div className="editor-tabs" role="tablist" aria-label={tr("Native settings editor")}>{tr(['fields', 'json', 'tools'].map(t => <button key={t} role="tab" aria-selected={tab === t} onClick={() => chooseTab(t)}>{tr(t === 'fields' ? 'Parameters' : t === 'json' ? 'JSON' : 'MCP tools')}</button>))}</div>
  {tr(tab === 'fields' ? <><input className="st-input" type="search" aria-label={tr("Search native parameters")} placeholder={tr("Search parameters…")} value={query} onChange={e => setQuery(e.target.value)}/>{tr(['Responses', 'GPU & memory', 'Serving', 'Image reading', 'Tools', 'Recovery'].map(group => rows.some(f => f.group === group) ? <section className="native-settings-group" key={group}><h3 className="h4">{tr(group)}</h3>{tr(rows.filter(f => f.group === group).map(f => <label className="native-setting-row" key={f.id}><span><strong>{tr(f.label)}</strong>{tr(f.supported === false ? <small>{tr("Requires a newer Strata installation.")}</small> : f.hint ? <small>{tr(f.hint)}</small> : null)}</span><NativeFieldInput field={f} value={values[f.id]} onChange={v => change(f.id, v)}/></label>))}</section> : null))}<section className="native-settings-group"><h3 className="h4">{tr("Sampling")}</h3>{tr([['temperature', 'Temperature'], ['top_p', 'Top-p'], ['top_k', 'Top-k'], ['min_p', 'Min-p'], ['repetition_penalty', 'Repetition penalty'], ['presence_penalty', 'Presence penalty'], ['frequency_penalty', 'Frequency penalty'], ['penalty_last_n', 'Penalty window'], ['seed', 'Seed']].map(([id, label]) => <label className="native-setting-row" key={id}><span><strong>{tr(label)}</strong></span><input className="st-input" aria-label={tr("Native {0}", [label])} inputMode="decimal" value={values.sampling?.[id] ?? ''} onChange={e => {
                    const sampling = { ...values.sampling };
                    if (e.target.value === '')
                        delete sampling[id];
                    else
                        sampling[id] = Number(e.target.value);
                    change('sampling', sampling);
                }}/></label>))}</section></> : tab === 'json' ? <><p className="small muted">{tr("Engine arguments and STRATA environment flags are editable here. Program paths and network access remain managed by the host.")}</p><textarea className="st-input native-json" aria-label={tr("Native configuration JSON")} spellCheck={false} value={text} onChange={e => { setText(e.target.value); setNotice(''); }}/></> : <><p className="small muted">{tr("Add local programs or HTTP MCP servers. Connected tools can read or change the resources you give them access to. Enable tools in chat when you want the model to use them.")}</p><NativeMcpEditor text={mcp} onChange={value => { setMcp(value); setNotice(''); }}/><details><summary className="small">{tr("Configuration example")}</summary><pre className="native-example">{'{\n  "files": {\n    "command": "npx",\n    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/your/folder"],\n    "disabled": false\n  }\n}'}</pre><p className="xs muted">{tr("Saved credentials are masked. Keep the mask to preserve an existing value; type a new value to replace it.")}</p></details></>)}
  <div className="native-settings-actions"><Btn disabled={pending || !original || !dirty} variant="primary" onClick={() => void save()}>{tr(pending ? 'Saving…' : 'Save settings')}</Btn><Btn disabled={pending} onClick={() => { const saved = JSON.parse(original || '{"values":{},"mcp":{}}'); setValues(saved.values); setText(JSON.stringify(saved.values, null, 2)); setMcp(JSON.stringify(saved.mcp, null, 2)); setError(''); }}>{tr("Restore saved")}</Btn></div>{tr(notice ? <p role="status" className="small">{tr(notice)}</p> : null)}{tr(error ? <ErrorNotice message={error} onRetry={() => setRevision(value => value + 1)}/> : null)}
 </details>;
}
function NativeFieldInput({ field, value, onChange }: {
    field: NativeField;
    value: any;
    onChange: (v: any) => void;
}) {
    if (field.type === 'array')
        return <NativeAliases field={field} value={value} onChange={onChange}/>;
    if (field.type === 'boolean')
        return <Select disabled={field.supported === false} ariaLabel={tr(field.label)} value={value === null || value === undefined ? '' : String(value)} options={[{ value: '', label: 'Default' }, { value: 'true', label: 'On' }, { value: 'false', label: 'Off' }]} onChange={v => onChange(v === '' ? null : v === 'true')}/>;
    if (field.type === 'enum')
        return <Select disabled={field.supported === false} ariaLabel={tr(field.label)} value={value ?? ''} options={[{ value: '', label: 'Default' }, ...(field.choices ?? []).map(v => ({ value: v, label: v }))]} onChange={v => onChange(v || null)}/>;
    return <input disabled={field.supported === false} className="st-input" aria-label={tr(field.label)} inputMode={field.type === 'number' ? 'numeric' : undefined} placeholder={tr("Default")} value={Array.isArray(value) ? value.join(', ') : value ?? ''} onChange={e => onChange(e.target.value === '' ? null : field.type === 'array' ? e.target.value.split(',').map(s => s.trim()).filter(Boolean) : field.type === 'number' ? Number(e.target.value) : e.target.value)}/>;
}
function NativeAliases({ field, value, onChange }: {
    field: NativeField;
    value: any;
    onChange: (value: any) => void;
}) { const [draft, setDraft] = useState(Array.isArray(value) ? value.join(', ') : ''); useEffect(() => setDraft(Array.isArray(value) ? value.join(', ') : ''), [JSON.stringify(value)]); return <input className="st-input" aria-label={tr(field.label)} disabled={field.supported === false} placeholder={tr("chat, coding")} value={draft} onChange={e => setDraft(e.target.value)} onBlur={() => onChange(draft.trim() ? draft.split(',').map(v => v.trim()).filter(Boolean) : null)}/>; }
