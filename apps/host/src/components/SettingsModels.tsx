import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { removeModelConnection, saveModelConnection, useStore } from '../lib/store';
import type { ApiModel, ModelConnection, ModelConnectionDraft, ModelProtocol } from '../lib/types';
import { Btn, Dialog, ErrorNotice, Icon, Select } from './ui';
const protocols: {
    value: ModelProtocol;
    label: string;
}[] = [
    { value: 'openai-chat', label: 'OpenAI Chat Completions' },
    { value: 'openai-responses', label: 'OpenAI Responses' },
    { value: 'anthropic', label: 'Anthropic Messages' },
];
export function SettingsModels() {
    const providers = useStore(s => s.providers), streaming = useStore(s => s.streaming);
    const [draft, setDraft] = useState<ModelConnectionDraft | null>(null);
    const [available, setAvailable] = useState<ApiModel[]>([]), [filter, setFilter] = useState('');
    const [modelId, setModelId] = useState(''), [modelName, setModelName] = useState('');
    const [pending, setPending] = useState(false), [fetching, setFetching] = useState(false);
    const [error, setError] = useState(''), [notice, setNotice] = useState(''), [remove, setRemove] = useState(false);
    const discovery = useRef<AbortController | null>(null);
    const editor = useRef<HTMLDivElement>(null);
    useEffect(() => () => discovery.current?.abort(), []);
    const saved = providers?.connections.find(c => c.id === draft?.id);
    const preset = providers?.catalog.find(p => p.id === draft?.provider);
    function reset() { discovery.current?.abort(); setFetching(false); setAvailable([]); setFilter(''); setModelId(''); setModelName(''); setError(''); setNotice(''); }
    function add() {
        reset();
        const preset = providers?.catalog[0];
        setDraft({ kind: 'provider', provider: preset?.id ?? null, name: preset?.name ?? '', baseUrl: preset?.baseUrl ?? '', protocol: preset?.protocol ?? 'openai-chat', apiKey: '', models: [] });
    }
    function edit(connection: ModelConnection) { reset(); setDraft({ ...connection, apiKey: '', models: [...connection.models] }); }
    function update(patch: Partial<ModelConnectionDraft>) {
        discovery.current?.abort();
        setFetching(false);
        setError('');
        setNotice('');
        setDraft(d => d ? { ...d, ...patch } : null);
        if (patch.provider !== undefined || patch.baseUrl !== undefined || patch.protocol !== undefined || patch.apiKey !== undefined)
            setAvailable([]);
    }
    useEffect(() => {
        if (draft)
            editor.current?.querySelector<HTMLElement>('input:not(:disabled),button:not(:disabled)')?.focus();
    }, [draft?.id, !!draft]);
    async function fetchModels() {
        if (!draft)
            return;
        discovery.current?.abort();
        const controller = new AbortController();
        discovery.current = controller;
        setFetching(true);
        setError('');
        setNotice('');
        try {
            const response = await fetch('/api/model-providers/discover', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(draft), signal: controller.signal });
            const data = await response.json() as {
                models?: ApiModel[];
                error?: string;
            };
            controller.signal.throwIfAborted();
            if (!response.ok)
                throw new Error(data.error ?? 'Could not fetch the model list.');
            setAvailable(data.models ?? []);
            setNotice(data.models?.length ? `Found ${data.models.length} models. Choose the ones to show in chat.` : 'This API returned no models. Add a model ID below.');
        }
        catch (err) {
            if (!controller.signal.aborted)
                setError(err instanceof Error ? err.message : 'Could not fetch models. You can add a model ID below.');
        }
        finally {
            if (discovery.current === controller)
                setFetching(false);
        }
    }
    async function apply() {
        if (!draft)
            return;
        setPending(true);
        setError('');
        try {
            await saveModelConnection(draft);
            setDraft(null);
            reset();
            setNotice('Models added. Choose one from the model picker in chat.');
        }
        catch (err) {
            setError(err instanceof Error ? err.message : 'Could not save the connection.');
        }
        finally {
            setPending(false);
        }
    }
    const connectionFields = draft ? <div className="provider-fields">
        <label className="input-field"><span>{tr("Display name")}</span><input className="st-input" value={draft.name} placeholder={tr(draft.kind === 'custom' ? 'My other PC' : 'Provider name')} onChange={e => update({ name: e.target.value })}/></label>
        <label className="input-field"><span>{tr("Base URL")}</span><input className="st-input" type="url" value={draft.baseUrl} placeholder={tr("http://192.168.1.20:1234/v1")} spellCheck={false} onChange={e => update({ baseUrl: e.target.value })}/><small className="muted">{tr(preset?.baseUrlHint ?? 'Include the API prefix, such as /v1, but leave off /models or /chat/completions.')}</small></label>
        <div className="input-field"><span>{tr("API protocol")}</span><Select ariaLabel={tr("API protocol")} value={draft.protocol} options={protocols} onChange={value => update({ protocol: value })}/></div>
  </div> : null;
    return <div className="stack stack--lg settings-models">
    <div><h2 className="h3">{tr("Models")}</h2><p className="small muted settings-models__intro">{tr("Connect an API provider or a model server on another computer.")}</p></div>
    {tr(providers?.connections.length ? <div className="stack stack--sm">{tr(providers.connections.map(connection => <div className="st-card provider-card" key={connection.id}>
      <Icon name={connection.kind === 'custom' ? 'computer' : 'models'} size={22}/>
      <div className="provider-card__text"><strong>{connection.name}</strong><p className="small muted">{tr(connection.models.length)} {tr(connection.models.length === 1 ? 'model' : 'models')} · {tr(connection.apiKeySet ? 'API key saved' : 'No API key')}</p><p className="xs muted provider-address" title={connection.baseUrl}>{connection.baseUrl}</p></div>
      <Btn size="sm" disabled={pending || fetching} onClick={() => edit(connection)} title={tr("Edit {0}", [connection.name])}>{tr("Edit")}</Btn>
    </div>))}</div> : !draft ? <div className="provider-empty"><Icon name="models" size={26}/><strong>{tr("Add models to your chat picker")}</strong><p className="small muted">{tr("Your local model stays available. Add an API connection to use more models in Strata.")}</p></div> : null)}
    {tr(!draft ? <Btn icon="plus" block disabled={!providers} onClick={add}>{tr("Add model provider")}</Btn> : <div className="st-card provider-editor" ref={editor}>
      <div className="provider-kind" aria-label={tr("Connection type")}>
        <button type="button" aria-pressed={draft.kind === 'provider'} disabled={!!draft.id || fetching || pending} onClick={() => { const p = providers!.catalog[0]; reset(); setDraft({ kind: 'provider', provider: p.id, name: p.name, baseUrl: p.baseUrl, protocol: p.protocol, apiKey: '', models: [] }); }}>{tr("API provider")}</button>
        <button type="button" aria-pressed={draft.kind === 'custom'} disabled={!!draft.id || fetching || pending} onClick={() => { reset(); setDraft({ kind: 'custom', provider: null, name: '', baseUrl: '', protocol: 'openai-chat', apiKey: '', models: [] }); }}>{tr("Custom model API")}</button>
      </div>
      <p className="small muted">{tr(draft.kind === 'provider' ? 'Choose a provider, enter its API key, then select models to add.' : 'Use the server’s API address. For another PC, enter its network address instead of localhost.')}</p>
      <fieldset disabled={pending} className="provider-fields">
        {tr(draft.kind === 'provider' ? <div className="input-field"><span>{tr("Provider")}</span><Select ariaLabel={tr("Provider")} value={draft.provider ?? ''} options={providers!.catalog.map(p => ({ value: p.id, label: p.name }))} onChange={value => { const p = providers!.catalog.find(p => p.id === value)!; update({ provider: p.id, name: p.name, baseUrl: p.baseUrl, protocol: p.protocol, apiKey: '', clearKey: true, models: [] }); }}/></div> : null)}
        {tr(draft.kind === 'custom' ? connectionFields : null)}
        <label className="input-field"><span>{tr("API key ")}{tr(draft.kind === 'custom' ? <span className="muted">{tr("(optional)")}</span> : null)}</span><input className="st-input" type="password" value={draft.apiKey} autoComplete="new-password" placeholder={tr(saved?.apiKeySet && !draft.clearKey ? 'Leave blank to keep the saved key' : 'Enter API key')} spellCheck={false} onChange={e => update({ apiKey: e.target.value, clearKey: false })}/><small className="muted">{tr("Saved on the Strata host, never in browser storage.")}</small></label>
        {tr(draft.kind === 'provider' ? <details className="provider-options" key={draft.provider} open={!preset?.baseUrl}><summary>{tr("Connection settings")}</summary>{tr(connectionFields)}</details> : null)}
        {tr(saved?.apiKeySet && draft.kind === 'custom' ? <label className="row small"><input type="checkbox" checked={draft.clearKey ?? false} onChange={e => update({ clearKey: e.target.checked, apiKey: '' })}/>{tr("Remove saved API key")}</label> : null)}
        <div className="provider-models">
          <div className="provider-models__head"><h3 className="h4">{tr("Models in chat")}</h3><Btn size="sm" icon="refresh" disabled={fetching || !draft.baseUrl.trim() || (draft.kind === 'provider' && !draft.apiKey.trim() && (!saved?.apiKeySet || !!draft.clearKey))} onClick={() => void fetchModels()}>{tr(fetching ? 'Fetching…' : 'Fetch available models')}</Btn></div>
          {tr(notice ? <p className="small muted" role="status">{tr(notice)}</p> : null)}
          {tr(available.length ? <><input className="st-input" type="search" aria-label={tr("Filter available models")} placeholder={tr("Filter models…")} value={filter} onChange={e => setFilter(e.target.value)}/><div className="provider-model-list">{tr(available.filter(m => `${m.name} ${m.id}`.toLowerCase().includes(filter.toLowerCase())).map(m => <label className="provider-model-row" key={m.id}><input type="checkbox" checked={draft.models.some(v => v.id === m.id)} onChange={e => update({ models: e.target.checked ? [...draft.models, m] : draft.models.filter(v => v.id !== m.id) })}/><span><strong>{m.name}</strong><small className="muted">{m.id}</small></span></label>))}</div></> : null)}
          {tr(draft.models.length ? <div className="stack stack--sm">{tr(draft.models.map(m => <div className="provider-selected" key={m.id}><Icon name="check" size={15}/><span><strong>{m.name}</strong><small className="muted">{m.id}</small>{tr(draft.protocol === 'openai-chat' ? <label className="row xs"><input type="checkbox" checked={m.vision ?? m.reads ?? true} onChange={e => update({ models: draft.models.map(v => v.id === m.id ? { ...v, vision: e.target.checked } : v) })}/>{tr("Read pictures with this model")}{tr(m.vision != null ? null : <span className="muted"> {tr(m.reads === false ? "(its server says it cannot)" : "(on by default)")}</span>)}</label> : null)}</span><button type="button" className="st-iconbtn" aria-label={tr("Remove {0} from chat picker", [m.name])} onClick={() => update({ models: draft.models.filter(v => v.id !== m.id) })}><Icon name="x" size={16}/></button></div>))}</div> : <p className="small muted">{tr("No models selected yet. Fetch the list or add a model ID.")}</p>)}
          <div className="provider-manual"><label className="input-field"><span>{tr("Model ID")}</span><input className="st-input" value={modelId} placeholder={tr("Model ID from your provider or server")} spellCheck={false} onChange={e => setModelId(e.target.value)} onKeyDown={e => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    if (modelId.trim() && !draft.models.some(m => m.id === modelId.trim())) {
                        update({ models: [...draft.models, { id: modelId.trim(), name: modelName.trim() || modelId.trim() }] });
                        setModelId('');
                        setModelName('');
                    }
                }
            }}/></label><label className="input-field"><span>{tr("Model name ")}<span className="muted">{tr("(optional)")}</span></span><input className="st-input" value={modelName} placeholder={tr("Name shown in chat")} onChange={e => setModelName(e.target.value)}/></label><Btn size="sm" icon="plus" disabled={!modelId.trim() || draft.models.some(m => m.id === modelId.trim())} onClick={() => { update({ models: [...draft.models, { id: modelId.trim(), name: modelName.trim() || modelId.trim() }] }); setModelId(''); setModelName(''); }}>{tr("Add model")}</Btn></div>
        </div>
      </fieldset>
      {tr(error ? <ErrorNotice message={error}/> : null)}
      <div className="provider-actions">{tr(draft.id ? <Btn variant="ghost" size="sm" disabled={pending || streaming} onClick={() => setRemove(true)}>{tr("Remove connection")}</Btn> : null)}<span className="spacer"/><Btn disabled={pending} onClick={() => { reset(); setDraft(null); }}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={pending || fetching || !draft.models.length || !draft.name.trim() || !draft.baseUrl.trim()} onClick={() => void apply()}>{tr(pending ? 'Saving…' : 'Apply')}</Btn></div>
    </div>)}
    {tr(!draft && notice ? <p role="status" className="small muted">{tr(notice)}</p> : null)}
    {tr(!draft && error ? <ErrorNotice message={error}/> : null)}
    <Dialog open={remove} compact title={tr("Remove {0}?", [draft?.name])} onClose={() => !pending && setRemove(false)} actions={<><Btn disabled={pending} onClick={() => setRemove(false)}>{tr("Keep connection")}</Btn><Btn variant="danger" disabled={pending} onClick={async () => {
                if (!draft?.id)
                    return;
                setPending(true);
                try {
                    await removeModelConnection(draft.id);
                    setRemove(false);
                    setDraft(null);
                    reset();
                }
                catch (err) {
                    setRemove(false);
                    setError(err instanceof Error ? err.message : 'Could not remove connection.');
                }
                finally {
                    setPending(false);
                }
            }}>{tr("Remove connection")}</Btn></>}><p>{tr("Its models will be removed from the chat picker. Your conversations stay saved.")}</p></Dialog>
  </div>;
}
