import { tr } from '../lib/i18n';
import { appBaseUrl } from '../lib/connect-urls';
import { deviceLabel, instanceLabel } from '../lib/runtime-models';
import { nativeRouting, useNative } from '../lib/native';
/** Connection facts first. Instructions and request details open only on demand. */
import { CloudflareDialog } from '../components/CloudflareDialog';
import { useEffect, useRef, useState } from 'react';
import { Btn, Dialog, Empty, Icon, IconBtn, Seg, Select, ErrorNotice } from '../components/ui';
import { Screen, SectionHead } from '../components/layout';
import { clockTime, duration, tokens } from '../lib/format';
import { cancelRequest, setLocalSharing, setServingAccess, useStore, pushToast } from '../lib/store';
export function ServingScreen() {
    const serving = useStore(s => s.serving);
    const storedRuntime = useStore(s => s.runtime), hardware = useStore(s => s.hardware);
    const native = useNative();
    const runtime = nativeRouting(native, hardware) ?? storedRuntime;
    const [modelFilter, setModelFilter] = useState('all'), [guideModel, setGuideModel] = useState('');
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [guide, setGuide] = useState(false);
    const [protocol, setProtocol] = useState<'openai' | 'responses' | 'anthropic'>('openai');
    const [filter, setFilter] = useState<'all' | 'active' | 'done' | 'error'>('all');
    const [query, setQuery] = useState('');
    const [sort, setSort] = useState<'at' | 'durationMs'>('at');
    const [cloudflareOpen, setCloudflareOpen] = useState(false);
    const [returnToTunnel, setReturnToTunnel] = useState(false);
    const [keyOpen, setKeyOpen] = useState(false);
    const [keyVisible, setKeyVisible] = useState(false);
    const [sharingOpen, setSharingOpen] = useState(false);
    const [generated, setGenerated] = useState(false);
    const [removeKeyOpen, setRemoveKeyOpen] = useState(false);
    const [keyDraft, setKeyDraft] = useState('');
    const [pending, setPending] = useState(false);
    const [error, setError] = useState('');
    function generateKey() { setKeyVisible(false); const bytes = crypto.getRandomValues(new Uint8Array(32)); setKeyDraft('sk-strata-' + Array.from(bytes, v => v.toString(16).padStart(2, '0')).join('')); setGenerated(true); }
    function closeKey() {
        setKeyVisible(false);
        setKeyOpen(false);
        setKeyDraft('');
        if (returnToTunnel) {
            setReturnToTunnel(false);
            setCloudflareOpen(true);
        }
    }
    const selected = serving?.requests.find(r => r.id === selectedId);
    const requestedCancel = useRef<string | null>(null);
    const cancelResult = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (selected?.id === requestedCancel.current && selected?.cancellable === false && cancelResult.current) {
            cancelResult.current.focus();
            requestedCancel.current = null;
        }
    }, [selected?.id, selected?.cancellable]);
    if (!serving)
        return <Screen title={tr("Serving")}><p className="muted">{tr("Waiting for API status\u2026")}</p></Screen>;
    const multipleModels = (runtime?.instances.length ?? 0) > 1;
    const filterId = multipleModels && runtime?.instances.some(i => i.id === modelFilter) ? modelFilter : 'all';
    const loaded = runtime?.instances.find(i => i.apiModelId === guideModel) ?? runtime?.instances.find(i => i.id === runtime.defaultId);
    const scopeRequests = filterId === 'all' ? serving.requests : serving.requests.filter(r => r.instanceId === filterId);
    const scopeStats = filterId === 'all' ? serving.stats : { running: scopeRequests.filter(r => r.state === 'generating').length, waiting: scopeRequests.filter(r => r.state === 'queued').length, completed: scopeRequests.filter(r => r.state === 'done').length };
    const modelOptions = (runtime?.instances ?? []).map(i => ({ value: i.apiModelId, label: `${instanceLabel(i)} · ${deviceLabel(i.gpuIds, hardware)}` }));
    const requests = serving.requests.filter(r => (filterId === 'all' || r.instanceId === filterId) && (filter === 'all' || (filter === 'active' ? ['generating', 'queued'].includes(r.state) : r.state === filter)) && `${r.client} ${r.id} ${r.endpoint} ${r.model}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => sort === 'durationMs' ? b.durationMs - a.durationMs : b.at.localeCompare(a.at));
    async function operation(action: () => Promise<void>) {
        setPending(true);
        setError('');
        try {
            await action();
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    async function copyConnectionValue(value: string, label: string) {
        try {
            await navigator.clipboard.writeText(value);
            pushToast('success', 'Copied', label);
        }
        catch {
            pushToast('warn', 'Couldn’t copy', 'Select the value and copy it manually.');
        }
    }
    const code = `curl ${serving.baseUrl}${protocol === 'responses' ? '/responses' : protocol === 'openai' ? serving.endpoints.openai : serving.endpoints.anthropic} \\\n  -H 'Content-Type: application/json'${serving.access.apiKeySet ? " \\\n  -H 'Authorization: Bearer YOUR_API_KEY'" : ''} \\\n  -d '${JSON.stringify({ model: loaded?.apiModelId ?? 'YOUR_MODEL_ID', ...(protocol === 'responses' ? { input: 'Hello', store: false } : { ...(protocol === 'anthropic' ? { max_tokens: 256 } : {}), messages: [{ role: 'user', content: 'Hello' }] }) }, null, 2)}'`;
    return <Screen title={tr("Serving")} actions={<><Btn icon="link" onClick={() => setCloudflareOpen(true)}>{tr("Cloudflare tunnel")}</Btn><IconBtn name="info" title={tr("Connection guide")} onClick={() => setGuide(true)}/></>}><div className="stack stack--lg">
    <section className="st-card serving-overview"><dl className="summary-list"><div><dt>{tr("API base URL")}</dt><dd className="row"><code style={{ overflowWrap: 'anywhere' }}>{serving.baseUrl}</code><IconBtn name="copy" title={tr("Copy API base URL")} onClick={() => void copyConnectionValue(serving.baseUrl, 'API base URL')}/></dd></div>{tr(!multipleModels ? <div><dt>{tr("Model ID")}</dt><dd className="row"><code>{loaded?.apiModelId ?? 'No model loaded'}</code><IconBtn name="copy" title={tr("Copy Model ID")} disabled={!loaded} onClick={() => {
                if (loaded)
                    void copyConnectionValue(loaded.apiModelId, 'Model ID');
            }}/></dd></div> : null)}<div><dt>{tr("Status")}</dt><dd><span className="api-status" data-active={serving.running}><span className="api-status__dot"/>{tr(serving.running ? 'API listening' : 'API stopped')} · {tr(multipleModels ? `${runtime?.instances.length} models loaded` : loaded ? 'Model ready' : 'Model not ready')}</span></dd></div><div><dt>{tr("Local network")}</dt><dd className="row"><span className="request-status" data-state={serving.sharing?.active ? "shared" : "queued"}>{tr(serving.sharing?.active ? "Shared on local network" : "Off")}</span><Btn size="sm" onClick={() => { setError(''); setSharingOpen(true); }}>{tr(serving.sharing?.active ? "Manage / stop" : "Share")}</Btn></dd></div>{tr(serving.tunnel?.status === "connected" && serving.tunnel.url ? <div><dt>{tr("Cloudflare tunnel")}</dt><dd className="row"><code style={{ overflowWrap: "anywhere" }}>{serving.tunnel.url}</code><IconBtn name="copy" title={tr("Copy Cloudflare tunnel URL")} onClick={() => void navigator.clipboard.writeText(serving.tunnel!.url!)}/><IconBtn name="settings" title={tr("Manage Cloudflare tunnel")} onClick={() => setCloudflareOpen(true)}/></dd></div> : serving.tunnel?.status === "starting" ? <div><dt>{tr("Cloudflare tunnel")}</dt><dd role="status">{tr("Connecting\u2026")}</dd></div> : null)}<div className="serving-access-row"><dt>{tr("API key")}</dt><dd><div className="serving-access-status">{tr(serving.access.apiKeySet ? <span className="small muted">{tr("Required \u00B7 ")}{tr(serving.access.apiKeyHint)}</span> : null)}<div className="row"><Btn size="sm" onClick={() => { setError(''); generateKey(); setKeyOpen(true); }}>{tr(serving.access.apiKeySet ? 'Change key' : 'Set key')}</Btn>{tr(serving.access.apiKeySet ? <Btn size="sm" onClick={() => { setError(''); setRemoveKeyOpen(true); }}>{tr("Remove key")}</Btn> : null)}</div></div>{tr(serving.access.otherDevices ? <span className="xs muted">{tr("Keep a key while the API is available to other devices.")}</span> : null)}</dd></div></dl></section>
    {tr(multipleModels ? <section className="st-card serving-models" aria-label={tr("Available API models")}><div className="section-toolbar"><h2 className="h4">{tr("Available models")}</h2><span className="xs muted">{tr("One API URL \u00B7 route by model ID")}</span></div>
      {tr(runtime?.instances.length ? <div className="serving-models__list">{tr(runtime.instances.map(i => <div key={i.id} className="serving-model"><div><strong>{tr(i.family)}</strong><small>{tr(i.name)} · {tr(i.memory.context / 1024)}{tr("K \u00B7 ")}{tr(deviceLabel(i.gpuIds, hardware))}</small></div><code>{i.apiModelId}</code><IconBtn name="copy" title={tr("Copy model ID {0}", [i.apiModelId])} onClick={() => void copyConnectionValue(i.apiModelId, 'Model ID')}/></div>))}</div> : <p className="small muted">{tr("Load a model in Models to make it available here.")}</p>)}
      <p className="xs muted">{tr("Set ")}<code>model</code>{tr(" in each request. Each model has its own queue.")}{tr(runtime?.mode === 'simulated' ? ' Design preview · responses are illustrative.' : '')}</p>
    </section> : null)}
    <section className="request-section"><SectionHead title={tr("Requests")}/><p className="small muted">{tr(scopeStats.running)}{tr(" active \u00B7 ")}{tr(scopeStats.waiting)}{tr(" queued \u00B7 ")}{tr(scopeStats.completed)}{tr(" completed")}</p><div className="section-toolbar"><Seg value={filter} ariaLabel={tr("Request status filter")} options={[{ value: 'all', label: 'All' }, { value: 'active', label: 'Active' }, { value: 'done', label: 'Done' }, { value: 'error', label: 'Errors' }]} onChange={setFilter}/>{tr(multipleModels ? <Select ariaLabel={tr("Serving model filter")} value={filterId} options={[{ value: 'all', label: 'All models' }, ...(runtime?.instances ?? []).map(i => ({ value: i.id, label: instanceLabel(i) }))]} onChange={setModelFilter}/> : null)}<input className="st-input" aria-label={tr("Search requests")} placeholder={tr("Client, model or request ID…")} value={query} onChange={e => setQuery(e.target.value)}/></div>
    {tr(requests.length ? <div className="st-tablewrap"><table className="st-table"><thead><tr><th aria-sort={sort === 'at' ? 'descending' : 'none'}><button className="st-btn st-btn--ghost st-btn--sm" onClick={() => setSort('at')}>{tr("Time")}</button></th><th>{tr("Client / endpoint")}</th>{tr(multipleModels ? <th>{tr("Model")}</th> : null)}<th>{tr("Status")}</th><th>{tr("Tokens in / out")}</th><th aria-sort={sort === 'durationMs' ? 'descending' : 'none'}><button className="st-btn st-btn--ghost st-btn--sm" onClick={() => setSort('durationMs')}>{tr("Duration")}</button></th></tr></thead><tbody>{tr(requests.map(r => <tr key={r.id}><td><button className="st-btn st-btn--ghost st-btn--sm" aria-label={tr("View request {0}", [r.id])} onClick={() => setSelectedId(r.id)}>{tr(clockTime(r.at))}</button></td><td>{tr(r.client)}<div className="xs muted">{tr(r.endpoint)}</div></td>{tr(multipleModels ? <td><code className="xs">{r.model}</code></td> : null)}<td>{tr(<span className="request-status" data-state={r.state}>{tr(r.cancelRequested ? r.cancellable ? 'Cancelling…' : 'Cancelled' : r.state.charAt(0).toUpperCase() + r.state.slice(1))}</span>)}</td><td className="num">{tr(tokens(r.tokensIn))} / {tr(tokens(r.tokensOut))}<div className="xs muted">{tr(r.usageSource === 'demo' ? 'Simulated' : r.usageSource === 'reported' ? 'Reported by model' : 'Usage unavailable')}</div></td><td className="num">{tr(r.durationMs ? duration(r.durationMs) : '—')}</td></tr>))}</tbody></table></div> : <Empty icon="link" title={tr(serving.requests.length ? 'No matching requests' : 'No requests yet')} body={serving.requests.length ? 'Try another filter or client name.' : 'Requests appear here when a client or Strata chat uses the API.'} action={tr(serving.requests.length ? <Btn onClick={() => { setQuery(''); setFilter('all'); setModelFilter('all'); }}>{tr("Clear filters")}</Btn> : <Btn onClick={() => setGuide(true)}>{tr("Connect an app")}</Btn>)}/>)}</section>

    {tr(error ? <ErrorNotice message={error}/> : null)}
  </div><Dialog open={guide} title={tr("Connect Strata to an app")} onClose={() => setGuide(false)} headerActions={<div className="protocol-switch"><Seg panelId="api-protocol-panel" value={protocol} ariaLabel={tr("API protocol")} options={[{ value: 'openai', label: 'Chat Completions' }, { value: 'responses', label: 'Responses' }, { value: 'anthropic', label: 'Anthropic' }]} onChange={setProtocol}/></div>} sheet><div className="stack">{tr(multipleModels ? <label className="input-field"><span>{tr("Model for this app")}</span><Select ariaLabel={tr("Connection guide model")} value={loaded?.apiModelId ?? ''} options={modelOptions} disabled={!modelOptions.length} onChange={setGuideModel}/></label> : null)}<ol><li>{tr("In your client, choose a compatible provider.")}</li><li>{tr("Set its base URL to ")}<code>{appBaseUrl(serving.baseUrl, protocol)}</code>.</li><li>{tr("Use model ID ")}<code>{loaded?.apiModelId ?? 'from the Models screen after loading'}</code>.</li><li>{tr(serving.access.apiKeySet ? 'Enter your existing API key in the client.' : 'This local API has no key configured.')}</li></ol>{tr(protocol === 'anthropic' ? <p className="small muted">{tr("Anthropic apps add /v1/messages themselves, so this address has no /v1 at the end. In Claude Code, set ANTHROPIC_BASE_URL to it.")}</p> : null)}{tr(protocol === 'responses' && (!native?.instances.some(i => i.status === 'ready') || native.capabilities.responses === false) ? <p className="small muted">{tr("Responses requires a running native model and a Strata version that supports this API. Update the native installation in Settings \u2192 About if needed.")}</p> : null)}<div id="api-protocol-panel" role="tabpanel" aria-labelledby={`api-protocol-panel-tab-${protocol}`} tabIndex={0}><div className="st-code"><div className="st-code__head"><span>{tr("curl")}</span><IconBtn name="copy" title={tr("Copy request template")} onClick={() => void navigator.clipboard.writeText(code)}/></div><pre>{code}</pre></div><p className="xs muted">{tr("Model list: ")}{tr(serving.baseUrl)}{tr(serving.endpoints.models)}{tr(". Network access depends on the host\u2019s listen address and your firewall.")}</p></div></div></Dialog>
    <Dialog open={!!selectedId} title={tr("Request details")} onClose={() => setSelectedId(null)} sheet>{tr(selected ? <><dl className="summary-list"><div><dt>{tr("ID")}</dt><dd className="mono">{tr(selected.id)}</dd></div><div><dt>{tr("State")}</dt><dd>{tr(selected.cancelRequested ? selected.cancellable ? 'Cancelling…' : 'Cancelled' : selected.state)}</dd></div><div><dt>{tr("Model")}</dt><dd>{tr(selected.model)}</dd></div><div><dt>{tr("Tokens in / out")}</dt><dd>{tr(tokens(selected.tokensIn))} / {tr(tokens(selected.tokensOut))}</dd></div><div><dt>{tr("Usage source")}</dt><dd>{tr(selected.usageSource === 'demo' ? 'Simulated' : selected.usageSource === 'reported' ? 'Reported by model' : 'Unavailable')}</dd></div><div><dt>{tr("Recorded duration")}</dt><dd>{tr(selected.durationMs ? duration(selected.durationMs) : 'Not yet available')}</dd></div></dl>{tr(selected.error ? <div ref={cancelResult} tabIndex={-1}><ErrorNotice message={selected.error} next={tr("Check the client connection and try the request again.")}/></div> : null)}<p className="small muted">{tr("Prompt content and historical queue position are not retained. This view updates while the request runs.")}</p>{tr(selected.cancellable !== false && ['queued', 'generating'].includes(selected.state) ? <Btn disabled={pending || selected.cancelRequested} onClick={() => { requestedCancel.current = selected.id; void operation(() => cancelRequest(selected.id)); }}>{tr(selected.cancelRequested ? 'Cancelling…' : 'Cancel request')}</Btn> : null)}{tr(error ? <ErrorNotice message={error}/> : null)}</> : <p>{tr("This request is no longer retained in the session log.")}</p>)}</Dialog>
    <Dialog open={sharingOpen} title={tr("Local sharing")} onClose={() => !pending && setSharingOpen(false)} actions={<><Btn disabled={pending} onClick={() => setSharingOpen(false)}>{tr("Close")}</Btn><Btn variant="primary" disabled={pending} onClick={() => void operation(async () => { await setLocalSharing(!serving.sharing?.active); setSharingOpen(false); })}>{tr(pending ? 'Updating…' : serving.sharing?.active ? 'Stop sharing' : 'Start sharing')}</Btn></>}><div className="form-stack"><span className="dialog-feature-icon"><Icon name="link" size={28}/></span><p>{tr(serving.sharing?.active ? 'Your inference API is available to devices on this network.' : 'Let another device on your network use your loaded models.')}</p><p className="small muted">{tr("Only the inference API is shared. Clients need your API key when one is set. Use a trusted local network; this connection uses HTTP.")}</p>{tr(serving.sharing?.active ? <div className="row"><code className="key-url">{serving.sharing.url}</code><IconBtn name="copy" title={tr("Copy local network URL")} onClick={() => void navigator.clipboard.writeText(serving.sharing!.url!)}/></div> : null)}{tr(!serving.access.apiKeySet ? <><p className="small muted"><Icon name="warning" size={16}/>{tr(" Without an API key, any device on this network can use your model.")}</p><Btn onClick={() => { generateKey(); setSharingOpen(false); setKeyOpen(true); }}>{tr("Create an API key (recommended)")}</Btn></> : <p className="small muted"><Icon name="shield" size={16}/>{tr(" API key required")}</p>)}{tr(error ? <ErrorNotice message={error}/> : null)}</div></Dialog>
    <Dialog open={keyOpen} className="serving-key-dialog" title={tr(serving.access.apiKeySet ? 'Change API key' : 'Create an API key')} onClose={() => {
            if (!pending)
                closeKey();
        }} actions={<><Btn disabled={pending} onClick={closeKey}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={pending || !keyDraft.trim()} onClick={() => void operation(async () => { await setServingAccess({ apiKey: keyDraft.trim() }); closeKey(); })}>{tr(pending ? 'Saving…' : 'Use this key')}<Icon name="chevron" className="wizard-next" size={14}/></Btn></>}>
      <div className="tunnel-wizard"><section className="tunnel-stage key-stage" aria-labelledby="key-stage-title">
        <span className="tunnel-stage__symbol"><Icon name="shield" size={26}/></span>
        <h3 id="key-stage-title" className="h3">{tr("Protect your connection")}</h3>
        <p className="small muted">{tr("A key lets your other apps connect to Strata.")}</p>
        <div className="input-field"><label htmlFor="api-key">{tr(generated ? 'Your new key' : 'Your own key')}</label><div className="key-input">
          <input id="api-key" aria-label={tr("API key")} className="st-input" type={keyVisible ? 'text' : 'password'} autoComplete="new-password" spellCheck={false} value={keyDraft} onChange={e => { setKeyDraft(e.target.value); setGenerated(false); }}/>
          <button type="button" className="st-btn st-btn--secondary st-btn--sm" aria-label={tr(keyVisible ? 'Hide API key' : 'Show API key')} aria-pressed={keyVisible} disabled={!keyDraft || pending} onClick={() => setKeyVisible(v => !v)}>{tr(keyVisible ? 'Hide' : 'Show')}</button>
          {tr(keyDraft ? <IconBtn name="copy" title={tr("Copy new API key")} onClick={() => void copyConnectionValue(keyDraft, 'API key')}/> : null)}
        </div></div>
        <div className="key-stage__choices"><Btn size="sm" icon="bolt" disabled={pending} onClick={generateKey}>{tr("Generate another")}</Btn><Btn size="sm" variant="ghost" disabled={pending} onClick={() => { setKeyDraft(''); setGenerated(false); setKeyVisible(false); }}>{tr("Use my own key")}</Btn></div>
        <p className="small muted">{tr("Copy this key into your client. It takes effect when you choose Use this key.")}{tr(serving.access.apiKeySet ? ' Your old key will stop working.' : '')}</p>
        {tr(error ? <ErrorNotice message={error}/> : null)}
      </section></div>
    </Dialog>
    <Dialog open={removeKeyOpen} compact title={tr("Remove API key?")} onClose={() => !pending && setRemoveKeyOpen(false)} actions={<><Btn disabled={pending} onClick={() => setRemoveKeyOpen(false)}>{tr("Keep key")}</Btn><Btn variant="danger" disabled={pending} onClick={() => void operation(async () => { await setServingAccess({ apiKey: null }); setRemoveKeyOpen(false); })}>{tr(pending ? 'Removing…' : 'Remove key')}</Btn></>}><p>{tr("Apps on this computer will be able to send requests without a key. Existing clients can keep their current key field.")}</p>{tr(serving.access.otherDevices ? <p className="small muted"><Icon name="warning" size={16}/>{tr(" Without an API key, anyone who can reach this address can use your model.")}</p> : null)}{tr(error ? <ErrorNotice message={error}/> : null)}</Dialog>
    <CloudflareDialog open={cloudflareOpen} onClose={() => setCloudflareOpen(false)} onCreateKey={() => { setReturnToTunnel(true); setCloudflareOpen(false); generateKey(); setKeyOpen(true); }}/>
  </Screen>;
}
