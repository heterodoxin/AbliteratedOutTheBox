import { tr } from '../lib/i18n';
import { modelFamilyIcon } from '../lib/model-icons';
import { useId, useRef, useState } from 'react';
import { FloatingMenu } from './FloatingMenu';
import { Bar, Btn, Dialog, ErrorNotice, Icon } from './ui';
import { go, useLifecycle, loadVariant, selectRemoteModel, setSettingsSection, useStore } from '../lib/store';
import { pickerGroups, pickerVersionState } from '../lib/model-picker';
import { versionFit } from '../lib/model-versions';
import type { ModelVariant } from '../lib/types';
import { sizeGb } from '../lib/format';
import { contextTokens } from '../lib/experience';
import { nativeAction, useNative } from '../lib/native';
/** The loaded model is shared by all chats; switching uses the existing lifecycle gates. */
export function ChatModelPicker() {
    const native = useNative(), remote = useStore(s => s.remoteModel);
    return native?.instances.length && !remote ? <NativeChatModelPicker /> : <LegacyChatModelPicker />;
}
function NativeChatModelPicker() {
    const native = useNative(), providers = useStore(s => s.providers), streaming = useStore(s => s.streaming);
    const [open, setOpen] = useState(false), [error, setError] = useState('');
    const ref = useRef<HTMLButtonElement>(null);
    const current = native?.instances.find(i => i.id === native.defaultId);
    return <><button className="chat-model-badge chat-model-picker" ref={ref} aria-label={tr("Choose model: {0}", [current?.label ?? 'Native model'])} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(v => !v)}><Icon name="models" size={16}/><span>{tr(current?.label ?? 'Native model')}</span><Icon name="chevron" size={12}/></button>{tr(open ? <FloatingMenu anchor={ref.current} title={tr("Chat model")} width={340} onClose={() => setOpen(false)}>{tr(native?.instances.filter(i => i.status !== 'error').map(i => <button className="menu__item" role="menuitemradio" aria-checked={current?.id === i.id} disabled={streaming || i.status !== 'ready'} key={i.id} onClick={() => void nativeAction(`instances/${i.id}/default`).then(() => setOpen(false)).catch(e => setError(e.message))}><Icon name="models"/><span><strong>{tr(i.label)}</strong><small>{tr(i.gpus.map(g => `GPU ${g + 1}`).join(" + ") || "Automatic GPU")}</small></span></button>))}{tr(providers?.connections.flatMap(c => c.models.map(m => <button className="menu__item" role="menuitem" disabled={streaming} key={`${c.id}-${m.id}`} onClick={() => { selectRemoteModel({ connectionId: c.id, modelId: m.id }); setOpen(false); }}><Icon name="link"/>{m.name} · {c.name}</button>)))}<button className="menu__item" role="menuitem" onClick={() => { setOpen(false); go('models'); }}>{tr("Manage models")}</button>{tr(error ? <p className="small inline-error">{tr(error)}</p> : null)}</FloatingMenu> : null)}</>;
}
function LegacyChatModelPicker() {
    const native = useNative();
    const library = useStore(s => s.library), hardware = useStore(s => s.hardware), engine = useStore(s => s.engine);
    const providers = useStore(s => s.providers), remote = useStore(s => s.remoteModel), streaming = useStore(s => s.streaming);
    const connection = providers?.connections.find(c => c.id === remote?.connectionId);
    const apiModel = connection?.models.find(m => m.id === remote?.modelId);
    const lifecycle = useLifecycle();
    const [open, setOpen] = useState(false), [selected, setSelected] = useState<string | null>(null);
    const [pending, setPending] = useState(false), [error, setError] = useState('');
    const [expanded, setExpanded] = useState<string[]>([]);
    const groupsId = useId();
    const trigger = useRef<HTMLButtonElement>(null);
    const running = library?.variants.find(v => v.id === library.loaded);
    const version = library?.variants.find(v => v.id === selected);
    const groups = library ? pickerGroups(library) : [];
    const status = version ? pickerVersionState(version, library?.loaded ?? null) : null;
    const tooBig = !!version && library?.mode === 'real' && versionFit(version, hardware, contextTokens(engine?.beginner.contextLength ?? '32K')).zone === 'too-big';
    async function apply() {
        if (!version)
            return;
        setPending(true);
        setError('');
        try {
            const success = await loadVariant(version.id);
            if (!success)
                throw new Error('The version could not load. Try again when the engine is ready.');
            selectRemoteModel(null);
            setSelected(null);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    function chooseVersion(v: ModelVariant) {
        setOpen(false);
        if (pickerVersionState(v, library?.loaded ?? null).running)
            selectRemoteModel(null);
        else {
            setSelected(v.id);
            setError('');
        }
    }
    function versionOption(v: ModelVariant) {
        const state = pickerVersionState(v, library?.loaded ?? null);
        return <button key={v.id} role="menuitemradio" disabled={streaming || state.busy} aria-checked={state.running && !apiModel} className="menu__item model-picker-option" onClick={() => chooseVersion(v)}>
      <span><strong>{tr(v.name)}</strong><span className="model-picker-spec"> · {tr(v.quant)} · {tr(sizeGb(v.diskGb))}</span></span>
      <span className="model-picker-state">{tr(state.label)}</span>
    </button>;
    }
    return <>
    <button ref={trigger} className="chat-model-badge chat-model-picker" aria-label={tr("Choose model: {0}", [apiModel?.name ?? (running?.local ? running.name : running?.family) ?? 'No model loaded'])} title={tr(apiModel ? `${connection?.name} · ${apiModel.id}` : 'View and switch models')} disabled={!library && !providers} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(v => !v)}><Icon name={apiModel ? 'link' : modelFamilyIcon(running?.family ?? groups[0]?.family)} size={16}/><span>{tr(apiModel?.name ?? (running?.local ? running.name : running?.family) ?? groups[0]?.family ?? 'No model loaded')}</span><Icon name="chevron" size={12}/></button>
    {tr(open ? <FloatingMenu anchor={trigger.current} title={tr("Models")} width={380} onClose={() => setOpen(false)}>
      {tr(native?.instances.filter(i => i.status !== 'error').map(i => <button key={i.id} className="menu__item" role="menuitem" disabled={streaming || i.status !== 'ready'} onClick={() => void nativeAction(`instances/${i.id}/default`).then(() => { selectRemoteModel(null); setOpen(false); }).catch(e => setError(e.message))}><Icon name="models"/><span>{tr(i.label)}<small>{tr(i.gpus.map(g => `GPU ${g + 1}`).join(" + ") || "Automatic GPU")}</small></span></button>))}<p className="model-picker-heading">{tr("Local models")}</p>
      {tr(!groups.length ? <p className="model-picker-empty">{tr("No local models downloaded yet.")}</p> : null)}
      {tr(groups.map(({ family, versions }, index) => {
                const main = versions[0], state = pickerVersionState(main, library!.loaded);
                const others = versions.slice(1), isExpanded = expanded.includes(family), id = `${groupsId}-${index}`;
                return <div key={family} role="group" aria-label={tr(family)} className="model-picker-group">
          <div className="model-picker-family-row">
            <button role="menuitemradio" disabled={streaming || state.busy} aria-checked={state.running && !apiModel} className="menu__item model-picker-family" onClick={() => chooseVersion(main)}>
              <Icon name={modelFamilyIcon(family)} size={20}/>
              <span className="model-picker-family-copy"><strong>{tr(family)}</strong><small><span>{tr(main.name)}</span><span>{tr(main.quant)} · {tr(sizeGb(main.diskGb))}</span></small></span>
              <span className="model-picker-state">{tr(state.running && !apiModel ? <Icon name="check" size={14}/> : null)}{tr(state.label)}</span>
            </button>
            {tr(others.length ? <button role="menuitem" className="menu__item model-picker-expand" aria-label={tr("Downloaded versions for {0}", [family])} aria-expanded={isExpanded} aria-controls={isExpanded ? id : undefined} onClick={() => setExpanded(current => isExpanded ? current.filter(f => f !== family) : [...current, family])}><Icon name="chevron" size={14}/></button> : null)}
          </div>
          {tr(isExpanded && others.length ? <div id={id} className="model-picker-versions">{tr(others.map(versionOption))}</div> : null)}
        </div>;
            }))}
      {tr(providers?.connections.map(c => <div key={c.id}><p className="model-picker-heading">{c.name}</p>{tr(c.models.map(m => <button key={m.id} role="menuitemradio" disabled={streaming} aria-checked={remote?.connectionId === c.id && remote.modelId === m.id} className="menu__item model-picker-option model-picker-api" onClick={() => { selectRemoteModel({ connectionId: c.id, modelId: m.id }); setOpen(false); }}><span><strong>{m.name}</strong><small>{tr(m.id)}</small></span><span className="model-picker-state">{tr(remote?.connectionId === c.id && remote.modelId === m.id ? <Icon name="check" size={15}/> : 'API')}</span></button>))}</div>))}
      <button role="menuitem" className="menu__item model-picker-manage" onClick={() => { setOpen(false); setSettingsSection('Models'); go('settings'); }}><Icon name="plus" size={16}/>{tr("Manage API models")}</button>
    </FloatingMenu> : null)}
    <Dialog open={!!version} compact title={tr(status?.busy ? `${status.label} ${version?.name}` : `Switch to ${version?.name}?`)} onClose={() => !pending && setSelected(null)} actions={<><Btn disabled={pending} onClick={() => setSelected(null)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={pending || status?.busy || !status?.installed || tooBig || !lifecycle.allowed} onClick={() => void apply()}>{tr(pending ? 'Working…' : status?.busy ? status.label : status?.action)}</Btn></>}>
      {tr(version ? <div className="stack stack--sm"><p><strong>{tr(version.family)}</strong><br /><span className="small muted">{tr(version.name)} · {tr(version.quant)} · {tr(sizeGb(version.diskGb))}</span></p>
        <p className="small">{tr("This switches the model version for all chats. Your conversations stay saved.")}</p>
        {tr(status?.busy ? <Bar value={version.progress?.pct ?? 0} label={tr(version.progress?.stage ?? status.label)}/> : null)}
        {tr(tooBig ? <p className="small inline-error">{tr("This version needs more memory at the current context. Choose a smaller version or reduce context in Engine.")}</p> : null)}
        {tr(!lifecycle.allowed ? <p className="small muted">{tr(lifecycle.reason)}</p> : null)}
        {tr(error || version.error ? <ErrorNotice message={error || version.error!}/> : null)}
      </div> : null)}
    </Dialog>
  </>;
}
