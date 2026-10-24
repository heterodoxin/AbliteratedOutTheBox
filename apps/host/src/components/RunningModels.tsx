import { tr } from '../lib/i18n';
import { useState } from 'react';
import { Btn, ErrorNotice, Icon } from './ui';
import { useStore, runtimeModelAction, pushToast } from '../lib/store';
import { deviceLabel } from '../lib/runtime-models';
import { modelFamilyIcon } from '../lib/model-icons';
/** Shared instance list; device measurements remain owned by the host. */
export function RunningModels({ onSelect, selected, controls = false }: {
    onSelect?: (id: string) => void;
    selected?: string;
    controls?: boolean;
}) {
    const runtime = useStore(s => s.runtime), hardware = useStore(s => s.hardware), serving = useStore(s => s.serving), connected = useStore(s => s.connection === 'live');
    const [pending, setPending] = useState(''), [error, setError] = useState('');
    if (!runtime?.instances.length)
        return null;
    async function act(id: string, action: 'unload' | 'default') {
        setPending(id);
        setError('');
        try {
            await runtimeModelAction(id, action);
            pushToast('success', action === 'unload' ? 'Model unloaded' : 'Chat model changed');
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending('');
        }
    }
    return <section className="running-models" aria-label={tr("Loaded models")}>
    <div className="running-models__heading"><h2>{tr("Loaded models ")}<span>{tr(runtime.instances.length)}</span></h2></div>
    <div className="running-models__list">{tr(runtime.instances.map(i => {
            const log = serving?.requests.filter(r => r.instanceId === i.id) ?? [];
            const active = log.filter(r => r.state === 'generating').length, queued = log.filter(r => r.state === 'queued').length;
            return <div key={i.id} className="running-model" data-selected={selected === i.id || undefined}>
        <Icon name={modelFamilyIcon(i.family)} size={20}/>
        <div className="running-model__identity">{tr(onSelect ? <button className="running-model__select" onClick={() => onSelect(i.id)}>{tr(i.family)}</button> : <strong>{tr(i.family)}</strong>)}<small>{tr(i.name)} · {tr(i.memory.context / 1024)}{tr("K \u00B7 ")}{tr(deviceLabel(i.gpuIds, hardware))}{tr(runtime.defaultId === i.id ? ' · Chat default' : '')}</small></div>
        <span className="running-model__status" data-active={active > 0}>{tr(active ? `${active} active` : queued ? 'Queued' : 'Idle')}{tr(queued ? ` · ${queued} queued` : '')}</span>
        {tr(controls ? <div className="running-model__actions">{tr(runtime.defaultId !== i.id ? <Btn size="sm" variant="ghost" disabled={!!pending || !connected} onClick={() => void act(i.id, 'default')}>{tr("Use in chat")}</Btn> : null)}<Btn size="sm" disabled={!!pending || active > 0 || queued > 0 || !connected} onClick={() => void act(i.id, 'unload')}>{tr(pending === i.id ? 'Updating…' : 'Unload')}</Btn></div> : null)}
      </div>;
        }))}</div>
    {tr(error ? <ErrorNotice message={error}/> : null)}
  </section>;
}
