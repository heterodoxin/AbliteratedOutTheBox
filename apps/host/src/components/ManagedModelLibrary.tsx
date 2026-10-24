import { tr } from '../lib/i18n';
import { useRef, useState } from 'react';
import { ActionError, Bar, Btn, Dialog, ErrorNotice, Icon, Select } from './ui';
import { FloatingMenu } from './FloatingMenu';
import { LifecycleNotice } from './LifecycleNotice';
import { VersionLadder } from './VersionLadder';
import { LocalModelImport } from './LocalModelImport';
import { modelFamilyIcon } from '../lib/model-icons';
import { familyVersion, modelFamilies, modelVersions, versionFit } from '../lib/model-versions';
import { contextTokens, estimate, execution } from '../lib/experience';
import { planMemory } from '../lib/run-memory';
import type { ModelVariant } from '../lib/types';
import { cancelInstall, installVariant, useLifecycle, loadVariant, removeVariant, runtimeModelAction, useStore } from '../lib/store';
export function ManagedModelLibrary({ contextOverride, familyOverride, selectedVariant, onChoose }: {
    contextOverride?: number;
    familyOverride?: string;
    selectedVariant?: string;
    onChoose?: (id: string) => void;
}) {
    const library = useStore(s => s.library), hardware = useStore(s => s.hardware), engine = useStore(s => s.engine);
    const runtime = useStore(s => s.runtime), serving = useStore(s => s.serving);
    const lifecycle = useLifecycle();
    const [selected, setSelected] = useState(selectedVariant ?? '');
    const [familyChoice, setFamilyChoice] = useState(familyOverride ?? '');
    const [pending, setPending] = useState(false), [error, setError] = useState('');
    const [localModelOpen, setLocalModelOpen] = useState(false);
    const [menu, setMenu] = useState(false), [confirm, setConfirm] = useState<'delete' | 'switch' | null>(null);
    const more = useRef<HTMLButtonElement>(null);
    if (!library)
        return <p className="small muted">{tr("Loading model versions\u2026")}</p>;
    const families = modelFamilies(library);
    const family = families.includes(familyChoice) ? familyChoice : library.variants.find(v => v.id === library.loaded)?.family ?? families[0];
    const versions = modelVersions(library, family);
    const version = versions.find(v => v.id === selected) ?? familyVersion(library, family);
    if (!version)
        return <p className="small muted">{tr("No versions are available.")}</p>;
    const context = contextOverride ?? contextTokens(engine?.beginner.contextLength ?? 'recommended');
    const fit = versionFit(version, hardware, context);
    const instances = runtime?.instances.filter(i => i.variantId === version.id) ?? [];
    const loaded = instances.length > 0 || version.state === 'loaded';
    const instanceBusy = instances.some(i => serving?.requests.some(r => r.instanceId === i.id && ['generating', 'queued'].includes(r.state)));
    const canUnload = instances.length > 0 && !instanceBusy;
    const working = ['downloading', 'loading', 'unloading'].includes(version.state);
    const installed = ['loaded', 'installed', 'loading', 'unloading'].includes(version.state) || !!version.installedAt;
    const noSpace = version.downloadGb > library.disk.freeGb;
    function memoryNote(v: ModelVariant) {
        if (!['installed', 'loaded'].includes(v.state) && !v.installedAt)
            return;
        const memory = planMemory(estimate(v, hardware, execution(undefined, context)), hardware, 'shared');
        if (memory && memory.ssd > .05)
            return 'SSD spill likely. Part of this model may be read from storage, so responses can be slower.';
    }
    async function act(action: () => Promise<boolean | void>) {
        setPending(true);
        setError('');
        try {
            const ok = await action();
            if (ok === false)
                throw new Error('The operation could not finish. Check the engine connection and try again.');
            setConfirm(null);
        }
        catch (reason) {
            setError(reason instanceof Error ? reason.message : String(reason));
        }
        finally {
            setPending(false);
        }
    }
    const useVersion = () => { if (onChoose)
        onChoose(version.id);
    else
        setConfirm('switch'); };
    return <div className="managed-models">
    <ActionError />
    <div className="managed-models__family"><Icon name={modelFamilyIcon(family)} size={23}/><Select ariaLabel={tr("Manage model family")} value={family} disabled={pending} menuWidth={340} menuClassName="model-family-menu" optionIcon={modelFamilyIcon} options={families.map(f => ({ value: f, label: f }))} onChange={value => { setFamilyChoice(value); setSelected(''); setError(''); setMenu(false); }}/><span>{tr(versions.length)}{tr(" versions")}</span></div>
    <div className="managed-version-head" aria-hidden="true"><span>{tr("Version")}</span><span>{tr("Download size")}</span><span>{tr("Status")}</span></div>
    <VersionLadder compact versions={versions} hardware={hardware} context={context} selected={version.id} disabled={pending} memoryNote={memoryNote} onSelect={id => { setSelected(id); setError(''); setMenu(false); }}/>
    {tr(version.state === 'downloading' ? <div className="managed-models__progress" role="status"><Bar value={version.progress?.pct ?? 0} label={tr("Downloading")}/><span>{tr(version.progress?.stage ?? 'Downloading model files…')}</span></div> : null)}
    <div className="managed-models__footer">
      <div className="managed-models__memory"><span>{tr("Estimated memory")}</span><strong>{tr(fit.needed.toFixed(1))} <small>{tr("GB")}</small></strong></div>
      <div className="managed-models__actions">
        {tr((installed || version.local) && !working ? <button ref={more} className="st-btn st-btn--secondary managed-models__more" aria-label={tr("Version actions")} aria-haspopup="menu" aria-expanded={menu} disabled={pending} onClick={() => setMenu(!menu)}><Icon name="ellipsis" size={19}/></button> : null)}
        {tr(working ? version.state === 'downloading' ? <Btn disabled={pending} onClick={() => void act(() => cancelInstall(version.id))}>{tr("Cancel download")}</Btn> : <Btn disabled>{tr(version.state === 'loading' ? 'Loading…' : 'Unloading…')}</Btn> : installed ? <Btn variant="primary" iconRight="chevron" className="managed-models__use" disabled={pending} onClick={useVersion}>{tr("Use version")}</Btn> : version.local ? <Btn icon="folder" onClick={() => setLocalModelOpen(true)}>{tr("Reconnect file")}</Btn> : <Btn variant="primary" icon="download" disabled={pending || noSpace} onClick={() => void act(() => installVariant(version.id))}>{tr(noSpace ? 'Not enough storage' : pending ? 'Starting…' : 'Download')}</Btn>)}
      </div>
    </div>
    {tr(version.error || error ? <ErrorNotice message={error || version.error!}/> : null)}
    <LifecycleNotice showScope={false}/>
    {tr(menu ? <FloatingMenu anchor={more.current} title={tr("Version actions")} width={230} onClose={() => setMenu(false)}>
      {tr(loaded ? <button role="menuitem" className="menu__item" disabled={!canUnload} onClick={() => { setMenu(false); void act(async () => { for (const instance of instances)
            await runtimeModelAction(instance.id, 'unload'); }); }}><Icon name="stop" size={17}/>{tr(instances.length > 1 ? `Unload ${instances.length} instances` : 'Unload model')}</button> : null)}
      <button role="menuitem" className="menu__item" disabled={loaded || pending} title={tr(loaded ? 'Unload this version first' : undefined)} onClick={() => { setMenu(false); setConfirm('delete'); }}><Icon name="trash" size={17}/>{tr(version.local ? 'Remove from Strata…' : 'Delete files…')}</button>
    </FloatingMenu> : null)}
    <Dialog compact open={!!confirm} title={tr(confirm === 'delete' ? version.local ? 'Remove from Strata?' : 'Delete this version?' : 'Switch version?')} onClose={() => !pending && setConfirm(null)} actions={<><Btn disabled={pending} onClick={() => setConfirm(null)}>{tr("Cancel")}</Btn><Btn variant={confirm === 'delete' ? 'danger' : 'primary'} disabled={pending || (confirm === 'switch' && !lifecycle.allowed)} onClick={() => void act(() => confirm === 'delete' ? removeVariant(version.id) : loadVariant(version.id))}>{tr(pending ? 'Working…' : confirm === 'delete' ? version.local ? 'Remove model' : 'Delete files' : 'Switch version')}</Btn></>}>
      <p>{tr(confirm === 'delete' ? version.local ? 'Remove this model from Strata. Your original GGUF files stay in their folder.' : `Remove ${Math.round(version.diskGb)} GB of downloaded files. You can download this version again.` : `Load ${version.name} for all chats. Loading may take a few moments.`)}</p>
      {tr(error ? <ErrorNotice message={error}/> : null)}
    </Dialog>
    <LocalModelImport open={localModelOpen} directory={library.disk.modelsDir} onClose={() => setLocalModelOpen(false)} onChoose={id => { setFamilyChoice('Local models'); setSelected(id); if (onChoose)
        onChoose(id); }}/>
  </div>;
}
