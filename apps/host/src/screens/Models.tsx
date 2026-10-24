import { tr } from '../lib/i18n';
import { modelFamilyIcon } from '../lib/model-icons';
import { ManagedModelLibrary } from '../components/ManagedModelLibrary';
import { useState } from 'react';
import { ActionError, Bar, Btn, Dialog, ErrorNotice, Icon, Select } from '../components/ui';
import { Screen } from '../components/layout';
import { VersionLadder } from '../components/VersionLadder';
import { LifecycleNotice } from '../components/LifecycleNotice';
import { modelFamilies, modelVersions, recommendedVersion, versionFit, VERSION_FIT_LABEL } from '../lib/model-versions';
import { contextTokens, estimate, execution } from '../lib/experience';
import { planMemory } from '../lib/run-memory';
import { useMedia } from '../lib/responsive';
import { formatModelName, sizeGb } from '../lib/format';
import { cancelInstall, go, installVariant, loadVariant, removeVariant, unloadModel, useStore, useLifecycle } from '../lib/store';
export function ModelsScreen({ embedded = false, ...props }: {
    embedded?: boolean;
    contextOverride?: number;
    familyOverride?: string;
    selectedVariant?: string;
    onChoose?: (id: string) => void;
} = {}) {
    return embedded ? <ManagedModelLibrary {...props}/> : <ModelLibraryScreen {...props}/>;
}
function ModelLibraryScreen({ embedded = false, contextOverride, onChoose }: {
    embedded?: boolean;
    contextOverride?: number;
    onChoose?: (id: string) => void;
} = {}) {
    const inlineDetails = useMedia('(max-width: 850px)');
    const library = useStore(s => s.library), hardware = useStore(s => s.hardware), engine = useStore(s => s.engine);
    const lifecycle = useLifecycle();
    const [selected, setSelected] = useState(''), [selectedFamily, setSelectedFamily] = useState(''), [confirm, setConfirm] = useState<'switch' | 'delete' | null>(null), [pending, setPending] = useState(false), [error, setError] = useState('');
    if (!library)
        return <Screen title={tr("Model")}><p>{tr("Loading model versions\u2026")}</p></Screen>;
    const running = library.variants.find(v => v.id === library.loaded), families = modelFamilies(library), family = families.includes(selectedFamily) ? selectedFamily : running?.family ?? families[0];
    const context = contextOverride ?? contextTokens(engine?.beginner.contextLength ?? 'recommended'), versions = modelVersions(library, family), recommended = recommendedVersion(versions, hardware, context);
    const version = versions.find(v => v.id === selected) ?? versions.find(v => v.id === running?.id) ?? recommended ?? versions[0];
    if (!version)
        return <Screen title={tr("Model")}><p>{tr("No model versions are available.")}</p></Screen>;
    function planningLabel(v: typeof version) { const preview = planMemory(estimate(v, hardware, execution(undefined, context)), hardware, 'shared'); return context > v.contextMax ? 'Context too long' : !preview ? 'Fit unknown' : !preview.cacheFits ? 'Context needs more memory' : preview.ssd > .05 ? 'SSD spill likely' : 'Fits in memory'; }
    const fit = versionFit(version, hardware, context), loaded = version.id === library.loaded, installed = ['loaded', 'installed'].includes(version.state), working = ['loading', 'downloading', 'unloading'].includes(version.state);
    async function act(action: () => Promise<boolean | void>) { setPending(true); setError(''); try {
        const ok = await action();
        if (ok === false)
            throw new Error('The operation could not finish. Check the engine connection and try again.');
        setConfirm(null);
    }
    catch (e) {
        setError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setPending(false);
    } }
    const versionDetails = <section id="model-version-details" className="version-detail" aria-label={tr("Version details")}><div className="section-toolbar"><span className="eyebrow">{tr(version.quant)}</span><span className="model-status">{tr(loaded ? 'Running' : installed ? 'Downloaded' : embedded ? planningLabel(version) : VERSION_FIT_LABEL[fit.zone])}</span></div><div className="version-detail__title"><h2>{tr(version.name)}</h2><p className="small muted">{tr(formatModelName(version))}</p></div><div className="version-numbers"><div><strong>{tr(sizeGb(version.diskGb))}</strong><span>{tr("Download size")}</span></div><div><strong>{tr(sizeGb(fit.needed))}</strong><span>{tr("Estimated memory")}</span></div></div><p className="xs muted">{tr("Select this version to review context, memory and performance.")}</p>
 {tr(onChoose && installed ? <div className="row version-detail__actions"><Btn variant="primary" size="sm" disabled={pending} onClick={() => onChoose(version.id)}>{tr("Use this version")}</Btn><Btn icon="trash" size="sm" disabled={pending || loaded} onClick={() => setConfirm('delete')}>{tr("Delete")}</Btn>{tr(loaded ? <Btn size="sm" disabled={pending || !lifecycle.allowed} onClick={() => void act(unloadModel)}>{tr("Unload model")}</Btn> : null)}</div> : working ? <><Bar value={version.progress?.pct ?? 0} label={tr(version.state)}/><p className="small" role="status">{tr(version.progress?.stage ?? 'Working…')}</p>{tr(version.state === 'downloading' ? <Btn disabled={pending} onClick={() => void act(() => cancelInstall(version.id))}>{tr("Cancel download")}</Btn> : null)}</> : loaded ? <Btn disabled={pending || !lifecycle.allowed} onClick={() => void act(unloadModel)}>{tr("Unload model")}</Btn> : installed ? <div className="row version-detail__actions"><Btn variant="primary" disabled={pending || !lifecycle.allowed || (!embedded && fit.zone === 'too-big')} onClick={() => setConfirm('switch')}>{tr("Switch to ")}{tr(version.name)}</Btn><Btn icon="trash" disabled={pending} onClick={() => setConfirm('delete')}>{tr("Delete")}</Btn></div> : <Btn variant="primary" icon="download" disabled={pending || (!embedded && fit.zone === 'too-big') || version.downloadGb > library.disk.freeGb} onClick={() => void act(() => installVariant(version.id))}>{tr(version.downloadGb > library.disk.freeGb ? 'Not enough disk space' : !embedded && fit.zone === 'too-big' ? 'Too big for this PC' : `Download · ${sizeGb(version.downloadGb)}`)}</Btn>)}
 {tr(version.error || error ? <ErrorNotice message={error || version.error!} next={tr("Your downloaded files are retained.")}/> : null)}<LifecycleNotice showScope={false}/>
 </section>;
    const content = <><div className="stack stack--lg model-page"><ActionError />
 <section className="model-hero" aria-label={tr("Model family")}><span className="model-hero__icon"><Icon name={modelFamilyIcon(family)} size={24}/></span><div className="model-hero__body"><div className="model-family-selector"><Select ariaLabel={tr("Model family")} menuWidth={360} menuClassName="model-family-menu" optionIcon={modelFamilyIcon} value={family} valueSuffix={<span className="model-family-meta">{tr(running?.family === family ? `${running.quant} · ${sizeGb(running.diskGb)}` : `${versions.length} versions`)}</span>} options={families.map(f => ({ value: f, label: f }))} onChange={f => { setSelectedFamily(f); setSelected(''); setError(''); }}/></div></div></section>
 {tr(!hardware ? <ErrorNotice message="Hardware has not been detected." next={tr("Check this computer to see which versions fit.")} action={tr("Run guided setup")} onRetry={() => go('setup')}/> : null)}
 <div className="model-version-heading"><h2 className="h3">{tr("Choose a version")}</h2></div>
 {tr(!embedded && library.mode === 'real' && hardware && versions.every(v => versionFit(v, hardware, context).zone === 'too-big') ? <ErrorNotice message={`${context / 1024}K context exceeds the available model budget.`} next={tr("A shorter context may make room for a smaller version.")} action={tr("Review context")} onRetry={() => go('tuning')}/> : null)}
 <div className="model-version-layout"><VersionLadder versions={versions} hardware={hardware} context={context} selected={version.id} recommended={recommended?.id} onSelect={id => { setSelected(id); setError(''); }} selectedDetail={inlineDetails ? versionDetails : undefined} detailId="model-version-details" fitLabel={embedded ? planningLabel : undefined}/>{tr(inlineDetails ? null : versionDetails)}</div>
 <details className="model-installed"><summary><span className="model-installed__label"><Icon name="chevron" size={16}/><span className="h3">{tr("On this computer")}</span></span><span className="small muted">{tr(sizeGb(library.disk.usedGb))}{tr(" model files \u00B7 ")}{tr(sizeGb(library.disk.freeGb))}{tr(" free")}</span></summary><div className="model-installed__files">{tr(library.variants.filter(v => ['installed', 'loaded'].includes(v.state)).map(v => <div className="installed-version" key={v.id}><span>{tr(v.family)} · {tr(v.quant)}</span><span className="small muted">{tr(sizeGb(v.diskGb))}</span></div>))}{tr(!library.variants.some(v => ['installed', 'loaded'].includes(v.state)) ? <p className="small muted">{tr("No versions downloaded yet. Choose one above to get started.")}</p> : null)}</div></details>
 </div><Dialog open={!!confirm} title={tr(confirm === 'delete' ? 'Delete this version?' : `Switch to ${version.name}?`)} onClose={() => !pending && setConfirm(null)} actions={<><Btn disabled={pending} onClick={() => setConfirm(null)}>{tr("Cancel")}</Btn><Btn variant={confirm === 'delete' ? 'danger' : 'primary'} disabled={pending} onClick={() => void act(() => confirm === 'delete' ? removeVariant(version.id) : loadVariant(version.id))}>{tr(pending ? 'Working…' : error ? 'Try again' : confirm === 'delete' ? 'Delete version' : 'Switch version')}</Btn></>}><p>{tr(confirm === 'delete' ? `Remove ${sizeGb(version.diskGb)} of downloaded files. You can download this version again.` : `Switch all chats to ${version.name} (${sizeGb(version.diskGb)}). Loading may take a few moments.`)}</p>{tr(confirm === 'switch' ? <details className="disclosure"><summary>{tr("Run settings")}</summary><p className="small muted">{tr("Uses your saved ")}{tr(engine?.beginner.contextLength ?? 'recommended')}{tr(" context. Change context on the Model page.")}</p></details> : null)}{tr(error ? <ErrorNotice message={error}/> : null)}</Dialog></>;
    return embedded ? content : <Screen title={tr("Model")}>{tr(content)}</Screen>;
}
