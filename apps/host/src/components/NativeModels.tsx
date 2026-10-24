import { tr } from '../lib/i18n';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Screen } from './layout';
import { VersionPicker } from './VersionPicker';
import { FamilySwitch } from './FamilySwitch';
import { PcFitCard } from './PcFitCard';
import { fitOn, memoryNeeded } from '../lib/native-fit';
import { GpuRoles } from './GpuRoles';
import { fromLegacy, normalize as normalizeRoles, order as roleOrder, recommend as recommendRoles, rows as roleRows, sameRoles, vramOn, type RoleCard, type Roles } from '../lib/gpu-roles';
import { Btn, Dialog, ErrorNotice, Icon, Select } from './ui';
import { installProgress, installStage, stageText, useDownloadEta } from '../lib/download-eta';
import { useKept } from '../lib/kept-state';
import { DownloadParts } from './DownloadParts';
import { EngineWarning } from './EngineWarning';
import { sizeGb } from '../lib/format';
import { NativeDeleteDialog } from './NativeDelete';
import { plainFailureText } from '../lib/plain-errors';
import { api } from '../lib/client';
import { go, setSettingsSection, selectRemoteModel, pushToast } from '../lib/store';
import { nativeAction, useNative, type NativeJob } from '../lib/native';
import { tuneEffective } from '../lib/native-tune';
import { tunedFields } from '../lib/tune-values';
import { draftingChoices as draftingOptions } from '../lib/drafting';
import { NativeSettings } from './NativeSettings';
import { NativeSettingsFile } from './NativeSettingsFile';
import { RunImageModel } from './RunImageModel';
import { IMAGE_GENERATION_VISIBLE } from '../lib/features';
import { useStore } from '../lib/store';
import { NativePresets } from './NativePresets';
import { NativeTune } from './NativeTune';
import { NativeTuneBadge } from './NativeTuneBadge';
import { nativeCatalogModels, catalogVersionLabel } from '../lib/native-catalog';
export function NativeModels({ onOtherModels, onReady, onImport, onConnectEngine }: {
    onOtherModels: () => void;
    onImport?: () => void;
    onConnectEngine?: () => void;
    onReady?: () => void;
}) {
    const native = useNative(), imageLibrary = useStore(s => s.imageModels), hardware = useStore(s => s.hardware);
    // The choices are kept while the page is closed (Models -> Monitor -> Models used to start them over and a person lost track of what was downloading).
    const [family, setFamily] = useKept('models.family', ''), [size, setSize] = useKept('models.size', ''), [context, setContext] = useKept('models.context', 32768), [vision, setVision] = useKept('models.vision', 'yes');
    const manual = useStore(s => s.setup?.choices?.onboarding?.manual ?? false);
    const [expert, setExpert] = useKept('models.expert', manual), [multiple, setMultiple] = useKept('models.multiple', false), [roles, setRoles] = useKept<Roles | null>('models.roles', null), [kv, setKv] = useKept('models.kv', 'int8'), [lowRam, setLowRam] = useKept('models.lowRam', 'auto'), [split, setSplit] = useKept('models.split', 'auto');
    const [resourceProfile, setResourceProfile] = useKept<'work' | 'performance'>('models.profile', 'work');
    const [drafting, setDrafting] = useState('auto');
    const [pending, setPending] = useState(false), [error, setError] = useState(''), [plan, setPlan] = useState<Record<string, any> | null>(null), [loadedError, setLoadedError] = useState(''), [tuneOpen, setTuneOpen] = useState(false), [jobId, setJobId] = useKept('models.job', ''), [manage, setManage] = useState(false);
    const [toastedJob, setToastedJob] = useKept('models.toasted', ''), [appliedSig, setAppliedSig] = useKept('models.applied', '');
    const presetPending = useRef(false);
    const [deleting, setDeleting] = useState<string | null>(null);
    const families = nativeCatalogModels(native?.catalog?.families ?? []);
    const expertSection = useRef<HTMLElement>(null);
    useEffect(() => { if (expert)
        expertSection.current?.scrollIntoView({ block: 'start', behavior: 'smooth' }); }, [expert]);
    const selectedFamily = families.find(f => f.family === family || f.sizes.some(size => size.sourceFamily === family)) ?? families[0], selectedSize = selectedFamily?.sizes.find(s => s.id === size) ?? selectedFamily?.sizes[0];
    const installed = native?.installed?.models.find(m => m.model === selectedSize?.id);
    // The model that is running is shown as running, with Unload: not offered for loading again.
    const downloaded = native?.installed?.models ?? [], allSizes = families.flatMap(f => f.sizes);
    // This computer for the fit figures: the biggest usable graphics card, or the ones chosen under Expert settings.
    const engineVendor = native?.hardware?.engine_vendor ?? null, roleCards: RoleCard[] = roleRows(native?.hardware?.gpus as RoleCard[] | undefined, engineVendor);
    const pcGpus = (native?.hardware?.gpus ?? []).filter(g => g.usable !== false), pcGpu = [...pcGpus].sort((x, y) => y.vram_gb - x.vram_gb)[0] ?? null;
    const pcRam = native?.hardware?.ram_gb || hardware?.memory.totalGb || null, pcVram = pcGpu?.vram_gb ?? 0;
    // What the models that stay loaded (Run multiple models) already take: the fit figures are for what is left.
    const sizeOf = (id: string) => allSizes.find(s => s.id === id);
    const keptModels = expert && multiple ? (native?.instances ?? []).filter(i => i.status !== 'error' && i.model !== selectedSize?.id) : [];
    const keptUse = keptModels.map(i => { const s = sizeOf(i.model); const card = pcGpus.find(g => g.index === i.gpus[0]); const f = s && pcRam ? fitOn(s, { ram_gb: pcRam, vram_gb: card?.vram_gb ?? pcVram }) : null; return { gpus: i.gpus, ram: f?.ramUseGb ?? 0, vram: f?.vramUseGb ?? 0 }; });
    const keptRam = keptUse.reduce((n, u) => n + u.ram, 0);
    const keptVramOn = (cards: number[]) => keptUse.filter(u => u.gpus.some(g => cards.includes(g))).reduce((n, u) => n + u.vram, 0);
    // The versions that are really on this computer: a config whose files are all there (every part of a model in parts).
    const readyIds = useMemo(() => new Set(downloaded.filter(m => m.ready).map(m => m.model)), [downloaded]);
    // What a running job works on, by its name in the list: "Qwen3.8-Flash-Next · Gyro · Smallest".
    const jobLabel = (j: NativeJob) => { const v = allSizes.find(s => s.id === j.model || (s.model === j.model && (!j.family || s.sourceFamily === j.family))); const f = v && families.find(x => x.sizes.includes(v)); return v ? [f?.title, catalogVersionLabel(v)].filter(Boolean).join(' · ') : ''; };
    const loadedSelected = native?.instances.find(i => i.model === selectedSize?.id && i.status === 'ready');
    // What the running model was really started with is what the page shows: after Load the highlight must still be the profile that was applied.
    useEffect(() => { if (loadedSelected?.resourceProfile)
        setResourceProfile(loadedSelected.resourceProfile); }, [loadedSelected?.id, loadedSelected?.resourceProfile]);
    // After a tuning, Expert mode shows the values the tuning set (the active preset's), not the ones the model was prepared with.
    const [tunedNow, setTunedNow] = useState<{ name: string; fields: ReturnType<typeof tunedFields> } | null>(null);
    useEffect(() => {
        if (!selectedSize || !installed?.ready) { setTunedNow(null); return; }
        let alive = true;
        tuneEffective(selectedSize.id, resourceProfile, selectedSize.images ? vision : 'no').then(t => { if (alive) setTunedNow(t.applied && t.preset ? { name: t.preset.name, fields: tunedFields(t.changed) } : null); }).catch(() => { if (alive) setTunedNow(null); });
        return () => { alive = false; };
    }, [selectedSize?.id, installed?.ready, resourceProfile, vision, native?.jobs.filter(j => j.state === 'done').length]);
    useEffect(() => { if (tunedNow?.fields.kv) setKv(tunedNow.fields.kv as typeof kv); if (tunedNow?.fields.lowRam) setLowRam(tunedNow.fields.lowRam as typeof lowRam); }, [tunedNow]);
    const active = native?.jobs.find(j => j.state === 'running');
    const job = native?.jobs.find(j => j.id === jobId) ?? (!jobId ? native?.jobs.filter(j => ['load', 'install'].includes(j.kind)).at(-1) : undefined);
    const noFit = selectedSize?.on_this_pc === 'does not fit';
    // Graphics cards (strata_control/gpu_fit.py: the one rule of the app, the Terminal and setup): the recommended option is pre-selected, the person may pick another
    // A card a loaded model already uses (a second model on it usually does not fit): said on its row, and not the pre-selected one while another is free.
    const usedBy = (cards: number[]) => (native?.instances ?? []).filter(i => i.status !== 'error' && i.model !== selectedSize?.id && i.gpus.some(g => cards.includes(g))).map(i => i.label);
    // Main / Helper / Off per card (src/lib/gpu-roles.ts, strata_control/gpu_roles.py): the recommendation until the person changes a role; a loaded model shows what it was started with.
    // With Run multiple models, a card another loaded model already uses starts Off for this one (while a free card exists).
    const busy = (i: number) => expert && multiple && usedBy([i]).length > 0, freeCards = roleCards.filter(c => !busy(c.index));
    const recommended = freeCards.length && freeCards.length < roleCards.length ? { ...recommendRoles(freeCards, engineVendor), roles: { ...Object.fromEntries(roleCards.map(c => [c.index, 'off'])), ...recommendRoles(freeCards, engineVendor).roles } as Roles } : recommendRoles(roleCards, engineVendor);
    const loadedRoles = loadedSelected && roleCards.length > 1 ? fromLegacy(loadedSelected.gpus, roleCards, 'as_given') : null;
    const effectiveRoles: Roles = roleCards.length > 1 ? (roles ? normalizeRoles(roles, roleCards) : loadedRoles ?? recommended.roles) : recommended.roles;
    const customRoles = !sameRoles(effectiveRoles, recommended.roles);
    const changeRoles = (next: Roles) => setRoles(sameRoles(next, recommended.roles) ? null : next);
    const cardsGpus: number[] = roleCards.length > 1 ? roleOrder(effectiveRoles, roleCards) : [];
    const chosenCards = cardsGpus;
    const fitHardware = { ram_gb: pcRam ? Math.max(0, pcRam - keptRam) : pcRam, vram_gb: Math.max(0, (roleCards.length > 1 ? vramOn(effectiveRoles, roleCards) : pcVram) - keptVramOn(chosenCards)) };
    // A layer split written for another number of cards no longer fits: back to automatic.
    useEffect(() => { if (cardsGpus.length <= 1 && split !== 'auto') setSplit('auto'); }, [cardsGpus.length]);
    // Drafting (which drafter guesses ahead): DFlash2 is for Qwen3.8-Flash-Next only (the qwen and gyro families), so no other model is offered it.
    const draftingChoices = selectedSize ? draftingOptions(selectedSize.sourceFamily ?? selectedFamily?.family, native?.capabilities?.dflash2) : null;
    const draftingWords: Record<string, string> = { auto: 'Auto (recommended)', mtp: 'MTP', dflash2: 'DFlash2', off: 'Off' };
    const values = useMemo(() => ({ family: selectedSize?.sourceFamily, model: selectedSize?.model, context, vision: selectedSize?.images ? vision : 'no', kv, low_ram: lowRam, resourceProfile, ...(cardsGpus.length > 1 ? { gpus: cardsGpus.join(',') } : cardsGpus.length === 1 ? { gpu: cardsGpus[0] } : {}), loadAfter: true, multiple: expert && multiple }), [selectedSize?.sourceFamily, selectedSize?.model, context, vision, kv, lowRam, resourceProfile, cardsGpus.join(','), expert, multiple]);
    const downloadValues = { ...values, ...(draftingChoices ? { drafting } : {}), vision: selectedSize?.images && vision !== 'no' ? 'yes' : 'no', loadVision: selectedSize?.images ? vision : 'no' };
    useEffect(() => { if (!native?.catalog || family)
        return; const current = native.instances.find(i => i.id === native.defaultId); const rec = native.catalog.recommendation ?? { family: null, model: null }; const loadedFamily = families.find(f => f.sizes.some(s => s.id === current?.model)); const have = new Set((native.installed?.models ?? []).filter(m => m.ready).map(m => m.model)); const installedFamily = families.find(f => f.sizes.some(s => have.has(s.id))); const f = loadedFamily ?? installedFamily ?? families.find(f => f.family === rec.family || f.sizes.some(s => s.sourceFamily === rec.family)) ?? families[0]; if (f) {
        setFamily(f.family);
        setSize(f.sizes.find(s => s.id === current?.model)?.id ?? f.sizes.find(s => have.has(s.id))?.id ?? f.sizes.find(s => s.model === rec.model)?.id ?? f.sizes[0]?.id ?? '');
        setContext(('context' in rec ? rec.context : undefined) ?? 32768);
    } }, [native?.catalog]);
    // The version that is selected: a model already on this PC shows the settings it was prepared with (so loading it never asks to download);
    // a model that is not yet downloaded starts with Read pictures ON (owner rule: pictures on by default). This runs when the SELECTION or the
    // installed settings change, not when the page opens again: coming back from another tab keeps what was chosen.
    useEffect(() => {
        if (presetPending.current) {
            presetPending.current = false;
            return;
        }
        if (!selectedSize)
            return;
        const sig = installed ? ['have', installed.model, installed.ready, installed.context, installed.kv, installed.images, installed.visionEnabled, installed.visionDevice, installed.lowRamMode, installed.drafting, installed.resourceProfile].join('|') : `new|${selectedSize.id}`;
        if (sig === appliedSig)
            return;
        setAppliedSig(sig);
        if (installed) {
            if (installed.context)
                setContext(installed.context);
            // Off only when the person turned it off (a saved choice); a model prepared without the picture files (images 'off') still starts with it ON.
            setVision(installed.visionEnabled === false ? 'no' : installed.visionDevice ?? (installed.images && installed.images !== 'off' ? installed.images : 'yes'));
            setKv(installed.kv ?? 'int8');
            setLowRam(installed.lowRamMode ?? 'auto');
            setDrafting(installed.drafting ?? 'auto');
            // A model never tuned has no saved choice: keep the page's own (Leave 20%) rather than flipping to Full performance.
            if (installed.resourceProfile && !loadedSelected?.resourceProfile)
                setResourceProfile(installed.resourceProfile);
        }
        else
            setVision('yes');
    }, [selectedSize?.id, installed?.model, installed?.ready, installed?.context, installed?.kv, installed?.images, installed?.visionEnabled, installed?.visionDevice, installed?.lowRamMode, installed?.resourceProfile]);
    useEffect(() => { if (jobId && job?.state === 'done' && job.id !== toastedJob && (job.kind === 'load' || job.kind === 'install')) {
        setToastedJob(job.id);
        selectRemoteModel(null);
        pushToast('success', 'Model ready', 'Start a new chat when you’re ready.');
    } }, [job?.state]);
    async function perform(action: () => Promise<unknown>) { setPending(true); setError(''); try {
        await action();
    }
    catch (e) {
        setError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setPending(false);
    } }
    async function load() { if (!selectedSize)
        return; await perform(async () => { if (installed?.ready && context === installed.context && (!selectedSize?.images || vision === 'no' || installed.images && installed.images !== 'off') && (!installed.kv || kv === installed.kv || kv === tunedNow?.fields.kv) && (lowRam === (installed.lowRamMode ?? 'auto') || lowRam === tunedNow?.fields.lowRam)) {
        const result = await nativeAction<NativeJob>('operations', { kind: 'load', args: { model: selectedSize.id, resourceProfile, vision: selectedSize?.images ? vision : 'no', multiple: expert && multiple, ...(draftingChoices ? { values: { drafting } } : {}), ...(roleCards.length > 1 ? (roles && customRoles ? { gpuRoles: effectiveRoles, layerSplit: expert && cardsGpus.length > 1 ? split : 'auto' } : cardsGpus.length ? { gpus: cardsGpus, gpusFromRecommendation: true, layerSplit: expert && cardsGpus.length > 1 ? split : 'auto' } : {}) : {}) } });
        setJobId(result.id);
    }
    else
        setPlan(await api.post('/api/native/plan', downloadValues)); }); }
    /* Unload / Use in chat: a refusal (a chat is still using the model) is said here, beside the button, not far below the page. */
    async function loadedAction(path: string) { setLoadedError(''); setPending(true); try {
        await nativeAction(path);
    }
    catch (e) {
        setLoadedError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setPending(false);
    } }
    async function download() { await perform(async () => { const result = await nativeAction<NativeJob>('operations', { kind: 'install', args: downloadValues }); setJobId(result.id); setPlan(null); }); }
    return <Screen plainHeader title={<span className="sr-only">{tr("Models")}</span>} titleDetail={<FamilySwitch families={families} selected={selectedFamily?.family} onPick={f => { setFamily(f.family); setSize((f.sizes.find(s => native?.installed?.models.some(m => m.model === s.id && m.ready)) ?? f.sizes[0])?.id ?? ''); setVision('yes'); setError(''); }}/>} actions={<Btn icon="sliders" aria-pressed={expert} aria-expanded={expert} aria-controls="native-expert-settings" onClick={() => { setExpert(v => !v); setMultiple(false); }}>{tr(expert ? 'Close expert mode' : 'Expert mode')}</Btn>}>
  <div className="native-model-layout native-model-layout--fit">
   <div className="stack">
    {tr(native?.instances.length && expert ? <section className="st-card native-running"><h2 className="h4">{tr(native.instances.length === 1 ? 'Loaded model' : 'Loaded models')}</h2>{tr(native.instances.map(i => <div key={i.id}><span><strong>{tr(i.label)}</strong><small>{tr(i.gpus.length ? i.gpus.map((g, n) => `GPU ${g + 1}${i.gpus.length > 1 && n === i.gpus.length - 1 ? ' (Main)' : ''}`).join(' + ') : 'Automatic GPU')} · {tr(i.status === 'ready' ? i.phase : i.status)}{tr(native.defaultId === i.id ? ' · Chat default' : '')}</small>{tr(i.error ? <small className="inline-error">{tr(plainFailureText(i.error).message)}</small> : null)}</span>{tr(expert && multiple && native.defaultId !== i.id ? <Btn size="sm" variant="ghost" onClick={() => void loadedAction(`instances/${i.id}/default`)}>{tr("Use in chat")}</Btn> : null)}<Btn size="sm" disabled={pending || !!active} onClick={() => void loadedAction(`instances/${i.id}/unload`)}>{tr("Unload")}</Btn></div>))}{tr(loadedError ? <ErrorNotice message={loadedError} next={tr("Stop or answer the chat that is still using this model, then try again.")}/> : null)}</section> : null)}
    <section className="st-card native-choice">{tr(selectedFamily && (!selectedSize || selectedSize.needsNewerEngine) ? <><h2 className="h3">{tr("This model needs a newer engine.")}</h2><p className="small muted">{tr("Update your engine or import a local GGUF for this model.")}</p><Btn onClick={() => { setSettingsSection('Engine'); go('settings'); }}>{tr("Update engine")}</Btn><Btn variant="ghost" onClick={onImport ?? onOtherModels}>{tr("Use a local GGUF\u2026")}</Btn></> : null)}{tr(!selectedFamily ? <><h2 className="h3">{tr("Set up Strata")}</h2><p className="small muted">{tr("Connect your local installation to choose an optimized model.")}</p><Btn variant="primary" onClick={onConnectEngine ?? (() => { setSettingsSection('Engine'); go('settings'); })}>{tr("Connect engine")}</Btn>{tr(native?.error ? <p className="small inline-error">{tr(native.error)}</p> : null)}<Btn variant="ghost" onClick={onOtherModels}>{tr("Other local models")}</Btn></> : null)}
     {tr(selectedSize ? <>
      <div className="native-choice__head"><span className="native-choice__label">{tr("Version")}</span><span className="native-choice__tools"><NativePresets values={values} onApply={v => { const f = families.find(f => f.family === v.family || f.sizes.some(s => s.sourceFamily === v.family)); if (!f)
        return; presetPending.current = true; setFamily(f.family); setSize(f.sizes.find(s => s.model === v.model && (s.sourceFamily === v.family || f.family === v.family))?.id ?? ''); setContext(v.context ?? 32768); setVision(v.vision ?? 'no'); setKv(v.kv ?? 'int8'); setLowRam(v.low_ram ?? 'auto'); }}/><Btn icon="models" onClick={() => setManage(true)}>{tr("Manage models")}</Btn></span></div>
      <div className="native-version"><VersionPicker pick={native?.catalog?.recommendation} versions={allSizes} selectedId={selectedSize.id} hardware={fitHardware} downloaded={readyIds} onSelect={id => { const f = families.find(f => f.sizes.some(s => s.id === id)); if (f) { setFamily(f.family); setSize(id); setError(''); } }} onOwnGguf={() => (onImport ?? onOtherModels)()}/></div>
      {tr(noFit ? <p className="small muted native-spill-note"><Icon name="disk" size={16}/>{tr("May run slower \u00B7 Model data can be read from SSD.")}</p> : null)}
      <div className="native-choice__row">
       <div className="native-choice__col"><span className="native-choice__label">{tr("Conversation context")}</span><Select ariaLabel={tr("Conversation context")} value={String(context)} options={contextChoices(native?.catalog?.contexts ?? [8192, 32768, 65536, 131072, 262144], context).map(n => ({ value: String(n), label: n % 1024 === 0 ? `${n / 1024}K` : `${n.toLocaleString()} tokens` }))} onChange={n => setContext(Number(n))}/></div>
       {tr(draftingChoices ? <div className="native-choice__col"><span className="native-choice__label">{tr("Drafting")}</span><Select ariaLabel={tr("Drafting")} value={drafting} disabled={!!active} options={draftingChoices.map(v => ({ value: v, label: tr(draftingWords[v]) }))} onChange={v => setDrafting(String(v))}/></div> : null)}
       <div className="native-choice__col"><span className="native-choice__label">{tr("Computer use")}</span><div className="native-resource-choice"><button aria-pressed={resourceProfile === 'work'} data-selected={resourceProfile === 'work'} onClick={() => setResourceProfile('work')}><Icon name="computer" size={20}/><span><strong>{tr("Leave 20% for my work")}</strong><small>{tr("Headroom for your other apps")}</small></span>{tr(resourceProfile === 'work' ? <Icon name="check" size={15}/> : null)}</button><button aria-pressed={resourceProfile === 'performance'} data-selected={resourceProfile === 'performance'} onClick={() => setResourceProfile('performance')}><Icon name="bolt" size={20}/><span><strong>{tr("Full performance")}</strong><small>{tr("Give Strata priority")}</small></span>{tr(resourceProfile === 'performance' ? <Icon name="check" size={15}/> : null)}</button></div></div>
      </div>
      {tr(loadedSelected?.metrics?.engine?.drafting_note ? <p className="small muted native-spill-note">{tr(String(loadedSelected.metrics.engine.drafting_note))}</p> : null)}
      {roleCards.length > 1 ? <GpuRoles cards={roleCards} roles={effectiveRoles} custom={customRoles} recommendedRoles={recommended.roles} vendor={engineVendor} disabled={!!active} onChange={changeRoles} onBack={() => setRoles(null)} memoryGb={selectedSize ? memoryNeeded(selectedSize) || null : null} ramGb={pcRam} spillGb={selectedSize ? fitOn(selectedSize, fitHardware).spillGb : 0} usedBy={i => usedBy([i])}/> : null}
      {tr(roleCards.length > 1 && (native?.hardware?.gpus ?? []).length > roleCards.length ? <p className="small muted native-cards__note">{tr('Not used: {0} (another make of graphics card than this engine drives).', [(native?.hardware?.gpus ?? []).filter(g => !roleCards.some(r => r.index === g.index && r.vendor === g.vendor)).map(g => g.name).join(', ')])}</p> : null)}
      {tr(selectedSize?.imagesCapable && !selectedSize.images ? <p className="small muted native-spill-note">{tr("This model can read pictures, but the engine installed here does not include the picture reader yet, so Read pictures is off. Update the engine in Settings → About to turn it on.")}</p> : null)}
      {tr(selectedSize?.images ? <><div className="native-image-reading"><label><input type="checkbox" checked={vision !== 'no'} disabled={!selectedSize?.images || !!active} onChange={e => setVision(e.target.checked ? 'yes' : 'no')}/><span><strong>{tr("Read pictures ")}<small>{tr("On by default")}</small></strong><small>{tr("Understand photos, screenshots and charts. The picture files download automatically. Turn it off to save memory.")}</small></span></label>{tr(expert && vision !== 'no' ? <Select ariaLabel={tr("Image encoder device")} value={vision === 'yes' ? 'yes' : vision} options={[{ value: 'yes', label: 'Automatic' }, { value: 'gpu', label: 'GPU' }, { value: 'cpu', label: 'CPU' }]} onChange={setVision}/> : null)}</div>{tr(vision !== 'no' && installed?.ready && installed.visionPrepared === false ? <div className="native-vision-missing" role="status"><span>{tr("Picture reading needs a few more files that are not on this computer yet.")}</span><Btn size="sm" disabled={pending || !!active} onClick={() => void download()}>{tr("Download picture-reading files")}</Btn></div> : null)}</> : null)}
     </> : null)}
     <EngineWarning native={native}/>
     {tr(active ? <NativeProgress job={active} label={jobLabel(active)}/> : null)}
     {tr(job?.state === 'error' ? <ErrorNotice message={job.error ?? job.summary} action={tr("Retry")} onRetry={() => void load()}/> : null)}
     {tr(error ? <ErrorNotice message={error}/> : null)}
     {selectedSize && installed?.ready ? <NativeTuneBadge model={selectedSize.id} resourceProfile={resourceProfile} vision={selectedSize.images ? vision : 'no'}/> : null}
     <div className="native-model-actions">{tr(loadedSelected ? <><Btn disabled={pending || !!active} onClick={() => void perform(() => nativeAction(`instances/${loadedSelected.id}/unload`))}>{tr("Unload")}</Btn>{tr(onReady ? <Btn variant="primary" onClick={onReady}>{tr("Start chatting")}</Btn> : <span className="native-ready">{tr("Ready")}</span>)}</> : onReady && native?.instances.some(i => i.status === 'ready') ? <Btn variant="primary" onClick={onReady}>{tr("Start chatting")}</Btn> : <><Btn icon="gauge" disabled={!installed?.ready || !native?.available || !!active || !selectedSize || selectedSize.needsNewerEngine} onClick={() => setTuneOpen(true)}>{tr("Tune model")}</Btn><Btn variant="primary" icon="models" disabled={pending || !native?.available || !!active || !selectedSize || !!selectedSize.needsNewerEngine} onClick={() => void load()}>{tr(pending ? 'Checking…' : installed?.ready ? 'Load model' : 'Download & load')}</Btn></>)}</div>
    </section>
    {tr(expert && IMAGE_GENERATION_VISIBLE ? <><details className="native-generation"><summary>{tr("Image generation ")}<span className="muted">{tr("Optional")}</span></summary><RunImageModel library={imageLibrary} disabled={pending || !!active} onChange={() => { }}/></details></> : null)}
    {tr(expert ? <section id="native-expert-settings" ref={expertSection} className="st-card native-execution"><h2 className="h4">{tr("Expert settings")}</h2>{tr(tunedNow ? <p className="small muted native-tuned-note">{tr("Showing the values of your tuning '{0}'.", [tunedNow.name])}</p> : null)}<label className="native-toggle"><span><strong>{tr("Run multiple models")}</strong><small>{tr("Loading another model keeps the others loaded.")}</small></span><button className="st-toggle" role="switch" aria-label={tr("Run multiple models")} aria-checked={multiple} onClick={() => setMultiple(v => !v)}/></label>{tr(cardsGpus.length > 1 ? <label className="input-field"><span>{tr("Layer split")}</span><input className="st-input" value={split} placeholder={tr("auto or 16,32")} onChange={e => setSplit(e.target.value)}/></label> : null)}<div className="native-field-grid"><label className="input-field"><span>{tr("Context memory")}</span><Select ariaLabel={tr("Native KV format")} value={kv} options={[{ value: 'int8', label: 'INT8' }, { value: 'q4_0', label: 'Q4' }, { value: 'k8v4', label: 'K8 / V4' }]} onChange={setKv}/></label><label className="input-field"><span>{tr("Low-RAM mode")}</span><Select ariaLabel={tr("Native low RAM mode")} value={lowRam} options={['auto', 'resident', 'mmap'].map(v => ({ value: v, label: v === 'auto' ? 'Automatic' : v === 'resident' ? 'Keep experts in RAM' : 'Read experts from disk' }))} onChange={setLowRam}/></label></div>{tr(installed ? <><NativeSettingsFile model={installed.model}/><NativeSettings model={installed.model}/></> : <p className="small muted">{tr("More engine parameters become available after preparing this version.")}</p>)}</section> : null)}
   </div>
   <aside className="stack stack--lg"><PcFitCard gpuName={pcGpu?.name} ramGb={pcRam} vramGb={pcVram} inUse={keptModels.length ? { ramGb: Math.round(keptRam), vramGb: Math.round(keptVramOn(chosenCards)) } : null} fit={selectedSize ? fitOn(selectedSize, fitHardware) : null}/><section className="st-card"><h2 className="h4">{tr("Model files")}</h2><p className="small muted">{tr("Strata prepares all shards, the draft layer and shared files. Completed downloads are reused.")}</p><code className="native-path">{installed?.dataFolder ?? native?.downloadDirectory ?? native?.installed?.data_folder ?? 'Choose an installation'}</code><Btn size="sm" variant="ghost" onClick={() => { setSettingsSection('Storage'); go('settings'); }}>{tr("Storage settings")}</Btn></section><Btn size="sm" variant="ghost" onClick={onOtherModels}>{tr("Other local models\u2026")}</Btn></aside>
  </div>
  
<Dialog open={!!plan} title={tr(installed?.ready ? "Apply settings and load" : "Download and prepare")} compact onClose={() => !pending && setPlan(null)} actions={<><Btn disabled={pending} onClick={() => setPlan(null)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={pending || !!plan?.plan?.disk_short} onClick={() => void download()}>{tr(installed?.ready ? "Apply & load" : "Download & load")}</Btn></>}><p><strong>{tr(plan?.plan?.title ?? selectedFamily?.title)}</strong>{tr(installed?.ready ? null : <>{tr(" \u00B7 approximately ")}{tr(plan?.plan?.download_gb ?? selectedSize?.download_gb)}{tr(" GB")}</>)}</p>{tr(plan?.drafting?.needed ? <p className="small muted">{tr("Choosing DFlash2 also downloads its drafter: about {0} GB, once. If it does not fit in graphics memory, the model uses the MTP instead.", [plan.drafting.gb])}</p> : null)}{tr(installed?.ready ? <p className="small muted">{tr("The model files are already on this computer. Nothing will be downloaded; only the settings you chose are applied.")}</p> : null)}{tr(plan?.plan?.disk_short ? <ErrorNotice message={plan.plan.disk_short}/> : null)}{tr(installed?.ready ? <p className="small muted">{tr("Any currently loaded native models will be unloaded first.")}</p> : <><p className="small muted">{tr("Existing files are checked and reused. Any currently loaded native models will be unloaded before preparing files.")}</p><p className="small muted">{tr("Your computer may feel slower while weights load. You can stop the download and resume later.")}</p></>)}{tr(selectedSize?.images ? <p className="small muted">{tr("Vision encoder files are included. Picture reading loads only when enabled.")}</p> : null)}</Dialog>
  <NativeDeleteDialog id={deleting} onClose={() => setDeleting(null)}/>
  <Dialog open={manage} title={tr("Manage models")} onClose={() => setManage(false)} actions={<Btn onClick={() => setManage(false)}>{tr("Done")}</Btn>}><div className="native-manage">
   {tr(downloaded.length ? <div className="native-manage-list" role="list">{tr(downloaded.map(m => { const v = allSizes.find(x => x.id === m.model), loaded = native?.instances.some(i => i.model === m.model && i.status === 'ready'); return <div className="native-manage-row" role="listitem" key={m.model} data-selected={selectedSize?.id === m.model || undefined}>
     <span className="native-manage-name"><strong>{tr(v ? `${catalogVersionLabel(v)}` : m.model_name ?? m.model)}</strong><small>{tr(v ? v.model : m.model)}{tr(m.diskGb ? ` · ${sizeGb(m.diskGb)}` : '')}</small></span>
     <span className={loaded ? 'st-pill st-pill--accent' : 'st-pill'}>{tr(loaded ? 'Loaded' : 'Downloaded')}</span>
     <span className="native-manage-actions"><Btn size="sm" disabled={!!active || !v} onClick={() => { if (v) setSize(v.id); setManage(false); }}>{tr(selectedSize?.id === m.model ? 'Selected' : 'Use version')}</Btn><Btn size="sm" variant="danger" disabled={!!active} onClick={() => setDeleting(m.model)}>{tr("Delete")}</Btn></span>
    </div>; }))}</div> : <p className="small muted native-manage-empty">{tr("No downloaded models")}</p>)}
   <div className="native-manage-foot"><Btn variant="ghost" icon="folder" onClick={() => { setManage(false); (onImport ?? onOtherModels)(); }}>{tr("Use a local GGUF…")}</Btn></div>
  </div></Dialog>
 {selectedSize ? <NativeTune open={tuneOpen} onClose={() => setTuneOpen(false)} model={selectedSize.id} resourceProfile={resourceProfile} vision={selectedSize.images ? vision : 'no'}/> : null}
 </Screen>;
}
/** setup's "5/7: downloading …" as a sentence a person can read. */
/** The listed context sizes, plus the model's own when it is a size set by hand in its run config (the select would show nothing, and a load would silently pick another). */
export function contextChoices(listed: number[], current: number): number[] {
    return current > 0 && !listed.includes(current) ? [...listed, current].sort((a, b) => a - b) : listed;
}
function installStepLabel(step?: string): string {
    const n = Number(step?.match(/^(\d+)\//)?.[1]);
    if (!n)
        return step ? 'Getting things ready…' : 'Downloading and preparing model files…';
    return n <= 2 ? 'Checking your computer…' : n <= 4 ? 'Getting things ready…' : n === 5 ? 'Downloading your model…' : n === 6 ? 'Preparing your model for Strata…' : 'Finishing up…';
}
export function NativeProgress({ job, label }: {
    job: NativeJob;
    label?: string;
}) { const download = installProgress(job.result?.install); const detailsText = [job.details, job.result?.install?.engine?.details].filter(Boolean).join('\n'); const eta = useDownloadEta(job.kind === 'install' ? download : undefined, job.id); const [stopping, setStopping] = useState(false), [error, setError] = useState(''); async function stop() { setStopping(true); setError(''); try {
    await nativeAction(`operations/${job.id}/cancel`);
}
catch (e) {
    setError(e instanceof Error ? e.message : String(e));
}
finally {
    setStopping(false);
} } return <section className="st-card native-progress" aria-live="polite"><div className="section-toolbar"><h2 className="h4">{tr(job.kind === 'install' ? (label ? tr('Preparing {0}', [label]) : 'Preparing your model') : job.kind === 'load' ? 'Loading model' : job.kind === 'autotune' || job.kind === 'recheck' ? 'Tuning your model' : 'Engine operation')}</h2>{tr(job.kind === 'install' || job.kind === 'load' || job.kind === 'autotune' || job.kind === 'recheck' ? <div className="onboard-progress-actions">{tr(eta ? <span className="small muted">{eta}</span> : null)}<Btn size="sm" disabled={stopping} onClick={() => void stop()}>{tr(stopping ? 'Stopping…' : 'Stop')}</Btn></div> : null)}</div><p className="small">{tr(job.kind === 'install' ? (job.result?.install?.label ?? (download && /^5\//.test(job.result?.install?.step ?? '') ? stageText(installStage(job.result?.install)) : installStepLabel(job.result?.install?.step))) : job.summary)}</p>{tr(download ? <><progress max={100} value={download.percent ?? 0}/><p className="xs muted">{tr(download.file === 'model files' ? 'Model files' : download.file === 'picture reader' ? 'Picture reader' : download.file === 'checking files' ? 'Checking files' : download.file === 'MTP draft layer' ? 'Draft layer' : download.file)} · {tr(download.percent ?? 0)}%{tr(Number.isFinite(download.done_gb) && Number.isFinite(download.total_gb) ? ` · ${download.done_gb!.toFixed(1)} / ${download.total_gb!.toFixed(1)} GB` : '')}</p></> : null)}<DownloadParts parts={job.result?.install?.parts}/>{tr(detailsText ? <details><summary className="small">{tr("Details")}</summary><pre className="small muted error-detail-text">{detailsText}</pre></details> : null)}{tr(error ? <ErrorNotice message={error}/> : null)}</section>; }
