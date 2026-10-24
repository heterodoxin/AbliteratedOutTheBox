import { tr } from '../lib/i18n';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ActionError, Btn, Dialog, ErrorNotice, Icon, Select } from '../components/ui';
import { Screen } from '../components/layout';
import { DemoTest } from '../components/EngineExperience';
import { RunExpertEditor } from '../components/RunExpertEditor';
import { RunPreview } from '../components/RunPreview';
import { api } from '../lib/client';
import { RunningModels } from '../components/RunningModels';
import { LocalModelImport } from '../components/LocalModelImport';
import { gpuId } from '../lib/monitor-hardware';
import { GpuRoles } from '../components/GpuRoles';
import { normalize as normalizeRoles, order as roleOrder, recommend as recommendRoles, sameRoles, type RoleCard, type Roles } from '../lib/gpu-roles';
import { RunImageModel } from '../components/RunImageModel';
import { IMAGE_GENERATION_VISIBLE } from '../lib/features';
import { FloatingMenu } from '../components/FloatingMenu';
import { ModelsScreen } from './Models';
import { configDocument, configPatch, readConfig, type ConfigDocument } from '../lib/config';
import { contextTokens, estimate, execution } from '../lib/experience';
import { demoRunExperience, planMemory } from '../lib/run-memory';
import { computerIdentity, configIdentity, defaultProfileKey, emptyProfiles, readProfiles, recommendedConfig, resourceConfig, RUN_PROFILES_KEY, saveProfile, type ComputerUse, type RunProfiles } from '../lib/run-profiles';
import { modelFamilyIcon } from '../lib/model-icons';
import { familyVersion, modelFamilies } from '../lib/model-versions';
import { sizeGb } from '../lib/format';
import { applyImageModel, getState, useLifecycle, loadRuntimeModel, loadVariant, pushToast, restartEngine, saveEngine, selectImageModel, setConfigDraft, useStore, validateEngine } from '../lib/store';
import type { ImageModelChoice, RuntimeModels, RuntimeModel } from '../lib/types';
import { useNative } from '../lib/native';
import { NativeModels } from '../components/NativeModels';
import { NativeLocalModel } from '../components/NativeLocalModel';
import { parseLocation } from '../lib/router';
type Issue = {
    level: 'error' | 'warn';
    message: string;
};
type RunStage = 'idle' | 'checking' | 'loading' | 'ready';
export function ModelRunScreen() {
    const native = useNative();
    const location = parseLocation(window.location.hash), entry = location.param;
    const [other, setOther] = useState(entry === 'local' || entry === 'local-import'), [importFirst, setImportFirst] = useState(entry === 'local-import');
    if (native && other)
        return <NativeLocalModel onBack={() => { setOther(false); setImportFirst(false); }}/>;
    if (native && !other)
        return <NativeModels onOtherModels={() => setOther(true)} onImport={() => { setImportFirst(true); setOther(true); }}/>;
    return <><LegacyModelRunScreen importFirst={importFirst} initialVariant={location.query.get('variant') ?? undefined}/>{tr(other ? <button className="st-btn native-return" onClick={() => setOther(false)}>{tr("Back")}</button> : null)}</>;
}
function LegacyModelRunScreen({ importFirst = false, initialVariant }: {
    importFirst?: boolean;
    initialVariant?: string;
}) {
    const engine = useStore(s => s.engine), hardware = useStore(s => s.hardware), library = useStore(s => s.library);
    const runtime = useStore(s => s.runtime);
    const [deviceChoice, setDeviceChoice] = useState('auto');
    const [customGpus, setCustomGpus] = useState<string[]>([]);
    // Two or more cards (Expert settings): a role for each card instead of the "Run on" list. null = the recommended roles.
    const [runRoles, setRunRoles] = useState<Roles | null>(null);
    const placementInitialized = useRef('');
    const imageModels = useStore(s => s.imageModels), imagePending = useStore(s => !!s.busy['image:select']);
    const imageChoice = imageModels?.selected ?? 'off';
    const imageVersion = imageModels?.variants.find(v => v.id === imageChoice);
    const imageNeedsDownload = imageChoice !== 'off' && imageVersion?.state !== 'installed';
    const imageUnavailable = imageChoice !== 'off' && !imageModels?.generationAvailable;
    const rawDraft = useStore(s => s.configDraft), connected = useStore(s => s.connection === 'live');
    const lifecycle = useLifecycle();
    const [selected, setSelected] = useState(() => {
        if (initialVariant)
            return initialVariant;
        try {
            return sessionStorage.getItem('strata.run-variant.v1') ?? '';
        }
        catch {
            return '';
        }
    });
    const [expert, setExpert] = useState(false);
    const [allowMultiple, setAllowMultiple] = useState(false);
    const executionMode = expert && allowMultiple ? 'multiple' : 'single';
    const [basicsOpen, setBasicsOpen] = useState(false);
    const [libraryOpen, setLibraryOpen] = useState(false);
    const [localModelOpen, setLocalModelOpen] = useState(importFirst);
    const [testOpen, setTestOpen] = useState(false);
    const [profileOpen, setProfileOpen] = useState(false);
    const [profileMenu, setProfileMenu] = useState(false);
    const profileTrigger = useRef<HTMLButtonElement>(null);
    const familySelections = useRef(new Map<string, string>());
    const familyDrafts = useRef(new Map<string, {
        variantId: string;
        draft: string | null;
        profile: string;
        image: ImageModelChoice;
    }>());
    const [profileName, setProfileName] = useState('Everyday');
    const [makeDefault, setMakeDefault] = useState(false);
    const [updateExisting, setUpdateExisting] = useState(false);
    const [activeProfile, setActiveProfile] = useState('current');
    const [profiles, setProfiles] = useState<RunProfiles>(emptyProfiles);
    const [profileError, setProfileError] = useState('');
    const [issues, setIssues] = useState<Issue[]>([]);
    const [pending, setPending] = useState(false);
    const [placementError, setPlacementError] = useState('');
    const [placementPlan, setPlacementPlan] = useState<{
        memory: RuntimeModel['memory'];
        budget: RuntimeModels['memory'];
        allowed: boolean;
        alreadyLoaded: boolean;
        reloads: boolean;
    } | null>(null);
    const [stage, setStage] = useState<RunStage>('idle');
    const [discardOpen, setDiscardOpen] = useState(false);
    const initialized = useRef(false);
    const saved = useMemo(() => engine ? configDocument(engine) : null, [engine]);
    const text = rawDraft ?? (saved ? JSON.stringify(saved, null, 2) : '');
    const parsed = useMemo(() => {
        try {
            return { config: readConfig(text), error: '' };
        }
        catch (error) {
            return { config: null, error: error instanceof Error ? error.message : String(error) };
        }
    }, [text]);
    const config = parsed.config;
    const running = library?.variants.find(v => v.id === library.loaded);
    const model = library?.variants.find(v => v.id === selected) ?? running ?? library?.variants.find(v => v.state === 'installed') ?? library?.variants[0];
    const computer = hardware ? computerIdentity(hardware) : '';
    const localProfiles = profiles.profiles.filter(p => p.variantId === model?.id && p.hardware === computer);
    const selectedProfile = localProfiles.find(p => p.id === activeProfile);
    const personalDefault = model ? profiles.defaults[defaultProfileKey(computer, model.id)] : undefined;
    const profileChanged = !!selectedProfile && !!config && (configIdentity(config) !== configIdentity(selectedProfile.config) || imageChoice !== (selectedProfile.imageModel ?? 'off'));
    const use = config?.headroom === 'full' ? 'full' : 'shared';
    const ctx = contextTokens(config?.beginner.contextLength ?? 'recommended');
    const roleCards: RoleCard[] = (hardware?.gpus ?? []).map((g, i) => ({ index: i, name: g.name, vram_gb: g.vramGb ?? 0, integrated: !!g.unified, vendor: g.vendor }));
    const rolesMode = expert && roleCards.length > 1;
    const recommendedRoles = recommendRoles(roleCards).roles, effectiveRoles: Roles = runRoles ? normalizeRoles(runRoles, roleCards) : recommendedRoles;
    const effectiveDevice = expert ? deviceChoice : 'auto';
    const gpuIds = hardware ? (rolesMode ? roleOrder(effectiveRoles, roleCards).map(i => gpuId(hardware.gpus[i])) : effectiveDevice === 'auto' ? hardware.gpus.map(gpuId) : effectiveDevice === 'cpu' ? [] : effectiveDevice === 'custom' ? customGpus : [effectiveDevice]) : [];
    const executionHardware = hardware ? { ...hardware, gpus: hardware.gpus.filter(g => gpuIds.includes(gpuId(g))), accelerator: { ...hardware.accelerator, backend: gpuIds.length ? hardware.accelerator.backend : 'cpu', unified: gpuIds.length ? hardware.accelerator.unified : false, usableVramGb: hardware.gpus.filter(g => gpuIds.includes(gpuId(g))).reduce((n, g) => n + g.vramGb, 0) } } : null;
    const cfg = execution({ ...config?.expert, ...(!expert ? { 'gpu.layers': gpuIds.length ? 'Automatic' : 'CPU only' } : !gpuIds.length ? { 'gpu.layers': 'CPU only' } : {}) }, ctx);
    const baseExperience = estimate(model, executionHardware, cfg, 'document', library?.mode === 'simulated' && !model?.local);
    const baseMemory = planMemory(baseExperience, executionHardware, use, Number(config?.expert['server.parallel'] ?? 1), config?.expert['mem.mmap'] !== false && config?.expert['mem.lock'] !== true);
    const memory = baseMemory && placementPlan ? { ...baseMemory, total: placementPlan.memory.totalGb, ram: placementPlan.memory.ramGb, gpu: placementPlan.memory.gpuGb, ssd: placementPlan.memory.spillGb } : baseMemory;
    const experience = demoRunExperience(baseExperience, memory, use);
    const contextMax = Math.floor((model?.contextMax ?? 1048576) / 1024);
    const unsupportedContext = !!model && ctx > model.contextMax;
    const invalid = expert && deviceChoice === 'custom' && !gpuIds.length || !!parsed.error || issues.some(i => i.level === 'error') || unsupportedContext;
    useEffect(() => {
        if (!hardware || !model || !engine || initialized.current)
            return;
        initialized.current = true;
        try {
            const stored = readProfiles(localStorage.getItem(RUN_PROFILES_KEY));
            setProfiles(stored);
            const id = stored.defaults[defaultProfileKey(computer, model.id)];
            const profile = stored.profiles.find(p => p.id === id);
            if (profile && (rawDraft === null || parsed.config && configIdentity(parsed.config) === configIdentity(profile.config))) {
                setActiveProfile(profile.id);
                if (rawDraft === null)
                    setConfigDraft(JSON.stringify(profile.config, null, 2));
                // Retain an image selection that has not yet been applied.
                if (imageModels?.selected === imageModels?.applied && (profile.imageModel ?? 'off') !== imageChoice)
                    void chooseImage(profile.imageModel ?? 'off');
            }
        }
        catch (error) {
            setProfileError(error instanceof Error ? error.message : 'Run profiles could not be loaded.');
        }
    }, [hardware, model?.id, engine, computer, rawDraft]);
    useEffect(() => {
        if (model && (!selected || !library?.variants.some(v => v.id === selected)))
            setSelected(model.id);
    }, [model?.id, selected, library]);
    useEffect(() => {
        if (selected)
            try {
                sessionStorage.setItem('strata.run-variant.v1', selected);
            }
            catch { /* Selection remains available for this session. */ }
    }, [selected]);
    useEffect(() => {
        if (!model || !runtime || placementInitialized.current === model.id)
            return;
        placementInitialized.current = model.id;
        const existing = runtime.instances.find(i => i.variantId === model.id);
        const ids = existing?.gpuIds;
        setDeviceChoice(runtime.mode === 'real' || !ids ? 'auto' : !ids.length ? 'cpu' : ids.length === 1 ? ids[0] : 'custom');
        setCustomGpus(ids ?? []);
    }, [model?.id, runtime?.mode]);
    useEffect(() => {
        if (model)
            familySelections.current.set(model.family, model.id);
    }, [model?.id]);
    useEffect(() => {
        if (!config || !model || !connected)
            return;
        let active = true;
        const timer = setTimeout(() => void validateEngine({ ...configPatch(config), variantId: model.id }).then(result => {
            if (active)
                setIssues(result.issues.filter(i => !i.message.startsWith('Estimated working memory (') && !i.message.includes('is fast memory. Actual fit')));
        }).catch(error => {
            if (active)
                setIssues([{ level: 'error', message: error.message }]);
        }), 180);
        return () => { active = false; clearTimeout(timer); };
    }, [text, model?.id, connected]);
    useEffect(() => {
        setPlacementPlan(null);
        setPlacementError('');
        if (!config || !model || !hardware || !connected)
            return;
        let active = true;
        const timer = setTimeout(() => {
            const patch = loadPatch(config);
            void api.post<NonNullable<typeof placementPlan>>('/api/runtime/plan', { variantId: model.id, gpuIds, config: patch, executionMode }).then(plan => {
                if (active)
                    setPlacementPlan(plan);
            }).catch(error => {
                if (active) {
                    setPlacementPlan(null);
                    if (library?.mode === 'simulated')
                        setPlacementError(error instanceof Error ? error.message : String(error));
                }
            });
        }, 180);
        return () => { active = false; clearTimeout(timer); };
    }, [text, model?.id, deviceChoice, customGpus.join(','), JSON.stringify(runRoles), runtime?.revision, connected, expert, executionMode]);
    function loadPatch(doc: ConfigDocument) {
        const patch = configPatch(doc);
        if (!expert || !gpuIds.length)
            patch.expert = patch.expert.map(p => p.id === 'gpu.layers' ? { ...p, value: gpuIds.length ? 'Automatic' : 'CPU only' } : p);
        return patch;
    }
    function update(next: ConfigDocument) { setIssues([]); setStage('idle'); setConfigDraft(JSON.stringify(next, null, 2)); }
    async function chooseImage(choice: ImageModelChoice) {
        setStage('idle');
        try {
            await selectImageModel(choice);
        }
        catch (error) {
            setProfileError(error instanceof Error ? error.message : String(error));
        }
    }
    function setContext(k: number) {
        if (config)
            update({ ...config, beginner: { ...config.beginner, contextLength: `${k}k` }, expert: { ...config.expert, 'ctx.length': `${k}K` } });
    }
    function chooseProfile(id: string) {
        if (!engine || !hardware)
            return;
        setActiveProfile(id);
        setStage('idle');
        const profile = profiles.profiles.find(p => p.id === id);
        if (profile) {
            update(structuredClone(profile.config));
            void chooseImage(profile.imageModel ?? 'off');
        }
        else if (id === 'current') {
            setConfigDraft(null);
            void chooseImage(imageModels?.applied ?? 'off');
        }
        else {
            update(recommendedConfig(engine, hardware, 'recommended', id === 'maximum' ? 'full' : 'shared'));
            void chooseImage('off');
        }
    }
    function chooseModel(id: string) {
        if (id === model?.id)
            return;
        const next = getState().library?.variants.find(v => v.id === id);
        if (model && next && next.family !== model.family) {
            familyDrafts.current.set(model.family, { variantId: model.id, draft: rawDraft, profile: activeProfile, image: imageChoice });
            const previous = familyDrafts.current.get(next.family);
            if (previous?.variantId === id) {
                setSelected(id);
                setStage('idle');
                setIssues([]);
                setActiveProfile(previous.profile);
                setConfigDraft(previous.draft);
                void chooseImage(previous.image);
                return;
            }
        }
        setSelected(id);
        setStage('idle');
        setIssues([]);
        const profileId = profiles.defaults[defaultProfileKey(computer, id)];
        if (profileId)
            chooseProfile(profileId);
        else {
            setActiveProfile('current');
            if (engine && hardware) {
                const nextConfig = recommendedConfig(engine, hardware);
                if (next?.local) {
                    const k = [4, 8, 16, 32, 64, 128, 256, 512, 1024].filter(k => k * 1024 <= next.contextMax).at(-1) ?? 4;
                    const context = Math.min(32, k);
                    nextConfig.beginner.contextLength = `${context}k`;
                    nextConfig.expert['ctx.length'] = `${context}K`;
                }
                update(nextConfig);
            }
        }
    }
    function chooseFamily(family: string) {
        if (!library || family === model?.family)
            return;
        const version = familyVersion(library, family, model?.quant, familySelections.current.get(family));
        if (version)
            chooseModel(version.id);
    }
    function chooseComputerUse(value: ComputerUse) {
        if (config && hardware)
            update(resourceConfig(config, value, hardware));
    }
    function persist(next: RunProfiles) {
        localStorage.setItem(RUN_PROFILES_KEY, JSON.stringify(next));
        setProfiles(next);
        setProfileError('');
    }
    function openSave(asNew = false) {
        setUpdateExisting(!asNew && !!selectedProfile);
        setProfileName(!asNew && selectedProfile ? selectedProfile.name : use === 'full' ? 'Maximum performance' : 'Everyday');
        setMakeDefault(!asNew && !!selectedProfile && selectedProfile.id === personalDefault);
        setProfileOpen(true);
    }
    async function storeProfile() {
        if (!model || !config || !hardware || !engine)
            return;
        setPending(true);
        try {
            const checked = await validateEngine({ ...configPatch(config), variantId: model.id });
            if (!checked.ok) {
                setIssues(checked.issues);
                throw new Error('Correct the configuration before saving a profile.');
            }
            const profile = { id: updateExisting && selectedProfile ? selectedProfile.id : crypto.randomUUID(), name: profileName.trim(), variantId: model.id, hardware: computer, engineVersion: engine.engine.version, config: structuredClone(config), imageModel: imageChoice, updatedAt: new Date().toISOString() };
            persist(saveProfile(profiles, profile, makeDefault));
            setActiveProfile(profile.id);
            setProfileOpen(false);
            pushToast('success', 'Settings saved', makeDefault ? 'Selected automatically for this model on this computer.' : 'Ready to restore from Saved settings.');
        }
        catch (error) {
            setProfileError(error instanceof Error ? error.message : String(error));
        }
        finally {
            setPending(false);
        }
    }
    async function tune() {
        if (!engine || !hardware || !config || !model || pending)
            return;
        setPending(true);
        setStage('checking');
        setIssues([]);
        try {
            const tuned = recommendedConfig(engine, executionHardware ?? hardware, config.beginner.contextLength, use);
            const checked = await validateEngine({ ...configPatch(tuned), variantId: model.id });
            if (!checked.ok) {
                setIssues(checked.issues);
                throw new Error('Review the highlighted settings before loading.');
            }
            update(tuned);
            setStage('ready');
            pushToast('success', 'Model tuned', 'Settings are ready. Load model when you’re ready.');
        }
        catch (error) {
            setStage('idle');
            setIssues([{ level: 'error', message: error instanceof Error ? error.message : String(error) }]);
        }
        finally {
            setPending(false);
        }
    }
    async function run() {
        if (!config || !model || pending || imagePending || imageNeedsDownload || imageUnavailable)
            return;
        setPending(true);
        setStage('loading');
        setIssues([]);
        try {
            const patch = loadPatch(config);
            const checked = await validateEngine({ ...patch, variantId: model.id });
            if (!checked.ok) {
                setIssues(checked.issues);
                throw new Error('Review the highlighted settings before loading.');
            }
            if (library?.mode === 'simulated')
                await loadRuntimeModel({ variantId: model.id, gpuIds, config: patch, executionMode });
            else {
                if (!lifecycle.allowed)
                    throw new Error(lifecycle.reason);
                if (expert && (rolesMode ? !!runRoles : deviceChoice !== 'auto'))
                    throw new Error('GPU assignment requires a native engine adapter. Use Automatic with this host.');
                await saveEngine(patch, true, true);
                const ok = getState().library?.loaded === model.id ? await restartEngine() : await loadVariant(model.id);
                if (!ok)
                    throw new Error(getState().actionError?.message ?? 'The model could not be loaded.');
            }
            await applyImageModel();
            setStage('ready');
            pushToast('success', 'Model loaded', executionMode === 'multiple' ? 'Other loaded models stay available.' : 'Ready for requests.');
        }
        catch (error) {
            setStage('idle');
            setIssues([{ level: 'error', message: error instanceof Error ? error.message : String(error) }]);
        }
        finally {
            setPending(false);
        }
    }
    if (!engine || !library || !model || !saved)
        return <Screen title={tr("Model")}><p className="small muted">{tr(connected ? 'Loading your models and settings…' : 'Connect the engine to configure a model.')}</p></Screen>;
    const versions = library.variants.filter(v => v.family === model.family || !!v.local);
    const installed = ['loaded', 'installed', 'loading', 'unloading'].includes(model.state) || !!model.installedAt;
    const working = ['downloading', 'loading', 'unloading'].includes(model.state);
    const modelSummary = <><Icon name={model.local ? 'model-file' : modelFamilyIcon(model.family)} size={21}/><span><strong>{tr(model.local ? model.name : model.family)}</strong><small>{tr(model.local ? model.quant : model.name)} · {tr(ctx / 1024)}{tr("K \u00B7 ")}{tr(use === 'full' ? 'Maximum performance' : 'Room for other apps')}{tr(imageChoice !== 'off' ? ` · Image ${imageChoice.toUpperCase()}` : '')}</small></span></>;
    return <Screen plainHeader title={tr("Model")} titleDetail={<span className="run-family-switcher"><Select ariaLabel={tr("Model family")} value={model.family} disabled={pending || imagePending} menuWidth={340} menuClassName="model-family-menu" optionIcon={modelFamilyIcon} options={modelFamilies(library).map(family => ({ value: family, label: family }))} onChange={chooseFamily}/></span>} actions={<>{tr(expert ? <><Btn size="sm" disabled={!config || invalid || pending} onClick={() => setTestOpen(true)}><span className="run-header-full">{tr("Benchmark")}</span><span className="run-header-short">{tr("Benchmark")}</span></Btn><Btn size="sm" disabled={!hardware || pending} onClick={() => {
                    if (hardware) {
                        setActiveProfile('recommended');
                        update(recommendedConfig(engine, hardware));
                        void chooseImage('off');
                    }
                }}><span className="run-header-full">{tr("Reset to recommended")}</span><span className="run-header-short">{tr("Reset defaults")}</span></Btn></> : null)}<button className="st-btn st-btn--secondary" aria-pressed={expert} disabled={pending} onClick={() => setExpert(!expert)}><Icon name="sliders" size={16}/>{tr(expert ? <><span className="run-header-full">{tr("Close expert mode")}</span><span className="run-header-short">{tr("Easy mode")}</span></> : 'Expert mode')}</button></>}>
    <div className="model-run">
      <ActionError />
      <div className="run-toolbar">
        {tr(expert ? <button className="run-basics-summary run-toolbar-summary" aria-expanded={basicsOpen} aria-controls="run-basic-settings" onClick={() => setBasicsOpen(!basicsOpen)}>{tr(modelSummary)}<Icon name="chevron" size={15}/></button> : <div className="run-basics-summary run-toolbar-summary run-model-summary">{tr(modelSummary)}</div>)}
        <button ref={profileTrigger} className="st-btn st-btn--secondary run-saved-settings" aria-label={tr("Saved settings{0}{1}", [selectedProfile ? `: ${selectedProfile.name}` : tr(''), profileChanged ? tr(' — unsaved changes') : tr('')])} aria-haspopup="menu" aria-expanded={profileMenu} disabled={pending} onClick={() => setProfileMenu(!profileMenu)}><Icon name="layers" size={17}/><span>{tr(selectedProfile?.name ?? 'Saved settings')}</span>{tr(profileChanged ? <i title={tr("Unsaved changes")}/> : null)}<Icon name="chevron" size={14}/></button>
        <Btn icon="models" size="sm" onClick={() => setLibraryOpen(true)}>{tr("Manage models")}</Btn>
      </div>
      {tr(expert ? <RunningModels controls/> : null)}
      {tr(profileError && !profileOpen ? <ErrorNotice message={profileError}/> : null)}
      <div className="run-layout">
        <div className="run-setup" data-expert={expert || undefined}>
          <div className="run-basics" id="run-basic-settings" hidden={expert && !basicsOpen}>
          <section className="run-card run-model-choice">
            <div className="run-version-choice"><span>{tr("Version")}</span><Select ariaLabel={tr("Model version")} value={model.id} disabled={pending} menuClassName="run-version-menu" options={[...versions.map(v => ({ value: v.id, label: `${v.name} · ${v.quant} · ${sizeGb(v.diskGb)}${v.local ? ' · Local' : ''}` })), { value: 'use-local-model', label: 'Use a local model…' }]} onChange={id => id === 'use-local-model' ? setLocalModelOpen(true) : chooseModel(id)}/></div>
            {tr(model.local ? <div className="run-local-model-note"><Icon name="model-file" size={15}/><span>{tr("Local model \u00B7 ")}{tr(model.local.architecture)}{tr(" \u00B7 Compatibility checked on load.")}</span><button className="text-btn" onClick={() => setLocalModelOpen(true)}>{tr("Change file")}</button></div> : null)}
            {IMAGE_GENERATION_VISIBLE ? <RunImageModel library={imageModels} disabled={pending || imagePending || !connected} onChange={() => setStage('idle')}/> : null}
          </section>
          <section className="run-card run-context"><div className="run-card__head"><span><Icon name="chat" size={19}/><h2>{tr("Conversation context")}</h2></span><strong>{tr(ctx / 1024)}{tr("K ")}<small>{tr("tokens")}</small></strong></div><p>{tr("How much your model can keep in mind.")}</p>
            <div className="run-context-presets" role="group" aria-label={tr("Context presets")}>{tr([32, 64, 128, 256].map(k => <button key={k} disabled={k > contextMax} aria-pressed={ctx === k * 1024} onClick={() => setContext(k)}><span>{tr(k)}{tr("K")}</span></button>))}</div>
            <div className="run-context-bottom"><label className="run-context-exact">{tr("Custom ")}<input className="st-input" aria-label={tr("Custom context in K tokens")} type="number" min="4" max={contextMax} step="1" value={Number.isFinite(ctx) ? ctx / 1024 : ''} onChange={e => {
            if (e.target.value)
                setContext(Number(e.target.value));
        }}/><span>{tr("K")}</span></label></div>
            {tr(unsupportedContext ? <p className="inline-error xs">{tr("This model supports up to ")}{tr(contextMax)}{tr("K. Choose a supported context to run it.")}</p> : null)}
          </section>
          <section className="run-card run-resources"><div className="run-card__head"><span><Icon name="computer" size={19}/><h2>{tr("Computer use")}</h2></span></div>
            <div className="run-resource-options">{tr([{ id: 'shared' as const, icon: 'computer' as const, title: 'Leave room for other apps', hint: 'Work alongside your model' }, { id: 'full' as const, icon: 'bolt' as const, title: 'Maximum performance', hint: 'Give your model more resources' }].map(option => <button key={option.id} aria-pressed={use === option.id} disabled={!hardware} onClick={() => chooseComputerUse(option.id)}><span><Icon name={option.icon} size={20}/><i>{tr(use === option.id ? <Icon name="check" size={12}/> : null)}</i></span><strong>{tr(option.title)}</strong><small>{tr(option.hint)}</small></button>))}</div>
            <p className="run-resources__hint"><Icon name="cpu" size={14}/>{tr(cfg.threads)}{tr(" CPU threads")}{tr(memory ? ` · ${memory.reserve.toFixed(0)} GB RAM headroom in the estimate` : '')}</p>
          </section>
          </div>
          {tr(expert ? <section className="run-card run-placement">
            <div className="run-card__head"><span><Icon name="gpu" size={19}/><h2>{tr("Model execution")}</h2></span></div>
            <label className="run-multiple-choice"><span><strong>{tr("Run multiple models")}</strong><small>{tr("Keep models loaded when you add another.")}</small></span><input type="checkbox" role="switch" aria-label={tr("Run multiple models")} checked={allowMultiple} disabled={pending || runtime?.capabilities.multipleModels === false} onChange={e => { setAllowMultiple(e.target.checked); setIssues([]); setStage('idle'); }}/></label>
            {rolesMode ? <GpuRoles cards={roleCards} roles={effectiveRoles} custom={!!runRoles && !sameRoles(effectiveRoles, recommendedRoles)} disabled={pending || runtime?.capabilities.deviceAssignment === false} onChange={next => { setRunRoles(sameRoles(next, recommendedRoles) ? null : next); setStage('idle'); setIssues([]); }} onBack={() => { setRunRoles(null); setStage('idle'); setIssues([]); }}/> : <>
            <span className="run-device-label">{tr("Run on")}</span>
            <Select ariaLabel={tr("Execution device")} value={deviceChoice} disabled={pending || runtime?.capabilities.deviceAssignment === false} options={[{ value: 'auto', label: hardware?.gpus.length ? hardware.gpus.length > 1 ? 'Automatic · all GPUs' : 'Automatic · GPU 1' : 'CPU only' }, { value: 'cpu', label: 'CPU only' }, ...(hardware?.gpus ?? []).map((g, i) => ({ value: gpuId(g), label: `GPU ${i + 1} · ${g.name}` })), ...((hardware?.gpus.length ?? 0) > 1 ? [{ value: 'custom', label: 'Choose GPUs…' }] : [])]} onChange={id => { setDeviceChoice(id); setStage('idle'); setIssues([]); }}/>
            {tr(deviceChoice === 'custom' ? <div className="run-gpu-options">{tr(hardware?.gpus.map((g, i) => <label key={gpuId(g)}><input type="checkbox" checked={customGpus.includes(gpuId(g))} onChange={e => { setCustomGpus(ids => e.target.checked ? [...ids, gpuId(g)] : ids.filter(id => id !== gpuId(g))); setStage('idle'); }}/><span>{tr("GPU ")}{tr(i + 1)} · {tr(g.name)}<small>{tr(g.unified ? 'Shared system memory' : `${sizeGb(g.vramGb)} VRAM`)}</small></span></label>))}</div> : null)}
            </>}
            <p className="xs muted">{tr(runtime?.capabilities.multipleModels === false ? 'This native host currently supports one model with automatic GPU placement.' : !allowMultiple ? 'One model at a time. Loading switches to the selected model.' : placementPlan?.reloads ? 'Loading applies these settings to this instance. Other models stay loaded.' : 'Loading adds a model. Existing models keep their settings and GPU assignments.')}</p>
            {tr(runtime?.instances.length ? <div className="run-memory-reservations"><span>{tr(placementPlan && !placementPlan.alreadyLoaded ? 'Shared RAM after load' : 'Shared RAM reserved')} <strong>{tr((placementPlan?.budget.ramGb ?? runtime.memory.ramGb).toFixed(1))} / {tr((placementPlan?.budget.budgetGb ?? runtime.memory.budgetGb).toFixed(1))}{tr(" GB")}</strong></span><small>{tr(runtime.memory.reserveGb.toFixed(0))}{tr(" GB RAM headroom \u00B7 estimates include all loaded models")}</small>{tr((placementPlan?.budget.devices ?? runtime.memory.devices).map(d => <span key={d.id}>{tr(d.name)}<strong>{tr(d.usedGb.toFixed(1))} / {tr(d.budgetGb.toFixed(1))}{tr(" GB")}</strong></span>))}</div> : null)}
          </section> : null)}
          {tr(expert ? <RunExpertEditor engine={engine} config={config} text={text} onChange={update} onText={value => { setStage('idle'); setConfigDraft(value); }} onRestore={() => setDiscardOpen(true)} onError={message => setIssues([{ level: 'error', message }])}/> : null)}
          {tr(parsed.error ? <ErrorNotice message={parsed.error}/> : null)}
          {tr(issues.length ? <div className="run-issues" role="status">{tr(issues.map((issue, i) => <p key={i} data-tone={issue.level}><Icon name={issue.level === 'error' ? 'error' : 'info'} size={16}/>{tr(issue.message)}</p>))}</div> : null)}
          {tr((executionMode === 'single' || library.mode === 'real') && !lifecycle.allowed ? <p className="small muted">{tr(lifecycle.reason)}</p> : null)}
          {tr(placementError ? <p className="small inline-error" role="status">{tr(placementError)}</p> : null)}
          {tr(placementPlan && !placementPlan.allowed ? <p className="small inline-error" role="status">{tr(executionMode === 'multiple' ? 'Other loaded models have reserved this memory. Choose another GPU, reduce context, or unload a model.' : 'Reduce context or choose a smaller version to fit this model.')}</p> : null)}
          <section className="run-actions" aria-label={tr("Run model")}>
            <Btn icon="sliders" disabled={invalid || pending || !connected} onClick={() => void tune()}>{tr(pending && stage === 'checking' ? 'Tuning…' : 'Tune model')}</Btn>
            <Btn variant="primary" icon="models" disabled={!installed || invalid || !!placementError || placementPlan?.allowed === false || pending || imagePending || imageNeedsDownload || imageUnavailable || !connected || ((executionMode === 'single' || library.mode === 'real') && !lifecycle.allowed) || working} onClick={() => void run()}>{tr(pending && stage === 'loading' ? 'Loading…' : 'Load model')}</Btn>
          </section>
          {tr(imageUnavailable ? <p className="small muted">{tr("This engine cannot run image models yet. Choose Off to run the chat model.")}</p> : null)}
          {tr(!installed ? <Btn icon={model.local ? 'folder' : 'download'} onClick={() => model.local ? setLocalModelOpen(true) : setLibraryOpen(true)}>{tr(model.local ? 'Reconnect local file' : 'Download model version')}</Btn> : null)}
        </div>
        <RunPreview experience={experience} memory={memory} hardware={hardware} imageEnabled={imageChoice !== 'off'}/>
      </div>
    </div>
    {tr(profileMenu ? <FloatingMenu anchor={profileTrigger.current} title={tr("Saved settings")} width={320} onClose={() => setProfileMenu(false)}>
      <div className="run-settings-menu-label">{tr("Saved settings")}</div>
      <button role="menuitemradio" aria-checked={!selectedProfile} className="menu__item run-settings-option" onClick={() => { chooseProfile('current'); setProfileMenu(false); }}><span className="run-settings-icon"><Icon name="sliders" size={18}/></span><span>{tr("Current settings")}</span>{tr(!selectedProfile ? <Icon name="check" size={15} className="run-settings-selected"/> : null)}</button>
      {tr(localProfiles.map(profile => <button key={profile.id} role="menuitemradio" aria-checked={profile.id === activeProfile} className="menu__item run-settings-option" onClick={() => { chooseProfile(profile.id); setProfileMenu(false); }}><span className="run-settings-icon"><Icon name="layers" size={18}/></span><span>{profile.name}{tr(profile.id === personalDefault ? <small>{tr("Default for this model")}</small> : null)}{tr(profile.engineVersion !== engine.engine.version ? <small>{tr("Engine changed \u00B7 tune again before running")}</small> : null)}</span>{tr(profile.id === activeProfile ? <Icon name="check" size={15} className="run-settings-selected"/> : null)}</button>))}
      <div className="run-settings-menu-divider" role="separator"/>
      {tr(selectedProfile ? <button role="menuitem" className="menu__item run-settings-action" disabled={!config} onClick={() => { setProfileMenu(false); openSave(); }}><span className="run-settings-icon"><Icon name="pencil" size={18}/></span>{tr("Edit saved settings\u2026")}</button> : null)}
      <button role="menuitem" className="menu__item run-settings-action" disabled={!config} onClick={() => { setProfileMenu(false); openSave(true); }}><span className="run-settings-icon"><Icon name={selectedProfile ? 'copy' : 'plus'} size={18}/></span>{tr(selectedProfile ? 'Save as new…' : 'Save current settings…')}</button>
      {tr(selectedProfile && profileChanged ? <button role="menuitem" className="menu__item run-settings-action" onClick={() => { setProfileMenu(false); chooseProfile(selectedProfile.id); }}><span className="run-settings-icon"><Icon name="refresh" size={18}/></span>{tr("Restore saved settings")}</button> : null)}
    </FloatingMenu> : null)}
    <Dialog open={libraryOpen} className="run-library-dialog" title={tr("Manage models")} onClose={() => setLibraryOpen(false)}><ModelsScreen embedded familyOverride={model.family} selectedVariant={model.id} contextOverride={ctx} onChoose={id => { chooseModel(id); setLibraryOpen(false); }}/></Dialog>
    <LocalModelImport open={localModelOpen} directory={library.disk.modelsDir} onClose={() => setLocalModelOpen(false)} onChoose={id => chooseModel(id)}/>
    <Dialog open={profileOpen} compact title={tr(updateExisting ? 'Update saved settings' : 'Save settings')} onClose={() => !pending && setProfileOpen(false)} actions={<><Btn disabled={pending} onClick={() => setProfileOpen(false)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!profileName.trim() || pending || imagePending || invalid} onClick={() => void storeProfile()}>{tr(pending ? 'Saving…' : updateExisting ? 'Update settings' : 'Save settings')}</Btn></>}><div className="run-save-profile"><label className="input-field"><span>{tr("Name")}</span><input className="st-input" aria-label={tr("Settings name")} maxLength={64} value={profileName} onChange={e => setProfileName(e.target.value)}/></label><p className="small muted">{tr(model.name)} · {tr(ctx / 1024)}{tr("K \u00B7 ")}{tr(use === 'full' ? 'Maximum performance' : 'Room for other apps')}</p><label className="run-default-choice"><input type="checkbox" checked={makeDefault} onChange={e => setMakeDefault(e.target.checked)}/><span>{tr("Use by default for this model")}<small>{tr("On this computer. Restore it whenever you open Models.")}</small></span></label>{tr(profileError ? <ErrorNotice message={profileError}/> : null)}</div></Dialog>
    <DemoTest open={testOpen} onClose={() => setTestOpen(false)} experience={experience}/>
    <Dialog open={discardOpen} compact title={tr("Restore saved settings?")} onClose={() => setDiscardOpen(false)} actions={<><Btn onClick={() => setDiscardOpen(false)}>{tr("Keep editing")}</Btn><Btn onClick={() => { setConfigDraft(null); void chooseImage(imageModels?.applied ?? 'off'); setActiveProfile('current'); setIssues([]); setDiscardOpen(false); }}>{tr("Restore settings")}</Btn></>}><p>{tr("Replace this draft with the settings last saved to the engine.")}</p></Dialog>
  </Screen>;
}
