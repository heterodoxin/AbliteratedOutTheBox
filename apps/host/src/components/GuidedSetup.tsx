import { tr } from '../lib/i18n';
import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { Brand } from './Brand';
import { Btn, ErrorNotice, Icon, Select } from './ui';
import { OnboardingArt, OnboardingTour } from './OnboardingArt';
import { OnboardingMemory } from './OnboardingMemory';
import { OnboardingResources } from './OnboardingResources';
import { OnboardingCheckbox } from './OnboardingCheckbox';
import { api } from '../lib/client';
import { navigate } from '../lib/router';
import { nativeAction, useNative, type NativeJob } from '../lib/native';
import { completeSetup, completeSetupStep, go, newSession, openSetup, selectRemoteModel, setSettingsSection, useStore } from '../lib/store';
import type { LocalModelFolder, SetupState } from '../lib/types';
import { downloadCounter, modelFolderLabel, onboardingError } from '../lib/onboarding-download';
import { installProgress, installStage, stageText, useDownloadEta } from '../lib/download-eta';
import { VersionPicker } from './VersionPicker';
import { DownloadParts } from './DownloadParts';
import { EngineWarning } from './EngineWarning';
import { OTHER_NOTE, PICK_NOTE, recommendedVersions } from '../lib/native-fit';
import { nativeCatalogModels, catalogVersionLabel } from '../lib/native-catalog';
type Choices = NonNullable<NonNullable<SetupState['choices']>['onboarding']>;
function radioKeys(event: KeyboardEvent<HTMLDivElement>) { if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key))
    return; event.preventDefault(); const items = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role=radio]')); const current = items.indexOf(event.target as HTMLButtonElement); const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (current + (['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1) + items.length) % items.length; items[next]?.focus(); items[next]?.click(); }
const steps = ['Welcome', 'Your model', 'Prepare', 'Make it yours'];
export function GuidedSetup() {
    const native = useNative(), setup = useStore(s => s.setup), hardware = useStore(s => s.hardware);
    const families = nativeCatalogModels(native?.catalog?.families ?? []);
    const saved = !setup?.complete && !setup?.choices?.onboarding?.manual ? setup?.choices?.onboarding : undefined;
    const [step, setStep] = useState(saved?.manual || setup?.complete ? 1 : saved?.step ?? 1);
    const [family, setFamily] = useState(saved?.family ?? native?.catalog?.recommendation?.family ?? native?.catalog?.families?.[0]?.family ?? '');
    const initialFamily = families.find(f => f.family === family || f.sizes.some(size => size.sourceFamily === family));
    const [size, setSize] = useState(saved?.size ?? initialFamily?.sizes.find(s => s.model === native?.catalog?.recommendation?.model)?.id ?? initialFamily?.sizes[0]?.id ?? '');
    const [context, setContext] = useState(saved?.context ?? native?.catalog?.recommendation?.context ?? 32768);
    const [vision, setVision] = useState(saved?.vision ?? true), [directory, setDirectory] = useState(saved?.directory ?? '');
    const [profile, setProfile] = useState<'work' | 'performance'>(saved?.profile ?? 'work');
    const [installJobId, setInstallJobId] = useState(saved?.installJobId ?? ''), [tuneJobId, setTuneJobId] = useState(saved?.tuneJobId ?? '');
    const [plan, setPlan] = useState<Record<string, any> | null>(null), [tour, setTour] = useState(0), [folderOpen, setFolderOpen] = useState(false);
    const [pending, setPending] = useState(false), [checkingPlan, setCheckingPlan] = useState(false), [error, setError] = useState(''), [calibrate, setCalibrate] = useState(saved?.calibrate ?? true);
    const writes = useRef(Promise.resolve()), leaving = useRef(false), saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null), titleRef = useRef<HTMLHeadingElement>(null);
    const selectedFamily = families.find(f => f.family === family || f.sizes.some(size => size.sourceFamily === family)) ?? families[0];
    const version = selectedFamily?.sizes.find(s => s.id === size) ?? selectedFamily?.sizes[0];
    const installed = native?.installed?.models.find(m => m.model === version?.id);
    const installJob = native?.jobs.find(j => j.id === installJobId) ?? (step === 3 ? native?.jobs.find(j => j.kind === 'install' && j.state === 'running' && j.guided && j.model === version?.model) : undefined), tuneJob = native?.jobs.find(j => j.id === tuneJobId) ?? (step === 4 ? native?.jobs.find(j => j.kind === 'tune' && j.state === 'running' && j.guided && j.model === version?.id) : undefined);
    const active = native?.jobs.find(j => j.state === 'running');
    const ready = native?.instances.some(i => i.model === version?.id && i.status === 'ready');
    const folder = directory || native?.downloadDirectory || native?.installed?.data_folder || setup?.app.modelsDir || '';
    const choices: Choices = { step, family: selectedFamily?.family, size: version?.id, context, vision, directory, profile, installJobId, tuneJobId, calibrate };
    function persist(next: number, patch: Partial<Choices> = {}, dismissed = false) {
        const value = { ...choices, step: next, ...patch };
        const work = writes.current.catch(() => { }).then(() => completeSetupStep(Math.min(4, next), { onboarding: value }, dismissed));
        writes.current = work;
        return work;
    }
    async function perform(action: () => Promise<void>) { setPending(true); setError(''); try {
        await action();
    }
    catch (e) {
        setError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setPending(false);
    } }
    async function advance(next: number, patch: Partial<Choices> = {}) { await persist(next, patch); setStep(next); setError(''); }
    async function skip(manual = false, section?: 'Engine' | 'Models') {
        await perform(async () => { leaving.current = true; if (saveTimer.current)
            clearTimeout(saveTimer.current); try {
            await persist(step, { manual }, true);
            openSetup(false);
            if (section) {
                setSettingsSection(section);
                navigate('chat', undefined, { settings: '1' });
            }
            else
                go(manual || preparing ? 'models' : 'chat');
        }
        catch (e) {
            leaving.current = false;
            throw e;
        } });
    }
    /* Files the person already has are checked and adopted by the "Use a local GGUF" page, not made the data folder (that downloaded them again, #3507). */
    async function ownModel(path?: string) {
        await perform(async () => { leaving.current = true; if (saveTimer.current)
            clearTimeout(saveTimer.current); try {
            await persist(step, { manual: true }, true);
            openSetup(false);
            navigate('models', 'local', path ? { path } : undefined);
        }
        catch (e) {
            leaving.current = false;
            throw e;
        } });
    }
    async function finish() {
        if (leaving.current)
            return;
        leaving.current = true;
        if (saveTimer.current)
            clearTimeout(saveTimer.current);
        try {
            await writes.current;
            await completeSetup();
            selectRemoteModel(null);
            openSetup(false);
            const chat = newSession();
            go('chat', chat.id);
        }
        catch (e) {
            leaving.current = false;
            setError(e instanceof Error ? e.message : String(e));
        }
    }
    function chooseNative(family: string, size: string) { setFamily(family); setSize(size); setVision(true); setPlan(null); setError(''); }
    useEffect(() => { titleRef.current?.focus({ preventScroll: true }); titleRef.current?.closest('.setupmodal__surface')?.scrollTo({ top: 0 }); }, [step]);
    useEffect(() => {
        if (step < 2 || leaving.current)
            return;
        saveTimer.current = setTimeout(() => { void persist(step).catch(e => setError(e.message)); }, 450);
        return () => { if (saveTimer.current)
            clearTimeout(saveTimer.current); };
    }, [step, family, size, context, vision, directory, profile, calibrate, installJobId, tuneJobId]);
    useEffect(() => {
        if (step === 3 && installJob?.state === 'done' && installed?.ready)
            void advance(4).catch(e => setError(e.message));
    }, [installJob?.state, installed?.ready, step]);
    useEffect(() => { if (step === 4 && tuneJob?.state === 'done' && ready)
        void finish(); }, [tuneJob?.state, ready, step]);
    useEffect(() => { if (step === 3 && installJob?.state === 'running' && !installJobId) {
        setInstallJobId(installJob.id);
        void persist(3, { installJobId: installJob.id }).catch(e => setError(e.message));
    } if (step === 4 && tuneJob?.state === 'running' && !tuneJobId) {
        setTuneJobId(tuneJob.id);
        void persist(4, { tuneJobId: tuneJob.id }).catch(e => setError(e.message));
    } }, [step, installJob?.id, tuneJob?.id]);
    const values = { guided: true, family: version?.sourceFamily, model: version?.model, context, vision: version?.images ? 'yes' : 'no', kv: 'int8', low_ram: 'auto', loadAfter: false, multiple: false, ...(directory ? { data_dir: directory } : {}) };
    async function review() {
        await perform(async () => {
            setInstallJobId('');
            setTuneJobId('');
            if (installed?.ready && installed.context === context && (!version?.images || installed.images && installed.images !== 'off') && (!directory || directory === installed.dataFolder)) {
                await advance(4, { installJobId: '', tuneJobId: '' });
                return;
            }
            setPlan(await api.post('/api/native/plan', values));
            await advance(3, { installJobId: '', tuneJobId: '' });
        });
    }
    async function download() {
        await perform(async () => {
            const result = await nativeAction<NativeJob>('operations', { kind: 'install', args: values });
            setInstallJobId(result.id);
            await persist(3, { installJobId: result.id });
        });
    }
    async function tune() {
        await perform(async () => {
            const result = await nativeAction<NativeJob>('operations', { kind: 'tune', args: { guided: true, model: version?.id, profile, calibrate, vision: version?.images && vision ? 'yes' : 'no', multiple: false } });
            setTuneJobId(result.id);
            await persist(4, { tuneJobId: result.id });
        });
    }
    const progress = installProgress(installJob?.result?.install);
    const preparing = installJob?.state === 'running';
    const stopped = installJob && ['error', 'cancelled'].includes(installJob.state);
    useEffect(() => {
        if (step !== 3 || plan || preparing || !native?.available || !version) {
            setCheckingPlan(false);
            return;
        }
        let cancelled = false;
        setCheckingPlan(true);
        api.post<Record<string, any>>('/api/native/plan', values).then(value => { if (!cancelled) {
            setPlan(value);
            setCheckingPlan(false);
        } }).catch(e => { if (!cancelled) {
            setError(e.message);
            setCheckingPlan(false);
        } });
        return () => { cancelled = true; };
    }, [step, plan, preparing, native?.available, version?.id, context, directory]);
    const recommendedRows = recommendedVersions(selectedFamily?.sizes ?? [], native?.catalog?.recommendation);
    const recommendedCards = recommendedRows.map(x => x.v);
    const pickNote = (v: { id: string }) => recommendedRows.find(x => x.v.id === v.id && [PICK_NOTE, OTHER_NOTE].includes(x.r.note))?.r.note ?? '';
    // The recommended four first, then a list with every other version (the Models page's version list); a family without a recommended set shows them all.
    const cards = recommendedCards.length ? recommendedCards : selectedFamily?.sizes ?? [];
    const others = recommendedCards.length ? (selectedFamily?.sizes ?? []).filter(v => !recommendedCards.includes(v)) : [];
    const readyIds = new Set((native?.installed?.models ?? []).filter(m => m.ready).map(m => m.model));
    const versionCard = (v: NonNullable<typeof version>) => <button key={v.id} type="button" role="radio" tabIndex={version?.id === v.id ? 0 : -1} aria-checked={version?.id === v.id} data-selected={version?.id === v.id} onClick={() => { setSize(v.id); setPlan(null); }}><span className="onboard-version-top"><span className="onboard-version-copy"><strong>{tr(catalogVersionLabel(v))}</strong><small>{tr(v.model)}{tr(pickNote(v) ? <> · {tr(pickNote(v))}</> : null)}</small></span><span className="onboard-version-radio">{tr(version?.id === v.id ? <Icon name="check" size={12}/> : null)}</span></span><span className="onboard-version-bottom"><span className="onboard-version-size"><Icon name="download" size={15}/>{tr(v.download_gb.toFixed(1))} <small>{tr("GB")}</small></span><span className="onboard-version-fit">{tr(v.on_this_pc === 'does not fit' || v.on_this_pc === 'tight' || v.on_this_pc?.includes('slower') || v.on_this_pc?.startsWith('experimental:') ? <><Icon name="clock" size={13}/>{tr("May run slower")}</> : v.installed ? <><Icon name="check" size={13}/>{tr("On this computer")}</> : <>{tr("Download size")}</>)}</span></span></button>;
    const counter = downloadCounter(plan?.plan?.download_gb ?? version?.download_gb, progress, preparing);
    const eta = useDownloadEta(preparing ? progress : undefined, installJob?.id);
    const title = step === 1 ? 'State of the art. At home.' : step === 2 ? 'Select the model you want to run.' : step === 3 ? 'Downloading the model...' : tuneJob?.state === 'running' ? 'A little tuning. A better fit.' : 'How you would like to work with Strata?';
    function changeDownloadFolder() { void perform(async () => { await advance(2); setFolderOpen(true); }); }
    return <div className="setup guided-onboarding" data-step={step}>
  <header className="onboard-header"><Brand variant="lockup" height={22}/>{tr(step !== 1 ? <button className="onboard-later" disabled={pending} onClick={() => void skip(false)}>{tr("Finish later")}<Icon name="chevron" size={13}/></button> : null)}</header>
  <div className="onboard-stepper" aria-label={tr("Setup progress")}>{tr(steps.map((label, i) => <div key={label} data-current={step === i + 1} data-done={step > i + 1} aria-current={step === i + 1 ? 'step' : undefined}><span>{tr(step > i + 1 ? <Icon name="check" size={12}/> : i + 1)}</span><small>{tr(label)}</small></div>))}</div>
  {tr(step === 1 ? <>
   <OnboardingArt />
   <div className="onboard-welcome-copy"><h1 ref={titleRef} tabIndex={-1} data-initial-focus>{tr(title)}</h1><p>{tr("Get stuff done on hardware you own.")}<br />{tr("Use intelligence the way you want.")}</p><div className="onboard-benefits"><span><Icon name="shield" size={17}/>{tr("Privacy first")}</span><span><Icon name="bolt" size={17}/>{tr("Fastest AI")}</span><span><Icon name="check" size={17}/>{tr("Easy to use")}</span></div></div>
   <footer className="onboard-footer onboard-footer--welcome"><Btn variant="primary" disabled={pending || !native} onClick={() => void perform(() => advance(2))}>{tr("Let\u2019s get started")}<Icon name="chevron" size={16}/></Btn><small>{tr("Already know your way around? ")}<button disabled={pending} onClick={() => void skip(true)}>{tr("Go straight to the app")}</button></small></footer>
  </> : <>
   <div className="onboard-stage-heading"><h1 ref={titleRef} tabIndex={-1} data-initial-focus>{tr(title)}</h1>{tr(step === 2 ? <p>{tr("You can always change your models later.")}</p> : step === 4 && tuneJob?.state === 'running' ? <p>{tr("We\u2019re measuring, applying your preference, and getting ready to chat.")}</p> : null)}</div>
   {tr(step === 2 ? <div className="onboard-model-stage">
    {tr(selectedFamily ? <>
     <div className="onboard-model-top"><div className="onboard-model-picker"><span className="onboard-control-label">{tr("Model")}</span><div><span className="onboard-model-symbol"><Icon name="models" size={23}/></span><Select ariaLabel={tr("Setup model")} value={selectedFamily.family} options={families.map(f => ({ value: f.family, label: f.title }))} onChange={value => chooseNative(value, '')}/></div></div><div className="onboard-context"><span className="onboard-control-label">{tr("Conversation context")}</span><Select ariaLabel={tr("Setup conversation context")} value={String(context)} options={(native?.catalog?.contexts ?? [32768]).map(n => ({ value: String(n), label: `${n / 1024}K` }))} onChange={value => setContext(Number(value))}/></div></div>
     <OnboardingMemory hardware={hardware} native={native?.hardware}/>{tr(!version || version.needsNewerEngine ? <p className="small muted">{tr("This model needs a newer engine. Update your engine or import a local GGUF for this model.")}</p> : null)}
     <div className="onboard-version-heading"><strong>{tr("Choose a version")}</strong><span>{tr(version ? `${catalogVersionLabel(version)} selected` : '')}</span></div>
     <div className="onboard-versions" role="radiogroup" aria-label={tr("Model version")} onKeyDown={radioKeys}>{tr(cards.map(v => versionCard(v)))}</div>
     {tr(others.length ? <div className="onboard-more-versions"><VersionPicker versions={others} selectedId={version?.id ?? ''} hardware={hardware ? { ram_gb: native?.hardware?.ram_gb ?? hardware.memory.totalGb, vram_gb: Math.max(0, ...(native?.hardware?.gpus ?? []).filter(g => g.usable !== false).map(g => g.vram_gb)) } : null} downloaded={readyIds} showRecommended={false} moreLabel="More versions" onSelect={id => { setSize(id); setPlan(null); }} onOwnGguf={() => void ownModel()}/></div> : null)}
     <div className="onboard-folder"><Icon name="folder" size={20}/><div><strong>{tr(directory ? 'Your model folder' : 'Already downloaded the model?')}</strong><small>{tr(directory || "Choose the folder and we'll start your models with Strata.")}</small></div><Btn size="sm" onClick={() => setFolderOpen(v => !v)}>{tr(folderOpen ? 'Close' : directory ? 'Change' : 'Choose folder')}</Btn></div>
     {tr(folderOpen ? <OnboardingFolder directory={folder} onChoose={(path, ggufFiles) => ggufFiles > 0 && !/strata/i.test(path.split(/[\\/]/).filter(Boolean).pop() ?? '') ? void ownModel(path) : void perform(async () => { await nativeAction('directory', { directory: path }); setDirectory(path); setFolderOpen(false); setPlan(null); })}/> : null)}
    </> : <div className="onboard-engine-connect"><Icon name="models" size={24}/><strong>{tr("Connect your Strata installation")}</strong><Btn onClick={() => void skip(false, 'Engine')}>{tr("Connect engine")}</Btn></div>)}
   </div> : step === 3 ? <div className="onboard-prepare-stage">
    <section className="onboard-download"><EngineWarning native={native}/><div className="onboard-download-identity"><Icon name="models" size={19}/><strong>{tr(selectedFamily?.title)}</strong><span>{tr(version ? catalogVersionLabel(version) : '')}</span><small>{tr(version?.model)}</small></div><OnboardingArt kind="download" progress={progress?.percent ?? 0} counter={counter}/><div className="onboard-download-info">
     {tr(preparing ? <div><strong className="onboard-download-status">{tr(progress ? stageText(installStage(installJob?.result?.install)) : 'Preparing your model…')}</strong><div className="onboard-progress" data-indeterminate={!progress}>{tr(progress ? <progress max={100} value={progress.percent ?? 0} aria-label={tr("Current file download progress")}/> : <span />)}</div><DownloadParts parts={installJob?.result?.install?.parts}/><div className="onboard-progress-meta"><span>{tr(progress ? `${progress.percent ?? 0}%${Number.isFinite(progress.done_gb) && Number.isFinite(progress.total_gb) ? ` · ${progress.done_gb!.toFixed(1)} / ${progress.total_gb!.toFixed(1)} GB` : ''}` : 'Getting files ready…')}</span><div className="onboard-progress-actions">{tr(eta ? <span aria-live="off">{eta}</span> : null)}<button disabled={pending} onClick={() => void perform(async () => { await nativeAction(`operations/${installJob!.id}/cancel`); })}>{tr("Stop")}</button></div></div></div> : <>{tr(stopped ? <OnboardingError message={installJob?.error ?? 'Download stopped. Ready when you are.'} onFolder={changeDownloadFolder}/> : null)}{tr(plan?.plan?.disk_short ? <OnboardingError message={plan.plan.disk_short} onFolder={changeDownloadFolder}/> : null)}</>)}
     <small className="onboard-download-location" title={tr(folder)}><Icon name="folder" size={16}/><span><span>{tr("Save to")}</span><strong>{tr(modelFolderLabel(folder))}</strong></span>{tr(!preparing ? <button type="button" onClick={changeDownloadFolder}>{tr("Change")}</button> : null)}</small>
    </div></section><OnboardingTour page={tour} onChange={setTour}/>
   </div> : <div className="onboard-tune-stage">
    {tr(tuneJob?.state === 'running' ? <><OnboardingArt kind="tune"/><div className="onboard-tuning-status" aria-live="polite"><span className="onboard-pulse-dot"/><strong>{tr(tuneJob.summary)}</strong><small>{tr("Your choice: ")}{tr(profile === 'work' ? '20% headroom for other work' : 'Full performance')}</small><div className="onboard-progress" data-indeterminate><span /></div><button disabled={pending} onClick={() => void perform(async () => { await nativeAction(`operations/${tuneJob.id}/cancel`); })}>{tr("Stop tuning")}</button></div><OnboardingTour page={tour} onChange={setTour}/></> : <>
     <OnboardingResources profile={profile}/>
     <div className="onboard-resource-options" role="radiogroup" aria-label={tr("Computer resource preference")} onKeyDown={radioKeys}>{tr(([{ id: 'work', icon: 'computer', title: 'Leave 20% for my work', hint: 'Use Strata while browsing, writing or working.', badge: 'Everyday choice' }, { id: 'performance', icon: 'bolt', title: 'Use full performance', hint: 'Focus this computer on your AI.', badge: '100% priority' }] as const).map(option => <button key={option.id} role="radio" tabIndex={profile === option.id ? 0 : -1} aria-checked={profile === option.id} data-selected={profile === option.id} onClick={() => setProfile(option.id)}><span className="onboard-resource-option-top"><span className="onboard-resource-option-icon"><Icon name={option.icon} size={22}/></span><span className="onboard-version-radio">{tr(profile === option.id ? <Icon name="check" size={12}/> : null)}</span></span><strong>{tr(option.title)}</strong><p>{tr(option.hint)}</p><small>{tr(option.badge)}</small></button>))}</div>
     <details className="onboard-resource-honesty"><summary>{tr("How resource sharing works")}</summary><p>{tr("We reserve GPU memory and limit CPU expert workers to leave headroom for other apps. Actual usage varies with your model. Model RAM requirements and system safety reserves still apply.")}</p></details>
     {tr(version?.images ? <label className="onboard-picture onboard-load-vision"><OnboardingCheckbox checked={vision} onChange={setVision}/><Icon name="image" size={20}/><span><strong>{tr("Read pictures when this model loads")}</strong><small>{tr("Understand photos and screenshots. Vision files are already downloaded.")}</small></span></label> : null)}
     {tr(native?.capabilities.calibration ? <label className="onboard-calibration"><OnboardingCheckbox checked={calibrate} onChange={setCalibrate}/><Icon name="gauge" size={20}/><span><strong>{tr("Find the fastest settings for this PC")}</strong><small>{tr("One-time measurement \u00B7 about 5\u201310 minutes. Your computer will be busy.")}</small></span></label> : <p className="onboard-calibration-note">{tr("We\u2019ll apply your preference and run a short speed check.")}</p>)}
     {tr(tuneJob && ['error', 'cancelled'].includes(tuneJob.state) ? <ErrorNotice message={tuneJob.error ?? 'Tuning stopped. Your model files are safe.'} next={tr("Try again, or turn off the longer measurement above.")}/> : null)}
    </>)}
   </div>)}
   <footer className="onboard-footer"><Btn variant="ghost" disabled={pending || !!active} onClick={() => void perform(() => advance(step === 4 ? 2 : step - 1))}>{tr("Back")}</Btn>{tr(step === 3 ? <span>{tr(preparing ? 'Take a look around while we prepare.' : 'Downloads start only when you’re ready.')}</span> : step === 4 && tuneJob?.state === 'running' ? <span>{tr("We\u2019ll open chat when your model is ready.")}</span> : null)}{tr(step === 2 ? <Btn variant="primary" disabled={pending || !!active || !version || !!version.needsNewerEngine || !native?.available || folderOpen} onClick={() => void review()}>{tr(pending ? 'Checking…' : 'Continue')}<Icon name="chevron" size={15}/></Btn> : step === 3 ? <Btn variant="primary" disabled={pending || checkingPlan || !!active || !!plan?.plan?.disk_short || !version || !!version.needsNewerEngine} onClick={() => void download()}>{tr(preparing ? 'Preparing…' : checkingPlan ? 'Checking storage…' : pending ? 'Starting…' : stopped ? 'Resume download' : 'Download model')}{tr(!preparing ? <Icon name="download" size={16}/> : null)}</Btn> : <Btn variant="primary" disabled={pending || !!active || !installed?.ready} onClick={() => void tune()}>{tr(tuneJob?.state === 'running' ? 'Tuning…' : pending ? 'Starting…' : 'Tune & start chatting')}<Icon name="chevron" size={15}/></Btn>)}</footer>
   {tr(step === 4 && ready && ['error', 'cancelled'].includes(tuneJob?.state ?? '') ? <button className="onboard-continue-anyway" onClick={() => void finish()}>{tr("Continue to chat without a speed result")}</button> : null)}
  </>)}
  {tr(error ? <div className="onboard-error"><OnboardingError message={error} onFolder={changeDownloadFolder}/></div> : null)}
 </div>;
}
function OnboardingError({ message, onFolder }: {
    message: string;
    onFolder: () => void;
}) {
    const detail = onboardingError(message);
    if (!detail.storage)
        return <ErrorNotice message={detail.message}/>;
    return <div className="onboard-storage-error" role="alert"><Icon name="disk" size={20}/><div><strong>{tr(detail.title)}</strong><p>{tr(detail.message)}</p><small>{tr(detail.next)}</small><button type="button" onClick={onFolder}>{tr("Choose another folder")}<Icon name="chevron" size={13}/></button></div></div>;
}
function OnboardingFolder({ directory, onChoose }: {
    directory: string;
    onChoose: (path: string, ggufFiles: number) => void;
}) {
    const [path, setPath] = useState(directory), [folder, setFolder] = useState<LocalModelFolder | null>(null), [pending, setPending] = useState(false), [error, setError] = useState('');
    async function browse(input: string, initial = false) { setPending(true); setError(''); try {
        let value: LocalModelFolder;
        try {
            value = await api.get<LocalModelFolder>(`/api/local-models/browse?directory=${encodeURIComponent(input)}`);
        }
        catch (e) {
            if (!initial)
                throw e;
            value = await api.get<LocalModelFolder>('/api/local-models/browse');
        }
        setFolder(value);
        setPath(value.directory);
    }
    catch (e) {
        setFolder(null);
        setError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setPending(false);
    } }
    useEffect(() => { void browse(directory, true); }, []);
    return <section className="onboard-folder-browser" aria-label={tr("Choose your model folder")}><form onSubmit={e => { e.preventDefault(); void browse(path); }}><input className="st-input" aria-label={tr("Existing model folder")} value={path} onChange={e => { setPath(e.target.value); setFolder(null); }} placeholder={tr("Full folder path")} spellCheck={false}/><Btn type="submit" size="sm" disabled={pending || !path.trim()}>{tr(pending ? 'Checking…' : 'Open')}</Btn></form><p>{tr("Choose the Strata data folder that holds your model downloads. Matching files are reused; missing files are prepared automatically.")}</p><div className="onboard-folder-shortcuts">{folder?.shortcuts?.map(item => <Btn key={item.path} size="sm" disabled={pending} onClick={() => void browse(item.path)}>{tr(item.name)}</Btn>)}</div><div className="onboard-folder-files">{tr(folder?.parent ? <button onClick={() => void browse(folder.parent!)}><Icon name="folder" size={16}/>{tr(" Up a folder")}</button> : null)}{folder?.entries.filter(e => e.kind === 'folder').slice(0, 30).map(e => <button key={e.path} onClick={() => void browse(e.path)}><Icon name="folder" size={16}/>{tr(e.name)}<Icon name="chevron" size={12}/></button>)}{tr(folder ? <small>{tr(folder.entries.filter(e => e.kind === 'model').length)}{tr(" GGUF files in this folder. Nested downloads are checked by the installer.")}</small> : null)}</div>{tr(folder ? <Btn variant="primary" size="sm" disabled={pending} onClick={() => onChoose(folder.directory, folder.entries.filter(e => e.kind === 'model').length)}>{tr("Use this folder")}</Btn> : null)}{tr(error ? <ErrorNotice message={error}/> : null)}</section>;
}
