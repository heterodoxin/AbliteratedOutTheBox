import { tr, localeTag } from '../lib/i18n';
import { useEffect, useMemo, useState } from 'react';
import { Bar, Btn, Dialog, Icon } from './ui';
import { api, ApiError } from '../lib/client';
import { nativeAction, refreshNative, useNative, type NativeJob, type NativeState } from '../lib/native';
import { nativeShortName } from '../lib/native-names';
import { sizeGb } from '../lib/format';

type Choice = 'move' | 'keep' | 'empty';
interface Plan {
    ok: boolean;
    issue: { code: string; message: string; details: string } | null;
    old: string; new: string; same_drive: boolean;
    free_bytes: number; models_bytes: number; needed_bytes: number;
    models: { group: string; name: string; bytes: number }[];
    choices: Record<Choice, { possible: boolean; reason: string }>;
    folder: string; picked: string; wholeDrive: boolean; shown: string; current: string; running: number; busy: boolean;
}
interface Outcome { done?: boolean; stopped?: boolean; failed?: boolean; code?: string; moved?: string[]; remaining?: string[]; bytes?: number; total?: number; from?: string; to?: string; sameDrive?: boolean }

const gb = (bytes: number) => sizeGb(bytes / 1e9);
const driveOf = (path: string) => /^[A-Za-z]:/.test(path) ? path.slice(0, 2) : path;
const list = (items: string[]) => { try { return new Intl.ListFormat(localeTag(), { type: 'conjunction' }).format(items); } catch { return items.join(', '); } };

/** The model a folder name belongs to, in the words the Storage list uses ("Gyro-S"); the draft layer and loose files have no name of their own. */
function modelName(group: string, native: NativeState | null) {
    const hit = native?.installed?.models.find(m => m.model.toLowerCase() === group.toLowerCase());
    return hit ? nativeShortName(hit.model) : null;
}

function timeLeft(seconds: number) {
    if (seconds <= 0) return null;
    if (seconds < 60) return tr('Less than a minute left');
    const minutes = Math.ceil(seconds / 60);
    if (minutes <= 1) return tr('About a minute left');
    if (minutes < 120) return tr('About {0} minutes left', [minutes]);
    return tr('About {0} hours left', [Math.ceil(minutes / 60)]);
}

function Failure({ message, next, details }: { message: string; next?: string; details?: string }) {
    return <div className="error-notice" role="alert"><Icon name="error" size={18}/><div><strong>{tr(message)}</strong>{next ? <p>{tr(next)}</p> : null}{details ? <details className="error-detail"><summary className="small">{tr('Details')}</summary><pre className="small muted">{details}</pre></details> : null}</div></div>;
}

/** Settings > Storage > Models folder: the card, the folder dialog with its three choices, the progress of a move and what came of it. */
export function ModelsFolderCard({ blocked }: { blocked: boolean }) {
    const native = useNative();
    const [plan, setPlan] = useState<Plan | null>(null);
    const [stage, setStage] = useState<'issue' | 'drive' | 'choose' | null>(null);
    const [choice, setChoice] = useState<Choice>('move');
    const [pending, setPending] = useState(false);
    const [error, setError] = useState<{ message: string; details?: string } | null>(null);
    const [typing, setTyping] = useState(false), [typed, setTyped] = useState('');
    const [jobId, setJobId] = useState<string | null>(null);
    const [outcome, setOutcome] = useState<{ job: NativeJob } | null>(null);
    const [notice, setNotice] = useState('');
    const [stopping, setStopping] = useState(false);
    const [unfinished, setUnfinished] = useState<{ from: string; to: string } | null>(null);
    const [resumed, setResumed] = useState<{ from: string; to: string } | null>(null);
    const folder = native?.downloadDirectory ?? '';
    const free = native?.downloadFreeBytes ?? null;
    const usage = native?.installed?.usage?.total_bytes ?? 0;
    const moving = native?.jobs.find(j => j.kind === 'models-move' && j.state === 'running') ?? null;
    const finished = jobId ? native?.jobs.find(j => j.id === jobId && j.kind === 'models-move' && j.state !== 'running') : undefined;
    useEffect(() => { if (finished && !outcome) { setOutcome({ job: finished }); setStopping(false); void refreshNative(true); } }, [finished, outcome]);
    const total = native?.downloadTotalBytes ?? null;
    const used = free !== null && total ? Math.min(100, Math.max(1, ((total - free) / total) * 100)) : 0;
    useEffect(() => { if (moving && !jobId) setJobId(moving.id); }, [moving, jobId]);

    /* A move that a crash or a stop left half done: found again at the next start, with its two choices. */
    useEffect(() => {
        if (moving || outcome) return;
        let live = true;
        api.get<{ interrupted: { from: string; to: string } | null }>('/api/native/models-folder/interrupted')
            .then(r => { if (live) setUnfinished(r.interrupted); }).catch(() => { if (live) setUnfinished(null); });
        return () => { live = false; };
    }, [moving, outcome, folder]);
    async function finishMove() {
        if (!unfinished) return;
        setPending(true);
        setError(null);
        try {
            const result = await api.post<NativeJob>('/api/native/models-folder/resume', { stopRunning: true });
            setResumed(unfinished);
            setUnfinished(null);
            if (result.id) { setJobId(result.id); setOutcome(null); await refreshNative(true); }
        }
        catch (e) { setError({ message: (e as Error).message, details: ((e as ApiError).body as { details?: string } | undefined)?.details }); }
        finally { setPending(false); }
    }
    async function leaveMove() {
        if (!unfinished) return;
        setPending(true);
        setError(null);
        try {
            await api.post('/api/native/models-folder/leave', {});
            setNotice(tr('The partial copies were removed. Your models are in {0}.', [unfinished.from]));
            setUnfinished(null);
            await refreshNative(true);
        }
        catch (e) { setError({ message: (e as Error).message, details: ((e as ApiError).body as { details?: string } | undefined)?.details }); }
        finally { setPending(false); }
    }

    function leave() { setPlan(null); setStage(null); setError(null); setTyping(false); }
    async function pick(path = '') {
        setPending(true);
        setError(null);
        setNotice('');
        try {
            const result = await api.post<{ cancelled?: boolean; plan?: Plan }>('/api/native/models-folder/plan', { path });
            if (result.cancelled || !result.plan) return;
            const next = result.plan;
            setTyping(false);
            setPlan(next);
            if (next.issue?.code === 'same') { setPlan(null); return; }
            setChoice(next.choices.move.possible ? 'move' : 'keep');
            setStage(next.issue ? 'issue' : next.wholeDrive ? 'drive' : 'choose');
        }
        catch (e) {
            if (e instanceof ApiError && e.status === 501) { setTyping(true); setError({ message: e.message }); }
            else setError({ message: (e as Error).message, details: ((e as ApiError).body as { details?: string } | undefined)?.details });
        }
        finally { setPending(false); }
    }
    async function start() {
        if (!plan) return;
        setPending(true);
        setError(null);
        try {
            const result = await api.post<NativeJob & { done?: boolean; setAside?: string[] }>('/api/native/models-folder/start', { path: plan.picked, choice, stopRunning: true });
            if (result.id) { setJobId(result.id); setOutcome(null); setStage(null); await refreshNative(true); setPlan(prev => prev); return; }
            setNotice(choice === 'keep' ? tr('Strata now also uses {0}. New downloads go there, and your current models stay where they are.', [plan.shown])
                : tr('Strata now uses {0}. Your old model files are still in {1}; Strata does not delete them.', [plan.shown, plan.current]));
            await refreshNative(true);
            leave();
        }
        catch (e) { setError({ message: (e as Error).message, details: ((e as ApiError).body as { details?: string } | undefined)?.details }); }
        finally { setPending(false); }
    }
    async function stop() {
        if (!moving) return;
        setStopping(true);
        try { await nativeAction(`operations/${moving.id}/cancel`); }
        catch (e) { setError({ message: (e as Error).message }); }
    }
    function closeResult() { setOutcome(null); setJobId(null); setResumed(null); leave(); setNotice(''); }

    const progress = moving?.progress as { done: number; total: number; file: string; group: string; index: number; count: number; secondsLeft: number; sameDrive: boolean } | null | undefined;
    const percent = progress && progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;
    const planNames = useMemo(() => (plan?.models ?? []).map(m => ({ ...m, label: modelName(m.group, native) })).filter(m => m.label), [plan, native]);
    const savedPlan = useMemo(() => plan, [plan]);

    return <>
    <section className="storage-directory" aria-labelledby="models-folder-title">
      <div className="storage-directory__heading"><span><Icon name="folder" size={18}/><strong id="models-folder-title">{tr('Models folder')}</strong></span><Btn size="sm" disabled={!folder || blocked || pending || !!moving} onClick={() => void pick()}>{tr(pending && !plan ? 'Waiting for the folder…' : 'Change folder')}</Btn></div>
      <p className="small muted">{tr('Strata saves the big model files here.')}</p>
      <code className="storage-directory__path">{folder || tr('Reading model directory…')}</code>
      {free !== null ? <div className="models-free"><Bar value={used} label={tr('Disk use')}/><div className="models-free__row small"><span><strong>{tr('{0} free', [gb(free)])}</strong> {tr('on this drive')}</span><span className="muted">{tr('Your models use {0}', [gb(usage)])}</span></div></div> : null}
      {blocked && !moving ? <p className="small muted">{tr('Finish or stop the download to change the folder.')}</p> : null}
      {typing ? <form className="storage-directory__edit" onSubmit={e => { e.preventDefault(); if (typed.trim()) void pick(typed.trim()); }}>
        <label htmlFor="models-folder-path" className="small">{tr('Directory path')}</label>
        <input id="models-folder-path" className="st-input" value={typed} onChange={e => setTyped(e.target.value)} placeholder={tr('Full path to an existing folder')} spellCheck={false} autoFocus/>
        <div className="row"><Btn variant="primary" type="submit" disabled={pending || !typed.trim()}>{tr('Use this directory')}</Btn></div>
      </form> : null}
      {unfinished && !moving ? <div className="models-notice models-notice--info" role="alert"><Icon name="info" size={18}/><div>
        <strong>{tr('A move was interrupted. Finish it or leave it.')}</strong><br/>
        {tr('Strata was moving your models from {0} to {1} when it stopped. Nothing is lost.', [unfinished.from, unfinished.to])}
        <p className="small muted">{tr('Finishing puts all your models in {1}. Leaving it keeps them in {0} and removes the partial copies from {1}.', [unfinished.from, unfinished.to])}</p>
        <div className="row"><Btn variant="primary" size="sm" disabled={pending || blocked} onClick={() => void finishMove()}>{tr('Finish the move')}</Btn><Btn size="sm" disabled={pending || blocked} onClick={() => void leaveMove()}>{tr('Leave it')}</Btn></div>
      </div></div> : null}
      {notice ? <p className="small storage-directory__notice" role="status"><Icon name="check" size={15}/>{notice}</p> : null}
      {error && !stage ? <Failure message={error.message} details={error.details}/> : null}
    </section>

    <Dialog className="models-dialog" open={stage === 'issue' && !!plan?.issue} compact title={tr(plan?.issue?.code === 'not_found' ? "That folder isn't there" : "That folder can't be used")} onClose={leave}
      actions={<>{plan?.issue?.code === 'not_found' ? null : <Btn onClick={leave}>{tr('Cancel')}</Btn>}<Btn variant="primary" onClick={() => { leave(); void pick(); }}>{tr('Choose another folder')}</Btn></>}>
      {plan?.issue ? <Failure message={plan.issue.message} details={plan.issue.details}
        next={plan.issue.code === 'not_writable' ? 'Choose a folder you own, for example one in Documents or on another drive.' : plan.issue.code === 'not_found' ? 'Check that the drive is plugged in, then pick the folder again. Your models were not touched.' : undefined}/> : null}
    </Dialog>

    <Dialog className="models-dialog" open={stage === 'drive' && !!plan} compact title={tr('Use this folder for your models?')} onClose={leave}
      actions={<><Btn onClick={leave}>{tr('Cancel')}</Btn><Btn variant="primary" onClick={() => setStage('choose')}>{tr('Continue')}</Btn></>}>
      {plan ? <>
        <div className="models-picked"><span className="small muted">{tr('New folder')}</span><code className="storage-directory__path">{plan.shown}</code><span className="small muted">{tr('{0} free on this drive', [gb(plan.free_bytes)])}</span></div>
        <div className="models-notice models-notice--info"><Icon name="info" size={18}/><div>{tr('You picked the whole drive {0}. Strata keeps its files in a folder of their own, so your models go in {1}.', [plan.picked, plan.shown])}</div></div>
        <p className="small muted">{tr('The three choices come next.')}</p>
      </> : null}
    </Dialog>

    <Dialog className="models-dialog" open={stage === 'choose' && !!plan} title={tr('Use this folder for your models?')} onClose={() => { if (!pending) leave(); }}
      actions={plan ? <>{plan.choices.move.possible ? <Btn disabled={pending} onClick={leave}>{tr('Cancel')}</Btn> : <Btn disabled={pending} onClick={() => { leave(); void pick(); }}>{tr('Choose another folder')}</Btn>}<Btn variant="primary" disabled={pending} onClick={() => void start()}>{tr(pending ? 'Working…' : choice === 'move' ? 'Move my models' : choice === 'keep' ? 'Use both folders' : 'Start empty')}</Btn></> : null}>
      {plan ? <ChooseBody plan={plan} choice={choice} setChoice={setChoice} names={planNames} error={error}/> : null}
    </Dialog>

    <Dialog className="models-dialog" open={!!moving && !outcome} compact noClose title={tr('Moving your models')} onClose={() => undefined}
      actions={<Btn disabled={stopping} onClick={() => void stop()}>{tr(stopping ? 'Stopping…' : 'Stop moving')}</Btn>}>
      <div className="models-free"><Bar value={percent} label={tr('Progress')}/><div className="models-free__row small"><span><strong>{tr('{0} of {1} GB', [progress ? (progress.done / 1e9 >= 10 ? Math.round(progress.done / 1e9) : (progress.done / 1e9).toFixed(1)) : 0, Math.round((progress?.total ?? 0) / 1e9)])}</strong></span><span className="muted">{progress && !progress.sameDrive ? timeLeft(progress.secondsLeft) : null}</span></div></div>
      <p className="small">{tr('Keep Strata open. You can keep using other apps. Your models cannot be loaded until the move is done.')}</p>
      {progress ? <details className="models-details"><summary>{tr('Details')}</summary><ul>
        <li>{tr('Now copying: {0}, file {1} of {2}', [modelName(progress.group, native) ?? progress.group, progress.index, progress.count])}</li>
        <li>{tr('From {0} to {1}', [savedPlan?.current ?? resumed?.from ?? '', savedPlan?.folder ?? resumed?.to ?? ''])}</li></ul></details> : null}
    </Dialog>

    <ResultDialog outcome={outcome?.job ?? null} plan={savedPlan} native={native} onClose={closeResult} onAgain={() => { closeResult(); void pick(); }}/>
    </>;
}

function ChooseBody({ plan, choice, setChoice, names, error }: { plan: Plan; choice: Choice; setChoice: (c: Choice) => void; names: (Plan['models'][number] & { label: string | null })[]; error: { message: string; details?: string } | null }) {
    const none = plan.models_bytes === 0;
    const short = !none && !plan.choices.move.possible;
    const lacking = Math.max(0, plan.needed_bytes - plan.free_bytes);
    const options: { id: Choice; title: string; text: string; need: string }[] = [
        { id: 'move', title: 'Move my models there', text: none ? 'You have no models to move.' : plan.same_drive ? 'Strata moves your {0} of models to the new folder. This is the same drive, so it takes a moment.' : 'Strata copies your {0} of models to the new folder, checks them, then removes the old copies.',
          need: none ? '' : plan.same_drive ? 'Needs no extra space.' : 'Needs {0} free in the new folder. You have {1}.' },
        { id: 'keep', title: 'Keep them where they are and also use the new folder', text: 'Nothing is copied. Your current models keep working, and new downloads go to the new folder.', need: short ? 'Needs no space now. New downloads must fit in {1}.' : 'Needs no space now.' },
        { id: 'empty', title: 'Start empty', text: 'Strata uses the new folder from now on and does not list your current models. Your old files are not deleted.', need: 'Needs no space now. Each model you download later needs its own space.' },
    ];
    const values = [gb(plan.models_bytes), gb(plan.free_bytes)];
    const running = plan.running > 0;
    return <>
    <div className="models-picked"><span className="small muted">{tr('New folder')}</span><code className="storage-directory__path">{plan.shown}</code><span className="small muted">{tr('{0} free on this drive', [gb(plan.free_bytes)])}</span></div>
    {short ? <div className="models-notice models-notice--warn" role="alert"><Icon name="warning" size={18}/><div><strong>{tr('Your models need {0}, this drive has {1} free.', [gb(plan.needed_bytes), gb(plan.free_bytes)])}</strong><br/>{tr('Free up about {0} there, pick another folder, or choose one of the other two options.', [gb(lacking)])}</div></div> : null}
    <fieldset className="models-choices" aria-label={tr('What should happen to your models')}>
      {options.map(o => { const possible = plan.choices[o.id].possible; return <label className="models-choice" key={o.id} data-disabled={!possible || undefined}>
        <input type="radio" name="models-folder-choice" checked={choice === o.id} disabled={!possible} onChange={() => setChoice(o.id)}/>
        <span><strong>{tr(o.title)}</strong>{o.id === 'move' && possible && !none ? <span className="models-choice__tag">{tr('Recommended')}</span> : null}<br/><span className="small">{tr(o.text, o.id === 'move' ? [gb(plan.models_bytes)] : undefined)}</span></span>
        {o.need ? <span className="models-choice__need">{tr(o.need, o.id === 'move' ? [gb(plan.needed_bytes), gb(plan.free_bytes)] : [values[0], values[1]])}</span> : null}
      </label>; })}
    </fieldset>
    {running ? <p className="small muted">{tr('A running model is stopped first; load it again when the move is done.')}</p> : null}
    {error ? <Failure message={error.message} details={error.details}/> : null}
    <details className="models-details"><summary>{tr('Details')}</summary><ul>
      {names.length ? <li>{tr('{0} would move.', [list(names.map(m => `${m.label}, ${gb(m.bytes)}`))])}</li> : null}
      {short ? <li>{tr('Free space is measured on the drive that holds {0}.', [plan.shown])}</li> : <>
        <li>{tr('Old folder: {0}', [plan.current])}</li>
        <li>{tr('Strata copies one file at a time and checks its size before it removes the old one.')}</li>
        <li>{tr('You can stop at any time; files already moved stay usable.')}</li></>}
    </ul></details>
    </>;
}

function ResultDialog({ outcome, plan, native, onClose, onAgain }: { outcome: NativeJob | null; plan: Plan | null; native: NativeState | null; onClose: () => void; onAgain: () => void }) {
    if (!outcome) return null;
    const result = (outcome.result ?? {}) as Outcome;
    const sized = (groups: string[]) => list(groups.map(g => { const m = plan?.models.find(x => x.group === g); const name = modelName(g, native); return name ? (m ? `${name} (${gb(m.bytes)})` : name) : null; }).filter((x): x is string => !!x));
    if (outcome.state === 'error') {
        const full = result.code === 'disk_full';
        const where = driveOf(result.to ?? '');
        const some = (result.moved?.length ?? 0) > 0;
        return <Dialog open compact title={tr(full ? 'The drive ran out of space' : 'The move stopped')} onClose={onClose}
          actions={<><Btn onClick={onClose}>{tr('Close')}</Btn><Btn variant="primary" onClick={onAgain}>{tr('Choose another folder')}</Btn></>}>
          <div className="error-notice" role="alert"><Icon name="error" size={18}/><div>
            <strong>{full ? tr('{0} filled up after {1}, so the move stopped.', [where, gb(result.bytes ?? 0)]) : tr(outcome.error ?? 'The models could not be moved.')}</strong>
            <p>{full ? (some ? tr('Nothing was lost. Models already moved are in the new folder, the rest are still in {0}, and the half-copied file was removed. Free up space on {1} or choose another folder, then try again.', [result.from ?? '', where])
              : tr('Nothing was lost. All your models are still in {0}, and the half-copied file was removed. Free up space on {1} or choose another folder, then try again.', [result.from ?? '', where]))
              : tr('Nothing was lost. Your models that were not moved are still in {0}. Choose another folder to try again.', [result.from ?? ''])}</p>
            {outcome.details ? <details className="error-detail"><summary className="small">{tr('Details')}</summary><pre className="small muted">{outcome.details}</pre></details> : null}
          </div></div>
        </Dialog>;
    }
    if (result.stopped) {
        const moved = sized(result.moved ?? []), left = sized(result.remaining ?? []);
        return <Dialog open compact title={tr('Moving stopped')} onClose={onClose} actions={<Btn variant="primary" onClick={onClose}>{tr('Close')}</Btn>}>
          <div className="models-notice models-notice--info"><Icon name="info" size={18}/><div><strong>{tr('Nothing was lost.')}</strong><br/>
            {moved ? tr('{0} is in the new folder.', [moved]) + ' ' : ''}{left ? tr('{0} is still in the old folder.', [left]) + ' ' : ''}{moved && left ? tr('Both work.') : ''}</div></div>
          <p className="small muted">{tr('Models in both folders are listed as normal. You can finish moving later with Change folder.')}</p>
        </Dialog>;
    }
    return <Dialog open compact title={tr('Your models are moved')} onClose={onClose} actions={<Btn variant="primary" onClick={onClose}>{tr('Done')}</Btn>}>
      <div className="models-ok"><span className="models-ok__mark"><Icon name="check" size={20}/></span><div><strong>{(() => { const [before, after] = tr('{0} now live in {1}', [gb(result.total ?? 0), '@@']).split('@@'); return <>{before}<code className="models-path">{plan?.shown ?? result.to ?? ''}</code>{after}</>; })()}</strong>
        {result.sameDrive ? null : <><br/><span className="small muted">{tr('The old copies were removed and {0} is free again on {1}.', [gb(result.total ?? 0), driveOf(result.from ?? '')])}</span></>}</div></div>
    </Dialog>;
}
