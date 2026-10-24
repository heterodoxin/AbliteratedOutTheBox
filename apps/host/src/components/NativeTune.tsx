import { tr } from '../lib/i18n';
import { useCallback, useEffect, useState } from 'react';
import { Btn, Dialog, Select } from './ui';
import { TuneRunView, TuneOutcomeCard } from './TuneRunView';
import { readOutcome } from '../lib/tune-outcome';
import { go } from '../lib/store';
import { nativeAction, useNative, type NativeJob } from '../lib/native';
import { answerNotice, tunePresetAction, tuneStatus, tuneUndo, type TuneProgress, type TunePreset, type TuneResult, type TuneStatus } from '../lib/native-tune';

type Phase = 'idle' | 'running' | 'result';

/**
 * Tune model: find the fastest settings for this PC and keep them. The tuner is Strata Control's (the same one the Terminal runs,
 * with the same saved tunings); this dialog only starts it, shows its progress and result in plain words, and lists, switches,
 * renames, deletes and undoes the saved tunings. It reuses the Auto-tune dialog's look.
 */
export function NativeTune({ open, onClose, model, resourceProfile, vision }: {
    open: boolean;
    onClose: () => void;
    model: string;
    resourceProfile: 'work' | 'performance';
    vision: string;
}) {
    const native = useNative();
    const [status, setStatus] = useState<TuneStatus | null>(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
    const [mode, setMode] = useState<'quick' | 'thorough'>('quick'), [use, setUse] = useState<'chat' | 'coding'>('chat'), [allowSmall, setAllowSmall] = useState(false);
    const [jobId, setJobId] = useState(''), [renaming, setRenaming] = useState<TunePreset | null>(null), [name, setName] = useState('');
    const running = native?.jobs.find(j => (j.kind === 'autotune' || j.kind === 'recheck') && j.state === 'running');
    const job: NativeJob | undefined = native?.jobs.find(j => j.id === jobId) ?? running;
    const phase: Phase = job?.state === 'running' ? 'running' : job && job.id === jobId ? 'result' : 'idle';
    const other = native?.jobs.find(j => j.state === 'running' && j.kind !== 'autotune' && j.kind !== 'recheck');

    const reload = useCallback(async () => {
        try { setStatus(await tuneStatus(model, resourceProfile, vision)); setError(''); }
        catch (e) { setStatus(null); setError(e instanceof Error ? e.message : String(e)); }
    }, [model, resourceProfile, vision]);
    useEffect(() => { if (open && phase !== 'running') void reload(); }, [open, phase, reload, job?.state]);
    useEffect(() => { if (open && running && !jobId) setJobId(running.id); }, [open, running?.id]);

    async function start(kind: 'autotune' | 'recheck') {
        setBusy(true); setError('');
        try {
            const started = await nativeAction<NativeJob>('operations', { kind, args: { model, resourceProfile, vision, mode, use, allowSmallDifferences: allowSmall, presetId: status?.active?.id } });
            setJobId(started.id);
        } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setBusy(false); }
    }
    async function act(fn: () => Promise<unknown>) {
        setBusy(true); setError('');
        try { await fn(); await reload(); }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setBusy(false); }
    }
    function close() { if (phase !== 'running') setJobId(''); onClose(); }

    const progress = job?.progress as TuneProgress | undefined;
    const result = job?.state === 'done' ? (job.result as TuneResult | null) : null;
    const ratio = progress && progress.runsPlanned ? Math.min(0.96, progress.runsDone / progress.runsPlanned) : 0;
    const active = status?.active ?? null;

    const notice = (cls?: string, steady?: boolean) => cls ? <p className="small muted">{tr(answerNotice(cls, steady))}</p> : null;
    let body;
    let actions;
    if (phase === 'running') {
        body = <TuneRunView progress={progress} ratio={ratio} saving={false}/>;
        actions = <><Btn onClick={() => void act(() => nativeAction(`operations/${job!.id}/cancel`))} disabled={busy}>{tr("Stop")}</Btn></>;
    } else if (phase === 'result') {
        const outcome = readOutcome(result, job?.state);
        const ok = outcome.tone === 'ok';
        body = <div className="stack"><TuneOutcomeCard outcome={outcome} details={job?.state === 'error' ? (job.details || job.error) : null}/>{outcome.code === 'faster' ? notice(result?.answers?.class, result?.answers?.default_repeatable) : null}</div>;
        const again = <Btn variant={ok ? undefined : 'primary'} disabled={busy || !!other} onClick={() => void start('autotune')}>{tr(outcome.code === 'faster' || outcome.code === 'already_fastest' || outcome.code === 'still_fast' ? 'Tune again' : 'Try again')}</Btn>;
        actions = <>
            {status?.can_undo && outcome.code === 'faster' ? <Btn disabled={busy} onClick={() => void act(async () => { await tuneUndo(model, resourceProfile, vision); setJobId(''); })}>{tr("Undo")}</Btn> : null}
            <span className="tune-actions-gap"/>
            {ok ? <>{again}<Btn variant="primary" onClick={() => { close(); go('chat'); }}>{tr(outcome.code === 'faster' ? 'Start chatting' : 'Close')}</Btn></> : <><Btn onClick={close}>{tr("Close")}</Btn>{again}</>}
        </>;
    } else {
        const stale = active && status?.state && status.state !== 'fresh';
        body = <div className="stack">
            {error ? <p className="small inline-error" role="alert">{error}</p> : null}
            {active ? <div><p className="strong">{tr("Saved tuning in use: {0}", [active.name])}</p>
                {active.gain_pct ? <p className="small muted">{tr("+{0}% faster answers", [Math.round(active.gain_pct)])}</p> : null}
                {notice(active.answers?.class, active.answers?.default_repeatable)}
                {stale ? <p className="small">{status?.state === 'hard' ? tr("This PC changed since it was tuned, so these settings are not used until you tune again.") : tr("This PC or the engine changed since it was tuned. A short check takes about 2 minutes.")}</p> : null}</div>
                : <p className="small muted">{tr("No saved tuning for these choices yet.")}</p>}
            <p className="small muted">{tr("Strata tries different settings on this PC and keeps the ones that make answers faster. Your model is unloaded while this runs, and the computer will be busy.")}</p>
            {other ? <p className="small inline-error" role="alert">{tr("Finish the current operation first.")}</p> : null}
            <label className="input-field"><span>{tr("How long to spend")}</span><Select value={mode} ariaLabel={tr("How long to spend")} options={[{ value: 'quick', label: 'Quick (about 20 minutes)' }, { value: 'thorough', label: 'Thorough (up to 2 hours)' }]} onChange={v => setMode(v as 'quick' | 'thorough')}/></label>
            <label className="input-field"><span>{tr("What you mostly do")}</span><Select value={use} ariaLabel={tr("What you mostly do")} options={[{ value: 'chat', label: 'Chat' }, { value: 'coding', label: 'Coding with long prompts' }]} onChange={v => setUse(v as 'chat' | 'coding')}/></label>
            <label className="check-label"><input type="checkbox" checked={allowSmall} onChange={e => setAllowSmall(e.target.checked)}/>{tr("Allow settings that change answers very slightly (same quality)")}</label>
            {status && status.presets.length ? <section className="stack"><h2 className="h4">{tr("Saved tunings")}</h2>
                {status.presets.map(p => <div className="row row--between" key={p.id}>
                    <span>{p.name}{p.active ? <small> {'·'} {tr("In use")}</small> : null}</span>
                    <span className="row">{!p.active ? <Btn size="sm" variant="ghost" disabled={busy} onClick={() => void act(() => tunePresetAction(p.id, 'apply'))}>{tr("Use")}</Btn> : null}
                        <Btn size="sm" variant="ghost" disabled={busy} onClick={() => { setRenaming(p); setName(p.name); }}>{tr("Rename")}</Btn>
                        <Btn size="sm" variant="ghost" disabled={busy} onClick={() => void act(() => tunePresetAction(p.id, 'delete'))}>{tr("Delete")}</Btn></span></div>)}
            </section> : null}
            {renaming ? <div className="row"><input className="st-input" aria-label={tr("Name")} value={name} maxLength={80} onChange={e => setName(e.target.value)}/><Btn size="sm" variant="primary" disabled={!name.trim() || busy} onClick={() => void act(async () => { await tunePresetAction(renaming.id, 'rename', { name: name.trim() }); setRenaming(null); })}>{tr("Save")}</Btn><Btn size="sm" onClick={() => setRenaming(null)}>{tr("Cancel")}</Btn></div> : null}
        </div>;
        actions = <>
            {status?.can_undo ? <Btn disabled={busy || !!other} onClick={() => void act(async () => { await tuneUndo(model, resourceProfile, vision); })}>{tr("Undo")}</Btn> : null}
            {active && status?.state === 'soft' ? <Btn disabled={busy || !!other} onClick={() => void start('recheck')}>{tr("Check again")}</Btn> : null}
            <Btn variant="primary" icon="gauge" disabled={busy || !!other || !status} onClick={() => void start('autotune')}>{tr(active ? "Tune again" : "Start tuning")}</Btn></>;
    }
    return <Dialog open={open} wide={phase !== 'idle'} title={phase === 'result' && readOutcome(result, job?.state).code === 'faster' ? tr("Tuned and saved") : tr("Tune this model")} onClose={close} actions={actions}>{body}</Dialog>;
}
