import { tr } from '../lib/i18n';
import { useCallback, useEffect, useState } from 'react';
import { Btn, Icon } from './ui';
import { answerNotice, tunePresetAction, tuneOff, tuneStatus, type TuneStatus } from '../lib/native-tune';
import { useNative } from '../lib/native';

/**
 * Which saved tuning a start will use, in plain words, and a way to switch it off (back to the standard settings) or on again.
 * Shown on the Models screen so nobody has to guess what runs after an unload or a restart. The tuning is applied by Strata at every
 * start (see strata_control/presets.py); this only reads it and flips it. Layout is the UI worker's: it is one line of text and one button.
 */
export function NativeTuneBadge({ model, resourceProfile, vision }: { model: string; resourceProfile: 'work' | 'performance'; vision: string }) {
    const native = useNative();
    const [status, setStatus] = useState<TuneStatus | null>(null), [busy, setBusy] = useState(false), [error, setError] = useState('');
    const jobs = native?.jobs.filter(j => j.state === 'running').length ?? 0;
    const read = useCallback(async () => {
        try { setStatus(await tuneStatus(model, resourceProfile, vision)); setError(''); }
        catch { setStatus(null); }
    }, [model, resourceProfile, vision]);
    useEffect(() => { void read(); }, [read, jobs, native?.instances.length]);
    async function flip(fn: () => Promise<unknown>) {
        setBusy(true); setError('');
        try { await fn(); await read(); }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setBusy(false); }
    }
    if (!status || (!status.active && !status.last_active)) return null;
    const active = status.active;
    return <p className="native-tune-line" data-tuning-active={!!active} role="status"><Icon name="gauge" size={15}/><span>
        {active ? <>{tr("Using your tuning '{0}'", [active.name])}{status.state === 'hard' ? <> {'·'} {tr("This PC changed since it was tuned, so these settings are not used until you tune again.")}</> : null}
            {active.answers && active.answers.class !== 'exact' ? <> {'·'} {tr(answerNotice(active.answers.class, active.answers.default_repeatable))}</> : null}{' '}
            <Btn size="sm" variant="ghost" disabled={busy} onClick={() => void flip(() => tuneOff(model, resourceProfile, vision))}>{tr("Switch off")}</Btn></>
            : <>{tr("Your tuning is off. The standard settings are used.")}{' '}
            <Btn size="sm" variant="ghost" disabled={busy} onClick={() => void flip(() => tunePresetAction(status.last_active!.id, 'apply'))}>{tr("Switch on")}</Btn></>}
        {error ? <span className="inline-error" role="alert"> {error}</span> : null}
    </span></p>;
}
