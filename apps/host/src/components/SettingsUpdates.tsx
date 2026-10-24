import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { Btn, Dialog, Icon, Select } from './ui';
import { useStore } from '../lib/store';
import type { SoftwareUpdate, UpdateCheckResult } from '../lib/types';
import { nativeAction, useNative } from '../lib/native';
import { NativeProgress } from './NativeModels';
import { jobErrorText } from '../lib/native-job-diagnostics';
import { appVersionLabel } from '../lib/app-version';
import { fetchUpdateCheck, publishUpdateCheck, useInstalledVersions, useLastUpdateCheck } from '../lib/update-check';
/** A check from an earlier day shows its date too, because the result is kept between visits. */
function checkedLabel(at: string) {
    const when = new Date(at), sameDay = when.toDateString() === new Date().toDateString();
    return sameDay ? when.toLocaleTimeString(localeTag(), { hour: '2-digit', minute: '2-digit' }) : when.toLocaleString(localeTag(), { dateStyle: 'medium', timeStyle: 'short' });
}
/** Installed versions are real host values. Installation needs a host updater. */
export function SettingsUpdates() {
    const native = useNative();
    const [confirm, setConfirm] = useState('');
    const appVersion = appVersionLabel(useStore(s => s.setup?.app.version), native?.version);
    const engineVersion = useStore(s => s.engine?.engine.version);
    const [rollback, setRollback] = useState(false);
    const [component, setComponent] = useState<'strata' | 'engine'>('strata');
    const [checking, setChecking] = useState(false);
    // Opening Settings shows the last check kept on this computer. The network is touched only when "Check for updates" is pressed.
    const versions = useInstalledVersions();
    const last = useLastUpdateCheck(versions);
    const [fresh, setFresh] = useState<UpdateCheckResult | null>(null);
    const [error, setError] = useState('');
    const request = useRef<AbortController | null>(null);
    useEffect(() => () => request.current?.abort(), []);
    const installed = component === 'strata' ? appVersion : native?.available ? (native.installed?.engine?.version ?? 'Not installed') : engineVersion;
    const active = native?.jobs.find(j => j.state === 'running');
    const result = checking || error ? null : (fresh ?? last);
    async function operate(kind: string) {
        setError('');
        try {
            await nativeAction('operations', { kind });
            setConfirm('');
            setRollback(false);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }
    async function check() {
        request.current?.abort();
        const controller = new AbortController();
        request.current = controller;
        setChecking(true);
        setError('');
        setFresh(null);
        try {
            const data = await fetchUpdateCheck(controller.signal);
            if (!controller.signal.aborted) {
                setFresh(data);
                publishUpdateCheck(data, versions);
            }
        }
        catch (e) {
            if (!controller.signal.aborted)
                setError(e instanceof Error ? e.message : 'Could not check for updates.');
        }
        finally {
            if (!controller.signal.aborted)
                setChecking(false);
        }
    }
    function versionRow(name: string, version: string | undefined, update?: SoftwareUpdate) {
        const labels: Record<SoftwareUpdate['status'], string> = { available: 'Update available', current: 'Up to date', unavailable: 'Check unavailable', error: 'Check failed', 'no-releases': 'No release yet' };
        return <div className="settings-version" data-update={update?.status}>
      <Icon name={name === 'Strata' ? 'layers' : 'engine'} size={20}/>
      <span><strong>{tr(name)}</strong><small>{tr(version ?? 'Version unavailable')}</small>{tr(!checking && update ? <small className="settings-version__message">{tr(update.message)}</small> : null)}</span>
      {tr(!checking && update?.status === 'available' ? (native?.available ? <Btn size="sm" variant="primary" disabled={!!active || (name === 'Engine' && !native.installed?.models.length && !native.capabilities.componentUpdates)} onClick={() => setConfirm(name === 'Strata' ? 'update-strata' : 'update-engine')}>{tr("Update available")}</Btn> : update.releaseUrl ? <a className="st-btn st-btn--secondary st-btn--sm" href={update.releaseUrl} target="_blank" rel="noreferrer" title={tr("Open the release page to update this installation")}>{tr("Update available")}<Icon name="download" size={15}/></a> : <span className="settings-version__label">{tr(labels[update.status])}</span>) : <span className="settings-version__label">{tr(checking ? 'Checking…' : update ? labels[update.status] : version === 'Not installed' ? 'Not installed' : 'Installed')}</span>)}
    </div>;
    }
    return <>
    <div className="settings-versions" aria-busy={checking}>
      {tr(versionRow('Strata', appVersion, result?.strata))}
      {tr(versionRow('Engine', native?.available ? (native.installed?.engine?.version ?? 'Not installed') : engineVersion, result?.engine))}
      <div className="settings-update-actions"><Btn size="sm" icon="refresh" disabled={checking} className={checking ? 'settings-update-checking' : undefined} onClick={() => void check()}>{tr(checking ? 'Checking…' : error || result?.strata.status === 'error' ? 'Try again' : result ? 'Check again' : 'Check for updates')}</Btn><Btn size="sm" variant="ghost" onClick={() => setRollback(true)}>{tr("Roll back\u2026")}</Btn></div>
      <div className="settings-update-inline" role="status" aria-live="polite">{tr(checking ? 'Checking for new versions…' : error ? <span className="inline-error">{tr(error)}</span> : result ? <><span className="sr-only">{tr(result.strata.message)} {tr(result.engine.message)} </span>{tr("Checked at ")}{tr(checkedLabel(result.checkedAt))}</> : null)}</div>
    </div>
    <Dialog open={rollback} compact title={tr("Roll back a version")} onClose={() => setRollback(false)} actions={<Btn onClick={() => setRollback(false)}>{tr("Done")}</Btn>}>
      <div className="settings-update-dialog">
        <label className="input-field"><span>{tr("Software")}</span><Select ariaLabel={tr("Software to manage")} value={component} options={[{ value: 'strata', label: 'Strata' }, { value: 'engine', label: 'Engine' }]} onChange={setComponent}/></label>
        <div className="settings-update-installed"><span>{tr("Installed version")}</span><strong>{tr(installed ?? 'Unavailable')}</strong></div>
        <div className="settings-update-status" role="status"><Icon name="info" size={18}/><div><strong>{tr(component === 'engine' && native?.capabilities.rollback ? 'Previous engine available' : 'No saved previous version')}</strong><p>{tr(component === 'engine' && native?.capabilities.rollback ? 'The native installer can restore its previous engine. Model files stay in place.' : 'Rollback requires a previous engine saved by a version of Strata that supports rollback.')}</p></div></div>
        <Btn variant="primary" disabled={component !== 'engine' || !native?.capabilities.rollback || !!active || (native?.instances.length ?? 0) > 0} onClick={() => void operate('rollback')}>{tr("Restore previous engine")}</Btn>
      </div>
    </Dialog>
    {tr(active ? <NativeProgress job={active}/> : null)}{tr(native?.jobs.filter(j => j.state === 'error' && ['update-strata', 'update-engine', 'rollback'].includes(j.kind)).slice(-1).map(j => <p role="alert" key={j.id} className="small inline-error">{tr(jobErrorText(j, native.jobs))}</p>))}
    <Dialog open={!!confirm} compact title={tr(confirm === 'update-strata' ? 'Update Strata?' : 'Update engine?')} onClose={() => setConfirm('')} actions={<><Btn onClick={() => setConfirm('')}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!!active || (native?.instances.length ?? 0) > 0} onClick={() => void operate(confirm)}>{tr("Update")}</Btn></>}><p>{tr("The native installer updates the installation. Model files and settings are kept. Unload native models before updating.")}</p>{tr((native?.instances.length ?? 0) > 0 ? <p className="small muted">{tr("Open Models and unload the running native models first.")}</p> : null)}</Dialog>
  </>;
}

/** "Strata 0.2.3 · Engine 0.1.48": the app's own version first, the engine's beside it (About). */
export function AboutVersions() {
    const native = useNative();
    const app = appVersionLabel(useStore(s => s.setup?.app.version), native?.version);
    const engineVersion = useStore(s => s.engine?.engine.version);
    const engine = native?.available ? native.installed?.engine?.version : engineVersion;
    return <p className="settings-about-version"><strong>{tr("Strata")} {app ?? ''}</strong>{engine ? <span className="muted"> · {tr("Engine")} {engine}</span> : null}</p>;
}
