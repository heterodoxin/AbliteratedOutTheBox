import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { fetchUpdateCheck, publishUpdateCheck, useInstalledVersions, useLastUpdateCheck } from '../lib/update-check';
import { nativeAction, useNative, type NativeJob } from '../lib/native';
import { Btn, Dialog, ErrorNotice, Select } from './ui';
import { NativeProgress } from './NativeModels';
import { jobErrorText } from '../lib/native-job-diagnostics';
export function NativeBenchmark({ open, onClose }: {
    open: boolean;
    onClose: () => void;
}) {
    const native = useNative();
    const [selected, setSelected] = useState(''), [error, setError] = useState(''), [jobId, setJobId] = useState(''), [report, setReport] = useState<Record<string, any> | null>(null);
    const instance = native?.instances.find(i => i.id === selected) ?? native?.instances.find(i => i.id === native.defaultId) ?? native?.instances[0];
    const job = native?.jobs.find(j => j.id === jobId), active = native?.jobs.some(j => j.state === 'running');
    useEffect(() => {
        if (open)
            void api.get<Record<string, any> | null>('/api/native/benchmark').then(setReport).catch(() => { });
    }, [open]);
    useEffect(() => {
        if (job?.state === 'done' && job.result)
            setReport(job.result);
        if (job?.state === 'error')
            setError(job.error ?? job.summary);
    }, [job?.state]);
    async function run() {
        setError('');
        try {
            const job = await nativeAction<NativeJob>('operations', { kind: 'benchmark', args: { instanceId: instance?.id } });
            setJobId(job.id);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }
    function download() {
        if (!report)
            return;
        const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = 'strata-benchmark.json';
        a.click();
        URL.revokeObjectURL(url);
    }
    return <Dialog open={open} title={tr("Measure performance")} onClose={onClose} actions={<><Btn onClick={onClose}>{tr("Close")}</Btn>{tr(job?.state === 'running' ? <Btn onClick={() => void nativeAction(`operations/${job.id}/cancel`).catch(e => setError(e.message))}>{tr("Stop benchmark")}</Btn> : null)}{tr(report ? <Btn icon="download" onClick={download}>{tr("Save report")}</Btn> : null)}<Btn variant="primary" disabled={!instance || instance.status !== 'ready' || active} onClick={() => void run()}>{tr(active ? 'Measuring…' : 'Run short benchmark')}</Btn></>}><div className="stack"><p className="small muted">{tr("A fixed prompt with thinking off and up to 200 output tokens. Usually takes 10\u201360 seconds. Results are measured by your engine.")}</p>{tr((native?.instances.length ?? 0) > 1 ? <Select ariaLabel={tr("Benchmark model")} value={instance?.id ?? ''} options={(native?.instances ?? []).map(i => ({ value: i.id, label: i.label }))} onChange={setSelected}/> : null)}{tr(!instance ? <p>{tr("Load a native model first.")}</p> : null)}{tr(job?.state === 'running' ? <p role="status">{tr(job.summary)}</p> : null)}{tr(report ? <div className="native-benchmark"><div><span>{tr("Generating")}</span><strong>{tr(report.outputTps == null ? '—' : Number(report.outputTps).toFixed(1))} <small>{tr("tok/s")}</small></strong></div><div><span>{tr("Reading")}</span><strong>{tr(report.promptTps == null ? '—' : Number(report.promptTps).toFixed(0))} <small>{tr("tok/s")}</small></strong></div><p className="xs muted">{tr(report.model)} · {tr(new Date(report.at).toLocaleString(localeTag()))} · {tr(report.settings)}</p></div> : null)}{tr(error ? <ErrorNotice message={error}/> : null)}</div></Dialog>;
}
export function NativeOperations() {
    const native = useNative();
    const [error, setError] = useState(''), [saved, setSaved] = useState(''), [saving, setSaving] = useState(false);
    const [logs, setLogs] = useState<{
        at: string;
        instanceId: string;
        line: string;
    }[]>([]), [showLogs, setShowLogs] = useState(false), [confirm, setConfirm] = useState(false);
    // Opening this page shows the last check kept on this computer and makes no network call. The check runs when "Check for updates" is pressed.
    const versions = useInstalledVersions();
    const last = useLastUpdateCheck(versions);
    const [checking, setChecking] = useState(false), [checkError, setCheckError] = useState('');
    const update: 'checking' | 'available' | 'current' | 'unknown' = checking ? 'checking' : last?.engine.status === 'available' ? 'available' : last?.engine.status === 'current' ? 'current' : 'unknown';
    const active = native?.jobs.find(j => j.state === 'running');
    async function checkForUpdates() {
        setChecking(true);
        setCheckError('');
        try {
            const data = await fetchUpdateCheck();
            publishUpdateCheck(data, versions);
            if (data.engine.status === 'error')
                setCheckError(data.engine.message);
        }
        catch (e) {
            setCheckError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setChecking(false);
        }
    }
    useEffect(() => {
        if (!showLogs)
            return;
        const read = () => void api.get<{
            logs: typeof logs;
        }>('/api/native/logs').then(value => { setLogs(value.logs); }).catch(() => { });
        read();
        const timer = setInterval(read, 2000);
        return () => clearInterval(timer);
    }, [showLogs]);
    async function updateEngine() {
        setError('');
        try {
            await nativeAction('operations', { kind: 'update-engine' });
            setConfirm(false);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }
    // The window cannot save a download, so the host writes the file to Downloads and says where.
    async function saveLogs() {
        setSaving(true);
        setSaved('');
        setError('');
        try {
            const result = await api.post<{ path: string }>('/api/native/logs/save', {});
            setSaved(result.path);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setSaving(false);
        }
    }
    return <div className="stack stack--lg native-engine-settings">
 <section className="st-card"><div className="section-toolbar"><h2 className="h3">{tr("Update component")}</h2>{tr(update === 'available' ? <Btn size="sm" variant="primary" disabled={!!active || (native?.instances.length ?? 0) > 0} onClick={() => setConfirm(true)}>{tr("Update available")}</Btn> : null)}{tr(update !== 'available' ? <Btn size="sm" disabled={update === 'checking'} onClick={() => void checkForUpdates()}>{tr(update === 'checking' ? 'Checking…' : 'Check for updates')}</Btn> : null)}</div><p className="small muted">{tr("Update the engine for this computer. Your models and settings are kept.")}</p>{tr(update === 'current' ? <p className="small muted">{tr("Up to date")}</p> : null)}{tr(checkError ? <p className="small inline-error" role="alert">{tr(checkError)}</p> : null)}{tr(update === 'available' && (native?.instances.length ?? 0) > 0 ? <p className="small muted">{tr("Open Models and unload the running native models first.")}</p> : null)}</section>
 <section className="st-card"><div className="section-toolbar"><h2 className="h3">{tr("Diagnostics")}</h2><Btn size="sm" onClick={() => setShowLogs(true)}>{tr("View logs")}</Btn></div><p className="small muted">{tr("Downloads, preparation and engine failures include their original messages here.")}</p>{tr(native?.jobs.filter(j => j.state === 'error').slice(-3).map(j => <ErrorNotice key={j.id} message={jobErrorText(j, native.jobs)} detail={j.details}/>))}</section>
 <Dialog open={showLogs} wide title={tr("Engine logs")} onClose={() => setShowLogs(false)} actions={<><Btn icon="download" disabled={saving} onClick={() => void saveLogs()}>{tr(saving ? 'Working…' : 'Save logs')}</Btn><Btn variant="primary" onClick={() => setShowLogs(false)}>{tr("Close")}</Btn></>}>
  <pre className="native-log" aria-label={tr("Engine log")}>{logs.length ? logs.map(l => `${l.at} · ${l.line}`).join(String.fromCharCode(10)) : tr('No engine messages yet.')}</pre>
  {tr(saved ? <p className="small" role="status">{tr("Saved to {0}", [saved])}</p> : null)}
 </Dialog>
 {tr(active ? <NativeProgress job={active}/> : null)}{tr(error ? <ErrorNotice message={error}/> : null)}
 <Dialog open={confirm} compact title={tr("Update engine?")} onClose={() => setConfirm(false)} actions={<><Btn onClick={() => setConfirm(false)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!!active || (native?.instances.length ?? 0) > 0} onClick={() => void updateEngine()}>{tr("Update")}</Btn></>}><p>{tr("The native installer updates the installation. Model files and settings are kept. Unload native models before updating.")}</p></Dialog>
 </div>;
}
