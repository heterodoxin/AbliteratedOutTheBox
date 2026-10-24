import { tr } from '../lib/i18n';
import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/client';
import { Btn, ErrorNotice } from './ui';
import { issueText, type SettingsFileInfo } from '../lib/run-config-issues';

/**
 * Expert settings: the model's settings live in one text file (its run configuration). A power user opens it in the system
 * text editor and changes any engine option; the change applies the next time the model starts. The file is checked here and
 * again before the model starts, and a mistake is shown in plain words with its line.
 */
export function NativeSettingsFile({ model }: { model: string }) {
    const [info, setInfo] = useState<SettingsFileInfo | null>(null), [pending, setPending] = useState(false), [error, setError] = useState('');
    const check = useCallback(() => api.get<SettingsFileInfo>(`/api/native/models/${model}/settings-file`).then(value => { setInfo(value); setError(''); }).catch(() => setInfo(null)), [model]);
    useEffect(() => {
        void check();
        // Back from the editor: look at the file again.
        const again = () => void check();
        window.addEventListener('focus', again);
        return () => window.removeEventListener('focus', again);
    }, [check]);
    async function open() {
        setPending(true);
        setError('');
        try {
            setInfo(await api.post<SettingsFileInfo>(`/api/native/models/${model}/settings-file`, {}));
        }
        catch (e) {
            setError((e as Error).message);
        }
        finally {
            setPending(false);
        }
    }
    const errors = info?.issues.filter(i => i.level === 'error') ?? [], warnings = info?.issues.filter(i => i.level === 'warning') ?? [];
    return <div className="native-settings-file">
    <h3 className="small strong">{tr("Settings file")}</h3>
    <div className="row"><Btn size="sm" disabled={pending || !info} onClick={() => void open()}>{tr("Open settings file")}</Btn><span className="small muted">{tr("Changes you make in the file apply the next time the model starts.")}</span></div>
    {tr(error ? <ErrorNotice message={error} next={tr("Open the file from its folder instead, or try again.")}/> : null)}
    {tr(errors.length ? <><ErrorNotice message={tr("Your settings file has a mistake, so the model will not start until it is fixed.")} next={tr("Open the settings file, fix the lines below, and load the model again.")}/>{tr(errors.map((issue, i) => <p className="small" key={`e${i}`}>{tr(issueText(issue))}</p>))}</> : null)}
    {tr(warnings.map((issue, i) => <p className="small muted" key={`w${i}`}>{tr(issueText(issue))}</p>))}
  </div>;
}
