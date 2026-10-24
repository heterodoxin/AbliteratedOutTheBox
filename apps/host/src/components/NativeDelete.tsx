import { tr } from '../lib/i18n';
import { useState } from 'react';
import { Btn, Dialog, ErrorNotice } from './ui';
import { nativeAction, nativeState, refreshNative, useNative, type NativeJob } from '../lib/native';
import { catalogVersionLabel, nativeCatalogModels } from '../lib/native-catalog';
import { sizeGb } from '../lib/format';

/** Delete a downloaded (native) model: its own files and run config. What another installed model still uses stays. A loaded model
 *  is unloaded first by the host. Same dialog and words as deleting a version of the managed library. */
export function NativeDeleteDialog({ id, onClose }: { id: string | null; onClose: () => void }) {
    const native = useNative();
    const [pending, setPending] = useState(false), [error, setError] = useState('');
    const entry = native?.installed?.models.find(m => m.model === id);
    const label = (() => { const sizes = nativeCatalogModels(native?.catalog?.families ?? []).flatMap(f => f.sizes); const v = sizes.find(s => s.id === id); return v ? `${catalogVersionLabel(v)} · ${v.model}` : entry?.model_name ?? id ?? ''; })();
    const running = !!native?.instances.some(i => i.model === id);
    const busy = !!native?.jobs.some(j => j.state === 'running');
    const size = entry?.diskGb ?? 0, gb = entry?.removableGb ?? size;
    // Files in the person's own folder (an adopted GGUF) stay where they are: only the entry is removed.
    const keepsFiles = entry?.removableGb === 0 && size > 0;
    function close() { if (!pending) { onClose(); setError(''); } }
    async function remove() {
        setPending(true);
        setError('');
        try {
            const job = await nativeAction<NativeJob>('operations', { kind: 'remove', args: { model: id } });
            for (let i = 0; i < 600; i++) {
                await new Promise(r => setTimeout(r, 500));
                await refreshNative(true).catch(() => undefined);
                const current = nativeState()?.jobs.find(j => j.id === job.id);
                if (current && current.state !== 'running') {
                    if (current.state === 'error') throw new Error(current.error ?? 'Could not delete this version.');
                    break;
                }
            }
            onClose();
        }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setPending(false); }
    }
    return <Dialog open={!!id} compact title={tr("Delete this version?")} onClose={close} actions={<><Btn disabled={pending} onClick={close}>{tr("Keep version")}</Btn><Btn variant="danger" disabled={pending || busy || !entry} onClick={() => void remove()}>{tr(pending ? 'Working…' : 'Delete version')}</Btn></>}>
      <p>{tr(label)}{tr(size ? ` · ${sizeGb(size)}` : '')}</p>
      {tr(!entry ? <p className="small muted">{tr("This version is no longer downloaded.")}</p> : <p className="small muted storage-delete-note">{keepsFiles ? tr('These files are in your own folder, so they stay there. Only the entry in Strata is removed.') : tr('Remove {0} GB of downloaded files. You can download this version again.', [gb ? gb.toFixed(1) : '0'])}{tr(running ? ' ' : '')}{tr(running ? (keepsFiles ? 'The running model will be unloaded.' : 'The running model will be unloaded and its files deleted.') : '')}</p>)}
      {tr(error ? <ErrorNotice message={error}/> : null)}
    </Dialog>;
}
