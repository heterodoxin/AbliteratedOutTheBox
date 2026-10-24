import { tr } from '../lib/i18n';
import { useState } from 'react';
import { Btn, Dialog, ErrorNotice, Field, Icon } from './ui';
import { changeModelDirectory, getState, lifecycleStatus, removeVariant, runtimeModelAction, unloadModel, useStore } from '../lib/store';
import { pickerGroups, pickerVersionState } from '../lib/model-picker';
import { sizeGb } from '../lib/format';
import { useNative } from '../lib/native';
import { NativeDeleteDialog } from './NativeDelete';
import { ModelsFolderCard } from './ModelsFolderDialog';
import { nativeShortName } from '../lib/native-names';
export function SettingsStorage() {
    const state = useStore(s => s);
    const native = useNative();
    const [selected, setSelected] = useState<string | null>(null);
    const [nativeDeleting, setNativeDeleting] = useState<string | null>(null);
    const [pending, setPending] = useState(false), [error, setError] = useState('');
    const [editing, setEditing] = useState(false), [directory, setDirectory] = useState('');
    const [directoryPending, setDirectoryPending] = useState(false), [directoryError, setDirectoryError] = useState(''), [notice, setNotice] = useState('');
    const library = state.library;
    const versions = library ? pickerGroups(library).flatMap(group => group.versions) : [];
    // ONE list and ONE total: the optimized models (sizes from their real files, shared files counted once) and the other local models together.
    const nativeModels = native?.installed?.models ?? [];
    const sharedGb = (native?.installed?.usage?.shared_bytes ?? 0) / 1e9;
    const nativeGb = native?.installed?.usage ? native.installed.usage.total_gb : nativeModels.reduce((n, m) => n + (m.diskGb ?? 0), 0);
    const totalGb = nativeGb + versions.reduce((n, v) => n + (v.diskGb ?? 0), 0);
    const rows = nativeModels.length + versions.length;
    const version = versions.find(v => v.id === selected);
    const running = !!version && (state.runtime?.instances.some(i => i.variantId === version.id) || library?.loaded === version.id);
    const lifecycle = lifecycleStatus(state);
    const currentDirectory = library?.disk.modelsDir ?? '';
    const loadedCount = (state.runtime?.instances.length ?? (library?.loaded ? 1 : 0)) + (native?.instances.length ?? 0);
    const directoryBlocked = !!native?.jobs.some(j => j.state === 'running') || !lifecycle.allowed || !!library?.downloads.active || !!state.imageModels?.variants.some(v => v.state === 'downloading') || !!library?.variants.some(v => v.state === 'loading');
    const changing = editing && directory.trim() !== currentDirectory;
    async function saveDirectory(path: string) {
        setDirectoryPending(true);
        setDirectoryError('');
        setNotice('');
        try {
            const result = await changeModelDirectory(path, path.trim() !== currentDirectory);   // checking the same folder unloads nothing
            const count = result.library.variants.filter(v => ['installed', 'loaded'].includes(v.state)).length + result.imageModels.variants.filter(v => v.state === 'installed').length;
            setEditing(false);
            setNotice(`${count} ${count === 1 ? 'version' : 'versions'} found${result.library.mode === 'simulated' ? ' in this example library' : ''}. Future downloads will use this folder.`);
        }
        catch (e) {
            setDirectoryError((e as Error).message);
        }
        finally {
            setDirectoryPending(false);
        }
    }
    const unavailable = !version || pickerVersionState(version, library?.loaded ?? null).busy || (running && !lifecycle.allowed);
    function close() {
        if (!pending) {
            setSelected(null);
            setError('');
        }
    }
    async function remove() {
        if (!version || unavailable)
            return;
        setPending(true);
        setError('');
        try {
            const instances = getState().runtime?.instances.filter(i => i.variantId === version.id) ?? [];
            if (instances.length)
                for (const instance of instances)
                    await runtimeModelAction(instance.id, 'unload');
            else if (getState().library?.loaded === version.id && !await unloadModel())
                throw new Error('Could not unload the model. Try again when the engine is idle.');
            if (!await removeVariant(version.id))
                throw new Error('Could not delete this version. Check the notification and try again.');
            setSelected(null);
        }
        catch (e) {
            setError((e as Error).message);
        }
        finally {
            setPending(false);
        }
    }
    const downloading = !!native?.jobs.some(j => j.state === 'running' && j.kind !== 'models-move') || !!library?.downloads.active || !!state.imageModels?.variants.some(v => v.state === 'downloading');
    return <>
    {native?.available && native.downloadDirectory ? <ModelsFolderCard blocked={downloading}/> : null}
    <section className="storage-directory" aria-labelledby="storage-directory-title">
      <div className="storage-directory__heading"><span><Icon name="folder" size={18}/><strong id="storage-directory-title">{tr("Model directory")}</strong></span><Btn size="sm" disabled={!currentDirectory || directoryPending || pending} onClick={() => { setDirectory(currentDirectory); setEditing(!editing); setDirectoryError(''); setNotice(''); }}>{tr(editing ? 'Cancel' : 'Change directory')}</Btn></div>
      <p className="small muted">{tr("Your models are saved here. Use an existing folder to reuse supported downloads.")}</p>
      <code className="storage-directory__path">{currentDirectory || 'Reading model directory…'}</code>
      {tr(native?.available && native.downloadDirectory ? <div className="storage-native-directory"><span className="small muted">{tr("Optimized Strata model files")}</span><code className="storage-directory__path">{native.downloadDirectory}</code><p className="xs muted">{tr("Chosen during setup. Prepared models keep using their existing files.")}</p></div> : null)}
      {tr(editing ? <form className="storage-directory__edit" onSubmit={e => {
                e.preventDefault();
                if (!directoryBlocked && directory.trim())
                    void saveDirectory(directory);
            }}>
        <label htmlFor="model-directory-path" className="small">{tr("Directory path")}</label>
        <input id="model-directory-path" className="st-input" value={directory} onChange={e => { setDirectory(e.target.value); setDirectoryError(''); }} autoFocus disabled={directoryPending} placeholder={tr("Full path to an existing folder")} spellCheck={false}/>
        <p className="small muted">{tr("Existing files stay where they are. Strata checks this folder for supported GGUF files.")}</p>
        {tr(changing && loadedCount > 0 ? <p className="small">{tr(loadedCount === 1 ? 'The loaded model will be unloaded.' : `All ${loadedCount} loaded models will be unloaded.`)}{tr(" Load a model again after switching.")}</p> : null)}
        <div className="row"><Btn variant="primary" type="submit" disabled={directoryPending || directoryBlocked || !directory.trim()}>{tr(directoryPending ? 'Checking files…' : changing && loadedCount > 0 ? 'Unload and change directory' : changing ? 'Use this directory' : 'Check folder')}</Btn></div>
      </form> : <Btn size="sm" icon="refresh" disabled={directoryPending || directoryBlocked || !currentDirectory} onClick={() => void saveDirectory(currentDirectory)}>{tr(directoryPending ? 'Checking files…' : 'Check folder')}</Btn>)}
      {tr(directoryBlocked && !directoryPending ? <p className="small muted">{tr(!lifecycle.allowed ? lifecycle.reason : 'Finish or cancel downloads and loading to change the directory.')}</p> : null)}
      {tr(state.setup?.app.modelsDirNotice && !notice ? <p className="small storage-directory__notice" role="status"><Icon name="info" size={15}/>{tr(state.setup.app.modelsDirNotice)}</p> : null)}
      {tr(notice ? <p className="small storage-directory__notice" role="status"><Icon name="check" size={15}/>{tr(notice)}</p> : null)}
      {tr(directoryError ? <ErrorNotice message={directoryError}/> : null)}
    </section>
    <Field wide label={tr("Downloaded models")} hint={tr("{0} on this computer. Choose a version to delete its files.", [sizeGb(totalGb)])}>
      {tr(rows ? <ul className="storage-models" aria-label={tr("Downloaded models")}>
        {tr(nativeModels.map(m => <li className="storage-model" key={m.model}><span className="storage-model__name" title={m.model_name || m.model}><strong>{nativeShortName(m.model)}</strong>{tr(native?.instances.some(i => i.model === m.model) ? <small className="muted">{tr('Loaded')}</small> : null)}</span><span className="storage-model__size muted">{tr(sizeGb(m.diskGb ?? 0))}</span><Btn size="sm" variant="danger" disabled={!!native?.jobs.some(j => j.state === 'running')} onClick={() => setNativeDeleting(m.model)}>{tr("Delete")}</Btn></li>))}
        {tr(versions.map(v => <li className="storage-model" key={v.id}><span className="storage-model__name" title={`${v.family} · ${v.quant}`}><strong>{`${v.local ? v.name : v.family} · ${v.quant}`}</strong>{tr(v.local ? <small className="muted">{tr('Local')}</small> : null)}{tr(v.id === library?.loaded ? <small className="muted">{tr('Loaded')}</small> : null)}</span><span className="storage-model__size muted">{tr(sizeGb(v.diskGb))}</span><Btn size="sm" variant="danger" disabled={pickerVersionState(v, library?.loaded ?? null).busy || pending || directoryPending} onClick={() => { setError(''); setSelected(v.id); }}>{tr("Delete")}</Btn></li>))}
        {tr(sharedGb > 0 ? <li className="storage-model storage-model--shared"><span className="storage-model__name"><strong>{tr("Shared files")}</strong><small className="muted">{tr("Used by more than one model")}</small></span><span className="storage-model__size muted">{tr(sizeGb(sharedGb))}</span><span/></li> : null)}
      </ul> : <p className="small muted">{tr("No downloaded models")}</p>)}
    </Field>
    <NativeDeleteDialog id={nativeDeleting} onClose={() => setNativeDeleting(null)}/>
    <Dialog open={!!selected} compact title={tr(version?.local ? 'Remove from Strata?' : `Delete ${version?.quant ?? 'model version'}?`)} onClose={close} actions={<><Btn disabled={pending} onClick={close}>{tr("Keep version")}</Btn><Btn variant="danger" disabled={pending || unavailable} onClick={() => void remove()}>{tr(pending ? 'Working…' : version?.local ? 'Remove model' : 'Delete version')}</Btn></>}>
      <p>{tr(version ? `${version.local ? version.name : version.family} · ${version.quant} · ${sizeGb(version.diskGb)}` : 'This version is no longer downloaded.')}</p>
      {tr(version ? <p className="small muted storage-delete-note">{tr(version.local ? `${running ? 'The model will be unloaded and removed from Strata.' : 'The model will be removed from Strata.'} Your original GGUF files stay in their folder.` : `${running ? 'The running model will be unloaded and its files deleted.' : 'Its files will be deleted from this computer.'} You can download it again in Models.`)}</p> : null)}
      {tr(running && !lifecycle.allowed ? <p className="small storage-delete-note">{tr(lifecycle.reason)}</p> : null)}
      {tr(error ? <ErrorNotice message={error}/> : null)}
    </Dialog>
  </>;
}
