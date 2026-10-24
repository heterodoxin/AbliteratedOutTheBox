import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useEffect, useState } from 'react';
import { Btn, Dialog, ErrorNotice, Icon } from './ui';
import { api } from '../lib/client';
import { addLocalModel } from '../lib/store';
import { sizeGb } from '../lib/format';
import type { LocalModelFolder, LocalModelPreview } from '../lib/types';
const fileSize = (bytes: number) => bytes < 1024 ** 2 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : bytes < 1024 ** 3 ? `${Math.round(bytes / 1024 ** 2)} MB` : sizeGb(bytes / 1024 ** 3);
export function LocalModelImport({ open, directory, onClose, onChoose }: {
    open: boolean;
    directory: string;
    onClose: () => void;
    onChoose: (id: string) => void;
}) {
    const [folder, setFolder] = useState<LocalModelFolder | null>(null), [path, setPath] = useState(''), [preview, setPreview] = useState<LocalModelPreview | null>(null);
    const [name, setName] = useState(''), [pending, setPending] = useState(false), [error, setError] = useState('');
    useEffect(() => {
        if (open) {
            setPreview(null);
            setError('');
            void browse(directory);
        }
    }, [open]);
    async function browse(input: string) {
        setPending(true);
        setError('');
        try {
            const next = await api.get<LocalModelFolder>(`/api/local-models/browse?directory=${encodeURIComponent(input)}`);
            setFolder(next);
            setPath(next.directory);
            setPreview(null);
        }
        catch (e) {
            setPath(input);
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    async function inspect(file: string) {
        setPending(true);
        setError('');
        try {
            const next = await api.post<LocalModelPreview>('/api/local-models/inspect', { path: file });
            setPreview(next);
            setName(next.variant.name);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    async function useModel() {
        if (!preview)
            return;
        setPending(true);
        setError('');
        try {
            const id = await addLocalModel(preview.variant.local!.path, name);
            onChoose(id);
            onClose();
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    const close = () => {
        if (!pending)
            onClose();
    };
    return <Dialog open={open} title={tr("Use a local model")} className="local-model-dialog" onClose={close} actions={<><Btn disabled={pending} onClick={preview ? () => { setPreview(null); setError(''); } : close}>{tr(preview ? 'Choose another file' : 'Cancel')}</Btn>{tr(preview ? <Btn variant="primary" icon="models" disabled={pending || !name.trim()} onClick={() => void useModel()}>{tr(pending ? 'Adding…' : 'Use this model')}</Btn> : null)}</>}>
  {preview ? <div className="local-model-review">
   <div className="local-model-review__identity"><Icon name="model-file" size={25}/><div><strong>{preview.variant.name}</strong><small>{tr("Local GGUF \u00B7 ")}{tr(preview.variant.local!.shards === 1 ? 'One file' : `${preview.variant.local!.shards} files found`)}</small></div></div>
   <dl className="local-model-facts"><div><dt>{tr("Architecture")}</dt><dd>{tr(preview.variant.local!.architecture)}</dd></div><div><dt>{tr("Quantization")}</dt><dd>{tr(preview.variant.quant)}</dd></div><div><dt>{tr("Size")}</dt><dd>{tr(fileSize(preview.variant.local!.files.reduce((n, f) => n + f.bytes, 0)))}</dd></div><div><dt>{tr("Context limit")}</dt><dd>{tr(Math.floor(preview.variant.local!.contextLength / 1024).toLocaleString(localeTag()))}{tr("K tokens")}</dd></div></dl>
   <label className="local-model-name">{tr("Display name")}<input className="st-input" value={name} maxLength={160} onChange={e => setName(e.target.value)} disabled={pending}/></label>
   <details className="local-model-files"><summary>{tr(preview.variant.local!.shards === 1 ? 'File location' : 'Model files')}</summary><code>{preview.variant.local!.path}</code>{tr(preview.files.length > 1 ? <ul>{tr(preview.files.map(file => <li key={file}>{tr(file)}</li>))}</ul> : null)}</details>
   <p className="small muted">{tr("Strata uses these files in place. You can load the model after adding it.")}</p>
   <p className="local-model-compatibility small"><Icon name="info" size={16}/><span>{tr("Custom model \u00B7 engine compatibility is checked when loading. Memory estimates use the file\u2019s metadata.")}</span></p>
   {tr(!preview.variant.local!.hasChatTemplate ? <p className="small muted">{tr("No chat template is declared; response formatting depends on the engine.")}</p> : null)}
  </div> : <div className="local-model-browser">
   <p className="small muted">{tr("Choose the model’s .gguf file. For a split model, choose any shard; Strata finds the first and checks the full set.")}</p>
   <form className="local-model-browser__path" onSubmit={e => {
                e.preventDefault();
                if (!pending)
                    void (/\.gguf$/i.test(path.trim()) ? inspect(path) : browse(path));
            }}><input className="st-input" aria-label={tr("Model file or folder path")} placeholder={tr("Model file or folder path")} value={path} disabled={pending} onChange={e => setPath(e.target.value)}/><Btn type="submit" disabled={pending || !path.trim()}>{tr("Go")}</Btn></form>
   <div className="local-model-browser__navigation"><div>{folder?.shortcuts.map(shortcut => <Btn key={shortcut.path} size="sm" icon="folder" disabled={pending} onClick={() => void browse(shortcut.path)}>{tr(shortcut.name)}</Btn>)}</div><Btn size="sm" disabled={pending || !folder?.parent} onClick={() => folder?.parent && void browse(folder.parent)}>{tr("Up a folder")}</Btn></div>
   <div className="local-model-browser__list" aria-label={tr("Local model files")} aria-busy={pending}>{pending ? <p className="small muted" role="status">{tr("Reading files\u2026")}</p> : folder?.entries.length ? folder.entries.map(entry => <button key={entry.path} className="local-model-file" aria-label={tr("{0} {1}", [entry.kind === 'folder' ? tr('Open folder') : tr('Choose file'), entry.name])} onClick={() => void (entry.kind === 'folder' ? browse(entry.path) : inspect(entry.path))}><Icon name={entry.kind === 'folder' ? 'folder' : 'model-file'} size={19}/><span title={entry.name}>{entry.name}</span>{tr(entry.kind === 'folder' ? <Icon name="chevron" size={15}/> : <small>{tr(fileSize(entry.bytes ?? 0))}</small>)}</button>) : <div className="local-model-browser__empty"><Icon name="folder" size={26}/><strong>{tr("No GGUF files here")}</strong><p className="small muted">{tr("Open another folder or enter a model\u2019s full path above.")}</p></div>}</div>
   {tr(folder?.truncated ? <p className="small muted">{tr("Showing the first 1,000 items. Enter a path to open a specific file.")}</p> : null)}
   <p className="xs muted">{tr("Only folders and GGUF files are shown. LoRA adapters and image projectors need separate support.")}</p>
  </div>}
  {tr(error ? <ErrorNotice message={error} next={tr("Choose a complete text model GGUF file.")}/> : null)}
 </Dialog>;
}
