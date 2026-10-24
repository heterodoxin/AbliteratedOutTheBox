import { tr } from '../lib/i18n';
import { useState } from 'react';
import { Btn, ErrorNotice, Icon, Select } from './ui';
import type { ImageModelChoice, ImageModelLibrary } from '../lib/types';
import { cancelImageModel, installImageModel, selectImageModel } from '../lib/store';
export function RunImageModel({ library, disabled, onChange }: {
    library: ImageModelLibrary | null;
    disabled: boolean;
    onChange: () => void;
}) {
    const [pending, setPending] = useState(false);
    const [error, setError] = useState('');
    const version = library?.variants.find(v => v.id === library.selected);
    const downloading = library?.variants.find(v => v.state === 'downloading');
    async function perform(action: () => Promise<void>) {
        setPending(true);
        setError('');
        try {
            await action();
        }
        catch (reason) {
            setError(reason instanceof Error ? reason.message : String(reason));
        }
        finally {
            setPending(false);
        }
    }
    return <div className="run-image-model">
    <div className="run-image-model__choice">
      <div className="run-image-model__label"><Icon name="image" size={18}/><span><strong>{tr("Image model ")}<small>{tr("Optional")}</small></strong><a href={library?.sourceUrl ?? 'https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF'} target="_blank" rel="noreferrer">{tr(library?.family ?? 'Qwen-Image-2.1-Turbo')}</a></span></div>
      <Select<ImageModelChoice> ariaLabel={tr("Image model version")} value={library?.selected ?? 'off'} disabled={disabled || pending || !!downloading || !library} menuWidth={240} menuSize="comfortable" options={[{ value: 'off', label: 'Off' }, ...(library?.variants.map(v => ({ value: v.id, label: `${v.label} · ${v.downloadGb.toFixed(1)} GB`, title: `${v.quant}${v.state === 'installed' ? ' · Downloaded' : ''}` })) ?? [])]} onChange={choice => { onChange(); void perform(() => selectImageModel(choice)); }}/>
    </div>
    {tr(downloading ? <div className="run-image-download" role="status" aria-live="polite"><div><span>{tr(downloading.progress?.stage ?? 'Downloading')} {tr(downloading.label)} <b>{tr(Math.floor(downloading.progress?.pct ?? 0))}%</b></span><Btn size="sm" disabled={pending} onClick={() => void perform(() => cancelImageModel(downloading.id))}>{tr("Cancel")}</Btn></div><progress aria-label={tr("Image model download progress")} max={100} value={downloading.progress?.pct ?? 0}/></div> : version ? <div className="run-image-model__action">
      <span className="run-image-model__status">{tr(version.state === 'installed' ? <><Icon name="check" size={15}/>{tr(library?.generationAvailable ? 'Downloaded' : 'Downloaded · Image engine required')}</> : 'Download this version to use it.')}</span>
      {tr(version.state !== 'installed' ? <Btn size="sm" icon="download" disabled={disabled || pending} onClick={() => void perform(() => installImageModel(version.id))}>{tr(version.state === 'error' ? 'Retry download' : `Download ${version.label}`)}</Btn> : null)}
    </div> : null)}
    {tr(version?.error && !error ? <ErrorNotice message={version.error}/> : null)}
    {tr(error ? <ErrorNotice message={error}/> : null)}
  </div>;
}
