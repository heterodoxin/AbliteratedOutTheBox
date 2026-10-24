import { tr } from '../lib/i18n';
import { useEffect, useState } from 'react';
import { Screen } from './layout';
import { Btn, ErrorNotice, Icon } from './ui';
import { api } from '../lib/client';
import { parseLocation } from '../lib/router';
import { nativeAction, useNative, type NativeJob } from '../lib/native';
import { nativeCatalogModels, catalogVersionLabel } from '../lib/native-catalog';
import { sizeGb } from '../lib/format';
import { NativeProgress } from './NativeModels';

interface Preview { path: string; preview: { model_dir?: string; plain_issues?: string[]; already_set_up?: { data_dir?: string | null; pack?: string | null } | null; file_count: number; total_bytes: number; file_set_status: string; issues: string[]; messages?: string[]; projectors?: { name: string; status: string; pinned: boolean; plain: string }[]; reads_pictures?: boolean; candidate_selection: { family: string; model: string } | null } }

/** The words for why a folder of GGUF files cannot be used: a sentence and what to do, the checker's own codes under Details. */
function problem(issues: string[]): { message: string; next: string } {
    const has = (re: RegExp) => issues.some(i => re.test(i));
    if (has(/multiple_model_groups/))
        return { message: 'This folder holds more than one model.', next: 'Type the full path of the .gguf file of the one you want, or choose the folder of that model.' };
    if (has(/incomplete_or_invalid_shard_set/))
        return { message: 'Some parts of this model are missing.', next: 'Put every part of one model (all the -0000N-of-0000M files) in one folder, then check again.' };
    if (has(/unsupported_architecture|unsupported_or_missing_architecture/))
        return { message: 'This model type isn’t supported by the Strata engine.', next: 'Add it as a Custom model API instead: start it with llama-server, then open Settings > Models > Add model provider > Custom model API.' };
    if (has(/unrecognized_or_unsupported_model_variant|expert_quantization|tokenizer/))
        return { message: 'Strata can’t run these files yet.', next: 'Strata runs the Qwen3.8-Flash-Next models. Download one from Models, or choose a folder with its files.' };
    return { message: 'These files could not be checked.', next: 'Check that the folder holds complete GGUF files, then try again. The file named under Details says what is wrong with it.' };
}

/** "Use a local GGUF": choose the folder of a model you already have, Strata checks it, and prepares and loads it without copying
 *  or downloading the model files. The same page layout and cards as the optimized models page. */
export function NativeLocalModel({ onBack }: { onBack: () => void }) {
    const native = useNative();
    const [typed, setTyped] = useState(''), [checked, setChecked] = useState<Preview | null>(null), [pending, setPending] = useState(false), [error, setError] = useState(''), [jobId, setJobId] = useState('');
    const job = native?.jobs.find(j => j.id === jobId);
    const active = native?.jobs.find(j => j.state === 'running');
    const selection = checked?.preview.candidate_selection;
    const usable = !!checked && checked.preview.file_set_status === 'plausible' && !!selection;
    const version = selection ? nativeCatalogModels(native?.catalog?.families ?? []).flatMap(f => f.sizes).find(s => s.sourceFamily === selection.family && s.model === selection.model) : undefined;
    const family = selection ? nativeCatalogModels(native?.catalog?.families ?? []).find(f => f.sizes.some(s => s.sourceFamily === selection.family && s.model === selection.model)) : undefined;
    async function check(path: string) {
        setPending(true);
        setError('');
        setChecked(null);
        try {
            const result = await api.post<Preview & { cancelled?: boolean }>('/api/native/local/preview', { path });
            if (!result.cancelled) {
                setChecked(result);
                setTyped(result.path);
            }
        }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setPending(false); }
    }
    /* The first-run page hands over the folder the person chose (#3507): it is checked at once, as if they had pressed Check files. */
    useEffect(() => { const path = parseLocation(window.location.hash).query.get('path'); if (path) { setTyped(path); void check(path); } }, []);
    /* Files that sit in a Strata data folder (<data>/models/<tag>) with their prepared model data beside them: that folder is used as the data
     * folder, so the prepared data is not made a second time. A folder the installer would not accept as a data folder is not passed. */
    const ownDataFolder = (() => { const d = checked?.preview.already_set_up?.pack ? checked.preview.already_set_up.data_dir : null; return d && /strata/i.test(d.split(/[\\/]/).filter(Boolean).pop() ?? '') ? d : null; })();
    async function prepare() {
        if (!usable || !selection || !checked)
            return;
        setPending(true);
        setError('');
        try {
            const result = await nativeAction<NativeJob>('operations', { kind: 'install', args: { family: selection.family, model: selection.model, gguf_dir: checked.preview.model_dir ?? checked.path, ...(ownDataFolder ? { data_dir: ownDataFolder } : {}), context: native?.catalog?.recommendation?.context ?? 32768, vision: checked.preview.projectors?.some(p => p.status === 'ok' && p.pinned) && native?.visionReader ? 'yes' : 'no', kv: 'int8', loadAfter: true } });
            setJobId(result.id);
        }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setPending(false); }
    }
    const found = checked ? problem(checked.preview.issues) : null;
    return <Screen plainHeader title={tr("Model")} actions={<Btn onClick={onBack}>{tr("Back")}</Btn>}>
  <div className="native-model-layout">
   <div className="stack stack--lg">
    <section className="st-card native-choice"><h2 className="h3">{tr("Use a local GGUF…")}</h2>
     <p className="small muted">{tr("Already have a supported model on this computer? Choose the folder that holds its files, or one of its .gguf files. Strata checks them, prepares the model and loads it. Nothing is copied or downloaded.")}</p>
     <label className="input-field"><span>{tr("Folder with the GGUF files, or one .gguf file")}</span><input className="st-input" value={typed} placeholder={tr("Full path to the folder or the file")} spellCheck={false} onChange={e => setTyped(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && typed.trim()) void check(typed); }}/></label>
     <div className="native-settings-actions"><Btn icon="folder" disabled={pending || !!active} onClick={() => void check('')}>{tr("Choose folder")}</Btn><Btn disabled={pending || !typed.trim() || !!active} onClick={() => void check(typed)}>{tr(pending && !jobId ? 'Checking…' : 'Check files')}</Btn></div>
    </section>
    {tr(checked ? <section className="st-card native-choice">{tr(usable ? <>
      <h2 className="h3">{tr(family ? family.title : 'Model found')}</h2>
      <p><strong>{tr(version ? catalogVersionLabel(version) : selection?.model ?? '')}</strong> · {tr(selection?.model ?? '')}</p>
      <p className="small muted">{tr("{0} files · {1}", [checked.preview.file_count, sizeGb(checked.preview.total_bytes / 1024 ** 3)])}</p>
      {tr(checked.preview.messages?.map(m => <p className="small muted" key={m}>{tr(m)}</p>))}
      <div className="native-model-actions"><Btn variant="primary" icon="models" disabled={pending || !!active} onClick={() => void prepare()}>{tr(pending ? 'Checking…' : 'Prepare and load')}</Btn></div>
     </> : <>
      <h2 className="h3">{tr(found!.message)}</h2>
      <p className="small muted">{tr(found!.next)}</p>
      {tr(checked.preview.plain_issues?.map(m => <p className="small muted" key={m}>{tr(m)}</p>))}
      {tr(checked.preview.messages?.map(m => <p className="small muted" key={m}>{tr(m)}</p>))}
      {tr(checked.preview.issues.length ? <details><summary className="small">{tr("Details")}</summary><pre className="small muted error-detail-text">{checked.preview.issues.join(String.fromCharCode(10))}</pre></details> : null)}
     </>)}</section> : null)}
    {tr(active ? <NativeProgress job={active}/> : job?.state === 'error' ? <ErrorNotice message={job.error ?? job.summary} detail={job.details}/> : job?.state === 'done' ? <p className="small" role="status"><Icon name="check" size={15}/> {tr(job.summary)}</p> : null)}
    {tr(error ? <ErrorNotice message={error}/> : null)}
   </div>
   <aside className="stack stack--lg"><section className="st-card"><h2 className="h4">{tr("This computer")}</h2><p>{tr(native?.hardware?.ram_gb || '—')}{tr(" GB RAM")}</p>{tr(native?.hardware?.gpus.length ? native.hardware.gpus.map(g => <p className="small" key={g.index}>{tr(g.name)} · {tr(g.vram_gb)}{tr(" GB")}</p>) : null)}</section></aside>
  </div>
 </Screen>;
}
