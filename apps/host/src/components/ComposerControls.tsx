import { tr } from '../lib/i18n';
import { modelFamilyIcon } from '../lib/model-icons';
import { useEffect, useRef, useState } from 'react';
import { FloatingMenu } from './FloatingMenu';
import { DictationControl } from './DictationControl';
import { Btn, Dialog, Icon, ErrorNotice } from './ui';
import { MOD } from '../lib/commands';
import { setChatPreferences, useStore } from '../lib/store';
import { artifactFromUrl, type PreviewArtifact } from '../lib/artifacts';
import type { ChatMode, Session } from '../lib/types';
import { attachChatPhoto } from '../lib/chat-photos';
import { useNative } from '../lib/native';
import { pictureStatus } from '../lib/pictures';
const modes: {
    value: ChatMode;
    label: string;
    hint: string;
}[] = [
    { value: 'manual', label: 'Manual', hint: 'Ask before making changes' },
    { value: 'plan', label: 'Plan', hint: 'Create a plan before making changes' },
    { value: 'bypass', label: 'Bypass permissions', hint: 'Skip permission prompts' },
];
const efforts = ['Off', 'Low', 'Medium', 'High', 'Max'];
/** The most tokens a reply may spend thinking (the server's reasoning_budget_tokens); 0 = no limit. */
const THINKING_BUDGETS: [number, string][] = [[0, 'No limit'], [2048, '2K'], [8192, '8K'], [32768, '32K']];
const textTypes = /\.(txt|md|markdown|csv|tsx?|jsx?|py|json|css|rs|go|sh|html?|ya?ml|toml|xml|log)$/i;
const photoTypes = /\.(png|jpe?g|webp|gif|svg)$/i;
export function ComposerControls({ session, onPreview, onAttach, onError, onSlash, streaming, draft, onDraft }: {
    draft: string;
    onDraft: (text: string) => void;
    session: Session | null;
    onPreview: (a: PreviewArtifact) => void;
    streaming: boolean;
    onAttach: (name: string, text: string) => void;
    onError: (error: string) => void;
    onSlash: () => void;
}) {
    const remoteModel = useStore(s => s.remoteModel);
    const native = useNative();
    const nativeModel = native?.instances.find(i => i.id === native.defaultId);
    const [toolsOpen, setToolsOpen] = useState(false);
    const library = useStore(s => s.library);
    const loaded = library?.variants.find(v => v.id === library.loaded);
    const remote = useStore(s => s.remoteModel), providers = useStore(s => s.providers);
    const apiModel = providers?.connections.find(c => c.id === remote?.connectionId)?.models.find(m => m.id === remote?.modelId);
    const [menu, setMenu] = useState(false), [preferences, setPreferences] = useState<'mode' | 'thinking' | null>(null), [reading, setReading] = useState(false);
    const [urlOpen, setUrlOpen] = useState(false), [url, setUrl] = useState(''), [error, setError] = useState('');
    const files = useRef<HTMLInputElement>(null), folder = useRef<HTMLInputElement>(null), previewFile = useRef<HTMLInputElement>(null);
    const trigger = useRef<HTMLButtonElement>(null), modeTrigger = useRef<HTMLButtonElement>(null), thinkingTrigger = useRef<HTMLButtonElement>(null);
    const attachLatest = useRef(onAttach);
    attachLatest.current = onAttach;
    const addFilesLatest = useRef<(files: File[]) => Promise<void>>(async () => { });
    // A model on an API takes a reasoning effort when its provider says it can think (OpenRouter's model list); otherwise the provider decides.
    const apiOnlyDefault = !!apiModel && apiModel.thinks !== true;
    const pictures = pictureStatus(native, remoteModel, providers);
    const mode = modes.find(m => m.value === session?.preferences?.mode) ?? modes[0];
    const effort = efforts.includes(session?.preferences?.thinking ?? '') ? session!.preferences!.thinking! : 'Medium';
    const budget = session?.preferences?.thinkingBudget ?? 0;
    const shortcut = MOD === '⌘' ? '⌘⇧A' : 'Ctrl+Shift+A';
    async function openFile(selected: File | undefined) {
        if (!selected)
            return;
        if (selected.size > 20 * 1024 * 1024) {
            onError('Choose a file smaller than 20 MB.');
            return;
        }
        const isPdf = /\.pdf$/i.test(selected.name), isImage = photoTypes.test(selected.name), isTable = /\.csv$/i.test(selected.name), isCode = /\.(tsx?|jsx?|py|json|css|rs|go|sh)$/i.test(selected.name), isHtml = /\.html?$/i.test(selected.name);
        try {
            onPreview({ id: crypto.randomUUID(), kind: isPdf ? 'pdf' : isImage ? 'image' : isTable ? 'table' : isCode ? 'code' : isHtml ? 'website' : 'markdown', title: selected.name, ...(isPdf || isImage ? { url: URL.createObjectURL(selected) } : { content: await selected.text() }) });
        }
        catch {
            onError('Could not read the file.');
        }
    }
    async function addFiles(selected: File[], directory = false) {
        if (!selected.length)
            return;
        setReading(true);
        onError('');
        const texts: string[] = [];
        let bytes = 0;
        let skipped = 0;
        try {
            for (const file of selected.slice(0, 64)) {
                const path = file.webkitRelativePath || file.name;
                if (directory && /(^|\/)(node_modules|\.git|\.strata)(\/|$)/.test(path)) {
                    skipped++;
                    continue;
                }
                if (!directory && photoTypes.test(file.name) && pictures.ok) {
                    await attachChatPhoto(file, session?.id);
                    continue;
                }
                if (!directory && (photoTypes.test(file.name) || /\.pdf$/i.test(file.name))) {
                    await openFile(file);
                    if (photoTypes.test(file.name))
                        onError(`${pictures.ok ? '' : tr(pictures.message)} ${tr('It is open in preview.')}`);
                    continue;
                }
                if (!textTypes.test(file.name) || bytes + file.size > 256 * 1024) {
                    skipped++;
                    continue;
                }
                const text = await file.text();
                if (text.includes('\u0000')) {
                    skipped++;
                    continue;
                }
                bytes += file.size;
                texts.push(`File: ${path}\n\n${text}`);
            }
            if (texts.length)
                attachLatest.current(directory ? (selected[0].webkitRelativePath.split('/')[0] || 'Folder') : selected.filter(f => textTypes.test(f.name)).map(f => f.name).join(', '), texts.join('\n\n'));
            skipped += Math.max(0, selected.length - 64);
            if (skipped)
                onError(`${skipped} file${skipped === 1 ? ' was' : 's were'} skipped. Add up to 64 text files, with 256 KB of text in total. Photos and PDFs can be opened separately.`);
        }
        catch {
            onError('Some files could not be read. Try adding them separately.');
        }
        finally {
            setReading(false);
        }
    }
    addFilesLatest.current = files => addFiles(files);
    useEffect(() => {
        folder.current?.setAttribute('webkitdirectory', '');
        const key = (e: KeyboardEvent) => {
            if (!e.defaultPrevented && (e.metaKey || e.ctrlKey) && e.shiftKey && !e.altKey && e.key.toLowerCase() === 'a' && !document.querySelector('.dialog-layer') && !streaming && !reading) {
                e.preventDefault();
                files.current?.click();
            }
        };
        const website = () => { setUrlOpen(true); setError(''); };
        const preview = () => previewFile.current?.click();
        const dropped = (e: Event) => { if (!streaming && !reading) void addFilesLatest.current((e as CustomEvent<File[]>).detail ?? []); };
        window.addEventListener('strata:drop-files', dropped);
        window.addEventListener('keydown', key);
        window.addEventListener('strata:open-website', website);
        window.addEventListener('strata:open-preview', preview);
        return () => { window.removeEventListener('strata:drop-files', dropped); window.removeEventListener('keydown', key); window.removeEventListener('strata:open-website', website); window.removeEventListener('strata:open-preview', preview); };
    }, [streaming, reading]);
    useEffect(() => {
        if (preferences !== 'mode')
            return;
        const key = (e: KeyboardEvent) => {
            const i = Number(e.key) - 1;
            if (i >= 0 && i < modes.length && !e.metaKey && !e.ctrlKey && !e.altKey) {
                e.preventDefault();
                setChatPreferences(session?.id, { mode: modes[i].value });
                setPreferences(null);
            }
        };
        window.addEventListener('keydown', key);
        return () => window.removeEventListener('keydown', key);
    }, [preferences, session?.id]);
    return <>
    {tr(!remoteModel && nativeModel?.metrics?.mcp?.servers?.length ? <button className="composer-choice" aria-label={tr("Use connected MCP tools")} aria-pressed={session?.preferences?.useTools === true} onClick={() => session?.preferences?.useTools ? setChatPreferences(session.id, { useTools: false }) : setToolsOpen(true)}><Icon name="skills" size={16}/>{tr("Tools")}{tr(session?.preferences?.useTools ? ' on' : '')}</button> : null)}
    <Dialog open={toolsOpen} compact title={tr("Enable connected tools?")} onClose={() => setToolsOpen(false)} actions={<><Btn onClick={() => setToolsOpen(false)}>{tr("Keep tools off")}</Btn><Btn variant="primary" onClick={() => { setChatPreferences(session?.id, { useTools: true, mode: 'bypass' }); setToolsOpen(false); }}>{tr("Enable tools")}</Btn></>}><p>{tr("Connected servers can act on the resources you configured. Calls run automatically when the model chooses them, and their results appear in chat.")}</p><p className="small muted">{tr("This switches this conversation to Bypass permissions. Servers: ")}{tr((nativeModel?.metrics?.mcp?.servers ?? []).map((s: any) => s.name).join(', '))}.</p></Dialog>
    <input ref={files} type="file" multiple hidden accept="image/*,.pdf,.txt,.md,.markdown,.csv,.ts,.tsx,.js,.jsx,.py,.json,.css,.rs,.go,.sh,.html,.htm,.yaml,.yml,.toml,.xml,.log" onChange={e => { void addFiles(Array.from(e.target.files ?? [])); e.target.value = ''; }}/>
    <input ref={folder} type="file" multiple hidden onChange={e => { void addFiles(Array.from(e.target.files ?? []), true); e.target.value = ''; }}/>
    <input ref={previewFile} type="file" hidden onChange={e => { void openFile(e.target.files?.[0]); e.target.value = ''; }}/>
    <button ref={trigger} className="composer-add" aria-label={tr("Add or open")} title={tr("Add or open")} disabled={reading || streaming} aria-haspopup="menu" aria-expanded={menu} onClick={() => { setPreferences(null); setMenu(v => !v); }}><Icon name="plus" size={22}/></button>
    <DictationControl draft={draft} onDraft={onDraft} sessionId={session?.id} disabled={streaming}/>
    <button ref={modeTrigger} className="composer-choice composer-mode" aria-label={tr("Mode: {0}", [tr(mode.label)])} aria-haspopup="menu" aria-expanded={preferences === 'mode'} onClick={() => { setMenu(false); setPreferences(v => v === 'mode' ? null : 'mode'); }}><span>{tr(mode.label)}</span><Icon name="chevron" size={13}/></button>
    <span className="spacer"/>
    <button ref={thinkingTrigger} className="composer-choice composer-effort" disabled={apiOnlyDefault} title={tr(apiOnlyDefault ? "Reasoning uses the API provider’s defaults." : undefined)} aria-label={tr("Reasoning effort: {0}", [apiOnlyDefault ? tr('Provider default') : effort])} aria-haspopup="menu" aria-expanded={preferences === 'thinking'} onClick={() => { setMenu(false); setPreferences(v => v === 'thinking' ? null : 'thinking'); }}><Icon name="thinking" size={16}/><span>{tr(apiOnlyDefault ? 'Default' : effort)}</span><Icon name="chevron" size={13}/></button>
    {tr(menu ? <FloatingMenu anchor={trigger.current} title={tr("Add or open")} width={292} onClose={() => setMenu(false)}>
      <button role="menuitem" className="menu__item" onClick={() => { setMenu(false); files.current?.click(); }}><Icon name="attach"/><span>{tr("Add files or photos")}</span><kbd className="menu-shortcut">{tr(shortcut)}</kbd></button>
      <button role="menuitem" className="menu__item" onClick={() => { setMenu(false); folder.current?.click(); }}><Icon name="folder-plus"/>{tr("Add folder")}</button>
      <button role="menuitem" className="menu__item" onClick={() => { setMenu(false); onSlash(); }}><span className="slash-command-icon" aria-hidden="true">/</span>{tr("Slash commands")}</button>
    </FloatingMenu> : null)}
    {tr(preferences ? <FloatingMenu anchor={preferences === 'mode' ? modeTrigger.current : thinkingTrigger.current} title={tr(preferences === 'mode' ? 'Mode' : 'Reasoning effort')} width={preferences === 'mode' ? 330 : 280} onClose={() => setPreferences(null)}>
      {tr(preferences === 'mode' ? <>{tr(modes.map((option, index) => <button key={option.value} role="menuitemradio" aria-checked={mode.value === option.value} className="menu__item composer-mode-option" onClick={() => { setChatPreferences(session?.id, { mode: option.value }); setPreferences(null); }}><span><strong>{tr(option.label)}</strong><small>{tr(option.hint)}</small></span>{tr(mode.value === option.value ? <Icon name="check" size={16}/> : null)}<kbd className="menu-shortcut">{tr(index + 1)}</kbd></button>))}</> : <><div className="reasoning-menu-model"><Icon name={apiModel ? 'link' : modelFamilyIcon(loaded?.family ?? nativeModel?.label)} size={18}/><strong>{tr(apiModel?.name ?? (nativeModel && nativeModel.status === 'ready' ? nativeModel.label : null) ?? loaded?.family ?? 'No model loaded')}</strong></div><p className="composer-menu-heading">{tr("Reasoning effort")}</p>{tr(efforts.map(option => <button key={option} role="menuitemradio" aria-checked={effort === option} className="menu__item" onClick={() => { setChatPreferences(session?.id, { thinking: option }); setPreferences(null); }}>{tr(option)}<span className="spacer"/>{tr(effort === option ? <Icon name="check" size={16}/> : null)}</button>))}{tr(apiModel ? null : <><p className="composer-menu-heading">{tr("Thinking budget")}</p>{tr(THINKING_BUDGETS.map(([value, label]) => <button key={value} role="menuitemradio" aria-checked={budget === value} disabled={effort === 'Off'} className="menu__item" onClick={() => { setChatPreferences(session?.id, { thinkingBudget: value }); setPreferences(null); }}>{tr(label)}<span className="spacer"/>{tr(budget === value ? <Icon name="check" size={16}/> : null)}</button>))}</>)}</>)}
    </FloatingMenu> : null)}

    <Dialog open={urlOpen} title={tr("Open a website")} onClose={() => setUrlOpen(false)} actions={<><Btn onClick={() => setUrlOpen(false)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!url.trim()} onClick={() => {
                const a = artifactFromUrl(url.trim());
                if (!a || !/^https?:\/\//i.test(url.trim())) {
                    setError('Enter a full http:// or https:// address.');
                    return;
                }
                onPreview(a);
                setUrlOpen(false);
                setUrl('');
            }}>{tr("Open preview")}</Btn></>}><div className="form-stack"><label className="input-field"><span>{tr("Website address")}</span><input className="st-input" type="url" aria-invalid={!!error} value={url} onChange={e => { setUrl(e.target.value); setError(''); }} placeholder={tr("https://example.com")}/></label>{tr(error ? <ErrorNotice message={error}/> : null)}</div></Dialog>
  </>;
}
