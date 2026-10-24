import { tr } from '../lib/i18n';
import { PreviewControls } from './PreviewControls';
import { lazy, Suspense, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, Icon, IconBtn, Seg, type IconName } from './ui';
import { Markdown } from './Markdown';
import { SyntaxCode } from './SyntaxCode';
import { FloatingMenu } from './FloatingMenu';
import { TablePreview } from './TablePreview';
import { artifactFilename, artifactIcon, safePreviewUrl, type PreviewArtifact } from '../lib/artifacts';
import { scrollBehavior } from '../lib/responsive';
const PdfPreview = lazy(() => import('./PdfPreview'));
export function ArtifactPanel({ items, selected, onSelect, onClose, onCloseTab, inline }: {
    items: PreviewArtifact[];
    selected: string | null;
    onSelect: (id: string) => void;
    onClose: () => void;
    onCloseTab: (id: string) => void;
    inline: boolean;
}) {
    const current = items.find(a => a.id === selected);
    const [text, setText] = useState(''), [error, setError] = useState(''), [mode, setMode] = useState<'preview' | 'source'>('preview'), [revision, setRevision] = useState(0), [loading, setLoading] = useState(false), [imageSize, setImageSize] = useState('fit');
    const [toolbarHost, setToolbarHost] = useState<HTMLElement | null>(null);
    const [fileMenu, setFileMenu] = useState(false);
    const fileMenuRef = useRef<HTMLButtonElement>(null);
    useLayoutEffect(() => { setToolbarHost(inline && current ? document.getElementById('preview-file-actions') : null); }, [inline, current?.id]);
    useEffect(() => {
        setError('');
        setMode('preview');
        setLoading(false);
        setImageSize('fit');
        setFileMenu(false);
        if (!current || !['markdown', 'code', 'table'].includes(current.kind))
            return;
        if (current.content !== undefined) {
            setText(current.content);
            return;
        }
        const url = current.url && safePreviewUrl(current.url);
        if (!url) {
            setError('This file cannot be opened.');
            return;
        }
        const abort = new AbortController();
        setLoading(true);
        fetch(url, { signal: abort.signal, credentials: 'omit' }).then(r => {
            if (!r.ok)
                throw new Error(`File returned ${r.status}`);
            return r.text();
        }).then(content => {
            if (content.length > 500000)
                throw new Error('This text file is too large to preview.');
            if (!abort.signal.aborted)
                setText(content);
        }).catch(e => {
            if (!abort.signal.aborted)
                setError(e.message);
        }).finally(() => {
            if (!abort.signal.aborted)
                setLoading(false);
        });
        return () => abort.abort();
    }, [current?.id, revision]);
    if (!current)
        return null;
    const url = current.url ? safePreviewUrl(current.url) : null;
    const viewControls = current.kind === 'markdown' ? <Seg ariaLabel={tr("Markdown view")} value={mode} options={[{ value: 'preview', label: 'Read' }, { value: 'source', label: 'Source' }]} onChange={setMode}/> : current.kind === 'image' ? <PreviewControls label={tr("Image")} zoom={imageSize} onZoom={setImageSize}/> : null;
    const actions = <><div className="preview-file-actions__inline">{tr(current.kind === 'code' ? <IconBtn name="copy" title={tr("Copy code")} disabled={loading || !!error} onClick={() => void navigator.clipboard.writeText(text)}/> : null)}{tr(url ? <a className="st-btn st-btn--icon" href={url} target="_blank" rel="noreferrer" aria-label={tr("Open outside preview")} title={tr("Open outside preview")}><Icon name="link" size={17}/></a> : null)}<IconBtn name="refresh" title={tr("Reload preview")} onClick={() => setRevision(v => v + 1)}/></div><button ref={fileMenuRef} className="st-btn st-btn--icon preview-file-actions__more" aria-label={tr("Preview file actions")} title={tr("Preview file actions")} aria-haspopup="menu" aria-expanded={fileMenu} onClick={() => setFileMenu(v => !v)}><Icon name="ellipsis"/></button></>;
    const actionMenu = fileMenu ? <FloatingMenu anchor={fileMenuRef.current} title={tr("Preview file actions")} onClose={() => setFileMenu(false)}>{tr(current.kind === 'code' ? <button role="menuitem" className="menu__item" disabled={loading || !!error} onClick={() => { void navigator.clipboard.writeText(text); setFileMenu(false); }}><Icon name="copy"/>{tr("Copy code")}</button> : null)}{tr(url ? <a role="menuitem" className="menu__item" href={url} target="_blank" rel="noreferrer" onClick={() => setFileMenu(false)}><Icon name="link"/>{tr("Open outside preview")}</a> : null)}<button role="menuitem" className="menu__item" onClick={() => { setRevision(v => v + 1); setFileMenu(false); }}><Icon name="refresh"/>{tr("Reload preview")}</button></FloatingMenu> : null;
    const content = <aside className="artifact-panel" aria-label={tr("{0} preview", [current.kind])}>
    {tr(!inline ? <ArtifactTabs onCloseTab={onCloseTab} items={items} selected={selected} onSelect={onSelect}/> : null)}
    <div id="artifact-content" role="tabpanel" aria-labelledby={`artifact-tab-${items.findIndex(a => a.id === selected)}`} className="artifact-content">
      {tr(current.kind === 'website' ? <><iframe key={`${current.id}-${revision}`} title={current.title} src={current.content ? undefined : url ?? undefined} srcDoc={current.content} sandbox="allow-scripts" referrerPolicy="no-referrer"/></> : current.kind === 'pdf' ? url ? <Suspense fallback={<p className="muted preview-placeholder">{tr("Opening PDF viewer\u2026")}</p>}><PdfPreview key={`${current.id}-${revision}`} url={url}/></Suspense> : <p>{tr("No PDF to display.")}</p> : current.kind === 'image' ? <div className="image-preview" data-size={imageSize === '1' ? 'actual' : 'fit'}>{tr(url ? <img key={`${current.id}-${revision}`} src={url} alt={current.title} onError={() => setError('Couldn’t load this image.')}/> : null)}{tr(error ? <p role="alert">{tr(error)}</p> : null)}</div> : loading ? <p className="muted preview-placeholder">{tr("Loading document\u2026")}</p> : error ? <p role="alert" className="inline-error preview-placeholder">{tr(error)}</p> : current.kind === 'code' ? <SyntaxCode text={text} filename={artifactFilename(current)} lineNumbers/> : current.kind === 'table' ? <TablePreview key={current.id} text={text}/> : mode === 'source' ? <SyntaxCode text={text} language="markdown" lineNumbers/> : <article className="document-markdown"><Markdown text={text}/></article>)}
    </div>
    {tr(viewControls ? <div className="artifact-view-controls">{tr(viewControls)}</div> : null)}
  </aside>;
    return inline ? <>{tr(toolbarHost ? createPortal(actions, toolbarHost) : null)}{tr(content)}{tr(actionMenu)}</> : <Dialog open title={tr("Preview")} onClose={onClose} headerActions={actions} sheet hideTitle>{tr(content)}{tr(actionMenu)}</Dialog>;
}
export interface ExtraPreviewTab {
    id: string;
    title: string;
    icon: IconName;
    controls: string;
}
export function ArtifactTabs({ items, selected, onSelect, onCloseTab, extraTabs = [] }: {
    items: PreviewArtifact[];
    selected: string | null;
    onSelect: (id: string) => void;
    onCloseTab: (id: string) => void;
    extraTabs?: ExtraPreviewTab[];
}) {
    const tabs = [...items.map(a => ({ id: a.id, title: artifactFilename(a), icon: artifactIcon(a), controls: 'artifact-content' })), ...extraTabs];
    const ref = useRef<HTMLDivElement>(null), focusAfterClose = useRef(false);
    const [edges, setEdges] = useState({ overflow: false, left: false, right: false });
    const tabIds = tabs.map(t => t.id).join('|');
    const revealSelected = useCallback(() => {
        const strip = ref.current, tab = strip?.querySelector<HTMLButtonElement>('[aria-selected="true"]');
        if (!strip || !tab)
            return;
        const r = tab.parentElement!.getBoundingClientRect(), box = strip.getBoundingClientRect();
        if (r.left < box.left)
            strip.scrollLeft -= box.left - r.left + 4;
        else if (r.right > box.right)
            strip.scrollLeft += r.right - box.right + 4;
    }, []);
    const measure = useCallback(() => {
        const strip = ref.current;
        if (!strip)
            return;
        const max = strip.scrollWidth - strip.clientWidth;
        const next = { overflow: max > 1, left: strip.scrollLeft > 1, right: strip.scrollLeft < max - 1 };
        setEdges(old => old.overflow === next.overflow && old.left === next.left && old.right === next.right ? old : next);
    }, []);
    useLayoutEffect(() => {
        const strip = ref.current;
        if (!strip)
            return;
        const observer = new ResizeObserver(() => { measure(); revealSelected(); });
        observer.observe(strip);
        for (const tab of strip.children)
            observer.observe(tab);
        measure();
        return () => observer.disconnect();
    }, [tabIds, measure, revealSelected]);
    useEffect(() => {
        const tab = ref.current?.querySelector<HTMLButtonElement>('[aria-selected="true"]');
        revealSelected();
        if (focusAfterClose.current) {
            tab?.focus({ preventScroll: true });
            focusAfterClose.current = false;
        }
        measure();
    }, [selected, tabIds, edges.overflow, measure, revealSelected]);
    function scroll(direction: number) {
        const strip = ref.current;
        if (strip)
            strip.scrollBy({ left: direction * Math.max(120, strip.clientWidth * .8), behavior: scrollBehavior() });
    }
    function close(id: string) { focusAfterClose.current = true; onCloseTab(id); }
    return <div className="artifact-tab-strip">{tr(edges.overflow ? <button type="button" className="st-btn st-btn--icon artifact-scroll artifact-scroll--left" aria-label={tr("Scroll preview tabs left")} title={tr("Scroll preview tabs left")} disabled={!edges.left} onClick={() => scroll(-1)}><Icon name="chevron" size={16}/></button> : null)}<div ref={ref} className="artifact-tabs" role="tablist" aria-label={tr("Open previews")} onScroll={measure}>{tr(tabs.map((a, i) => <div className="artifact-tab" role="presentation" data-selected={a.id === selected || undefined} key={a.id}>
    <button className="artifact-tab__select" role="tab" title={a.title} aria-selected={a.id === selected} aria-controls={a.controls} id={`artifact-tab-${i}`} tabIndex={a.id === selected ? 0 : -1} onClick={() => onSelect(a.id)} onKeyDown={e => {
                if (e.key === 'Delete') {
                    e.preventDefault();
                    close(a.id);
                    return;
                }
                if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
                    e.preventDefault();
                    const next = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
                    onSelect(tabs[next].id);
                    ref.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
                }
            }}><Icon name={a.icon} size={15}/><span>{a.title}</span></button>
    <button className="artifact-tab__close" aria-label={tr("Close {0}", [a.title])} title={tr("Close {0}", [a.title])} tabIndex={a.id === selected ? 0 : -1} onClick={() => close(a.id)}><Icon name="x" size={13}/></button>
  </div>))}</div>{tr(edges.overflow ? <button type="button" className="st-btn st-btn--icon artifact-scroll artifact-scroll--right" aria-label={tr("Scroll preview tabs right")} title={tr("Scroll preview tabs right")} disabled={!edges.right} onClick={() => scroll(1)}><Icon name="chevron" size={16}/></button> : null)}</div>;
}
