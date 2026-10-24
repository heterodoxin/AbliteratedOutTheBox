import { tr } from '../lib/i18n';
import { useRef, useState } from 'react';
import { FloatingMenu } from './FloatingMenu';
import { Icon } from './ui';
import { artifactIcon, artifactLabel, safePreviewUrl, type PreviewArtifact } from '../lib/artifacts';
import { api } from '../lib/client';
export function ArtifactCard({ artifact: a, onPreview }: {
    artifact: PreviewArtifact;
    onPreview: (a: PreviewArtifact) => void;
}) {
    const [menu, setMenu] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState('');
    const trigger = useRef<HTMLButtonElement>(null);
    const url = a.url ? safePreviewUrl(a.url) : null;
    const localPath = url && new URL(url).origin === location.origin && new URL(url).pathname.startsWith('/examples/') ? new URL(url).pathname : null;
    const canReveal = !!localPath || a.content !== undefined || !!url?.startsWith('blob:');
    async function open(action: 'open' | 'reveal') {
        setMenu(false);
        setError('');
        if (!canReveal && url && action === 'open') {
            window.open(url, '_blank', 'noopener,noreferrer');
            return;
        }
        setBusy(true);
        try {
            const body: Record<string, string> = { action, name: a.title };
            if (localPath)
                body.path = localPath;
            else if (a.content !== undefined)
                body.content = a.content;
            else if (url?.startsWith('blob:')) {
                const blob = await (await fetch(url)).blob();
                body.base64 = await new Promise<string>((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(String(r.result).split(',')[1]); r.onerror = () => reject(new Error('Could not read this file.')); r.readAsDataURL(blob); });
            }
            else
                throw new Error('No local file is available.');
            await api.post('/api/artifacts/open', body);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
        }
    }
    return <div className="artifact-card-wrap"><div className="artifact-card"><button className="artifact-card__open" onClick={() => onPreview(a)} aria-label={tr("Open {0} in sidebar", [a.title])}><span className="artifact-card__icon"><Icon name={artifactIcon(a)} size={22}/></span><span><strong>{a.title}</strong><small>{tr(artifactLabel(a.kind))}{tr(" \u00B7 Open in sidebar")}</small></span></button><button ref={trigger} className="st-btn st-btn--icon" aria-label={tr("Options for {0}", [a.title])} title={tr("Options for {0}", [a.title])} aria-haspopup="menu" aria-expanded={menu} disabled={busy} onClick={() => setMenu(v => !v)}><Icon name="ellipsis" size={16}/></button></div>{tr(error ? <p className="xs inline-error" role="alert">{tr(error)}</p> : null)}{tr(menu ? <FloatingMenu anchor={trigger.current} title={tr("Open {0}", [a.title])} onClose={() => setMenu(false)}><button role="menuitem" className="menu__item" onClick={() => { setMenu(false); onPreview(a); }}><Icon name="panel"/>{tr("Open in sidebar")}</button><button role="menuitem" className="menu__item" disabled={!canReveal} title={tr(!canReveal ? 'This website has no local file' : undefined)} onClick={() => void open('reveal')}><Icon name="folder"/>{tr("Show in file explorer")}</button><button role="menuitem" className="menu__item" onClick={() => void open('open')}><Icon name="computer"/>{tr("Open in default app")}</button>{tr(!canReveal ? <p className="composer-menu-note xs muted">{tr("Websites open in your default browser.")}</p> : a.content !== undefined || a.url?.startsWith('blob:') ? <p className="composer-menu-note xs muted">{tr("Opens a local copy of this preview.")}</p> : null)}</FloatingMenu> : null)}</div>;
}
