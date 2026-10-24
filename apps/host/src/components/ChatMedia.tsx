import { tr } from '../lib/i18n';
import { useState } from 'react';
import { Btn } from './ui';
import { safePreviewUrl, type PreviewArtifact } from '../lib/artifacts';
import type { ChatMessage } from '../lib/types';
export function ChatMedia({ items, onPreview }: {
    items: NonNullable<ChatMessage['media']>;
    onPreview: (a: PreviewArtifact) => void;
}) {
    return <div className="chat-media">{items.map(m => <Media key={m.src} item={m} onPreview={onPreview}/>)}</div>;
}
function Media({ item: m, onPreview }: {
    item: NonNullable<ChatMessage['media']>[number];
    onPreview: (a: PreviewArtifact) => void;
}) {
    const [playing, setPlaying] = useState(() => !matchMedia('(prefers-reduced-motion: reduce)').matches), [error, setError] = useState(false);
    const src = safePreviewUrl(m.src), poster = m.poster ? safePreviewUrl(m.poster) : null;
    if (!src)
        return null;
    return <figure className="chat-media__item"><div className="chat-media__head"><span>{tr(m.kind === 'gif' ? 'Animated GIF' : m.kind === 'video' ? 'Video' : 'Image')}</span><span className="muted">{m.title}</span></div>{error ? <p className="preview-placeholder" role="alert">{tr("This media could not load.")}</p> : m.kind === 'video' ? <video src={src} poster={poster ?? undefined} controls playsInline preload="metadata" aria-label={m.title} onError={() => setError(true)}/> : <button className="chat-media__image" aria-label={tr("Open {0} in sidebar", [m.title])} onClick={() => onPreview({ id: m.src, kind: 'image', url: src, title: m.title, demo: true })}><img src={m.kind === 'gif' && !playing && poster ? poster : src} alt={m.caption ?? m.title} loading="lazy" onError={() => setError(true)}/></button>}<figcaption><span>{m.caption}</span>{tr(m.kind === 'gif' && poster ? <Btn size="sm" variant="ghost" onClick={() => setPlaying(v => !v)}>{tr(playing ? 'Pause animation' : 'Play animation')}</Btn> : null)}</figcaption></figure>;
}
