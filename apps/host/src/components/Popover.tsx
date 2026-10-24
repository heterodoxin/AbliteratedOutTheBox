import { tr } from '../lib/i18n';
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { IconBtn } from './ui';
/** Anchored, non-modal surface. Focus and dismissal stay with its own trigger. */
export function Popover({ anchor, title, detail, onClose, children, above = false, width = 360 }: {
    anchor: HTMLElement | null;
    title: string;
    /** A short line to the right of the title, on the same row (the Subagents list puts its counts here). */
    detail?: ReactNode;
    onClose: () => void;
    children: ReactNode;
    above?: boolean;
    width?: number;
}) {
    const ref = useRef<HTMLDivElement>(null);
    const [position, setPosition] = useState({ left: 12, top: 12 });
    useLayoutEffect(() => {
        const place = () => { const r = anchor?.getBoundingClientRect(); const h = ref.current?.offsetHeight ?? 200; const w = Math.min(width, window.innerWidth - 24); const y = above ? (r?.top ?? h) - h - 10 : (r?.bottom ?? 12) + 8; setPosition({ left: Math.max(12, Math.min(r?.left ?? 12, window.innerWidth - w - 12)), top: Math.max(12, Math.min(y, window.innerHeight - h - 12)) }); };
        place();
        ref.current?.focus({ preventScroll: true });
        const outside = (e: PointerEvent) => {
            if (!ref.current?.contains(e.target as Node) && !anchor?.contains(e.target as Node))
                onClose();
        };
        const resize = new ResizeObserver(place);
        if (ref.current)
            resize.observe(ref.current);
        document.addEventListener('pointerdown', outside);
        window.addEventListener('resize', place);
        return () => { resize.disconnect(); document.removeEventListener('pointerdown', outside); window.removeEventListener('resize', place); };
    }, [anchor, width, above]);
    function close() { onClose(); anchor?.focus({ preventScroll: true }); }
    return createPortal(<div ref={ref} role="dialog" aria-label={tr(title)} tabIndex={-1} className="telepop anchored-popover" style={{ ...position, zIndex: anchor?.closest('[role="dialog"]') ? 'var(--layer-dialog-popover)' : 'var(--layer-popover)', width, maxWidth: 'calc(100vw - 24px)' }} onKeyDown={e => {
            if (e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                close();
            }
        }}><div className="telepop__head"><strong>{tr(title)}</strong>{detail ? <span className="telepop__detail">{detail}</span> : null}<IconBtn name="x" title={tr("Close {0}", [title.toLowerCase()])} onClick={close}/></div>{tr(children)}</div>, document.body);
}
