import { tr } from '../lib/i18n';
import { useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
export function FloatingMenu({ anchor, title, onClose, children, width = 232 }: {
    anchor: HTMLElement | null;
    title: string;
    onClose: () => void;
    children: React.ReactNode;
    width?: number;
}) {
    const ref = useRef<HTMLDivElement>(null);
    const [placed, setPlaced] = useState({ left: 12, top: 12 });
    useLayoutEffect(() => {
        const rect = anchor?.getBoundingClientRect();
        const height = ref.current?.getBoundingClientRect().height ?? 200;
        const left = Math.max(12, Math.min(rect?.left ?? 12, window.innerWidth - Math.min(width, window.innerWidth - 24) - 12));
        const below = (rect?.bottom ?? 12) + 8; // a little air between the button and its panel
        const top = below + height + 12 <= window.innerHeight ? below : Math.max(12, (rect?.top ?? window.innerHeight) - height - 6);
        setPlaced({ left, top });
        ref.current?.querySelector<HTMLElement>('[role="menuitem"]:not(:disabled), [role="menuitemradio"]:not(:disabled)')?.focus();
        const outside = (e: PointerEvent) => { if (!ref.current?.contains(e.target as Node) && !anchor?.contains(e.target as Node))
            onClose(); };
        document.addEventListener('pointerdown', outside);
        return () => { document.removeEventListener('pointerdown', outside); if (anchor?.isConnected)
            anchor.focus({ preventScroll: true }); };
    }, [anchor, width]);
    return createPortal(<div className="menu" role="menu" aria-label={tr(title)} ref={ref} style={{ ...placed, zIndex: anchor?.closest('[role="dialog"]') ? 'var(--layer-dialog-popover)' : 'var(--layer-popover)', width, maxWidth: 'calc(100dvw - 24px)', maxHeight: `calc(100dvh - ${placed.top + 12}px)`, overflowY: 'auto' }} onKeyDown={e => {
            const nodes = Array.from(ref.current?.querySelectorAll<HTMLElement>('[role="menuitem"]:not(:disabled), [role="menuitemradio"]:not(:disabled)') ?? []);
            const at = nodes.indexOf(document.activeElement as HTMLElement);
            if (['Escape', 'Tab'].includes(e.key)) {
                e.preventDefault();
                e.stopPropagation();
                onClose();
            }
            if (['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) {
                e.preventDefault();
                nodes[e.key === 'Home' ? 0 : e.key === 'End' ? nodes.length - 1 : (at + (e.key === 'ArrowDown' ? 1 : -1) + nodes.length) % nodes.length]?.focus();
            }
        }}>{tr(children)}</div>, document.body);
}
