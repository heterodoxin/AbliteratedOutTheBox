import { tr } from '../lib/i18n';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
export function VersionMemoryTooltip({ id, anchor, text, onClose, onEnter, onLeave }: {
    id: string;
    anchor: HTMLElement;
    text: string;
    onClose: () => void;
    onEnter: () => void;
    onLeave: () => void;
}) {
    const ref = useRef<HTMLDivElement>(null);
    const [position, setPosition] = useState({ left: 12, top: 12 });
    useLayoutEffect(() => {
        const rect = anchor.getBoundingClientRect(), width = Math.min(260, window.innerWidth - 24);
        const height = ref.current?.getBoundingClientRect().height ?? 80;
        const left = Math.max(12, Math.min(rect.right - width, window.innerWidth - width - 12));
        const top = rect.bottom + height + 20 <= window.innerHeight ? rect.bottom + 8 : Math.max(12, rect.top - height - 8);
        setPosition({ left, top });
    }, [anchor, text]);
    useEffect(() => {
        const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') {
            event.preventDefault();
            event.stopImmediatePropagation();
            onClose();
        } };
        document.addEventListener('keydown', escape, true);
        window.addEventListener('scroll', onClose, true);
        window.addEventListener('resize', onClose);
        return () => { document.removeEventListener('keydown', escape, true); window.removeEventListener('scroll', onClose, true); window.removeEventListener('resize', onClose); };
    }, [onClose]);
    return createPortal(<div ref={ref} id={id} role="tooltip" className="version-memory-tooltip" style={position} onPointerEnter={onEnter} onPointerLeave={onLeave}>{tr(text)}</div>, document.body);
}
