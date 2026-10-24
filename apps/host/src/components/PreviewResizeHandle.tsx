import { tr } from '../lib/i18n';
import { useEffect, useRef, type RefObject } from 'react';
import { DEFAULT_PREVIEW_FRACTION, pointerPreviewFraction, previewWidth } from '../lib/preview-layout';
export function PreviewResizeHandle({ container, total, fraction, onChange, controls }: {
    container: RefObject<HTMLDivElement | null>;
    total: number;
    fraction: number;
    onChange: (value: number) => void;
    controls: string;
}) {
    const pointer = useRef<number | null>(null);
    const width = previewWidth(total, fraction), min = previewWidth(total, 0), max = previewWidth(total, 1);
    const cleanup = () => { pointer.current = null; document.documentElement.removeAttribute('data-preview-resizing'); };
    useEffect(() => cleanup, []);
    return <div className="preview-resize-handle" role="separator" tabIndex={0} aria-label={tr("Resize preview sidebar")} aria-orientation="vertical" aria-controls={controls} aria-valuemin={min} aria-valuemax={max} aria-valuenow={width} aria-valuetext={`Preview width ${width} pixels`} title={tr("Drag to resize · arrow keys adjust · double-click to reset")} style={{ right: width - 6 }} onPointerDown={e => { if (e.button !== 0)
        return; e.preventDefault(); e.currentTarget.focus({ preventScroll: true }); pointer.current = e.pointerId; e.currentTarget.setPointerCapture(e.pointerId); document.documentElement.setAttribute('data-preview-resizing', 'true'); }} onPointerMove={e => { if (pointer.current !== e.pointerId || !container.current)
        return; const box = container.current.getBoundingClientRect(); onChange(pointerPreviewFraction(box.width, box.right, e.clientX)); }} onPointerUp={e => { if (pointer.current === e.pointerId) {
        if (e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.releasePointerCapture(e.pointerId);
        cleanup();
    } }} onPointerCancel={cleanup} onLostPointerCapture={cleanup} onDoubleClick={() => onChange(DEFAULT_PREVIEW_FRACTION)} onKeyDown={e => { let next: number | undefined; if (e.key === 'ArrowLeft')
        next = (width + (e.shiftKey ? 64 : 24)) / total;
    else if (e.key === 'ArrowRight')
        next = (width - (e.shiftKey ? 64 : 24)) / total;
    else if (e.key === 'Home')
        next = 0;
    else if (e.key === 'End')
        next = 1;
    else if (e.key === 'Enter')
        next = DEFAULT_PREVIEW_FRACTION; if (next !== undefined) {
        e.preventDefault();
        onChange(previewWidth(total, next) / total);
    } }}><span /></div>;
}
