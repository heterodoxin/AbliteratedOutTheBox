import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import type { PDFDocumentProxy, RenderTask } from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { PreviewControls } from './PreviewControls';
/** Imported only when a PDF is opened. One page renders at a time. */
export default function PdfPreview({ url }: {
    url: string;
}) {
    const [doc, setDoc] = useState<PDFDocumentProxy | null>(null), [page, setPage] = useState(1), [zoom, setZoom] = useState('fit'), [width, setWidth] = useState(600), [error, setError] = useState(''), [loading, setLoading] = useState(true);
    const canvas = useRef<HTMLCanvasElement>(null), container = useRef<HTMLDivElement>(null);
    useEffect(() => {
        let alive = true;
        let task: ReturnType<typeof import('pdfjs-dist').getDocument> | undefined;
        setLoading(true);
        setError('');
        setDoc(null);
        setPage(1);
        void import('pdfjs-dist').then(pdf => {
            if (!alive)
                return;
            pdf.GlobalWorkerOptions.workerSrc = workerUrl;
            task = pdf.getDocument({ url });
            return task.promise;
        }).then(pdf => {
            if (alive && pdf)
                setDoc(pdf);
        }).catch(e => {
            if (alive) {
                setError(e.message);
                setLoading(false);
            }
        });
        return () => { alive = false; void task?.destroy(); };
    }, [url]);
    useEffect(() => {
        if (!container.current)
            return;
        const observer = new ResizeObserver(entries => setWidth(entries[0].contentRect.width));
        observer.observe(container.current);
        return () => observer.disconnect();
    }, []);
    useEffect(() => {
        if (!doc || !canvas.current)
            return;
        let alive = true;
        let task: RenderTask | undefined;
        setLoading(true);
        setError('');
        void doc.getPage(page).then(async (p) => {
            if (!alive || !canvas.current)
                return;
            const viewport = p.getViewport({ scale: 1 });
            const scale = zoom === 'fit' ? Math.max(.2, (width - 48) / viewport.width) : Number(zoom);
            const view = p.getViewport({ scale });
            const ratio = Math.min(devicePixelRatio || 1, 2);
            const el = canvas.current;
            el.width = Math.floor(view.width * ratio);
            el.height = Math.floor(view.height * ratio);
            el.style.width = `${view.width}px`;
            el.style.height = `${view.height}px`;
            task = p.render({ canvas: el, viewport: view, transform: ratio === 1 ? undefined : [ratio, 0, 0, ratio, 0, 0] });
            await task.promise;
            if (alive)
                setLoading(false);
        }).catch(e => {
            if (alive && e.name !== 'RenderingCancelledException') {
                setError(e.message);
                setLoading(false);
            }
        });
        return () => { alive = false; task?.cancel(); };
    }, [doc, page, zoom, width]);
    return <div className="pdf-view"><div className="pdf-controls" role="group" aria-label={tr("PDF controls")}><PreviewControls label={tr("PDF")} zoom={zoom} onZoom={setZoom} page={page} pages={doc?.numPages} onPage={setPage}/></div><div ref={container} className="pdf-canvas-wrap" aria-busy={loading}>{tr(loading ? <p className="small muted pdf-loading" role="status">{tr("Rendering page\u2026")}</p> : null)}{tr(error ? <p className="inline-error" role="alert">{tr("Couldn\u2019t open this PDF. ")}{tr(error)}</p> : <canvas ref={canvas} role="img" aria-label={tr("PDF page {0}", [page])}/>)}</div></div>;
}
