import { tr } from '../lib/i18n';
import { IconBtn, Seg } from './ui';
export function PreviewControls({ zoom, onZoom, page, pages, onPage, label }: {
    zoom: string;
    onZoom: (v: string) => void;
    page?: number;
    pages?: number;
    onPage?: (n: number) => void;
    label: string;
}) { return <div className="preview-controls">{tr(page !== undefined && onPage ? <><IconBtn name="chevron" title={tr("Previous page")} disabled={page <= 1} onClick={() => onPage(page - 1)}/><span className="small num">{tr(page)} / {tr(pages ?? '—')}</span><IconBtn name="chevron" title={tr("Next page")} disabled={!pages || page >= pages} onClick={() => onPage(page + 1)}/></> : null)}<Seg ariaLabel={tr("{0} zoom", [label])} value={zoom} onChange={onZoom} options={[{ value: 'fit', label: 'Fit' }, { value: '1', label: '100%' }]}/></div>; }
