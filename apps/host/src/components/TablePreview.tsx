import { tr } from '../lib/i18n';
import { useMemo, useState } from 'react';
/** Handles quoted fields and embedded newlines without evaluating cell content. */
export function parseCsv(text: string): string[][] {
    const rows: string[][] = [];
    let row: string[] = [], cell = '', quoted = false;
    for (let i = 0; i < text.length; i++) {
        const c = text[i];
        if (c === '"') {
            if (quoted && text[i + 1] === '"') {
                cell += '"';
                i++;
            }
            else
                quoted = !quoted;
        }
        else if (!quoted && (c === ',' || c === '\n')) {
            row.push(cell.replace(/\r$/, ''));
            cell = '';
            if (c === '\n') {
                rows.push(row);
                row = [];
            }
        }
        else
            cell += c;
    }
    if (cell || row.length) {
        row.push(cell.replace(/\r$/, ''));
        rows.push(row);
    }
    return rows;
}
export function TablePreview({ text }: {
    text: string;
}) {
    const [query, setQuery] = useState(''), [sort, setSort] = useState<{
        column: number;
        desc: boolean;
    } | null>(null);
    const [headers, ...rows] = useMemo(() => parseCsv(text), [text]);
    const filtered = rows.filter(r => r.join(' ').toLowerCase().includes(query.toLowerCase()));
    if (sort)
        filtered.sort((a, b) => (a[sort.column] ?? '').localeCompare(b[sort.column] ?? '', undefined, { numeric: true }) * (sort.desc ? -1 : 1));
    return <div className="table-preview"><div className="section-toolbar"><span className="small muted">{tr(filtered.length)}{tr(" rows \u00B7 ")}{tr(headers?.length ?? 0)}{tr(" columns")}</span><input className="st-input" type="search" aria-label={tr("Filter table")} placeholder={tr("Filter rows…")} value={query} onChange={e => setQuery(e.target.value)}/></div><div className="st-tablewrap"><table className="st-table"><thead><tr>{tr(headers?.map((h, i) => <th key={i} aria-sort={sort?.column === i ? sort.desc ? 'descending' : 'ascending' : 'none'}><button className="table-sort" onClick={() => setSort({ column: i, desc: sort?.column === i ? !sort.desc : false })}>{h}{tr(sort?.column === i ? sort.desc ? ' ↓' : ' ↑' : '')}</button></th>))}</tr></thead><tbody>{tr(filtered.slice(0, 1000).map((r, i) => <tr key={i}>{tr(headers?.map((_, j) => <td key={j}>{r[j] ?? ''}</td>))}</tr>))}</tbody></table></div>{tr(filtered.length > 1000 ? <p className="xs muted">{tr("Showing the first 1,000 matching rows.")}</p> : null)}</div>;
}
