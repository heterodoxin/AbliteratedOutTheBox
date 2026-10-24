import { tr } from '../lib/i18n';
/** Safe, dependency-free rendering of the Markdown structures used in answers. */
import { Fragment, type ReactNode } from 'react';
import { Btn, IconBtn } from './ui';
import { SyntaxCode } from './SyntaxCode';
import { MathView } from './MathView';
import { displayBlock, hasMath, splitMath } from '../lib/math';
import { imagePlan } from '../lib/markdown-images';
function inlineMarks(text: string, onOpenLink?: (url: string) => void, imageSession?: string | null): ReactNode {
    return text.split(/(!\[[^\]]*\]\([^\s)]+\)|`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*|\[[^\]]+\]\([^\s)]+\))/g).map((part, i) => {
        const image = part.match(/^!\[([^\]]*)\]\(([^\s)]+)\)$/);
        if (image) {
            const plan = imagePlan(image[2], imageSession);
            if (plan.kind === 'img')
                return <img key={i} src={plan.src} alt={image[1]} className="md-image" loading="lazy" referrerPolicy="no-referrer"/>;
            if (plan.kind === 'link')
                return <a key={i} href={plan.href} target="_blank" rel="noreferrer" onClick={e => { if (onOpenLink) {
                    e.preventDefault();
                    onOpenLink(plan.href);
                } }}>{image[1] || plan.href}</a>;
            return <Fragment key={i}>{image[1] || part}</Fragment>;
        }
        if (part.startsWith('`') && part.endsWith('`'))
            return <code key={i} className="md-code">{part.slice(1, -1)}</code>;
        if (part.startsWith('**') && part.endsWith('**'))
            return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (part.startsWith('*') && part.endsWith('*'))
            return <em key={i}>{part.slice(1, -1)}</em>;
        const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        if (link && /^(https?:\/\/|mailto:|#)/i.test(link[2]))
            return <a key={i} href={link[2]} target="_blank" rel="noreferrer" onClick={e => { if (onOpenLink && /^https?:/i.test(link[2])) {
                e.preventDefault();
                onOpenLink(link[2]);
            } }}>{link[1]}</a>;
        return <Fragment key={i}>{part}</Fragment>;
    });
}
/** Text with its inline marks and, outside code spans, its math (#3268). Math is cut out first: a * or _ inside a formula is not emphasis. */
function inline(text: string, onOpenLink?: (url: string) => void, imageSession?: string | null): ReactNode {
    if (!hasMath(text))
        return inlineMarks(text, onOpenLink, imageSession);
    return text.split(/(`[^`]+`)/g).map((chunk, c) => {
        if (chunk.length > 1 && chunk.startsWith('`') && chunk.endsWith('`'))
            return <Fragment key={c}>{inlineMarks(chunk, onOpenLink, imageSession)}</Fragment>;
        return <Fragment key={c}>{splitMath(chunk).map((part, p) => part.kind === 'math'
            ? <MathView key={p} tex={part.tex} display={part.display}/>
            : <Fragment key={p}>{inlineMarks(part.text, onOpenLink, imageSession)}</Fragment>)}</Fragment>;
    });
}
export function Markdown({ text, onOpenLink, onPreviewHtml, imageSession }: {
    text: string;
    onOpenLink?: (url: string) => void;
    onPreviewHtml?: (html: string) => void;
    /** The chat whose folder a picture path in the answer is read from (the host serves only files inside it). */
    imageSession?: string | null;
}) {
    const nodes: ReactNode[] = [];
    const lines = text.split('\n');
    for (let i = 0; i < lines.length;) {
        const line = lines[i];
        if (!line.trim()) {
            i++;
            continue;
        }
        if (/^\s*```/.test(line)) {
            const language = line.trim().slice(3);
            const body: string[] = [];
            const key = i++;
            while (i < lines.length && !/^\s*```/.test(lines[i]))
                body.push(lines[i++]);
            i++;
            nodes.push(<div className="st-code" key={key}><div className="st-code__head"><span>{language || tr('code')}</span>{tr(/^html$/i.test(language) && onPreviewHtml ? <Btn size="sm" onClick={() => onPreviewHtml(body.join('\n'))}>{tr("Open preview")}</Btn> : null)}<IconBtn name="copy" title={tr("Copy code")} onClick={() => void navigator.clipboard.writeText(body.join('\n'))}/></div><SyntaxCode text={body.join('\n')} language={language}/></div>);
            continue;
        }
        const block = displayBlock(lines, i);
        if (block) {
            nodes.push(<div className="md-math-block" key={i}><MathView tex={block.tex.trim()} display/></div>);
            i += block.used;
            continue;
        }
        const heading = line.match(/^(#{1,6})\s+(.+)$/);
        if (heading) {
            const Tag = heading[1].length <= 2 ? 'h2' : heading[1].length === 3 ? 'h3' : 'h4';
            nodes.push(<Tag key={i}>{tr(inline(heading[2], onOpenLink, imageSession))}</Tag>);
            i++;
            continue;
        }
        if (/^\s*[-*+]\s+\S/.test(line) || /^\s*\d+[.)]\s+\S/.test(line)) {
            const ordered = /^\s*\d+[.)]/.test(line);
            const items: ReactNode[] = [];
            const key = i;
            const re = ordered ? /^\s*\d+[.)]\s+(.+)$/ : /^\s*[-*+]\s+(.+)$/;
            while (i < lines.length && re.test(lines[i])) {
                items.push(<li key={i}>{tr(inline(lines[i].replace(re, '$1'), onOpenLink))}</li>);
                i++;
            }
            nodes.push(ordered ? <ol key={key}>{tr(items)}</ol> : <ul key={key}>{tr(items)}</ul>);
            continue;
        }
        if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
            const cells = (row: string) => row.trim().replace(/^\||\|$/g, '').split('|').map(s => s.trim());
            const header = cells(line);
            const rows: string[][] = [];
            const key = i;
            i += 2;
            while (i < lines.length && lines[i].includes('|'))
                rows.push(cells(lines[i++]));
            nodes.push(<div className="md-table-wrap" key={key}><table className="md-table"><thead><tr>{tr(header.map((c, j) => <th key={j}>{tr(inline(c, onOpenLink, imageSession))}</th>))}</tr></thead><tbody>{tr(rows.map((r, j) => <tr key={j}>{tr(r.map((c, k) => <td key={k}>{tr(inline(c, onOpenLink, imageSession))}</td>))}</tr>))}</tbody></table></div>);
            continue;
        }
        if (/^>\s?/.test(line)) {
            nodes.push(<blockquote key={i}>{tr(inline(line.replace(/^>\s?/, ''), onOpenLink))}</blockquote>);
            i++;
            continue;
        }
        nodes.push(<p key={i}>{tr(inline(line, onOpenLink, imageSession))}</p>);
        i++;
    }
    return <>{tr(nodes)}</>;
}
