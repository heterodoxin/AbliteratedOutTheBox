/**
 * One formula of an answer (#3268), drawn by KaTeX. KaTeX and its fonts are bundled with the app (no network) and load as a separate
 * chunk the first time an answer holds math, so a chat without math pays nothing. Until the chunk is there, and if it cannot load,
 * the formula shows as its source text, as before.
 */
import { useEffect, useState } from 'react';

type Katex = typeof import('katex').default;
let loading: Promise<Katex> | null = null;
const drawn = new Map<string, string>();

function loadKatex(): Promise<Katex> {
    loading ??= Promise.all([import('katex'), import('katex/dist/katex.min.css')]).then(([module]) => module.default);
    return loading;
}

function draw(katex: Katex, tex: string, display: boolean): string {
    const key = (display ? 'D' : 'I') + tex;
    const known = drawn.get(key);
    if (known !== undefined)
        return known;
    // trust stays off (no \href, \url, \includegraphics); a broken formula is shown as its source, not as an error page
    const html = katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore', trust: false, maxSize: 30, maxExpand: 500 });
    if (drawn.size > 400)
        drawn.clear();
    drawn.set(key, html);
    return html;
}

const source = (tex: string, display: boolean) => (display ? `$$${tex}$$` : `$${tex}$`);

export function MathView({ tex, display }: { tex: string; display: boolean }) {
    const [html, setHtml] = useState<string | null>(() => drawn.get((display ? 'D' : 'I') + tex) ?? null);
    useEffect(() => {
        let live = true;
        loadKatex().then(katex => { if (live) setHtml(draw(katex, tex, display)); }).catch(() => { /* stays as source text */ });
        return () => { live = false; };
    }, [tex, display]);
    const cls = display ? 'md-math md-math--display' : 'md-math';
    if (html === null)
        return <code className={cls + ' md-math--source'}>{source(tex, display)}</code>;
    return <span className={cls} dangerouslySetInnerHTML={{ __html: html }}/>;
}
