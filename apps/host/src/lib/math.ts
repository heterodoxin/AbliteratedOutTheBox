/**
 * Finding the math in an answer (#3268). Pure text work, no rendering: the pieces are drawn by components/MathView.tsx.
 *
 * Recognised, as models write them:
 *   $$ ... $$ and \[ ... \]   display math (may span lines)
 *   $ ... $  and \( ... \)    inline math (one line)
 * A dollar sign that is money stays text: "costs $5 and $10" has no math, because an opening $ must be followed by a non-space,
 * a closing $ must follow a non-space and must not be followed by a digit. `\$` is a literal dollar sign.
 * Code (backtick spans, fenced blocks) is never searched: the caller splits it off first.
 */
export type MathPart = { kind: 'text'; text: string } | { kind: 'math'; tex: string; display: boolean };

const isSpace = (c: string | undefined) => c === undefined || /\s/.test(c);
const isDigit = (c: string | undefined) => c !== undefined && c >= '0' && c <= '9';

/** Where the unescaped `close` starts at or after `from`, or -1. `\\` and `\$` style escapes are skipped. */
function findClose(text: string, from: number, close: string, oneLine: boolean): number {
    for (let i = from; i < text.length; i++) {
        const c = text[i];
        if (oneLine && c === '\n')
            return -1;
        if (c === '\\' && close !== '\\)' && close !== '\\]') {
            i++;
            continue;
        }
        if (text.startsWith(close, i))
            return i;
    }
    return -1;
}

export function splitMath(text: string): MathPart[] {
    const parts: MathPart[] = [];
    let plain = '';
    const flush = () => {
        if (plain) {
            parts.push({ kind: 'text', text: plain });
            plain = '';
        }
    };
    const push = (tex: string, display: boolean) => {
        if (!tex.trim())
            return false;
        flush();
        parts.push({ kind: 'math', tex: tex.trim(), display });
        return true;
    };
    for (let i = 0; i < text.length;) {
        const c = text[i];
        if (c === '\\') {
            const next = text[i + 1];
            if (next === '$') {                       // \$ is a dollar sign
                plain += '$';
                i += 2;
                continue;
            }
            if (next === '(' || next === '[') {
                const close = next === '(' ? '\\)' : '\\]';
                const end = findClose(text, i + 2, close, next === '(');
                if (end > 0 && push(text.slice(i + 2, end), next === '[')) {
                    i = end + 2;
                    continue;
                }
            }
            plain += c + (next ?? '');
            i += 2;
            continue;
        }
        if (c === '$') {
            if (text[i + 1] === '$') {
                const end = findClose(text, i + 2, '$$', false);
                if (end > 0 && push(text.slice(i + 2, end), true)) {
                    i = end + 2;
                    continue;
                }
                plain += '$$';
                i += 2;
                continue;
            }
            if (!isSpace(text[i + 1])) {
                const end = findClose(text, i + 1, '$', true);
                if (end > i + 1 && !isSpace(text[end - 1]) && !isDigit(text[end + 1]) && push(text.slice(i + 1, end), false)) {
                    i = end + 1;
                    continue;
                }
            }
        }
        plain += c;
        i++;
    }
    flush();
    return parts;
}

/** True when the text holds any math, so a plain answer costs nothing. */
export function hasMath(text: string): boolean {
    return text.includes('$') || text.includes('\\(') || text.includes('\\[');
}

/**
 * A block of display math that starts a line: `$$ ... $$` on one line, or `$$` / `\[` on its own line up to the line with the closing
 * delimiter. Returns the formula and the number of lines used, or null (an unclosed block, e.g. while the answer is still streaming,
 * stays text).
 */
export function displayBlock(lines: string[], at: number): { tex: string; used: number } | null {
    const first = lines[at].trim();
    const opens = first.startsWith('$$') ? '$$' : first.startsWith('\\[') ? '\\[' : '';
    if (!opens)
        return null;
    const closes = opens === '$$' ? '$$' : '\\]';
    const rest = first.slice(2);
    const sameLine = rest.indexOf(closes);
    if (sameLine >= 0) {
        // "$$ x $$" alone on the line; text after the closing delimiter makes it an inline display in a paragraph instead
        return rest.slice(sameLine + closes.length).trim() ? null : { tex: rest.slice(0, sameLine), used: 1 };
    }
    const body = [rest];
    for (let j = at + 1; j < lines.length && j < at + 60; j++) {
        const t = lines[j].trim();
        const end = t.indexOf(closes);
        if (end >= 0) {
            if (t.slice(end + closes.length).trim())
                return null;
            body.push(t.slice(0, end));
            return { tex: body.join('\n'), used: j - at + 1 };
        }
        body.push(lines[j]);
    }
    return null;
}
