/**
 * The pure parts of the Time-skip A versus B view: what kind of file it is (the view adapts to it), which words changed, which lines
 * changed, and how an HTML page is made safe to show. The same window serves every kind, so none of this decides a size.
 */
export type FileKind = 'markdown' | 'text' | 'html' | 'image' | 'code' | 'other';

const IMAGE = /\.(png|jpe?g|gif|webp|svg|bmp)$/i;
const MARKDOWN = /\.(md|markdown|mdx)$/i;
const TEXT = /\.(txt|text|log|rst|csv|tsv)$/i;
const HTML = /\.(html?|xhtml)$/i;

export function fileKind(path: string, file?: { image?: boolean; unreadable?: boolean }): FileKind {
  if (file?.image || IMAGE.test(path)) return 'image';
  if (file?.unreadable) return 'other';
  if (MARKDOWN.test(path)) return 'markdown';
  if (HTML.test(path)) return 'html';
  if (TEXT.test(path)) return 'text';
  return 'code';
}

export interface Segment { text: string; changed: boolean }

/** Cap on the work a diff may do (a product of the two lengths); past it the whole text counts as changed rather than freezing the page. */
const WORK_LIMIT = 6_000_000;

function lcsPairs(a: string[], b: string[]): [number, number][] | null {
  if (a.length * b.length > WORK_LIMIT) return null;
  const w = b.length + 1;
  const table = new Uint32Array((a.length + 1) * w);
  for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--) {
    table[i * w + j] = a[i] === b[j] ? table[(i + 1) * w + j + 1] + 1 : Math.max(table[(i + 1) * w + j], table[i * w + j + 1]);
  }
  const pairs: [number, number][] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { pairs.push([i, j]); i++; j++; }
    else if (table[(i + 1) * w + j] >= table[i * w + j + 1]) i++;
    else j++;
  }
  return pairs;
}

const tokens = (text: string) => text.split(/(\s+)/).filter(t => t !== '');

function joinSegments(parts: string[], keep: Set<number>): Segment[] {
  const flags = parts.map((text, i) => /\S/.test(text) && !keep.has(i));
  // Whitespace between two changed words belongs to the mark, so a changed phrase reads as one.
  parts.forEach((text, i) => { if (!/\S/.test(text) && i > 0 && i < parts.length - 1 && flags[i - 1] && flags[i + 1]) flags[i] = true; });
  const out: Segment[] = [];
  parts.forEach((text, i) => {
    const last = out.at(-1);
    if (last && last.changed === flags[i]) last.text += text; else out.push({ text, changed: flags[i] });
  });
  return out;
}

/** The words of both versions with the changed ones marked. Unchanged text is identical on both sides. */
export function wordDiff(original: string, updated: string): { original: Segment[]; updated: Segment[] } {
  const a = tokens(original), b = tokens(updated);
  const pairs = lcsPairs(a, b);
  if (!pairs) return { original: [{ text: original, changed: original !== updated }], updated: [{ text: updated, changed: original !== updated }] };
  const keepA = new Set(pairs.map(p => p[0])), keepB = new Set(pairs.map(p => p[1]));
  return { original: joinSegments(a, keepA), updated: joinSegments(b, keepB) };
}

/** Split marked segments into lines, so a page can lay out headings, lists and paragraphs and still mark words inside them. */
export function segmentLines(segments: Segment[]): Segment[][] {
  const lines: Segment[][] = [[]];
  for (const seg of segments) {
    const parts = seg.text.split(/\r?\n/);
    parts.forEach((part, i) => {
      if (i > 0) lines.push([]);
      if (part) lines[lines.length - 1].push({ text: part, changed: seg.changed });
    });
  }
  return lines;
}

export type DiffRow =
  | { t: 'same' | 'add' | 'del'; text: string; a?: number; b?: number }
  | { t: 'gap'; n: number };

/** A line diff for code and other text, with unchanged stretches folded to one "n unchanged lines" row so it stays short. */
export function lineDiff(original: string, updated: string, context = 2): DiffRow[] {
  const a = original === '' ? [] : original.split(/\r?\n/), b = updated === '' ? [] : updated.split(/\r?\n/);
  const pairs = lcsPairs(a, b);
  const rows: DiffRow[] = [];
  if (!pairs) {
    a.forEach((text, i) => rows.push({ t: 'del', text, a: i + 1 }));
    b.forEach((text, j) => rows.push({ t: 'add', text, b: j + 1 }));
    return rows;
  }
  let i = 0, j = 0;
  const flush = (ei: number, ej: number) => {
    while (i < ei) { rows.push({ t: 'del', text: a[i], a: i + 1 }); i++; }
    while (j < ej) { rows.push({ t: 'add', text: b[j], b: j + 1 }); j++; }
  };
  for (const [pi, pj] of pairs) { flush(pi, pj); rows.push({ t: 'same', text: a[pi], a: pi + 1, b: pj + 1 }); i = pi + 1; j = pj + 1; }
  flush(a.length, b.length);
  // fold unchanged stretches longer than twice the context
  const out: DiffRow[] = [];
  let k = 0;
  while (k < rows.length) {
    if (rows[k].t !== 'same') { out.push(rows[k]); k++; continue; }
    let end = k;
    while (end < rows.length && rows[end].t === 'same') end++;
    const run = rows.slice(k, end);
    const head = k === 0 ? 0 : context, tail = end === rows.length ? 0 : context;
    if (run.length > head + tail + 1) {
      out.push(...run.slice(0, head), { t: 'gap', n: run.length - head - tail }, ...run.slice(run.length - tail));
    } else out.push(...run);
    k = end;
  }
  return out;
}

export const diffStats = (rows: DiffRow[]) => ({ added: rows.filter(r => r.t === 'add').length, removed: rows.filter(r => r.t === 'del').length });

/**
 * A page for the preview frame. The frame is sandboxed without a same-origin right, forms, navigation or pop-ups; it may run the page's own
 * scripts, because most pages people ask for (a 3D scene, a game, a chart) draw themselves with a script and would show as an empty dark
 * page without one. The content policy still blocks every request to the internet: only what was written is shown.
 */
export const HTML_PREVIEW_SANDBOX = 'allow-scripts';
export function htmlPreviewDocument(html: string): string {
  const policy = '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'unsafe-inline\' \'unsafe-eval\'; img-src data: blob:; style-src \'unsafe-inline\'; font-src data:; media-src data: blob:">';
  const base = '<base target="_blank">';
  const head = /<head[^>]*>/i.exec(html);
  if (head) return html.slice(0, head.index + head[0].length) + policy + base + html.slice(head.index + head[0].length);
  return `<!doctype html><html><head><meta charset="utf-8">${policy}${base}</head><body>${html}</body></html>`;
}

/** True when the page asks for something from the internet (a script, style or picture by web address): the preview cannot load it. */
export const needsOutsideFiles = (html: string) => /<(?:script|link|img|iframe|source)\b[^>]*(?:src|href)\s*=\s*["']?(?:https?:)?\/\//i.test(html);

/** The address of one side of a changed picture on the host. */
export const timeSkipImageUrl = (jobId: string, path: string, version: 'original' | 'updated') =>
  `/api/timeskip/file?id=${encodeURIComponent(jobId)}&path=${encodeURIComponent(path)}&version=${version}`;
