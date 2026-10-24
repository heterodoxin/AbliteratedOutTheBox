import { useEffect, useMemo, useState } from 'react';
import { tr } from '../lib/i18n';
import type { TimeSkipFile } from '../lib/time-skip';
import { HTML_PREVIEW_SANDBOX, diffStats, fileKind, htmlPreviewDocument, needsOutsideFiles, lineDiff, segmentLines, timeSkipImageUrl, wordDiff, type Segment } from '../lib/time-skip-compare';
import { Btn, Dialog, Icon } from './ui';

const baseName = (path: string) => path.split('/').pop() || path;
const statusText = (status: TimeSkipFile['status']) => tr(status === 'added' ? 'new' : status === 'removed' ? 'removed' : 'changed');

/** Lines of marked words laid out as the document they are: headings, list items, code fences and paragraphs (Markdown), or plain lines. */
function MarkedDoc({ segments, markdown }: { segments: Segment[]; markdown: boolean }) {
  const lines = segmentLines(segments);
  let fenced = false;
  return <div className={markdown ? 'tsc-doc tsc-doc--md' : 'tsc-doc'}>
    {lines.map((line, i) => {
      const plain = line.map(s => s.text).join('');
      const marks = (parts: Segment[]) => parts.map((s, k) => s.changed ? <mark key={k}>{s.text}</mark> : <span key={k}>{s.text}</span>);
      if (!markdown) return <p key={i}>{plain ? marks(line) : <br/>}</p>;
      if (/^\s*```/.test(plain)) { fenced = !fenced; return <p key={i} className="tsc-fence">{plain ? marks(line) : null}</p>; }
      if (fenced) return <p key={i} className="tsc-code">{plain ? marks(line) : <br/>}</p>;
      const heading = /^(#{1,6})\s+/.exec(plain);
      if (heading) return <p key={i} className={`tsc-h tsc-h${Math.min(3, heading[1].length)}`}>{marks(stripLead(line, heading[0].length))}</p>;
      const item = /^(\s*)([-*+]|\d+[.)])\s+/.exec(plain);
      if (item) return <p key={i} className="tsc-li">{marks(stripLead(line, item[0].length))}</p>;
      return <p key={i}>{plain ? marks(line) : <br/>}</p>;
    })}
  </div>;
}

/** Drop the first `count` characters (the Markdown marker) from the front of a line of segments. */
function stripLead(line: Segment[], count: number): Segment[] {
  const out: Segment[] = []; let left = count;
  for (const seg of line) {
    if (left >= seg.text.length) { left -= seg.text.length; continue; }
    out.push({ text: seg.text.slice(left), changed: seg.changed }); left = 0;
  }
  return out;
}

function DiffView({ original, updated }: { original: string; updated: string }) {
  const rows = useMemo(() => lineDiff(original, updated), [original, updated]);
  const stats = diffStats(rows);
  return <div className="tsc-diff" role="table" aria-label={tr("Changes")}>
    <div className="tsc-diff__stats"><span className="tsc-add">+{stats.added}</span><span className="tsc-del">−{stats.removed}</span></div>
    {rows.map((r, i) => r.t === 'gap'
      ? <div className="tsc-diff__gap" role="row" key={i}>{tr('{0} unchanged lines', [r.n])}</div>
      : <div className={`tsc-diff__row tsc-diff__row--${r.t}`} role="row" key={i}><span className="tsc-diff__no" aria-hidden="true">{r.a ?? ''}</span><span className="tsc-diff__no" aria-hidden="true">{r.b ?? ''}</span><span className="tsc-diff__sign" aria-hidden="true">{r.t === 'add' ? '+' : r.t === 'del' ? '−' : ''}</span><code>{r.text || ' '}</code></div>)}
  </div>;
}

function Picture({ src, label, missing, onOpen }: { src: string; label: string; missing?: string; onOpen: () => void }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => { setFailed(false); }, [src]);
  if (missing || failed) return <div className="tsc-picture tsc-picture--empty"><span>{missing ?? tr("This picture could not be loaded.")}</span></div>;
  return <button type="button" className="tsc-picture" onClick={onOpen} aria-label={tr("Enlarge the {0} picture", [label])}><img src={src} alt={label} onError={() => setFailed(true)}/></button>;
}

function FileView({ jobId, file }: { jobId: string; file: TimeSkipFile }) {
  const kind = fileKind(file.path, file);
  const [code, setCode] = useState(false);
  const [big, setBig] = useState<'original' | 'updated' | null>(null);
  useEffect(() => { setCode(false); setBig(null); }, [file.path]);
  const original = file.original ?? '', updated = file.updated ?? '';
  const words = useMemo(() => (kind === 'markdown' || kind === 'text') ? wordDiff(original, updated) : null, [kind, original, updated]);
  const heads = <div className="tsc-heads"><span><b>A</b> · {tr("Original")}</span><span><b>B</b> · {tr("Updated")}</span></div>;
  if (kind === 'other') return <p className="tsc-note">{tr("This file is too big, or not text, so it is listed but not shown.")}</p>;
  if (kind === 'image') {
    const a = timeSkipImageUrl(jobId, file.path, 'original'), b = timeSkipImageUrl(jobId, file.path, 'updated');
    return <>{heads}<div className="tsc-pair tsc-pair--pictures">
      <Picture src={a} label={tr("Original")} missing={file.status === 'added' ? tr("Not there before.") : undefined} onOpen={() => setBig('original')}/>
      <Picture src={b} label={tr("Updated")} missing={file.status === 'removed' ? tr("Removed.") : undefined} onOpen={() => setBig('updated')}/>
    </div>
    <Dialog open={!!big} title={`${baseName(file.path)} · ${big === 'original' ? tr('Original') : tr('Updated')}`} onClose={() => setBig(null)} sheet>
      {big ? <img className="tsc-enlarged" src={big === 'original' ? a : b} alt={baseName(file.path)}/> : null}
    </Dialog></>;
  }
  if (kind === 'html') return <>
    <div className="tsc-tools"><Btn size="sm" variant="ghost" icon={code ? 'image' : 'file'} aria-pressed={code} onClick={() => setCode(v => !v)}>{tr(code ? 'View preview' : 'View code')}</Btn></div>
    {code ? <DiffView original={original} updated={updated}/> : <>{heads}<div className="tsc-pair tsc-pair--html">
      {/* The page may run its own scripts in an isolated frame; the policy inside blocks every outside request. */}
      {file.status === 'added' ? <div className="tsc-picture tsc-picture--empty"><span>{tr("Not there before.")}</span></div>
        : <iframe title={tr("Original page")} sandbox={HTML_PREVIEW_SANDBOX} srcDoc={htmlPreviewDocument(original)} loading="lazy"/>}
      {file.status === 'removed' ? <div className="tsc-picture tsc-picture--empty"><span>{tr("Removed.")}</span></div>
        : <iframe title={tr("Updated page")} sandbox={HTML_PREVIEW_SANDBOX} srcDoc={htmlPreviewDocument(updated)} loading="lazy"/>}
    </div>{needsOutsideFiles(updated) ? <p className="tsc-note">{tr("This page asks for files from the internet, which the preview does not load, so it may look incomplete. View code shows exactly what was written.")}</p> : null}</>}
  </>;
  if (words) return <>{heads}<div className="tsc-pair tsc-pair--text">
    <MarkedDoc segments={words.original} markdown={kind === 'markdown'}/>
    <MarkedDoc segments={words.updated} markdown={kind === 'markdown'}/>
  </div></>;
  return <DiffView original={original} updated={updated}/>;
}

/**
 * The finished A versus B, in one window of a fixed, modest size whatever was made: text and Markdown rendered with the changed words marked,
 * a web page as two sandboxed previews (with a code toggle), pictures side by side to enlarge, anything else as a diff, and a compact switcher
 * when several files changed. It scrolls inside itself; it never grows to fill the screen.
 */
export function TimeSkipCompare({ jobId, files, more = 0 }: { jobId: string; files: TimeSkipFile[]; more?: number }) {
  const [at, setAt] = useState(0);
  useEffect(() => { setAt(0); }, [jobId]);
  const file = files[Math.min(at, files.length - 1)];
  if (!file) return null;
  return <section className="tsc" aria-label={tr("Original and updated, side by side")}>
    {files.length > 1 ? <div className="tsc-files" role="tablist" aria-label={tr("Changed files")}>
      {files.map((f, i) => <button key={f.path} type="button" role="tab" aria-selected={i === at} tabIndex={i === at ? 0 : -1} title={f.path} onClick={() => setAt(i)}
        onKeyDown={e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); setAt((i + (e.key === 'ArrowRight' ? 1 : files.length - 1)) % files.length); } }}>
        <Icon name={fileKind(f.path, f) === 'image' ? 'image' : 'file'} size={13}/><span>{baseName(f.path)}</span><small>{statusText(f.status)}</small></button>)}
      {more > 0 ? <span className="tsc-more">{tr('+{0} more', [more])}</span> : null}
    </div> : <div className="tsc-one"><Icon name={fileKind(file.path, file) === 'image' ? 'image' : 'file'} size={14}/><strong title={file.path}>{baseName(file.path)}</strong><small>{statusText(file.status)}</small></div>}
    <div className="tsc-stage"><FileView jobId={jobId} file={file}/></div>
  </section>;
}
