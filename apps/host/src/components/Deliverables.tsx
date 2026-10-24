import { useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import { fileAction, fileSize, fileTypeLabel, previewOf, rawUrlOf, type ActivityGroup, type Deliverable } from '../lib/deliverables';
import type { PreviewArtifact } from '../lib/artifacts';
import { pushToast } from '../lib/store';
import { plainFailureText } from '../lib/plain-errors';
import { FloatingMenu } from './FloatingMenu';
import { Icon } from './ui';

const problem = (title: string, e: unknown) => pushToast('error', tr(title), plainFailureText(e instanceof Error ? e.message : String(e)).message);

/** A line's numbers ("+225 −0") shown as a pair of small figures; the sentence itself stays plain grey. */
export function Plusminus({ added, removed }: { added: number; removed: number }) {
  return <span className="pm" aria-label={tr('{0} lines added, {1} removed', [added, removed])}><span className="pm__a">+{added}</span><span className="pm__d">−{removed}</span></span>;
}

const showOf = (file: { path: string; name: string }, sessionId: string, onPreview: (a: PreviewArtifact) => void, kind?: string) =>
  previewOf(sessionId, file).then(onPreview, e => {
    // A file with no preview opens in its own app instead.
    if (kind === 'other') return fileAction.open(sessionId, file.path, 'app').catch(err => problem('The file was not opened', err));
    return problem('The file has no preview', e);
  });

/** One file the assistant made or changed: its type, name and size, and Open (in the file's own app or the browser) with a menu for the other things to do with it, among them Open in sidebar. */
export function FileCard({ file, sessionId, onPreview }: { file: Deliverable; sessionId: string; onPreview: (a: PreviewArtifact) => void }) {
  const more = useRef<HTMLButtonElement>(null);
  const [menu, setMenu] = useState(false), [busy, setBusy] = useState(false);
  const run = async (fn: () => Promise<unknown>, failed: string) => { setMenu(false); try { await fn(); } catch (e) { problem(failed, e); } };
  return <div className="fcard" data-kind={file.kind}>
    <span className="fcard__type">{fileTypeLabel(file)}</span>
    <span className="fcard__name"><strong title={file.path}>{file.name}</strong><span>{fileSize(file.size)}</span></span>
    <span className="fcard__split">
      <button type="button" className="fcard__open" disabled={busy} onClick={() => { setBusy(true); void fileAction.open(sessionId, file.path, 'app').catch(err => problem('The file was not opened', err)).finally(() => setBusy(false)); }}>{tr("Open")}</button>
      <button ref={more} type="button" className="fcard__more" aria-haspopup="menu" aria-expanded={menu} aria-label={tr("More ways to open {0}", [file.name])} onClick={() => setMenu(v => !v)}><Icon name="chevron" size={14} className="fcard__chev"/></button>
    </span>
    {tr(menu ? <FloatingMenu anchor={more.current} title={tr("File actions")} width={220} onClose={() => setMenu(false)}>
      <button role="menuitem" className="menu__item" onClick={() => { setMenu(false); void showOf(file, sessionId, onPreview, file.kind); }}><Icon name="panel"/>{tr("Open in sidebar")}</button>
      <button role="menuitem" className="menu__item" onClick={() => void run(() => fileAction.open(sessionId, file.path, 'folder'), 'The folder was not opened')}><Icon name="folder"/>{tr("Show in folder")}</button>
      <button role="menuitem" className="menu__item" onClick={() => void run(async () => { await navigator.clipboard.writeText(await fileAction.fullPath(sessionId, file.path)); pushToast('info', tr('Path copied')); }, 'The path was not copied')}><Icon name="copy"/>{tr("Copy path")}</button>
      <button role="menuitem" className="menu__item" onClick={() => void run(async () => { await fileAction.attach(sessionId, file.path); pushToast('info', tr('Attached to your next message'), file.name); }, 'The file was not attached')}><Icon name="attach"/>{tr("Attach as context")}</button>
    </FloatingMenu> : null)}
  </div>;
}

/** "Edited N files": each with its lines added and removed, and the total under See all. */
export function EditedFiles({ files, sessionId, onPreview }: { files: Deliverable[]; sessionId: string; onPreview: (a: PreviewArtifact) => void }) {
  const [all, setAll] = useState(false);
  const textual = files.filter(f => f.lines !== null);
  if (textual.length < 2) return null;
  const shown = all ? textual : textual.slice(0, 3);
  const added = textual.reduce((n, f) => n + f.added, 0), removed = textual.reduce((n, f) => n + f.removed, 0);
  return <div className="edited">
    <div className="edited__head"><b>{tr('Edited {0} files', [textual.length])}</b></div>
    {shown.map(f => <button type="button" className="edited__row" key={f.path} onClick={() => void showOf(f, sessionId, onPreview, f.kind)}><Icon name="file" size={16}/><span>{f.name}</span><Plusminus added={f.added} removed={f.removed}/></button>)}
    <button type="button" className="edited__row edited__row--all" disabled={textual.length <= 3 && !all} onClick={() => setAll(v => !v)}><span>{tr(all ? 'Show fewer' : 'See all')}</span><Plusminus added={added} removed={removed}/></button>
  </div>;
}

/**
 * The saved skills an answer used, as one quiet line under the answer in the same style as the activity line ("Ran 2 commands, read a file"):
 * "Used 2 skills: Weekly report, Brand voice"; opened, each skill with its one-line description.
 */
export function SkillsUsed({ skills }: { skills: { name: string; description: string }[] }) {
  const [open, setOpen] = useState(false);
  if (!skills.length) return null;
  const names = skills.map(s => s.name).join(', ');
  return <div className="act" data-open={open || undefined}>
    <button type="button" className="act__line" aria-expanded={open} onClick={() => setOpen(v => !v)}>
      <Icon name="skills" size={15}/><span>{skills.length === 1 ? tr('Used 1 skill: {0}', [names]) : tr('Used {0} skills: {1}', [skills.length, names])}</span>
      <Icon name="chevron" size={14} className="act__chev"/>
    </button>
    {open ? <ul className="act__files">{skills.map(s => <li key={s.name}><span className="act__skill"><Icon name="skills" size={14}/>{s.name}{s.description ? <span className="small muted">{s.description}</span> : null}</span></li>)}</ul> : null}
  </div>;
}

/** A run of tool use between two pieces of text, as one quiet line: "Ran 2 commands, read monitor.png", with a thumbnail when it touched a picture. */
export function ActivityLine({ group, sessionId, onPreview }: { group: ActivityGroup; sessionId: string; onPreview: (a: PreviewArtifact) => void }) {
  const [open, setOpen] = useState(false);
  const picture = group.files.find(f => f.kind === 'image');
  // The host's sentence may end with the numbers ("… +225 −0"): they are drawn as figures instead.
  const sentence = group.line.replace(/\s*\+\d+\s*[−-]\s*\d+\s*$/, '');
  return <div className="act" data-open={open || undefined}>
    <button type="button" className="act__line" aria-expanded={open} disabled={!group.files.length} onClick={() => setOpen(v => !v)}>
      <span>{sentence}</span>{group.added || group.removed ? <Plusminus added={group.added} removed={group.removed}/> : null}
      {group.files.length ? <Icon name="chevron" size={14} className="act__chev"/> : null}
      {!group.closed ? <span className="act__live" aria-hidden="true"/> : null}
    </button>
    {picture ? <img className="act__thumb" src={rawUrlOf(sessionId, picture.path)} alt="" loading="lazy" onClick={() => void showOf(picture, sessionId, onPreview)}/> : null}
    {open ? <ul className="act__files">{group.files.map(f => <li key={f.path}><button type="button" onClick={() => void showOf(f, sessionId, onPreview, f.kind)}><Icon name="file" size={14}/>{f.name}{f.added || f.removed ? <Plusminus added={f.added} removed={f.removed}/> : null}</button></li>)}</ul> : null}
  </div>;
}
