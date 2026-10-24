import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import type { PreviewArtifact } from '../lib/artifacts';
import { plainTimeSkipError, replayTimeSkip, setTimeSkipResult, stopTimeSkip } from '../lib/store';
import { realTimeSkipDocuments, timeSkipCheckOutcomes, timeSkipDocumentMarkdown, timeSkipDocuments, timeSkipExample, timeSkipResultState, type TimeSkipDocument, type TimeSkipRun, type TimeSkipVersion } from '../lib/time-skip';
import { TimeSkipWorking } from './TimeSkipWorking';
import { isForkJob } from '../lib/time-skip';
import { FloatingMenu } from './FloatingMenu';
import { TimeSkipCompare } from './TimeSkipCompare';
import { Btn, Dialog, Icon } from './ui';
function VersionTabs({ version, onChange, done, fork }: {
    fork?: boolean;
    version: TimeSkipVersion;
    onChange: (version: TimeSkipVersion) => void;
    done: boolean;
}) {
    const options = ['original', 'updated'] as const;
    return <div className="ts-version-tabs" role="tablist" aria-label={tr("Result version")} data-version={version}>
    <span className="ts-tab-selection" aria-hidden="true"/>
    {tr(options.map((which, i) => <button key={which} type="button" role="tab" id={`ts-result-tab-${which}`} aria-controls="ts-result" aria-selected={version === which} tabIndex={version === which ? 0 : -1} onClick={() => onChange(which)} onKeyDown={event => {
                if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key))
                    return;
                event.preventDefault();
                const next = event.key === 'Home' ? 0 : event.key === 'End' ? 1 : (i + 1) % 2;
                onChange(options[next]);
                event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
            }}><Icon name={which === 'original' ? 'file' : 'pencil'} size={14}/><span>{tr(fork ? (which === 'original' ? 'A' : 'B') : which === 'original' ? 'Original' : done ? 'Updated' : 'Update')}</span></button>))}
  </div>;
}
function ImageResult({ document, original, compare }: {
    document: TimeSkipDocument;
    original?: TimeSkipDocument;
    compare: boolean;
}) {
    const [position, setPosition] = useState(50);
    if (!document.image)
        return null;
    return <figure className="ts-image-result">
    {tr(compare ? <div className="ts-image-labels"><span>{tr("Original")}</span><span>{tr("Updated")}</span></div> : null)}
    <div className="ts-image-frame">
      <img src={document.image.url} alt={tr(document.image.alt)}/>
      {tr(compare && original?.image ? <>
        <img className="ts-image-original" src={original.image.url} alt={tr(original.image.alt)} style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}/>
        <span className="ts-image-divider" style={{ left: `${position}%` }} aria-hidden="true"><span><Icon name="chevron" size={11}/><Icon name="chevron" size={11}/></span></span>
        <input type="range" min="0" max="100" value={position} aria-label={tr("Image comparison divider")} aria-valuetext={`${position}% original, ${100 - position}% updated`} onChange={event => setPosition(Number(event.target.value))}/>
      </> : null)}
    </div>
    <figcaption>{tr(compare ? <><Icon name="sliders" size={12}/>{tr("Drag to compare")}</> : 'Sample image · 1200 × 900')}</figcaption>
  </figure>;
}
function ResultDocument({ document, original, highlight = false, label, imageCompare = false }: {
    document: TimeSkipDocument;
    original?: TimeSkipDocument;
    highlight?: boolean;
    label?: string;
    imageCompare?: boolean;
}) {
    const oldLines = new Set(original?.code?.split('\n'));
    const goal = document.successCheck.match(/^(\d+) of (\d+) testers complete setup\.$/);
    const total = goal ? Math.min(20, Number(goal[2])) : 0;
    return <article className="ts-document" data-updated={highlight || undefined}>
    <header><Icon name={document.image ? 'image' : 'file'} size={15}/><h3>{tr(document.title)}</h3><span className="ts-doc-label">{tr(label ?? (imageCompare ? 'Compare' : highlight ? 'Updated' : 'Original'))}</span></header>
    {document.image ? <ImageResult document={document} original={original} compare={imageCompare}/>
            : document.code ? <pre aria-label={`${document.title} · ${tr('code')}`}><code>{document.code.split('\n').map((line, i) => <span key={i} data-changed={highlight && !oldLines.has(line) || undefined}>{line}{'\n'}</span>)}</code></pre>
                : <ol>{tr(document.steps.map((step, i) => <li key={step} style={{ '--ts-order': i } as React.CSSProperties} data-changed={highlight && original?.steps[i] !== step || undefined}><span className="ts-step-icon"><Icon name={step.startsWith('Invite') ? 'chat' : step.startsWith('Collect') ? 'clock' : 'tool'} size={15}/></span><span>{tr(step)}</span><span className="ts-step-number" aria-hidden="true">{tr(String(i + 1).padStart(2, '0'))}</span></li>))}</ol>}
    {tr(!document.image && !document.hideCheck ? <section className="ts-success" data-defined={document.successCheck !== 'Not defined' || undefined}><h4><Icon name="shield" size={13}/>{tr("Success check")}{tr(goal ? <span>{tr("Setup goal")}</span> : null)}</h4><p>{tr(document.successCheck)}</p>{tr(goal ? <div className="ts-goal" aria-hidden="true"><div>{tr(Array.from({ length: total }, (_, i) => <i key={i} data-filled={i < Number(goal[1]) || undefined}/>))}</div><span>{tr(goal[1])}<small>/{tr(goal[2])}</small></span></div> : null)}</section> : null)}
  </article>;
}
function RealDetails({ run }: { run: TimeSkipRun }) {
    const job = run.job!;
    const outcomes = timeSkipCheckOutcomes(run);
    return <>
    <h4>{tr("Checks")}</h4>
    {outcomes.length ? <div className="ts-check-table" role="table" aria-label={tr("Checks")}>
      <div role="row"><span role="columnheader">{tr("Check")}</span><span role="columnheader">{tr("Result")}</span></div>
      {outcomes.map(c => <div role="row" key={c.id}><strong role="rowheader">{c.id}</strong><span role="cell">{c.outcome}</span></div>)}
    </div> : <p className="ts-prompt">{tr("No checks are set up for this folder, so nothing was judged automatically. Look at the two versions and choose which to keep.")}</p>}
    <h4>{tr("History")}</h4>
    <ol className="ts-history"><li><Icon name="file" size={14}/><span>{tr("Original saved")}<span>{tr("Preserved before any changes.")}</span></span></li>
      {job.attempts.map(a => <li key={a.index}><Icon name={a.decision === 'reject' ? 'x' : 'check'} size={14}/><span>{tr('Revision {0}', [a.index])}<span>{a.reason}</span></span></li>)}</ol>
    {job.error ? <p className="ts-prompt"><strong>{tr("What went wrong")}</strong>{plainTimeSkipError(job.error)}</p> : null}
  </>;
}
export function TimeSkipPanel({ run, sessionId, inline, expanded, onCompare, onClose, onPreview }: {
    run: TimeSkipRun;
    sessionId: string;
    inline: boolean;
    expanded: boolean;
    onCompare: (expanded: boolean) => void;
    onClose: () => void;
    onPreview: (a: PreviewArtifact) => void;
}) {
    const [now, setNow] = useState(Date.now);
    const [version, setVersion] = useState<TimeSkipVersion>(run.retainedVersion ?? 'updated');
    const [details, setDetails] = useState(false);
    const [menu, setMenu] = useState(false);
    const [imageCompare, setImageCompare] = useState(false);
    const [wide, setWide] = useState(false);
    const panel = useRef<HTMLElement>(null);
    const menuAnchor = useRef<HTMLButtonElement>(null);
    const { frame, ready, kept, status } = timeSkipResultState(run, now);
    const real = !!run.job;
    const documents = run.job ? realTimeSkipDocuments(run.job) : timeSkipDocuments(run.prompt);
    const checks = timeSkipExample(run.prompt);
    const sideBySide = expanded && wide;
    // A finished real run that changed files: the A versus B view, one window whatever was made. Everything else keeps the older layout.
    const compareReal = real && frame.done && (run.job?.files?.length ?? 0) > 0;
    const workingReal = real && !frame.done;
    const realEmpty = real && frame.done && !compareReal;
    useEffect(() => {
        if (!panel.current)
            return;
        const observer = new ResizeObserver(entries => setWide(entries[0].contentRect.width > 650));
        observer.observe(panel.current);
        return () => observer.disconnect();
    }, [inline]);
    useEffect(() => {
        setNow(Date.now());
        if (run.completedAt)
            return;
        const timer = setInterval(() => setNow(Date.now()), 250);
        return () => clearInterval(timer);
    }, [run.id, run.startedAt, run.completedAt]);
    useEffect(() => { setVersion(run.retainedVersion ?? 'updated'); setDetails(false); setMenu(false); setImageCompare(false); }, [run.id, run.startedAt, run.retainedVersion]);
    function openResult(which: TimeSkipVersion) {
        const document = documents[which];
        if (real) {
            onPreview({ id: `${run.id}-${run.startedAt}-${which}`, kind: 'markdown', title: `${which === 'original' ? 'Original' : 'Updated'} · ${document.title}`, content: timeSkipDocumentMarkdown(document) });
            return;
        }
        if (document.image) {
            onPreview({ id: `${run.id}-${run.startedAt}-${which}`, kind: 'image', title: `${which}-fieldnotes-cover.svg`, url: document.image.url, demo: true });
            return;
        }
        // Snapshot IDs include the run clock, so replaying never reuses an old artifact.
        onPreview({ id: `${run.id}-${run.startedAt}-${which}`, kind: 'markdown', title: `${which}-${document.code ? 'counter' : 'launch-plan'}.md`, content: timeSkipDocumentMarkdown(document), demo: true });
    }
    const forkRun = isForkJob(run.job);
    function restore(which: TimeSkipVersion) {
        setTimeSkipResult(sessionId, run.id, which);
        setVersion(which);
        setMenu(false);
    }
    const result = (which: TimeSkipVersion) => which === 'updated' && !ready ? <div className="ts-building" role="status"><div className="ts-building-visual" aria-hidden="true"><span className="ts-paper ts-paper--saved"><Icon name="file" size={15}/><i /><i /><i /></span><span className="ts-paper ts-paper--update"><Icon name="pencil" size={15}/><i /><i /><i /></span></div><strong>{tr("Working on the update")}</strong><p>{tr("The original is saved. The new result will appear here when it\u2019s ready to compare.")}</p></div>
        : <ResultDocument document={documents[which]} original={documents.original} highlight={which === 'updated'} imageCompare={imageCompare && !expanded && which === 'updated'}/>;
    const content = <aside ref={panel} className="ts-panel" id="time-skip-content" aria-label={tr("Time-skip")} data-expanded={expanded || undefined}>
    <div className="ts-scroll">
      <div className="ts-content">
        <div className="ts-heading"><h2 aria-live="polite"><span className="ts-status-icon">{tr(!frame.done ? <span className="ts-activity"/> : <Icon name={kept === 'original' ? 'file' : 'shield'} size={21}/>)}</span><span>{tr(forkRun && frame.done && (run.job?.files?.length ?? 0) > 0 ? (run.retainedVersion === 'original' ? 'Version A kept' : run.retainedVersion === 'updated' ? 'Version B kept' : 'Two versions to compare') : status)}</span></h2></div>
        <p className="ts-caption">{tr(forkRun && frame.done ? 'A and B each went all the way through. Keep the one you like.' : frame.done ? kept === 'original' ? 'The updated result is still available.' : 'The original is always available.' : frame.pass === 2 ? 'Trying one more revision. Your progress is saved.' : run.job ? 'Working on a copy of your files.' : 'Improving the saved draft.')}</p>
        {tr(workingReal ? <TimeSkipWorking run={run} sessionId={sessionId}/> : compareReal ? <><TimeSkipCompare jobId={run.job!.id} files={run.job!.files!} more={run.job!.moreFiles ?? 0}/></> : realEmpty ? <p className="ts-empty">{tr(frame.done && run.job?.phase && ['failed', 'cancelled', 'interrupted'].includes(run.job.phase) ? 'Nothing was kept from this run. Your files were not changed.' : 'The assistant did not change anything, so there is nothing to compare. Your files were not changed.')}</p> : <>
        {tr(expanded ? <div className="ts-compare-heading"><h3>{tr("Compare results")}</h3><Btn size="sm" variant="ghost" icon="contract" onClick={() => onCompare(false)}>{tr(inline ? 'Back to chat' : 'Done comparing')}</Btn></div> : null)}
        <div className="ts-switch-row" data-comparing={expanded || undefined}>
          <VersionTabs version={version} onChange={which => { setVersion(which); setImageCompare(false); }} done={frame.done} fork={forkRun}/>
          {tr(!expanded ? <button type="button" className="st-btn st-btn--ghost st-btn--sm ts-compare-button" aria-pressed={documents.updated.image ? imageCompare : undefined} disabled={!ready} onClick={() => { if (documents.updated.image) {
        setVersion('updated');
        setImageCompare(value => !value);
    }
    else
        onCompare(true); }}><Icon name="expand" size={13}/>{tr("Compare")}</button> : null)}
        </div>
        {tr(expanded ? <div id="ts-result" className="ts-comparison" role={sideBySide ? 'group' : 'tabpanel'} aria-label={tr(sideBySide ? 'Result comparison' : undefined)} aria-labelledby={sideBySide ? undefined : `ts-result-tab-${version}`}>
          {tr((['original', 'updated'] as const).map(which => <section key={which} className="ts-comparison-column" data-version={which} data-selected={version === which || undefined}>
            <div className="ts-column-heading"><h4>{tr(forkRun ? (which === 'original' ? 'A' : 'B') : which === 'original' ? 'Original' : 'Updated')}</h4><span>{tr(which === kept && frame.done ? <><Icon name="check" size={12}/>{tr("Kept result")}</> : which === 'original' ? 'Saved draft' : ready ? 'Changes highlighted' : 'In progress')}</span></div>
            {tr(result(which))}
          </section>))}
        </div> : <div id="ts-result" role="tabpanel" aria-labelledby={`ts-result-tab-${version}`} className="ts-selected" key={version}>{tr(result(version))}</div>)}
        </>)}
        {tr(!compareReal && ready && (!real || documents.changes.length > 0) ? <section className="ts-changes"><h3><Icon name="sliders" size={14}/>{tr("What changed")}<span>{tr(documents.changes.length)}{tr(" updates")}</span></h3><ul>{tr(documents.changes.map((change, i) => <li key={change}><Icon name={i === documents.changes.length - 1 ? 'shield' : 'pencil'} size={14}/><span>{tr(change)}</span><Icon name="check" size={11}/></li>))}</ul></section> : null)}
        {tr(!real && kept === 'original' && frame.done ? <div className="ts-restored" role="status"><span>{tr("Original restored in this example.")}</span><button onClick={() => restore('updated')}>{tr("Undo")}</button></div> : null)}
        {real && forkRun && frame.done && (run.job?.files?.length ?? 0) > 0 ? <section className="ts-keep" aria-label={tr('Keep a version')}>
          <p>{tr(run.retainedVersion === 'original' ? 'Version A is in your files.' : run.retainedVersion === 'updated' ? 'Version B is in your files.' : 'Your files are unchanged. Choose the version to put in your folder; you can switch later.')}</p>
          <span className="ts-keep__buttons"><Btn variant="secondary" icon={run.retainedVersion === 'original' ? 'check' : undefined} disabled={run.retainedVersion === 'original'} onClick={() => restore('original')}>{tr('Keep A')}</Btn><Btn variant="secondary" icon={run.retainedVersion === 'updated' ? 'check' : undefined} disabled={run.retainedVersion === 'updated'} onClick={() => restore('updated')}>{tr('Keep B')}</Btn></span>
        </section> : null}
        {real && !forkRun && frame.done && (run.job?.files?.length ?? 0) > 0 ? <section className="ts-keep" aria-label={tr('Keep the update')}>
          <p>{tr(run.retainedVersion === 'updated' ? 'The update is in your files. You can put the original back.' : 'Your files are unchanged. Keep the update to put it in your folder; you can put the original back later.')}</p>
          {run.retainedVersion === 'updated' ? <Btn variant="secondary" icon="refresh" onClick={() => restore('original')}>{tr('Put the original back')}</Btn> : <Btn variant="primary" icon="check" onClick={() => restore('updated')}>{tr('Keep the update')}</Btn>}
        </section> : null}
        <section className="ts-details">
          <button className="ts-disclosure" aria-expanded={details} aria-controls="ts-details-content" onClick={() => setDetails(v => !v)}><Icon name="clock" size={14}/><strong>{tr("Details")}</strong><span>{tr(frame.done ? 'Checks & history' : 'Progress')}</span><Icon name="chevron" size={13}/></button>
          <div className="ts-details-reveal" data-open={details || undefined} aria-hidden={!details} inert={!details}><div className="ts-details-clip"><div id="ts-details-content" className="ts-details-content">
            <p className="ts-prompt"><strong>{tr("Request")}</strong>{run.prompt}</p>
            {real ? <RealDetails run={run}/> : <><h4>{tr("Same checks, both versions")}</h4>
            <div className="ts-check-table" role="table" aria-label={tr("Simulated comparison checks")}>
              <div role="row"><span role="columnheader">{tr("Check")}</span><span role="columnheader">{tr("Original")}</span><span role="columnheader">{tr("Updated")}</span></div>
              {tr(checks.labels.map((label, i) => <div role="row" key={label}><strong role="rowheader">{tr(label)}</strong><span role="cell">{tr(checks.original[i])}</span><span role="cell">{tr(frame.index >= 4 ? checks.improved[i] : 'Pending')}</span></div>))}
            </div>
            <h4>{tr("History")}</h4>
            <ol className="ts-history"><li><Icon name="file" size={14}/><span>{tr("Original saved")}<span>{tr("Preserved before any revisions.")}</span></span></li>{tr(ready ? <li><Icon name="check" size={14}/><span>{tr(frame.index >= 4 ? 'Update passed the example checks' : 'Update ready for checking')}<span>{tr(checks.correction)}</span></span></li> : null)}{tr(frame.index >= 5 ? <li><Icon name={frame.index >= 7 ? 'x' : 'clock'} size={14}/><span>{tr(frame.index >= 7 ? 'Later revision not kept' : 'Checking a later revision')}<span>{tr(frame.index >= 7 ? checks.rejection : 'The saved update stays available.')}</span></span></li> : null)}</ol>
            {tr(frame.index >= 7 ? <details className="ts-rejected"><summary>{tr("View the rejected revision")}<Icon name="chevron" size={13}/></summary><ResultDocument document={documents.revision} label={tr("Not kept")}/></details> : null)}</>}
          </div></div></div>
        </section>
      </div>
    </div>
    <footer className="ts-footer"><div>{tr(sideBySide ? <Btn onClick={() => openResult('original')} icon="file">{tr("Open original")}</Btn> : null)}<Btn variant="secondary" className="btn-neutral" block disabled={(sideBySide || version === 'updated') && !ready} onClick={() => openResult(sideBySide ? 'updated' : version)}><Icon name="file" size={15}/>{tr(!sideBySide && version === 'original' ? 'Open original' : 'Open result')}<Icon name="chevron" size={12} className="timeskip-card-arrow"/></Btn><button ref={menuAnchor} className="st-btn st-btn--icon" aria-label={tr("More Time-skip actions")} aria-haspopup="menu" aria-expanded={menu} onClick={() => setMenu(v => !v)}><Icon name="ellipsis" size={17}/></button></div></footer>
    {tr(menu ? <FloatingMenu anchor={menuAnchor.current} title={tr("Time-skip actions")} onClose={() => setMenu(false)}><button role="menuitem" className="menu__item" disabled={!frame.done} onClick={() => restore(kept === 'original' ? 'updated' : 'original')}><Icon name="branch"/>{tr(kept === 'original' ? 'Keep updated result' : 'Restore original result')}</button>{real ? (!frame.done ? <button role="menuitem" className="menu__item" onClick={() => { setMenu(false); void stopTimeSkip(sessionId, run.id); }}><Icon name="stop"/>{tr("Stop Time-skip")}</button> : null) : <button role="menuitem" className="menu__item" onClick={() => { setMenu(false); onCompare(false); replayTimeSkip(sessionId, run.id); }}><Icon name="refresh"/>{tr("Replay example")}</button>}</FloatingMenu> : null)}
  </aside>;
    return inline ? content : <Dialog open title={tr("Time-skip")} onClose={onClose} sheet wide={expanded} className="ts-dialog">{tr(content)}</Dialog>;
}
