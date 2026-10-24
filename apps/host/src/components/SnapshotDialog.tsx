import { tr } from '../lib/i18n';
import { pushToast } from '../lib/store';
import { Btn, Dialog, Icon } from './ui';
/** Saves the picture and says so: a download gives no sign of its own in some windows (QA O02), so a short message follows it. */
export function saveSnapshotWithNotice(data: string) {
    try {
        downloadSnapshot(data);
        pushToast('success', tr('Screenshot saved'), tr('It is in your Downloads folder.'));
    }
    catch {
        pushToast('error', tr('The screenshot was not saved'), tr('Try again, or take a screenshot of this window instead.'));
    }
}
export function downloadSnapshot(data: string) {
    const encoded = data.slice(data.indexOf(',') + 1);
    const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: 'image/png' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `Strata-Monitor-${new Date().toISOString().slice(0, 10)}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function SnapshotDialog({ open, data, pending, error, onClose, onRetry }: {
    open: boolean;
    data: string;
    pending: boolean;
    error: string;
    onClose: () => void;
    onRetry: () => void;
}) {
    const caption = 'My local AI performance with Strata';
    const project = 'https://github.com/Niko1221/Strata';
    return <Dialog open={open} title={tr("Monitor screenshot")} onClose={onClose} actions={<><Btn onClick={onClose}>{tr("Close")}</Btn><Btn icon="download" variant="primary" disabled={!data || pending} onClick={() => saveSnapshotWithNotice(data)}>{tr("Download")}</Btn></>}>
    <div className="snapshot-preview" aria-busy={pending}>
      {tr(pending ? <p className="small muted" role="status">{tr("Preparing your screenshot\u2026")}</p> : data ? <img src={data} alt={tr("Captured Strata Monitor readings")}/> : <div><p className="inline-error" role="alert">{tr(error || 'Screenshot could not be prepared.')}</p><Btn onClick={onRetry}>{tr("Try again")}</Btn></div>)}
    </div>
    {tr(data && !pending ? <><div className="snapshot-share-actions">
      <a className="st-btn st-btn--secondary" href={`https://x.com/intent/post?text=${encodeURIComponent(`${caption}\n${project}`)}`} target="_blank" rel="noreferrer" onClick={() => downloadSnapshot(data)}>{tr("Share to X ")}<Icon name="link" size={15}/></a>
      <a className="st-btn st-btn--secondary" href={`https://www.reddit.com/submit?title=${encodeURIComponent(caption)}&url=${encodeURIComponent(project)}`} target="_blank" rel="noreferrer" onClick={() => downloadSnapshot(data)}>{tr("Share to Reddit ")}<Icon name="link" size={15}/></a>
    </div><p className="small muted">{tr("Sharing downloads the image and opens a draft. Attach the image to your post.")}</p></> : null)}
  </Dialog>;
}
