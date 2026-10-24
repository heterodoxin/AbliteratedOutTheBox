import { tr } from '../lib/i18n';

export interface InstallPart { kind?: string; name?: string; state?: string; done_gb?: number; total_gb?: number; files_done?: number; files_total?: number }

/**
 * The parts of a download, one line each: the model (with "2/4" when it comes in parts), the picture reader, the engine. A model in several
 * parts is only downloaded when every line says Done, so a person can see which part is under way instead of one bar that restarts.
 */
export function DownloadParts({ parts }: { parts: InstallPart[] | undefined }) {
  if (!parts || parts.length < 2) return null;
  return <ul className="native-parts" aria-label={tr("Download parts")}>
    {parts.map(p => <li key={`${p.kind}-${p.name}`} data-state={p.state}>
      <span>{tr(p.name)}{(p.files_total ?? 0) > 1 ? ` ${p.files_done ?? 0}/${p.files_total}` : ''}</span>
      <small className="muted">{p.state === 'done' || p.state === 'current' ? tr('Done') : p.state === 'waiting' || p.state === 'pending' ? tr('Waiting') : Number.isFinite(p.done_gb) && Number.isFinite(p.total_gb) ? `${p.done_gb!.toFixed(1)} / ${p.total_gb!.toFixed(1)} GB` : ''}</small>
    </li>)}
  </ul>;
}
