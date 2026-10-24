import { useEffect, useState } from 'react';
import { tr } from '../lib/i18n';
import { elapsedText, isPlayful, statusWords, useLiveStatus } from '../lib/live-status';

/**
 * The line under the newest message while the model works: a small pulse, what it is doing (a playful word while it only thinks,
 * the real action once a tool runs), how long, and how many tokens. It shows the latest status frame only and nothing older.
 */
export function StatusLine({ sessionId, startedAt, tokens }: { sessionId: string | null; startedAt: number; tokens?: number }) {
  const frame = useLiveStatus(sessionId);
  const [now, setNow] = useState(() => Date.now());
  const [rotation, setRotation] = useState(0);
  useEffect(() => {
    if (!frame) return;
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, [!!frame]);
  const playful = !!frame && isPlayful(frame);
  useEffect(() => {
    if (!playful) return;
    const spin = setInterval(() => setRotation(r => r + 1), 3500);
    return () => clearInterval(spin);
  }, [playful]);
  if (!frame) return null;
  const count = frame.tokens ?? tokens;
  return <div className="live-status" data-phase={frame.phase} aria-live="off">
    <span className="live-status__pulse" aria-hidden="true"><i/><i/></span>
    <span className="live-status__verb" key={playful ? rotation : frame.code + String(frame.args?.target ?? '')} title={statusWords(frame, rotation)}>{statusWords(frame, rotation)}</span>
    <span className="live-status__time">{elapsedText(startedAt, now)}</span>
    <span className="live-status__tokens">{count ? tr('{0} tokens', [count.toLocaleString()]) : ''}</span>
  </div>;
}
