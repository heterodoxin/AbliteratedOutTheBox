import { useEffect, useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import type { ChatMessage } from '../lib/types';
import { atBottom, lastReasoningLine, thoughtDuration } from '../lib/reasoning';
import { Icon } from './ui';

/**
 * The model's reasoning as one quiet line above its answer. Closed by default: while it reasons the line shows what it is thinking about,
 * afterwards "Thought for 38s". Opened, it is a short muted box that follows the newest line unless the person scrolls up.
 */
export function ReasoningBlock({ message }: { message: ChatMessage }) {
  const text = message.reasoning ?? '';
  const body = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [stick, setStick] = useState(true);
  const thinking = !!message.streaming && message.reasoningMs === undefined && !message.content;
  useEffect(() => { if (open && stick && body.current) body.current.scrollTop = body.current.scrollHeight; }, [text, open, stick]);
  useEffect(() => { if (open) { setStick(true); requestAnimationFrame(() => { if (body.current) body.current.scrollTop = body.current.scrollHeight; }); } }, [open]);
  const took = thoughtDuration(message.reasoningMs);
  const label = thinking ? lastReasoningLine(text) || tr('Thinking…') : took ? tr('Thought for {0}', [took]) : tr('Reasoning');
  return <details className="rz" open={open} data-live={thinking || undefined} onToggle={e => setOpen(e.currentTarget.open)}>
    <summary><Icon name="thinking" size={14}/><span className="rz__label">{label}</span><Icon name="chevron" size={12} className="rz__chev"/></summary>
    <div className="rz__body" ref={body} tabIndex={0} aria-label={tr('Model reasoning')} onScroll={e => setStick(atBottom(e.currentTarget))}>{text}</div>
    {open && thinking && !stick ? <button type="button" className="rz__jump" onClick={() => { setStick(true); }}><Icon name="chevron" size={11}/>{tr('Jump to latest')}</button> : null}
  </details>;
}
