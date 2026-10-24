import {sessionTitle} from '../lib/session-title';
import { tr } from '../lib/i18n';
/** Conversation statistics are deliberately separate from engine telemetry. */
import { useState } from 'react';
import { Btn, Dialog, IconBtn } from '../components/ui';
import { tokens } from '../lib/format';
import { go, useStore } from '../lib/store';
import type { Session } from '../lib/types';
export function ContextPanel({ session, open, inline = false, onClose }: {
    session: Session | null;
    open: boolean;
    inline?: boolean;
    onClose: () => void;
}) {
    const library = useStore(s => s.library);
    const [expanded, setExpanded] = useState<Record<string, boolean>>(() => {
        try {
            return JSON.parse(localStorage.getItem('strata.inspector-sections') ?? '{"conversation":true}');
        }
        catch {
            return { conversation: true };
        }
    });
    const calls = session?.messages.flatMap(m => m.tools ?? []) ?? [];
    const replies = session?.messages.filter(m => m.role === 'assistant' && !m.streaming) ?? [];
    const lastRequest = [...replies].reverse().find(m => m.requestId);
    const usageKnown = session?.statsAvailable !== false && replies.length > 0 && replies.every(m => ['reported', 'demo'].includes(m.usageSource ?? ''));
    const usageLabel = replies.some(m => m.usageSource === 'demo') ? 'Tokens in / out (example)' : 'Recorded tokens in / out';
    const requestSize = lastRequest?.inputTokens !== undefined && lastRequest.tokens !== undefined ? lastRequest.inputTokens + lastRequest.tokens : null;
    const model = library?.variants.find(v => v.id === library.loaded);
    function change(key: string, value: boolean) {
        const next = { ...expanded, [key]: value };
        setExpanded(next);
        localStorage.setItem('strata.inspector-sections', JSON.stringify(next));
    }
    const content = <aside className="contextpanel" aria-label={tr("Conversation context")}>
    <header className="contextpanel__head st-card__head"><div><h2 className="h4">{tr("Conversation context")}</h2><p className="xs muted">{session ? sessionTitle(session) : tr('No conversation selected')}</p></div><IconBtn name="x" title={tr("Close conversation context")} onClick={onClose}/></header>
    <div className="contextpanel__body">
      <details className="disclosure" open={expanded.conversation} onToggle={e => change('conversation', e.currentTarget.open)}><summary>{tr("This conversation")}</summary><div>
        <dl className="summary-list"><div><dt>{tr("Messages")}</dt><dd>{tr(session?.messages.length ?? 0)}</dd></div><div><dt>{tr(usageLabel)}</dt><dd>{tr(usageKnown ? `${tokens(session?.tokensIn)} / ${tokens(session?.tokensOut)}` : 'Not available for every request')}</dd></div><div><dt>{tr("Last request size")}</dt><dd>{tr(requestSize === null ? 'Not available' : `${tokens(requestSize)}${lastRequest?.contextMax ? ` / ${tokens(lastRequest.contextMax)}` : ''} tokens${lastRequest?.demo ? ' · Example' : ''}`)}</dd></div>{tr(lastRequest?.model ? <div><dt>{tr("Answered by")}</dt><dd>{tr(lastRequest.model)}</dd></div> : null)}</dl>
        <p className="xs muted">{tr("Request size is input plus output at completion. It is not a live reading of the model cache. Older messages may have no recorded usage.")}</p>
        {tr(lastRequest?.skillsUsed?.length ? <p className="small">{tr("Instructions sent: ")}{tr(lastRequest.skillsUsed.join(', '))}.</p> : null)}
      </div></details>
      <details className="disclosure" open={!!expanded.tools} onToggle={e => change('tools', e.currentTarget.open)}><summary>{tr("Tool activity \u00B7 ")}{tr(calls.length)}</summary><div>{tr(calls.length ? calls.map(c => <div className="small" key={c.id}>{c.name} · {tr(c.state)}</div>) : <p className="small muted">{tr("No tool calls in this conversation.")}</p>)}<p className="xs muted">{tr("Tool execution is not connected in this build. Recorded proposals and approvals are examples. Availability does not grant permission.")}</p></div></details>
      <details className="disclosure" open={!!expanded.engine} onToggle={e => change('engine', e.currentTarget.open)}><summary>{tr("Current engine")}</summary><div><p className="small">{tr(model?.label ?? 'No model loaded')}{tr(library?.mode === 'simulated' ? ' · Example' : '')}</p><p className="xs muted">{tr("The currently loaded model may differ from the model that answered earlier messages.")}</p><div className="row"><Btn size="sm" onClick={() => go('monitor')}>{tr("Monitor")}</Btn><Btn size="sm" onClick={() => go('tuning')}>{tr("Engine")}</Btn></div></div></details>
    </div>
  </aside>;
    if (inline)
        return open ? content : null;
    return <Dialog open={open} title={tr("Conversation context")} onClose={onClose} sheet>{tr(content)}</Dialog>;
}
