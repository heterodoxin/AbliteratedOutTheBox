import { tr } from '../lib/i18n';
import { useRef, useState } from 'react';
import { Btn, Dialog, type IconName } from './ui';
import { compactTokens, elapsed } from '../lib/chat-metrics';
import type { Session } from '../lib/types';
type Agent = NonNullable<Session['agents']>[number];
const states: Record<Agent['status'], {
    label: string;
    icon: IconName;
}> = {
    working: { label: 'In progress', icon: 'clock' }, completed: { label: 'Completed', icon: 'check' },
    failed: { label: 'Failed', icon: 'error' }, queued: { label: 'Queued', icon: 'clock' }, cancelled: { label: 'Cancelled', icon: 'x' },
};
export function SubagentRow({ agent: a, now = Date.now() }: {
    agent: Agent;
    now?: number;
}) {
    const [log, setLog] = useState(false);
    const observedAt = useRef(Date.now());
    const duration = a.durationMs + (a.status === 'working' ? Math.max(0, now - (a.updatedAt ?? observedAt.current)) : 0);
    const state = states[a.status] ?? states.queued;
    return <div className="subagent-row" data-status={a.status}>
    <span className="subagent-row__dot" aria-label={tr(state.label)}/>
    <div className="subagent-row__copy"><div className="subagent-row__title"><strong title={a.name}>{a.name}</strong><span className="subagent-row__status">{tr(state.label)}</span></div><p title={a.description}>{a.description}</p>{tr(a.status === 'failed' ? <button className="text-action" onClick={() => setLog(true)}>{tr("View log")}</button> : null)}</div>
    <div className="subagent-row__metrics"><span>{tr(compactTokens(a.tokens))}{tr(" tok")}</span><span>{tr(elapsed(duration))}</span></div>
  <Dialog open={log} title={tr("{0} \u00B7 log", [a.name])} onClose={() => setLog(false)} actions={<Btn onClick={() => setLog(false)}>{tr("Done")}</Btn>}><p>{a.description}</p><p className="small muted">{tr("Status: ")}{tr(state.label)} · {tr(elapsed(a.durationMs))} · {tr(compactTokens(a.tokens))}{tr(" tokens")}</p></Dialog></div>;
}
