import { useEffect, useState } from 'react';
import { tr } from '../lib/i18n';
import { goalAction, useGoal, type Goal } from '../lib/goal';
import { pushToast, resumeGoal, useStore } from '../lib/store';
import { plainFailureText } from '../lib/plain-errors';
import { Btn, Dialog, Icon, IconBtn } from './ui';

/** The words after the objective: where the goal is, in one short phrase. */
export function goalStatusText(goal: Goal): string {
  switch (goal.phase) {
    case 'paused': return tr('Paused');
    case 'blocked': return goal.blocked || tr('Needs you');
    case 'complete': return tr('Done');
    default: return tr('Working');   // no "step 1 of 256": the host's round count is not a measure of progress
  }
}

/**
 * The goal of this chat, as a slim bar attached above the message box: part of the chat, but separate. Icon-only Pause or Resume, Edit and
 * Remove. The host owns the goal; every button asks it and draws the answer.
 */
export function GoalBar({ chatId }: { chatId: string | null | undefined }) {
  const goal = useGoal(chatId);
  const streaming = useStore(s => s.streaming);
  const [editing, setEditing] = useState(false), [removing, setRemoving] = useState(false), [text, setText] = useState(''), [busy, setBusy] = useState(false);
  useEffect(() => { setEditing(false); setRemoving(false); }, [chatId]);
  if (!chatId || !goal) return null;
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); }
    catch (e) { pushToast('error', tr("The goal was not changed"), plainFailureText(e instanceof Error ? e.message : String(e)).message); }
    finally { setBusy(false); }
  };
  const resumable = goal.phase === 'paused' || goal.phase === 'blocked';
  return <>
    <div className="goalbar" data-phase={goal.phase} role="group" aria-label={tr("Goal")}>
      <Icon name={goal.phase === 'complete' ? 'check' : 'bolt'} size={16}/>
      <span className="goalbar__text" title={goal.objective}><strong>{tr("Goal")}</strong><span> · {goal.objective}</span><span className="goalbar__state"> · {goalStatusText(goal)}</span></span>
      {goal.phase !== 'complete' ? (resumable
        ? <IconBtn name="play" title={tr("Resume the goal")} size={16} disabled={busy || streaming} onClick={() => void run(() => resumeGoal())}/>
        : <IconBtn name="pause" title={tr("Pause the goal")} size={16} disabled={busy} onClick={() => void run(() => goalAction(chatId, 'pause'))}/>) : null}
      {goal.phase !== 'complete' ? <IconBtn name="pencil" title={tr("Edit the goal")} size={16} disabled={busy} onClick={() => { setText(goal.objective); setEditing(true); }}/> : null}
      <IconBtn name="x" title={tr("Remove the goal")} size={16} disabled={busy} onClick={() => setRemoving(true)}/>
    </div>
    <Dialog open={editing} compact title={tr("Edit the goal")} onClose={() => setEditing(false)} actions={<><Btn onClick={() => setEditing(false)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={busy || !text.trim()} onClick={() => void run(async () => { await goalAction(chatId, 'edit', text.trim()); setEditing(false); })}>{tr("Save goal")}</Btn></>}>
      <label className="input-field"><span>{tr("What should the assistant keep working towards?")}</span><textarea className="st-input" rows={3} maxLength={4000} value={text} onChange={e => setText(e.target.value)} autoFocus/></label>
    </Dialog>
    <Dialog open={removing} compact title={tr("Remove this goal?")} onClose={() => setRemoving(false)} actions={<><Btn onClick={() => setRemoving(false)}>{tr("Keep goal")}</Btn><Btn variant="danger" disabled={busy} onClick={() => void run(async () => { await goalAction(chatId, 'clear'); setRemoving(false); })}>{tr("Remove goal")}</Btn></>}>
      <p>{tr("The assistant stops working towards it. Your chat and files stay as they are.")}</p>
    </Dialog>
  </>;
}
