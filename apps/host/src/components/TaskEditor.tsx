import { tr } from '../lib/i18n';
import { useState } from 'react';
import { Btn, Dialog, Icon, Select, ErrorNotice } from './ui';
import { api } from '../lib/client';
import { useStore } from '../lib/store';
import type { Task } from '../lib/tasks';
export function TaskEditor({ task, assistant = false, onClose, onSaved }: {
    task: Task;
    assistant?: boolean;
    onClose: () => void;
    onSaved?: () => void;
}) {
    const [editor, setEditor] = useState<Task>({ ...task });
    const [step, setStep] = useState<0 | 1>(0), [ai, setAi] = useState(assistant);
    const [description, setDescription] = useState(''), [suggestion, setSuggestion] = useState(false), [pending, setPending] = useState(false), [error, setError] = useState('');
    const library = useStore(s => s.library);
    const valid = !!editor.title.trim() && !!editor.prompt.trim();
    async function action(fn: () => Promise<unknown>) { setPending(true); setError(''); try {
        await fn();
        window.dispatchEvent(new Event('strata:tasks'));
        onSaved?.();
    }
    catch (e) {
        setError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setPending(false);
    } }
    async function save() { await (editor.id ? api.patch(`/api/tasks/${editor.id}`, editor) : api.post('/api/tasks', editor)); onClose(); }
    return <Dialog open title={tr(editor.id ? 'Edit task' : ai ? 'Create with AI' : 'New task')} onClose={() => !pending && onClose()} actions={<>
    <Btn disabled={pending} onClick={() => step ? setStep(0) : onClose()}>{tr(step ? 'Back' : 'Cancel')}</Btn>
    {tr(step === 0 ? <Btn variant="primary" disabled={pending || !valid} onClick={() => setStep(1)}>{tr("Choose schedule ")}<Icon name="chevron" className="wizard-next" size={14}/></Btn> : <Btn variant="primary" disabled={pending || !valid} onClick={() => void action(save)}>{tr(pending ? 'Saving…' : editor.id ? 'Save changes' : 'Create task')}</Btn>)}
  </>}>
    <div className="st-tabs task-editor-steps" role="tablist" aria-label={tr("Task setup steps")}>
      {tr((['Instructions', 'Schedule'] as const).map((label, index) => <button className="st-tab" type="button" key={label} id={`task-tab-${index}`} role="tab" aria-selected={step === index} aria-controls={`task-step-${index}`} tabIndex={step === index ? 0 : -1} disabled={pending || index === 1 && !valid} onClick={() => setStep(index as 0 | 1)} onKeyDown={e => { if (['ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        if (valid) {
            const next = step === 0 ? 1 : 0;
            setStep(next);
            document.getElementById(`task-tab-${next}`)?.focus();
        }
    } }}>{tr(label)}</button>))}
    </div>
    <div className="form-stack task-editor-panel settings-tab-panel" role="tabpanel" id={`task-step-${step}`} aria-labelledby={`task-tab-${step}`}>
      {tr(step === 0 ? <>
        {tr(!editor.id && !ai && !!library?.loaded ? <button type="button" className="text-action task-ai-link" onClick={() => setAi(true)}><Icon name="thinking" size={14}/>{tr("Draft it with AI")}</button> : null)}
        {tr(ai && !suggestion ? <div className="task-assistant"><label className="input-field"><span>{tr("What would you like Strata to do?")}</span><textarea className="st-input task-instructions" rows={5} value={description} onChange={e => setDescription(e.target.value)} placeholder={tr("Help me practise Spanish every morning…")}/></label><Btn disabled={!description.trim() || pending} onClick={() => void action(async () => { const draft = await api.post<{
                draft: Task;
            }>('/api/tasks/draft', { description, timeZone: editor.timeZone }); setEditor({ ...editor, ...draft.draft }); setSuggestion(true); })}>{tr(pending ? 'Drafting…' : 'Suggest a task')}</Btn></div> : <>
          <label className="input-field"><span>{tr("Name")}</span><input className="st-input" maxLength={120} disabled={pending} value={editor.title} onChange={e => setEditor({ ...editor, title: e.target.value })}/></label>
          <label className="input-field"><span>{tr("Instructions")}</span><textarea className="st-input task-instructions" rows={5} maxLength={12000} placeholder={tr("Example: Review this project and summarize what changed.")} disabled={pending} value={editor.prompt} onChange={e => setEditor({ ...editor, prompt: e.target.value })}/></label>
          {tr(ai ? <p className="xs muted">{tr(library?.mode === 'simulated' ? 'Example suggestion. ' : '')}{tr("Review the instructions, then choose a schedule.")}</p> : null)}
        </>)}
      </> : <>
        <div className="task-schedule-summary"><strong>{editor.title}</strong><p className="small muted">{editor.prompt}</p></div>
        <div className="task-schedule"><div className="input-field"><span>{tr("Repeat")}</span><Select ariaLabel={tr("Repeat task")} value={editor.frequency} options={[{ value: 'hourly', label: 'Every hour' }, { value: 'daily', label: 'Every day' }, { value: 'weekdays', label: 'Weekdays' }, { value: 'weekly', label: 'Every week' }]} onChange={frequency => setEditor({ ...editor, frequency })}/></div>{tr(editor.frequency !== 'hourly' ? <label className="input-field"><span>{tr("Time")}</span><input type="time" className="st-input" disabled={pending} value={editor.at} onChange={e => setEditor({ ...editor, at: e.target.value })}/></label> : null)}{tr(editor.frequency === 'weekly' ? <div className="input-field"><span>{tr("Day")}</span><Select ariaLabel={tr("Day of week")} value={String(editor.weekday)} options={['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'].map((label, i) => ({ value: String(i), label }))} onChange={v => setEditor({ ...editor, weekday: Number(v) })}/></div> : null)}</div>
        <div className="input-field"><span>{tr("What this task may do")}</span><Select ariaLabel={tr("What this task may do")} value={editor.mode ?? 'plan'} options={[{ value: 'plan', label: 'Look only, change nothing' }, { value: 'manual', label: 'Stop and tell me before changing anything' }, { value: 'bypass', label: 'Bypass permissions' }]} onChange={mode => setEditor({ ...editor, mode: mode as Task['mode'] })}/></div>
        <label className="row small"><input type="checkbox" disabled={pending} checked={editor.enabled} onChange={e => setEditor({ ...editor, enabled: e.target.checked })}/>{tr("Run on schedule")}</label>
        <p className="xs muted">{tr(editor.timeZone)}{tr(" \u00B7 Missed runs are skipped while Strata is closed.")}</p>
      </>)}
      {tr(error ? <ErrorNotice message={error}/> : null)}
    </div>
  </Dialog>;
}
