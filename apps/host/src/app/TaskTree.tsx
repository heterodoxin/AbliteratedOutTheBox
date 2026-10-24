import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { Btn, Dialog, Icon, IconBtn } from '../components/ui';
import { api } from '../lib/client';
import { navigate, href, useLocation } from '../lib/router';
import { blankTask, type Task } from '../lib/tasks';
import { TaskEditor } from '../components/TaskEditor';
import { pushToast, setNavOpen } from '../lib/store';
import { useBoard } from '../lib/use-board';
type TaskLeaf = Task & {
    id: string;
};
export function TaskTree({ collapsed }: {
    collapsed: boolean;
}) {
    const [editor, setEditor] = useState<Task | null>(null);
    const location = useLocation();
    const { board } = useBoard();
    const [open, setOpen] = useState(() => localStorage.getItem('strata.tasks-open') !== 'false');
    const [tasks, setTasks] = useState<TaskLeaf[]>([]), [remove, setRemove] = useState<TaskLeaf | null>(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
    const [leaving, setLeaving] = useState<string | null>(null);
    const leavingRef = useRef<string | null>(null), animationTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    useEffect(() => {
        let alive = true;
        const refresh = () => {
            void api.get<{
                tasks: TaskLeaf[];
            }>('/api/tasks').then(r => {
                if (alive && !leavingRef.current) {
                    setTasks(r.tasks.filter(t => !t.archived));
                    setError('');
                }
            }).catch(e => {
                if (alive)
                    setError(e.message);
            });
        };
        refresh();
        const timer = setInterval(refresh, 5000);
        window.addEventListener('strata:tasks', refresh);
        return () => {
            alive = false;
            clearInterval(timer);
            if (animationTimer.current)
                clearTimeout(animationTimer.current);
            window.removeEventListener('strata:tasks', refresh);
        };
    }, []);
    async function deleteTask() {
        if (!remove)
            return;
        const task = remove;
        setBusy(true);
        setError('');
        try {
            await api.patch(`/api/tasks/${task.id}`, { archived: true, enabled: false });
            leavingRef.current = task.id;
            setLeaving(task.id);
            setRemove(null);
            pushToast('success', 'Task removed');
            if (location.param === task.id)
                navigate('tasks');
            animationTimer.current = setTimeout(() => { setTasks(ts => ts.filter(t => t.id !== task.id)); leavingRef.current = null; setLeaving(null); window.dispatchEvent(new Event('strata:tasks')); }, 240);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
        }
    }
    // A dialog opened from the narrow-window menu closes the menu first (QA F03: the form opened behind it and the keyboard stayed there). The dialogs
    // live here, so they stay mounted while the menu is closed.
    const dialogs = <>
    {tr(editor ? <TaskEditor task={editor} onClose={() => setEditor(null)}/> : null)}
    <Dialog open={!!remove} compact title={tr("Remove this task?")} onClose={() => !busy && setRemove(null)} actions={<><Btn disabled={busy} onClick={() => setRemove(null)}>{tr("Keep task")}</Btn><Btn variant="danger" disabled={busy} onClick={() => void deleteTask()}>{tr(busy ? 'Removing…' : 'Remove task')}</Btn></>}><div className="stack stack--sm"><p><strong>“{remove?.title}”</strong>{tr(" will stop running and be removed.")}</p></div>{tr(error ? <p className="inline-error" role="alert">{tr(error)}</p> : null)}</Dialog>
  </>;
    if (collapsed)
        return dialogs;
    return <section className="task-tree" aria-label={tr("Scheduled tasks")}>
    <div className="ws__head"><button className="navgroup__label task-tree__heading" aria-expanded={open} onClick={() => { setOpen(!open); localStorage.setItem('strata.tasks-open', String(!open)); }}>{tr("Tasks")}{tr(board.need ? <span className="navcount" title={tr("{0} need your attention", [board.need])}><span className="sr-only">{tr("{0} need your attention", [board.need])}</span><span aria-hidden="true">{board.need}</span></span> : null)}<Icon name="chevron" size={12} className={`ws__chev ${open ? 'is-open' : ''}`}/></button><div className="ws__tools"><IconBtn name="plus" title={tr("Add a task")} onClick={() => { setNavOpen(false); setEditor(blankTask()); }}/></div></div>
    {tr(open ? <div className="navsessions"><div className="ws__session row task-tree-row"><a className="navsession" href="#/tasks?view=board" aria-current={location.route === 'tasks' && !location.param && location.query.get('view') === 'board' ? 'page' : undefined} onClick={() => setNavOpen(false)}><span className="navsession__title">{tr("Task board")}</span></a></div>{tr(tasks.map(t => <div className="ws__session row task-tree-row" data-leaving={leaving === t.id || undefined} key={t.id}><a className="navsession" href={href('tasks', t.id)} aria-current={location.route === 'tasks' && location.param === t.id ? 'page' : undefined} title={t.title}><span className="navsession__title">{t.title}</span>{tr(board.problemTasks.includes(t.id) ? <span className="navsession__dot navsession__dot--warn" role="img" aria-label={tr("Needs attention")} title={tr("Needs attention")}/> : null)}{tr(!t.enabled ? <Icon name="clock" size={12}/> : null)}</a><button className="st-btn st-btn--icon session-overflow" aria-label={tr("Edit task {0}", [t.title])} title={tr("Edit {0}", [t.title])} disabled={t.status === 'running'} onClick={() => { setNavOpen(false); setEditor({ ...t }); }}><Icon name="pencil" size={15}/></button><button className="st-btn st-btn--icon session-overflow" aria-label={tr("Delete task {0}", [t.title])} disabled={t.status === 'running'} title={tr(t.status === 'running' ? 'Stop this task before deleting' : `Delete ${t.title}`)} onClick={() => { setNavOpen(false); setRemove(t); setError(''); }}><Icon name="trash" size={15}/></button></div>))}{tr(!tasks.length ? <p className="small muted ws__empty">{tr("No tasks yet")}</p> : null)}{tr(error && !remove ? <p className="xs inline-error">{tr("Couldn\u2019t load tasks.")}</p> : null)}</div> : null)}
    {dialogs}
  </section>;
}
