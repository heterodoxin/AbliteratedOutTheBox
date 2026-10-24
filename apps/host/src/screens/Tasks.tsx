import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useLocation, navigate } from '../lib/router';
import { useEffect, useState } from 'react';
import { Btn, Dialog, Icon, IconBtn, Select, Seg, ErrorNotice } from '../components/ui';
import { TaskBoard } from '../components/TaskBoard';
import { useNative } from '../lib/native';
import { Markdown } from '../components/Markdown';
import { Screen } from '../components/layout';
import { api } from '../lib/client';
import { continueTaskInNewChat, pushToast, useStore } from '../lib/store';
import { blankTask as blank, type Task } from '../lib/tasks';
import { TaskEditor } from '../components/TaskEditor';
import { TaskDetail } from '../components/TaskDetail';
export function TasksScreen() {
    const location = useLocation();
    const library = useStore(s => s.library);
    const native = useNative();
    const loaded = library?.variants.find(v => v.id === library.loaded);
    const [tasks, setTasks] = useState<Task[]>([]), [query, setQuery] = useState(''), [sort, setSort] = useState('next'), archived = false;
    const [editor, setEditor] = useState<Task | null>(null), [assistant, setAssistant] = useState(false), [pending, setPending] = useState(false), [error, setError] = useState(''), [ready, setReady] = useState(false), [result, setResult] = useState<string | null>(null), [removeTask, setRemoveTask] = useState(false);
    async function refresh() {
        const data = await api.get<{
            tasks: Task[];
        }>('/api/tasks');
        setTasks(data.tasks);
        setReady(true);
    }
    useEffect(() => {
        let alive = true;
        const load = () => {
            if (alive)
                void refresh().catch(e => {
                    if (alive) {
                        setError(e.message);
                        setReady(true);
                    }
                });
        };
        load();
        const timer = setInterval(load, 3000);
        window.addEventListener('strata:tasks', load);
        return () => { alive = false; clearInterval(timer); window.removeEventListener('strata:tasks', load); };
    }, []);
    async function action(fn: () => Promise<unknown>) {
        setPending(true);
        setError('');
        try {
            await fn();
            await refresh();
            window.dispatchEvent(new Event('strata:tasks'));
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setPending(false);
        }
    }
    function open(task: Task, assist = false) { setEditor({ ...task }); setAssistant(assist); setError(''); }
    useEffect(() => {
        if (location.query.has('new'))
            open(blank());
    }, [location.query.get('new')]);
    function closeEditor() {
        setEditor(null);
        if (location.query.has('new'))
            navigate('tasks', location.param ?? undefined);
    }
    const currentTask = tasks.find(t => t.id === location.param);
    // Schedule | Board: the board is #/tasks?view=board, the same screen (no route of its own).
    const view: 'schedule' | 'board' = location.query.get('view') === 'board' && !location.param ? 'board' : 'schedule';
    const setView = (next: 'schedule' | 'board') => navigate('tasks', undefined, next === 'board' ? { view: 'board' } : undefined);
    const visible = tasks.filter(t => (!location.param || t.id === location.param) && !!t.archived === archived && `${t.title} ${t.prompt}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => sort === 'name' ? a.title.localeCompare(b.title) : (a.nextRunAt ?? 'z').localeCompare(b.nextRunAt ?? 'z'));
    const selected = tasks.find(t => t.id === result);
    const taskHeader = currentTask ? <div className="task-header-actions">
      <IconBtn name={currentTask.enabled ? 'pause' : 'play'} title={tr(currentTask.enabled ? "Pause {0}" : "Resume {0}", [currentTask.title])} disabled={pending || currentTask.status === 'running' || !!currentTask.archived} onClick={() => void action(() => api.patch(`/api/tasks/${currentTask.id}`, { enabled: !currentTask.enabled }))}/>
      <IconBtn name="trash" title={tr("Delete {0}", [currentTask.title])} disabled={pending || !!currentTask.archived} onClick={() => setRemoveTask(true)}/>
      <Btn icon="pencil" disabled={pending || currentTask.status === 'running'} onClick={() => open(currentTask)}>{tr("Edit")}</Btn>
      {tr(currentTask.lastRunAt || currentTask.output || currentTask.error ? <Btn icon="chat" disabled={pending || currentTask.status === 'running'} onClick={() => continueTaskInNewChat(currentTask)}>{tr("Continue in a new chat")}</Btn> : null)}
      <Btn variant="primary" icon={currentTask.status === 'running' ? 'stop' : 'bolt'} disabled={pending || !loaded && currentTask.status !== 'running'} onClick={() => void action(() => api.post(`/api/tasks/${currentTask.id}/${currentTask.status === 'running' ? 'stop' : 'run'}`))}>{tr(currentTask.status === 'running' ? 'Stop' : 'Run now')}</Btn>
    </div> : null;
    return <Screen width={view === 'board' && !currentTask ? 'full' : 'normal'} title={currentTask ? <span className="sr-only">{currentTask.title}</span> : "Tasks"} sub={currentTask ? undefined : view === 'board' ? "Everything that is running, waiting for you, or needs a look." : "Run a prompt on a schedule, or whenever you need it."} actions={currentTask ? taskHeader : <div className="tasks-actions"><Seg ariaLabel="View" value={view} onChange={setView} options={[{ value: 'schedule', label: 'Schedule', icon: 'clock' }, { value: 'board', label: 'Board', icon: 'panel' }]}/><input className="st-input tasks-search" type="search" aria-label={tr("Search tasks")} placeholder={tr("Search tasks…")} value={query} onChange={e => setQuery(e.target.value)}/>{tr(view === 'board' ? null : <Select ariaLabel={tr("Sort tasks")} value={sort} options={[{ value: 'next', label: 'Next run' }, { value: 'name', label: 'Name' }]} onChange={setSort}/>)}<IconBtn name="plus" title={tr("New task")} onClick={() => open(blank())}/></div>}>
    <div className="tasks-page">
    
    {tr(!currentTask && view !== 'board' && <p className="task-runtime small muted"><Icon name="clock" size={16}/>{tr(library?.mode === 'simulated' ? 'Example model · sample answers. ' : '')}{tr("Runs while Strata\u2019s host is open. Runs saved instructions using the loaded model.")}</p>)}
    {tr(error && !editor ? <ErrorNotice message={error} onRetry={() => void refresh()}/> : null)}
    {tr(currentTask || view === 'board' ? null : !ready ? <p className="muted">{tr("Loading tasks\u2026")}</p> : !visible.length ? <div className="tasks-empty"><Icon name="clock" size={72}/><h2 className="h3">{tr(query ? 'No matching tasks' : 'No scheduled tasks yet')}</h2><p className="muted">{tr(query ? 'Try another search.' : 'Create one with the + button.')}</p></div> : <div className="task-list">{tr(visible.map(t => <article key={t.id} className="task-row"><span className="task-icon"><Icon name="clock" size={24}/></span><div className="task-copy"><h2 className="h4">{t.title}</h2><p className="small muted">{tr(t.status === 'running' ? 'Running…' : t.enabled ? (t.nextRunAt ? `Next: ${new Date(t.nextRunAt).toLocaleString(localeTag(), { dateStyle: 'medium', timeStyle: 'short' })}` : 'Scheduled · calculating next run') : 'Paused')} · {tr(t.timeZone)}</p>{tr(t.error ? <button className="task-failed text-action" onClick={() => setResult(t.id!)}>{tr("Last run failed \u00B7 View")}</button> : null)}{tr(t.output ? <button className="text-action" onClick={() => setResult(t.id!)}>{tr("View last result")}{tr(t.demo ? ' · Example' : '')}</button> : null)}</div><div className="row"><><IconBtn name="pencil" title={tr("Edit {0}", [t.title])} disabled={pending || t.status === 'running'} onClick={() => open(t)}/><Btn size="sm" icon={t.status === 'running' ? 'stop' : 'bolt'} title={tr(t.status === 'running' ? `Stop ${t.title}` : `Run ${t.title} now`)} disabled={pending || !loaded} onClick={() => void action(() => api.post(`/api/tasks/${t.id}/${t.status === 'running' ? 'stop' : 'run'}`))}>{tr(t.status === 'running' ? 'Stop' : 'Run now')}</Btn><label className="row small"><span>{tr("On schedule")}</span><button className="st-toggle" role="switch" aria-label={tr("Schedule {0}", [t.title])} aria-checked={t.enabled} disabled={pending || t.status === 'running'} onClick={() => void action(() => api.patch(`/api/tasks/${t.id}`, { enabled: !t.enabled }))}/></label></></div></article>))}</div>)}
    {tr(!currentTask && view === 'board' ? <TaskBoard query={query} canRun={!!loaded || !!native?.instances.some(i => i.status === 'ready')} onViewResult={setResult}/> : null)}
    {tr(currentTask ? <TaskDetail task={currentTask}/> : null)}
    </div>
    {tr(editor ? <TaskEditor task={editor} assistant={assistant} onClose={closeEditor}/> : null)}
    <Dialog open={!!result} title={selected ? <span>{selected.title}</span> : tr('Task result')} onClose={() => setResult(null)} sheet>{tr(selected ? <><p className="small muted">{tr(selected.lastRunAt ? new Date(selected.lastRunAt).toLocaleString(localeTag()) : '')}{tr(selected.demo ? ' · Example result' : '')}</p><div className="document-markdown">{tr(selected.error ? <ErrorNotice message={selected.error} next={tr("Try running this task again.")}/> : <Markdown text={selected.output ?? tr('No result yet.')}/>)}</div></> : null)}</Dialog>
    <Dialog open={removeTask && !!currentTask} compact title={tr("Remove this task?")} onClose={() => !pending && setRemoveTask(false)} actions={<><Btn disabled={pending} onClick={() => setRemoveTask(false)}>{tr("Keep task")}</Btn><Btn variant="danger" disabled={pending} onClick={() => { const id = currentTask?.id; void action(() => api.patch(`/api/tasks/${id}`, { archived: true, enabled: false })).then(() => { setRemoveTask(false); pushToast('success', 'Task removed'); navigate('tasks'); }); }}>{tr("Remove task")}</Btn></>}><div className="stack stack--sm"><p><strong>“{currentTask?.title}”</strong>{tr(" will stop running and be removed.")}</p></div></Dialog>
  </Screen>;
}
