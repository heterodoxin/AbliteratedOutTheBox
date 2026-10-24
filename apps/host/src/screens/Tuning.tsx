import { tr } from '../lib/i18n';
import { FloatingMenu } from '../components/FloatingMenu';
import { LifecycleNotice } from '../components/LifecycleNotice';
/** One configuration draft shared by simple choices, parameters and JSON. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { DemoTest, PerformancePreview } from '../components/EngineExperience';
import { AutoTune } from '../components/AutoTune';
import { contextTokens, estimate, execution, RECOMMENDED_CONTEXT_TOKENS } from '../lib/experience';
import type { Workload } from '../lib/experience';
import { ActionError, Btn, Dialog, ErrorNotice, Icon, Select } from '../components/ui';
import { Screen, SectionHead } from '../components/layout';
import { configDocument, configPatch, draftParams, readConfig, type ConfigDocument } from '../lib/config';
import { useStore, setConfigDraft, saveEngine, restartEngine, validateEngine, pushToast, lifecycleAvailable, useLifecycle } from '../lib/store';
import type { BeginnerConfig, EngineConfig, ExpertParam } from '../lib/types';
type Issue = {
    level: 'warn' | 'error';
    key?: string;
    message: string;
};
export function TuningScreen() {
    const engine = useStore(s => s.engine);
    const hardware = useStore(s => s.hardware);
    const [workload, setWorkload] = useState<Workload>('document');
    const [testOpen, setTestOpen] = useState(false);
    const [tuneOpen, setTuneOpen] = useState(false);
    const [unsupported, setUnsupported] = useState(false);
    const [updating, setUpdating] = useState(false);
    const library = useStore(s => s.library);
    const rawDraft = useStore(s => s.configDraft);
    const connected = useStore(s => s.connection === 'live');
    const [mode, setMode] = useState<'beginner' | 'expert'>('beginner');
    const [view, setView] = useState<'parameters' | 'raw'>('parameters');
    const [query, setQuery] = useState('');
    const [group, setGroup] = useState('all');
    const [changedOnly, setChangedOnly] = useState(false);
    const [descriptions, setDescriptions] = useState(false);
    const [moreOpen, setMoreOpen] = useState(false);
    const lifecycle = useLifecycle();
    const [issues, setIssues] = useState<Issue[]>([]);
    const [preview, setPreview] = useState<EngineConfig['predicted'] | null>(null);
    const [pending, setPending] = useState(false);
    const [checked, setChecked] = useState(false);
    const [applyOpen, setApplyOpen] = useState(false);
    const [discardOpen, setDiscardOpen] = useState(false);
    const importRef = useRef<HTMLInputElement>(null);
    const moreRef = useRef<HTMLButtonElement>(null);
    function closeMore() { setMoreOpen(false); moreRef.current?.focus(); }
    const saved = useMemo(() => engine ? configDocument(engine) : null, [engine]);
    const text = rawDraft ?? (saved ? JSON.stringify(saved, null, 2) : '');
    const parsed = useMemo(() => {
        try {
            return { doc: readConfig(text), error: null };
        }
        catch (e) {
            return { doc: null, error: e instanceof Error ? e.message : String(e) };
        }
    }, [text]);
    const doc = parsed.doc;
    const params = useMemo(() => engine && doc ? draftParams(engine, doc) : [], [engine, doc]);
    const changed = useMemo(() => params.filter(p => p.value !== saved?.expert[p.id]), [params, saved]);
    const dirty = rawDraft !== null && (!doc || JSON.stringify(doc) !== JSON.stringify(saved));
    const groups = [...new Set(params.map(p => p.group))];
    const rows = params.filter(p => p.id !== 'server.host' && (unsupported || !!p.applies && !p.applies.startsWith('Preserved'))).filter(p => (group === 'all' || p.group === group) && (!changedOnly || p.value !== saved?.expert[p.id]) && `${p.id} ${p.label} ${p.effect} ${p.group}`.toLowerCase().includes(query.toLowerCase()));
    const current = library?.variants.find(v => v.id === library?.loaded);
    useEffect(() => {
        setChecked(false);
        setUpdating(true);
        if (!doc || !engine || !connected) {
            setPreview(null);
            setUpdating(false);
            return;
        }
        let active = true;
        const timer = setTimeout(() => {
            void validateEngine(configPatch(doc)).then(result => {
                if (!active)
                    return;
                setUpdating(false);
                setIssues(result.issues);
                setPreview(result.predicted ?? null);
            }).catch(e => { if (active) {
                setUpdating(false);
                setIssues([{ level: 'error', message: e.message }]);
            } });
        }, 180);
        return () => { active = false; clearTimeout(timer); };
    }, [text, connected, engine?.variantId]);
    function update(next: ConfigDocument) { setConfigDraft(JSON.stringify(next, null, 2)); }
    function changeParam(id: string, value: ExpertParam['value']) {
        if (!doc)
            return;
        const next = { ...doc, beginner: { ...doc.beginner }, expert: { ...doc.expert, [id]: value } };
        if (id === 'perf.batch')
            Object.assign(next, { batchAutomatic: false });
        if (id === 'ctx.length')
            next.beginner.contextLength = String(value).toLowerCase() as BeginnerConfig['contextLength'];
        update(next);
    }
    async function check() {
        if (!doc) {
            setChecked(true);
            return false;
        }
        const result = await validateEngine(configPatch(doc));
        setIssues(result.issues);
        setPreview(result.predicted ?? null);
        setChecked(true);
        return result.ok;
    }
    async function save(apply = false) {
        if (!doc || pending)
            return;
        setPending(true);
        try {
            if (!await check())
                return;
            if (apply && !lifecycleAvailable())
                return;
            await saveEngine(configPatch(doc), true, apply);
            setConfigDraft(null);
            setApplyOpen(false);
            if (apply)
                await restartEngine();
            else
                pushToast('success', 'Settings saved', 'Used the next time this model loads. Sampling defaults affect the next request.');
        }
        catch (e) {
            setIssues([{ level: 'error', message: e instanceof Error ? e.message : String(e) }]);
        }
        finally {
            setPending(false);
        }
    }
    function download() {
        if (!doc)
            return;
        const url = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = 'strata-engine.json';
        a.click();
        URL.revokeObjectURL(url);
    }
    function resetDraft() {
        if (!engine || !doc)
            return;
        update({ ...doc, batchAutomatic: true, beginner: { priority: 'balanced', contextLength: 'recommended', readImages: false, gpuIdle: 'never' }, expert: Object.fromEntries(engine.expert.map(p => [p.id, p.recommended])) });
    }
    async function importFile(file: File | undefined) {
        if (!file)
            return;
        try {
            update(readConfig(await file.text()));
            setMode('expert');
            setView('raw');
        }
        catch (e) {
            setIssues([{ level: 'error', message: `Import failed: ${e instanceof Error ? e.message : String(e)}` }]);
        }
    }
    if (!engine || !saved)
        return <Screen plainHeader title={tr("Engine")}><div className="setup__note">{tr(connected ? 'Loading settings…' : 'Connect the engine to edit its settings.')}</div></Screen>;
    const predicted = preview ?? engine.predicted;
    const cfg = execution(doc?.expert, contextTokens(doc?.beginner.contextLength ?? engine.beginner.contextLength));
    const savedCfg = execution(saved.expert, contextTokens(saved.beginner.contextLength));
    const contextMax = Math.max(4, Math.floor((current?.contextMax ?? 32768) / 1024));
    const recommendedContext = RECOMMENDED_CONTEXT_TOKENS / 1024;
    const recommendedPosition = contextMax === 4 ? 0 : Math.min(100, (recommendedContext - 4) / (contextMax - 4) * 100);
    const experience = estimate(current, hardware, cfg, workload, library?.mode === 'simulated');
    const savedExperience = estimate(current, hardware, savedCfg, workload, library?.mode === 'simulated');
    const showPreview = <PerformancePreview experience={experience} saved={savedExperience} workload={workload} onWorkload={setWorkload} updating={updating}/>;
    const errors = parsed.error ? [{ level: 'error', message: parsed.error } as Issue] : issues;
    const visibleIssues = errors.filter(i => !i.message.startsWith('Estimated working memory (') && !i.message.includes('is fast memory. Actual fit and offload behavior'));
    const invalid = !!parsed.error || errors.some(i => i.level === 'error');
    return <Screen plainHeader title={tr("Engine")} actions={<><button className="st-btn st-btn--secondary" aria-pressed={mode === 'expert'} onClick={() => setMode(mode === 'expert' ? 'beginner' : 'expert')}>{tr(mode === 'expert' ? 'Easy mode' : 'Expert mode')}</button><Btn disabled={!doc} onClick={resetDraft}>{tr("Reset to recommended")}</Btn><Btn disabled={!doc} onClick={() => setTestOpen(true)}>{tr("Run speed test")}</Btn></>}>
    <div className="stack stack--lg tuning" data-mode={mode}>
      <ActionError />
      {tr(mode === 'beginner' && doc ? <div className="instrument-layout">
        <section className="instrument-controls simple-engine"><div className="auto-tune-card"><Icon name="gauge" size={28}/><h2 className="h3">{tr("Make the most of this computer")}</h2><p className="small muted">{tr("Find a starting point for your hardware, then review before saving.")}</p><Btn variant="primary" disabled={!hardware || !current} onClick={() => setTuneOpen(true)}>{tr("Auto-tune for my PC")}</Btn>
          <label className="input-field"><span>{tr("Conversation context \u00B7 ")}{tr(cfg.context / 1024)}{tr("K")}</span>
            <div className="context-slider"><input type="range" className="st-slider" aria-label={tr("Conversation context")} aria-valuetext={`${cfg.context / 1024}K tokens`} aria-describedby="context-recommendation context-help" min={4} max={contextMax} step={1} value={cfg.context / 1024} onChange={e => changeParam('ctx.length', `${e.target.value}K`)}/><span className="context-recommendation-mark" aria-hidden="true" style={{ left: `calc(${recommendedPosition}% + ${8 - recommendedPosition * .16}px)` }}/></div>
            <div className="context-guidance"><div className="context-scale" aria-hidden="true"><span>{tr("4K")}</span><span>{tr(contextMax)}{tr("K")}</span></div><p className="context-recommendation" id="context-recommendation"><span aria-hidden="true"/>{tr("Recommended: ")}{tr(recommendedContext)}{tr("K")}</p><p className="xs muted" id="context-help">{tr("More conversation history uses more memory.")}</p></div>
          </label></div>
          <section><h2 className="h4">{tr("Resources")}</h2><div className="workload-cards">{tr([{ id: 'shared', title: 'Leave room for other apps', detail: 'Keep this computer responsive' }, { id: 'full', title: 'Full speed', detail: 'Give the engine more CPU threads' }].map(h => <button key={h.id} aria-pressed={(doc.headroom ?? 'shared') === h.id} onClick={() => update({ ...doc, headroom: h.id, expert: { ...doc.expert, 'perf.threads': Math.max(1, (hardware?.cpu.cores ?? 4) - (h.id === 'shared' ? 2 : 0)) } })}><strong>{tr(h.title)}</strong><span>{tr(h.detail)}</span></button>))}</div></section>
        </section>{tr(showPreview)}
      </div> : <div className="stack expert-engine">
        <div className="engine-subtabs" role="tablist" aria-label={tr("Expert editor")}>{tr((['parameters', 'raw'] as const).map(v => <button role="tab" aria-selected={view === v} key={v} onClick={() => setView(v)}>{tr(v === 'raw' ? 'Raw config' : 'Parameters')}</button>))}</div>
        {tr(view === 'raw' ? <section className="stack">
          <SectionHead title={tr("Configuration draft")} right={<div className="row"><Btn size="sm" onClick={() => setDiscardOpen(true)}>{tr("Reload saved")}</Btn><Btn size="sm" onClick={download} disabled={!doc}>{tr("Export")}</Btn><Btn size="sm" onClick={() => importRef.current?.click()}>{tr("Import")}</Btn></div>}/>
          {tr(parsed.error ? <ErrorNotice message={parsed.error} next={tr("Correct the JSON before saving or applying.")}/> : null)}<textarea className="st-input config-editor" aria-label={tr("Configuration draft")} aria-invalid={!!parsed.error} aria-describedby={visibleIssues.length ? 'config-errors' : undefined} spellCheck={false} value={text} onChange={e => setConfigDraft(e.target.value)}/>
        </section> : <>
          <div className="row small muted"><a href="#/serving">{tr("Server access is managed in Serving")}</a><label className="check-label"><input type="checkbox" checked={unsupported} onChange={e => setUnsupported(e.target.checked)}/>{tr("Show unsupported")}</label></div><div className="parameter-toolbar">
            <input className="st-input" type="search" aria-label={tr("Search parameters")} placeholder={tr("Search name or technical key…")} value={query} onChange={e => setQuery(e.target.value)}/>
            <Select value={group} ariaLabel={tr("Parameter group")} options={[{ value: 'all', label: 'All groups' }, ...groups.map(g => ({ value: g, label: g }))]} onChange={setGroup}/>
            <label className="check-label"><input type="checkbox" checked={changedOnly} onChange={e => setChangedOnly(e.target.checked)}/>{tr("Changed only (")}{tr(changed.length)})</label><label className="check-label"><input type="checkbox" checked={descriptions} onChange={e => setDescriptions(e.target.checked)}/>{tr("Show descriptions")}</label>
          </div>
          {tr(groups.filter(g => rows.some(p => p.group === g)).map(g => <section className="parameter-group" key={g}>
            <h2 className="h4">{tr(g)}</h2>
            {tr(rows.filter(p => p.group === g).map(p => <div className="parameter-row" key={p.id}>
              <div><label className="strong" htmlFor={`param-${p.id}`}>{tr(p.label)}</label><code className="parameter-key">{p.id}</code>{tr(descriptions ? <div className="small muted">{tr(p.effect)}</div> : null)}<div className="parameter-support xs muted" data-unsupported={!p.applies || p.applies.startsWith('Preserved') || undefined}>{tr(!p.applies || p.applies.startsWith('Preserved') ? 'Not used by this engine' : p.applies)}{tr(" \u00B7 recommended ")}{tr(String(p.recommended))}</div></div>
              <div><div className="parameter-control"><ExpertInput param={p} disabled={!p.applies || p.applies.startsWith('Preserved')} onChange={value => changeParam(p.id, value)}/><Btn variant="ghost" size="sm" disabled={p.value === saved.expert[p.id]} onClick={() => changeParam(p.id, saved.expert[p.id] ?? p.recommended)}>{tr("Revert")}</Btn></div>{tr(errors.filter(i => i.message.startsWith(p.id + ':')).map(i => <p className="small inline-error" role="alert" key={i.message}>{tr(i.message)}</p>))}</div>
            </div>))}
          </section>))}
          {tr(!rows.length ? <div className="empty-inline">{tr("No parameters match. ")}<Btn variant="ghost" onClick={() => { setQuery(''); setGroup('all'); setChangedOnly(false); }}>{tr("Clear filters")}</Btn></div> : null)}
        </>)}
      </div>)}
      <input type="file" accept=".json,application/json" ref={importRef} hidden onChange={e => { void importFile(e.target.files?.[0]); e.target.value = ''; }}/>
      {tr(mode === 'expert' && descriptions && predicted.notes.length ? <p className="xs muted">{tr(predicted.notes.join(' '))}</p> : null)}
      {tr(visibleIssues.length ? <div className="validation-list" id="config-errors" role="status">{tr(visibleIssues.map((issue, i) => <div key={i} data-tone={issue.level}><Icon name={issue.level === 'error' ? 'error' : 'info'} size={16}/><span>{tr(issue.message)}</span></div>))}</div> : checked ? <p className="small" role="status">{tr("Configuration checked. No validation errors.")}</p> : null)}
      {tr(dirty ? <div className="config-savebar"><span className="small">{tr(changed.length || 'Configuration')} {tr(changed.length === 1 ? 'change' : 'changes')}</span><span className="spacer"/>{tr(library?.loaded && library.mode !== 'simulated' ? <Btn size="sm" disabled={invalid || pending || !connected || !lifecycle.allowed} onClick={() => setApplyOpen(true)}>{tr("Apply now\u2026")}</Btn> : null)}<Btn size="sm" variant="ghost" disabled={pending} onClick={() => setDiscardOpen(true)}>{tr("Discard")}</Btn><Btn size="sm" variant="primary" disabled={invalid || pending || !connected} onClick={() => void save()}>{tr(pending ? 'Saving…' : 'Save')}</Btn></div> : null)}

    </div>
    {tr(moreOpen ? <FloatingMenu anchor={moreRef.current} title={tr("More actions")} onClose={closeMore}>
      <button role="menuitem" className="menu__item" disabled={pending || !doc} onClick={() => { closeMore(); resetDraft(); }}>{tr("Recommended defaults")}</button>
      <button role="menuitem" className="menu__item" disabled={!connected || pending} onClick={() => { closeMore(); void check().catch(e => setIssues([{ level: 'error', message: e.message }])); }}>{tr("Validate draft")}</button>
      <button role="menuitem" className="menu__item" disabled={!doc || invalid} onClick={() => { closeMore(); setTestOpen(true); }}>{tr("Preview test \u00B7 Example")}</button>
    </FloatingMenu> : null)}
    <DemoTest open={testOpen} onClose={() => setTestOpen(false)} experience={experience}/>
    <AutoTune open={tuneOpen} onClose={() => setTuneOpen(false)} onDone={() => { resetDraft(); setTuneOpen(false); }}/>
    <Dialog open={discardOpen} compact title={tr("Discard configuration changes?")} onClose={() => setDiscardOpen(false)} actions={<><Btn onClick={() => setDiscardOpen(false)}>{tr("Keep editing")}</Btn><Btn variant="danger" onClick={() => { setConfigDraft(null); setIssues([]); setDiscardOpen(false); }}>{tr("Discard changes")}</Btn></>}><p>{tr("Discard all unsaved engine changes and return to the saved configuration.")}</p></Dialog>
    <Dialog open={applyOpen} title={tr("Apply settings now?")} onClose={() => setApplyOpen(false)} actions={<><Btn onClick={() => setApplyOpen(false)}>{tr("Keep editing")}</Btn><Btn variant="primary" disabled={pending || invalid || !lifecycle.allowed} onClick={() => void save(true)}>{tr("Save and restart")}</Btn></>}><p>{tr("Save the draft and restart ")}{tr(current?.label ?? 'the loaded model')}.</p><LifecycleNotice /></Dialog>
  </Screen>;
}
function ExpertInput({ param, onChange, disabled = false }: {
    param: ExpertParam;
    disabled?: boolean;
    onChange: (value: ExpertParam['value']) => void;
}) {
    if (param.type === 'select' && param.options)
        return <Select disabled={disabled} value={String(param.value)} ariaLabel={tr(param.label)} options={param.options.map(value => ({ value, label: value }))} onChange={onChange}/>;
    if (param.type === 'toggle')
        return <button disabled={disabled} className="st-toggle" type="button" role="switch" aria-checked={!!param.value} aria-label={tr(param.label)} onClick={() => onChange(!param.value)}/>;
    return <input disabled={disabled} id={`param-${param.id}`} className="st-input" aria-label={tr(param.label)} inputMode={param.type === 'number' ? 'decimal' : undefined} value={String(param.value)} onChange={e => { const v = e.target.value; onChange(param.type === 'number' && v.trim() !== '' && Number.isFinite(Number(v)) ? Number(v) : v); }}/>;
}
