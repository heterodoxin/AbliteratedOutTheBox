import { tr } from '../lib/i18n';
/**
 * Settings -> Tools: what the assistant may use in chats, in plain words.
 *
 * Every row is a switch in the existing Field pattern. Nothing here is a new component: switches, selects, dialogs and buttons are
 * the ones the other Settings panels use. The host owns the truth (GET/PUT /api/agent/tools); a change applies from the next message.
 */
import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { pushToast } from '../lib/store';
import { Btn, Dialog, ErrorNotice, Field, Icon, Select } from './ui';

export interface ToolGroup {
  id: string;
  name: string;
  about: string;
  area: 'computer' | 'web' | 'together';
  enabled: boolean;
  /** Manual mode asks before this tool changes anything. */
  asks: boolean;
}
export interface CustomTool {
  id: string;
  name: string;
  about: string;
  kind: 'instructions' | 'command' | 'server';
  enabled: boolean;
  /** Shown in the list: the command, the server address or the first words of the instructions. */
  summary: string;
  instructions?: string;
  command?: string;
  address?: string;
  args?: string[];
}
export interface ToolsState {
  groups: ToolGroup[];
  search: { provider: 'builtin' | 'exa' | 'perplexity' | 'deepseek'; keySet: boolean };
  /** Skip the once-per-chat question before the first web search or page in Manual mode. Off unless the person turned it on. */
  webWithoutAsking: boolean;
  browser: { enabled: boolean; found: boolean; name: string | null; missing?: 'component' };
  custom: CustomTool[];
}

const AREAS: { id: ToolGroup['area']; title: string }[] = [
  { id: 'computer', title: 'On this computer' },
  { id: 'web', title: 'On the web' },
  { id: 'together', title: 'Working on bigger jobs' },
];
const SEARCH_CHOICES = [
  { value: 'builtin', label: 'DuckDuckGo (no key)' },
  { value: 'exa', label: 'Exa (your own key)' },
  { value: 'perplexity', label: 'Perplexity (your own key)' },
  { value: 'deepseek', label: 'DeepSeek (your own key)' },
];
const KIND_CHOICES = [
  { value: 'instructions', label: 'Follow my instructions' },
  { value: 'command', label: 'Run a command' },
  { value: 'server', label: 'Connect a tool server (MCP)' },
];
const KIND_NAMES: Record<CustomTool['kind'], string> = { instructions: 'Instructions', command: 'Command', server: 'Tool server' };

interface Draft {
  id?: string;
  name: string;
  about: string;
  kind: CustomTool['kind'];
  instructions: string;
  command: string;
  address: string;
  args: string;
}
const blank = (): Draft => ({ name: '', about: '', kind: 'instructions', instructions: '', command: '', address: '', args: '' });

export function SettingsTools({ preview }: { preview?: ToolsState }) {
  const [state, setState] = useState<ToolsState | null>(preview ?? null);
  const [error, setError] = useState('');
  const [key, setKey] = useState('');
  const [draft, setDraft] = useState<Draft | null>(null);
  const [formError, setFormError] = useState('');
  const [remove, setRemove] = useState<CustomTool | null>(null);
  const [busy, setBusy] = useState(false);

  async function load() {
    try { setState(await api.get<ToolsState>('/api/agent/tools')); setError(''); }
    catch (e) { setError(e instanceof Error ? e.message : 'The tool list could not be loaded.'); }
  }
  useEffect(() => { if (!preview) void load(); }, []);

  async function save(change: Record<string, unknown>) {
    if (preview) return;
    setBusy(true);
    try { setState(await api.put<ToolsState>('/api/agent/tools', change)); setError(''); }
    catch (e) { setError(e instanceof Error ? e.message : 'The change could not be saved.'); void load(); }
    finally { setBusy(false); }
  }
  const toggle = (id: string, on: boolean) => {
    setState(s => s && { ...s, groups: s.groups.map(g => g.id === id ? { ...g, enabled: on } : g) });
    void save({ groups: { [id]: on } });
  };

  async function submit() {
    if (!draft) return;
    if (!/^[A-Za-z][A-Za-z0-9 _-]{0,39}$/.test(draft.name.trim())) { setFormError('Give the tool a short name: letters, numbers, spaces or hyphens.'); return; }
    if (!draft.about.trim()) { setFormError('Say what the tool does, in a sentence. The assistant reads this to decide when to use it.'); return; }
    if (draft.kind === 'instructions' && !draft.instructions.trim()) { setFormError('Write the instructions the assistant should follow when it uses this tool.'); return; }
    if (draft.kind === 'command' && !draft.command.trim()) { setFormError('Enter the command to run.'); return; }
    if (draft.kind === 'server' && !draft.address.trim()) { setFormError('Enter the tool server’s web address, or the program that starts it.'); return; }
    setBusy(true);
    try {
      const body = { name: draft.name.trim(), about: draft.about.trim(), kind: draft.kind, instructions: draft.instructions, command: draft.command.trim(), address: draft.address.trim(), args: draft.args.split('\n').map(a => a.trim()).filter(Boolean) };
      setState(draft.id ? await api.patch<ToolsState>(`/api/agent/tools/custom/${encodeURIComponent(draft.id)}`, body) : await api.post<ToolsState>('/api/agent/tools/custom', body));
      setDraft(null); setFormError('');
    } catch (e) { setFormError(e instanceof Error ? e.message : 'The tool could not be saved.'); }
    finally { setBusy(false); }
  }

  if (!state) return <div className="stack stack--lg settings-models">{error ? <ErrorNotice message={tr(error)} onRetry={() => void load()} /> : <p className="small muted">{tr('Loading the tool list…')}</p>}</div>;
  const keyed = state.search.provider !== 'builtin';

  return <div className="stack stack--lg settings-tools">
    <div><h2 className="h3">{tr('Tools')}</h2><p className="small muted settings-models__intro">{tr('Choose what the assistant may use in your chats. A change applies from your next message. In Manual mode, anything that changes your computer still asks you first.')}</p></div>
    {error ? <ErrorNotice message={tr(error)} onRetry={() => void load()} /> : null}
    {AREAS.map(area => <section className="settings-section" key={area.id}>
      <h2 className="h4" style={{ marginBottom: 10 }}>{tr(area.title)}</h2>
      <div className="st-card st-card--flush">
        {state.groups.filter(g => g.area === area.id).map(g => <Field key={g.id} label={tr(g.name)} hint={tr(g.about)}>
          <button className="st-toggle" role="switch" aria-checked={g.enabled} aria-label={tr(g.name)} disabled={busy} onClick={() => toggle(g.id, !g.enabled)} />
        </Field>)}
      </div>
    </section>)}

    <section className="settings-section">
      <h2 className="h4" style={{ marginBottom: 10 }}>{tr('Web search')}</h2>
      <div className="st-card st-card--flush">
        <Field label={tr('Where results come from')} hint={tr(keyed ? 'Searches use your own account with this service.' : 'Results come from DuckDuckGo, or from Bing when DuckDuckGo is busy. You do not need an account or a key.')}>
          <Select ariaLabel={tr('Where results come from')} value={state.search.provider} options={SEARCH_CHOICES} onChange={value => { setKey(''); void save({ search: { provider: value } }); }} />
        </Field>
        {keyed ? <Field label={tr('Your key')} hint={tr(state.search.keySet ? 'A key is saved on this computer. Enter a new one to replace it.' : 'Saved on this computer only. Strata never shows it again.')}>
          <div className="row">
            <input className="st-input" style={{ width: 190 }} type="password" aria-label={tr('Your key')} autoComplete="new-password" spellCheck={false} placeholder={tr(state.search.keySet ? 'Leave blank to keep the saved key' : 'Paste your key')} value={key} onChange={e => setKey(e.target.value)} />
            <Btn size="sm" disabled={busy || !key.trim()} onClick={() => { void save({ search: { provider: state.search.provider, key: key.trim() } }); setKey(''); pushToast('success', tr('Key saved')); }}>{tr('Save key')}</Btn>
          </div>
        </Field> : null}
        <Field label={tr('Let the assistant search the web without asking')} hint={tr('In Manual mode the assistant asks once per chat before it searches the web or opens a web page. Turn this on to skip that question.')}>
          <button className="st-toggle" role="switch" aria-checked={state.webWithoutAsking} aria-label={tr('Let the assistant search the web without asking')} disabled={busy} onClick={() => void save({ webWithoutAsking: !state.webWithoutAsking })} />
        </Field>
      </div>
    </section>

    <section className="settings-section">
      <h2 className="h4" style={{ marginBottom: 10 }}>{tr('Web browser')}</h2>
      <div className="st-card st-card--flush">
        <Field label={tr('Use a web browser')} hint={state.browser.found ? tr('The assistant can open pages, click and type in {0}. In Manual mode every action asks first.', [state.browser.name ?? tr('your browser')]) : tr(state.browser.missing === 'component' ? 'The browser tool is not part of this installation.' : 'Chrome or Edge was not found on this computer. Install one, then come back and turn this on.')}>
          <button className="st-toggle" role="switch" aria-checked={state.browser.enabled && state.browser.found} aria-label={tr('Use a web browser')} disabled={busy || !state.browser.found} onClick={() => void save({ browser: { enabled: !state.browser.enabled } })} />
        </Field>
      </div>
    </section>

    <section className="settings-section">
      <h2 className="h4" style={{ marginBottom: 10 }}>{tr('Your own tools')}</h2>
      <p className="small muted" style={{ marginBottom: 10 }}>{tr('Teach the assistant something new: give it instructions to follow, a command to run, or connect a tool server. Commands and servers ask first in Manual mode.')}</p>
      {state.custom.length ? <div className="st-card st-card--flush">
        {state.custom.map(t => <Field key={t.id} label={tr(t.name)} hint={<>{tr(t.about)} <span className="muted">· {tr(KIND_NAMES[t.kind])}</span></>}>
          <div className="row">
            <Btn size="sm" disabled={busy} onClick={() => { setFormError(''); setDraft({ id: t.id, name: t.name, about: t.about, kind: t.kind, instructions: t.instructions ?? '', command: t.command ?? '', address: t.address ?? '', args: (t.args ?? []).join('\n') }); }}>{tr('Edit')}</Btn>
            <Btn size="sm" disabled={busy} onClick={() => setRemove(t)}>{tr('Remove')}</Btn>
            <button className="st-toggle" role="switch" aria-checked={t.enabled} aria-label={tr(t.name)} disabled={busy} onClick={() => void save({ custom: { [t.id]: !t.enabled } })} />
          </div>
        </Field>)}
      </div> : <div className="provider-empty"><Icon name="tool" size={26} /><strong>{tr('No tools of your own yet')}</strong><p className="small muted">{tr('For example: “Write release notes in our house style”, or a command that checks your website is up.')}</p></div>}
      <div style={{ marginTop: 12 }}><Btn icon="plus" block onClick={() => { setFormError(''); setDraft(blank()); }}>{tr('Add a tool')}</Btn></div>
    </section>

    <Dialog open={draft !== null} title={tr(draft?.id ? 'Edit tool' : 'Add a tool')} compact onClose={() => setDraft(null)} actions={<><Btn onClick={() => setDraft(null)}>{tr('Cancel')}</Btn><Btn variant="primary" disabled={busy} onClick={() => void submit()}>{tr('Save tool')}</Btn></>}>
      {draft ? <div className="stack">
        <label className="input-field"><span>{tr('Name')}</span><input className="st-input" aria-label={tr('Name')} value={draft.name} placeholder={tr('For example: Release notes')} spellCheck={false} onChange={e => setDraft({ ...draft, name: e.target.value })} /></label>
        <label className="input-field"><span>{tr('What it does')}</span><input className="st-input" aria-label={tr('What it does')} value={draft.about} placeholder={tr('One sentence. The assistant reads it to decide when to use the tool.')} onChange={e => setDraft({ ...draft, about: e.target.value })} /></label>
        <div className="input-field"><span>{tr('How it works')}</span><Select ariaLabel={tr('How it works')} value={draft.kind} options={KIND_CHOICES} onChange={value => setDraft({ ...draft, kind: value as Draft['kind'] })} /></div>
        {draft.kind === 'instructions' ? <label className="input-field"><span>{tr('Instructions')}</span><textarea className="st-input" rows={6} aria-label={tr('Instructions')} value={draft.instructions} placeholder={tr('Write what the assistant should do when it uses this tool.')} onChange={e => setDraft({ ...draft, instructions: e.target.value })} /></label> : null}
        {draft.kind === 'command' ? <>
          <label className="input-field"><span>{tr('Command to run')}</span><input className="st-input" aria-label={tr('Command to run')} value={draft.command} placeholder={tr('For example: curl -I https://example.com')} spellCheck={false} onChange={e => setDraft({ ...draft, command: e.target.value })} /></label>
          <p className="small muted">{tr('The assistant can pass text to the command. The command reads it from the STRATA_INPUT setting. Its printed output goes back to the assistant.')}</p>
        </> : null}
        {draft.kind === 'server' ? <>
          <label className="input-field"><span>{tr('Web address or program')}</span><input className="st-input" aria-label={tr('Web address or program')} value={draft.address} placeholder={tr('https://… or the program that starts the server')} spellCheck={false} onChange={e => setDraft({ ...draft, address: e.target.value })} /></label>
          <label className="input-field"><span>{tr('Extra settings for the program ')}<span className="muted">{tr('(optional, one per line)')}</span></span><textarea className="st-input" rows={3} aria-label={tr('Extra settings for the program')} value={draft.args} spellCheck={false} onChange={e => setDraft({ ...draft, args: e.target.value })} /></label>
        </> : null}
        {formError ? <p className="small" role="alert" style={{ color: 'var(--danger, #b3261e)' }}>{tr(formError)}</p> : null}
      </div> : null}
    </Dialog>
    <Dialog open={remove !== null} title={tr('Remove this tool?')} compact onClose={() => setRemove(null)} actions={<><Btn onClick={() => setRemove(null)}>{tr('Keep it')}</Btn><Btn variant="primary" onClick={() => { const id = remove!.id; setRemove(null); void (async () => { try { setState(await api.del<ToolsState>(`/api/agent/tools/custom/${encodeURIComponent(id)}`)); } catch (e) { setError(e instanceof Error ? e.message : 'The tool could not be removed.'); } })(); }}>{tr('Remove')}</Btn></>}>
      <p>{remove ? tr('“{0}” will no longer be available to the assistant.', [remove.name]) : null}</p>
    </Dialog>
  </div>;
}
