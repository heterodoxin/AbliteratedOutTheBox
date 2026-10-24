import {sessionTitle} from '../lib/session-title';
import { tr } from '../lib/i18n';
import { bindingKeys, useBindings } from '../lib/shortcut-bindings';
/** ⌘K palette — jump to a screen or run an engine action. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useSurfaceFocus } from '../lib/focus';
import { cx } from '../lib/format';
import { go, selectSession, setActiveWorkspace, setPalette, useStore } from '../lib/store';
import { highlight, parseScope, rank } from '../lib/fuzzy';
import { Icon, type IconName } from '../components/ui';
import { getCommands, runCommand, MOD } from '../lib/commands';
interface Command {
    id: string;
    label: string;
    hint?: string;
    icon: IconName;
    /** Marks an action that changes or removes something, so it is not mistaken
     *  for navigation. */
    danger?: boolean;
    /** The keyboard binding, shown so the palette teaches the shortcut. */
    keys?: string[];
    /** Which scope this belongs to; `>` narrows to commands. */
    scope: 'command' | 'session' | 'workspace';
    /** Extra words that should find this entry. The label stays human; aliases
     *  cover what people actually type. */
    keywords?: string;
    group: 'Go to' | 'Session' | 'Engine' | 'Model' | 'Navigation' | 'Chat';
    run: () => void;
}
export function CommandPalette({ open }: {
    open: boolean;
}) {
    const bindings = useBindings();
    const language = useStore(s => s.language);
    const [query, setQuery] = useState('');
    const [cursor, setCursor] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);
    const library = useStore((s) => s.library);
    const theme = useStore((s) => s.theme);
    const loaded = library?.loaded ?? null;
    // Focus and query reset follow the open state now that the palette stays
    // mounted (so closing can animate rather than unmount mid-flight).
    // Focus, trapping and restoring are shared with every other overlay; see
    // src/lib/focus.ts for why a plain focus() call here did not work.
    const modalRef = useRef<HTMLDivElement>(null);
    useSurfaceFocus(open, modalRef, inputRef);
    useEffect(() => {
        if (!open) {
            setQuery('');
            setCursor(0);
        }
    }, [open]);
    const activeId = useStore(s => s.activeSession);
    const sessionList = useStore(s => s.sessions);
    const commands = useMemo<Command[]>(() => getCommands()
        .filter(c => c.palette !== false && (!c.available || c.available()))
        .map(c => ({ ...c, label: tr(c.label), hint: tr(c.hint), scope: 'command', keys: c.binding ? bindingKeys(c.binding, MOD) : undefined, run: () => runCommand(c) })), [library, theme, loaded, activeId, sessionList, bindings, language]);
    const sessions = useStore((s) => s.sessions);
    const workspaces = useStore((s) => s.workspaces);
    // Conversations and workspaces are reachable from here so the palette is a
    // real launcher rather than a screen switcher.
    const extras = useMemo<Command[]>(() => {
        const rows: Command[] = sessions.map((sess) => ({
            id: `session:${sess.id}`,
            label: sessionTitle(sess),
            icon: 'chat',
            group: 'Go to',
            scope: 'session',
            run: () => {
                selectSession(sess.id);
                go('chat', sess.id);
            },
        }));
        for (const w of workspaces) {
            rows.push({
                id: `workspace:${w.id}`,
                label: w.name,
                hint: w.path,
                icon: 'folder',
                group: 'Go to',
                scope: 'workspace',
                run: () => {
                    setActiveWorkspace(w.id);
                    go('chat');
                },
            });
        }
        return rows;
    }, [sessions, workspaces, language]);
    const all = useMemo(() => [...commands, ...extras], [commands, extras]);
    /** Ranked, scoped and explained: the match indices drive the highlighting. */
    const filtered = useMemo(() => {
        const { scope, term } = parseScope(query);
        const pool = !query.trim() ? [...commands.filter(c => ['new', 'go:models', 'go:monitor', 'go:tasks'].includes(c.id)), ...extras.filter(c => c.scope === 'session').slice(0, 4)] : scope === 'all' ? all : all.filter((c) => c.scope === scope);
        return rank(term, pool, (c) => [c.label, c.keywords ?? '', c.hint ?? '']).map(({ item, indices }) => ({
            ...item,
            indices,
        }));
    }, [all, query, commands, extras]);
    useEffect(() => {
        setCursor(0);
    }, [query]);
    const grouped = useMemo(() => {
        const map = new Map<string, (Command & {
            indices: number[];
        })[]>();
        for (const c of filtered) {
            const arr = map.get(c.group) ?? [];
            arr.push(c);
            map.set(c.group, arr);
        }
        return [...map.entries()];
    }, [filtered]);
    const flat = grouped.flatMap(([, items]) => items);
    const onKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setCursor((c) => Math.min(flat.length - 1, c + 1));
        }
        else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setCursor((c) => Math.max(0, c - 1));
        }
        else if (e.key === 'Enter') {
            e.preventDefault();
            setPalette(false);
            flat[cursor]?.run();
        }
    };
    let index = -1;
    return (<>
      <div className="scrim" data-open={open} onClick={() => setPalette(false)} aria-hidden="true"/>
      <div ref={modalRef} className="modal" data-open={open} role="dialog" aria-modal="true" aria-label={tr("Command palette")} inert={!open}>
        <div className="modal__head">
          <Icon name="chat" size={18} style={{ color: 'var(--st-ink-muted)' }}/>
          <input ref={inputRef} className="st-input st-input--bare" style={{ flex: 1 }} placeholder={tr("Search chats, screens or actions…")} aria-label={tr("Search commands")} aria-activedescendant={flat[cursor] ? `command-${flat[cursor].id}` : undefined} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={onKeyDown}/>
          <kbd className="kbd">{tr("esc")}</kbd>
        </div>
        <div style={{ overflowY: 'auto', padding: 8 }}>
          {tr(flat.length === 0 ? (<div className="small muted" style={{ padding: 18 }}>{tr("Nothing matches “{0}”. Try > for commands, # for conversations, or @ for workspaces.", [query])}</div>) : (grouped.map(([group, items]) => (<div key={group} style={{ marginBottom: 6 }}>
                <div className="navgroup__label" style={{ padding: '8px 12px 4px' }}>
                  {tr(group)}
                </div>
                {tr(items.map((c) => {
                index += 1;
                const active = index === cursor;
                const i = index;
                return (<button key={c.id} id={`command-${c.id}`} type="button" className={cx('navitem', 'palette__row')} data-danger={c.danger || undefined} aria-selected={active} style={{
                        height: 42,
                        background: active ? 'var(--st-surface-2)' : undefined,
                        boxShadow: active ? 'inset 0 0 0 1px var(--st-line)' : undefined,
                    }} onMouseEnter={() => setCursor(i)} onClick={() => {
                        setPalette(false);
                        c.run();
                    }}>
                      <Icon name={c.icon}/>
                      <span className="navitem__text">
                        {/* The matched characters are emphasised so a fuzzy hit
                explains itself instead of looking like a mistake. */}
                        {tr(highlight(c.label, c.indices).map((part, k) => part.hit ? (<mark key={k}>{part.text}</mark>) : (<span key={k}>{part.text}</span>)))}
                      </span>
                      {tr(c.hint ? <span className="navitem__meta">{c.hint}</span> : null)}
                      {tr(c.keys?.length ? (<span className="palette__keys">
                          {tr(c.keys.map((k) => (<kbd key={k}>{tr(k)}</kbd>)))}
                        </span>) : null)}
                    </button>);
            }))}
              </div>))))}
        </div>
      </div>
    </>);
}
