import {sessionTitle} from '../lib/session-title';
import { tr } from '../lib/i18n';
import { useNative } from '../lib/native';
import { modelFamilyIcon } from '../lib/model-icons';
import { TaskTree } from './TaskTree';
/** Quiet navigation, with the same destinations at every window size. */
import { useEffect, useId, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { FloatingMenu } from '../components/FloatingMenu';
import { QuitStrataDialog } from '../components/QuitStrata';
import { useAppMode } from '../lib/app-mode';
import { href, useLocation, type Route } from '../lib/router';
import { cx, relTime } from '../lib/format';
import { useMedia } from '../lib/responsive';
import { recentChats, recentChatDescription } from '../lib/recent-chats';
import type { Session } from '../lib/types';
import { MOD, runCommand } from '../lib/commands';
import { bindingFor, bindingKeys, useBindings } from '../lib/shortcut-bindings';
import { Btn, Dialog, Icon, IconBtn, type IconName } from '../components/ui';
import { Brand } from '../components/Brand';
import { addWorkspace, deleteSession, go, isUnread, pickWorkspace, renameSession, removeWorkspace, renameWorkspace, selectSession, setActiveWorkspace, setNavOpen, togglePin, toggleRail, setTheme, useStore } from '../lib/store';
export function Sidebar({ containerRef }: {
    containerRef: RefObject<HTMLElement | null>;
}) {
    useBindings();
    const location = useLocation();
    const compact = useMedia('(max-width: 1279px)');
    const rail = useStore(s => s.rail);
    const theme = useStore(s => s.theme);
    const systemDark = useMedia('(prefers-color-scheme: dark)');
    const dark = theme === 'dark' || theme === 'system' && systemDark;
    const navOpen = useStore(s => s.navOpen);
    const collapsed = compact ? !navOpen : rail;
    const sessions = useStore(s => s.sessions);
    const library = useStore(s => s.library);
    const monitor = useStore(s => s.monitor);
    const workspaces = useStore(s => s.workspaces);
    const activeWorkspace = useStore(s => s.activeWorkspace);
    const activeSession = useStore(s => s.activeSession);
    const [query, setQuery] = useState('');
    const [searchOpen, setSearchOpen] = useState(false);
    const searchInput = useRef<HTMLInputElement>(null);
    const searchTrigger = useRef<HTMLButtonElement>(null);
    useEffect(() => {
        if (searchOpen)
            searchInput.current?.focus();
    }, [searchOpen, collapsed]);
    function closeSearch() { setQuery(''); setSearchOpen(false); requestAnimationFrame(() => searchTrigger.current?.focus()); }
    const [recentOpen, setRecentOpen] = useState(false);
    const [includeArchived, setIncludeArchived] = useState(false);
    const [removeWorkspaceId, setRemoveWorkspaceId] = useState<string | null>(null);
    const [menuId, setMenuId] = useState<string | null>(null);
    const [renameId, setRenameId] = useState<string | null>(null);
    const [renameWsId, setRenameWsId] = useState<string | null>(null), [wsName, setWsName] = useState('');
    const [name, setName] = useState('');
    const [deleteId, setDeleteId] = useState<string | null>(null);
    const [folders, setFolders] = useState<Record<string, boolean>>({});
    const menuTrigger = useRef<HTMLElement | null>(null);
    const current = library?.variants.find(v => v.id === library.loaded);
    const native = useNative(), nativeCurrent = native?.instances.find(i => i.id === native.defaultId);
    const modelName = nativeCurrent?.label ?? (current?.local ? current.name : current?.family) ?? tr('No model loaded');
    // The version from the catalog and the card it runs on ("Q2_0 · GPU 2"), not the file id.
    const nativeSize = nativeCurrent ? native?.catalog?.families.flatMap(f => f.sizes).find(s => s.id === nativeCurrent.model) : undefined;
    const nativeSecond = nativeCurrent ? [nativeSize?.model ?? nativeCurrent.model.toUpperCase(), nativeCurrent.gpus.length ? nativeCurrent.gpus.map(g => `GPU ${g + 1}`).join(' + ') : ''].filter(Boolean).join(' · ') : null;
    const modelQuant = nativeSecond ?? current?.quant ?? tr('View model versions');
    const phase = nativeCurrent ? (nativeCurrent.status === 'ready' ? nativeCurrent.phase : nativeCurrent.status) : monitor?.phase ?? 'idle';
    const status = phase === 'unavailable' ? 'Status unavailable' : phase === 'error' ? 'Error' : phase === 'loading' ? 'Loading' : !current && !nativeCurrent ? 'No model loaded' : phase === 'reading' ? 'Reading prompt' : phase === 'thinking' ? 'Thinking' : phase === 'generating' ? 'Generating' : 'Ready';
    const menuSession = sessions.find(s => s.id === menuId);
    const q = query.trim().toLowerCase();
    const matching = sessions.filter(s => (!s.archived || q || includeArchived) && (!q || `${sessionTitle(s)} ${s.messages.map(m => m.content).join(' ')}`.toLowerCase().includes(q))).sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned) || b.updatedAt.localeCompare(a.updatedAt));
    function openMenu(id: string, trigger: HTMLElement) { menuTrigger.current = trigger; setMenuId(id); }
    function closeMenu() { setMenuId(null); menuTrigger.current?.focus(); }
    return <>
    <aside className="nav" ref={containerRef} aria-label={tr("Main navigation")} aria-modal={navOpen && compact ? true : undefined} role={navOpen && compact ? 'dialog' : undefined} tabIndex={-1}>
      <div className="nav__top">{tr(!collapsed ? <a className="nav__brand" href={href('chat')} aria-label={tr("Strata — open chat")}><Brand variant="lockup" height={28} decorative className="nav__brand-lockup"/></a> : null)}<button type="button" className="nav__collapse" aria-label={tr(collapsed ? 'Show sidebar' : 'Hide sidebar')} title={tr(collapsed ? 'Show sidebar' : 'Hide sidebar')} onClick={() => compact ? setNavOpen(!navOpen) : toggleRail()}><Icon name="panel" size={20}/></button></div>
      <div className="nav__scroll">
        <button className="st-btn st-btn--primary nav-new" aria-label={tr("New chat")} title={tr("New chat")} onClick={() => runCommand('new')}><Icon name="new-chat" size={18}/><span className="nav__rail-hide">{tr("New chat")}</span></button>
        <TaskTree collapsed={collapsed}/>
        {tr(collapsed ? <div className="nav-recent" aria-label={tr("Recent chats")}>{tr(recentChats(sessions, activeSession).map((session, index) => <RecentChatLink key={session.id} session={session} slot={index + 1} active={activeSession === session.id && location.route === 'chat'}/>))}</div> : null)}
        {!collapsed ? <div className="ws">
          <div className="ws__head">{tr(searchOpen ? <div className="nav-search"><Icon name="search" size={16}/><input ref={searchInput} className="st-input st-input--bare" type="search" placeholder={tr("Search chats…")} aria-label={tr("Search chats")} value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => {
                    if (e.key === 'Escape') {
                        e.preventDefault();
                        closeSearch();
                    }
                }}/><IconBtn name="x" title={tr("Close chat search")} onClick={closeSearch}/></div> : <><span className="navgroup__label">{tr("Workspaces")}</span><div className="ws__tools"><button ref={searchTrigger} type="button" className="st-btn st-btn--icon" aria-label={tr("Search chats")} title={tr("Search chats")} onClick={() => setSearchOpen(true)}><Icon name="search" size={18}/></button><IconBtn name="folder-plus" title={tr("Add a workspace folder")} onClick={() => { void pickWorkspace().then(path => path ? addWorkspace(path) : undefined); }}/></div></>)}</div>
          {workspaces.map(w => {
                const expanded = folders[w.id] !== false || !!q;
                const rows = matching.filter(s => s.workspace === w.path || !!w.aliases?.includes(s.workspace));
                const shown = q ? rows : rows.slice(0, 20);
                return <div className="ws__group" key={w.id}>
              <div className={cx('ws__row', activeWorkspace === w.id && 'is-active')}><button className="ws__name" aria-expanded={expanded} onClick={() => { setActiveWorkspace(w.id); setFolders(f => ({ ...f, [w.id]: !expanded })); }}><Icon name="chevron" size={14} className={cx('ws__chev', expanded && 'is-open')}/><Icon name="folder" size={17}/><span>{w.name}</span></button><IconBtn name="pencil" title={tr("Rename {0}", [w.name])} onClick={() => { setWsName(w.name); setNavOpen(false); setRenameWsId(w.id); }}/>{tr(w.id !== 'root' ? <IconBtn name="x" title={tr("Remove {0} from sidebar", [w.name])} onClick={() => { setNavOpen(false); setRemoveWorkspaceId(w.id); }}/> : null)}</div>
              {tr(expanded ? <div className="navsessions">
                {tr(shown.map(s => <div key={s.id}>
                  <div className="ws__session row"><a className="navsession" href={href('chat', s.id)} aria-current={activeSession === s.id && location.route === 'chat' ? 'page' : undefined} onClick={() => selectSession(s.id)} onContextMenu={e => { e.preventDefault(); openMenu(s.id, e.currentTarget); }} title={sessionTitle(s)}><span className="navsession__title">{sessionTitle(s)}</span>{tr(s.pinned ? <span aria-label={tr("Pinned")}><Icon name="pin" size={12}/></span> : <span className="navsession__time">{tr(relTime(s.updatedAt))}</span>)}{tr(isUnread(s, activeSession) ? <span className="navsession__unread" aria-label={tr("Unread")}/> : null)}{tr(s.archived ? <span className="xs muted">{tr("Archived")}</span> : null)}</a><button className="st-btn st-btn--icon session-overflow" aria-label={tr("Actions for {0}", [sessionTitle(s)])} onClick={e => openMenu(s.id, e.currentTarget)}><Icon name="ellipsis" size={16}/></button></div>
                </div>))}
                {tr(!shown.length ? <p className="small muted ws__empty">{tr(q ? 'No matching chats' : 'No chats yet')}</p> : null)}
                {tr(!q && rows.length > shown.length ? <button className="ws-view small" onClick={() => { setNavOpen(false); setRecentOpen(true); }}>{tr("View all chats")}</button> : null)}
              </div> : null)}
            </div>;
            })}
        </div> : null}
      </div>
      <div className="nav__foot">
        <button className="enginechip" data-state={status === 'Ready' ? 'ready' : ['Reading prompt', 'Thinking', 'Generating', 'Loading'].includes(status) ? 'busy' : 'off'} onClick={() => go('models')} title={tr("{0} \u00B7 {1} \u00B7 {2}", [modelName, modelQuant, tr(status)])} aria-label={tr("{0} \u00B7 {1}", [modelName, modelQuant])}><span className="enginechip__family"><Icon name={modelFamilyIcon(current?.family)} size={20}/><span className="enginechip__dot"/></span><span className="enginechip__text nav__rail-hide"><span className="enginechip__title">{modelName}</span><span className="enginechip__sub">{modelQuant}</span></span></button>
        <NavLink route="skills" label={tr("Skills")} icon="skills" current={location.route}/>
        <NavLink route="monitor" label={tr("Monitor")} icon="activity" current={location.route}/>
        <NavLink route="models" label={tr("Models")} icon="models" current={location.route === 'tuning' ? 'models' : location.route}/>
        <div className="nav__bottom"><MoreMenu current={location.route}/><button className="st-btn st-btn--icon theme-switch" aria-label={tr(dark ? 'Switch to light mode' : 'Switch to dark mode')} title={tr(dark ? 'Switch to light mode' : 'Switch to dark mode')} onClick={() => setTheme(dark ? 'light' : 'dark')}><span key={String(dark)} className="theme-switch__icon"><Icon name={dark ? 'sun' : 'moon'} size={19}/></span></button></div>
      </div>
    </aside>
    {tr(menuId && menuSession ? <FloatingMenu anchor={menuTrigger.current} title={tr("Actions for {0}", [sessionTitle(menuSession)])} onClose={closeMenu}>
      <button role="menuitem" className="menu__item" onClick={() => { setName(sessionTitle(menuSession)); setNavOpen(false); setRenameId(menuId); setMenuId(null); }}><Icon name="pencil"/>{tr("Rename")}</button>
      <button role="menuitem" className="menu__item" onClick={() => { togglePin(menuId); closeMenu(); }}><Icon name="pin"/>{tr(menuSession.pinned ? 'Unpin' : 'Pin')}</button>
      <button role="menuitem" className="menu__item menu__item--danger" onClick={() => { setNavOpen(false); setDeleteId(menuId); setMenuId(null); }}><Icon name="trash"/>{tr("Delete chat")}</button>
    </FloatingMenu> : null)}
    <Dialog open={!!renameId} compact title={tr("Rename chat")} onClose={() => setRenameId(null)} actions={<><Btn onClick={() => setRenameId(null)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!name.trim()} onClick={() => {
                if (renameId)
                    renameSession(renameId, name.trim());
                setRenameId(null);
            }}>{tr("Save name")}</Btn></>}><label className="input-field" htmlFor="chat-name"><span>{tr("Chat name")}</span><input className="st-input" id="chat-name" value={name} onFocus={e => e.currentTarget.select()} onChange={e => setName(e.target.value)}/></label></Dialog>
    <Dialog open={!!renameWsId} compact title={tr("Rename workspace")} onClose={() => setRenameWsId(null)} actions={<><Btn onClick={() => setRenameWsId(null)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!wsName.trim()} onClick={() => {
                if (renameWsId)
                    void renameWorkspace(renameWsId, wsName.trim());
                setRenameWsId(null);
            }}>{tr("Save name")}</Btn></>}><label className="input-field" htmlFor="workspace-name"><span>{tr("Workspace name")}</span><input className="st-input" id="workspace-name" value={wsName} onFocus={e => e.currentTarget.select()} onChange={e => setWsName(e.target.value)}/></label><p className="small muted">{tr("Only the name changes. The folder stays where it is.")}</p></Dialog>
    <Dialog open={!!deleteId} compact title={tr("Delete this chat?")} onClose={() => setDeleteId(null)} actions={<><Btn onClick={() => setDeleteId(null)}>{tr("Keep chat")}</Btn><Btn variant="danger" onClick={() => {
                if (deleteId)
                    deleteSession(deleteId);
                setDeleteId(null);
            }}>{tr("Delete chat")}</Btn></>}><p>{tr("This chat will be permanently deleted.")}</p></Dialog>
    <Dialog open={!!removeWorkspaceId} title={tr("Remove workspace from sidebar?")} onClose={() => setRemoveWorkspaceId(null)} actions={<><Btn onClick={() => setRemoveWorkspaceId(null)}>{tr("Keep workspace")}</Btn><Btn onClick={() => {
                if (removeWorkspaceId)
                    void removeWorkspace(removeWorkspaceId).then(() => setRemoveWorkspaceId(null));
            }}>{tr("Remove from sidebar")}</Btn></>}><p>{tr("Files stay on your computer. Add this folder again to reopen its chats.")}</p></Dialog>
    <Dialog open={recentOpen} title={tr("All chats")} onClose={() => setRecentOpen(false)}><div className="stack"><label className="row small"><input type="checkbox" checked={includeArchived} onChange={e => setIncludeArchived(e.target.checked)}/>{tr("Include archived chats")}</label><input type="search" className="st-input" placeholder={tr("Search chats…")} aria-label={tr("Search all chats")} value={query} onChange={e => setQuery(e.target.value)}/>{tr(matching.map(s => <button className="recent-chat" key={s.id} onClick={() => { selectSession(s.id); go('chat', s.id); setRecentOpen(false); }}><span>{sessionTitle(s)}</span><span className="xs muted">{tr(relTime(s.updatedAt))}</span></button>))}{tr(!matching.length ? <p className="small muted">{tr("No matching chats. Try another search.")}</p> : null)}</div></Dialog>
  </>;
}
function RecentChatLink({ session, active, slot }: {
    session: Session;
    active: boolean;
    slot: number;
}) {
    const binding = bindingFor('chat:slot:' + slot, { key: String(slot), mod: true, shift: false, alt: false });
    const keys = bindingKeys(binding, MOD);
    const anchor = useRef<HTMLAnchorElement>(null);
    const tooltip = useRef<HTMLDivElement>(null);
    const id = useId();
    const [open, setOpen] = useState(false);
    const [position, setPosition] = useState({ left: 0, top: 0 });
    useLayoutEffect(() => {
        if (!open)
            return;
        const place = () => {
            const rect = anchor.current?.getBoundingClientRect();
            if (!rect)
                return;
            const height = tooltip.current?.getBoundingClientRect().height ?? 80;
            setPosition({ left: rect.right + 12, top: Math.max(12, Math.min(rect.top, window.innerHeight - height - 12)) });
        };
        place();
        window.addEventListener('resize', place);
        window.addEventListener('scroll', place, true);
        return () => { window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true); };
    }, [open]);
    return <>
    <a ref={anchor} className="navitem nav-recent__chat" href={href('chat', session.id)} aria-label={sessionTitle(session)} aria-current={active ? 'page' : undefined} aria-describedby={open ? id : undefined} onClick={() => { setOpen(false); selectSession(session.id); }} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onKeyDown={event => {
            if (event.key === 'Escape' && open) {
                event.preventDefault();
                event.stopPropagation();
                setOpen(false);
            }
        }}><Icon name="chat"/><span className="nav-chat-number" aria-hidden="true">{tr(slot)}</span></a>
    {open ? createPortal(<div ref={tooltip} id={id} role="tooltip" className="nav-chat-tooltip" style={position}><strong>{sessionTitle(session)}</strong><span>{session.messages.length ? recentChatDescription(session) : tr(recentChatDescription(session))}</span>{tr(binding ? <kbd className="nav-chat-shortcut">{tr(keys.join(" "))}</kbd> : null)}</div>, document.body) : null}
  </>;
}
function NavLink({ route, label, icon, current, meta }: {
    route: Route;
    label: string;
    icon: IconName;
    current: string;
    meta?: string;
}) {
    return <a className="navitem" href={href(route)} aria-current={current === route ? 'page' : undefined} title={tr(label)} aria-label={tr(label)}><Icon name={icon}/><span className="navitem__text">{tr(label)}</span>{tr(meta ? <span className="navitem__meta">{tr(meta)}</span> : null)}</a>;
}
function MoreMenu({ current }: {
    current: string;
}) {
    const [open, setOpen] = useState(false);
    const [quitOpen, setQuitOpen] = useState(false);
    const appMode = useAppMode();
    const ref = useRef<HTMLButtonElement>(null);
    return <><button ref={ref} className="navitem nav__more" aria-label={tr("More — Serving and Settings")} aria-haspopup="menu" aria-expanded={open} data-active={['serving', 'settings'].includes(current) || undefined} onClick={() => setOpen(v => !v)}><Icon name="ellipsis"/><span className="navitem__text">{tr("More")}</span></button>{tr(open ? <FloatingMenu anchor={ref.current} title={tr("More")} onClose={() => { setOpen(false); ref.current?.focus(); }}>{tr(([{ route: 'serving', label: 'Serving', icon: 'link' }, { route: 'settings', label: 'Settings', icon: 'settings' }] as const).map(d => <a key={d.route} role="menuitem" className="menu__item" href={href(d.route)} onClick={() => setOpen(false)}><Icon name={d.icon}/>{tr(d.label)}</a>))}{appMode.canQuit ? <button type="button" role="menuitem" className="menu__item" onClick={() => { setOpen(false); setNavOpen(false); setQuitOpen(true); }}><Icon name="x"/>{tr("Quit Strata")}</button> : null}</FloatingMenu> : null)}<QuitStrataDialog open={quitOpen} onClose={() => setQuitOpen(false)}/></>;
}
