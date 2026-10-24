import { tr } from '../lib/i18n';
/** App shell: routes, setup takeover, palette, toasts. */
import { useNative } from '../lib/native';
import { Component, useEffect, useRef, useState, type ReactNode } from 'react';
import { useSurfaceFocus } from '../lib/focus';
import { closeSettings, useLocation } from '../lib/router';
import { bootOnce, bootstrap, setNavOpen, setFocus, useStore, } from '../lib/store';
import { Sidebar } from './Sidebar';
import { CommandPalette } from './CommandPalette';
import { ShortcutsDialog } from './Shortcuts';
import { FolderPathDialog } from '../components/FolderPathDialog';
import { handleAppShortcut, runCommand } from '../lib/commands';
import { useMedia } from '../lib/responsive';
import { Btn, Dialog, IconBtn, Toasts } from '../components/ui';
import { Brand } from '../components/Brand';
import { SetupScreen } from '../screens/Setup';
import { ChatScreen } from '../screens/Chat';
import { ModelRunScreen } from '../screens/ModelRun';
import { ServingScreen } from '../screens/Serving';
import { MonitorScreen } from '../screens/Monitor';
import { TasksScreen } from '../screens/Tasks';
import { SkillsScreen } from '../screens/Skills';
import { SettingsScreen } from '../screens/Settings';
function AppShell() {
    useStore(s => s.language);
    const location = useLocation();
    const rail = useStore((s) => s.rail);
    const ready = useStore((s) => s.ready);
    const setup = useStore((s) => s.setup);
    const paletteOpen = useStore((s) => s.paletteOpen);
    const focus = useStore((s) => s.focus);
    const firstRunOpen = useStore((s) => s.setupOpen);
    const setupRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        bootOnce();
    }, []);
    useEffect(() => {
        window.addEventListener('keydown', handleAppShortcut);
        return () => window.removeEventListener('keydown', handleAppShortcut);
    }, []);
    const compact = useMedia('(max-width: 1279px)');
    const mobile = useMedia('(max-width: 767px)');
    const navOpen = useStore(s => s.navOpen);
    const navRef = useRef<HTMLElement>(null);
    useSurfaceFocus(navOpen && compact, navRef);
    useEffect(() => { setNavOpen(false); }, [location.route, location.param, compact]);
    // First run is a modal over the working shell rather than a takeover: the app
    // is visibly there behind it, which is far less daunting than a blank window
    // demanding five steps before anything appears.
    const setupOpen = location.route === 'setup' || (!!setup && !setup.complete && !setup.dismissed) || firstRunOpen;
    // Hook runs here, after setupOpen exists — see src/lib/focus.ts.
    useSurfaceFocus(setupOpen, setupRef);
    return (<div className="app" data-rail={compact ? !navOpen : rail} data-layout={mobile ? 'mobile' : compact ? 'compact' : 'desktop'} data-nav-open={navOpen || undefined} data-focus={focus && location.route === 'chat' || undefined}>
      <Sidebar containerRef={navRef}/>
      {tr(navOpen && compact ? <button className="nav-scrim" tabIndex={-1} aria-hidden="true" aria-label={tr("Close navigation")} onClick={() => setNavOpen(false)}/> : null)}
      <main className="main">
        <div className="mobile-bar">
          <IconBtn name="panel" title={tr("Open navigation")} onClick={() => setNavOpen(true)}/>
          <Brand variant="lockup" height={24}/>
          <span className="spacer"/>
          <IconBtn name="new-chat" title={tr("New chat")} onClick={() => runCommand('new')}/>
        </div>
        <ConnectionBar />
        {tr(focus && location.route === 'chat' ? <FocusBar /> : null)}
        {tr(!ready ? <BootSkeleton /> : <RouteView route={location.route}/>)}
      </main>
      <CommandPalette open={paletteOpen}/>
      <Dialog open={location.query.get('settings') === '1'} title={tr("Settings")} className="settings-dialog" onClose={closeSettings}><SettingsScreen /></Dialog>
      <ShortcutsDialog />
      <FolderPathDialog />
      {tr(setupOpen ? (<div className="setupmodal" role="dialog" aria-modal="true" aria-label={tr("Set up Strata")}>
          <div className="scrim" data-open="true" aria-hidden="true"/>
          <div className="setupmodal__surface" ref={setupRef}>
            <SetupScreen />
          </div>
        </div>) : null)}
      <Toasts />
    </div>);
}
/**
 * Says when the host is not answering.
 *
 * `connection` has been tracked since the beginning and never rendered, so a
 * dead host looked exactly like an idle one: every panel simply empty, with no
 * statement that anything was wrong. For someone who does not know what a host
 * process is, "nothing works and nothing says why" is the worst possible state.
 */
function ConnectionBar() {
    const connection = useStore((s) => s.connection);
    const [retrying, setRetrying] = useState(false);
    if (connection === 'live')
        return null;
    const offline = connection === 'offline';
    return (<div className="connbar" data-state={connection} role="status">
      <span className="connbar__dot"/>
      <span className="connbar__text">
        {tr(offline ? (<>
            <strong>{tr("Strata isn\u2019t connected to its engine.")}</strong>{tr(" Your conversation and drafts are kept while it reconnects.")}</>) : (<>{tr("Connecting to the local engine\u2026")}</>))}
      </span>
      {tr(offline ? (<Btn variant="secondary" size="sm" icon="refresh" disabled={retrying} onClick={() => {
                setRetrying(true);
                void bootstrap().finally(() => setRetrying(false));
            }}>
          {tr(retrying ? 'Retrying…' : 'Retry')}
        </Btn>) : null)}
      {tr(offline ? <details className="connection-details"><summary>{tr("Connection details")}</summary><p className="xs">{tr("App host: ")}{tr(window.location.origin)}{tr(". Retained readings may be stale. Start the configured engine host, then retry.")}</p></details> : null)}
    </div>);
}
/**
 * The only chrome focus mode keeps: what is loaded, what it is doing, and the
 * way out. Everything else on this screen is the conversation.
 */
function FocusBar() {
    const native = useNative(), nativeCurrent = native?.instances.find(i => i.id === native.defaultId);
    const library = useStore((s) => s.library);
    const monitor = useStore((s) => s.monitor);
    const model = library?.variants.find((v) => v.id === library.loaded) ?? null;
    const phase = nativeCurrent ? (nativeCurrent.status === 'ready' ? nativeCurrent.phase : nativeCurrent.status) : monitor?.phase ?? 'idle';
    const state = ['error', 'unavailable'].includes(phase) ? 'idle' : phase === 'generating' || phase === 'reading' || phase === 'thinking' ? 'busy' : model || nativeCurrent ? 'ready' : 'idle';
    return (<div className="focusbar" data-state={state}>
      <span className="focusbar__dot"/>
      <span className="strong" style={{ color: 'var(--st-ink)' }}>
        {tr(nativeCurrent?.label ?? model?.label ?? 'No model loaded')}
      </span>
      <span>
        {tr(phase === 'reading'
            ? 'reading prompt'
            : phase === 'generating'
                ? `generating · ${(nativeCurrent ? nativeCurrent.metrics?.live?.tok_s ?? 0 : monitor?.outputTps ?? 0).toFixed(1)} tok/s`
                : phase === 'thinking'
                    ? 'thinking'
                    : phase === 'error' ? 'engine error' : phase === 'unavailable' ? 'status unavailable' : phase === 'loading' ? 'loading' : model || nativeCurrent ? 'ready' : 'no model loaded')}
      </span>
      <span className="spacer"/>
      {/* Grouped, so each hint reads as one instruction rather than a row of
            loose keycaps. */}
      <span className="focusbar__keys">
        <span>
          {/* A chord reads as one key: the two caps sit tight together, with
            the space before the label doing the separating. */}
          <span className="focusbar__chord">
            <kbd>⌥↑</kbd>
            <kbd>⌥↓</kbd>
          </span>{tr("turns")}</span>
        <span>
          <kbd>⌘↓</kbd>{tr(" latest")}</span>
        <span>
          <kbd>{tr("Esc")}</kbd>{tr(" leave")}</span>
      </span>
      {/* The quietest control in the bar. It was the loudest thing in a mode
            whose whole purpose is to remove chrome. */}
      <IconBtn name="x" title={tr("Leave focus mode")} onClick={() => setFocus(false)}/>
    </div>);
}
function RouteView({ route }: {
    route: string;
}) {
    switch (route) {
        case 'chat':
            return <ChatScreen />;
        case 'models':
        case 'tuning':
            return <ModelRunScreen />;
        case 'serving':
            return <ServingScreen />;
        case 'monitor':
            return <MonitorScreen />;
        case 'tasks':
            return <TasksScreen />;
        case 'skills':
            return <SkillsScreen />;
        default:
            return <ChatScreen />;
    }
}
function BootSkeleton() {
    return (<>
      <header className="main__head">
        <div className="col" style={{ gap: 8 }}>
          <div className="skel" style={{ width: 180, height: 24 }}/>
          <div className="skel" style={{ width: 320, height: 14 }}/>
        </div>
      </header>
      <div className="main__body">
        <div className="wrap stack">
          <div className="skel" style={{ height: 20, width: '60%' }}/><div className="skel" style={{ height: 20, width: '80%' }}/><div className="skel" style={{ height: 20, width: '70%' }}/>
        </div>
      </div>
    </>);
}
class AppBoundary extends Component<{
    children: ReactNode;
}, {
    failed: boolean;
}> {
    state = { failed: false };
    static getDerivedStateFromError() { return { failed: true }; }
    render() {
        if (!this.state.failed)
            return this.props.children;
        return <div className="app-recovery"><h1 className="h2">{tr("Strata couldn\u2019t show this screen")}</h1><p>{tr("Your saved conversations and settings are still available.")}</p><Btn variant="primary" onClick={() => { void bootstrap().then(() => this.setState({ failed: false })); }}>{tr("Retry")}</Btn></div>;
    }
}
export function App() { return <AppBoundary><AppShell /></AppBoundary>; }
