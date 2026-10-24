import { bindingFor, bindingEqual, eventBinding, type Binding } from './shortcut-bindings';
/** Shared action descriptors for keyboard shortcuts, palette and chat commands. */
import type { IconName } from '../components/ui';
import type { Route } from './router';
import {
  exportConversation, getState, go, newSession, pushToast,
  selectSession,
  restartEngine, setFocus, setNavOpen, setPalette, setShortcuts, setTheme,
  toggleFocus, toggleRail, unloadModel,
} from './store';
import { recentChats } from './recent-chats';

export const MOD = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl';
interface AppCommand {
  id: string;
  label: string;
  icon: IconName;
  group: 'Session' | 'Engine' | 'Navigation' | 'Model' | 'Chat';
  hint?: string;
  keywords?: string;
  key?: string;
  binding?: Binding | null;
  shift?: boolean;
  alt?: boolean;
  slash?: string[];
  palette?: boolean;
  danger?: boolean;
  available?: () => boolean;
  unavailable?: string;
  run: () => void;
}
export const DESTINATIONS: { route: Route; label: string; icon: IconName; keywords: string; key: string; slash?: string }[] = [
  { route: 'chat', label: 'Chat', icon: 'chat', keywords: 'conversation prompt ask', key: '1' },
  { route: 'monitor', label: 'Monitor', icon: 'activity', keywords: 'stats performance speed hardware', key: '2', slash: 'monitor' },
  { route: 'serving', label: 'Serving', icon: 'link', keywords: 'api server endpoint openai anthropic port', key: '3', slash: 'serve' },
  { route: 'models', label: 'Models', icon: 'models', keywords: 'download weights gguf choose tune engine parameters context sampling', key: '4', slash: 'model' },
  { route: 'tasks', label: 'Tasks', icon: 'clock', keywords: 'schedule recurring loop automation', key: '6', slash: 'tasks' },
  { route: 'skills', label: 'Skills', icon: 'skills', keywords: 'instructions prompts abilities', key: '7', slash: 'skills' },
  { route: 'settings', label: 'Settings', icon: 'settings', keywords: 'preferences theme appearance text size', key: '8' },
];
export const APP_COMMANDS: AppCommand[] = [
  { id: 'new', label: 'New chat', hint: 'Start a new conversation; keep this one', group: 'Session', key: 'n', icon: 'new-chat', slash: ['new'], run: () => { const s = newSession(); go('chat', s.id); } },
  { id: 'palette', label: 'Command palette', group: 'Session', key: 'k', icon: 'search', palette: false, run: () => setPalette(!getState().paletteOpen) },
  { id: 'sidebar', label: 'Show or hide sidebar', group: 'Session', key: 'b', icon: 'panel', run: () => window.innerWidth < 1280 ? setNavOpen(!getState().navOpen) : toggleRail() },
  { id: 'settings', label: 'Settings', group: 'Session', key: ',', icon: 'settings', palette: false, run: () => go('settings') },
  { id: 'shortcuts', label: 'Keyboard shortcuts', keywords: 'keys bindings help', group: 'Session', key: '/', icon: 'bolt', run: () => setShortcuts(!getState().shortcutsOpen) },
  { id: 'dictation', label: 'Start or stop dictation', hint: 'Speak into the message box without sending', keywords: 'speech microphone voice transcribe', group: 'Session', key: 'd', shift: true, icon: 'mic', available: () => window.location.hash.startsWith('#/chat') && !getState().streaming, unavailable: 'Open a chat and wait for the current response to finish.', run: () => window.dispatchEvent(new Event('strata:dictation')) },
  { id: 'restart', label: 'Restart engine', hint: 'Reloads the model into memory', keywords: 'reboot refresh', group: 'Engine', key: 'r', shift: true, icon: 'refresh', danger: true, run: () => void restartEngine() },
  { id: 'unload', label: 'Unload model', hint: 'Frees model memory', keywords: 'eject stop vram', group: 'Engine', key: 'u', shift: true, icon: 'stop', danger: true, available: () => !!getState().library?.loaded, unavailable: 'No model is loaded.', run: () => void unloadModel() },
  { id: 'go:tuning', label: 'Model settings', group: 'Navigation', icon: 'engine', key: '5', alt: true, palette: false, slash: ['tune'], run: () => go('models') },
  { id: 'monitor', label: 'Open Monitor', group: 'Engine', key: 'm', shift: true, icon: 'activity', palette: false, run: () => go('monitor') },
  { id: 'focus', label: 'Focus on the conversation', group: 'Session', key: 'f', shift: true, icon: 'chat', run: () => { go('chat', getState().activeSession ?? undefined); toggleFocus(); } },
  { id: 'export', label: 'Export conversation', hint: 'Download the active conversation as Markdown', keywords: 'save download markdown', group: 'Session', icon: 'download', slash: ['export'], available: () => getState().sessions.some(s => s.id === getState().activeSession && s.messages.length > 0), unavailable: 'Open a conversation with messages first.', run: () => exportConversation() },
  { id: 'setup', label: 'Run setup', hint: 'Check computer, pick a version and load', keywords: 'wizard onboarding install resume', group: 'Engine', icon: 'bolt', slash: ['setup'], run: () => go('setup') },
];

/** All model version changes happen on the Model screen. */
export function getCommands(): AppCommand[] {
  const state = getState();
  const dark = state.theme === 'dark' || (state.theme === 'system' && matchMedia('(prefers-color-scheme: dark)').matches);
  const commands: AppCommand[] = [
    ...APP_COMMANDS,
    ...Array.from({ length: 6 }, (_, index) => ({ id: `chat:slot:${index + 1}`, label: `Chat ${index + 1}`, icon: 'chat' as const, group: 'Chat' as const, key: String(index + 1), hint: recentChats(state.sessions, state.activeSession)[index]?.title, available: () => !!recentChats(getState().sessions, getState().activeSession)[index], unavailable: 'No saved chat in this position.', run: () => { const session = recentChats(getState().sessions, getState().activeSession)[index]; if (session) { selectSession(session.id); go('chat', session.id); } } })),
    ...DESTINATIONS.map(d => ({ id: `go:${d.route}`, label: d.label, icon: d.icon, keywords: d.keywords, key: d.key, alt: true, group: 'Navigation' as const, slash: d.slash ? [d.slash] : undefined, run: () => go(d.route) })),
    { id: 'theme', label: dark ? 'Switch to light theme' : 'Switch to dark theme', keywords: 'appearance colours', icon: dark ? 'sun' : 'moon', group: 'Session', run: () => setTheme(dark ? 'light' : 'dark') },

  ];
  return commands.map(c=>({...c,binding:bindingFor(c.id,c.key?{key:c.key,mod:true,shift:!!c.shift,alt:!!c.alt}:null)}));
}
export function runCommand(command: AppCommand | string) {
  const action = typeof command === 'string' ? getCommands().find(c => c.id === command) : command;
  if (!action) return;
  if (action.available && !action.available()) { pushToast('error', action.unavailable ?? 'This action is unavailable.'); return; }
  action.run();
}
export function getSlashCommands() {
  return getCommands().filter(c => !c.available || c.available()).flatMap(c => (c.slash ?? []).map(alias => ({
    id: `${c.id}:${alias}`, label: `/${alias}`, hint: c.hint ?? `Open ${c.label}`, icon: c.icon, run: () => runCommand(c),
  })));
}
export function handleAppShortcut(e: KeyboardEvent) {
  if (e.defaultPrevented || e.isComposing || getState().shortcutsOpen) return;
  const meta = e.metaKey || e.ctrlKey;
  const typing = (e.target as HTMLElement)?.matches('input, textarea, [contenteditable="true"]');
  if (!meta) {
    if (e.key === '?' && !typing) { e.preventDefault(); setShortcuts(true); }
    if (e.key === 'Escape') {
      const state = getState();
      if (state.paletteOpen) setPalette(false);
      else if (state.shortcutsOpen) setShortcuts(false);
      else if (state.navOpen) setNavOpen(false);
      else if (state.focus) setFocus(false);
    }
    return;
  }
  const command = getCommands().find(c => bindingEqual(c.binding??null,eventBinding(e)));
  if (command) { e.preventDefault(); runCommand(command); }
}
