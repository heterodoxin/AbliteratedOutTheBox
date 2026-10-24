import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { UsageDashboard } from '../components/UsageDashboard';
import { FloatingMenu } from '../components/FloatingMenu';
import { TimeSkipPanel } from '../components/TimeSkipPanel';
import { timeSkipPrompt, timeSkipTabId, type TimeSkipRun } from '../lib/time-skip';
import { TimeSkipWorking } from '../components/TimeSkipWorking';
import { ChatToolbar } from '../components/ChatToolbar';
import { ChatTelemetry } from '../components/TelemetryPopover';
import { ArtifactCard } from '../components/ArtifactCard';
import { ChatMedia } from '../components/ChatMedia';
import { closePreviewTab } from '../lib/preview-tabs';
import { ArtifactPanel } from '../components/ArtifactPanel';
import { PreviewResizeHandle } from '../components/PreviewResizeHandle';
import { DEFAULT_PREVIEW_FRACTION, PREVIEW_SPLIT_KEY, previewWidth } from '../lib/preview-layout';
import { ComposerControls } from '../components/ComposerControls';
import { ChatPhotos } from '../components/ChatPhotos';
import { useChatPhotos, removeChatPhoto } from '../lib/chat-photos';
import { useNative } from '../lib/native';
import { mainResult, previewOf } from '../lib/deliverables';
import { noModelNotice } from '../lib/no-model';
import { ReasoningBlock } from '../components/ReasoningBlock';
import { StatusLine } from '../components/StatusLine';
import { GoalBar } from '../components/GoalBar';
import { PlanCard, PlanPanel, QueuedMessages } from '../components/PlanCard';
import { FolderChip } from '../components/FolderChip';
import { DropZone, DROP_EVENT, pastedFiles } from '../components/DropZone';
import { ActivityLine, EditedFiles, FileCard, SkillsUsed } from '../components/Deliverables';
import { queueMessage, usePlan, usePlanQuiet } from '../lib/plan';
import { plainFailureText } from '../lib/plain-errors';
import { contextRecovery } from '../lib/context-recovery';
import { useLiveStatus } from '../lib/live-status';
import { askQuestion, askSub, askIcon } from '../lib/ask';
import { artifactFromUrl, type PreviewArtifact } from '../lib/artifacts';
/** Chat — talk to the one loaded model. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Markdown } from '../components/Markdown';
import { Btn, Icon, IconBtn, ErrorNotice, type IconName } from '../components/ui';
import { composerCap, scrollBehavior } from '../lib/responsive';
import { relTime, cx, duration, tokens, tps } from '../lib/format';
import { createSubagentExample, pushToast, setPreviewTabClosed, branchFrom, continueLast, markRead, resolveTool, retryLast, selectSession, rewindTo, sendMessage, stopStreaming, useStore, } from '../lib/store';
import { getSlashCommands } from '../lib/commands';
import { useLocation } from '../lib/router';
import { composerSends } from '../lib/shortcut-bindings';
import type { ChatMessage, Session, ToolCall } from '../lib/types';
export function ChatScreen() {
    const location = useLocation();
    const sessions = useStore((s) => s.sessions);
    const activeId = useStore((s) => s.activeSession);
    const library = useStore((s) => s.library);
    const streaming = useStore((s) => s.streaming);
    const setup = useStore(s => s.setup);
    const setupFlag = useStore(s => s.setupOpen);
    const setupVisible = location.route === 'setup' || setupFlag || !!setup && !setup.complete && !setup.dismissed;
    const session: Session | null = useMemo(() => {
        const wanted = location.param ?? activeId;
        return sessions.find((s) => s.id === wanted) ?? sessions[0] ?? null;
    }, [sessions, activeId, location.param]);
    /* A conversation you are watching is read by definition — including while a
       reply streams into it. Without this, switching away from a session that
       answered you would mark it unread. */
    const messageCount = session?.messages.length ?? 0;
    useEffect(() => {
        if (session && session.id === activeId && !streaming)
            markRead(session.id);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [session?.id, messageCount, streaming]);
    const [draft, setDraft] = useState('');
    const draftKey = `strata.chat-draft.${session?.id ?? 'new'}`;
    useEffect(() => {
        setDraft(sessionStorage.getItem(draftKey) ?? '');
        if (session && session.id !== activeId)
            selectSession(session.id);
    }, [draftKey]);
    const updateDraft = (value: string) => { setDraft(value); sessionStorage.setItem(draftKey, value); };
    const splitRef = useRef<HTMLDivElement>(null);
    const [splitWidth, setSplitWidth] = useState(0);
    const [previewFraction, setPreviewFraction] = useState(() => {
        try {
            const saved = localStorage.getItem(PREVIEW_SPLIT_KEY);
            return saved && Number.isFinite(Number(saved)) ? Number(saved) : DEFAULT_PREVIEW_FRACTION;
        }
        catch {
            return DEFAULT_PREVIEW_FRACTION;
        }
    });
    useEffect(() => {
        const timer = setTimeout(() => {
            try {
                localStorage.setItem(PREVIEW_SPLIT_KEY, String(previewFraction));
            }
            catch { }
        }, 180);
        return () => clearTimeout(timer);
    }, [previewFraction]);
    useEffect(() => {
        if (!splitRef.current)
            return;
        const ro = new ResizeObserver(entries => setSplitWidth(entries[0].contentRect.width));
        ro.observe(splitRef.current);
        return () => ro.disconnect();
    }, [session?.id]);
    const [expanded, setExpanded] = useState(false);
    useEffect(() => { setExpanded(false); }, [session?.id]);
    useEffect(() => {
        if (!expanded)
            return;
        const exit = (e: KeyboardEvent) => {
            if (e.key === 'Escape')
                setExpanded(false);
        };
        document.addEventListener('keydown', exit);
        return () => document.removeEventListener('keydown', exit);
    }, [expanded]);
    const [previews, setPreviews] = useState<PreviewArtifact[]>([]);
    const [selectedPreview, setSelectedPreview] = useState<string | null>(null);
    const timeSkipRuns = (session?.timeSkips ?? []).filter(run => session?.messages.some(m => m.timeSkipId === run.id && m.role === 'assistant') && (!run.job || (run.job.files?.length ?? 0) > 0));
    const runTabs = timeSkipRuns.map((run, i) => ({ id: timeSkipTabId(run.id), title: timeSkipRuns.length > 1 ? `Time-skip ${i + 1}` : 'Time-skip', icon: 'branch' as const, controls: 'time-skip-content' })).filter(tab => !session?.closedPreviews?.includes(tab.id));
    const plan = usePlan(session?.id);
    const planTabId = session ? `plan:${session.id}` : '';
    const planTab = plan && plan.state !== 'rejected' && !session?.closedPreviews?.includes(planTabId) ? [{ id: planTabId, title: 'Plan', icon: 'check' as const, controls: 'plan-content' }] : [];
    const planShown = planTab.length > 0 && selectedPreview === planTabId;
    const planRunId = plan?.state === 'running' ? plan.id : '';
    useEffect(() => { if (planRunId && planTabId && !session?.closedPreviews?.includes(planTabId)) setSelectedPreview(planTabId); }, [planRunId]);
    // The chat shows every run of this chat (a running one has no files yet, so no tab); the sidebar tabs only those with something to compare.
    const allTimeSkipRuns = session?.timeSkips ?? [];
    const extraTabs = [...runTabs, ...planTab];
    const selectedRun = timeSkipRuns.find(run => timeSkipTabId(run.id) === selectedPreview);
    const inlinePreview = splitWidth >= (selectedRun || planShown ? 760 : 1120);
    const paneWidth = previewWidth(splitWidth, previewFraction);
    const splitStyle = inlinePreview && selectedPreview && !expanded ? { gridTemplateColumns: `minmax(0,1fr) ${paneWidth}px` } : undefined;
    const resizeHandle = !setupVisible && inlinePreview && selectedPreview && !expanded ? <PreviewResizeHandle container={splitRef} total={splitWidth} fraction={previewFraction} onChange={setPreviewFraction} controls={selectedRun ? 'time-skip-content' : planShown ? 'plan-content' : 'artifact-content'}/> : null;
    const runCount = timeSkipRuns.length;
    const previewCount = session?.messages.reduce((n, m) => n + (m.artifacts?.length ?? 0), 0) ?? 0;
    useEffect(() => { const items = [...new Map((session?.messages.flatMap(m => m.artifacts ?? []) ?? []).map(a => [a.id, a])).values()]; setPreviews(items.filter(a => !session?.closedPreviews?.includes(a.id))); setSelectedPreview(session?.closedPreviews?.includes(session.previewOpen ?? '') ? null : session?.previewOpen?.startsWith('time-skip:') && !timeSkipRuns.some(r => timeSkipTabId(r.id) === session.previewOpen) ? null : session?.previewOpen ?? null); }, [session?.id]);
    useEffect(() => {
        const items = [...new Map((session?.messages.flatMap(m => m.artifacts ?? []) ?? []).map(a => [a.id, a])).values()];
        if (items.length)
            setPreviews(old => [...old.filter(a => !items.some(b => a.id === b.id)), ...items.filter(a => !session?.closedPreviews?.includes(a.id))]);
    }, [previewCount]);
    const localUrls = useRef<string[]>([]);
    useEffect(() => {
        const id = session?.previewOpen;
        if (id?.startsWith('time-skip:') && timeSkipRuns.some(r => timeSkipTabId(r.id) === id) && !session?.closedPreviews?.includes(id))
            setSelectedPreview(id);
    }, [session?.id, runCount]);
    useEffect(() => () => localUrls.current.forEach(url => URL.revokeObjectURL(url)), []);
    function openPreview(item: PreviewArtifact) {
        if (item.url?.startsWith('blob:'))
            localUrls.current.push(item.url);
        setPreviewTabClosed(session?.id, item.id, false, item.id);
        setPreviews(old => old.some(a => a.id === item.id) ? old : [...old, item]);
        setSelectedPreview(item.id);
    }
    /* A run that finishes with a file the sidebar can show opens its main result there by itself (one file, never several). Only a run seen
       finishing here does it: opening an old chat shows nothing new. A file that cannot be read just stays a card with Open. */
    const replyWasStreaming = useRef<Map<string, boolean>>(new Map());
    const lastReply = [...(session?.messages ?? [])].reverse().find(m => m.role === 'assistant');
    useEffect(() => {
        if (!session || !lastReply) return;
        const before = replyWasStreaming.current.get(lastReply.id);
        replyWasStreaming.current.set(lastReply.id, !!lastReply.streaming);
        if (before !== true || lastReply.streaming || lastReply.error) return;
        const main = mainResult(lastReply.deliverables, lastReply.content);
        if (!main) return;
        const sessionId = session.id;
        void previewOf(sessionId, main).then(item => { if (session.id === sessionId) openPreview(item); }, () => { });
    }, [lastReply?.id, lastReply?.streaming]);
    function closeTab(id: string) {
        const next = closePreviewTab([...previews.map(p => ({ id: p.id })), ...extraTabs], selectedPreview, id);
        setPreviews(previews.filter(p => p.id !== id));
        setSelectedPreview(next.selected);
        setPreviewTabClosed(session?.id, id, true, next.selected);
        if (!next.items.length) {
            setExpanded(false);
            requestAnimationFrame(() => textarea.current?.focus());
        }
    }
    function selectPreview(id: string) {
        setSelectedPreview(id);
        setPreviewTabClosed(session?.id, id, false, id);
    }
    function openTimeSkip(id: string) { selectPreview(timeSkipTabId(id)); }
    function openLink(url: string) {
        const item = artifactFromUrl(url);
        if (item)
            openPreview(item);
    }
    const scroller = useRef<HTMLDivElement>(null);
    const focusMode = useStore((s) => s.focus);
    /** Per-conversation scroll offsets, so returning to one restores the view. */
    const scrollPositions = useRef<Map<string, number>>(new Map());
    /* Scrolling back to reread something is normal; being stranded there while a
       reply arrives is not. The pill only appears once you are far enough up that
       the newest message is genuinely off screen. */
    const [awayFromBottom, setAwayFromBottom] = useState(false);
    /* A reply that finishes while you are reading further up has no way to say so
       — the text simply stops arriving off screen. The pill pulses once, on
       opacity alone, so the cue is noticed without anything moving. */
    const [justFinished, setJustFinished] = useState(false);
    const wasStreaming = useRef(false);
    useEffect(() => {
        if (wasStreaming.current && !streaming) {
            setJustFinished(true);
            const t = setTimeout(() => setJustFinished(false), 700);
            wasStreaming.current = streaming;
            return () => clearTimeout(t);
        }
        wasStreaming.current = streaming;
        return undefined;
    }, [streaming]);
    /** Whether the reader is following the newest message rather than reading back. */
    const pinned = useRef(true);
    /**
     * Scrolls to the newest message and stays there.
     *
     * `content-visibility: auto` means the transcript's height is an estimate
     * until each turn renders, so a single `scrollTo(scrollHeight)` lands short
     * and the reader ends up above the reply they asked to see. Re-aiming until
     * the height stops growing settles it; the loop is bounded because a
     * transcript that never settles must not spin.
     */
    const scrollToLatest = useCallback((smooth = true) => {
        const el = scroller.current;
        if (!el)
            return;
        pinned.current = true;
        el.scrollTo({ top: el.scrollHeight, behavior: smooth ? scrollBehavior() : 'auto' });
        setAwayFromBottom(false);
    }, []);
    /**
     * Keeps a pinned transcript pinned.
     *
     * With `content-visibility: auto`, an off-screen turn has an *estimated*
     * height until it renders, so the scroll height grows as you approach the
     * bottom and any single scroll lands short. Watching the content box and
     * re-aiming while the reader is pinned is what makes the two compatible —
     * and it fixes the same drift while a long reply streams in.
     */
    useEffect(() => {
        const el = scroller.current;
        // The content, not the scroller's first child — which is the reading
        // progress hairline, and never changes size.
        const inner = el?.querySelector('.chat__inner');
        if (!el || !inner)
            return undefined;
        const ro = new ResizeObserver(() => {
            if (pinned.current)
                el.scrollTop = el.scrollHeight;
        });
        ro.observe(inner);
        return () => ro.disconnect();
    }, [session?.id, session?.messages.length === 0]);
    const textarea = useRef<HTMLTextAreaElement>(null);
    const remote = useStore(s => s.remoteModel), providers = useStore(s => s.providers);
    const apiModel = providers?.connections.find(c => c.id === remote?.connectionId)?.models.find(m => m.id === remote?.modelId);
    const loaded = apiModel ?? library?.variants.find((v) => v.id === library.loaded) ?? null;
    /**
     * Follow the stream only when the reader is already at the bottom. Scrolling
     * back through a long conversation should not be yanked forward by the next
     * token — which is the single most irritating thing a chat UI can do over a
     * long session.
     */
    useEffect(() => {
        const el = scroller.current;
        if (!el || (session?.showcase && !streaming))
            return;
        const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
        if (nearBottom)
            scrollToLatest();
    }, [session?.messages.length, session?.messages.at(-1)?.content, scrollToLatest]);
    // Scroll position is remembered per conversation, so returning to one puts you
    // back where you were reading rather than at the top or the bottom.
    useEffect(() => {
        const el = scroller.current;
        if (!el || !session)
            return;
        const saved = scrollPositions.current.get(session.id);
        el.scrollTop = saved ?? (session.showcase ? 0 : el.scrollHeight);
        pinned.current = !session.showcase || saved !== undefined;
        // Deliberately keyed on the conversation only: re-running on every message
        // would fight the follow-the-stream effect above.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [session?.id]);
    // The box grows with the draft but never past 30% of the window, and scrolls inside after that, so Send and the toolbar stay on screen (QA F08).
    useEffect(() => {
        const fit = () => {
            const el = textarea.current;
            if (!el)
                return;
            el.style.height = 'auto';
            el.style.height = `${Math.min(composerCap(window.innerHeight), el.scrollHeight)}px`;
        };
        fit();
        window.addEventListener('resize', fit);
        return () => window.removeEventListener('resize', fit);
    }, [draft]);
    /** Hands a rewound prompt back to the composer so it can be changed. */
    function editPrompt(text: string) {
        setDraft(text);
        requestAnimationFrame(() => textarea.current?.focus());
    }
    /**
     * Turn navigation, for reading rather than writing.
     *
     * Focus mode is for reading a long conversation, and reading needs a next.
     * `Alt+↑`/`Alt+↓` step between your own turns — the boundaries a reader
     * actually navigates by, rather than by message — and `Cmd+↓` returns to the
     * latest. Only bound while focus mode is on, so the keys stay free elsewhere.
     */
    useEffect(() => {
        if (!focusMode)
            return undefined;
        const onKey = (e: KeyboardEvent) => {
            const el = scroller.current;
            if (!el)
                return;
            const atLatest = () => {
                pinned.current = true;
                el.scrollTo({ top: el.scrollHeight, behavior: scrollBehavior() });
            };
            if ((e.metaKey || e.ctrlKey) && e.key === 'ArrowDown') {
                e.preventDefault();
                atLatest();
                return;
            }
            if (!e.altKey || (e.key !== 'ArrowUp' && e.key !== 'ArrowDown'))
                return;
            e.preventDefault();
            const turns = [...el.querySelectorAll<HTMLElement>('.st-msg--user')];
            if (!turns.length)
                return;
            const top = el.scrollTop;
            /* The tolerance has to exceed TURN_OFFSET, or the key sticks on the turn
               it just moved to: positioning a turn at `scrollTop + 16` leaves it
               satisfying `offsetTop > scrollTop + 8` on the next press. */
            const TURN_OFFSET = 16;
            const seen = TURN_OFFSET + 24;
            const next = e.key === 'ArrowDown'
                ? turns.find((t) => t.offsetTop > top + seen)
                : [...turns].reverse().find((t) => t.offsetTop < top - seen);
            if (next) {
                pinned.current = false;
                el.scrollTo({ top: Math.max(0, next.offsetTop - TURN_OFFSET), behavior: scrollBehavior() });
                setAwayFromBottom(true);
            }
            else if (e.key === 'ArrowDown') {
                atLatest();
            }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [focusMode]);
    const chatNative = useNative(), queuedPhotos = useChatPhotos(session?.id);
    const readyForChat = !!loaded || !!chatNative?.instances.some(i => i.status === 'ready');
    useEffect(() => {
        if (setupVisible || !readyForChat || messageCount)
            return;
        if (scroller.current?.querySelector('.first-chat-start'))
            textarea.current?.focus({ preventScroll: true });
    }, [setupVisible, readyForChat, session?.id, messageCount]);
    function submit() {
        const text = draft.trim();
        if (!text && !queuedPhotos.length)
            return;
        if (streaming) {
            // Typed while a goal runs: it waits in the queue, shown under the conversation. With no goal running there is nothing to queue behind.
            if (!text || !session)
                return;
            updateDraft('');
            void queueMessage(session.id, text).then(ok => { if (!ok) { updateDraft(text); pushToast('info', tr('The assistant is still answering'), tr('Send your message when it has finished.')); } }, e => { updateDraft(text); pushToast('error', tr('The message was not queued'), plainFailureText(e instanceof Error ? e.message : String(e)).message); });
            return;
        }
        const timeSkip = timeSkipPrompt(text);
        if (timeSkip === '') {
            pushToast('info', 'Add a prompt after /time-skip');
            return;
        }
        if (timeSkip === null) {
            // No model to answer: say so plainly (or show the download), and do not start a run.
            const why = noModelNotice(chatNative ?? null, { libraryLoaded: !!loaded });
            if (why) { pushToast('info', tr(why.title), tr(why.message)); return; }
        }
        updateDraft('');
        void sendMessage(text);
    }
    function onComposerKey(_e: React.KeyboardEvent) { }
    if (!session) {
        return (<>
        <ChatToolbar previewWidth={paneWidth} extraTabs={extraTabs} session={session} items={previews} onCloseTab={closeTab} selected={setupVisible ? null : selectedPreview} onSelect={selectPreview} onClose={() => { setSelectedPreview(null); setExpanded(false); }} inline={inlinePreview} expanded={expanded} onExpand={() => setExpanded(v => !v)}/>
        <div className="chatsplit" style={splitStyle} data-timeskip={!!selectedRun || undefined} ref={splitRef} data-preview={!!selectedPreview && inlinePreview || undefined} data-expanded={expanded && !!selectedPreview && inlinePreview || undefined}>
         <DropZone className="chatcol">
          <div className="main__body chat" ref={scroller}>
            {tr(awayFromBottom ? (<button type="button" className="jumplatest" data-finished={justFinished || undefined} aria-label={tr("Scroll to the latest message")} title={tr("Scroll to the latest message")} onClick={() => scrollToLatest()}>
                <Icon name="chevron" size={18}/></button>) : null)}
            <div className="chat__inner">
              <UsageDashboard loaded={!!loaded}/>
            </div>
          </div>
          <Composer draft={draft} setDraft={updateDraft} loaded={!!loaded} streaming={streaming} onSubmit={submit} onKey={onComposerKey} session={session} onPreview={openPreview} textareaRef={textarea}/>
         </DropZone>
        <ArtifactPanel items={previews} onCloseTab={closeTab} selected={setupVisible ? null : selectedPreview} onSelect={selectPreview} onClose={() => setSelectedPreview(null)} inline={inlinePreview}/>
         {tr(resizeHandle)}
        </div>
      </>);
    }
    return (<>
      <ChatToolbar previewWidth={paneWidth} extraTabs={extraTabs} session={session} items={previews} onCloseTab={closeTab} selected={setupVisible ? null : selectedPreview} onSelect={selectPreview} onClose={() => { setSelectedPreview(null); setExpanded(false); }} inline={inlinePreview} expanded={expanded} onExpand={() => setExpanded(v => !v)}/>

      {/* The composer belongs to the conversation column, not the window: with
                the context panel open, a full-width composer sat underneath the panel
                and drifted out of alignment with the transcript it belongs to. */}
      <div className="chatsplit" style={splitStyle} data-timeskip={!!selectedRun || undefined} ref={splitRef} data-preview={!!selectedPreview && inlinePreview || undefined} data-expanded={expanded && !!selectedPreview && inlinePreview || undefined}>
        <DropZone className="chatcol">
          <div className="main__body chat" ref={scroller} onScroll={(e) => {
            const el = e.currentTarget;
            if (session)
                scrollPositions.current.set(session.id, el.scrollTop);
            // How far you are from the newest message, which decides whether
            // the way back is worth offering.
            // The threshold has to tolerate the height estimate moving under
            // us, or the pill flickers as turns render during a scroll.
            const away = el.scrollHeight - el.scrollTop - el.clientHeight > 400;
            pinned.current = !away;
            setAwayFromBottom(away);
        }}>

            <div className="chat__inner">
              {tr(session.messages.length === 0 ? (session.statsAvailable === false ? <p className="small muted">{tr("Editing from an earlier message. Update the prompt below.")}</p> :
            <UsageDashboard loaded={!!loaded}/>) : (session.messages.map((m, i) => (<Message key={m.id} message={m} isLast={i === session.messages.length - 1} onEditPrompt={editPrompt} onPreview={openPreview} onOpenTimeSkip={openTimeSkip} timeSkipDone={!!allTimeSkipRuns.find(r => r.id === m.timeSkipId)?.completedAt} timeSkipReal={!!allTimeSkipRuns.find(r => r.id === m.timeSkipId)?.job} timeSkipRun={allTimeSkipRuns.find(r => r.id === m.timeSkipId)} onOpenLink={openLink}/>))))}
              <QueuedMessages chatId={session.id}/>
            </div>
          </div>
          <div className="jumplatest-row">{tr(awayFromBottom ? (<button type="button" className="jumplatest" data-finished={justFinished || undefined} aria-label={tr("Scroll to the latest message")} title={tr("Scroll to the latest message")} onClick={() => scrollToLatest()}>
                <Icon name="chevron" size={18}/></button>) : null)}</div>
          <Composer draft={draft} setDraft={updateDraft} loaded={!!loaded} streaming={streaming} onSubmit={submit} onKey={onComposerKey} session={session} onPreview={openPreview} textareaRef={textarea}/>
        </DropZone>
        {tr(!setupVisible && selectedRun && session ? <TimeSkipPanel key={selectedRun.id} run={selectedRun} sessionId={session.id} inline={inlinePreview} expanded={expanded} onCompare={setExpanded} onPreview={openPreview} onClose={() => { setSelectedPreview(null); setExpanded(false); }}/> : null)}
        {tr(!setupVisible && planShown ? <PlanPanel chatId={session?.id}/> : null)}
        <ArtifactPanel items={previews} onCloseTab={closeTab} selected={setupVisible || planShown ? null : selectedPreview} onSelect={selectPreview} onClose={() => setSelectedPreview(null)} inline={inlinePreview}/>
        {tr(resizeHandle)}
      </div>
    </>);
}
interface SlashCommand {
    id: string;
    label: string;
    hint: string;
    icon: IconName;
    run: () => void;
}
function useSlashCommands(clear: () => void, setDraft: (value: string) => void): SlashCommand[] {
    const demo = useStore(s => s.library?.mode === 'simulated');
    return [
        { id: 'time-skip', label: '/time-skip', hint: 'Try a change on a copy, check it, and keep it only if it is better', icon: 'branch', run: () => setDraft('/time-skip ') },
        { id: 'goal', label: '/goal', hint: 'Set a goal and let the assistant keep working until it is done', icon: 'bolt', run: () => setDraft('/goal ') },
        { id: 'plan', label: '/plan', hint: 'Have the assistant write a plan first; nothing runs until you approve it', icon: 'check', run: () => setDraft('/plan ') },
        { id: 'research', label: '/research', hint: 'Search the web, read the pages and answer with sources', icon: 'search', run: () => setDraft('/research ') },
        { id: 'loop', label: '/loop', hint: 'Repeat a message every few minutes while this chat is open', icon: 'refresh', run: () => setDraft('/loop 10m ') },
        { id: 'website', label: '/website', hint: 'Open a website in the preview sidebar', icon: 'link', run: () => { clear(); window.dispatchEvent(new Event('strata:open-website')); } },
        { id: 'preview', label: '/preview', hint: 'Open a local file in the preview sidebar', icon: 'file', run: () => { clear(); window.dispatchEvent(new Event('strata:open-preview')); } },
        ...(demo ? [{ id: 'subagents', label: '/subagents', hint: 'Preview subagent statistics', icon: 'experts' as const, run: () => { clear(); createSubagentExample(); } }] : []),
        ...getSlashCommands().map(command => ({ ...command, run: () => { clear(); command.run(); } }))
    ];
}
function SlashMenu({ query, commands, cursor, onPick, onHover, }: {
    query: string;
    commands: SlashCommand[];
    cursor: number;
    onPick: (c: SlashCommand) => void;
    onHover: (i: number) => void;
}) {
    const matches = commands.filter((c) => c.label.slice(1).startsWith(query.toLowerCase()));
    if (!matches.length) {
        return (<div className="slashmenu">
        <div className="slashmenu__empty small muted">{tr("No command matches \u201C/")}{query}”</div>
      </div>);
    }
    return (<div className="slashmenu" role="listbox" aria-label={tr("Commands")}>
      <div className="slashmenu__label">{tr("Commands")}</div>
      {tr(matches.map((c, i) => (<button key={c.id} type="button" role="option" aria-selected={i === cursor} className="slashmenu__item" data-active={i === cursor} onMouseEnter={() => onHover(i)} onMouseDown={e => e.preventDefault()} onClick={() => onPick(c)}>
          <Icon name={c.icon} size={16}/>
          <span className="slashmenu__name">{tr(c.label)}</span>
          <span className="slashmenu__hint">{tr(c.hint)}</span>
        </button>)))}
    </div>);
}
function Composer({ draft, setDraft, loaded, streaming, onSubmit, onKey, session, onPreview, textareaRef, }: {
    draft: string;
    setDraft: (v: string) => void;
    loaded: boolean;
    streaming: boolean;
    onSubmit: () => void;
    onKey: (e: React.KeyboardEvent) => void;
    session: Session | null;
    onPreview: (a: PreviewArtifact) => void;
    textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
    const clearDraft = useCallback(() => setDraft(''), [setDraft]);
    const photos = useChatPhotos(session?.id), native = useNative();
    const canChat = loaded || native?.instances.some(i => i.status === 'ready');
    const commands = useSlashCommands(clearDraft, setDraft);
    const [cursor, setCursor] = useState(0);
    const [fileError, setFileError] = useState('');
    const [slashOpen, setSlashOpen] = useState(false);
    const [slashDismissed, setSlashDismissed] = useState(false);
    const slashHostRef = useRef<HTMLDivElement>(null);
    const slashQuery = slashDismissed ? null : slashOpen ? '' : draft.startsWith('/') && !/\s/.test(draft) ? draft.slice(1) : null;
    const matches = slashQuery === null ? [] : commands.filter((c) => c.label.slice(1).startsWith(slashQuery.toLowerCase()));
    useEffect(() => { setSlashDismissed(false); }, [draft, session?.id]);
    useEffect(() => {
        if (slashQuery === null)
            return;
        const outside = (event: PointerEvent) => {
            if (slashHostRef.current?.contains(event.target as Node) || event.target === textareaRef.current)
                return;
            setSlashOpen(false);
            setSlashDismissed(true);
        };
        document.addEventListener('pointerdown', outside);
        return () => document.removeEventListener('pointerdown', outside);
    }, [slashQuery, textareaRef]);
    useEffect(() => {
        setCursor(0);
    }, [slashQuery]);
    return (<div className="composerwrap">
      <div className="composer">
        <div className="composer__inner">
          {tr(fileError ? <ErrorNotice message={fileError} action={tr("Dismiss")} onRetry={() => setFileError('')}/> : null)}
          <FolderChip chatId={session?.id} disabled={streaming}/>
          <GoalBar chatId={session?.id}/>
          <div className="st-composer" style={{ boxShadow: 'var(--st-sh-bar)' }}>
          {tr(photos.length ? <ChatPhotos photos={photos} onRemove={id => removeChatPhoto(id, session?.id)}/> : null)}
          <textarea ref={textareaRef} rows={1} value={draft} aria-label={tr("Message Strata")} placeholder={tr(canChat ? 'Message Strata…' : 'Try /time-skip, or load a model to chat…')} onFocus={() => setSlashDismissed(false)} onPaste={(e) => {
            // A picture on the clipboard (a screenshot, a copied image) is attached like a dropped file; text pastes as usual.
            const files = pastedFiles(e.clipboardData);
            if (!files.length)
                return;
            e.preventDefault();
            window.dispatchEvent(new CustomEvent(DROP_EVENT, { detail: files }));
        }} onChange={(e) => { setSlashOpen(false); setSlashDismissed(false); setDraft(e.target.value); e.currentTarget.style.height = "auto"; e.currentTarget.style.height = `${Math.min(180, e.currentTarget.scrollHeight)}px`; }} onKeyDown={(e) => {
            // The command menu owns the keyboard while it is open.
            if (slashQuery !== null && e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                setSlashOpen(false);
                setSlashDismissed(true);
                return;
            }
            if (slashQuery !== null && matches.length) {
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    setCursor((c) => (c + 1) % matches.length);
                    return;
                }
                if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    setCursor((c) => (c - 1 + matches.length) % matches.length);
                    return;
                }
                if (e.key === 'Enter' || e.key === 'Tab') {
                    e.preventDefault();
                    setSlashOpen(false);
                    setSlashDismissed(true);
                    matches[cursor].run();
                    return;
                }
            }
            onKey(e);
            if (!e.nativeEvent.isComposing && composerSends(e)) {
                e.preventDefault();
                onSubmit();
            }
        }}/>
          <div className="st-composer__bar">
            <ComposerControls draft={draft} onDraft={setDraft} session={session} onPreview={onPreview} streaming={streaming} onSlash={() => { setSlashDismissed(false); setSlashOpen(true); requestAnimationFrame(() => textareaRef.current?.focus()); }} onError={setFileError} onAttach={(name, text) => { setDraft(`${draft}${draft ? '\n\n' : ''}Attached ${name}\n\n${text}`); requestAnimationFrame(() => textareaRef.current?.focus()); }}/>
            {tr(streaming ? (<Btn variant="secondary" size="sm" icon="stop" onClick={stopStreaming}>{tr("Stop")}</Btn>) : (<Btn variant="primary" size="sm" icon="send" disabled={(!draft.trim() && !photos.length) || draft.trim() === '/' || (!canChat && timeSkipPrompt(draft) === null)} onClick={onSubmit}>{tr("Send")}</Btn>))}
          </div>
        </div>
        <ChatTelemetry session={session}/>
        </div>
      </div>
      {/* Outside the composer's glass on purpose: a material sampling another
                material blurs an already-blurred backdrop, which Liquid Glass
                prohibits and which doubles the compositor cost for a muddier result.
                The host repeats the composer's padding so the menu keeps its column. */}
      {tr(slashQuery !== null ? (<div className="slashmenu-host" ref={slashHostRef}>
          <SlashMenu query={slashQuery} commands={commands} cursor={cursor} onHover={setCursor} onPick={(c) => {
                setSlashOpen(false);
                setSlashDismissed(true);
                c.run();
            }}/>
        </div>) : null)}
    </div>);
}
function Message({ message: m, isLast, onEditPrompt, onPreview, onOpenLink, onOpenTimeSkip, timeSkipDone, timeSkipReal, timeSkipRun, }: {
    message: ChatMessage;
    isLast: boolean;
    onEditPrompt: (text: string) => void;
    onPreview: (a: PreviewArtifact) => void;
    onOpenLink: (url: string) => void;
    onOpenTimeSkip: (id: string) => void;
    timeSkipDone: boolean;
    timeSkipReal?: boolean;
    timeSkipRun?: TimeSkipRun;
}) {
    const streaming = useStore(s => s.streaming);
    const activeSession = useStore(s => s.activeSession);
    const hostLive = useStore(s => s.connection !== 'offline');
    const liveStatus = useLiveStatus(m.role === 'assistant' && m.streaming ? activeSession : null);
    // A reply that proposed a plan shows the plan card in place of its closing text (which is the plan in prose).
    const storedPlan = usePlanQuiet(m.planId ? activeSession : null);
    const planCard = !!m.planId && storedPlan?.id === m.planId;
    // A reply with activity lines is drawn piece by piece in the order it came; one without is the plain text it always was.
    const grouped = !!m.blocks?.some(b => b.kind === 'group');
    const [sentTime, setSentTime] = useState(() => relTime(m.at));
    if (m.role === 'user') {
        return (<div className="st-msg st-msg--user" onMouseEnter={() => setSentTime(relTime(m.at))} onFocus={() => setSentTime(relTime(m.at))}>
        <div className="st-bubble">{tr(m.photos?.length ? <ChatPhotos photos={m.photos}/> : null)}{m.markdown ? <Markdown text={m.content} onOpenLink={onOpenLink} imageSession={activeSession}/> : m.content}</div>
        <div className="st-msg__meta">
          <time dateTime={m.at} title={tr(new Date(m.at).toLocaleString(localeTag()))}>{tr(sentTime)}</time>
          <span className="st-msg__actions">
            <IconBtn name="copy" title={tr("Copy message")} onClick={() => void navigator.clipboard.writeText(m.content)}/>
            <IconBtn name="refresh" title={tr("Rewind to here")} disabled={streaming} onClick={() => {
                const prompt = rewindTo(m.id);
                if (prompt !== null)
                    onEditPrompt(prompt);
            }}/>
            <MessageMenu label={tr("More message actions")}><button role="menuitem" className="menu__item" onClick={() => branchFrom(m.id)}><Icon name="branch"/>{tr("Branch from here")}</button></MessageMenu>
          </span>
        </div>
      </div>);
    }
    return (<div className="st-msg st-msg--assistant" data-error={m.error ? 'true' : undefined}>
      {tr(m.reasoning ? <ReasoningBlock message={m}/> : null)}


      {tr((m.tools ?? []).filter(t => !grouped || t.state !== 'done').map((t) => (<ToolCard key={t.id} messageId={m.id} tool={t}/>)))}

      {tr(!(m.timeSkipId && timeSkipReal && !timeSkipDone) && (m.content || m.error || !m.streaming || !liveStatus) ? <div className="st-bubble">
        {tr(planCard ? <PlanCard chatId={activeSession} planId={m.planId!}/> : null)}
        {tr(m.error ? (<ErrorNotice message={m.error} detail={m.errorDetail} {...contextRecovery(m.errorCode)}/>) : null)}
        {tr(grouped && !planCard && !m.timeSkipId ? m.blocks!.map((b, i) => b.kind === 'group' ? <ActivityLine key={b.id} group={b} sessionId={activeSession ?? ''} onPreview={onPreview}/> : <Markdown key={i} text={b.text} onOpenLink={onOpenLink} imageSession={activeSession} onPreviewHtml={content => onPreview({ id: m.id + '-html-' + i, kind: 'website', title: 'HTML preview', content })}/>) : null)}
        {tr(m.content && !planCard && !grouped ? (<Markdown text={m.timeSkipId && !timeSkipReal ? hostLive ? tr('This was a sample run made while the app could not reach its host.') : timeSkipDone ? tr('The result is ready. Switch between Original and Updated, or compare them side by side. This example uses sample outputs and simulated checks.') : tr('Revising the saved draft. Follow the result in the sidebar. This example uses sample outputs and simulated checks.') : m.content} onOpenLink={onOpenLink} imageSession={activeSession} onPreviewHtml={content => onPreview({ id: m.id + '-html', kind: 'website', title: 'HTML preview', content })}/>) : !m.error && m.streaming && !liveStatus ? (<span className="typing" aria-label={tr("Generating")}>
            <i />
            <i />
            <i />
          </span>) : null)}
        {tr(m.streaming && m.content ? <span className="caret" aria-hidden="true"/> : null)}
      </div> : null)}
      {tr(m.skills?.length && !m.streaming ? <SkillsUsed skills={m.skills}/> : null)}
      {tr(m.deliverables?.length && !m.streaming ? <div className="deliverables">{m.deliverables.map(f => <FileCard key={f.path} file={f} sessionId={activeSession ?? ''} onPreview={onPreview}/>)}<EditedFiles files={m.deliverables} sessionId={activeSession ?? ''} onPreview={onPreview}/></div> : null)}
      {tr(m.streaming && isLast && !m.timeSkipId ? <StatusLine sessionId={activeSession} startedAt={Date.parse(m.at) || Date.now()} tokens={m.tokens}/> : null)}

      {tr(m.timeSkipId && timeSkipReal && !timeSkipDone && timeSkipRun ? <TimeSkipWorking run={timeSkipRun} sessionId={activeSession ?? ''}/> : m.timeSkipId && (!timeSkipReal || (timeSkipRun?.job?.files?.length ?? 0) > 0) ? <button className="timeskip-chat-card" onClick={() => onOpenTimeSkip(m.timeSkipId!)}><Icon name="branch" size={20}/><span><strong>{tr("Time-skip")}</strong><span>{tr(timeSkipDone ? 'Result ready · compare versions' : 'Revising · follow progress')}</span></span><Icon name="chevron" size={16} className="timeskip-card-arrow"/></button> : null)}
      {tr(m.media?.length ? <ChatMedia items={m.media} onPreview={onPreview}/> : null)}
      {tr(m.artifacts?.length ? <div className="artifact-cards">{tr(m.artifacts.map(a => <ArtifactCard key={a.id} artifact={a} onPreview={onPreview}/>))}</div> : null)}

      {tr(!m.streaming && !m.timeSkipId && (m.content || m.error) ? (<div className="st-msg__meta">
          <span className="st-msg__actions">
            {tr(isLast ? <>
              <IconBtn name="copy" title={tr("Copy response")} onClick={() => void navigator.clipboard.writeText(m.content)}/>
              <IconBtn name="refresh" title={tr("Ask again")} onClick={() => void retryLast(m.id)}/>
            </> : null)}
            <MessageMenu label={tr("More response actions")}>
              {tr(!isLast ? <><button role="menuitem" className="menu__item" onClick={() => void navigator.clipboard.writeText(m.content)}>{tr("Copy response")}</button><button role="menuitem" className="menu__item" onClick={() => void retryLast(m.id)}>{tr("Ask again")}</button></> : null)}
              {tr(isLast ? <button role="menuitem" className="menu__item" onClick={() => void continueLast()}>{tr("Continue")}</button> : null)}
              <button role="menuitem" className="menu__item" onClick={() => branchFrom(m.id)}>{tr("Branch from here")}</button>
              {tr(m.tokens || m.tps || m.durationMs ? <p className="xs muted response-stats">{tr(m.tokens ? `${tokens(m.tokens)} tokens` : '')}{tr(m.tps ? ` · ${tps(m.tps)} tok/s` : '')}{tr(m.durationMs ? ` · ${duration(m.durationMs)}` : '')}{tr(m.demo ? ' · Example' : '')}</p> : null)}
            </MessageMenu>
          </span>
        </div>) : null)}
    </div>);
}
function MessageMenu({ label, children }: {
    label: string;
    children: React.ReactNode;
}) {
    const [open, setOpen] = useState(false);
    const trigger = useRef<HTMLButtonElement>(null);
    return <><button ref={trigger} className="st-btn st-btn--icon" aria-label={tr(label)} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(v => !v)}><Icon name="ellipsis" size={16}/></button>{tr(open ? <FloatingMenu anchor={trigger.current} title={tr(label)} onClose={() => setOpen(false)}><div onClick={() => setOpen(false)}>{tr(children)}</div></FloatingMenu> : null)}</>;
}
const SCOPE: Record<ToolCall['scope'], {
    label: string;
    tone: 'plain' | 'info' | 'warn';
}> = {
    read: { label: 'Reads', tone: 'plain' },
    write: { label: 'Writes', tone: 'warn' },
    shell: { label: 'Runs a command', tone: 'warn' },
    connected: { label: 'Connected tool', tone: 'info' },
};
/**
 * A tool call, with the grant it needs. Approvals are per action and per turn —
 * the model never gets standing permission.
 */
function ToolCard({ messageId, tool: t }: {
    messageId: string;
    tool: ToolCall;
}) {
    const scope = SCOPE[t.scope];
    const pending = t.state === 'proposed';
    if (!pending && t.state !== 'running')
        return <details className="disclosure tool-summary"><summary><Icon name="chevron" size={14}/>{tr(t.verb ?? t.name)}{t.target ? <><span>·</span><span className="tool-summary__target" title={t.target}>{t.target}</span></> : null}<span>·</span><Icon name={t.state === 'denied' || t.state === 'error' ? 'x' : 'check'} size={14}/>{tr(t.state === 'denied' ? 'Denied' : t.state === 'error' ? 'Failed' : t.live && t.state === 'done' && t.approved ? 'Approved' : 'Done')}</summary><div><code>{t.preview}</code><p className="small muted">{tr(t.result ?? t.effect)}</p></div></details>;
    if (pending && t.ask)
        return <section className="askcard" data-state={t.state} aria-label={tr(askQuestion(t.ask))}>
      <div className="askcard__top"><span className="askcard__icon"><Icon name={askIcon(t.ask)} size={18}/></span>
        <div className="askcard__text"><p className="askcard__q">{askQuestion(t.ask)}</p>{askSub(t.ask) ? <p className="askcard__why" title={askSub(t.ask)}>{askSub(t.ask)}</p> : null}</div>
        <div className="askcard__acts"><Btn size="sm" onClick={() => resolveTool(messageId, t.id, false)}>{tr("Deny")}</Btn><Btn variant="primary" size="sm" onClick={() => resolveTool(messageId, t.id, true)}>{tr("Approve")}</Btn></div></div>
      <details className="askcard__details"><summary><Icon name="chevron" size={11}/>{tr("Details")}</summary><pre>{t.preview}</pre></details>
    </section>;
    return <section className="toolcard toolcard--compact" data-state={t.state} aria-label={t.name}>
    <details className="toolcard__disclosure">
      <summary><Icon name={t.scope === 'shell' ? 'bolt' : 'tool'} size={17}/><span className="toolcard__title">{tr(t.verb ?? t.name)}</span><span className="toolcard__scope xs muted">{tr(scope.label)}</span><Icon name="chevron" className="toolcard__chevron" size={15}/></summary>
      <div className="toolcard__body"><code className="toolcard__preview">{t.preview}</code><p className="small muted">{tr(t.effect)}</p>{tr(t.result ? <p className="small">{t.result}</p> : null)}</div>
    </details>
    {tr(pending && t.target ? <code className="toolcard__what" title={t.target}>{t.target}</code> : null)}
    {tr(pending ? <div className="toolcard__quick-actions"><span className="xs muted">{tr("Needs your approval")}</span><Btn size="sm" onClick={() => resolveTool(messageId, t.id, false)}>{tr("Deny")}</Btn><Btn variant="primary" size="sm" icon="check" onClick={() => resolveTool(messageId, t.id, true)}>{tr("Approve")}</Btn></div> : <span className="xs muted toolcard__running" role="status">{tr("Running\u2026")}</span>)}
  </section>;
}
export { cx };
