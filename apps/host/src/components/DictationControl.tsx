import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Btn, Icon, IconBtn } from './ui';
import { getCommands, MOD } from '../lib/commands';
import { bindingKeys, useBindings } from '../lib/shortcut-bindings';
import { createDictation, type DictationState, type RecognizerConstructor } from '../lib/dictation';
export function DictationControl({ draft, onDraft, sessionId, disabled }: {
    draft: string;
    onDraft: (text: string) => void;
    sessionId?: string;
    disabled: boolean;
}) {
    useBindings();
    const [visible, setVisible] = useState(false), [state, setState] = useState<DictationState | 'unsupported'>('finished'), [error, setError] = useState(''), [spoken, setSpoken] = useState('');
    const controller = useRef<ReturnType<typeof createDictation> | null>(null);
    const latest = useRef({ draft, onDraft });
    latest.current = { draft, onDraft };
    const active = useRef(false), base = useRef(''), written = useRef(''), manualEdit = useRef(false);
    const toggleRef = useRef<() => void>(() => { });
    const constructor = (window as unknown as {
        SpeechRecognition?: RecognizerConstructor;
        webkitSpeechRecognition?: RecognizerConstructor;
    }).SpeechRecognition ?? (window as unknown as {
        webkitSpeechRecognition?: RecognizerConstructor;
    }).webkitSpeechRecognition;
    const binding = getCommands().find(c => c.id === 'dictation')?.binding ?? null;
    const shortcut = bindingKeys(binding, MOD).map(k => MOD === '⌘' && k === 'Shift' ? '⇧' : MOD === '⌘' && k === 'Alt' ? '⌥' : k).join(MOD === '⌘' ? '' : '+');
    const ariaShortcut = binding ? [...(binding.mod ? [MOD === '⌘' ? 'Meta' : 'Control'] : []), ...(binding.alt ? ['Alt'] : []), ...(binding.shift ? ['Shift'] : []), binding.key.toUpperCase()].join('+') : undefined;
    const listening = ['starting', 'listening', 'speaking', 'finishing'].includes(state);
    function close() { controller.current?.dispose(); controller.current = null; active.current = false; setState('finished'); setVisible(false); document.querySelector<HTMLTextAreaElement>('.st-composer textarea')?.focus(); }
    function toggle() {
        if (active.current) {
            controller.current?.stop();
            return;
        }
        if (disabled)
            return;
        controller.current?.dispose();
        controller.current = null;
        setVisible(true);
        setError('');
        setSpoken('');
        if (!constructor) {
            setState('unsupported');
            return;
        }
        base.current = latest.current.draft;
        written.current = base.current;
        manualEdit.current = false;
        try {
            const recognizer = new constructor();
            controller.current = createDictation(recognizer, { lang: navigator.language || 'en-US', onState: (next, message) => {
                    active.current = ['starting', 'listening', 'speaking', 'finishing'].includes(next);
                    setState(next);
                    if (message)
                        setError(message);
                }, onText: text => {
                    setSpoken(text);
                    if (manualEdit.current)
                        return;
                    const value = `${base.current}${base.current && text && !/\s$/.test(base.current) ? ' ' : ''}${text}`;
                    written.current = value;
                    latest.current.onDraft(value);
                } });
            controller.current.start();
        }
        catch {
            active.current = false;
            setState('error');
            setError('Speech recognition is unavailable in this browser. Use your computer’s built-in dictation.');
        }
    }
    toggleRef.current = toggle;
    useEffect(() => {
        const event = () => {
            if (!document.querySelector('.dialog-layer'))
                toggleRef.current();
        };
        window.addEventListener('strata:dictation', event);
        return () => window.removeEventListener('strata:dictation', event);
    }, []);
    useEffect(() => { controller.current?.dispose(); controller.current = null; active.current = false; setState('finished'); setVisible(false); return () => controller.current?.dispose(); }, [sessionId]);
    useEffect(() => {
        if (!visible)
            return;
        const escape = (event: KeyboardEvent) => {
            if (event.key === 'Escape' && !event.defaultPrevented && !document.querySelector('.dialog-layer')) {
                event.preventDefault();
                close();
            }
        };
        document.addEventListener('keydown', escape);
        return () => document.removeEventListener('keydown', escape);
    }, [visible]);
    useEffect(() => {
        if (disabled && active.current)
            controller.current?.stop();
    }, [disabled]);
    useEffect(() => {
        if (active.current && draft !== written.current) {
            manualEdit.current = true;
            controller.current?.stop();
        }
    }, [draft]);
    const title = state === 'unsupported' ? 'Dictation unavailable' : state === 'error' ? 'Dictation stopped' : state === 'finished' ? 'Dictation finished' : state === 'starting' ? 'Starting microphone…' : state === 'finishing' ? 'Finishing dictation…' : state === 'speaking' ? 'Listening to you…' : 'Ready to listen';
    return <>
    <button className="composer-mic" data-speech-supported={!!constructor} aria-label={tr("Speech to text")} title={tr("Speech to text \u00B7 {0}", [shortcut])} aria-keyshortcuts={ariaShortcut} disabled={disabled && !listening} aria-pressed={listening} aria-controls={visible ? 'dictation-feedback' : undefined} onClick={toggle}><Icon name="mic" size={19}/></button>
    {tr(visible ? createPortal(<aside id="dictation-feedback" className="dictation-feedback" aria-label={tr("Dictation")} data-active={listening || undefined}>
      <div className="dictation-feedback__head"><span className="dictation-feedback__icon"><Icon name={state === 'finished' ? 'check' : 'mic'} size={19}/></span><strong role="status">{tr(title)}</strong><IconBtn name="x" title={tr("Close dictation")} onClick={close}/></div>
      {tr(listening ? <div className="voice-wave" data-speaking={state === 'speaking' || undefined} aria-hidden="true">{tr([0, 1, 2, 3, 4].map(i => <span key={i} style={{ animationDelay: `${i * 100}ms` }}/>))}</div> : null)}
      <p className="small muted">{tr(state === 'unsupported' ? 'This browser does not provide speech recognition. Use your computer’s built-in dictation in the message box.' : state === 'error' ? error : state === 'finished' ? (spoken ? 'Text is in your message box. Review it before sending.' : 'No text was added. Your draft is ready to edit.') : 'Speak naturally. A 3.5-second pause finishes dictation. Text appears in your message box without sending.')}</p>
      {tr(spoken ? <p className="dictation-feedback__transcript">{tr(spoken)}</p> : null)}
      <div className="dictation-feedback__actions"><kbd>{tr(shortcut)}</kbd>{tr(listening ? <Btn size="sm" onClick={() => controller.current?.stop()} disabled={state === 'finishing'}>{tr("Stop")}</Btn> : <>{tr(constructor ? <Btn size="sm" onClick={toggle}>{tr(state === 'error' ? 'Try again' : 'Speak again')}</Btn> : null)}<Btn size="sm" onClick={close}>{tr("Done")}</Btn></>)}</div>
      {tr(listening ? <p className="xs muted dictation-feedback__provider">{tr("Uses your browser\u2019s speech service.")}</p> : null)}
    </aside>, document.body) : null)}
  </>;
}
