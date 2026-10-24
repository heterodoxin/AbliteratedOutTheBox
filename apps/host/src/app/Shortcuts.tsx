import { tr } from '../lib/i18n';
import { useState } from 'react';
import { Dialog, Btn, IconBtn } from '../components/ui';
import { setShortcuts, useStore } from '../lib/store';
import { getCommands, MOD } from '../lib/commands';
import { bindingFor, bindingEqual, bindingKeys, eventBinding, resetBindings, setBinding, useBindings } from '../lib/shortcut-bindings';
export function ShortcutsDialog() {
    const open = useStore(s => s.shortcutsOpen);
    useBindings();
    const [recording, setRecording] = useState<string | null>(null), [error, setError] = useState('');
    const rows = [...getCommands().filter(c => c.key).map(c => ({ id: c.id, label: c.label, binding: c.binding ?? null, group: c.group })), { id: 'chat:send', label: 'Send message', binding: bindingFor('chat:send', { key: 'enter', mod: false, alt: false, shift: false }), group: 'Chat' }];
    function record(id: string, e: React.KeyboardEvent) {
        e.preventDefault();
        e.stopPropagation();
        if (['Meta', 'Control', 'Shift', 'Alt'].includes(e.key))
            return;
        if (e.key === 'Escape') {
            setRecording(null);
            setError('');
            return;
        }
        const b = eventBinding(e);
        if (id === 'chat:send' ? b.key !== 'enter' || b.alt || b.shift : !b.mod || ['tab', 'escape', 'enter'].includes(b.key)) {
            setError(id === 'chat:send' ? 'Use Enter or Command/Ctrl + Enter. Shift + Enter stays available for a new line.' : 'Use Command/Ctrl with a letter, number or punctuation key.');
            return;
        }
        const duplicate = rows.find(r => r.id !== id && bindingEqual(r.binding, b));
        if (duplicate) {
            setError(`Already assigned to ${duplicate.label}. Clear or change that binding first.`);
            return;
        }
        setBinding(id, b);
        setRecording(null);
        setError('');
    }
    return <Dialog open={open} title={tr("Keyboard shortcuts")} onClose={() => { setRecording(null); setShortcuts(false); }} actions={<><Btn onClick={() => { resetBindings(); setRecording(null); setError(''); }}>{tr("Restore defaults")}</Btn><Btn variant="primary" onClick={() => { setRecording(null); setError(''); setShortcuts(false); }}>{tr("Done")}</Btn></>}>
    <p className="small muted">{tr("Click a shortcut, then press your preferred keys. Changes are saved in this browser. Escape cancels recording.")}</p>
    {tr(error ? <p className="inline-error" role="alert">{tr(error)}</p> : null)}
    <div className="shortcut-editor">{tr([...new Set(rows.map(r => r.group))].map(group => <section key={group}><h3 className="modal__group">{tr(group)}</h3>{tr(rows.filter(r => r.group === group).map(r => <div className="shortcutrow" key={r.id}><span className="shortcutrow__label">{tr(r.label)}</span><button className="shortcut-capture" aria-label={tr("Change shortcut for {0}", [tr(r.label)])} data-recording={recording === r.id || undefined} onClick={() => { setRecording(r.id); setError(''); }} onKeyDown={e => {
                    if (recording === r.id)
                        record(r.id, e);
                }}>{tr(recording === r.id ? 'Press keys…' : bindingKeys(r.binding, MOD).map(k => <kbd key={k} className="kbd">{tr(k)}</kbd>))}</button>{tr(r.id !== 'chat:send' ? <IconBtn name="x" title={tr("Clear shortcut for {0}", [tr(r.label)])} disabled={!r.binding} onClick={() => setBinding(r.id, null)}/> : null)}</div>))}</section>))}</div>
    <p className="xs muted">{tr("Shift + Enter adds a new line. Type / for chat commands. Escape closes panels.")}</p>
  </Dialog>;
}
export { MOD };
