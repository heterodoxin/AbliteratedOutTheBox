import { tr } from '../lib/i18n';
import { useEffect, useState } from 'react';
import { Btn, Dialog, Icon } from './ui';
const stages = ['Checking hardware', 'Reviewing supported parameters', 'Choosing recommended settings', 'Ready to review'];
/** Proposes catalog defaults; the animation never represents a measured benchmark. */
export function AutoTune({ open, onClose, onDone }: {
    open: boolean;
    onClose: () => void;
    onDone: () => void;
}) {
    const [step, setStep] = useState(0);
    useEffect(() => { if (!open)
        return; setStep(0); const timer = setInterval(() => setStep(s => Math.min(3, s + 1)), 900); return () => clearInterval(timer); }, [open]);
    return <Dialog open={open} title={tr("Auto-tune for this computer")} onClose={onClose} actions={<><Btn onClick={onClose}>{tr("Close")}</Btn><Btn variant="primary" disabled={step < 3} onClick={onDone}>{tr("Review recommended settings")}</Btn></>}><div className="tune-progress"><div className="tune-orbit" data-complete={step === 3}><svg viewBox="0 0 120 120" aria-hidden="true"><circle cx="60" cy="60" r="50"/><circle cx="60" cy="60" r="50" pathLength="100" strokeDasharray={`${(step + 1) * 25} 100`}/></svg><Icon name={step === 3 ? 'check' : 'engine'} size={42}/></div><h3 role="status">{tr(stages[step])}</h3><p className="small muted">{tr("A recommendation based on detected hardware and supported defaults. Save to apply it; speed has not been benchmarked.")}</p><div className="tune-steps">{tr(stages.map((s, i) => <span key={s} data-done={i <= step}/>))}</div></div></Dialog>;
}
