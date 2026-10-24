import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { Brand } from './Brand';
import { Icon } from './ui';
/** Vector artwork stays sharp, interactive and honours reduced motion. */
export function OnboardingArt({ kind = 'welcome', progress = 0, counter }: {
    kind?: 'welcome' | 'download' | 'tune';
    progress?: number;
    counter?: {
        value: number | undefined;
        label: string;
        key: string;
    };
}) {
    return <div className={`onboard-art onboard-art--${kind}`} aria-hidden={kind !== 'download'}>
  <div aria-hidden="true" className="onboard-orbit onboard-orbit--one"/><div className="onboard-orbit onboard-orbit--two"/>
  <div className="onboard-art-grid"/>
  {tr(kind === 'welcome' ? <>
   <div className="onboard-float onboard-float--chat"><Icon name="chat" size={19}/><span>{tr("A little help.")}<br /><strong>{tr("A lot more possible.")}</strong></span></div>
   <div className="onboard-float onboard-float--file"><Icon name="file" size={20}/><span>{tr("your-next-idea.md")}<small>{tr("Right here, on your computer")}</small></span></div>
   <div className="onboard-local-machine"><div className="onboard-machine-screen"><Brand variant="mark" height={36} decorative/><div className="onboard-machine-line"/><div className="onboard-machine-line short"/><div className="onboard-machine-answer"><i /><i /><i /></div></div><div className="onboard-machine-base"/><span className="onboard-local-badge"><span />{tr(" Local AI. Yours.")}</span></div>
   <div className="onboard-float onboard-float--spark"><Icon name="bolt" size={20}/></div>
  </> : kind === 'download' ? <>
   <div aria-hidden="true" className="onboard-data-packet packet-one"><Icon name="file" size={18}/></div><div aria-hidden="true" className="onboard-data-packet packet-two"><Icon name="image" size={18}/></div><div aria-hidden="true" className="onboard-data-packet packet-three"><Icon name="file" size={18}/></div>
   <div aria-hidden="true" className="onboard-download-core"><div className="onboard-download-ring" style={{ '--download-progress': `${Math.min(100, Math.max(0, progress))}%` } as React.CSSProperties}><Icon name="models" size={40}/></div><div className="onboard-download-stack"><i /><i /><i /></div></div>
   {tr(counter ? <DownloadCounter key={counter.key} value={counter.value} label={tr(counter.label)}/> : null)}
  </> : <>
   <div className="onboard-chip"><div className="onboard-chip-pins"/><Icon name="cpu" size={56}/><div className="onboard-chip-light"/></div>
   <div className="onboard-float onboard-float--cpu"><Icon name="cpu" size={18}/><span>{tr("Your CPU")}<small>{tr("Finding its rhythm")}</small></span></div>
   <div className="onboard-float onboard-float--gpu"><Icon name="gpu" size={18}/><span>{tr("Your GPU")}<small>{tr("Working together")}</small></span></div>
   <span className="onboard-local-badge"><span />{tr(" Tuned to your computer")}</span>
  </>)}
 </div>;
}
function DownloadCounter({ value, label }: {
    value: number | undefined;
    label: string;
}) {
    // Reset at each shard; animate only from reported download data.
    const [shown, setShown] = useState(value), previous = useRef(value);
    useEffect(() => {
        const from = previous.current;
        previous.current = value;
        if (value === undefined || from === undefined || matchMedia('(prefers-reduced-motion: reduce)').matches) {
            setShown(value);
            return;
        }
        const start = performance.now();
        let frame = 0;
        const tick = (now: number) => { const t = Math.min(1, (now - start) / 650); setShown(from + (value - from) * (1 - Math.pow(1 - t, 3))); if (t < 1)
            frame = requestAnimationFrame(tick); };
        frame = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(frame);
    }, [value]);
    return <div className="onboard-download-counter" role="img" aria-label={tr(value === undefined ? label : `${value.toFixed(1)} ${label}`)}><Icon name="download" size={18}/><strong aria-hidden="true">{tr(shown === undefined ? '···' : shown.toFixed(1))}</strong><span aria-hidden="true">{tr(label)}</span></div>;
}
const tour = [
    { label: 'Time-skip', command: '/time-skip', prompt: 'Improve this launch plan.', title: 'Keep the stronger idea.', description: 'Type /time-skip and what you want to improve. Strata explores a revision, compares it with the original, and keeps the stronger version.', badge: '', steps: ['Save the original', 'Explore a revision', 'Compare and keep'] },
    { label: 'Goal', command: '/goal', prompt: 'Get my portfolio ready to publish.', title: 'Give your AI a finish line.', description: 'A goal describes the outcome you want. Strata will plan the work, check its progress, and keep going until that outcome is reached.', badge: '', steps: ['Set the outcome', 'Work through the plan', 'Check the result'] },
    { label: 'Loop', command: '/loop', prompt: '10m check whether the build passes.', title: 'Let useful work repeat.', description: 'Type /loop, how often, and what to do — for example /loop 10m check the build. Strata sends it again and again while the chat is open. For work that must go on with Strata closed, use Tasks.', badge: '', steps: ['Say how often', 'Say what to do', 'Stop any time'] },
    { label: 'Subagents', command: 'Spawn subagents', prompt: 'to research, draft and review my launch plan.', title: 'More hands, one conversation.', description: 'Ask “spawn subagents” in chat and describe the jobs. Separate agents can research, write and review in parallel, then bring their work back to you.', badge: '', steps: ['Describe the jobs', 'Agents work in parallel', 'Bring the results together'] },
] as const;
export function OnboardingTour({ page, onChange }: {
    page: number;
    onChange: (page: number) => void;
}) {
    const [phase, setPhase] = useState(0), [replay, setReplay] = useState(0);
    useEffect(() => { const motion = matchMedia('(prefers-reduced-motion: reduce)'); setPhase(motion.matches ? 2 : 0); if (motion.matches)
        return; const timer = setInterval(() => setPhase(value => (value + 1) % 3), 2600); return () => clearInterval(timer); }, [page, replay]);
    const item = tour[page % tour.length];
    return <section className="onboard-tour" aria-label={tr("A quick tour of Strata")}>
  <div className="onboard-tour-top"><span>{tr("Try this in Strata")}</span><div role="tablist" aria-label={tr("Tour topic")}>{tr(tour.map((t, i) => <button key={t.label} type="button" role="tab" id={`onboard-tour-tab-${i}`} aria-controls="onboard-tour-content" aria-selected={page === i} tabIndex={page === i ? 0 : -1} onClick={() => onChange(i)} onKeyDown={e => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
        e.preventDefault();
        const next = e.key === 'Home' ? 0 : e.key === 'End' ? tour.length - 1 : (page + (e.key === 'ArrowLeft' ? -1 : 1) + tour.length) % tour.length;
        onChange(next);
        e.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role=tab]')[next]?.focus();
    } }}>{tr(t.label)}</button>))}</div></div>
  <div className="onboard-tour-content" role="tabpanel" id="onboard-tour-content" aria-labelledby={`onboard-tour-tab-${page}`} key={`${page}-${replay}`}>
   <div className="onboard-tour-demo-label"><span>{tr("Example")}</span>{tr(item.badge ? <span>{tr(item.badge)}</span> : null)}</div>
   <div className={`onboard-tour-preview preview-${page}`} data-phase={phase} aria-hidden="true">
    <div className="tour-prompt"><Icon name="chat" size={15}/><span><strong>{tr(item.command)}</strong>{tr(item.prompt ? ` ${item.prompt}` : '')}</span><Icon name="send" size={14}/></div>
    {tr(page === 0 ? <><div className="tour-directions"><div data-active={phase === 0}><Icon name="file" size={19}/><strong>{tr("Original")}</strong><small>{tr("Saved safely")}</small><i /><i /></div><div data-active={phase > 0}><Icon name="branch" size={19}/><strong>{tr("Revision")}</strong><small>{tr(phase === 2 ? 'Clearer next steps' : 'Exploring an idea')}</small><i /><i /></div></div><div className="tour-outcome" key={phase}><Icon name={phase === 2 ? 'check' : 'branch'} size={15}/>{tr(['Original saved', 'Trying a new direction', 'Stronger version kept'][phase])}</div></> : page === 1 ? <><div className="tour-goal"><Icon name="gauge" size={22}/><span>{tr("Portfolio ready to publish")}<small>{tr("Clear outcome \u00B7 measurable progress")}</small></span></div><div className="tour-goal-checks">{tr(['Choose the best work', 'Write project stories', 'Check every page'].map((label, i) => <span key={label} data-complete={phase >= i}><Icon name={phase >= i ? 'check' : 'clock'} size={13}/>{tr(label)}</span>))}</div></> : page === 2 ? <><div className="tour-loop"><div><Icon name="clock" size={20}/><strong>{tr("Daily briefing")}</strong><span>08:00</span></div><div className="tour-week">{tr(['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((day, i) => <span key={i} data-selected={i < 5} data-active={i === phase}>{tr(day)}</span>))}</div><small>{tr("Repeats every weekday")}</small></div><div className="tour-outcome" key={phase}><Icon name={phase === 2 ? 'check' : 'clock'} size={15}/>{tr(['Schedule your task', 'Tomorrow is already planned', 'A fresh briefing, every morning'][phase])}</div></> : <><div className="tour-agents">{tr([{ name: 'Research', icon: 'search' }, { name: 'Draft', icon: 'pencil' }, { name: 'Review', icon: 'check' }].map((agent, i) => <div key={agent.name} style={{ '--agent-delay': `${i * .18}s` } as React.CSSProperties}><Icon name={agent.icon as 'search' | 'pencil' | 'check'} size={19}/><strong>{tr(agent.name)}</strong><span>{tr(phase === 2 ? 'Done' : 'Working…')}</span><i /></div>))}</div><div className="tour-outcome" key={phase}><Icon name={phase === 2 ? 'check' : 'experts'} size={15}/>{tr(phase === 2 ? 'One plan, brought together' : 'Three jobs moving together')}</div></>)}
   </div><h3>{tr(item.title)}</h3><p>{tr(item.description)}</p><ol className="onboard-tour-steps">{tr(item.steps.map((text, i) => <li key={text} data-active={phase === i}><span>{tr(i + 1)}</span>{tr(text)}</li>))}</ol>
  </div><div className="onboard-tour-bottom"><button type="button" aria-label={tr("Replay this tip")} onClick={() => setReplay(value => value + 1)}><Icon name="refresh" size={13}/>{tr("Replay")}</button><span>{tr(page + 1)} / {tr(tour.length)}</span><button type="button" onClick={() => onChange((page + 1) % tour.length)}>{tr(page === tour.length - 1 ? 'Start again' : 'Next tip')}<Icon name="chevron" size={14}/></button></div>
 </section>;
}
