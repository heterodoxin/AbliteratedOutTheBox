import { useEffect, useState } from 'react';
import { tr } from '../lib/i18n';
import { isForkJob, type ForkLane, type TimeSkipJob, type TimeSkipRun } from '../lib/time-skip';
import { activityWords, phaseFraction } from '../lib/time-skip-activity';
import { stopTimeSkip } from '../lib/store';
import { Btn } from './ui';

const clock = (ms: number) => { const t = Math.max(0, Math.round(ms / 1000)); return t < 60 ? `${t}s` : `${Math.floor(t / 60)}m ${String(t % 60).padStart(2, '0')}s`; };

/**
 * While a Time-skip run works, in the chat: the saved version (A) and the one being written (B) with a stream of light between them, what
 * the assistant is doing right now, how long, and Stop. It is shown from the first moment, so a long run never looks like nothing happening.
 */
export function TimeSkipWorking({ run, sessionId }: { run: TimeSkipRun; sessionId: string }) {
  const [now, setNow] = useState(() => Date.now());
  const [stopping, setStopping] = useState(false);
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t); }, []);
  const fraction = phaseFraction(run.job?.phase);
  const fork = isForkJob(run.job);
  return <section className="tsw" aria-label={tr("Time-skip is working")}>
    <div className="tsw__stage" aria-hidden="true">
      <div className="tsw__ver"><span>A · {tr("Original")}</span><i style={{ width: '92%' }}/><i style={{ width: '100%' }}/><i style={{ width: '78%' }}/><i style={{ width: '96%' }}/></div>
      <div className="tsw__flow"><i/><i/><i/></div>
      <div className="tsw__ver tsw__ver--b"><span>B · {tr("Revising")}</span><i style={{ width: '70%' }}/><i style={{ width: '88%' }}/><i style={{ width: '55%' }}/></div>
    </div>
    {fork ? <ForkProgress job={run.job!}/> : null}
    <div className="tsw__bar">
      <div className="tsw__step" role="status"><strong>{activityWords(run.job)}</strong><span className="small muted">{clock(now - run.startedAt)}</span>
        {fork ? null : <div className="tsw__prog" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)}><i style={{ width: `${Math.round(fraction * 100)}%` }}/></div>}</div>
      <Btn size="sm" icon="stop" disabled={stopping || !run.job?.id} onClick={() => { setStopping(true); void stopTimeSkip(sessionId, run.id).finally(() => setStopping(false)); }}>{tr("Stop")}</Btn>
    </div>
  </section>;
}

const laneFraction = (l: ForkLane | null | undefined) => (l && l.total > 0 ? Math.min(1, (l.state === 'done' ? l.total : Math.max(0, l.step - 1)) / l.total) : 0);

function Lane({ name, lane, waitingText }: { name: string; lane: ForkLane | null | undefined; waitingText: string }) {
  const pct = Math.round(laneFraction(lane) * 100);
  const words = !lane ? waitingText : lane.waiting ? waitingText : lane.line || (lane.text ? lane.text : tr('Starting…'));
  return <div className="tsf__lane" data-state={lane?.state ?? 'waiting'}>
    <span className="tsf__name">{name}</span>
    <div className="tsf__prog" role="progressbar" aria-label={tr('Progress of {0}', [name])} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}><i style={{ width: `${pct}%` }}/></div>
    <span className="tsf__words">{words}</span>
  </div>;
}

/**
 * The fork design (notes-split-agent8.md, item 11): the plan as a list, the one step the assistant chose to check and branch at (with its one-line
 * why), and the two agents' progress. B starts at that step from A's work so far.
 */
export function ForkProgress({ job }: { job: TimeSkipJob }) {
  const steps = job.plan?.steps ?? [], lever = job.lever;
  const reachedA = job.a?.step ?? 0;
  const short = !lever && steps.length > 0 && steps.length < 3;
  return <div className="tsf">
    {steps.length ? <ol className="tsf__plan" aria-label={tr("The plan")}>
      {steps.map((s, i) => {
        const n = i + 1, state = job.a && job.a.state === 'done' || n < reachedA ? 'done' : n === reachedA ? 'running' : 'pending';
        return <li key={s.id} data-state={state} data-lever={lever?.step === n || undefined}>
          <span className="tsf__no" aria-hidden="true">{state === 'done' ? '✓' : n}</span>
          <span>{s.text}{lever?.step === n ? <small className="tsf__why"><strong>{tr("Checking here matters most")}</strong>{lever.why ? ' · ' + lever.why : ''}{lever.defaulted ? ' · ' + tr('chosen by default') : ''}</small> : null}</span>
        </li>;
      })}
    </ol> : null}
    {short ? <p className="small muted tsf__note">{tr("This is a short job, so there is one version. Nothing is worth branching.")}</p> : null}
    <div className="tsf__lanes">
      <Lane name="A" lane={job.a} waitingText={tr('Waiting to start')}/>
      {short ? null : <Lane name="B" lane={job.b} waitingText={lever ? tr('Starts when A reaches step {0}', [lever.step]) : tr('Waiting for the plan')}/>}
    </div>
  </div>;
}
