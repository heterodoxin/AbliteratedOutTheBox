import { tr } from '../lib/i18n';
import { Icon, type IconName } from './ui';
import type { TuneProgress } from '../lib/native-tune';
import type { Outcome } from '../lib/tune-outcome';
import { tokS } from '../lib/tune-outcome';
import { tunePhase, tuneSpeeds } from '../lib/tune-phase';

const STEPS: { icon: IconName; title: string; running: string }[] = [
  { icon: 'cpu', title: 'Checking your PC', running: 'Looking at your graphics card and memory' },
  { icon: 'gauge', title: 'Measuring the base speed', running: 'Timing your current settings' },
  { icon: 'sliders', title: 'Trying settings', running: 'Trying other settings, one at a time' },
  { icon: 'thinking', title: 'Confirming the best one', running: 'Repeating the test to be sure' },
  { icon: 'disk', title: 'Saving', running: 'Keeping it for this model; you can undo it' },
];

const fmt = (n: number) => (Math.round(n * 10) / 10).toFixed(1);

function Spark({ values }: { values: number[] }) {
  if (values.length < 2) return <div className="tune-spark tune-spark--empty" aria-hidden="true"/>;
  const lo = Math.min(...values), hi = Math.max(...values), span = Math.max(0.0001, hi - lo);
  const pts = values.map((v, i) => [(i / (values.length - 1)) * 220, 56 - ((v - lo) / span) * 46] as const);
  const line = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
  const [lx, ly] = pts[pts.length - 1];
  return <svg className="tune-spark" viewBox="0 0 220 64" preserveAspectRatio="none" aria-hidden="true"><path d={`${line} L220 64 L0 64Z`} className="tune-spark__fill"/><path d={line} className="tune-spark__line"/><circle cx={lx} cy={ly} r="4"/></svg>;
}

/** The tuner at work: five steps, each with its own icon and state, and the speed it has measured so far. */
export function TuneRunView({ progress, ratio, saving }: { progress: TuneProgress | undefined; ratio: number; saving: boolean }) {
  const at = saving ? 4 : tunePhase(progress?.stage);
  const speeds = tuneSpeeds(progress);
  const base = speeds[0]?.tok_s ?? null;
  const now = speeds.length ? speeds[speeds.length - 1].tok_s : null;
  const left = progress?.minutesLeft ?? null;
  const sub = (i: number): string => {
    if (i < at) return i === 1 && base != null ? tr("Starting point: {0} tokens per second", [fmt(base)]) : '';
    if (i === at && progress?.step) return progress.step;
    return i === at ? tr(STEPS[i].running) : '';
  };
  return <div className="tune-run">
    <ol className="tune-steps2" aria-label={tr("Tuning steps")}>
      {STEPS.map((s, i) => {
        const state = i < at ? 'done' : i === at ? 'run' : 'pending';
        return <li key={s.title} className="tune-step" data-s={state} data-step={s.icon} aria-current={state === 'run' ? 'step' : undefined}>
          <span className="tune-step__badge"><Icon name={state === 'done' ? 'check' : s.icon} size={22}/></span>
          <span className="tune-step__text"><strong>{tr(s.title)}</strong><small>{sub(i)}</small></span>
          <span className="tune-step__state">{tr(state === 'done' ? 'Done' : state === 'run' ? 'Running' : 'Waiting')}</span>
        </li>;
      })}
    </ol>
    <aside className="tune-live" aria-live="polite">
      <span className="label">{tr("Speed now")}</span>
      <div className="tune-live__big">{now != null ? fmt(now) : '—'}<small>{tr("tokens per second")}</small></div>
      <Spark values={speeds.map(x => x.tok_s)}/>
      <div className="tune-live__bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(ratio * 100)}><i style={{ width: `${Math.max(3, Math.round(ratio * 100))}%` }}/></div>
      <p className="small muted">{left != null ? tr("about {0} min left", [left]) : tr("This takes about 20 minutes.")}</p>
      <ul className="tune-live__list">{speeds.slice(-6).map((x, i) => <li key={`${x.label}-${i}`}><span>{x.label === 'base' ? tr("Your settings") : x.label}</span><b data-up={base != null && x.label !== 'base' && x.tok_s > base ? 'true' : undefined}>{fmt(x.tok_s)}</b></li>)}</ul>
    </aside>
  </div>;
}

/**
 * The result of a tune, in one layout whatever happened: a round tile (green, amber, red or quiet), a headline, one sentence, up to three facts.
 * The buttons are the dialog's footer.
 */
export function TuneOutcomeCard({ outcome, details }: { outcome: Outcome; details?: string | null }) {
  const o = outcome;
  const headline = ({
    faster: o.gainPct != null ? tr('{0}% faster', [Math.round(o.gainPct)]) : tr('Faster answers'),
    already_fastest: tr('Already the fastest'), still_fast: tr('Still as fast as before'), slower: tr('Slower than before'),
    busy_pc: tr("Couldn't measure reliably"), unsure: tr("Couldn't tell right now"), failed: tr("Tuning didn't finish"), stopped: tr('Stopped'),
  } as Record<string, string>)[o.code];
  const lead = ({
    faster: tr('Your answers now arrive sooner. Quality is the same.'),
    already_fastest: tr('Nothing we tried beat your current settings, so nothing was changed.'),
    still_fast: tr('Your saved tuning is still working as well as when it was saved.'),
    slower: tr('It is slower than before. You can tune again.'),
    unsure: tr('Your tuning was kept. Try the check again later.'),
    busy_pc: o.reason === 'not_tried' ? tr('Your PC was too busy to try other settings, so nothing was changed. Try again when other programs are closed.') : tr("Your PC was busy during the test, so we couldn't measure reliably. Nothing was changed. Try again when other programs are closed."),
    failed: o.reason === 'no_memory' ? tr("There wasn't enough free memory to test. Nothing was changed. Close other programs and try again.") : o.reason === 'start_failed' ? tr('The engine could not start with these settings. Nothing was changed.') : tr('The engine stopped while we were measuring. Nothing was changed.'),
    stopped: tr('Nothing was changed.'),
  } as Record<string, string>)[o.code];
  const facts: [string, string][] = [];
  if (o.code === 'faster') { facts.push([tr('Before'), `${tokS(o.baseTokS)} ${tr('tok/s')}`], [tr('After'), `${tokS(o.tunedTokS)} ${tr('tok/s')}`]); if (o.presetName) facts.push([tr('Saved as'), o.presetName]); }
  else if (o.code === 'already_fastest' || o.code === 'still_fast') { facts.push([tr('Speed'), `${tokS(o.baseTokS)} ${tr('tok/s')}`]); if (o.minutes != null) facts.push([tr('Took'), tr('{0} min', [Math.max(1, Math.round(o.minutes))])]); if (o.presetName) facts.push([tr('Saved as'), o.presetName]); }
  else if (o.code === 'busy_pc') { facts.push([tr('Speed'), o.baseTokS != null ? `${tokS(o.baseTokS)} ${tr('tok/s')}` : '—']); if (o.noisePct != null) facts.push([tr('Variation'), `${Math.round(o.noisePct)}%`]); facts.push([tr('Changed'), tr('Nothing')]); }
  else if (o.code === 'failed' || o.code === 'stopped' || o.code === 'unsure' || o.code === 'slower') facts.push([tr('Changed'), tr('Nothing')]);
  return <div className="tune-outcome" data-tone={o.tone} data-code={o.code}>
    <div className="tune-outcome__tile" aria-hidden="true"><Icon name={o.icon} size={34}/></div>
    <div className="tune-outcome__body" role="status">
      <h3>{headline}</h3>
      <p>{lead}</p>
      {facts.length ? <dl className="tune-outcome__facts">{facts.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl> : null}
      {details ? <details className="tune-outcome__details"><summary>{tr('Details')}</summary><pre>{details}</pre></details> : null}
    </div>
  </div>;
}
