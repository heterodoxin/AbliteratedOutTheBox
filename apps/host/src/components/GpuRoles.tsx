import { tr } from '../lib/i18n';
import { useId, useState } from 'react';
import { Btn, Icon } from './ui';
import {
  CPU_ONLY_LINE, NO_CARD_LINE, OFF_AT_OR_BELOW_GB, gpuLabel, noteText, order, recommend, setRole, shortName, vramOn, warnings,
  type Role, type RoleCard, type Roles, type RoleWarning,
} from '../lib/gpu-roles';

const ROLE_WORD: Record<Role, string> = { main: 'Main', helper: 'Helper', off: 'Off' };
const HINT: Record<Role, string> = { main: 'Does the main work', helper: 'Holds part of the model', off: 'Not used' };

/** The one plain line above the cards, built from translated pieces. */
export function recommendationLine(cards: RoleCard[], vendor?: string | null): string {
  const rec = recommend(cards, vendor);
  if (rec.main === null) return tr('No graphics card that Strata can use was found.');
  const byi = new Map(cards.map(c => [c.index, c]));
  const m = byi.get(rec.main)!;
  if (cards.length === 1) return tr('Strata will use your {0}.', [shortName(m)]);
  const why = { same: 'the cards are alike', memory: 'it has more memory', faster: 'it is faster', only: 'it is the only card', none: '' }[rec.why];
  let s = tr('{0} is Main: {1}.', [shortName(m), tr(why)]);
  const helpers = Object.keys(rec.roles).map(Number).filter(i => rec.roles[i] === 'helper').map(i => shortName(byi.get(i)!));
  if (helpers.length) s += ' ' + tr(helpers.length > 1 ? '{0} help.' : '{0} helps.', [helpers.join(', ')]);
  for (const i of Object.keys(rec.roles).map(Number)) if (rec.roles[i] === 'off') {
    const code = rec.reasons[i], c = byi.get(i)!;
    s += ' ' + tr('{0} starts Off: {1}', [shortName(c), code === 'cannot_run' ? tr('Strata cannot run it.') : code === 'igpu' ? tr('built-in graphics.') : tr('{0} GB or less.', [OFF_AT_OR_BELOW_GB])]);
  }
  return s;
}

function warningText(w: RoleWarning, cards: RoleCard[]): string {
  const c = w.card !== null ? cards.find(x => x.index === w.card) : undefined;
  if (w.code === 'cannot_run') return tr('{0} cannot run Strata.', [c ? shortName(c) : '']);
  if (w.code === 'igpu_main') return tr('{0} is built-in graphics; it is slow as Main.', [c ? shortName(c) : '']);
  if (w.code === 'cpu_only') return tr(w.text === CPU_ONLY_LINE ? CPU_ONLY_LINE : NO_CARD_LINE);
  return tr('Not enough memory for this setup: about {0} GB is read from the drive.', [w.text.match(/about (\d+) GB/)?.[1] ?? '']);
}

export function noteLine(code: string | null, cards: RoleCard[]): string {
  if (!code) return '';
  const kind = code.split(':')[0], c = cards.find(x => x.index === Number(code.split(':')[1]));
  if (kind === 'old_main_helper') return tr('{0} is now a Helper.', [c ? shortName(c) : '']);
  if (kind === 'promoted') return tr('{0} is now Main.', [c ? shortName(c) : '']);
  if (kind === 'only_main') return tr('The only card that is on has to be Main.');
  return noteText(code, cards);
}

/**
 * Main / Helper / Off for every graphics card (the mockup GpuRoles-mockup.html). One click changes a role; "Back to recommended" restores the
 * recommendation. Warnings are short and never block Load. `roles` are the roles in force (the recommendation until the person changes one).
 */
export function GpuRoles({ cards, roles, custom, recommendedRoles, vendor, onChange, onBack, memoryGb, ramGb, spillGb = 0, engineRunsCpuOnly = false, disabled = false, usedBy }: {
  cards: RoleCard[];
  roles: Roles;
  /** The person changed a role: "Your choice", and "Back to recommended" is offered. */
  custom: boolean;
  /** The roles Strata recommends (the tags and "Back to recommended" use them); the pure recommendation when not given. */
  recommendedRoles?: Roles;
  vendor?: string | null;
  onChange: (next: Roles) => void;
  onBack: () => void;
  memoryGb?: number | null;
  ramGb?: number | null;
  /** About how many GB of the model are read from the drive with these cards (0 when everything fits or it is unknown). */
  spillGb?: number;
  engineRunsCpuOnly?: boolean;
  disabled?: boolean;
  /** Names of the loaded models that already use a card ("In use by ..."). */
  usedBy?: (card: number) => string[];
}) {
  const titleId = useId();
  const [note, setNote] = useState('');
  const rec = { ...recommend(cards, vendor), ...(recommendedRoles ? { roles: recommendedRoles } : {}) };
  const single = cards.length === 1;
  const warns = warnings(roles, cards, { spillGb, engineRunsCpuOnly });
  const cardWarn = (i: number) => warns.filter(w => w.card === i);
  const globalWarn = warns.filter(w => w.card === null);
  const lst = order(roles, cards);
  const change = (idx: number, role: Role) => {
    const out = setRole(roles, cards, idx, role);
    setNote(noteLine(out.note, cards));
    onChange(out.roles);
  };
  const back = () => { setNote(''); onBack(); };
  const fix = (w: RoleWarning) => w.fix && change(w.fix.card, w.fix.role);
  const fixLabel = (w: RoleWarning) => {
    if (!w.fix) return '';
    const target = cards.find(c => c.index === w.fix!.card);
    return w.fix.role === 'off' ? tr('Turn it off') : tr('Make {0} Main', [target ? shortName(target) : '']);
  };
  const have = vramOn(roles, cards) + Math.max(0, (ramGb ?? 0) - 10);
  return <div className="gpuroles" aria-labelledby={titleId} data-custom={custom}>
    <div className="gpuroles__head">
      <span className="native-choice__label" id={titleId}>{tr('Graphics cards')}</span>
      {custom ? <button type="button" className="gpuroles__link" disabled={disabled} onClick={back}>{tr('Back to recommended')}</button> : null}
    </div>
    <div className="gpuroles__reco" data-custom={custom}>
      <Icon name="bolt" size={18}/>
      <p><strong>{tr(custom ? 'Your choice.' : 'Recommended.')}</strong> {custom ? tr('You changed the roles. Strata will use exactly these.') : recommendationLine(cards, vendor)}</p>
    </div>
    <div className="gpuroles__rows" role="list">
      {cards.map(c => {
        const role = roles[c.index] ?? 'off', isRec = rec.roles[c.index] === role, w = cardWarn(c.index), other = usedBy?.(c.index) ?? [];
        const shared = !!(c.integrated || c.uma || c.unified_gb);
        const options: Role[] = single ? ['main', 'off'] : ['main', 'helper', 'off'];
        return <div className="gpuroles__row" role="listitem" data-role={role} key={c.index}>
          <div className="gpuroles__top">
            <span className="gpuroles__ico"><Icon name="gpu" size={20}/></span>
            <span className="gpuroles__name">
              <strong>{gpuLabel(c)} · {c.name}{isRec ? <span className="gpuroles__tag">{tr('Recommended')}</span> : <span className="gpuroles__tag gpuroles__tag--neutral">{tr('Recommended: {0}', [tr(ROLE_WORD[rec.roles[c.index] ?? 'off'])])}</span>}</strong>
              <small>{shared ? tr("Uses your computer's memory (built-in graphics)") : tr('{0} GB memory', [Math.round(c.vram_gb)])} · {tr(HINT[role])}</small>
            </span>
            <span className="gpuroles__roles" role="radiogroup" aria-label={tr('Role of {0}', [gpuLabel(c)])}>
              {options.map(r => <button type="button" role="radio" key={r} data-r={r} aria-checked={role === r} disabled={disabled} onClick={() => change(c.index, r)}>{tr(ROLE_WORD[r])}</button>)}
            </span>
          </div>
          {other.length ? <div className="gpuroles__note gpuroles__note--info"><Icon name="info" size={16}/><p>{tr('In use by {0}', [other.join(', ')])}</p></div> : null}
          {w.map(x => <div className="gpuroles__note gpuroles__note--warn" key={x.code}><Icon name="warning" size={16}/><p>{warningText(x, cards)}</p>{x.fix ? <Btn size="sm" disabled={disabled} onClick={() => fix(x)}>{fixLabel(x)}</Btn> : null}</div>)}
        </div>;
      })}
    </div>
    <div className="gpuroles__toast" role="status" aria-live="polite">{note ? <div className="gpuroles__note gpuroles__note--info"><Icon name="info" size={16}/><p>{note}</p></div> : null}</div>
    {globalWarn.map(x => <div className="gpuroles__note gpuroles__note--warn" key={x.code}><Icon name="warning" size={16}/><p>{warningText(x, cards)}</p>{x.code === 'cpu_only' && custom ? <Btn size="sm" disabled={disabled} onClick={back}>{tr('Back to recommended')}</Btn> : null}</div>)}
    <details className="gpuroles__details">
      <summary>{tr('Details')}</summary>
      <div className="gpuroles__body">
        <dl>
          <dt>{tr('Main')}</dt><dd>{tr('The card that does the final steps of every answer: it writes the next words, runs the draft-ahead step and checks it, and keeps your conversation. Pick your fastest card with plenty of memory. There is only ever one.')}</dd>
          <dt>{tr('Helper')}</dt><dd>{tr('Holds a share of the model and its fast memory, so more of it fits on graphics cards instead of being read from your drive. Prompts are read about 20% faster.')}</dd>
          <dt>{tr('Off')}</dt><dd>{tr('Strata does not touch this card. Good for a card you use for something else, such as a game or your screen.')}</dd>
        </dl>
        {memoryGb ? <div className="gpuroles__fit" data-state={lst.length ? (spillGb >= 1 ? 'bad' : 'ok') : 'bad'}><span className="gpuroles__dot" aria-hidden="true"/><span>{lst.length ? (spillGb >= 1 ? tr('The model needs about {0} GB. These cards and your memory give about {1} GB, so about {2} GB would be read from your drive.', [Math.round(memoryGb), Math.round(have), Math.round(spillGb)]) : tr('The model needs about {0} GB. These cards and your memory give about {1} GB. It fits.', [Math.round(memoryGb), Math.round(have)])) : tr('Only your processor and memory are used.')}</span></div> : null}
        <div className="gpuroles__expert xs muted">{lst.length ? tr('For experts: the engine starts on cards {0} (Main last).', [lst.join(',')]) : null} {custom ? '' : tr('Not saved while you keep the recommendation: Strata picks again if you add or remove a card.')}</div>
      </div>
    </details>
  </div>;
}
