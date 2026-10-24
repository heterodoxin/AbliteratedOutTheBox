import { useRef, useState, type ReactNode } from 'react';
import { tr, localeTag } from '../lib/i18n';
import { api } from '../lib/client';
import { go, answerChatTool, continueChatInNewChat, continueTaskInNewChat, dismissChat, selectSession, stopChat, stopLoop } from '../lib/store';
import { pushToast } from '../lib/store';
import { askQuestion } from '../lib/ask';
import { LOOP_MAX_FAILURES, loopEvery } from '../lib/loop';
import { plainFailure } from '../lib/plain-errors';
import { userError } from '../lib/format';
import { type BoardAction, type BoardCard, type BoardColumn, type BoardReason } from '../lib/board';
import { useBoard, refreshBoard } from '../lib/use-board';
import { Btn, Icon, IconBtn, type IconName } from './ui';
import { FloatingMenu } from './FloatingMenu';

/**
 * The task board: what is running, what waits for a yes, what found a problem and what finished, in four columns. The cards come from
 * src/lib/board.ts; this file only draws them and calls the same endpoints and store functions the rest of the app uses. Cards are not
 * dragged: they move by themselves when the facts change.
 */

const COLUMNS: { id: BoardColumn; name: string; hint?: string; tone: 'run' | 'wait' | 'bad' | 'done' }[] = [
  { id: 'running', name: 'Running', tone: 'run' },
  { id: 'waiting', name: 'Waiting for you', hint: 'Stopped at a question only you can answer.', tone: 'wait' },
  { id: 'attention', name: 'Needs attention', hint: 'Found a problem. Stays here until you deal with it.', tone: 'bad' },
  { id: 'done', name: 'Done', hint: 'Finished cleanly in the last 7 days.', tone: 'done' },
];

const ACTION: Record<BoardAction, { label: string; icon?: IconName }> = {
  'stop-task': { label: 'Stop', icon: 'stop' },
  'run-again': { label: 'Run again', icon: 'refresh' },
  'open-task': { label: 'Open task', icon: 'pencil' },
  'open-chat': { label: 'Open chat', icon: 'chat' },
  'stop-chat': { label: 'Stop chat', icon: 'stop' },
  'stop-loop': { label: 'Stop loop', icon: 'stop' },
  approve: { label: 'Approve', icon: 'check' },
  deny: { label: 'Deny' },
  continue: { label: 'Continue in a new chat', icon: 'chat' },
  dismiss: { label: 'Dismiss', icon: 'x' },
  'view-result': { label: 'View result' },
};

/* ------------------------------------------------------------------ words */

/** "every 10 min": short enough for a card; whole hours keep the loop's own words ("every hour"). */
const everyShort = (ms: number) => (ms % 3_600_000 === 0 || ms < 60_000 ? loopEvery(ms) : tr('every {0} min', [Math.round(ms / 60_000)]));

const span = (ms: number) => {
  const min = Math.floor(ms / 60_000);
  if (min < 1) return tr('under 1 min');
  if (min < 60) return tr('{0} min', [min]);
  if (min < 48 * 60) return tr('{0} h', [Math.floor(min / 60)]);
  return tr('{0} d', [Math.floor(min / 1440)]);
};
const ago = (ms: number) => (ms < 60_000 ? tr('just now') : tr('{0} ago', [span(ms)]));
const clock = (at: number) => new Date(at).toLocaleTimeString(localeTag(), { hour: '2-digit', minute: '2-digit' });
/** "today 07:00", "yesterday", "Mon 07:00" for a day that has a time on it. */
function day(at: number, now: number) {
  const d = new Date(at), n = new Date(now);
  const midnight = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((midnight(n) - midnight(d)) / 86_400_000);
  if (days <= 0) return tr('today {0}', [clock(at)]);
  if (days === 1) return tr('yesterday');
  return d.toLocaleDateString(localeTag(), { weekday: 'short', day: 'numeric', month: 'short' });
}

export function cardTime(card: BoardCard, now: number): string {
  switch (card.timeKind) {
    case 'started': return tr('started {0}', [ago(now - card.at)]);
    case 'waiting': return tr('waiting {0}', [span(now - card.at)]);
    case 'next': return card.nextAt && card.nextAt > now ? tr('next in {0}', [span(card.nextAt - now)]) : tr('answering now');
    case 'stopped': return tr('stopped {0}', [now - card.at < 86_400_000 ? clock(card.at) : day(card.at, now)]);
    default: return card.source === 'task' ? day(card.at, now) : now - card.at < 86_400_000 ? ago(now - card.at) : day(card.at, now);
  }
}

function ReasonText({ reason }: { reason: BoardReason }) {
  let text = '', raw: string | undefined;
  switch (reason.kind) {
    case 'task-failed': case 'chat-failed': {
      const plain = plainFailure(reason.detail ?? '');
      const known = plain.kind !== 'plain';
      text = known ? tr(plain.message) : tr(userError(String(reason.detail ?? '').slice(0, 240)) || 'It did not finish.');
      // The words a person reads are the sentence; the original text stays one click away for experts.
      raw = known ? (plain.detail ?? reason.detail) : undefined;
      if (!known && !reason.detail) text = tr('It did not finish.');
      break;
    }
    case 'task-stopped': text = tr('This run was stopped before it finished.'); break;
    case 'task-closed': text = tr('Strata closed while this was running, so it did not finish.'); break;
    case 'task-needs-yes': text = tr('Finished, but wanted to change something and could not ask. Nothing was changed.'); break;
    case 'loop-failed': text = tr('Stopped: {0} runs in a row did not work.', [LOOP_MAX_FAILURES]); break;
    default: text = reason.ask ? askQuestion(reason.ask) : reason.fallback ?? '';
  }
  return <span>{text}{raw ? <details className="bcard__detail"><summary>{tr('Details')}</summary><pre>{raw}</pre></details> : null}</span>;
}

/* ------------------------------------------------------------------- card */

function Card({ card, now, busy, canRun, run, openMenu }: { card: BoardCard; now: number; busy: boolean; canRun: boolean; run: (card: BoardCard, action: BoardAction) => void; openMenu: (card: BoardCard, anchor: HTMLElement) => void }) {
  const titleId = `bcard-${card.key.replace(/[^a-z0-9]+/gi, '-')}`;
  const source = card.source === 'task' ? <><Icon name="clock" size={14}/>{tr('Scheduled task')}</> : card.source === 'chat' ? <><Icon name="chat" size={14}/>{tr('Chat')}</> : <><Icon name="refresh" size={14}/>{tr('Loop · {0}', [everyShort(card.everyMs ?? 0)])}</>;
  const where = card.folder || card.meta ? <div className="bcard__where">{card.folder ? <><Icon name="folder" size={14}/>{card.folder}</> : null}{card.folder && card.meta ? ' · ' : ''}{card.meta ? tr(card.meta.text, card.meta.params) : null}</div> : null;
  const button = (action: BoardAction, variant: 'primary' | 'secondary' | 'ghost') => {
    const a = ACTION[action];
    const off = busy || (action === 'run-again' && !canRun);
    return <Btn key={action} size="sm" variant={variant} icon={variant === 'ghost' ? undefined : a.icon} disabled={off} title={action === 'run-again' && !canRun ? tr('Load a model first') : undefined} onClick={() => run(card, action)}>{tr(a.label)}</Btn>;
  };
  const iconFor = (action: BoardAction, label?: string) => <IconBtn key={action} name={ACTION[action].icon ?? 'x'} title={tr(label ?? ACTION[action].label)} disabled={busy} onClick={() => run(card, action)}/>;

  let actions: ReactNode;
  if (card.column === 'waiting') {
    actions = card.approve
      ? <>{button('approve', 'primary')}{button('deny', 'secondary')}<span className="spacer"/>{iconFor('open-chat')}</>
      : <>{button(card.primary, 'secondary')}</>;
  } else if (card.column === 'done') {
    actions = <>{card.chip ? <span className={`bchip${card.chip.tone === 'good' ? ' bchip--good' : ''}`}><Icon name="check" size={12}/>{tr(card.chip.text)}</span> : null}<span className="spacer"/>{button(card.primary, 'ghost')}{iconFor('dismiss')}</>;
  } else {
    const variant = card.column === 'attention' ? 'primary' : 'secondary';
    actions = <>{button(card.primary, variant)}<span className="spacer"/>
      {card.menu.length === 1 ? iconFor(card.menu[0]) : card.menu.length > 1
        ? <button type="button" className="st-btn st-btn--icon" aria-haspopup="menu" aria-label={tr('More actions for {0}', [card.title])} title={tr('More actions')} disabled={busy} onClick={e => openMenu(card, e.currentTarget)}><Icon name="ellipsis" size={19}/></button> : null}</>;
  }

  return <article className="bcard" aria-labelledby={titleId} data-source={card.source}>
    <div className="bcard__top"><span className="bsrc">{source}</span><span className="btime">{cardTime(card, now)}</span></div>
    <h3 className="bcard__title" id={titleId}>{card.title}</h3>
    {where}
    {card.line ? <p className="bcard__line">{card.line}</p> : null}
    {card.reason ? <div className={`bcard__why bcard__why--${card.reason.tone}`}><Icon name={card.reason.tone === 'bad' ? 'error' : card.reason.tone === 'warn' ? 'warning' : 'shield'} size={16}/><ReasonText reason={card.reason}/></div> : null}
    {card.busy ? <div className="bcard__bar bcard__bar--busy" aria-hidden="true"><i/></div> : null}
    <div className="bcard__actions">{actions}</div>
  </article>;
}

/* ------------------------------------------------------------------ board */

export function TaskBoard({ query, canRun, onViewResult }: { query: string; canRun: boolean; onViewResult: (taskId: string) => void }) {
  const { board, ready, tasks } = useBoard();
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [menu, setMenu] = useState<{ card: BoardCard; anchor: HTMLElement } | null>(null);
  const now = Date.now();
  const q = query.trim().toLowerCase();
  const shown = (list: BoardCard[]) => (q ? list.filter(c => `${c.title} ${c.line ?? ''} ${c.folder ?? ''}`.toLowerCase().includes(q)) : list);
  const guard = useRef(new Set<string>());

  async function run(card: BoardCard, action: BoardAction) {
    if (guard.current.has(card.key)) return;
    const task = card.source === 'task' ? tasks.find(t => t.id === card.id) : undefined;
    const local = () => {
      if (action === 'open-chat') { selectSession(card.id); go('chat', card.id); }
      else if (action === 'open-task') go('tasks', card.id);
      else if (action === 'view-result') onViewResult(card.id);
      else if (action === 'stop-chat') stopChat(card.id);
      else if (action === 'stop-loop') stopLoop(card.id);
      else if (action === 'approve' || action === 'deny') { if (card.approve) answerChatTool(card.id, card.approve.messageId, card.approve.toolId, action === 'approve'); }
      else if (action === 'continue') { if (task) continueTaskInNewChat(task); else continueChatInNewChat(card.id); }
      else if (action === 'dismiss' && card.source !== 'task') dismissChat(card.id);
      else return false;
      return true;
    };
    if (local()) return;
    guard.current.add(card.key);
    setBusy(b => ({ ...b, [card.key]: true }));
    try {
      if (action === 'stop-task') await api.post(`/api/tasks/${card.id}/stop`);
      else if (action === 'run-again') await api.post(`/api/tasks/${card.id}/run`);
      else if (action === 'dismiss') await api.patch(`/api/tasks/${card.id}`, { ackedAt: new Date().toISOString() });
      window.dispatchEvent(new Event('strata:tasks'));
    } catch (e) {
      pushToast('error', 'That did not work', e instanceof Error ? e.message : String(e));
    } finally {
      guard.current.delete(card.key);
      setBusy(b => ({ ...b, [card.key]: false }));
      refreshBoard();
    }
  }

  const openMenu = (card: BoardCard, anchor: HTMLElement) => setMenu(m => (m?.card.key === card.key ? null : { card, anchor }));
  const total = Object.values(board.columns).reduce((n, c) => n + c.length, 0);

  return <div className="board-wrap">
    <p className="board-note"><Icon name="info" size={16}/>{tr('Cards move on their own as things happen. A card in Needs attention stays there until you deal with it or dismiss it.')}</p>
    <p className="sr-only" role="status" aria-live="polite">{board.need ? tr('{0} need your attention', [board.need]) : tr('Nothing needs you right now.')}</p>
    <div className="board">
      {COLUMNS.map(col => {
        const cards = shown(board.columns[col.id]);
        const headId = `bcol-${col.id}`;
        return <section key={col.id} className={`bcol bcol--${col.id}`} data-tone={col.tone} aria-labelledby={headId}>
          <div className="bcol__head"><span className="bcol__dot" aria-hidden="true"/><h2 className="bcol__name" id={headId}>{tr(col.name)}</h2><span className="bcol__count" aria-label={tr('{0} cards', [cards.length])}>{cards.length}</span></div>
          {col.hint ? <p className="bcol__hint">{tr(col.hint)}</p> : null}
          {cards.map(card => <Card key={card.key} card={card} now={now} busy={!!busy[card.key]} canRun={canRun} run={run} openMenu={openMenu}/>)}
          {!cards.length ? <p className="bcol__empty">{ready ? tr(q && total ? 'No matching cards.' : 'Nothing here right now.') : tr('Loading tasks…')}</p> : null}
        </section>;
      })}
    </div>
    {board.upcoming.length ? <div className="upnext" role="region" aria-label={tr('Coming up')}><Icon name="clock" size={18}/><span className="navgroup__label">{tr('Coming up')}</span>
      {board.upcoming.map((u, i) => <span className="upnext__item" key={u.id}>{i ? <span className="sep" aria-hidden="true"/> : null}<a href={`#/tasks/${u.id}`}>{u.title}</a> <span className="i">{day(u.at, now)}</span></span>)}</div> : null}
    {menu ? <FloatingMenu anchor={menu.anchor} title={tr('More actions for {0}', [menu.card.title])} width={230} onClose={() => setMenu(null)}>
      {menu.card.menu.map(action => <div key={action}>{action === 'dismiss' && menu.card.menu.length > 1 ? <hr className="board-menu-sep"/> : null}<button type="button" role="menuitem" className="menu__item" onClick={() => { const card = menu.card; setMenu(null); void run(card, action); }}><Icon name={ACTION[action].icon ?? 'x'}/>{tr(ACTION[action].label)}</button></div>)}
    </FloatingMenu> : null}
  </div>;
}
