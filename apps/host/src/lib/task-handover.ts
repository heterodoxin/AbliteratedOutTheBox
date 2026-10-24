import { tr, localeTag } from './i18n';
import type { Task } from './tasks';
import type { Session } from './types';
import { loopEndNote } from './loop';
import { sessionTitle } from './session-title';

/**
 * "Continue in a new chat" on a task: the first message of the new chat. It carries what the task found, what it did and the files it made, in the
 * words the person reads, so the conversation can go on from the result (a finished task, or one that stopped and needs a person).
 */
const when = (iso?: string) => (iso ? new Date(iso).toLocaleString(localeTag(), { dateStyle: 'medium', timeStyle: 'short' }) : '');

/** The task's saved answer as its parts: the answer, the list of what it did, and the note that something needed a yes ("What it did" is how the host writes it, server/task-report.mjs). */
export function splitTaskOutput(output: string): { found: string; did: string[]; attention: string } {
  let text = String(output ?? '').trim();
  let attention = '';
  const note = /\n\n> ([\s\S]*)$/.exec(text);
  if (note) { attention = note[1].replace(/\n> /g, '\n').trim(); text = text.slice(0, note.index).trim(); }
  const cut = /\n\n---\n\n\*\*What it did\*\*\n\n([\s\S]*)$/.exec(text);
  const did = cut ? cut[1].split('\n').map(l => l.replace(/^-\s*/, '').trim()).filter(Boolean) : [];
  return { found: (cut ? text.slice(0, cut.index) : text).trim(), did, attention };
}

const MAX_STEPS = 10, MAX_STEP = 160;

export const taskHandoverTitle = (task: Task) => tr('Continuing from the task “{0}”', [task.title]);

export function taskHandoverMessage(task: Task): string {
  const failed = !!task.error;
  const ran = when(task.lastRunAt);
  const head = `**${taskHandoverTitle(task)}**${ran ? ` ${failed ? tr('(ran {0}, did not finish).', [ran]) : tr('(ran {0}, finished).', [ran])}` : ''}`;
  const parts = [head];
  if (failed) parts.push(`**${tr('What went wrong')}**\n\n${String(task.error).trim()}`);
  const { found, did, attention } = splitTaskOutput(task.output ?? '');
  if (found) parts.push(`**${tr('What it found')}**\n\n${found}`);
  if (attention) parts.push(`**${tr('Needs your attention')}**\n\n${attention}`);
  if (did.length) {
    // the steps of a long run are many and long: the first ones, shortened
    const shown = did.slice(0, MAX_STEPS).map(l => `- ${l.length > MAX_STEP ? `${l.slice(0, MAX_STEP - 1)}…` : l}`);
    if (did.length > MAX_STEPS) shown.push(`- ${tr('…and {0} more steps', [did.length - MAX_STEPS])}`);
    parts.push(`**${tr('What it did')}**\n\n${shown.join('\n')}`);
  }
  const files = (task.files ?? []).filter(f => f && typeof f.path === 'string' && f.path);
  if (files.length) parts.push(`**${tr('Files')}**\n\n${files.map(f => `- ${f.path}`).join('\n')}`);
  parts.push(tr('Please help me take it from here.'));
  return parts.join('\n\n');
}

export const chatHandoverTitle = (session: Pick<Session, 'title' | 'messages' | 'defaultTitle'>) => tr('Continuing from the chat “{0}”', [sessionTitle(session)]);

/**
 * "Continue in a new chat" on a chat or a loop that went wrong (the board): what was asked last, what went wrong or what the last answer
 * was, and, for a loop, how it ended, so the new chat can take it from there.
 */
export function chatHandoverMessage(session: Session): string {
  const real = session.messages.filter(m => !m.local);
  const answer = [...real].reverse().find(m => m.role === 'assistant');
  const asked = [...real].reverse().find(m => m.role === 'user');
  const parts = [`**${chatHandoverTitle(session)}**`];
  const loop = session.loop;
  if (loop) {
    const ended = loop.endedWhy && loop.endedWhy !== 'stopped' && loop.endedWhy !== 'closed' ? `\n\n${loopEndNote(loop.endedWhy, loop.answers)}` : '';
    parts.push(`**${tr('The loop')}**\n\n${loop.prompt}${ended}`);
  } else if (asked?.content) parts.push(`**${tr('What I asked')}**\n\n${asked.content.trim().slice(0, 1500)}`);
  if (answer?.error) parts.push(`**${tr('What went wrong')}**\n\n${String(answer.error).trim()}`);
  const said = (answer?.content ?? '').trim() || (loop?.lastText ?? '');
  if (said) parts.push(`**${tr('The last answer')}**\n\n${said.slice(0, 2000)}`);
  parts.push(tr('Please help me take it from here.'));
  return parts.join('\n\n');
}
