import type { Session } from './types';

export function sessionLog(session: Session) {
  return JSON.stringify({ schema: 'strata.session-log/1', exportedAt: new Date().toISOString(), session }, null, 2);
}

export function downloadSessionLog(session: Session) {
  const url = URL.createObjectURL(new Blob([sessionLog(session)], { type: 'application/json' }));
  const link = document.createElement('a');
  const name = session.title.replace(/[^\p{L}\p{N}._-]+/gu, '-').slice(0, 80) || 'chat';
  link.href = url; link.download = `${name}-session-log.json`;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
