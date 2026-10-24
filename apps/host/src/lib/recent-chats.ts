import type { Session } from './types';

export function recentChats(sessions: Session[], activeId?: string | null) {
  const ordered = sessions.filter(session => !session.archived)
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  const current = ordered.find(session => session.id === activeId);
  const latest = ordered.filter(session => session.id !== current?.id).slice(0, 5);
  const ids = new Set([...latest, ...(current ? [current] : [])].map(session => session.id));
  // Activity order keeps number shortcuts stable when simply switching chats.
  return ordered.filter(session => ids.has(session.id));
}

export function recentChatDescription(session: Session) {
  const message = session.messages.filter(message => message.role === 'user').at(-1) ?? session.messages.at(-1);
  const text = message?.content.replace(/\s+/g, ' ').trim() || 'No messages yet';
  return text.length > 160 ? `${text.slice(0, 157).trimEnd()}…` : text;
}
