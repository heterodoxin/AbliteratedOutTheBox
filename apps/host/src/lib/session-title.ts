import {tr} from './i18n';
import type {Session} from './types';
/** Untitled, empty chats are interface copy; chosen names and prompt titles are user content. */
export function sessionTitle(session:Pick<Session,'title'|'messages'|'defaultTitle'>){
 return session.title==='New conversation'&&!session.messages.length&&session.defaultTitle!==false?tr('New conversation'):session.title;
}
