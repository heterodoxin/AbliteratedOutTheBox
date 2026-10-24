import { useState } from 'react';
import { tr } from '../lib/i18n';
import { chooseChatFolder, setChatFolder, useChatFolder } from '../lib/chat-folder';
import { pushToast } from '../lib/store';
import { plainFailureText } from '../lib/plain-errors';
import { Icon } from './ui';

/**
 * The chat's project folder, above the message box on the left: its name (the full path on hover), or "Select a folder". A click opens the
 * system folder picker; the small x goes back to the default folder. The assistant works inside this folder and nowhere else.
 */
export function FolderChip({ chatId, disabled }: { chatId: string | null | undefined; disabled?: boolean }) {
  const folder = useChatFolder(chatId);
  const [busy, setBusy] = useState(false);
  if (!chatId) return null;
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); } catch (e) { pushToast('error', tr("The folder was not changed"), plainFailureText(e instanceof Error ? e.message : String(e)).message); }
    finally { setBusy(false); }
  };
  const label = folder ? folder.name : tr("Select a folder");
  return <div className="folderchip" data-set={folder ? 'true' : undefined} data-missing={folder && !folder.exists ? 'true' : undefined}>
    <button type="button" className="folderchip__pick" disabled={busy || disabled} title={folder ? folder.path : tr("Choose the folder this chat works in")} aria-label={folder ? tr("Project folder: {0}. Change it.", [folder.name]) : tr("Select a folder")} onClick={() => void run(() => chooseChatFolder(chatId))}>
      <Icon name="folder" size={15}/><span>{label}</span>
    </button>
    {folder ? <button type="button" className="folderchip__x" disabled={busy || disabled} title={tr("Use the default folder")} aria-label={tr("Use the default folder")} onClick={() => void run(() => setChatFolder(chatId, null))}><Icon name="x" size={12}/></button> : null}
  </div>;
}
