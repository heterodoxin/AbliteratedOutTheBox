import { useEffect, useState } from 'react';
import { tr } from '../lib/i18n';
import { useFolderPrompt } from '../lib/folder-pick';
import { Btn, Dialog } from './ui';

/** The fallback for a system without a folder chooser: say so in plain words and take the folder's path as text. */
export function FolderPathDialog() {
  const prompt = useFolderPrompt();
  const [path, setPath] = useState('');
  useEffect(() => { if (prompt) setPath(''); }, [prompt]);
  const submit = () => { const text = path.trim(); if (text) prompt?.resolve(text); };
  return <Dialog open={!!prompt} compact title={tr("Type the folder path")} onClose={() => prompt?.resolve(null)}
    actions={<><Btn onClick={() => prompt?.resolve(null)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!path.trim()} onClick={submit}>{tr("Use this folder")}</Btn></>}>
    <form className="stack" onSubmit={event => { event.preventDefault(); submit(); }}>
      <p className="small" role="status">{tr(prompt?.message ?? '')}</p>
      <label className="input-field"><span>{tr("Folder path")}</span>
        <input className="st-input" autoFocus value={path} onChange={event => setPath(event.target.value)} placeholder={tr("/home/you/Documents")} spellCheck={false} aria-label={tr("Folder path")}/>
      </label>
    </form>
  </Dialog>;
}
