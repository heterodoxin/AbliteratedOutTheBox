/** Open a file in the person's text editor (not the program a .json happens to belong to: that can be a browser). */
import { execFile } from 'node:child_process';
import { launch } from './chat-files.mjs';
import { externalEnvironment } from './external-env.mjs';

/** The command that opens `file` as text on this system, or null when only the file's own program is known. */
export async function textEditorCommand(file, { platform = process.platform, ask = askSystem } = {}) {
  if (platform === 'win32') return ['notepad.exe', [file]];
  if (platform === 'darwin') return ['open', ['-t', file]];
  // Linux: the program the system opens plain text with (gedit, kate, mousepad, ...), through its desktop entry.
  const entry = (await ask('xdg-mime', ['query', 'default', 'text/plain'])).trim();
  if (/^[\w.+-]+\.desktop$/.test(entry)) {
    const gtk = (await ask('sh', ['-c', 'command -v gtk-launch'])).trim();
    if (gtk) return [gtk, [entry, file]];
    const gio = (await ask('sh', ['-c', 'command -v gio'])).trim();
    if (gio) {
      for (const dir of ['/usr/share/applications', '/usr/local/share/applications', `${process.env.HOME ?? ''}/.local/share/applications`]) {
        const path = `${dir}/${entry}`;
        if ((await ask('test', ['-f', path], true)) === 'yes') return [gio, ['launch', path, file]];
      }
    }
  }
  return null;
}

function askSystem(command, args, yesNo = false) {
  return new Promise(resolve => {
    execFile(command, args, { timeout: 3000, env: externalEnvironment() }, (error, stdout) => resolve(yesNo ? (error ? 'no' : 'yes') : (error ? '' : String(stdout))));
  });
}

/** Start the editor and return at once. Falls back to the file's own program (xdg-open) when no text editor is known. */
export async function openTextFile(file, { platform = process.platform, ask, recorder = null } = {}) {
  const found = await textEditorCommand(file, { platform, ask });
  const [command, args] = found ?? ['xdg-open', [file]];
  launch(command, args, { recorder });
  return { command, args };
}
