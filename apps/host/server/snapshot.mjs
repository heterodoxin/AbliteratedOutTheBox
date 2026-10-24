/** Native Save As for a rendered Monitor PNG. The client cannot choose a host path. */
import { writeFile } from 'node:fs/promises';
import { exec } from './util.mjs';
import { utf8Script } from './powershell-utf8.mjs';

export function snapshotBytes(data) {
  if (typeof data !== 'string' || data.length > 24 * 1024 * 1024 || !/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(data)) throw new Error('Expected a PNG screenshot smaller than 18 MB.');
  const bytes = Buffer.from(data.slice(data.indexOf(',') + 1), 'base64');
  if (bytes.length < 24 || !bytes.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10])) || bytes.toString('ascii', 12, 16) !== 'IHDR') throw new Error('Invalid PNG screenshot.');
  return bytes;
}
export async function saveSnapshot(data, { platform = process.platform, run = exec, write = writeFile } = {}) {
  const bytes = snapshotBytes(data);
  const name = `Strata-Monitor-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  let result;
  try {
    if (platform === 'darwin') result = await run('osascript', ['-e', `POSIX path of (choose file name with prompt "Save Monitor screenshot" default name "${name}")`]);
    else if (platform === 'win32') result = await run('powershell.exe', ['-NoProfile', '-STA', '-Command', utf8Script(`Add-Type -AssemblyName System.Windows.Forms; $dialog = New-Object System.Windows.Forms.SaveFileDialog; $dialog.Filter = 'PNG image (*.png)|*.png'; $dialog.FileName = '${name}'; if ($dialog.ShowDialog() -eq 'OK') { $dialog.FileName }`)]);
    else result = await run('zenity', ['--file-selection', '--save', '--confirm-overwrite', '--title=Save Monitor screenshot', `--filename=${name}`, '--file-filter=PNG images | *.png']);
  } catch (e) {
    if (/User canceled|\(-128\)/i.test(e.stderr ?? '') || (platform === 'linux' && e.code === 1)) return { cancelled: true };
    if (e.code === 'ENOENT') return { download: true, name };
    throw new Error('The save dialog could not open. Try saving the screenshot again.');
  }
  const path = result.stdout.trim();
  if (!path) return { cancelled: true };
  await write(path, bytes);
  return { saved: true, name: path.split(/[\\/]/).at(-1) };
}
