import { tr } from './i18n';
import type { IconName } from '../components/ui';

/** The question and its one muted line for an approval card. The host names a sentence (server/agent/ask.mjs); the words are here, translated. */
export interface Ask { id: string; params?: (string | number)[]; why?: string; detail?: string }

export function askQuestion(ask: Ask): string {
  const p = (ask.params ?? []).map(String);
  switch (ask.id) {
    case 'open-known-folder': return tr("Open your {0} folder in File Explorer?", p);
    case 'open-folder': return tr("Open the folder {0} in File Explorer?", p);
    case 'open-file': return tr("Open the file {0}?", p);
    case 'open-explorer': return tr("Open File Explorer?");
    case 'run-program': return tr("Run a program: {0}?", p);
    case 'run-script': return tr("Run the script {0}?", p);
    case 'run-steps': return tr("Run {0} steps on your computer?", p);
    case 'run-command': return tr("Run a command on your computer?");
    case 'delete': return tr("Delete {0}?", p);
    case 'create-file': return tr("Create the file {0}?", p);
    case 'create-folder': return tr("Create the folder {0}?", p);
    case 'write-file': return tr("Write to the file {0}?", p);
    case 'change-file': return tr("Change the file {0}?", p);
    case 'copy': return tr("Copy {0}?", p);
    case 'move': return tr("Move {0}?", p);
    case 'rename': return tr("Rename {0}?", p);
    case 'git': return tr("Run git {0}?", p);
    case 'install': return tr("Install packages with {0}?", p);
    case 'stop-task': return tr("Stop a background task?");
    case 'open-page': return tr("Open the web page {0}?", p);
    case 'use-tool': return tr("Use the connected tool {0}?", p);
    default: return tr("Use the tool {0}?", p);
  }
}

function askWhy(id?: string): string | null {
  switch (id) {
    case 'opens-window': return tr("A window will open. Nothing is changed.");
    case 'run-program-why': return tr("It can change your computer.");
    case 'delete-why': return tr("It is removed from your computer.");
    case 'replaces-if-exists': return tr("Replaces the file if it already exists.");
    case 'changes-repo': return tr("It changes the project's files or history.");
    case 'install-why': return tr("It downloads software and changes your computer.");
    case 'run-steps-why': return tr("More than one thing happens. Details shows each step.");
    case 'run-command-why': return tr("Details shows exactly what it does.");
    case 'use-tool-why': return tr("It can act on the service you connected.");
    default: return null;
  }
}

/** The muted line under the question: the folder or file for "open" questions, otherwise what to know before answering. */
export function askSub(ask: Ask): string {
  const why = askWhy(ask.why);
  if (ask.id.startsWith('open-') && ask.detail) return ask.detail;
  return why ?? ask.detail ?? '';
}

export function askIcon(ask: Ask): IconName {
  if (ask.id.startsWith('open-')) return ask.id === 'open-page' ? 'link' : 'folder';
  if (ask.id === 'delete') return 'trash';
  if (['create-file', 'write-file', 'change-file', 'copy', 'move', 'rename'].includes(ask.id)) return 'file';
  if (ask.id === 'create-folder') return 'folder-plus';
  if (['run-program', 'run-script', 'run-steps', 'run-command', 'git', 'install'].includes(ask.id)) return 'bolt';
  return 'tool';
}
