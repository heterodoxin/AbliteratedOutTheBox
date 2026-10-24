/**
 * The tools a person can switch on and off (Settings -> Tools), in plain words, and which Harness plugins each one is.
 *
 * `plugins` are plugin ids of the app's agent preset (server/agent/preset.mjs). `web` marks the two halves of the Harness's one
 * `tool-web` plugin (open pages and search), which the preset switches through that plugin's own `fetch` / `search` options.
 * Everything not listed here (the persona, compaction, instructions, the clock) is not a tool and is always on.
 *
 * `asks` is only a plain-words hint for the screen: in Manual mode the approval gate asks before anything that is not provably look-only.
 */
export const TOOL_GROUPS = [
  { id: 'shell', area: 'computer', name: 'Run commands', about: 'Lets the assistant run commands, for example to open a folder or start a program.', asks: true, plugins: ['tool-pwsh', 'tool-bash'] },
  { id: 'files', area: 'computer', name: 'Read and change files', about: 'Lets the assistant read, create and edit files in your chat folder.', asks: true, plugins: ['tool-fs'] },
  { id: 'find', area: 'computer', name: 'Find files and text', about: 'Lets the assistant look for files by name and search inside them. It only reads.', asks: false, plugins: ['tool-fs-search'] },
  { id: 'jobs', area: 'computer', name: 'Long-running jobs', about: 'Lets the assistant start a slow command in the background and check on it later.', asks: true, plugins: ['tool-jobs'] },
  { id: 'fetch', area: 'web', name: 'Open web pages', about: 'Lets the assistant read a web page you point it to.', asks: true, web: 'fetch' },
  { id: 'search', area: 'web', name: 'Search the web', about: 'Lets the assistant search the web when it needs fresh information.', asks: true, web: 'search' },
  { id: 'helpers', area: 'together', name: 'Helper assistants', about: 'Lets the assistant start helpers that work on parts of a task at the same time.', asks: false, plugins: ['delegation'] },
  { id: 'goals', area: 'together', name: 'Goals', about: 'Lets you give a goal with /goal. The assistant keeps working until it is done.', asks: false, plugins: ['command-goal', 'tool-goal'] },
  { id: 'todo', area: 'together', name: 'To-do list', about: 'Lets the assistant keep a checklist while it works on a big task.', asks: false, plugins: ['tool-todo'] },
  { id: 'skills', area: 'together', name: 'Skills', about: 'Lets the assistant use your saved skills when they fit the task.', asks: false, plugins: ['skill-filesystem', 'tool-skill'] },
  { id: 'present', area: 'together', name: 'Show results', about: 'Lets the assistant show a page or file in the preview next to the chat.', asks: false, plugins: ['present'] },
];

export const GROUP_IDS = TOOL_GROUPS.map(g => g.id);

/** The preset's plugin rows without the groups the person switched off. */
export function withoutDisabled(plugins, off) {
  const gone = new Set();
  for (const group of TOOL_GROUPS) if (off.has(group.id)) for (const id of group.plugins ?? []) gone.add(id);
  return plugins.filter(p => !gone.has(p.id));
}
