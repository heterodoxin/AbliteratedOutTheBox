// QA report F03: a dialog opened from the narrow-window menu closes the menu and takes the keyboard.
// (The real behaviour is checked in a browser by scripts/qa2-flows.mjs; these keep the pieces from being taken out again.)
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const read = path => readFile(new URL(path, import.meta.url), 'utf8').then(s => s.replace(/\r\n/g, '\n'));

test('every dialog the menu opens closes the menu first, and the task dialogs stay mounted while it is closed', async () => {
  const tree = await read('../src/app/TaskTree.tsx');
  assert.match(tree, /import \{ pushToast, setNavOpen \} from '\.\.\/lib\/store'/);
  for (const opener of ['setEditor(blankTask())', 'setEditor({ ...t })', 'setRemove(t)']) assert.ok(tree.includes(`setNavOpen(false); ${opener}`), `${opener} closes the menu`);
  assert.match(tree, /if \(collapsed\)\n\s+return dialogs;/, 'a closed menu still renders the dialogs');
  const side = await read('../src/app/Sidebar.tsx');
  for (const opener of ['setRenameId(menuId)', 'setDeleteId(menuId)', 'setRecentOpen(true)', 'setRemoveWorkspaceId(w.id)', 'setRenameWsId(w.id)', 'setQuitOpen(true)']) assert.ok(side.includes(`setNavOpen(false); ${opener}`), `${opener} closes the menu`);
});

test('a surface that opens moves the keyboard in even when animation frames are held back', async () => {
  const focus = await read('../src/lib/focus.ts');
  assert.match(focus, /const fallback = setTimeout\(/);
  assert.match(focus, /clearTimeout\(fallback\)/);
  assert.match(focus, /!el\.contains\(document\.activeElement\) && isTopSurface\(el\)/);
});
