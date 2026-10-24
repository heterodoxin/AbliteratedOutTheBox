# 06 · Workspace

**Route** `#/workspace` · **File** `src/screens/Workspace.tsx` (219 lines)

What the model can see. The screen that makes "local" concrete: these are your
files, and this is the boundary.

## What it is today

- **A file tree** — `treerow` per node, directories expandable, sizes on files,
  a `Select` for workspace root.
- **A "What Strata may touch" card** — the scope explanation.
- **Filters** — a few `Select` controls and a search field.

## Where it falls short

1. **Directories use the `layers` icon.** The kit gained real `folder` and `file`
   glyphs two rounds ago and the sprite carries them; only the sidebar uses
   `folder`. This is the most visible inconsistency in the app — the file tree is
   the one place a folder icon is unambiguous.
2. **Files have no icon at all**, only a size. A directory and a file differ by
   the presence of a chevron.
3. **Tree expand/collapse is instant.** The workspace tree in the *sidebar*
   animates a grid track; this one, the main event, does not. The two should not
   behave differently.
4. **The detail panel shows metadata, not the file.** Selecting a node *does*
   render a panel (`Workspace.tsx:168`) with kind, path, item count or size,
   changed time, Copy path and "Add as reference" — but never the file's
   contents. Reading a file the model can read is the obvious next step and the
   panel is already there to hold it.
5. **Modified time is shown in the detail panel but not in the tree**, so
   scanning for what changed recently means clicking through nodes one at a
   time.
6. **No indication of what the model has actually read.** The tool runs are in
   the transcript; the workspace does not reflect them, though it is the natural
   place to see "these three files were opened".
7. **The scope card states the rule but not the consequence** — no count of files
   or total size in reach.

## The plan

### Substance — make the boundary legible
1. **Real icons**: `folder` for directories, `file` for files, both already in
   the sprite. Distinguish common types by tint only, never by adding a second
   icon family.
2. **Extend the existing detail panel with contents.** It already carries the
   header (kind, path, size, changed) and two actions; add the file body beneath
   as a read-only code surface, fetched on selection. Metadata without contents
   is a panel that answers every question except the one that was asked.
3. ~~"Read by the model" markers.~~ **Done.** Files that appear in a tool call in the
   current session get a subtle marker and a count, so the boundary is not just
   stated but demonstrated.
4. **The scope card gains figures**: `128 files · 4.2 MB in reach`.

### Hierarchy
5. **Tree is the hero**, filters are secondary. Today the filter row and the tree
   have comparable weight.

### Fluidity
6. **Tree rows animate open and closed** using the same grid-track technique as
   the sidebar tree and the context rail — one technique, three places.
7. **The preview pane slides in from the right** like the context rail, on the
   same `--st-t-surface` track animation, rather than appearing.
8. **Keyboard navigation**: arrows to move and expand, Enter to preview. A tree
   without keyboard traversal is a list with extra steps.

## Motion

| Interaction | Spec |
|---|---|
| Node expand/collapse | grid track `0fr → 1fr`, `--st-t-local` |
| Preview open/close | grid track, `--st-t-surface` — same as the context rail |
| Row selection | Background only, `--st-t-micro` |
| Icon swap on expand | None. The chevron rotates, `--st-t-control`; the icon does not change |

## Verification

- Assert **no directory renders the `layers` glyph** — the audit can check icon
  names against node kind.
- Assert selecting a file renders a preview with non-empty content.
- Add `flow workspace`: expand a directory, select a file, assert the preview
  path matches, collapse again.
