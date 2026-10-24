# 11 · Command palette

**Shortcut** `⌘K` · **File** `src/app/CommandPalette.tsx` (222 lines)

The keyboard path to everything. For a power user it replaces the sidebar
entirely; for a beginner it is the fastest way to discover what exists.

## What it is today

- **A modal with a bare search field** (`.st-input--bare`) and grouped results.
- **Groups** — Go to, Engine, Model, plus command entries.
- **Always mounted** and driven by `data-open`, so it animates both ways.
- **Portal-free** — it renders inside `.app`, and the modal is opaque.
- **Fuzzy-ish matching**, cursor movement with `↑`/`↓`, `Enter` to run, `Esc` to
  close, focus returned to the trigger on close.

## Where it falls short

1. ~~Matching is prefix-based on the label.~~ **Fixed.** `src/lib/fuzzy.ts` adds
   subsequence matching with a score that rewards consecutive characters and word
   starts, returning the matched indices so hits can be explained. Verified:
   `mtr` → **Monitor**, `wsp` → **Workspace**, `dl` → **Models**, each with the
   matched characters highlighted. The plan's own example was wrong — `mtx`
   cannot match *Monitor*, which has no `x` — and was replaced with `mtr`.
2. ~~Commands and destinations are mixed.~~ **Fixed.** `>` scopes to commands,
   `#` to conversations, `@` to workspaces, and the empty state names all three.
   Verified: `>unload` → *Unload model*, `@` → the workspace.
3. **No recent or frequent section.** The first row is always `Chat`, whether or
   not that is where anyone goes.
4. ~~Results do not show what they will do.~~ **Fixed.** Entries carry a `danger`
   flag; restart, unload and re-run-setup render with a danger-toned icon and a
   distinct label weight, so an action that changes something does not read as
   navigation.
5. ~~No hint of the shortcut.~~ **Fixed.** A `keys` field renders real `<kbd>`
   chips, starting with `⌘ /` on Keyboard shortcuts.
6. **The accent colour is used for the selected row**, which conflicts with the
   rule that accent marks the primary action — in a list, selection is not
   primary action.
7. **Empty state is a single line** with no suggestion of what can be typed.

## The plan

### Substance
1. **Fuzzy matching with match highlighting** — subsequence matching with the
   matched characters emphasised, so `mtx` finds *Monitor* and shows why.
2. **Scoped prefixes**: `>` for commands, `#` for conversations, `@` for
   workspaces. Three prefixes cover everything the palette can reach.
3. **A Recents group at the top**, from actual navigation history, collapsed once
   the user starts typing.
4. **Destructive commands are marked** with the danger tone and their effect
   stated (`Restart engine — reloads the model, about 20 s`).
5. **Show the shortcut** on the right of every entry that has one, so the palette
   teaches the keyboard rather than replacing it.
6. ~~Search conversations.~~ **Fixed.** Sessions and workspaces are palette
   entries, reached with `#` and `@`.

### Hierarchy
7. **Selection uses the neutral raised surface**, not the accent. Accent stays
   for the primary action on a screen.

## Motion

| Interaction | Spec |
|---|---|
| Open/close | `st-enter` / reverse, `--st-t-local` — built and interruptible |
| Filtering | Rows fade, `--st-t-control`; results do not animate position |
| Selection moving | Background only, `--st-t-micro`. `↑`/`↓` must not animate |
| Scoped prefix entered | Group headers collapse, `--st-t-control` |

Selection deliberately does **not** move smoothly: at keyboard repeat rates an
animated highlight lags the key and feels broken.

## Verification

- Shipped as `npm run flow palette`. It asserts focus lands inside on open and
  stays inside after fourteen `Tab` presses, then checks `mtr` → *Monitor* with
  the match highlighted, `tune` → *Engine tuning* by alias, `>unload` scoped, and
  `@` → the workspace.
- **Three of the original items were also fixed beyond the plan:** aliases
  (`tune`, `api`, `stats` find entries whose labels do not contain them), which
  the plan had not anticipated but which the alias test immediately exposed.
- Assert destructive entries render a danger marker.
- Assert the palette is fully operable by keyboard with no pointer.
