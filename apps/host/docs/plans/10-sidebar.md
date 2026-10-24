# 10 · Sidebar and navigation

**File** `src/app/Sidebar.tsx` (417 lines — the largest app-level surface)

Restructured last round to the harness shape: logo → hide → New session → Skills
→ Workspaces → footer. It is the spine of the app and the thing a user touches
most often after the composer.

## What it is today

- **Collapse row** with the `panel` icon, its own row under the logo.
- **New session** in the accent material, using the `new-chat` duotone glyph.
- **Skills** as a single flat top-level row.
- **Workspaces** — header with search / collapse-all / add-folder, then a tree of
  workspaces with sessions nested under each, relative-time meta, hover-reveal
  delete.
- **Footer** — engine chip, Monitor, Serving, `… More` (Models, Engine tuning,
  Settings), theme toggle.
- **Rail mode** at 64 px, with labels hidden and titles as tooltips.

## Where it falls short

1. **Skills sits alone.** One flat row between New session and Workspaces. It was
   meant to pair with Plugins, which was cut, so the row now reads as an orphan.
2. **"Hide sidebar" is a permanent label.** Once collapsed, the control that
   restores it is an icon in a 64 px rail; the expanded label is doing no work
   after the first week.
3. **The `… More` label is inherited from the harness**, where it holds a dozen
   items. Ours holds three. It is a menu with a shrug for a name.
4. ~~No pinning.~~ **Fixed.** `Session.pinned` had existed in the type since the
   beginning and was never written to. Rows now carry a pin control, pinned
   conversations sort above recency within their workspace, and the pin stays
   visible once set. Verified: pinning the older of two conversations lifts it
   to the top. **Archive now ships too** — see below.
5. ~~Hover-reveal delete is hard to discover.~~ **Fixed.** Right-click opens a
   menu (Pin · Rename · Delete) and `Delete`/`Backspace` on a focused row arms
   the confirm, so the destructive action is reachable without a pointer and
   still requires confirming. The hover button remains for pointer users.
6. **The engine chip is the fourth place** the loaded model is reported (header,
   sidebar, telemetry, Monitor).
7. ~~No keyboard traversal.~~ **Fixed.** Arrow keys move between rows in visual
   order, Home and End jump to the ends, and entering with no row focused starts
   from the appropriate end. Verified: `ArrowDown` walks *Skills → workspace →
   conversation → conversation → Monitor*, and `End` reaches *More*.
8. **The theme toggle is an icon with no label**, next to a text row, at the
   bottom where it is least expected.

## The plan

### Substance
1. **Resolve Skills.** Either fold it into the More menu (three items becomes
   four and the orphan row disappears), or pair it with a second item. A single
   flat row is the one layout that reads as unfinished either way.
2. **Rename `… More` to `Settings`**, which is what three of its items are, and
   move Engine tuning and Models where they already are. If a menu is needed,
   name it for its contents.
3. **Pin and archive sessions.** Pinned rows sit above the workspace tree;
   archive hides without deleting. `Session.pinned` already exists in the type
   and is unused.
4. ~~Make delete discoverable.~~ **Done**, along with *Rename*, *Pin* and
   *Archive*, which had no affordance at all.

   Archive is distinct from delete on purpose: archiving is *"done with this for
   now"*, deleting is *"never want to see it again"*, and conflating the two
   makes people keep things they have finished with. An archived conversation
   leaves the list but **stays findable by search** — typing a name is an
   explicit request to find it, and hiding it then would be the opposite of
   helpful. Verified both directions.
5. **Full keyboard traversal** — `↑`/`↓` between rows, `→`/`←` to expand and
   collapse, `Enter` to open, `⌫` to delete. The tree is a tree; it should behave
   like one everywhere, including the two other trees in the app.

### Hierarchy
6. **One place reports the model.** Keep the engine chip — it is the most useful
   of the four — and let the header chip and telemetry reference it rather than
   restate it.

### Fluidity
7. **Session rows animate into their workspace group** when created, so the
   connection between "New session" and the list is visible.
8. **The rail transition already animates** the grid track; make sure labels
   cross-fade rather than being clipped, which is visible at 64 px today.

## Motion

| Interaction | Spec |
|---|---|
| Rail collapse | grid track 268 ↔ 64 px, `--st-t-surface` — built |
| Tree expand/collapse | grid track `0fr → 1fr`, `--st-t-local` — built |
| New session appearing | `st-enter`, `--st-t-local`, 6 px |
| More menu | `st-enter`, `--st-t-control`, 6 px — built |
| Row delete | Row collapses to zero height via grid track, `--st-t-local`, then unmounts |

## Verification

- Assert every row is reachable **and actionable by keyboard alone**.
- Assert the rail restores to the state it was left in across a reload.
- Assert session rows never exceed a bounded list without pagination or archive.
- `flow sidebar`: collapse, expand, open More, create a session, delete it.
