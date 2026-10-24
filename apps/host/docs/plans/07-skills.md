# 07 · Skills

**Route** `#/skills` · **File** `src/screens/Skills.tsx` (121 lines — the smallest screen)

Reusable instruction sets the model can be given. Today it is a single list,
which makes it the thinnest screen in the app and the one where "substance" is
most obviously missing.

## What it is today

- **Header** with a count pill (`3 enabled`) and an "Add from folder" action.
- **A search field.**
- **One `listrow` per skill** — name, description, an enable toggle, and a source
  badge (`workspace`).
- **An empty state** that distinguishes "no skills" from "no match".
- **A "How skills work" explainer card** at the bottom.

## Where it falls short

1. ~~A skill is a name and a sentence.~~ **Fixed.** The disclosure existed but
   showed *triggers* — the words that activate a skill, not what it does. The
   host now sends the instruction text itself (bundled skills carry it;
   workspace skills read their own `SKILL.md`, which the description had always
   claimed was the source without ever reading it). Verified: 364 characters of
   real instruction render, and **no skill lacks a body**.
2. **Enabled/disabled is the only control.** No ordering, no scoping to a
   workspace, no per-conversation activation.
3. **The list is flat and unsorted.** No grouping by source, no indication of
   which fired recently.
4. **"Add from folder" is opaque.** It gives no hint of the expected layout, file
   format, or where to put files.
5. **No feedback loop.** Nothing shows whether a skill has ever been used, or
   which conversations it affected.
6. **The explainer card is a wall of prose at the bottom**, below the fold on a
   short window, where it cannot help anyone who is confused at the top.
7. **The screen sits alone in the sidebar** with no sibling, which makes the
   single flat row read as unfinished.

## The plan

### Substance — make a skill inspectable
1. ~~A detail view per skill.~~ **Done.** *Read instructions* opens the
   instruction text, with the triggers kept beneath it as a secondary
   detail — they answer "when does this fire", which is a different question
   from "what does it do".
2. ~~Usage evidence.~~ **Done** — see the shipped note above. The triggers were
   decorative until this was built: nothing consulted them at runtime.
3. **Scope control** — enable per workspace, since a skill written for one
   project is noise in another.

### Hierarchy
4. **Move the explainer to the top as a one-line subtitle**, and put the detail
   in a disclosure for people who want it. Prose at the bottom cannot answer a
   question asked at the top.
5. **Group by source** (`Workspace` / `Built in`) once there is more than one
   source; a single flat list will not scale.

### Fluidity
6. **Toggling a skill shows the change immediately** and reconciles quietly — the
   optimistic update is the interaction, and a spinner on a switch is a worse
   switch.
7. **Adding a skill animates the new row in** with the standard entrance, so the
   result of "Add from folder" is visibly connected to the action.

## Motion

| Interaction | Spec |
|---|---|
| Toggle | Knob travel `--st-t-control`; no page movement — built |
| New skill added | `st-enter`, `--st-t-local`, 6 px |
| Detail expand | grid track `0fr → 1fr`, `--st-t-local` |
| Search filtering | Rows fade, `--st-t-control`. No reflow animation |

## Verification

- Assert each skill row can reveal its full instruction text.
- **Shipped as `npm run flow skilluse`**: every skill renders a usage line, and
  the line is `Not used yet` rather than blank when it has never fired.
- Assert the empty state distinguishes *no skills* from *no match* — already
  correct; keep it under test.
