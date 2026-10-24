# 01 · Chat

**Route** `#/chat` — the landing screen · **File** `src/screens/Chat.tsx` (779 lines, the largest in the app)

Chat is where the product either holds up over a long session or does not. It is
the only screen a user might sit in for hours, so it carries the strictest rules
in the system: the reading measure, the follow-the-stream behaviour, focus mode,
and the telemetry row.

## What it is today

- **Header** — title (editable inline), a subtitle carrying model · message count ·
  token count, and two controls: a generating pill that appears only while
  working, and the context-rail toggle. The `···` overflow was removed last round.
- **Transcript** — `.chat__inner`, 920 px max, centre-aligned. Assistant prose is
  capped at **56ch**, measured at **68 characters** at 10, 16 and 22 px. Code
  blocks and tables escape the cap and use the full bubble.
- **Tool cards** — inline approval cards with Enter/Esc, a scope badge, a result
  line and a duration.
- **Composer** — `Do anything` placeholder, keyboard hints as the box's second
  line, toolbar row, Send. Glass material at the regular tier.
- **Telemetry** — three centred, clickable groups (turns/steps/tok/s, tokens/cache
  hit, context %) opening detail popovers.
- **Context rail** — a grid track that animates 0 → 340 px, with a per-conversation
  breakdown.
- **Focus mode** — `⌘⇧F` withdraws the chrome to an ambient bar.
- **Scroll behaviour** — follows the stream only when already at the bottom;
  per-conversation scroll memory; a scroll-driven reading-progress hairline.

## Where it falls short

1. **The header subtitle is dense and low-value.** `Starter · 0.5B Q4_K_M · 2
   messages · 244 tokens` duplicates what the telemetry row and the sidebar
   engine chip already say. Three places report the same state.
2. **The header has no model identity while idle.** The generating pill only
   exists mid-request, so at rest the bar says nothing about what is loaded.
3. ~~Assistant messages have no per-message time or cost.~~ **Checked and
   wrong.** A two-turn conversation renders a footer under *both* assistant
   messages with identical structure (`236 tokens · 56.6 tok/s · 4.2 s`), because
   the `done` frame writes `tokens`, `tps` and `durationMs` onto every message and
   the footer renders whenever any of the three exists. The figures were always
   per-turn; only the *actions* below are missing.
4. ~~No way to retry or edit a turn.~~ **Fixed.** Every assistant turn carries
   *Ask again* (`retryLast`) and every user turn carries *Edit and ask again*
   (`rewindTo`). Verified: retry drops the reply and replaces it without
   duplicating the turn; edit rewinds the transcript and refills the composer
   focused. Actions reveal on hover and on focus-within, so they are reachable by
   keyboard. *Branch* and *Continue* are now shipped — see below.
5. **Reasoning is a collapsed block with no affordance.** The disclosure animates
   open but there is no indication of how long the model thought.
6. **The intro state is four static suggestion cards.** They do not reflect the
   workspace, and they do not change with the loaded model.
7. **Nothing marks unread turns** when a long generation finishes while the user
   is scrolled up.
8. **The transcript has no `content-visibility`.** In a 200-turn conversation
   every message stays in layout, which is the one place the app can get slow.
9. **Focus mode dims earlier turns to 45% but offers no way to move between
   turns** — no next/previous, no jump-to-latest.

## The plan

### Hierarchy — one hero, less restatement
1. **Strip the subtitle to the conversation only.** `2 messages · 244 tokens`
   moves entirely into the telemetry popover; the subtitle becomes the model name
   and nothing else, or disappears when the rail toggle is present.
2. **Add a persistent model chip to the header**, left of the controls: the
   loaded model with its state dot. This is the DSH shape and it means the bar is
   informative at rest, not only mid-request.

### Substance — make the transcript auditable
3. **Per-turn actions on the existing footer** — retry, edit-and-resend, branch.
   The figures are already correct and per-turn; what the footer lacks is
   anything to *do* with a turn after reading it.
4. ~~Retry and edit-and-resend.~~ **Done.** Both live in the store rather than
   the screen, because each has to touch the message list — doing it from a
   component ends up duplicating the stream handling that already lives there.

5. ~~Branch and continue.~~ **Done.** *Branch from here* forks a session at any
   turn, leaving the original intact — verified: the branch carries only the
   turns up to that point, the original is byte-identical afterwards, and both
   are listed. *Continue from here* appears on the newest reply only, since an
   older one already has its answer beneath it, and is sent as an ordinary user
   turn rather than a hidden mode, because a continuation *is* a request and the
   transcript should show it.

   **Branching exposed a real bug.** `Chat` resolves its session as
   `location.param ?? activeId`, so the URL wins when it names a session. Every
   other path into a session navigates (`newSession(); go('chat', id)`); the
   first version of `branchFrom` only set `activeSession`, so the sidebar showed
   the new branch while the transcript kept showing the original *indefinitely* —
   a feature that looked broken while its data was correct. `branchFrom` now
   navigates, and the flow asserts the URL follows.
5. **Reasoning blocks show their duration** (`Thought for 4.2s`) in the summary
   row, using the disclosure already built.
6. **`content-visibility: auto` on message blocks** with `contain-intrinsic-size`
   so off-screen turns cost nothing. This is the single largest performance win
   available and it is invisible when it works.

### Fluidity
7. ~~Jump-to-latest pill.~~ **Done** — and it only appears once the newest
   message is genuinely off screen, so it does not sit over a short transcript
   with nothing to jump to. When scrolled up during or after a stream, a floating
   capsule appears above the composer with the count of new turns. Clicking scrolls
   smoothly; the pill fades on a token.
8. **Unread mark on the session row** in the sidebar while the user is elsewhere.
9. ~~Turn navigation in focus mode.~~ **Done.** Focus mode is for reading a long
   conversation, and reading needs a next. `⌥↑`/`⌥↓` step between your own turns
   and `⌘↓` returns to the latest; the focus bar teaches the keys, since that bar
   is the only chrome the mode keeps. Verified: five distinct stops going down,
   each press retreating on the way back, and `⌘↓` landing at gap 0.

   **The first version stuck.** It scrolled a turn to `offsetTop − 16` and then
   tested `offsetTop > scrollTop + 8` — which the turn it had just moved to still
   satisfied, so the second press did nothing. The tolerance has to exceed the
   offset the turn is positioned at.

### Substance — the intro
10. **Suggestion cards become workspace-aware** — derived from the actual file
    list rather than a fixed array, and one of them becomes "Continue where you
    left off" when a previous session exists.

## Motion

| Interaction | Spec |
|---|---|
| Turn entering | `st-enter`, `--st-t-local`, 6 px lift — already correct |
| Jump-to-latest appear | `st-enter`, `--st-t-control`, `--st-from-y: 4px` |
| Reasoning disclosure | `interpolate-size` on `::details-content`, `--st-t-local` — built |
| Retry / branch | No transition; the transcript re-renders. Movement here would be noise |
| Rail open/close | grid track 0 ↔ 340 px, `--st-t-surface` — built |

Everything is dropped to opacity-only under reduced motion, per rule 6.

## Verification

- The per-turn footer assertion is **already satisfied** and should be kept as a
  regression guard rather than written as a new check: every assistant message
  with a completed stream must render `.st-msg__meta`.
- New assertion: with 100 seeded turns, **paint time on scroll stays within 1.5×
  of a 10-turn conversation** — the `content-visibility` proof.
- `npm run measure` must still report 68 characters after any header or column
  change.
- Focus-mode turn navigation added to the shortcut table and to `flow focus`.
