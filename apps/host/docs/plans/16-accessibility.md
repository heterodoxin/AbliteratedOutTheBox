# 16 · Accessibility and long sessions

**Across the app** · **Harness** `npm run contrast`, `npm run measure`, `npm run luminance`

Two audiences that overlap more than they look: people who need the interface to
accommodate them, and people who will sit in it for six hours. Both are served by
the same things — contrast, size, stillness, and not being shouted at.

## What is today — verified, not assumed

| Claim | Evidence |
|---|---|
| Every text pairing meets WCAG AA | `npm run contrast`: **37 pairings pass**, both themes, every interactive state |
| Contrast is measured on rendered pairings | Alpha is composited over ancestors; hover measured separately |
| Reading measure is 68 characters | `npm run measure` at 10, 16 and 22 px, normal and focus mode |
| Dark-mode brightness is budgeted | `npm run luminance`: no surface > 4,000 px² exceeds luminance 0.15 |
| Reduced motion substitutes, not silences | `transform: none`, `opacity: 1` verified under the OS setting |
| Materials are gated | Reduced-transparency and a manual override produce identical geometry |
| Controls are capsules, surfaces are not | `npm run audit` |
| No glass samples glass | `npm run luminance` walks every route |

## Where it falls short

1. ~~The command palette does not focus its own search field, and nothing traps
   focus.~~ **Fixed.** `src/lib/focus.ts` provides `useSurfaceFocus`, which waits
   for the surface to actually report `visible` before focusing, traps `Tab`
   inside while open, and restores the trigger on close. Measured after the fix:
   focus is on the palette input on open, is still inside after fourteen `Tab`
   presses, and returns to the trigger on `Escape`. Applied to the palette, the
   shortcuts dialog, the sampling drawer and the setup modal.
2. **Closed overlays are correctly inert, open ones are not.** `inert={!open}` is
   applied on the palette, context rail, sampling drawer, shortcuts dialog and the
   serving sheet — so a *closed* surface is out of the tab order, which is right.
   An *open* one still lets focus walk out into the page behind it.
3. **Landmarks are adequate; the outline is not.** One `<main>`, one `<aside>`,
   and `SectionHead` correctly renders `<h2 className="h3">`. What is missing is
   the level beneath: card titles are styled `<div>`s, so `<h2>` is the deepest
   level in the document and a screen reader cannot navigate within a section.
4. **Toast announcements exist; engine and stream announcements do not.**
   `.st-toasts` carries `aria-live="polite"` and is correct. The engine phase and
   the arrival of a completed response change silently.
4. **Contrast is verified for text but not for non-text UI** — icon buttons,
   toggle tracks, borders. WCAG 1.4.11 asks for 3:1 on those and nothing measures
   them.
5. **The reading text size tops out at 22 px** and the UI chrome does not scale
   with it — correct by design, but there is no way to scale the chrome for
   someone who needs a larger interface overall.
6. **No high-contrast mode beyond the OS media query**, which cannot be verified
   on a machine where it is not enabled.
7. **Long-session fatigue is handled ad hoc** — dark accent tinting, ambient
   rules, no idle loops — but nothing measures it.
8. **The tool-approval card is the highest-stakes interaction** in the app and has
   the least generous target sizing and no visible focus order.

## The plan

### Substance
1. ~~Fix the focus drop and add a trap.~~ **Done.**
2. ~~Deepen the heading hierarchy.~~ **Done, and it uncovered a second defect:**
   every screen had **two `<h1>` elements**, because the decorative compact title
   copy in the scroll collapse was also an `<h1>`. It is now a `<span
   aria-hidden>`. Verified per route: one `h1` each, with `h2` and `h3` beneath.
3. **Announce what changes without user action** — engine phase and response
   completion (`aria-live="polite"`). Toasts already do this correctly; leave
   them alone and do not announce streaming tokens.
4. **Extend the contrast probe to non-text UI** — 3:1 for icon buttons, toggle
   tracks, input borders and focus indicators, measured the same way text is.
5. **An interface scale** independent of the reading size: 90 / 100 / 110 / 125%,
   driving a root `--st-ui-scale` that the type and spacing tokens multiply
   through. The reading size stays conversation-only, as designed.

### Long sessions
6. **A fatigue budget, measured.** Extend `npm run luminance` with a second
   assertion: in dark mode, **the total area of saturated surface on screen** must
   stay under a threshold. This is what the green-button complaint actually was,
   expressed as a number.
7. **Contrast mode verification**: the probe runs a pass with
   `prefers-contrast: more` emulated, so the stronger-border path is tested rather
   than assumed.

## Motion

Nothing here is decorative:

| Concern | Rule |
|---|---|
| Reduced motion | Opacity substitutes for movement; loops that carry no state stop entirely |
| Focus movement | Never animated. Focus is instant, always |
| Live-region updates | No animation on the announced element, so the announcement is not delayed |
| Long-session calm | One loop per region; nothing animates while generating except the caret and the status dot |

## Verification

- `npm run contrast` extended: **non-text UI at 3:1**, and a `prefers-contrast`
  pass.
- New `flow a11y`: for each modal, assert `document.activeElement` is **inside
  the surface** on open (this fails today for the palette), then tab to the end
  and assert focus never leaves; assert every screen has a heading outline deeper
  than `<h2>`; assert the approval card is operable by keyboard alone.
- New luminance assertion: **saturated surface area under threshold in dark mode**.
- Manual pass with VoiceOver on Chat: sending, tool approval, completion.
