# Interaction guidelines

The rules behind the app's motion, fluidity and focus behaviour. Every one of
them is checked by `npm run audit` (static) or `npm run contrast` / `npm run
measure` / `npm run motion` (runtime), so a violation fails a command rather than
a code review.

Companion documents: [`design-system-notes.md`](design-system-notes.md) for how
the app maps onto the kit, [`dsh-recon.md`](dsh-recon.md) for the product this
one is modelled on.

---

## 1. Motion is feedback, not decoration

Every animation must answer one of three questions: *what changed*, *where did it
come from*, *what should I do next*. If it answers none of them, it does not
ship. There are no animated backgrounds, no parallax, no scroll-jacking, no
idle loops.

## 2. Duration is chosen by distance, not by taste

| Band | Token | Travel | Used for |
| --- | --- | --- | --- |
| micro | `--st-t-micro` 100 ms | ≤ 4 px | hover, press, colour and opacity on controls |
| control | `--st-t-control` 160 ms | ≤ 16 px | menus, rows appearing, small transforms |
| local | `--st-t-local` 240 ms | ≤ 24 px | drawers, sheets, disclosure |
| surface | `--st-t-surface` 320 ms | — | a whole panel or screen |
| choreographed | `--st-t-choreo` 460 ms | — | multi-element sequences, first paint only |

No raw millisecond value may appear in a `transition` or `animation` outside the
loop allowlist. If a duration is not on this scale, the scale is wrong — change
the scale, not the rule.

## 3. One easing family

- `--st-ease-standard` — `cubic-bezier(.4, 0, .2, 1)`, the default for everything
- `--st-ease-enter` — decelerate in
- `--st-ease-exit` — accelerate out
- `--st-ease-spring` — direct manipulation only (a toggle, a drag release). Never
  on an entrance.
- `--st-ease-in-out` — loops only

## 4. One moving thing at a time

Per region, one thing animates. Two panels sliding at once reads as noise, and on
a machine that is also running a local model it costs real frames.

**Nothing animates during generation** except the caret and the status dot. The
generation path is the one place where motion competes with the work.

## 5. Everything the user triggers is interruptible

Overlays are driven by `[data-open]` transitions, not one-shot keyframes, so
closing mid-open reverses from wherever it got to. `visibility` flips discretely
at the correct end of each run, which also keeps a closed drawer out of the tab
order.

## 6. Reduced motion substitutes opacity for movement

It never silences the interface. Distances collapse to zero and durations to a
single frame; opacity and colour still transition, so a state change remains
perceivable. Loops that carry no state stop entirely.

The OS preference is honoured by default, with a manual override in
**Settings → Reading & focus → Motion** for people who want stillness without
changing a system setting. `data-motion="full"` deliberately wins over the OS.

## 7. Continuity over re-painting

Route changes morph (`view-transition`), a disclosure eases open
(`interpolate-size` with `::details-content`), and the header's elevation follows
the scroll position through a CSS scroll timeline — no scroll listener, no layout
read on the main thread. `timeline-scope` hoists the timeline so a *sibling* of
the scroller can use it.

Transitions are skipped while a response is streaming: snapshotting a transcript
that is being appended to on every token would fight the SSE updates.

## 8. Focused views

- **One hero per screen.** Each screen names its primary object and gives it the
  only large type. Everything else drops a step.
- **Focus mode** (`⌘⇧F`) is a reading mode, not a different app: the same
  conversation with the chrome withdrawn. An ambient bar keeps what you need
  while reading — what is loaded, what it is doing, and the way out.
- Escape unwinds one layer at a time: an overlay first, then focus mode. A
  pending tool approval outranks both.
- Navigating to another screen leaves focus mode, rather than stranding a bare
  shell.

## 9. Long sessions

- **Prose is capped at a comfortable measure** — 68 characters per line, verified
  at every text size in the 10–22 px range. Code blocks, tables and metric grids
  keep the full width, because a wide table beats a wrapped one.
- **Text size affects the conversation only.** The interface keeps its own scale;
  a reader who wants larger prose does not want larger chrome.
- **Density** trades row height for rows on screen, for the screens you scan
  rather than read.
- **Follow the reader, not the stream.** New output scrolls into view only when
  you are already at the bottom. Scrolling back through a long conversation is
  never yanked forward by the next token.
- **Scroll position is remembered per conversation**, so returning to one puts
  you back where you were reading.
- **Large surfaces are never pure white or black.** `#f9fafb` and `#0e1113` are
  the extremes; the off-scale audit enforces that no literal colour sneaks in.
- `prefers-contrast: more` raises muted text and strengthens hairlines.

## 10. Liquid Glass: capsules, concentricity, and one hard rule

Apple's Liquid Glass has three principles — **Hierarchy, Harmony, Consistency** —
and one rule that overrides enthusiasm:

> **Glass is for the floating navigation/control layer only. Never apply it to
> content.** Glass cannot sample glass.

**Strata complies, and the audit keeps it that way.** The surfaces carrying a
material are exactly six, all control-layer: `.main__head`, `.focusbar`,
`.composer`, `.slashmenu`, `.scrim`, `.st-toast--toned`. No card, list, table or
media container has one. `npm run luminance` walks every route and fails if a
material is ever found inside another material.

- **Capsules are for controls; the kit's 8/14/20/24 scale is for surfaces.** A
  button, field, nav row or menu item is something you press or type into and is
  a capsule; a card, drawer, modal or toast is something content sits on and
  stays a rounded rectangle. Capsule-on-a-surface reads as a chip floating in a
  box. `npm run audit` fails if a control is not a capsule.
- **Concentric corners.** A rounded element inside a padded rounded parent takes
  the parent's radius minus the gap, floored at zero (`--st-r-nested`), so the
  two curves share a centre instead of merely nesting.
- **Press compresses the control** (`scale(0.97)`) rather than nudging it. Under
  reduced motion the scale is dropped; nothing else about the press changes.
- **The title collapses on scroll** by cross-fading two stacked copies. Animating
  `font-size` would reflow the header every frame; opacity is compositor-only.
- **Content dissolves under the bar** via a `mask-image` fade rather than meeting
  a hard rule. A mask is not a gradient, so the flat-surface rule still holds.
- **`.clear` glass is not implemented.** Apple reserves it for bold foreground
  over media-rich content; Strata has no media.

### Lensing: measured, then declined

Refraction is the material's signature, and it *is* reachable here:
`backdrop-filter: url(#svgfilter)` executes in this runtime — verified with both
an `feTurbulence` displacement (13.6% of pixels changed over text) and an
`feImage` displacement map (20.6%).

It was still declined, on evidence:

| Finding | Result |
| --- | --- |
| Frame cost while scrolling | p95 **1.03×** — entirely affordable |
| Visual change over `.composer` | **mean delta 0.00/255, zero pixels changed** |
| Why | the composer sits over a *uniform* backdrop — there is nothing to refract |
| Where it *would* show | over the suggestion cards behind the slash menu — a text menu, where a 40px displacement hurts legibility |
| Portability | Chromium-only, and the displacement map is shape- and size-specific, so it must be rebuilt on every resize |

The effect works and costs nothing; it simply has nothing to act on. Shipping it
would add a per-resize canvas rebuild and a Chromium-only code path for a change
no one can see. Revisit if Strata ever puts glass over media.

## 11. Materials are rationed

Blur is the only decorative device the system allows itself, and only where it
does something. Three tiers, six surfaces, every one of them overlaying content
that moves underneath:

| Tier | Value | Surfaces |
| --- | --- | --- |
| thin | `blur(12px)` | screen header, focus bar |
| regular | `blur(24px)` | composer, slash menu, toasts |
| thick | `blur(40px)` | scrim behind modals and the drawer |

**Excluded on purpose:** the sidebar (never overlaps anything), every card (flat
background — a blur would cost a compositor pass and show nothing), and the
transcript (repaints on every streamed token; the one place blur genuinely
hurts).

Every material pairs its blur with `saturate(180%)` — without it blurred content
goes grey — and a **specular edge**, a 1px inset highlight that reads as light
catching the surface. A hairline, not a gradient.

The blur value must be a `--st-material-*` token, and the audit enforces it: the
token is what collapses to `none` under reduced transparency, so gating cannot be
forgotten. `prefers-reduced-transparency` and the manual override in Settings
both produce a flat surface with **identical geometry**.

## 12. A large bright fill is what fatigues

Brightness is a function of *area*, not just colour. A 2px progress bar at full
saturation is fine; the same green filling a 9,700px² button is glare.

In dark mode, large accent-filled surfaces — the primary button and the reader's
own message bubble — are therefore a **tinted material** rather than a solid
fill, with the accent showing as light through the surface:

| | Luminance | vs page | Label |
| --- | --- | --- | --- |
| solid fill (before) | **0.496** | 9.86:1 | 7.70:1 |
| tinted, at rest | **0.055** | 1.90:1 | 5.73:1 |
| tinted, hover | 0.079 | 2.33:1 | 4.67:1 |

Thin accent strokes — progress bars, gauge arcs, toggles, dots, sparklines —
keep the bright accent in both themes. Light mode keeps the solid fill: 2.5:1
against its background is gentle, and solid reads as confident on white.

`npm run luminance` fails the build if any surface over 4,000px² exceeds
luminance 0.15 in dark mode.

## 13. Navigation makes the hierarchy obvious

Drawn from the harness this app is modelled on, and written for someone who finds
computers daunting: **every row says what it does, and configuration lives one
level down.**

- The collapse control gets its own row under the logo. Wedged beside the
  wordmark it read as a peer of the brand rather than a window control.
- One obvious action at the top — **New session** — in the accent material.
- Daily destinations are flat rows. Configuration (Models, Engine tuning,
  Settings) lives behind **… More**, so eight rows became three and a menu.
- **Workspaces group conversations by the folder they belong to**, which is the
  mental model a person already has. Search filters them; the folder-plus opens
  the *system* folder chooser, because asking someone to type an absolute path is
  not a real option.
- The footer holds what is live: the engine chip, Monitor, Serving.

## 14. Controls own their own popups

A native `<select>` popup is drawn by the operating system — on macOS a dark
system menu with a blue highlight that shares nothing with the rest of the app.
`Select` replaces it with a real listbox: capsule trigger, glass popup rendered
through a **portal** so it can never become glass-inside-glass, full keyboard
support (↑/↓, Home/End, Enter, Escape, typeahead), `aria-activedescendant`
tracking, and focus returned to the trigger on close.

`npm run audit` fails the build if a native `<select>` reappears anywhere in
`src/`. It strips comments first — the component's own doc comment *describes*
the element it replaces, and matching that prose would be a permanent false
alarm.

## 15. Contrast is a property of the pairing, not the token

The failures are always pairings — a label on a filled button, text on a
translucent tint, muted copy on a quiet panel. So contrast is measured on the
rendered result, with translucent backgrounds composited over their ancestors,
and every interactive state measured separately. The kit's primary button once
passed at rest and failed on hover; a single measurement would have missed it.

Minimums: **4.5:1** for body text, **3:1** for text ≥ 24 px or ≥ 18.66 px bold.
Current status: all pairings pass, light and dark, in every state — see
`npm run contrast`.

One measurement trap worth knowing: Chrome serialises a `color-mix()` result as
`color(srgb 0.117 0.29 0.24)` with 0–1 components. Reading those as 0–255 makes
a tinted fill look almost black, which silently defeats both the contrast and
luminance checks. Both probes parse that form explicitly.
