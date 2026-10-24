# 13 · Overlays — drawers, menus, selects, popovers

**Files** `SamplingDrawer.tsx`, `Sidebar.tsx` (More menu), `ui.tsx` (Select),
`TelemetryPopover.tsx`, `Shortcuts.tsx`

The surfaces that float above content. All were converted to the same pattern
last round: **stays mounted, driven by `data-open`, animates both ways, portals
out of any glass ancestor**.

## What they are today

| Surface | Container | Material | Dismissal |
|---|---|---|---|
| Sampling drawer | `.st-drawer.sheet` | regular | Esc, scrim, Close |
| More menu | `.menu` | regular | click-away, Esc |
| Select popup | `.select__pop` | regular | select, click-away, Esc, scroll |
| Telemetry popover | `.telepop` | regular | click-away, Esc, scroll |
| Shortcuts dialog | `.modal` | opaque + specular | Esc, scrim |
| Command palette | `.modal` | opaque + specular | Esc, scrim |

Every one is portalled where it could otherwise nest inside glass, and
`npm run luminance` fails the build if that ever regresses.

## Where it falls short

1. ~~Six overlay types, three different dismissal grammars.~~ **Measured, and
   documented here.** The plan guessed at two grammars; there are three, and the
   third is the interesting one.

   | Overlay | Escape | Outside click | Scroll |
   |---|---|---|---|
   | Command palette | closes | closes | stays |
   | Shortcuts | closes | closes | stays |
   | Sampling drawer | closes | — | stays |
   | Telemetry popover | closes | closes | **closes** |
   | Select popup | closes | closes | **repositions** |
   | Menus (More, session) | closes | closes | stays |

   The rule beneath it: **an overlay anchored to content follows or leaves with
   that content; one that is not anchored ignores scrolling.** A telemetry
   popover is a readout of a row that has just scrolled away, so it closes. A
   Select is a control you are mid-interaction with, so it moves with its trigger
   rather than discarding your place. Modals are not anchored to anything, so
   scroll is none of their business.

   **Measuring this took three attempts, and the first two were wrong.** Both
   reported "scroll: stays" for every overlay — because the transcript was
   already scrolled to its maximum, so `scrollTop += 80` clamped to no movement
   and no scroll event ever fired. A test that asserts nothing happened, when
   nothing was attempted, is the most convincing kind of wrong.
2. ~~None of them trap focus.~~ **Wrong — they all do.** Measured against the
   *visible* overlay: the palette, the shortcuts dialog and the sampling drawer
   each land focus inside on open and keep it there across 18 Tab presses.
   `useSurfaceFocus` has done this since the first pass; the plan asserted the
   opposite without testing. (Testing this needs care: querying `.modal` finds
   the hidden palette rather than the open dialog, which is how the claim
   survived — an early probe of mine "confirmed" the bug the same wrong way.)
3. **The Select popup is the only one without a visible focus ring on its active
   option** when driven by keyboard — `data-active` sets a background, which is
   the same treatment as hover.
4. **No overlay is repositioned on resize** except the Select, which repositions
   on scroll and resize.
5. **Escape precedence is implemented in two places** — the App key handler and
   each overlay — with the ordering implicit rather than stated.
6. ~~Toasts have no tray.~~ **Fixed.** The stack caps at three — further toasts
   displace the oldest rather than pushing everything off-screen — and a hovered
   or focused toast has its dismissal clock stopped and restarted on leave, so
   reading one does not race its own timer. Verified: five pushes leave three on
   screen, and a hovered toast is still alive at 6.3 s against a 5.2 s timer.
7. **No overlay announces itself** beyond `role="dialog"`; none uses
   `aria-modal` with a labelled heading consistently.

## The plan

### Substance — one grammar, written down
1. **Document the dismissal rules** in the interaction guidelines: *transient
   anchored surfaces (menu, select, popover) close on scroll; modal surfaces do
   not; everything closes on Escape and click-away.* Then verify it.
2. **Focus management for every modal surface**: focus moves into the surface on
   open, is trapped while open, and returns to the invoking control on close. The
   `Select` already returns focus; the rest do not trap.
3. **A visible keyboard focus ring** inside listbox popups, distinct from the
   hover background — hover and keyboard selection must not look identical.
4. **Centralise Escape precedence**: overlay stack first, then focus mode, then
   nothing. One place decides, and the rule is stated in the guidelines.
5. ~~Toasts gain pause-on-hover and a stack cap.~~ **Done.** The oldest is
   dropped rather than collapsed into a count — a counter the user cannot open is
   not worth the space it takes.

### Fluidity
6. **Every anchored surface grows from its anchor** rather than the window edge —
   `transform-origin` set from the trigger's position, so a menu from the bottom
   bar opens upward *from that bar*.
7. ~~Exit animations for modals.~~ **Already correct.** Measured: on `Escape` the
   modal passes through opacity **0.35** at 90 ms before settling at 0 with
   `visibility: hidden`. The plan asserted they "cut out" without checking;
   because they stay mounted and are driven by `data-open`, the discrete
   visibility transition gives them a real exit already.

## Motion

| Surface | Enter | Exit |
|---|---|---|
| Drawer | slide 20 px + fade, `--st-t-local` | reverse, interruptible |
| Menu / popover | 6 px lift + fade, `--st-t-control` | reverse |
| Select popup | 4 px lift + fade, `--st-t-control` | fade only |
| Modal | scale .98 + fade, `--st-t-local` | currently instant — needs a reverse |
| Toast | 10 px lift, `--st-t-local` | 6 px, `--st-t-control` |

All collapse to opacity-only under reduced motion.

## Verification

- New `flow overlays2`: opens each surface, asserts focus lands inside, tabs
  past the last control and asserts focus stays trapped, then closes and asserts
  focus returns to the trigger.
- Assert no anchored surface overflows the viewport at 900 px — the telemetry
  popover clamp is the model to copy.
